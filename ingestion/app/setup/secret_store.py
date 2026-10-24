"""Encrypted secret store for the portopt install wizard (SPEC D5).

Secrets are serialised to JSON, encrypted with Fernet under a key derived from
the user's master passphrase (scrypt), and written to ``~/.portopt/secrets.enc``
as ``magic || version || log2(N) || r || p || salt || token``. The passphrase is
never persisted — only the scrypt cost, the random salt and the ciphertext are
stored, so the cost can be raised later without orphaning existing stores. Files
are created owner-only (0600) and replaced atomically, so an interrupted save
leaves the previous store intact. Stores written before the header existed
(``salt || token``, N=2**14) still load and are upgraded on the next save.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

DEFAULT_SECRETS_PATH = Path.home() / ".portopt" / "secrets.enc"

# The passphrase is the only thing between a copied secrets.enc and its keys, and no
# strength estimator ships with the CLI, so length is the enforceable floor.
MIN_PASSPHRASE_LENGTH = 12

_SALT_BYTES = 16
_KEY_BYTES = 32
# OWASP's minimum scrypt cost (128 MiB per guess); the file is an offline target.
_SCRYPT_N = 2**17
_SCRYPT_R = 8
_SCRYPT_P = 1

_MAGIC = b"PTSS"
_FORMAT_VERSION = 1
_HEADER_BYTES = len(_MAGIC) + 4  # version, log2(N), r, p

# A header read back from disk is untrusted: bound it so a corrupt or hostile file
# fails as a corrupt store instead of exhausting memory inside scrypt.
_MAX_KDF_MEMORY = 2**30
_MAX_SCRYPT_P = 16


class SecretStoreError(Exception):
    """Base error for the secret store."""


class SecretStoreNotFoundError(SecretStoreError):
    """Raised when loading a store that does not exist."""


class InvalidPassphraseError(SecretStoreError):
    """Raised when the passphrase cannot decrypt the store (wrong or corrupt)."""


@dataclass(frozen=True)
class _KdfParams:
    """scrypt cost parameters recorded in the store header."""

    n: int
    r: int
    p: int

    def header(self) -> bytes:
        return _MAGIC + bytes(
            [_FORMAT_VERSION, self.n.bit_length() - 1, self.r, self.p]
        )

    def is_sane(self) -> bool:
        return (
            self.n >= 2
            and self.r >= 1
            and 1 <= self.p <= _MAX_SCRYPT_P
            and 128 * self.n * self.r <= _MAX_KDF_MEMORY
        )


# The cost every store used before the header existed.
_LEGACY_KDF = _KdfParams(n=2**14, r=8, p=1)


def _derive_key(passphrase: str, salt: bytes, params: _KdfParams) -> bytes:
    kdf = Scrypt(salt=salt, length=_KEY_BYTES, n=params.n, r=params.r, p=params.p)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def _replace_atomically(path: Path, data: bytes) -> None:
    """Swap ``data`` into ``path`` through a sibling temp file, never in place.

    The store holds every key behind one passphrase: truncating it first and
    crashing mid-write would lose them all. ``mkstemp`` creates the temp file 0600,
    so the data is never readable by other users, and the rename is atomic on the
    same filesystem.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _split_blob(blob: bytes) -> tuple[_KdfParams, bytes, bytes]:
    """Split a stored blob into ``(kdf params, salt, token)``."""
    if not blob.startswith(_MAGIC):
        return _LEGACY_KDF, blob[:_SALT_BYTES], blob[_SALT_BYTES:]
    if len(blob) < _HEADER_BYTES + _SALT_BYTES:
        raise InvalidPassphraseError("Wrong passphrase or corrupt store")
    version, log2_n, r, p = blob[len(_MAGIC) : _HEADER_BYTES]
    params = _KdfParams(n=1 << log2_n, r=r, p=p)
    if version != _FORMAT_VERSION or not params.is_sane():
        raise InvalidPassphraseError("Wrong passphrase or corrupt store")
    salt_end = _HEADER_BYTES + _SALT_BYTES
    return params, blob[_HEADER_BYTES:salt_end], blob[salt_end:]


def store_exists(*, path: Path | None = None) -> bool:
    """Return whether a secret store has already been written at ``path``."""
    return (path or DEFAULT_SECRETS_PATH).exists()


def save_secrets(
    secrets: Mapping[str, str], passphrase: str, *, path: Path | None = None
) -> Path:
    """Encrypt ``secrets`` under ``passphrase`` and atomically write to ``path`` (0600).

    Raises:
        OSError: If the store cannot be written; any previous store is left intact.
    """
    path = path or DEFAULT_SECRETS_PATH
    params = _KdfParams(n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    salt = os.urandom(_SALT_BYTES)
    token = Fernet(_derive_key(passphrase, salt, params)).encrypt(
        json.dumps(dict(secrets)).encode("utf-8")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # best effort — POSIX only
        path.parent.chmod(0o700)
    _replace_atomically(path, params.header() + salt + token)
    return path


def load_secrets(passphrase: str, *, path: Path | None = None) -> dict[str, str]:
    """Decrypt and return the stored secrets, or raise a typed error."""
    path = path or DEFAULT_SECRETS_PATH
    if not path.exists():
        raise SecretStoreNotFoundError(f"No secret store at {path}")
    params, salt, token = _split_blob(path.read_bytes())
    try:
        payload = Fernet(_derive_key(passphrase, salt, params)).decrypt(token)
    except InvalidToken as exc:
        raise InvalidPassphraseError("Wrong passphrase or corrupt store") from exc
    return json.loads(payload)
