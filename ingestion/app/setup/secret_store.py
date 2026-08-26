"""Encrypted secret store for the portopt install wizard (SPEC D5).

Secrets are serialised to JSON, encrypted with Fernet under a key derived from
the user's master passphrase (scrypt), and written to ``~/.portopt/secrets.enc``
as ``salt || token``. The passphrase is never persisted — only the random salt
and the ciphertext are stored. Files are created owner-only (0600).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

DEFAULT_SECRETS_PATH = Path.home() / ".portopt" / "secrets.enc"

_SALT_BYTES = 16
_KEY_BYTES = 32
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


class SecretStoreError(Exception):
    """Base error for the secret store."""


class SecretStoreNotFoundError(SecretStoreError):
    """Raised when loading a store that does not exist."""


class InvalidPassphraseError(SecretStoreError):
    """Raised when the passphrase cannot decrypt the store (wrong or corrupt)."""


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=_KEY_BYTES, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def save_secrets(
    secrets: Mapping[str, str], passphrase: str, *, path: Path | None = None
) -> Path:
    """Encrypt ``secrets`` under ``passphrase`` and write to ``path`` (0600)."""
    path = path or DEFAULT_SECRETS_PATH
    salt = os.urandom(_SALT_BYTES)
    token = Fernet(_derive_key(passphrase, salt)).encrypt(
        json.dumps(dict(secrets)).encode("utf-8")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # best effort — POSIX only
        path.parent.chmod(0o700)
    path.write_bytes(salt + token)
    path.chmod(0o600)
    return path


def load_secrets(passphrase: str, *, path: Path | None = None) -> dict[str, str]:
    """Decrypt and return the stored secrets, or raise a typed error."""
    path = path or DEFAULT_SECRETS_PATH
    if not path.exists():
        raise SecretStoreNotFoundError(f"No secret store at {path}")
    blob = path.read_bytes()
    salt, token = blob[:_SALT_BYTES], blob[_SALT_BYTES:]
    try:
        payload = Fernet(_derive_key(passphrase, salt)).decrypt(token)
    except InvalidToken as exc:
        raise InvalidPassphraseError("Wrong passphrase or corrupt store") from exc
    return json.loads(payload)
