"""Encrypted secret store contract (SPEC D5).

Secrets are Fernet-encrypted at rest under a passphrase-derived key. These tests
pin the security guarantees: round-trip, wrong-passphrase rejection, no plaintext
(secret value, secret name, or passphrase) on disk, owner-only file mode, and a
typed error when the store is missing.
"""

import base64
import json
import os
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from app.setup import secret_store

# Captured at import, before the autouse fixture lowers the cost for speed, so the
# shipped default can still be asserted.
_SHIPPED_SCRYPT_N = secret_store._SCRYPT_N


@pytest.fixture(autouse=True)
def _cheap_kdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop scrypt to a toy cost: the shipped one takes seconds per derivation."""
    monkeypatch.setattr(secret_store, "_SCRYPT_N", 2**10)


def _store(tmp_path: Path) -> Path:
    return tmp_path / "secrets.enc"


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = _store(tmp_path)
    secrets = {"TRADING_212_API_KEY": "t212-abc123", "FRED_API_KEY": "fred-xyz"}
    secret_store.save_secrets(secrets, "correct horse battery staple", path=path)
    loaded = secret_store.load_secrets("correct horse battery staple", path=path)
    assert loaded == secrets


def test_wrong_passphrase_raises(tmp_path: Path) -> None:
    path = _store(tmp_path)
    secret_store.save_secrets({"A": "1"}, "right-pass", path=path)
    with pytest.raises(secret_store.InvalidPassphraseError):
        secret_store.load_secrets("wrong-pass", path=path)


def test_ciphertext_contains_no_plaintext(tmp_path: Path) -> None:
    path = _store(tmp_path)
    secret_store.save_secrets({"FRED_API_KEY": "supersecretvalue"}, "pw", path=path)
    raw = path.read_bytes()
    assert b"supersecretvalue" not in raw
    assert b"FRED_API_KEY" not in raw


def test_passphrase_not_persisted(tmp_path: Path) -> None:
    path = _store(tmp_path)
    secret_store.save_secrets({"A": "1"}, "myS3cretPass", path=path)
    assert b"myS3cretPass" not in path.read_bytes()


def test_missing_file_load_raises(tmp_path: Path) -> None:
    with pytest.raises(secret_store.SecretStoreNotFoundError):
        secret_store.load_secrets("pw", path=_store(tmp_path))


def test_corrupt_file_raises_invalid_passphrase(tmp_path: Path) -> None:
    path = _store(tmp_path)
    path.write_bytes(b"not-a-valid-encrypted-store-blob")
    with pytest.raises(secret_store.InvalidPassphraseError):
        secret_store.load_secrets("pw", path=path)


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode not enforced on Windows"
)
def test_secret_file_is_owner_only(tmp_path: Path) -> None:
    path = _store(tmp_path)
    secret_store.save_secrets({"A": "1"}, "pw", path=path)
    assert (path.stat().st_mode & 0o777) == 0o600


def test_save_never_rewrites_the_existing_store_in_place(tmp_path: Path) -> None:
    # A hard link shares the inode: truncate-then-write would change both names, so
    # an interrupted save could leave a partial store with the old keys already gone.
    path = _store(tmp_path)
    secret_store.save_secrets({"A": "1"}, "pw", path=path)
    snapshot = tmp_path / "snapshot.enc"
    try:
        os.link(path, snapshot)
    except OSError:
        pytest.skip("hard links unsupported on this filesystem")
    secret_store.save_secrets({"A": "2"}, "pw", path=path)
    assert secret_store.load_secrets("pw", path=snapshot) == {"A": "1"}
    assert secret_store.load_secrets("pw", path=path) == {"A": "2"}


def test_a_failed_save_keeps_the_previous_store_and_leaves_no_litter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _store(tmp_path)
    secret_store.save_secrets({"A": "1"}, "pw", path=path)

    def _crash(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError, match="disk went away"):
        secret_store.save_secrets({"A": "2"}, "pw", path=path)
    monkeypatch.undo()

    assert secret_store.load_secrets("pw", path=path) == {"A": "1"}
    assert [p.name for p in tmp_path.iterdir()] == ["secrets.enc"]


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode not enforced on Windows"
)
def test_store_is_created_owner_only_not_tightened_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With chmod disabled the final mode is the creation mode, so a file that is only
    # tightened after the write would be readable by other users in between.
    monkeypatch.setattr(os, "chmod", lambda *_a, **_kw: None)
    previous = os.umask(0)
    try:
        path = secret_store.save_secrets({"A": "1"}, "pw", path=_store(tmp_path))
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600


def test_shipped_scrypt_cost_meets_owasp_minimum() -> None:
    # OWASP's scrypt minimum is N=2**17 with r=8, p=1 (128 MiB per guess).
    assert _SHIPPED_SCRYPT_N >= 2**17
    assert secret_store._SCRYPT_R >= 8
    assert secret_store._SCRYPT_P >= 1


def test_load_uses_kdf_parameters_recorded_in_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cost must travel with the file, otherwise raising the module default later
    # makes every existing store undecryptable.
    path = _store(tmp_path)
    secret_store.save_secrets({"A": "1"}, "pw", path=path)
    monkeypatch.setattr(secret_store, "_SCRYPT_N", 2**11)
    assert secret_store.load_secrets("pw", path=path) == {"A": "1"}


def _legacy_blob(secrets: dict[str, str], passphrase: str) -> bytes:
    """A store as written before the KDF header existed: ``salt || token``."""
    salt = os.urandom(16)
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))
    return salt + Fernet(key).encrypt(json.dumps(secrets).encode("utf-8"))


def test_legacy_headerless_store_still_loads(tmp_path: Path) -> None:
    path = _store(tmp_path)
    path.write_bytes(_legacy_blob({"A": "1"}, "old-pass"))
    assert secret_store.load_secrets("old-pass", path=path) == {"A": "1"}


def test_resaving_a_legacy_store_upgrades_it_to_the_versioned_format(
    tmp_path: Path,
) -> None:
    path = _store(tmp_path)
    path.write_bytes(_legacy_blob({"A": "1"}, "old-pass"))
    loaded = secret_store.load_secrets("old-pass", path=path)
    secret_store.save_secrets(loaded, "old-pass", path=path)
    assert path.read_bytes().startswith(secret_store._MAGIC)
    assert secret_store.load_secrets("old-pass", path=path) == {"A": "1"}


@pytest.mark.parametrize(
    ("version", "log2_n", "r", "p"),
    [
        (99, 10, 8, 1),  # unknown format version
        (1, 40, 8, 1),  # N = 2**40 would exhaust memory
        (1, 10, 0, 1),  # r = 0 is not a valid scrypt parameter
        (1, 10, 8, 0),  # p = 0 likewise
        (1, 20, 255, 1),  # memory cost far beyond any legitimate store
    ],
)
def test_malformed_kdf_header_is_a_corrupt_store_not_a_crash(
    tmp_path: Path, version: int, log2_n: int, r: int, p: int
) -> None:
    path = _store(tmp_path)
    header = secret_store._MAGIC + bytes([version, log2_n, r, p])
    path.write_bytes(header + os.urandom(16) + b"token")
    with pytest.raises(secret_store.InvalidPassphraseError):
        secret_store.load_secrets("pw", path=path)


def test_truncated_versioned_store_is_a_corrupt_store(tmp_path: Path) -> None:
    path = _store(tmp_path)
    secret_store.save_secrets({"A": "1"}, "pw", path=path)
    path.write_bytes(path.read_bytes()[:10])
    with pytest.raises(secret_store.InvalidPassphraseError):
        secret_store.load_secrets("pw", path=path)


def test_store_exists_reports_whether_a_store_was_written(tmp_path: Path) -> None:
    path = _store(tmp_path)
    assert secret_store.store_exists(path=path) is False
    secret_store.save_secrets({"A": "1"}, "pw", path=path)
    assert secret_store.store_exists(path=path) is True
