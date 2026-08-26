"""Encrypted secret store contract (SPEC D5, task T2).

Secrets are Fernet-encrypted at rest under a passphrase-derived key. These tests
pin the security guarantees: round-trip, wrong-passphrase rejection, no plaintext
(secret value, secret name, or passphrase) on disk, owner-only file mode, and a
typed error when the store is missing.
"""

import sys
from pathlib import Path

import pytest

from app.setup import secret_store


def _store(tmp_path: Path) -> Path:
    return tmp_path / "secrets.enc"


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = _store(tmp_path)
    secrets = {"OPENAI_API_KEY": "sk-abc123", "FRED_API_KEY": "fred-xyz"}
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
    secret_store.save_secrets(
        {"OPENAI_API_KEY": "sk-supersecretvalue"}, "pw", path=path
    )
    raw = path.read_bytes()
    assert b"sk-supersecretvalue" not in raw
    assert b"OPENAI_API_KEY" not in raw


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
