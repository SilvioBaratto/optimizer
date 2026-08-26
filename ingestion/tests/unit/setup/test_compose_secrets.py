"""Compose-secret rendering contract (SPEC D6, task T6).

`render` writes every declared secret to `<secrets_dir>/<name>` (empty for
unconfigured ones so `docker compose up` never fails on a missing file), each
owner-only (0600); `cleanup` removes them on `portopt stop`.
"""

import sys
from pathlib import Path

import pytest

from app.setup import compose_secrets as cs


def test_render_writes_all_declared_secret_files(tmp_path: Path) -> None:
    written = cs.render(
        {"openai_api_key": "sk-x", "fred_api_key": "f"}, secrets_dir=tmp_path
    )
    assert {p.name for p in written} == set(cs.SECRET_NAMES)
    assert (tmp_path / "openai_api_key").read_text(encoding="utf-8") == "sk-x"
    # Unconfigured secret still rendered, but empty.
    assert (tmp_path / "trading_212_api_key").read_text(encoding="utf-8") == ""


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode not enforced on Windows"
)
def test_rendered_files_are_owner_only(tmp_path: Path) -> None:
    cs.render({"openai_api_key": "x"}, secrets_dir=tmp_path)
    assert (tmp_path / "openai_api_key").stat().st_mode & 0o777 == 0o600


def test_cleanup_removes_rendered_files(tmp_path: Path) -> None:
    cs.render({"openai_api_key": "x"}, secrets_dir=tmp_path)
    cs.cleanup(secrets_dir=tmp_path)
    assert not any((tmp_path / name).exists() for name in cs.SECRET_NAMES)


def test_cleanup_missing_dir_is_noop(tmp_path: Path) -> None:
    cs.cleanup(secrets_dir=tmp_path / "does-not-exist")  # no raise
