"""Compose-secret rendering contract (SPEC D6, task T6).

`render` writes every declared secret to `<secrets_dir>/<name>` (empty for
unconfigured ones so `docker compose up` never fails on a missing file), each
owner-only (0600); `cleanup` removes them on `portopt stop`.
"""

import sys
from pathlib import Path

import pytest

from app.setup import compose_secrets as cs

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def test_render_writes_all_declared_secret_files(tmp_path: Path) -> None:
    written = cs.render({"fred_api_key": "f"}, secrets_dir=tmp_path)
    assert {p.name for p in written} == set(cs.SECRET_NAMES)
    assert (tmp_path / "fred_api_key").read_text(encoding="utf-8") == "f"
    # Every other declared secret (trading212 + the new provider keys) is still
    # rendered, but empty when unconfigured — so `docker compose up` never fails
    # on a missing `file:`.
    for name in cs.SECRET_NAMES:
        if name != "fred_api_key":
            assert (tmp_path / name).read_text(encoding="utf-8") == ""


def test_compose_declares_a_file_for_every_secret_name() -> None:
    # Parity: every rendered secret must have a top-level compose declaration, or
    # `docker compose up` fails on the missing `file:` mapping. Read as text
    # (no yaml dep in ingestion); names come from SECRET_NAMES, not literals.
    compose = _COMPOSE.read_text(encoding="utf-8")
    for name in cs.SECRET_NAMES:
        assert f"file: ./secrets/{name}" in compose


def test_secret_names_grew_beyond_the_original_three() -> None:
    # T3 added the fund's per-provider auth secret keys; guard against a
    # regression that drops them back to the original trading212 + fred trio.
    assert len(cs.SECRET_NAMES) >= 12
    assert len(set(cs.SECRET_NAMES)) == len(cs.SECRET_NAMES)  # no dupes


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode not enforced on Windows"
)
def test_rendered_files_are_owner_only(tmp_path: Path) -> None:
    cs.render({"fred_api_key": "x"}, secrets_dir=tmp_path)
    assert (tmp_path / "fred_api_key").stat().st_mode & 0o777 == 0o600


def test_cleanup_removes_rendered_files(tmp_path: Path) -> None:
    cs.render({"fred_api_key": "x"}, secrets_dir=tmp_path)
    cs.cleanup(secrets_dir=tmp_path)
    assert not any((tmp_path / name).exists() for name in cs.SECRET_NAMES)


def test_cleanup_missing_dir_is_noop(tmp_path: Path) -> None:
    cs.cleanup(secrets_dir=tmp_path / "does-not-exist")  # no raise
