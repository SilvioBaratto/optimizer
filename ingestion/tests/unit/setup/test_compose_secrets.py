"""Compose-secret rendering contract (SPEC D6).

`render` writes every declared secret to `<secrets_dir>/<name>` (empty for
unconfigured ones so `docker compose up` never fails on a missing file), each
owner-only (0600); `cleanup` removes them (all, or just the ones named).
"""

import re
import shutil
import subprocess
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


def test_render_overwrites_a_docker_planted_directory(tmp_path: Path) -> None:
    # Docker creates a missing secret mount source as a directory when the stack comes
    # up before render, so render must replace that placeholder with the value file
    # instead of failing with a directory write error.
    (tmp_path / "trading_212_api_key").mkdir(parents=True)
    written = cs.render({"trading_212_api_key": "k"}, secrets_dir=tmp_path)
    assert (tmp_path / "trading_212_api_key").is_file()
    assert (tmp_path / "trading_212_api_key").read_text(encoding="utf-8") == "k"
    assert {p.name for p in written} == set(cs.SECRET_NAMES)


def test_compose_declares_a_file_for_every_secret_name() -> None:
    # Parity: every rendered secret must have a top-level compose declaration, or
    # `docker compose up` fails on the missing `file:` mapping. Read as text
    # (no yaml dep in ingestion); names come from SECRET_NAMES, not literals.
    compose = _COMPOSE.read_text(encoding="utf-8")
    for name in cs.SECRET_NAMES:
        assert f"file: ./secrets/{name}" in compose


def test_secret_names_grew_beyond_the_original_three() -> None:
    # The fund's per-provider auth secret keys must be present alongside
    # the original trading212 + fred trio.
    assert len(cs.SECRET_NAMES) >= 12
    assert len(set(cs.SECRET_NAMES)) == len(cs.SECRET_NAMES)


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


def test_cleanup_with_names_removes_only_those_files(tmp_path: Path) -> None:
    cs.render({"fred_api_key": "x", "openai_api_key": "y"}, secrets_dir=tmp_path)
    cs.cleanup(secrets_dir=tmp_path, names=["openai_api_key"])
    assert not (tmp_path / "openai_api_key").exists()
    assert (tmp_path / "fred_api_key").read_text(encoding="utf-8") == "x"


def _scheduler_mounted_secrets() -> set[str]:
    """Secret names listed under the `scheduler` service in docker-compose.yml."""
    mounted: set[str] = set()
    in_scheduler = in_secrets = False
    for line in _COMPOSE.read_text(encoding="utf-8").splitlines():
        if line.startswith("  ") and not line.startswith("   ") and line.strip():
            in_scheduler = line.strip() == "scheduler:"
            in_secrets = False
        elif in_scheduler and line.startswith("    ") and not line.startswith("     "):
            in_secrets = line.strip() == "secrets:"
        elif in_scheduler and in_secrets and line.strip().startswith("- "):
            mounted.add(line.strip()[2:])
    return mounted


def test_daemon_and_fund_only_names_partition_the_declared_secrets() -> None:
    assert set(cs.DAEMON_SECRET_NAMES) == _scheduler_mounted_secrets()
    assert set(cs.DAEMON_SECRET_NAMES) | set(cs.FUND_ONLY_SECRET_NAMES) == set(
        cs.SECRET_NAMES
    )
    assert not set(cs.DAEMON_SECRET_NAMES) & set(cs.FUND_ONLY_SECRET_NAMES)


def test_cleanup_missing_dir_is_noop(tmp_path: Path) -> None:
    cs.cleanup(secrets_dir=tmp_path / "does-not-exist")  # no raise


def test_dockerignore_keeps_plaintext_secrets_out_of_the_build_context() -> None:
    """The compose build context is the repo root, so an unignored `secrets/` or
    `.certs/` is uploaded to the daemon on every `docker compose build`."""
    entries = {
        line.strip()
        for line in (_REPO_ROOT / ".dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    # The recursive form also matches at the context root and additionally drops
    # the nested copies under each member tree.
    assert entries & {"secrets/", "**/secrets/"}
    assert entries & {".certs/", "**/.certs/"}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize(
    "stray",
    [
        "secrets/openai_api_key",
        "ingestion/secrets/openai_api_key",
        "fund/secrets/openai_api_key",
    ],
)
def test_plaintext_secret_files_are_git_ignored_in_every_directory(stray: str) -> None:
    # A render that lands in a subdirectory (e.g. a start from `ingestion/`) must not
    # become committable; only the root `secrets/` was ignored before.
    inside = subprocess.run(  # noqa: S603 - static argv, no shell
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--is-inside-work-tree"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if inside.returncode != 0:
        pytest.skip("not a git work tree")
    ignored = subprocess.run(  # noqa: S603 - static argv, no shell
        ["git", "-C", str(_REPO_ROOT), "check-ignore", "-q", stray],  # noqa: S607
        check=False,
    )
    assert ignored.returncode == 0, f"{stray} is not git-ignored"


def _ignore_file_of_each_compose_build() -> list[Path]:
    # BuildKit reads `<dockerfile>.dockerignore` instead of the root `.dockerignore`
    # whenever that file exists, so the root file alone does not describe every build.
    compose = _COMPOSE.read_text(encoding="utf-8")
    dockerfiles = re.findall(r"^\s*dockerfile:\s*(\S+)\s*$", compose, re.MULTILINE)
    specific = [_REPO_ROOT / f"{dockerfile}.dockerignore" for dockerfile in dockerfiles]
    return [
        path if path.is_file() else _REPO_ROOT / ".dockerignore" for path in specific
    ]


def test_every_compose_build_is_checked_for_its_ignore_file() -> None:
    ignore_files = _ignore_file_of_each_compose_build()
    # Scheduler and fund: a regex that matches nothing must not pass vacuously.
    assert len(ignore_files) >= 2
    assert _REPO_ROOT / "fund" / "Dockerfile.dockerignore" in ignore_files


@pytest.mark.parametrize(
    "ignore_file",
    _ignore_file_of_each_compose_build(),
    ids=lambda path: path.relative_to(_REPO_ROOT).as_posix(),
)
def test_docker_build_context_excludes_every_secrets_directory(
    ignore_file: Path,
) -> None:
    # `COPY ingestion/` or `COPY fund/` would otherwise bake a stray `<dir>/secrets/*`
    # into the image, and the root `secrets/` would travel to the daemon as context.
    lines = ignore_file.read_text(encoding="utf-8").splitlines()
    # The trailing-slash spelling names the same directories; either one counts.
    assert {"**/secrets", "**/secrets/"} & {line.strip() for line in lines}
