"""Compose-secret rendering contract (SPEC D6).

`render` writes every declared secret to `<secrets_dir>/<name>` (empty for
unconfigured ones so `docker compose up` never fails on a missing file) inside an
owner-only directory, with a mode the containers' non-root user can read;
`cleanup` removes them (all, or just the ones named).
"""

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import IO, Any

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


_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode not enforced on Windows"
)


@_POSIX_ONLY
def test_rendered_files_are_readable_by_the_container_user(tmp_path: Path) -> None:
    # Compose file secrets are plain bind mounts: the host owner and mode apply inside
    # the container, whose appuser (uid 1000) is not the host operator on Linux. A
    # 0600 file owned by another uid is unreadable there, so the worker crash-loops.
    cs.render({"fred_api_key": "x"}, secrets_dir=tmp_path / "secrets")
    assert (tmp_path / "secrets" / "fred_api_key").stat().st_mode & 0o777 == 0o644


@_POSIX_ONLY
def test_secrets_dir_is_owner_only_even_when_it_pre_existed_world_readable(
    tmp_path: Path,
) -> None:
    # The directory is what keeps other host users out once the files are readable by
    # the container; a git checkout creates it 0755.
    target = tmp_path / "secrets"
    target.mkdir()
    target.chmod(0o755)
    cs.render({"fred_api_key": "x"}, secrets_dir=target)
    assert target.stat().st_mode & 0o777 == 0o700


@_POSIX_ONLY
def test_files_stay_owner_only_when_the_dir_cannot_be_locked_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On filesystems where a directory chmod does not take effect, widening the files
    # would expose the keys to other users, so they fall back to 0600.
    target = tmp_path / "secrets"
    target.mkdir()
    target.chmod(0o755)
    real_chmod = os.chmod

    def _chmod_files_only(path: str | os.PathLike[str], mode: int, **kw: Any) -> None:
        if not Path(path).is_dir():
            real_chmod(path, mode, **kw)

    monkeypatch.setattr(os, "chmod", _chmod_files_only)
    cs.render({"fred_api_key": "x"}, secrets_dir=target)
    assert (target / "fred_api_key").stat().st_mode & 0o777 == 0o600


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


def test_render_requires_an_explicit_directory_and_never_uses_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A default of `./secrets` would drop plaintext keys into whatever directory the
    # operator happened to be in while compose mounted another checkout's secrets.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError):
        cs.render({"fred_api_key": "x"})  # type: ignore[call-arg]
    assert list(tmp_path.iterdir()) == []


@_POSIX_ONLY
def test_render_replaces_a_symlinked_secret_instead_of_writing_through_it(
    tmp_path: Path,
) -> None:
    # A planted symlink in a group-writable secrets/ must not redirect the write onto
    # whatever it points at.
    victim = tmp_path / "victim"
    victim.write_text("precious", encoding="utf-8")
    target = tmp_path / "secrets"
    target.mkdir()
    (target / "fred_api_key").symlink_to(victim)
    cs.render({"fred_api_key": "x"}, secrets_dir=target)
    assert victim.read_text(encoding="utf-8") == "precious"
    assert not (target / "fred_api_key").is_symlink()
    assert (target / "fred_api_key").read_text(encoding="utf-8") == "x"


@_POSIX_ONLY
def test_files_are_created_with_their_final_mode_not_tightened_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With chmod disabled the final mode is the creation mode, so a file that is only
    # tightened after the write is readable by other users in between. The directory
    # cannot be locked down here, so the files must come out owner-only.
    target = tmp_path / "secrets"
    target.mkdir()
    target.chmod(0o755)
    monkeypatch.setattr(os, "chmod", lambda *_a, **_kw: None)
    previous = os.umask(0)
    try:
        cs.render({"fred_api_key": "x"}, secrets_dir=target)
    finally:
        os.umask(previous)
    assert (target / "fred_api_key").stat().st_mode & 0o777 == 0o600


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


@_POSIX_ONLY
@pytest.mark.parametrize(("dir_is_gate", "final_mode"), [(True, 0o644), (False, 0o600)])
def test_a_looser_leftover_file_gets_its_final_mode_before_the_value_lands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dir_is_gate: bool,
    final_mode: int,
) -> None:
    leftover = tmp_path / "fred_api_key"
    leftover.write_text("stale", encoding="utf-8")
    leftover.chmod(0o666)
    monkeypatch.setattr(cs, "_restrict_dir", lambda _target: dir_is_gate)

    # The final mode cannot tell settle-then-write from write-then-settle, so sample
    # the mode of the open file at the moment each value is written.
    modes_at_write: list[int] = []
    real_fdopen = os.fdopen

    class _SpyHandle:
        def __init__(self, handle: IO[str]) -> None:
            self._handle = handle

        def __enter__(self) -> "_SpyHandle":
            self._handle.__enter__()
            return self

        def __exit__(self, *exc: Any) -> None:
            self._handle.__exit__(*exc)

        def write(self, data: str) -> int:
            modes_at_write.append(os.fstat(self._handle.fileno()).st_mode & 0o777)
            return self._handle.write(data)

    monkeypatch.setattr(os, "fdopen", lambda *a, **k: _SpyHandle(real_fdopen(*a, **k)))
    cs.render({"fred_api_key": "fresh"}, secrets_dir=tmp_path)

    assert len(modes_at_write) == len(cs.SECRET_NAMES)
    # Leftover is 0666; a freshly created file gets the creation mode.
    assert set(modes_at_write) == {final_mode}
    assert leftover.read_text(encoding="utf-8") == "fresh"


def _deny_dir_chmod(monkeypatch: pytest.MonkeyPatch, denied: Path) -> None:
    """Make `denied.chmod` fail like a directory owned by another host user."""
    real_chmod = Path.chmod

    def _chmod(self: Path, *args: object, **kwargs: object) -> None:
        if self == denied:
            raise PermissionError(1, "Operation not permitted", str(self))
        real_chmod(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "chmod", _chmod)


@_POSIX_ONLY
def test_unlockable_open_directory_is_reported_to_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "secrets"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    _deny_dir_chmod(monkeypatch, target)
    with caplog.at_level(logging.WARNING):
        cs.render({"fred_api_key": "x"}, secrets_dir=target)
    assert str(target) in caplog.text


@_POSIX_ONLY
def test_files_stay_readable_when_the_directory_is_already_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A failed chmod is harmless when the directory is already closed to other
    # users, so the container-readable mode must not be downgraded needlessly.
    target = tmp_path / "secrets"
    target.mkdir(mode=0o700)
    target.chmod(0o700)
    _deny_dir_chmod(monkeypatch, target)
    with caplog.at_level(logging.WARNING):
        cs.render({"fred_api_key": "x"}, secrets_dir=target)
    assert (target / "fred_api_key").stat().st_mode & 0o777 == 0o644
    assert caplog.text == ""
