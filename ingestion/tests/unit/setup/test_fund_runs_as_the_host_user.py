"""The fund container writes `./exports` as the host user on a Linux host.

A bind mount keeps the host's numeric owner on Linux, so the image's user (uid 1000)
cannot write a directory owned by any other uid, and what it does write is unreadable
to the host user. Running the container as the invoking user, through an interpolated
`user:`, fixes both; macOS and a uid-1000 host keep the image's default, and root
never becomes the container user.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.setup import docker_bootstrap as db

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root.
_COMPOSE = Path(__file__).resolve().parents[4] / "docker-compose.yml"


def test_the_fund_service_runs_as_an_interpolated_user() -> None:
    """Compose takes the ids from the environment and defaults to the image's 1000."""
    fund = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]["fund"]
    assert fund["user"] == "${PORTOPT_FUND_UID:-1000}:${PORTOPT_FUND_GID:-1000}"


def _env(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    *,
    platform: str,
    uid: int,
    gid: int = 4343,
) -> dict[str, str] | None:
    monkeypatch.setattr(db.sys, "platform", platform)
    monkeypatch.setattr(db.os, "getuid", lambda: uid, raising=False)
    monkeypatch.setattr(db.os, "getgid", lambda: gid, raising=False)
    monkeypatch.delenv("PORTOPT_FUND_UID", raising=False)
    monkeypatch.delenv("PORTOPT_FUND_GID", raising=False)
    monkeypatch.delenv("CA_BUNDLE_FILE", raising=False)
    return db._compose_env(repo)


def test_a_linux_host_user_is_handed_to_compose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Linux host uid other than 0 and 1000 reaches compose's interpolation."""
    env = _env(monkeypatch, tmp_path, platform="linux", uid=4242, gid=4343)
    assert env is not None
    assert env["PORTOPT_FUND_UID"] == "4242"
    assert env["PORTOPT_FUND_GID"] == "4343"


@pytest.mark.parametrize(
    ("platform", "uid"),
    [
        (
            "darwin",
            501,
        ),  # Docker Desktop and Colima present files as the container user
        ("linux", 1000),  # already the image's user
        ("linux", 0),  # never run the container as root because the CLI was
        ("win32", 1000),
    ],
)
def test_no_other_host_changes_the_container_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, uid: int
) -> None:
    """Only a non-root, non-1000 Linux host overrides the image's user."""
    env = _env(monkeypatch, tmp_path, platform=platform, uid=uid)
    assert env is None or "PORTOPT_FUND_UID" not in env


def test_an_operator_supplied_user_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A uid the operator exported is never replaced by the invoking user's."""
    _env(monkeypatch, tmp_path, platform="linux", uid=4242)
    monkeypatch.setenv("PORTOPT_FUND_UID", "7777")
    env = db._compose_env(tmp_path)
    assert env is None or env["PORTOPT_FUND_UID"] == "7777"


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_every_compose_call_gets_the_user(
    mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Not only `up`: a later recreate must not silently fall back to uid 1000."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(db.sys, "platform", "linux")
    monkeypatch.setattr(db.os, "getuid", lambda: 4242, raising=False)
    monkeypatch.setattr(db.os, "getgid", lambda: 4343, raising=False)
    monkeypatch.delenv("PORTOPT_FUND_UID", raising=False)
    monkeypatch.delenv("PORTOPT_FUND_GID", raising=False)

    db._compose("ps", repo=tmp_path)

    assert mock_run.call_args.kwargs["env"]["PORTOPT_FUND_UID"] == "4242"
