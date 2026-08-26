"""Docker + database bootstrap for the portopt install wizard (SPEC D4/D11/D12).

Cross-platform Docker verification (works under Docker Desktop and Linux
Engine), then bring up Postgres and run ``alembic upgrade head`` (migrate-only —
no data seeding). All commands are static argv lists run without a shell.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


class DockerError(RuntimeError):
    """Raised when Docker is unavailable or a bootstrap command fails."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - static, trusted argv; never shell=True
        cmd, capture_output=True, text=True, check=False
    )


def _install_hint() -> str:
    if sys.platform in ("win32", "darwin"):
        return "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    return (
        "Install Docker Engine + the compose plugin: "
        "https://docs.docker.com/engine/install/"
    )


def _start_hint() -> str:
    if sys.platform in ("win32", "darwin"):
        return "Start Docker Desktop and retry."
    return "Start the Docker daemon (e.g. `sudo systemctl start docker`) and retry."


def check_docker() -> None:
    """Verify the Docker CLI, a reachable daemon, and the compose v2 plugin."""
    if shutil.which("docker") is None:
        raise DockerError(f"Docker CLI not found on PATH. {_install_hint()}")
    if _run(["docker", "info"]).returncode != 0:
        raise DockerError(f"Docker daemon not reachable. {_start_hint()}")
    if _run(["docker", "compose", "version"]).returncode != 0:
        raise DockerError(f"`docker compose` (v2) not available. {_install_hint()}")


def bring_up_db() -> None:
    """Start the Postgres service and wait for it to become healthy."""
    result = _run(["docker", "compose", "up", "-d", "--wait", "db"])
    if result.returncode != 0:
        raise DockerError(f"Failed to start the db service:\n{result.stderr}")


def migrate() -> None:
    """Run `alembic upgrade head` (migrate-only — no data seeding)."""
    result = _run(["alembic", "upgrade", "head"])
    if result.returncode != 0:
        raise DockerError(f"`alembic upgrade head` failed:\n{result.stderr}")
