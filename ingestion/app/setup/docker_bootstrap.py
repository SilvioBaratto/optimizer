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


def compose_up() -> None:
    """Bring up all services in the background (`portopt start`)."""
    result = _run(["docker", "compose", "up", "-d"])
    if result.returncode != 0:
        raise DockerError(f"`docker compose up` failed:\n{result.stderr}")


def compose_down() -> None:
    """Stop and remove all services (`portopt stop`)."""
    result = _run(["docker", "compose", "down"])
    if result.returncode != 0:
        raise DockerError(f"`docker compose down` failed:\n{result.stderr}")


def running_services() -> set[str]:
    """Return the set of compose services currently running (empty on error)."""
    result = _run(
        ["docker", "compose", "ps", "--services", "--filter", "status=running"]
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def docker_available() -> bool:
    """True if Docker + the compose plugin are usable (never raises)."""
    try:
        check_docker()
    except DockerError:
        return False
    return True
