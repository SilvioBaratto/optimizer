"""Docker + database bootstrap for the portopt install wizard (SPEC D4/D11/D12).

Cross-platform Docker verification (works under Docker Desktop and Linux
Engine), then bring up Postgres and run ``alembic upgrade head`` (migrate-only —
no data seeding). All commands are static argv lists run without a shell.
"""

from __future__ import annotations

import os
import re
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


# The stack's Dockerfiles use `--mount=type=secret` (BuildKit-only), and the
# helpers below lean on `up --wait`; below these, tooling degrades but still works.
_ENGINE_MIN = (23, 0)
_COMPOSE_MIN = (2, 1, 1)

# Images each profile builds (db/adminer are pulled, never built), so `build_and_up`
# knows what "absent" means before deciding to run the slow build.
_BUILT_IMAGES = {
    "fund": ("optimizer-fund:latest",),
    "ingestion": ("optimizer-scheduler:latest",),
}


def _parse_version(text: str) -> tuple[int, ...] | None:
    """Extract the first ``major.minor[.patch]`` triple from version output."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def _fmt_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def _server_version() -> tuple[int, ...] | None:
    """Docker Engine (server) version, or None if the daemon can't report it."""
    result = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    if result.returncode != 0:
        return None
    return _parse_version(result.stdout)


def _compose_version() -> tuple[int, ...] | None:
    """Docker Compose plugin version, or None if undetectable."""
    result = _run(["docker", "compose", "version", "--short"])
    if result.returncode != 0:
        return None
    return _parse_version(result.stdout)


def warn_stale_versions() -> list[str]:
    """Return advisory warnings for Docker/Compose below the versions this stack
    prefers.

    Never raises — old tooling still brings the stack up, just with caveats
    (legacy builder can't read the secret mount; `up --wait` may be a no-op). An
    undetectable version yields no warning (no false alarms). Setting
    ``DOCKER_BUILDKIT=1`` is the escape hatch that suppresses the Engine warning:
    it forces BuildKit on an older Engine so the secret-mount builds still work.
    """
    warnings: list[str] = []
    engine = _server_version()
    if (
        engine is not None
        and engine < _ENGINE_MIN
        and os.environ.get("DOCKER_BUILDKIT") != "1"
    ):
        warnings.append(
            f"Docker Engine {_fmt_version(engine)} < {_fmt_version(_ENGINE_MIN)}: "
            "BuildKit may be off, but the Dockerfiles use `--mount=type=secret` "
            "(BuildKit-only). Set DOCKER_BUILDKIT=1 or upgrade Docker."
        )
    compose = _compose_version()
    if compose is not None and compose < _COMPOSE_MIN:
        warnings.append(
            f"Docker Compose {_fmt_version(compose)} < {_fmt_version(_COMPOSE_MIN)}: "
            "`up --wait` may be unsupported; the stack still starts but readiness "
            "isn't gated."
        )
    return warnings


def _image_exists(ref: str) -> bool:
    """True if a local image with `ref` is present (`docker image inspect`)."""
    return _run(["docker", "image", "inspect", ref]).returncode == 0


def build_and_up(*, profile: str = "fund", wait_timeout: int = 300) -> list[str]:
    """Build the profile's images when absent, then bring the stack up and wait.

    Surfaces (prints and returns) any stale-version warnings, builds with
    ``--pull`` only when one of the profile's built images is missing — the slow
    first run; later starts reuse the images — then ``up -d --wait`` so the caller
    blocks until every service reports healthy.

    Args:
        profile: The compose profile to build and start (default ``"fund"`` =
            db + fund).
        wait_timeout: Seconds ``up --wait`` waits for health before failing.

    Returns:
        The advisory version warnings surfaced (empty when tooling is current).

    Raises:
        DockerError: If the build or the ``up --wait`` command fails.
    """
    warnings = warn_stale_versions()
    for warning in warnings:
        print(f"warning: {warning}")

    if any(not _image_exists(ref) for ref in _BUILT_IMAGES.get(profile, ())):
        print("Building images — slow the first time; later starts reuse them.")
        build = _run(
            [
                "docker",
                "compose",
                "--profile",
                profile,
                "--progress",
                "plain",
                "build",
                "--pull",
            ]
        )
        if build.returncode != 0:
            raise DockerError(f"`docker compose build` failed:\n{build.stderr}")

    up = _run(
        [
            "docker",
            "compose",
            "--profile",
            profile,
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            str(wait_timeout),
        ]
    )
    if up.returncode != 0:
        raise DockerError(f"`docker compose up --wait` failed:\n{up.stderr}")
    return warnings
