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
from collections.abc import Sequence
from pathlib import Path


class DockerError(RuntimeError):
    """Raised when Docker is unavailable or a bootstrap command fails."""


def _run(
    cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - static, trusted argv; never shell=True
        cmd, capture_output=True, text=True, check=False, cwd=cwd, env=env
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


_HOST_DB_URL = "postgresql://postgres:postgres@localhost:54320/optimizer_db"


def _find_db_package() -> Path | None:
    """Locate ``packages/portopt-db`` — it owns alembic.ini + the migration tree.

    Only the ``setup.*`` front doors export ``OPTIMIZER_REPO``; a direct
    ``portopt setup`` does not, so fall back to walking up from the current dir.
    """
    bases: list[Path] = []
    repo = os.environ.get("OPTIMIZER_REPO")
    if repo:
        bases.append(Path(repo))
    cwd = Path.cwd()
    bases.extend([cwd, *cwd.parents])
    for base in bases:
        candidate = base / "packages" / "portopt-db"
        if (candidate / "alembic.ini").is_file():
            return candidate
    return None


def migrate() -> None:
    """Run ``alembic upgrade head`` from packages/portopt-db (migrate-only).

    Best-effort. alembic's ``script_location`` is relative (it must run from the
    package dir), env.py imports ``portopt_db`` and refuses to run without
    ``DATABASE_URL``, and a bare ``alembic`` on PATH may belong to an unrelated
    environment (or be absent). So run it as ``sys.executable -m alembic`` — the
    interpreter running ``portopt``, which has ``portopt_db`` + alembic — from the
    package dir with the host DB URL injected. On any failure (missing dir, alembic
    not importable, or a non-zero exit) warn and continue rather than aborting
    setup: the fund container runs the same migration on start and its healthcheck
    gates on it, so the DB reaches head there regardless.
    """
    db_dir = _find_db_package()
    if db_dir is None:
        print(
            "warning: packages/portopt-db/alembic.ini not found; skipping host-side "
            "migration — the fund container migrates on start."
        )
        return
    env = {**os.environ}
    env.setdefault("DATABASE_URL", _HOST_DB_URL)
    try:
        result = _run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(db_dir),
            env=env,
        )
    except OSError as exc:
        print(
            f"warning: could not run host-side migration ({exc}) — the fund "
            "container migrates on start."
        )
        return
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        print(
            "warning: host-side `alembic upgrade head` did not complete — the fund "
            f"container will migrate on start. Detail: {detail}"
        )


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


# The stack's Dockerfiles use `--mount=type=secret` (BuildKit-only); below the
# Engine floor BuildKit may be off. The Compose floor is a HARD parse floor, not a
# soft `up --wait` degradation: docker-compose.yml uses the long-form `env_file`
# `required:` mapping, which only parses on Compose >= 2.24 (Jan 2024) — older
# Compose fails `docker compose config`/`up` outright.
_ENGINE_MIN = (23, 0)
_COMPOSE_MIN = (2, 24, 0)

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
    needs.

    Never raises, but the two warnings differ in severity. The Engine one is
    advisory: an old builder can't read the secret mount, and ``DOCKER_BUILDKIT=1``
    is the escape hatch that suppresses it (it forces BuildKit on an older Engine
    so the secret-mount builds still work). The Compose one is a HARD blocker:
    docker-compose.yml uses the long-form ``env_file`` ``required:`` mapping, which
    only parses on Compose >= 2.24 — below that ``docker compose config``/``up``
    fails outright and nothing starts, so the warning is a heads-up before that
    hard failure, not a soft ``up --wait`` degradation. An undetectable version
    yields no warning (no false alarms).
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
            "docker-compose.yml uses the long-form `env_file` `required:` syntax "
            "(Compose >= 2.24); older Compose can't parse it — upgrade Docker "
            "Compose."
        )
    return warnings


def _image_exists(ref: str) -> bool:
    """True if a local image with `ref` is present (`docker image inspect`)."""
    return _run(["docker", "image", "inspect", ref]).returncode == 0


def stop_services(services: Sequence[str]) -> None:
    """Stop the named compose services, leaving the rest of the stack running.

    Unlike ``docker compose down`` (which removes every project container), this
    targets services by name so ``portopt stop`` can halt the fund decision engine
    without taking down the persistent ingestion daemon or the shared db.
    """
    result = _run(["docker", "compose", "stop", *services])
    if result.returncode != 0:
        raise DockerError(f"`docker compose stop` failed:\n{result.stderr}")


def build_and_up(
    *, profiles: Sequence[str] = ("fund",), wait_timeout: int = 300
) -> list[str]:
    """Build each profile's images when absent, then bring the stack up and wait.

    Surfaces (prints and returns) any stale-version warnings, builds with
    ``--pull`` only when one of the profiles' built images is missing — the slow
    first run; later starts reuse the images — then ``up -d --wait`` so the caller
    blocks until every service reports healthy.

    Args:
        profiles: The compose profiles to build and start (default ``("fund",)``;
            ``portopt start`` passes ``("fund", "ingestion")`` so the persistent
            data daemon comes up alongside the decision engine).
        wait_timeout: Seconds ``up --wait`` waits for health before failing.

    Returns:
        The advisory version warnings surfaced (empty when tooling is current).

    Raises:
        DockerError: If the build or the ``up --wait`` command fails.
    """
    warnings = warn_stale_versions()
    for warning in warnings:
        print(f"warning: {warning}")

    profile_args = [arg for profile in profiles for arg in ("--profile", profile)]
    built = [img for profile in profiles for img in _BUILT_IMAGES.get(profile, ())]
    if any(not _image_exists(ref) for ref in built):
        print("Building images — slow the first time; later starts reuse them.")
        build = _run(
            [
                "docker",
                "compose",
                *profile_args,
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
            *profile_args,
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
