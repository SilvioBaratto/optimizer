"""Runtime lifecycle for portopt: start / stop / status (SPEC D6/D10).

`run_start` boots a stopped Docker runtime, decrypts `~/.portopt/secrets.enc`,
renders the compose secret files, and brings the stack up: the persistent ingestion
daemon always, plus the fund engine when an LLM backend is configured.
`run_stop` stops only the fund engine, leaving the scheduler + db running, and
`run_stop(all_services=True)` takes the whole stack down.
`run_status` reports Docker + service health.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.setup import (
    compose_env,
    compose_secrets,
    config_file,
    docker_bootstrap,
    llm_backend,
    path_install,
    secret_store,
)

FUND_SKIPPED_NOTICE = (
    "fund decision engine not started: no LLM backend is configured "
    "(the data pipeline is running). "
    "Run `portopt setup --reconfigure` to configure one, then `portopt start`."
)


# Directory the fund service bind-mounts for its TUI exports (docker-compose.yml).
EXPORTS_DIRNAME = "exports"
# The non-root user fund/Dockerfile runs the container as.
_CONTAINER_UID = 1000

STATUS_OK = "ok"
STATUS_DOWN = "down"
STATUS_DISABLED = "disabled"
FUND_DISABLED_HINT = (
    "fund is off by design when no LLM backend is configured or after "
    "`portopt stop`; `portopt start` brings it up once one is."
)


class LifecycleError(RuntimeError):
    """Raised when a lifecycle command cannot proceed."""


def run_start(passphrase: str, *, rebuild: bool = False) -> None:
    """Decrypt secrets, render compose secret files, and bring the stack up.

    Boots the local Docker runtime first when its daemon is down (Colima, OrbStack,
    Docker Desktop), so a start works from a cold machine. Always brings up the
    persistent ingestion daemon so the scheduler is never left down by a start. The
    fund decision engine comes up with it only when :func:`llm_backend.is_configured`
    says its worker can build a model; otherwise its container would restart forever
    without turning healthy, so it is left off and ``FUND_SKIPPED_NOTICE`` is
    printed. A fund container still running from an earlier start is then stopped,
    because ``up`` never touches a service outside the requested profiles and it
    would keep serving the provider the operator just configured away (unless a
    fund round is running in it, which is left to finish). A running service whose
    mounted secret was rotated since the last render is restarted first, so it picks
    up the new key instead of keeping the one it read at startup.

    Args:
        passphrase: The master passphrase that decrypts the secret store.
        rebuild: Force an image rebuild before starting, so a start after a base
            image refresh does not reuse a stale layer. Images whose sources changed
            are rebuilt without it.
    """
    if not passphrase:
        raise LifecycleError(
            "A master passphrase is required (set PORTOPT_PASSPHRASE)."
        )
    docker_bootstrap.ensure_docker()
    # Resolved before anything is decrypted: compose mounts <checkout>/secrets/*, so
    # plaintext rendered into the cwd would both miss the mounts and litter a random
    # directory with live keys.
    repo = docker_bootstrap.compose_dir()
    secrets = secret_store.load_secrets(passphrase)
    secrets_dir = repo / compose_secrets.DEFAULT_SECRETS_DIR
    rotated = compose_secrets.changed_names(secrets, secrets_dir=secrets_dir)
    compose_secrets.render(secrets, secrets_dir=secrets_dir)
    # Containers never read config.toml — the fund's non-secret LLM selection
    # reaches it only through the generated .env.fund env_file (the compose
    # `environment:` block omits these vars so nothing shadows the file), and the
    # scheduler's Trading212 mode through .env.scheduler.
    config = config_file.load_config()
    compose_env.render(config, path=repo / compose_env.DEFAULT_ENV_FUND_PATH)
    compose_env.render_scheduler(
        config, path=repo / compose_env.DEFAULT_ENV_SCHEDULER_PATH
    )
    _ensure_exports_dir(repo)
    # Every service is profile-gated (a bare `up` starts nothing). The ingestion
    # profile (db + the scheduler daemon) always comes up: the scheduler must stay up
    # for the daily/macro/news jobs, so a `portopt start` must never leave the data
    # pipeline down. The fund profile joins it only with a usable LLM backend: its
    # worker builds the model before the first heartbeat, so without one the container
    # never passes its healthcheck and `up --wait` would block on it. Images build on
    # first run; `up --wait` blocks on each service's healthcheck.
    fund_enabled = llm_backend.is_configured(config, secrets)
    # `up` recreates a service only when its compose config changes, and a
    # rewritten secret file is not one. Daemons read their keys once at startup,
    # so a rotated key would reach only new processes (the TUI, `fund run`) while
    # the scheduled jobs kept failing on the old one. A `restart` keeps the
    # container's old env_file values, so a fund about to be left off is not
    # restarted.
    if rotated:
        stale = compose_secrets.services_mounting(rotated)
        stale &= docker_bootstrap.running_services(repo=repo)
        if not fund_enabled:
            stale.discard("fund")
        if stale:
            docker_bootstrap.restart_services(sorted(stale), repo=repo)
    docker_bootstrap.build_and_up(
        profiles=("fund", "ingestion") if fund_enabled else ("ingestion",),
        force_build=rebuild,
        repo=repo,
    )
    if not fund_enabled:
        print(FUND_SKIPPED_NOTICE)
        _stop_leftover_fund(repo)


def _ensure_exports_dir(repo: Path) -> None:
    """Create the fund's export directory as the invoking user, writable by the fund.

    Compose creates a missing bind-mount source as root, after which the fund
    container's non-root user cannot write its TUI exports there. On a Linux host the
    mount keeps the host's numeric owner, so the container runs as the invoking user
    (``docker_bootstrap.fund_user_env``) and a directory that user owns is writable as
    it is. For a host that stays on the image's user (uid 1000 or root) a directory
    owned by someone else and not world-writable is opened up (sticky, like ``/tmp``,
    so the container's files stay its own); if the invoking user cannot change it, for
    example because root created it, the command that does is printed. Docker Desktop
    and Colima present host files as the container user and need nothing.
    """
    exports = repo / EXPORTS_DIRNAME
    exports.mkdir(exist_ok=True)
    if sys.platform != "linux":
        return
    host_user = docker_bootstrap.fund_user_env().get("PORTOPT_FUND_UID")
    container_uid = int(host_user) if host_user else _CONTAINER_UID
    info = exports.stat()
    if info.st_uid == container_uid or info.st_mode & 0o002:
        return
    try:
        exports.chmod(0o1777)
    except OSError:
        print(
            f"warning: {exports} is owned by uid {info.st_uid}, but the fund container "
            f"writes as uid {container_uid}, so TUI exports will fail with "
            f"'permission denied'. Fix it with `sudo chown {container_uid} {exports}` "
            f"(or `sudo chmod o+w {exports}`)."
        )


def _stop_leftover_fund(repo: Path) -> None:
    """Stop a fund container from an earlier start, unless it is mid-round."""
    if "fund" not in docker_bootstrap.running_services(repo=repo):
        return
    drivers = docker_bootstrap.fund_drivers(repo=repo)
    if drivers:
        running = "\n  ".join(drivers)
        print(
            "warning: the fund container is still running with its earlier "
            "configuration because these are running in it:\n"
            f"  {running}\n"
            "Run `portopt stop` once they finish."
        )
        return
    docker_bootstrap.stop_services(["fund"], repo=repo)


def run_stop(*, all_services: bool = False, force: bool = False) -> None:
    """Stop the fund decision engine, leaving the ingestion daemon + db running.

    The scheduler is a persistent data daemon (daily pipeline, macro/news) that must
    stay up, so stopping the fund no longer tears down the whole stack. The fund's
    own ``.env.fund`` and the LLM keys only it mounts are wiped; the three scheduler
    secrets and the scheduler's ``.env.scheduler`` stay in place because the daemon
    restarts on its own and needs them (``portopt start`` re-renders everything
    anyway), so a later plain ``docker compose up`` recreates it with the same
    credentials and mode.

    Args:
        all_services: Also stop the scheduler, adminer and db, then wipe every rendered
            plaintext secret, so no key outlives the stack.
        force: Stop even while a fund round, approval or TUI runs in the fund
            container; it is killed mid-flight.

    Raises:
        LifecycleError: A fund CLI/TUI process is running in the fund container and
            ``force`` is not set; nothing has been stopped or wiped.
    """
    repo = docker_bootstrap.compose_dir()
    if not force:
        drivers = docker_bootstrap.fund_drivers(repo=repo)
        if drivers:
            running = "\n  ".join(drivers)
            raise LifecycleError(
                "the fund container is still running these, and stopping it would "
                f"kill them mid-flight:\n  {running}\n"
                "Let them finish, or rerun with --force to stop anyway."
            )
    # `compose stop db` leaves its dependents running, so adminer (a db client that
    # `portopt start` also launches) is named explicitly, and db goes last.
    docker_bootstrap.stop_services(
        ["fund", "scheduler", "adminer", "db"] if all_services else ["fund"],
        repo=repo,
    )
    compose_secrets.cleanup(
        secrets_dir=repo / compose_secrets.DEFAULT_SECRETS_DIR,
        names=None if all_services else compose_secrets.FUND_ONLY_SECRET_NAMES,
    )
    compose_env.cleanup(path=repo / compose_env.DEFAULT_ENV_FUND_PATH)


def cleanup_rendered() -> None:
    """Remove every rendered plaintext secret/env file from the checkout.

    Best effort and silent when no checkout is found: nothing could have been
    rendered there, so there is nothing to wipe.
    """
    repo = path_install.locate_repo()
    if repo is None:
        return
    compose_secrets.cleanup(secrets_dir=repo / compose_secrets.DEFAULT_SECRETS_DIR)
    compose_env.cleanup(path=repo / compose_env.DEFAULT_ENV_FUND_PATH)


def run_status() -> dict[str, str]:
    """Report Docker + service health as a name -> status mapping.

    Each service is judged by its container's state and healthcheck, not by whether
    it is merely listed as running: a crash-looping or hung container is not up.

    Returns:
        ``docker``, ``db``, ``scheduler`` and ``fund`` mapped to ``"ok"``,
        ``"down"`` (not running, or Docker unreachable), ``"starting"``,
        ``"unhealthy"`` or ``"restarting"``. ``fund`` is ``"disabled"`` instead of
        ``"down"`` when it has no running container: without an LLM backend, or after
        ``portopt stop``, it is off on purpose.
    """
    if not docker_bootstrap.docker_available():
        return {
            "docker": STATUS_DOWN,
            "db": STATUS_DOWN,
            "scheduler": STATUS_DOWN,
            "fund": STATUS_DOWN,
        }
    states = docker_bootstrap.service_states(repo=path_install.locate_repo())
    fund = states.get("fund")
    fund_status = (
        STATUS_DISABLED
        if fund is None or fund.state in ("exited", "created")
        else _service_status(fund)
    )
    return {
        "docker": STATUS_OK,
        "db": _service_status(states.get("db")),
        "scheduler": _service_status(states.get("scheduler")),
        "fund": fund_status,
    }


def _service_status(state: docker_bootstrap.ServiceState | None) -> str:
    if state is None:
        return STATUS_DOWN
    if state.state == "running":
        return {"unhealthy": "unhealthy", "starting": "starting"}.get(
            state.health, STATUS_OK
        )
    if state.state == "restarting":
        return "restarting"
    return STATUS_DOWN


def status_ok(report: dict[str, str]) -> bool:
    """Return whether every entry of a :func:`run_status` report is fine.

    A service that is ``disabled`` by design does not count against the stack.
    """
    return all(status in (STATUS_OK, STATUS_DISABLED) for status in report.values())
