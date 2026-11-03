"""Runtime lifecycle for portopt: start / stop / status (SPEC D6/D10).

`run_start` boots a stopped Docker runtime, decrypts `~/.portopt/secrets.enc`,
renders the compose secret files, and brings the stack up: the persistent ingestion
daemon always, plus the fund engine when an LLM backend is configured.
`run_stop` stops only the fund engine, leaving the scheduler + db running, and
`run_stop(all_services=True)` takes the whole stack down.
`run_status` reports Docker + service health.
"""

from __future__ import annotations

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
    printed. A running service whose mounted secret was rotated since the last
    render is restarted first, so it picks up the new key instead of keeping the one
    it read at startup.

    Args:
        passphrase: The master passphrase that decrypts the secret store.
        rebuild: Force an image rebuild before starting, so a start after a code or
            migration change does not reuse a stale image.
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
    # `up` recreates a service only when its compose config changes, and a
    # rewritten secret file is not one. Daemons read their keys once at startup,
    # so a rotated key would reach only new processes (the TUI, `fund run`) while
    # the scheduled jobs kept failing on the old one.
    if rotated:
        stale = compose_secrets.services_mounting(rotated)
        stale &= docker_bootstrap.running_services(repo=repo)
        if stale:
            docker_bootstrap.restart_services(sorted(stale), repo=repo)
    # Every service is profile-gated (a bare `up` starts nothing). The ingestion
    # profile (db + the scheduler daemon) always comes up: the scheduler must stay up
    # for the daily/macro/news jobs, so a `portopt start` must never leave the data
    # pipeline down. The fund profile joins it only with a usable LLM backend: its
    # worker builds the model before the first heartbeat, so without one the container
    # never passes its healthcheck and `up --wait` would block on it. Images build on
    # first run; `up --wait` blocks on each service's healthcheck.
    fund_enabled = llm_backend.is_configured(config, secrets)
    docker_bootstrap.build_and_up(
        profiles=("fund", "ingestion") if fund_enabled else ("ingestion",),
        force_build=rebuild,
        repo=repo,
    )
    if not fund_enabled:
        print(FUND_SKIPPED_NOTICE)


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


def run_status() -> dict[str, bool]:
    """Report Docker + service health as a name -> ok mapping."""
    docker_ok = docker_bootstrap.docker_available()
    running = (
        docker_bootstrap.running_services(repo=path_install.locate_repo())
        if docker_ok
        else set()
    )
    return {
        "docker": docker_ok,
        "db": "db" in running,
        "scheduler": "scheduler" in running,
        "fund": "fund" in running,
    }
