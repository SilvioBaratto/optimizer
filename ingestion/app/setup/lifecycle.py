"""Runtime lifecycle for portopt: start / stop / status (SPEC D6/D10).

`run_start` decrypts `~/.portopt/secrets.enc`, renders the compose secret files,
and brings the full stack up (the fund engine + the persistent ingestion daemon).
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
    path_install,
    secret_store,
)


class LifecycleError(RuntimeError):
    """Raised when a lifecycle command cannot proceed."""


def run_start(passphrase: str, *, rebuild: bool = False) -> None:
    """Decrypt secrets, render compose secret files, and bring the full stack up.

    Brings up both the fund decision engine and the persistent ingestion daemon so
    the scheduler is never left down by a start. A running service whose mounted
    secret was rotated since the last render is restarted first, so it picks up
    the new key instead of keeping the one it read at startup.

    Args:
        passphrase: The master passphrase that decrypts the secret store.
        rebuild: Force an image rebuild before starting, so a start after a code or
            migration change does not reuse a stale image.
    """
    if not passphrase:
        raise LifecycleError(
            "A master passphrase is required (set PORTOPT_PASSPHRASE)."
        )
    docker_bootstrap.check_docker()
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
    # Every service is profile-gated (a bare `up` starts nothing). Bring up BOTH
    # the fund profile (db + decision engine) AND the ingestion profile (the
    # scheduler daemon): the scheduler must stay up for the daily/macro/news jobs,
    # so a `portopt start` must never leave the data pipeline down. Images build on
    # first run; `up --wait` blocks on each service's healthcheck.
    docker_bootstrap.build_and_up(
        profiles=("fund", "ingestion"), force_build=rebuild, repo=repo
    )


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
