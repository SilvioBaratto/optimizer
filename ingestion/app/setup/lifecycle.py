"""Runtime lifecycle for portopt: start / stop / status (SPEC D6/D10).

`run_start` decrypts `~/.portopt/secrets.enc`, renders the compose secret files,
and brings the full stack up (the fund engine + the persistent ingestion daemon).
`run_stop` stops only the fund engine, leaving the scheduler + db running.
`run_status` reports Docker + service health.
"""

from __future__ import annotations

from app.setup import (
    compose_env,
    compose_secrets,
    config_file,
    docker_bootstrap,
    secret_store,
)


class LifecycleError(RuntimeError):
    """Raised when a lifecycle command cannot proceed."""


def run_start(passphrase: str) -> None:
    """Decrypt secrets, render compose secret files, and bring the full stack up.

    Brings up both the fund decision engine and the persistent ingestion daemon so
    the scheduler is never left down by a start.
    """
    if not passphrase:
        raise LifecycleError(
            "A master passphrase is required (set PORTOPT_PASSPHRASE)."
        )
    docker_bootstrap.check_docker()
    secrets = secret_store.load_secrets(passphrase)
    compose_secrets.render(secrets)
    # The fund container never reads config.toml — its non-secret LLM selection
    # reaches it only through the generated .env.fund env_file (the compose
    # `environment:` block omits these vars so nothing shadows the file).
    compose_env.render(config_file.load_config())
    # Every service is profile-gated (a bare `up` starts nothing). Bring up BOTH
    # the fund profile (db + decision engine) AND the ingestion profile (the
    # scheduler daemon): the scheduler must stay up for the daily/macro/news jobs,
    # so a `portopt start` must never leave the data pipeline down. Images build on
    # first run; `up --wait` blocks on each service's healthcheck.
    docker_bootstrap.build_and_up(profiles=("fund", "ingestion"))


def run_stop() -> None:
    """Stop the fund decision engine, leaving the ingestion daemon + db running.

    The scheduler is a persistent data daemon (daily pipeline, macro/news) that must
    stay up, so stopping the fund no longer tears down the whole stack. Only the
    fund's own ``.env.fund`` is removed; the shared ``./secrets`` files stay in place
    for the still-running scheduler (``portopt start`` re-renders them anyway).
    """
    docker_bootstrap.stop_services(["fund"])
    compose_env.cleanup()


def run_status() -> dict[str, bool]:
    """Report Docker + service health as a name -> ok mapping."""
    docker_ok = docker_bootstrap.docker_available()
    running = docker_bootstrap.running_services() if docker_ok else set()
    return {
        "docker": docker_ok,
        "db": "db" in running,
        "scheduler": "scheduler" in running,
        "fund": "fund" in running,
    }
