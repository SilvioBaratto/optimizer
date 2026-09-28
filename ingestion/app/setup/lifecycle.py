"""Runtime lifecycle for portopt: start / stop / status (SPEC D6/D10).

`run_start` decrypts `~/.portopt/secrets.enc`, renders the compose secret files,
and brings the Docker stack up. `run_stop` tears it down and wipes the plaintext
secret files. `run_status` reports Docker + service health.
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
    """Decrypt secrets, render compose secret files, and bring the fund stack up."""
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
    # Every service is profile-gated (a bare `up` starts nothing), so target the
    # fund profile (db + fund), building the image on first run and waiting for
    # the alembic-at-head healthcheck.
    docker_bootstrap.build_and_up(profile="fund")


def run_stop() -> None:
    """Stop the stack and remove the rendered plaintext secret files."""
    docker_bootstrap.compose_down()
    compose_secrets.cleanup()
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
