"""Runtime lifecycle for portopt: start / stop / status (SPEC D6/D10).

`run_start` decrypts `~/.portopt/secrets.enc`, renders the compose secret files,
and brings the Docker stack up. `run_stop` tears it down and wipes the plaintext
secret files. `run_status` reports Docker + service health.
"""

from __future__ import annotations

from app.setup import compose_secrets, docker_bootstrap, secret_store


class LifecycleError(RuntimeError):
    """Raised when a lifecycle command cannot proceed."""


def run_start(passphrase: str) -> None:
    """Decrypt secrets, render compose secret files, and bring the stack up."""
    if not passphrase:
        raise LifecycleError(
            "A master passphrase is required (set PORTOPT_PASSPHRASE)."
        )
    docker_bootstrap.check_docker()
    secrets = secret_store.load_secrets(passphrase)
    compose_secrets.render(secrets)
    docker_bootstrap.compose_up()


def run_stop() -> None:
    """Stop the stack and remove the rendered plaintext secret files."""
    docker_bootstrap.compose_down()
    compose_secrets.cleanup()


def run_status() -> dict[str, bool]:
    """Report Docker + service health as a name -> ok mapping."""
    docker_ok = docker_bootstrap.docker_available()
    running = docker_bootstrap.running_services() if docker_ok else set()
    return {
        "docker": docker_ok,
        "db": "db" in running,
        "scheduler": "scheduler" in running,
    }
