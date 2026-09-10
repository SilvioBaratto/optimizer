"""Install-wizard orchestration for ``portopt setup`` (SPEC D4).

Two entry points share one persist+bootstrap core:
- ``run_setup_noninteractive`` — flags/env for CI; fails loud, never loops.
- ``run_setup_interactive`` — drives a `Prompter`.

Secrets are validated live *before* anything is written, so a failure at any
step leaves nothing persisted.
"""

from __future__ import annotations

import os

from app.setup import (
    config_file,
    docker_bootstrap,
    secret_store,
    validators,
)
from app.setup.prompts import Prompter

# Stable prompt messages (also used as keys by NonInteractivePrompter in tests).
_MSG_PASSPHRASE = "Master passphrase:"  # noqa: S105 - UI label, not a secret
_MSG_CONNECT_T212 = "Connect Trading212?"
_MSG_T212_KEY = "TRADING_212_API_KEY:"
_MSG_T212_SECRET = "TRADING_212_SECRET_KEY:"  # noqa: S105 - UI label, not a secret
_MSG_CONNECT_FRED = "Configure FRED (optional)?"
_MSG_FRED_KEY = "FRED_API_KEY:"


class SetupError(RuntimeError):
    """Raised when the wizard cannot complete (validation or config error)."""


def _persist_and_bootstrap(
    secrets: dict[str, str], config: dict[str, object], passphrase: str
) -> None:
    secret_store.save_secrets(secrets, passphrase)
    config_file.save_config(config)
    docker_bootstrap.bring_up_db()
    docker_bootstrap.migrate()


def run_setup_noninteractive(
    *,
    passphrase: str | None,
    t212_key: str | None = None,
    t212_secret: str | None = None,
    fred_key: str | None = None,
) -> None:
    """Non-interactive setup from flags/env — fails loud, persists nothing on error."""
    if not passphrase:
        raise SetupError("A master passphrase is required (set PORTOPT_PASSPHRASE).")
    docker_bootstrap.check_docker()

    secrets: dict[str, str] = {}
    config: dict[str, object] = {}

    if t212_key or t212_secret:
        if not (t212_key and t212_secret):
            raise SetupError("Trading212 needs both an API key and a secret key.")
        if not validators.validate_t212(t212_key, t212_secret):
            raise SetupError("Trading212 credentials failed validation.")
        secrets["trading_212_api_key"] = t212_key
        secrets["trading_212_secret_key"] = t212_secret

    if fred_key:
        if not validators.validate_fred(fred_key):
            raise SetupError("FRED API key failed validation.")
        secrets["fred_api_key"] = fred_key

    _persist_and_bootstrap(secrets, config, passphrase)


def run_setup_interactive(prompter: Prompter, *, passphrase: str | None = None) -> None:
    """Interactive setup via the prompt seam; each credential validates before persist."""
    docker_bootstrap.check_docker()

    pw = (
        passphrase
        or os.getenv("PORTOPT_PASSPHRASE")
        or prompter.password(_MSG_PASSPHRASE)
    )
    if not pw:
        raise SetupError("A master passphrase is required.")

    secrets: dict[str, str] = {}
    config: dict[str, object] = {}

    if prompter.confirm(_MSG_CONNECT_T212, default=False):
        # Rule 2: auto-detect exported env vars before prompting.
        t212_key = os.getenv("TRADING_212_API_KEY") or prompter.password(_MSG_T212_KEY)
        t212_secret = os.getenv("TRADING_212_SECRET_KEY") or prompter.password(
            _MSG_T212_SECRET
        )
        if not validators.validate_t212(t212_key, t212_secret):
            raise SetupError("Trading212 credentials failed validation.")
        secrets["trading_212_api_key"] = t212_key
        secrets["trading_212_secret_key"] = t212_secret

    if prompter.confirm(_MSG_CONNECT_FRED, default=False):
        fred_key = os.getenv("FRED_API_KEY") or prompter.password(_MSG_FRED_KEY)
        if not validators.validate_fred(fred_key):
            raise SetupError("FRED API key failed validation.")
        secrets["fred_api_key"] = fred_key

    _persist_and_bootstrap(secrets, config, pw)
