"""Install-wizard orchestration for ``portopt setup`` (SPEC D4).

Two entry points share one persist+bootstrap core:
- ``run_setup_noninteractive`` — flags/env for CI; fails loud, never loops.
- ``run_setup_interactive`` — drives a `Prompter`; the LLM step loops until a
  cloud key validates (mandatory, no local models).

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

_CLOUD_PROVIDERS = ("openai", "anthropic")

# Stable prompt messages (also used as keys by NonInteractivePrompter in tests).
_MSG_PASSPHRASE = "Master passphrase:"  # noqa: S105 - UI label, not a secret
_MSG_CONNECT_T212 = "Connect Trading212?"
_MSG_T212_KEY = "TRADING_212_API_KEY:"
_MSG_T212_SECRET = "TRADING_212_SECRET_KEY:"  # noqa: S105 - UI label, not a secret
_MSG_LLM_PROVIDER = "LLM provider:"
_MSG_LLM_KEY = "LLM API key:"
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
    llm_provider: str | None,
    llm_key: str | None,
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

    if llm_provider not in _CLOUD_PROVIDERS:
        raise SetupError(
            f"LLM provider must be one of {_CLOUD_PROVIDERS} (cloud only)."
        )
    if not llm_key or not validators.validate_llm(llm_provider, llm_key):
        raise SetupError(f"{llm_provider} API key failed validation.")
    config["llm_provider"] = llm_provider
    secrets[f"{llm_provider}_api_key"] = llm_key

    if fred_key:
        if not validators.validate_fred(fred_key):
            raise SetupError("FRED API key failed validation.")
        secrets["fred_api_key"] = fred_key

    _persist_and_bootstrap(secrets, config, passphrase)


def run_setup_interactive(prompter: Prompter, *, passphrase: str | None = None) -> None:
    """Interactive setup via the prompt seam; the LLM step loops until valid."""
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

    # Mandatory LLM gate — loop until a cloud key validates. The provider's env
    # var (e.g. OPENAI_API_KEY) is tried once before falling back to a prompt, so
    # an invalid env value cannot spin the loop forever.
    tried_env: set[str] = set()
    while True:
        provider = prompter.select(_MSG_LLM_PROVIDER, list(_CLOUD_PROVIDERS))
        env_key = os.getenv(f"{provider.upper()}_API_KEY")
        if env_key and provider not in tried_env:
            tried_env.add(provider)
            llm_key = env_key
        else:
            llm_key = prompter.password(_MSG_LLM_KEY)
        if validators.validate_llm(provider, llm_key):
            break
        prompter.error("Invalid provider/key — try again.")
    config["llm_provider"] = provider
    secrets[f"{provider}_api_key"] = llm_key

    if prompter.confirm(_MSG_CONNECT_FRED, default=False):
        fred_key = os.getenv("FRED_API_KEY") or prompter.password(_MSG_FRED_KEY)
        if not validators.validate_fred(fred_key):
            raise SetupError("FRED API key failed validation.")
        secrets["fred_api_key"] = fred_key

    _persist_and_bootstrap(secrets, config, pw)
