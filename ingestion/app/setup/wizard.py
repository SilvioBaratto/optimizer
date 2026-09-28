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
_MSG_CONFIGURE_LLM = "Configure the fund LLM backend?"
_MSG_LLM_PROVIDER = "LLM provider:"
_MSG_LLM_MODEL = "LLM model id:"
_MSG_LLM_BASE_URL = "LLM base URL (blank for the hosted default):"
_MSG_LLM_KEY = "LLM API key:"
_MSG_AWS_REGION = "AWS region:"
_MSG_AZURE_ENDPOINT = "Azure OpenAI endpoint:"
_MSG_AZURE_API_VERSION = "Azure OpenAI API version:"
_MSG_AZURE_DEPLOYMENT = "Azure OpenAI deployment name:"

# Provider slug -> compose secret name for its auth key (matches
# compose_secrets.SECRET_NAMES + fund.config._read_secret). Providers absent here
# are keyless: `aws` authenticates via the boto3 chain (IAM/region). The env var
# a key may also arrive in is the secret name upper-cased (e.g. OPENAI_API_KEY).
_LLM_SECRET_NAMES = {
    "ollama": "ollama_api_key",
    "openai": "openai_api_key",
    "openrouter": "openrouter_api_key",
    "anthropic": "anthropic_api_key",
    "google": "google_api_key",
    "groq": "groq_api_key",
    "nvidia": "nvidia_api_key",
    "huggingface": "huggingfacehub_api_token",
    "microsoft": "azure_openai_api_key",
}

# A local/self-hosted base_url on this host is the keyless local ollama path.
_OLLAMA_CLOUD_HOST = "ollama.com"
# Sensible model default per provider (blank = the operator must type one).
_LLM_DEFAULT_MODEL = {"ollama": "deepseek-v4.1-flash:cloud"}
# Interactive re-prompts for a rejected key before giving up (bounded so a
# NonInteractivePrompter can never spin forever).
_LLM_KEY_ATTEMPTS = 3


class SetupError(RuntimeError):
    """Raised when the wizard cannot complete (validation or config error)."""


def _persist_and_bootstrap(
    secrets: dict[str, str], config: dict[str, object], passphrase: str
) -> None:
    secret_store.save_secrets(secrets, passphrase)
    config_file.save_config(config)
    docker_bootstrap.bring_up_db()
    docker_bootstrap.migrate()


def _llm_key_env(provider: str) -> str:
    """The env var a provider's key may also arrive in (upper-cased secret name)."""
    return _LLM_SECRET_NAMES[provider].upper()


def _llm_needs_key(provider: str, base_url: str | None) -> bool:
    """Whether ``provider`` requires an auth key given ``base_url``.

    ``aws`` never does (boto3 chain); ``ollama`` only on the cloud host; ``nvidia``
    only when hosted (a self-hosted NIM ``base_url`` is keyless); all others do.
    """
    if provider == "aws":
        return False
    if provider == "ollama":
        return not base_url or _OLLAMA_CLOUD_HOST in base_url
    if provider == "nvidia":
        return base_url is None
    return True


def _stage_llm(
    config: dict[str, object],
    secrets: dict[str, str],
    *,
    provider: str,
    model: str | None,
    base_url: str | None,
    fields: dict[str, str | None],
    secret_name: str | None,
    key: str | None,
) -> None:
    """Stage validated LLM choices: non-secret → ``config``, key → ``secrets``."""
    config["llm_provider"] = provider
    if model:
        config["llm_model"] = model
    if base_url:
        config["llm_base_url"] = base_url
    if provider == "aws" and fields.get("region"):
        config["aws_region"] = fields["region"]
    if provider == "microsoft":
        for cfg_key, field_key in (
            ("azure_openai_endpoint", "endpoint"),
            ("azure_openai_api_version", "api_version"),
            ("azure_openai_deployment_name", "deployment_name"),
        ):
            if fields.get(field_key):
                config[cfg_key] = fields[field_key]
    if secret_name and key:
        secrets[secret_name] = key


def _configure_llm_noninteractive(
    config: dict[str, object],
    secrets: dict[str, str],
    *,
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_key: str | None,
) -> None:
    """Stage the fund LLM backend from flags/env; validate before anything persists.

    ``llm_provider`` unset → no LLM config (fund keeps its env-free default).
    """
    if not llm_provider:
        return
    if llm_provider not in validators.SUPPORTED_LLM_PROVIDERS:
        raise SetupError(
            f"Unknown LLM provider {llm_provider!r}; expected one of "
            f"{', '.join(validators.SUPPORTED_LLM_PROVIDERS)}."
        )
    fields = _llm_fields_from_env(llm_provider, llm_base_url)
    secret_name = _LLM_SECRET_NAMES.get(llm_provider)
    key: str | None = None
    if secret_name and _llm_needs_key(llm_provider, llm_base_url):
        key = llm_key or os.getenv(_llm_key_env(llm_provider))
        if not key:
            raise SetupError(
                f"{llm_provider} needs an API key "
                f"(--llm-key or {_llm_key_env(llm_provider)})."
            )
    if not validators.validate_llm(
        llm_provider, key=key, base_url=llm_base_url, **fields
    ):
        raise SetupError(f"{llm_provider} credentials failed validation.")
    _stage_llm(
        config,
        secrets,
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        fields=fields,
        secret_name=secret_name,
        key=key,
    )


def _llm_fields_from_env(provider: str, base_url: str | None) -> dict[str, str | None]:
    """Provider-specific non-secret fields, sourced from env in non-interactive runs."""
    if provider == "aws":
        return {"region": os.getenv("AWS_REGION") or base_url}
    if provider == "microsoft":
        return {
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT") or base_url,
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        }
    return {}


def _prompt_llm_fields(prompter: Prompter, provider: str) -> dict[str, str | None]:
    """Prompt provider-specific non-secret fields (env auto-detected first)."""
    if provider == "aws":
        return {"region": os.getenv("AWS_REGION") or prompter.text(_MSG_AWS_REGION)}
    if provider == "microsoft":
        return {
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT")
            or prompter.text(_MSG_AZURE_ENDPOINT),
            "api_version": os.getenv("OPENAI_API_VERSION")
            or prompter.text(_MSG_AZURE_API_VERSION),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or prompter.text(_MSG_AZURE_DEPLOYMENT),
        }
    return {}


def _validate_llm_key_with_retry(
    prompter: Prompter,
    provider: str,
    base_url: str | None,
    fields: dict[str, str | None],
) -> str:
    """Prompt for the key and validate; re-prompt on rejection (network → abort)."""
    env_key = os.getenv(_llm_key_env(provider))
    for _ in range(_LLM_KEY_ATTEMPTS):
        key = env_key or prompter.password(_MSG_LLM_KEY)
        if validators.validate_llm(provider, key=key, base_url=base_url, **fields):
            return key
        env_key = None  # discard a bad env key so the next round prompts
        prompter.error("That LLM key was rejected; try again.")
    raise SetupError(
        f"{provider} key failed validation after {_LLM_KEY_ATTEMPTS} attempts."
    )


def _configure_llm_interactive(
    prompter: Prompter, config: dict[str, object], secrets: dict[str, str]
) -> None:
    """Prompt for the fund LLM backend, validate before persist.

    Declining keeps fund.config's env-free default. A rejected key re-prompts; an
    unreachable service raises ``ValidationNetworkError`` so setup aborts clean.
    """
    if not prompter.confirm(_MSG_CONFIGURE_LLM, default=False):
        return
    provider = prompter.select(_MSG_LLM_PROVIDER, validators.SUPPORTED_LLM_PROVIDERS)
    model = prompter.text(
        _MSG_LLM_MODEL, default=_LLM_DEFAULT_MODEL.get(provider, "")
    ).strip()
    base_url = prompter.text(_MSG_LLM_BASE_URL, default="").strip() or None
    fields = _prompt_llm_fields(prompter, provider)

    secret_name = _LLM_SECRET_NAMES.get(provider)
    key: str | None = None
    if secret_name and _llm_needs_key(provider, base_url):
        key = _validate_llm_key_with_retry(prompter, provider, base_url, fields)
    elif not validators.validate_llm(provider, base_url=base_url, **fields):
        raise SetupError(f"{provider} configuration failed validation.")

    _stage_llm(
        config,
        secrets,
        provider=provider,
        model=model or None,
        base_url=base_url,
        fields=fields,
        secret_name=secret_name,
        key=key,
    )


def run_setup_noninteractive(
    *,
    passphrase: str | None,
    t212_key: str | None = None,
    t212_secret: str | None = None,
    fred_key: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_key: str | None = None,
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

    _configure_llm_noninteractive(
        config,
        secrets,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_key=llm_key,
    )

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

    _configure_llm_interactive(prompter, config, secrets)

    _persist_and_bootstrap(secrets, config, pw)
