"""Install-wizard orchestration for ``portopt setup`` (SPEC D4).

Two entry points share one persist+bootstrap core:
- ``run_setup_noninteractive`` — flags/env for CI; fails loud, never loops.
- ``run_setup_interactive`` — drives a `Prompter`.

Secrets are validated live *before* anything is written, so a failure at any
step leaves nothing persisted.
"""

from __future__ import annotations

import logging
import os

from app.setup import (
    config_file,
    docker_bootstrap,
    lifecycle,
    path_install,
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


def _is_first_run() -> bool:
    """True when neither the secret store nor the config file exists yet.

    Gates the one-time auto-launch: a first interactive setup brings the stack up,
    but a re-run never does (the plan's ``interactive + TTY + not --no-launch``, once).
    """
    return not (
        secret_store.DEFAULT_SECRETS_PATH.exists()
        or config_file.DEFAULT_CONFIG_PATH.exists()
    )


def _merge_with_existing_secrets(
    new: dict[str, str], passphrase: str
) -> dict[str, str]:
    """Merge ``new`` over the already-stored secrets so a re-run never wipes them.

    A missing store (first run) starts empty. A store that will not decrypt with the
    given passphrase fails loud — overwriting it would silently destroy the user's
    secrets (the Immich anti-pattern the spec forbids).
    """
    try:
        existing = secret_store.load_secrets(passphrase)
    except secret_store.SecretStoreNotFoundError:
        existing = {}
    except secret_store.InvalidPassphraseError as exc:
        raise SetupError(
            "The existing secret store could not be decrypted with this passphrase; "
            "re-run with the original passphrase."
        ) from exc
    return {**existing, **new}


def _persist_and_bootstrap(
    secrets: dict[str, str],
    config: dict[str, object],
    passphrase: str,
    *,
    skip_path_install: bool = False,
) -> None:
    merged = _merge_with_existing_secrets(secrets, passphrase)
    secret_store.save_secrets(merged, passphrase)
    # Merge, never clobber: a re-run that omits a section must keep the config the
    # earlier run persisted (and the repo_path the launcher install records).
    config_file.update_config(config)
    docker_bootstrap.bring_up_db()
    docker_bootstrap.migrate()
    if not skip_path_install:
        # Best-effort: secrets are already encrypted and the DB migrated, so a PATH
        # failure must not fail the whole setup — warn and let the user add it manually.
        try:
            path_install.install_launcher()
        except path_install.PathInstallError as exc:
            logging.getLogger(__name__).warning(
                "Could not install the `optimizer` launcher on PATH: %s. "
                "Add scripts/optimizer to PATH manually.",
                exc,
            )


def _auto_launch(passphrase: str) -> None:
    """Bring the stack up after a first setup; a failure is a warning, not a rollback.

    Setup has already persisted secrets and migrated, so a launch failure (Docker
    down, image build error) must not undo it — the operator can retry with
    ``portopt start``.
    """
    try:
        lifecycle.run_start(passphrase)
    except Exception as exc:
        # Deliberately broad: setup is already persisted + migrated, so no launch
        # error (Docker down, build failure, bad passphrase) may undo it.
        logging.getLogger(__name__).warning(
            "Could not auto-launch the stack: %s. Run `portopt start` to bring it up.",
            exc,
        )


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
    skip_validation: bool = False,
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
        llm_provider,
        key=key,
        base_url=llm_base_url,
        skip_validation=skip_validation,
        **fields,
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
    *,
    skip_validation: bool = False,
) -> str:
    """Prompt for the key and validate; re-prompt on rejection (network → abort)."""
    env_key = os.getenv(_llm_key_env(provider))
    for _ in range(_LLM_KEY_ATTEMPTS):
        key = env_key or prompter.password(_MSG_LLM_KEY)
        if validators.validate_llm(
            provider,
            key=key,
            base_url=base_url,
            skip_validation=skip_validation,
            **fields,
        ):
            return key
        env_key = None  # discard a bad env key so the next round prompts
        prompter.error("That LLM key was rejected; try again.")
    raise SetupError(
        f"{provider} key failed validation after {_LLM_KEY_ATTEMPTS} attempts."
    )


def _configure_llm_interactive(
    prompter: Prompter,
    config: dict[str, object],
    secrets: dict[str, str],
    *,
    existing: dict[str, object] | None = None,
    reconfigure: bool = False,
    skip_validation: bool = False,
) -> None:
    """Prompt for the fund LLM backend, validate before persist.

    Declining keeps fund.config's env-free default (or, on a re-run, the provider the
    earlier run persisted — the merge preserves it). The "configure?" prompt defaults
    to reconfigure: off on a first run / a reuse re-run, on when ``--reconfigure`` asks
    to change it. Prompt defaults are pre-filled from ``existing`` config. A rejected
    key re-prompts; an unreachable service raises ``ValidationNetworkError``.
    """
    existing = existing or {}
    if not prompter.confirm(_MSG_CONFIGURE_LLM, default=reconfigure):
        return
    provider = prompter.select(_MSG_LLM_PROVIDER, validators.SUPPORTED_LLM_PROVIDERS)
    model = prompter.text(
        _MSG_LLM_MODEL,
        default=_LLM_DEFAULT_MODEL.get(provider)
        or str(existing.get("llm_model") or ""),
    ).strip()
    base_url = (
        prompter.text(
            _MSG_LLM_BASE_URL, default=str(existing.get("llm_base_url") or "")
        ).strip()
        or None
    )
    fields = _prompt_llm_fields(prompter, provider)

    secret_name = _LLM_SECRET_NAMES.get(provider)
    key: str | None = None
    if secret_name and _llm_needs_key(provider, base_url):
        key = _validate_llm_key_with_retry(
            prompter, provider, base_url, fields, skip_validation=skip_validation
        )
    elif not validators.validate_llm(
        provider, base_url=base_url, skip_validation=skip_validation, **fields
    ):
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
    skip_path_install: bool = False,
    skip_validation: bool = False,
    reconfigure: bool = False,
) -> None:
    """Non-interactive setup from flags/env — fails loud, persists nothing on error.

    A re-run merges the given flags over the existing store/config (never wiping an
    untouched secret); ``skip_validation`` bypasses the live credential checks; CI
    never auto-launches (that is an interactive-only, first-run step).
    """
    if not passphrase:
        raise SetupError("A master passphrase is required (set PORTOPT_PASSPHRASE).")
    docker_bootstrap.check_docker()

    secrets: dict[str, str] = {}
    config: dict[str, object] = {}

    if t212_key or t212_secret:
        if not (t212_key and t212_secret):
            raise SetupError("Trading212 needs both an API key and a secret key.")
        if not skip_validation and not validators.validate_t212(t212_key, t212_secret):
            raise SetupError("Trading212 credentials failed validation.")
        secrets["trading_212_api_key"] = t212_key
        secrets["trading_212_secret_key"] = t212_secret

    if fred_key:
        if not skip_validation and not validators.validate_fred(fred_key):
            raise SetupError("FRED API key failed validation.")
        secrets["fred_api_key"] = fred_key

    _configure_llm_noninteractive(
        config,
        secrets,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_key=llm_key,
        skip_validation=skip_validation,
    )

    _persist_and_bootstrap(
        secrets, config, passphrase, skip_path_install=skip_path_install
    )


def run_setup_interactive(
    prompter: Prompter,
    *,
    passphrase: str | None = None,
    skip_path_install: bool = False,
    skip_validation: bool = False,
    reconfigure: bool = False,
    no_launch: bool = False,
) -> None:
    """Interactive setup via the prompt seam; each credential validates before persist.

    A first run (no store/config) brings the stack up at the end unless ``no_launch``;
    a re-run never auto-launches and merges over the existing store/config, pre-filling
    the LLM prompts from what was persisted. ``skip_validation`` bypasses live checks.
    """
    docker_bootstrap.check_docker()
    first_run = _is_first_run()
    existing = config_file.load_config()

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
        if not skip_validation and not validators.validate_t212(t212_key, t212_secret):
            raise SetupError("Trading212 credentials failed validation.")
        secrets["trading_212_api_key"] = t212_key
        secrets["trading_212_secret_key"] = t212_secret

    if prompter.confirm(_MSG_CONNECT_FRED, default=False):
        fred_key = os.getenv("FRED_API_KEY") or prompter.password(_MSG_FRED_KEY)
        if not skip_validation and not validators.validate_fred(fred_key):
            raise SetupError("FRED API key failed validation.")
        secrets["fred_api_key"] = fred_key

    _configure_llm_interactive(
        prompter,
        config,
        secrets,
        existing=existing,
        reconfigure=reconfigure,
        skip_validation=skip_validation,
    )

    _persist_and_bootstrap(secrets, config, pw, skip_path_install=skip_path_install)

    if first_run and not no_launch:
        _auto_launch(pw)
