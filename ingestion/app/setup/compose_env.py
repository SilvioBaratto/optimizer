"""Render the wizard's non-secret settings into compose env files (SPEC D4).

The install wizard persists its non-secret choices to `~/.portopt/config.toml`.
Containers never read that file — they read environment variables — so
`portopt start` translates those keys into env files that compose mounts as the
LAST `env_file` entry of the consuming service (so they override `.env`):

* `./.env.fund` — the fund's LLM selection (provider/model/base-url/region/Azure),
  under the exact env-var NAMES `fund.config.load_config` reads. `portopt stop` and
  a failed setup delete it.
* `./.env.scheduler` — the ingestion scheduler's `TRADING_212_MODE`. It is a
  separate file because the scheduler outlives `portopt stop`: if the mode shared
  the fund's file, stopping the fund would delete it, and the next plain
  `docker compose up` would recreate the scheduler against the live host.

Provider auth keys are NOT handled here — they flow only through file-based docker
secrets (`compose_secrets`).

Only keys the wizard actually set are written. An absent setting must stay absent
so the consumer's own code default applies; writing it as an empty value would
instead pin the field to the empty string.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path

from app.setup.host_urls import to_container_url

# config.toml key -> the env-var name fund.config.load_config reads for it.
# `azure_openai_api_version` is deliberately mapped to OPENAI_API_VERSION (the name
# fund.config actually reads), NOT AZURE_OPENAI_API_VERSION — the one translation a
# reader would get wrong.
_ENV_BY_CONFIG_KEY = {
    "llm_provider": "LLM_PROVIDER",
    "aws_region": "AWS_REGION",
    "azure_openai_endpoint": "AZURE_OPENAI_ENDPOINT",
    "azure_openai_api_version": "OPENAI_API_VERSION",
    "azure_openai_deployment_name": "AZURE_OPENAI_DEPLOYMENT_NAME",
}

# The wizard stores one generic `llm_base_url`, but fund.config reads a different
# env var per provider. Providers absent here have no base-url field in fund.config
# (openrouter/anthropic/google/groq/huggingface route to a fixed endpoint; aws uses
# AWS_REGION), so a base_url set for them has nowhere to go and is dropped.
_BASE_URL_ENV_BY_PROVIDER = {
    "ollama": "OLLAMA_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "nvidia": "NVIDIA_BASE_URL",
}

# config.toml key -> the env-var name the ingestion daemon reads. Trading212 keys
# are scoped to one environment (demo vs live host), so the daemon must be told
# which one the wizard validated them against.
_SCHEDULER_ENV_BY_CONFIG_KEY = {
    "trading_212_mode": "TRADING_212_MODE",
}

DEFAULT_ENV_FUND_PATH = Path(".env.fund")
DEFAULT_ENV_SCHEDULER_PATH = Path(".env.scheduler")


def _translate(
    config: Mapping[str, object], names: Mapping[str, str]
) -> dict[str, str]:
    env: dict[str, str] = {}
    for cfg_key, env_var in names.items():
        value = config.get(cfg_key)
        if value:
            env[env_var] = str(value)
    return env


def _write_env_file(target: Path, env: Mapping[str, str]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(f"{key}={value}\n" for key, value in env.items()), encoding="utf-8"
    )
    return target


def config_to_env(config: Mapping[str, object]) -> dict[str, str]:
    """Translate persisted config.toml keys to the fund's env-var names.

    Returns only the vars the wizard set (truthy values); keys that are not the
    fund's (the scheduler's `trading_212_mode`, unrelated config) are ignored and
    absent settings are omitted, never emitted as empty.
    """
    env = _translate(config, _ENV_BY_CONFIG_KEY)

    provider = str(config.get("llm_provider", ""))

    model = config.get("llm_model")
    if model:
        env["FUND_PRIMARY_MODEL"] = str(model)
        # fund.config's fallback default (deepseek-*:cloud) is an Ollama model id;
        # on any other provider it builds a fallback that only fails at request
        # time, so mirror the primary unless the wizard captured an explicit one.
        fallback = config.get("llm_fallback_model") or (
            model if provider and provider != "ollama" else None
        )
        if fallback:
            env["FUND_FALLBACK_MODEL"] = str(fallback)

    base_url = config.get("llm_base_url")
    if base_url:
        base_url_var = _BASE_URL_ENV_BY_PROVIDER.get(provider)
        if base_url_var:
            env[base_url_var] = to_container_url(str(base_url))
    return env


def config_to_scheduler_env(config: Mapping[str, object]) -> dict[str, str]:
    """Translate persisted config.toml keys to the ingestion scheduler's env vars.

    Returns only what the wizard set (today just `TRADING_212_MODE`); absent
    settings are omitted so the daemon's own default applies.
    """
    return _translate(config, _SCHEDULER_ENV_BY_CONFIG_KEY)


def render(config: Mapping[str, object], *, path: Path | None = None) -> Path:
    """Write the fund's translated env vars to `<path>` (`.env.fund`).

    Always writes the file (empty when nothing is configured) so the fund service's
    `env_file` entry resolves. Returns the path written.
    """
    return _write_env_file(path or DEFAULT_ENV_FUND_PATH, config_to_env(config))


def render_scheduler(config: Mapping[str, object], *, path: Path | None = None) -> Path:
    """Write the scheduler's translated env vars to `<path>` (`.env.scheduler`).

    Always rewrites the file (empty when no mode is configured) so the scheduler's
    `env_file` entry resolves and a mode dropped from config.toml cannot linger
    from an earlier start. Nothing here deletes it: the scheduler keeps running
    after `portopt stop`, and a plain `docker compose up` must still find it.
    Returns the path written.
    """
    return _write_env_file(
        path or DEFAULT_ENV_SCHEDULER_PATH, config_to_scheduler_env(config)
    )


def cleanup(*, path: Path | None = None) -> None:
    """Remove the rendered `.env.fund` file (best effort).

    Deliberately leaves `.env.scheduler` alone: callers (`portopt stop`, a failed
    setup) run while the scheduler is still up.
    """
    target = path or DEFAULT_ENV_FUND_PATH
    with contextlib.suppress(OSError):
        target.unlink(missing_ok=True)
