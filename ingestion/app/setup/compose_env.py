"""Render the wizard's non-secret LLM selection into a compose env file (SPEC D4).

The install wizard persists the chosen provider/model/base-url/region/Azure fields
to `~/.portopt/config.toml`. The fund container never reads that file — it reads
environment variables. `portopt start` calls `render` to translate those config
keys into the exact env-var NAMES `fund.config.load_config` reads and write them to
`./.env.fund`, which the fund service mounts as its last `env_file` entry (so it
overrides `.env`). Provider auth keys are NOT handled here — they flow only through
file-based docker secrets (`compose_secrets`).

Only keys the wizard actually set are written. An absent setting must stay absent
so the fund's own code default applies; writing it as an empty value would instead
pin the field to the empty string.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path

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

DEFAULT_ENV_FUND_PATH = Path(".env.fund")


def config_to_env(config: Mapping[str, object]) -> dict[str, str]:
    """Translate persisted config.toml keys to the fund's env-var names.

    Returns only the vars the wizard set (truthy values); unknown/non-LLM keys are
    ignored and absent settings are omitted, never emitted as empty.
    """
    env: dict[str, str] = {}
    for cfg_key, env_var in _ENV_BY_CONFIG_KEY.items():
        value = config.get(cfg_key)
        if value:
            env[env_var] = str(value)

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
            env[base_url_var] = str(base_url)
    return env


def render(config: Mapping[str, object], *, path: Path | None = None) -> Path:
    """Write the translated non-secret LLM env vars to `<path>` (`.env.fund`).

    Always writes the file (empty when nothing is configured) so the fund service's
    `env_file` entry resolves. Returns the path written.
    """
    target = path or DEFAULT_ENV_FUND_PATH
    env = config_to_env(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(f"{key}={value}\n" for key, value in env.items()), encoding="utf-8"
    )
    return target


def cleanup(*, path: Path | None = None) -> None:
    """Remove the rendered `.env.fund` file (best effort)."""
    target = path or DEFAULT_ENV_FUND_PATH
    with contextlib.suppress(OSError):
        target.unlink(missing_ok=True)
