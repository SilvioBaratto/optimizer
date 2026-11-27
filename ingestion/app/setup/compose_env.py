"""Render the wizard's non-secret settings into the fund's compose env file (SPEC D4).

The install wizard persists its non-secret choices to `~/.portopt/config.toml`.
Containers never read that file — they read environment variables — so
`portopt start` translates the fund's LLM selection (provider/model/base-url/
region/Azure) into `./.env.fund`, under the exact env-var NAMES
`fund.config.load_config` reads. Compose mounts it as the LAST `env_file` entry of
the fund service (so it overrides `.env`); `portopt stop` and a failed setup delete
it.

The scheduler's one setting from config.toml, the acknowledgement of Yahoo's terms,
needs no file: `config_to_scheduler_env` gives it to `portopt start`, which adds it
to the environment of `docker compose up`.

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

from app.setup import terms
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

# config.toml boolean -> the env-var name the ingestion scheduler reads for it. These
# travel in the environment of `docker compose up`, which docker-compose.yml
# interpolates into the scheduler service, so CI and a plain `docker compose up` can
# set them the same way.
_SCHEDULER_ENV_BY_CONFIG_KEY = {
    terms.ACKNOWLEDGEMENT_CONFIG_KEY: terms.ACKNOWLEDGEMENT_ENV_VAR,
}
_ENV_TRUE = "true"

# Relative to the compose project directory (the repo root), like the secrets dir.
DEFAULT_ENV_FUND_PATH = Path(".env.fund")


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
    fund's are ignored and absent settings are omitted, never emitted as empty.
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
    """Translate persisted config.toml booleans to the scheduler's env-var names.

    Only a value stored as ``true`` is passed. A missing or false one is omitted, so
    a value the operator exported for the compose run is not overridden and the
    compose file's own default applies otherwise.
    """
    return {
        env_var: _ENV_TRUE
        for cfg_key, env_var in _SCHEDULER_ENV_BY_CONFIG_KEY.items()
        if config.get(cfg_key) is True
    }


def render(config: Mapping[str, object], *, path: Path) -> Path:
    """Write the fund's translated env vars to `<path>` (`.env.fund`).

    ``path`` has no default on purpose: a cwd-relative one would write the file
    wherever the operator happened to run, not where compose reads it.

    Always writes the file (empty when nothing is configured) so the fund service's
    `env_file` entry resolves. Returns the path written.
    """
    return _write_env_file(path, config_to_env(config))


def cleanup(*, path: Path) -> None:
    """Remove the rendered `.env.fund` file (best effort)."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
