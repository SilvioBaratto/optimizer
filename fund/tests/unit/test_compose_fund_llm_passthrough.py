"""Contract: the ``fund`` compose service forwards the D4 switchable-LLM env vars.

The deferred half of Task 13 (SPEC "Switchable LLM Backends"): the ``fund``
service in the repo-root ``docker-compose.yml`` must pass the D4 backend
configuration into the container. Two mechanisms, deliberately split:

* ``env_file: .env`` forwards **every** provider var present in ``.env`` (the
  shared model-id pair, each provider's auth var, ``OLLAMA_BASE_URL``) exactly as
  ``fund.config.load_config`` reads them — and *only when set*, so an unset var
  falls through to ``fund.config``'s own default.
* the switchable selector ``LLM_PROVIDER`` is **additionally** declared as an
  explicit ``environment`` entry so its default (``ollama``) is visible in compose
  and overridable from the host shell.

We intentionally do **not** give the model pair / auth vars empty-default
``environment`` entries: ``environment:`` overrides ``env_file:``, so
``FUND_PRIMARY_MODEL: ${FUND_PRIMARY_MODEL:-}`` would clobber ``fund.config``'s
code default with an empty string whenever the var is unset. ``env_file`` is the
correct pass-through-if-present channel for those.

``Path(__file__).resolve().parents[3]`` resolves to the repo root
(unit -> tests -> fund -> optimizer), never ``Path.cwd()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"

# T3: the fund service consumes these as file-based docker secrets (rendered by
# `portopt start` to /run/secrets/<name>). Keys are the lower-case secret-file
# names (name-agree with compose_secrets.SECRET_NAMES); values are the env-var
# prefix fund.config._read_secret reads (<PREFIX>_FILE wins over inline). aws is
# intentionally absent — it authenticates via IAM/region, not a secret file.
_FUND_SECRET_ENV = {
    "ollama_api_key": "OLLAMA_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "groq_api_key": "GROQ_API_KEY",
    "nvidia_api_key": "NVIDIA_API_KEY",
    "huggingfacehub_api_token": "HUGGINGFACEHUB_API_TOKEN",
    "azure_openai_api_key": "AZURE_OPENAI_API_KEY",
    "fred_api_key": "FRED_API_KEY",
}


def _compose() -> dict[str, Any]:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


def _fund_service() -> dict[str, Any]:
    return _compose()["services"]["fund"]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def test_fund_service_forwards_dotenv():
    """env_file includes .env, so D4 vars set there reach the container."""
    assert ".env" in _as_list(_fund_service().get("env_file"))


def test_fund_service_declares_llm_provider_selector():
    """The switchable selector is an explicit environment entry, not left implicit."""
    environment = _fund_service().get("environment", {})
    assert "LLM_PROVIDER" in environment


def test_fund_llm_provider_default_matches_config():
    """LLM_PROVIDER default is ollama (fund.config default) and host-overridable."""
    environment = _fund_service().get("environment", {})
    assert environment["LLM_PROVIDER"] == "${LLM_PROVIDER:-ollama}"


# --- T3: per-provider file-based docker secrets reach the fund container -------


def test_fund_service_mounts_every_provider_secret():
    """The fund service mounts each provider key + fred as a docker secret."""
    mounted = set(_as_list(_fund_service().get("secrets")))
    assert set(_FUND_SECRET_ENV) <= mounted


def test_fund_service_maps_secret_file_env():
    """Each secret is wired as <NAME>_FILE=/run/secrets/<name> (fund._read_secret)."""
    environment = _fund_service().get("environment", {})
    for name, env in _FUND_SECRET_ENV.items():
        assert environment.get(f"{env}_FILE") == f"/run/secrets/{name}"


def test_fund_service_has_no_bare_secret_key_env():
    """Secrets flow ONLY via <NAME>_FILE — never a bare `<NAME>:` under environment
    (which would bake a value into compose / clobber the file-based path)."""
    environment = _fund_service().get("environment", {})
    for env in _FUND_SECRET_ENV.values():
        assert env not in environment


def test_top_level_secrets_declare_every_fund_secret():
    """Every fund secret has a top-level `secrets: <name>: {file: ./secrets/<name>}`."""
    declared = _compose().get("secrets", {})
    for name in _FUND_SECRET_ENV:
        assert declared.get(name) == {"file": f"./secrets/{name}"}
