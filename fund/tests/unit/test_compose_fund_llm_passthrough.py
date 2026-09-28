"""Contract: the ``fund`` compose service forwards the D4 switchable-LLM env vars.

The deferred half of Task 13 (SPEC "Switchable LLM Backends"): the ``fund``
service in the repo-root ``docker-compose.yml`` must pass the D4 backend
configuration into the container. Two mechanisms, deliberately split:

* ``env_file: .env`` forwards **every** provider var present in ``.env`` (the
  shared model-id pair, each provider's auth var, ``OLLAMA_BASE_URL``) exactly as
  ``fund.config.load_config`` reads them — and *only when set*, so an unset var
  falls through to ``fund.config``'s own default.
* ``env_file: .env.fund`` (rendered by ``portopt start`` from the wizard's
  ``config.toml``, listed **last** so it overrides ``.env``) carries the switchable
  selector ``LLM_PROVIDER`` and the rest of the non-secret LLM selection.

``LLM_PROVIDER`` is delivered through ``.env.fund``, **not** an ``environment``
entry: ``environment:`` overrides ``env_file:`` AND ``${...}`` interpolation never
reads ``.env.fund`` (Compose auto-loads only ``.env``), so an
``LLM_PROVIDER: ${LLM_PROVIDER:-ollama}`` entry would pin the container to
``ollama`` and silently discard ``.env.fund``. The same reasoning bars empty-default
``environment`` entries for the model pair / auth vars (``${FUND_PRIMARY_MODEL:-}``
would clobber ``fund.config``'s code default with ``""``). ``env_file`` is the sole
non-secret channel; secrets stay on the ``<NAME>_FILE`` docker-secret channel.

``Path(__file__).resolve().parents[3]`` resolves to the repo root
(unit -> tests -> fund -> optimizer), never ``Path.cwd()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
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


def _env_file_paths() -> list[str]:
    """env_file entries as paths (each is a bare string or a {path, required} map)."""
    return [
        entry if isinstance(entry, str) else entry.get("path")
        for entry in _as_list(_fund_service().get("env_file"))
    ]


def test_fund_service_forwards_dotenv():
    """env_file includes .env, so D4 vars set there reach the container."""
    assert ".env" in _env_file_paths()


def test_fund_service_appends_generated_env_fund_last():
    """The wizard's non-secret LLM selection reaches the container via .env.fund,
    listed last so it overrides .env."""
    assert _env_file_paths()[-1] == ".env.fund"


def test_env_fund_is_optional():
    """A pre-setup `docker compose up` (no rendered .env.fund) must still parse."""
    env_fund = next(
        e
        for e in _as_list(_fund_service().get("env_file"))
        if isinstance(e, dict) and e.get("path") == ".env.fund"
    )
    assert env_fund.get("required") is False


# The non-secret D4 vars fund.config reads. NONE may sit under `environment:`: it
# outranks env_file, and `${...}` interpolation never reads .env.fund, so an entry
# there would shadow the wizard's file (or an empty `${VAR:-}` would clobber the
# fund.config code default). They flow only via env_file (.env / .env.fund).
_NON_SECRET_D4_VARS = (
    "LLM_PROVIDER",
    "FUND_PRIMARY_MODEL",
    "FUND_FALLBACK_MODEL",
    "OLLAMA_BASE_URL",
    "OPENAI_BASE_URL",
    "NVIDIA_BASE_URL",
    "AWS_REGION",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
)


@pytest.mark.parametrize("var", _NON_SECRET_D4_VARS)
def test_non_secret_llm_var_not_declared_under_environment(var: str):
    """No non-secret LLM var is declared under `environment:` (would shadow
    .env.fund or clobber the fund.config default)."""
    assert var not in _fund_service().get("environment", {})


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
