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


def _fund_service() -> dict[str, Any]:
    data = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return data["services"]["fund"]


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
