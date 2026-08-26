"""Runtime BAML provider routing for the ingestion daemon (SPEC D8).

BAML functions bind a static default client at compile time; at call time we
override the primary via a per-call ``ClientRegistry`` built from
``settings.llm_provider`` (cloud only: openai/anthropic) with the matching model
and key. Callers pass ``baml_options=llm_call_options()`` so the existing
``b.<Function>(...)`` call shape (and its tests) stay intact.

The registry is built per call — cheap and thread-safe (the scheduler runs
BAML calls from daemon threads; a shared mutable client would not be safe).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from baml_py import ClientRegistry

from app.config import settings

if TYPE_CHECKING:
    from app.config import Settings

_CLIENT_NAME = "PortoptRuntime"


def _resolve_client(cfg: Settings) -> tuple[str, dict[str, str]]:
    """Map the configured provider to (baml provider, client options)."""
    provider = cfg.llm_provider
    if provider == "openai":
        return "openai", {"model": cfg.openai_model, "api_key": cfg.openai_api_key}
    if provider == "anthropic":
        return "anthropic", {
            "model": cfg.anthropic_model,
            "api_key": cfg.anthropic_api_key,
        }
    raise ValueError(
        f"Unsupported LLM provider: {provider!r} (cloud only: openai, anthropic)"
    )


def build_client_registry(cfg: Settings | None = None) -> ClientRegistry:
    """Build a ClientRegistry whose primary is the configured cloud provider."""
    cfg = cfg or settings
    provider, options = _resolve_client(cfg)
    registry = ClientRegistry()
    registry.add_llm_client(_CLIENT_NAME, provider, options)
    registry.set_primary(_CLIENT_NAME)
    return registry


def llm_call_options(cfg: Settings | None = None) -> dict[str, Any]:
    """`baml_options` dict carrying a runtime ClientRegistry for the provider."""
    return {"client_registry": build_client_registry(cfg)}
