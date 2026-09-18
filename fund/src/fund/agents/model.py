"""Model factory — DeepSeek on Ollama Cloud via ``ChatOllama`` (SPEC D4).

Builds the primary + fallback chat models every Phase-7 agent runs on, threading
the frozen :class:`~fund.config.FundConfig` contract onto ``ChatOllama``:
``base_url`` (Ollama Cloud), the pinned model ids, ``temperature=0`` (reproducible
runs), and the **non-thinking flash route** (``reasoning=False`` — thinking mode
rejects ``tool_choice`` and deepagents needs reliable tool-calling).

Construction only — no DB, no network, no ``.invoke``. ``ChatOllama`` is built
lazily; the first request happens later at call time. ``langchain_ollama`` is
imported **inside** the builder so a bare ``import fund.agents.model`` never drags
in the model stack and needs no environment. The ``OLLAMA_API_KEY`` secret is
validated at **build time** (like ``fund.audit.persistence`` does for
``DATABASE_URL``), never at import — CI keeps ``settings.ollama_api_key`` ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fund.config import FundConfig, settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "build_chat_model",
    "build_fallback",
    "build_primary",
]


def build_chat_model(config: FundConfig, *, model_name: str) -> BaseChatModel:
    """Return a ``ChatOllama`` for ``model_name`` configured from ``config``.

    The ``OLLAMA_API_KEY`` is required and sent as a bearer ``Authorization``
    header (Ollama Cloud auth); a missing key raises :class:`RuntimeError` here at
    build time, never at import.
    """
    if not config.ollama_api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is required to build a chat model; "
            "set it in the environment."
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name,
        base_url=config.ollama_base_url,
        client_kwargs={"headers": {"Authorization": f"Bearer {config.ollama_api_key}"}},
        temperature=config.model_temperature,
        reasoning=config.model_reasoning,
    )


def build_primary(config: FundConfig = settings) -> BaseChatModel:
    """Build the primary model (``config.primary_model``)."""
    return build_chat_model(config, model_name=config.primary_model)


def build_fallback(config: FundConfig = settings) -> BaseChatModel:
    """Build the fallback model (``config.fallback_model``)."""
    return build_chat_model(config, model_name=config.fallback_model)
