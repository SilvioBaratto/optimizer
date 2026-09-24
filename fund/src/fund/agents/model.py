"""Model factory — a lazy-dispatch provider registry over LangChain chat models.

``build_chat_model`` dispatches on :attr:`FundConfig.llm_provider` through the
``_BUILDERS`` registry: one small ``_build_<provider>`` per LangChain
``BaseChatModel`` backend (SPEC D4, "Switchable LLM Backends"). ``build_primary``
/ ``build_fallback`` keep their signatures so the four call sites (``worker``,
``scheduler``, ``cli``, ``tui``) and the ``ModelFallbackMiddleware`` wiring in
``graph.py`` are untouched. The env-free default (`llm_provider="ollama"`,
`ollama_base_url="https://ollama.com"`) reproduces today's DeepSeek-on-Ollama-Cloud
path byte-for-byte.

**Load-bearing lazy imports.** Each builder imports its provider SDK **inside**
the function body (mirroring the original ``from langchain_ollama import
ChatOllama``), so a bare ``import fund.agents.model`` pulls in **no** provider
package — ``worker.py`` / ``scheduler.py`` stay agent-stack-free. This invariant
is guarded by ``tests/unit/hygiene/test_model_no_provider_import.py``.

**Build-time auth, never import-time.** Required secrets are validated inside the
builder via :func:`_require` (raising a clear :class:`RuntimeError`), like
``fund.audit.persistence`` does for ``DATABASE_URL`` — CI keeps every provider key
unset. An **uninstalled** optional extra surfaces as the builder's ``ImportError``,
which ``build_chat_model`` re-raises as a ``RuntimeError`` naming the ``uv sync``
command to install it.

Construction only — no DB, no network, no ``.invoke``. The chat model is built
lazily; the first request happens later at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fund.config import FundConfig, settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

__all__ = [
    "build_chat_model",
    "build_fallback",
    "build_primary",
]

# Ollama Cloud host marker: a `base_url` on this host is the cloud path and so
# *requires* a key; a local `base_url` (e.g. http://localhost:11434) does not.
_OLLAMA_CLOUD_HOST = "ollama.com"

# OpenRouter has no dedicated LangChain package — it is `ChatOpenAI` aimed at this
# fixed OpenAI-compatible gateway with its own `OPENROUTER_API_KEY`.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _require(value: str | None, env_name: str) -> str:
    """Return ``value`` or raise a clear build-time error naming ``env_name``.

    Every builder funnels its required-secret checks through here so the message
    is uniform and the failure happens at build time, never at import.
    """
    if not value:
        raise RuntimeError(
            f"{env_name} is required to build a chat model; set it in the environment."
        )
    return value


def _build_ollama(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatOllama`` (SPEC D4 default provider).

    Cloud (``base_url`` on ``ollama.com``) **requires** ``OLLAMA_API_KEY``; a local
    ``base_url`` needs none. The bearer ``Authorization`` header is attached iff a
    key is present (so a local run may still authenticate a proxy). ``reasoning``
    (the non-thinking flash route) and ``temperature=0`` thread the D4 contract;
    ``reasoning=`` is Ollama-only — never pass it to another provider's class.
    """
    from langchain_ollama import ChatOllama

    if _OLLAMA_CLOUD_HOST in config.ollama_base_url:
        _require(config.ollama_api_key, "OLLAMA_API_KEY")

    client_kwargs: dict[str, Any] = {}
    if config.ollama_api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {config.ollama_api_key}"}

    return ChatOllama(
        model=model_name,
        base_url=config.ollama_base_url,
        client_kwargs=client_kwargs,
        temperature=config.model_temperature,
        reasoning=config.model_reasoning,
    )


def _build_openai(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatOpenAI`` (OpenAI, or any OpenAI-compatible endpoint).

    Requires ``OPENAI_API_KEY``; honours ``OPENAI_BASE_URL`` when set (a proxy or
    self-hosted gateway), otherwise ``ChatOpenAI`` targets OpenAI's own host.
    ``temperature=0`` threads the D4 contract; ``reasoning=`` is Ollama-only and is
    never passed here — the o-series non-thinking route is an operator model-id
    choice (SPEC Open Q2), not a constructor flag.
    """
    from langchain_openai import ChatOpenAI

    api_key = _require(config.openai_api_key, "OPENAI_API_KEY")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": config.model_temperature,
    }
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    return ChatOpenAI(**kwargs)


def _build_openrouter(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatOpenAI`` pointed at OpenRouter's OpenAI-compatible gateway.

    OpenRouter reuses ``ChatOpenAI`` with a fixed ``base_url`` and its own
    ``OPENROUTER_API_KEY`` (there is no dedicated LangChain package). ``temperature
    =0``; no ``reasoning=`` (Ollama-only).
    """
    from langchain_openai import ChatOpenAI

    api_key = _require(config.openrouter_api_key, "OPENROUTER_API_KEY")
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=_OPENROUTER_BASE_URL,
        temperature=config.model_temperature,
    )


def _build_anthropic(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatAnthropic`` (Anthropic's Claude models).

    Requires ``ANTHROPIC_API_KEY``; ``temperature=0`` threads the D4 contract.
    ``reasoning=`` is Ollama-only and is never passed here — Anthropic's extended
    thinking is an operator model-id / separate-flag choice (SPEC Open Q2), not
    this constructor flag.
    """
    from langchain_anthropic import ChatAnthropic

    api_key = _require(config.anthropic_api_key, "ANTHROPIC_API_KEY")
    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        temperature=config.model_temperature,
    )


def _build_groq(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatGroq`` (Groq's hosted low-latency inference).

    Requires ``GROQ_API_KEY``; ``temperature=0`` threads the D4 contract.
    ``reasoning=`` is Ollama-only and is never passed here — Groq exposes its
    reasoning models through the model id (SPEC Open Q2), not this constructor.
    """
    from langchain_groq import ChatGroq

    api_key = _require(config.groq_api_key, "GROQ_API_KEY")
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=config.model_temperature,
    )


def _build_google(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatGoogleGenerativeAI`` (Google's Gemini models).

    Requires ``GOOGLE_API_KEY``; ``temperature=0`` threads the D4 contract.
    ``api_key`` is the accepted alias for the class's ``google_api_key`` field, so
    this stays consistent with the other builders. ``reasoning=`` is Ollama-only
    and is never passed here — Gemini's thinking budget is an operator model-id /
    separate-parameter choice (SPEC Open Q2), not this constructor flag.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = _require(config.google_api_key, "GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=config.model_temperature,
    )


# Provider registry: slug → builder. Referencing a builder does NOT import its
# SDK (the import is lazy inside the body); only the selected provider's package
# is ever loaded. New providers land as one `_build_*` + one entry here.
_BUILDERS: dict[str, Callable[[FundConfig, str], BaseChatModel]] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
    "openrouter": _build_openrouter,
    "anthropic": _build_anthropic,
    "groq": _build_groq,
    "google": _build_google,
}


def build_chat_model(config: FundConfig, *, model_name: str) -> BaseChatModel:
    """Return a ``BaseChatModel`` for ``model_name`` from the provider registry.

    Dispatches on ``config.llm_provider``:

    * unknown provider → :class:`RuntimeError` listing the valid providers;
    * a provider whose optional extra is not installed → the builder's
      ``ImportError`` is caught and re-raised as a :class:`RuntimeError` naming the
      ``uv sync --package portopt-fund --extra <provider>`` command;
    * required-secret checks happen inside the builder (build time, not import).
    """
    try:
        builder = _BUILDERS[config.llm_provider]
    except KeyError:
        valid = ", ".join(sorted(_BUILDERS))
        raise RuntimeError(
            f"Unknown LLM provider {config.llm_provider!r}; valid providers: {valid}."
        ) from None

    try:
        return builder(config, model_name)
    except ImportError as exc:
        raise RuntimeError(
            f"The {config.llm_provider!r} provider needs its optional extra; "
            f"install it with `uv sync --package portopt-fund "
            f"--extra {config.llm_provider}`."
        ) from exc


def build_primary(config: FundConfig = settings) -> BaseChatModel:
    """Build the primary model (``config.primary_model``)."""
    return build_chat_model(config, model_name=config.primary_model)


def build_fallback(config: FundConfig = settings) -> BaseChatModel:
    """Build the fallback model (``config.fallback_model``)."""
    return build_chat_model(config, model_name=config.fallback_model)
