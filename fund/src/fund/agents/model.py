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

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fund.config import FundConfig, settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

__all__ = [
    "build_chat_model",
    "build_fallback",
    "build_primary",
    "effective_temperature",
]

# Ollama Cloud host marker: a `base_url` on this host is the cloud path and so
# *requires* a key; a local `base_url` (e.g. http://localhost:11434) does not.
_OLLAMA_CLOUD_HOST = "ollama.com"

# Install-hint overrides for providers whose SDK is not shipped by an extra named
# after them: openrouter/microsoft reuse `langchain-openai` (the `openai` extra);
# ollama is a hard dependency, so there is no extra to suggest (None).
_INSTALL_EXTRA: dict[str, str | None] = {
    "ollama": None,
    "openrouter": "openai",
    "microsoft": "openai",
}

# OpenRouter has no dedicated LangChain package — it is `ChatOpenAI` aimed at this
# fixed OpenAI-compatible gateway with its own `OPENROUTER_API_KEY`.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# httpx and google-genai default to *no* request deadline, so a half-open
# connection would block the daemon's single worker thread forever (the heartbeat
# thread keeps the job lease fresh, so nothing reaps it). The read timeout is
# per received chunk, not per generation: it bounds a stalled stream while still
# leaving room for a slow time-to-first-token on a large context.
_CONNECT_TIMEOUT_SECONDS = 10.0
_READ_TIMEOUT_SECONDS = 180.0

# Model families whose APIs reject any non-default sampling temperature: OpenAI's
# o-series ("o1", "o3-mini", "o4-mini", ...) answers HTTP 400, and langchain-anthropic
# raises client-side for the Fable family. The fallback model goes through the same
# builder, so sending the temperature=0 contract anyway would fail every request
# with nothing for ModelFallbackMiddleware to fall back to.
_OPENAI_REASONING_MODEL = re.compile(r"o\d")
_ANTHROPIC_FIXED_SAMPLING_PREFIX = "claude-fable-5"

# Explicit completion budget for the providers whose SDK wrapper otherwise caps
# output silently (ChatNVIDIA at 1024 tokens, the Hugging Face wrapper at 512). A
# PM `task` brief, an allocator report or a tool-call argument above the cap is
# cut off mid-JSON and surfaces as an invalid tool call that loops until the
# recursion limit. Reasoning models also spend this budget on thinking tokens.
_MAX_COMPLETION_TOKENS = 4096

# ChatNVIDIA treats a falsy `api_key` as unset and falls back to NVIDIA_API_KEY in
# the process env, which would put an operator's hosted key on the wire to a keyless
# NIM. Any truthy value keeps the env var out of the client; the placeholder is
# stripped again before the model is returned, so no Authorization header is sent.
_KEYLESS_API_KEY_PLACEHOLDER = "unused"

# ChatNVIDIA's own definition of the hosted endpoint (`is_hosted` in its client).
_NVIDIA_HOSTED_HOSTS = frozenset({"integrate.api.nvidia.com", "ai.api.nvidia.com"})


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


class _BearerAuth:
    """httpx auth hook that attaches a bearer token while keeping it out of ``repr``.

    ``ChatOllama`` stores ``client_kwargs`` as a plain dict, and LangChain falls
    back to ``repr(model)`` as the serialized form it hands every callback and
    tracer (LangSmith, debug logging). A token placed in a ``headers`` dict would
    therefore be exported verbatim; httpx accepts any ``Request -> Request``
    callable as ``auth``, so the secret lives only on this object.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def __call__(self, request: Any) -> Any:
        request.headers["Authorization"] = f"Bearer {self._token}"
        return request

    def __repr__(self) -> str:
        return "_BearerAuth(token='***')"


def _rejects_temperature(provider: str, model_name: str) -> bool:
    """Return whether ``provider`` refuses a non-default temperature for the model."""
    if provider in ("openai", "microsoft"):
        return _OPENAI_REASONING_MODEL.match(model_name) is not None
    if provider == "anthropic":
        return model_name.startswith(_ANTHROPIC_FIXED_SAMPLING_PREFIX)
    return False


def effective_temperature(config: FundConfig, model_name: str) -> float | None:
    """Return the temperature the selected provider is actually sent for a model.

    The builders and the run audit record both go through this, so the audit row
    cannot claim a deterministic ``temperature=0`` for a model that was built
    without one.

    Args:
        config: Run config carrying the D4 ``model_temperature`` contract and the
            selected ``llm_provider``.
        model_name: Model id the chat model is built for.

    Returns:
        ``config.model_temperature``, or ``None`` when the model family rejects a
        non-default value and the SDK's own sampling default applies.
    """
    if _rejects_temperature(config.llm_provider, model_name):
        return None
    return config.model_temperature


def _temperature_kwargs(config: FundConfig, model_name: str) -> dict[str, float]:
    """Return the ``temperature`` kwarg, or nothing for a model that rejects it."""
    temperature = effective_temperature(config, model_name)
    return {} if temperature is None else {"temperature": temperature}


def _build_ollama(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatOllama`` (SPEC D4 default provider).

    Cloud (``base_url`` on ``ollama.com``) **requires** ``OLLAMA_API_KEY``; a local
    ``base_url`` needs none but is given an explicit ``num_ctx``
    (:attr:`FundConfig.ollama_num_ctx`): a local server otherwise applies its own
    small default window and silently truncates the oldest tokens (system prompt,
    tool schemas). The bearer ``Authorization`` header is attached iff a key is
    present (so a local run may still authenticate a proxy). ``reasoning``
    (the non-thinking flash route) and ``temperature=0`` thread the D4 contract;
    ``reasoning=`` is Ollama-only — never pass it to another provider's class.
    """
    import httpx
    from langchain_ollama import ChatOllama

    is_cloud = _OLLAMA_CLOUD_HOST in config.ollama_base_url
    if is_cloud:
        _require(config.ollama_api_key, "OLLAMA_API_KEY")

    client_kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(
            _READ_TIMEOUT_SECONDS, connect=_CONNECT_TIMEOUT_SECONDS
        ),
    }
    if config.ollama_api_key:
        client_kwargs["auth"] = _BearerAuth(config.ollama_api_key)

    context_kwargs: dict[str, Any] = (
        {} if is_cloud else {"num_ctx": config.ollama_num_ctx}
    )

    return ChatOllama(
        model=model_name,
        base_url=config.ollama_base_url,
        client_kwargs=client_kwargs,
        temperature=config.model_temperature,
        reasoning=config.model_reasoning,
        **context_kwargs,
    )


def _build_openai(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatOpenAI`` (OpenAI, or any OpenAI-compatible endpoint).

    Requires ``OPENAI_API_KEY``; honours ``OPENAI_BASE_URL`` when set (a proxy or
    self-hosted gateway), otherwise ``ChatOpenAI`` targets OpenAI's own host.
    ``temperature=0`` threads the D4 contract except for o-series reasoning models,
    which reject it; ``reasoning=`` is Ollama-only and is never passed here — the
    o-series non-thinking route is an operator model-id choice (SPEC Open Q2), not a
    constructor flag.
    """
    from langchain_openai import ChatOpenAI

    api_key = _require(config.openai_api_key, "OPENAI_API_KEY")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        **_temperature_kwargs(config, model_name),
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
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": _OPENROUTER_BASE_URL,
        "temperature": config.model_temperature,
    }
    return ChatOpenAI(**kwargs)


def _build_anthropic(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatAnthropic`` (Anthropic's Claude models).

    Requires ``ANTHROPIC_API_KEY``; ``temperature=0`` threads the D4 contract except
    for the Fable family, which rejects it. ``reasoning=`` is Ollama-only and is
    never passed here — Anthropic's extended thinking is an operator model-id /
    separate-flag choice (SPEC Open Q2), not this constructor flag.
    """
    from langchain_anthropic import ChatAnthropic

    api_key = _require(config.anthropic_api_key, "ANTHROPIC_API_KEY")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        **_temperature_kwargs(config, model_name),
    }
    return ChatAnthropic(**kwargs)


def _build_groq(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatGroq`` (Groq's hosted low-latency inference).

    Requires ``GROQ_API_KEY``; ``temperature=0`` threads the D4 contract.
    ``reasoning=`` is Ollama-only and is never passed here — Groq exposes its
    reasoning models through the model id (SPEC Open Q2), not this constructor.
    """
    from langchain_groq import ChatGroq

    api_key = _require(config.groq_api_key, "GROQ_API_KEY")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": config.model_temperature,
    }
    return ChatGroq(**kwargs)


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
        request_timeout=_READ_TIMEOUT_SECONDS,
    )


def _build_aws(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatBedrockConverse`` (AWS Bedrock, Converse API).

    Bedrock is **region-scoped, not key-scoped**: credentials come from the boto3
    chain (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``
    or an IAM role), so the only secret this builder *requires* is the region
    (``AWS_REGION``). ``model`` is the accepted alias for the class's ``model_id``
    field, keeping this consistent with the other builders. ``temperature=0``
    threads the D4 contract; ``reasoning=`` is Ollama-only and is never passed here.
    TLS to Bedrock is a boto3 concern — set ``AWS_CA_BUNDLE`` (documented, not code).
    """
    from langchain_aws import ChatBedrockConverse

    region = _require(config.aws_region, "AWS_REGION")
    return ChatBedrockConverse(
        model=model_name,
        region_name=region,
        temperature=config.model_temperature,
    )


def _build_microsoft(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build an ``AzureChatOpenAI`` (Azure OpenAI Service).

    Azure reuses the ``openai`` extra (``langchain-openai``) — no separate package.
    Unlike public OpenAI, Azure is wired from **four** env vars, each *required*
    (``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``, ``OPENAI_API_VERSION``,
    ``AZURE_OPENAI_DEPLOYMENT_NAME``); a missing one raises via :func:`_require`
    naming it. The deployment is always the one in ``config``; ``build_fallback``
    swaps in ``AZURE_OPENAI_FALLBACK_DEPLOYMENT_NAME`` (see :func:`_fallback_config`).
    ``model`` / ``api_key`` / ``api_version`` / ``azure_deployment`` are the class's
    accepted aliases, so this stays consistent with the other builders.
    ``temperature=0`` threads the D4 contract except for o-series reasoning models,
    which reject it; ``reasoning=`` is Ollama-only and is never passed here.
    """
    from langchain_openai import AzureChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": _require(config.azure_openai_api_key, "AZURE_OPENAI_API_KEY"),
        "azure_endpoint": _require(
            config.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT"
        ),
        "api_version": _require(config.azure_openai_api_version, "OPENAI_API_VERSION"),
        "azure_deployment": _require(
            config.azure_openai_deployment_name, "AZURE_OPENAI_DEPLOYMENT_NAME"
        ),
        **_temperature_kwargs(config, model_name),
    }
    return AzureChatOpenAI(**kwargs)


def _is_nvidia_hosted(base_url: str) -> bool:
    """Return whether ``base_url`` is NVIDIA's own hosted API, reached over TLS."""
    parsed = urlparse(base_url)
    return parsed.scheme == "https" and parsed.hostname in _NVIDIA_HOSTED_HOSTS


def _nvidia_api_key(config: FundConfig) -> str | None:
    """Return the key to send to NVIDIA, or ``None`` for an endpoint that gets none.

    ``NVIDIA_API_KEY`` authenticates to NVIDIA's hosted API. ``load_config`` fills
    it from the same variable an operator keeps their hosted key in, so a custom
    ``NVIDIA_BASE_URL`` (a self-hosted NIM, often plain HTTP) only receives it when
    the operator opts in with ``NVIDIA_BASE_URL_SEND_KEY`` for a keyed gateway.

    Raises:
        RuntimeError: A key is called for (hosted API, or opted in) but unset.
    """
    base_url = config.nvidia_base_url
    if not base_url or _is_nvidia_hosted(base_url) or config.nvidia_base_url_send_key:
        return _require(config.nvidia_api_key, "NVIDIA_API_KEY")
    return None


def _build_nvidia(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatNVIDIA`` (NVIDIA NIM), hosted or self-hosted.

    Branches on ``nvidia_base_url``:

    * **hosted** (``build.nvidia.com``, whether ``NVIDIA_BASE_URL`` is unset or spells
      out NVIDIA's own https URL) *requires* ``NVIDIA_API_KEY``;
    * **custom endpoint** (``NVIDIA_BASE_URL`` set, e.g. ``http://host:8000/v1``)
      routes by ``base_url`` and is sent **no** credential, not even an
      ``NVIDIA_API_KEY`` present in the process env. A keyed gateway opts in with
      ``NVIDIA_BASE_URL_SEND_KEY=true`` (see :func:`_nvidia_api_key`).

    ``temperature=0`` threads the D4 contract and ``max_completion_tokens`` is set
    explicitly because ``ChatNVIDIA`` otherwise caps every completion at 1024 tokens;
    ``reasoning=`` is Ollama-only and is never passed here.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": config.model_temperature,
        "max_completion_tokens": _MAX_COMPLETION_TOKENS,
    }
    if config.nvidia_base_url:
        kwargs["base_url"] = config.nvidia_base_url
    api_key = _nvidia_api_key(config)
    if api_key:
        return ChatNVIDIA(api_key=api_key, **kwargs)

    llm = ChatNVIDIA(api_key=_KEYLESS_API_KEY_PLACEHOLDER, **kwargs)
    # Both clients keep their own copy. A renamed attribute in a future SDK leaves
    # the placeholder in place, which is harmless, rather than leaking a real key.
    for client_attr in ("_client", "_async_client"):
        client = getattr(llm, client_attr, None)
        if client is not None:
            client.api_key = None
    return llm


def _build_huggingface(config: FundConfig, model_name: str) -> BaseChatModel:
    """Build a ``ChatHuggingFace`` over the Hugging Face Inference API (cloud-only).

    Cloud path only: a ``HuggingFaceEndpoint`` (the Inference API, ``repo_id=`` the
    model, ``HUGGINGFACEHUB_API_TOKEN`` required) wrapped in ``ChatHuggingFace``.
    ``FUND_HF_MODE=local`` is **deferred** and raises a clear :class:`RuntimeError`
    *before* the import — the ``huggingface`` extra deliberately ships no
    torch/transformers, so local inference is not yet supported. ``model_id`` is
    passed explicitly so construction resolves no tokenizer over the network
    (build time stays request-free). ``temperature=0`` threads the D4 contract and
    ``max_tokens`` is set explicitly, both on the wrapper itself: ``ChatHuggingFace``
    inherits endpoint properties only when truthy, so a ``0.0`` temperature would be
    dropped and the endpoint's 512-token default would cap every completion.
    ``reasoning=`` is Ollama-only and is never passed here.
    """
    if config.hf_mode != "cloud":
        raise RuntimeError(
            f"FUND_HF_MODE={config.hf_mode!r} (local Hugging Face inference) is not "
            "yet supported; only cloud mode (the Inference API) is available. Set "
            "FUND_HF_MODE=cloud."
        )

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    token = _require(config.hf_api_token, "HUGGINGFACEHUB_API_TOKEN")
    endpoint_kwargs: dict[str, Any] = {
        "repo_id": model_name,
        "huggingfacehub_api_token": token,
        "temperature": config.model_temperature,
        "max_new_tokens": _MAX_COMPLETION_TOKENS,
    }
    endpoint = HuggingFaceEndpoint(**endpoint_kwargs)
    return ChatHuggingFace(
        llm=endpoint,
        model_id=model_name,
        temperature=config.model_temperature,
        max_tokens=_MAX_COMPLETION_TOKENS,
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
    "aws": _build_aws,
    "microsoft": _build_microsoft,
    "nvidia": _build_nvidia,
    "huggingface": _build_huggingface,
}


def build_chat_model(config: FundConfig, *, model_name: str) -> BaseChatModel:
    """Return a ``BaseChatModel`` for ``model_name`` from the provider registry.

    Dispatches on ``config.llm_provider``:

    * unknown provider → :class:`RuntimeError` listing the valid providers;
    * a provider whose optional extra is not installed → the builder's
      ``ImportError`` is caught and re-raised as a :class:`RuntimeError` naming the
      ``uv sync --package portopt-fund --extra <extra>`` command, where ``<extra>``
      is the extra that actually ships the SDK (``openai`` for ``openrouter`` and
      ``microsoft``);
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
        extra = _INSTALL_EXTRA.get(config.llm_provider, config.llm_provider)
        if extra is None:
            raise RuntimeError(
                f"The {config.llm_provider!r} provider's package is a core "
                f"dependency of portopt-fund but failed to import; reinstall "
                f"with `uv sync --package portopt-fund`."
            ) from exc
        raise RuntimeError(
            f"The {config.llm_provider!r} provider needs its optional extra; "
            f"install it with `uv sync --package portopt-fund --extra {extra}`."
        ) from exc


def _fallback_config(config: FundConfig) -> FundConfig:
    """Return the config the fallback model is built from.

    Azure routes a request by the deployment in the URL path and treats the model
    label as metadata, so the fallback only differs from the primary when it is built
    on its own deployment. The role is known here and not in the builder, which sees
    only a model label that may legitimately equal the primary's: the same model on a
    second deployment is the usual Azure resilience setup.
    """
    if (
        config.llm_provider == "microsoft"
        and config.azure_openai_fallback_deployment_name
    ):
        return replace(
            config,
            azure_openai_deployment_name=config.azure_openai_fallback_deployment_name,
        )
    return config


def build_primary(config: FundConfig = settings) -> BaseChatModel:
    """Build the primary model (``config.primary_model``)."""
    return build_chat_model(config, model_name=config.primary_model)


def build_fallback(config: FundConfig = settings) -> BaseChatModel:
    """Build the fallback model (``config.fallback_model``).

    On Azure a configured ``AZURE_OPENAI_FALLBACK_DEPLOYMENT_NAME`` replaces the
    primary deployment for this model, whatever the two model ids are.
    """
    return build_chat_model(_fallback_config(config), model_name=config.fallback_model)
