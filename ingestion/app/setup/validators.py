"""Live API-key validators for the portopt install wizard (SPEC D8).

Each validator issues one cheap request and returns ``True`` on HTTP 200,
``False`` on a definitive non-200 answer (invalid key), or raises
``ValidationNetworkError`` when the service gave no verdict: the request itself
failed (network/timeout) or it answered 429/5xx (throttled/outage). That lets the
wizard tell "bad key" (retry) apart from "service unavailable" (abort/inform).

The LLM-backend validator (``validate_llm()``, SPEC "Switchable LLM Backends")
probes the fund's ten switchable providers over **httpx only** — never an
agent-stack or provider SDK import, so ingestion keeps its ``⊬ agent stack``
boundary. Most providers get a cheap authed GET (200 → ok, 401/403 → bad key);
``aws`` (Bedrock, region-scoped via the boto3 chain) and ``microsoft`` (Azure,
four wired fields) have no cheap authed REST probe, so they are presence-only
(+ endpoint reachability for Azure) — SPEC Open Q2.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit

import httpx
import requests

_TIMEOUT = 10

_T212_BASE = {
    "live": "https://live.trading212.com",
    "demo": "https://demo.trading212.com",
}


class ValidationNetworkError(Exception):
    """Raised when a validation request cannot reach the service."""


def _get(url: str, **kwargs: object) -> requests.Response:
    try:
        return requests.get(url, timeout=_TIMEOUT, **kwargs)  # type: ignore[arg-type]
    except requests.RequestException as exc:
        # str(exc) embeds the full request URL, and the FRED key rides in its query
        # string; the CLI echoes this message to stderr/CI logs. Raise `from None`
        # so a traceback cannot resurface the URL through the chained cause either.
        host = urlsplit(url).hostname
        raise ValidationNetworkError(
            f"{type(exc).__name__} contacting {host}"
        ) from None


def _accepted(status_code: int, url: str) -> bool:
    """Map a probe's HTTP status to "key accepted".

    Raises:
        ValidationNetworkError: The service answered 429 or 5xx, which says nothing
            about the key; reporting it as invalid would send the user hunting for
            a typo in a correct credential. Only the host is named because ``url``
            can carry the key in its query string.
    """
    if status_code == 429 or status_code >= 500:
        host = urlsplit(url).hostname
        raise ValidationNetworkError(
            f"{host} answered HTTP {status_code} (rate-limited or unavailable), so "
            "the key could not be checked; retry shortly or use --skip-validation."
        )
    return status_code == 200


def validate_t212(api_key: str, secret_key: str, *, mode: str = "live") -> bool:
    """True if the Trading212 metadata endpoint accepts the key/secret pair.

    Raises:
        ValidationNetworkError: Trading212 is unreachable, throttling or erroring.
    """
    base = _T212_BASE.get(mode, _T212_BASE["live"])
    token = base64.b64encode(f"{api_key}:{secret_key}".encode()).decode()
    url = f"{base}/api/v0/equity/metadata/exchanges"
    resp = _get(url, headers={"Authorization": f"Basic {token}"})
    return _accepted(resp.status_code, url)


def validate_fred(api_key: str) -> bool:
    """True if the FRED API accepts the key on a minimal series request.

    Raises:
        ValidationNetworkError: FRED is unreachable, throttling or erroring.
    """
    url = "https://api.stlouisfed.org/fred/series"
    resp = _get(
        url,
        params={"series_id": "GNPCA", "api_key": api_key, "file_type": "json"},
    )
    return _accepted(resp.status_code, url)


# --- LLM backend validators (SPEC "Switchable LLM Backends", httpx only) -------

# The ten switchable providers the fund can select (matches fund's builder
# registry). Exposed so the wizard offers exactly this set (T5).
SUPPORTED_LLM_PROVIDERS = (
    "ollama",
    "openai",
    "openrouter",
    "anthropic",
    "google",
    "groq",
    "nvidia",
    "huggingface",
    "aws",
    "microsoft",
)

# Per-provider cheapest authed endpoints. A local/self-hosted `base_url` (ollama,
# nvidia NIM, an OpenAI-compatible proxy) overrides the hosted default.
_OLLAMA_CLOUD_HOST = "ollama.com"
_OLLAMA_BASE = "https://ollama.com"
_OPENAI_BASE = "https://api.openai.com/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_ANTHROPIC_BASE = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
_GROQ_BASE = "https://api.groq.com/openai/v1"
_GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta"
_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
_HF_WHOAMI = "https://huggingface.co/api/whoami-v2"


def _llm_get(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    """One cheap httpx GET; connect/timeout → ``ValidationNetworkError``.

    Isolated so tests mock a single seam and the network→exception mapping lives
    in one place. httpx only — no agent-stack/provider SDK ever imported here.
    """
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            return client.get(url, headers=headers, params=params)
    except httpx.RequestError as exc:
        raise ValidationNetworkError(str(exc)) from exc


def _bearer_get_ok(url: str, key: str | None) -> bool:
    """True iff ``key`` is present and a bearer GET to ``url`` returns 200."""
    if not key:
        return False
    resp = _llm_get(url, headers={"Authorization": f"Bearer {key}"})
    return _accepted(resp.status_code, url)


def _v_openai(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    return _bearer_get_ok(f"{(base_url or _OPENAI_BASE).rstrip('/')}/models", key)


def _v_openrouter(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    # OpenRouter's /models is public, so probe the authed /auth/key instead.
    return _bearer_get_ok(f"{(base_url or _OPENROUTER_BASE).rstrip('/')}/auth/key", key)


def _v_groq(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    return _bearer_get_ok(f"{(base_url or _GROQ_BASE).rstrip('/')}/models", key)


def _v_anthropic(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    if not key:
        return False
    url = f"{(base_url or _ANTHROPIC_BASE).rstrip('/')}/v1/models"
    resp = _llm_get(
        url, headers={"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION}
    )
    return _accepted(resp.status_code, url)


def _v_google(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    if not key:
        return False
    url = f"{(base_url or _GOOGLE_BASE).rstrip('/')}/models"
    resp = _llm_get(url, params={"key": key})
    return _accepted(resp.status_code, url)


def _v_huggingface(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    # Cloud-only; a shallow authed whoami GET is the cheapest key check.
    return _bearer_get_ok(_HF_WHOAMI, key)


def _v_ollama(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    # Cloud (ollama.com) requires a key; a local base_url needs none. Attach the
    # bearer only when a key is present (server-down → connect → raise).
    base = (base_url or _OLLAMA_BASE).rstrip("/")
    if _OLLAMA_CLOUD_HOST in base and not key:
        return False
    headers = {"Authorization": f"Bearer {key}"} if key else None
    url = f"{base}/api/tags"
    return _accepted(_llm_get(url, headers=headers).status_code, url)


def _v_nvidia(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    # Self-hosted NIM (base_url set) needs no key — reachability only; hosted
    # (build.nvidia.com) requires the key.
    if base_url:
        url = f"{base_url.rstrip('/')}/models"
        return _accepted(_llm_get(url).status_code, url)
    return _bearer_get_ok(f"{_NVIDIA_BASE}/models", key)


def _v_aws(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    # Presence-only (Open Q2): Bedrock is region-scoped; credentials come from the
    # boto3 chain (env / IAM role), so there is no cheap authed REST probe (SigV4).
    # `region` is the one required field — pass it via base_url or region=.
    return bool(fields.get("region") or base_url)


def _v_microsoft(
    key: str | None, base_url: str | None, fields: Mapping[str, str | None]
) -> bool:
    # Presence-only + endpoint reachability (Open Q2): Azure OpenAI is wired from
    # four required fields. A bad key can't be told apart cheaply, so any HTTP
    # response = reachable = ok; an unreachable endpoint raises.
    endpoint = fields.get("endpoint") or base_url
    if not (
        key and endpoint and fields.get("api_version") and fields.get("deployment_name")
    ):
        return False
    _llm_get(endpoint)  # reachability only (raises ValidationNetworkError if not)
    return True


_LLM_VALIDATORS: dict[
    str, Callable[[str | None, str | None, Mapping[str, str | None]], bool]
] = {
    "ollama": _v_ollama,
    "openai": _v_openai,
    "openrouter": _v_openrouter,
    "anthropic": _v_anthropic,
    "google": _v_google,
    "groq": _v_groq,
    "nvidia": _v_nvidia,
    "huggingface": _v_huggingface,
    "aws": _v_aws,
    "microsoft": _v_microsoft,
}


def validate_llm(
    provider: str,
    key: str | None = None,
    base_url: str | None = None,
    *,
    skip_validation: bool = False,
    **fields: str | None,
) -> bool:
    """Validate an LLM provider's credentials with one cheap httpx probe.

    Returns ``True`` when the provider accepts the credentials (or, for the
    presence-only providers, when the required fields are present and the endpoint
    is reachable); ``False`` on a bad/absent key (the wizard re-prompts); raises
    ``ValidationNetworkError`` when the service is unreachable or answers 429/5xx
    (abort/inform; neither says anything about the key).

    Args:
        provider: One of ``SUPPORTED_LLM_PROVIDERS``.
        key: The provider auth key/token (``None`` for keyless local/self-hosted
            or presence-only providers).
        base_url: Override the hosted default (local ollama, self-hosted NIM, an
            OpenAI-compatible proxy, or the Azure/AWS endpoint/region).
        skip_validation: When ``True``, return ``True`` immediately with **no**
            network call — the wizard's ``--skip-validation`` escape hatch.
        **fields: Provider-specific extras — ``region`` (aws); ``endpoint`` /
            ``api_version`` / ``deployment_name`` (microsoft/Azure).

    httpx only — never an agent-stack or provider SDK import (ingestion ⊬ agent stack).
    """
    if skip_validation:
        return True
    try:
        validator = _LLM_VALIDATORS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}; expected one of "
            f"{', '.join(SUPPORTED_LLM_PROVIDERS)}"
        ) from None
    return validator(key, base_url, fields)
