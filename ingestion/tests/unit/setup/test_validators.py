"""On-the-spot key-validation contract (SPEC D8 wizard).

Each validator makes one cheap request and returns True (200) / False (auth or
other non-200), or raises `ValidationNetworkError` when the request itself fails
(so the wizard distinguishes "bad key" from "network down"). No real network:
`requests.get` is patched, matching the repo's Trading212-client test style.
"""

from __future__ import annotations

import traceback
from unittest.mock import MagicMock, patch

import requests

from app.setup import validators


def _resp(status: int) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    return r


@patch("app.setup.validators.requests.get")
def test_validate_t212_success_sends_basic_auth(mock_get: MagicMock) -> None:
    mock_get.return_value = _resp(200)
    assert validators.validate_t212("k", "s") is True
    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["Authorization"].startswith("Basic ")


@patch("app.setup.validators.requests.get")
def test_validate_t212_auth_fail(mock_get: MagicMock) -> None:
    mock_get.return_value = _resp(403)
    assert validators.validate_t212("k", "s") is False


@patch("app.setup.validators.requests.get")
def test_validate_fred_success_sends_series_and_key(mock_get: MagicMock) -> None:
    mock_get.return_value = _resp(200)
    assert validators.validate_fred("fred-key") is True
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["series_id"] == "GNPCA"
    assert kwargs["params"]["api_key"] == "fred-key"


@patch("app.setup.validators.requests.get")
def test_validate_fred_bad_key(mock_get: MagicMock) -> None:
    mock_get.return_value = _resp(400)
    assert validators.validate_fred("bad") is False


# --- validate_llm: httpx-only per-provider probe, no SDK import ---------------

import httpx
import pytest

from app.setup.validators import ValidationNetworkError

# Providers that do a real authed probe (200 -> ok, 401/403 -> bad, connect ->
# raise). ollama(cloud+key) and nvidia(hosted+key) belong here; aws + microsoft
# are presence-only and tested separately.
_AUTHED_PROVIDERS = (
    "ollama",
    "openai",
    "openrouter",
    "anthropic",
    "google",
    "groq",
    "huggingface",
    "nvidia",
)


def test_supported_llm_providers_are_the_ten() -> None:
    assert set(validators.SUPPORTED_LLM_PROVIDERS) == {
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
    }


@pytest.mark.parametrize("provider", _AUTHED_PROVIDERS)
def test_validate_llm_authed_ok_on_200(provider: str) -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(200)) as m:
        assert validators.validate_llm(provider, key="k") is True
    m.assert_called_once()


@pytest.mark.parametrize("provider", _AUTHED_PROVIDERS)
@pytest.mark.parametrize("status", [401, 403])
def test_validate_llm_authed_bad_key_on_401_403(provider: str, status: int) -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(status)):
        assert validators.validate_llm(provider, key="k") is False


@pytest.mark.parametrize("provider", _AUTHED_PROVIDERS)
def test_validate_llm_authed_connect_error_raises(provider: str) -> None:
    with patch(
        "app.setup.validators._llm_get",
        side_effect=ValidationNetworkError("down"),
    ):
        with pytest.raises(ValidationNetworkError):
            validators.validate_llm(provider, key="k")


def test_validate_llm_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError):
        validators.validate_llm("not-a-provider", key="k")


def test_validate_llm_skip_validation_never_touches_network() -> None:
    with patch("app.setup.validators._llm_get") as m:
        assert validators.validate_llm("openai", key="k", skip_validation=True) is True
    m.assert_not_called()


def test_validate_llm_openai_uses_bearer_and_models_endpoint() -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(200)) as m:
        assert validators.validate_llm("openai", key="sk-x") is True
    (url,), kwargs = m.call_args
    assert url.endswith("/models")
    assert kwargs["headers"]["Authorization"] == "Bearer sk-x"


def test_validate_llm_anthropic_uses_x_api_key_header() -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(200)) as m:
        assert validators.validate_llm("anthropic", key="sk-a") is True
    _, kwargs = m.call_args
    assert kwargs["headers"]["x-api-key"] == "sk-a"
    assert "anthropic-version" in kwargs["headers"]


def test_validate_llm_google_passes_key_as_query_param() -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(200)) as m:
        assert validators.validate_llm("google", key="goog") is True
    _, kwargs = m.call_args
    assert kwargs["params"]["key"] == "goog"


def test_validate_llm_ollama_cloud_without_key_is_bad_without_network() -> None:
    # Cloud (ollama.com) requires a key; no key -> bad, and no probe is issued.
    with patch("app.setup.validators._llm_get") as m:
        assert validators.validate_llm("ollama") is False
    m.assert_not_called()


def test_validate_llm_ollama_local_needs_no_key() -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(200)) as m:
        assert (
            validators.validate_llm("ollama", base_url="http://localhost:11434") is True
        )
    (url,), kwargs = m.call_args
    assert url.endswith("/api/tags")
    assert kwargs["headers"] is None  # no bearer attached for a keyless local run


def test_validate_llm_nvidia_self_hosted_needs_no_key() -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(200)) as m:
        assert validators.validate_llm("nvidia", base_url="http://nim:8000/v1") is True
    m.assert_called_once()


def test_validate_llm_aws_presence_only_no_network() -> None:
    with patch("app.setup.validators._llm_get") as m:
        assert validators.validate_llm("aws", region="eu-west-1") is True
        assert validators.validate_llm("aws") is False
    m.assert_not_called()


def test_validate_llm_microsoft_presence_plus_reachability() -> None:
    full = {
        "key": "az-key",
        "endpoint": "https://az.openai.azure.com",
        "api_version": "2024-06-01",
        "deployment_name": "gpt4o",
    }
    with patch("app.setup.validators._llm_get", return_value=_resp(404)) as m:
        # Reachable (any HTTP response) with all four fields present -> ok.
        assert validators.validate_llm("microsoft", **full) is True
    m.assert_called_once()


def test_validate_llm_microsoft_missing_field_is_bad_without_network() -> None:
    with patch("app.setup.validators._llm_get") as m:
        assert (
            validators.validate_llm(
                "microsoft",
                key="az-key",
                endpoint="https://az.openai.azure.com",
                api_version="2024-06-01",
                # deployment_name missing
            )
            is False
        )
    m.assert_not_called()


def test_validate_llm_microsoft_unreachable_endpoint_raises() -> None:
    with patch(
        "app.setup.validators._llm_get",
        side_effect=ValidationNetworkError("no route"),
    ):
        with pytest.raises(ValidationNetworkError):
            validators.validate_llm(
                "microsoft",
                key="az-key",
                endpoint="https://unreachable.example",
                api_version="2024-06-01",
                deployment_name="gpt4o",
            )


def test_llm_get_maps_httpx_request_error_to_validation_network_error() -> None:
    class _BoomClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _BoomClient:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, *a: object, **k: object) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

    with patch("app.setup.validators.httpx.Client", _BoomClient):
        with pytest.raises(ValidationNetworkError):
            validators._llm_get("https://api.openai.com/v1/models")


# --- requests network failures must never echo the FRED key -------------------

_FRED_KEY = "SECRETFREDKEY123"


@pytest.mark.parametrize(
    "exc_cls",
    [
        requests.ConnectionError,
        requests.exceptions.SSLError,
        requests.exceptions.ProxyError,
        requests.ConnectTimeout,
    ],
)
@patch("app.setup.validators.requests.get")
def test_validate_fred_network_failure_never_leaks_api_key(
    mock_get: MagicMock, exc_cls: type[requests.RequestException]
) -> None:
    # requests embeds the full request URL, query string included, in its
    # connection-error text; the wizard CLI echoes this message to stderr/CI logs.
    mock_get.side_effect = exc_cls(
        "HTTPSConnectionPool(host='api.stlouisfed.org', port=443): Max retries "
        f"exceeded with url: /fred/series?series_id=GNPCA&api_key={_FRED_KEY}"
        "&file_type=json (Caused by NewConnectionError('boom'))"
    )
    with pytest.raises(ValidationNetworkError) as info:
        validators.validate_fred(_FRED_KEY)
    assert _FRED_KEY not in str(info.value)
    assert _FRED_KEY not in "".join(traceback.format_exception(info.value))
    assert "api.stlouisfed.org" in str(info.value)
    assert exc_cls.__name__ in str(info.value)


@patch("app.setup.validators.requests.get")
def test_validate_t212_network_failure_raises_validation_network_error(
    mock_get: MagicMock,
) -> None:
    mock_get.side_effect = requests.ConnectionError("no route")
    with pytest.raises(ValidationNetworkError, match="live.trading212.com"):
        validators.validate_t212("k", "s")


# --- throttling / outages are not "invalid key" ---------------------------------

_UNAVAILABLE_STATUSES = (429, 500, 502, 503, 504)


@pytest.mark.parametrize("status", _UNAVAILABLE_STATUSES)
@patch("app.setup.validators.requests.get")
def test_validate_t212_rate_limit_or_outage_is_not_an_invalid_key(
    mock_get: MagicMock, status: int
) -> None:
    mock_get.return_value = _resp(status)
    with pytest.raises(
        ValidationNetworkError, match=rf"live\.trading212\.com.*{status}"
    ):
        validators.validate_t212("k", "s")


@pytest.mark.parametrize("status", _UNAVAILABLE_STATUSES)
@patch("app.setup.validators.requests.get")
def test_validate_fred_rate_limit_or_outage_is_not_an_invalid_key(
    mock_get: MagicMock, status: int
) -> None:
    mock_get.return_value = _resp(status)
    with pytest.raises(
        ValidationNetworkError, match=rf"api\.stlouisfed\.org.*{status}"
    ) as info:
        validators.validate_fred(_FRED_KEY)
    assert _FRED_KEY not in str(info.value)


@pytest.mark.parametrize("status", [401, 403])
@patch("app.setup.validators.requests.get")
def test_validate_t212_definitive_auth_answers_stay_an_invalid_key(
    mock_get: MagicMock, status: int
) -> None:
    mock_get.return_value = _resp(status)
    assert validators.validate_t212("k", "s") is False


@pytest.mark.parametrize("status", _UNAVAILABLE_STATUSES)
@pytest.mark.parametrize("provider", _AUTHED_PROVIDERS)
def test_validate_llm_rate_limit_or_outage_is_not_a_rejected_key(
    provider: str, status: int
) -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(status)):
        with pytest.raises(ValidationNetworkError, match=str(status)):
            validators.validate_llm(provider, key="k")


@pytest.mark.parametrize("status", _UNAVAILABLE_STATUSES)
def test_validate_llm_keyless_local_probes_surface_outages_too(status: int) -> None:
    with patch("app.setup.validators._llm_get", return_value=_resp(status)):
        with pytest.raises(ValidationNetworkError, match=str(status)):
            validators.validate_llm("ollama", base_url="http://localhost:11434")
        with pytest.raises(ValidationNetworkError, match=str(status)):
            validators.validate_llm("nvidia", base_url="http://nim:8000/v1")


@pytest.mark.parametrize("provider", _AUTHED_PROVIDERS)
def test_validate_llm_bad_request_still_means_a_rejected_key(provider: str) -> None:
    # Google answers 400 (not 401) for an unknown API key.
    with patch("app.setup.validators._llm_get", return_value=_resp(400)):
        assert validators.validate_llm(provider, key="k") is False
