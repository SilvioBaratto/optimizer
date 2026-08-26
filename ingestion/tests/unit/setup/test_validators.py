"""On-the-spot key-validation contract (SPEC D8 wizard, task T4).

Each validator makes one cheap request and returns True (200) / False (auth or
other non-200), or raises `ValidationNetworkError` when the request itself fails
(so the wizard distinguishes "bad key" from "network down"). No real network:
`requests.get` is patched, matching the repo's Trading212-client test style.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.setup import validators


def _resp(status: int) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    return r


@patch("app.setup.validators.requests.get")
def test_validate_openai_success_sends_bearer(mock_get: MagicMock) -> None:
    mock_get.return_value = _resp(200)
    assert validators.validate_llm("openai", "sk-x") is True
    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-x"


@patch("app.setup.validators.requests.get")
def test_validate_openai_auth_fail(mock_get: MagicMock) -> None:
    mock_get.return_value = _resp(401)
    assert validators.validate_llm("openai", "bad") is False


@patch("app.setup.validators.requests.get")
def test_validate_anthropic_success_sends_key_and_version(mock_get: MagicMock) -> None:
    mock_get.return_value = _resp(200)
    assert validators.validate_llm("Anthropic", "sk-ant") is True
    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["x-api-key"] == "sk-ant"
    assert "anthropic-version" in kwargs["headers"]


def test_validate_llm_rejects_non_cloud_provider() -> None:
    with pytest.raises(ValueError, match="cloud"):
        validators.validate_llm("ollama", "x")


@patch("app.setup.validators.requests.get")
def test_validate_llm_network_error_raises(mock_get: MagicMock) -> None:
    mock_get.side_effect = requests.ConnectionError("down")
    with pytest.raises(validators.ValidationNetworkError):
        validators.validate_llm("openai", "sk-x")


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
