"""Unit tests for Trading212Client — all HTTP calls mocked via requests.

The client's surface is intentionally tiny: the two metadata endpoints the
ingestion daemon actually uses, plus the shared ``_get`` retry helper and the
``from_settings`` factory. The account / portfolio / order / dividend methods
were removed (daemon ingests, does not trade), so their tests are gone too.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.universe.trading212.client import Trading212Client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(mode: str = "live", max_retries: int = 3) -> Trading212Client:
    return Trading212Client(
        api_key="test-key-abc",
        api_secret="test-secret-xyz",
        mode=mode,
        max_retries=max_retries,
    )


def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _mock_429_error(retry_after: str | None = None) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"Retry-After": retry_after} if retry_after else {}
    err = requests.HTTPError("429 Too Many Requests")
    err.response = response
    return err


def _mock_http_error(status_code: int) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    err = requests.HTTPError(f"{status_code} Error")
    err.response = response
    return err


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestTrading212ClientInit:
    def test_live_mode_sets_live_base_url(self):
        c = _make_client(mode="live")
        assert c.base_url == "https://live.trading212.com"

    def test_demo_mode_sets_demo_base_url(self):
        c = _make_client(mode="demo")
        assert c.base_url == "https://demo.trading212.com"

    def test_headers_contain_basic_auth(self):
        c = _make_client()
        import base64

        expected = base64.b64encode(b"test-key-abc:test-secret-xyz").decode()
        assert c.headers == {"Authorization": f"Basic {expected}"}

    def test_default_max_retries(self):
        c = Trading212Client(api_key="k", api_secret="s")
        assert c.max_retries == 5


# ---------------------------------------------------------------------------
# Metadata endpoints — the live surface
# ---------------------------------------------------------------------------


class TestMetadataEndpoints:
    @patch.object(Trading212Client, "_get")
    def test_get_exchanges_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = [{"id": 1, "name": "NASDAQ"}]
        c = _make_client()
        assert c.get_exchanges() == [{"id": 1, "name": "NASDAQ"}]
        mock_get.assert_called_once_with("/api/v0/equity/metadata/exchanges")

    @patch.object(Trading212Client, "_get")
    def test_get_instruments_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = [{"ticker": "AAPL_US_EQ", "type": "STOCK"}]
        c = _make_client()
        assert c.get_instruments() == [{"ticker": "AAPL_US_EQ", "type": "STOCK"}]
        mock_get.assert_called_once_with("/api/v0/equity/metadata/instruments")


# ---------------------------------------------------------------------------
# _get — retry / error behaviour
# ---------------------------------------------------------------------------


@patch("app.services.universe.trading212.client.time.sleep")
@patch("app.services.universe.trading212.client.requests.get")
class TestGetInternal:
    def test_success_on_first_attempt(self, mock_get, _sleep):
        mock_get.return_value = _mock_response({"ok": True})
        c = _make_client()
        assert c._get("/path") == {"ok": True}
        assert mock_get.call_count == 1

    def test_passes_params_to_requests(self, mock_get, _sleep):
        mock_get.return_value = _mock_response([])
        c = _make_client()
        c._get("/path", params={"limit": 50})
        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {"limit": 50}

    def test_retries_on_network_error(self, mock_get, _sleep):
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("fail"),
            _mock_response({"retry": True}),
        ]
        c = _make_client()
        assert c._get("/p") == {"retry": True}
        assert mock_get.call_count == 2

    def test_raises_after_max_retries_network_error(self, mock_get, _sleep):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        c = _make_client(max_retries=2)
        with pytest.raises(requests.exceptions.ConnectionError):
            c._get("/p")
        assert mock_get.call_count == 2

    def test_handles_429_with_retry_after_header(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _raise(_mock_429_error(retry_after="3")),
            _mock_response({"ok": True}),
        ]
        c = _make_client()
        assert c._get("/p") == {"ok": True}
        mock_sleep.assert_called_once_with(3)

    def test_handles_429_with_exponential_fallback(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _raise(_mock_429_error()),
            _mock_response({"ok": True}),
        ]
        c = _make_client()
        assert c._get("/p") == {"ok": True}
        mock_sleep.assert_called_once_with(2)  # (2**0) * 2

    def test_raises_non_429_http_error_immediately(self, mock_get, _sleep):
        mock_get.return_value.raise_for_status.side_effect = _mock_http_error(404)
        c = _make_client()
        with pytest.raises(requests.HTTPError):
            c._get("/p")
        assert mock_get.call_count == 1

    def test_raises_on_last_429_attempt(self, mock_get, _sleep):
        mock_get.side_effect = [
            _raise(_mock_429_error()),
            _raise(_mock_429_error()),
        ]
        c = _make_client(max_retries=2)
        with pytest.raises(requests.HTTPError):
            c._get("/p")


def _raise(err):
    """Helper: make side_effect raise an exception from a mock_get call."""
    resp = MagicMock()
    resp.raise_for_status.side_effect = err
    return resp


# ---------------------------------------------------------------------------
# _fetch_json
# ---------------------------------------------------------------------------


class TestFetchJson:
    @patch.object(Trading212Client, "_get", return_value=[{"id": 1}])
    def test_delegates_to_get(self, mock_get):
        c = _make_client()
        result = c._fetch_json("/path")
        mock_get.assert_called_once_with("/path")
        assert result == [{"id": 1}]


# ---------------------------------------------------------------------------
# from_settings factory
# ---------------------------------------------------------------------------


class TestFromSettings:
    @patch("app.services.universe.trading212.client.settings")
    def test_returns_none_when_no_api_key(self, mock_settings):
        mock_settings.trading_212_api_key = ""
        assert Trading212Client.from_settings() is None

    @patch("app.services.universe.trading212.client.settings")
    def test_returns_client_when_key_set(self, mock_settings):
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "live"
        client = Trading212Client.from_settings()
        assert client is not None
        assert client.api_key == "key123"

    @patch("app.services.universe.trading212.client.settings")
    def test_mode_from_settings_when_no_override(self, mock_settings):
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "demo"
        client = Trading212Client.from_settings()
        assert client is not None
        assert client.base_url == "https://demo.trading212.com"

    @patch("app.services.universe.trading212.client.settings")
    def test_mode_override_takes_precedence(self, mock_settings):
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "live"
        client = Trading212Client.from_settings(mode="demo")
        assert client is not None
        assert client.base_url == "https://demo.trading212.com"
