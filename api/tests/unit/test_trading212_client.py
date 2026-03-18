"""Unit tests for Trading212Client — all HTTP calls mocked via requests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.trading212.client import Trading212Client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(mode: str = "live", max_retries: int = 3) -> Trading212Client:
    return Trading212Client(api_key="test-key-abc", mode=mode, max_retries=max_retries)


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

    def test_headers_contain_api_key(self):
        c = _make_client()
        assert c.headers == {"Authorization": "test-key-abc"}

    def test_default_max_retries(self):
        c = Trading212Client(api_key="k")
        assert c.max_retries == 5


# ---------------------------------------------------------------------------
# _get — retry / error behaviour
# ---------------------------------------------------------------------------


@patch("app.services.trading212.client.time.sleep")
@patch("app.services.trading212.client.requests.get")
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
# _fetch_json backward compat
# ---------------------------------------------------------------------------


class TestFetchJsonBackwardCompat:
    @patch.object(Trading212Client, "_get", return_value=[{"id": 1}])
    def test_delegates_to_get(self, mock_get):
        c = _make_client()
        result = c._fetch_json("/path")
        mock_get.assert_called_once_with("/path")
        assert result == [{"id": 1}]


# ---------------------------------------------------------------------------
# _get_paginated
# ---------------------------------------------------------------------------


class TestGetPaginated:
    @patch.object(Trading212Client, "_get")
    def test_single_page_no_cursor(self, mock_get):
        mock_get.return_value = {"items": [{"a": 1}], "nextPageCursor": None}
        c = _make_client()
        assert c._get_paginated("/p") == [{"a": 1}]
        assert mock_get.call_count == 1

    @patch.object(Trading212Client, "_get")
    def test_multi_page_follows_cursor(self, mock_get):
        mock_get.side_effect = [
            {"items": [{"a": 1}], "nextPageCursor": 42},
            {"items": [{"b": 2}], "nextPageCursor": None},
        ]
        c = _make_client()
        assert c._get_paginated("/p") == [{"a": 1}, {"b": 2}]
        assert mock_get.call_count == 2

    @patch.object(Trading212Client, "_get")
    def test_stops_on_empty_items(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": 99}
        c = _make_client()
        assert c._get_paginated("/p") == []

    @patch.object(Trading212Client, "_get")
    def test_passes_ticker_to_every_page(self, mock_get):
        mock_get.side_effect = [
            {"items": [{"a": 1}], "nextPageCursor": 10},
            {"items": [{"b": 2}], "nextPageCursor": None},
        ]
        c = _make_client()
        c._get_paginated("/p", ticker="AAPL_US_EQ")
        for call in mock_get.call_args_list:
            assert call.kwargs["params"]["ticker"] == "AAPL_US_EQ"

    @patch.object(Trading212Client, "_get")
    def test_custom_limit_forwarded(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c._get_paginated("/p", limit=10)
        assert mock_get.call_args.kwargs["params"]["limit"] == 10

    @patch.object(Trading212Client, "_get")
    def test_cursor_passed_to_second_page(self, mock_get):
        mock_get.side_effect = [
            {"items": [{"a": 1}], "nextPageCursor": 77},
            {"items": [], "nextPageCursor": None},
        ]
        c = _make_client()
        c._get_paginated("/p")
        second_call_params = mock_get.call_args_list[1].kwargs["params"]
        assert second_call_params["cursor"] == 77


# ---------------------------------------------------------------------------
# Account endpoints
# ---------------------------------------------------------------------------


class TestGetAccountCash:
    @patch.object(Trading212Client, "_get")
    def test_returns_cash_dict(self, mock_get):
        data = {"blocked": 0.0, "free": 1000.0, "invested": 5000.0,
                "pieCash": 0.0, "result": 200.0, "total": 6200.0}
        mock_get.return_value = data
        c = _make_client()
        assert c.get_account_cash() == data

    @patch.object(Trading212Client, "_get")
    def test_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = {}
        c = _make_client()
        c.get_account_cash()
        mock_get.assert_called_once_with("/api/v0/equity/account/cash")


class TestGetAccountInfo:
    @patch.object(Trading212Client, "_get")
    def test_returns_info_dict(self, mock_get):
        data = {"currencyCode": "GBP", "id": 12345}
        mock_get.return_value = data
        c = _make_client()
        assert c.get_account_info() == data

    @patch.object(Trading212Client, "_get")
    def test_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = {}
        c = _make_client()
        c.get_account_info()
        mock_get.assert_called_once_with("/api/v0/equity/account/info")


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


class TestGetPortfolioPositions:
    @patch.object(Trading212Client, "_get")
    def test_returns_position_list(self, mock_get):
        data = [{"ticker": "AAPL_US_EQ", "quantity": 10, "averagePrice": 150.0}]
        mock_get.return_value = data
        c = _make_client()
        assert c.get_portfolio_positions() == data

    @patch.object(Trading212Client, "_get")
    def test_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = []
        c = _make_client()
        c.get_portfolio_positions()
        mock_get.assert_called_once_with("/api/v0/equity/portfolio")


# ---------------------------------------------------------------------------
# Order history
# ---------------------------------------------------------------------------


class TestGetOrderHistory:
    @patch.object(Trading212Client, "_get")
    def test_returns_page_dict(self, mock_get):
        data = {"items": [{"id": 1}], "nextPageCursor": None}
        mock_get.return_value = data
        c = _make_client()
        assert c.get_order_history() == data

    @patch.object(Trading212Client, "_get")
    def test_default_limit_is_50(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_order_history()
        params = mock_get.call_args.kwargs["params"]
        assert params["limit"] == 50

    @patch.object(Trading212Client, "_get")
    def test_cursor_included_when_provided(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_order_history(cursor=42)
        params = mock_get.call_args.kwargs["params"]
        assert params["cursor"] == 42

    @patch.object(Trading212Client, "_get")
    def test_ticker_included_when_provided(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_order_history(ticker="AAPL_US_EQ")
        params = mock_get.call_args.kwargs["params"]
        assert params["ticker"] == "AAPL_US_EQ"

    @patch.object(Trading212Client, "_get")
    def test_cursor_omitted_when_none(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_order_history()
        params = mock_get.call_args.kwargs["params"]
        assert "cursor" not in params

    @patch.object(Trading212Client, "_get")
    def test_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_order_history()
        assert mock_get.call_args.args[0] == "/api/v0/equity/history/orders"


class TestGetAllOrderHistory:
    @patch.object(Trading212Client, "_get_paginated")
    def test_delegates_to_get_paginated(self, mock_pag):
        mock_pag.return_value = []
        c = _make_client()
        c.get_all_order_history()
        mock_pag.assert_called_once_with(
            "/api/v0/equity/history/orders", ticker=None,
        )

    @patch.object(Trading212Client, "_get_paginated")
    def test_returns_flat_list(self, mock_pag):
        mock_pag.return_value = [{"id": 1}, {"id": 2}]
        c = _make_client()
        assert c.get_all_order_history() == [{"id": 1}, {"id": 2}]

    @patch.object(Trading212Client, "_get_paginated")
    def test_ticker_forwarded(self, mock_pag):
        mock_pag.return_value = []
        c = _make_client()
        c.get_all_order_history(ticker="MSFT_US_EQ")
        mock_pag.assert_called_once_with(
            "/api/v0/equity/history/orders", ticker="MSFT_US_EQ",
        )


# ---------------------------------------------------------------------------
# Dividend history
# ---------------------------------------------------------------------------


class TestGetDividendHistory:
    @patch.object(Trading212Client, "_get")
    def test_returns_page_dict(self, mock_get):
        data = {"items": [{"amount": 1.5}], "nextPageCursor": None}
        mock_get.return_value = data
        c = _make_client()
        assert c.get_dividend_history() == data

    @patch.object(Trading212Client, "_get")
    def test_default_limit_is_50(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_dividend_history()
        params = mock_get.call_args.kwargs["params"]
        assert params["limit"] == 50

    @patch.object(Trading212Client, "_get")
    def test_cursor_included_when_provided(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_dividend_history(cursor=99)
        params = mock_get.call_args.kwargs["params"]
        assert params["cursor"] == 99

    @patch.object(Trading212Client, "_get")
    def test_ticker_included_when_provided(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_dividend_history(ticker="AAPL_US_EQ")
        params = mock_get.call_args.kwargs["params"]
        assert params["ticker"] == "AAPL_US_EQ"

    @patch.object(Trading212Client, "_get")
    def test_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = {"items": [], "nextPageCursor": None}
        c = _make_client()
        c.get_dividend_history()
        assert mock_get.call_args.args[0] == "/api/v0/history/dividends"


class TestGetAllDividendHistory:
    @patch.object(Trading212Client, "_get_paginated")
    def test_delegates_to_get_paginated(self, mock_pag):
        mock_pag.return_value = []
        c = _make_client()
        c.get_all_dividend_history()
        mock_pag.assert_called_once_with(
            "/api/v0/history/dividends", ticker=None,
        )

    @patch.object(Trading212Client, "_get_paginated")
    def test_returns_flat_list(self, mock_pag):
        mock_pag.return_value = [{"amount": 1.5}, {"amount": 2.0}]
        c = _make_client()
        assert c.get_all_dividend_history() == [{"amount": 1.5}, {"amount": 2.0}]

    @patch.object(Trading212Client, "_get_paginated")
    def test_ticker_forwarded(self, mock_pag):
        mock_pag.return_value = []
        c = _make_client()
        c.get_all_dividend_history(ticker="VOD_L_EQ")
        mock_pag.assert_called_once_with(
            "/api/v0/history/dividends", ticker="VOD_L_EQ",
        )


# ---------------------------------------------------------------------------
# from_settings factory
# ---------------------------------------------------------------------------


class TestFromSettings:
    @patch("app.services.trading212.client.settings")
    def test_returns_none_when_no_api_key(self, mock_settings):
        mock_settings.trading_212_api_key = ""
        assert Trading212Client.from_settings() is None

    @patch("app.services.trading212.client.settings")
    def test_returns_client_when_key_set(self, mock_settings):
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "live"
        client = Trading212Client.from_settings()
        assert client is not None
        assert client.api_key == "key123"

    @patch("app.services.trading212.client.settings")
    def test_mode_from_settings_when_no_override(self, mock_settings):
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "demo"
        client = Trading212Client.from_settings()
        assert client.base_url == "https://demo.trading212.com"

    @patch("app.services.trading212.client.settings")
    def test_mode_override_takes_precedence(self, mock_settings):
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "live"
        client = Trading212Client.from_settings(mode="demo")
        assert client.base_url == "https://demo.trading212.com"
