"""ScreenerClient contract (SPEC D1, task T8).

Wraps yfinance 1.6.0 ``yf.screen`` (there is no ``yf.Screener`` class) with the
shared rate-limiter / circuit-breaker / retry infrastructure, and is exposed on
the facade as ``.screener``. yfinance calls are patched.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.screener.screener_client import (
    ScreenerClient,
)

_SCREEN = "app.services.market_data.yfinance.screener.screener_client.yf.screen"


def _client() -> ScreenerClient:
    rate_limiter = MagicMock()
    circuit_breaker = MagicMock()
    circuit_breaker.check.return_value = None
    return ScreenerClient(
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        default_max_retries=1,
    )


@patch(_SCREEN)
def test_screen_forwards_query_and_pagination(mock_screen: MagicMock) -> None:
    mock_screen.return_value = {"quotes": [{"symbol": "AAPL"}]}
    result = _client().screen("day_gainers", size=100, offset=50)
    assert result == {"quotes": [{"symbol": "AAPL"}]}
    assert mock_screen.call_args.args[0] == "day_gainers"
    assert mock_screen.call_args.kwargs["size"] == 100
    assert mock_screen.call_args.kwargs["offset"] == 50


@patch(_SCREEN)
def test_screen_uses_rate_limiter_and_circuit_breaker(mock_screen: MagicMock) -> None:
    mock_screen.return_value = {"quotes": []}
    client = _client()
    client.screen("most_actives")
    client.circuit_breaker.check.assert_called_once()
    client.rate_limiter.acquire.assert_called_once_with("screener")


def test_screen_predefined_rejects_unknown() -> None:
    assert _client().screen_predefined("not-a-real-screen") is None


@patch(_SCREEN)
def test_screen_predefined_valid_delegates(mock_screen: MagicMock) -> None:
    import yfinance as yf

    mock_screen.return_value = {"quotes": []}
    name = next(iter(yf.PREDEFINED_SCREENER_QUERIES))
    assert _client().screen_predefined(name) == {"quotes": []}
    assert mock_screen.call_args.args[0] == name


@patch(_SCREEN)
def test_screen_predefined_uses_count_not_size(mock_screen: MagicMock) -> None:
    import yfinance as yf

    mock_screen.return_value = {"quotes": []}
    name = next(iter(yf.PREDEFINED_SCREENER_QUERIES))
    _client().screen_predefined(name, count=50)
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("count") == 50
    assert "size" not in kwargs  # predefined screens size via count, not size


@patch(_SCREEN)
def test_custom_query_uses_size_not_count(mock_screen: MagicMock) -> None:
    mock_screen.return_value = {"quotes": []}
    _client().screen(
        "day_gainers", size=80
    )  # a Query would be an object; string still size-path here
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("size") == 80
    assert "count" not in kwargs


def test_facade_exposes_screener_and_tracks_it_for_reset() -> None:
    from app.services.market_data.yfinance import YFinanceClient

    YFinanceClient.reset_instance()
    client = YFinanceClient.get_instance()
    assert isinstance(client.screener, ScreenerClient)
    assert "screener" in YFinanceClient._CACHED_SUBCLIENT_NAMES
    YFinanceClient.reset_instance()
