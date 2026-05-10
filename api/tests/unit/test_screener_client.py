"""Tests for issue #592: branch ScreenerClient.screen on query type.

yfinance 1.3.0 split the ``size`` kwarg of ``yf.screen`` into two:
``count=`` for predefined string screens and ``size=`` for ``*Query``
instances. The wrapper must forward the correct kwarg.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("yfinance")
from yfinance import EquityQuery, ETFQuery, FundQuery

from app.services.market_data.yfinance.market.screener import ScreenerClient


def _build_client() -> ScreenerClient:
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None
    return ScreenerClient(
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        default_max_retries=1,
    )


def _patch_yf_screen(return_value: pd.DataFrame | None = None):
    if return_value is None:
        return_value = pd.DataFrame({"symbol": ["AAPL"]})
    return patch(
        "app.services.market_data.yfinance.market.screener.yf.screen",
        return_value=return_value,
    )


def test_when_query_is_predefined_string_then_yf_screen_receives_count_kwarg() -> None:
    client = _build_client()
    with _patch_yf_screen() as mock_screen:
        client.screen(
            "day_gainers",
            offset=10,
            size=50,
            sort_field="percentchange",
            sort_asc=False,
        )
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("count") == 50
    assert "size" not in kwargs
    assert kwargs.get("offset") == 10
    assert kwargs.get("sortField") == "percentchange"
    assert kwargs.get("sortAsc") is False


def test_when_query_is_equity_query_instance_then_yf_screen_receives_size_kwarg() -> (
    None
):
    client = _build_client()
    query = MagicMock(spec=EquityQuery)
    with _patch_yf_screen() as mock_screen:
        client.screen(query, offset=5, size=100, sort_field="ticker", sort_asc=True)
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("size") == 100
    assert "count" not in kwargs
    assert kwargs.get("offset") == 5
    assert kwargs.get("sortField") == "ticker"
    assert kwargs.get("sortAsc") is True


def test_when_query_is_fund_query_instance_then_yf_screen_receives_size_kwarg() -> None:
    client = _build_client()
    query = MagicMock(spec=FundQuery)
    with _patch_yf_screen() as mock_screen:
        client.screen(query, offset=0, size=25, sort_field="ticker", sort_asc=True)
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("size") == 25
    assert "count" not in kwargs


def test_when_query_is_etf_query_instance_then_yf_screen_receives_size_kwarg() -> None:
    client = _build_client()
    query = MagicMock(spec=ETFQuery)
    with _patch_yf_screen() as mock_screen:
        client.screen(query, offset=0, size=75, sort_field="ticker", sort_asc=True)
    kwargs = mock_screen.call_args.kwargs
    assert kwargs.get("size") == 75
    assert "count" not in kwargs
