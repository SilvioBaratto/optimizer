"""Tests for issue #603: FundsClient._get_funds_data quote_type allowlist guard.

Equity / index / crypto / future / currency tickers must short-circuit to None
without ever hitting yfinance's funds_data endpoint. ETF and MUTUALFUND
quote_types pass through. Fail-open if fast_info is missing or quote_type
absent.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.services.infrastructure.cache import LRUCache
from app.services.yfinance.protocols import FundsClientProtocol
from app.services.yfinance.ticker.funds import FundsClient


class _StubFastInfo:
    """Mimics yfinance's FastInfo proxy with optional ``quote_type``."""

    def __init__(self, quote_type: str | None) -> None:
        if quote_type is not None:
            self.quote_type = quote_type


class _StubTicker:
    """Counts ``fast_info`` and ``funds_data`` attribute reads."""

    def __init__(self, fast_info: Any, funds_data: Any) -> None:
        self._fast_info = fast_info
        self._funds_data = funds_data
        self.fast_info_reads = 0
        self.funds_data_reads = 0

    @property
    def fast_info(self) -> Any:
        self.fast_info_reads += 1
        return self._fast_info

    @property
    def funds_data(self) -> Any:
        self.funds_data_reads += 1
        return self._funds_data


def _build_client(symbol: str, ticker: Any) -> tuple[FundsClient, LRUCache]:
    cache = LRUCache(capacity=10, default_ttl=3600.0)
    cache.put(symbol, ticker)
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None
    client = FundsClient(
        cache=cache,
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        default_max_retries=1,
    )
    return client, cache


def test_when_quote_type_is_equity_then_get_funds_data_returns_none() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("EQUITY"), fd)
    client, _ = _build_client("AAPL", ticker)

    assert client._get_funds_data("AAPL") is None
    assert ticker.funds_data_reads == 0


def test_when_quote_type_is_etf_then_get_funds_data_returns_funds_data() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("ETF"), fd)
    client, _ = _build_client("SPY", ticker)

    assert client._get_funds_data("SPY") is fd


def test_when_quote_type_is_mutualfund_then_get_funds_data_returns_funds_data() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("MUTUALFUND"), fd)
    client, _ = _build_client("VTSAX", ticker)

    assert client._get_funds_data("VTSAX") is fd


def test_when_quote_type_is_lowercase_etf_then_normalised_and_passes() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("etf"), fd)
    client, _ = _build_client("VOO", ticker)

    assert client._get_funds_data("VOO") is fd


def test_when_quote_type_is_index_then_short_circuits() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("INDEX"), fd)
    client, _ = _build_client("^GSPC", ticker)

    assert client._get_funds_data("^GSPC") is None
    assert ticker.funds_data_reads == 0


def test_when_fast_info_is_none_then_fail_open_and_returns_funds_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(None, fd)
    client, _ = _build_client("XYZ", ticker)

    with caplog.at_level(logging.WARNING):
        result = client._get_funds_data("XYZ")

    assert result is fd
    assert any("XYZ" in rec.message for rec in caplog.records)


def test_when_quote_type_attribute_missing_then_fail_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo(None), fd)
    client, _ = _build_client("XYZ", ticker)

    with caplog.at_level(logging.WARNING):
        result = client._get_funds_data("XYZ")

    assert result is fd


def test_when_called_twice_for_equity_then_quote_type_cached_under_qt_key() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("EQUITY"), fd)
    client, cache = _build_client("AAPL", ticker)

    assert client._get_funds_data("AAPL") is None
    assert client._get_funds_data("AAPL") is None

    assert cache.get("qt:AAPL") == "EQUITY"
    assert ticker.fast_info_reads == 1


def test_when_quote_type_cached_then_fast_info_not_re_read() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("ETF"), fd)
    client, cache = _build_client("SPY", ticker)

    client._get_funds_data("SPY")
    client._get_funds_data("SPY")
    client._get_funds_data("SPY")

    assert ticker.fast_info_reads == 1
    assert cache.get("qt:SPY") == "ETF"


def test_when_equity_then_all_fetch_fund_methods_short_circuit_to_none() -> None:
    fd = MagicMock(name="funds_data")
    ticker = _StubTicker(_StubFastInfo("EQUITY"), fd)
    client, _ = _build_client("AAPL", ticker)

    methods = (
        "fetch_fund_overview",
        "fetch_fund_top_holdings",
        "fetch_fund_sector_weightings",
        "fetch_fund_bond_holdings",
        "fetch_fund_bond_ratings",
        "fetch_fund_equity_holdings",
        "fetch_fund_operations",
        "fetch_fund_asset_classes",
        "fetch_fund_description",
    )
    for name in methods:
        result = getattr(client, name)("AAPL")
        assert result is None, f"{name} did not short-circuit"

    assert ticker.funds_data_reads == 0


def test_when_funds_client_built_then_protocol_is_satisfied() -> None:
    cache = LRUCache(capacity=10, default_ttl=3600.0)
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    client = FundsClient(
        cache=cache,
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        default_max_retries=1,
    )
    assert isinstance(client, FundsClientProtocol)
