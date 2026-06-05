"""Unit tests for ticker sub-clients: corporate_actions, holders, financials.

Coverage targets
----------------
* app.services.market_data.yfinance.ticker.corporate_actions
* app.services.market_data.yfinance.ticker.holders
* app.services.market_data.yfinance.ticker.financials

DB / repository note: these modules contain no database or repository logic;
the upsert layer lives in yfinance_data_service.py which is tested separately.

Design
------
All clients extend BaseClient.  We construct them with mock infra deps and
set ``cache.get.return_value = fake_ticker`` so ``_get_ticker`` never calls
``yf.Ticker``.  ``max_retries=1`` keeps every path to a single attempt with
no sleep, making the suite fast and deterministic.

Raw pandas objects (DataFrame / Series) are returned as-is, so we use
``pd.testing.assert_frame_equal`` / ``assert series.equals(...)`` rather than
``assert_json_safe`` (which rejects DataFrame/Series).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.ticker.corporate_actions import (
    CorporateActionsClient,
)
from app.services.market_data.yfinance.ticker.financials import FinancialsClient
from app.services.market_data.yfinance.ticker.holders import HoldersClient

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_infra(ticker_attrs: dict | None = None):
    """Return (cache, rate_limiter, circuit_breaker, fake_ticker)."""
    fake_ticker = MagicMock(name="yf.Ticker")
    if ticker_attrs:
        for attr, value in ticker_attrs.items():
            setattr(fake_ticker, attr, value)

    cache = MagicMock(name="cache")
    cache.get.return_value = fake_ticker

    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None

    return cache, rate_limiter, circuit_breaker, fake_ticker


def _corp_client(**attrs) -> tuple[CorporateActionsClient, MagicMock]:
    cache, rl, cb, ticker = _make_infra(attrs)
    client = CorporateActionsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
    return client, ticker


def _holders_client(**attrs) -> tuple[HoldersClient, MagicMock]:
    cache, rl, cb, ticker = _make_infra(attrs)
    client = HoldersClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
    return client, ticker


def _fin_client(**attrs) -> tuple[FinancialsClient, MagicMock]:
    cache, rl, cb, ticker = _make_infra(attrs)
    client = FinancialsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
    return client, ticker


# Convenience DataFrames / Series for success / empty paths
_DF = pd.DataFrame({"a": [1, 2]})
_SERIES = pd.Series([0.5, 1.0], name="val")
_EMPTY_DF = pd.DataFrame()
_EMPTY_SERIES = pd.Series(dtype=float)


# ===========================================================================
# CorporateActionsClient
# ===========================================================================


class TestFetchDividends:
    def test_when_series_non_empty_then_returns_series(self) -> None:
        client, _ = _corp_client(dividends=_SERIES)
        result = client.fetch_dividends("AAPL", max_retries=1)
        assert result is not None and result.equals(_SERIES)

    def test_when_series_empty_then_returns_none(self) -> None:
        client, _ = _corp_client(dividends=_EMPTY_SERIES)
        assert client.fetch_dividends("AAPL", max_retries=1) is None

    def test_when_attribute_raises_then_returns_none(self) -> None:
        cache, rl, cb, ticker = _make_infra()
        type(ticker).dividends = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        client = CorporateActionsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
        assert client.fetch_dividends("AAPL", max_retries=1) is None


class TestFetchSplits:
    def test_when_series_non_empty_then_returns_series(self) -> None:
        client, _ = _corp_client(splits=_SERIES)
        result = client.fetch_splits("AAPL", max_retries=1)
        assert result is not None and result.equals(_SERIES)

    def test_when_series_empty_then_returns_none(self) -> None:
        client, _ = _corp_client(splits=_EMPTY_SERIES)
        assert client.fetch_splits("AAPL", max_retries=1) is None

    def test_when_attribute_raises_then_returns_none(self) -> None:
        cache, rl, cb, ticker = _make_infra()
        type(ticker).splits = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        client = CorporateActionsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
        assert client.fetch_splits("AAPL", max_retries=1) is None


class TestFetchActions:
    def test_when_dataframe_non_empty_then_returns_dataframe(self) -> None:
        client, _ = _corp_client(actions=_DF)
        result = client.fetch_actions("AAPL", max_retries=1)
        pd.testing.assert_frame_equal(result, _DF)

    def test_when_dataframe_empty_then_returns_none(self) -> None:
        client, _ = _corp_client(actions=_EMPTY_DF)
        assert client.fetch_actions("AAPL", max_retries=1) is None

    def test_when_attribute_raises_then_returns_none(self) -> None:
        cache, rl, cb, ticker = _make_infra()
        type(ticker).actions = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("err"))
        )
        client = CorporateActionsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
        assert client.fetch_actions("AAPL", max_retries=1) is None


class TestFetchCapitalGains:
    def test_when_series_non_empty_then_returns_series(self) -> None:
        client, _ = _corp_client(capital_gains=_SERIES)
        result = client.fetch_capital_gains("AAPL", max_retries=1)
        assert result is not None and result.equals(_SERIES)

    def test_when_series_empty_then_returns_none(self) -> None:
        client, _ = _corp_client(capital_gains=_EMPTY_SERIES)
        assert client.fetch_capital_gains("AAPL", max_retries=1) is None

    def test_when_attribute_raises_then_returns_none(self) -> None:
        cache, rl, cb, ticker = _make_infra()
        type(ticker).capital_gains = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("err"))
        )
        client = CorporateActionsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
        assert client.fetch_capital_gains("AAPL", max_retries=1) is None


class TestFetchSharesFull:
    def test_when_dataframe_non_empty_then_returns_dataframe(self) -> None:
        fake_ticker = MagicMock(name="yf.Ticker")
        fake_ticker.get_shares_full.return_value = _DF
        cache = MagicMock()
        cache.get.return_value = fake_ticker
        cb = MagicMock()
        cb.check.return_value = None
        client = CorporateActionsClient(
            cache=cache, rate_limiter=MagicMock(), circuit_breaker=cb
        )
        result = client.fetch_shares_full("AAPL", start="2020-01-01", max_retries=1)
        pd.testing.assert_frame_equal(result, _DF)

    def test_when_dataframe_empty_then_returns_none(self) -> None:
        fake_ticker = MagicMock(name="yf.Ticker")
        fake_ticker.get_shares_full.return_value = _EMPTY_DF
        cache = MagicMock()
        cache.get.return_value = fake_ticker
        cb = MagicMock()
        cb.check.return_value = None
        client = CorporateActionsClient(
            cache=cache, rate_limiter=MagicMock(), circuit_breaker=cb
        )
        assert client.fetch_shares_full("AAPL", max_retries=1) is None

    def test_when_get_shares_full_raises_then_returns_none(self) -> None:
        fake_ticker = MagicMock(name="yf.Ticker")
        fake_ticker.get_shares_full.side_effect = RuntimeError("network")
        cache = MagicMock()
        cache.get.return_value = fake_ticker
        cb = MagicMock()
        cb.check.return_value = None
        client = CorporateActionsClient(
            cache=cache, rate_limiter=MagicMock(), circuit_breaker=cb
        )
        assert client.fetch_shares_full("AAPL", max_retries=1) is None


# ===========================================================================
# HoldersClient
# ===========================================================================

_HOLDERS_METHODS = [
    ("fetch_major_holders", "major_holders"),
    ("fetch_institutional_holders", "institutional_holders"),
    ("fetch_mutualfund_holders", "mutualfund_holders"),
    ("fetch_insider_transactions", "insider_transactions"),
    ("fetch_insider_purchases", "insider_purchases"),
    ("fetch_insider_roster_holders", "insider_roster_holders"),
]


@pytest.mark.parametrize("method_name,attr_name", _HOLDERS_METHODS)
def test_when_holders_dataframe_non_empty_then_returns_it(
    method_name: str, attr_name: str
) -> None:
    client, _ = _holders_client(**{attr_name: _DF})
    result = getattr(client, method_name)("AAPL", max_retries=1)
    pd.testing.assert_frame_equal(result, _DF)


@pytest.mark.parametrize("method_name,attr_name", _HOLDERS_METHODS)
def test_when_holders_dataframe_empty_then_returns_none(
    method_name: str, attr_name: str
) -> None:
    client, _ = _holders_client(**{attr_name: _EMPTY_DF})
    assert getattr(client, method_name)("AAPL", max_retries=1) is None


@pytest.mark.parametrize("method_name,attr_name", _HOLDERS_METHODS)
def test_when_holders_attr_raises_then_returns_none(
    method_name: str, attr_name: str
) -> None:
    cache, rl, cb, ticker = _make_infra()
    setattr(
        type(ticker),
        attr_name,
        property(lambda self: (_ for _ in ()).throw(RuntimeError("err"))),
    )
    client = HoldersClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
    assert getattr(client, method_name)("AAPL", max_retries=1) is None


# ===========================================================================
# FinancialsClient
# ===========================================================================

_FINANCIAL_STMT_CASES = [
    # (method_name, quarterly, expected_attr)
    ("fetch_income_stmt", False, "income_stmt"),
    ("fetch_income_stmt", True, "quarterly_income_stmt"),
    ("fetch_balance_sheet", False, "balance_sheet"),
    ("fetch_balance_sheet", True, "quarterly_balance_sheet"),
    ("fetch_cashflow", False, "cashflow"),
    ("fetch_cashflow", True, "quarterly_cashflow"),
]


@pytest.mark.parametrize("method_name,quarterly,attr_name", _FINANCIAL_STMT_CASES)
def test_when_financial_df_non_empty_then_returns_it(
    method_name: str, quarterly: bool, attr_name: str
) -> None:
    client, _ = _fin_client(**{attr_name: _DF})
    result = getattr(client, method_name)("AAPL", quarterly=quarterly, max_retries=1)
    pd.testing.assert_frame_equal(result, _DF)


@pytest.mark.parametrize("method_name,quarterly,attr_name", _FINANCIAL_STMT_CASES)
def test_when_financial_df_empty_then_returns_none(
    method_name: str, quarterly: bool, attr_name: str
) -> None:
    client, _ = _fin_client(**{attr_name: _EMPTY_DF})
    assert (
        getattr(client, method_name)("AAPL", quarterly=quarterly, max_retries=1) is None
    )


@pytest.mark.parametrize("method_name,quarterly,attr_name", _FINANCIAL_STMT_CASES)
def test_when_financial_attr_raises_then_returns_none(
    method_name: str, quarterly: bool, attr_name: str
) -> None:
    cache, rl, cb, ticker = _make_infra()
    setattr(
        type(ticker),
        attr_name,
        property(lambda self: (_ for _ in ()).throw(RuntimeError("err"))),
    )
    client = FinancialsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
    assert (
        getattr(client, method_name)("AAPL", quarterly=quarterly, max_retries=1) is None
    )


class TestFetchSecFilings:
    """fetch_sec_filings uses ``is_valid=lambda v: v is not None``, so [] is valid."""

    def test_when_sec_filings_list_non_empty_then_returns_list(self) -> None:
        filings = [{"date": "2024-01-01", "type": "10-K"}]
        client, _ = _fin_client(sec_filings=filings)
        assert client.fetch_sec_filings("AAPL", max_retries=1) == filings

    def test_when_sec_filings_empty_list_then_returns_empty_list(self) -> None:
        """Empty list is a valid result (is_valid only rejects None)."""
        client, _ = _fin_client(sec_filings=[])
        assert client.fetch_sec_filings("AAPL", max_retries=1) == []

    def test_when_sec_filings_none_then_returns_none(self) -> None:
        client, _ = _fin_client(sec_filings=None)
        assert client.fetch_sec_filings("AAPL", max_retries=1) is None

    def test_when_sec_filings_attr_raises_then_returns_none(self) -> None:
        cache, rl, cb, ticker = _make_infra()
        type(ticker).sec_filings = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("err"))
        )
        client = FinancialsClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
        assert client.fetch_sec_filings("AAPL", max_retries=1) is None
