"""Unit tests for ticker sub-clients: corporate_actions, holders, financials, metadata.

Coverage targets
----------------
* app.services.market_data.yfinance.ticker.corporate_actions
* app.services.market_data.yfinance.ticker.holders
* app.services.market_data.yfinance.ticker.financials
* app.services.market_data.yfinance.ticker.metadata (``fetch_isin``)

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
from app.services.market_data.yfinance.ticker.metadata import MetadataClient

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


def _metadata_client(**attrs) -> tuple[MetadataClient, MagicMock]:
    cache, rl, cb, ticker = _make_infra(attrs)
    client = MetadataClient(cache=cache, rate_limiter=rl, circuit_breaker=cb)
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

    def test_when_currency_tagged_frame_then_returns_frame_unchanged(self) -> None:
        # yfinance returns Dividends + currency columns when Yahoo tags payments.
        frame = pd.DataFrame(
            {"Dividends": [0.25, 0.30], "currency": ["GBp", "GBp"]},
            index=pd.to_datetime(["2024-01-15", "2024-04-15"]),
        )
        client, _ = _corp_client(dividends=frame)
        result = client.fetch_dividends("SHEL.L", max_retries=1)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, frame)

    def test_when_currency_tagged_frame_empty_then_returns_none(self) -> None:
        empty = pd.DataFrame({"Dividends": [], "currency": []})
        client, _ = _corp_client(dividends=empty)
        assert client.fetch_dividends("SHEL.L", max_retries=1) is None

    def test_when_attribute_raises_then_returns_none(self) -> None:
        cache, rl, cb, ticker = _make_infra()
        type(ticker).dividends = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        client = CorporateActionsClient(
            cache=cache, rate_limiter=rl, circuit_breaker=cb
        )
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
        client = CorporateActionsClient(
            cache=cache, rate_limiter=rl, circuit_breaker=cb
        )
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
        client = CorporateActionsClient(
            cache=cache, rate_limiter=rl, circuit_breaker=cb
        )
        assert client.fetch_actions("AAPL", max_retries=1) is None


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

    def test_when_no_start_given_then_history_is_requested_from_an_early_date(
        self,
    ) -> None:
        """yfinance alone stops at the trailing 548 days; "earliest available"
        needs an explicit start or the share-count series never gets older rows."""
        client, ticker = _corp_client(get_shares_full=MagicMock(return_value=_SERIES))

        client.fetch_shares_full("AAPL", max_retries=1)

        start = ticker.get_shares_full.call_args.kwargs["start"]
        assert start is not None
        assert pd.Timestamp(start) <= pd.Timestamp("2000-01-01")

    def test_when_start_given_then_it_is_passed_through_unchanged(self) -> None:
        client, ticker = _corp_client(get_shares_full=MagicMock(return_value=_SERIES))

        client.fetch_shares_full(
            "AAPL", start="2020-01-01", end="2021-01-01", max_retries=1
        )

        ticker.get_shares_full.assert_called_once_with(
            start="2020-01-01", end="2021-01-01"
        )

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
# MetadataClient.fetch_isin
# ===========================================================================


class TestFetchIsin:
    """yfinance scrapes Business Insider for ``Ticker.isin``.

    Business Insider's tickers carry no Yahoo exchange suffix, so a suffixed
    symbol never matches and yfinance falls back to the first hit *without* a
    ticker: a depositary receipt, preferred share or bond of the same issuer.
    Such codes are well formed, pass the check digit and often carry the
    listing's own country prefix, so only a suffix-less symbol's ISIN is kept.
    """

    @pytest.mark.parametrize(
        ("symbol", "isin"),
        [
            ("AAPL", "US0378331005"),
            ("MSFT", "US5949181045"),
            ("RY", "CA7800871021"),
            ("ASML", "USN070592100"),
        ],
    )
    def test_when_a_us_symbol_has_a_well_formed_isin_then_it_is_returned(
        self, symbol: str, isin: str
    ) -> None:
        client, _ = _metadata_client(isin=isin)
        assert client.fetch_isin(symbol, max_retries=1) == isin

    @pytest.mark.parametrize(
        ("symbol", "isin"),
        [
            pytest.param("ASML.AS", "AR0725224551", id="asml-cedear"),
            pytest.param("MC.PA", "CA50244Q1037", id="lvmh-cdr"),
            pytest.param("7203.T", "CA89238H1091", id="toyota-cdr-on-tokyo"),
            pytest.param("RY.TO", "CA78012H5675", id="rbc-preferred-share"),
            pytest.param("TD.TO", "CA8911605415", id="td-same-country-prefix"),
            pytest.param("SHOP.TO", "ARBCOM460341", id="shopify-cedear"),
            pytest.param("0700.HK", "KYG875721899", id="tencent-cayman-line"),
            pytest.param("FRO.OL", "PLCTHQM00018", id="frontline-polish-code"),
            pytest.param("ASML.AS", "NL0010273215", id="genuine-eu-line"),
            pytest.param("FRO.OL", "BMG3682E1921", id="genuine-bermuda-line"),
            pytest.param("VUSA.L", "IE00B3XXRP09", id="genuine-ucits-line"),
            pytest.param("APC.DE", "US0378331005", id="german-venue"),
            pytest.param("0NMK.IL", "US0378331005", id="london-iob"),
        ],
    )
    def test_when_a_symbol_has_an_exchange_suffix_then_no_isin_is_stored(
        self, symbol: str, isin: str
    ) -> None:
        """Even a genuine line is dropped: the scrape cannot be tied to the listing."""
        client, _ = _metadata_client(isin=isin)

        assert client.fetch_isin(symbol, max_retries=1) is None

        client.cache.get.assert_not_called()

    @pytest.mark.parametrize(
        "isin",
        [
            "NET000PLEN01",
            "NL0010273216",
            "NL001027321",
            "nl0010273215",
            "NL0010273215X",
            "1L0010273215",
        ],
    )
    def test_when_isin_is_malformed_or_fails_checksum_then_it_is_dropped(
        self, isin: str
    ) -> None:
        client, _ = _metadata_client(isin=isin)
        assert client.fetch_isin("AAPL", max_retries=1) is None

    @pytest.mark.parametrize("isin", ["-", "", None])
    def test_when_yfinance_reports_no_isin_then_none(self, isin: str | None) -> None:
        client, _ = _metadata_client(isin=isin)
        assert client.fetch_isin("AAPL", max_retries=1) is None


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
