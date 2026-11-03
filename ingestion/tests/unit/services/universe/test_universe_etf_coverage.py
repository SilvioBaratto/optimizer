"""US-ETF coverage regression for the Screener-backed universe build.

Large, liquid US ETFs (VOO, SPY, QQQ, GLD, sector and factor funds) were absent
from the stored universe: the venue re-bucket dropped funds listed on Cboe BZX
and the ETF classifier rejected every fund that was neither a bond nor a
multi-asset product. These tests run mocked Screener rows shaped like real Yahoo
output (exchange codes, ``netAssets`` and no ``marketCap`` for funds) through the
source, the canonical dedup, the anti-junk floor and the builder, and assert the
big funds come out the other side.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

from app.services._shared.trading_calendar import EXCHANGE_NAME_TO_MIC
from app.services.universe.trading212.builder import UniverseBuilder
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.yfinance_source import (
    _CODE_TO_CONFIG_NAME,
    _CODE_TO_REGION,
    _EXCHANGE_PREF,
    _STOCK_SORT_FIELD,
    PassThroughTickerMapper,
    YFinanceUniverseSource,
)

_ETF_SORT = "fundnetassets"


def _etf(
    symbol: str,
    exchange: str,
    long_name: str,
    *,
    net_assets: float,
    price: float,
    volume: float,
    shares: float | None = None,
    financial_currency: str | None = "USD",
) -> dict[str, Any]:
    # Shape of a real ETF screener row: no marketCap, netAssets carries the size,
    # and financialCurrency is often absent.
    return {
        "symbol": symbol,
        "exchange": exchange,
        "quoteType": "ETF",
        "currency": "USD",
        "financialCurrency": financial_currency,
        "longName": long_name,
        "regularMarketPrice": price,
        "averageDailyVolume3Month": volume,
        "marketCap": None,
        "sharesOutstanding": shares,
        "netAssets": net_assets,
    }


_US_ETFS: list[dict[str, Any]] = [
    _etf(
        "VOO",
        "PCX",
        "Vanguard S&P 500 ETF",
        net_assets=1.7e12,
        price=691.32,
        volume=7_128_884,
    ),
    _etf(
        "SPY",
        "PCX",
        "State Street SPDR S&P 500 ETF Trust",
        net_assets=8.1e11,
        price=752.1,
        volume=41_462_520,
        shares=917_782_016,
    ),
    _etf(
        "QQQ",
        "NGM",
        "Invesco QQQ Trust",
        net_assets=5.0e11,
        price=681.09,
        volume=35_113_323,
        shares=393_100_000,
    ),
    _etf(
        "GLD",
        "PCX",
        "SPDR Gold Shares",
        net_assets=1.0e11,
        price=440.0,
        volume=9_000_000,
        shares=230_000_000,
    ),
    _etf(
        "XLK",
        "PCX",
        "State Street Technology Select Sector SPDR ETF",
        net_assets=7.0e10,
        price=250.0,
        volume=8_000_000,
        shares=280_000_000,
    ),
    _etf(
        "MTUM",
        "BTS",
        "iShares MSCI USA Momentum Factor ETF",
        net_assets=1.5e10,
        price=240.0,
        volume=1_000_000,
        financial_currency=None,
    ),
    _etf(
        "IEFA",
        "BTS",
        "iShares Core MSCI EAFE ETF",
        net_assets=1.9e11,
        price=96.64,
        volume=8_579_660,
        financial_currency=None,
    ),
    _etf(
        "BND",
        "NGM",
        "Vanguard Total Bond Market Index Fund ETF Shares",
        net_assets=3.9e11,
        price=71.33,
        volume=10_135_307,
        financial_currency=None,
    ),
]

_NOT_INVESTABLE: list[dict[str, Any]] = [
    _etf(
        "TQQQ",
        "NGM",
        "ProShares UltraPro QQQ",
        net_assets=2.0e10,
        price=80.0,
        volume=60_000_000,
        shares=250_000_000,
    ),
    _etf(
        "IBIT",
        "NGM",
        "iShares Bitcoin Trust ETF",
        net_assets=7.0e10,
        price=55.0,
        volume=40_000_000,
        shares=1_200_000_000,
    ),
    _etf(
        "TINY",
        "PCX",
        "Tiny Thematic ETF",
        net_assets=1.0e6,
        price=20.0,
        volume=500,
        shares=50_000,
    ),
]


def _source(
    quotes: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> YFinanceUniverseSource:
    screener = MagicMock()
    screener.screen.side_effect = [{"quotes": []}, {"quotes": quotes}, {"quotes": []}]
    src = YFinanceUniverseSource(
        screener=screener, fx_resolver=lambda majors: dict.fromkeys(majors, 1.0)
    )
    monkeypatch.setattr(
        src,
        "_build_queries",
        lambda: [
            ("QS", "STOCK", _STOCK_SORT_FIELD, 10_000),
            ("QE", "ETF", _ETF_SORT, 10_000),
        ],
    )
    return src


def _build(src: YFinanceUniverseSource) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    repo = MagicMock()
    repo.get_active_tickers.return_value = set()
    repo.get_pinned_tickers.return_value = set()
    repo.save_instruments_batch.side_effect = lambda rows, _exchange_id: (
        saved.extend(rows) or len(rows)
    )
    UniverseBuilder(
        config=UniverseBuilderConfig(),
        api_client=src,
        ticker_mapper=PassThroughTickerMapper(),
        repository=repo,
        max_workers=1,
    ).build()
    return saved


class TestCboeBzxVenue:
    """Cboe BZX (Yahoo ``BTS``) is a covered venue like NYSE Arca."""

    def test_code_maps_to_its_own_exchange(self) -> None:
        """BTS resolves to a Cboe BZX exchange row with a US-style ticker."""
        assert _CODE_TO_CONFIG_NAME["BTS"] == "Cboe BZX"
        assert _CODE_TO_REGION["BTS"] == "us"
        assert UniverseBuilderConfig().get_yahoo_suffix("Cboe BZX") == ""

    def test_has_the_us_session_calendar(self) -> None:
        """BZX trades the NYSE sessions, so the history guard applies."""
        assert EXCHANGE_NAME_TO_MIC["Cboe BZX"] == "XNYS"

    def test_ranks_after_the_major_us_venues_on_a_tie(self) -> None:
        """A tie between BZX and NYSE Arca keeps the Arca line."""
        assert _EXCHANGE_PREF.index("PCX") < _EXCHANGE_PREF.index("BTS")


class TestLargeUsEtfsSurviveTheBuild:
    """VOO, GLD, SPY, QQQ and sector/factor funds are stored after a build."""

    def test_big_funds_are_admitted_as_etfs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every large fund passes venue, classification, dedup and the floor."""
        src = _source(_US_ETFS + _NOT_INVESTABLE, monkeypatch)

        stored = {i["ticker"]: i for i in src.get_instruments()}

        assert {"VOO", "GLD", "SPY", "QQQ", "XLK", "MTUM", "IEFA", "BND"} <= set(stored)
        assert all(stored[t]["type"] == "ETF" for t in stored)

    def test_cboe_bzx_funds_land_on_their_own_exchange(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Factor and international funds listed on BZX are no longer dropped."""
        src = _source(_US_ETFS, monkeypatch)
        src.get_instruments()

        names = {e["name"] for e in src.get_exchanges()}

        assert names == {"NYSE", "NASDAQ", "Cboe BZX"}

    def test_build_persists_each_fund_with_its_asset_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Equity funds are equity, gold is commodity and bond funds stay FI."""
        saved = _build(_source(_US_ETFS + _NOT_INVESTABLE, monkeypatch))

        by_ticker = {row["ticker"]: row for row in saved}

        assert by_ticker["VOO"]["assetClass"] == "equity"
        assert by_ticker["SPY"]["assetClass"] == "equity"
        assert by_ticker["QQQ"]["assetClass"] == "equity"
        assert by_ticker["XLK"]["assetClass"] == "equity"
        assert by_ticker["MTUM"]["assetClass"] == "equity"
        assert by_ticker["IEFA"]["assetClass"] == "equity"
        assert by_ticker["GLD"]["assetClass"] == "commodity"
        assert by_ticker["BND"]["assetClass"] == "fixed_income"
        assert all(row["type"] == "ETF" for row in saved)

    def test_leveraged_crypto_and_dust_funds_stay_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Daily-reset, crypto-trust and sub-floor funds are not admitted."""
        saved = _build(_source(_US_ETFS + _NOT_INVESTABLE, monkeypatch))

        assert {"TQQQ", "IBIT", "TINY"}.isdisjoint({row["ticker"] for row in saved})

    def test_a_fund_is_not_collapsed_into_a_sibling_with_a_similar_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dedup keys on the exact normalized name, so S&P 500 trackers coexist."""
        src = _source(_US_ETFS, monkeypatch)

        stored = {i["ticker"] for i in src.get_instruments()}

        assert {"VOO", "SPY"} <= stored

    def test_a_large_fund_with_no_financial_currency_clears_the_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Yahoo omits financialCurrency on many funds; netAssets still sizes them."""
        src = _source([_US_ETFS[-1]], monkeypatch)  # BND

        assert [i["ticker"] for i in src.get_instruments()] == ["BND"]
