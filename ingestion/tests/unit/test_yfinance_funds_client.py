"""T2 — yfinance funds sub-client (re-added after the strip).

Fetches ETF fund data (asset-class split, top holdings, sector weights) and the
headline profile (AUM/NAV/fund-family/…). ``yf.Ticker`` is mocked — no network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.market_data.yfinance._facade import YFinanceClient
from app.services.market_data.yfinance.funds import FundsClient


@pytest.fixture
def client() -> FundsClient:
    return FundsClient(
        cache=MagicMock(),
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=1,
    )


def _funds_data() -> MagicMock:
    fd = MagicMock()
    fd.asset_classes = {"stockPosition": 0.0, "bondPosition": 1.0}
    fd.sector_weightings = {"government": 0.6, "corporate": 0.4}
    fd.top_holdings = pd.DataFrame(
        {"Name": ["US Treasury", "Bund"], "Holding Percent": [0.05, 0.03]},
        index=pd.Index(["UST", "BUND"], name="Symbol"),
    )
    return fd


class TestFetchFundsData:
    def test_parses_asset_classes_holdings_and_sectors(
        self, client: FundsClient
    ) -> None:
        ticker = MagicMock()
        ticker.funds_data = _funds_data()

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_funds_data("JAGA.DE")

        assert out is not None
        assert out["asset_classes"]["bondPosition"] == 1.0
        assert out["sector_weightings"]["government"] == 0.6
        assert {h["symbol"] for h in out["top_holdings"]} == {"UST", "BUND"}
        assert out["top_holdings"][0]["weight"] == 0.05

    def test_equity_ticker_with_empty_funds_data_returns_none(
        self, client: FundsClient
    ) -> None:
        fd = MagicMock()
        fd.asset_classes = {}
        fd.sector_weightings = {}
        fd.top_holdings = pd.DataFrame()
        ticker = MagicMock()
        ticker.funds_data = fd

        with patch.object(client, "_get_ticker", return_value=ticker):
            assert client.fetch_funds_data("AAPL") is None

    def test_raising_funds_data_is_swallowed_to_none(self, client: FundsClient) -> None:
        ticker = MagicMock()
        type(ticker).funds_data = property(
            lambda self: (_ for _ in ()).throw(ValueError("not a fund"))
        )

        with patch.object(client, "_get_ticker", return_value=ticker):
            assert client.fetch_funds_data("AAPL") is None


class TestFetchFundProfile:
    def test_parses_info_fields(self, client: FundsClient) -> None:
        ticker = MagicMock()
        ticker.info = {
            "totalAssets": 79_000_000_000,
            "navPrice": 193.93,
            "fundFamily": "JPMorgan",
            "legalType": "Exchange Traded Fund",
            "annualReportExpenseRatio": 0.0025,
            "currency": "EUR",
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("JAGA.DE")

        assert out is not None
        assert out["aum"] == 79_000_000_000
        assert out["nav"] == 193.93
        assert out["fund_family"] == "JPMorgan"
        assert out["expense_ratio"] == 0.0025
        assert out["base_currency"] == "EUR"

    def test_empty_info_returns_none(self, client: FundsClient) -> None:
        ticker = MagicMock()
        ticker.info = {}

        with patch.object(client, "_get_ticker", return_value=ticker):
            assert client.fetch_fund_profile("AAPL") is None


class TestFacadeWiring:
    def test_facade_exposes_cached_funds_sub_client(self) -> None:
        facade = YFinanceClient(
            cache=MagicMock(),
            rate_limiter=MagicMock(),
            circuit_breaker=MagicMock(),
        )
        assert isinstance(facade.funds, FundsClient)
        assert facade.funds is facade.funds  # cached_property
