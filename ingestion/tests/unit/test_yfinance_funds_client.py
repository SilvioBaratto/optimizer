"""yfinance funds sub-client.

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
    # funds_data depth DataFrames are indexed by yfinance's human display labels
    # (Title Case, spaces, slashes) — not camelCase field names.
    fd.equity_holdings = pd.DataFrame(
        {"JAGA.DE": [15.0, 2.0]},
        index=["Price/Earnings", "Price/Book"],
    )
    fd.bond_holdings = pd.DataFrame(
        {"JAGA.DE": [5.5, 7.0]},
        index=["Duration", "Maturity"],
    )
    fd.fund_operations = pd.DataFrame(
        {"JAGA.DE": [0.001, 0.2]},
        index=["Annual Report Expense Ratio", "Annual Holdings Turnover"],
    )
    fd.bond_ratings = {"aaa": 0.5, "bbb": 0.2}
    fd.fund_overview = {
        "categoryName": "Ultrashort Bond",
        "family": "JPMorgan",
        "legalType": "Exchange Traded Fund",
    }
    fd.description = "A short-duration bond ETF."
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

    def test_parses_fund_depth(self, client: FundsClient) -> None:
        ticker = MagicMock()
        ticker.funds_data = _funds_data()

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_funds_data("JAGA.DE")

        assert out is not None
        assert out["equity_holdings"]["Price/Earnings"] == 15.0
        assert out["bond_holdings"]["Duration"] == 5.5
        assert out["fund_operations"]["Annual Report Expense Ratio"] == 0.001
        assert out["bond_ratings"]["aaa"] == 0.5
        assert out["fund_overview"]["categoryName"] == "Ultrashort Bond"
        assert out["description"] == "A short-duration bond ETF."

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

    def test_none_funds_data_returns_none(self, client: FundsClient) -> None:
        ticker = MagicMock()
        ticker.funds_data = None
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
            "financialCurrency": "USD",
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("JAGA.DE")

        assert out is not None
        assert out["aum"] == 79_000_000_000
        assert out["nav"] == 193.93
        assert out["fund_family"] == "JPMorgan"
        assert out["expense_ratio"] == 0.0025
        assert out["base_currency"] == "USD"

    def test_expense_ratio_prefers_net_expense_ratio(self, client: FundsClient) -> None:
        """netExpenseRatio is the key Yahoo actually populates for ETFs (SPY/AGG);
        the legacy mutual-fund keys are fallbacks and must not shadow it."""
        ticker = MagicMock()
        ticker.info = {
            "navPrice": 1.0,
            "netExpenseRatio": 0.0945,
            "annualReportExpenseRatio": 0.5,
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("SPY")

        assert out is not None
        assert out["expense_ratio"] == pytest.approx(0.000945)

    def test_net_expense_ratio_percent_points_become_a_decimal_fraction(
        self, client: FundsClient
    ) -> None:
        """Yahoo quotes netExpenseRatio in percent points (SPY 0.0945 = 0.0945%)
        while the legacy keys are already fractions; the stored column is a
        fraction, so the percent value is divided by 100."""
        ticker = MagicMock()
        ticker.info = {"navPrice": 96.0, "netExpenseRatio": 0.03}

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("AGG")

        assert out is not None
        assert out["expense_ratio"] == pytest.approx(0.0003)

    def test_net_and_fraction_keys_agree_on_the_same_fee(
        self, client: FundsClient
    ) -> None:
        """A mutual fund exposing both keys (VFIAX 0.04 vs 0.0004) must yield the
        same stored value whichever key is read."""
        net_only = MagicMock()
        net_only.info = {"navPrice": 1.0, "netExpenseRatio": 0.04}
        fraction_only = MagicMock()
        fraction_only.info = {"navPrice": 1.0, "annualReportExpenseRatio": 0.0004}

        with patch.object(client, "_get_ticker", return_value=net_only):
            from_net = client.fetch_fund_profile("VFIAX")
        with patch.object(client, "_get_ticker", return_value=fraction_only):
            from_fraction = client.fetch_fund_profile("VFIAX")

        assert from_net is not None and from_fraction is not None
        assert from_net["expense_ratio"] == pytest.approx(
            from_fraction["expense_ratio"]
        )

    def test_listing_currency_is_not_reported_as_the_fund_base_currency(
        self, client: FundsClient
    ) -> None:
        """navPrice/totalAssets are in the fund's own currency, not the quote
        currency: IUSA.L is quoted in pence but its NAV is USD. Mislabelling
        NAV/AUM with the listing code ('GBp') would feed FX the wrong pair."""
        ticker = MagicMock()
        ticker.info = {
            "currency": "GBp",
            "navPrice": 76.3169,
            "totalAssets": 22_874_429_440,
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("IUSA.L")

        assert out is not None
        assert out["base_currency"] is None

    def test_base_currency_comes_from_financial_currency_as_major_unit(
        self, client: FundsClient
    ) -> None:
        ticker = MagicMock()
        ticker.info = {"navPrice": 96.1, "currency": "USD", "financialCurrency": "GBp"}

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("AGG")

        assert out is not None
        assert out["base_currency"] == "GBP"

    def test_quote_currency_alone_does_not_make_an_equity_look_like_a_fund(
        self, client: FundsClient
    ) -> None:
        ticker = MagicMock()
        ticker.info = {"currency": "USD", "quoteType": "EQUITY"}

        with patch.object(client, "_get_ticker", return_value=ticker):
            assert client.fetch_fund_profile("AAPL") is None

    def test_us_listed_usd_fund_without_financial_currency_is_labelled_usd(
        self, client: FundsClient
    ) -> None:
        """Yahoo leaves financialCurrency empty for many US-listed funds (BND,
        TDEM, FAAA) though they quote and report in USD; those must not lose
        their base currency."""
        ticker = MagicMock()
        ticker.info = {
            "navPrice": 72.4,
            "totalAssets": 3.2e11,
            "currency": "USD",
            "market": "us_market",
            "exchange": "NGM",
            "financialCurrency": None,
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("BND")

        assert out is not None
        assert out["base_currency"] == "USD"

    def test_usd_quote_on_a_non_us_listing_is_not_assumed_to_be_the_nav_currency(
        self, client: FundsClient
    ) -> None:
        """A UCITS share class quoted in USD on the LSE says nothing certain
        about its NAV currency, so only a US listing earns the fallback."""
        ticker = MagicMock()
        ticker.info = {
            "navPrice": 600.0,
            "currency": "USD",
            "market": "gb_market",
            "exchange": "LSE",
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("CSPX.L")

        assert out is not None
        assert out["base_currency"] is None

    def test_us_listing_quoted_in_another_currency_is_not_assumed_usd(
        self, client: FundsClient
    ) -> None:
        ticker = MagicMock()
        ticker.info = {"navPrice": 50.0, "currency": "EUR", "market": "us_market"}

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("XYZ")

        assert out is not None
        assert out["base_currency"] is None

    def test_financial_currency_wins_over_the_us_listing_fallback(
        self, client: FundsClient
    ) -> None:
        ticker = MagicMock()
        ticker.info = {
            "navPrice": 50.0,
            "currency": "USD",
            "market": "us_market",
            "financialCurrency": "EUR",
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("XYZ")

        assert out is not None
        assert out["base_currency"] == "EUR"

    def test_us_listed_equity_with_only_quote_currency_is_still_not_a_fund(
        self, client: FundsClient
    ) -> None:
        """The US-listing fallback must not by itself turn a bare equity quote
        into a fund profile."""
        ticker = MagicMock()
        ticker.info = {
            "currency": "USD",
            "market": "us_market",
            "exchange": "NMS",
            "quoteType": "EQUITY",
        }

        with patch.object(client, "_get_ticker", return_value=ticker):
            assert client.fetch_fund_profile("AAPL") is None

    def test_aum_falls_back_to_net_assets(self, client: FundsClient) -> None:
        """Some UCITS listings expose netAssets but not totalAssets."""
        ticker = MagicMock()
        ticker.info = {"navPrice": 1.0, "netAssets": 1_045_968_830.0}

        with patch.object(client, "_get_ticker", return_value=ticker):
            out = client.fetch_fund_profile("V60A.DE")

        assert out is not None
        assert out["aum"] == 1_045_968_830.0

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
