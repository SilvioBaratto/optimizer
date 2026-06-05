"""Unit tests for market sub-clients: search, sector_industry, screener.

Coverage targets
----------------
* app.services.market_data.yfinance.market.search
* app.services.market_data.yfinance.market.sector_industry
* app.services.market_data.yfinance.market.screener

DB / repository note: these modules contain no database or repository logic.

Design
------
Each client accepts rate_limiter + circuit_breaker mocks.  ``yf.*`` classes
are patched at their import site within each module namespace.
``max_retries=1`` ensures single-attempt, sleep-free execution.

``assert_json_safe`` is applied to dict / list returns only (search results,
sector overview), never to raw pandas DataFrames.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.market.screener import ScreenerClient
from app.services.market_data.yfinance.market.search import SearchClient
from app.services.market_data.yfinance.market.sector_industry import (
    SectorIndustryClient,
)
from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _build_search_client() -> SearchClient:
    rl = MagicMock(name="rate_limiter")
    cb = MagicMock(name="circuit_breaker")
    cb.check.return_value = None
    return SearchClient(rate_limiter=rl, circuit_breaker=cb, default_max_retries=1)


def _build_sector_client() -> SectorIndustryClient:
    rl = MagicMock(name="rate_limiter")
    cb = MagicMock(name="circuit_breaker")
    cb.check.return_value = None
    return SectorIndustryClient(rate_limiter=rl, circuit_breaker=cb, default_max_retries=1)


def _build_screener_client() -> ScreenerClient:
    rl = MagicMock(name="rate_limiter")
    cb = MagicMock(name="circuit_breaker")
    cb.check.return_value = None
    return ScreenerClient(rate_limiter=rl, circuit_breaker=cb, default_max_retries=1)


# Patch helpers
_SEARCH_PATH = "app.services.market_data.yfinance.market.search.yf.Search"
_LOOKUP_PATH = "app.services.market_data.yfinance.market.search.yf.Lookup"
_SECTOR_PATH = "app.services.market_data.yfinance.market.sector_industry.yf.Sector"
_INDUSTRY_PATH = "app.services.market_data.yfinance.market.sector_industry.yf.Industry"
_SCREEN_PATH = "app.services.market_data.yfinance.market.screener.yf.screen"
_PREDEFINED_PATH = (
    "app.services.market_data.yfinance.market.screener.yf.PREDEFINED_SCREENER_QUERIES"
)


def _mock_search_instance(
    quotes: list | None = None,
    news: list | None = None,
    **extras: Any,
) -> MagicMock:
    inst = MagicMock(name="yf.Search.instance")
    inst.quotes = quotes if quotes is not None else [{"symbol": "AAPL"}]
    inst.news = news if news is not None else [{"title": "Apple news"}]
    for k, v in extras.items():
        setattr(inst, k, v)
    return inst


# ===========================================================================
# SearchClient
# ===========================================================================


class TestSearchSuccess:
    def test_when_search_succeeds_then_returns_quotes_and_news(self) -> None:
        client = _build_search_client()
        inst = _mock_search_instance(
            quotes=[{"symbol": "AAPL", "shortname": "Apple"}],
            news=[{"title": "Apple beats estimates"}],
        )
        with patch(_SEARCH_PATH, return_value=inst):
            result = client.search("Apple")
        assert result is not None
        assert "quotes" in result and "news" in result
        assert_json_safe(result)

    def test_when_search_returns_result_then_quotes_match_mock(self) -> None:
        client = _build_search_client()
        quotes = [{"symbol": "MSFT"}]
        inst = _mock_search_instance(quotes=quotes)
        with patch(_SEARCH_PATH, return_value=inst):
            result = client.search("Microsoft", max_retries=1)
        assert result is not None
        assert result["quotes"] == quotes

    def test_when_include_lists_false_then_lists_key_absent(self) -> None:
        client = _build_search_client()
        with patch(_SEARCH_PATH, return_value=_mock_search_instance()):
            result = client.search("q")
        assert result is not None
        assert "lists" not in result

    def test_when_include_lists_true_then_lists_key_present(self) -> None:
        client = _build_search_client()
        inst = _mock_search_instance()
        inst.lists = [{"id": "L1"}]
        with patch(_SEARCH_PATH, return_value=inst):
            result = client.search("q", include_lists=True)
        assert result is not None
        assert "lists" in result
        assert result["lists"] == [{"id": "L1"}]

    def test_when_include_research_true_then_research_key_present(self) -> None:
        client = _build_search_client()
        inst = _mock_search_instance()
        inst.research = [{"id": "R1"}]
        with patch(_SEARCH_PATH, return_value=inst):
            result = client.search("q", include_research=True)
        assert result is not None
        assert result["research"] == [{"id": "R1"}]
        assert "lists" not in result
        assert "nav" not in result

    def test_when_include_nav_true_then_nav_key_present(self) -> None:
        client = _build_search_client()
        inst = _mock_search_instance()
        inst.nav = [{"id": "N1"}]
        with patch(_SEARCH_PATH, return_value=inst):
            result = client.search("q", include_nav=True)
        assert result is not None
        assert "nav" in result
        assert result["nav"] == [{"id": "N1"}]

    def test_when_search_raises_then_returns_none(self) -> None:
        client = _build_search_client()
        with patch(_SEARCH_PATH, side_effect=RuntimeError("network")):
            result = client.search("q", max_retries=1)
        assert result is None


class TestLookup:
    def test_when_unknown_asset_type_then_returns_none_without_calling_yf(
        self,
    ) -> None:
        client = _build_search_client()
        with patch(_LOOKUP_PATH) as mock_lk:
            result = client.lookup("Apple", asset_type="unicorn", max_retries=1)
        assert result is None
        mock_lk.assert_not_called()

    def test_when_valid_asset_type_then_calls_lookup_and_returns_list(self) -> None:
        client = _build_search_client()
        expected = [{"symbol": "AAPL", "shortname": "Apple"}]
        mock_lk_inst = MagicMock()
        mock_lk_inst.get_stock.return_value = expected
        with patch(_LOOKUP_PATH, return_value=mock_lk_inst):
            result = client.lookup("Apple", asset_type="stock", max_retries=1)
        assert result == expected

    @pytest.mark.parametrize(
        "asset_type,method_name",
        [
            ("etf", "get_etf"),
            ("mutualfund", "get_mutualfund"),
            ("index", "get_index"),
            ("future", "get_future"),
            ("currency", "get_currency"),
            ("cryptocurrency", "get_cryptocurrency"),
        ],
    )
    def test_when_asset_type_valid_then_correct_method_called(
        self, asset_type: str, method_name: str
    ) -> None:
        client = _build_search_client()
        mock_lk_inst = MagicMock()
        getattr(mock_lk_inst, method_name).return_value = [{"symbol": "X"}]
        with patch(_LOOKUP_PATH, return_value=mock_lk_inst):
            result = client.lookup("X", asset_type=asset_type, max_retries=1)
        assert result == [{"symbol": "X"}]
        getattr(mock_lk_inst, method_name).assert_called_once()

    def test_when_lookup_raises_then_returns_none(self) -> None:
        client = _build_search_client()
        mock_lk_inst = MagicMock()
        mock_lk_inst.get_stock.side_effect = RuntimeError("timeout")
        with patch(_LOOKUP_PATH, return_value=mock_lk_inst):
            result = client.lookup("Apple", asset_type="stock", max_retries=1)
        assert result is None


# ===========================================================================
# SectorIndustryClient
# ===========================================================================


class TestSectorOverview:
    def test_when_overview_dict_returned_then_passes_through(self) -> None:
        client = _build_sector_client()
        overview = {"key": "technology", "name": "Technology", "description": "..."}
        sector_inst = MagicMock()
        sector_inst.overview = overview
        with patch(_SECTOR_PATH, return_value=sector_inst):
            result = client.fetch_sector_overview("technology", max_retries=1)
        assert result == overview
        assert_json_safe(result)

    def test_when_overview_raises_then_returns_none(self) -> None:
        client = _build_sector_client()
        sector_inst = MagicMock()
        type(sector_inst).overview = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("err"))
        )
        with patch(_SECTOR_PATH, return_value=sector_inst):
            assert client.fetch_sector_overview("technology", max_retries=1) is None


class TestSectorTopCompanies:
    def test_when_df_non_empty_then_returns_df(self) -> None:
        client = _build_sector_client()
        df = pd.DataFrame({"symbol": ["AAPL"], "name": ["Apple"]})
        sector_inst = MagicMock()
        sector_inst.top_companies = df
        with patch(_SECTOR_PATH, return_value=sector_inst):
            result = client.fetch_sector_top_companies("technology", max_retries=1)
        pd.testing.assert_frame_equal(result, df)

    def test_when_top_companies_none_then_returns_none(self) -> None:
        client = _build_sector_client()
        sector_inst = MagicMock()
        sector_inst.top_companies = None
        with patch(_SECTOR_PATH, return_value=sector_inst):
            assert client.fetch_sector_top_companies("technology", max_retries=1) is None


class TestSectorIndustries:
    """Uses _is_non_empty_df guard — empty DataFrame yields None."""

    def test_when_industries_df_non_empty_then_returns_df(self) -> None:
        client = _build_sector_client()
        df = pd.DataFrame({"key": ["software"]})
        sector_inst = MagicMock()
        sector_inst.industries = df
        with patch(_SECTOR_PATH, return_value=sector_inst):
            result = client.fetch_sector_industries("technology", max_retries=1)
        pd.testing.assert_frame_equal(result, df)

    def test_when_industries_df_empty_then_returns_none(self) -> None:
        client = _build_sector_client()
        sector_inst = MagicMock()
        sector_inst.industries = pd.DataFrame()
        with patch(_SECTOR_PATH, return_value=sector_inst):
            assert client.fetch_sector_industries("technology", max_retries=1) is None

    def test_when_industries_raises_then_returns_none(self) -> None:
        client = _build_sector_client()
        sector_inst = MagicMock()
        type(sector_inst).industries = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("err"))
        )
        with patch(_SECTOR_PATH, return_value=sector_inst):
            assert client.fetch_sector_industries("technology", max_retries=1) is None


class TestIndustryOverview:
    def test_when_overview_dict_returned_then_passes_through(self) -> None:
        client = _build_sector_client()
        overview = {"key": "software-infrastructure", "name": "Software Infrastructure"}
        ind_inst = MagicMock()
        ind_inst.overview = overview
        with patch(_INDUSTRY_PATH, return_value=ind_inst):
            result = client.fetch_industry_overview(
                "software-infrastructure", max_retries=1
            )
        assert result == overview
        assert_json_safe(result)

    def test_when_overview_raises_then_returns_none(self) -> None:
        client = _build_sector_client()
        ind_inst = MagicMock()
        type(ind_inst).overview = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("err"))
        )
        with patch(_INDUSTRY_PATH, return_value=ind_inst):
            assert (
                client.fetch_industry_overview(
                    "software-infrastructure", max_retries=1
                )
                is None
            )


class TestIndustryTopEtfs:
    def test_when_df_non_empty_then_returns_df(self) -> None:
        client = _build_sector_client()
        df = pd.DataFrame({"symbol": ["QQQ"]})
        ind_inst = MagicMock()
        ind_inst.top_etfs = df
        with patch(_INDUSTRY_PATH, return_value=ind_inst):
            result = client.fetch_industry_top_etfs(
                "software-infrastructure", max_retries=1
            )
        pd.testing.assert_frame_equal(result, df)

    def test_when_top_etfs_none_then_returns_none(self) -> None:
        client = _build_sector_client()
        ind_inst = MagicMock()
        ind_inst.top_etfs = None
        with patch(_INDUSTRY_PATH, return_value=ind_inst):
            assert (
                client.fetch_industry_top_etfs(
                    "software-infrastructure", max_retries=1
                )
                is None
            )


class TestIndustryTopGrowthCompanies:
    """Uses _is_non_empty_df guard."""

    def test_when_growth_df_empty_then_returns_none(self) -> None:
        client = _build_sector_client()
        ind_inst = MagicMock()
        ind_inst.top_growth_companies = pd.DataFrame()
        with patch(_INDUSTRY_PATH, return_value=ind_inst):
            assert (
                client.fetch_industry_top_growth_companies(
                    "software-infrastructure", max_retries=1
                )
                is None
            )

    def test_when_growth_df_non_empty_then_returns_df(self) -> None:
        client = _build_sector_client()
        df = pd.DataFrame({"symbol": ["MSFT"], "growth": [0.3]})
        ind_inst = MagicMock()
        ind_inst.top_growth_companies = df
        with patch(_INDUSTRY_PATH, return_value=ind_inst):
            result = client.fetch_industry_top_growth_companies(
                "software-infrastructure", max_retries=1
            )
        pd.testing.assert_frame_equal(result, df)


# ===========================================================================
# ScreenerClient
# ===========================================================================


class TestScreenerScreen:
    def test_when_screen_succeeds_then_returns_dataframe(self) -> None:
        client = _build_screener_client()
        df = pd.DataFrame({"symbol": ["SPY", "QQQ"]})
        with patch(_SCREEN_PATH, return_value=df):
            result = client.screen("most_actives", max_retries=1)
        pd.testing.assert_frame_equal(result, df)

    def test_when_screen_raises_then_returns_none(self) -> None:
        client = _build_screener_client()
        with patch(_SCREEN_PATH, side_effect=RuntimeError("timeout")):
            result = client.screen("most_actives", max_retries=1)
        assert result is None

    def test_when_string_query_then_count_kwarg_used_not_size(self) -> None:
        client = _build_screener_client()
        df = pd.DataFrame({"symbol": ["A"]})
        with patch(_SCREEN_PATH, return_value=df) as mock_screen:
            client.screen("most_actives", size=10, max_retries=1)
        kwargs = mock_screen.call_args.kwargs
        assert "count" in kwargs
        assert "size" not in kwargs

    def test_when_non_string_query_then_size_kwarg_used(self) -> None:
        client = _build_screener_client()
        df = pd.DataFrame({"symbol": ["A"]})
        mock_query = MagicMock()  # non-str
        with patch(_SCREEN_PATH, return_value=df) as mock_screen:
            client.screen(mock_query, size=15, max_retries=1)
        kwargs = mock_screen.call_args.kwargs
        assert "size" in kwargs
        assert kwargs["size"] == 15


class TestScreenerEtfsPredefined:
    def test_when_unknown_name_then_raises_value_error(self) -> None:
        client = _build_screener_client()
        with pytest.raises(ValueError, match="Unknown ETF predefined"):
            client.screen_etfs_predefined("not_real")

    def test_when_valid_name_then_yf_screen_called_with_count(self) -> None:
        client = _build_screener_client()
        df = pd.DataFrame({"symbol": ["SPY"]})
        with patch(_SCREEN_PATH, return_value=df) as mock_screen:
            client.screen_etfs_predefined("top_etfs_us", size=20, max_retries=1)
        kwargs = mock_screen.call_args.kwargs
        assert kwargs.get("count") == 20


class TestGetPredefinedScreeners:
    def test_when_called_then_returns_patched_dict(self) -> None:
        client = _build_screener_client()
        expected = {"most_actives": {"query": {}}}
        with patch(_PREDEFINED_PATH, expected):
            result = client.get_predefined_screeners()
        assert result == expected
