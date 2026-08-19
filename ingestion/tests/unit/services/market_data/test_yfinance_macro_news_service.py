"""Unit tests for the macro-news sub-client.

Coverage target
---------------
* app.services.market_data.yfinance.news.macro_news

Split out of ``test_yfinance_news_service.py`` to keep each test file under
the 500-line cap.  DB / repository note: this module contains no database or
repository logic.

Design
------
* MacroNewsFetcher: replace ``fetcher._news_client`` after construction and
  supply a mock ``search_client`` so no yfinance call is made.
* ``assert_json_safe`` is applied to dict / list returns only.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.news.macro_news import (
    MACRO_SEARCH_QUERIES,
    MACRO_TICKERS,
    TRUSTED_PUBLISHERS,
    MacroNewsFetcher,
    MacroTheme,
    _classify_themes,
    _make_news_id,
)
from tests._fixtures.assertions import assert_json_safe


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _stale_iso() -> str:
    return (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%S")


# ===========================================================================
# Static configuration: theme-enum → search-term / ticker mappings
# ===========================================================================


class TestMacroSearchQueriesMapping:
    def test_when_fed_rate_query_then_maps_to_monetary_policy(self) -> None:
        themes = MACRO_SEARCH_QUERIES["Federal Reserve interest rate decision"]
        assert MacroTheme.monetary_policy in themes

    def test_when_ism_query_then_maps_to_growth_and_business_cycle(self) -> None:
        themes = MACRO_SEARCH_QUERIES["ISM manufacturing index"]
        assert MacroTheme.growth_indicators in themes
        assert MacroTheme.business_cycle in themes

    def test_when_ecb_query_then_maps_to_monetary_policy(self) -> None:
        themes = MACRO_SEARCH_QUERIES["European Central Bank rate"]
        assert MacroTheme.monetary_policy in themes


class TestMacroTickersMapping:
    def test_when_vix_ticker_then_maps_to_volatility_risk(self) -> None:
        assert MacroTheme.volatility_risk in MACRO_TICKERS["^VIX"]

    def test_when_tnx_ticker_then_includes_yield_curve_and_monetary_policy(
        self,
    ) -> None:
        themes = MACRO_TICKERS["^TNX"]
        assert MacroTheme.yield_curve in themes
        assert MacroTheme.monetary_policy in themes


# ===========================================================================
# _classify_themes
# ===========================================================================


class TestClassifyThemes:
    def test_when_text_contains_federal_reserve_then_adds_monetary_policy(
        self,
    ) -> None:
        themes = _classify_themes("Federal Reserve interest rate cut", [])
        assert MacroTheme.monetary_policy in themes

    def test_when_text_contains_vix_volatility_then_adds_volatility_risk(
        self,
    ) -> None:
        themes = _classify_themes("VIX volatility index spikes", [])
        assert MacroTheme.volatility_risk in themes

    def test_when_seed_themes_provided_then_always_present_in_result(self) -> None:
        seed = [MacroTheme.yield_curve]
        themes = _classify_themes("neutral text with no keywords", seed)
        assert MacroTheme.yield_curve in themes

    def test_when_result_then_sorted_and_deduplicated(self) -> None:
        seed = [MacroTheme.monetary_policy, MacroTheme.monetary_policy]
        themes = _classify_themes("Federal Reserve interest rate", seed)
        assert themes.count(MacroTheme.monetary_policy) == 1
        values = [t.value for t in themes]
        assert values == sorted(values)


# ===========================================================================
# _make_news_id
# ===========================================================================


class TestMakeNewsId:
    def test_when_same_inputs_then_deterministic(self) -> None:
        assert _make_news_id("Title A", "https://a.com") == _make_news_id(
            "Title A", "https://a.com"
        )

    def test_when_different_inputs_then_different_id(self) -> None:
        assert _make_news_id("Title A", "https://a.com") != _make_news_id(
            "Title B", "https://a.com"
        )

    def test_when_called_then_returns_32_char_hex_string(self) -> None:
        result = _make_news_id("Title", "https://x.com")
        assert len(result) == 32
        int(result, 16)  # raises ValueError if not valid hex


# ===========================================================================
# MacroNewsFetcher.fetch_all
# ===========================================================================


def _build_macro_fetcher(
    news_articles: list | None = None,
    search_news: list | None = None,
) -> MacroNewsFetcher:
    """Build MacroNewsFetcher with mocked internals."""
    search_client = MagicMock(name="search_client")
    search_client.search.return_value = {"news": search_news or []}

    fetcher = MacroNewsFetcher(
        yf_client=MagicMock(name="yf_client"),
        search_client=search_client,
        scraper=MagicMock(),
    )

    mock_news_client = MagicMock()
    mock_news_client.fetch.return_value = news_articles or []
    fetcher._news_client = mock_news_client
    return fetcher


def _trusted_article(title: str = "Rate decision imminent") -> dict[str, Any]:
    return {
        "title": title,
        "pubDate": _today_iso(),
        "publisher": "Reuters",
        "link": f"https://reuters.com/{title.replace(' ', '-')}",
    }


def _untrusted_article(title: str = "Untrusted article") -> dict[str, Any]:
    return {
        "title": title,
        "pubDate": _today_iso(),
        "publisher": "RandomBlog",
        "link": "https://randomblog.com/article",
    }


class TestMacroNewsFetcherFetchAll:
    def test_when_trusted_article_returned_then_included_in_result(self) -> None:
        fetcher = _build_macro_fetcher(news_articles=[_trusted_article()])
        result = fetcher.fetch_all(max_articles=30)
        assert any(a["publisher"] == "Reuters" for a in result)

    def test_when_untrusted_publisher_then_article_dropped(self) -> None:
        fetcher = _build_macro_fetcher(news_articles=[_untrusted_article()])
        result = fetcher.fetch_all(max_articles=30)
        assert not any(a["publisher"] == "RandomBlog" for a in result)

    def test_when_articles_returned_then_output_is_json_safe(self) -> None:
        fetcher = _build_macro_fetcher(
            news_articles=[_trusted_article("Fed holds rates")]
        )
        result = fetcher.fetch_all(max_articles=10)
        if result:
            assert_json_safe(result)

    def test_when_duplicate_titles_then_deduplicated(self) -> None:
        article = _trusted_article("Fed holds rates steady")
        fetcher = _build_macro_fetcher(news_articles=[article, article])
        result = fetcher.fetch_all(max_articles=30)
        titles = [a["title"] for a in result if a["title"] == "Fed holds rates steady"]
        assert len(titles) <= 1

    def test_when_article_has_news_id_then_it_is_32_char_hex(self) -> None:
        fetcher = _build_macro_fetcher(
            news_articles=[_trusted_article("Inflation rises")]
        )
        result = fetcher.fetch_all(max_articles=10)
        for article in result:
            nid = article["news_id"]
            assert len(nid) == 32
            int(nid, 16)

    def test_when_article_from_search_included_then_themes_field_present(self) -> None:
        search_article = _trusted_article("Federal Reserve interest rate decision")
        fetcher = _build_macro_fetcher(search_news=[search_article])
        result = fetcher.fetch_all(max_articles=30)
        for article in result:
            assert "themes" in article

    def test_when_stale_article_then_excluded_from_results(self) -> None:
        stale = {
            "title": "Old macro news",
            "pubDate": _stale_iso(),
            "publisher": "Reuters",
            "link": "https://reuters.com/old",
        }
        fetcher = _build_macro_fetcher(news_articles=[stale])
        result = fetcher.fetch_all(max_articles=30)
        assert not any(a.get("title") == "Old macro news" for a in result)

    def test_when_reuters_is_trusted_publisher_then_in_trusted_set(self) -> None:
        assert "Reuters" in TRUSTED_PUBLISHERS

    def test_when_fetch_all_called_then_search_client_invoked_per_query(
        self,
    ) -> None:
        search_client = MagicMock()
        search_client.search.return_value = {"news": []}
        fetcher = MacroNewsFetcher(
            yf_client=MagicMock(),
            search_client=search_client,
            scraper=MagicMock(),
        )
        fetcher._news_client = MagicMock()
        fetcher._news_client.fetch.return_value = []
        fetcher.fetch_all(max_articles=5)
        assert search_client.search.call_count == len(MACRO_SEARCH_QUERIES)
