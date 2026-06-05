"""Unit tests for news sub-clients: client + aggregator.

Coverage targets
----------------
* app.services.market_data.yfinance.news.client
* app.services.market_data.yfinance.news.aggregator

(``news.macro_news`` lives in ``test_yfinance_macro_news_service.py`` to keep
each file under the 500-line cap.)

DB / repository note: these modules contain no database or repository logic.

Design
------
* NewsClient is a dataclass; pass ``scraper=MagicMock()`` to suppress the
  lazy ``ArticleScraper()`` import in ``__post_init__``.
* CountryNewsFetcher: replace ``fetcher._news_client`` after construction to
  control article flow without touching the yf_client chain.
* MacroNewsFetcher: similarly replace ``fetcher._news_client`` and supply a
  mock ``search_client``.
* ``max_retries=1`` on every NewsClient.fetch call (no sleep on failure paths).
* ``assert_json_safe`` is applied to dict / list returns only.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.news.aggregator import (
    CountryNewsFetcher,
    _is_article_recent,
    _parse_article_date,
)
from app.services.market_data.yfinance.news.client import NewsClient
from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _stale_iso() -> str:
    return (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%S")


def _make_article(
    title: str = "Test article",
    pub_date: str | None = None,
    publisher: str = "Reuters",
    link: str = "https://example.com/article",
) -> dict[str, Any]:
    return {
        "title": title,
        "pubDate": pub_date or _today_iso(),
        "publisher": publisher,
        "link": link,
    }


_SENTINEL = object()  # distinguishes "caller passed None" from "not passed"


def _make_yf_client(news: list | None = _SENTINEL) -> MagicMock:  # type: ignore[assignment]
    """Return a mock YFinanceClientProtocol with ticker.news configured.

    Pass ``news=None`` to simulate yfinance returning None for the news feed.
    Omit the argument (default) to get an empty list.
    """
    yf_client = MagicMock(name="yf_client")
    ticker = MagicMock(name="ticker")
    ticker.news = [] if news is _SENTINEL else news
    yf_client.get_ticker.return_value = ticker
    return yf_client


# ===========================================================================
# news/client.py — NewsClient
# ===========================================================================


class TestNewsClientFetch:
    def test_when_news_returned_then_returns_list(self) -> None:
        articles = [_make_article()]
        yf_client = _make_yf_client(news=articles)
        client = NewsClient(yf_client=yf_client, scraper=MagicMock())
        result = client.fetch("^GSPC", max_retries=1)
        assert result == articles

    def test_when_news_is_none_then_returns_none(self) -> None:
        yf_client = _make_yf_client(news=None)
        client = NewsClient(yf_client=yf_client, scraper=MagicMock())
        result = client.fetch("^GSPC", max_retries=1)
        assert result is None

    def test_when_get_ticker_raises_then_returns_none(self) -> None:
        yf_client = MagicMock()
        yf_client.get_ticker.side_effect = RuntimeError("network")
        client = NewsClient(yf_client=yf_client, scraper=MagicMock())
        result = client.fetch("^GSPC", max_retries=1)
        assert result is None

    def test_when_fetch_full_content_true_and_scraper_succeeds_then_enriches(
        self,
    ) -> None:
        article = _make_article(link="https://example.com/1")
        yf_client = _make_yf_client(news=[article])
        scraper = MagicMock()
        scraper.fetch.return_value = {
            "success": True,
            "content": "Full text here",
            "content_length": 14,
            "error": None,
        }
        client = NewsClient(yf_client=yf_client, scraper=scraper)
        result = client.fetch("^GSPC", max_retries=1, fetch_full_content=True)
        assert result is not None
        assert result[0]["full_content"] == "Full text here"
        assert result[0]["content_length"] == 14

    def test_when_fetch_full_content_true_and_scraper_fails_then_sets_error(
        self,
    ) -> None:
        article = _make_article(link="https://example.com/2")
        yf_client = _make_yf_client(news=[article])
        scraper = MagicMock()
        scraper.fetch.return_value = {
            "success": False,
            "content": None,
            "content_length": 0,
            "error": "403 Forbidden",
        }
        client = NewsClient(yf_client=yf_client, scraper=scraper)
        result = client.fetch("^GSPC", max_retries=1, fetch_full_content=True)
        assert result is not None
        assert result[0]["full_content"] is None
        assert result[0]["content_error"] == "403 Forbidden"

    def test_when_max_articles_set_then_only_that_many_enriched(self) -> None:
        articles = [_make_article(title=f"Art {i}") for i in range(5)]
        yf_client = _make_yf_client(news=articles)
        scraper = MagicMock()
        scraper.fetch.return_value = {
            "success": True,
            "content": "x",
            "content_length": 1,
            "error": None,
        }
        client = NewsClient(yf_client=yf_client, scraper=scraper)
        client.fetch("^GSPC", max_retries=1, fetch_full_content=True, max_articles=2)
        assert scraper.fetch.call_count == 2


class TestExtractLink:
    def _client(self) -> NewsClient:
        return NewsClient(yf_client=MagicMock(), scraper=MagicMock())

    def test_when_canonical_url_is_dict_then_extracts_url(self) -> None:
        article = {"content": {"canonicalUrl": {"url": "https://example.com/a"}}}
        assert self._client()._extract_link(article) == "https://example.com/a"

    def test_when_canonical_url_is_string_then_returns_string(self) -> None:
        article = {"content": {"canonicalUrl": "https://example.com/b"}}
        assert self._client()._extract_link(article) == "https://example.com/b"

    def test_when_no_content_key_then_falls_back_to_article_link(self) -> None:
        article = {"link": "https://example.com/c"}
        assert self._client()._extract_link(article) == "https://example.com/c"

    def test_when_no_link_anywhere_then_returns_none(self) -> None:
        article: dict[str, Any] = {}
        assert self._client()._extract_link(article) is None


# ===========================================================================
# news/aggregator.py — helpers
# ===========================================================================


class TestParsArticleDate:
    def test_when_int_epoch_then_returns_datetime(self) -> None:
        epoch = int(datetime(2024, 3, 1).timestamp())
        result = _parse_article_date(epoch)
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_when_iso_string_then_returns_datetime(self) -> None:
        result = _parse_article_date("2024-06-01T12:00:00")
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_when_datetime_object_then_returns_same(self) -> None:
        dt = datetime(2024, 1, 1)
        assert _parse_article_date(dt) == dt

    def test_when_none_then_returns_none(self) -> None:
        assert _parse_article_date(None) is None

    def test_when_na_string_then_returns_none(self) -> None:
        assert _parse_article_date("N/A") is None

    def test_when_invalid_string_then_returns_none(self) -> None:
        assert _parse_article_date("not-a-date") is None


class TestIsArticleRecent:
    def test_when_today_iso_then_returns_true(self) -> None:
        assert _is_article_recent(_today_iso()) is True

    def test_when_stale_date_then_returns_false(self) -> None:
        assert _is_article_recent(_stale_iso()) is False

    def test_when_unparseable_then_returns_false(self) -> None:
        assert _is_article_recent("garbage") is False

    def test_when_none_then_returns_false(self) -> None:
        assert _is_article_recent(None) is False

    def test_when_epoch_today_then_returns_true(self) -> None:
        epoch = int(datetime.now().timestamp())
        assert _is_article_recent(epoch) is True


# ===========================================================================
# news/aggregator.py — CountryNewsFetcher
# ===========================================================================


def _build_country_fetcher(
    country_tickers: dict[str, list[str]] | None = None,
) -> CountryNewsFetcher:
    yf_client = MagicMock(name="yf_client")
    fetcher = CountryNewsFetcher(
        yf_client=yf_client,
        scraper=MagicMock(),
        country_tickers=country_tickers or {"USA": ["^GSPC"]},
    )
    return fetcher


class TestCountryNewsFetcher:
    def test_when_unknown_country_then_returns_empty_list(self) -> None:
        fetcher = _build_country_fetcher()
        assert fetcher.fetch_for_country("Atlantis", fetch_full_content=False) == []

    def test_when_news_client_returns_articles_then_deduplicates_by_title(
        self,
    ) -> None:
        fetcher = _build_country_fetcher(country_tickers={"USA": ["^GSPC", "^DJI"]})
        article = {
            "title": "Markets rise",
            "pubDate": _today_iso(),
            "publisher": "Reuters",
            "link": "https://example.com/1",
        }
        mock_news_client = MagicMock()
        mock_news_client.fetch.return_value = [article]
        fetcher._news_client = mock_news_client
        result = fetcher.fetch_for_country("USA", fetch_full_content=False)
        titles = [a["title"] for a in result]
        assert titles.count("Markets rise") == 1

    def test_when_article_is_recent_then_included_in_result(self) -> None:
        fetcher = _build_country_fetcher()
        article = {
            "title": "Fresh news",
            "pubDate": _today_iso(),
            "publisher": "Reuters",
            "link": "https://example.com/2",
        }
        mock_news_client = MagicMock()
        mock_news_client.fetch.return_value = [article]
        fetcher._news_client = mock_news_client
        result = fetcher.fetch_for_country("USA", fetch_full_content=False)
        assert len(result) == 1
        assert result[0]["title"] == "Fresh news"
        assert_json_safe(result)

    def test_when_article_is_stale_then_excluded_from_result(self) -> None:
        fetcher = _build_country_fetcher()
        article = {
            "title": "Old news",
            "pubDate": _stale_iso(),
            "publisher": "Reuters",
            "link": "https://example.com/3",
        }
        mock_news_client = MagicMock()
        mock_news_client.fetch.return_value = [article]
        fetcher._news_client = mock_news_client
        result = fetcher.fetch_for_country("USA", fetch_full_content=False)
        assert result == []

    def test_when_news_client_returns_none_then_returns_empty_list(self) -> None:
        fetcher = _build_country_fetcher()
        mock_news_client = MagicMock()
        mock_news_client.fetch.return_value = None
        fetcher._news_client = mock_news_client
        result = fetcher.fetch_for_country("USA", fetch_full_content=False)
        assert result == []

    def test_when_fetch_full_content_true_and_scraper_present_then_scraper_called(
        self,
    ) -> None:
        fetcher = _build_country_fetcher()
        article = {
            "title": "News with content",
            "pubDate": _today_iso(),
            "publisher": "Reuters",
            "link": "https://example.com/4",
        }
        mock_news_client = MagicMock()
        mock_news_client.fetch.return_value = [article]
        fetcher._news_client = mock_news_client
        fetcher.scraper = MagicMock()
        fetcher.scraper.fetch.return_value = {
            "success": True,
            "content": "article body",
            "content_length": 12,
            "error": None,
        }
        result = fetcher.fetch_for_country("USA", fetch_full_content=True)
        assert len(result) == 1
        fetcher.scraper.fetch.assert_called_once()
