"""Which page each news path takes from a Yahoo news item.

Validates the link choice of the four news paths, each through its own public
entry point, on the same twelve item shapes:

* the yfinance sweep (``YFinanceDataService.fetch_and_store``): the page it
  scrapes for a new article;
* ``NewsClient.fetch`` with full content: the page it scrapes;
* ``CountryNewsFetcher.fetch_for_country``: the ``link`` it returns;
* ``MacroNewsFetcher.fetch_all``: the ``link`` it returns and the ``news_id``
  derived from it.

The paths agree that a nested ``canonicalUrl`` object wins and disagree on
everything else, so each expectation table pins one path's behaviour. No
requirement id is registered for link selection yet.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.news.aggregator import CountryNewsFetcher
from app.services.market_data.yfinance.news.client import NewsClient
from app.services.market_data.yfinance.news.macro_news import (
    MACRO_TICKERS,
    MacroNewsFetcher,
    _make_news_id,
)
from app.services.market_data.yfinance_data_service import YFinanceDataService
from tests._fixtures.yfinance_fakes import make_repo, make_silent_yf_client

_SCRAPER_CLASS = "app.services.market_data.yfinance.news.scraper.ArticleScraper"

_TITLE = "Fed holds rates steady"
_CANONICAL = "https://canonical.example/story"
_CANONICAL_TEXT = "https://canonical-text.example/story"
_PREVIEW = "https://preview.example/story"
_CONTENT_LINK = "https://content.example/story"
_ITEM_LINK = "https://item.example/story"
_CLICK_THROUGH = "https://click.example/story"
_NOT_HTTP = "ftp://files.example/story"


def _metadata() -> dict[str, Any]:
    return {
        "title": _TITLE,
        "pubDate": datetime.now(timezone.utc).isoformat(),
        "provider": {"displayName": "Reuters"},
    }


def _nested(item_link: str | None = None, **link_fields: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"content": {**_metadata(), **link_fields}}
    if item_link is not None:
        item["link"] = item_link
    return item


def _flat(**link_fields: Any) -> dict[str, Any]:
    return {
        "title": _TITLE,
        "pubDate": datetime.now(timezone.utc).isoformat(),
        "publisher": "Reuters",
        **link_fields,
    }


_ITEMS: dict[str, Callable[[], dict[str, Any]]] = {
    "canonical-object": lambda: _nested(
        _ITEM_LINK,
        canonicalUrl={"url": _CANONICAL},
        previewUrl=_PREVIEW,
        link=_CONTENT_LINK,
    ),
    "canonical-object-without-url": lambda: _nested(
        _ITEM_LINK, canonicalUrl={}, previewUrl=_PREVIEW, link=_CONTENT_LINK
    ),
    "canonical-string": lambda: _nested(
        _ITEM_LINK,
        canonicalUrl=_CANONICAL_TEXT,
        previewUrl=_PREVIEW,
        link=_CONTENT_LINK,
    ),
    "canonical-none": lambda: _nested(
        _ITEM_LINK, canonicalUrl=None, previewUrl=_PREVIEW
    ),
    "preview-only": lambda: _nested(previewUrl=_PREVIEW),
    "item-link-beside-content": lambda: _nested(_ITEM_LINK),
    "content-link-only": lambda: _nested(link=_CONTENT_LINK),
    "flat-item": lambda: _flat(link=_ITEM_LINK),
    "flat-item-with-canonical-object": lambda: _flat(
        canonicalUrl={"url": _CANONICAL}, link=_ITEM_LINK
    ),
    "click-through-only": lambda: _nested(clickThroughUrl={"url": _CLICK_THROUGH}),
    "not-http-canonical": lambda: _nested(canonicalUrl={"url": _NOT_HTTP}),
    "no-link": lambda: _nested(),
}

# Page scraped by the yfinance sweep; None means nothing is scraped.
_SWEEP_SCRAPES: dict[str, str | None] = {
    "canonical-object": _CANONICAL,
    "canonical-object-without-url": None,
    "canonical-string": _PREVIEW,
    "canonical-none": _PREVIEW,
    "preview-only": _PREVIEW,
    "item-link-beside-content": _ITEM_LINK,
    "content-link-only": None,
    "flat-item": _ITEM_LINK,
    "flat-item-with-canonical-object": _CANONICAL,
    "click-through-only": None,
    "not-http-canonical": _NOT_HTTP,
    "no-link": None,
}

# Page scraped by NewsClient.fetch; None means "No link available".
_NEWS_CLIENT_SCRAPES: dict[str, str | None] = {
    "canonical-object": _CANONICAL,
    "canonical-object-without-url": None,
    "canonical-string": _CANONICAL_TEXT,
    "canonical-none": None,
    "preview-only": None,
    "item-link-beside-content": _ITEM_LINK,
    "content-link-only": _CONTENT_LINK,
    "flat-item": _ITEM_LINK,
    "flat-item-with-canonical-object": _CANONICAL,
    "click-through-only": None,
    "not-http-canonical": _NOT_HTTP,
    "no-link": None,
}

# ``link`` returned by both the country and the macro fetchers.
_STORED_LINK: dict[str, str] = {
    "canonical-object": _CANONICAL,
    "canonical-object-without-url": "",
    "canonical-string": _CONTENT_LINK,
    "canonical-none": "",
    "preview-only": "",
    "item-link-beside-content": "",
    "content-link-only": _CONTENT_LINK,
    "flat-item": _ITEM_LINK,
    "flat-item-with-canonical-object": _CANONICAL,
    "click-through-only": "",
    "not-http-canonical": _NOT_HTTP,
    "no-link": "",
}

_CASES = list(_ITEMS)


def _scraped_by_sweep(item: dict[str, Any]) -> list[str]:
    feed = MagicMock()
    feed.news = [item]
    yf_client = make_silent_yf_client()
    yf_client.get_ticker.return_value = feed
    service = YFinanceDataService(make_repo(), yf_client, news_scrape_delay=0.0)
    with patch(_SCRAPER_CLASS) as scraper_class:
        scraper_class.return_value.fetch.return_value = {
            "success": True,
            "content": "body",
        }
        service.fetch_and_store(
            instrument_id=uuid4(), yfinance_ticker="AAPL", mode="full"
        )
    return [call.args[0] for call in scraper_class.return_value.fetch.call_args_list]


def _yf_client_with_news(news: list[dict[str, Any]]) -> MagicMock:
    yf_client = MagicMock(name="yf_client")
    yf_client.get_ticker.return_value.news = news
    return yf_client


class TestSweepLink:
    """The sweep scrapes the canonical object's url, else previewUrl, else link."""

    @pytest.mark.parametrize("case", _CASES)
    def test_the_sweep_scrapes_the_page_its_rule_picks(self, case: str) -> None:
        """Scrape at most one page per item, the one ``_SWEEP_SCRAPES`` names."""
        expected = _SWEEP_SCRAPES[case]

        scraped = _scraped_by_sweep(_ITEMS[case]())

        assert scraped == ([] if expected is None else [expected])


class TestNewsClientLink:
    """NewsClient also takes a non-object canonicalUrl and either ``link``."""

    @pytest.mark.parametrize("case", _CASES)
    def test_the_news_client_scrapes_the_page_its_rule_picks(self, case: str) -> None:
        """Scrape the page ``_NEWS_CLIENT_SCRAPES`` names, else record no link."""
        item = _ITEMS[case]()
        scraper = MagicMock()
        scraper.fetch.return_value = {
            "success": True,
            "content": "body",
            "content_length": 4,
        }
        client = NewsClient(yf_client=_yf_client_with_news([item]), scraper=scraper)

        client.fetch("^GSPC", max_retries=1, fetch_full_content=True)

        expected = _NEWS_CLIENT_SCRAPES[case]
        scraped = [call.args[0] for call in scraper.fetch.call_args_list]
        assert scraped == ([] if expected is None else [expected])
        if expected is None:
            assert item["content_error"] == "No link available"


class TestCountryNewsLink:
    """The country fetcher reads only the canonical object or the content's link."""

    @pytest.mark.parametrize("case", _CASES)
    def test_the_country_fetcher_returns_the_link_its_rule_picks(
        self, case: str
    ) -> None:
        """Return the ``link`` that ``_STORED_LINK`` names, ``""`` for none."""
        fetcher = CountryNewsFetcher(
            yf_client=MagicMock(name="yf_client"),
            scraper=MagicMock(),
            country_tickers={"USA": ["^GSPC"]},
        )
        fetcher._news_client = MagicMock()
        fetcher._news_client.fetch.return_value = [_ITEMS[case]()]

        articles = fetcher.fetch_for_country("USA", fetch_full_content=False)

        assert [a["link"] for a in articles] == [_STORED_LINK[case]]


class TestMacroNewsLink:
    """The macro fetcher shares the country rule, and the id is hashed from it."""

    @pytest.mark.parametrize("case", _CASES)
    def test_the_macro_fetcher_returns_the_link_its_rule_picks(self, case: str) -> None:
        """Return the ``_STORED_LINK`` link and the id hashed from title and link."""
        first_ticker = next(iter(MACRO_TICKERS))
        item = _ITEMS[case]()
        search_client = MagicMock(name="search_client")
        search_client.search.return_value = {"news": []}
        fetcher = MacroNewsFetcher(
            yf_client=MagicMock(name="yf_client"),
            search_client=search_client,
            scraper=MagicMock(),
        )
        fetcher._news_client = MagicMock()
        fetcher._news_client.fetch.side_effect = lambda ticker: (
            [item] if ticker == first_ticker else []
        )

        articles = fetcher.fetch_all(max_articles=30)

        expected = _STORED_LINK[case]
        assert [(a["link"], a["news_id"]) for a in articles] == [
            (expected, _make_news_id(_TITLE, expected))
        ]
