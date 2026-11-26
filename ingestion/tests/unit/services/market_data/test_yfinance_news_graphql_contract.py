"""The ticker news feed of yfinance 1.8.0 against the code that consumes it.

Validates ``fetch_yahoo_news`` (``app.services.market_data.yfinance.news.client``),
``extract_article_link`` and ``CountryNewsFetcher``.

yfinance 1.8.0 reads ``Ticker.news`` from Yahoo's GraphQL news API again, after
Yahoo retired the ``/xhr/ncp`` route. These tests run the real ``Ticker.get_news``
parser over a canned GraphQL answer, with only the HTTP POST replaced, so they pin
what the repo relies on from the library: the article shape its readers accept, and
the ``YFDataException`` that sends a failed feed to the search fallback. What
Yahoo's live endpoint answers cannot be checked offline. No requirement id is
registered for the news feed yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

import yfinance as yf
from yfinance.data import YfData

from app.services.market_data.yfinance.news import client as news_client_module
from app.services.market_data.yfinance.news._links import extract_article_link
from app.services.market_data.yfinance.news.aggregator import CountryNewsFetcher
from app.services.market_data.yfinance.news.client import fetch_yahoo_news
from app.services.market_data.yfinance.news.feed_health import TickerFeedHealth

_GRAPHQL_URL = "https://nexus-finance-prod.media.yahoo.com/"
_CANONICAL = "https://finance.yahoo.com/news/apple-beats.html"
_OUTLINK = "https://publisher.example/outlink"
_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class _Answer:
    """Stands in for the HTTP response ``YfData.post`` returns."""

    def __init__(self, text: str, payload: Any = None) -> None:
        self.text = text
        self._payload = payload

    def json(self) -> Any:
        """Return the canned JSON body."""
        return self._payload


def _graphql_answer(edges: list[dict[str, Any]]) -> _Answer:
    """Wrap *edges* in the envelope Yahoo's news GraphQL query answers with."""
    body = {
        "data": {
            "lightyearList": {
                "main": {
                    "edges": edges,
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
    }
    return _Answer("{}", body)


def _story_edge(published: datetime) -> dict[str, Any]:
    """A story whose optional attributes are present but null, as GraphQL sends them."""
    stamp = published.strftime(_STAMP_FORMAT)
    return {
        "node": {
            "asset": {
                "__typename": "Story",
                "id": "uuid-story",
                "title": "Apple beats expectations",
                "previewUrl": "https://finance.yahoo.com/m/uuid-story/apple.html",
                "isPremiumNews": False,
                "contentAttributes": {
                    "description": None,
                    "summary": "The quarter came in above estimates.",
                    "publishTime": stamp,
                    "displayTime": stamp,
                    "isHosted": True,
                    "canonicalUrl": _CANONICAL,
                    "clickthroughUrl": _CANONICAL,
                    "provider": {
                        "displayName": "Reuters",
                        "url": "https://reuters.com",
                    },
                    "thumbnail": None,
                },
                "finance": {"stockTickers": [{"symbol": "AAPL"}]},
            }
        }
    }


def _outlink_edge(published: datetime) -> dict[str, Any]:
    """An outlink, which Yahoo serves without a provider or content attributes."""
    return {
        "node": {
            "asset": {
                "__typename": "Outlink",
                "id": "uuid-outlink",
                "title": "Outlink headline",
                "description": None,
                "displayTime": published.strftime(_STAMP_FORMAT),
                "url": _OUTLINK,
                "provider": None,
                "contentAttributes": {"thumbnail": None},
            }
        }
    }


def _serve(monkeypatch: pytest.MonkeyPatch, answer: _Answer) -> list[str]:
    """Make every ``YfData.post`` return *answer*; the list records the URLs posted."""
    posted: list[str] = []

    def _post(self: YfData, url: str, *args: Any, **kwargs: Any) -> _Answer:
        posted.append(url)
        return answer

    monkeypatch.setattr(YfData, "post", _post)
    return posted


def _searching(
    monkeypatch: pytest.MonkeyPatch, articles: list[dict[str, Any]]
) -> list[str]:
    """Stub the search fallback; the list records the symbols it was asked for."""
    searched: list[str] = []

    def _search(symbol: str, count: int) -> list[dict[str, Any]]:
        searched.append(symbol)
        return articles

    monkeypatch.setattr(news_client_module, "_search_news", _search)
    return searched


def _client_for(symbol: str) -> MagicMock:
    """A yfinance client double that hands out a real, unconnected ``yf.Ticker``."""
    client = MagicMock(name="yf_client")
    client.get_ticker.return_value = yf.Ticker(symbol)
    return client


@pytest.fixture
def published() -> datetime:
    """A publish time recent enough to pass the aggregator's age filter."""
    return datetime.now(timezone.utc) - timedelta(hours=2)


class TestFeedAnswers:
    """The GraphQL feed answers with articles, so search is not needed."""

    def test_articles_are_returned_in_the_nested_shape(
        self, monkeypatch: pytest.MonkeyPatch, published: datetime
    ) -> None:
        """Items Yahoo cannot serve are dropped; the rest keep ``id`` and ``content``."""
        posted = _serve(
            monkeypatch,
            _graphql_answer(
                [
                    _story_edge(published),
                    _outlink_edge(published),
                    {"node": {"asset": {"__typename": "AssetError"}}},
                ]
            ),
        )
        searched = _searching(monkeypatch, [{"uuid": "from-search"}])

        articles = fetch_yahoo_news("AAPL", _client_for("AAPL"))

        assert articles is not None
        assert [a["id"] for a in articles] == ["uuid-story", "uuid-outlink"]
        story = articles[0]["content"]
        assert story["contentType"] == "STORY"
        assert story["provider"] == {
            "displayName": "Reuters",
            "url": "https://reuters.com",
        }
        assert story["canonicalUrl"] == {"url": _CANONICAL}
        assert story["pubDate"] == published.strftime(_STAMP_FORMAT)
        assert posted == [_GRAPHQL_URL]
        assert searched == []

    def test_readers_take_the_link_and_the_nulls_in_their_stride(
        self, monkeypatch: pytest.MonkeyPatch, published: datetime
    ) -> None:
        """Null provider and description must not break the link or country readers."""
        _serve(
            monkeypatch,
            _graphql_answer([_story_edge(published), _outlink_edge(published)]),
        )
        client = _client_for("^GSPC")
        articles = fetch_yahoo_news("^GSPC", client)
        assert articles is not None

        assert extract_article_link(articles[0]) == _CANONICAL
        assert extract_article_link(articles[1]) == _OUTLINK

        fetcher = CountryNewsFetcher(
            yf_client=client, scraper=MagicMock(), country_tickers={"USA": ["^GSPC"]}
        )
        rows = fetcher.fetch_for_country("USA", fetch_full_content=False)

        by_title = {row["title"]: row for row in rows}
        assert set(by_title) == {"Apple beats expectations", "Outlink headline"}
        assert by_title["Apple beats expectations"]["publisher"] == "Reuters"
        assert by_title["Apple beats expectations"]["link"] == _CANONICAL
        assert by_title["Outlink headline"]["publisher"] == ""
        assert by_title["Outlink headline"]["link"] == _OUTLINK


class TestFeedHasNothingOrFails:
    """An empty or failed feed still reaches the search fallback."""

    def test_empty_answer_is_a_quiet_feed_and_search_fills_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A feed that answered with no articles counts as healthy, not failed."""
        _serve(monkeypatch, _graphql_answer([]))
        found = [{"uuid": "from-search", "title": "Search hit"}]
        searched = _searching(monkeypatch, found)
        gate = TickerFeedHealth(failure_threshold=1)

        result = fetch_yahoo_news("VWCE.DE", _client_for("VWCE.DE"), feed=gate)

        assert result == found
        assert searched == ["VWCE.DE"]
        assert gate.allows_request() is True

    def test_error_page_raises_the_exception_the_fallback_catches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """yfinance keeps reporting Yahoo's error page as ``YFDataException``."""
        _serve(monkeypatch, _Answer("<html>Will be right back...</html>"))
        found = [{"uuid": "from-search", "title": "Search hit"}]
        searched = _searching(monkeypatch, found)
        gate = TickerFeedHealth(failure_threshold=1)

        result = fetch_yahoo_news("AAPL", _client_for("AAPL"), feed=gate)

        assert result == found
        assert searched == ["AAPL"]
        assert gate.allows_request() is False
