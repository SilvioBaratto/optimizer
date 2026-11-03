"""``fetch_yahoo_news`` when the ticker news feed fails instead of coming back empty.

Yahoo retired the ``/xhr/ncp`` route behind ``Ticker.news``. Depending on what the
error page says, yfinance 1.7.0 either returns an empty feed or raises
``YFDataException("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***")`` even though Yahoo
itself is up. An empty feed already falls back to Yahoo search; a raised error has
to reach the same fallback instead of failing every ticker.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

from yfinance.exceptions import YFDataException, YFRateLimitError

from app.services.market_data.yfinance.news import client as news_client_module
from app.services.market_data.yfinance.news.client import NewsClient, fetch_yahoo_news
from app.services.market_data.yfinance.news.feed_health import TickerFeedHealth

_DOWN = "*** YAHOO! FINANCE IS CURRENTLY DOWN! ***"


class _FeedTicker:
    """Ticker whose ``news`` property raises or returns, like yfinance's own."""

    def __init__(self, *, error: Exception | None = None, news: Any = None) -> None:
        self._error = error
        self._news = news

    @property
    def news(self) -> Any:
        """Raise the configured error, otherwise return the configured feed."""
        if self._error is not None:
            raise self._error
        return self._news


def _client_with(ticker: _FeedTicker) -> MagicMock:
    """A yfinance client double whose ``get_ticker`` returns *ticker*."""
    yf_client = MagicMock(name="yf_client")
    yf_client.get_ticker.return_value = ticker
    return yf_client


def _search_returning(
    monkeypatch: pytest.MonkeyPatch, articles: list[dict[str, Any]]
) -> list[str]:
    """Stub the search fallback; the returned list records the symbols searched."""
    searched: list[str] = []

    def _search(symbol: str, count: int) -> list[dict[str, Any]]:
        searched.append(symbol)
        return articles

    monkeypatch.setattr(news_client_module, "_search_news", _search)
    return searched


class TestFeedRaisesDataError:
    """A ``YFDataException`` from the ticker feed is treated like an empty feed."""

    def test_search_articles_are_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The search fallback delivers the news the dead feed could not."""
        articles = [{"uuid": "u1", "title": "From search", "link": "https://x/1"}]
        searched = _search_returning(monkeypatch, articles)
        yf_client = _client_with(_FeedTicker(error=YFDataException(_DOWN)))

        assert fetch_yahoo_news("AAPL", yf_client) == articles
        assert searched == ["AAPL"]

    def test_no_search_results_gives_none_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ticker with no news anywhere is 'nothing usable', not an error."""
        _search_returning(monkeypatch, [])
        yf_client = _client_with(_FeedTicker(error=YFDataException(_DOWN)))

        assert fetch_yahoo_news("VWCE.DE", yf_client) is None

    def test_search_failures_still_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the fallback fails too the caller must still see a real error."""

        def _boom(symbol: str, count: int) -> list[dict[str, Any]]:
            raise YFDataException(_DOWN)

        monkeypatch.setattr(news_client_module, "_search_news", _boom)
        yf_client = _client_with(_FeedTicker(error=YFDataException(_DOWN)))

        with pytest.raises(YFDataException):
            fetch_yahoo_news("AAPL", yf_client)

    def test_news_client_gets_articles_through_the_same_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Macro news and the aggregator reach Yahoo through ``NewsClient``."""
        articles = [{"uuid": "u2", "title": "Macro", "link": "https://x/2"}]
        _search_returning(monkeypatch, articles)
        yf_client = _client_with(_FeedTicker(error=YFDataException(_DOWN)))
        client = NewsClient(yf_client=yf_client, scraper=MagicMock())

        assert client.fetch("^GSPC", max_retries=1) == articles


class TestOtherErrorsAreNotHidden:
    """Only the 'feed unavailable' error is rerouted; real faults keep their path."""

    def test_rate_limit_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rate limits must reach the retry and circuit-breaker logic."""
        searched = _search_returning(monkeypatch, [{"uuid": "u"}])
        yf_client = _client_with(_FeedTicker(error=YFRateLimitError()))

        with pytest.raises(YFRateLimitError):
            fetch_yahoo_news("AAPL", yf_client)
        assert searched == []

    def test_network_errors_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dropped connection is transient and is retried by the caller."""
        searched = _search_returning(monkeypatch, [{"uuid": "u"}])
        yf_client = _client_with(
            _FeedTicker(error=RuntimeError("Failed to perform, curl: (28)"))
        )

        with pytest.raises(RuntimeError):
            fetch_yahoo_news("AAPL", yf_client)
        assert searched == []

    def test_rate_limit_error_does_not_count_towards_opening_the_gate(self) -> None:
        """A rate limit says nothing about whether the feed route still exists."""
        gate = TickerFeedHealth(failure_threshold=2)
        yf_client = _client_with(_FeedTicker(error=YFRateLimitError()))

        for _ in range(5):
            with pytest.raises(YFRateLimitError):
                fetch_yahoo_news("AAPL", yf_client, feed=gate)

        assert gate.allows_request() is True


class TestDeadFeedIsSkipped:
    """Once the feed has failed repeatedly it stops being called."""

    def test_feed_is_not_called_after_the_gate_opens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every skipped call saves a request and a cookie-strategy flip."""
        articles = [{"uuid": "u3", "title": "Search", "link": "https://x/3"}]
        searched = _search_returning(monkeypatch, articles)
        yf_client = _client_with(_FeedTicker(error=YFDataException(_DOWN)))
        gate = TickerFeedHealth(failure_threshold=3, cooldown_seconds=1_000.0)

        results = [fetch_yahoo_news("AAPL", yf_client, feed=gate) for _ in range(8)]

        assert results == [articles] * 8
        assert searched == ["AAPL"] * 8
        assert yf_client.get_ticker.call_count == 3

    def test_feed_is_probed_again_after_the_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If Yahoo or yfinance gets fixed, news comes back without a restart."""
        _search_returning(monkeypatch, [])
        now = [0.0]
        gate = TickerFeedHealth(
            failure_threshold=2, cooldown_seconds=60.0, clock=lambda: now[0]
        )
        failing = _client_with(_FeedTicker(error=YFDataException(_DOWN)))
        for _ in range(2):
            fetch_yahoo_news("AAPL", failing, feed=gate)
        assert fetch_yahoo_news("AAPL", failing, feed=gate) is None
        assert failing.get_ticker.call_count == 2

        now[0] = 60.0
        articles = [{"uuid": "u4", "title": "Back", "link": "https://x/4"}]
        recovered = _client_with(_FeedTicker(news=articles))

        assert fetch_yahoo_news("AAPL", recovered, feed=gate) == articles
        assert gate.allows_request() is True

    def test_a_populated_feed_keeps_the_gate_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healthy answers between failures must never accumulate towards opening."""
        searched = _search_returning(monkeypatch, [])
        gate = TickerFeedHealth(failure_threshold=2)
        failing = _client_with(_FeedTicker(error=YFDataException(_DOWN)))
        healthy = _client_with(_FeedTicker(news=[{"uuid": "ok"}]))

        for _ in range(4):
            fetch_yahoo_news("AAPL", failing, feed=gate)
            fetch_yahoo_news("AAPL", healthy, feed=gate)

        assert failing.get_ticker.call_count == 4
        assert searched == ["AAPL"] * 4
