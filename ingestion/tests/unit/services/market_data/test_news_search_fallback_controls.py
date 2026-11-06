"""The news search fallback runs under the same request controls as every other fetch.

Once the ticker feed is skipped, every ticker's news is a ``yf.Search`` call. It has
to wait on the shared limiter, drop yfinance's cached error pages, surface the
5xx/429 answers yfinance would otherwise fold into ``[]``, and report them to the
breaker and the request-outcome tally, both when called through ``NewsClient`` and
from the bulk sweep.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("yfinance")

from yfinance.data import YfData
from yfinance.exceptions import YFDataException, YFRateLimitError

from app.services.market_data.yfinance._base import TICKER_REQUEST_KEY
from app.services.market_data.yfinance.infrastructure import (
    CircuitBreaker,
    YahooTransientError,
    install_fault_witness,
    track_request_outcomes,
)
from app.services.market_data.yfinance.news import client as news_client_module
from app.services.market_data.yfinance.news.client import (
    NewsClient,
    fetch_ticker_news,
    fetch_yahoo_news,
)
from app.services.market_data.yfinance_data_service import YFinanceDataService

_DOWN = "*** YAHOO! FINANCE IS CURRENTLY DOWN! ***"


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the backoff pause between attempts of a failing request."""
    monkeypatch.setattr("app.services.infrastructure.retry.time.sleep", lambda s: None)


def _dead_feed_client(breaker: CircuitBreaker | None = None) -> MagicMock:
    """A client whose ticker feed raises the retired-route error."""
    client = MagicMock(name="yf_client")
    if breaker is not None:
        client.circuit_breaker = breaker
    ticker = MagicMock()
    type(ticker).news = property(
        lambda self: (_ for _ in ()).throw(YFDataException(_DOWN))
    )
    client.get_ticker.return_value = ticker
    return client


def _answer_with_status(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    """Make every Yahoo request, as seen by the fault witness, answer *status*."""

    class _Response:
        status_code = status

    monkeypatch.setattr(YfData, "_make_request", lambda self, *a, **k: _Response())
    install_fault_witness()


def _search_hitting_yahoo(symbol: str, count: int) -> list[dict[str, Any]]:
    """Stand-in for ``yf.Search`` that issues one request and swallows the answer."""
    YfData._make_request(object())  # type: ignore[call-arg]
    return []


class TestFallbackRequestControls:
    """The search fallback is paced, uncached and fault-aware."""

    def test_the_fallback_waits_on_the_shared_ticker_limiter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pace the search with the limiter that paces the ticker feed."""
        client = _dead_feed_client()
        monkeypatch.setattr(news_client_module, "_search_news", lambda s, c: [])

        fetch_yahoo_news("AAPL", client)

        client.rate_limiter.acquire.assert_called_once_with(TICKER_REQUEST_KEY)

    def test_cached_error_pages_are_dropped_before_the_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not replay a failed search answer that yfinance memoised."""
        events: list[str] = []
        monkeypatch.setattr(
            news_client_module,
            "forget_cached_responses",
            lambda: events.append("forget"),
        )
        monkeypatch.setattr(
            news_client_module,
            "_search_news",
            lambda s, c: events.append("search") or [],
        )

        fetch_yahoo_news("AAPL", _dead_feed_client())

        assert events[:2] == ["forget", "search"]

    def test_a_server_error_swallowed_by_yfinance_is_raised_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 503 that leaves an empty result is a lost request, not 'no news'."""
        _answer_with_status(monkeypatch, 503)
        monkeypatch.setattr(news_client_module, "_search_news", _search_hitting_yahoo)

        with pytest.raises(YahooTransientError):
            fetch_yahoo_news("AAPL", _dead_feed_client())

    def test_a_healthy_empty_search_is_not_a_fault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keep a 200 answer with no articles as an empty result."""
        _answer_with_status(monkeypatch, 200)
        monkeypatch.setattr(news_client_module, "_search_news", _search_hitting_yahoo)

        assert fetch_yahoo_news("AAPL", _dead_feed_client()) == []


class TestFallbackReportsFailures:
    """Lost search requests reach the breaker and the outcome tally."""

    def test_a_lost_search_is_a_failure_in_the_outcome_tally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Count a 503 search as a request Yahoo never answered."""
        _answer_with_status(monkeypatch, 503)
        monkeypatch.setattr(news_client_module, "_search_news", _search_hitting_yahoo)
        client = _dead_feed_client(CircuitBreaker())

        with track_request_outcomes() as outcomes:
            assert fetch_ticker_news("AAPL", client, max_retries=2) is None

        assert len(outcomes.failures) == 1
        assert outcomes.failures[0].startswith("YahooTransientError")

    def test_a_rate_limited_search_trips_the_breaker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Let a 429 from the fallback open the breaker like any ticker request."""

        def _limited(symbol: str, count: int) -> list[dict[str, Any]]:
            raise YFRateLimitError()

        monkeypatch.setattr(news_client_module, "_search_news", _limited)
        breaker = CircuitBreaker(service_name="yfinance")
        client = _dead_feed_client(breaker)

        assert fetch_ticker_news("AAPL", client, max_retries=1) is None
        assert breaker.attempt_count == 1

    def test_news_client_uses_the_same_guarded_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Macro and country news share the sweep's request controls."""
        _answer_with_status(monkeypatch, 503)
        monkeypatch.setattr(news_client_module, "_search_news", _search_hitting_yahoo)
        client = _dead_feed_client(CircuitBreaker())

        with track_request_outcomes() as outcomes:
            news = NewsClient(yf_client=client, scraper=MagicMock()).fetch(
                "^GSPC", max_retries=1
            )

        assert news is None
        assert len(outcomes.failures) == 1


class TestSearchAnswersEmpty:
    """A search that answered with no articles is an answer, not a failure."""

    def test_failed_feed_and_empty_search_give_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Report 'no news' instead of 'source failed' when search answered []."""
        monkeypatch.setattr(news_client_module, "_search_news", lambda s, c: [])

        assert fetch_yahoo_news("VWCE.DE", _dead_feed_client()) == []

    def test_skipped_feed_and_empty_search_give_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Treat a ticker whose dead feed is gated off the same way."""
        monkeypatch.setattr(news_client_module, "_search_news", lambda s, c: [])
        gate = news_client_module.TickerFeedHealth(failure_threshold=1)
        gate.record_failure()
        client = _dead_feed_client()

        assert fetch_yahoo_news("AAPL", client, feed=gate) == []
        client.get_ticker.assert_not_called()


class TestSweepNewsFetch:
    """The bulk sweep fetches ticker news through the guarded path."""

    def _service(self, breaker: CircuitBreaker) -> tuple[YFinanceDataService, Any]:
        client = _dead_feed_client(breaker)
        client.fetch_history.return_value = None
        client.fetch_info.return_value = None
        for attr in (
            client.financials.fetch_income_stmt,
            client.financials.fetch_balance_sheet,
            client.financials.fetch_cashflow,
        ):
            attr.return_value = None
        return YFinanceDataService(MagicMock(), client), client

    def test_a_rate_limited_news_request_trips_the_breaker_and_is_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Surface a 429 from the news fallback to the breaker and the sweep tally."""

        def _limited(symbol: str, count: int) -> list[dict[str, Any]]:
            raise YFRateLimitError()

        monkeypatch.setattr(news_client_module, "_search_news", _limited)
        breaker = CircuitBreaker(service_name="yfinance")
        service, _ = self._service(breaker)

        result = service.fetch_and_store(
            uuid4(), "AAPL", mode="full", staleness=None, period="5y"
        )

        assert breaker.attempt_count >= 1
        assert result["transport_failures"] >= 1

    def test_the_sweep_stores_search_articles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keep delivering the fallback's articles to the repository."""
        articles = [{"uuid": "u1", "title": "t", "link": "https://x.test/1"}]
        monkeypatch.setattr(news_client_module, "_search_news", lambda s, c: articles)
        service, _ = self._service(CircuitBreaker())
        service.repo.get_scraped_news_uuids.return_value = {"u1"}
        service.repo.upsert_news.return_value = 1

        result = service.fetch_and_store(uuid4(), "AAPL", mode="full", staleness=None)

        service.repo.upsert_news.assert_called_once()
        assert result["counts"]["news"] == 1
