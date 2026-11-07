"""The macro-news step reports progress while it walks its sources.

The step runs under the scheduler heartbeat, which renews the job lease only while
the step has reported progress within the stall timeout. A step that reports
nothing until it ends has the whole stall timeout as its time budget, and a long
rate-limited sweep would lose its lease and open the job slot for a second run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("yfinance")

from app.services.macro.macro_regime_service import (
    MacroRegimeService,
    run_macro_news_fetch,
)
from app.services.market_data.yfinance.news.macro_news import (
    MACRO_SEARCH_QUERIES,
    MACRO_TICKERS,
    MacroNewsFetcher,
)


def _fetcher() -> MacroNewsFetcher:
    search_client = MagicMock(name="search_client")
    search_client.search.return_value = {"news": []}
    fetcher = MacroNewsFetcher(
        yf_client=MagicMock(name="yf_client"),
        search_client=search_client,
        scraper=MagicMock(),
    )
    news_client = MagicMock()
    news_client.fetch.return_value = [
        {
            "title": "Fed holds rates",
            "pubDate": "2099-01-01T00:00:00",
            "publisher": "Reuters",
            "link": "https://reuters.com/a",
        }
    ]
    fetcher._news_client = news_client
    return fetcher


class TestFetchAllProgress:
    """One progress report per source, so the lease follows the work."""

    def test_every_source_reports_before_it_is_fetched(self) -> None:
        """Touch the stall watch for each ticker and each search query."""
        fetcher = _fetcher()
        on_progress = MagicMock()

        fetcher.fetch_all(max_articles=10, on_progress=on_progress)

        sources = len(MACRO_TICKERS) + len(MACRO_SEARCH_QUERIES)
        reported = [
            c.kwargs for c in on_progress.call_args_list if "current" in c.kwargs
        ]
        assert [r["current"] for r in reported] == list(range(1, sources + 1))
        assert on_progress.call_args_list[0].kwargs == {"total": sources}

    def test_progress_is_optional(self) -> None:
        """Keep the signature usable without a callback."""
        assert _fetcher().fetch_all(max_articles=10)


class TestProgressIsForwarded:
    """The job's callback reaches the fetcher from the step."""

    def test_service_hands_the_callback_to_the_fetcher(self) -> None:
        """Pass on_progress from fetch_macro_news to MacroNewsFetcher.fetch_all."""
        repo = MagicMock()
        repo.upsert_macro_news.return_value = 0
        service = MacroRegimeService(repo)
        fetcher = MagicMock()
        fetcher.fetch_all.return_value = []
        on_progress = MagicMock()

        with (
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.market_data.yfinance.news.macro_news.MacroNewsFetcher",
                return_value=fetcher,
            ),
        ):
            service.fetch_macro_news(on_progress=on_progress)

        assert fetcher.fetch_all.call_args.kwargs["on_progress"] is on_progress

    def test_the_step_hands_its_callback_to_the_service(self) -> None:
        """Pass on_progress from run_macro_news_fetch to the service."""
        request = SimpleNamespace(max_articles=5, fetch_full_content=False)
        service: Any = MagicMock()
        service.fetch_macro_news.return_value = {"count": 0, "errors": []}
        on_progress = MagicMock()

        with (
            patch("app.database.database_manager"),
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=service,
            ),
        ):
            run_macro_news_fetch(request, on_progress=on_progress)

        assert service.fetch_macro_news.call_args.kwargs["on_progress"] is on_progress
