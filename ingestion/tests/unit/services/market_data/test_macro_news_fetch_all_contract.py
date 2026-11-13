"""Exact output of ``MacroNewsFetcher.fetch_all`` on a mixed sweep.

Validates ``app/services/market_data/yfinance/news/macro_news.py``
(``MacroNewsFetcher.fetch_all``): the order and content of the selection under
several budgets, the theme pass and the final cut, scraping order with full
content, progress reports, partial source failures and the outage message.
``test_yfinance_macro_news_service.py`` covers each rule on its own; these cases
pin how the rules combine. No requirement id is registered for macro news yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, call

import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.news.macro_news import (
    MACRO_SEARCH_QUERIES,
    MACRO_TICKERS,
    MacroNewsFetcher,
    MacroNewsUnavailableError,
)

_TICKERS = list(MACRO_TICKERS)
_QUERIES = list(MACRO_SEARCH_QUERIES)
_FED_QUERY = "Federal Reserve interest rate decision"
_EM_QUERY = "MSCI emerging markets"

_BROKEN_ITEM = {"title": "Broken", "content": "not a mapping"}


def _article(title: str, hours: float) -> dict[str, Any]:
    published = datetime.now(timezone.utc) - timedelta(hours=hours)
    return {
        "title": title,
        "pubDate": published.isoformat(),
        "publisher": "Reuters",
        "link": f"https://reuters.com/{title.replace(' ', '-').lower()}",
    }


def _answer(value: Any) -> Any:
    if isinstance(value, Exception):
        raise value
    return value


def _fetcher(
    tickers: dict[str, Any], searches: dict[str, Any], scraper: Any = None
) -> MacroNewsFetcher:
    """Build a fetcher whose feeds answer from the two maps; missing ones are empty.

    A value that is an exception is raised instead of returned.
    """
    search_client = MagicMock(name="search_client")
    search_client.search.side_effect = lambda query, max_results=8: _answer(
        searches.get(query, {"news": []})
    )
    fetcher = MacroNewsFetcher(
        yf_client=MagicMock(name="yf_client"),
        search_client=search_client,
        scraper=scraper,
    )
    fetcher._news_client = MagicMock()
    fetcher._news_client.fetch.side_effect = lambda ticker: _answer(
        tickers.get(ticker, [])
    )
    return fetcher


def _mixed_sweep(scraper: Any = None) -> MacroNewsFetcher:
    """Three delivering tickers, a failed and a raising one, and four searches."""
    tickers = {
        "^VIX": [_article(f"Vol story {i}", i) for i in range(1, 6)],
        "^TNX": [_article("Yield story 1", 0.5), _article("Yield story 2", 6)],
        "GC=F": [_article("Gold story", 48)],
        "^TYX": None,
        "^IRX": RuntimeError("feed down"),
    }
    searches = {
        _FED_QUERY: {
            "news": [_article(f"Fed search {i}", 1.5 + i) for i in range(1, 5)]
            + [_article("Yield story 1", 0.5)]
        },
        _EM_QUERY: {"news": [_article("EM search", 72)]},
        "ISM manufacturing index": None,
        "10-year Treasury yield": {"news": None},
    }
    return _fetcher(tickers, searches, scraper)


def _picked(articles: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (a["title"], a["source_ticker"], a["source_query"], a["themes"])
        for a in articles
    ]


_YIELD_1 = ("Yield story 1", "^TNX", None, "monetary_policy,yield_curve")
_YIELD_2 = ("Yield story 2", "^TNX", None, "monetary_policy,yield_curve")
_GOLD = ("Gold story", "GC=F", None, "commodity_inflation")
_EM = ("EM search", None, _EM_QUERY, "geographic_allocation")


def _vol(i: int) -> tuple[Any, ...]:
    return (f"Vol story {i}", "^VIX", None, "volatility_risk")


def _fed(i: int) -> tuple[Any, ...]:
    return (f"Fed search {i}", None, _FED_QUERY, "monetary_policy")


class TestSelectionUnderBudgets:
    """Search first, then capped ticker articles, then the theme pass and the cut."""

    @pytest.mark.parametrize(
        ("max_articles", "expected"),
        [
            pytest.param(0, [], id="no-budget"),
            # The theme pass adds older articles of missing themes, and the final
            # newest-first cut then drops the search articles taken first.
            pytest.param(2, [_YIELD_1, _vol(1)], id="theme-pass-outranks-search"),
            pytest.param(
                6,
                [_YIELD_1, _vol(1), _fed(1), _fed(2), _fed(3), _fed(4)],
                id="one-ticker-slot",
            ),
            pytest.param(
                30,
                [
                    _YIELD_1,
                    _vol(1),
                    _vol(2),
                    _fed(1),
                    _vol(3),
                    _fed(2),
                    _fed(3),
                    _fed(4),
                    _YIELD_2,
                    _GOLD,
                    _EM,
                ],
                id="everything-fits",
            ),
        ],
    )
    def test_the_sweep_returns_exactly_this_selection(
        self, max_articles: int, expected: list[tuple[Any, ...]]
    ) -> None:
        """Pin the selected articles, newest first, for each budget."""
        articles = _mixed_sweep().fetch_all(max_articles=max_articles)

        assert _picked(articles) == expected

    def test_a_theme_article_beyond_a_full_budget_is_cut_again(self) -> None:
        """The theme pass may add past the budget; the final cut keeps the newest."""
        tickers = {
            "^VIX": [_article(f"Vol story {i}", i) for i in range(1, 4)],
            "^TNX": [_article("Yield story", 10)],
        }

        articles = _fetcher(tickers, {}).fetch_all(max_articles=2)

        assert _picked(articles) == [_vol(1), _vol(2)]

    def test_a_second_run_selects_the_same_articles(self) -> None:
        """Titles kept by one run do not shadow the next run's copies."""
        fetcher = _mixed_sweep()

        first = _picked(fetcher.fetch_all(max_articles=6))
        second = _picked(fetcher.fetch_all(max_articles=6))

        assert second == first


class _Scraper:
    """Fails the ``vol-story-1`` and ``fed-search-2`` pages, stubs ``fed-search-3``."""

    def __init__(self) -> None:
        self.fetched: list[str] = []

    def fetch(self, url: str) -> dict[str, Any]:
        """Record the page's last path segment and answer as the class says."""
        self.fetched.append(url.rsplit("/", 1)[1])
        if "vol-story-1" in url or "fed-search-2" in url:
            return {"success": False, "content": None}
        if "fed-search-3" in url:
            return {"success": True, "content": "short"}
        return {"success": True, "content": "body " * 60}


_SCRAPE_ORDER = [
    "yield-story-1",
    "yield-story-2",
    "vol-story-1",
    "vol-story-2",
    "vol-story-3",
    "vol-story-4",
    "gold-story",
    "fed-search-1",
    "fed-search-2",
    "fed-search-3",
    "fed-search-4",
    "em-search",
]


class TestFullContentSelection:
    """With full content, pages are scraped per feed in recency order."""

    @pytest.mark.parametrize(
        ("max_articles", "expected"),
        [
            (4, [_YIELD_1, _vol(2), _fed(1), _fed(4)]),
            (6, [_YIELD_1, _vol(2), _fed(1), _vol(3), _fed(4), _GOLD]),
        ],
    )
    def test_only_scrapable_articles_are_selected(
        self, max_articles: int, expected: list[tuple[Any, ...]]
    ) -> None:
        """Pin the selection and every page fetched, in fetch order."""
        scraper = _Scraper()

        articles = _mixed_sweep(scraper).fetch_all(
            max_articles=max_articles, fetch_full_content=True
        )

        assert _picked(articles) == expected
        assert scraper.fetched == _SCRAPE_ORDER
        assert all(len(a["full_content"]) >= 200 for a in articles)


class TestSourceWalk:
    """Every ticker, then every query, each announced before it is fetched."""

    def test_progress_names_each_source_in_order(self) -> None:
        """Report the total once, then each source with its 1-based position."""
        on_progress = MagicMock()

        _mixed_sweep().fetch_all(max_articles=6, on_progress=on_progress)

        sources = _TICKERS + _QUERIES
        assert on_progress.call_args_list == [
            call(total=len(sources)),
            *(
                call(current=i, current_ticker=source)
                for i, source in enumerate(sources, start=1)
            ),
        ]

    def test_each_feed_and_query_is_requested_once_in_order(self) -> None:
        """Ask each ticker feed, then search each query for at most eight items."""
        fetcher = _mixed_sweep()

        fetcher.fetch_all(max_articles=6)

        assert fetcher._news_client.fetch.call_args_list == [call(t) for t in _TICKERS]
        assert fetcher.search_client.search.call_args_list == [  # type: ignore[attr-defined]
            call(q, max_results=8) for q in _QUERIES
        ]


class TestPartialSourceFailure:
    """A source that fails while its articles are processed is a failed source."""

    def test_a_ticker_feed_that_fails_midway_loses_all_its_articles(self) -> None:
        """The feed delivered, so the run is no outage, but nothing of it is kept."""
        fetcher = _fetcher({"^VIX": [_article("Vol story 1", 1), _BROKEN_ITEM]}, {})

        assert fetcher.fetch_all() == []

    def test_a_search_that_fails_midway_keeps_what_it_processed(self) -> None:
        """Articles before the broken item are kept; the ones after it are not."""
        searches = {
            _FED_QUERY: {
                "news": [
                    _article("Fed search 1", 1),
                    _BROKEN_ITEM,
                    _article("Fed search 2", 2),
                ]
            }
        }

        articles = _fetcher({}, searches).fetch_all()

        assert _picked(articles) == [
            ("Fed search 1", None, _FED_QUERY, "monetary_policy")
        ]


class TestOutageMessage:
    """The outage error counts failed and empty sources separately."""

    def test_the_message_counts_each_kind_of_silent_source(self) -> None:
        """None and a raise fail a source; ``[]`` and ``{"news": None}`` are empty."""
        tickers = {
            **dict.fromkeys(_TICKERS[:5]),
            **{t: RuntimeError("feed down") for t in _TICKERS[5:10]},
        }
        searches = {
            **dict.fromkeys(_QUERIES[:3]),
            **{q: RuntimeError("search down") for q in _QUERIES[3:5]},
            **{q: {"news": None} for q in _QUERIES[5:7]},
        }

        with pytest.raises(MacroNewsUnavailableError) as raised:
            _fetcher(tickers, searches).fetch_all()

        assert str(raised.value) == (
            "No article from any macro news source: 15 failed, 16 empty of 31 sources"
        )
