"""Low-level yfinance news client with optional full-content scraping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yfinance as yf
from yfinance.exceptions import YFDataException

from .._base import fetch_with_resilience
from ..protocols import ArticleScraperProtocol, YFinanceClientProtocol
from .feed_health import TickerFeedHealth

logger = logging.getLogger(__name__)

TICKER_FEED_HEALTH = TickerFeedHealth()


def _search_news(symbol: str, count: int) -> list[dict[str, Any]]:
    """Fetch articles from Yahoo's search endpoint.

    ``Ticker.news`` requests ``/xhr/ncp``, which Yahoo now answers with 404, so it
    yields nothing for every symbol: an empty list, or a raised
    ``YFDataException`` when the error page carries yfinance's "right back" text.
    The search endpoint still serves the same flat article shape
    (``uuid``/``link``/``providerPublishTime``).
    """
    return list(yf.Search(symbol, max_results=0, news_count=count).news or [])


def fetch_yahoo_news(
    symbol: str,
    yf_client: YFinanceClientProtocol,
    count: int = 10,
    *,
    feed: TickerFeedHealth | None = None,
) -> list[dict[str, Any]] | None:
    """Fetch a ticker's news, falling back to the search endpoint.

    The search endpoint is used when the ticker feed is empty and also when it
    fails with ``YFDataException``, the error yfinance raises for any non-JSON
    answer, including the 404 page of the retired route. After repeated failures
    the feed is skipped for a while so a sweep does not pay for a dead route on
    every ticker. Rate limits and network errors are not caught: they belong to the
    caller's retry and circuit-breaker logic.

    Args:
        symbol: Yahoo Finance ticker symbol.
        yf_client: Client used to obtain the ``Ticker`` object.
        count: Maximum articles requested from the search fallback.
        feed: Memory of whether the ticker feed answers; defaults to the
            process-wide :data:`TICKER_FEED_HEALTH`.

    Returns:
        Article dicts from the ticker feed, else from search. When both come back
        empty, the ticker feed's own empty value (``[]`` or ``None``) is returned
        so callers keep seeing the signal they always did; ``None`` when the feed
        failed.

    Raises:
        YFDataException: The search fallback failed as well.
    """
    health = feed if feed is not None else TICKER_FEED_HEALTH
    news: list[dict[str, Any]] | None = None
    if health.allows_request():
        try:
            news = yf_client.get_ticker(symbol).news
        except YFDataException as exc:
            health.record_failure()
            logger.debug("ticker news feed failed for %s: %s", symbol, exc)
        else:
            health.record_success()
    if news:
        return news
    return _search_news(symbol, count) or news


@dataclass
class NewsClient:
    """Fetches ticker news via yfinance with optional article scraping.

    Attributes:
        yf_client: YFinanceClientProtocol used to retrieve ticker objects.
        scraper: Article scraper; when ``None``, a default ``ArticleScraper``
            is instantiated in ``__post_init__``.
        default_max_retries: Retry count used when callers pass ``None``.
        empty_feed_alert_after: Number of consecutive tickers without any
            article after which a single warning is logged.
        news_count: Maximum articles requested from the search fallback.
    """

    yf_client: YFinanceClientProtocol
    scraper: ArticleScraperProtocol | None = None
    default_max_retries: int = 3
    empty_feed_alert_after: int = 3
    news_count: int = 10
    _empty_feed_streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scraper is None:
            from .scraper import ArticleScraper

            self.scraper = ArticleScraper()

    def fetch(
        self,
        symbol: str,
        max_retries: int | None = None,
        fetch_full_content: bool = False,
        max_articles: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Fetch news articles for a ticker symbol.

        Args:
            symbol: Yahoo Finance ticker symbol.
            max_retries: Override the instance default retry count.
            fetch_full_content: When ``True``, scrape full article body for
                each article (up to ``max_articles``).
            max_articles: Cap on the number of articles to scrape when
                ``fetch_full_content`` is ``True``. ``None`` means all.

        Returns:
            List of article dicts, or ``None`` when Yahoo has no usable answer
            or every attempt failed.
        """
        max_retries = max_retries or self.default_max_retries

        def _action() -> list[dict[str, Any]] | None:
            return fetch_yahoo_news(symbol, self.yf_client, self.news_count)

        news = fetch_with_resilience(
            _action,
            max_retries,
            circuit_breaker=self.yf_client.circuit_breaker,
        )

        self._record_feed(symbol, news)

        if news is not None and fetch_full_content:
            self._enrich_with_content(news, max_articles)

        return news

    def _record_feed(self, symbol: str, news: list[dict[str, Any]] | None) -> None:
        # yfinance turns a failing news endpoint (HTTP 404) into an empty
        # stream, so one empty feed looks exactly like a quiet ticker. A run of
        # them is the only observable symptom of a dead endpoint.
        if news:
            self._empty_feed_streak = 0
            return

        self._empty_feed_streak += 1
        if self._empty_feed_streak == self.empty_feed_alert_after:
            logger.warning(
                "Yahoo news returned no articles for %d consecutive tickers "
                "(latest: %s); the news endpoint may be failing",
                self._empty_feed_streak,
                symbol,
            )

    def _enrich_with_content(
        self,
        articles: list[dict[str, Any]],
        max_articles: int | None,
    ) -> None:
        articles_to_fetch = (
            len(articles) if max_articles is None else min(len(articles), max_articles)
        )

        for article in articles[:articles_to_fetch]:
            link = self._extract_link(article)

            if link:
                result = self.scraper.fetch(link)  # type: ignore[union-attr]
                if result["success"]:
                    article["full_content"] = result["content"]
                    article["content_length"] = result["content_length"]
                else:
                    article["full_content"] = None
                    article["content_error"] = result["error"]
            else:
                article["full_content"] = None
                article["content_error"] = "No link available"

    def _extract_link(self, article: dict[str, Any]) -> str | None:
        content = article.get("content", article)

        if isinstance(content.get("canonicalUrl"), dict):
            return content["canonicalUrl"].get("url")
        if "canonicalUrl" in content:
            return content["canonicalUrl"]
        if "link" in article:
            return article["link"]
        if "link" in content:
            return content["link"]

        return None
