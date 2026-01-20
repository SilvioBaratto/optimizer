"""
News client for fetching stock/index news with optional full content.

Separates news fetching concern from the core yfinance client,
providing cleaner interface segregation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from optimizer.src.yfinance.article_scraper import ArticleScraper

if TYPE_CHECKING:
    from optimizer.src.yfinance.client import YFinanceClient


@dataclass
class NewsClient:
    """
    Fetches news for stocks/indices with optional full content.

    Composes YFinanceClient for ticker access and ArticleScraper
    for full content fetching. Follows Single Responsibility Principle
    by focusing only on news operations.

    Attributes:
        yf_client: YFinanceClient for ticker operations
        scraper: Optional ArticleScraper for full content (lazy-initialized)
        default_max_retries: Default retry attempts for fetching

    Example:
        from optimizer.src.yfinance import get_yfinance_client
        from optimizer.src.yfinance.news_client import NewsClient

        client = get_yfinance_client()
        news_client = NewsClient(yf_client=client)

        # Basic usage (metadata only)
        news = news_client.fetch("AAPL")

        # With full content
        news = news_client.fetch("AAPL", fetch_full_content=True, max_articles=10)
    """

    yf_client: YFinanceClient
    scraper: ArticleScraper = field(default_factory=ArticleScraper)
    default_max_retries: int = 3

    def fetch(
        self,
        symbol: str,
        max_retries: int | None = None,
        fetch_full_content: bool = False,
        max_articles: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Fetch news for a ticker with retry logic.

        Args:
            symbol: Ticker symbol (stock ticker or index like '^GSPC')
            max_retries: Maximum retry attempts (uses default if None)
            fetch_full_content: If True, fetch full article content from URLs
            max_articles: If set with fetch_full_content, only fetch content
                          for first N articles

        Returns:
            List of news articles or None if fetch failed.
            Each article dict includes: title, publisher, link, providerPublishTime
            If fetch_full_content=True, also includes: full_content, content_length
            (or content_error if fetching failed)
        """
        max_retries = max_retries or self.default_max_retries

        for attempt in range(max_retries):
            try:
                ticker = self.yf_client.get_ticker(symbol)
                news = ticker.news

                if news is None:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return None

                if fetch_full_content:
                    self._enrich_with_content(news, max_articles)

                return news

            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return None

        return None

    def _enrich_with_content(
        self,
        articles: list[dict[str, Any]],
        max_articles: int | None,
    ) -> None:
        """
        Enrich articles with full content from URLs.

        Modifies articles in-place, adding full_content and content_length
        (or content_error if fetching failed).

        Args:
            articles: List of article dicts to enrich
            max_articles: Maximum number of articles to fetch content for
        """
        articles_to_fetch = (
            len(articles) if max_articles is None else min(len(articles), max_articles)
        )

        for article in articles[:articles_to_fetch]:
            link = self._extract_link(article)

            if link:
                result = self.scraper.fetch(link)
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
        """
        Extract link from article dict.

        Handles different link formats used by yfinance.

        Args:
            article: Article dict from yfinance

        Returns:
            Article URL or None if not found
        """
        content = article.get("content", article)

        # Try different link formats
        if isinstance(content.get("canonicalUrl"), dict):
            return content["canonicalUrl"].get("url")
        if "canonicalUrl" in content:
            return content["canonicalUrl"]
        if "link" in article:
            return article["link"]
        if "link" in content:
            return content["link"]

        return None
