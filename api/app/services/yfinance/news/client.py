from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..protocols import ArticleScraperProtocol, YFinanceClientProtocol
from ..infrastructure import retry_with_backoff


@dataclass
class NewsClient:
    yf_client: YFinanceClientProtocol
    scraper: ArticleScraperProtocol | None = None
    default_max_retries: int = 3

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
        max_retries = max_retries or self.default_max_retries

        def _action() -> list[dict[str, Any]] | None:
            ticker = self.yf_client.get_ticker(symbol)
            return ticker.news

        news = retry_with_backoff(
            _action,
            max_retries,
            is_valid=lambda n: n is not None,
        )

        if news is not None and fetch_full_content:
            self._enrich_with_content(news, max_articles)

        return news

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
