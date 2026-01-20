"""
Article scraper for fetching full content from news URLs.

Separate concern from yfinance client - handles web scraping
using BeautifulSoup.
"""

import time
from dataclasses import dataclass, field
from typing import TypedDict

import requests
from bs4 import BeautifulSoup

# Default HTTP headers for article fetching
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Common article content selectors (ordered by specificity)
ARTICLE_SELECTORS = [
    "article",
    "div.article-body",
    "div.article-content",
    "div.entry-content",
    "div.post-content",
    "div.content-body",
    "div.story-body",
    "div.caas-body",  # Yahoo Finance specific
    "main",
]


class ArticleResult(TypedDict):
    """Result from article fetch operation."""

    success: bool
    content: str | None
    content_length: int | None
    error: str | None


@dataclass
class ArticleScraper:
    """
    Fetches and parses article content from URLs.

    Uses BeautifulSoup to extract article text from common HTML patterns.
    Includes rate limiting delay to be respectful to servers.

    Attributes:
        timeout: Request timeout in seconds
        delay: Delay before each request in seconds
        headers: HTTP headers to use for requests

    Example:
        scraper = ArticleScraper()
        result = scraper.fetch("https://example.com/article")
        if result["success"]:
            content = result["content"]
    """

    timeout: int = 10
    delay: float = 1.0
    headers: dict[str, str] = field(default_factory=lambda: DEFAULT_HEADERS.copy())

    def fetch(self, url: str) -> ArticleResult:
        """
        Fetch and parse article content from URL.
        """
        try:
            # Add delay to be respectful to servers
            if self.delay > 0:
                time.sleep(self.delay)

            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()

            # Try to find article content using common selectors
            article_content = self._find_article_content(soup)

            if article_content:
                paragraphs = article_content.find_all("p")
                full_text = "\n\n".join(
                    [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                )

                return ArticleResult(
                    success=True,
                    content=full_text,
                    content_length=len(full_text),
                    error=None,
                )

            return ArticleResult(
                success=False,
                content=None,
                content_length=None,
                error="Could not find article content",
            )

        except requests.exceptions.Timeout:
            return ArticleResult(
                success=False,
                content=None,
                content_length=None,
                error="Request timeout",
            )
        except requests.exceptions.RequestException as e:
            return ArticleResult(
                success=False,
                content=None,
                content_length=None,
                error=f"Request failed: {str(e)}",
            )
        except Exception as e:
            return ArticleResult(
                success=False,
                content=None,
                content_length=None,
                error=f"Parsing error: {str(e)}",
            )

    def _find_article_content(self, soup: BeautifulSoup):
        """
        Find article content using common CSS selectors.
        """
        for selector in ARTICLE_SELECTORS:
            content_div = soup.select_one(selector)
            if content_div:
                return content_div

        # Fallback: get all paragraphs from body
        return soup.find("body")

    def fetch_multiple(
        self,
        urls: list[str],
        max_articles: int | None = None,
    ) -> list[ArticleResult]:
        """
        Fetch multiple articles.

        Args:
            urls: List of article URLs to fetch
            max_articles: Maximum number of articles to fetch (None = all)

        Returns:
            List of ArticleResult objects
        """
        results = []
        fetch_count = len(urls) if max_articles is None else min(len(urls), max_articles)

        for url in urls[:fetch_count]:
            result = self.fetch(url)
            results.append(result)

        return results
