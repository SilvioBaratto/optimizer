"""Article fetching and content extraction from financial news URLs."""

import time
from dataclasses import dataclass, field
from typing import TypedDict

import requests
from bs4 import BeautifulSoup

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
    """Result returned by ``ArticleScraper.fetch``.

    Attributes:
        success: Whether the fetch and parse succeeded.
        content: Extracted article body text, or ``None`` on failure.
        content_length: Character count of ``content``, or ``None``.
        error: Failure message when ``success`` is ``False``.
    """

    success: bool
    content: str | None
    content_length: int | None
    error: str | None


@dataclass
class ArticleScraper:
    """Scrapes full article text from a URL using common CSS selectors.

    Attributes:
        timeout: Request timeout in seconds.
        delay: Seconds to wait between requests to avoid overwhelming servers.
        headers: HTTP headers forwarded with each request.
    """

    timeout: int = 10
    delay: float = 1.0
    headers: dict[str, str] = field(default_factory=lambda: DEFAULT_HEADERS.copy())

    def fetch(self, url: str) -> ArticleResult:
        """Fetch and extract the main article text from a URL.

        Args:
            url: Article URL to fetch.

        Returns:
            ``ArticleResult`` with ``success=True`` and the extracted text,
            or ``success=False`` with an error description.
        """
        try:
            # Add delay to be respectful to servers
            if self.delay > 0:
                time.sleep(self.delay)

            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside"]
            ):
                element.decompose()

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
                error=f"Request failed: {e!s}",
            )
        except Exception as e:
            return ArticleResult(
                success=False,
                content=None,
                content_length=None,
                error=f"Parsing error: {e!s}",
            )

    def _find_article_content(self, soup: BeautifulSoup):
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
        """Fetch and extract article text from multiple URLs sequentially.

        Args:
            urls: List of article URLs to fetch.
            max_articles: Cap on the number of URLs to process. ``None``
                means all.

        Returns:
            List of ``ArticleResult`` instances in the same order as ``urls``.
        """
        results = []
        fetch_count = (
            len(urls) if max_articles is None else min(len(urls), max_articles)
        )

        for url in urls[:fetch_count]:
            result = self.fetch(url)
            results.append(result)

        return results
