"""Article fetching and content extraction from financial news URLs."""

import re
import time
from dataclasses import dataclass, field
from typing import TypedDict

import requests
import urllib3.exceptions
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


# Postgres text columns reject NUL outright, and one rejected row fails the
# whole multi-row upsert, so a single hostile page could block every article
# in the batch. Tab, newline and carriage return are legitimate whitespace.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Small on purpose: urllib3 bounds each decoded read to this size, so it is
# also the most a gzip bomb can inflate between two cap checks.
_READ_CHUNK_BYTES = 8192


class _ResponseTooLarge(Exception):
    """Raised when a response body exceeds ``ArticleScraper.max_bytes``."""


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
        timeout: Per-socket-operation timeout in seconds. It does not bound the
            total download time, see ``max_seconds``.
        delay: Seconds to wait between requests to avoid overwhelming servers.
        max_bytes: Cap on the decoded (post-decompression) response body; a
            larger response is rejected rather than truncated.
        max_seconds: Wall-clock budget for downloading the response body.
        max_text_chars: Cap on the extracted article text, which is stored in
            an unbounded ``Text`` column.
        headers: HTTP headers forwarded with each request.
    """

    timeout: int = 10
    delay: float = 1.0
    max_bytes: int = 5 * 1024 * 1024
    max_seconds: float = 30.0
    max_text_chars: int = 200_000
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

            with requests.get(
                url, headers=self.headers, timeout=self.timeout, stream=True
            ) as response:
                response.raise_for_status()
                body = self._read_body(response)

            soup = BeautifulSoup(body, "html.parser")

            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside"]
            ):
                element.decompose()

            article_content = self._find_article_content(soup)

            if article_content:
                paragraphs = article_content.find_all("p")
                texts = (
                    _CONTROL_CHARS.sub("", p.get_text()).strip() for p in paragraphs
                )
                full_text = "\n\n".join(t for t in texts if t)[: self.max_text_chars]

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

        except _ResponseTooLarge:
            return ArticleResult(
                success=False,
                content=None,
                content_length=None,
                error=f"Response too large (over {self.max_bytes} bytes)",
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

    def _read_body(self, response: requests.Response) -> bytes:
        """Read the decoded body, enforcing ``max_bytes`` and ``max_seconds``.

        Reading ``response.raw`` skips the translation ``requests`` applies to
        urllib3 errors, so it is repeated here to keep a broken or stalled body
        reported as a request failure or timeout instead of a parsing error.
        """
        deadline = time.monotonic() + self.max_seconds
        body = bytearray()
        while True:
            # read1 returns after a single socket read, so the deadline is
            # re-checked between trickled bytes; iter_content would block
            # until a whole chunk had arrived.
            try:
                chunk = response.raw.read1(_READ_CHUNK_BYTES, decode_content=True)
            except urllib3.exceptions.ReadTimeoutError as e:
                raise requests.exceptions.ReadTimeout(str(e)) from e
            except urllib3.exceptions.ProtocolError as e:
                raise requests.exceptions.ChunkedEncodingError(str(e)) from e
            except urllib3.exceptions.DecodeError as e:
                raise requests.exceptions.ContentDecodingError(str(e)) from e
            except urllib3.exceptions.HTTPError as e:
                raise requests.exceptions.ConnectionError(str(e)) from e
            if not chunk:
                return bytes(body)
            body += chunk
            if len(body) > self.max_bytes:
                raise _ResponseTooLarge
            if time.monotonic() > deadline:
                raise requests.exceptions.Timeout(
                    f"Download exceeded {self.max_seconds}s"
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
