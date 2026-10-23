"""Article fetching and content extraction from financial news URLs."""

import ipaddress
import re
import socket
import time
from dataclasses import dataclass, field
from typing import TypedDict
from urllib.parse import urljoin, urlsplit

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


_ALLOWED_SCHEMES = frozenset({"http", "https"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class _ResponseTooLarge(Exception):
    """Raised when a response body exceeds ``ArticleScraper.max_bytes``."""


class _BlockedDestination(Exception):
    """Raised when a URL points somewhere the scraper must not connect to."""


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
        max_redirects: Redirect hops followed before the fetch fails. Each hop
            is validated like the original URL.
        allow_private_hosts: Skip the check that every resolved address is
            public. Only for tests and local mirrors; the scheme check stays.
    """

    timeout: int = 10
    delay: float = 1.0
    max_bytes: int = 5 * 1024 * 1024
    max_seconds: float = 30.0
    max_text_chars: int = 200_000
    headers: dict[str, str] = field(default_factory=lambda: DEFAULT_HEADERS.copy())
    max_redirects: int = 5
    allow_private_hosts: bool = False

    def fetch(self, url: str) -> ArticleResult:
        """Fetch and extract the main article text from a URL.

        Only http and https URLs whose host resolves exclusively to public
        addresses are fetched, and the same rule applies to every redirect
        hop, because article links come from third-party feeds and must not
        steer the daemon at loopback, private-network or metadata services.

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

            # One session per fetch: cookies set by a redirecting page (consent
            # and cookie-check hops) reach the next hop, but never another article.
            with requests.Session() as session, self._open(session, url) as response:
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

        except _BlockedDestination as e:
            return ArticleResult(
                success=False,
                content=None,
                content_length=None,
                error=f"Blocked URL: {e!s}",
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

    def _open(self, session: requests.Session, url: str) -> requests.Response:
        """Open ``url`` as a stream, following redirects one validated hop at a time."""
        for _ in range(self.max_redirects + 1):
            self._check_destination(url)
            # requests would follow redirects without re-running the check.
            response = session.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                stream=True,
                allow_redirects=False,
            )
            location = response.headers.get("Location")
            if response.status_code not in _REDIRECT_STATUSES or not location:
                return response
            response.close()
            url = urljoin(url, location)
        raise requests.exceptions.TooManyRedirects(
            f"Exceeded {self.max_redirects} redirects"
        )

    def _check_destination(self, url: str) -> None:
        """Raise ``_BlockedDestination`` unless ``url`` is a fetchable public target.

        The name is resolved here and again by ``requests`` when it connects,
        so a resolver that answers differently the second time is not covered.
        """
        try:
            # urlsplit and urllib3 disagree on where the authority ends for a
            # URL like ``http://127.0.0.1:80\@93.184.216.34/``: urlsplit sees
            # the public host, the connection goes to loopback. Checking the
            # URL as requests will send it leaves no second parse to differ.
            parts = urlsplit(requests.Request("GET", url).prepare().url or "")
            host = parts.hostname
            port = parts.port
        except (requests.exceptions.RequestException, ValueError) as e:
            raise _BlockedDestination(f"unsupported URL {url!r}") from e
        if parts.scheme not in _ALLOWED_SCHEMES or not host:
            raise _BlockedDestination(f"unsupported URL {url!r}")
        if self.allow_private_hosts:
            return
        try:
            port = port or (443 if parts.scheme == "https" else 80)
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (ValueError, OSError) as e:
            raise _BlockedDestination(f"cannot resolve {host!r}: {e}") from e
        for info in infos:
            address = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                address = address.ipv4_mapped
            if not address.is_global or address.is_multicast:
                raise _BlockedDestination(f"{host!r} resolves to non-public {address}")

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
