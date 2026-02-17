"""Country-level news aggregation with deduplication and recency filtering.

Extracted from ``services/macro_regime/news_fetcher.py`` and refactored to
accept dependencies via constructor injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from dateutil import parser as dateutil_parser

from ..protocols import ArticleScraperProtocol, YFinanceClientProtocol

# Default country → index ticker mapping used for news retrieval.
DEFAULT_COUNTRY_TICKERS: dict[str, list[str]] = {
    "USA": ["^GSPC", "^DJI"],
    "Germany": ["^GDAXI"],
    "France": ["^FCHI"],
    "UK": ["^FTSE"],
    "Japan": ["^N225"],
}

# Maximum age for news articles (60 days / ~2 months).
_MAX_AGE_DAYS = 60


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_article_date(pub_time: Any) -> datetime | None:
    if pub_time is None or pub_time == "N/A":
        return None
    try:
        if isinstance(pub_time, int):
            return datetime.fromtimestamp(pub_time)
        if isinstance(pub_time, str):
            return dateutil_parser.isoparse(pub_time)
        if isinstance(pub_time, datetime):
            return pub_time
    except Exception:
        return None
    return None


def _is_article_recent(pub_time: Any, max_days: int = _MAX_AGE_DAYS) -> bool:
    pub_date = _parse_article_date(pub_time)
    if pub_date is None:
        return False
    if pub_date.tzinfo is not None:
        pub_date = pub_date.replace(tzinfo=None)
    cutoff_date = datetime.now() - timedelta(days=max_days)
    return pub_date >= cutoff_date


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CountryNewsFetcher:
    """Fetches, deduplicates, and enriches financial news per country.

    Parameters
    ----------
    yf_client:
        A ``YFinanceClientProtocol``-compatible client used to create
        a ``NewsClient`` internally.
    scraper:
        Optional article scraper.  When *None*, a default
        ``ArticleScraper`` is created lazily.
    country_tickers:
        Mapping of country name → list of index tickers whose news
        feeds are aggregated.  Defaults to ``DEFAULT_COUNTRY_TICKERS``.
    """

    yf_client: YFinanceClientProtocol
    scraper: ArticleScraperProtocol | None = None
    country_tickers: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_COUNTRY_TICKERS),
    )

    def __post_init__(self) -> None:
        # Lazy-import to avoid circular dependency at module level.
        from .client import NewsClient

        self._news_client = NewsClient(
            yf_client=self.yf_client,
            scraper=self.scraper,
        )

    # -- public interface ---------------------------------------------------

    def fetch_for_country(
        self,
        country: str,
        max_articles: int = 50,
        fetch_full_content: bool = True,
    ) -> list[dict[str, Any]]:
        tickers = self.country_tickers.get(country, [])
        if not tickers:
            return []

        all_news: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        for ticker in tickers:
            try:
                news = self._news_client.fetch(ticker)
                if not news:
                    continue

                for article in news:
                    article_dict = self._process_article(
                        article, ticker, seen_titles, fetch_full_content,
                    )
                    if article_dict is not None:
                        all_news.append(article_dict)
            except Exception:
                continue

        all_news.sort(key=lambda x: x["date"], reverse=True)
        return all_news[:max_articles]

    def fetch_for_all_countries(
        self,
        max_articles_per_country: int = 50,
        fetch_full_content: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            country: self.fetch_for_country(
                country,
                max_articles=max_articles_per_country,
                fetch_full_content=fetch_full_content,
            )
            for country in self.country_tickers
        }

    # -- private helpers ----------------------------------------------------

    def _process_article(
        self,
        article: dict[str, Any],
        ticker_source: str,
        seen_titles: set[str],
        fetch_full_content: bool,
    ) -> dict[str, Any] | None:
        content = article.get("content", article)

        title = content.get("title", article.get("title", ""))
        pub_time = content.get("pubDate", content.get("providerPublishTime", "N/A"))

        if not _is_article_recent(pub_time):
            return None

        if title in seen_titles:
            return None
        seen_titles.add(title)

        publisher = (
            content.get("provider", {}).get("displayName", "")
            if isinstance(content.get("provider"), dict)
            else content.get("publisher", "")
        )
        link = (
            content.get("canonicalUrl", {}).get("url", "")
            if isinstance(content.get("canonicalUrl"), dict)
            else content.get("link", "")
        )

        pub_date = _parse_article_date(pub_time)
        date_str = pub_date.strftime("%Y-%m-%d %H:%M") if pub_date else str(pub_time)

        article_dict: dict[str, Any] = {
            "title": title,
            "publisher": publisher,
            "date": date_str,
            "link": link,
            "ticker_source": ticker_source,
        }

        if fetch_full_content and link and self.scraper is not None:
            content_result = self.scraper.fetch(link)
            if content_result["success"]:
                article_dict["full_content"] = content_result["content"]
                article_dict["content_length"] = content_result["content_length"]
            else:
                article_dict["full_content"] = None
                article_dict["content_error"] = content_result["error"]

        return article_dict
