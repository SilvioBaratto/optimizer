"""News fetching and article scraping."""

from .aggregator import CountryNewsFetcher
from .client import NewsClient
from .scraper import ArticleResult, ArticleScraper

__all__ = [
    "ArticleResult",
    "ArticleScraper",
    "NewsClient",
    "CountryNewsFetcher",
]
