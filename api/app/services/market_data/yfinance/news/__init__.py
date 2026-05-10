"""News fetching and article scraping."""

from .aggregator import CountryNewsFetcher
from .client import NewsClient
from .macro_news import MacroNewsFetcher, MacroTheme
from .scraper import ArticleResult, ArticleScraper

__all__ = [
    "ArticleResult",
    "ArticleScraper",
    "CountryNewsFetcher",
    "MacroNewsFetcher",
    "MacroTheme",
    "NewsClient",
]
