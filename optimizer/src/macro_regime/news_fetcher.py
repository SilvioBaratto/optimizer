#!/usr/bin/env python3
from datetime import datetime, timedelta
from dateutil import parser
from typing import List, Dict, Optional

from optimizer.src.yfinance import YFinanceClient
from optimizer.src.yfinance.news_client import NewsClient
from optimizer.src.yfinance.article_scraper import ArticleScraper

# Maximum age for news articles (60 days / ~2 months)
MAX_AGE_DAYS = 60

COUNTRY_NEWS_TICKERS = {
    "USA": ["^GSPC", "^DJI"],  # S&P 500, Dow Jones (55-65% allocation) - Primary news sources
    "Germany": ["^GDAXI"],  # DAX Performance Index (Europe's largest economy) - Primary news source
    "France": ["^FCHI"],  # CAC 40 Index (major European economy) - Primary news source
    "UK": ["^FTSE"],  # FTSE 100 Index (major European economy) - Primary news source
    "Japan": ["^N225"],  # Nikkei 225 Index (8-12% allocation) - Primary news source
}


def parse_article_date(pub_time) -> Optional[datetime]:
    """
    Parse article publication time from various formats.
    """
    if pub_time is None or pub_time == "N/A":
        return None

    try:
        # Unix timestamp (integer)
        if isinstance(pub_time, int):
            return datetime.fromtimestamp(pub_time)

        # ISO format string (e.g., '2025-10-08T22:00:47Z')
        if isinstance(pub_time, str):
            return parser.isoparse(pub_time)

        # Already a datetime
        if isinstance(pub_time, datetime):
            return pub_time
    except Exception:
        return None

    return None


def is_article_recent(pub_time, max_days: int = MAX_AGE_DAYS) -> bool:
    """
    Check if article is within the last max_days.
    """
    pub_date = parse_article_date(pub_time)
    if pub_date is None:
        return False  # Filter out unparseable dates

    # Remove timezone info for comparison
    if pub_date.tzinfo is not None:
        pub_date = pub_date.replace(tzinfo=None)

    cutoff_date = datetime.now() - timedelta(days=max_days)
    return pub_date >= cutoff_date


def fetch_news_for_country(
    country: str, max_articles: int = 50, fetch_full_content: bool = True
) -> List[Dict]:
    """
    Fetch macroeconomic news for a specific country.
    """
    tickers = COUNTRY_NEWS_TICKERS.get(country, [])
    if not tickers:
        return []

    all_news = []
    seen_titles = set()  # Deduplicate by title

    # Get singleton YFinanceClient and create NewsClient
    yf_client = YFinanceClient.get_instance()
    news_client = NewsClient(yf_client=yf_client)
    scraper = ArticleScraper() if fetch_full_content else None

    for ticker in tickers:
        try:
            news = news_client.fetch(ticker)

            if not news or len(news) == 0:
                continue

            # Process articles
            for article in news:
                content = article.get("content", article)

                # Extract fields
                title = content.get("title", article.get("title", ""))
                pub_time = content.get("pubDate", content.get("providerPublishTime", "N/A"))

                # Filter by date
                if not is_article_recent(pub_time):
                    continue

                # Deduplicate by title
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # Extract other fields
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

                # Format date
                pub_date = parse_article_date(pub_time)
                if pub_date:
                    date_str = pub_date.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = str(pub_time)

                # Build article dictionary
                article_dict = {
                    "title": title,
                    "publisher": publisher,
                    "date": date_str,
                    "link": link,
                    "ticker_source": ticker,
                }

                # Fetch full content if requested
                if fetch_full_content and link and scraper:
                    content_result = scraper.fetch(link)
                    if content_result["success"]:
                        article_dict["full_content"] = content_result["content"]
                        article_dict["content_length"] = content_result["content_length"]
                    else:
                        article_dict["full_content"] = None
                        article_dict["content_error"] = content_result["error"]

                # Add to collection
                all_news.append(article_dict)

        except Exception:
            continue

    # Sort by date (most recent first)
    all_news.sort(key=lambda x: x["date"], reverse=True)

    # Limit to max_articles
    return all_news[:max_articles]


def fetch_news_for_all_countries(
    max_articles_per_country: int = 50, fetch_full_content: bool = True
) -> Dict[str, List[Dict]]:
    """
    Fetch news for all portfolio countries.
    """
    all_country_news = {}

    for country in COUNTRY_NEWS_TICKERS.keys():
        news = fetch_news_for_country(
            country, max_articles=max_articles_per_country, fetch_full_content=fetch_full_content
        )
        all_country_news[country] = news

    return all_country_news


if __name__ == "__main__":
    # Test news fetcher for portfolio countries
    all_results = {}
    for country in COUNTRY_NEWS_TICKERS.keys():
        news = fetch_news_for_country(country, max_articles=10, fetch_full_content=False)
        all_results[country] = news
