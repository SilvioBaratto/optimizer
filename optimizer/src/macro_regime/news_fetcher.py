#!/usr/bin/env python3
"""
News Fetcher for Macro Regime Analysis
=======================================
Fetches macroeconomic news from yfinance for portfolio countries.
Includes full article content fetching.

Supports PORTFOLIO_COUNTRIES: USA, Germany, France, UK, Japan
Note: China and India excluded (not available in Trading212)
Based on test_yfinance_macro_news.py methodology.

Uses YFinanceClient.fetch_article_content() for generic content scraping.
"""

from datetime import datetime, timedelta
from dateutil import parser
from typing import List, Dict, Optional, Any

from src.yfinance import YFinanceClient

# NOTE: COUNTRY_NEWS_TICKERS must match PORTFOLIO_COUNTRIES from ilsole_scraper.py
# PORTFOLIO_COUNTRIES = ['USA', 'Germany', 'France', 'UK', 'Japan']
# Note: China and India excluded (not available in Trading212)
# Based on portfolio guideline document pages 91-105 allocation strategy

# Maximum age for news articles (60 days / ~2 months)
MAX_AGE_DAYS = 60

# Country proxies for PORTFOLIO_COUNTRIES (use major indices + ETFs for best macro coverage)
# Based on portfolio_guideline document pages 91-105 allocation strategy
# NOTE: Indices (^symbols) typically provide better news coverage than ETFs in yfinance
COUNTRY_NEWS_TICKERS = {
    'USA': ['^GSPC', '^DJI'],  # S&P 500, Dow Jones (55-65% allocation) - Primary news sources
    'Germany': ['^GDAXI'],  # DAX Performance Index (Europe's largest economy) - Primary news source
    'France': ['^FCHI'],  # CAC 40 Index (major European economy) - Primary news source
    'UK': ['^FTSE'],  # FTSE 100 Index (major European economy) - Primary news source
    'Japan': ['^N225'],  # Nikkei 225 Index (8-12% allocation) - Primary news source
}


def parse_article_date(pub_time) -> Optional[datetime]:
    """
    Parse article publication time from various formats.

    Parameters
    ----------
    pub_time : various
        Publication time (int timestamp, ISO string, or datetime)

    Returns
    -------
    datetime or None
        Parsed datetime object, or None if parsing fails
    """
    if pub_time is None or pub_time == 'N/A':
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

    Parameters
    ----------
    pub_time : various
        Publication time
    max_days : int
        Maximum age in days (default 60)

    Returns
    -------
    bool
        True if recent, False otherwise
    """
    pub_date = parse_article_date(pub_time)
    if pub_date is None:
        return False  # Filter out unparseable dates

    # Remove timezone info for comparison
    if pub_date.tzinfo is not None:
        pub_date = pub_date.replace(tzinfo=None)

    cutoff_date = datetime.now() - timedelta(days=max_days)
    return pub_date >= cutoff_date




def fetch_news_for_country(country: str, max_articles: int = 50, fetch_full_content: bool = True) -> List[Dict]:
    """
    Fetch macroeconomic news for a specific country.

    Parameters
    ----------
    country : str
        Country code (USA, Germany, etc.)
    max_articles : int
        Maximum number of articles to return (default 50)
    fetch_full_content : bool
        If True, fetch full article content from URLs (default True)

    Returns
    -------
    list
        List of news article dicts with keys: title, publisher, date, link, full_content (optional)
    """
    tickers = COUNTRY_NEWS_TICKERS.get(country, [])
    if not tickers:
        print(f"  Warning: No news tickers configured for {country}")
        return []

    all_news = []
    seen_titles = set()  # Deduplicate by title

    # Get singleton YFinanceClient instance
    client = YFinanceClient.get_instance()

    for ticker in tickers:
        try:
            news = client.fetch_news(ticker)

            if not news or len(news) == 0:
                continue

            # Process articles
            for article in news:
                content = article.get('content', article)

                # Extract fields
                title = content.get('title', article.get('title', ''))
                pub_time = content.get('pubDate', content.get('providerPublishTime', 'N/A'))

                # Filter by date
                if not is_article_recent(pub_time):
                    continue

                # Deduplicate by title
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # Extract other fields
                publisher = content.get('provider', {}).get('displayName', '') if isinstance(content.get('provider'), dict) else content.get('publisher', '')
                link = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else content.get('link', '')

                # Format date
                pub_date = parse_article_date(pub_time)
                if pub_date:
                    date_str = pub_date.strftime('%Y-%m-%d %H:%M')
                else:
                    date_str = str(pub_time)

                # Build article dictionary
                article_dict = {
                    'title': title,
                    'publisher': publisher,
                    'date': date_str,
                    'link': link,
                    'ticker_source': ticker
                }

                # Fetch full content if requested (using generic client method)
                if fetch_full_content and link:
                    content_result = client.fetch_article_content(link)
                    if content_result['success']:
                        article_dict['full_content'] = content_result['content']
                        article_dict['content_length'] = content_result['content_length']
                    else:
                        article_dict['full_content'] = None
                        article_dict['content_error'] = content_result['error']

                # Add to collection
                all_news.append(article_dict)

        except Exception as e:
            print(f"  Warning: Failed to fetch news from {ticker}: {e}")
            continue

    # Sort by date (most recent first)
    all_news.sort(key=lambda x: x['date'], reverse=True)

    # Limit to max_articles
    return all_news[:max_articles]


def fetch_news_for_all_countries(max_articles_per_country: int = 50, fetch_full_content: bool = True) -> Dict[str, List[Dict]]:
    """
    Fetch news for all portfolio countries.

    Parameters
    ----------
    max_articles_per_country : int
        Maximum articles per country (default 50)
    fetch_full_content : bool
        If True, fetch full article content from URLs (default True)

    Returns
    -------
    dict
        Dictionary mapping country code to list of news articles
        Countries: USA, Germany, France, UK, Japan
        Note: China and India excluded (not available in Trading212)
    """
    all_country_news = {}

    for country in COUNTRY_NEWS_TICKERS.keys():
        print(f"Fetching news for {country}...")
        news = fetch_news_for_country(country, max_articles=max_articles_per_country, fetch_full_content=fetch_full_content)
        all_country_news[country] = news
        print(f"  Found {len(news)} recent articles")

    return all_country_news


if __name__ == "__main__":
    """
    Test news fetcher for all PORTFOLIO_COUNTRIES.

    Fetches macroeconomic news for:
    - USA (55-65% allocation)
    - Germany, France, UK (Europe: 15-20% allocation)
    - Japan (8-12% allocation)

    Note: China and India excluded (not available in Trading212)
    """
    print("\n" + "="*100)
    print("NEWS FETCHER TEST - PORTFOLIO COUNTRIES")
    print("="*100)
    print("\nTesting news fetch for all 5 portfolio countries (Trading212 universe)")
    print("(Fetching 10 articles per country for speed)")
    print("\nNote: China and India excluded (not available in Trading212)")
    print("="*100)

    # Test all portfolio countries
    all_results = {}

    for country in COUNTRY_NEWS_TICKERS.keys():
        print(f"\n{'='*100}")
        print(f"FETCHING NEWS: {country}")
        print(f"{'='*100}")
        print(f"Tickers: {', '.join(COUNTRY_NEWS_TICKERS[country])}")

        try:
            news = fetch_news_for_country(country, max_articles=10, fetch_full_content=False)
            all_results[country] = news

            print(f"\n✅ Fetched {len(news)} articles for {country}")

            if news:
                print("\nTop 3 articles:")
                for i, article in enumerate(news[:3], 1):
                    print(f"\n  [{i}] {article['title'][:80]}...")
                    print(f"      Publisher: {article['publisher']}")
                    print(f"      Date: {article['date']}")
                    print(f"      Source: {article['ticker_source']}")
            else:
                print(f"⚠️  No recent articles found for {country}")

        except Exception as e:
            print(f"\n❌ ERROR fetching {country} news: {e}")
            all_results[country] = []

    # Summary
    print("\n" + "="*100)
    print("SUMMARY - NEWS COVERAGE BY COUNTRY")
    print("="*100)

    total_articles = 0
    for country in COUNTRY_NEWS_TICKERS.keys():
        count = len(all_results.get(country, []))
        total_articles += count
        status = "✅" if count > 0 else "⚠️ "
        print(f"{status} {country:<10} {count:>3} articles")

    print(f"\n📊 Total articles fetched: {total_articles}")
    print(f"📊 Average per country: {total_articles/len(COUNTRY_NEWS_TICKERS):.1f}")

    # Coverage check
    print("\n" + "="*100)
    print("TICKER VALIDATION")
    print("="*100)

    for country, tickers in COUNTRY_NEWS_TICKERS.items():
        print(f"\n{country}:")
        for ticker in tickers:
            articles_from_ticker = sum(1 for a in all_results.get(country, []) if a.get('ticker_source') == ticker)
            status = "✅" if articles_from_ticker > 0 else "⚠️ "
            print(f"  {status} {ticker:<15} {articles_from_ticker} articles")

    print("\n" + "="*100)
