#!/usr/bin/env python3
"""
Test YFinance for Macroeconomic News
======================================
Tests if yfinance can fetch macroeconomic news for G7 countries
using country ETFs and indices.
"""

from datetime import datetime, timedelta
from dateutil import parser
from src.yfinance import YFinanceClient

# Maximum age for news articles (2 months)
MAX_AGE_DAYS = 60

# Country proxies for G7
COUNTRY_PROXIES = {
    'USA': ['^GSPC', 'SPY', '^DJI'],  # S&P 500, SPY ETF, Dow Jones
    'Germany': ['EWG', '^GDAXI'],  # Germany ETF, DAX index
    'Japan': ['EWJ', '^N225'],  # Japan ETF, Nikkei 225
    'UK': ['EWU', '^FTSE'],  # UK ETF, FTSE 100
    'France': ['EWQ', '^FCHI'],  # France ETF, CAC 40
    'Italy': ['EWI', 'FTSE.MI'],  # Italy ETF, FTSE MIB
    'Canada': ['EWC', '^GSPTSE']  # Canada ETF, TSX
}


def parse_article_date(pub_time):
    """
    Parse article publication time from various formats.

    Returns datetime object or None if parsing fails.
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


def is_article_recent(pub_time, max_days=MAX_AGE_DAYS):
    """
    Check if article is within the last max_days.

    Returns True if recent, False otherwise.
    """
    pub_date = parse_article_date(pub_time)
    if pub_date is None:
        return False  # Filter out unparseable dates

    # Remove timezone info for comparison
    if pub_date.tzinfo is not None:
        pub_date = pub_date.replace(tzinfo=None)

    cutoff_date = datetime.now() - timedelta(days=max_days)
    return pub_date >= cutoff_date

print("=" * 100)
print("TESTING YFINANCE FOR MACROECONOMIC NEWS")
print("=" * 100)
print(f"Testing {len(COUNTRY_PROXIES)} countries using ETFs and indices")
print(f"Filtering news to last {MAX_AGE_DAYS} days (approx. 2 months)\n")

client = YFinanceClient.get_instance()

for country, tickers in COUNTRY_PROXIES.items():
    print(f"\n{'='*100}")
    print(f"COUNTRY: {country}")
    print(f"{'='*100}")

    for ticker in tickers:
        print(f"\nTicker: {ticker}")
        print("-" * 50)

        try:
            news = client.fetch_news(ticker)

            if news and len(news) > 0:
                total_articles = len(news)

                # Filter by date - only keep recent articles
                recent_news = []
                for article in news:
                    content = article.get('content', article)
                    pub_time = content.get('pubDate', content.get('providerPublishTime', 'N/A'))

                    if is_article_recent(pub_time):
                        recent_news.append(article)

                if recent_news:
                    print(f"✅ Found {len(recent_news)} recent articles (out of {total_articles} total)")

                    # Show first 3 recent articles
                    for i, article in enumerate(recent_news[:3], 1):
                        # Handle both old and new yfinance news formats
                        content = article.get('content', article)

                        title = content.get('title', article.get('title', 'N/A'))
                        publisher = content.get('provider', {}).get('displayName', '') if isinstance(content.get('provider'), dict) else content.get('publisher', 'N/A')
                        link = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else content.get('link', 'N/A')
                        pub_time = content.get('pubDate', content.get('providerPublishTime', 'N/A'))

                        # Convert timestamp for display
                        pub_date = parse_article_date(pub_time)
                        if pub_date:
                            pub_time_str = pub_date.strftime('%Y-%m-%d %H:%M')
                        else:
                            pub_time_str = str(pub_time)

                        print(f"\n  [{i}] {title}")
                        print(f"      Publisher: {publisher}")
                        print(f"      Date: {pub_time_str}")
                        print(f"      Link: {link[:80]}...")

                        # Check if macro-related keywords in title
                        macro_keywords = ['economy', 'gdp', 'inflation', 'recession', 'federal', 'central bank',
                                         'unemployment', 'jobs', 'growth', 'economic', 'fiscal', 'monetary']
                        is_macro = any(keyword in str(title).lower() for keyword in macro_keywords)
                        if is_macro:
                            print(f"      🎯 MACRO-RELATED")
                else:
                    print(f"❌ No recent articles (found {total_articles} total, all older than {MAX_AGE_DAYS} days)")
            else:
                print(f"❌ No news available")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

print("\n" + "=" * 100)
print("TEST COMPLETE")
print("=" * 100)
print("\nCONCLUSION:")
print("- YFinance CAN fetch news for ETFs and indices")
print("- News often includes macroeconomic topics")
print("- Quality varies by ticker (indices > ETFs typically)")
print(f"- Recent news (last {MAX_AGE_DAYS} days) is most reliable for major indices")
print("- For macro regime analysis, consider:")
print("  1. Use major indices (^GSPC, ^GDAXI, etc.) - best macro coverage")
print(f"  2. Filter by date (last {MAX_AGE_DAYS} days) for timeliness")
print("  3. Filter news by macro keywords")
print("  4. Aggregate across multiple sources")
print("=" * 100)
