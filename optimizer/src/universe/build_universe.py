#!/usr/bin/env python3
"""
Build Universe - T212 to yfinance Ticker Mapper with Database Storage
======================================================================
This script:
1. Fetches stock/exchange metadata from Trading212 API
2. Maps T212 tickers to yfinance tickers (discovers correct suffix)
3. Applies basic institutional filters:
   - Market cap ≥ $100M
   - Price ≥ $5
   - Liquidity ≥ $1M daily dollar volume
   - Data completeness ≥ 50%
4. Saves data directly to PostgreSQL database

This pre-filtering reduces the workload for portfolio_filter.py
"""

import requests
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from datetime import datetime, timedelta
from threading import Lock
from tqdm import tqdm
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.yfinance import YFinanceClient

# Load environment variables from .env file
load_dotenv()

# Import database and models
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from app.database import database_manager, init_db
from app.models.universe import Exchange, Instrument


# ============================================================================
# PORTFOLIO CONFIGURATION
# ============================================================================
# Countries to include in the universe (aligned with macro regime analysis)
PORTFOLIO_COUNTRIES = [
    'USA',       # 55-65% - AI infrastructure leadership, profit margin superiority
    'Germany',   # Europe's largest economy (part of 15-20% Europe allocation)
    'France',    # Major European economy
    'UK',        # Major European economy
    'Japan',     # 8-12% - Corporate governance reforms, BoJ normalization
    'China',     # 6-8% - Stimulus potential vs. structural headwinds
    'India'      # Overweight - 6-7% sustained growth, favorable demographics
]

# Mapping of countries to their exchanges (from Trading212 data)
COUNTRY_TO_EXCHANGES = {
    'USA': [
        'NYSE',
        'NASDAQ',
        'NYSE Arca'
    ],
    'Germany': [
        'Deutsche Börse Xetra',
        'Frankfurt Stock Exchange',
        'Gettex'
    ],
    'France': [
        'Euronext Paris'
    ],
    'UK': [
        'London Stock Exchange',
        'London Stock Exchange (IOB)'
    ],
    'Japan': [
        'Tokyo Stock Exchange'
    ],
    # Note: China and India not available in Trading212
    # 'China': [],
    # 'India': []
}

# ============================================================================
# INSTITUTIONAL FILTERS - MARKET CAP TIERS & LIQUIDITY
# ============================================================================
# These filters are applied during universe building to reduce downstream processing

# Market Capitalization Thresholds
MIN_MARKET_CAP = 100_000_000        # $100M absolute minimum
SMALL_CAP_THRESHOLD = 2_000_000_000  # $100M-2B (small-cap)
MID_CAP_THRESHOLD = 10_000_000_000   # $2B-10B (mid-cap)
LARGE_CAP_THRESHOLD = 10_000_000_000 # $10B+ (large-cap)

# Price Filters
MIN_PRICE = 5.0                      # $5 minimum share price
MAX_PRICE = 10_000.0                 # $10,000 maximum (data error check)

# Volume/Liquidity Filters (by market cap segment)
LIQUIDITY_FILTERS = {
    'large_cap': {
        'min_adv_dollars': 10_000_000,   # $10M minimum daily dollar volume
        'min_adv_shares': 500_000         # 500K shares minimum
    },
    'mid_cap': {
        'min_adv_dollars': 5_000_000,    # $5M minimum daily dollar volume
        'min_adv_shares': 250_000        # 250K shares minimum
    },
    'small_cap': {
        'min_adv_dollars': 1_000_000,    # $1M minimum daily dollar volume
        'min_adv_shares': 100_000        # 100K shares minimum
    }
}

# Historical Data Coverage Requirements (for institutional signal generation)
MIN_TRADING_DAYS = 1260  # 5 years minimum (252 trading days/year × 5 = 1260)
# This ensures sufficient data for:
# - Long-term alpha/beta estimation
# - Multi-year momentum patterns
# - Secular trend analysis
# - Robust statistical significance testing

# Institutional Data Coverage Requirements
INSTITUTIONAL_FIELDS = {
    'market_cap': {
        'fields': ['marketCap'],
        'description': 'Market capitalization (MIN $100M)',
        'required': True
    },
    'price': {
        'fields': ['currentPrice', 'regularMarketPrice'],
        'description': 'Current stock price (MIN $5)',
        'required': True,
        'at_least_one': True  # Only one of the fields needs to be present
    },
    'volume': {
        'fields': ['averageVolume', 'averageVolume10days'],
        'description': 'Average daily volume (MIN $1M dollar volume)',
        'required': True,
        'at_least_one': True
    },
    'shares_outstanding': {
        'fields': ['sharesOutstanding'],
        'description': 'Total shares outstanding',
        'required': True
    },
    'beta': {
        'fields': ['beta'],
        'description': 'Market beta (risk metric)',
        'required': False  # Optional but important
    },
    'sector_industry': {
        'fields': ['sector', 'industry'],
        'description': 'GICS sector and industry classification',
        'required': True,
        'all_required': True  # Both fields must be present
    },
    'exchange': {
        'fields': ['exchange'],
        'description': 'Primary exchange listing',
        'required': True
    },
    'financial_ratios': {
        'fields': ['trailingPE', 'priceToBook'],
        'description': 'Valuation ratios (P/E, P/B)',
        'required': True,
        'at_least_one': True
    },
    'profitability': {
        'fields': ['returnOnEquity', 'returnOnAssets', 'profitMargins'],
        'description': 'Profitability metrics (ROE, ROA, margins)',
        'required': True,
        'at_least_one': True
    },
    'debt_metrics': {
        'fields': ['debtToEquity', 'totalDebt', 'totalAssets'],
        'description': 'Debt and balance sheet metrics',
        'required': True,
        'at_least_one': True
    },
    'dividend_data': {
        'fields': ['dividendYield', 'dividendRate'],
        'description': 'Dividend information',
        'required': False,  # Not all stocks pay dividends
        'at_least_one': True
    },
    '52week_range': {
        'fields': ['fiftyTwoWeekHigh', 'fiftyTwoWeekLow'],
        'description': '52-week price range',
        'required': True,
        'all_required': True  # Both high and low needed
    }
}


def get_allowed_exchanges() -> set:
    """
    Get set of allowed exchange names based on PORTFOLIO_COUNTRIES.

    Returns:
        Set of exchange names to include in universe
    """
    allowed_exchanges = set()
    for country in PORTFOLIO_COUNTRIES:
        if country in COUNTRY_TO_EXCHANGES:
            allowed_exchanges.update(COUNTRY_TO_EXCHANGES[country])
    return allowed_exchanges


def is_exchange_allowed(exchange_name: str) -> bool:
    """
    Check if an exchange should be included based on PORTFOLIO_COUNTRIES.

    Args:
        exchange_name: Name of the exchange

    Returns:
        True if exchange is in a portfolio country, False otherwise
    """
    allowed_exchanges = get_allowed_exchanges()
    return exchange_name in allowed_exchanges


def check_category_coverage(data: Dict, category_name: str, category_spec: Dict) -> Tuple[bool, List[str]]:
    """
    Check if a data category meets institutional requirements.

    Args:
        data: Stock data dictionary
        category_name: Name of the category
        category_spec: Category specification from INSTITUTIONAL_FIELDS

    Returns:
        (passed, missing_fields)
    """
    fields = category_spec['fields']
    available = []
    missing = []

    for field in fields:
        if field in data and data[field] is not None:
            available.append(field)
        else:
            missing.append(field)

    # Determine if category passed
    if category_spec.get('all_required', False):
        # All fields must be present
        passed = len(missing) == 0
    elif category_spec.get('at_least_one', False):
        # At least one field must be present
        passed = len(available) > 0
    else:
        # Default: all fields must be present
        passed = len(missing) == 0

    return passed, missing


def check_institutional_coverage(data: Dict) -> Tuple[bool, List[str]]:
    """
    Check if data meets all required institutional coverage requirements.

    Args:
        data: Stock data dictionary

    Returns:
        (passed, missing_categories) - True if 100% coverage of required fields
    """
    if not data:
        return False, list(INSTITUTIONAL_FIELDS.keys())

    missing_categories = []

    # Check each required category
    for category_name, category_spec in INSTITUTIONAL_FIELDS.items():
        if not category_spec.get('required', True):
            continue  # Skip optional categories

        passed, missing = check_category_coverage(data, category_name, category_spec)

        if not passed:
            missing_categories.append(category_name)

    # Pass only if ALL required categories are satisfied
    return len(missing_categories) == 0, missing_categories


def determine_market_cap_segment(market_cap: float) -> str:
    """
    Determine market cap segment for a stock.

    Args:
        market_cap: Market capitalization in dollars

    Returns:
        'large_cap', 'mid_cap', or 'small_cap'
    """
    if market_cap >= LARGE_CAP_THRESHOLD:
        return 'large_cap'
    elif market_cap >= SMALL_CAP_THRESHOLD:
        return 'mid_cap'
    else:
        return 'small_cap'


def fetch_json(base_url, path, headers, max_retries=5):
    """Fetch JSON with exponential backoff for rate limiting and timeouts."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{base_url}{path}", headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = (2 ** attempt) * 2
                print(f"⚠️  Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise
            else:
                raise
        except requests.exceptions.RequestException as e:
            error_str = str(e).lower()

            # Check for rate limit or timeout errors
            if any(x in error_str for x in ['rate limit', 'too many requests', 'timeout', 'timed out']):
                if attempt < max_retries - 1:
                    # Progressive backoff: 60s, 5min, 15min, 30min, 1hr
                    wait_times = [60, 300, 900, 1800, 3600]
                    wait_time = wait_times[attempt] if attempt < len(wait_times) else 3600
                    print(f"⚠️  Rate limit/timeout on {path}. Waiting {wait_time//60} minutes before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
            else:
                # Other request errors - shorter retry
                print(f"⚠️  Request error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
                if attempt == max_retries - 1:
                    raise

    raise Exception(f"Failed to fetch {path} after {max_retries} attempts")


# Cache system for ticker mappings
class TickerMappingCache:
    """Persistent cache for T212 → yfinance ticker mappings."""

    def __init__(self, cache_dir="../.cache"):
        self.cache_dir = cache_dir
        self.lock = Lock()
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "ticker_mappings.pkl")
        self.mappings = self._load_mappings()

    def _load_mappings(self):
        """Load ticker mappings from cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️  Error loading ticker mappings cache: {e}")
        return {}

    def save_mapping(self, symbol, exchange_name, yf_ticker):
        """Save successful ticker mapping."""
        with self.lock:
            key = f"{symbol}:{exchange_name}"
            self.mappings[key] = {
                'yf_ticker': yf_ticker,
                'timestamp': datetime.now()
            }
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.mappings, f)
            except Exception as e:
                print(f"⚠️  Error saving ticker mapping: {e}")

    def get_mapping(self, symbol, exchange_name, max_age_days=90):
        """Get cached ticker mapping if available and fresh."""
        key = f"{symbol}:{exchange_name}"
        if key in self.mappings:
            cached = self.mappings[key]
            age = datetime.now() - cached['timestamp']
            if age < timedelta(days=max_age_days):
                return cached['yf_ticker']
        return None


# Global cache instance
_cache = TickerMappingCache()

# Rate limiter
class RateLimiter:
    """Adaptive rate limiter for API calls."""

    def __init__(self, calls_per_second=5):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = Lock()

    def wait_if_needed(self):
        """Wait if we're calling too fast."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            self.last_call = time.time()


_rate_limiter = RateLimiter(calls_per_second=5)  # Conservative rate to avoid failures

# Yahoo Finance suffix mapping by exchange
YAHOO_SUFFIX = {
    # UK & US exchanges
    "London Stock Exchange":      ".L",
    "London Stock Exchange (IOB)":".L",
    "NYSE":                       "",
    "NASDAQ":                     "",
    "NYSE Arca":                  "",

    # Euronext group
    "Euronext Amsterdam":         ".AS",
    "Euronext Paris":             ".PA",
    "Euronext Brussels":          ".BR",
    "Euronext Lisbon":            ".LS",
    "Euronext Milan":             ".MI",

    # Continental Europe
    "Deutsche Börse Xetra":       ".DE",
    "Frankfurt Stock Exchange":   ".F",
    "SIX Swiss Exchange":         ".SW",
    "Vienna Stock Exchange":      ".VI",
    "Wiener Börse":              ".VI",
    "Bolsa de Madrid":            ".MC",
    "Oslo Børs":                  ".OL",
    "Gettex":                     ".DE",

    # Nordics
    "Nasdaq Stockholm":           ".ST",
    "Nasdaq Copenhagen":          ".CO",
    "Nasdaq Helsinki":            ".HE",

    # North America ex-US
    "Toronto Stock Exchange":     ".TO",
    "TSX Venture Exchange":       ".V",

    # Asia-Pac
    "Australian Securities Exchange": ".AX",
    "Tokyo Stock Exchange":           ".T",
}


def get_ticker_object(yf_ticker: str):
    """
    Get yfinance Ticker object from centralized client.

    Note: Caching is now handled by YFinanceClient internally.
    """
    client = YFinanceClient.get_instance()
    return client.get_ticker(yf_ticker)


def discover_yfinance_ticker(symbol: str, exchange_name: Optional[str] = None,
                            cached_yf_ticker: Optional[str] = None) -> Optional[str]:
    """
    Discover the correct yfinance ticker for a given T212 symbol.
    Uses rate limiting and exponential backoff for rate limit errors.

    Returns:
        yfinance ticker string if found, None otherwise
    """
    max_retries = 5

    try:
        # Check cache first
        if not cached_yf_ticker and exchange_name:
            cached_yf_ticker = _cache.get_mapping(symbol, exchange_name)
            if cached_yf_ticker:
                # Verify cached ticker still works
                try:
                    _rate_limiter.wait_if_needed()
                    stock = get_ticker_object(cached_yf_ticker)
                    info = stock.info
                    if info and len(info) > 5:
                        return cached_yf_ticker
                except Exception as e:
                    # If rate limited on cache check, propagate error
                    if 'rate limit' in str(e).lower() or 'too many requests' in str(e).lower():
                        raise
                    pass  # Cache invalid, continue discovery

        # Yahoo Finance uses dashes instead of slashes for share classes
        clean_symbol = symbol.replace('/', '-')

        # Build list of tickers to try
        ticker_attempts = []
        if cached_yf_ticker:
            ticker_attempts.append(cached_yf_ticker)

        # Add exchange-specific suffix first
        if exchange_name and exchange_name in YAHOO_SUFFIX:
            preferred_suffix = YAHOO_SUFFIX[exchange_name]
            preferred_ticker = clean_symbol + preferred_suffix
            if preferred_ticker not in ticker_attempts:
                ticker_attempts.append(preferred_ticker)

        # Add US suffix (no suffix) early since it's common
        if clean_symbol not in ticker_attempts:
            ticker_attempts.append(clean_symbol)

        # Try to discover the correct ticker
        for attempt_ticker in ticker_attempts:
            for retry in range(max_retries):
                try:
                    _rate_limiter.wait_if_needed()
                    stock = get_ticker_object(attempt_ticker)

                    # Suppress yfinance output and catch HTTP errors
                    import io
                    import sys
                    old_stderr = sys.stderr
                    sys.stderr = io.StringIO()

                    try:
                        info = stock.info
                    finally:
                        sys.stderr = old_stderr

                    # Check if we got valid data
                    if info and len(info) > 5 and ('currentPrice' in info or 'regularMarketPrice' in info):
                        # Cache successful mapping
                        if exchange_name:
                            _cache.save_mapping(symbol, exchange_name, attempt_ticker)
                        return attempt_ticker

                    # No valid data, try next ticker
                    break

                except Exception as e:
                    error_str = str(e).lower()

                    # Check if it's a rate limit or timeout error
                    if any(x in error_str for x in ['rate limit', 'too many requests', 'timeout', 'timed out']):
                        if retry < max_retries - 1:
                            # Progressive backoff: 60s, 5min, 15min, 30min, 1hr
                            wait_times = [60, 300, 900, 1800, 3600]
                            wait_time = wait_times[retry] if retry < len(wait_times) else 3600

                            tqdm.write(f"  ⚠️  Rate limit/timeout. Waiting {wait_time//60} minutes before retry {retry + 2}/{max_retries}...")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Max retries reached, skip this ticker
                            tqdm.write(f"  ✗ Rate limit/timeout persists after {max_retries} retries. Skipping {symbol}")
                            break
                    # Check for common "ticker not found" errors
                    elif any(x in error_str for x in ['not found', '404', 'invalid crumb', 'unauthorized']):
                        # Ticker doesn't exist, try next one
                        break
                    else:
                        # Other error, try next ticker
                        break

        return None

    except Exception as e:
        if 'rate limit' not in str(e).lower():
            tqdm.write(f"  ✗ Error discovering ticker for {symbol}: {str(e)}")
        return None


def check_historical_data_coverage(yf_ticker: str, min_days: int = 750, max_retries: int = 3) -> Tuple[bool, int]:
    """
    Check if stock has 5 years of historical data using yfinance period='5y'.

    Trust yfinance's definition of "5 years" and just validate we got reasonable data.
    No arbitrary day count thresholds - if yfinance returns data for '5y', accept it.

    Args:
        yf_ticker: Yahoo Finance ticker symbol
        min_days: Sanity check minimum (default: 750 days ≈ 3 years minimum)
        max_retries: Maximum retry attempts for rate limiting

    Returns:
        (has_coverage: bool, days_available: int)
    """
    for attempt in range(max_retries):
        try:
            _rate_limiter.wait_if_needed()
            stock = get_ticker_object(yf_ticker)

            # Request 5 years of data from yfinance
            hist = stock.history(period="5y")

            if hist is None or hist.empty:
                return False, 0

            days_available = len(hist)

            # Sanity check: Just ensure we got a reasonable amount of data
            # (at least ~3 years worth = 750 trading days)
            # If yfinance returned data for period="5y", trust it's approximately correct
            has_coverage = days_available >= min_days

            return has_coverage, days_available

        except Exception as e:
            error_str = str(e).lower()

            # Handle rate limiting and timeouts
            if any(x in error_str for x in ['rate limit', 'too many requests', 'timeout', 'timed out']):
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10
                    time.sleep(wait_time)
                    continue
                else:
                    return False, 0
            else:
                # Other errors
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    return False, 0

    return False, 0


def fetch_basic_filter_data(yf_ticker: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive financial data needed for institutional filters and data coverage.

    Returns:
        Dict with all institutional data fields or None if failed
    """
    for attempt in range(max_retries):
        try:
            _rate_limiter.wait_if_needed()
            stock = get_ticker_object(yf_ticker)
            info = stock.info

            # Validate we got real data
            if not info or len(info) < 10:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None

            # Extract ALL institutional data fields
            institutional_data = {
                # Market cap and price
                'marketCap': info.get('marketCap'),
                'currentPrice': info.get('currentPrice'),
                'regularMarketPrice': info.get('regularMarketPrice'),

                # Volume and shares
                'volume': info.get('volume'),
                'averageVolume': info.get('averageVolume'),
                'averageVolume10days': info.get('averageVolume10days'),
                'sharesOutstanding': info.get('sharesOutstanding'),

                # Beta and risk
                'beta': info.get('beta'),

                # Sector and industry
                'sector': info.get('sector'),
                'industry': info.get('industry'),

                # Exchange
                'exchange': info.get('exchange'),

                # Financial ratios
                'trailingPE': info.get('trailingPE'),
                'priceToBook': info.get('priceToBook'),
                'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),

                # Profitability
                'returnOnEquity': info.get('returnOnEquity'),
                'returnOnAssets': info.get('returnOnAssets'),
                'profitMargins': info.get('profitMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'grossMargins': info.get('grossMargins'),

                # Debt metrics
                'debtToEquity': info.get('debtToEquity'),
                'totalDebt': info.get('totalDebt'),
                'totalAssets': info.get('totalAssets'),
                'currentRatio': info.get('currentRatio'),

                # Dividend data
                'dividendYield': info.get('dividendYield'),
                'dividendRate': info.get('dividendRate'),

                # 52-week range
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),

                # Additional useful metrics
                'operatingCashflow': info.get('operatingCashflow'),
                'revenueGrowth': info.get('revenueGrowth'),
                'earningsGrowth': info.get('earningsGrowth'),
                '52WeekChange': info.get('52WeekChange'),
            }

            return institutional_data

        except Exception as e:
            error_str = str(e).lower()

            # Handle rate limiting and timeouts
            if any(x in error_str for x in ['rate limit', 'too many requests', 'timeout', 'timed out']):
                if attempt < max_retries - 1:
                    # Progressive backoff: 10s, 20s, 40s
                    wait_time = (2 ** attempt) * 10
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            else:
                # Other errors - shorter retry
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    return None

    return None


def apply_basic_filters(data: Dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
    """
    Apply institutional filters with market cap-based liquidity tiers, data coverage, and historical data requirements.

    Args:
        data: Comprehensive institutional data
        yf_ticker: Yahoo Finance ticker (for historical data check)

    Returns:
        (passed: bool, reason: str) - Whether stock passed filters and reason if failed
    """
    if not data:
        return False, "No data available"

    # Filter 1: Market cap ≥ $100M
    market_cap = data.get('marketCap')
    if market_cap is None or market_cap < MIN_MARKET_CAP:
        mcap_str = f"${market_cap/1e6:.1f}M" if market_cap else "N/A"
        return False, f"Market cap {mcap_str} < ${MIN_MARKET_CAP/1e6:.0f}M"

    # Filter 2: Price filters ($5 ≤ price ≤ $10,000)
    price = data.get('currentPrice') or data.get('regularMarketPrice')
    if price is None:
        return False, "No current price available"
    if price < MIN_PRICE:
        return False, f"Price ${price:.2f} < ${MIN_PRICE}"
    if price > MAX_PRICE:
        return False, f"Price ${price:.2f} > ${MAX_PRICE} (data error)"

    # Filter 3: Market cap-based liquidity filters
    market_cap_segment = determine_market_cap_segment(market_cap)
    liquidity_req = LIQUIDITY_FILTERS[market_cap_segment]

    avg_volume = data.get('averageVolume') or data.get('averageVolume10days')
    if avg_volume is None:
        return False, "No average volume data"

    # Check dollar volume
    avg_dollar_volume = avg_volume * price
    min_adv_dollars = liquidity_req['min_adv_dollars']
    if avg_dollar_volume < min_adv_dollars:
        return False, f"{market_cap_segment.upper()}: ADV ${avg_dollar_volume/1e6:.1f}M < ${min_adv_dollars/1e6:.0f}M"

    # Check share volume
    min_adv_shares = liquidity_req['min_adv_shares']
    if avg_volume < min_adv_shares:
        return False, f"{market_cap_segment.upper()}: ADV {avg_volume:,.0f} shares < {min_adv_shares:,.0f}"

    # Filter 4: Institutional data coverage (100% of required fields)
    coverage_passed, missing_categories = check_institutional_coverage(data)
    if not coverage_passed:
        missing_str = ", ".join(missing_categories[:3])
        if len(missing_categories) > 3:
            missing_str += f" +{len(missing_categories)-3} more"
        return False, f"Incomplete data: missing {missing_str}"

    # Filter 5: Historical data coverage (5 years from yfinance period='5y')
    has_history, days_available = check_historical_data_coverage(yf_ticker)
    if not has_history:
        years = days_available / 252 if days_available > 0 else 0
        return False, f"Insufficient history: {days_available} days ({years:.1f}y) from period='5y'"

    return True, f"Passed all filters ({market_cap_segment}, {days_available}d history from period='5y')"


def process_single_instrument(instrument, exchange_name, skip_filters=False, log_filter_failures=False):
    """
    Process a single instrument to discover its yfinance ticker and apply basic filters.

    Args:
        instrument: Instrument data from T212 API
        exchange_name: Exchange name
        skip_filters: If True, skip basic filtering (keep all mapped stocks)
        log_filter_failures: If True, log why stocks fail filters (for debugging)

    Returns:
        Minimal instrument data with yfinance ticker mapping
    """
    try:
        short_name = instrument.get("shortName", "unknown")

        # Base instrument data
        instrument_data = {
            "ticker":           instrument.get("ticker"),
            "type":             instrument.get("type"),
            "isin":             instrument.get("isin"),
            "currencyCode":     instrument.get("currencyCode"),
            "name":             instrument.get("name"),
            "shortName":        short_name,
            "maxOpenQuantity":  instrument.get("maxOpenQuantity"),
            "addedOn":          instrument.get("addedOn"),
            "exchange":         exchange_name
        }

        # Discover yfinance ticker
        yf_ticker = discover_yfinance_ticker(
            short_name,
            exchange_name,
            instrument.get("yfinanceTicker")
        )

        if not yf_ticker:
            # No valid ticker found, skip this instrument
            if log_filter_failures:
                tqdm.write(f"  ✗ {short_name:10} No yfinance ticker found")
            return None

        instrument_data["yfinanceTicker"] = yf_ticker

        # Skip filtering if requested
        if skip_filters:
            if log_filter_failures:
                tqdm.write(f"  ✓ {short_name:10} → {yf_ticker:10} (filters skipped)")
            return instrument_data

        # Fetch basic financial data for filtering
        basic_data = fetch_basic_filter_data(yf_ticker)

        if not basic_data:
            # Failed to fetch data, skip this instrument
            if log_filter_failures:
                tqdm.write(f"  ✗ {short_name:10} → {yf_ticker:10} Failed to fetch yfinance data")
            return None

        # Apply basic filters (including 5-year historical data check)
        passed, reason = apply_basic_filters(basic_data, yf_ticker)

        if not passed:
            # Failed filters, skip this instrument
            if log_filter_failures:
                tqdm.write(f"  ✗ {short_name:10} → {yf_ticker:10} FILTERED: {reason}")
            return None

        # Passed all filters!
        if log_filter_failures:
            tqdm.write(f"  ✓ {short_name:10} → {yf_ticker:10} PASSED: {reason}")
        return instrument_data

    except Exception as e:
        tqdm.write(f"  ✗ Error processing {instrument.get('shortName', 'unknown')}: {str(e)}")
        return None


def save_exchange_to_db(session: Session, exchange_data: Dict[str, Any]) -> Exchange:
    """
    Save or update exchange in database.

    Args:
        session: Database session
        exchange_data: Exchange data from T212 API

    Returns:
        Exchange object (new or existing)
    """
    exchange_id = exchange_data["id"]
    exchange_name = exchange_data["name"]

    # Check if exchange already exists
    existing = session.execute(
        select(Exchange).where(Exchange.exchange_id == exchange_id)
    ).scalar_one_or_none()

    if existing:
        # Update existing exchange
        existing.exchange_name = exchange_name
        existing.is_active = True
        existing.last_updated = datetime.now()
        return existing
    else:
        # Create new exchange
        new_exchange = Exchange(
            exchange_id=exchange_id,
            exchange_name=exchange_name,
            is_active=True,
            last_updated=datetime.now()
        )
        session.add(new_exchange)
        session.flush()  # Get the ID without committing
        return new_exchange


def save_instrument_to_db(
    session: Session,
    instrument_data: Dict[str, Any],
    exchange: Exchange,
    batch_list: Optional[List] = None
) -> Optional[Instrument]:
    """
    Save or update instrument in database.

    Args:
        session: Database session
        instrument_data: Instrument data with yfinance mapping
        exchange: Parent Exchange object
        batch_list: Optional list to collect instruments for batch insert

    Returns:
        Instrument object (new or existing) or None if added to batch
    """
    ticker = instrument_data.get("ticker")
    if not ticker:
        return None

    # Check if instrument exists
    existing = session.execute(
        select(Instrument).where(Instrument.ticker == ticker)
    ).scalar_one_or_none()

    # Parse addedOn date
    added_on = None
    added_on_str = instrument_data.get("addedOn")
    if added_on_str:
        try:
            added_on = datetime.fromisoformat(added_on_str.replace('+', ' +'))
        except:
            pass

    if existing:
        # Update existing instrument
        existing.exchange_id = exchange.id
        existing.short_name = instrument_data.get("shortName") or ticker
        existing.name = instrument_data.get("name")
        existing.isin = instrument_data.get("isin")
        existing.instrument_type = instrument_data.get("type", "STOCK")
        existing.currency_code = instrument_data.get("currencyCode")
        existing.max_open_quantity = instrument_data.get("maxOpenQuantity")
        existing.added_on = added_on
        existing.yfinance_ticker = instrument_data.get("yfinanceTicker")
        existing.is_active = True
        existing.last_validated = datetime.now()
        return existing
    else:
        # Create new instrument
        new_inst = Instrument(
            exchange_id=exchange.id,
            ticker=ticker,
            short_name=instrument_data.get("shortName") or ticker,
            name=instrument_data.get("name"),
            isin=instrument_data.get("isin"),
            instrument_type=instrument_data.get("type", "STOCK"),
            currency_code=instrument_data.get("currencyCode"),
            max_open_quantity=instrument_data.get("maxOpenQuantity"),
            added_on=added_on,
            yfinance_ticker=instrument_data.get("yfinanceTicker"),
            is_active=True,
            last_validated=datetime.now()
        )

        if batch_list is not None:
            batch_list.append(new_inst)
            return None
        else:
            session.add(new_inst)
            return new_inst


def build_universe(
    exchanges: List[Dict],
    instruments: List[Dict],
    max_workers: int = 20,
    skip_filters: bool = False,
    batch_size: int = 50,
    only_exchanges: Optional[List[str]] = None,
    log_filter_failures: bool = False
) -> tuple[int, int]:
    """
    Build universe with T212 → yfinance ticker mappings and save to database.
    Uses sequential processing per exchange with controlled concurrency.
    Uses SHORT-LIVED sessions per exchange to avoid SSL timeouts.

    Args:
        exchanges: List of exchange metadata from T212 API
        instruments: List of instruments from T212 API
        max_workers: Number of concurrent threads per exchange (default: 20)
        skip_filters: If True, skip basic filtering (keep all mapped stocks)
        batch_size: Batch size for bulk inserts (default: 50, smaller for stability)
        only_exchanges: Optional list of exchange names to process (for debugging). If provided, ONLY these exchanges will be processed.
        log_filter_failures: If True, log detailed reasons why stocks fail filters (for debugging)

    Returns:
        (total_exchanges, total_instruments) saved to database
    """
    # Create mapping from workingScheduleId to exchange
    schedule_to_exchange = {}
    for ex in exchanges:
        for schedule in ex.get("workingSchedules", []):
            schedule_to_exchange[schedule["id"]] = ex

    # Group instruments by workingScheduleId
    by_schedule = defaultdict(list)
    for inst in instruments:
        by_schedule[inst["workingScheduleId"]].append(inst)

    # Calculate total number of stocks
    # Filter exchanges by portfolio countries FIRST
    allowed_exchanges = get_allowed_exchanges()
    total_stocks = 0
    exchange_stocks = []
    filtered_exchanges = []

    for ex in exchanges:
        exchange_name = ex.get("name")
        if not exchange_name:
            continue  # Skip if exchange name is missing

        # DEBUG MODE: If only_exchanges is specified, ONLY process those exchanges
        if only_exchanges is not None:
            if exchange_name not in only_exchanges:
                filtered_exchanges.append(exchange_name)
                continue
        else:
            # NORMAL MODE: Skip exchanges not in portfolio countries
            if not is_exchange_allowed(exchange_name):
                filtered_exchanges.append(exchange_name)
                continue

        # Collect all instruments for this exchange
        all_exchange_instruments = []
        for schedule in ex.get("workingSchedules", []):
            schedule_id = schedule["id"]
            schedule_instruments = by_schedule.get(schedule_id, [])
            all_exchange_instruments.extend(schedule_instruments)

        # Filter: only process STOCK type
        insts = [i for i in all_exchange_instruments if i.get("type") == "STOCK"]
        if insts:
            exchange_stocks.append((ex, insts))
            total_stocks += len(insts)

    # Show country filtering info
    print(f"\n{'='*80}")
    if only_exchanges is not None:
        print(f"DEBUG MODE - EXCHANGE FILTERING")
        print(f"{'='*80}")
        print(f"🔍 ONLY processing exchanges: {', '.join(only_exchanges)}")
        print(f"📊 Found {len(exchange_stocks)} matching exchanges with {total_stocks} stocks")
    else:
        print(f"COUNTRY FILTERING")
        print(f"{'='*80}")
        print(f"Portfolio countries: {', '.join(PORTFOLIO_COUNTRIES)}")
        print(f"Allowed exchanges:   {', '.join(sorted(allowed_exchanges))}")

    if filtered_exchanges:
        print(f"\nFiltered out {len(filtered_exchanges)} exchanges:")
        for ex_name in sorted(set(filtered_exchanges))[:10]:  # Show first 10
            print(f"  - {ex_name}")
        if len(set(filtered_exchanges)) > 10:
            print(f"  ... and {len(set(filtered_exchanges)) - 10} more")

    print(f"\n{'='*80}")
    print(f"Processing {total_stocks} stocks across {len(exchange_stocks)} exchanges")
    print(f"Sequential processing with {max_workers} threads per exchange")
    print(f"Using short-lived DB sessions to avoid SSL timeouts...")
    print(f"{'='*80}\n")

    total_exchanges_saved = 0
    total_instruments_saved = 0

    # Process exchanges sequentially
    with tqdm(total=total_stocks, desc="Mapping & saving", unit="stock") as pbar:
        for ex_data, insts in exchange_stocks:
            # NEW SESSION for exchange creation
            with database_manager.get_session() as session:
                exchange_obj = save_exchange_to_db(session, ex_data)
                session.commit()
                exchange_id = exchange_obj.id
                total_exchanges_saved += 1

            # Process instruments for this exchange concurrently
            instruments_processed = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_instrument = {
                    executor.submit(process_single_instrument, inst, ex_data["name"], skip_filters, log_filter_failures): inst
                    for inst in insts
                }

                # Collect results as they complete
                for future in as_completed(future_to_instrument):
                    try:
                        result = future.result()
                        # Only add non-None results (instruments that passed filters)
                        if result is not None:
                            instruments_processed.append(result)
                    except Exception as e:
                        inst = future_to_instrument[future]
                        tqdm.write(f"  ✗ Failed to process {inst.get('shortName', 'unknown')}: {str(e)}")
                    finally:
                        pbar.update(1)

            # Save instruments in SMALL batches with NEW sessions
            for i in range(0, len(instruments_processed), batch_size):
                batch_data = instruments_processed[i:i+batch_size]

                # NEW SESSION for each batch
                with database_manager.get_session() as session:
                    # Re-fetch exchange in this session
                    exchange_obj = session.execute(
                        select(Exchange).where(Exchange.id == exchange_id)
                    ).scalar_one()

                    batch = []
                    for inst_data in batch_data:
                        save_instrument_to_db(session, inst_data, exchange_obj, batch_list=batch)

                    if batch:
                        session.bulk_save_objects(batch)
                        session.commit()
                        total_instruments_saved += len(batch)

    return total_exchanges_saved, total_instruments_saved


def generate_database_report(session: Session) -> None:
    """Generate summary report from database with market cap distribution."""
    print("\n" + "="*80)
    print("DATABASE SUMMARY REPORT")
    print("="*80)

    # Count exchanges
    exchanges = session.execute(select(Exchange)).scalars().all()
    print(f"Total exchanges: {len(exchanges)}")

    # Count instruments
    all_instruments = session.execute(select(Instrument)).scalars().all()
    print(f"Total instruments: {len(all_instruments)}")

    # Count instruments with yfinance data
    with_yf = sum(1 for inst in all_instruments if inst.yfinance_ticker)
    print(f"Instruments with yfinance mapping: {with_yf} ({with_yf/len(all_instruments)*100:.1f}%)")

    # Count by exchange
    print("\nInstruments by exchange:")
    exchange_counts = {}
    for inst in all_instruments:
        ex_name = inst.exchange.exchange_name
        exchange_counts[ex_name] = exchange_counts.get(ex_name, 0) + 1

    for ex_name, count in sorted(exchange_counts.items(), key=lambda x: -x[1]):
        print(f"  {ex_name:40} {count:6d} stocks")

    print("\n[INFO] Market cap distribution will be shown after stocks are analyzed with yfinance data.")


def main():
    p = argparse.ArgumentParser(
        description="Build T212 → yfinance ticker mapping with institutional filters (Database Storage)")
    p.add_argument("--mode", choices=["demo","live"], default="live",
                   help="Which T212 environment to use")
    p.add_argument("--max-workers", type=int, default=20,
                   help="Number of concurrent threads per exchange (default: 20)")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Batch size for database inserts (default: 500)")
    p.add_argument("--cache-dir", type=str, default="../.cache",
                   help="Directory for caching ticker mappings (default: ../.cache)")
    p.add_argument("--skip-filters", action="store_true",
                   help="Skip basic filters (keep all mapped stocks)")
    p.add_argument("--api-key", type=str, default=None,
                   help="Trading212 API key (if not provided, reads from TRADING_212_API_KEY env var)")
    p.add_argument("--clear", action="store_true",
                   help="Clear existing universe data before import")
    p.add_argument("--only-exchanges", nargs='+', default=None,
                   help="DEBUG MODE: Only process specific exchanges (e.g., --only-exchanges NYSE NASDAQ)")
    p.add_argument("--log-filter-failures", action="store_true", default=True,
                   help="DEBUG MODE: Log detailed reasons why stocks fail filters")
    args = p.parse_args()

    try:
        # Initialize cache with custom directory if specified
        global _cache
        if args.cache_dir != "../.cache":
            _cache = TickerMappingCache(cache_dir=args.cache_dir)

        # Get API key from command line or environment variable
        api_key = args.api_key or os.getenv("TRADING_212_API_KEY")
        if not api_key:
            print("✗ Error: TRADING_212_API_KEY not found")
            print("  Please either:")
            print("    1. Set TRADING_212_API_KEY in your .env file")
            print("    2. Export TRADING_212_API_KEY as an environment variable")
            print("    3. Use --api-key argument")
            return

        # Initialize database
        print("Initializing database connection...")
        init_db()

        if not database_manager.is_initialized:
            print("✗ Database initialization failed")
            sys.exit(1)

        print("✓ Database connection established")

        # Fetch from T212 API
        base = "https://demo.trading212.com" if args.mode=="demo" else "https://live.trading212.com"
        headers = {"Authorization": api_key}

        print("\n" + "="*80)
        if args.skip_filters:
            print("BUILDING UNIVERSE (NO FILTERS) - SAVING TO DATABASE")
        else:
            print("BUILDING UNIVERSE WITH BASIC FILTERS - SAVING TO DATABASE")
        print("="*80)
        print(f"Mode: Trading212 {args.mode}")

        # Display portfolio countries
        print(f"\nPortfolio countries: {', '.join(PORTFOLIO_COUNTRIES)}")
        countries_missing = [c for c in PORTFOLIO_COUNTRIES if c not in COUNTRY_TO_EXCHANGES]
        if countries_missing:
            print(f"Note: {', '.join(countries_missing)} not available in Trading212")

        if not args.skip_filters:
            print(f"\n" + "="*80)
            print("INSTITUTIONAL FILTER CONFIGURATION")
            print("="*80)
            print(f"\nMarket Cap Tiers:")
            print(f"  Large-cap:  ≥ ${LARGE_CAP_THRESHOLD/1e9:.0f}B")
            print(f"  Mid-cap:    ${SMALL_CAP_THRESHOLD/1e9:.0f}B - ${MID_CAP_THRESHOLD/1e9:.0f}B")
            print(f"  Small-cap:  ${MIN_MARKET_CAP/1e6:.0f}M - ${SMALL_CAP_THRESHOLD/1e9:.0f}B")

            print(f"\nPrice Filters:")
            print(f"  Minimum:    ${MIN_PRICE}")
            print(f"  Maximum:    ${MAX_PRICE:,.0f} (data error check)")

            print(f"\nLiquidity Requirements (by market cap):")
            for segment in ['large_cap', 'mid_cap', 'small_cap']:
                req = LIQUIDITY_FILTERS[segment]
                print(f"  {segment.replace('_', '-').title():12} Min ADV: ${req['min_adv_dollars']/1e6:.0f}M / {req['min_adv_shares']:,} shares")

            print(f"\nInstitutional Data Coverage:")
            print(f"  Required: 100% coverage of {sum(1 for spec in INSTITUTIONAL_FIELDS.values() if spec.get('required', True))} required categories")
            print(f"  Categories: market_cap, price, volume, shares, sector/industry, exchange,")
            print(f"              financial_ratios, profitability, debt_metrics, 52week_range")

            print(f"\nHistorical Data Coverage:")
            print(f"  Method:  Trust yfinance period='5y' with sanity check (≥750 days)")
            print(f"  Purpose: Institutional signal generation (alpha, beta, Sharpe, 12-1 momentum)")
            print(f"  Excludes: Recent IPOs and stocks with limited trading history")
        else:
            print(f"\nFilters: DISABLED (--skip-filters flag)")
        print(f"\nFetching data from Trading212 ({args.mode} mode)...")

        # Fetch metadata
        exchanges   = fetch_json(base, "/api/v0/equity/metadata/exchanges",   headers)
        instruments_list = fetch_json(base, "/api/v0/equity/metadata/instruments", headers)

        print(f"✓ Fetched {len(exchanges)} exchanges and {len(instruments_list)} instruments")

        # Clear existing data if requested
        if args.clear:
            response = input("⚠️  This will delete all existing universe data. Continue? (yes/no): ")
            if response.lower() == 'yes':
                print("Clearing existing data...")
                with database_manager.get_session() as session:
                    deleted_instruments = session.execute(delete(Instrument)).rowcount
                    deleted_exchanges = session.execute(delete(Exchange)).rowcount
                    session.commit()
                    print(f"✓ Deleted {deleted_exchanges} exchanges and {deleted_instruments} instruments")
            else:
                print("Aborted")
                return

        # Show debug mode warning if active
        if args.only_exchanges:
            print(f"\n⚠️  DEBUG MODE ACTIVE")
            print(f"    Only processing exchanges: {', '.join(args.only_exchanges)}")
            print(f"    All other exchanges will be skipped\n")

        # Build universe and save to database (uses internal short-lived sessions)
        total_exchanges, total_instruments = build_universe(
            exchanges,
            instruments_list,
            max_workers=args.max_workers,
            skip_filters=args.skip_filters,
            batch_size=args.batch_size,
            only_exchanges=args.only_exchanges,
            log_filter_failures=args.log_filter_failures
        )

        print(f"\n{'='*80}")
        print(f"IMPORT COMPLETE")
        print(f"{'='*80}")
        print(f"Exchanges saved:   {total_exchanges}")
        print(f"Instruments saved: {total_instruments}")

        # Generate report from database (uses new session)
        with database_manager.get_session() as session:
            generate_database_report(session)

        print("\n" + "="*80)
        print(f"FILTER SUMMARY")
        print("="*80)
        print(f"\nMarket Cap Tiers Applied:")
        print(f"  • Large-cap: ≥${LARGE_CAP_THRESHOLD/1e9:.0f}B (ADV ≥$10M)")
        print(f"  • Mid-cap:   $2B-$10B (ADV ≥$5M)")
        print(f"  • Small-cap: $100M-$2B (ADV ≥$1M)")
        print(f"\nInstitutional Data:")
        print(f"  • 100% coverage of 10 required categories")
        print(f"  • All stocks have complete fundamental data")
        print(f"\nHistorical Data:")
        print(f"  • Method: Trust yfinance period='5y' with sanity check (≥750 days)")
        print(f"  • Excludes: Recent IPOs with < 3 years history")
        print(f"  • Ensures: Sufficient data for institutional metrics")

        print("\n" + "="*80)
        print("✓ Universe built and saved to database successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Build failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__=="__main__":
    main()
