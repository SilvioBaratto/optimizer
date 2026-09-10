# yfinance Implementation Patterns

## 1. Basic Ticker Usage

```python
import yfinance as yf

# Create a Ticker object
ticker = yf.Ticker("AAPL")

# Full info dict (slow — makes multiple API calls, cached after first call)
info = ticker.info
print(info["marketCap"])
print(info["sector"])
print(info["forwardPE"])

# Fast info (single API call, fewer fields)
fi = ticker.fast_info
print(fi.market_cap)
print(fi.last_price)
print(fi.previous_close)
print(fi.currency)

# ISIN identifier
print(ticker.isin)  # e.g., "US0378331005"
```

---

## 2. Price History

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# Default: 1 month of daily data
df = ticker.history()

# Specific period and interval
df = ticker.history(period="1y", interval="1d")

# Specific date range
df = ticker.history(start="2023-01-01", end="2024-01-01")

# Intraday data (max 60 days for most intervals)
df = ticker.history(period="5d", interval="15m")
df = ticker.history(period="7d", interval="1m")  # 1m max 7 days

# Include pre/post market
df = ticker.history(period="1d", interval="1m", prepost=True)

# With repair enabled
df = ticker.history(period="2y", repair=True)

# Weekly and monthly
df = ticker.history(period="5y", interval="1wk")
df = ticker.history(period="max", interval="1mo")

# Result columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
print(df.columns.tolist())
```

---

## 3. Bulk Download

```python
import yfinance as yf
import pandas as pd

# Download multiple tickers at once
data = yf.download(
    ["AAPL", "MSFT", "GOOG"],
    period="2y",
    interval="1d",
    auto_adjust=True,
    threads=True,
)

# Default group_by='column': multi-level columns (Price, Ticker)
aapl_close = data["Close"]["AAPL"]
msft_close = data["Close"]["MSFT"]

# group_by='ticker': multi-level columns (Ticker, Price)
data = yf.download(
    ["AAPL", "MSFT", "GOOG"],
    period="2y",
    group_by="ticker",
)
aapl_data = data["AAPL"]
aapl_close = data["AAPL"]["Close"]

# Disable multi-level index entirely
data = yf.download(
    ["AAPL", "MSFT"],
    period="1y",
    multi_level_index=False,
)

# Download with specific date range
data = yf.download(
    "AAPL MSFT",  # space-separated string also works
    start="2022-01-01",
    end="2024-01-01",
)

# Single ticker returns flat DataFrame (no multi-level)
aapl = yf.download("AAPL", period="1y")
print(aapl["Close"].head())
```

---

## 4. Financial Statements

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# Income statement
annual_income = ticker.income_stmt              # Annual (default)
quarterly_income = ticker.quarterly_income_stmt  # Quarterly
ttm_income = ticker.get_income_stmt(freq="trailing")  # TTM

# Balance sheet
annual_bs = ticker.balance_sheet
quarterly_bs = ticker.quarterly_balance_sheet
ttm_bs = ticker.get_balance_sheet(freq="trailing")

# Cash flow
annual_cf = ticker.cashflow
quarterly_cf = ticker.quarterly_cashflow
ttm_cf = ticker.get_cashflow(freq="trailing")

# Earnings — DEPRECATED in 1.6.0: ticker.earnings / ticker.quarterly_earnings
# now emit a DeprecationWarning and return None (data no longer served by the API).
# Use the "Net Income" row of the income statement instead:
net_income = ticker.get_income_stmt().loc["Net Income"]              # annual
net_income_q = ticker.quarterly_income_stmt.loc["Net Income"]        # quarterly

# Get as dict instead of DataFrame
income_dict = ticker.get_income_stmt(as_dict=True)

# Pretty column names (human-readable)
income_pretty = ticker.get_income_stmt(pretty=True)

# SEC filings
filings = ticker.sec_filings
for filing in filings[:5]:
    print(f"{filing['type']}: {filing['title']} ({filing['date']})")
```

---

## 5. Analyst & Estimates

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# Analyst recommendations (recent)
recs = ticker.recommendations
# Columns: period, strongBuy, buy, hold, sell, strongSell

# Recommendations summary
summary = ticker.recommendations_summary

# Upgrades and downgrades
changes = ticker.upgrades_downgrades
# Columns: Firm, ToGrade, FromGrade, Action

# Analyst price targets
targets = ticker.analyst_price_targets
print(f"Current: {targets['current']}")
print(f"Low: {targets['low']}")
print(f"High: {targets['high']}")
print(f"Mean: {targets['mean']}")
print(f"Median: {targets['median']}")

# Earnings estimates (current quarter, next quarter)
ee = ticker.earnings_estimate
# Columns: numberOfAnalysts, avg, low, high, yearAgoEps, growth

# Revenue estimates
re = ticker.revenue_estimate

# EPS surprise history
eh = ticker.earnings_history
# Columns: epsEstimate, epsActual, epsDifference, surprisePercent

# EPS trend over time
trend = ticker.eps_trend
# Rows: current, 7daysAgo, 30daysAgo, 60daysAgo, 90daysAgo

# EPS revisions (how many analysts revised up/down)
revisions = ticker.eps_revisions
# Rows: upLast7days, upLast30days, downLast7days, downLast30days

# Growth estimates vs sector/industry
growth = ticker.growth_estimates
# Compares stock growth to sector and industry averages
```

---

## 6. Ownership & Insider Data

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# High-level holder breakdown
major = ticker.major_holders
# Shows: % of shares held by insiders, institutions, etc.

# Top institutional holders
inst = ticker.institutional_holders
# Columns: Holder, Shares, Date Reported, % Out, Value

# Top mutual fund holders
mf = ticker.mutualfund_holders
# Columns: Holder, Shares, Date Reported, % Out, Value

# Recent insider transactions
insider_tx = ticker.insider_transactions
# Columns: Insider, Position, Transaction, Shares, Value, Date

# Insider purchases summary
insider_buys = ticker.insider_purchases

# Full insider roster
roster = ticker.insider_roster_holders
```

---

## 7. Dividends, Splits & Corporate Actions

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# Historical dividends
divs = ticker.dividends
# Series indexed by date with dividend amounts

# Historical stock splits
splits = ticker.splits
# Series indexed by date with split ratios

# Combined actions (dividends + splits)
actions = ticker.actions
# DataFrame with Dividends and Stock Splits columns

# Capital gains (for mutual funds / ETFs)
gains = ticker.capital_gains

# Shares outstanding over time (method — there is no .shares_full attribute)
shares = ticker.get_shares_full()
# Or with date range:
shares = ticker.get_shares_full(start="2020-01-01", end="2024-01-01")
```

---

## 8. Search & Lookup

```python
import yfinance as yf

# Full-text search across Yahoo Finance
search = yf.Search("Apple", include_research=True)   # research/nav need their include_* flag

# Matching stock symbols
for quote in search.quotes:
    print(f"{quote['symbol']}: {quote['shortname']} ({quote['exchange']})")

# Related news
for article in search.news:
    print(article["title"])

# Research reports (only populated because include_research=True)
for report in search.research:
    print(report["title"])

search.all            # filtered aggregate view
search.response       # raw payload

# Lookup — filter by asset class via PROPERTY or get_*(count=...), NOT a type= arg
lookup = yf.Lookup("semiconductor")
lookup.stock                      # full result set for stocks
lookup.etf                        # full result set for ETFs
lookup.mutualfund
df = lookup.get_stock(count=25)   # count-limited DataFrame
df = lookup.get_etf(count=10)
```

---

## 9. Market Status

```python
import yfinance as yf

# Identifiers: US, GB, ASIA, EUROPE, RATES, COMMODITIES, CURRENCIES, CRYPTOCURRENCIES
market = yf.Market("US")

# Market status — US ONLY. Non-US returns None + a logged warning (v1.4.0).
status = market.status
print(status)

# Market summary — regional, works for all eight identifiers
summary = market.summary
print(summary)

europe = yf.Market("EUROPE").summary   # use .summary for non-US regional data
```

---

## 10. Sector & Industry

```python
import yfinance as yf

# Sector data (region scopes the rollups; ISO 3166-1 alpha-2, defaults to US)
tech = yf.Sector("technology", region="US")
print(tech.overview)
print(tech.top_companies)      # Top companies by market cap (region-scoped)
print(tech.industries)         # Industries within sector
print(tech.top_etfs)           # Sector-tracking ETFs (region-scoped)
print(tech.top_mutual_funds)   # Sector-tracking mutual funds
print(tech.research_reports)   # Research reports (renamed from .research)

# Industry data
semis = yf.Industry("semiconductors", region="US")
print(semis.overview)
print(semis.top_companies)
print(semis.top_performing_companies)
print(semis.top_growth_companies)
print(semis.research_reports)
print(semis.sector_key, semis.sector_name)

# Available sector keys
sectors = [
    "technology", "healthcare", "financial-services",
    "consumer-cyclical", "communication-services", "industrials",
    "consumer-defensive", "energy", "basic-materials",
    "real-estate", "utilities",
]
```

---

## 11. Screener

```python
import yfinance as yf
from yfinance import EquityQuery

# Simple query: large-cap tech stocks
query = EquityQuery("and", [
    EquityQuery("gt", ["intradaymarketcap", 10_000_000_000]),  # > $10B
    EquityQuery("eq", ["sector", "Technology"]),
])

result = yf.screen(query, sortField="intradaymarketcap", sortAsc=False, size=25)
for stock in result["quotes"]:
    print(f"{stock['symbol']}: ${stock.get('marketCap', 0):,.0f}")

# Complex query with nested AND/OR
query = EquityQuery("and", [
    EquityQuery("gt", ["intradaymarketcap", 1_000_000_000]),
    EquityQuery("or", [
        EquityQuery("gt", ["dividendyield", 3]),
        EquityQuery("lt", ["peratio.lasttwelvemonths", 15]),
    ]),
    EquityQuery("eq", ["region", "us"]),
])

# NOTE: sortField (camelCase) + sortAsc (bool), NOT sort_field / sort_type.
# size for CUSTOM queries (default 100, max 250); count for PREDEFINED names (default 25).
result = yf.screen(query, sortField="dividendyield", sortAsc=False, size=50)

# Predefined screen — pass a name string and use count
gainers = yf.screen("day_gainers", count=50)

# Between operator
query = EquityQuery("and", [
    EquityQuery("btwn", ["intradayprice", 10, 50]),
    EquityQuery("gt", ["dayvolume", 1_000_000]),
])

# Paginated results
all_results = []
for offset in range(0, 500, 250):
    page = yf.screen(query, size=250, offset=offset)
    all_results.extend(page["quotes"])
    if len(page["quotes"]) < 250:
        break

# Fund screening
from yfinance import FundQuery

# FundQuery has a SMALL field set: exchange, categoryname,
# annualreturnnavy1categoryrank, performanceratingoverall, initialinvestment,
# riskratingoverall, intradaypricechange, eodprice, intradayprice.
# (No netassets / annualreturnnavy5 — those are ETFQuery-only.)
fund_query = FundQuery("and", [
    FundQuery("gt", ["performanceratingoverall", 3]),
    FundQuery("lt", ["riskratingoverall", 4]),
])
fund_result = yf.screen(fund_query, sortField="performanceratingoverall", sortAsc=False)

# ETF screening (v1.3.0+)
from yfinance import ETFQuery

etf_query = ETFQuery("and", [
    ETFQuery("gt", ["fundnetassets", 500_000_000]),
    ETFQuery("eq", ["region", "us"]),
])
etf_result = yf.screen(etf_query, size=50)
for etf in etf_result["quotes"]:
    print(f"{etf['symbol']}: {etf.get('shortName', '')}")
```

### Valuation Measures Table (v1.3.0+)

```python
import yfinance as yf

t = yf.Ticker("AAPL")
vm = t.get_valuation_measures()   # method (freq='quarterly', periods=5); no .valuation_measures attr
# DataFrame with rows for Market Cap, Enterprise Value, Trailing P/E,
# Forward P/E, PEG Ratio (5yr), Price/Sales, Price/Book, EV/Revenue, EV/EBITDA
# and columns for current + historical quarters and year-ends.

print(vm.loc["Trailing P/E"])       # time series of trailing P/E
print(vm.iloc[:, 0])                # all 9 metrics for the most recent period
```

---

## 12. WebSocket Real-Time Data

No `run()` / `on_message`. Pattern: `subscribe(...)` → `listen(handler)` → `close()`.

```python
import yfinance as yf

# Synchronous WebSocket (context manager handles close())
def on_message(msg):
    """Called for each price update (single dict arg)."""
    # decoded keys are snake_case proto names: id, price, time, day_volume, change, change_percent, ...
    print(f"{msg['id']}: {msg['price']} vol={msg.get('day_volume')}")

with yf.WebSocket() as ws:
    ws.subscribe(["AAPL", "MSFT", "GOOG"])
    ws.listen(on_message)          # blocks; handler passed HERE, not via on_message=

# Asynchronous WebSocket — coroutines must be awaited
import asyncio

async def stream_prices():
    async with yf.AsyncWebSocket() as ws:
        await ws.subscribe(["AAPL", "MSFT"])
        await ws.listen(lambda msg: print(f"{msg['id']}: {msg['price']}"))

asyncio.run(stream_prices())

# Convenience: stream straight off a Ticker / Tickers
yf.Ticker("AAPL").live()
yf.Tickers("AAPL MSFT").live()
```

---

## 13. Calendar Events

```python
import yfinance as yf

# Calendar data for a date range
cal = yf.Calendars(start="2026-01-01", end="2026-03-31")

# Convenience properties (default settings, no filtering/pagination)
earnings = cal.earnings_calendar
ipos = cal.ipo_info_calendar
splits = cal.splits_calendar
econ = cal.economic_events_calendar
print(earnings.head())

# Manual query methods — paginate (limit defaults to 12!) + filter + cache bypass
big_earnings = cal.get_earnings_calendar(
    market_cap=100_000_000, filter_most_active=True, limit=100,
)
more_ipos = cal.get_ipo_info_calendar(limit=50, offset=50, force=True)

# Single ticker calendar info (dict of next earnings/dividend events)
cal_info = yf.Ticker("AAPL").calendar
```

---

## 14. Fund Data

Fund data lives under `ticker.funds_data` (a `FundsData` object) — not flat `ticker.fund_*`.

```python
import yfinance as yf

# ETF example
fd = yf.Ticker("SPY").funds_data

overview   = fd.fund_overview        # dict[str, str | None] — family, category
holdings   = fd.top_holdings         # DataFrame — top holdings with weights
sectors    = fd.sector_weightings    # dict[str, float]
allocation = fd.asset_classes        # dict[str, float] — stocks/bonds/cash/other
operations = fd.fund_operations      # DataFrame — turnover, expense ratio, inception, AUM
eq_hold    = fd.equity_holdings      # DataFrame — P/E, P/B of holdings
desc       = fd.description          # str
qt         = fd.quote_type()         # str — METHOD, not a property

# NOTE: no fund_performance — use ticker.history() for returns.

# Bond holdings (for bond funds/ETFs)
bond_fd = yf.Ticker("AGG").funds_data
bond_hold    = bond_fd.bond_holdings    # DataFrame
bond_ratings = bond_fd.bond_ratings     # dict[str, float]
```

---

## 15. Configuration & Debugging

```python
import yfinance as yf

# Set proxy for all requests
yf.config.network.proxy = "http://proxy.company.com:8080"

# Increase retries
yf.config.network.retries = 5

# Hide exceptions (return empty results silently)
yf.config.debug.hide_exceptions = True

# Enable verbose logging
yf.config.debug.logging = True

# Full debug mode (sets both logging and exception display)
yf.enable_debug_mode()

# Custom timezone cache location
yf.set_tz_cache_location("/tmp/yf-cache")

# Custom requests session for all calls
import requests

session = requests.Session()
session.headers.update({"User-Agent": "MyApp/1.0"})

ticker = yf.Ticker("AAPL", session=session)
data = yf.download("AAPL", session=session)
```

---

## 16. Price Repair

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# Enable repair — detects and fixes common data issues
df = ticker.history(period="2y", repair=True)

# Check the Repaired? column
if "Repaired?" in df.columns:
    repaired_rows = df[df["Repaired?"] == True]
    print(f"Repaired {len(repaired_rows)} rows")
    print(repaired_rows)

# Repair categories detected:
# 1. Missing dividend adjustment — prices not adjusted after ex-date
# 2. Missing split adjustment — prices not adjusted after split date
# 3. Missing data — gaps filled from adjacent intervals
# 4. Corrupt data — outlier prices replaced
# 5. 100x currency errors — e.g., pence vs pounds on LSE
# 6. Dividend amount errors — incorrect dividend values

# Bulk download with repair
data = yf.download(
    ["AAPL", "MSFT"],
    period="5y",
    repair=True,
)

# Note: repair near split dates can sometimes be unreliable.
# If data around a split date looks wrong, try fetching a wider
# date range so yfinance has more context for detection.
```

---

## 17. Multi-Level Columns

```python
import yfinance as yf
import pandas as pd

# Multi-ticker download produces multi-level columns
data = yf.download(["AAPL", "MSFT", "GOOG"], period="1y")

# Default group_by='column':
# Level 0: Price type (Close, Open, High, Low, Volume)
# Level 1: Ticker (AAPL, MSFT, GOOG)
close_prices = data["Close"]           # DataFrame of all tickers
aapl_close = data["Close"]["AAPL"]     # Series for AAPL

# group_by='ticker':
# Level 0: Ticker (AAPL, MSFT, GOOG)
# Level 1: Price type (Close, Open, High, Low, Volume)
data_by_ticker = yf.download(["AAPL", "MSFT"], period="1y", group_by="ticker")
aapl_data = data_by_ticker["AAPL"]

# Flatten multi-level columns
data.columns = ["_".join(col).strip() for col in data.columns.values]
# Now: Close_AAPL, Close_MSFT, Close_GOOG, ...

# CSV round-tripping with multi-level headers
data = yf.download(["AAPL", "MSFT"], period="1y")
data.to_csv("prices.csv")

# Read back preserving multi-level structure
df = pd.read_csv("prices.csv", header=[0, 1], index_col=0, parse_dates=True)
# header=[0, 1] tells pandas to read 2 header rows as multi-level

# Disable multi-level entirely
flat_data = yf.download(
    ["AAPL", "MSFT"],
    period="1y",
    multi_level_index=False,
)
```

---

## 18. Session Reuse

```python
import yfinance as yf
import requests

# Create a custom session
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; MyApp/1.0)",
})

# Optional: add authentication or custom adapters
# session.auth = ("user", "pass")
# session.verify = "/path/to/cert.pem"

# Reuse session across all yfinance calls
ticker = yf.Ticker("AAPL", session=session)
info = ticker.info
history = ticker.history(period="1y")

tickers = yf.Tickers("AAPL MSFT", session=session)

market = yf.Market("US", session=session)

search = yf.Search("Apple", session=session)

sector = yf.Sector("technology", session=session)

# Session is reused for connection pooling, auth, proxy, etc.
```

---

## 19. Project Integration Patterns

This project wraps yfinance through `ingestion/app/services/market_data/yfinance/_facade.py` (package `app.services.market_data.yfinance`) with resilience patterns:

```python
from app.services.market_data.yfinance import YFinanceClient, get_yfinance_client

# Singleton access — preferred way
client = get_yfinance_client()
# Or: client = YFinanceClient.get_instance()

# Fetch ticker info with retry + circuit breaker + caching
info = client.fetch_info("AAPL")
# Returns None if all retries fail (validated: >= 10 fields)

# Fetch price history with retry + validation
history = client.fetch_history("AAPL", period="2y")
# Returns None if fewer than 10 rows

# Fetch aligned stock + benchmark data
stock_hist, bench_hist, stock_info = client.fetch_price_and_benchmark(
    symbol="AAPL",
    benchmark="SPY",
    period="2y",
)
# Aligns dates timezone-agnostically; returns (None, None, None) on failure

# Bulk download with rate limiting
data = client.bulk_download(
    symbols=["AAPL", "MSFT", "GOOG"],
    period="2y",
    group_by="ticker",
    auto_adjust=False,
)

# Direct Ticker access (cached in LRU cache)
ticker = client.get_ticker("AAPL")
# Subsequent calls return cached Ticker object

# News with full article content
from app.services.market_data.yfinance import NewsClient

news_client = NewsClient(yf_client=client)
articles = news_client.fetch(
    "AAPL",
    fetch_full_content=True,
    max_articles=5,
)
# Each article dict may include 'full_content' from scraping

# Country-level news aggregation
from app.services.market_data.yfinance import CountryNewsFetcher

fetcher = CountryNewsFetcher(yf_client=client)
us_news = fetcher.fetch_for_country("USA", max_articles=50)
all_news = fetcher.fetch_for_all_countries()
# Deduplicates by title, filters to last 60 days, sorts by date
```
