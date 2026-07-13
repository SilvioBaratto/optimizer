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

# Earnings (simplified income — Revenue and Earnings columns)
annual_earn = ticker.earnings
quarterly_earn = ticker.quarterly_earnings

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

# Shares outstanding over time
shares = ticker.shares_full
# Or with date range:
shares = ticker.get_shares_full(start="2020-01-01", end="2024-01-01")
```

---

## 8. Search & Lookup

```python
import yfinance as yf

# Full-text search across Yahoo Finance
search = yf.Search("Apple")

# Matching stock symbols
for quote in search.quotes:
    print(f"{quote['symbol']}: {quote['shortname']} ({quote['exchange']})")

# Related news
for article in search.news:
    print(f"{article['title']}")

# Research reports
for report in search.research:
    print(f"{report['title']}")

# Lookup — screen-like symbol search by type
lookup = yf.Lookup("semiconductor", type="equity")
for quote in lookup.quotes:
    print(f"{quote['symbol']}: {quote['shortname']}")

# Lookup for ETFs
etf_lookup = yf.Lookup("bond", type="etf")

# Lookup for mutual funds
fund_lookup = yf.Lookup("growth", type="mutualfund")
```

---

## 9. Market Status

```python
import yfinance as yf

# US market
market = yf.Market("us_market")

# Market status (open/closed/pre/post)
status = market.status
print(status)

# Market summary (major indices, key metrics)
summary = market.summary
print(summary)

# Other markets
uk = yf.Market("gb_market")
japan = yf.Market("jp_market")
germany = yf.Market("de_market")
```

---

## 10. Sector & Industry

```python
import yfinance as yf

# Sector data
tech = yf.Sector("technology")
print(tech.overview)
print(tech.top_companies)     # Top companies by market cap
print(tech.industries)        # Industries within sector
print(tech.top_etfs)          # Sector-tracking ETFs
print(tech.research)          # Research reports

# Industry data
semis = yf.Industry("semiconductors")
print(semis.overview)
print(semis.top_companies)
print(semis.top_etfs)
print(semis.research)

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
    EquityQuery("gt", ["marketcap", 10_000_000_000]),  # > $10B
    EquityQuery("eq", ["sector", "Technology"]),
])

result = yf.screen(query, sort_field="marketcap", sort_type="desc", size=25)
for stock in result["quotes"]:
    print(f"{stock['symbol']}: ${stock.get('marketCap', 0):,.0f}")

# Complex query with nested AND/OR
query = EquityQuery("and", [
    EquityQuery("gt", ["marketcap", 1_000_000_000]),
    EquityQuery("or", [
        EquityQuery("gt", ["dividendyield", 3]),
        EquityQuery("lt", ["peratio", 15]),
    ]),
    EquityQuery("eq", ["region", "us"]),
])

result = yf.screen(query, sort_field="dividendyield", sort_type="desc", size=50)
print(f"Total matches: {result['total']}")

# Between operator
query = EquityQuery("and", [
    EquityQuery("btwn", ["intradayprice", 10, 50]),
    EquityQuery("gt", ["volume", 1_000_000]),
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

fund_query = FundQuery("and", [
    FundQuery("gt", ["netassets", 1_000_000_000]),
    FundQuery("lt", ["annualreturnnavy5", 10]),
])
fund_result = yf.screen(fund_query, sort_field="netassets", sort_type="desc")

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
vm = t.valuation_measures
# DataFrame with rows for Market Cap, Enterprise Value, Trailing P/E,
# Forward P/E, PEG Ratio (5yr), Price/Sales, Price/Book, EV/Revenue, EV/EBITDA
# and columns for current + historical quarters and year-ends.

print(vm.loc["Trailing P/E"])       # time series of trailing P/E
print(vm.iloc[:, 0])                # all 9 metrics for the most recent period
```

---

## 12. WebSocket Real-Time Data

```python
import yfinance as yf

# Synchronous WebSocket
def on_message(ws, msg):
    """Called for each price update."""
    print(f"Symbol: {msg['id']}, Price: {msg['price']}, Volume: {msg['dayVolume']}")

ws = yf.WebSocket()
ws.subscribe(["AAPL", "MSFT", "GOOG"])
ws.on_message = on_message
ws.run()  # Blocks — runs until interrupted

# Asynchronous WebSocket
import asyncio

async def stream_prices():
    ws = yf.AsyncWebSocket()
    ws.subscribe(["AAPL", "MSFT"])

    async for msg in ws:
        print(f"{msg['id']}: {msg['price']}")
        # Break condition:
        # if some_condition:
        #     break

asyncio.run(stream_prices())
```

---

## 13. Calendar Events

```python
import yfinance as yf

# Calendar data for a date range
cal = yf.Calendars(start="2024-01-01", end="2024-03-31")

# Earnings calendar
earnings = cal.earnings
print(earnings.head())

# IPO calendar
ipos = cal.ipos
print(ipos.head())

# Stock splits
splits = cal.splits
print(splits.head())

# Economic events
econ = cal.economic_events
print(econ.head())

# Single ticker calendar info
ticker = yf.Ticker("AAPL")
cal_info = ticker.calendar
# Shows next earnings date, ex-dividend date, etc.
```

---

## 14. Fund Data

```python
import yfinance as yf

# ETF example
spy = yf.Ticker("SPY")

# Fund overview
overview = spy.fund_overview
print(overview)

# Top holdings with weights
holdings = spy.fund_top_holdings
print(holdings.head(10))

# Sector weightings
sectors = spy.fund_sector_weightings
print(sectors)

# Asset allocation (stocks/bonds/cash/other)
allocation = spy.fund_asset_allocation
print(allocation)

# Performance data
perf = spy.fund_performance
print(perf)

# Holding info (AUM, turnover, inception)
info = spy.fund_holding_info
print(info)

# Equity holdings characteristics (P/E, P/B of holdings)
eq_hold = spy.fund_equity_holdings
print(eq_hold)

# Bond holdings (for bond funds/ETFs)
bond_etf = yf.Ticker("AGG")
bond_hold = bond_etf.fund_bond_holdings
bond_ratings = bond_etf.fund_bond_ratings
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

market = yf.Market("us_market", session=session)

search = yf.Search("Apple", session=session)

sector = yf.Sector("technology", session=session)

# Session is reused for connection pooling, auth, proxy, etc.
```

---

## 19. Project Integration Patterns

This project wraps yfinance through `@yfinance_client/client.py` with resilience patterns:

```python
from yfinance_client import YFinanceClient, get_yfinance_client

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

# Adapter layer for dependency injection
from optimizer.adapters.yfinance import PriceAdapter, MarketCapAdapter

price_provider = PriceAdapter()  # uses singleton client internally
history = price_provider.fetch_history("AAPL", period="2y")
stock, bench, info = price_provider.fetch_price_and_benchmark("AAPL")

mcap_provider = MarketCapAdapter()
market_caps = mcap_provider.get_market_caps(["AAPL", "MSFT", "GOOG"])
# Returns pd.Series of market caps

# News with full article content
from yfinance_client import NewsClient

news_client = NewsClient(yf_client=client)
articles = news_client.fetch(
    "AAPL",
    fetch_full_content=True,
    max_articles=5,
)
# Each article dict may include 'full_content' from scraping

# Country-level news aggregation
from yfinance_client import CountryNewsFetcher

fetcher = CountryNewsFetcher(yf_client=client)
us_news = fetcher.fetch_for_country("USA", max_articles=50)
all_news = fetcher.fetch_for_all_countries()
# Deduplicates by title, filters to last 60 days, sorts by date
```
