---
name: yfinance
description: |
  Load proactively whenever the user works with yfinance or Yahoo Finance data — pulling price history, fetching financials or analyst data, screening stocks / funds / ETFs, streaming real-time quotes, or inspecting sector, industry, or fund rollups. Do not wait to be asked; apply this skill automatically whenever the user mentions yfinance, Yahoo Finance, OHLCV, stock data, ticker info, earnings estimates, valuation measures, an equity or ETF screener, or real-time quote streaming. Covers yfinance 1.3.0 (April 2026): Ticker / Tickers, yf.download, Market, Search, Lookup, Screener (EquityQuery, FundQuery, ETFQuery), WebSocket, Sector, Industry, Calendars, FundsData, caching, price repair, and the new config deprecations.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - WebFetch
  - WebSearch
  - mcp__ide__getDiagnostics
---

# yfinance

Expert guidance for **yfinance** — a Python library for retrieving financial data from Yahoo Finance.

**Covers yfinance 1.3.0 (April 2026).** Major additions since 1.0 (Dec 2025):

- `ETFQuery` screener class (1.3.0)
- `Ticker.valuation_measures` — 9-metric valuation history table (1.3.0)
- Currency column on analysis-data tables (1.2.2)
- Thread-safe `download()` (1.2.1 / 1.2.2)
- Price-repair: capital-gains double-counting fix + fewer false-positive splits (1.1.0)
- `curl_cffi ≥ 0.15` required (1.2.1) — CVE mitigation
- `history()` / `download()` output memory-consolidated (1.2.0) — pandas 3+ may surface read-only errors on in-place mutation
- New config method; flat `yf.config.*` attribute assignment still works but emits `DeprecationWarning` (1.0)

## Where to look

Keep this file open for orientation, decision guide, and gotchas. For deep detail jump into a topic file:

| You're working on... | Read |
|---|---|
| `Ticker` / `Tickers` basics, `fast_info`, options, `valuation_measures` | `references/ticker.md` |
| Pulling price history — `yf.download`, `Ticker.history`, multi-level columns | `references/price_history.md` |
| Income / balance / cashflow / earnings / SEC filings | `references/financials.md` |
| Analyst data — recommendations, estimates, ownership, ESG | `references/analysis.md` |
| Fund & ETF data — holdings, sector weightings, bond info | `references/funds.md` |
| Screeners — `EquityQuery`, `FundQuery`, `ETFQuery`, field lists | `references/screener.md` |
| Market status, Search, Lookup, Calendars, Sector, Industry | `references/market_search.md` |
| Real-time streaming — `WebSocket`, `AsyncWebSocket` | `references/websocket.md` |
| Proxy, retries, logging, caching, price repair, deprecations | `references/config.md` |
| Worked end-to-end patterns | `PATTERNS.md` |

## Official documentation

yfinance evolves fast — cross-check the upstream docs when something looks off.

| Topic | URL |
|---|---|
| API reference index | https://ranaroussi.github.io/yfinance/reference/index.html |
| User guide | https://ranaroussi.github.io/yfinance/advanced/index.html |
| Advanced config | https://ranaroussi.github.io/yfinance/advanced/config.html |
| GitHub releases | https://github.com/ranaroussi/yfinance/releases |

## Architecture

```
yfinance/
├── ticker.py           # Ticker class — central entry point
├── stock.py            # Stock info, fast_info, news, ISIN
├── market.py           # Market status and summary
├── financials.py       # Income stmt, balance sheet, cash flow
├── analysis.py         # Recommendations, price targets, estimates
├── price_history.py    # history(), download()
├── search.py           # Search and Lookup classes
├── screener/           # EquityQuery, FundQuery, ETFQuery, screen()
├── websocket.py        # WebSocket, AsyncWebSocket
├── sector_industry.py  # Sector, Industry classes
├── calendars.py        # Calendars (earnings, IPOs, splits, econ events)
├── funds_data.py       # Fund-specific data (holdings, weightings)
└── functions.py        # Module-level download() helper
```

## Decision guide

Start here — pick the right tool for the task, then dive into the matching reference file.

### Retrieving data

| You want to... | Use |
|---|---|
| One-shot OHLCV for many tickers | `yf.download(tickers, period="1y")` |
| Rich per-ticker object (info, financials, options, news) | `yf.Ticker(sym)` |
| Per-ticker OHLCV with pre/post, error control | `ticker.history(...)` |
| Live-ish summary metrics (price, vol, 52w range) | `ticker.fast_info` |
| Descriptive long-tail fields (officers, summary, ...) | `ticker.info` |
| Real-time quote stream | `yf.WebSocket()` / `yf.AsyncWebSocket()` |
| Find a ticker by name / fuzzy | `yf.Search(query)` or `yf.Lookup(query, type=...)` |
| Sector or industry roll-up (top companies, ETFs) | `yf.Sector(key)` / `yf.Industry(key)` |
| Earnings / IPO / split / econ calendar | `yf.Calendars(start, end)` |
| Market open/close status | `yf.Market("us_market")` |

### Screening

| You want to... | Use |
|---|---|
| Screen equities (market cap, P/E, sector, ...) | `EquityQuery` |
| Screen mutual funds (NAV returns, net assets) | `FundQuery` |
| Screen ETFs (expense ratio, fund net assets, category) | `ETFQuery` **(v1.3.0+)** |
| Combine conditions | Nested `and` / `or` queries |
| Paginate past 250 results | `offset` on `yf.screen(...)` |

Details: `references/screener.md`.

### Fundamentals

| You want to... | Use |
|---|---|
| Income statement, balance sheet, cash flow | `ticker.income_stmt`, `ticker.balance_sheet`, `ticker.cashflow` |
| Quarterly / TTM variants | `quarterly_*` properties or `get_*(freq="trailing")` |
| 9-metric valuation history (P/E, EV/EBITDA, Price/Book, ...) | `ticker.valuation_measures` **(v1.3.0+)** |
| Analyst recommendations / targets / estimates | `ticker.recommendations`, `analyst_price_targets`, `earnings_estimate`, ... |
| Ownership (insiders, institutions) | `ticker.major_holders`, `institutional_holders`, `insider_transactions` |
| Fund holdings, sector weightings, bond info | `ticker.fund_top_holdings`, `fund_sector_weightings`, `fund_bond_holdings` |

## Common imports cheat sheet

```python
import yfinance as yf
from yfinance import (
    EquityQuery, FundQuery, ETFQuery, screen,   # screeners
)

# Bulk prices
data = yf.download(["AAPL", "MSFT"], period="1y", auto_adjust=True)

# Per-ticker object
t = yf.Ticker("AAPL")
t.fast_info.lastPrice
t.valuation_measures                    # v1.3.0+
t.income_stmt
t.recommendations
```

For the full import surface see the matching reference file.

## Key constraints & gotchas

These cut across many entry points — internalize them before starting.

1. **Period ↔ interval limits are enforced silently.** `1m` → max 7 days, `2m–90m` → 60 days, `1h` → 730 days, `1d+` → full history. Passing a wider `period` than allowed truncates output without warning.
2. **Always pass linear returns / auto-adjusted prices downstream.** `yf.download` returns `auto_adjust=True` by default. If you disable it, remember to adjust for splits and dividends manually before computing returns.
3. **Multi-level columns when `tickers` is a list.** Use `multi_level_index=False` to flatten; round-trip CSVs with `header=[0, 1]`.
4. **pandas 3+ read-only errors:** `history()` output is memory-consolidated since v1.2.0. Call `.copy()` before mutating in place.
5. **Screener fields don't cross classes.** Passing `marketcap` to `FundQuery` raises `ValueError`. Pick the right `*Query` class first.
6. **Screener `size` max is 250.** Paginate with `offset` for larger result sets.
7. **WebSocket — bind handler before `subscribe` / `run`.** Otherwise the first messages are dropped.
8. **Fund-only properties** (`ticker.fund_*`) return empty structures on equities. Guard with `ticker.fast_info.quoteType in ("ETF", "MUTUALFUND")`.
9. **`yf.config.*` attribute assignment is deprecated since v1.0** but still functional. For new projects prefer the new config method per the advanced config docs.
10. **`curl_cffi ≥ 0.15` is required** since v1.2.1 (CVE fix). Do not pin older versions.
11. **Thread safety:** `download()` is thread-safe since v1.2.1 / 1.2.2. Earlier versions would corrupt DataFrames under concurrent use.
12. **`ticker.info` is slow and flaky** — it scrapes Yahoo's quote page. Prefer `fast_info`, `valuation_measures`, or the dedicated financials/analysis properties whenever they cover the field you need.

## Version changelog (1.0 → 1.3.0)

| Version | Date | Highlights |
|---|---|---|
| **1.3.0** | Apr 16, 2026 | `ETFQuery` screener; `Ticker.valuation_measures`; `Ticker.dividends` type regression fix |
| **1.2.2** | Apr 13, 2026 | Currency column on analysis data; `download()` thread-safety; `history()` TypeError fix |
| **1.2.1** | Apr 7, 2026 | `curl_cffi ≥ 0.15` (CVE mitigation); dividend currency preservation |
| **1.2.0** | Feb 16, 2026 | `history()` DataFrame consolidation (read-only surface on pandas 3+); expanded screener country/exchange coverage |
| **1.1.0** | Jan 24, 2026 | Price repair: capital-gains double-counting fix; fewer false-positive splits |
| **1.0** | Dec 22, 2025 | Stable 1.0 graduation; no breaking changes; new config method with deprecation warnings on flat `yf.config.*` attribute assignment |

## Dependencies

- `curl_cffi >= 0.15` (required since v1.2.1 — do **not** pin older)
- `pandas` — output is memory-consolidated; on pandas 3+ call `.copy()` before mutating in place
- Python 3.8+ (check `pyproject.toml` for current floor)

## Implementation patterns

End-to-end examples — bulk downloads, screeners, financials pipelines, valuation-history analysis, websocket streams, portfolio-data prep — in `PATTERNS.md`.
