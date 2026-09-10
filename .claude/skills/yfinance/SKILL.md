---
name: yfinance
description: |
  Load proactively whenever the user works with yfinance or Yahoo Finance data — pulling price history, fetching financials or analyst data, screening stocks / funds / ETFs, streaming real-time quotes, or inspecting sector, industry, or fund rollups. Do not wait to be asked; apply this skill automatically whenever the user mentions yfinance, Yahoo Finance, OHLCV, stock data, ticker info, earnings estimates, valuation measures, an equity or ETF screener, or real-time quote streaming. Covers yfinance 1.6.0 (Aug 2026): Ticker / Tickers (with live() streaming; global lang/region locale config via yf.config, not per-Ticker args), yf.download, Auth login, Market, Search, Lookup, Screener (EquityQuery, FundQuery, ETFQuery + PREDEFINED_SCREENER_QUERIES), WebSocket, Sector, Industry (region-scoped), Calendars, FundsData, caching, price repair, and the curl_cffi-optional / global-proxy config changes.
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

**Covers yfinance 1.6.0 (Aug 13, 2026).** Major changes since 1.3.0:

- **`yf.Auth`** — log in to a Yahoo account; unlocks subscription-tier data (1.4.0; login/tier via subscriptions API in 1.5.1)
- **`curl_cffi` is now OPTIONAL** (1.4.0) — falls back to the `requests` package when absent. It is **no longer a hard requirement**. If installed, use `curl_cffi >= 0.16` (1.5.2 compat fix)
- **`proxy` removed from per-call / constructor signatures** across `Ticker`, `Search`, `Lookup`, `Market`, `Sector`, `Industry`, `screen()`, and `download()` — configure proxy globally instead
- **`Ticker.live()` / `Tickers.live()`** — first-class WebSocket streaming entry points
- **`lang` / `region` are set via GLOBAL config** (`yf.config` locale), **not** `Ticker` / `Tickers` constructor args. `Sector` / `Industry` DO accept a `region=` constructor arg (default `'US'`) (1.4.0)
- **`yf.download()` is reentrant** (1.4.0) — shared module globals removed; safe under concurrency. `ignore_tz=False` now returns the most-common **exchange** timezone (was UTC before 1.4.0)
- **Valuation measures now sourced from the timeseries API** (1.5.1), not the HTML Statistics scrape — field shapes may differ
- **Price repair: `repair=True` no longer permanently converts GBp/ZAc/ILA sub-unit prices to the main currency** (1.6.0) — repaired sub-unit prices stay in their quoted sub-unit
- **`Calendars` property names**: `earnings_calendar`, `ipo_info_calendar`, `splits_calendar`, `economic_events_calendar` (+ paginated `get_*` methods)
- **Packaging migrated to `pyproject.toml`** (1.6.0); `frozendict` hard dependency dropped (1.4.0)

## Where to look

Keep this file open for orientation, decision guide, and gotchas. For deep detail jump into a topic file:

| You're working on... | Read |
|---|---|
| `Ticker` / `Tickers` basics, `fast_info`, options, `valuation`, `live()`, lang/region | `references/ticker.md` |
| Pulling price history — `yf.download`, `Ticker.history`, `PriceHistory`, multi-level columns | `references/price_history.md` |
| Income / balance / cashflow / earnings / earnings-dates / SEC filings | `references/financials.md` |
| Analyst data — recommendations, estimates, ownership, ESG, `as_dict` toggle | `references/analysis.md` |
| Fund & ETF data — the `funds_data` `FundsData` object | `references/funds.md` |
| Screeners — `EquityQuery`, `FundQuery`, `ETFQuery`, `screen()` signature, predefined screens | `references/screener.md` |
| Market status, Search, Lookup, Calendars, Sector, Industry | `references/market_search.md` |
| Real-time streaming — `WebSocket`, `AsyncWebSocket`, `Ticker.live()` | `references/websocket.md` |
| Proxy, retries, logging, caching, price repair, curl_cffi-optional, Auth | `references/config.md` |
| Worked end-to-end patterns | `PATTERNS.md` |

## Official documentation

yfinance evolves fast — cross-check the upstream docs when something looks off.

| Topic | URL |
|---|---|
| API reference index | https://ranaroussi.github.io/yfinance/reference/index.html |
| User guide | https://ranaroussi.github.io/yfinance/advanced/index.html |
| Advanced config (proxy, caching, sessions) | https://ranaroussi.github.io/yfinance/advanced/config.html |
| Auth (`yf.Auth`) | https://ranaroussi.github.io/yfinance/reference/yfinance.auth.html |
| GitHub releases | https://github.com/ranaroussi/yfinance/releases |

## Architecture

```
yfinance/
├── ticker.py                    # Ticker class — central entry point (+ live() streaming)
├── tickers.py                   # Tickers container (+ live())
├── auth.py                      # Auth — Yahoo account login (v1.4.0+)
├── stock.py                     # Stock info, fast_info, news, ISIN, corporate actions
├── market.py                    # Market status (US-only) and summary
├── financials.py                # Income stmt, balance sheet, cash flow, earnings dates, SEC filings
├── analysis.py                  # Recommendations, price targets, estimates, holdings
├── search.py                    # Search and Lookup classes
├── screener/                    # EquityQuery, FundQuery, ETFQuery, screen(), PREDEFINED_SCREENER_QUERIES
├── websocket.py                 # WebSocket, AsyncWebSocket
├── sector_industry.py           # Sector, Industry classes (region-scoped)
├── calendars.py                 # Calendars (earnings, IPOs, splits, econ events)
├── scrapers/funds.py            # FundsData (reached via Ticker.funds_data)
├── scrapers/history.py          # PriceHistory — low-level backing for Ticker.history()
└── functions.py                 # Module-level download(), enable_debug_mode(), set_tz_cache_location()
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
| Real-time quote stream | `ticker.live()`, `yf.WebSocket()` / `yf.AsyncWebSocket()` |
| Find a ticker by name / fuzzy | `yf.Search(query)` |
| Symbol lookup filtered by asset class | `yf.Lookup(query).stock` / `.etf` / `.get_etf(count=…)` |
| Sector or industry roll-up (top companies, ETFs) | `yf.Sector(key, region=…)` / `yf.Industry(key, region=…)` |
| Earnings / IPO / split / econ calendar | `yf.Calendars(start, end).earnings_calendar` |
| Market open/close status (US) + summary (regional) | `yf.Market("US")` |
| Log in to a Yahoo account (subscription data) | `yf.Auth(...)` |

### Screening

| You want to... | Use |
|---|---|
| Screen equities (market cap, P/E, sector, ...) | `EquityQuery` |
| Screen mutual funds (NAV returns, net assets) | `FundQuery` |
| Screen ETFs (expense ratio, fund net assets, category) | `ETFQuery` |
| Combine conditions | Nested `and` / `or` queries |
| Run a named prebuilt screen | `yf.screen("day_gainers")` (see `PREDEFINED_SCREENER_QUERIES`) |
| Sort / paginate | `yf.screen(q, sortField=…, sortAsc=…, size=…, offset=…)` |

Details: `references/screener.md`.

### Fundamentals

| You want to... | Use |
|---|---|
| Income statement, balance sheet, cash flow | `ticker.income_stmt`, `ticker.balance_sheet`, `ticker.cashflow` |
| Quarterly / TTM variants | `quarterly_*` / `ttm_*` properties, or `get_*(freq="trailing")` |
| 9-metric valuation history (P/E, EV/EBITDA, Price/Book, ...) | `ticker.valuation` (method form: `ticker.get_valuation_measures(freq, periods)`) |
| Analyst recommendations / targets / estimates | `ticker.recommendations`, `analyst_price_targets`, `earnings_estimate`, ... |
| Any accessor as a dict instead of a DataFrame | `ticker.get_<name>(as_dict=True)` |
| Ownership (insiders, institutions) | `ticker.major_holders`, `institutional_holders`, `insider_transactions` |
| Fund holdings, sector weightings, bond info | `ticker.funds_data.top_holdings`, `.sector_weightings`, `.bond_holdings` |

## Common imports cheat sheet

```python
import yfinance as yf
from yfinance import (
    EquityQuery, FundQuery, ETFQuery, screen,   # screeners
    PREDEFINED_SCREENER_QUERIES,                 # named prebuilt screens
)

# Bulk prices (NOTE: actions=False by default, unlike Ticker.history)
data = yf.download(["AAPL", "MSFT"], period="1y", auto_adjust=True)

# Per-ticker object
t = yf.Ticker("AAPL")            # constructor is (ticker, session=None); lang/region are global (yf.config)
t.fast_info.last_price
t.valuation
t.income_stmt
t.recommendations
t.funds_data.top_holdings        # ETF/fund holdings live under funds_data
```

For the full import surface see the matching reference file.

## Key constraints & gotchas

These cut across many entry points — internalize them before starting.

1. **Period ↔ interval limits are enforced silently.** `1m` → max 7 days, `2m–90m` → 60 days, `1h` → 730 days, `1d+` → full history. Intraday cannot extend beyond the last 60 days. Passing a wider `period` than allowed truncates output without warning. **`30m` bars are fetched as `15m` and resampled internally** (Yahoo API bug workaround) — they are derived, not native.
2. **`download()` defaults differ from `history()`.** `download()` uses **`actions=False`** (no dividends/splits columns unless you pass `actions=True`); `Ticker.history()` uses `actions=True`. Both default `auto_adjust=True`. Pass linear/auto-adjusted prices downstream — do not feed log returns to skfolio.
3. **Multi-level columns when `tickers` is a list.** `multi_level_index=True` by default (even for a single ticker in `download`). Use `multi_level_index=False` to flatten; round-trip CSVs with `header=[0, 1]`.
4. **pandas 3+ read-only errors:** `history()` output is memory-consolidated since v1.2.0. Call `.copy()` before mutating in place. (1.6.0 fixed an internal read-only `Adj Close` crash in dividend-adjust repair.)
5. **Screener fields don't cross classes.** Passing an equity field to `FundQuery` raises `ValueError`. Pick the right `*Query` class first; inspect `q.valid_fields` / `q.valid_values`.
6. **`screen()` uses camelCase sort params and split size/count.** It is `screen(query, sortField=…, sortAsc=…, size=…, count=…, offset=…)` — **not** `sort_field` / `sort_type`. Use **`size`** for custom queries (default 100, max 250) and **`count`** for predefined queries (default 25, max 250). `sortAsc` is a bool.
7. **WebSocket — no `run()` / `on_message` attribute.** Use `ws.subscribe([...])` then `ws.listen(handler)` (handler is an optional arg to `listen`, not an assigned attribute) and `ws.close()`. Async methods must be `await`ed. Both support `with` / `async with` context managers.
8. **Fund data lives under `ticker.funds_data`** (a `FundsData` object), not flat `ticker.fund_*` properties. Populated only for ETFs / mutual funds; guard with `ticker.funds_data.quote_type()` or `ticker.fast_info.quote_type`. **`fund_performance` is intentionally not implemented — use `history()`.**
9. **`proxy` is no longer a per-call or constructor argument** (removed across the 1.4.x–1.6.0 line). Configure proxy globally via yfinance config. Flat `yf.config.*` attribute assignment (deprecated since 1.0) still works but proxy is global-only.
10. **`curl_cffi` is OPTIONAL** since v1.4.0 — yfinance falls back to `requests` if it's absent. Do **not** treat it as required. If you do install it, use `curl_cffi >= 0.16` (v1.5.2 fixed a `>=0.16` breakage).
11. **`download()` is reentrant** since v1.4.0 (shared module globals removed) — safe from async workers / worker pools. `ignore_tz=False` returns the most-common **exchange** timezone across the requested tickers (was UTC before 1.4.0); `ignore_tz=True` returns a tz-naive index.
12. **`ticker.info` is slow and flaky** — it scrapes Yahoo's quote page. Prefer `fast_info`, `valuation`, or the dedicated financials/analysis properties whenever they cover the field you need.
13. **`Market.status` is US-only.** Yahoo's markettime endpoint ignores the `market` argument; for any non-US market `status` returns `None` and logs a warning (v1.4.0). Use `Market.summary` for regional data.
14. **Price repair keeps sub-unit currencies as-is (1.6.0).** `repair=True` no longer permanently converts GBp / ZAc / ILA (pence / cents / agorot) prices to the main currency — repaired sub-unit prices stay in their quoted sub-unit. Convert explicitly if you need main-currency values.
15. **Valuation measures come from the timeseries API (1.5.1)**, not the old HTML Statistics scrape — row/field shapes may differ from pre-1.5.1 code. The accessor is the `ticker.valuation` property (method form: `ticker.get_valuation_measures(freq='quarterly', periods=5)`); there is no `ticker.valuation_measures` attribute.

## Version changelog (1.3.0 → 1.6.0)

| Version | Date | Highlights |
|---|---|---|
| **1.6.0** | Aug 13, 2026 | Price-repair improvements; added Balance Sheet keys; screener `dividendyield` + `dividendpershare.lasttwelvemonths` fields; smarter "possibly delisted" messaging; 30m/`15m` interval fixes; pandas 3 / numpy≥2.5 warning fixes; `Lookup` error-handling fix. **Breaking:** `repair=True` keeps GBp/ZAc/ILA in sub-units; packaging → `pyproject.toml` |
| **1.5.2** | Jul 23, 2026 | Fix breakage with `curl_cffi >= 0.16` |
| **1.5.1** | Jun 28, 2026 | Valuation measures via timeseries API (was HTML scrape); chunked fundamentals fallback on timeout; login/subscription tier via subscriptions API; proxy-string normalization; several price/dividend-repair fixes. **Supersedes the retracted 1.5.0** |
| **1.5.0** | Jun 2026 | **RETRACTED / yanked** (dev branch not merged). Do not pin — use 1.5.1 |
| **1.4.1** | May 28, 2026 | Preserve the Date/Datetime index name in `download()` output |
| **1.4.0** | May 23, 2026 | `yf.Auth`; `region` scoping for Sector/Industry; `lang`/`region` for Ticker; `curl_cffi` made optional (fallback to `requests`); `repair` added to `get_history_metadata()`; `download()` reentrant; localized-intraday UTC fix; Market region validation. **Breaking:** curl_cffi optional; `frozendict` dropped; download reentrancy; new constructor scoping params |
| **1.3.0** | Apr 16, 2026 | `Ticker.valuation` (valuation-measures history; accessor property, method form `get_valuation_measures`); `ETFQuery` screener; `Ticker.dividends` type-regression fix |

## Dependencies

- `curl_cffi` — **optional** since v1.4.0 (falls back to `requests`). If installed, use `>= 0.16` (v1.5.2 compat fix). Not a hard requirement.
- `requests` — the fallback HTTP transport when `curl_cffi` is absent.
- `pandas` — `history()`/`download()` output is memory-consolidated; on pandas 3+ call `.copy()` before mutating in place.
- `frozendict` — **no longer a hard dependency** (dropped v1.4.0; internal fallback).
- Packaging is `pyproject.toml`-based (v1.6.0); there is no `setup.py` install path.
- Python 3.8+ (check `pyproject.toml` for the current floor).

## Auth (v1.4.0+)

`yf.Auth` logs in to a Yahoo account so subscription-tier data becomes available. Login state and subscription tier are resolved via Yahoo's subscriptions API (1.5.1), and login cookies are preserved across cookie-strategy switches. See the Auth reference page (URL above) for the exact constructor — do not assume credentials handling; consult the upstream docs before wiring secrets.

## Implementation patterns

End-to-end examples — bulk downloads, screeners, financials pipelines, valuation-history analysis, websocket streams, portfolio-data prep — in `PATTERNS.md`.
