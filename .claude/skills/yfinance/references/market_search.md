# Market, Search, Lookup, Calendars, Sector, Industry

Top-level utilities that don't hang off a `Ticker`. Use these for discovery, macro state, and sector/industry rollups.

## yf.Market — market status & summary

```python
market = yf.Market("us_market", session=None, timeout=10)
market.status            # dict — market open/close state
market.summary           # dict — market summary data (indices, sectors)
```

**Valid market identifiers:** `us_market`, `gb_market`, `de_market`, `fr_market`, `jp_market`, `hk_market`, `ca_market`, `in_market`, `br_market`, `au_market`

## yf.Search — full-text search

```python
search = yf.Search(
    query="Apple",
    max_results=8,
    news_count=8,
    enable_fuzzy_query=False,
    session=None,
    timeout=10,
)
search.quotes            # list[dict] — matching symbols
search.news              # list[dict] — related news articles
search.research          # list[dict] — research reports
```

Use when you only know the company name, not the ticker — returns best-effort matches plus related news.

## yf.Lookup — screen-like symbol lookup

```python
lookup = yf.Lookup(
    query="tech",
    type="equity",       # "equity", "mutualfund", "etf", "index", "future", "currency"
    session=None,
    timeout=10,
)
lookup.quotes
```

Unlike `Search`, `Lookup` filters by asset type — useful when building auto-complete over equities only (or ETFs only).

## yf.Calendars — upcoming events

```python
cal = yf.Calendars(
    start="2026-01-01",
    end="2026-12-31",
    session=None,
)
cal.earnings             # DataFrame — earnings calendar
cal.ipos                 # DataFrame — IPO calendar
cal.splits               # DataFrame — upcoming stock splits
cal.economic_events      # DataFrame — economic events (CPI, Fed, etc.)
```

## yf.Sector — sector-level rollup

```python
sector = yf.Sector(key="technology", session=None)
sector.overview          # dict — sector overview
sector.top_companies     # DataFrame — top companies in sector
sector.industries        # DataFrame — industries inside the sector
sector.top_etfs          # DataFrame — sector-tracking ETFs
sector.research          # list — research reports
```

**Valid sector keys:** `technology`, `healthcare`, `financial-services`, `consumer-cyclical`, `communication-services`, `industrials`, `consumer-defensive`, `energy`, `basic-materials`, `real-estate`, `utilities`

## yf.Industry — industry-level rollup

```python
industry = yf.Industry(key="semiconductors", session=None)
industry.overview
industry.top_companies
industry.top_etfs
industry.research
```

Industry keys are the hyphenated slugs Yahoo uses (`semiconductors`, `software-infrastructure`, `biotechnology`, ...). Get the full list for a given sector via `sector.industries`.
