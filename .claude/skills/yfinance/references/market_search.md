# Market, Search, Lookup, Calendars, Sector, Industry

Top-level utilities that don't hang off a `Ticker`. Use these for discovery, macro state, and
sector/industry rollups. **None of these take a `proxy` argument any more** — configure proxy
globally (see `config.md`).

## yf.Market — market status & summary

```python
market = yf.Market("US", session=None, timeout=30)
market.status            # dict — market open/close state (US ONLY)
market.summary           # dict — market summary data (regional, all markets)
```

**Eight documented market identifiers:** `US`, `GB`, `ASIA`, `EUROPE`, `RATES`,
`COMMODITIES`, `CURRENCIES`, `CRYPTOCURRENCIES`.

> **Gotcha (v1.4.0):** `Market.status` is backed by Yahoo's markettime endpoint, which
> **ignores the `market` argument and returns U.S. data only**. For any non-US market,
> `status` returns `None` and logs a warning. Only `Market.summary` returns regional data for
> all eight markets. Default `timeout=30`. No `proxy` parameter.

## yf.Search — full-text search

```python
search = yf.Search(
    query="Apple",
    max_results=8,
    news_count=8,
    lists_count=8,
    include_cb=True,               # company breakdown
    include_nav_links=False,
    include_research=False,
    include_cultural_assets=False,
    enable_fuzzy_query=False,      # tolerate typos
    recommended=8,
    session=None,
    timeout=30,
    raise_errors=True,
)
search.quotes            # list[dict] — matching symbols
search.news              # list[dict] — related news
search.lists             # list — populated via lists_count
search.research          # list — populated only when include_research=True
search.nav               # list — populated only when include_nav_links=True
search.all               # filtered aggregate view of the response
search.response          # raw unfiltered API payload
search.search()          # re-run the query (returns self); populates the properties
```

Use when you only know the company name, not the ticker. `research` / `nav` are empty unless
the corresponding `include_*` flag is set. `raise_errors=True` by default — set `False` to
suppress exceptions. **No `type` and no `proxy` parameter** (that filtering is `Lookup`'s job).

## yf.Lookup — asset-class-filtered symbol lookup

`Lookup` has **no `type=` argument**. You pick the asset class via a property (full result set)
or a `get_*` method (count-limited DataFrame).

```python
lookup = yf.Lookup(query="tech", session=None, timeout=30, raise_errors=True)

# Full result set per asset class (properties):
lookup.all
lookup.stock
lookup.mutualfund
lookup.etf
lookup.index
lookup.future
lookup.currency
lookup.cryptocurrency

# Count-limited pandas DataFrames (methods, default count=25):
lookup.get_all(count=25)
lookup.get_stock(count=25)
lookup.get_etf(count=25)
lookup.get_mutualfund(count=25)
lookup.get_index(count=25)
lookup.get_future(count=25)
lookup.get_currency(count=25)
lookup.get_cryptocurrency(count=25)
```

> **Breaking vs older skill docs:** the old `yf.Lookup(query, type="equity").quotes` form is
> gone. Use the per-asset-class property (`.stock`, `.etf`, ...) or `get_*(count=…)`.

## yf.Calendars — upcoming events

```python
cal = yf.Calendars(start=None, end=None, session=None)
# start/end accept str | datetime | date | None

# Convenience properties (default settings, no filtering/pagination):
cal.earnings_calendar
cal.ipo_info_calendar
cal.splits_calendar
cal.economic_events_calendar

# Manual query methods (date range + pagination + cache bypass):
cal.get_earnings_calendar(market_cap=None, filter_most_active=True,
                          start=None, end=None, limit=12, offset=0, force=False)
cal.get_ipo_info_calendar(start=None, end=None, limit=12, offset=0, force=False)
cal.get_splits_calendar(start=None, end=None, limit=12, offset=0, force=False)
cal.get_economic_events_calendar(start=None, end=None, limit=12, offset=0, force=False)
```

> **Gotchas:**
> - Property names are `earnings_calendar` / `ipo_info_calendar` / `splits_calendar` /
>   `economic_events_calendar` — **not** `earnings` / `ipos` / `splits` / `economic_events`.
> - All `get_*` methods paginate: **`limit` defaults to 12** (not unlimited). Raise it or page
>   with `offset` to pull more.
> - `force=True` bypasses the cache.
> - `market_cap` and `filter_most_active` exist **only** on `get_earnings_calendar`;
>   `filter_most_active` defaults to `True` (earnings restricted to most-active names unless
>   disabled).

`Ticker.calendar` remains a separate per-ticker property returning a dict of the next
earnings/dividend events.

## yf.Sector — sector-level rollup

```python
sector = yf.Sector(key="technology", session=None, region="US")
sector.key                 # sector identifier
sector.name                # human-readable name
sector.symbol              # sector symbol
sector.ticker              # associated Ticker object (property, not a call)
sector.overview            # dict
sector.top_companies       # DataFrame — region-scoped
sector.research_reports    # list[dict]   (renamed from .research)
sector.top_etfs            # dict symbol -> name — region-scoped
sector.top_mutual_funds    # dict symbol -> name — region-scoped
sector.industries          # industries within the sector
```

**Valid sector keys:** `technology`, `healthcare`, `financial-services`, `consumer-cyclical`,
`communication-services`, `industrials`, `consumer-defensive`, `energy`, `basic-materials`,
`real-estate`, `utilities`.

## yf.Industry — industry-level rollup

```python
industry = yf.Industry(key="semiconductors", session=None, region="US")
industry.key
industry.name
industry.symbol
industry.ticker
industry.overview
industry.top_companies              # region-scoped
industry.research_reports
industry.sector_key                 # parent sector key
industry.sector_name                # parent sector name
industry.top_performing_companies   # region-scoped
industry.top_growth_companies       # region-scoped
```

Industry keys are the hyphenated slugs Yahoo uses (`semiconductors`,
`software-infrastructure`, `biotechnology`, ...). Get the full list for a sector via
`sector.industries`.

> **region (v1.4.0):** both `Sector` and `Industry` accept a `region` = ISO 3166-1 alpha-2
> country code (case-insensitive); omitting it defaults to U.S. data. `region` only affects the
> list-style rollups (`top_companies`, `top_etfs`, `top_mutual_funds`, industry
> performing/growth lists) — not identity fields like `key` / `name` / `symbol`. The old
> `proxy` constructor argument is gone (replaced by `region` + global proxy config), and the
> `.research` property is now `.research_reports`.
