# Screener — EquityQuery, FundQuery, ETFQuery

Three query classes share the same operator surface but validate against different field maps. Using a field from the wrong class raises `ValueError`.

```python
from yfinance import EquityQuery, FundQuery, ETFQuery, screen
```

- `EquityQuery` — stocks, fields from `EQUITY_SCREENER_FIELDS`
- `FundQuery` — mutual funds, fields from `FUND_SCREENER_FIELDS`
- `ETFQuery` — ETFs (**v1.3.0+**), fields from `ETF_SCREENER_FIELDS`

## Operators

| Operator | Meaning |
|---|---|
| `gt`, `lt`, `gte`, `lte` | Comparison |
| `eq` | Exact match |
| `is-in` | Value in a list |
| `btwn` | Between two values (inclusive) |
| `and`, `or` | Logical composition |

## Executing a screen

```python
query = EquityQuery("and", [
    EquityQuery("gt", ["marketcap", 1_000_000_000]),
    EquityQuery("lt", ["peratio", 20]),
    EquityQuery("eq", ["sector", "Technology"]),
])

result = yf.screen(
    query,
    sort_field="marketcap",
    sort_type="desc",          # "asc" | "desc"
    offset=0,
    size=25,                   # max 250
)
# result["quotes"] — list of matching stocks
# result["total"]  — total matches
```

## Nested AND/OR

```python
query = EquityQuery("or", [
    EquityQuery("and", [
        EquityQuery("gt", ["intradayprice", 50]),
        EquityQuery("lt", ["intradayprice", 200]),
    ]),
    EquityQuery("gt", ["dividendyield", 3]),
])
```

## Pagination

Yahoo caps `size` at 250. Paginate with `offset`:

```python
all_results = []
for offset in range(0, 2000, 250):
    page = yf.screen(query, size=250, offset=offset)
    all_results.extend(page["quotes"])
    if len(page["quotes"]) < 250:
        break
```

## Fund screening

```python
fund_query = FundQuery("and", [
    FundQuery("gt", ["netassets", 1_000_000_000]),
    FundQuery("lt", ["annualreturnnavy5", 10]),
])
yf.screen(fund_query, sort_field="netassets", sort_type="desc")
```

## ETF screening (v1.3.0+)

```python
etf_query = ETFQuery("and", [
    ETFQuery("gt", ["fundnetassets", 500_000_000]),
    ETFQuery("eq", ["region", "us"]),
])
yf.screen(etf_query, size=50)
```

**Predefined screens shipped with 1.3.0:** Top US ETFs, Top Performing ETFs, Technology ETFs, Bond ETFs.

## EquityQuery field categories

- **Valuation:** `marketcap`, `peratio`, `pbratio`, `enterprisevalue`, `evtoebitda`, `evtorevenue`, `pricetosales`
- **Price:** `intradayprice`, `intradaymarketcap`, `fiftytwowkhigh`, `fiftytwoweeklow`, `intradaypricepctchange`
- **Dividends:** `dividendyield`, `trailingannualdividendyield`, `payoutratio`
- **Financials:** `revenue`, `ebitda`, `netincome`, `totaldebt`, `totalcash`, `grossprofitmargin`, `operatingmargin`, `profitmargin`
- **Growth:** `revenuegrowthquarterly`, `earningsgrowthquarterly`
- **Classification:** `sector`, `industry`, `exchange`, `region`
- **Analyst:** `recommendationkey`, `numberofanalystopinions`, `targetmeanprice`

## FundQuery fields

Commonly used: `netassets`, `category`, `performancerating`, `riskrating`, `annualreturnnavy1`, `annualreturnnavy3`, `annualreturnnavy5`, `initialinvestment`, `expenseratio`.

## ETFQuery fields (v1.3.0+)

Pulled from Yahoo's `instrument/etf/fields` endpoint — includes fund net assets, category, expense ratio, NAV-based returns, region, exchange, and ETF-specific performance metrics. The full list lives in `ETF_SCREENER_FIELDS` inside `yfinance/screener/query.py`.

## Cross-class mistake

```python
# BAD — raises ValueError: 'marketcap' not in FUND_SCREENER_FIELDS
FundQuery("gt", ["marketcap", 1_000_000_000])
```

When in doubt, inspect the field maps:

```python
from yfinance.screener.query import (
    EQUITY_SCREENER_FIELDS, FUND_SCREENER_FIELDS, ETF_SCREENER_FIELDS,
)
```
