# Screener — EquityQuery, FundQuery, ETFQuery, screen()

Three query classes share the same operator surface but validate against different field maps.
Using a field from the wrong class raises `ValueError`.

```python
from yfinance import EquityQuery, FundQuery, ETFQuery, screen, PREDEFINED_SCREENER_QUERIES
```

- `EquityQuery` — stocks
- `FundQuery` — mutual funds
- `ETFQuery` — ETFs

## Query construction

All three share the same constructor:

```python
Query(operator, operand)
# operator: Literal['eq','is-in','btwn','gt','lt','gte','lte','and','or']
# operand:  list[Query]         for logical ops ('and'/'or')
#           [field, value, ...] for value ops
```

```python
q = EquityQuery("and", [
    EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
    EquityQuery("lt", ["epsgrowth.lasttwelvemonths", 15]),
])
```

## Operators

| Operator | Meaning |
|---|---|
| `gt`, `lt`, `gte`, `lte` | Comparison |
| `eq` | Exact match |
| `is-in` | Value in a list (`["field", v1, v2, ...]`) |
| `btwn` | Between two values (inclusive) |
| `and`, `or` | Logical composition (operands are nested queries) |

## Introspecting a query class

Every query object exposes the valid surface so you never guess field names:

```python
q = EquityQuery("eq", ["region", "us"])
q.valid_fields       # dict: category -> [field names]  (valuation, price, profitability, esg, ...)
q.valid_values       # dict: field -> allowed values     (region/exchange/sector codes, ...)
q.to_dict()          # JSON-serialisable payload sent to Yahoo
q.operator           # the operator token; q.operands the operands
```

> The only public introspection members are `valid_fields`, `valid_values`, `to_dict`,
> plus `operator` / `operands`. There is **no `valid_operations`** (accessing it raises
> `AttributeError`). Valid operator tokens are: `eq`, `is-in`, `btwn`, `gt`, `lt`, `gte`,
> `lte`, `and`, `or`.

- `EquityQuery.valid_fields` groups: `eq_fields`, `price`, `trading`, `short_interest`,
  `valuation`, `profitability`, `leverage`, `liquidity`, `income_statement`,
  `balance_sheet`, `cash_flow`, `esg`.
- `FundQuery` / `ETFQuery` expose analogous category-grouped fields and restricted value sets.

## Executing a screen

```python
result = yf.screen(
    query,                 # str (predefined name) OR a Query object
    offset=0,              # pagination start (default 0)
    size=100,              # CUSTOM queries: default 100, max 250
    count=25,              # PREDEFINED queries: default 25, max 250
    sortField="ticker",    # field to sort by (default "ticker")  -- NOTE camelCase
    sortAsc=False,         # ascending? (default False)           -- NOTE bool, not "desc"
    session=None,
)
# result is the raw Yahoo response dict; records are under result["quotes"]
```

> **Breaking vs older skill docs:** the sort parameters are **`sortField`** (camelCase) and
> **`sortAsc`** (bool) — **not** `sort_field` / `sort_type="desc"`. There is **no `proxy`**
> argument (configure proxy globally). `userId` / `userIdType` (default `"guid"`) exist for
> user-identity but are rarely needed.

**`size` vs `count`:** use **`size`** for custom `Query` objects, **`count`** for predefined
names. Object-level defaults (`offset=0`, `size=100`, `count=25`, `sortField="ticker"`,
`sortAsc=False`) only apply when `query` is a `Query` object; a predefined-name string bypasses them.

```python
q = EquityQuery("and", [
    EquityQuery("gt", ["intradaymarketcap", 1_000_000_000]),
    EquityQuery("lt", ["peratio.lasttwelvemonths", 20]),
    EquityQuery("eq", ["sector", "Technology"]),
])
result = yf.screen(q, sortField="intradaymarketcap", sortAsc=False, size=25)
for stock in result["quotes"]:
    print(stock["symbol"], stock.get("marketCap"))
```

## Predefined screens

```python
list(yf.PREDEFINED_SCREENER_QUERIES.keys())
yf.screen("day_gainers", count=50)          # predefined -> use count
```

19 documented names:

- **Equity:** `aggressive_small_caps`, `day_gainers`, `day_losers`, `growth_technology_stocks`,
  `most_actives`, `most_shorted_stocks`, `small_cap_gainers`, `undervalued_growth_stocks`,
  `undervalued_large_caps`
- **Fund:** `conservative_foreign_funds`, `high_yield_bond`, `portfolio_anchors`,
  `solid_large_growth_funds`, `solid_midcap_growth_funds`, `top_mutual_funds`
- **ETF:** `top_etfs_us`, `top_performing_etfs`, `technology_etfs`, `bond_etfs`

## Nested AND/OR

```python
q = EquityQuery("or", [
    EquityQuery("and", [
        EquityQuery("gt", ["intradayprice", 50]),
        EquityQuery("lt", ["intradayprice", 200]),
    ]),
    EquityQuery("gt", ["dividendyield", 3]),
])
```

## Pagination

`size`/`count` cap at 250. Paginate with `offset`:

```python
all_results = []
for offset in range(0, 2000, 250):
    page = yf.screen(query, size=250, offset=offset)
    quotes = page["quotes"]
    all_results.extend(quotes)
    if len(quotes) < 250:
        break
```

## Field names

`EquityQuery.valid_fields` categories (dotted, timeframe-qualified suffixes are the norm —
there is **no bare `marketcap`, `peratio`, or `revenue`**):

- **eq_fields:** `exchange`, `region`, `sector`, `peer_group`, `industry`
- **price:** `intradayprice`, `intradaymarketcap`, `intradaypricechange`, `eodprice`, `percentchange`, `fiftytwowkpercentchange`, `lastclose52weekhigh.lasttwelvemonths`, `lastclose52weeklow.lasttwelvemonths`, `lastclosemarketcap.lasttwelvemonths`
- **trading:** `dayvolume`, `eodvolume`, `avgdailyvol3m`, `beta`, `pctheldinst`, `pctheldinsider`
- **short_interest:** `short_interest.value`, `short_percentage_of_float.value`, `short_percentage_of_shares_outstanding.value`, `short_interest_percentage_change.value`, `days_to_cover_short.value`
- **valuation:** `peratio.lasttwelvemonths`, `pricebookratio.quarterly`, `pegratio_5y`, `bookvalueshare.lasttwelvemonths`, `lastclosepriceearnings.lasttwelvemonths`, `lastclosepricetangiblebookvalue.lasttwelvemonths`, `lastclosetevtotalrevenue.lasttwelvemonths`, `lastclosemarketcaptotalrevenue.lasttwelvemonths`
- **profitability / dividends:** `returnonequity.lasttwelvemonths`, `returnonassets.lasttwelvemonths`, `returnontotalcapital.lasttwelvemonths`, `dividendyield`, `dividendpershare.lasttwelvemonths`, `forward_dividend_yield`, `forward_dividend_per_share`, `consecutive_years_of_dividend_growth_count`
- **leverage:** `totaldebtequity.lasttwelvemonths`, `ltdebtequity.lasttwelvemonths`, `netdebtebitda.lasttwelvemonths`, `totaldebtebitda.lasttwelvemonths`, `ebitinterestexpense.lasttwelvemonths`, `ebitdainterestexpense.lasttwelvemonths`, `lastclosetevebit.lasttwelvemonths`, `lastclosetevebitda.lasttwelvemonths`
- **liquidity:** `currentratio.lasttwelvemonths`, `quickratio.lasttwelvemonths`, `operatingcashflowtocurrentliabilities.lasttwelvemonths`, `altmanzscoreusingtheaveragestockinformationforaperiod.lasttwelvemonths`
- **income_statement:** `totalrevenues.lasttwelvemonths`, `ebitda.lasttwelvemonths`, `ebit.lasttwelvemonths`, `netincomeis.lasttwelvemonths`, `netincomemargin.lasttwelvemonths`, `grossprofitmargin.lasttwelvemonths`, `ebitdamargin.lasttwelvemonths`, `epsgrowth.lasttwelvemonths`, `quarterlyrevenuegrowth.quarterly`, `grossprofit.lasttwelvemonths`, `operatingincome.lasttwelvemonths`
- **balance_sheet:** `totalassets.lasttwelvemonths`, `totaldebt.lasttwelvemonths`, `totalequity.lasttwelvemonths`, `totalcommonequity.lasttwelvemonths`, `totalcurrentassets.lasttwelvemonths`, `totalcurrentliabilities.lasttwelvemonths`, `totalsharesoutstanding`
- **cash_flow:** `cashfromoperations.lasttwelvemonths`, `capitalexpenditure.lasttwelvemonths`, `leveredfreecashflow.lasttwelvemonths`, `unleveredfreecashflow.lasttwelvemonths`
- **esg:** `esg_score`, `governance_score`, `social_score`, `environmental_score`, `highest_controversy`

`FundQuery` has a **tiny** field set — no `netassets`, no `annualreturnnavy5` (those belong
to `ETFQuery`): `exchange`, `categoryname`, `annualreturnnavy1categoryrank`,
`performanceratingoverall`, `initialinvestment`, `riskratingoverall`, `intradaypricechange`,
`eodprice`, `intradayprice`.

`ETFQuery` fields include `fundnetassets`, `ticker`, `annualreturnnavy1`, `annualreturnnavy3`,
`annualreturnnavy5`, `annualreturnnavy1categoryrank`, `annualreportgrossexpenseratio`,
`annualreportnetexpenseratio`, `turnoverratio`, `trailing_3m_return`, `trailing_ytd_return`,
`morningstar_rating`, `morningstar_economic_moat`, `fundfamilyname`, `categoryname`,
`primary_sector`, `region`, `exchange`, plus the shared price/keystats fields.

For exact names/values per class, call `q.valid_fields` and `q.valid_values` rather than
memorising — they are authoritative and version-current.

## Cross-class mistake

```python
# BAD — an equity field on a FundQuery raises ValueError
FundQuery("gt", ["intradaymarketcap", 1_000_000_000])
```
