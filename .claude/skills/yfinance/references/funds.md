# Funds Data (ETFs & Mutual Funds)

When the ticker is a fund, fund-specific data lives under a single accessor:
**`ticker.funds_data`**, which returns a `FundsData` object (`yfinance.scrapers.funds.FundsData`).
The old flat `ticker.fund_*` properties are gone — go through `funds_data`.

```python
ticker = yf.Ticker("SPY")
fd = ticker.funds_data           # FundsData object
```

## FundsData surface

Nine read-only **properties** plus one **method**:

| Member | Kind | Returns | Description |
|---|---|---|---|
| `description` | property | `str` | Fund description |
| `fund_overview` | property | `dict[str, str \| None]` | Family, category, legal type (values may be `None`) |
| `fund_operations` | property | `DataFrame` | Turnover, expense ratio, inception, AUM-style operations |
| `asset_classes` | property | `dict[str, float]` | Stocks / bonds / cash / other allocation |
| `top_holdings` | property | `DataFrame` | Top holdings with weights |
| `equity_holdings` | property | `DataFrame` | P/E, P/B, price/sales of equity holdings |
| `bond_holdings` | property | `DataFrame` | Duration, maturity, credit quality |
| `bond_ratings` | property | `dict[str, float]` | Credit-rating breakdown |
| `sector_weightings` | property | `dict[str, float]` | Sector allocation |
| `quote_type()` | **method** | `str` | Fund quote type (call with parentheses) |

```python
fd.top_holdings                  # DataFrame indexed by holding
fd.sector_weightings             # {"Technology": 0.31, ...}
fd.asset_classes                 # {"stockPosition": 0.99, "bondPosition": 0.0, ...}
fd.bond_holdings                 # DataFrame (bond funds/ETFs)
fd.quote_type()                  # "ETF" / "MUTUALFUND"  -- METHOD, not a property
```

> **Gotchas:**
> - `quote_type` is a **method** — call `fd.quote_type()`. The other nine accessors are
>   properties (no parentheses).
> - Return shapes are mixed: `asset_classes` / `bond_ratings` / `sector_weightings` →
>   `dict[str, float]`; `fund_overview` → `dict[str, str | None]`; `top_holdings` /
>   `equity_holdings` / `bond_holdings` / `fund_operations` → `DataFrame`; `description` → `str`.
> - **There is no fund-performance member** — `fundPerformance` is intentionally not
>   implemented; use `ticker.history()` for performance/returns.
> - Populated only for ETFs / mutual funds; equity tickers yield empty structures.

## Detecting a fund

Guard fund access so equities don't return empty structures:

```python
if ticker.fast_info.quote_type in ("ETF", "MUTUALFUND"):
    holdings = ticker.funds_data.top_holdings
# or:
if ticker.funds_data.quote_type() in ("ETF", "MUTUALFUND"):
    ...
```

## Typical weights

`top_holdings` is the most-used — a DataFrame of holdings with a weight column. Yahoo usually
reports only the top ~10, so weights sum close to but not exactly 1.0. Multiply by NAV to
dollarize.

## Related

For screening funds by size, performance, or category, see `screener.md` — `FundQuery` and
`ETFQuery`.
