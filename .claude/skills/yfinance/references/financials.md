# Financial Statements

All three statements (income, balance sheet, cash flow) are available at yearly, quarterly, and trailing-twelve-month (TTM) granularity. The shorthand properties wrap the `get_*` methods with `freq` preset.

## Income statement

```python
ticker = yf.Ticker("AAPL")

ticker.income_stmt                                    # Annual
ticker.quarterly_income_stmt                          # Quarterly
ticker.ttm_income_stmt                                # TTM (property shortcut)
ticker.get_income_stmt(freq="trailing",               # TTM via method
                       as_dict=False, pretty=False)
```

## Balance sheet

```python
ticker.balance_sheet                                  # Annual
ticker.quarterly_balance_sheet                        # Quarterly
ticker.get_balance_sheet(freq="quarterly")
```

> The docs expose annual `balance_sheet` (+ `quarterly_balance_sheet` via the method); unlike
> income/cash-flow there is **no `ttm_balance_sheet` property** — use `get_balance_sheet(freq="trailing")` if Yahoo has it.

## Cash flow

```python
ticker.cashflow                                       # Annual
ticker.quarterly_cashflow                             # Quarterly
ticker.ttm_cashflow                                   # TTM (property shortcut)
ticker.get_cashflow(freq="trailing")
```

## Earnings — DEPRECATED (use income statement Net Income)

`ticker.earnings`, `ticker.quarterly_earnings`, and `ticker.get_earnings()` are **deprecated
in 1.6.0** and no longer return data. Each emits a `DeprecationWarning`
(`"'Ticker.earnings' is deprecated as not available via API. Look for \"Net Income\" in
Ticker.income_stmt."`) and returns **`None`** — the underlying earnings table is no longer
served by the API. Do not treat them as working data views.

Read the **`Net Income`** row of the income statement instead:

```python
ticker.income_stmt.loc["Net Income"]                  # Annual (property uses pretty=True)
ticker.quarterly_income_stmt.loc["Net Income"]        # Quarterly
ticker.get_income_stmt(freq="yearly", pretty=True).loc["Net Income"]
# NB: default pretty=False keys the row as "NetIncome" (no space)
```

## Earnings dates & calendar

```python
ticker.earnings_dates                                 # DataFrame (past + upcoming)
ticker.get_earnings_dates(limit=12, offset=0)         # paginated; offset added post-1.3.0
ticker.calendar                                       # dict — next earnings/dividend events
```

`calendar` returns a **dict** (events, earnings, dividends), not a DataFrame.

## SEC filings

```python
ticker.sec_filings                                    # list[dict]
ticker.get_sec_filings()                              # method form (no params)
# Each item: form type, filing date, URL, description
```

## Parameters

These apply to the still-functional statement getters (`get_income_stmt` / `get_balance_sheet` / `get_cashflow`):

| Parameter | Applies to | Meaning |
|---|---|---|
| `freq` | statement getters | `"yearly"`, `"quarterly"`, or `"trailing"` (TTM) |
| `as_dict` | all `get_*` statement getters | Return a `dict` instead of a `DataFrame` |
| `pretty` | `get_income_stmt` / `get_balance_sheet` / `get_cashflow` only | Human-readable row labels |
| `limit`, `offset` | `get_earnings_dates` | Row cap (default 12) + pagination offset |

Row labels are Yahoo's internal keys by default (e.g. `"NetIncome"`, `"OperatingIncome"`); set
`pretty=True` when displaying to humans. No `proxy` parameter on any getter — proxy is global.

`get_earnings(as_dict, freq)` still accepts these params but is **deprecated and returns `None`**
(see the Earnings section above), so `freq` / `as_dict` have no effect — it never takes `pretty`.

> **v1.6.0** added missing Balance Sheet keys (e.g. `FixedMaturityInvestments`,
> `EquityInvestments`) — more line items may now appear than in older pulls.
