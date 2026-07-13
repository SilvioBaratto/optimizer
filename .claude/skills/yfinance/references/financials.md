# Financial Statements

All three statements (income, balance sheet, cash flow) are available at yearly, quarterly, and trailing-twelve-month (TTM) granularity. The shorthand properties wrap the `get_*` methods with `freq` preset.

## Income statement

```python
ticker = yf.Ticker("AAPL")

ticker.income_stmt                                    # Annual
ticker.quarterly_income_stmt                          # Quarterly
ticker.get_income_stmt(freq="trailing",               # TTM
                       as_dict=False, pretty=False)
```

## Balance sheet

```python
ticker.balance_sheet                                  # Annual
ticker.quarterly_balance_sheet                        # Quarterly
ticker.get_balance_sheet(freq="quarterly")
```

## Cash flow

```python
ticker.cashflow                                       # Annual
ticker.quarterly_cashflow                             # Quarterly
ticker.get_cashflow(freq="trailing")
```

## Earnings (simplified income view)

```python
ticker.earnings                                       # Annual
ticker.quarterly_earnings                             # Quarterly
```

## SEC filings

```python
ticker.sec_filings                                    # list[dict]
# Each item: form type, filing date, URL, description
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `freq` | `"yearly"` | `"yearly"`, `"quarterly"`, or `"trailing"` (TTM) |
| `as_dict` | `False` | Return a `dict` instead of a `DataFrame` |
| `pretty` | `False` | Human-readable row labels ("Total Revenue" vs `"TotalRevenue"`) |

Row labels are Yahoo's internal keys by default (e.g. `"NetIncome"`, `"OperatingIncome"`). Set `pretty=True` when displaying to humans.
