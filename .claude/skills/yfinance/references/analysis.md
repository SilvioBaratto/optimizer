# Analysis Data

Analyst recommendations, price targets, earnings & revenue estimates, ownership, ESG scores. All accessed from a `yf.Ticker` instance.

> **`as_dict` toggle:** every accessor comes as a property **and** a `get_*()` method. The
> `get_*()` forms take an optional `as_dict` flag (dict instead of DataFrame). Exceptions:
> `get_analyst_price_targets()` and `get_funds_data()` take **no** parameters.
> `analyst_price_targets` already returns a plain dict. None of these take a `proxy` argument.

## Recommendations

```python
ticker = yf.Ticker("AAPL")

ticker.recommendations              # DataFrame — columns: period, strongBuy, buy, hold, sell, strongSell
ticker.recommendations_summary      # DataFrame — ALIAS of .recommendations (identical output)
ticker.upgrades_downgrades          # DataFrame — individual analyst grade changes
```

## Price targets

```python
ticker.analyst_price_targets
# dict with: current, low, high, mean, median
```

## Earnings & revenue estimates

```python
ticker.earnings_estimate            # current/next quarter estimates
ticker.revenue_estimate             # revenue estimates
ticker.earnings_history             # EPS surprise history
ticker.eps_trend                    # EPS trend (now vs 7/30/60/90 days ago)
ticker.eps_revisions                # EPS revision counts (up/down)
ticker.growth_estimates             # growth vs sector / industry
```

**Fixed forward-period indexes** (not a DatetimeIndex, except `earnings_history`):

| Accessor | Index | Columns |
|---|---|---|
| `earnings_estimate` | `0q, +1q, 0y, +1y` | `numberOfAnalysts, avg, low, high, yearAgoEps, growth` |
| `revenue_estimate` | `0q, +1q, 0y, +1y` | `numberOfAnalysts, avg, low, high, yearAgoRevenue, growth` |
| `earnings_history` | `DatetimeIndex` | `epsEstimate, epsActual, epsDifference, surprisePercent` |
| `eps_trend` | `0q, +1q, 0y, +1y` | `current, 7daysAgo, 30daysAgo, 60daysAgo, 90daysAgo` |
| `eps_revisions` | `0q, +1q, 0y, +1y` | `upLast7days, upLast30days, downLast7days, downLast30days` |
| `growth_estimates` | `0q, +1q, 0y, +1y, LTG` | `stockTrend, industryTrend, sectorTrend, indexTrend` |

> **v1.2.2+** — the periodic estimate tables (`earnings_estimate`, `revenue_estimate`, `eps_trend`, `eps_revisions`) include a **currency** column so you can disambiguate multi-listing equities (e.g. ADRs vs ordinary shares) without scraping `info["financialCurrency"]`. `earnings_history` (columns `epsActual, epsEstimate, epsDifference, surprisePercent`) and `growth_estimates` do **not** carry a currency column.

## ESG / Sustainability

```python
ticker.sustainability               # DataFrame — ESG scores
```

## Ownership

```python
ticker.major_holders                # % held by insiders / institutions
ticker.institutional_holders        # top institutional holders
ticker.mutualfund_holders           # top mutual-fund holders
ticker.insider_transactions         # recent insider transactions
ticker.insider_purchases            # insider purchase summary
ticker.insider_roster_holders       # insider roster
```

All return DataFrames.
