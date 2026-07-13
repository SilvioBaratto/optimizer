# Analysis Data

Analyst recommendations, price targets, earnings & revenue estimates, ownership, ESG scores. All accessed from a `yf.Ticker` instance.

## Recommendations

```python
ticker = yf.Ticker("AAPL")

ticker.recommendations              # DataFrame — recent recommendations
ticker.recommendations_summary      # DataFrame — Buy/Hold/Sell counts
ticker.upgrades_downgrades          # DataFrame — upgrades and downgrades
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

> **v1.2.2+** — all estimate / history tables now include a **currency** column so you can disambiguate multi-listing equities (e.g. ADRs vs ordinary shares) without scraping `info["financialCurrency"]`.

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
