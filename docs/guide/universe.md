# Universe Screening

The universe module implements investability screening with hysteresis-based entry/exit thresholds. It filters a raw stock universe down to securities that meet minimum standards of market capitalization, liquidity, price level, listing history, and data availability — the foundation for any systematic investment strategy.

## Overview

Before constructing a portfolio, you need a clean investable universe. Stocks that are too small, too illiquid, or too newly listed create problems: they may be impossible to trade at the quantities needed, they generate excessive transaction costs, or they lack sufficient history for reliable estimation.

The universe module enforces investability standards through 8 screens, each with separate entry and exit thresholds (hysteresis) to reduce turnover at screen boundaries.

### Why Hysteresis?

Without hysteresis, a stock hovering near a threshold (e.g., market cap of $199M vs $201M) would oscillate in and out of the universe each month. Hysteresis sets a lower exit threshold than the entry threshold — once a stock enters the universe, it stays until it drops below a more lenient exit level.

```
Entry threshold:  $200M ─────────────────
                                          │ Stock enters here
Exit threshold:   $150M ─────────────────
                          │ Stock exits here
```

## HysteresisConfig

Each screen uses a `HysteresisConfig` with entry and exit thresholds:

```python
from optimizer.universe import HysteresisConfig

config = HysteresisConfig(
    entry=200_000_000,  # must exceed to enter
    exit_=150_000_000,  # must drop below to exit
)
```

| Field | Type | Description |
|-------|------|-------------|
| `entry` | `float` | Threshold a stock must exceed to enter the universe |
| `exit_` | `float` | Threshold below which a current member is removed |

!!! warning "exit_ must be <= entry"
    The exit threshold must be less than or equal to the entry threshold. This is enforced at construction time.

## InvestabilityScreenConfig

The main configuration holds all 8 screen thresholds plus listing requirements:

```python
from optimizer.universe import InvestabilityScreenConfig

config = InvestabilityScreenConfig(
    market_cap=HysteresisConfig(entry=200_000_000, exit_=150_000_000),
    addv_12m=HysteresisConfig(entry=750_000, exit_=500_000),
    addv_3m=HysteresisConfig(entry=500_000, exit_=350_000),
    trading_frequency=HysteresisConfig(entry=0.95, exit_=0.90),
    price_us=HysteresisConfig(entry=3.0, exit_=2.0),
    price_europe=HysteresisConfig(entry=2.0, exit_=1.5),
    min_trading_history=252,
    min_ipo_seasoning=60,
    min_annual_reports=3,
    min_quarterly_reports=8,
    exchange_region=ExchangeRegion.US,
    mcap_percentile_entry=0.10,
    mcap_percentile_exit=0.075,
)
```

### Screen Details

| Screen | Default Entry | Default Exit | Description |
|--------|--------------|--------------|-------------|
| `market_cap` | $200M | $150M | Free-float market capitalization (USD) |
| `addv_12m` | $750K | $500K | 12-month average daily dollar volume |
| `addv_3m` | $500K | $350K | 3-month average daily dollar volume |
| `trading_frequency` | 95% | 90% | Fraction of trading days with nonzero volume |
| `price_us` | $3.00 | $2.00 | Minimum price for US equities |
| `price_europe` | $2.00 | $1.50 | Minimum price for European equities |
| `mcap_percentile_entry` | 10th | 7.5th | Exchange-relative market cap percentile |

### Non-Hysteresis Requirements

| Requirement | Default | Description |
|-------------|---------|-------------|
| `min_trading_history` | 252 days | Minimum trading days of price history |
| `min_ipo_seasoning` | 60 days | Minimum days since first price observation |
| `min_annual_reports` | 3 | Minimum annual financial statements |
| `min_quarterly_reports` | 8 | Minimum quarterly financial statements |
| `exchange_region` | `US` | Region for price threshold selection |

### Exchange Percentile Screen

The exchange percentile screen adds a relative dimension to the absolute market cap floor. A stock must exceed **both** the absolute market cap threshold **and** the percentile rank within its exchange to enter the universe. This prevents very small stocks from entering when listed on exchanges with low median capitalizations.

## Presets

| Preset | Market Cap Entry | ADDV 12m Entry | Use Case |
|--------|-----------------|----------------|----------|
| `for_developed_markets()` | $200M | $750K | Institutional-grade, strict liquidity |
| `for_broad_universe()` | $100M | $500K | Broader coverage, relaxed thresholds |
| `for_small_cap()` | $50M | $250K | Small-cap research, minimal screens |

### Preset Details

```python
# Strict institutional universe
config = InvestabilityScreenConfig.for_developed_markets()

# Broader coverage
config = InvestabilityScreenConfig.for_broad_universe()
# Relaxes: mcap to $100M, ADDV to $500K, history to 126 days, etc.

# Small-cap
config = InvestabilityScreenConfig.for_small_cap()
# Relaxes: mcap to $50M, ADDV to $250K, price to $1.00, etc.
```

## Screening Functions

### screen_universe (main entry point)

```python
from optimizer.universe import screen_universe, InvestabilityScreenConfig

investable = screen_universe(
    fundamentals=fundamentals_df,
    price_history=price_df,
    volume_history=volume_df,
    financial_statements=statements_df,
    config=InvestabilityScreenConfig.for_developed_markets(),
    current_members=None,  # pd.Index for hysteresis
)

print(f"Investable universe: {len(investable)} stocks")
print(investable)  # pd.Index of passing tickers
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fundamentals` | `pd.DataFrame` | Cross-sectional data indexed by ticker |
| `price_history` | `pd.DataFrame` | Price matrix (dates x tickers) |
| `volume_history` | `pd.DataFrame` | Volume matrix (dates x tickers) |
| `financial_statements` | `pd.DataFrame` or `None` | Statement-level data |
| `config` | `InvestabilityScreenConfig` or `None` | Screening config |
| `current_members` | `pd.Index` or `None` | Current universe for hysteresis |

### Lower-level functions

```python
from optimizer.universe import (
    apply_investability_screens,
    compute_addv,
    compute_listing_age,
    compute_trading_frequency,
    count_financial_statements,
    compute_exchange_mcap_percentile_thresholds,
)

# Compute individual metrics
addv = compute_addv(price_history, volume_history, window=252)
listing_age = compute_listing_age(price_history)
freq = compute_trading_frequency(volume_history, window=252)
```

## Code Examples

### Basic universe screening

```python
from optimizer.universe import screen_universe, InvestabilityScreenConfig

investable = screen_universe(
    fundamentals=fundamentals,
    price_history=prices,
    volume_history=volume,
    config=InvestabilityScreenConfig.for_developed_markets(),
)

# Use investable universe for optimization
selected_prices = prices[prices.columns.intersection(investable)]
```

### Screening with hysteresis

```python
import pandas as pd

# First month: no current members
month1_universe = screen_universe(
    fundamentals=fundamentals_jan,
    price_history=prices_jan,
    volume_history=volume_jan,
    config=InvestabilityScreenConfig.for_developed_markets(),
    current_members=None,
)

# Second month: pass previous universe for hysteresis
month2_universe = screen_universe(
    fundamentals=fundamentals_feb,
    price_history=prices_feb,
    volume_history=volume_feb,
    config=InvestabilityScreenConfig.for_developed_markets(),
    current_members=month1_universe,
)
```

### Full pipeline with universe screening

```python
from optimizer.pipeline import run_full_pipeline_with_selection
from optimizer.universe import InvestabilityScreenConfig

result = run_full_pipeline_with_selection(
    prices=prices,
    optimizer=optimizer,
    fundamentals=fundamentals,
    volume_history=volume,
    investability_config=InvestabilityScreenConfig.for_developed_markets(),
    scoring_config=CompositeScoringConfig(),
    selection_config=SelectionConfig(n_stocks=100),
)
```

## Gotchas and Tips

!!! warning "fundamentals DataFrame must be indexed by ticker"
    The `fundamentals` DataFrame should have tickers as the index, with columns for `market_cap`, `price`, and optionally `exchange` (for percentile screening).

!!! tip "Pass current_members for turnover reduction"
    Without `current_members`, every screening round applies entry thresholds to all stocks. Passing the previous universe enables hysteresis — existing members use the more lenient exit thresholds, reducing unnecessary churn.

!!! tip "Exchange percentile requires 'exchange' column"
    The exchange percentile screen requires an `exchange` column in the `fundamentals` DataFrame. Without it, only the absolute market cap floor is applied.

!!! tip "Combine with factor selection"
    Universe screening and factor selection are complementary: screening ensures investability, while factor selection picks the best stocks from the investable universe. Use `run_full_pipeline_with_selection()` to chain both steps.

## Quick Reference

| Task | Code |
|------|------|
| Developed markets | `InvestabilityScreenConfig.for_developed_markets()` |
| Broad universe | `InvestabilityScreenConfig.for_broad_universe()` |
| Small-cap | `InvestabilityScreenConfig.for_small_cap()` |
| Screen universe | `screen_universe(fundamentals, prices, volume, config=cfg)` |
| With hysteresis | `screen_universe(..., current_members=prev_universe)` |
| Compute ADDV | `compute_addv(prices, volume, window=252)` |
| Listing age | `compute_listing_age(prices)` |
| Trading frequency | `compute_trading_frequency(volume, window=252)` |
