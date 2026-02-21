# Factor Research

Comprehensive guide to the factors module. This module provides a complete factor research pipeline from raw fundamentals to optimization-ready inputs, covering 17 individual factors across 9 factor groups. Every component follows the same pattern: **frozen `@dataclass` config** + **factory function** + **`str, Enum` types**.

---

## Pipeline Overview

The factor pipeline is a sequential workflow where each stage transforms the output of the previous one:

```
fundamentals --> construction --> standardization --> scoring -->
selection --> regime tilts --> validation --> integration
```

| Stage | Input | Output | Key Function |
|-------|-------|--------|--------------|
| Construction | Fundamentals, prices, volume | Raw factor scores (`pd.DataFrame`) | `compute_all_factors()` |
| Standardization | Raw scores, sector labels | Standardized scores + coverage | `standardize_all_factors()` |
| Scoring | Standardized scores, IC history | Composite score per ticker (`pd.Series`) | `compute_composite_score()` |
| Selection | Composite scores | Selected tickers (`pd.Index`) | `select_stocks()` |
| Regime Tilts | Group weights, macro data | Tilted group weights | `apply_regime_tilts()` |
| Validation | Score history, return history | `FactorValidationReport` | `run_factor_validation()` |
| Integration | Scores, premia, weights | Constraints, views, net alpha | `build_factor_exposure_constraints()` |

---

## Factor Taxonomy

### FactorType (17 factors)

Each factor is computed from one of four data sources: fundamental data, price history, volume history, or alternative data (analyst/insider).

| Factor | Enum Value | Group | Data Source | Formula |
|--------|-----------|-------|-------------|---------|
| Book-to-Price | `BOOK_TO_PRICE` | Value | Fundamentals | book_value / market_cap |
| Earnings Yield | `EARNINGS_YIELD` | Value | Fundamentals | net_income / market_cap |
| Cash Flow Yield | `CASH_FLOW_YIELD` | Value | Fundamentals | operating_cashflow / market_cap |
| Sales-to-Price | `SALES_TO_PRICE` | Value | Fundamentals | total_revenue / market_cap |
| EBITDA-to-EV | `EBITDA_TO_EV` | Value | Fundamentals | ebitda / enterprise_value |
| Gross Profitability | `GROSS_PROFITABILITY` | Profitability | Fundamentals | gross_profit / total_assets (Novy-Marx) |
| ROE | `ROE` | Profitability | Fundamentals | net_income / total_equity |
| Operating Margin | `OPERATING_MARGIN` | Profitability | Fundamentals | operating_income / total_revenue |
| Profit Margin | `PROFIT_MARGIN` | Profitability | Fundamentals | net_income / total_revenue |
| Asset Growth | `ASSET_GROWTH` | Investment | Fundamentals | -YoY total asset growth (sign-flipped) |
| Momentum (12-1) | `MOMENTUM_12_1` | Momentum | Prices | 12-month return skipping most recent month |
| Volatility | `VOLATILITY` | Low Risk | Prices | -annualized std (sign-flipped, lower = better) |
| Beta | `BETA` | Low Risk | Prices | -market beta (sign-flipped, lower = better) |
| Amihud Illiquidity | `AMIHUD_ILLIQUIDITY` | Liquidity | Prices + Volume | avg(\|return\| / dollar_volume) |
| Dividend Yield | `DIVIDEND_YIELD` | Dividend | Fundamentals | trailing annual dividend yield |
| Recommendation Change | `RECOMMENDATION_CHANGE` | Sentiment | Analyst data | net upgrades - downgrades |
| Net Insider Buying | `NET_INSIDER_BUYING` | Ownership | Insider data | purchases - sales (shares) |

!!! note "Sign Conventions"
    Volatility, beta, and asset growth are **sign-flipped** so that higher values always indicate a more favorable factor exposure. For volatility and beta, lower raw values are better (less risk), so the sign is negated. For asset growth, conservative investment (lower growth) is favorable per the Hou-Xue-Zhang investment factor, so the sign is negated.

### FactorGroupType (9 groups)

Factors are organized into groups for hierarchical aggregation during composite scoring.

| Group | Enum Value | Weight Tier | Member Factors |
|-------|-----------|-------------|----------------|
| Value | `VALUE` | CORE | BOOK_TO_PRICE, EARNINGS_YIELD, CASH_FLOW_YIELD, SALES_TO_PRICE, EBITDA_TO_EV |
| Profitability | `PROFITABILITY` | CORE | GROSS_PROFITABILITY, ROE, OPERATING_MARGIN, PROFIT_MARGIN |
| Momentum | `MOMENTUM` | CORE | MOMENTUM_12_1 |
| Low Risk | `LOW_RISK` | CORE | VOLATILITY, BETA |
| Investment | `INVESTMENT` | SUPPLEMENTARY | ASSET_GROWTH |
| Liquidity | `LIQUIDITY` | SUPPLEMENTARY | AMIHUD_ILLIQUIDITY |
| Dividend | `DIVIDEND` | SUPPLEMENTARY | DIVIDEND_YIELD |
| Sentiment | `SENTIMENT` | SUPPLEMENTARY | RECOMMENDATION_CHANGE |
| Ownership | `OWNERSHIP` | SUPPLEMENTARY | NET_INSIDER_BUYING |

The `GROUP_WEIGHT_TIER` mapping assigns each group to either `CORE` or `SUPPLEMENTARY`. Core groups receive `core_weight` (default 1.0) and supplementary groups receive `supplementary_weight` (default 0.5) during composite scoring, reflecting the stronger empirical evidence behind core factors.

---

## 1. Construction

Factor construction computes raw factor scores from fundamentals, prices, volume, analyst data, and insider data. All construction respects point-in-time alignment to prevent look-ahead bias.

### FactorConstructionConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `factors` | `tuple[FactorType, ...]` | 8 core factors | Which factors to compute |
| `momentum_lookback` | `int` | `252` | Lookback window for momentum (trading days) |
| `momentum_skip` | `int` | `21` | Recent days to skip for momentum (reversal avoidance) |
| `volatility_lookback` | `int` | `252` | Lookback window for volatility (trading days) |
| `beta_lookback` | `int` | `252` | Lookback window for beta estimation (trading days) |
| `amihud_lookback` | `int` | `252` | Lookback window for Amihud illiquidity (trading days) |
| `publication_lag` | `PublicationLagConfig` | Default lags | Per-source publication lags for PIT correctness |

The default `factors` tuple includes: BOOK_TO_PRICE, EARNINGS_YIELD, GROSS_PROFITABILITY, ROE, ASSET_GROWTH, MOMENTUM_12_1, VOLATILITY, DIVIDEND_YIELD.

### Presets

```python
from optimizer.factors import FactorConstructionConfig

# Core factors with strongest empirical support (8 factors, default)
config = FactorConstructionConfig.for_core_factors()

# All 17 factors
config = FactorConstructionConfig.for_all_factors()
```

### PublicationLagConfig

Differentiated publication lags prevent look-ahead bias by ensuring that data is only used after it would realistically have been available.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `annual_days` | `int` | `90` | Lag for annual financial statements (10-K filing) |
| `quarterly_days` | `int` | `45` | Lag for quarterly financial statements (10-Q filing) |
| `analyst_days` | `int` | `5` | Lag for analyst estimates and recommendations |
| `macro_days` | `int` | `63` | Lag for macroeconomic indicators (release + revision lag) |

```python
from optimizer.factors import PublicationLagConfig

# Uniform lag across all sources
lag = PublicationLagConfig.uniform(days=60)

# Custom per-source lags
lag = PublicationLagConfig(
    annual_days=120,
    quarterly_days=60,
    analyst_days=2,
    macro_days=45,
)
```

!!! note "Backward Compatibility"
    `FactorConstructionConfig` accepts a plain `int` for `publication_lag`, which is automatically converted to `PublicationLagConfig.uniform(int_value)`.

### Point-in-Time Alignment

The `align_to_pit()` function filters time-series data to records that would have been published on or before a given computation date. For each ticker, it returns the most recent available record.

```python
from optimizer.factors import align_to_pit

# Get the most recent fundamentals available as of 2024-06-30,
# accounting for a 90-day publication lag
pit_data = align_to_pit(
    data=fundamentals_df,
    period_date_col="fiscal_period_end",
    as_of_date="2024-06-30",
    lag_days=90,
    ticker_col="ticker",
)
```

A record with period end date `D` is considered published `lag_days` calendar days after `D`. The function returns a cross-sectional view (one row per ticker) containing only the latest record for which `D + lag_days <= as_of_date`.

### Computing Factors

```python
from optimizer.factors import compute_all_factors, compute_factor, FactorConstructionConfig, FactorType

# Compute all configured factors at once
config = FactorConstructionConfig.for_all_factors()
raw_factors = compute_all_factors(
    fundamentals=fundamentals_df,      # Cross-sectional, indexed by ticker
    price_history=price_df,            # Dates x tickers matrix
    volume_history=volume_df,          # Dates x tickers matrix
    analyst_data=analyst_df,           # Optional
    insider_data=insider_df,           # Optional
    config=config,
)
# raw_factors: pd.DataFrame with tickers as rows, factor names as columns

# Compute a single factor
momentum = compute_factor(
    factor_type=FactorType.MOMENTUM_12_1,
    fundamentals=fundamentals_df,
    price_history=price_df,
    config=config,
)
```

!!! warning "Data Requirements"
    - `fundamentals` must be a cross-sectional DataFrame indexed by ticker with columns matching the factor formulas (e.g., `market_cap`, `book_value`, `net_income`).
    - `price_history` must be a dates x tickers DataFrame. Momentum requires at least `momentum_lookback` rows of data.
    - `volume_history` is only required for `AMIHUD_ILLIQUIDITY`. If `None`, that factor returns an empty Series.
    - `analyst_data` is only required for `RECOMMENDATION_CHANGE`. It must contain either a `recommendation_change` column or `strong_buy`/`buy`/`sell`/`strong_sell` counts.
    - `insider_data` is only required for `NET_INSIDER_BUYING`. It must contain `shares`, `ticker`, and optionally `transaction_type` columns.

---

## 2. Standardization

Cross-sectional standardization transforms raw factor scores into comparable, well-behaved distributions suitable for aggregation. The pipeline is: **winsorize** --> **z-score or rank-normal** --> **sector neutralize** --> **optional re-standardization**.

### StandardizationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `StandardizationMethod` | `Z_SCORE` | Z-score or rank-normal standardization |
| `winsorize_lower` | `float` | `0.01` | Lower percentile for winsorization (0-1) |
| `winsorize_upper` | `float` | `0.99` | Upper percentile for winsorization (0-1) |
| `neutralize_sector` | `bool` | `True` | Whether to sector-neutralize scores |
| `neutralize_country` | `bool` | `False` | Whether to country-neutralize scores |
| `re_standardize_after_neutralization` | `bool` | `False` | Re-apply z-score after neutralization |

### StandardizationMethod

| Value | Description | Best For |
|-------|-------------|----------|
| `Z_SCORE` | `(x - mean) / std` | Approximately normal factors (e.g., momentum) |
| `RANK_NORMAL` | `Phi^-1((rank - 0.5) / N)` inverse normal transform | Heavy-tailed distributions (e.g., value ratios) |

### Presets

```python
from optimizer.factors import StandardizationConfig

# Rank-normal for heavy-tailed distributions (value ratios, illiquidity)
config = StandardizationConfig.for_heavy_tailed()

# Z-score for approximately normal factors (momentum, profitability)
config = StandardizationConfig.for_normal()
```

### Standardization Pipeline Steps

#### Step 1: Winsorize

```python
from optimizer.factors import winsorize_cross_section

# Clip extremes at the 1st and 99th percentiles
clipped = winsorize_cross_section(raw_scores, lower_pct=0.01, upper_pct=0.99)
```

#### Step 2: Z-Score or Rank-Normal

```python
from optimizer.factors import z_score_standardize, rank_normal_standardize

# Z-score: mean 0, std 1
z_scored = z_score_standardize(clipped)

# Rank-normal: maps ranks to normal distribution, robust to outliers
rank_normed = rank_normal_standardize(clipped)
```

#### Step 3: Sector Neutralize

```python
from optimizer.factors import neutralize_sector

# Demean scores within each sector
neutral = neutralize_sector(
    scores=z_scored,
    sector_labels=sector_series,          # pd.Series: ticker -> sector
    country_labels=country_series,        # Optional: ticker -> country
)
```

Sector neutralization removes sector-level biases so that the factor captures stock-level characteristics rather than sector membership. When both `neutralize_sector` and `neutralize_country` are enabled, the function creates sector-country interaction groups (e.g., `"Technology_US"`) and demeans within each.

### Full Standardization

```python
from optimizer.factors import standardize_all_factors, StandardizationConfig

config = StandardizationConfig(
    method=StandardizationMethod.RANK_NORMAL,
    neutralize_sector=True,
)

standardized, coverage = standardize_all_factors(
    raw_factors=raw_factors,          # Tickers x factors DataFrame
    config=config,
    sector_labels=sector_series,      # pd.Series: ticker -> sector
)
# standardized: pd.DataFrame of standardized scores
# coverage: pd.DataFrame (boolean) indicating non-NaN values
```

### PCA Orthogonalization

For eliminating multicollinearity among factor scores, `orthogonalize_factors()` projects the scores onto principal components:

```python
from optimizer.factors import orthogonalize_factors

# Retain components explaining >= 95% of variance
orthogonal = orthogonalize_factors(
    factor_scores=standardized,
    method="pca",
    min_variance_explained=0.95,
)
# orthogonal: pd.DataFrame with columns PC1, PC2, ...
```

!!! warning "Orthogonalization Limitations"
    - Only `"pca"` is supported as the method. Other values raise `ConfigurationError`.
    - Requires at least 2 factors and 2 non-NaN observations.
    - Rows with NaN in the input produce NaN in the output but preserve the index.
    - After orthogonalization, factor scores lose their economic interpretation (they become statistical principal components).

---

## 3. Composite Scoring

Composite scoring aggregates standardized factor scores into a single composite score per ticker. The process is hierarchical: factors are first averaged within their group, then group scores are combined using configurable weighting schemes.

### CompositeScoringConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `CompositeMethod` | `EQUAL_WEIGHT` | Scoring method |
| `ic_lookback` | `int` | `36` | Number of periods for IC estimation (IC/ICIR methods) |
| `core_weight` | `float` | `1.0` | Relative weight for CORE factor groups |
| `supplementary_weight` | `float` | `0.5` | Relative weight for SUPPLEMENTARY factor groups |
| `ridge_alpha` | `float` | `1.0` | L2 regularization strength for RIDGE_WEIGHTED |
| `gbt_max_depth` | `int` | `3` | Maximum tree depth for GBT_WEIGHTED |
| `gbt_n_estimators` | `int` | `50` | Number of boosting rounds for GBT_WEIGHTED |

### CompositeMethod

| Method | Description | Requirements | Strengths |
|--------|-------------|-------------|-----------|
| `EQUAL_WEIGHT` | Core/supplementary tiered equal weighting | None | Robust, no estimation error |
| `IC_WEIGHTED` | Trailing IC magnitude as weights | `ic_history` | Adapts to recent predictive power |
| `ICIR_WEIGHTED` | `\|mean(IC) / std(IC)\|` as weights | `ic_history` | Penalizes inconsistent predictors |
| `RIDGE_WEIGHTED` | Ridge regression on historical returns | `training_scores`, `training_returns` | Captures linear factor interactions |
| `GBT_WEIGHTED` | Gradient-boosted trees on historical returns | `training_scores`, `training_returns` | Captures non-linear interactions |

### Presets

```python
from optimizer.factors import CompositeScoringConfig

config = CompositeScoringConfig.for_equal_weight()
config = CompositeScoringConfig.for_ic_weighted()
config = CompositeScoringConfig.for_icir_weighted()
config = CompositeScoringConfig.for_ridge_weighted()
config = CompositeScoringConfig.for_gbt_weighted()
```

### Scoring Workflow

#### Step 1: Compute Group Scores

Group scores are the coverage-weighted mean of factor scores within each group:

```python
from optimizer.factors import compute_group_scores

group_scores = compute_group_scores(standardized, coverage)
# group_scores: pd.DataFrame with tickers as rows, group names as columns
```

#### Step 2: Compute Composite Score

```python
from optimizer.factors import compute_composite_score, CompositeScoringConfig

# Equal-weight composite (simplest)
composite = compute_composite_score(
    standardized_factors=standardized,
    coverage=coverage,
)

# IC-weighted composite (requires IC history)
config = CompositeScoringConfig.for_ic_weighted()
composite = compute_composite_score(
    standardized_factors=standardized,
    coverage=coverage,
    config=config,
    ic_history=ic_df,             # Periods x groups DataFrame of IC values
)

# ML composite (requires training data)
config = CompositeScoringConfig.for_ridge_weighted()
composite = compute_composite_score(
    standardized_factors=standardized,
    coverage=coverage,
    config=config,
    training_scores=historical_scores,      # Historical tickers x factors
    training_returns=forward_returns,       # Forward return per ticker
)
```

!!! warning "Look-Ahead Bias in ML Scoring"
    For `RIDGE_WEIGHTED` and `GBT_WEIGHTED`, the training window must end **strictly before** the prediction date. The caller is responsible for ensuring temporal separation between `training_scores` and the current-period `standardized_factors`.

### IC-Weighted Scoring Details

The IC-weighted method uses trailing Information Coefficient (Spearman rank correlation between factor scores and forward returns) to dynamically weight factor groups:

1. Compute the mean IC over the trailing `ic_lookback` periods for each group
2. Clamp negative ICs to zero (negative-IC groups should not contribute positively)
3. Multiply by the core/supplementary tier weight
4. Normalize to sum to 1

If all groups have negative or zero IC, the method falls back to equal-weight scoring.

### ICIR-Weighted Scoring Details

ICIR (Information Coefficient Information Ratio) penalizes factors that are inconsistent predictors:

```
ICIR = |mean(IC) / std(IC)|
```

A factor with high mean IC but also high IC volatility receives a lower weight than a factor with moderate but stable IC. Falls back to equal-weight when all groups have ICIR = 0.

### ML Scoring Details

Both ML methods train a model on historical `(factor_scores, forward_returns)` pairs and predict on the current period. The raw predictions are standardized to zero mean and unit variance.

```python
from optimizer.factors import fit_ridge_composite, fit_gbt_composite, predict_composite_scores

# Fit ridge regression
model = fit_ridge_composite(
    scores=historical_scores,
    forward_returns=forward_returns,
    alpha=1.0,
)

# Or fit gradient-boosted trees
model = fit_gbt_composite(
    scores=historical_scores,
    forward_returns=forward_returns,
    max_depth=3,
    n_estimators=50,
)

# Predict on current-period scores
composite = predict_composite_scores(model, current_scores)
```

The `FittedMLModel` type alias covers both `RidgeCV` and `GradientBoostingRegressor`.

### Regime-Tilted Scoring

When regime tilts are applied, group weights can be passed through to the scoring functions:

```python
from optimizer.factors import (
    classify_regime,
    apply_regime_tilts,
    compute_composite_score,
    RegimeTiltConfig,
    FactorGroupType,
)

# Classify regime
regime = classify_regime(macro_data)

# Compute tilted weights
base_weights = {
    FactorGroupType.VALUE: 1.0,
    FactorGroupType.MOMENTUM: 1.0,
    FactorGroupType.LOW_RISK: 1.0,
    FactorGroupType.PROFITABILITY: 1.0,
}
tilted = apply_regime_tilts(
    base_weights, regime, RegimeTiltConfig.for_moderate_tilts()
)

# Convert to string keys for compute_composite_score
group_weights = {g.value: w for g, w in tilted.items()}
composite = compute_composite_score(
    standardized, coverage, group_weights=group_weights,
)
```

---

## 4. Stock Selection

Stock selection filters the scored universe down to a target number of stocks, with mechanisms to reduce unnecessary turnover.

### SelectionConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `SelectionMethod` | `FIXED_COUNT` | Fixed-count or quantile-based selection |
| `target_count` | `int` | `100` | Number of stocks to select (for FIXED_COUNT) |
| `target_quantile` | `float` | `0.8` | Quantile threshold for entry (for QUANTILE, 0-1) |
| `exit_quantile` | `float` | `0.7` | Exit quantile for hysteresis (for QUANTILE) |
| `buffer_fraction` | `float` | `0.1` | Buffer zone fraction around selection boundary |
| `sector_balance` | `bool` | `True` | Whether to enforce sector-proportional representation |
| `sector_tolerance` | `float` | `0.03` | Maximum deviation from parent universe sector weights |

### SelectionMethod

| Method | Description |
|--------|-------------|
| `FIXED_COUNT` | Select top N stocks by composite score |
| `QUANTILE` | Select all stocks above a quantile threshold |

### Presets

```python
from optimizer.factors import SelectionConfig

# Top 100 stocks (default)
config = SelectionConfig.for_top_100()

# Top quintile (top 20%)
config = SelectionConfig.for_top_quintile()

# Concentrated portfolio of top 30
config = SelectionConfig.for_concentrated()
```

### Buffer-Zone Hysteresis

Hysteresis prevents excessive turnover by creating a buffer zone around the selection boundary. Current members within the buffer are retained even if they would not qualify as new entrants.

**Fixed-Count hysteresis**: The top `target_count` stocks are always included. Current members ranking between `target_count` and `target_count + buffer_fraction * target_count` are retained.

```python
from optimizer.factors import select_fixed_count

selected = select_fixed_count(
    scores=composite_scores,
    target_count=100,
    buffer_fraction=0.1,                 # Buffer of 10 stocks
    current_members=previous_selection,   # pd.Index of previously selected tickers
)
```

**Quantile hysteresis**: New stocks must score above `target_quantile` (e.g., 80th percentile). Existing members survive as long as they stay above `exit_quantile` (e.g., 70th percentile).

```python
from optimizer.factors import select_quantile

selected = select_quantile(
    scores=composite_scores,
    target_quantile=0.8,                 # Entry threshold
    exit_quantile=0.7,                   # Exit threshold (lower = more sticky)
    current_members=previous_selection,
)
```

### Sector Balancing

When `sector_balance=True`, the selection is adjusted so that no sector is over- or under-represented relative to the parent universe by more than `sector_tolerance`:

```python
from optimizer.factors import apply_sector_balance

balanced = apply_sector_balance(
    selected=initial_selection,
    scores=composite_scores,
    sector_labels=sector_series,
    parent_universe=full_universe,
    tolerance=0.03,
)
```

Under-represented sectors gain their highest-scoring non-selected stocks. Over-represented sectors lose their lowest-scoring selected stocks.

### Full Selection Pipeline

```python
from optimizer.factors import select_stocks, SelectionConfig

config = SelectionConfig(
    method=SelectionMethod.FIXED_COUNT,
    target_count=100,
    buffer_fraction=0.1,
    sector_balance=True,
    sector_tolerance=0.03,
)

# Without turnover tracking
selected = select_stocks(
    scores=composite_scores,
    config=config,
    current_members=previous_selection,
    sector_labels=sector_series,
    parent_universe=full_universe,
)

# With turnover tracking
selected, turnover = select_stocks(
    scores=composite_scores,
    config=config,
    current_members=previous_selection,
    sector_labels=sector_series,
    parent_universe=full_universe,
    return_turnover=True,
)
```

### Selection Turnover

```python
from optimizer.factors import compute_selection_turnover

turnover = compute_selection_turnover(
    current=previous_selection,
    new=new_selection,
    universe=full_universe,
)
# turnover = len(added | removed) / len(universe)
```

---

## 5. Regime Tilts

Regime tilts apply macro-economic regime-conditional adjustments to factor group weights. The system classifies the current macro environment and applies multiplicative tilts to emphasize factors with stronger expected performance in that regime.

### MacroRegime

| Regime | Description | Factor Emphasis |
|--------|-------------|-----------------|
| `EXPANSION` | GDP above trend, accelerating | Momentum (1.2x), reduce Value/Low Risk |
| `SLOWDOWN` | GDP above trend, decelerating | Low Risk (1.3x), Dividend (1.2x), reduce Momentum |
| `RECESSION` | GDP below trend, decelerating | Low Risk (1.5x), Profitability (1.3x), Value (1.2x), reduce Momentum |
| `RECOVERY` | GDP below trend, accelerating | Value (1.3x), Momentum (1.2x), reduce Low Risk |

### RegimeTiltConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable` | `bool` | `False` | Whether to apply regime tilts |
| `expansion_tilts` | `tuple[tuple[str, float], ...]` | See defaults | Group tilts during expansion |
| `slowdown_tilts` | `tuple[tuple[str, float], ...]` | See defaults | Group tilts during slowdown |
| `recession_tilts` | `tuple[tuple[str, float], ...]` | See defaults | Group tilts during recession |
| `recovery_tilts` | `tuple[tuple[str, float], ...]` | See defaults | Group tilts during recovery |

Tilts are stored as tuples of `(group_name, tilt_factor)` for frozen-dataclass compatibility.

### Presets

```python
from optimizer.factors import RegimeTiltConfig

# Enable moderate tilts (uses the built-in tilt tables)
config = RegimeTiltConfig.for_moderate_tilts()

# Disable tilts (default)
config = RegimeTiltConfig.for_no_tilts()
```

### Regime Classification

```python
from optimizer.factors import classify_regime

regime = classify_regime(macro_data)
# macro_data: pd.DataFrame with date index and columns like
# 'gdp_growth', 'yield_spread', 'unemployment_rate'
```

The classification heuristic uses GDP growth as the primary signal:

1. If `gdp_growth` is available with 2+ observations:
    - Rising unemployment with positive GDP overrides to `SLOWDOWN`
    - Current > trend and current > previous --> `EXPANSION`
    - Current > trend and current <= previous --> `SLOWDOWN`
    - Current <= trend and current <= previous --> `RECESSION`
    - Current <= trend and current > previous --> `RECOVERY`
2. Fallback: `yield_spread` (10Y-2Y Treasury spread):
    - \> 1.0 --> `EXPANSION`
    - \> 0.0 --> `SLOWDOWN`
    - \> -0.5 --> `RECOVERY`
    - <= -0.5 --> `RECESSION`
3. Default: `EXPANSION`

### Applying Tilts

```python
from optimizer.factors import apply_regime_tilts, get_regime_tilts, FactorGroupType, MacroRegime

# Get the raw tilt dictionary for a regime
tilts = get_regime_tilts(MacroRegime.RECESSION)
# {FactorGroupType.LOW_RISK: 1.5, FactorGroupType.PROFITABILITY: 1.3, ...}
# Groups not listed receive a default tilt of 1.0

# Apply tilts to base group weights (with re-normalization)
base_weights = {
    FactorGroupType.VALUE: 1.0,
    FactorGroupType.PROFITABILITY: 1.0,
    FactorGroupType.MOMENTUM: 1.0,
    FactorGroupType.LOW_RISK: 1.0,
}
tilted = apply_regime_tilts(
    group_weights=base_weights,
    regime=MacroRegime.RECESSION,
    config=RegimeTiltConfig.for_moderate_tilts(),
)
```

!!! note "Re-Normalization"
    After applying multiplicative tilts, the total weight is re-normalized to preserve the original total. This ensures that tilts only change the relative allocation between groups, not the overall magnitude.

!!! warning "Disabled by Default"
    `RegimeTiltConfig.enable` defaults to `False`. When `enable=False`, `apply_regime_tilts()` returns a copy of the original weights unchanged. You must explicitly use `RegimeTiltConfig.for_moderate_tilts()` or set `enable=True`.

---

## 6. Validation

Factor validation assesses the statistical significance and economic value of factors before deploying them in production.

### FactorValidationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `newey_west_lags` | `int` | `6` | Number of lags for Newey-West HAC standard errors |
| `t_stat_threshold` | `float` | `2.0` | Minimum absolute t-statistic for significance |
| `fdr_alpha` | `float` | `0.05` | False discovery rate alpha level |
| `n_quantiles` | `int` | `5` | Number of quantiles for spread analysis |
| `fmp_top_pct` | `float` | `0.2` | Top percentile for factor-mimicking portfolios |
| `fmp_bottom_pct` | `float` | `0.2` | Bottom percentile for factor-mimicking portfolios |

### Presets

```python
from optimizer.factors import FactorValidationConfig

# Standard validation
config = FactorValidationConfig.for_standard()

# Strict validation (t > 3.0, FDR alpha = 1%)
config = FactorValidationConfig.for_strict()
```

### Information Coefficient (IC) Analysis

The Information Coefficient is the Spearman rank correlation between factor scores and subsequent forward returns. A positive IC indicates that higher factor scores predict higher returns.

```python
from optimizer.factors import compute_monthly_ic, compute_ic_series, compute_icir, compute_ic_stats

# Single-period IC
ic = compute_monthly_ic(factor_scores, forward_returns)

# IC time series (one IC per date)
ic_series = compute_ic_series(
    factor_scores_history=scores_df,    # Dates x tickers matrix
    returns_history=returns_df,         # Dates x tickers matrix
    factor_name="book_to_price",
)

# ICIR: mean(IC) / std(IC)
icir = compute_icir(ic_series)

# Full IC statistics with Newey-West inference
stats = compute_ic_stats(ic_series, lags=5)
# stats.mean, stats.variance_nw, stats.t_stat_nw, stats.p_value, stats.icir
```

### Newey-West t-Statistic

The Newey-West HAC (heteroscedasticity and autocorrelation consistent) estimator provides robust standard errors for IC significance testing, accounting for the serial correlation inherent in overlapping IC measurements.

```python
from optimizer.factors import compute_newey_west_tstat

t_stat, p_value = compute_newey_west_tstat(ic_series, n_lags=6)
```

The variance estimator uses Bartlett kernel weights:

```
Var_NW = gamma_0 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * gamma_j
```

where `gamma_j = E[(IC_t - mean)(IC_{t-j} - mean)]`.

### Multiple Testing Correction

When testing multiple factors simultaneously, p-values must be corrected for multiple comparisons.

```python
from optimizer.factors import correct_pvalues, benjamini_hochberg
import numpy as np

# Holm-Bonferroni (FWER) + Benjamini-Hochberg (FDR)
raw_pvalues = np.array([0.01, 0.04, 0.03, 0.15, 0.02])
corrected = correct_pvalues(raw_pvalues, alpha=0.05)
# corrected.holm: Holm-Bonferroni adjusted p-values (controls family-wise error rate)
# corrected.bh: Benjamini-Hochberg adjusted p-values (controls false discovery rate)

# Standalone BH correction (returns boolean series)
significant = benjamini_hochberg(p_values_series, alpha=0.05)
```

### Variance Inflation Factor (VIF)

VIF detects multicollinearity among factors. A VIF above 10 indicates that the factor's variance is largely explained by other factors.

```python
from optimizer.factors import compute_vif

vif = compute_vif(standardized_factors)
# pd.Series: VIF per factor (>= 1.0 by construction)
high_vif = vif[vif > 10]  # Candidates for removal or merging
```

### Quantile Spread Analysis

Quantile spreads measure the economic value of a factor by comparing returns across factor-sorted portfolios.

```python
from optimizer.factors import compute_quantile_spread

# Single-period spread: top quantile return - bottom quantile return
spread = compute_quantile_spread(
    factor_scores=scores_series,
    forward_returns=returns_series,
    n_quantiles=5,
)
```

### Factor Spread Benchmarks

The module includes annualized long-short quintile spread benchmarks derived from academic literature (Fama-French, AQR, Novy-Marx):

| Group | Low | High |
|-------|-----|------|
| value | 2% | 6% |
| profitability | 2% | 5% |
| investment | 1% | 4% |
| momentum | 4% | 10% |
| low_risk | 1% | 4% |
| liquidity | 1% | 3% |
| dividend | 1% | 3% |
| sentiment | 0.5% | 2% |
| ownership | 0.5% | 2% |

### Universe-Level Validation

`validate_factor_universe()` validates all factors simultaneously with Newey-West inference and multiple testing correction:

```python
from optimizer.factors import validate_factor_universe

summary = validate_factor_universe(
    ic_matrix=ic_matrix,     # Dates x factors matrix of IC values
    lags=5,
    alpha=0.05,
)
# Returns pd.DataFrame with columns:
# ic_mean, icir, t_stat_nw, p_value_raw, p_value_holm, p_value_bh,
# significant_holm, significant_bh
```

### Full Validation Report

```python
from optimizer.factors import run_factor_validation, FactorValidationConfig

report = run_factor_validation(
    factor_scores_history={
        "book_to_price": scores_bp_df,    # Dates x tickers per factor
        "momentum_12_1": scores_mom_df,
    },
    returns_history=returns_df,            # Dates x tickers forward returns
    config=FactorValidationConfig.for_standard(),
)

# report.ic_results: list[ICResult] with per-factor IC, t-stat, p-value
# report.quantile_spreads: list[QuantileSpreadResult] with per-factor spreads
# report.significant_factors: list[str] (BH FDR-significant factors)
# report.significant_factors_holm: list[str] (Holm FWER-significant factors)
```

### Out-of-Sample Validation

Rolling block or combinatorial purged cross-validation (CPCV) for out-of-sample factor assessment:

```python
from optimizer.factors import run_factor_oos_validation, FactorOOSConfig

# Rolling block OOS
config = FactorOOSConfig(
    train_months=36,     # 3-year training window
    val_months=12,       # 1-year validation window
    step_months=6,       # Roll forward 6 months per fold
)

result = run_factor_oos_validation(
    scores=panel_scores,     # MultiIndex (date, ticker) x factors
    returns=panel_returns,   # MultiIndex (date, ticker) x return column
    config=config,
)

# result.per_fold_ic: n_folds x factors DataFrame of mean IC per fold
# result.per_fold_spread: n_folds x factors DataFrame of mean spread per fold
# result.mean_oos_ic: pd.Series of mean OOS IC per factor
# result.mean_oos_icir: pd.Series of OOS ICIR per factor
# result.n_folds: int
```

#### FactorOOSConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_months` | `int` | `36` | Length of the training window in months |
| `val_months` | `int` | `12` | Length of the validation window in months |
| `step_months` | `int` | `6` | Number of months to roll forward between folds |

#### CPCV Mode

When a `CPCVConfig` is provided, CPCV is used instead of rolling blocks. CPCV generates all `C(n_folds, n_test_folds)` combinations with purging and embargo at train-test boundaries:

```python
from optimizer.validation import CPCVConfig

cpcv = CPCVConfig(
    n_folds=10,
    n_test_folds=2,
    purged_size=3,
    embargo_size=5,
)

result = run_factor_oos_validation(
    scores=panel_scores,
    returns=panel_returns,
    cpcv_config=cpcv,    # Overrides config when provided
)
```

!!! note "Input Format for OOS Validation"
    `scores` must have a two-level row MultiIndex `(date, ticker)` with one column per factor. `returns` must have the same MultiIndex with a single return column.

---

## 7. Diagnostics

Diagnostic tools for assessing factor quality, redundancy, and data integrity.

### PCA Analysis

```python
from optimizer.factors import compute_factor_pca

pca_result = compute_factor_pca(
    scores=standardized_factors,
    n_components=None,               # Keep all components
)

# pca_result.explained_variance_ratio: ndarray of variance per component
# pca_result.loadings: pd.DataFrame (factors x PCs) -- PCA loading matrix
# pca_result.n_components_95pct: smallest n components for >= 95% variance
```

### Redundant Factor Detection

```python
from optimizer.factors import flag_redundant_factors

redundant = flag_redundant_factors(
    scores=standardized_factors,
    vif_threshold=10.0,              # VIF cutoff (5 = conservative, 10 = standard)
)
# redundant: list[str] of factor names with VIF > threshold
```

### Survivorship Bias Check

```python
from optimizer.factors import check_survivorship_bias

has_bias = check_survivorship_bias(
    returns=returns_df,
    final_periods=12,                # Inspect last 12 periods
    zero_threshold=1e-10,
)
# True if no assets have near-zero returns in the tail (potential survivorship bias)
```

The heuristic is simple: if **no** asset appears to have stopped trading (near-zero returns in the final periods), the dataset may exclude delisted or failed companies. A `UserWarning` is emitted when survivorship bias is suspected.

---

## 8. Mimicking Portfolios

Factor-mimicking portfolios are long-short portfolios designed to isolate pure factor exposure. They are used for factor premium estimation, validation, and cross-factor correlation analysis.

### Building Mimicking Portfolios

```python
from optimizer.factors import build_factor_mimicking_portfolios

fmp_returns = build_factor_mimicking_portfolios(
    scores=scores_df,           # Dates x assets matrix for one factor
    returns=returns_df,         # Dates x assets return matrix
    quantile=0.30,              # 30% in each leg
    weighting="equal",          # "equal" or "value"
)
# fmp_returns: pd.DataFrame with column "factor_return"
```

For each date, the top `quantile` fraction of assets (by factor score) are held long and the bottom `quantile` fraction are held short. The function processes **one factor at a time**. For multiple factors, call once per factor and concatenate:

```python
import pandas as pd
from optimizer.factors import build_factor_mimicking_portfolios

factor_returns = pd.concat([
    build_factor_mimicking_portfolios(scores_value, returns)
        .rename(columns={"factor_return": "value"}),
    build_factor_mimicking_portfolios(scores_mom, returns)
        .rename(columns={"factor_return": "momentum"}),
], axis=1)
```

### Beta-Neutral Mimicking Portfolios

When `beta_neutral=True`, the hedge ratio adjusts the short-leg weight to approximate zero market beta exposure:

```python
fmp_returns = build_factor_mimicking_portfolios(
    scores=scores_df,
    returns=returns_df,
    quantile=0.30,
    beta_neutral=True,
    market_returns=market_series,    # Required when beta_neutral=True
)
```

The hedge ratio is computed as `beta_long / beta_short`, where each beta is the OLS regression coefficient of the leg returns against market returns.

### Quintile Spread Analysis

```python
from optimizer.factors import compute_quintile_spread

result = compute_quintile_spread(
    scores=scores_df,           # Dates x assets factor scores
    returns=returns_df,         # Dates x assets returns
    n_quantiles=5,
)

# result.quintile_returns: pd.DataFrame (Dates x Q1..Q5) -- per-bucket returns
# result.spread_returns: pd.Series (Q5 - Q1) -- long-short spread
# result.annualised_mean: mean daily spread * 252
# result.t_stat: mean / (std / sqrt(T))
# result.sharpe: mean * sqrt(252) / std
```

Assets are ranked by factor score at each date and split into `n_quantiles` equal-count buckets. Q1 = lowest scores (short), Qn = highest scores (long).

### Cross-Factor Correlation

```python
from optimizer.factors import compute_cross_factor_correlation

corr_matrix = compute_cross_factor_correlation(factor_returns)
# pd.DataFrame: factors x factors Pearson correlation matrix
```

---

## 9. Integration with Optimization

The integration layer bridges factor scores and analytics to portfolio optimization inputs: expected returns, exposure constraints, Black-Litterman views, and net alpha.

### FactorIntegrationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `risk_free_rate` | `float` | `0.04` | Annual risk-free rate |
| `market_risk_premium` | `float` | `0.05` | Annual equity risk premium |
| `use_black_litterman` | `bool` | `False` | Whether to generate BL views from factor scores |
| `exposure_lower_bound` | `float` | `-0.5` | Lower bound for factor exposure constraints |
| `exposure_upper_bound` | `float` | `0.5` | Upper bound for factor exposure constraints |

### Presets

```python
from optimizer.factors import FactorIntegrationConfig

# Direct factor score to expected return mapping
config = FactorIntegrationConfig.for_linear_mapping()

# Factor-based Black-Litterman views
config = FactorIntegrationConfig.for_black_litterman()
```

### Factor Scores to Expected Returns

Convert factor Z-scores to expected returns via a linear model:

```
E[r_i] = r_f + lambda_mkt * beta_i + sum_g lambda_g * z_{i,g}
```

```python
from optimizer.factors import factor_scores_to_expected_returns

expected_returns = factor_scores_to_expected_returns(
    scores=group_scores,           # Assets x factor-groups DataFrame
    betas=market_betas,            # pd.Series of CAPM beta per asset
    factor_premiums={
        "market": 0.05,
        "value": 0.03,
        "momentum": 0.04,
        "profitability": 0.02,
    },
    risk_free_rate=0.02,
)
```

Assets missing from `betas` are treated as having a beta of 1.0. The `"market"` key provides the market premium; all other keys are matched against columns in `scores`.

### Factor Exposure Constraints

Build linear inequality constraints that limit portfolio factor exposure, ready for `MeanRisk`:

```python
from optimizer.factors import build_factor_exposure_constraints

# Uniform bounds: all factors constrained to [-0.5, 0.5]
constraints = build_factor_exposure_constraints(
    factor_scores=standardized,
    bounds=(-0.5, 0.5),
)

# Per-factor bounds
constraints = build_factor_exposure_constraints(
    factor_scores=standardized,
    bounds={
        "book_to_price": (-0.3, 0.3),
        "momentum_12_1": (-0.5, 0.5),
        "volatility": (-0.2, 0.2),
    },
)

# Use with MeanRisk optimizer
from optimizer.optimization import MeanRiskConfig, build_mean_risk

model = build_mean_risk(
    MeanRiskConfig.for_max_sharpe(),
    factor_exposure_constraints=constraints,
)
```

The constraint encodes `lb_g <= sum_i w_i * z_{i,g} <= ub_g` as the pair `left_inequality @ w <= right_inequality` (two rows per factor: one for the lower bound, one for the upper bound).

!!! warning "Feasibility Warning"
    `build_factor_exposure_constraints()` checks whether the equal-weight portfolio exposure falls within the bounds for each factor. If not, a `UserWarning` is emitted indicating the constraint may be infeasible. Tighten bounds carefully.

### Black-Litterman Views from Factors

Generate relative views for Black-Litterman based on factor scores and factor premia:

```python
from optimizer.factors import build_factor_bl_views

views, confidences = build_factor_bl_views(
    factor_scores=standardized,
    factor_premia={"book_to_price": 0.03, "momentum_12_1": 0.06},
    selected_tickers=selected,
)
# views: list[tuple[str, ...]] -- top-quartile vs bottom-quartile tickers
# confidences: list[float] -- |premium| as confidence
```

For each factor, the function identifies top-quartile and bottom-quartile assets and generates a relative view that the top outperforms the bottom by the factor premium.

### Factor Premia Estimation

Estimate annualized factor premia from long-short factor-mimicking portfolio returns:

```python
from optimizer.factors import estimate_factor_premia

premia = estimate_factor_premia(factor_mimicking_returns)
# dict[str, float]: annualized premium per factor (mean_daily * 252)
```

### Net Alpha

Compute factor alpha after deducting turnover-based transaction costs:

```python
from optimizer.factors import compute_net_alpha

result = compute_net_alpha(
    ic_series=ic_series,              # Time series of IC values
    weights_history=weights_df,       # Dates x assets weight matrix
    cost_bps=10.0,                    # Round-trip cost in basis points
    annualisation=252,
)

# result.gross_alpha: mean(IC) * sqrt(252)
# result.avg_turnover: mean one-way turnover across rebalancing dates
# result.total_cost: avg_turnover * cost_bps / 10_000
# result.net_alpha: gross_alpha - total_cost
# result.net_icir: net_alpha / (std(IC) * sqrt(252))
```

!!! note "Net ICIR"
    `net_icir` divides the net alpha by the annualized IC volatility. A net ICIR above 0.5 is generally considered attractive for a factor strategy; above 1.0 is exceptional.

### Gross Alpha Recovery

```python
from optimizer.factors import compute_gross_alpha

gross = compute_gross_alpha(
    net_alpha=0.03,
    avg_turnover=0.50,
    cost_bps=10.0,
)
# gross = net_alpha + avg_turnover * cost_bps / 10_000
```

---

## End-to-End Example

A complete workflow from raw data to optimized portfolio:

```python
import pandas as pd
from optimizer.factors import (
    FactorConstructionConfig,
    StandardizationConfig,
    CompositeScoringConfig,
    SelectionConfig,
    RegimeTiltConfig,
    FactorValidationConfig,
    FactorIntegrationConfig,
    compute_all_factors,
    standardize_all_factors,
    compute_composite_score,
    select_stocks,
    classify_regime,
    apply_regime_tilts,
    run_factor_validation,
    build_factor_exposure_constraints,
    FactorGroupType,
)

# 1. Construction: compute raw factor scores
construction_config = FactorConstructionConfig.for_all_factors()
raw_factors = compute_all_factors(
    fundamentals=fundamentals_df,
    price_history=price_df,
    volume_history=volume_df,
    analyst_data=analyst_df,
    config=construction_config,
)

# 2. Standardization: winsorize, z-score, sector-neutralize
std_config = StandardizationConfig(neutralize_sector=True)
standardized, coverage = standardize_all_factors(
    raw_factors, config=std_config, sector_labels=sectors,
)

# 3. Regime tilts (optional)
regime = classify_regime(macro_data)
base_weights = {g: 1.0 for g in FactorGroupType}
tilted = apply_regime_tilts(
    base_weights, regime, RegimeTiltConfig.for_moderate_tilts(),
)
group_weights = {g.value: w for g, w in tilted.items()}

# 4. Composite scoring
scoring_config = CompositeScoringConfig.for_equal_weight()
composite = compute_composite_score(
    standardized, coverage, config=scoring_config,
    group_weights=group_weights,
)

# 5. Stock selection
selection_config = SelectionConfig.for_top_100()
selected = select_stocks(
    scores=composite,
    config=selection_config,
    sector_labels=sectors,
    parent_universe=standardized.index,
)

# 6. Validation (on historical data)
report = run_factor_validation(
    factor_scores_history=historical_scores,
    returns_history=historical_returns,
    config=FactorValidationConfig.for_standard(),
)
print(f"Significant factors (BH): {report.significant_factors}")

# 7. Integration: build constraints for optimizer
constraints = build_factor_exposure_constraints(
    factor_scores=standardized.loc[selected],
    bounds=(-0.5, 0.5),
)

# 8. Pass to optimizer
from optimizer.optimization import MeanRiskConfig, build_mean_risk

model = build_mean_risk(
    MeanRiskConfig.for_max_sharpe(),
    factor_exposure_constraints=constraints,
)
# model.fit(returns_selected) ...
```

---

## Gotchas and Tips

1. **Sign conventions matter.** Volatility, beta, and asset growth are sign-flipped internally so that higher values always indicate a more favorable exposure. Do not negate these yourself before passing to the pipeline.

2. **Point-in-time alignment is critical.** Always use `align_to_pit()` with appropriate publication lags when constructing factors from fundamental data. Using `PublicationLagConfig` with source-specific lags is more accurate than a single uniform lag.

3. **Coverage-weighted group aggregation.** `compute_group_scores()` uses a coverage-weighted mean, not a simple mean. Factors with NaN scores do not drag down the group score for tickers where they are missing -- they are simply excluded from the average.

4. **IC-weighted fallback.** When all factor groups have negative or zero IC, both `compute_ic_weighted_composite()` and `compute_icir_weighted_composite()` fall back to equal-weight scoring rather than producing degenerate weights.

5. **ML scoring requires temporal separation.** The `training_scores` and `training_returns` for RIDGE_WEIGHTED and GBT_WEIGHTED must not overlap with the current prediction period. The caller is responsible for this split.

6. **Hysteresis reduces turnover.** Both `select_fixed_count()` and `select_quantile()` accept `current_members` to implement buffer-zone hysteresis. Without passing previous members, every rebalancing produces a fresh selection from scratch, potentially causing excessive turnover.

7. **Sector balance adjustments are post-hoc.** `apply_sector_balance()` runs after the initial selection and may add or remove stocks to meet tolerance constraints. The final count may differ slightly from `target_count`.

8. **Regime tilts are disabled by default.** `RegimeTiltConfig.enable` is `False`. When disabled, `apply_regime_tilts()` returns the original weights unchanged, even if tilt tables are defined in the config.

9. **OOS validation input format.** `run_factor_oos_validation()` expects a two-level MultiIndex `(date, ticker)` on both `scores` and `returns`. This is different from other functions that use separate dates-x-tickers DataFrames.

10. **Factor exposure constraints require matching tickers.** The tickers in `factor_scores` passed to `build_factor_exposure_constraints()` must match the assets used in the optimizer `fit()` call. Mismatches produce incorrect constraint matrices.
