# portopt

[![CI](https://github.com/SilvioBaratto/optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/SilvioBaratto/optimizer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/portopt)](https://pypi.org/project/portopt/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![codecov](https://codecov.io/gh/SilvioBaratto/optimizer/branch/main/graph/badge.svg)](https://codecov.io/gh/SilvioBaratto/optimizer)
[![Docs](https://img.shields.io/badge/docs-silviobaratto.github.io%2Foptimizer-blue)](https://silviobaratto.github.io/optimizer)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)

Quantitative portfolio construction and optimization platform built on [skfolio](https://skfolio.org/) and scikit-learn. Every component follows the **frozen-config + factory** pattern and composes in standard sklearn pipelines.

## Installation

```bash
pip install portopt
```

For development (tests, linting, type checking, docs):

```bash
git clone https://github.com/SilvioBaratto/optimizer.git
cd optimizer
pip install -e ".[dev]"
```

## Quick Start

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

# Build optimizer from frozen config
optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())

# Run end-to-end: prices -> returns -> preprocess -> optimize -> backtest
result = run_full_pipeline(
    prices=price_df,
    optimizer=optimizer,
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)

print(result.weights)     # pd.Series of asset weights
print(result.summary)     # dict with Sharpe, max drawdown, etc.
print(result.backtest)    # out-of-sample MultiPeriodPortfolio
```

## Features

### Pipeline

Single entry point from raw prices to validated, rebalanced portfolio weights. Handles price-to-return conversion, preprocessing, pre-selection, optimization, cross-validation, and backtesting internally.

```
prices -> returns -> [preprocess -> pre-select -> optimize] -> backtest -> weights
                      \________ sklearn Pipeline ________/
```

Prices are converted to returns **outside** the pipeline (semantic change). Everything inside is a single sklearn `Pipeline` that can be cross-validated and tuned as one object.

### Preprocessing

Four sklearn-compatible transformers for return data cleaning:

- **DataValidator** -- replaces `inf` and extreme returns (|r| > 10) with `NaN`
- **OutlierTreater** -- three-group z-score methodology: remove data errors (>= 10 sigma), winsorize moderate outliers (3-10 sigma), keep normal observations
- **SectorImputer** -- leave-one-out sector-average NaN imputation with global mean fallback
- **RegressionImputer** -- OLS regression from top-5 correlated assets with cold-start fallback to sector imputation

### Pre-selection

Assembles data cleaning and asset filtering into a single sklearn pipeline:

`validate -> outliers -> impute -> select_complete -> drop_zero_variance -> drop_correlated -> [select_k] -> [select_pareto] -> [select_non_expiring]`

All steps run inside CV folds to prevent data leakage. Pipeline parameters are exposed via `get_params()` for hyperparameter tuning.

### Moment Estimation

5 expected return estimators and 11 covariance estimators with regime-aware blending:

| Expected Returns | Covariance |
|---|---|
| Empirical, Shrunk (James-Stein, Bayes-Stein), Exponentially Weighted, Equilibrium (CAPM), HMM-Blended | Empirical, EW, Ledoit-Wolf, OAS, Shrunk, Denoised (RMT), Detoned, Gerber, Graphical Lasso, Implied, HMM-Blended |

**HMM Regime Blending**: Fits a Gaussian HMM via Baum-Welch EM, then blends per-regime moments using filtered (causal) probabilities. `HMMBlendedCovariance` uses the full law of total variance including between-regime mean dispersion.

**Deep Markov Model** (optional, requires `torch` + `pyro`): Variational inference for state-space models with diagonal covariance.

**Log-normal scaling**: Multi-period moment projection with Jensen's inequality correction.

### View Integration

Three frameworks for incorporating forward-looking views:

- **Black-Litterman** -- Bayesian posterior combining market equilibrium with absolute/relative views. Three uncertainty methods: He-Litterman, Idzorek confidence, empirical track record
- **Entropy Pooling** -- 9 view types (mean, variance, correlation, skew, kurtosis, CVaR) via KL-divergence minimization
- **Opinion Pooling** -- Linear and logarithmic combination of multiple expert priors

### Optimization

13 portfolio optimization models across 4 categories:

| Category | Models |
|---|---|
| **Convex** | MeanRisk (4 objectives, 15 risk measures), Risk Budgeting, Maximum Diversification, Benchmark Tracker, DR-CVaR |
| **Hierarchical** | HRP, HERC, NCO |
| **Naive** | Equal Weighted, Inverse Volatility |
| **Ensemble** | Stacking Optimization |

**Robust variants**: Ellipsoidal mu uncertainty sets (kappa-scaled chi-squared confidence), bootstrap covariance uncertainty, distributionally robust CVaR over Wasserstein ball, HMM-driven regime-conditional risk measure selection.

Every model uses frozen `@dataclass` configs with named presets:

```python
MeanRiskConfig.for_max_sharpe()           # maximize Sharpe ratio
MeanRiskConfig.for_min_cvar(beta=0.95)    # minimize CVaR at 95%
RobustConfig.for_conservative()           # kappa=2 ellipsoidal uncertainty
DRCVaRConfig.for_moderate()               # Wasserstein ball epsilon=0.02
```

### Validation

Temporal cross-validation strategies that respect the time-series nature of financial data:

- **Walk-Forward** -- rolling or expanding window (quarterly, semiannual, annual presets)
- **Combinatorial Purged CV** -- multiple non-overlapping test paths with purging and embargoing to prevent leakage
- **Multiple Randomized CV** -- Monte Carlo evaluation with asset subsampling

### Scoring and Tuning

19 ratio measures (Sharpe, Sortino, Calmar, CVaR ratio, ...) for model selection. Grid search and randomized search with temporal CV enforced by default. Nested parameter addressing via sklearn's double-underscore syntax:

```python
param_grid = {
    "prior_estimator__mu_estimator__alpha": [0.01, 0.1],
    "risk_measure": [RiskMeasureType.CVAR, RiskMeasureType.SEMI_VARIANCE],
}
```

### Rebalancing

Three strategies for determining when to trade:

- **Calendar** -- fixed intervals (monthly, quarterly, semiannual, annual)
- **Threshold** -- drift-based (absolute or relative)
- **Hybrid** -- calendar-gated threshold (check drift only at review dates)

Plus utility functions: `compute_drifted_weights()`, `compute_turnover()`, `compute_rebalancing_cost()`.

### Factor Research

Complete factor research pipeline with 17 factors across 9 groups:

**Construction** -> **Standardization** (winsorize, z-score, sector neutralize) -> **Scoring** (equal-weight, IC-weighted, ICIR-weighted, Ridge, GBT) -> **Selection** (fixed-count or quantile with buffer hysteresis) -> **Regime Tilts** (GDP/yield-spread classification with multiplicative group tilts)

**Validation**: Information Coefficient analysis, Newey-West t-statistics, VIF collinearity, Benjamini-Hochberg FDR correction, out-of-sample rolling block validation.

**Integration**: Factor exposure constraints for MeanRisk, Black-Litterman views from factor premia, net alpha after turnover costs.

### Synthetic Data

Vine copula models for scenario generation. Decomposes the multivariate return distribution into marginal distributions and bivariate copulas organized in a tree structure. Supports conditional sampling for stress testing:

```python
# What if SPY drops 10%?
prior = build_synthetic_data(
    SyntheticDataConfig.for_stress_test(),
    sample_args={"conditioning": {"SPY": -0.10}},
)
```

### Universe Screening

8 investability screens with hysteresis entry/exit thresholds to reduce universe turnover: market cap, 12m/3m average daily dollar volume, trading frequency, price floors (US/Europe), listing age, IPO seasoning, financial statement coverage, exchange-relative percentile.

## Design Principles

**Config + Factory**: Every module uses frozen `@dataclass` configs holding only serializable primitives and enums. Factory functions create estimator instances. Configs can be serialized, logged, and swept over; non-serializable objects (estimators, arrays, callables) are passed as factory kwargs.

**sklearn compatibility**: All transformers follow `BaseEstimator + TransformerMixin`. The full preprocessing + optimization chain composes in `sklearn.pipeline.Pipeline` and can be cross-validated, tuned, and serialized as one object.

**skfolio foundation**: Optimization models wrap [skfolio](https://skfolio.org/) estimators. portopt adds regime blending, robust uncertainty sets, factor research, rebalancing, and universe screening on top.

## Architecture

```
optimizer/            Pure-Python library (DB-agnostic, sklearn/skfolio-based)
  pipeline/           End-to-end orchestration (prices -> validated weights)
  preprocessing/      Return data cleaning (validation, outliers, imputation)
  pre_selection/      Asset filtering pipeline (completeness, variance, correlation)
  moments/            Expected return + covariance estimation, HMM, DMM
  views/              Black-Litterman, Entropy Pooling, Opinion Pooling
  optimization/       13 optimization models + robust variants
  validation/         Walk-Forward, Combinatorial Purged CV, Randomized CV
  scoring/            19 ratio measures for model selection
  tuning/             Grid/randomized search with temporal CV
  rebalancing/        Calendar, threshold, and hybrid rebalancing
  factors/            17 factors, scoring, selection, regime tilts, validation
  synthetic/          Vine copula scenario generation + stress testing
  universe/           Investability screening with hysteresis

api/                  FastAPI backend (PostgreSQL, BAML, Trading 212)
cli/                  Typer CLI (data fetching, universe management)
tests/                Test suite (93%+ coverage)
theory/               LaTeX/Markdown theoretical documentation
examples/             Self-contained runnable scripts
```

## Examples

Self-contained scripts using real market data from skfolio datasets (no API keys required):

| Script | Description |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | MeanRisk optimization with walk-forward backtest |
| [`robust_optimization.py`](examples/robust_optimization.py) | Compare robust portfolios at different kappa values |
| [`regime_blending.py`](examples/regime_blending.py) | HMM fitting and regime-conditional moment estimation |
| [`factor_selection.py`](examples/factor_selection.py) | Factor construction, standardization, and stock selection |
| [`full_pipeline.py`](examples/full_pipeline.py) | End-to-end pipeline with pre-selection and rebalancing |

```bash
pip install portopt
python examples/quickstart.py
```

## Documentation

Full documentation with conceptual guides, configuration references, and code examples:

**[silviobaratto.github.io/optimizer](https://silviobaratto.github.io/optimizer)**

## Development

```bash
pip install -e ".[dev]"

# Tests
pytest tests/ -v

# Lint
ruff check optimizer/ tests/

# Type check
mypy optimizer/
```

## API + CLI

The project also includes a FastAPI backend and Typer CLI for data management and portfolio operations:

```bash
# Start PostgreSQL
docker compose up -d

# API
cd api && pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload     # http://localhost:8000

# CLI
python -m cli --help
python -m cli db health
python -m cli universe stats
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `FRED_API_KEY` | Federal Reserve Economic Data |
| `TRADING_212_API_KEY` | Trading 212 portfolio access |
| `TRADING_ECONOMICS_API_KEY` | Trading Economics macro data |

## Disclaimer

This software is provided for **educational and research purposes only**. It is not intended as, and shall not be understood or construed as, financial, investment, tax, or legal advice.

**No investment advice.** The authors and contributors are not registered investment advisors, broker-dealers, or financial planners. Nothing in this software or its documentation constitutes a recommendation to buy, sell, or hold any financial instrument.

**No liability for losses.** The authors and contributors accept no responsibility or liability whatsoever for any loss or damage arising from the use of this software. You may lose some or all of your invested capital. Use this software entirely at your own risk.

**Past performance is not indicative of future results.** Backtesting and historical analysis produced by this software do not guarantee future performance. Simulated results may not reflect the impact of real market conditions including liquidity, slippage, fees, and taxes.

**Seek professional advice.** Before making any investment decision, consult with a qualified, licensed financial advisor, accountant, or attorney.

By using this software, you acknowledge that you have read and understood this disclaimer and agree to be bound by its terms.

## License

[BSD-3-Clause](LICENSE)
