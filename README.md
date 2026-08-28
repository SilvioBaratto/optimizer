# portopt

[![CI](https://github.com/SilvioBaratto/optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/SilvioBaratto/optimizer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/portopt)](https://pypi.org/project/portopt/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![codecov](https://codecov.io/gh/SilvioBaratto/optimizer/branch/main/graph/badge.svg)](https://codecov.io/gh/SilvioBaratto/optimizer)
![License](https://img.shields.io/badge/license-PolyForm--Noncommercial--1.0.0-green)
[![oosmetrics](https://api.oosmetrics.com/api/v1/badge/achievement/c47694dc-b34e-481e-8907-2766ff13d4cd.svg)](https://oosmetrics.com/repo/SilvioBaratto/optimizer)

Quantitative portfolio construction and optimization built on [skfolio](https://skfolio.org/) and scikit-learn. Every component follows the **frozen-config + factory** pattern and composes in standard sklearn pipelines.

The repository is a **`uv` workspace** with three packages:

- **`optimizer/`** — the pure-Python optimization library, published to PyPI as **`portopt-core`** (import package `optimizer`). DB-agnostic, no API keys, no I/O.
- **`ingestion/`** — the **`portopt`** app: a yfinance-centric ingestion daemon (PostgreSQL + SQLAlchemy + APScheduler + BAML) plus a `uv`-installable CLI and install wizard. No HTTP API.
- **`packages/portopt-db/`** — **`portopt-db`** (import package `portopt_db`): the shared database layer — SQLAlchemy models, repositories, connection manager, and the single Alembic migration tree. Consumed by `ingestion`; carries no sklearn/skfolio stack.

The optimizer library is independent of the data side: neither `ingestion/` nor `portopt-db/` imports `optimizer`, and the daemon image carries none of the sklearn/skfolio optimization stack.

## Installation

The **`portopt` CLI** (ingestion daemon + install wizard) installs via [uv](https://docs.astral.sh/uv/):

```bash
# mac/linux
curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/main/install.sh | bash
# windows
powershell -c "irm https://raw.githubusercontent.com/SilvioBaratto/optimizer/main/install.ps1 | iex"
```

The bootstrap installs `uv` (if missing), runs `uv tool install portopt`, and launches
`portopt setup` — an interactive wizard that verifies Docker, validates your API keys,
encrypts your secrets (`~/.portopt/secrets.enc`), and migrates the database. A cloud LLM
provider (OpenAI or Anthropic) is **mandatory**. Re-run any time with `portopt setup`; manage
the stack with `portopt start` / `portopt stop` / `portopt status`.

The optimization **library** is a separate distribution, `portopt-core` (import package `optimizer`):

```bash
pip install portopt-core
```

For development (tests, linting, type checking):

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

`run_full_pipeline_with_selection()` extends the same entry point with an upstream stock-selection stage: fundamentals -> investability screening -> factor computation -> standardization -> regime tilts -> composite scoring -> selection.

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

4 expected-return estimators and 11 covariance estimators:

| Expected Returns | Covariance |
|---|---|
| Empirical, Shrunk (James-Stein, Bayes-Stein, Bodnar-Okhrin), Exponentially Weighted, Equilibrium (CAPM) | Empirical, EW, Ledoit-Wolf, OAS, Shrunk, Denoised (RMT), Detoned, Gerber, Graphical Lasso CV, Implied, Regime-Adjusted EW |

**Regime-adjusted EW**: short-term volatility uplift applied on top of an exponentially weighted covariance (multiplier internal to skfolio, clipped to `(0.7, 1.6)`).

**Log-normal scaling**: multi-period moment projection with Jensen's inequality correction (`apply_lognormal_correction`, `scale_moments_to_horizon`).

Separately, `build_variance_estimator()` returns 1-D `BaseVariance` estimators (`variance_`, not `covariance_`) — not interchangeable with covariance estimators inside priors.

### View Integration

Three frameworks for incorporating forward-looking views:

- **Black-Litterman** -- Bayesian posterior combining market equilibrium with absolute/relative views. Omega from He-Litterman, Idzorek confidence, or empirical track record (`calibrate_omega_from_track_record`)
- **Entropy Pooling** -- mean, variance, correlation, skew, kurtosis, and CVaR views via KL-divergence minimization
- **Opinion Pooling** -- linear and logarithmic combination of multiple expert priors

### Optimization

13 portfolio optimization models across 4 categories:

| Category | Models |
|---|---|
| **Convex** | MeanRisk, Risk Budgeting, Maximum Diversification, Benchmark Tracker, DR-CVaR |
| **Hierarchical** | HRP, HERC, NCO, Schur Complementary |
| **Naive** | Equal Weighted, Inverse Volatility, Random |
| **Ensemble** | Stacking Optimization |

**Robust variants**: ellipsoidal/bootstrap mu and covariance uncertainty sets (`RobustMeanRisk`), distributionally robust CVaR over a Wasserstein ball, and `RegimeBlendedMeanRisk` which consumes externally-supplied regime probabilities (the library does not fit HMMs itself).

**Constraint helpers**: `build_sector_constraints()` and `build_region_linear_constraints()` emit skfolio `linear_constraints` strings for group exposure bands.

Every model uses frozen `@dataclass` configs with named presets:

```python
MeanRiskConfig.for_max_sharpe()           # maximize Sharpe ratio
MeanRiskConfig.for_min_cvar(beta=0.95)    # minimize CVaR at 95%
RobustMeanRiskConfig.for_conservative()   # 99% uncertainty-set confidence
DRCVaRConfig.for_moderate()               # Wasserstein ball radius
```

### Validation

Temporal cross-validation strategies that respect the time-series nature of financial data:

- **Walk-Forward** -- rolling or expanding window (monthly, quarterly presets)
- **Combinatorial Purged CV** -- multiple non-overlapping test paths with purging and embargoing to prevent leakage
- **Multiple Randomized CV** -- Monte Carlo evaluation with asset subsampling

Plus covariance-forecast evaluation (offline and online).

### Scoring and Tuning

Ratio measures (Sharpe, Sortino, Calmar, CVaR ratio, ...) for model selection. Grid search and randomized search with temporal CV enforced by default. Nested parameter addressing via sklearn's double-underscore syntax:

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

**Integration**: factor exposure constraints for MeanRisk, Black-Litterman views from factor premia, net alpha after turnover costs.

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

### FX

Multi-currency handling: `FxPriceConverter` (sklearn transformer) converts a multi-currency price panel to a base currency (EUR/GBP/USD, optionally crossing via USD), and `decompose_fx_returns()` splits total return into stock-only and FX components.

## Design Principles

**Config + Factory**: Every module uses frozen `@dataclass` configs holding only serializable primitives and enums. Factory functions create estimator instances. Configs can be serialized, logged, and swept over; non-serializable objects (estimators, arrays, callables) are passed as factory kwargs.

**sklearn compatibility**: All transformers follow `BaseEstimator + TransformerMixin`. The full preprocessing + optimization chain composes in `sklearn.pipeline.Pipeline` and can be cross-validated, tuned, and serialized as one object.

**skfolio foundation**: Optimization models wrap [skfolio](https://skfolio.org/) estimators. portopt adds robust uncertainty sets, factor research, rebalancing, universe screening, and FX on top.

## Architecture

```
optimizer/            Pure-Python library (DB-agnostic, sklearn/skfolio-based)
  pipeline/           End-to-end orchestration (prices -> validated weights)
  preprocessing/      Return data cleaning (validation, outliers, imputation)
  pre_selection/      Asset filtering pipeline (completeness, variance, correlation)
  moments/            Expected return + covariance + variance estimation, prior construction
  views/              Black-Litterman, Entropy Pooling, Opinion Pooling
  optimization/       13 optimization models + robust variants + group constraints
  validation/         Walk-Forward, Combinatorial Purged CV, Randomized CV
  scoring/            Ratio measures for model selection
  tuning/             Grid/randomized search with temporal CV
  rebalancing/        Calendar, threshold, and hybrid rebalancing
  factors/            17 factors, scoring, selection, regime tilts, validation
  synthetic/          Vine copula scenario generation + stress testing
  universe/           Investability screening with hysteresis
  distance/           Distance estimators for hierarchical optimizers
  cluster/            Hierarchical clustering wrapper
  uncertainty_set/    Mu / covariance uncertainty sets for robust optimization
  linear_model/       Cross-sectional regression (factor IC)
  online/             partial_fit-based incremental workflows
  fx/                 Multi-currency conversion + FX return decomposition

ingestion/            Ingestion daemon (PostgreSQL, APScheduler, BAML) — services/scheduler/CLI
packages/portopt-db/  Shared DB layer (models, repositories, engine, single Alembic tree)
scheduler/            Shell wrappers over the daemon CLI (fetch, refetch)
scripts/              CI helpers (branch-coverage gate)
tests/                Library test suite
```

## Development

```bash
# uv workspace: one venv for all three packages
uv sync --all-packages --all-extras

# Tests (per package)
uv run --package portopt-core pytest tests/ -v       # optimizer library
uv run --package portopt-db   pytest                 # shared DB layer
uv run --package portopt      pytest                 # ingestion daemon

# Lint / type check
uv run --package portopt-core ruff check optimizer/ tests/
uv run --package portopt-core mypy optimizer/

# Everything (lint + typecheck + test)
make all
```

`pip install -e ".[dev]"` still works for the library alone if you are not on uv.

## Ingestion daemon

`ingestion/` is **yfinance-centric**: it builds its instrument universe from the yfinance
Screener and fetches market data, fundamentals, and macro series (FRED, Il Sole 24 Ore,
Trading Economics) into PostgreSQL on a schedule. APScheduler runs in-process; there is no
HTTP API. Job metrics are exposed to Prometheus, which is also the container healthcheck
target. Trading 212 is an optional add-on — when configured, its tickers are mapped onto the
yfinance universe *after* the build (it no longer sources it).

```bash
# PostgreSQL (host port 54320) + Adminer (18081) + scheduler (metrics 9000)
docker compose up -d
docker compose logs -f scheduler

# Or run the daemon directly (uv workspace)
uv sync --all-packages --all-extras
(cd packages/portopt-db && alembic upgrade head)   # migrations owned by portopt-db
uv run --package portopt python -m app.worker      # blocks until SIGTERM
```

Seven scheduled jobs: `daily_pipeline` (07:00), `midday_news` (14:00), `universe_build`
(Sun 02:00), `weekly_refetch` (Sun 03:00), `fred_monthly`, `news_refresh` (30 min), and
`orphan_reaper`. Cadence is configurable via `SCHEDULER_*` env vars.

Any step can be run by hand through the same job-slot and heartbeat path the scheduler
uses — so a manual run is refused rather than double-fetching if the scheduler is already
running that step:

```bash
docker compose exec scheduler python -m app.cli daily
docker compose exec scheduler python -m app.cli yfinance --mode full --period 5y
# also: refetch-all | universe | macro | fred | news | summarize | calibrate |
#       reference-indices
```

Run **exactly one daemon per database**: the orphan reaper fails any active job whose
worker host is not its own, so two instances will reap each other's jobs.

See `ingestion/README.md` for the full picture.

### Environment Variables

`portopt setup` collects and encrypts these; for CI / manual runs the daemon also reads them
from the environment (and Docker-compose `secrets:` at `/run/secrets/*`):

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `FRED_API_KEY` | Federal Reserve Economic Data |
| `LLM_PROVIDER` + `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (+ `OPENAI_MODEL` / `ANTHROPIC_MODEL`) | Cloud LLM (**mandatory**) — BAML news summarization + macro-regime calibration. Local models are not supported |
| `TRADING_212_API_KEY` / `TRADING_212_SECRET_KEY` / `TRADING_212_MODE` | Optional Trading 212 add-on — mapped onto the yfinance universe after the build |
| `METRICS_PORT` | Prometheus port (default `9000`) |
| `NOTIFICATION_WEBHOOK_URL` | Discord/Slack webhook for job-failure alerts (optional) |

Il Sole 24 Ore and Trading Economics are scraped from HTML and need no key.
Scheduler cadence is configurable via `SCHEDULER_*` env vars — see `CLAUDE.md`.

## Disclaimer

This software is provided for **educational and research purposes only**. It is not intended as, and shall not be understood or construed as, financial, investment, tax, or legal advice.

**No investment advice.** The authors and contributors are not registered investment advisors, broker-dealers, or financial planners. Nothing in this software or its documentation constitutes a recommendation to buy, sell, or hold any financial instrument.

**No liability for losses.** The authors and contributors accept no responsibility or liability whatsoever for any loss or damage arising from the use of this software. You may lose some or all of your invested capital. Use this software entirely at your own risk.

**Past performance is not indicative of future results.** Backtesting and historical analysis produced by this software do not guarantee future performance. Simulated results may not reflect the impact of real market conditions including liquidity, slippage, fees, and taxes.

**Seek professional advice.** Before making any investment decision, consult with a qualified, licensed financial advisor, accountant, or attorney.

By using this software, you acknowledge that you have read and understood this disclaimer and agree to be bound by its terms.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SilvioBaratto/optimizer&type=Date)](https://star-history.com/#SilvioBaratto/optimizer&Date)

## License

[PolyForm Noncommercial License 1.0.0](LICENSE)
