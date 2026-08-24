# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent & Skill Requirements

**MANDATORY**: For any finance-related or code implementation task in this repository, always use:
- `/skfolio` skill — for all portfolio optimization, risk models, and skfolio API usage
- `/yfinance` skill — for all Yahoo Finance data retrieval and yfinance API usage
- `python-pro` agent — for all Python code writing, reviewing, and debugging

These must be loaded proactively, not on request. Any work involving financial data, portfolio optimization, or Python implementation must go through these tools.

## Project Overview

Python-only repository. Two shipped things:

- **`optimizer/`** — Pure-Python optimization library (DB-agnostic, sklearn/skfolio-based). Published to PyPI as **`portopt`**
- **`ingestion/`** — Headless **ingestion daemon**. APScheduler in-process, no HTTP API. Fetches market / fundamental / macro data into PostgreSQL on a schedule. Entrypoint `ingestion/app/worker.py`; manual runs via `ingestion/app/cli.py`

The two do not depend on each other. `ingestion/` does **not** import `optimizer`, and the ingestion image carries no sklearn/skfolio/scipy stack.

Supporting directories:

- **`tests/`** — library test suite (mirrors `optimizer/` submodules) + `tests/scheduler/` (shell-wrapper contract)
- **`ingestion/tests/`** — daemon test suite (SQLite in-memory)
- **`scheduler/`** — thin shell wrappers over the CLI (`fetch.sh`, `refetch_all.sh`)
- **`scripts/`** — CI helpers (`check_branch_coverage.py`)

There is **no frontend, no docs site, no `examples/`, no `research/`, no `cli/`, and no HTTP API**. They were deleted in the strip (branch `refactor/strip-to-ingestion-pipeline`). If you find a reference to any of them — a route, a `TestClient`, `uvicorn`, `app.main`, `/api/v1/` — it is a leftover; delete it rather than reviving the dependency.

## Build & Run Commands

```bash
# Infrastructure
docker compose up -d              # PostgreSQL (54320) + Adminer (18081) + scheduler (metrics 9000)

# Optimizer library (root)
pip install -e ".[dev]"           # Install optimizer + dev deps
pytest tests/ -v                  # All optimizer tests
pytest tests/rebalancing/ -v      # Single module tests
pytest -k "test_name"             # Single test by name
ruff check optimizer/ tests/      # Lint (CI step)
ruff check . --fix                # Lint + auto-fix
mypy optimizer/                   # Type check strict mode (CI step)

# Makefile shortcuts (root)
make lint                         # ruff check + ruff format --check
make format                       # ruff format (writes changes)
make typecheck                    # mypy optimizer/
make test                         # pytest with coverage term-missing
make coverage                     # pytest with HTML coverage in htmlcov/
make all                          # lint + typecheck + test
make clean                        # remove caches, coverage, egg-info

# Ingestion daemon
cd ingestion && pip install -e ".[test]"
alembic upgrade head              # Run migrations
python -m app.worker              # Run the daemon (blocks; SIGTERM to stop)
cd ingestion && pytest            # Daemon tests

# Manual ingestion runs (same job-slot / heartbeat path as the scheduler)
docker compose exec scheduler python -m app.cli daily
docker compose exec scheduler python -m app.cli refetch-all
docker compose exec scheduler python -m app.cli yfinance --mode full --period 5y
# also: universe | macro | fred | news | summarize | calibrate | reference-indices

# BAML (regenerate after editing ingestion/baml_src/)
cd ingestion && baml-cli generate
```

## CI Pipeline

`.github/workflows/ci.yml` — triggers on push/PR to `main`, Ubuntu, Python 3.12. Jobs:

| Job | Steps |
|-----|-------|
| `lint` | `ruff check optimizer/ tests/` → `ruff format --check` → `pip-audit --strict` |
| `typecheck` | `mypy optimizer/` |
| `pyright` | `pyright` (scoped to `optimizer/` only) |
| `test` | `pytest tests/` with `--cov=optimizer --cov-fail-under=90`, then `scripts/check_branch_coverage.py coverage.xml 0.80` |
| `ingestion-test` | `pytest ingestion/tests/` with `--cov=app --cov-fail-under=80`, then the same branch-coverage gate |

Other workflows: `release.yml` (on `v*` tags). There is no `smoke.yml` — it drove `/optimize` and `/backtest`, which no longer exist.

**Dependencies**: the library's runtime deps live in the root `pyproject.toml` `[project.dependencies]` — CI installs via `pip install -e ".[dev]"`. The ingestion daemon's deps live in `ingestion/pyproject.toml` (`[project.dependencies]` + the `[test]` extra), installed separately by the `ingestion-test` job (`pip install -e "./ingestion[test]"`) and by `ingestion/Dockerfile` (`pip install .`, runtime-only). There is no `requirements.txt` anywhere; add library deps to the root `pyproject.toml` and daemon deps to `ingestion/pyproject.toml`.

## Architecture

### Optimizer Library (`optimizer/`)

Every module follows the same pattern: **frozen `@dataclass` config** + **factory function** + **`str, Enum` types**. Configs hold only primitives/enums/nested frozen dataclasses (serialisable). Non-serialisable objects (estimator instances, numpy arrays, callables) are passed as factory `**kwargs`. This boundary is strict and consistent across all modules.

All transformers follow the sklearn `BaseEstimator + TransformerMixin` API and compose in `sklearn.pipeline.Pipeline`. The pipeline flattens pre-selection + optimiser steps so `get_params()` exposes all nested parameters (e.g. `"optimizer__l2_coef"`, `"drop_correlated__threshold"`).

#### Pipeline flow

```
prices → preprocessing → pre_selection → moments → views →
optimization → validation → tuning → rebalancing → pipeline
```

Plus: `factors/`, `synthetic/`, `scoring/`, `universe/`, `distance/`, `cluster/`, `uncertainty_set/`, `linear_model/`, `online/`, `fx/`

#### Submodule index (skfolio 0.20.1)

| Submodule | Purpose |
|-----------|---------|
| `preprocessing/` | sklearn time-series + cross-sectional transformers for return cleaning. |
| `pre_selection/` | Asset pre-selection pipeline assembly. |
| `moments/` | Mu / covariance / variance estimators + prior construction. |
| `views/` | Black-Litterman, Entropy Pooling, Opinion Pooling. |
| `distance/` | `DistanceConfig` + `build_distance` wrapping skfolio's six distance estimators (Pearson, Kendall, Spearman, Covariance, Distance Correlation, Mutual Information). |
| `cluster/` | `HierarchicalClusteringConfig` + `build_hierarchical_clustering` wrapping `HierarchicalClustering` with seven linkage methods. |
| `uncertainty_set/` | `Mu/CovarianceUncertaintySetConfig` + factories for ellipsoidal / bootstrap uncertainty sets used by `RobustMeanRisk`. |
| `linear_model/` | `CSLinearRegressionConfig` + `build_cs_linear_regression` for cross-sectional regressions and factor IC. |
| `online/` | `partial_fit`-based incremental workflows: `run_online_predict`, `run_online_score`, `OnlineGridSearch`, `OnlineRandomizedSearch`. **Online instances are not thread-safe — one wrapper per thread.** |
| `optimization/` | Convex / hierarchical / ensemble / robust optimizers + sector & region constraint builders. |
| `synthetic/` | Vine copula models + synthetic data generation. |
| `validation/` | Walk-Forward / CPCV / MultipleRandomizedCV + covariance forecast evaluation (offline + online). |
| `scoring/` | Ratio-measure scorers for model selection. |
| `tuning/` | Grid / Randomized search with temporal CV enforced. |
| `rebalancing/` | Calendar / threshold / hybrid rebalancing decisions. |
| `factors/` | Factor research pipeline (construction → scoring → selection → integration). |
| `universe/` | Investability screening with hysteresis. |
| `fx/` | Multi-currency conversion + FX-vs-stock return decomposition. |
| `pipeline/` | End-to-end orchestration: prices → validated weights. |

#### Module details

- **`preprocessing/`** — sklearn transformers for return data cleaning:
  - `DataValidator` — replaces `inf` and extreme returns with `NaN`
  - `OutlierTreater` — three-group z-score methodology (remove / winsorize / keep)
  - `SectorImputer` — leave-one-out sector-average NaN imputation
  - `RegressionImputer` — OLS regression from top-K correlated assets (`n_neighbors=5`, `min_train_periods=60`); cold-start assets and rows with missing neighbors fall back to `SectorImputer`

- **`pre_selection/`** — `PreSelectionConfig` + `build_preselection_pipeline()` factory assembling sklearn `Pipeline` from config (composes custom transformers with skfolio selectors: `SelectComplete`, `DropZeroVariance`, `DropCorrelated`, `SelectKExtremes`, `SelectNonDominated`, `SelectNonExpiring`)

- **`moments/`** — Moment estimation and prior construction:
  - `MomentEstimationConfig` — selects mu/cov estimators; presets: `for_equilibrium_ledoitwolf`, `for_shrunk_denoised`, `for_adaptive`, `for_regime_adjusted_ew`
  - `MuEstimatorType` has 4 members (`EMPIRICAL`, `SHRUNK`, `EW`, `EQUILIBRIUM`); `CovEstimatorType` has 11. There is **no HMM-blended or Deep-Markov estimator** — those were removed
  - `build_mu_estimator()` — maps `MuEstimatorType` to skfolio `BaseMu` instances
  - `build_cov_estimator()` — maps `CovEstimatorType` to skfolio `BaseCovariance` instances
  - `build_prior()` — composes mu + cov into `EmpiricalPrior`, optionally wrapping in `TimeSeriesFactorModel`
  - `build_variance_estimator()` — returns a 1-D `BaseVariance` (`EmpiricalVariance`, `EWVariance`, or `RegimeAdjustedEWVariance`). **Gotcha**: variance estimators expose `variance_` (1-D), NOT `covariance_` (2-D); not interchangeable with covariance estimators inside priors that need a full matrix
  - `RegimeAdjustedEWCovariance` is reachable via `CovEstimatorType.REGIME_ADJUSTED_EW`. STVU multiplier is internal to skfolio (no `hmmlearn` import path) and clipped to `(0.7, 1.6)` by default
  - `apply_lognormal_correction()` / `scale_moments_to_horizon()` — multi-period variance scaling. **Gotcha**: inputs are log-return parameters, output is simple-return space (`E[R_T] = exp(...) - 1`)

- **`views/`** — View integration frameworks:
  - `BlackLittermanConfig` / `build_black_litterman()` — presets: `for_equilibrium`, `for_factor_model`. When inside `TimeSeriesFactorModel`, views must reference factor names (e.g. `MTUM`, `QUAL`), not asset names
  - `EntropyPoolingConfig` / `build_entropy_pooling()` — supports mean/variance/correlation/skew/kurtosis/cvar views. Correlation views use format `(ASSET1, ASSET2) == value`
  - `OpinionPoolingConfig` / `build_opinion_pooling()` — expert estimators passed as factory kwarg (not stored in config)
  - `calibrate_omega_from_track_record(view_history, return_history)` — empirical diagonal Ω matrix from forecast error variance; requires ≥5 aligned observations

- **`optimization/`** — Portfolio optimization models. Convex, hierarchical, ensemble, robust:
  - `MeanRiskConfig` / `build_mean_risk()` — base convex Mean-Risk optimizer with full constraint surface. Presets include `for_min_variance`, `for_max_sharpe`, `for_max_sharpe_diversified`, `for_concentrated_sharpe`, `for_max_sharpe_sector_constrained`, `for_max_utility`, `for_min_cvar`, `for_efficient_frontier`
  - `RiskBudgetingConfig` / `build_risk_budgeting()` — Equal-Risk-Contribution (ERC) by default; supports custom risk budgets via the `for_custom_budgets({"AAPL": 0.5, ...})` preset (sum-to-1 enforced)
  - `HRPConfig` / `build_hrp()` — Hierarchical Risk Parity (recursive bisection, no matrix inversion)
  - `HERCConfig` / `build_herc()` — Hierarchical Equal Risk Contribution (cluster-level ERC)
  - `NCOConfig` / `build_nco()` — Nested Clusters Optimization. Inner/outer estimators are factory kwargs (non-serialisable)
  - `SchurComplementaryConfig` / `build_schur_complementary()` — interpolates between HRP (`gamma=0`) and Minimum-Variance (`gamma=1`); `gamma=0.5` is the skfolio default (`for_balanced` preset)
  - `MaxDiversificationConfig` / `build_max_diversification()` — maximises diversification ratio. Long-only by default (ratio is undefined for short positions)
  - `BenchmarkTrackerConfig` / `build_benchmark_tracker()` — minimises tracking error vs a benchmark return series. **Gotcha**: benchmark returns are passed as `y` in `fit(X, y)`, not as a Config field
  - `EqualWeightedConfig` / `build_equal_weighted()` — uniform weights baseline
  - `InverseVolatilityConfig` / `build_inverse_volatility()` — weights ∝ 1/σᵢ; `for_ew_covariance(half_life=63)` preset uses an EW covariance prior
  - `RandomConfig` / `build_random()` — Dirichlet random weights for naive baseline. **Gotcha**: skfolio 0.20.1 `Random` draws a single sample with no `n_portfolios` / `random_state` constructor args; the Config fields are reserved for forward compatibility and are NOT forwarded
  - `StackingConfig` / `build_stacking()` — ensemble blending multiple base optimizers via a final estimator. Base `estimators` list is a factory kwarg (non-serialisable)
  - `RobustMeanRiskConfig` / `build_robust_mean_risk()` — wraps `MeanRisk` with mu / covariance uncertainty-set estimators (see `uncertainty_set/`). Presets: `for_conservative` (99%), `for_moderate` (95%), `for_aggressive` (90%), `for_bootstrap_covariance`. **Fallback contract**: both uncertainty configs `None` recovers plain `MeanRisk` exactly (atol=1e-8)
  - `DRCVaRConfig` / `build_dr_cvar()` — distributionally robust CVaR over a Wasserstein ball. **Fallback contract**: `epsilon=0` short-circuits to plain `MeanRisk(CVaR)` for exact equality (skfolio's DRCVaR with radius=0 differs at ~1e-3 from `MeanRisk(CVaR)`); return type is `BaseOptimization`
  - `RegimeBlendedMeanRiskConfig` / `build_regime_blended_mean_risk()` — regime-blended mean-risk with externally-supplied regime probabilities. Composes `_ExternallyControlledRegimeCovariance` → `EmpiricalPrior` → `TimeSeriesFactorModel` → `MeanRisk`. Caller supplies `regime_probabilities` and `factor_returns` as factory kwargs; the library does NOT fit HMMs internally
  - `build_sector_constraints()` / `build_region_linear_constraints()` (`_region_constraints.py`) — emit skfolio `linear_constraints` strings for group exposure bands. **Gotcha**: a `-` inside a group token is parsed as minus by skfolio and silently drops the constraint — sanitize tokens (hyphen → space) and only constrain groups actually present

- **`synthetic/`** — Vine copula models + synthetic data generation:
  - `VineCopulaConfig` / `SyntheticDataConfig` — presets: `for_scenario_generation`, `for_stress_test`
  - Stress testing: pass `sample_args={"conditioning": {"TICKER": value}}` to `build_synthetic_data()`

- **`validation/`** — Cross-validation:
  - `WalkForwardConfig` (`for_monthly_rolling`, `for_quarterly_rolling`, `for_quarterly_expanding`), `CPCVConfig`, `MultipleRandomizedCVConfig`
  - `run_cross_val()` — defaults to WalkForward (quarterly rolling) when no `cv` is passed
  - Covariance forecast evaluation (offline + online)

- **`scoring/`** — `ScorerConfig` / `build_scorer()` — ratio measures for model selection

- **`tuning/`** — `GridSearchConfig` / `RandomizedSearchConfig` — temporal CV enforced by default. Use sklearn `__` notation for nested tuning (e.g. `"prior_estimator__mu_estimator__alpha"`)

- **`rebalancing/`** — Calendar, threshold, and hybrid rebalancing:
  - `CalendarRebalancingConfig` — fixed-interval rebalancing (21/63/126/252 trading days)
  - `ThresholdRebalancingConfig` — drift-based rebalancing (absolute or relative thresholds)
  - `HybridRebalancingConfig` — calendar-gated threshold: checks drift only at review dates, always returns `False` between reviews. Presets: `for_monthly_with_5pct_threshold`, `for_quarterly_with_10pct_threshold`
  - `should_rebalance()` / `should_rebalance_hybrid()` — decision functions
  - `compute_drifted_weights()`, `compute_turnover()`, `compute_rebalancing_cost()`

- **`factors/`** — Complete factor research pipeline:
  - **Config types**: `FactorConstructionConfig`, `StandardizationConfig`, `CompositeScoringConfig`, `SelectionConfig`, `RegimeTiltConfig`, `FactorValidationConfig`, `FactorIntegrationConfig`, `PublicationLagConfig`
  - **Enums**: `FactorGroupType` (9 groups), `FactorType` (17 factors), `CompositeMethod` (EQUAL_WEIGHT, IC_WEIGHTED, ICIR_WEIGHTED, RIDGE_WEIGHTED, GBT_WEIGHTED), `MacroRegime` (4 regimes)
  - **Construction**: `compute_all_factors()` — builds factor scores from fundamentals + price data; `align_to_pit()` handles publication lag to prevent look-ahead bias
  - **Standardization**: `standardize_all_factors()` — winsorize → z-score/rank-normal → sector neutralize
  - **Scoring**: `compute_composite_score()` — dispatches to equal-weight, IC-weighted, ICIR-weighted, or ML (ridge/GBT) scoring
  - **Selection**: `select_stocks()` — fixed-count or quantile selection with buffer-zone hysteresis and sector balancing
  - **Regime tilts**: `classify_regime()` (GDP/yield-spread heuristic) + `apply_regime_tilts()` — multiplicative tilts on group weights
  - **Validation**: `run_factor_validation()` → `FactorValidationReport` with IC, Newey-West t-stats, VIF, Benjamini-Hochberg correction; `run_factor_oos_validation()` — rolling block OOS validation
  - **Mimicking portfolios**: `build_factor_mimicking_portfolios()`, `compute_quintile_spread()`
  - **Integration**: `build_factor_exposure_constraints()` → `FactorExposureConstraints` (ready for `MeanRisk`); `build_factor_bl_views()` → Black-Litterman views; `compute_net_alpha()` → net alpha after turnover costs

- **`universe/`** — Investability screening with hysteresis:
  - `InvestabilityScreenConfig` — 8 screens (market cap, ADDV, trading frequency, price, listing age, IPO seasoning, financial statements, exchange percentile) with `HysteresisConfig` entry/exit pairs
  - `screen_universe()` — factory returning `pd.Index` of passing tickers
  - Presets: `for_developed_markets`, `for_broad_universe`, `for_small_cap`

- **`pipeline/`** — End-to-end orchestration:
  - `run_full_pipeline(prices, optimizer, ...)` — single entry point: prices → returns → pipeline → backtest → weights → rebalancing. Accepts `cv_config`, `previous_weights`, `rebalancing_config` (threshold or hybrid), `current_date`, `last_review_date`, `y_prices`, `sector_mapping`, `n_jobs`
  - `run_full_pipeline_with_selection(...)` — extends with upstream stock selection: fundamentals → investability screening → factor computation → standardization → regime tilts → composite scoring → stock selection → `run_full_pipeline`. When `fundamentals=None`, skips all selection and delegates directly
  - `optimize()`, `backtest()`, `tune_and_optimize()`, `build_portfolio_pipeline()`, `compute_net_backtest_returns()` — lower-level composable functions

- **`distance/`** — `DistanceConfig` + `build_distance` wrapping skfolio's six distance estimators (`PearsonDistance`, `KendallDistance`, `SpearmanDistance`, `CovarianceDistance`, `DistanceCorrelation`, `MutualInformation`). MI-only fields (`n_bins`, `bandwidth`) raise `ConfigurationError` on non-MI estimators; `bandwidth` is reserved (skfolio 0.20.1 does not expose it). Six `for_<name>` presets.

- **`cluster/`** — `HierarchicalClusteringConfig` + `build_hierarchical_clustering` wrapping `skfolio.cluster.HierarchicalClustering`. `LinkageMethodType` mirrors `skfolio.cluster.LinkageMethod` exactly (7 members: `SINGLE`, `COMPLETE`, `AVERAGE`, `WEIGHTED`, `CENTROID`, `MEDIAN`, `WARD`). `min_cluster_size` field is reserved (skfolio 0.20.1 does not expose it).

- **`uncertainty_set/`** — Mu / Covariance uncertainty-set estimators for `RobustMeanRisk`. Two parallel Configs (`MuUncertaintySetConfig`, `CovarianceUncertaintySetConfig`), each with `EMPIRICAL` and `BOOTSTRAP` kinds. Bootstrap variants use `arch.StationaryBootstrap` (Politis-White block-size rule when `block_size=None`); `random_state` field maps to skfolio's `seed`. **Gotcha**: `confidence_level` is a probability (e.g. 0.95); the chi-squared κ scaling is internal to `MeanRisk` and not exposed.

- **`linear_model/`** — `CSLinearRegressionConfig` + `build_cs_linear_regression` wrapping `skfolio.linear_model.CSLinearRegression`. **Shape contract**: `X: (T, N, K)`, `y: (T, N)`, `cs_weights: (T, N)`. `weighted` and `min_observations` Config fields are caller-side hints (not skfolio constructor args). Used by the opt-in `compute_ic_series(use_cs_regression=True)` path in `factors/`.

- **`online/`** — `partial_fit`-based incremental workflows: `run_online_predict`, `run_online_score`, `build_online_grid_search`, `build_online_randomized_search`. **Pipeline is rejected** at the wrapper boundary (raises `ConfigurationError`). **Online instances are not thread-safe** — `partial_fit` accumulates mutable state and `OnlineGridSearch` mutates the wrapped estimator in place. Construct one wrapper per thread when running scheduled jobs.

- **`fx/`** — Multi-currency conversion + return decomposition:
  - `FxConfig` (frozen) with `BaseCurrency` (EUR/GBP/USD), `FxConversionMode` (NONE/TO_BASE/DECOMPOSE), `FxDataSource` (YFINANCE/FRED), `cross_via_usd`, `require_full_coverage`, `strict`
  - `FxPriceConverter` is sklearn-compatible (`BaseEstimator + TransformerMixin`) and slots into `sklearn.pipeline.Pipeline` upstream of optimization
  - `decompose_fx_returns()` / `FxReturnDecomposition` — splits total return into stock-only and FX components
  - `currency_map` and `fx_rates` are non-serialisable runtime kwargs to `build_fx_converter()`, NOT config fields

### Key conventions

- `prices_to_returns()` runs **outside** the pipeline (changes data semantics); pipeline operates on return DataFrames only
- Views use `tuple[str, ...]` in configs (hashable); factories convert to `list` for skfolio
- View configs embed `MomentEstimationConfig` for their inner prior (keeps configs serialisable)
- The fitted prior attribute is `return_distribution_` (not `prior_model_`), containing `mu`, `covariance`, `returns`, `sample_weight`, `cholesky`
- For `BenchmarkTracker`, benchmark returns are passed as `y` in `fit(X, y)`
- When `previous_weights` is passed to `run_full_pipeline()`, it auto-aligns on post-pre-selection universe and re-normalises
- Sector mapping is injected as a plain `dict[str, str]`, not queried from the database

#### Cross-cutting gotchas (skfolio 0.20.1)

- **Linear returns only**: pipelines and skfolio estimators consume linear (simple) returns as `X`. Use `prices_to_returns()` (default) — do NOT pass log returns
- **`shuffle=False` in cross-validation**: temporal CV must preserve order. `KFold(shuffle=True)` and `train_test_split(shuffle=True)` break causality and silently leak future data
- **Metadata routing**: call `sklearn.set_config(enable_metadata_routing=True)` BEFORE configuring `.set_fit_request(...)` on estimators. Required for `ImpliedCovariance.implied_vol`, `BenchmarkTracker.y`, etc.
- **`TimeSeriesFactorModel.fit(X, y)`** — `X` is asset returns, `y` is factor returns. When wrapped in Black-Litterman, views must reference factor names (not asset names)
- **`BenchmarkTracker.fit(X, y)`** — `y` is the benchmark return series. The `BenchmarkTrackerConfig` carries no benchmark field; pass at fit time
- **Variance estimators store `variance_` (1-D), NOT `covariance_` (2-D)** — `EmpiricalVariance` / `EWVariance` / `RegimeAdjustedEWVariance` are NOT interchangeable with covariance estimators inside priors that need a full matrix
- **`Pipeline` is rejected by `online_predict` / `OnlineGridSearch`** — skfolio routes `partial_fit` through a single estimator and cannot route through `Pipeline`. Apply pre-selection to `X` upstream before passing to online wrappers
- **Walk-forward CV cannot vary constraints per fold** — regime-dependent sector bands are fixed for a whole run; a backtest is single-regime, not per-rebalance

### Ingestion Daemon (`ingestion/app/`)

Headless. **No HTTP API, no FastAPI, no routes.** Layering is
**Scheduler/CLI → Services → Repositories → Models**, with `_shared/` in each layer
for cross-cutting code.

| Layer | Path | Domains |
|-------|------|---------|
| Entrypoints | `ingestion/app/` | `worker.py` (daemon), `cli.py` (manual runs) |
| Services | `ingestion/app/services/` | `jobs`, `macro`, `market_data`, `universe`, `infrastructure`, `_shared` |
| Repositories | `ingestion/app/repositories/` | `jobs`, `macro`, `market_data`, `universe`, `_shared` |
| Models | `ingestion/app/models/` | `jobs`, `macro`, `market_data`, `universe`, `_shared` |
| Schemas | `ingestion/app/schemas/` | `jobs`, `macro`, `market_data`, `universe`, `_shared` |

With the HTTP layer gone, schemas are no longer request/response bodies — they are the
typed argument objects the scheduler and CLI pass into service functions
(`YFinanceFetchRequest`, `MacroFetchRequest`, `UniverseBuildRequest`) plus the progress
payloads those services report back. `app/schemas/__init__.py` deliberately re-exports
nothing; import from the domain module.

**Conventions**:
- Synchronous SQLAlchemy sessions (`Session`, not `AsyncSession`). Everything opens its own session via `database_manager.get_session` — there is no request scope
- Repository pattern — all DB queries through typed repositories
- BAML — LLM function definitions in `ingestion/baml_src/`, generated client in `ingestion/baml_client/` (do not edit generated files). Only two functions survive: `SummarizeCountryNews` and `ClassifyMacroRegime`
- PostgreSQL 16 on port **54320** (not 5432). Connection: `postgresql://postgres:postgres@localhost:54320/optimizer_db`
- The library is configured by the root `pyproject.toml`; the daemon by `ingestion/pyproject.toml`
- **Do not reintroduce `optimizer` as an ingestion dependency.** The daemon ingests; it does not optimize

**Import-cycle gotcha**: `app/services/_shared/__init__.py` must NOT re-export `bootstrap_benchmarks`. `_benchmark_bootstrap` imports `market_data.reference_index_seeder`, which imports back into `_shared` for `ProgressCallback` — re-exporting makes that cycle load-bearing on import order. Import it from the module: `from app.services._shared._benchmark_bootstrap import bootstrap_benchmarks`.

### Scheduler & Background Jobs

APScheduler runs in-process inside `app/worker.py`, with a `SQLAlchemyJobStore` so misfired
runs execute at next startup within a configurable grace window.

**Seven scheduled jobs** (cron schedules configurable via `SCHEDULER_*` env vars):

| Job | Trigger | Default | Description |
|-----|---------|---------|-------------|
| `daily_pipeline` | Cron | `0 7 * * *` | Sequential: ref-indices → yfinance → macro → news → summarize → calibrate |
| `midday_news` | Cron | `0 14 * * *` | Afternoon news + summarize refresh |
| `universe_build` | Cron | `0 2 * * 0` | Trading 212 instrument-universe rebuild |
| `weekly_refetch` | Cron | `0 3 * * 0` | Full yfinance + macro rebuild (5-year history) |
| `fred_monthly` | Cron | `0 8 1 * *` | FRED economic data fetch |
| `news_refresh` | Interval | Every 30 min | Incremental news re-summarization |
| `orphan_reaper` | Interval | Every 300s | Fails jobs whose worker died without a terminal status |

`universe_build` is scheduled **before** `weekly_refetch` on purpose: every other step
iterates the `instruments` table, so a stale universe silently caps what yfinance fetches.

**Composable steps**: `scheduler.py` exposes one public function per step
(`run_yfinance_step`, `run_macro_step`, `run_fred_step`, `run_news_step`,
`run_summarize_step`, `run_calibrate_step`, `run_universe_step`,
`refresh_reference_indices`). Each returns `True` only on completion. The scheduled
pipelines and the CLI both compose these, so a manual run takes the identical
job-slot / heartbeat / progress path. **Add new work as a step, not as a CLI-only branch.**

**BackgroundJobService** (`ingestion/app/services/jobs/background_job.py`) — one instance per job
domain, at module level in `scheduler.py`:
- `create_job()` — atomically claims a slot; raises `JobAlreadyRunningError` if one is already pending/running
- `get_job(job_id)` → dict; `update_job(job_id, **kwargs)` — status, progress, errors
- `start_background(target, args)` — daemon thread + heartbeat companion

**Gotcha — synchronous steps need an explicit heartbeat.** Only the heartbeat thread stamps
`last_heartbeat_at`; `update_job` never does. A step run synchronously in the scheduler
thread (i.e. not via `start_background`) and outliving `SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS`
(300s) gets falsely reaped mid-run and flips to `failed` while still working — both the
yfinance fetch and the reference-index seed exceed that. `scheduler._heartbeat()` is the
shared context manager for this; `_run_step` and `refresh_reference_indices` both use it.
**Any new synchronous step must wrap its work in it.**

**Gotcha — the reaper is host-scoped.** `reconcile_orphans` fails any active row whose
`worker_host != socket.gethostname()`. Two containers pointed at the same database will
reap each other's jobs. Run exactly one daemon per DB.

**Progress callback**: `make_progress(job_id, job_svc)` in `app/services/_shared/_progress.py`
returns a closure forwarding kwargs to `update_job`. Service functions accept `on_progress=`
to report `current=`, `total=`, `current_country=`, etc.

**Persistence**: `background_jobs` table (UUID PK, `job_type` + `status` composite index,
JSONB `extra`/`result`/`errors`). Model uses `JSON().with_variant(JSONB, "postgresql")` for
SQLite test compatibility.

**Observability** — logs, the `background_jobs` table, and Prometheus. There is no job-polling
endpoint:
- Prometheus (`ingestion/app/metrics.py`): counters `jobs_started_total`, `jobs_completed_total`, `jobs_failed_total`; histogram `job_duration_seconds`; gauge `jobs_in_progress` — all labeled by `domain`. Served by `prometheus_client.start_http_server` on `METRICS_PORT` (default 9000); `worker.py` imports `app.metrics` eagerly so the families exist before the first job runs
- Webhook (`app/services/_shared/notifications.py`): Discord/Slack-compatible POST on job failure when `NOTIFICATION_WEBHOOK_URL` is set

### Shared Infrastructure (`ingestion/app/services/infrastructure/`)

Generalized resilience primitives, re-exported as shims in
`ingestion/app/services/market_data/yfinance/infrastructure/`:
- **`CircuitBreaker`** — exponential backoff (2^attempt), max_attempts safety limit, service-name in errors
- **`RateLimiter`** — thread-safe per-key delay enforcement (default 0.1s)
- **`retry_with_backoff()`** — retry with full-jitter exponential backoff, transient error detection
- **`LRUCache`** — in-memory TTL cache

All three scrapers (`services/macro/scrapers/{fred,ilsole,tradingeconomics}_scraper.py`)
instantiate module-level `CircuitBreaker` + `RateLimiter` from the shared package.

**Gotcha**: transient-error detection (`TRANSIENT_NETWORK_INDICATORS` in
`infrastructure/retry.py`) is a **case-sensitive substring match**. `"Too Many Requests"`
trips the breaker; `"too many requests"` does not.

**yfinance client surface** (`services/market_data/yfinance/`): the facade exposes only what
ingestion uses — `financials`, `analysis`, `holders`, `corporate_actions`, `metadata`,
`search`, plus `fetch_info` / `fetch_history` / `bulk_download` / `fetch_prices_dataframe`.
The `market`, `sectors`, `screener`, `calendars`, `streaming`, and `funds` sub-clients were
deleted — they wrote nothing to the database.

### Shell Scripts (`scheduler/`)

Thin wrappers over the CLI; step order and gating live in `scheduler.py`, not in the scripts.
Both accept `RUNNER=docker` (default, `docker compose exec`) or `RUNNER=local`.

- **`fetch.sh`** → `python -m app.cli daily`
- **`refetch_all.sh`** → `python -m app.cli refetch-all` (universe → yfinance+macro 5y → FRED)

`smoke.sh` was deleted with the endpoints it drove.

### Environment Variables

Configuration via `.env` at project root:
- `DATABASE_URL` — PostgreSQL connection string
- `TRADING_212_API_KEY` — Trading 212 API access. **Absent ⇒ `universe_build` skips without claiming a job slot** (a config state, not a failure)
- `FRED_API_KEY` — Federal Reserve Economic Data
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_API_KEY` — BAML LLM client (news summarize + macro calibrate). Read by BAML directly from `baml_src/clients.baml`, so they are deliberately absent from `app/config.py`

Il Sole 24 Ore and Trading Economics are scraped from HTML — **neither takes an API key**. (`TRADING_ECONOMICS_API_KEY` appeared in the old docs and `.env.example` but nothing ever read it.)

- `NOTIFICATION_WEBHOOK_URL` — Discord/Slack webhook for failure alerts (optional)
- `METRICS_PORT` — Prometheus port (default `9000`); also the container healthcheck target
- `BENCHMARK_TICKERS` — comma-separated reference indices (default: SPY,QQQ,IWM,EFA,EEM,AGG,VGK,VWO,TLT,GLD,URTH,VBINX)
- `SCHEDULER_DAILY_PIPELINE_CRON` — 5-field cron (default `0 7 * * *`)
- `SCHEDULER_MIDDAY_NEWS_CRON` — (default `0 14 * * *`)
- `SCHEDULER_UNIVERSE_BUILD_CRON` — (default `0 2 * * 0`) — must precede the weekly refetch
- `SCHEDULER_WEEKLY_REFETCH_CRON` — (default `0 3 * * 0`)
- `SCHEDULER_FRED_MONTHLY_CRON` — (default `0 8 1 * *`)
- `SCHEDULER_NEWS_REFRESH_INTERVAL_MIN` — minutes (default `30`)
- `SCHEDULER_MISFIRE_GRACE_TIME_SECONDS` — (default `3600`)
- `SCHEDULER_HEARTBEAT_CADENCE_SECONDS` — worker heartbeat cadence (default `30`)
- `SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS` — orphan reaper staleness threshold (default `300`)
- `YFINANCE_FETCH_WORKERS` — parallel fetch workers, 1-16 (default `4`)

### Docker Compose

- `db` — PostgreSQL 16 on host port **54320**
- `adminer` — DB admin UI on host port **18081**
- `scheduler` — the daemon (`python -m app.worker`). No API port; publishes **9000** for Prometheus, which is also the healthcheck target

### Database

~28 tables, all ingestion. The portfolio / execution / factor / risk / rebalancing /
api_keys tables (17 of them) were dropped by migration `d1e2f3a4b5c6`, which is
**destructive and one-way** — its `downgrade()` raises. Restore from a pre-upgrade dump
rather than downgrading.

### Testing

**Daemon tests** (`ingestion/tests/`) use **SQLite in-memory** with `StaticPool`:
- SAVEPOINT pattern (`session.begin_nested()`) so `session.commit()` in app code is rolled back between tests
- There is **no `client` fixture** — no HTTP layer. Tests drive service, repository, and scheduler functions directly
- `patched_session_factory` points `database_manager.get_session` at the test session, for code that opens its own session rather than receiving one

**Gotcha — JSONB columns**: `BackgroundJob` uses `JSON().with_variant(JSONB, "postgresql")` so SQLite tests can create the table. Do NOT use raw `JSONB` in new models that need test coverage.

**Gotcha — coverage floors**: CI enforces line ≥ 80% *and* branch ≥ 0.80 on `app/`. Protocol
stub files are omitted in `ingestion/pyproject.toml` — every member is a `...` body, so coverage
emits an unreachable branch arc per stub.

### Linting & Type Checking

- **ruff**: line-length 88, target py310, rules `E, F, I, N, W, UP, B, SIM, S, RUF, C4, PTH`. Per-file ignores: `N803, N806` for `optimizer/` and `tests/` (sklearn `X, y` convention), `S101` for tests
- **mypy**: strict mode, `ignore_missing_imports = true`. Module overrides relax `disallow_subclassing_any` for sklearn/skfolio base classes
- **pyright**: also run in CI (`pyright[nodejs]==1.1.408`)
- **Dependencies**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `skfolio` (>= 0.20.1), `jinja2` are declared runtime deps in `pyproject.toml`. `arch` is NOT declared — it reaches the bootstrap uncertainty-set classes transitively via skfolio. Code paths importing it directly should guard with `try/except ImportError` or declare it explicitly.
