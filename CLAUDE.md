# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent & Skill Requirements

**MANDATORY**: For any finance-related or code implementation task in this repository, always use:
- `/skfolio` skill — for all portfolio optimization, risk models, and skfolio API usage
- `/yfinance` skill — for all Yahoo Finance data retrieval and yfinance API usage
- `python-pro` agent — for all Python code writing, reviewing, and debugging

These must be loaded proactively, not on request. Any work involving financial data, portfolio optimization, or Python implementation must go through these tools.

## Project Overview

Portfolio optimizer platform with a FastAPI backend (synchronous SQLAlchemy + PostgreSQL), a Typer CLI, and a `skfolio`-based optimization library:

- **`optimizer/`** — Pure-Python optimization library (DB-agnostic, sklearn/skfolio-based)
- **`api/`** — FastAPI application (app factory in `api/app/main.py`, runs on port 8000)
- **`frontend/`** — Angular 21 dashboard (standalone components, signals, Tailwind CSS v4, ECharts)
- **`cli/`** — Typer CLI (`cli/__init__.py` creates the app, entry via `python -m cli`)
- **`theory/`** — LaTeX/Markdown theoretical documentation (not code)

## Build & Run Commands

```bash
# Infrastructure
docker compose up -d              # PostgreSQL (port 54320) + Adminer (port 18081)

# Optimizer library (root)
pip install -e ".[dev]"           # Install optimizer + dev deps (what CI uses)
pytest tests/ -v                  # All optimizer tests
pytest tests/rebalancing/ -v      # Single module tests
pytest -k "test_name"             # Single test by name
ruff check optimizer/ tests/      # Lint (CI step)
ruff check . --fix                # Lint + auto-fix
mypy optimizer/                   # Type check strict mode (CI step)

# API
cd api && pip install -r requirements.txt
alembic upgrade head              # Run migrations
uvicorn app.main:app --reload     # Start dev server on :8000
cd api && pytest                  # API tests

# Frontend
cd frontend && npm install --legacy-peer-deps
cd frontend && ng serve           # Dev server on :4200
cd frontend && ng build           # Production build
cd frontend && ng test            # Unit tests

# CLI
python -m cli --help              # Show all commands
python -m cli db health           # Check DB connectivity

# BAML (regenerate after editing api/baml_src/)
cd api && baml-cli generate
```

## CI Pipeline

`.github/workflows/ci.yml` — triggers on push/PR to `main`, runs on Ubuntu with Python 3.12:
```
pip install -e ".[dev]"    →    ruff check optimizer/ tests/    →    mypy optimizer/    →    pytest tests/ -v --tb=short
```
**Important**: CI installs via `pyproject.toml` (not `requirements.txt`). Any new dependency must be added to **both** `pyproject.toml` `[project.dependencies]` and `requirements.txt`.

## Architecture

### Optimizer Library (`optimizer/`)

Every module follows the same pattern: **frozen `@dataclass` config** + **factory function** + **`str, Enum` types**. Configs hold only primitives/enums/nested frozen dataclasses (serialisable). Non-serialisable objects (estimator instances, numpy arrays, callables) are passed as factory `**kwargs`. This boundary is strict and consistent across all modules.

All transformers follow the sklearn `BaseEstimator + TransformerMixin` API and compose in `sklearn.pipeline.Pipeline`. The pipeline flattens pre-selection + optimiser steps so `get_params()` exposes all nested parameters (e.g. `"optimizer__l2_coef"`, `"drop_correlated__threshold"`).

#### Pipeline flow

```
prices → preprocessing → pre_selection → moments → views →
optimization → validation → tuning → rebalancing → pipeline
```

Plus: `factors/`, `synthetic/`, `scoring/`, `universe/`

#### Module details

- **`preprocessing/`** — sklearn transformers for return data cleaning:
  - `DataValidator` — replaces `inf` and extreme returns with `NaN`
  - `OutlierTreater` — three-group z-score methodology (remove / winsorize / keep)
  - `SectorImputer` — leave-one-out sector-average NaN imputation
  - `RegressionImputer` — OLS regression from top-K correlated assets (`n_neighbors=5`, `min_train_periods=60`); cold-start assets and rows with missing neighbors fall back to `SectorImputer`

- **`pre_selection/`** — `PreSelectionConfig` + `build_preselection_pipeline()` factory assembling sklearn `Pipeline` from config (composes custom transformers with skfolio selectors: `SelectComplete`, `DropZeroVariance`, `DropCorrelated`, `SelectKExtremes`, `SelectNonDominated`, `SelectNonExpiring`)

- **`moments/`** — Moment estimation and prior construction:
  - `MomentEstimationConfig` — selects mu/cov estimators; presets: `for_equilibrium_ledoitwolf`, `for_shrunk_denoised`, `for_adaptive`
  - `build_mu_estimator()` — maps `MuEstimatorType` to skfolio `BaseMu` instances
  - `build_cov_estimator()` — maps `CovEstimatorType` to skfolio `BaseCovariance` instances
  - `build_prior()` — composes mu + cov into `EmpiricalPrior`, optionally wrapping in `FactorModel`
  - `HMMConfig` / `HMMResult` / `fit_hmm()` — Gaussian HMM via `hmmlearn` Baum-Welch EM
  - `HMMBlendedMu` / `HMMBlendedCovariance` — skfolio-compatible estimators using regime-probability-weighted blending. **Gotcha**: `HMMBlendedCovariance` uses full law of total variance (includes cross-state mean dispersion), while `blend_moments_by_regime()` does not — use the class for optimizer inputs
  - `DMMConfig` / `fit_dmm()` / `blend_moments_dmm()` — Deep Markov Model via Pyro SVI. **Optional**: requires `torch`+`pyro` which are NOT in `pyproject.toml`; produces **diagonal covariance only**
  - `apply_lognormal_correction()` / `scale_moments_to_horizon()` — multi-period variance scaling. **Gotcha**: inputs are log-return parameters, output is simple-return space (`E[R_T] = exp(...) - 1`)

- **`views/`** — View integration frameworks:
  - `BlackLittermanConfig` / `build_black_litterman()` — presets: `for_equilibrium`, `for_factor_model`. When inside `FactorModel`, views must reference factor names (e.g. `MTUM`, `QUAL`), not asset names
  - `EntropyPoolingConfig` / `build_entropy_pooling()` — supports mean/variance/correlation/skew/kurtosis/cvar views. Correlation views use format `(ASSET1, ASSET2) == value`
  - `OpinionPoolingConfig` / `build_opinion_pooling()` — expert estimators passed as factory kwarg (not stored in config)
  - `calibrate_omega_from_track_record(view_history, return_history)` — empirical diagonal Ω matrix from forecast error variance; requires ≥5 aligned observations

- **`optimization/`** — Portfolio optimization models:
  - `MeanRiskConfig` + 9 other config types — convex, hierarchical, ensemble optimisers with presets
  - `build_mean_risk()`, `build_risk_budgeting()`, `build_hrp()`, `build_herc()`, `build_nco()`, `build_max_diversification()`, `build_benchmark_tracker()`, `build_equal_weighted()`, `build_inverse_volatility()`, `build_stacking()`
  - `RobustConfig` / `build_robust_mean_risk()` — ellipsoidal μ uncertainty via κ-scaled chi-squared confidence sets + optional bootstrap covariance uncertainty (`arch.StationaryBootstrap`). `kappa=0` recovers standard MeanRisk exactly. Presets: `for_conservative` (κ=2), `for_moderate` (κ=1), `for_aggressive` (κ=0.5), `for_bootstrap_covariance`
  - `DRCVaRConfig` / `build_dr_cvar()` — distributionally robust CVaR over Wasserstein ball. `epsilon=0` falls back to standard empirical CVaR
  - `RegimeRiskConfig` / `build_regime_blended_optimizer()` — HMM-driven regime-conditional risk measure selection. `build_regime_risk_budgeting()` — probability-weighted blending of per-regime budget vectors

- **`synthetic/`** — Vine copula models + synthetic data generation:
  - `VineCopulaConfig` / `SyntheticDataConfig` — presets: `for_scenario_generation`, `for_stress_test`
  - Stress testing: pass `sample_args={"conditioning": {"TICKER": value}}` to `build_synthetic_data()`

- **`validation/`** — Cross-validation:
  - `WalkForwardConfig`, `CPCVConfig`, `MultipleRandomizedCVConfig` — temporal CV configs
  - `run_cross_val()` — defaults to WalkForward (quarterly rolling) when no `cv` is passed

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
  - `optimize()`, `backtest()`, `tune_and_optimize()` — lower-level composable functions

### Frontend (`frontend/`)

Angular 21 single-page dashboard with Tailwind CSS v4 and ECharts for data visualization. Deployed to Vercel.

**Pages**: `dashboard`, `portfolio-builder`, `optimization-studio`, `risk-center`, `factor-research`, `ai-control-room`, `backtesting`, `attribution`, `rebalancing`, `settings`

**Conventions**:
- All components are standalone (Angular 21 default — do NOT set `standalone: true` in decorators)
- Use `ChangeDetectionStrategy.OnPush` on all components
- Use signal-based `input()` / `output()` instead of decorators, `computed()` for derived state
- Use `inject()` instead of constructor injection
- Use native control flow (`@if`, `@for`, `@switch`) instead of structural directives
- Reactive forms over template-driven forms
- Use `class` bindings instead of `ngClass`, `style` bindings instead of `ngStyle`
- Shared reusable components in `shared/` (e.g. `echarts-heatmap`, `echarts-donut`, `tab-group`, `freshness-indicator`)
- Services in `services/` with `providedIn: 'root'` for singletons
- Models/interfaces in `models/`
- `--legacy-peer-deps` required for `npm install` due to dependency conflicts

### Key conventions

- `prices_to_returns()` runs **outside** the pipeline (changes data semantics); pipeline operates on return DataFrames only
- Views use `tuple[str, ...]` in configs (hashable); factories convert to `list` for skfolio
- View configs embed `MomentEstimationConfig` for their inner prior (keeps configs serialisable)
- The fitted prior attribute is `return_distribution_` (not `prior_model_`), containing `mu`, `covariance`, `returns`, `sample_weight`, `cholesky`
- For `BenchmarkTracker`, benchmark returns are passed as `y` in `fit(X, y)`
- When `previous_weights` is passed to `run_full_pipeline()`, it auto-aligns on post-pre-selection universe and re-normalises
- Sector mapping is injected as a plain `dict[str, str]`, not queried from the database

### API Layer (`api/app/`)

Layered architecture: **Routes → Services → Repositories → Models**

- All routes under `/api/v1/`. CLI client (`cli/client.py`) prepends this automatically
- Synchronous SQLAlchemy sessions (`Session`, not `AsyncSession`)
- Repository pattern — all DB queries through typed repositories
- BAML — LLM function definitions in `api/baml_src/`, generated client in `api/baml_client/` (do not edit generated files)
- PostgreSQL 16 on port **54320** (not 5432). Connection: `postgresql://postgres:postgres@localhost:54320/optimizer_db`
- Two separate `requirements.txt` and `pyproject.toml` — root for optimizer library, `api/` for FastAPI app

### Scheduler & Background Jobs

APScheduler runs inside the FastAPI process (started in lifespan), with a `SQLAlchemyJobStore` so misfired runs execute at next startup within a configurable grace window.

**Five scheduled jobs** (all configurable via `SCHEDULER_*` env vars):

| Job | Trigger | Default | Description |
|-----|---------|---------|-------------|
| `daily_pipeline` | Cron | `0 7 * * *` | Sequential: yfinance → macro → news → summarize → calibrate |
| `midday_news` | Cron | `0 14 * * *` | Afternoon news + summarize refresh |
| `weekly_refetch` | Cron | `0 3 * * 0` | Full yfinance + macro rebuild (5-year history) |
| `fred_monthly` | Cron | `0 8 1 * *` | FRED economic data fetch |
| `news_refresh` | Interval | Every 30 min | Incremental news re-summarization |

**BackgroundJobService** (`api/app/services/background_job.py`) — instantiated once per job domain (e.g. `macro_fetch`, `fred_fetch`, `yfinance_fetch`) at module level in route files:
- `create_job()` — atomically claims a slot; raises `JobAlreadyRunningError` if one is already pending/running
- `get_job(job_id)` → dict — returns job state for polling endpoints
- `update_job(job_id, **kwargs)` — updates status, progress, errors
- `start_background(target, args)` — launches worker in daemon thread

**Route pattern for async work** (used in `macro_regime.py`, `yfinance_data.py`, `macro_calibration.py`, `trading212.py`):
1. Module-level `_job_service = BackgroundJobService(job_type="...", session_factory=database_manager.get_session)`
2. POST endpoint → `create_job()` → `start_background(wrapper_fn)` → return `202 + job_id`
3. Wrapper function: set `status="running"` → call service function with `on_progress=make_progress(job_id, svc)` → catch errors → set `status="completed"` or `"failed"`
4. GET endpoint → `get_job(job_id)` → return progress

**Progress callback**: `make_progress(job_id, job_svc)` in `api/app/services/_progress.py` returns a closure forwarding kwargs to `update_job`. Service functions accept `on_progress=` to report `current=`, `total=`, `current_country=`, etc.

**Persistence**: `background_jobs` table (UUID PK, `job_type` + `status` composite index, JSONB `extra`/`result`/`errors` columns). Model uses `JSON().with_variant(JSONB, "postgresql")` for SQLite test compatibility.

**Observability**:
- Prometheus metrics (`api/app/metrics.py`): counters `jobs_started_total`, `jobs_completed_total`, `jobs_failed_total`; histogram `job_duration_seconds`; gauge `jobs_in_progress` — all labeled by `domain`. Exposed at `/metrics`
- Webhook notifications (`api/app/services/notifications.py`): Discord/Slack-compatible POST on job failure when `NOTIFICATION_WEBHOOK_URL` is set
- Read-only jobs API: `GET /api/v1/jobs` (domain/status filtering, pagination), `GET /api/v1/jobs/{job_id}`

### Shared Infrastructure (`api/app/services/infrastructure/`)

Generalized resilience primitives extracted from yfinance, re-exported as shims in `api/app/services/yfinance/infrastructure/`:
- **`CircuitBreaker`** — exponential backoff (2^attempt), max_attempts safety limit, service-name in errors
- **`RateLimiter`** — thread-safe per-key delay enforcement (default 0.1s)
- **`retry_with_backoff()`** — retry with full-jitter exponential backoff, transient error detection
- **`LRUCache`** — in-memory TTL cache

All three scrapers (`fred_scraper.py`, `ilsole_scraper.py`, `tradingeconomics_scraper.py`) instantiate module-level `CircuitBreaker` + `RateLimiter` from the shared package.

### Shell Scripts (`scheduler/`)

- **`fetch.sh`** — daily pipeline: health check → fire-and-poll yfinance → macro → news → summarize → calibrate (with dependency enforcement and 409 conflict handling). `MAX_POLL_SECONDS` env var (default 21600 = 6h)
- **`refetch_all.sh`** — full re-fetch: adds universe + fred stages. `MAX_POLL_SECONDS` default 21600

### Environment Variables

Configuration via `.env` at project root:
- `DATABASE_URL` — PostgreSQL connection string
- `TRADING_212_API_KEY` — Trading 212 API access
- `FRED_API_KEY` — Federal Reserve Economic Data
- `TRADING_ECONOMICS_API_KEY` — Trading Economics data
- `NOTIFICATION_WEBHOOK_URL` — Discord/Slack webhook for failure alerts (optional)
- `SCHEDULER_DAILY_PIPELINE_CRON` — 5-field cron (default `0 7 * * *`)
- `SCHEDULER_MIDDAY_NEWS_CRON` — (default `0 14 * * *`)
- `SCHEDULER_WEEKLY_REFETCH_CRON` — (default `0 3 * * 0`)
- `SCHEDULER_FRED_MONTHLY_CRON` — (default `0 8 1 * *`)
- `SCHEDULER_NEWS_REFRESH_INTERVAL_MIN` — minutes (default `30`)
- `SCHEDULER_MISFIRE_GRACE_TIME_SECONDS` — (default `3600`)

### Docker Compose

- `db` — PostgreSQL 16 on host port **54320**
- `adminer` — DB admin UI on host port **18081**
- `api` — FastAPI on host port **8005** (port 8000 is reserved on the host). Passes all `SCHEDULER_*` and `NOTIFICATION_WEBHOOK_URL` env vars
- `frontend` — Angular on host port **4300**

### Testing

**API tests** (`api/tests/`) use **SQLite in-memory** with `StaticPool`:
- SAVEPOINT pattern (`session.begin_nested()`) so `session.commit()` in app code is rolled back between tests
- `client` fixture overrides FastAPI `get_db` dependency to inject the test session

**Gotcha — BackgroundJobService in tests**: The service uses `database_manager.get_session` (bypasses FastAPI DI), so it does NOT use the test SQLite session. Tests that trigger background job creation/polling must mock `create_job`, `get_job`, and `start_background` on the module-level service instance (e.g. `patch("app.api.v1.macro_regime._summarize_job_service.create_job", ...)`).

**Gotcha — JSONB columns**: The `BackgroundJob` model uses `JSON().with_variant(JSONB, "postgresql")` so SQLite tests can create the table. Do NOT use raw `JSONB` type in new models that need test coverage.

### Linting & Type Checking

- **ruff**: line-length 88, target py310, rules `E, F, I, N, W, UP`. Per-file ignores: `N803, N806` for `optimizer/` and `tests/` (sklearn `X, y` convention)
- **mypy**: strict mode, `ignore_missing_imports = true`. Module overrides relax `disallow_subclassing_any` for sklearn/skfolio base classes. DMM module has broader relaxation for torch/pyro stubs
- **Dependencies**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `skfolio`, `hmmlearn`, `arch` are runtime deps in `pyproject.toml`. `torch`/`pyro` (for DMM) are **not declared** — DMM is effectively optional
