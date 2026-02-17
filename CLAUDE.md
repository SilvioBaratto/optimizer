# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent & Skill Requirements

**MANDATORY**: For any finance-related or code implementation task in this repository, always use:
- `/skfolio` skill — for all portfolio optimization, risk models, and skfolio API usage
- `/yfinance` skill — for all Yahoo Finance data retrieval and yfinance API usage
- `python-pro` agent — for all Python code writing, reviewing, and debugging

These must be loaded proactively, not on request. Any work involving financial data, portfolio optimization, or Python implementation must go through these tools.

## Project Overview

Portfolio optimizer platform with a FastAPI backend (synchronous SQLAlchemy + PostgreSQL), a Typer CLI that talks to the API (or falls back to direct DB access), and a `skfolio`-based optimization library. The codebase is organized into three top-level packages:

- **`api/`** — FastAPI application (app factory in `api/app/main.py`, runs on port 8000)
- **`cli/`** — Typer CLI (`cli/__init__.py` creates the app, entry via `python -m cli`)
- **`optimizer/`** — Pure-Python optimization library (DB-agnostic, sklearn/skfolio-based)
- **`theory/`** — LaTeX/Markdown theoretical documentation (not code)

## Build & Run Commands

### Infrastructure
```bash
docker compose up -d              # Start PostgreSQL (port 54320) + Adminer (port 18080)
```

### API Server
```bash
cd api
pip install -r requirements.txt
alembic upgrade head              # Run migrations
uvicorn app.main:app --reload     # Start dev server on :8000
```

### CLI
```bash
pip install -r requirements.txt   # Root requirements
python -m cli --help              # Show all commands
python -m cli db health           # Check DB connectivity
python -m cli universe stats      # Universe statistics
python -m cli yfinance fetch      # Bulk fetch yfinance data
python -m cli macro fetch         # Fetch macro data
```

### Tests
```bash
# Optimizer library tests (root)
pytest tests/ -v                  # All optimizer library tests

# API tests
cd api
pytest                            # Run all tests
pytest tests/unit/                # Unit tests only
pytest tests/integration/         # Integration tests only
pytest tests/e2e/                 # E2E tests only
pytest -k "test_name"             # Run single test by name
pytest -m slow                    # Run only slow-marked tests
```

### Linting & Type Checking (root optimizer package)
```bash
ruff check .                      # Lint
ruff check . --fix                # Lint + auto-fix
mypy .                            # Type check (strict mode)
```

### Database Migrations
```bash
cd api
alembic revision --autogenerate -m "description"   # Create migration
alembic upgrade head                                # Apply all
alembic downgrade -1                                # Rollback one
```

### BAML Client (regenerate after editing `api/baml_src/`)
```bash
cd api
baml-cli generate
```

## Architecture

### API Layer (`api/app/`)

Follows a layered architecture: **Routes → Services → Repositories → Models**

- `api/v1/router.py` — aggregates all v1 routers (trading212, yfinance_data, macro_regime, database)
- `services/` — business logic. The yfinance service is decomposed into sub-modules under `services/yfinance/` (ticker, market, news, infrastructure with cache/circuit-breaker/rate-limiter)
- `repositories/` — SQLAlchemy query layer. `base.py` has a generic `BaseRepository`
- `models/` — SQLAlchemy ORM models inheriting from `BaseModel` (UUID PK + timestamps via mixins in `models/base.py`)
- `schemas/` — Pydantic v2 request/response schemas
- `middleware/` — security headers, logging, rate limiting
- `dependencies.py` — DI: `get_db` session, `CurrentUser`/`DBSession` type aliases, `PaginationParams`, `RateLimiter`
- `database.py` — synchronous `DatabaseManager` with connection pooling (psycopg2, QueuePool)
- `config.py` — `pydantic-settings` based `Settings` class reading from `.env`

### CLI Layer (`cli/`)

Typer command groups (`db`, `universe`, `yfinance`, `macro`) that call API endpoints via `cli/client.py` (httpx). The `yfinance` commands fall back to `cli/direct_fetch.py` for direct DB access when the API is unavailable.

### Database

- PostgreSQL 16 via Docker on port **54320** (not 5432)
- Alembic migrations in `api/alembic/versions/`
- Default connection: `postgresql://postgres:postgres@localhost:54320/optimizer_db`
- Key domain tables: Exchange, Instrument, TickerProfile, PriceHistory, FinancialStatement, AnalystRecommendation, EconomicIndicator, BondYield, TradingEconomicsIndicator

### Key Patterns

- **Synchronous SQLAlchemy** — the API uses sync sessions (`Session`, not `AsyncSession`) despite being a FastAPI app
- **Repository pattern** — all DB queries go through typed repositories, not raw session calls in routes
- **BAML** — LLM function definitions in `api/baml_src/`, generated client in `api/baml_client/` (do not edit generated files)
- **Two separate requirements.txt** — root `requirements.txt` for the optimizer library + CLI; `api/requirements.txt` for the FastAPI app
- **Two separate pyproject.toml** — root configures the `optimizer` package (skfolio-based); `api/pyproject.toml` configures the API package

### Environment Variables

Configuration via `.env` at project root. Key variables:
- `DATABASE_URL` — PostgreSQL connection string
- `TRADING_212_API_KEY` — Trading 212 API access
- `FRED_API_KEY` — Federal Reserve Economic Data
- `TRADING_ECONOMICS_API_KEY` — Trading Economics data

### Optimizer Library (`optimizer/`)

Pure-Python, DB-agnostic library for data preparation and portfolio optimization. All transformers follow the sklearn `BaseEstimator + TransformerMixin` API and compose in `sklearn.pipeline.Pipeline`.

- `preprocessing/` — Custom transformers for return data cleaning:
  - `DataValidator` — replaces `inf` and extreme returns with `NaN`
  - `OutlierTreater` — three-group z-score methodology (remove / winsorize / keep)
  - `SectorImputer` — leave-one-out sector-average NaN imputation
- `pre_selection/` — Pipeline assembly and configuration:
  - `PreSelectionConfig` — frozen dataclass with all pipeline hyper-parameters
  - `build_preselection_pipeline()` — factory that assembles an sklearn `Pipeline` from config, composing custom transformers with skfolio selectors (`SelectComplete`, `DropZeroVariance`, `DropCorrelated`, `SelectKExtremes`, `SelectNonDominated`, `SelectNonExpiring`)
- `moments/` — Moment estimation and prior construction (typed config + factory layer over skfolio estimators):
  - `MomentEstimationConfig` — frozen dataclass selecting mu estimator, covariance estimator, and prior assembly options; includes factory presets (`for_equilibrium_ledoitwolf`, `for_shrunk_denoised`, `for_adaptive`)
  - `MuEstimatorType` / `CovEstimatorType` / `ShrinkageMethod` — `str, Enum` enums for estimator selection
  - `build_mu_estimator()` — factory mapping `MuEstimatorType` to skfolio `BaseMu` instances (`EmpiricalMu`, `ShrunkMu`, `EWMu`, `EquilibriumMu`)
  - `build_cov_estimator()` — factory mapping `CovEstimatorType` to skfolio `BaseCovariance` instances (`EmpiricalCovariance`, `LedoitWolf`, `OAS`, `ShrunkCovariance`, `EWCovariance`, `GerberCovariance`, `GraphicalLassoCV`, `DenoiseCovariance`, `DetoneCovariance`, `ImpliedCovariance`)
  - `build_prior()` — composes mu + cov into `EmpiricalPrior`, optionally wrapping in `FactorModel`
- `views/` — View integration frameworks (typed config + factory layer over skfolio prior estimators):
  - `BlackLittermanConfig` — frozen dataclass for Black-Litterman prior configuration; includes factory presets (`for_equilibrium`, `for_factor_model`)
  - `EntropyPoolingConfig` — frozen dataclass for Entropy Pooling prior configuration; includes factory presets (`for_mean_views`, `for_stress_test`)
  - `OpinionPoolingConfig` — frozen dataclass for Opinion Pooling prior configuration (estimator objects passed separately to factory)
  - `ViewUncertaintyMethod` — `str, Enum` for BL uncertainty calibration (`HE_LITTERMAN`, `IDZOREK`)
  - `build_black_litterman()` — factory building `BlackLitterman` from config, optionally wrapping in `FactorModel`
  - `build_entropy_pooling()` — factory building `EntropyPooling` from config with mean/variance/correlation/skew/kurtosis/cvar views
  - `build_opinion_pooling()` — factory building `OpinionPooling` from named expert estimators + config
- `optimization/` — Portfolio optimization models (typed config + factory layer over skfolio optimisers):
  - `ObjectiveFunctionType` / `RiskMeasureType` / `ExtraRiskMeasureType` / `DistanceType` / `LinkageMethodType` / `RatioMeasureType` — `str, Enum` enums mapping to skfolio counterparts
  - `DistanceConfig` / `ClusteringConfig` — reusable frozen-dataclass sub-configs for distance and clustering estimators
  - `MeanRiskConfig` — frozen dataclass for `MeanRisk`; includes `transaction_costs`, `management_fees`, `max_tracking_error`; includes presets (`for_min_variance`, `for_max_sharpe`, `for_max_utility`, `for_min_cvar`, `for_efficient_frontier`)
  - `RiskBudgetingConfig` — frozen dataclass for `RiskBudgeting`; includes presets (`for_risk_parity`, `for_cvar_parity`)
  - `MaxDiversificationConfig` — frozen dataclass for `MaximumDiversification`
  - `HRPConfig` / `HERCConfig` — frozen dataclasses for hierarchical methods; include presets (`for_variance`, `for_cvar`)
  - `NCOConfig` — frozen dataclass for `NestedClustersOptimization`
  - `BenchmarkTrackerConfig` — frozen dataclass for `BenchmarkTracker` (minimises tracking error; benchmark returns passed as `y` in `fit(X, y)`)
  - `EqualWeightedConfig` — frozen dataclass for `EqualWeighted` (1/N allocation, no parameters)
  - `InverseVolatilityConfig` — frozen dataclass for `InverseVolatility` (inverse-vol weighting)
  - `StackingConfig` — frozen dataclass for `StackingOptimization` (ensemble meta-optimiser; `estimators` and `final_estimator` passed as factory kwargs)
  - `build_distance_estimator()` / `build_clustering_estimator()` — helper factories for sub-components
  - `build_mean_risk()` / `build_risk_budgeting()` / `build_max_diversification()` — convex optimiser factories
  - `build_hrp()` / `build_herc()` / `build_nco()` — hierarchical/cluster optimiser factories
  - `build_benchmark_tracker()` — benchmark tracking factory
  - `build_equal_weighted()` / `build_inverse_volatility()` — naive baseline factories
  - `build_stacking()` — ensemble stacking factory
- `synthetic/` — Synthetic data generation and vine copula models (typed config + factory layer):
  - `DependenceMethodType` / `SelectionCriterionType` — `str, Enum` enums for vine copula configuration
  - `VineCopulaConfig` — frozen dataclass for `VineCopula` (marginal fitting, tree depth, dependence method, selection criterion)
  - `SyntheticDataConfig` — frozen dataclass for `SyntheticData`; embeds `VineCopulaConfig`; includes presets (`for_scenario_generation`, `for_stress_test`)
  - `build_vine_copula()` — factory building `VineCopula` from config; non-serialisable `marginal_candidates`, `copula_candidates`, `central_assets` passed as kwargs
  - `build_synthetic_data()` — factory building `SyntheticData` from config; `distribution_estimator` and `sample_args` (for conditional stress testing via `{"conditioning": {"AAPL": -0.10}}`) passed as kwargs
- `validation/` — Model selection and cross-validation (typed config + factory layer over skfolio cross-validators):
  - `WalkForwardConfig` — frozen dataclass for `WalkForward` (rolling/expanding windows with purge); includes presets (`for_monthly_rolling`, `for_quarterly_rolling`, `for_quarterly_expanding`)
  - `CPCVConfig` — frozen dataclass for `CombinatorialPurgedCV` (n_folds, n_test_folds, purge, embargo); includes presets (`for_statistical_testing`, `for_small_sample`)
  - `MultipleRandomizedCVConfig` — frozen dataclass for `MultipleRandomizedCV` (dual randomisation over time and assets); embeds `WalkForwardConfig`; includes preset (`for_robustness_check`)
  - `build_walk_forward()` / `build_cpcv()` / `build_multiple_randomized_cv()` — factory functions building skfolio cross-validators from config
  - `run_cross_val()` — convenience wrapper around `skfolio.model_selection.cross_val_predict` that enforces temporal splitting; returns `MultiPeriodPortfolio` (WalkForward) or `Population` (CPCV/MultipleRandomizedCV)
  - `compute_optimal_folds()` — wraps `optimal_folds_number` for CPCV calibration
- `scoring/` — Performance scoring for model selection (typed config + factory layer over skfolio metrics):
  - `ScorerConfig` — frozen dataclass selecting a `RatioMeasureType` or custom callable; includes presets (`for_sharpe`, `for_sortino`, `for_calmar`, `for_cvar_ratio`, `for_custom`)
  - `build_scorer()` — factory building a scorer callable from config; custom callables passed as `score_func` kwarg when `ratio_measure` is `None`
- `tuning/` — Hyperparameter tuning with temporal cross-validation (typed config + factory layer over sklearn search estimators):
  - `GridSearchConfig` — frozen dataclass embedding `WalkForwardConfig` + `ScorerConfig`; includes presets (`for_quick_search`, `for_thorough_search`)
  - `RandomizedSearchConfig` — frozen dataclass with `n_iter`, `random_state`, embedded `WalkForwardConfig` + `ScorerConfig`; includes presets (`for_quick_search`, `for_thorough_search`)
  - `build_grid_search_cv()` — factory building `GridSearchCV` with temporal CV from config; takes `estimator` + `param_grid`
  - `build_randomized_search_cv()` — factory building `RandomizedSearchCV` with temporal CV from config; takes `estimator` + `param_distributions`
- `rebalancing/` — Rebalancing frameworks (calendar-based, threshold-based, turnover computation):
  - `RebalancingFrequency` — `str, Enum` for calendar frequencies (MONTHLY=21, QUARTERLY=63, SEMIANNUAL=126, ANNUAL=252 trading days)
  - `ThresholdType` — `str, Enum` for drift threshold conventions (ABSOLUTE, RELATIVE)
  - `CalendarRebalancingConfig` — frozen dataclass with frequency; `trading_days` property; includes presets (`for_monthly`, `for_quarterly`, `for_annual`)
  - `ThresholdRebalancingConfig` — frozen dataclass with threshold_type + threshold value; includes presets (`for_absolute`, `for_relative`)
  - `compute_drifted_weights()` — computes weights after one period of returns
  - `compute_turnover()` — one-way turnover between current and target weights
  - `compute_rebalancing_cost()` — total transaction cost of rebalancing (uniform or per-asset costs)
  - `should_rebalance()` — checks whether any asset breaches drift threshold (absolute or relative)

**Usage pattern**: `prices_to_returns()` runs *outside* the pipeline (changes data semantics); the pipeline operates on return DataFrames only. Sector mapping is injected as a plain `dict[str, str]`, not queried from the database. The fitted prior attribute is `return_distribution_` (not `prior_model_`), containing `mu`, `covariance`, `returns`, `sample_weight`, and `cholesky`. Views use `tuple[str, ...]` in configs (hashable, frozen-dataclass-friendly); factory functions convert to `list` for skfolio. View configs embed `MomentEstimationConfig` for their inner prior (not raw `BasePrior`), keeping configs serialisable. For `OpinionPooling`, expert estimators are passed as a factory argument (not stored in the frozen dataclass) since they are not serialisable. Correlation views in Entropy Pooling use the format `(ASSET1, ASSET2) == value`. When using `BlackLitterman` inside a `FactorModel`, views must reference factor names (e.g. `MTUM`, `QUAL`), not asset names. Optimization configs follow the same serialisability boundary: configs hold only primitives/enums/nested frozen dataclasses; non-serialisable objects (`prior_estimator`, `previous_weights`, `risk_budget`, `inner_estimator`/`outer_estimator`, `estimators`/`final_estimator`, `groups`, `linear_constraints`) are passed as factory keyword arguments. Optimization configs embed `MomentEstimationConfig` for their inner prior; when `prior_estimator` is passed explicitly to the factory, it overrides config-built priors. For `BenchmarkTracker`, benchmark returns are passed as `y` in `fit(X, y)`. For `StackingOptimization`, the default `estimators` are `[("mean_risk", MeanRisk()), ("hrp", HierarchicalRiskParity())]`. For synthetic data stress testing, pass `sample_args={"conditioning": {"TICKER": value}}` to `build_synthetic_data()`. Validation configs follow the same serialisability boundary: all configs are frozen dataclasses with primitives only. `run_cross_val()` defaults to `WalkForward` (quarterly rolling) when no `cv` is passed. `build_grid_search_cv()` and `build_randomized_search_cv()` enforce temporal CV by default (walk-forward, not random shuffle). For `GridSearchCV`/`RandomizedSearchCV`, use sklearn double-underscore notation for nested parameter tuning (e.g. `"prior_estimator__mu_estimator__alpha"`). `CalendarRebalancingConfig.trading_days` maps frequency to trading days; use with `WalkForwardConfig.test_size` for aligned backtesting. `should_rebalance()` handles zero-weight targets safely in relative mode. Transaction costs in `MeanRiskConfig` integrate with rebalancing via `previous_weights` factory kwarg.

### API Prefix

All API routes are served under `/api/v1/`. The CLI client (`cli/client.py`) prepends this automatically.
