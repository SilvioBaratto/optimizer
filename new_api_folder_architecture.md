# API Folder Architecture — Proposed Reorganization

**Status:** proposal (no code or import changes yet).
**Scope:** `api/app/api/v1/`, `api/app/models/`, `api/app/repositories/`, `api/app/services/`, `api/app/schemas/`.
**Goal:** make files findable by folder name (domain) instead of by file name.

## Why

`api/app/` has accumulated ~170 files across four flat domain layers (`api/v1/`, `models/`, `repositories/`, `services/`) plus `schemas/`. Today, locating a file requires remembering its name (e.g. `services/macro_calibration.py`, `services/factor_scoring_service.py`). After the reorg, navigating to a domain folder (`services/macro/`, `services/factors/`) is enough — the file inside is then trivial to choose.

## Design choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Top-level shape | **Layer-first** with domain sub-folders inside each layer | Preserves the `Routes → Services → Repositories → Models` convention from `CLAUDE.md`. Smallest blast radius. |
| Cluster granularity | **Fine (17 clusters)** | Optimises for "land on the right folder by name". Tolerates small folders (e.g. `auth/` = 1 file). |
| Shared helpers | **Dedicated `_shared/` sub-folder per layer** | Makes cross-cutting utilities discoverable; avoids burying them inside an arbitrary domain. |

## Layer parity rule

These 16 domain names appear in the same form across every layer (a layer omits domains it doesn't own):

```
auth, market_data, universe, macro, portfolio, optimization, backtest,
factors, risk, rebalancing, views, attribution, dashboard, scenarios,
reports, jobs
```

Plus `execution/` under `models/` + `repositories/` only (shared persistence parent of `optimization` + `backtest`). `execution/` is **deliberately absent** from `api/v1/`, `services/`, and `schemas/` — those layers route through `optimization/` and `backtest/` directly.

### Asymmetries (intentional)

- `auth/` exists only under `models/`. Middleware lives in `middleware/auth.py` (transport concern, not a domain).
- `execution/` exists only under `models/` + `repositories/`.
- `attribution/`, `dashboard/`, `views/`, `scenarios/`, `reports/` have no `models/<domain>/` folder — these clusters do not own ORM tables (they read from other domains' models).
- `views/` has `repositories/views/` (for `view_generation_repository`) but no `models/views/` — view generation reads `Instrument` + `PriceHistory` + `TickerProfile` from `market_data/` + `universe/`.
- `repositories/macro/sentiment_repository.py` exists without a corresponding `models/macro/sentiment.py` — sentiment data is sourced via the existing `MacroNews` model from `models/macro/macro_regime.py`; no separate ORM class.

## Top-level tree

```
api/app/
├── api/v1/
│   ├── _shared/              # router.py, metrics.py, database.py, test.py
│   ├── market_data/          # yfinance_data.py, reference_indices.py
│   ├── universe/             # trading212.py, universe_screen.py
│   ├── macro/                # macro_regime.py, macro_calibration.py
│   ├── portfolio/            # portfolio.py
│   ├── optimization/         # optimize.py, tune.py, validate.py
│   ├── backtest/             # backtest.py
│   ├── factors/              # factors.py
│   ├── risk/                 # risk.py, risk_analytics.py
│   ├── rebalancing/          # rebalance.py, rebalance_policy.py
│   ├── views/                # views.py, opinion_pooling.py, llm_moments.py
│   ├── attribution/          # attribution.py
│   ├── dashboard/            # dashboard.py
│   ├── scenarios/            # stress_scenarios.py, synthetic.py
│   ├── reports/              # reports.py
│   └── jobs/                 # jobs.py, scheduler.py
├── models/
│   ├── _shared/              # base.py
│   ├── auth/                 # api_key.py
│   ├── market_data/          # yfinance_data.py
│   ├── universe/             # universe.py
│   ├── macro/                # macro_regime.py
│   ├── portfolio/            # portfolio.py
│   ├── execution/            # execution.py
│   ├── factors/              # factor.py
│   ├── risk/                 # risk.py
│   ├── rebalancing/          # rebalancing.py
│   └── jobs/                 # background_job.py
├── repositories/
│   ├── _shared/              # base.py, database_admin_repository.py
│   ├── market_data/          # yfinance_repository.py
│   ├── universe/             # universe_repository.py
│   ├── macro/                # macro_regime_repository.py, sentiment_repository.py
│   ├── portfolio/            # portfolio_repository.py
│   ├── execution/            # execution_repository.py
│   ├── factors/              # factor_repository.py
│   ├── risk/                 # risk_repository.py
│   ├── rebalancing/          # rebalancing_repository.py
│   ├── views/                # view_generation_repository.py
│   ├── dashboard/            # dashboard_repository.py
│   └── jobs/                 # background_job_repository.py
├── services/
│   ├── _shared/              # _price_fetcher.py, _sector_resolver.py, _json_safe.py,
│   │                         # _progress.py, _benchmark_bootstrap.py,
│   │                         # trading_calendar.py, notifications.py
│   ├── infrastructure/       # (already structured — leave in place)
│   ├── market_data/
│   │   ├── yfinance/         # (existing services/yfinance/ pkg moves here)
│   │   ├── yfinance_data_service.py
│   │   └── reference_index_seeder.py
│   ├── universe/
│   │   ├── trading212/       # (existing services/trading212/ pkg moves here)
│   │   └── universe_screening_service.py
│   ├── macro/
│   │   ├── scrapers/         # (existing services/scrapers/ pkg moves here)
│   │   ├── macro_regime_service.py
│   │   ├── macro_calibration.py
│   │   └── macro_news_summary.py
│   ├── portfolio/            # broker_sync_service.py
│   ├── optimization/         # optimization_service.py, tuning_service.py, validation_service.py
│   ├── backtest/             # backtest_service.py
│   ├── factors/              # factor_service.py, factor_compute_service.py,
│   │                         # factor_scoring_service.py, factor_analysis_service.py,
│   │                         # _factor_helpers.py
│   ├── risk/                 # risk_analytics_service.py
│   ├── rebalancing/          # rebalancing_service.py
│   ├── views/                # view_generation.py, entropy_pooling_service.py,
│   │                         # opinion_pooling.py, llm_moments.py, sentiment.py
│   ├── attribution/          # attribution_service.py
│   ├── dashboard/            # dashboard_service.py
│   ├── scenarios/            # stress_scenarios.py, synthetic_service.py
│   ├── reports/              # report_service.py
│   └── jobs/                 # background_job.py, scheduler.py
├── schemas/
│   ├── _shared/              # base.py, base_job.py
│   ├── market_data/          # yfinance_data.py, reference_index.py
│   ├── universe/             # trading212.py, universe_screen.py
│   ├── macro/                # macro_regime.py
│   ├── portfolio/            # portfolio.py
│   ├── optimization/         # optimization.py, tuning.py, validation.py
│   ├── backtest/             # backtest.py
│   ├── factors/              # factors.py
│   ├── risk/                 # risk.py, risk_analytics.py
│   ├── rebalancing/          # rebalancing.py
│   ├── views/                # views.py, llm_moments.py
│   ├── attribution/          # attribution.py
│   ├── dashboard/            # dashboard.py
│   ├── scenarios/            # stress_scenarios.py, synthetic.py
│   ├── reports/              # reports.py
│   └── jobs/                 # jobs.py, scheduler.py
├── middleware/               # (already structured — leave in place)
├── utils/                    # (already structured — leave in place)
├── core/                     # (currently empty — leave or remove)
├── config.py
├── database.py
├── dependencies.py
├── exceptions.py
├── metrics.py
├── main.py
└── __init__.py
```

## Cluster details (file-level mapping)

### 1. market_data — Yahoo Finance ingest + benchmark seeding
| Source | Target |
|--------|--------|
| `api/v1/yfinance_data.py` | `api/v1/market_data/yfinance_data.py` |
| `api/v1/reference_indices.py` | `api/v1/market_data/reference_indices.py` |
| `models/yfinance_data.py` | `models/market_data/yfinance_data.py` |
| `repositories/yfinance_repository.py` | `repositories/market_data/yfinance_repository.py` |
| `services/yfinance/` (pkg) | `services/market_data/yfinance/` |
| `services/yfinance_data_service.py` | `services/market_data/yfinance_data_service.py` |
| `services/reference_index_seeder.py` | `services/market_data/reference_index_seeder.py` |
| `schemas/yfinance_data.py` | `schemas/market_data/yfinance_data.py` |
| `schemas/reference_index.py` | `schemas/market_data/reference_index.py` |

### 2. universe — Instrument universe + Trading 212 + investability screening
| Source | Target |
|--------|--------|
| `api/v1/trading212.py` | `api/v1/universe/trading212.py` |
| `api/v1/universe_screen.py` | `api/v1/universe/universe_screen.py` |
| `models/universe.py` | `models/universe/universe.py` |
| `repositories/universe_repository.py` | `repositories/universe/universe_repository.py` |
| `services/trading212/` (pkg) | `services/universe/trading212/` |
| `services/universe_screening_service.py` | `services/universe/universe_screening_service.py` |
| `schemas/trading212.py` | `schemas/universe/trading212.py` |
| `schemas/universe_screen.py` | `schemas/universe/universe_screen.py` |

### 3. macro — Macro regime data + FRED + scrapers + news summarisation + LLM calibration
| Source | Target |
|--------|--------|
| `api/v1/macro_regime.py` | `api/v1/macro/macro_regime.py` |
| `api/v1/macro_calibration.py` | `api/v1/macro/macro_calibration.py` |
| `models/macro_regime.py` | `models/macro/macro_regime.py` |
| `repositories/macro_regime_repository.py` | `repositories/macro/macro_regime_repository.py` |
| `repositories/sentiment_repository.py` | `repositories/macro/sentiment_repository.py` |
| `services/macro_regime_service.py` | `services/macro/macro_regime_service.py` |
| `services/macro_calibration.py` | `services/macro/macro_calibration.py` |
| `services/macro_news_summary.py` | `services/macro/macro_news_summary.py` |
| `services/scrapers/` (pkg) | `services/macro/scrapers/` |
| `schemas/macro_regime.py` | `schemas/macro/macro_regime.py` |

Note: `macro_calibration.py` route registers under the `/views/macro-calibration` URL prefix but lives in `macro/` by **data ownership** (depends on `MacroRegimeRepository`). Folder location follows ownership, not URL prefix.

### 4. portfolio — Portfolio CRUD + snapshots + broker sync
| Source | Target |
|--------|--------|
| `api/v1/portfolio.py` | `api/v1/portfolio/portfolio.py` |
| `models/portfolio.py` | `models/portfolio/portfolio.py` |
| `repositories/portfolio_repository.py` | `repositories/portfolio/portfolio_repository.py` |
| `services/broker_sync_service.py` | `services/portfolio/broker_sync_service.py` |
| `schemas/portfolio.py` | `schemas/portfolio/portfolio.py` |

### 5. optimization — Optimize + tune + validate (sync/async portfolio optimisation)
| Source | Target |
|--------|--------|
| `api/v1/optimize.py` | `api/v1/optimization/optimize.py` |
| `api/v1/tune.py` | `api/v1/optimization/tune.py` |
| `api/v1/validate.py` | `api/v1/optimization/validate.py` |
| `services/optimization_service.py` | `services/optimization/optimization_service.py` |
| `services/tuning_service.py` | `services/optimization/tuning_service.py` |
| `services/validation_service.py` | `services/optimization/validation_service.py` |
| `schemas/optimization.py` | `schemas/optimization/optimization.py` |
| `schemas/tuning.py` | `schemas/optimization/tuning.py` |
| `schemas/validation.py` | `schemas/optimization/validation.py` |

### 6. backtest — Walk-forward backtest jobs
| Source | Target |
|--------|--------|
| `api/v1/backtest.py` | `api/v1/backtest/backtest.py` |
| `services/backtest_service.py` | `services/backtest/backtest_service.py` |
| `schemas/backtest.py` | `schemas/backtest/backtest.py` |

### 7. execution — Persistence parent for optimization + backtest (data-layer only)
| Source | Target |
|--------|--------|
| `models/execution.py` | `models/execution/execution.py` |
| `repositories/execution_repository.py` | `repositories/execution/execution_repository.py` |

`execution/` exists under `models/` + `repositories/` only. The `OptimizationRun` and `BacktestRun` ORM rows share a model + repository (both are persisted run records); the user-facing routes + schemas stay split into `optimization/` and `backtest/` because they are distinct entry points. **This is the only cross-cluster split in the proposal.**

### 8. factors — Factor research pipeline (compute, score, validate, select, integrate)
| Source | Target |
|--------|--------|
| `api/v1/factors.py` | `api/v1/factors/factors.py` |
| `models/factor.py` | `models/factors/factor.py` |
| `repositories/factor_repository.py` | `repositories/factors/factor_repository.py` |
| `services/factor_service.py` | `services/factors/factor_service.py` |
| `services/factor_compute_service.py` | `services/factors/factor_compute_service.py` |
| `services/factor_scoring_service.py` | `services/factors/factor_scoring_service.py` |
| `services/factor_analysis_service.py` | `services/factors/factor_analysis_service.py` |
| `services/_factor_helpers.py` | `services/factors/_factor_helpers.py` |
| `schemas/factors.py` | `schemas/factors/factors.py` |

### 9. risk — Risk limits CRUD + analytics (VaR, correlation, concentration, liquidity)
| Source | Target |
|--------|--------|
| `api/v1/risk.py` | `api/v1/risk/risk.py` |
| `api/v1/risk_analytics.py` | `api/v1/risk/risk_analytics.py` |
| `models/risk.py` | `models/risk/risk.py` |
| `repositories/risk_repository.py` | `repositories/risk/risk_repository.py` |
| `services/risk_analytics_service.py` | `services/risk/risk_analytics_service.py` |
| `schemas/risk.py` | `schemas/risk/risk.py` |
| `schemas/risk_analytics.py` | `schemas/risk/risk_analytics.py` |

### 10. rebalancing — Policy CRUD + decision engine
| Source | Target |
|--------|--------|
| `api/v1/rebalance.py` | `api/v1/rebalancing/rebalance.py` |
| `api/v1/rebalance_policy.py` | `api/v1/rebalancing/rebalance_policy.py` |
| `models/rebalancing.py` | `models/rebalancing/rebalancing.py` |
| `repositories/rebalancing_repository.py` | `repositories/rebalancing/rebalancing_repository.py` |
| `services/rebalancing_service.py` | `services/rebalancing/rebalancing_service.py` |
| `schemas/rebalancing.py` | `schemas/rebalancing/rebalancing.py` |

### 11. views — Black-Litterman + Entropy Pooling + Opinion Pooling + LLM moments + sentiment
| Source | Target |
|--------|--------|
| `api/v1/views.py` | `api/v1/views/views.py` |
| `api/v1/opinion_pooling.py` | `api/v1/views/opinion_pooling.py` |
| `api/v1/llm_moments.py` | `api/v1/views/llm_moments.py` |
| `repositories/view_generation_repository.py` | `repositories/views/view_generation_repository.py` |
| `services/view_generation.py` | `services/views/view_generation.py` |
| `services/entropy_pooling_service.py` | `services/views/entropy_pooling_service.py` |
| `services/opinion_pooling.py` | `services/views/opinion_pooling.py` |
| `services/llm_moments.py` | `services/views/llm_moments.py` |
| `services/sentiment.py` | `services/views/sentiment.py` |
| `schemas/views.py` | `schemas/views/views.py` |
| `schemas/llm_moments.py` | `schemas/views/llm_moments.py` |

### 12. attribution — Brinson + factor attribution
| Source | Target |
|--------|--------|
| `api/v1/attribution.py` | `api/v1/attribution/attribution.py` |
| `services/attribution_service.py` | `services/attribution/attribution_service.py` |
| `schemas/attribution.py` | `schemas/attribution/attribution.py` |

### 13. dashboard — Equity curve, allocation, drift, market snapshot
| Source | Target |
|--------|--------|
| `api/v1/dashboard.py` | `api/v1/dashboard/dashboard.py` |
| `repositories/dashboard_repository.py` | `repositories/dashboard/dashboard_repository.py` |
| `services/dashboard_service.py` | `services/dashboard/dashboard_service.py` |
| `schemas/dashboard.py` | `schemas/dashboard/dashboard.py` |

### 14. scenarios — Stress scenarios + synthetic data (vine copula)
| Source | Target |
|--------|--------|
| `api/v1/stress_scenarios.py` | `api/v1/scenarios/stress_scenarios.py` |
| `api/v1/synthetic.py` | `api/v1/scenarios/synthetic.py` |
| `services/stress_scenarios.py` | `services/scenarios/stress_scenarios.py` |
| `services/synthetic_service.py` | `services/scenarios/synthetic_service.py` |
| `schemas/stress_scenarios.py` | `schemas/scenarios/stress_scenarios.py` |
| `schemas/synthetic.py` | `schemas/scenarios/synthetic.py` |

### 15. reports — Background PDF reports
| Source | Target |
|--------|--------|
| `api/v1/reports.py` | `api/v1/reports/reports.py` |
| `services/report_service.py` | `services/reports/report_service.py` |
| `schemas/reports.py` | `schemas/reports/reports.py` |

### 16. jobs — Background-job infra + APScheduler
| Source | Target |
|--------|--------|
| `api/v1/jobs.py` | `api/v1/jobs/jobs.py` |
| `api/v1/scheduler.py` | `api/v1/jobs/scheduler.py` |
| `models/background_job.py` | `models/jobs/background_job.py` |
| `repositories/background_job_repository.py` | `repositories/jobs/background_job_repository.py` |
| `services/background_job.py` | `services/jobs/background_job.py` |
| `services/scheduler.py` | `services/jobs/scheduler.py` |
| `schemas/jobs.py` | `schemas/jobs/jobs.py` |
| `schemas/scheduler.py` | `schemas/jobs/scheduler.py` |

### 17. auth — API-key model
| Source | Target |
|--------|--------|
| `models/api_key.py` | `models/auth/api_key.py` |

`middleware/auth.py` stays in `middleware/` — auth verification is a transport concern, not a domain.

## `_shared/` placement (per layer)

### Summary by layer

| Layer | Files in `_shared/` | Purpose |
|-------|---------------------|---------|
| `api/v1/_shared/` | `router.py`, `metrics.py`, `database.py`, `test.py`, `__init__.py` | App-level FastAPI router aggregator + admin/diagnostic endpoints |
| `models/_shared/` | `base.py` | `Base`, `BaseModel`, `TimestampMixin`, `UUIDPrimaryKeyMixin` |
| `repositories/_shared/` | `base.py`, `database_admin_repository.py` | `RepositoryBase` session holder + DB admin queries |
| `services/_shared/` | `_price_fetcher.py`, `_sector_resolver.py`, `_json_safe.py`, `_progress.py`, `_benchmark_bootstrap.py`, `trading_calendar.py`, `notifications.py` | Cross-domain helpers: price fetcher (used by 6 services), sector resolver (attribution + dashboard), JSON-safe coercion, progress callbacks, lifespan benchmark seeding, exchange calendar, webhook notifications |
| `schemas/_shared/` | `base.py`, `base_job.py` | `CamelCaseModel` Pydantic base + `AsyncJobCreateResponse` / `AsyncJobProgress` |

### Explicit source → target mapping for `_shared/` files

The cluster tables above cover domain files. The 16 cross-cutting files below land in `_shared/` and are listed explicitly so no source path is implicit:

| Source | Target |
|--------|--------|
| `api/v1/router.py` | `api/v1/_shared/router.py` |
| `api/v1/metrics.py` | `api/v1/_shared/metrics.py` |
| `api/v1/database.py` | `api/v1/_shared/database.py` |
| `api/v1/test.py` | `api/v1/_shared/test.py` |
| `models/base.py` | `models/_shared/base.py` |
| `repositories/base.py` | `repositories/_shared/base.py` |
| `repositories/database_admin_repository.py` | `repositories/_shared/database_admin_repository.py` |
| `services/_price_fetcher.py` | `services/_shared/_price_fetcher.py` |
| `services/_sector_resolver.py` | `services/_shared/_sector_resolver.py` |
| `services/_json_safe.py` | `services/_shared/_json_safe.py` |
| `services/_progress.py` | `services/_shared/_progress.py` |
| `services/_benchmark_bootstrap.py` | `services/_shared/_benchmark_bootstrap.py` |
| `services/trading_calendar.py` | `services/_shared/trading_calendar.py` |
| `services/notifications.py` | `services/_shared/notifications.py` |
| `schemas/base.py` | `schemas/_shared/base.py` |
| `schemas/base_job.py` | `schemas/_shared/base_job.py` (rename pending — see open question 2) |

`services/infrastructure/` (cache, circuit_breaker, rate_limiter, retry) is **already** structured as a sub-package and stays at the top of `services/`. It is a deliberately public utility surface (re-exported via `services/yfinance/infrastructure/` shims for the existing CLI/agent integrations) and does **not** move under `_shared/`.

## Already-structured sub-packages

These move as a unit (no internal restructuring):

| Source pkg | Target pkg | File count |
|------------|-----------|------------|
| `services/yfinance/` | `services/market_data/yfinance/` | 29 |
| `services/trading212/` | `services/universe/trading212/` | 11 |
| `services/scrapers/` | `services/macro/scrapers/` | 5 |
| `services/infrastructure/` | (unchanged — stays at top of `services/`) | 5 |

The internal layout of these packages is preserved verbatim. For reference, the files implicitly carried along include (non-exhaustive):

- `services/yfinance/protocols/` (`__init__.py`, `interfaces.py`)
- `services/yfinance/market/` (`streaming.py`, `screener.py`, plus existing modules)
- `services/yfinance/ticker/` (`financials.py`, `metadata.py`, `analysis.py`, `corporate_actions.py`, `funds.py`, `holders.py`)
- `services/yfinance/news/` (`aggregator.py`, plus existing modules)
- `services/yfinance/infrastructure/` (re-export shims of `services/infrastructure/`)
- `services/trading212/cache/`, `services/trading212/filters/`, `services/trading212/ticker_mapper.py`, `services/trading212/config.py`, `services/trading212/builder.py`, `services/trading212/client.py`, `services/trading212/protocols.py`
- `services/scrapers/fred_scraper.py`, `ilsole_scraper.py`, `tradingeconomics_scraper.py`, `exceptions.py`
- `services/infrastructure/cache.py`, `circuit_breaker.py`, `rate_limiter.py`, `retry.py`

## Files NOT moved

Top-level `api/app/*.py` infrastructure files are app-wide scaffolding, not domain code. They stay in place:

- `config.py`
- `database.py`
- `dependencies.py`
- `exceptions.py`
- `metrics.py`
- `main.py`
- `__init__.py`

`middleware/` is untouched (6 files: `__init__.py`, `auth.py`, `logging.py`, `rate_limiting.py`, `security.py`, `metrics_middleware.py`).

`utils/` is untouched (3 files: `__init__.py`, `currency.py`, `date_parsing.py`). Import paths `app.utils.currency.to_major_currency` and `app.utils.date_parsing.parse_reference_date` survive the reorg unchanged.

`core/` contains only `__init__.py` (no substantive code). Open question 1 covers its disposition.

`api/__init__.py` (the intermediate package init between `app/` and `api/v1/`) stays in place; it must continue to exist for `from app.api.v1.<...>` imports to resolve.

## Layer-root `__init__.py` handling

Each of the five reorganised layers currently has a `__init__.py` at its root:

- `api/v1/__init__.py`
- `models/__init__.py`
- `repositories/__init__.py`
- `services/__init__.py`
- `schemas/__init__.py`

These stay at the layer root (not moved into `_shared/`). After the reorg, two of them require functional updates rather than mechanical moves:

- **`models/__init__.py`** — must re-export every ORM class (or at minimum import every submodule containing ORM classes) so Alembic autogenerate sees the full `Base.metadata` graph. Currently a thin file; after the reorg it must import from `models/auth/`, `models/market_data/`, `models/universe/`, `models/macro/`, `models/portfolio/`, `models/execution/`, `models/factors/`, `models/risk/`, `models/rebalancing/`, `models/jobs/`, `models/_shared/`. Missing any sub-folder = silent table omission in migrations.
- **`api/v1/__init__.py`** — if it currently aggregates routers (or `router.py` does), the aggregation logic moves to the new `api/v1/_shared/router.py` and references the new domain-folder paths.

The other three (`repositories/__init__.py`, `services/__init__.py`, `schemas/__init__.py`) can stay empty or thin re-export files — they have no Alembic-style discovery requirement.

New `__init__.py` files must be added to every newly-created sub-folder (one per cluster-folder per layer, plus one per `_shared/`).

## Open questions — resolved

Each question below has been resolved against FastAPI production architecture best practices (`fastapi-expert-agent`): explicit-over-implicit naming, domain-driven separation of concerns, repository pattern locality, Pydantic v2 public-base re-exports, ownership-by-data-dependency, and bounded-context folder structure.

### 1. `core/` empty — **REMOVE**

`core/` currently contains only `__init__.py` with no substantive code. Empty packages are an intent-without-delivery anti-pattern: they advertise capability the codebase does not have, and any reader scanning the tree wastes attention on a dead folder.

FastAPI production projects commonly use `core/` for cross-cutting security helpers, custom exception hierarchies, or settings glue. The optimizer project does not have those concerns centralised there — security lives in `middleware/`, settings in `config.py`, exceptions in `exceptions.py`, all top-level. The reorg's `_shared/` per layer covers the rest.

**Decision:** delete the `core/` directory during migration. If future cross-cutting concerns emerge (e.g. centralised dependency-injection providers, domain-event bus), recreate `core/` with concrete content at that point.

### 2. `schemas/base_job.py` rename — **KEEP NAME (no leading underscore)**

Leading underscore = PEP 8 "private to module/package" marker. `schemas/base_job.py` is the **public** parent class for every domain job schema (`AsyncJobCreateResponse`, `AsyncJobProgress`) and is imported by ~10 cluster schema modules. Renaming to `_base_job.py` would lie about its access level.

The `_shared/` folder name carries the underscore (the folder is package-private — callers should import from cluster-specific schemas, not directly from `schemas._shared`). Files **inside** `_shared/` retain their public/private status as before. `base.py` (`CamelCaseModel`) and `base_job.py` (`AsyncJob*`) are public re-export bases; they keep their public names.

**Decision:** target path is `schemas/_shared/base_job.py` (no rename). Same rule applies to `models/_shared/base.py`, `repositories/_shared/base.py`, `schemas/_shared/base.py`.

The `services/_shared/` files that already carry leading underscores (`_price_fetcher.py`, `_sector_resolver.py`, `_json_safe.py`, `_progress.py`, `_benchmark_bootstrap.py`) keep them — those are private internals, callers must import via the owning service or via an explicit `_shared` re-export.

### 3. `services/_factor_helpers.py` placement — **`services/factors/_factor_helpers.py`**

Repository-pattern locality rule: helpers used by exactly one cluster live inside that cluster. `_factor_helpers.py` provides `FactorDataError` and shared dataclasses imported by `factor_compute_service`, `factor_scoring_service`, `factor_analysis_service`, and the `factor_service` facade — all inside the `factors/` cluster. No external caller.

Moving it to `_shared/` would advertise reusability the file does not deliver, and would force factors-cluster contributors to scan two folders for related code.

The leading underscore is retained as the private-internal marker. The full source→target row:

| Source | Target |
|--------|--------|
| `services/_factor_helpers.py` | `services/factors/_factor_helpers.py` |

This is already reflected in cluster table 8 — confirmed.

### 4. `reference_indices` ownership — **`market_data/` (confirmed)**

The `reference_index_seeder.py` service fetches SPY/QQQ/IWM price history from yfinance and persists it via `yfinance_repository`. Writing rows into `instruments` is a side-effect of the ingest pipeline, not a universe-management operation: there is no investability screening, no Trading 212 ticker mapping, no exchange enrichment.

Repository-pattern principle: data flow direction determines ownership. The seeder reads from yfinance and writes to two tables (Instrument + PriceHistory) but the **operation** is "ingest market data", not "construct the universe". Routes that build/screen the tradeable universe (`trading212.py`, `universe_screen.py`) live in `universe/`. The seeder hooks into `_benchmark_bootstrap.py` at lifespan startup for benchmark coverage of dashboard/risk-analytics.

**Decision:** `market_data/` confirmed. Cross-references in `_benchmark_bootstrap.py` (lifespan hook) update to import from `services.market_data.reference_index_seeder`.

### 5. `macro_calibration` URL prefix mismatch — **folder follows ownership; URL unchanged**

FastAPI separation rule: URL prefix = transport contract; folder = code ownership. They do not have to match. Routes register with explicit `prefix=...` regardless of file location, so the existing `/views/macro-calibration` URL stays stable for frontend consumers.

The file's data dependency is `MacroRegimeRepository` (it pulls macro indicators to feed BAML `ClassifyMacroRegime`). Code ownership belongs to the macro cluster; URL grouping under `/views/` reflects that calibration produces inputs consumed by the views/Black-Litterman pipeline.

**Decision:** target path `api/v1/macro/macro_calibration.py`. URL prefix `/views/macro-calibration` is preserved by the route's `prefix=` argument — no frontend or OpenAPI client changes. Add a one-line comment at the top of the route file explaining the deliberate URL/folder split so future readers don't "fix" it.

### 6. `auth/` cluster size — **KEEP as own cluster**

Domain-driven design: `auth/` is its own bounded context even with one current file (`api_key.py`). Single-file folders today are not waste — they are pre-created shape for predictable growth. Authentication concerns expand naturally: user accounts, roles, scopes, refresh tokens, audit logs, session tables, OAuth provider integrations.

Merging `api_key.py` into `models/_shared/` would dilute `_shared/` (which is for cross-cutting infrastructure bases, not domain entities) and force a future split when the second auth model arrives. Pre-creating the folder is cheaper than the future move.

**Decision:** `models/auth/api_key.py` confirmed. The folder remains a one-file cluster until auth grows. `middleware/auth.py` stays put — it is transport-layer auth verification, not the domain definition.

### Summary table

| # | Question | Decision | Best-practice basis |
|---|----------|----------|---------------------|
| 1 | `core/` disposition | Delete | Explicit-over-implicit; empty packages are noise |
| 2 | `base_job.py` rename | Keep public name | PEP 8 underscore = private; bases are public re-exports |
| 3 | `_factor_helpers.py` location | `services/factors/` | Locality rule; single-cluster usage |
| 4 | `reference_indices` cluster | `market_data/` | Ownership follows ingest direction, not write target |
| 5 | `macro_calibration` URL | Folder by ownership, URL preserved | Transport ≠ code structure |
| 6 | `auth/` single-file cluster | Keep as cluster | DDD bounded context; future growth |

## Migration notes (for future reorg work)

When the actual reorg is executed, the following will be required (not done now):

1. **`git mv`** all listed source paths to target paths (preserves history).
2. **Add `__init__.py`** to every new sub-folder (Python package marker).
3. **Rewrite imports** across the codebase. Affected import roots: `app.api.v1.*`, `app.models.*`, `app.repositories.*`, `app.services.*`, `app.schemas.*` — every domain file gains a sub-folder segment in its dotted path.
4. **Update `app/api/v1/router.py`** to register routers from the new sub-folder paths.
5. **Update Alembic migrations** that import from `app.models.*` — Alembic auto-detects models via `Base.metadata` so importing `app.models` (the package `__init__.py`) is sufficient as long as `__init__.py` re-exports from the new sub-folders.
6. **Update tests**:
   - `api/tests/unit/*` and `api/tests/integration/*` — patch targets reference `app.services.*` and `app.repositories.*` strings; all need rewriting.
   - `tests/cli/*` (legacy folder name from the old CLI; now references `research.*`) is unaffected.
7. **Update CLAUDE.md** "Architecture" section to reflect the new layout.

This document is the source of truth for the target structure. Migration work will reference each cluster table line-by-line.

## Verification of this document

This is a doc-only deliverable. Verify the document by checking:

1. Every file currently in the five flat layers is mapped to a target path (count: ~170 source files → cluster tables + explicit `_shared/` source-target table + sub-package unit moves).
2. The 16 domain folder names appear in the same form across every layer that owns them; intentional asymmetries are listed.
3. Already-structured sub-packages (`yfinance/`, `trading212/`, `infrastructure/`, `scrapers/`) are preserved as units; their internal layouts including `protocols/`, `market/`, `ticker/`, `news/`, `cache/`, `filters/` sub-packages are implicitly carried.
4. The cross-cluster split (`execution/` model + repo, shared by optimization + backtest) is documented as a layered asymmetry.
5. Top-level `api/app/*.py` infrastructure files are explicitly listed as un-moved.
6. Layer-root `__init__.py` files are addressed (kept in place; `models/__init__.py` and `api/v1/__init__.py` require functional updates).
7. `middleware/`, `utils/`, `core/`, and intermediate `api/__init__.py` are accounted for.

## Audit trail

This document was reviewed against the live filesystem on 2026-05-10 by `feature-dev:code-explorer` agent. The audit found four categories of gaps in the prior draft, all addressed in this revision:

1. **Implicit `_shared/` mappings** — 16 cross-cutting files appeared only as target filenames in the layer summary table without explicit source rows. Now listed in the "Explicit source → target mapping for `_shared/` files" subsection.
2. **Layer-root `__init__.py` disposition** — five layer-root inits were never mentioned. Now covered in the "Layer-root `__init__.py` handling" section, with explicit notes on `models/__init__.py` (Alembic discovery) and `api/v1/__init__.py` (router aggregation).
3. **Sub-package internal contents** — `services/yfinance/protocols/`, `market/streaming.py`, `ticker/financials.py`, etc. are not visible in the doc. Now enumerated in the "Already-structured sub-packages" section as implicitly carried.
4. **Layer-parity asymmetries** — `execution/` (models+repos only), `auth/` (models only), domain folders without models, and the `sentiment_repository` orphan are now explicitly called out in the new "Asymmetries (intentional)" subsection.

`middleware/metrics_middleware.py`, `utils/currency.py`, `utils/date_parsing.py`, `core/__init__.py`, and `api/__init__.py` are now explicitly listed in the "Files NOT moved" section.
