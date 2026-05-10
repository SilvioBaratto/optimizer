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

Plus `execution/` under `models/` + `repositories/` only (shared persistence parent of `optimization` + `backtest`).

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

| Layer | Files in `_shared/` | Purpose |
|-------|---------------------|---------|
| `api/v1/_shared/` | `router.py`, `metrics.py`, `database.py`, `test.py`, `__init__.py` | App-level FastAPI router aggregator + admin/diagnostic endpoints |
| `models/_shared/` | `base.py` | `Base`, `BaseModel`, `TimestampMixin`, `UUIDPrimaryKeyMixin` |
| `repositories/_shared/` | `base.py`, `database_admin_repository.py` | `RepositoryBase` session holder + DB admin queries |
| `services/_shared/` | `_price_fetcher.py`, `_sector_resolver.py`, `_json_safe.py`, `_progress.py`, `_benchmark_bootstrap.py`, `trading_calendar.py`, `notifications.py` | Cross-domain helpers: price fetcher (used by 6 services), sector resolver (attribution + dashboard), JSON-safe coercion, progress callbacks, lifespan benchmark seeding, exchange calendar, webhook notifications |
| `schemas/_shared/` | `base.py`, `base_job.py` | `CamelCaseModel` Pydantic base + `AsyncJobCreateResponse` / `AsyncJobProgress` |

`services/infrastructure/` (cache, circuit_breaker, rate_limiter, retry) is **already** structured as a sub-package and stays at the top of `services/`. It is a deliberately public utility surface (re-exported via `services/yfinance/infrastructure/` shims for the existing CLI/agent integrations) and does **not** move under `_shared/`.

## Already-structured sub-packages

These move as a unit (no internal restructuring):

| Source pkg | Target pkg |
|------------|-----------|
| `services/yfinance/` | `services/market_data/yfinance/` |
| `services/trading212/` | `services/universe/trading212/` |
| `services/scrapers/` | `services/macro/scrapers/` |
| `services/infrastructure/` | (unchanged — stays at top of `services/`) |

## Files NOT moved

Top-level `api/app/*.py` infrastructure files are app-wide scaffolding, not domain code. They stay in place:

- `config.py`
- `database.py`
- `dependencies.py`
- `exceptions.py`
- `metrics.py`
- `main.py`
- `__init__.py`

`middleware/` and `utils/` are already structured and untouched. `core/` is currently empty — open question whether to keep or remove.

## Open questions / call-outs

These are deliberately left for the user to decide before any future migration:

1. **`core/` is empty** — remove the directory, or keep it as a placeholder for future cross-cutting concerns?
2. **`schemas/_shared/base_job.py` rename** — currently `schemas/base_job.py` (no leading underscore). Rename for consistency with `_shared/` convention, or keep as-is?
3. **`services/_factor_helpers.py` placement** — used only inside the factors cluster. Proposal moves it into `services/factors/` (private helper, leading underscore retained). Alternative: keep at `services/_shared/`.
4. **`reference_indices` route ownership** — borderline between `market_data/` (data layer affinity, depends on `yfinance_repository`) and `universe/` (it seeds benchmark `Instrument` rows). Proposal places it under `market_data/`. Alternative: `universe/`.
5. **`macro_calibration` URL-vs-folder mismatch** — file lives in `macro/` (data ownership) but registers under `/views/macro-calibration` URL. Folder location follows ownership; URL prefix unchanged.
6. **`auth/` cluster size** — only one file (`api_key.py`). Kept as its own cluster per the "fine granularity" choice. Alternative: merge into `_shared/`.

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

1. Every file currently in the five flat layers is mapped to a target path (count: ~170 source files → ~170 mapped lines across the cluster tables and `_shared/` table).
2. The 16 domain folder names appear in the same form across every layer that owns them.
3. Already-structured sub-packages (`yfinance/`, `trading212/`, `infrastructure/`, `scrapers/`) are preserved as units.
4. The cross-cluster split (`execution/` model + repo, shared by optimization + backtest) is documented as the single intentional asymmetry.
5. Top-level `api/app/*.py` infrastructure files are explicitly listed as un-moved.
