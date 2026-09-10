# CLAUDE.md

Guidance for Claude Code in this repository. These instructions OVERRIDE default behavior.

> **Deep reference lives in [`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)** — per-module
> API detail, scheduler internals, full env-var list, DB/ingestion layer tables. Read the
> relevant section there on demand; this file is the contract + commands + gotchas only.

## Agent & Skill Requirements

**MANDATORY**, loaded proactively (not on request) for any finance-related or code task:
- `/skfolio` skill — all portfolio optimization, risk models, skfolio API usage
- `/yfinance` skill — all Yahoo Finance data retrieval and yfinance API usage
- `python-pro` agent — all Python writing, reviewing, debugging

## Project Overview

Python-only **uv workspace**, three packages (one shared venv):

- **`optimizer/`** — Pure-Python optimization library (DB-agnostic, sklearn/skfolio-based). Dist **`portopt-core`** (import `optimizer`)
- **`ingestion/`** — Headless **ingestion daemon**. APScheduler in-process, no HTTP API. Fetches market / fundamental / macro data into PostgreSQL on a schedule. Entrypoint `ingestion/app/worker.py`; manual runs via `ingestion/app/cli.py`. Dist **`portopt`**
- **`packages/portopt-db/`** — Shared **DB layer** (import `portopt_db`): SQLAlchemy `Base` + all models + all repositories + `DatabaseManager`/`RepositoryBase` + the single Alembic tree. Dist **`portopt-db`** (internal; only consumer today is `ingestion`, a future `fund/` is planned)

**Boundary (guarded, load-bearing)**: `ingestion/` and `portopt-db/` do **not** import `optimizer`; `portopt-db/` carries no sklearn/skfolio stack. The daemon **does** ship `scikit-learn` (scipy transitively): yfinance's price-repair path (`repair=True`) imports `sklearn.cluster.DBSCAN`, so without it ~22% of tickers return empty history and get dropped. sklearn here is a data-layer dep, not optimization. With a single shared venv there is no install isolation — three static import-scan tests are the **sole** enforcement: `ingestion/tests/unit/hygiene/test_no_optimizer_import.py`, `packages/portopt-db/tests/test_no_optimizer_import.py`, root `tests/test_no_portopt_db_import.py`.

Supporting dirs: `tests/` (library suite, mirrors `optimizer/` + `tests/scheduler/`), `ingestion/tests/` (SQLite in-memory), `packages/portopt-db/tests/` (SQLite in-memory), `scheduler/` (shell wrappers over the CLI), `scripts/` (CI helpers).

**No frontend, no docs site, no `examples/`/`research/`/`cli/`, no HTTP API** — deleted in the strip (branch `refactor/strip-to-ingestion-pipeline`). Any leftover reference (a route, `TestClient`, `uvicorn`, `app.main`, `/api/v1/`) is dead — delete it, don't revive the dependency.

## Build & Run Commands

uv workspace — `uv sync` resolves all three members into one venv. Run tools per-package with `uv run --package <name>` (`portopt-core`, `portopt`, `portopt-db`). `pip install -e` still works per-package.

```bash
# Infrastructure
docker compose up -d              # PostgreSQL (54320) + Adminer (18081) + scheduler (metrics 9000)

# Workspace setup (all three packages + every extra into one venv)
uv sync --all-packages --all-extras

# Optimizer library (root, package portopt-core)
uv run --package portopt-core pytest tests/ -v      # All optimizer tests
uv run --package portopt-core pytest -k "test_name" # Single test by name
uv run --package portopt-core ruff check optimizer/ tests/
uv run --package portopt-core mypy optimizer/       # Type check strict mode (CI step)

# Makefile shortcuts (root)
make lint | format | typecheck | test | all | clean

# Shared DB layer (package portopt-db)
uv run --package portopt-db pytest
cd packages/portopt-db && alembic upgrade head      # single migration owner; head: d3e4f5a6b7c8

# Ingestion daemon (package portopt)
uv run --package portopt pytest
uv run --package portopt python -m app.worker       # Run the daemon (blocks; SIGTERM to stop)

# Manual ingestion runs (same job-slot / heartbeat path as scheduler)
docker compose exec scheduler python -m app.cli daily
docker compose exec scheduler python -m app.cli refetch-all
docker compose exec scheduler python -m app.cli yfinance --mode full --period 5y
# also: universe | macro | fred | news | summarize | calibrate | reference-indices

# BAML (regenerate after editing ingestion/baml_src/)
cd ingestion && baml-cli generate
```

## CI Pipeline

`.github/workflows/ci.yml` — push/PR to `main`, Ubuntu, Python **3.12 & 3.13**, uv-driven (`uv sync --all-packages --all-extras`). Jobs: `lint` (ruff check → ruff format --check → pip-audit --strict), `typecheck` (mypy optimizer/), `pyright` (scoped to optimizer/, pinned), `test` (pytest tests/ `--cov=optimizer --cov-fail-under=90` then branch-coverage 0.80), `ingestion-test` (`--cov=app --cov-fail-under=80`), `portopt-db-test` (`--cov=portopt_db --cov-fail-under=90`). Other: `release.yml` (on `v*`). No `smoke.yml`.

**Deps**: each member declares its own. Library runtime → root `pyproject.toml`; DB layer → `packages/portopt-db/pyproject.toml` (SQLAlchemy, psycopg2-binary, alembic, pandas, pydantic); daemon → `ingestion/pyproject.toml` (`portopt-db` is a `[tool.uv.sources]` workspace dep; psycopg2-binary/alembic arrive transitively). No `requirements.txt`. `requires-python >= 3.12` across all three.

## Optimizer Library conventions

Every module: **frozen `@dataclass` config** + **factory function** + **`str, Enum` types**. Configs hold only primitives/enums/nested frozen dataclasses (serialisable). Non-serialisable objects (estimator instances, numpy arrays, callables) are factory `**kwargs`. Strict and consistent across all modules.

All transformers follow sklearn `BaseEstimator + TransformerMixin` and compose in `sklearn.pipeline.Pipeline`. The pipeline flattens pre-selection + optimiser steps so `get_params()` exposes nested params (`"optimizer__l2_coef"`, `"drop_correlated__threshold"`).

**Pipeline flow**: `prices → preprocessing → pre_selection → moments → views → optimization → validation → tuning → rebalancing → pipeline`. Plus `factors/`, `synthetic/`, `scoring/`, `universe/`, `distance/`, `cluster/`, `uncertainty_set/`, `linear_model/`, `online/`, `fx/`.

Per-submodule detail (configs, presets, factories, shape contracts) → **[`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)**.

### Key conventions

- `prices_to_returns()` runs **outside** the pipeline (changes data semantics); pipeline operates on return DataFrames only
- Views use `tuple[str, ...]` in configs (hashable); factories convert to `list` for skfolio
- View configs embed `MomentEstimationConfig` for their inner prior (keeps configs serialisable)
- The fitted prior attribute is `return_distribution_` (not `prior_model_`): `mu`, `covariance`, `returns`, `sample_weight`, `cholesky`
- For `BenchmarkTracker`, benchmark returns are passed as `y` in `fit(X, y)`
- When `previous_weights` is passed to `run_full_pipeline()`, it auto-aligns on post-pre-selection universe and re-normalises
- Sector mapping is injected as a plain `dict[str, str]`, not queried from the database

### Cross-cutting gotchas (skfolio 1.0)

- **Linear returns only**: pipelines/estimators consume linear (simple) returns as `X`. Use `prices_to_returns()` — do NOT pass log returns
- **`shuffle=False` in CV**: temporal CV must preserve order. `KFold(shuffle=True)` / `train_test_split(shuffle=True)` break causality and silently leak future data
- **Metadata routing**: call `sklearn.set_config(enable_metadata_routing=True)` BEFORE `.set_fit_request(...)`. Required for `ImpliedCovariance.implied_vol`, `BenchmarkTracker.y`, etc.
- **`TimeSeriesFactorModel.fit(X, y)`**: `X` asset returns, `y` factor returns. Wrapped in Black-Litterman, views reference factor names (not asset names)
- **`BenchmarkTracker.fit(X, y)`**: `y` is the benchmark return series; the Config carries no benchmark field — pass at fit time
- **Variance estimators store `variance_` (1-D), NOT `covariance_` (2-D)** — `EmpiricalVariance`/`EWVariance`/`RegimeAdjustedEWVariance` are NOT interchangeable with covariance estimators inside priors needing a full matrix
- **`Pipeline` is rejected by `online_predict` / `OnlineGridSearch`** — skfolio routes `partial_fit` through a single estimator. Apply pre-selection to `X` upstream. **Online instances are not thread-safe — one wrapper per thread**
- **Walk-forward CV cannot vary constraints per fold** — regime-dependent sector bands are fixed for a whole run; a backtest is single-regime, not per-rebalance

## Ingestion / DB / Scheduler (summary)

Full tables and internals → **[`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)**. Load-bearing points:

- **DB layer (`portopt-db`)**: single schema + connection manager + Alembic tree (head `d3e4f5a6b7c8`, runs from `packages/portopt-db`). Pure structural extraction, no sklearn/optimizer import (guarded). `background_jobs` *model* lives here but `BackgroundJobRepository` *behavior* stays in `ingestion/app/repositories/jobs/`. Coverage floor line ≥ 90%
- **Daemon layering**: Scheduler/CLI → Services → Repositories → Models, `_shared/` per layer. Models + most repos live in `portopt_db`; ingestion keeps only the `jobs` repo. No HTTP API. Sync SQLAlchemy sessions — everything opens its own via `database_manager.get_session`. PostgreSQL 16 on port **54320**. **Do not reintroduce `optimizer` as an ingestion dep**
- **Import-cycle gotcha**: `app/services/_shared/__init__.py` must NOT re-export `bootstrap_benchmarks` — import from `app.services._shared._benchmark_bootstrap`
- **Scheduler**: APScheduler in-process in `worker.py`, `SQLAlchemyJobStore`. Seven jobs; `universe_build` runs **before** `weekly_refetch` (every step iterates `instruments`, a stale universe caps yfinance). One public step function each in `scheduler.py` — **add new work as a step, not a CLI-only branch**
- **Gotcha — sync steps need an explicit heartbeat**: only the heartbeat thread stamps `last_heartbeat_at`. A sync step outliving `SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS` (300s) gets falsely reaped and flips to `failed` mid-run. Wrap work in `scheduler._heartbeat()`
- **Gotcha — reaper is host-scoped**: `reconcile_orphans` fails any active row whose `worker_host != socket.gethostname()`. Run exactly one daemon per DB
- **Gotcha — JSONB in test-covered models**: use `JSON().with_variant(JSONB, "postgresql")` so SQLite tests can create the table
- **Gotcha — transient-error detection** (`infrastructure/retry.py`): case-sensitive substring match. `"Too Many Requests"` trips the breaker; `"too many requests"` does not
- **Cron weekday gotcha**: APScheduler `from_crontab` numbers days `0=Mon..6=Sun`. Use weekday names (`sat`); a bare `0` fires Monday

Key env vars: `DATABASE_URL`, `TRADING_212_API_KEY` (absent ⇒ `universe_build` skips without claiming a slot), `FRED_API_KEY`, `LLM_PROVIDER` (`openai`|`anthropic` — cloud-only, local/Ollama not supported), `METRICS_PORT` (9000), `YFINANCE_FETCH_WORKERS` (1-16, default 4). Full list + `SCHEDULER_*` crons → ARCHITECTURE.md.

## Linting & Type Checking

- **ruff**: line-length 88, target py310, rules `E, F, I, N, W, UP, B, SIM, S, RUF, C4, PTH`. Per-file ignores: `N803, N806` for `optimizer/` and `tests/` (sklearn `X, y`), `S101` for tests
- **mypy**: strict, `ignore_missing_imports = true`. Overrides relax `disallow_subclassing_any` for sklearn/skfolio bases
- **pyright**: run in CI (`pyright[nodejs]==1.1.408`)
- **Runtime deps** (root `pyproject.toml`): `numpy`, `pandas`, `scipy`, `scikit-learn`, `skfolio` (==1.0.6), `jinja2`. `arch` is NOT declared — it reaches bootstrap uncertainty-set classes transitively via skfolio. Code importing it directly must guard with `try/except ImportError` or declare it explicitly
- Coverage floors: library/daemon/db all gated line + branch ≥ 0.80 via `scripts/check_branch_coverage.py`
