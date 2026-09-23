# CLAUDE.md

Guidance for Claude Code in this repository. These instructions OVERRIDE default behavior.

> **Deep reference lives in [`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)** — per-module
> API detail, scheduler internals, full env-var list, DB/ingestion layer tables. Read the
> relevant section there on demand; this file is the contract + commands + gotchas only.

## Agent & Skill Requirements

**MANDATORY**, loaded proactively (not on request) for any finance-related or code task:
- `/skfolio` skill — all portfolio optimization, risk models, skfolio API usage
- `/yfinance` skill — all Yahoo Finance data retrieval and yfinance API usage

## Project Overview

Python-only **uv workspace**, **four packages** (one shared venv):

- **`packages/portopt-core/`** (declared from the repo-root `pyproject.toml`) — Pure-Python optimization library (DB-agnostic, sklearn/skfolio-based). Dist **`portopt-core`**, import `optimizer`. **The `optimizer` package lives at `packages/portopt-core/optimizer/`, NOT at the repo root** (src-layout move; `[tool.setuptools.packages.find] where = ["packages/portopt-core"]`).
- **`ingestion/`** — Headless **ingestion daemon**. APScheduler in-process, no HTTP API. Fetches market / fundamental / macro data into PostgreSQL on a schedule. Entrypoint `ingestion/app/worker.py`; manual runs via `ingestion/app/cli.py`. Dist **`portopt`**, import `app`.
- **`packages/portopt-db/`** — Shared **DB layer** (import `portopt_db`): SQLAlchemy `Base` + all models + all repositories + `DatabaseManager`/`RepositoryBase` + the single Alembic tree. Dist **`portopt-db`** (internal; consumed by both `ingestion` and `fund`).
- **`fund/`** — **LIVE** deepagents/langgraph bridge (Phases 0–9 code complete). *The LLM chooses, the optimizer computes.* Dist **`portopt-fund`**, import `fund` (src-layout `fund/src/fund/`). Entrypoints: `fund` CLI, `fund-worker` daemon, `fund-tui`.

**Boundary (guarded, load-bearing)**: `ingestion/` and `portopt-db/` do **not** import `optimizer`; `portopt-db/` carries no sklearn/skfolio stack. `optimizer` (`portopt-core`) does **not** import `portopt_db`. The daemon **does** ship `scikit-learn` (scipy transitively): yfinance's price-repair path (`repair=True`) imports `sklearn.cluster.DBSCAN`, so without it ~22% of tickers return empty history and get dropped. sklearn here is a data-layer dep, not optimization. With a single shared venv there is no install isolation — static import-scan tests are the **sole** enforcement: `ingestion/tests/unit/hygiene/test_no_optimizer_import.py`, `packages/portopt-db/tests/test_no_optimizer_import.py`, root `tests/test_no_portopt_db_import.py`. The `fund/` bridge adds a second axis: `fund` **may** import `optimizer` + `portopt_db` (the only member that may), but `deepagents`/`langgraph` live only in `fund/` — guarded by `fund/tests/unit/hygiene/test_no_ingestion_import.py` (fund ⊬ `app`) plus `test_no_agent_stack_import.py` in both `ingestion` and `packages/portopt-db` (neither imports `deepagents`/`langgraph`/`fund`).

> **Gotcha — the optimizer⊬portopt_db guard is currently a no-op.** `tests/test_no_portopt_db_import.py` scans `_OPTIMIZER_SRC = _REPO_ROOT / "optimizer"`, a path that no longer exists after the src-layout move. `Path.rglob` on a missing dir yields nothing, so `test_when_optimizer_src_is_scanned_then_no_db_import_is_found` passes trivially without scanning a single file. Repoint it at `packages/portopt-core/optimizer` to restore the guard. (The pyproject-dependency half of the test still works.)

Supporting dirs: `tests/` (library suite, mirrors `optimizer/` + `tests/scheduler/`), `ingestion/tests/` (SQLite in-memory), `packages/portopt-db/tests/` (SQLite in-memory), `fund/tests/` (SQLite in-memory + `integration` marker for live-Postgres round-trips), `scheduler/` (shell wrappers over the CLI), `scripts/` (CI helpers).

**No frontend, no docs site, no `examples/`/`research/`/`cli/`, no HTTP API** — deleted in the strip (branch `refactor/strip-to-ingestion-pipeline`). Any leftover reference (a route, `TestClient`, `uvicorn`, `app.main`, `/api/v1/`) is dead — delete it, don't revive the dependency.

## Build & Run Commands

uv workspace — `uv sync` resolves all four members into one venv. Run tools per-package with `uv run --package <name>` (`portopt-core`, `portopt`, `portopt-db`, `portopt-fund`). `pip install -e` still works per-package.

```bash
# Infrastructure
docker compose up -d              # PostgreSQL (54320) + Adminer (18081) + scheduler (metrics 9000)

# Workspace setup (all four packages + every extra into one venv)
uv sync --all-packages --all-extras

# Optimizer library (package portopt-core; source at packages/portopt-core/optimizer/)
uv run --package portopt-core pytest tests/ -v      # All optimizer tests
uv run --package portopt-core pytest -k "test_name" # Single test by name
uv run --package portopt-core ruff check packages/portopt-core/optimizer/ tests/
uv run --package portopt-core mypy packages/portopt-core/optimizer/
uv run pyright                                       # scoped via [tool.pyright] include

# Shared DB layer (package portopt-db)
uv run --package portopt-db pytest
cd packages/portopt-db && alembic upgrade head      # single migration owner; head: b3c4d5e6f7a8

# Ingestion daemon (package portopt)
uv run --package portopt pytest
uv run --package portopt python -m app.worker       # Run the daemon (blocks; SIGTERM to stop)

# Fund bridge (package portopt-fund)
uv run --package portopt-fund pytest
uv run --package portopt-fund python -m fund.worker # Drift-monitor daemon (fund-worker)

# Manual ingestion runs (same job-slot / heartbeat path as scheduler)
docker compose exec scheduler python -m app.cli daily
docker compose exec scheduler python -m app.cli refetch-all
docker compose exec scheduler python -m app.cli yfinance --mode full --period 5y
# also: universe | macro | fred | news | market-structure | calendars | market-summary | options
# lifecycle: setup | start | stop | status   (there is NO `reference-indices` command)

# Fund CLI (package portopt-fund)
fund profile <portfolio_id>          # build/refresh an investor profile
fund run <portfolio_id> --asof DATE  # run a decision round for a rebalance bar
fund approve <run_id> | reject <run_id> | status [portfolio_id] | report <run_id>
```

> **Gotcha — stale lint/typecheck paths.** The root `Makefile` (`make lint | typecheck`) and the CI `lint`/`typecheck` jobs still pass the bare path `optimizer/`, which no longer exists after the src-layout move — `ruff check optimizer/` errors with `E902` (file not found) locally on `development`. It has not turned CI red because **CI runs only on `main`** (`on: push/pull_request: branches: [main]`) where the move may not have landed. Use `packages/portopt-core/optimizer/` explicitly; fixing `Makefile` + `ci.yml` is pending.

## CI Pipeline

`.github/workflows/ci.yml` — push/PR to `main`, Ubuntu, Python **3.12 & 3.13**, uv-driven (`uv sync --all-packages --all-extras`). **Seven jobs**: `lint` (ruff check → ruff format --check → pip-audit --strict), `typecheck` (`mypy optimizer/` + `cd packages/portopt-db && mypy src/portopt_db`), `pyright` (`uv run pyright`, scoped via config, pinned **1.1.398**), `test` (matrix 3.12/3.13, `pytest tests/ --cov=optimizer --cov-fail-under=90` then branch-coverage ≥ 0.80 on the 3.12 leg), `ingestion-test` (`--cov=app --cov-fail-under=80` + branch 0.80), `portopt-db-test` (`--cov=portopt_db --cov-fail-under=90` + branch 0.80), `fund-test` (matrix 3.12/3.13, `--cov=fund --cov-fail-under=80` + branch 0.80 on 3.12). Other: `release.yml` (on `v*`, builds `portopt-core`). No `smoke.yml`.

**Deps**: each member declares its own. Library runtime → root `pyproject.toml`; DB layer → `packages/portopt-db/pyproject.toml` (SQLAlchemy, psycopg2-binary, alembic, pandas, pydantic); daemon → `ingestion/pyproject.toml` (`portopt-db` is a `[tool.uv.sources]` workspace dep; psycopg2-binary/alembic arrive transitively); bridge → `fund/pyproject.toml` (`portopt-core` + `portopt-db` workspace deps; `deepagents==0.7.14` + langgraph/langchain-ollama/psycopg). No `requirements.txt`. `requires-python >= 3.12` across all **four**.

## Optimizer Library conventions

Every module: **frozen `@dataclass` config** + **factory function** + **`str, Enum` types**. Configs hold only primitives/enums/nested frozen dataclasses (serialisable). Non-serialisable objects (estimator instances, numpy arrays, callables) are factory `**kwargs`. Strict and consistent across all modules.

All transformers follow sklearn `BaseEstimator + TransformerMixin` and compose in `sklearn.pipeline.Pipeline`. `build_portfolio_pipeline()` (in `pre_selection/`) flattens pre-selection + optimiser steps so `get_params()` exposes nested params (`"optimizer__l2_coef"`, `"drop_correlated__threshold"`).

**Library is composable primitives, not a fixed end-to-end runner** — there is no `pipeline/` module and no `run_full_pipeline`. Opinionated, DB-connected orchestration (FX, delisting, universe/factor selection, rebalancing decisions, persistence) belongs to the `fund/` bridge, keeping `optimizer` DB-agnostic.

**18 submodules.** Module flow: `prices → preprocessing → pre_selection → moments → views → optimization → validation → tuning → rebalancing`. Plus `factors/`, `synthetic/`, `scoring/`, `universe/`, `distance/`, `cluster/`, `uncertainty_set/`, `linear_model/`, `online/`, `fx/`. The `optimization` module ships **fifteen optimizer builders** (config + `build_*`): mean-risk (+ regime-blended, robust), HRP, HERC, NCO, Schur-complementary, risk-budgeting, max-diversification, DR-CVaR, stacking, benchmark-tracker, and three naive baselines (equal-weighted, inverse-volatility, random).

Per-submodule detail (configs, presets, factories, shape contracts, exact estimator/factor/screen inventories) → **[`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)**.

### Key conventions

- `prices_to_returns()` runs **outside** the pipeline (changes data semantics); pipelines operate on return DataFrames only
- Views use `tuple[str, ...]` in configs (hashable); factories convert to `list` for skfolio
- View configs embed `MomentEstimationConfig` for their inner prior (keeps configs serialisable)
- The fitted prior attribute is `return_distribution_` (not `prior_model_`): `mu`, `covariance`, `returns`, `sample_weight`, `cholesky`
- For `BenchmarkTracker`, benchmark returns are passed as `y` in `fit(X, y)`
- `build_portfolio_pipeline(optimizer, pre_selection_config=None, sector_mapping=None, expiration_dates=None, outlier_protection_mask=None)` (in `pre_selection/`) composes pre-selection + optimiser into one flat sklearn `Pipeline`; `sector_mapping` is injected as a plain `dict[str, str]`, not queried from the database

### Cross-cutting gotchas (skfolio 1.0)

- **Linear returns only**: pipelines/estimators consume linear (simple) returns as `X`. Use `prices_to_returns()` — do NOT pass log returns
- **`shuffle=False` in CV**: temporal CV must preserve order. `KFold(shuffle=True)` / `train_test_split(shuffle=True)` break causality and silently leak future data
- **Metadata routing**: call `sklearn.set_config(enable_metadata_routing=True)` BEFORE `.set_fit_request(...)`. Required for `ImpliedCovariance.implied_vol`, `BenchmarkTracker.y`, etc.
- **`TimeSeriesFactorModel.fit(X, factors=...)`**: `X` asset returns, `factors` factor returns (keyword-only in 1.0). Wrapped in Black-Litterman, views reference factor names (not asset names)
- **`BenchmarkTracker.fit(X, y)`**: `y` is the benchmark return series; the Config carries no benchmark field — pass at fit time
- **Variance estimators store `variance_` (1-D), NOT `covariance_` (2-D)** — `EmpiricalVariance`/`EWVariance`/`RegimeAdjustedEWVariance` are NOT interchangeable with covariance estimators inside priors needing a full matrix
- **`Pipeline` is rejected by `online_predict` / `OnlineGridSearch`** — skfolio routes `partial_fit` through a single estimator. Apply pre-selection to `X` upstream. **Online instances are not thread-safe — one wrapper per thread**
- **Walk-forward CV cannot vary constraints per fold** — regime-dependent sector bands are fixed for a whole run; a backtest is single-regime, not per-rebalance
- **EW estimators take `half_life`, not `alpha`** (1.0 breaking change; `alpha` raises `TypeError`)

## Ingestion / DB / Scheduler (summary)

Full tables and internals → **[`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)**. Load-bearing points:

- **DB layer (`portopt-db`)**: single schema + connection manager + Alembic tree (**head `b3c4d5e6f7a8`**, runs from `packages/portopt-db`). Pure structural extraction, no sklearn/optimizer import (guarded). `background_jobs` *model* lives here but `BackgroundJobRepository` *behavior* stays in `ingestion/app/repositories/jobs/`. Coverage floor line ≥ 90%
- **Daemon layering**: Scheduler/CLI → Services → Repositories → Models, `_shared/` per layer. Models + most repos live in `portopt_db`; ingestion keeps only the `jobs` repo. No HTTP API. Sync SQLAlchemy sessions — everything opens its own via `database_manager.get_session`. PostgreSQL 16 on port **54320**. **Do not reintroduce `optimizer` as an ingestion dep**
- **Import-cycle gotcha**: `app/services/_shared/__init__.py` must NOT re-export `bootstrap_benchmarks` — import from `app.services._shared._benchmark_bootstrap`
- **Scheduler**: APScheduler in-process in `worker.py`, `SQLAlchemyJobStore`. **Seven jobs**: `daily_pipeline` (daily 07:00), `midday_news` (daily 14:00), `universe_build` (**Sat** 02:00), `weekly_refetch` (**Sat** 03:00), `weekly_market_wide` (**Sat** 04:00), `fred_monthly` (1st 08:00), `orphan_reaper` (interval). `universe_build` runs **before** `weekly_refetch` (every step iterates `instruments`, a stale universe caps yfinance). One public step function each in `scheduler.py` — **add new work as a step, not a CLI-only branch**
- **Gotcha — sync steps need an explicit heartbeat**: only the heartbeat thread stamps `last_heartbeat_at`. A sync step outliving `SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS` (300s) gets falsely reaped and flips to `failed` mid-run. Wrap work in `scheduler._heartbeat()`
- **Gotcha — reaper is a heartbeat lease** (NOT host-scoped): `reconcile_orphans` fails any active `(pending|running)` row whose `last_heartbeat_at` is NULL or older than the lease TTL — there is **no** `worker_host`/`worker_pid` reap predicate (those columns are observability-only; the removed host-scoped design is stale). Run exactly one daemon per DB
- **Gotcha — JSONB in test-covered models**: use `JSON().with_variant(JSONB, "postgresql")` so SQLite tests can create the table
- **Gotcha — transient-error detection** (`infrastructure/retry.py`): case-sensitive substring match. `"Too Many Requests"` trips the breaker; `"too many requests"` does not
- **Cron weekday gotcha**: APScheduler `from_crontab` numbers days `0=Mon..6=Sun`. Use weekday names (`sat`); a bare `0` fires Monday

Key env vars: `DATABASE_URL`, `TRADING_212_API_KEY` (absent ⇒ `universe_build` skips without claiming a slot), `FRED_API_KEY`, `METRICS_PORT` (9000), `YFINANCE_FETCH_WORKERS` (1-16, default 4). Full list + `SCHEDULER_*` crons → ARCHITECTURE.md.

## Fund bridge (`fund/`, LIVE)

The bridge where an LLM makes decisions and delegates numerics to `optimizer`, persisting to Postgres via `portopt_db`. Only member that may import both `optimizer` and `portopt_db`; `deepagents`/`langgraph` are confined here. Model path: DeepSeek on Ollama Cloud via `langchain-ollama`. Durable checkpointer + store on a **dedicated `langgraph` Postgres schema** (`langgraph-checkpoint-postgres` over `psycopg`). Deep spec: `SPEC.md` + `todo/deep_agent.md`.

- **Gotcha — importing `fund.audit` pulls langgraph** via `__init__ → persistence`. Keep audit repos **lazy** (the `observe` pattern) so a module's bare import stays agent-stack-free (needed to keep `worker.py`/`scheduler.py` importable without the agent stack)
- **Gotcha — test UUID/SQLite affinity**: seed `agent_runs.portfolio_id` with a **letter-bearing** UUID (not all-decimal), or SQLite REAL-coerces it and the UUID result processor crashes
- **Gotcha — committing repos + shared session**: `FundJobRepository` commits; the shared SAVEPOINT `db_session` leaks commits across the session-scoped engine — use a private function-scoped engine (`job_session`) in tests, and `synchronize_session=False` on the reaper UPDATE. The fund worker's job slot is a **heartbeat lease**, mirroring the ingestion reaper
- **Theory-consultation protocol**: the `THEORY_CONSULTATION_PROTOCOL` is wired onto all agent prompts and the profiler (openwiki-first). The `openwiki` MCP server backs it — if it fails to connect, note that rather than assuming the capability is absent

## Linting & Type Checking

- **ruff**: line-length 88, **target py310** (deliberately below the 3.12 runtime floor — py312 turns on UP042 `class X(str, Enum) → StrEnum`, a behavior-changing library-wide migration deferred to its own task). Rules `E, F, I, N, W, UP, B, SIM, S, RUF, C4, PTH`; ignores `RUF002, RUF003`. Per-file ignores: `N803, N806` for `packages/portopt-core/optimizer/**` and `tests/**` (sklearn `X, y`), `S101` for tests, `E402` for the library `__init__.py`
- **mypy**: strict, `python_version = "3.12"`, `ignore_missing_imports = true`. Overrides relax `disallow_subclassing_any` for sklearn/skfolio bases. `portopt-db` has its own strict `[tool.mypy]` gate (`mypy src/portopt_db`)
- **pyright**: run in CI as `uv run pyright`, pinned **`pyright[nodejs]==1.1.398`**, `pythonVersion = "3.10"`, `include = ["packages/portopt-core/optimizer"]`
- **Runtime deps** (root `pyproject.toml`): `numpy==2.5.2`, `pandas==3.0.5`, `scipy==1.18.1`, `scikit-learn==1.9.0`, `skfolio==1.0.6`, `jinja2==3.1.6`. `arch` is NOT declared — it reaches bootstrap uncertainty-set classes transitively via skfolio. Code importing it directly must guard with `try/except ImportError` or declare it explicitly
- Coverage floors: `portopt-core` ≥ 90, `portopt-db` ≥ 90, `ingestion` ≥ 80, `fund` ≥ 80 — all also gated branch ≥ 0.80 via `scripts/check_branch_coverage.py`
