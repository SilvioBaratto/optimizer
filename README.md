# Optimizer

> A portfolio-optimization platform in one **uv workspace**: a composable
> optimization **library**, a headless market-data **ingestion daemon**, a shared
> **database layer**, and an agentic **fund manager** — four packages, one shared
> virtual environment.

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-PolyForm--Noncommercial--1.0.0-orange)
![Lint: Ruff](https://img.shields.io/badge/lint-ruff-000000)
![Types: mypy | pyright](https://img.shields.io/badge/types-mypy%20%7C%20pyright-2a6db2)

---

## What's in here

`optimizer` is not a single program — it is a workspace of four cooperating
Python packages that share one virtual environment but keep strict import
boundaries:

| You want to… | Use | Needs a database / Docker? |
|---|---|---|
| Build and tune portfolio-optimization pipelines in your own code | the **`optimizer`** library (`portopt-core`) | **No** — pure Python (numpy/pandas/skfolio) |
| Keep a PostgreSQL store of market / fundamental / macro data fresh on a schedule | the **`portopt`** ingestion daemon | Yes — PostgreSQL (via Docker) |
| Share models, repositories and migrations across packages | the **`portopt-db`** layer | Yes — it *is* the DB layer |
| Run an LLM-driven fund manager that decides and lets the optimizer compute | the **`fund`** bridge (`portopt-fund`) | Yes — PostgreSQL + an Ollama model endpoint |

The **library installs and runs with zero Docker/DB dependency**. Docker +
PostgreSQL are required only for the ingestion daemon and the fund bridge, which
persist data. If all you want is optimization, install `portopt-core` and skip
the rest.

---

## Architecture

Four workspace members, one shared venv (`uv sync` resolves them together):

| Directory | Dist name | Import name | Role |
|---|---|---|---|
| `packages/portopt-core/` (declared from the repo-root `pyproject.toml`) | `portopt-core` | `optimizer` | Pure-Python optimization library (DB-agnostic, sklearn/skfolio-based) |
| `ingestion/` | `portopt` | `app` | Headless ingestion daemon (APScheduler in-process, no HTTP API) + the `portopt` CLI |
| `packages/portopt-db/` | `portopt-db` | `portopt_db` | Shared SQLAlchemy `Base` + models + repositories + `DatabaseManager` + the single Alembic tree |
| `fund/` | `portopt-fund` | `fund` | deepagents/langgraph bridge: *the LLM chooses, the optimizer computes* |

### Import boundaries (enforced by static import-scan tests)

Because there is one shared venv, install-time isolation does not exist — the
boundaries below are guarded by source-scan hygiene tests, not by what's
installed:

- `ingestion/` and `portopt-db/` do **not** import `optimizer`.
- `portopt-db/` carries no sklearn/skfolio stack.
- `optimizer` (`portopt-core`) does **not** import `portopt_db` — the library
  stays DB-free.
- `fund/` is the **only** member allowed to import both `optimizer` **and**
  `portopt_db`; `deepagents`/`langgraph` live only in `fund/`.

> The ingestion daemon *does* ship `scikit-learn` — not for optimization, but
> because yfinance's price-repair path (`repair=True`) imports
> `sklearn.cluster.DBSCAN`. Without it ~22% of tickers return empty history and
> get dropped. It is a data-layer dependency there.

### The optimization library

`optimizer` is a set of **composable primitives**, not a fixed end-to-end
runner. There is no `run_full_pipeline`; opinionated, DB-connected orchestration
lives in `fund/`. Every module follows the same convention: a **frozen
`@dataclass` config** + a **factory function** + **`str, Enum` types**.
Transformers are sklearn `BaseEstimator + TransformerMixin` and compose in
`sklearn.pipeline.Pipeline`.

**Module flow:**

```
prices → preprocessing → pre_selection → moments → views
       → optimization → validation → tuning → rebalancing
```

Plus the cross-cutting modules `factors/`, `synthetic/`, `scoring/`,
`universe/`, `distance/`, `cluster/`, `uncertainty_set/`, `linear_model/`,
`online/`, `fx/` — **18 submodules** in total.

The `optimization` module exposes **fifteen optimizer builders** (each a frozen
config + `build_*` factory): Mean-Risk (`build_mean_risk`) and its
regime-blended and robust variants, HRP, HERC, NCO, Schur-Complementary, Risk
Budgeting, Max-Diversification, Distributionally-Robust CVaR, Stacking,
Benchmark-Tracker, and three naive baselines (Equal-Weighted,
Inverse-Volatility, Random). Full per-module inventories (covariance/mu
estimators, factor definitions, investability screens, presets) live in
[`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md).

---

## Requirements

- **Python ≥ 3.12** (all four packages; the numpy 2.5 / pandas 3.0 pins make
  3.12 a hard floor). CI runs 3.12 and 3.13.
- **[uv](https://docs.astral.sh/uv/)** — the workspace resolver / runner.
- **Docker + the `docker compose` v2 plugin** — only for the ingestion daemon
  and the fund bridge (they need PostgreSQL 16). Not needed for the library.
- Optional API keys (ingestion): `TRADING_212_API_KEY` (+ secret),
  `FRED_API_KEY`. Absent Trading 212 ⇒ the universe-build step skips cleanly.

---

## Installation

### 1. The optimization library only (no Docker, no DB)

```bash
# From a clone, into the current environment:
pip install -e packages/portopt-core        # editable install of dist `portopt-core`
```

Then `import optimizer` works with nothing else running.

### 2. The full platform (ingestion daemon)

The daemon ships an install wizard. The one-line bootstrap installs `uv`, runs
`uv tool install portopt`, then launches the wizard:

```bash
# macOS / Linux
curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/main/install.sh | bash
# Windows (PowerShell)
powershell -c "irm https://raw.githubusercontent.com/SilvioBaratto/optimizer/main/install.ps1 | iex"
```

Equivalently, by hand:

```bash
uv tool install portopt      # installs the `portopt` CLI on your PATH
portopt setup                # interactive wizard (see below)
```

**`portopt setup`** verifies Docker + the compose plugin, validates your
Trading 212 / FRED keys live, encrypts them (Fernet + scrypt) to
`~/.portopt/secrets.enc` (mode `0600`, passphrase never persisted), brings up
PostgreSQL (`docker compose up -d --wait db`), and runs `alembic upgrade head`.
It seeds no data. Re-run any time with `portopt setup`.

Lifecycle:

```bash
portopt start     # decrypt secrets → render as compose secrets → docker compose up -d
portopt stop      # docker compose down + delete the rendered plaintext secret files
portopt status    # report Docker + service health (non-zero exit if anything is down)
```

> `portopt start` renders the encrypted secrets into git-ignored
> `./secrets/<name>` files only for the lifetime of the stack; `portopt stop`
> deletes them.

### 3. Developer setup (all four packages)

```bash
git clone https://github.com/SilvioBaratto/optimizer
cd optimizer
uv sync --all-packages --all-extras     # one venv with every member + every extra
docker compose up -d                     # PostgreSQL (54320) + Adminer (18081) + scheduler (metrics 9000)
cd packages/portopt-db && alembic upgrade head   # apply migrations (single owner)
```

---

## Usage

### The optimization library

Feed **linear** returns (never log returns), keep temporal order, and compose
pre-selection + an optimizer into one flat sklearn `Pipeline`:

```python
from skfolio.preprocessing import prices_to_returns
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pre_selection import PreSelectionConfig, build_portfolio_pipeline

# `prices`: a wide DataFrame — DatetimeIndex, one column per ticker.
X = prices_to_returns(prices)            # linear returns; runs OUTSIDE the pipeline

estimator = build_mean_risk(MeanRiskConfig())          # frozen config + factory
pipeline = build_portfolio_pipeline(                   # flattens pre-selection + optimizer
    estimator,
    pre_selection_config=PreSelectionConfig(),
)
pipeline.fit(X)
weights = pipeline[-1].weights_          # fitted skfolio optimizer exposes weights_
```

`build_portfolio_pipeline` flattens the steps so nested params are tunable via
`get_params()` (e.g. `"optimizer__l2_coef"`, `"drop_correlated__threshold"`).
`sector_mapping`, `expiration_dates` and an `outlier_protection_mask` are
optional keyword arguments (plain values, not queried from any database).

See [`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md) for every config,
factory, preset and shape contract, and the `skfolio`/`yfinance` skills under
[`.claude/skills/`](.claude/skills/) for API-level guidance.

### The ingestion daemon

Run the daemon (blocks; SIGTERM to stop) — normally via Docker Compose, but it
can run in-process:

```bash
uv run --package portopt python -m app.worker
```

It schedules seven jobs with APScheduler in-process (a `SQLAlchemyJobStore`):

| Job | Cadence |
|---|---|
| `daily_pipeline` | daily 07:00 |
| `midday_news` | daily 14:00 |
| `universe_build` | Saturday 02:00 |
| `weekly_refetch` | Saturday 03:00 |
| `weekly_market_wide` | Saturday 04:00 |
| `fred_monthly` | 1st of month 08:00 |
| `orphan_reaper` | on an interval |

`universe_build` runs **before** `weekly_refetch` (a stale universe caps
yfinance). The reaper is a pure **heartbeat lease**: it fails any active
(`pending`/`running`) job whose `last_heartbeat_at` is NULL or older than the
lease TTL — so **run exactly one daemon per database**.

Trigger the same work manually (same job-slot / heartbeat path):

```bash
docker compose exec scheduler python -m app.cli daily
docker compose exec scheduler python -m app.cli refetch-all
docker compose exec scheduler python -m app.cli yfinance --mode full --period 5y
```

Full CLI surface (`portopt <command>` or `python -m app.cli <command>`):
`daily`, `refetch-all`, `universe`, `yfinance`, `macro`, `fred`, `news`,
`market-structure`, `calendars`, `market-summary`, `options`, plus the lifecycle
commands `setup`, `start`, `stop`, `status`.

### The fund manager (`fund`)

The `fund` bridge lets an LLM (DeepSeek via Ollama) make portfolio decisions
and delegate the numeric work to `optimizer`, persisting runs to PostgreSQL via
`portopt_db`. CLI (`fund <command>`):

| Command | What it does |
|---|---|
| `fund profile <portfolio_id>` | Build/refresh an investor profile from a questionnaire |
| `fund run <portfolio_id> --asof <date>` | Run a decision round for a rebalance bar |
| `fund approve <run_id>` / `fund reject <run_id>` | Resolve a paused (human-in-the-loop) run |
| `fund status [portfolio_id]` | Show the paused-run queue |
| `fund report <run_id>` | Tabular audit of one run |

Also ships `fund-worker` (the drift-monitoring daemon) and `fund-tui` (a watch
TUI).

---

## Dependencies

Each member declares its own; the single shared venv installs one version of
each shared pin. Exact runtime pins:

**`portopt-core`** (library): `numpy==2.5.2`, `pandas==3.0.5`, `scipy==1.18.1`,
`scikit-learn==1.9.0`, `skfolio==1.0.6`, `jinja2==3.1.6`.
*(`arch` is **not** declared — it reaches bootstrap uncertainty-set classes
transitively via skfolio. Code importing it directly must guard with
`try/except ImportError` or declare it.)*

**`portopt-db`** (DB layer): `SQLAlchemy==2.0.52`, `psycopg2-binary==2.9.12`,
`alembic==1.19.1`, `pandas==3.0.5`, `pydantic==2.13.4`. Owns the SQLAlchemy pin
the whole workspace inherits.

**`portopt`** (ingestion): `yfinance==1.6.0`, `scikit-learn==1.9.0` (declared,
for DBSCAN price-repair), `numpy==2.5.2`, `pandas==3.0.5`, `APScheduler==3.11.3`,
`SQLAlchemy==2.0.52`, `exchange_calendars==4.13.2`, `httpx==0.28.1`,
`beautifulsoup4==4.15.0`, `requests==2.34.2`, `prometheus-client==0.26.0`,
`typer==0.27.1`, `questionary==2.1.1`, `rich==15.0.0`, `cryptography==45.0.7`,
`pydantic==2.13.4`, `pydantic-settings==2.15.0`, `python-dotenv==1.2.3`, plus
`portopt-db` (workspace; psycopg2-binary/alembic arrive transitively).

**`portopt-fund`** (bridge): `deepagents==0.7.14` (the only exact pin),
`langgraph`, `langgraph-checkpoint-postgres`, `psycopg[binary]`, `psycopg-pool`,
`langchain-ollama`, `APScheduler`, `typer`, `textual`, `pydantic`,
`python-dotenv`, plus `portopt-core` + `portopt-db` (workspace).

---

## Development

`uv` drives everything. Run per-package with `uv run --package <name> …`.

```bash
# Library (portopt-core) — tests, lint, types
uv run --package portopt-core pytest tests/ -v
uv run --package portopt-core pytest -k "test_name"
uv run --package portopt-core ruff check packages/portopt-core/optimizer/ tests/
uv run --package portopt-core mypy packages/portopt-core/optimizer/
uv run pyright                              # scoped to the library via [tool.pyright]

# Ingestion daemon (portopt)
uv run --package portopt pytest

# Shared DB layer (portopt-db)
uv run --package portopt-db pytest
cd packages/portopt-db && alembic upgrade head     # head: b3c4d5e6f7a8

# Fund bridge (portopt-fund)
uv run --package portopt-fund pytest
```

There is a root `Makefile` (`make lint | format | typecheck | test | all`), but
note that it and the CI `lint`/`typecheck` jobs currently pass the bare path
`optimizer/`, which no longer exists after the library moved to
`packages/portopt-core/optimizer/`. Use the explicit paths above until that is
fixed.

**CI** (`.github/workflows/ci.yml`, push/PR to `main`, Ubuntu, Python 3.12 &
3.13, uv-driven) runs seven jobs: `lint` (ruff check → ruff format --check →
pip-audit), `typecheck` (mypy), `pyright` (pinned `1.1.398`), `test`
(`--cov=optimizer` ≥ 90% + branch ≥ 0.80), `ingestion-test` (`--cov=app` ≥ 80%),
`portopt-db-test` (`--cov=portopt_db` ≥ 90%), and `fund-test` (`--cov=fund` ≥
80%). `release.yml` builds `portopt-core` on `v*` tags.

---

## Repository layout

```
optimizer/
├── pyproject.toml                 # root = dist `portopt-core`; declares the uv workspace
├── packages/
│   ├── portopt-core/optimizer/    # the optimization library (import `optimizer`)
│   └── portopt-db/                # shared DB layer + the single Alembic tree
├── ingestion/                     # the `portopt` daemon + CLI (import `app`)
├── fund/                          # the `portopt-fund` bridge (import `fund`)
├── tests/                         # library test suite (mirrors optimizer/ + scheduler/)
├── scheduler/                     # shell wrappers over the CLI
├── scripts/                       # CI helpers (e.g. branch-coverage gate)
├── docker-compose.yml             # db + adminer + scheduler
└── .claude/ARCHITECTURE.md        # deep per-module reference
```

---

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
