# Optimizer

> Portfolio optimization in one **uv workspace**: a composable skfolio /
> scikit-learn optimization **library**, a scheduled market-data **ingestion
> daemon** that keeps PostgreSQL fresh, a shared **database layer**, and an agentic
> **fund** where an LLM chooses, the optimizer computes, and paper trades wait
> behind human approval gates — four packages, one shared virtual environment.

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
| Run an LLM-driven fund manager that decides and lets the optimizer compute | the **`fund`** bridge (`portopt-fund`) | Yes — PostgreSQL + a chat-model endpoint (DeepSeek on Ollama Cloud by default; any of ten providers via `LLM_PROVIDER`) |

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
estimators, factor definitions, investability screens, presets) live in each
submodule's source and docstrings under `packages/portopt-core/optimizer/`.

---

## Requirements

- **Python ≥ 3.12** (all four packages; the numpy 2.5 / pandas 3.0 pins make
  3.12 a hard floor). CI runs 3.12 and 3.13.
- **[uv](https://docs.astral.sh/uv/)** — the workspace resolver / runner.
- **Docker + the `docker compose` v2 plugin** — only for the ingestion daemon
  and the fund bridge (they need PostgreSQL 16). Not needed for the library.
- Optional API keys (ingestion): `TRADING_212_API_KEY` (+ secret),
  `FRED_API_KEY`. Absent Trading 212 ⇒ only the `t212_annotate` step skips cleanly;
  the universe itself is built from the yfinance Screener either way.

---

## Installation

### 1. The optimization library only (no Docker, no DB)

```bash
# From a clone, into the current environment (the repo root declares `portopt-core`):
pip install -e .                            # editable install of dist `portopt-core`
```

Then `import optimizer` works with nothing else running. The library has no
sibling-package dependencies, so plain pip is safe here. The daemon and the fund
bridge are not: they depend on `portopt-db` / `portopt-core`, which exist only as
workspace paths that uv resolves (pip ignores `[tool.uv.sources]` and would look
those unpublished names up on PyPI), so install those two through uv.

### 2. The full platform (ingestion daemon)

The daemon ships an install wizard. The one-line bootstrap needs **git**: it clones
this repo, installs `uv`, installs the `portopt` CLI straight from it (the CLI and the
daemon are never published to PyPI; `release.yml` builds only the `portopt-core`
library), then launches the wizard:

```bash
# macOS / Linux
curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/development/install.sh | bash
# Windows (PowerShell)
powershell -c "irm https://raw.githubusercontent.com/SilvioBaratto/optimizer/development/install.ps1 | iex"
```

Both scripts pull from the `development` branch by default; set `PORTOPT_REF`
(a branch, tag or commit SHA) or `PORTOPT_REPO` before running to install a
different ref or fork.

`portopt` runs `docker compose` and renders its plaintext secrets inside a **checkout**
of this repo (it owns `docker-compose.yml`), and installing the CLI into a `uv` tool
venv does not provide one. The scripts therefore clone into `~/optimizer`
(`%USERPROFILE%\optimizer` on Windows; set `PORTOPT_HOME` to choose elsewhere, which
`install.sh` also accepts as `PORTOPT_DIR`), reuse that directory if it already holds a
checkout, and export `OPTIMIZER_REPO` for the wizard, which then remembers the location
in `~/.portopt/config.toml`. `install.sh` also launches the wizard from inside the
checkout and, on a re-run, updates an existing git checkout to `PORTOPT_REF`; it refuses
to touch a directory that is a checkout of some other repo. Without a checkout,
`portopt setup` stops with "Cannot locate the optimizer checkout" before it persists
anything.

Equivalently, by hand — clone first, then run the from-clone front door:

```bash
git clone https://github.com/SilvioBaratto/optimizer && cd optimizer
./setup.sh      # setup.cmd / setup.ps1 on Windows; interactive wizard (see below)
```

To install the CLI yourself, point it at an existing checkout:

```bash
uv tool install "git+https://github.com/SilvioBaratto/optimizer.git@development#subdirectory=ingestion"
OPTIMIZER_REPO=/path/to/optimizer portopt setup
```

**`portopt setup`** verifies Docker + the compose plugin, validates your
Trading 212 / FRED keys live, encrypts them (Fernet + scrypt) to
`~/.portopt/secrets.enc` (mode `0600`, passphrase never persisted), brings up
PostgreSQL (`docker compose up -d --wait db`), and applies the Alembic migrations.
It seeds no data. Re-run any time with `portopt setup`.

Lifecycle:

```bash
portopt start     # decrypt secrets → render as compose secrets → docker compose --profile fund up -d --wait
portopt stop      # stop the fund engine + delete its plaintext LLM keys (daemon + db keep running)
portopt stop --all  # stop the whole stack + delete every rendered plaintext secret
portopt status    # report Docker + service health (non-zero exit if anything is down)
```

> `portopt start` renders the encrypted secrets into git-ignored
> `./secrets/<name>` files. `portopt stop` deletes the LLM keys only the fund
> mounts, but keeps the three the ingestion daemon mounts (Trading 212 key and
> secret, FRED key): the daemon restarts on its own and needs them. Run
> `portopt stop --all` to stop the daemon too and leave no plaintext on disk.
> Both refuse while a `fund run`, `fund approve` or `fund-tui` is running inside
> the fund container, because stopping it would kill that round mid-flight; add
> `--force` to stop anyway.

### 3. Developer setup (all four packages)

```bash
git clone https://github.com/SilvioBaratto/optimizer
cd optimizer
uv sync --all-packages --all-extras     # one venv with every member + every extra
docker compose --profile ingestion up -d # PostgreSQL (54320) + Adminer (18081) + scheduler (metrics 9000)
export DATABASE_URL=postgresql://postgres:postgres@localhost:54320/optimizer_db   # alembic reads only this variable, never .env
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

See each submodule's source and docstrings under
`packages/portopt-core/optimizer/` for every config, factory, preset and shape
contract, and the `skfolio`/`yfinance` skills under
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
`daily`, `refetch-all`, `universe`, `yfinance`, `macro`, `fx`, `fred`, `news`,
`market-structure`, `calendars`, `market-summary`, `options`, `daily-events`
(`--backfill START:END` rebuilds a date range), plus the lifecycle commands
`setup`, `start`, `stop`, `status`.

### The fund manager (`fund`)

The `fund` bridge lets an LLM make portfolio decisions and delegate the numeric
work to `optimizer`, persisting runs to PostgreSQL via `portopt_db`. Orders are
paper tickets, and by default a round pauses at a human-in-the-loop gate before
`place_orders` (each portfolio's mandate sets its gates). The chat
model is chosen by `LLM_PROVIDER` from a ten-provider registry (ollama, openai,
openrouter, anthropic, google, groq, nvidia, huggingface, aws, microsoft); set
nothing and it stays on the verified default — DeepSeek on Ollama Cloud. Install
a provider's SDK with the matching extra (`uv sync --package portopt-fund
--extra <provider>`, or `--extra all-llm`; `openrouter` and `microsoft` use
`--extra openai`, `ollama` needs none). CLI (`fund <command>`):

| Command | What it does |
|---|---|
| `fund profile <portfolio_id>` | Build/refresh an investor profile from a questionnaire |
| `fund mandate set <portfolio_id>` / `fund mandate show <portfolio_id>` | Write or display a portfolio's mandate (capital, base currency, drift threshold, benchmark) |
| `fund run <portfolio_id> --asof <date>` | Run a decision round for a rebalance bar |
| `fund approve <run_id>` / `fund reject <run_id>` | Resolve a paused (human-in-the-loop) run |
| `fund status [portfolio_id]` | Show the paused-run queue |
| `fund report <run_id>` | Tabular audit of one run |
| `fund nav-mark [--rebuild]` | Mark every active mandate's NAV now (the daily step, run by hand) |

Two more entry points ship with it:

- **`fund-worker`** — the daemon: a weekly rebalance sweep, an interval drift
  monitor, a daily NAV mark and an orphan-job reaper (a heartbeat lease, like the
  ingestion reaper). The NAV mark waits for the day's ingestion price sweep to
  finish (up to `FUND_NAV_MARK_WAIT_SECONDS`, 4 h by default) so it values holdings
  at fresh closes.
- **`fund-tui`** — a Textual oversight cockpit with eight views (Posture,
  Deliberation, Decisions, Mandate, Audit, Performance, Ask, Daily brief). Every
  action it takes (approve, reject, freeze, mandate edits) goes through a confirm
  and lands in the audit log; **Ask** answers questions about the fund in plain
  language. Press `?` for the key map.

---

## Dependencies

Each member declares its own; the single shared venv installs one version of
each shared pin. Exact runtime pins:

**`portopt-core`** (library): `numpy==2.5.3`, `pandas==3.0.6`, `scipy==1.18.1`,
`scikit-learn==1.9.1`, `skfolio==1.9.10`, `jinja2==3.1.6`.
*(`arch` is **not** declared — it reaches bootstrap uncertainty-set classes
transitively via skfolio. Code importing it directly must guard with
`try/except ImportError` or declare it.)*

**`portopt-db`** (DB layer): `SQLAlchemy==2.0.52`, `psycopg2-binary==2.9.12`,
`alembic==1.19.1`, `pandas==3.0.6`, `pydantic==2.13.4`. Owns the SQLAlchemy pin
the whole workspace inherits.

**`portopt`** (ingestion): `yfinance==1.7.0`, `scikit-learn==1.9.1` (declared,
for DBSCAN price-repair), `numpy==2.5.3`, `pandas==3.0.6`, `APScheduler==3.11.3`,
`SQLAlchemy==2.0.52`, `exchange_calendars==4.13.2`, `httpx==0.28.1`,
`beautifulsoup4==4.15.0`, `requests==2.34.2`, `prometheus-client==0.26.0`,
`typer==0.27.1`, `questionary==2.1.1`, `rich==15.0.0`, `cryptography==50.0.1`,
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
uv run pyright                              # all four packages, via [tool.pyright] include

# Ingestion daemon (portopt) — member suites run from the member directory
cd ingestion && uv run pytest tests/

# Shared DB layer (portopt-db)
cd packages/portopt-db && uv run pytest
cd packages/portopt-db && DATABASE_URL=postgresql://postgres:postgres@localhost:54320/optimizer_db alembic upgrade head   # head: a7c9e1b3d5f2

# Fund bridge (portopt-fund)
cd fund && uv run pytest tests/
```

`--package` only selects the dependency set; it does not change the working
directory. Run from the repo root, `pytest` always collects the root
`testpaths = ["tests"]` (the library suite), so a member suite must be started
from its own directory, exactly as CI does.

The root `Makefile` (`make lint | format | typecheck | test | all`) runs the same
ruff and mypy gates as the CI `lint`/`typecheck` jobs: ruff over the library,
`ingestion/`, `fund/` and `packages/portopt-db/`, and mypy over the library,
`portopt-db` and `ingestion`. `make test` runs only the library
suite.

**CI** (`.github/workflows/ci.yml`, push/PR to `main` and `development`, Ubuntu, Python 3.12 &
3.13, uv-driven) runs nine jobs: `lint` (ruff check → ruff format --check on
all four members → shellcheck → pip-audit), `typecheck` (mypy), `pyright` (pinned `1.1.414`), `test`
(`--cov=optimizer` ≥ 90%), `ingestion-test` (`--cov=app` ≥ 80%),
`portopt-db-test` (`--cov=portopt_db` ≥ 90%), `fund-test` (`--cov=fund` ≥
80%), `postgres-integration` (the Alembic upgrade to head on a real PostgreSQL plus
the `integration`-marked tests), and `docker-images` (compose config + build of
both images). `release.yml` publishes `portopt-core` to PyPI on `v*` tags, but only
after checking that the tagged commit is on `main`, that the tag equals
`v<pyproject version>`, and that the whole CI workflow passes on it.

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
├── install.sh / install.ps1       # one-line bootstrap (clone + uv + the setup wizard)
├── setup.sh / .cmd / .ps1         # the from-clone setup front door
└── docker-compose.yml             # db + adminer + scheduler + fund (every service profile-gated)
```

---

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
