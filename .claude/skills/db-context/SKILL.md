---
name: db-context
description: >
  Complete knowledge of the optimizer PostgreSQL database: 28 ingestion tables, schema,
  relationships, live row counts, query patterns, and conventions. Load this skill proactively
  whenever working with database models (api/app/models/), repositories (api/app/repositories/),
  Alembic migrations (api/alembic/), SQL queries, or any code that reads from or writes to the
  database. Also load when discussing table structure, data contents, schema design, or debugging
  data issues. This skill eliminates the need to read model files for schema questions.
---

# Optimizer Database Reference

PostgreSQL 16 (Alpine) in Docker on host port **54320**.
Connection: `postgresql://postgres:postgres@localhost:54320/optimizer_db`

**Every table here is ingestion.** The database is written by the headless daemon in `api/`
(`app/worker.py` on a schedule, `app/cli.py` by hand) and by nothing else. There is no HTTP API
and no portfolio/optimization/backtest state — migration `d1e2f3a4b5c6` dropped those 17 tables,
and it is one-way (its `downgrade()` raises). If you find a reference to `portfolios`,
`portfolio_snapshots`, `broker_positions`, `optimization_runs`, `backtest_runs`, `factor_scores`,
`risk_limits`, `rebalancing_policies`, `api_keys`, or `regime_states`, it is a leftover — delete it
rather than reviving the table.

Verified against the live database on 2026-07-11 (migration head `d1e2f3a4b5c6`). If models have
changed since, re-read `api/app/models/__init__.py`.

- Column-by-column schema → `references/full-schema.md`
- Live volumes, FRED series, indicator lists → `references/data-inventory.md`

---

## 1. Table Catalog (28 tables)

### Core (2)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `exchanges` | Exchange | 6 | name | NYSE, NASDAQ, LSE, Xetra, Euronext Paris, … |
| `instruments` | Instrument | 2,914 | (ticker, exchange_id) | Securities with yfinance mapping + delisting tracking |

`instruments` is the head of the pipeline: every other ingestion step iterates it, so a stale
universe silently caps what yfinance fetches. It is rebuilt weekly from Trading 212
(`universe_build`, Sun 02:00 — deliberately ahead of `weekly_refetch`).

### yfinance Market Data (11, all FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `ticker_profiles` | TickerProfile | 2,909 | instrument_id | 70+ fields: fundamentals, valuation, margins, dividends |
| `price_history` | PriceHistory | 3.72M | (instrument_id, date) | Daily OHLCV, Numeric(20,6) |
| `financial_statements` | FinancialStatement | 5.18M | (instrument_id, statement_type, period_type, period_date, line_item) | EAV: income/balance/cashflow/earnings, Numeric(38,6) |
| `dividends` | Dividend | 140,540 | (instrument_id, date) | Payment amounts, Numeric(20,6) |
| `stock_splits` | StockSplit | 4,555 | (instrument_id, date) | Split ratios, Numeric(20,6) |
| `analyst_recommendations` | AnalystRecommendation | 10,789 | (instrument_id, period) | Strong buy/buy/hold/sell/strong sell counts |
| `analyst_price_targets` | AnalystPriceTarget | 2,897 | instrument_id | Current/low/high/mean/median targets |
| `institutional_holders` | InstitutionalHolder | 29,093 | (instrument_id, holder_name) | Name, shares, value, pct_held |
| `mutual_fund_holders` | MutualFundHolder | 29,695 | (instrument_id, holder_name) | Same shape as institutional |
| `insider_transactions` | InsiderTransaction | 179,763 | (instrument_id, insider_name, start_date, transaction_type) | Insider trades with position/ownership |
| `ticker_news` | TickerNews | 109,726 | (instrument_id, news_uuid) | Per-stock news with `full_content` scraping |

### Macro (11)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `economic_indicators` | EconomicIndicator | 4 | country | Il Sole 24 Ore forecast snapshot (latest per country) |
| `economic_indicator_observations` | EconomicIndicatorObservation | 64 | (country, date) | Il Sole forecast time-series |
| `trading_economics_indicators` | TradingEconomicsIndicator | 138 | (country, indicator_key) | Latest macro indicator values |
| `trading_economics_observations` | TradingEconomicsObservation | 2,208 | (country, indicator_key, date) | Macro indicator time-series |
| `bond_yields` | BondYield | 16 | (country, maturity) | Latest yield + day/month/year changes |
| `bond_yield_observations` | BondYieldObservation | 256 | (country, maturity, date) | Yield-curve history |
| `fred_observations` | FredObservation | 150,952 | (series_id, date) | 14 FRED series (spreads, VIX, recession) |
| `macro_news` | MacroNews | 569 | news_id | Macro articles with `full_content` |
| `macro_news_themes` | MacroNewsTheme | 995 | (news_id, theme) | Theme tags (junction) |
| `macro_news_summaries` | MacroNewsSummary | 64 | (country, summary_date) | **LLM-generated** daily country summaries (BAML `SummarizeCountryNews`) |
| `macro_calibrations` | MacroCalibration | 5 | country | **LLM-generated** regime classification (BAML `ClassifyMacroRegime`) |

Il Sole 24 Ore and Trading Economics are **scraped from HTML and take no API key**. FRED needs
`FRED_API_KEY`. The two LLM tables are written by the `summarize` and `calibrate` steps via Ollama.

### Operations (2)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `background_jobs` | BackgroundJob | ~7 | (none) | Job tracking: pending/running/completed/failed, + `worker_pid`, `worker_host`, `last_heartbeat_at` |
| `background_job_errors` | BackgroundJobError | 0 | (job_id, error_index) | Child: ordered error messages |

`background_jobs` is the **only** place to read job progress — there is no polling endpoint.

### Infrastructure (2, not in models)
| Table | Purpose |
|-------|---------|
| `alembic_version` | Migration version tracking |
| `apscheduler_jobs` | APScheduler persistent job store (survives restarts → misfired runs replay) |

---

## 2. Relationship Map

```
exchanges 1--* instruments (CASCADE)
    instruments 1--* ticker_profiles         (CASCADE, passive_deletes)
    instruments 1--* price_history           (CASCADE, passive_deletes)
    instruments 1--* financial_statements    (CASCADE, passive_deletes)
    instruments 1--* dividends               (CASCADE, passive_deletes)
    instruments 1--* stock_splits            (CASCADE, passive_deletes)
    instruments 1--* analyst_recommendations (CASCADE, passive_deletes)
    instruments 1--* analyst_price_targets   (CASCADE, passive_deletes)
    instruments 1--* institutional_holders   (CASCADE, passive_deletes)
    instruments 1--* mutual_fund_holders     (CASCADE, passive_deletes)
    instruments 1--* insider_transactions    (CASCADE, passive_deletes)
    instruments 1--* ticker_news             (CASCADE, passive_deletes)

background_jobs 1--* background_job_errors (CASCADE, delete-orphan, lazy="selectin")
macro_news      1--* macro_news_themes     (CASCADE, delete-orphan, lazy="selectin")
```

**The macro observation tables have no FK at all** — `economic_indicator_observations`,
`trading_economics_observations`, `bond_yield_observations`, and `fred_observations` are
standalone time-series keyed by (country/series, date).

**Key distinction**: instrument children use `passive_deletes=True` (the DB cascades). Job and
news children use `cascade="all, delete-orphan"` with `lazy="selectin"` (SQLAlchemy manages the
lifecycle and eager-loads).

---

## 3. Base Model Pattern

All models inherit `BaseModel` from `api/app/models/_shared/base.py`:

```python
class Base(DeclarativeBase):
    type_annotation_map = {datetime: DateTime(timezone=True)}

class TimestampMixin:
    created_at = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class UUIDPrimaryKeyMixin:
    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)

class BaseModel(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __abstract__ = True
    def to_dict(self) -> dict[str, Any]: ...
```

Every table gets `id` (UUID PK), `created_at`, `updated_at` (both DateTime TZ).

Models live in domain folders: `app/models/{universe,market_data,macro,jobs}/`, plus `_shared/`.

---

## 4. Design Patterns

### Upsert (ON CONFLICT)
All bulk writes go through `RepositoryBase._upsert()` (`app/repositories/_shared/base.py`):
```python
stmt = pg_insert(Model).values(rows)
stmt = stmt.on_conflict_do_update(
    constraint="<constraint_name>",       # must match the UniqueConstraint name= in the model
    set_={col: stmt.excluded[col] for col in update_columns} | {"updated_at": func.now()},
)
```
The constraint name is the `name=` from the model's `UniqueConstraint`. `id` and `created_at` are
always excluded from updates. This is what makes every fetch step safely re-runnable.

### Repositories
- `RepositoryBase(session)` — session + `_upsert()`
- `BaseRepository(RepositoryBase, Generic[M, C, U])` — generic CRUD
- Domain repos extend `RepositoryBase` directly:
  `YFinanceRepository`, `MacroRegimeRepository`, `SentimentRepository`, `UniverseRepository`,
  `BackgroundJobRepository`, `DatabaseAdminRepository`

### Session Management
- **Synchronous** `Session` (not `AsyncSession`)
- `autoflush=False`, `autocommit=False`, `expire_on_commit=False`
- **There is no request scope and no `get_db()` dependency** — the HTTP layer is gone. Everything
  opens its own session via `database_manager.get_session()` (a context manager: rolls back on
  error, invalidates on disconnect, always closes)
- Lazy init: the first `get_session()` initializes the engine if startup did not

### Child-table properties
`errors` and `themes` are stored as child rows, not JSONB, and reconstructed via `@property`:
- `BackgroundJob.errors` → `[message, ...]` from `error_entries`
- `MacroNews.themes` → comma-joined string from `theme_entries`

### EAV
`financial_statements` is the one EAV table:
`(instrument_id, statement_type, period_type, period_date, line_item)` → `value` as Numeric(38,6).
It is by far the largest table (5.18M rows).

### Time-Series Deduplication
Composite unique constraints make re-fetch upserts safe:
- `(series_id, date)` — fred_observations
- `(country, indicator_key, date)` — trading_economics_observations
- `(country, maturity, date)` — bond_yield_observations
- `(country, date)` — economic_indicator_observations
- `(instrument_id, date)` — price_history, dividends, stock_splits

### Survivorship Bias
`instruments.delisted_at` (Date) and `delisting_return` (Float) are set when a ticker disappears
from Trading 212, and cleared on re-import (re-activation). Do not filter them out blindly —
excluding delisted rows from a backtest is exactly how survivorship bias gets in.

---

## 5. Connection & Configuration

```
Engine defaults (app/config.py): QueuePool, pool_size=5, max_overflow=10,
                                 pool_timeout=30s, pool_recycle=3600s
Pre-ping: True (detects stale connections)   Reset on return: rollback
Driver:   psycopg2 with keepalives (idle=30s, interval=10s, count=3)
Health:   cached 30s
Docker:   postgres:16-alpine, container "optimizer_db", volume "postgres_data", host port 54320
```

Defaults are overridable via `.env` (`DATABASE_POOL_SIZE`, …); the running container may show
larger values than the defaults above.

---

## 6. Migration Conventions

```bash
cd api && alembic upgrade head        # apply
cd api && alembic current             # show version
cd api && alembic revision --autogenerate -m "add_foo_table"
```

- **40 migrations** in `api/alembic/versions/`; current head **`d1e2f3a4b5c6`** (drop non-ingestion tables)
- **`d1e2f3a4b5c6` is destructive and one-way** — its `downgrade()` raises rather than recreate 17
  empty tables whose rows and owning code are both gone. To go back, restore a pre-upgrade dump
- All FKs use `ondelete="CASCADE"`; all timestamps get `server_default=sa.func.now()`
- UUID PKs use `UUID(as_uuid=True)` from `sqlalchemy.dialects.postgresql`
- Data migrations use `op.execute()` with raw SQL

---

## 7. Common Query Patterns

```python
# Lookup by FK
session.execute(select(PriceHistory).where(PriceHistory.instrument_id == iid))

# Date range
select(PriceHistory).where(
    PriceHistory.instrument_id == iid,
    PriceHistory.date >= start,
    PriceHistory.date <= end,
)

# Staleness check — drives incremental fetch
select(func.max(PriceHistory.date)).where(PriceHistory.instrument_id == iid)

# Every instrument worth fetching
repo.get_instruments_with_yfinance_ticker()   # non-null, non-empty yfinance_ticker, exchange eager-loaded

# Benchmark coverage (drives reference-index re-seeding)
repo.get_benchmark_coverage(["SPY", "QQQ"])   # -> {ticker: (row_count, latest_date)}

# Bulk upsert
repo._upsert(Model, rows, constraint="uq_constraint_name", update_columns=[...])

# Idempotent insert (ignore duplicates)
pg_insert(Model).values(rows).on_conflict_do_nothing(constraint="uq_...")

# Job state — the only way to read progress
select(BackgroundJob).where(BackgroundJob.job_type == "yfinance_fetch").order_by(BackgroundJob.started_at.desc())
```

---

## 8. Adding a New Table

1. **Model**: class in `api/app/models/<domain>/<file>.py` inheriting `BaseModel`
2. **UniqueConstraint**: in `__table_args__`, explicit `name="uq_<table>_<cols>"` — the upsert path needs it
3. **Indexes**: `Index("ix_<table>_<col>", "<col>")` in `__table_args__`
4. **Register**: import in `api/app/models/__init__.py` and add to `__all__` (Alembic autogenerate reads `Base.metadata`)
5. **Repository**: in `api/app/repositories/<domain>/` extending `RepositoryBase`
6. **Migration**: `alembic revision --autogenerate` → review types/constraints/indexes before applying
7. **SQLite compat**: if the table needs JSONB, use `JSON().with_variant(JSONB, "postgresql")` or the test suite cannot create it

---

## 9. Gotchas

- **`Column(..., index=True)` + a separate `op.create_index`** in the same migration throws
  `DuplicateTable` on a fresh DB — `index=True` already auto-creates `ix_<table>_<col>`.
- **TickerProfile is 1:1** — unique on `instrument_id`, but `Instrument` declares
  `profiles: Mapped[list[TickerProfile]]` (a list, not a scalar).
- **Numeric, not Float** — `price_history`, `dividends`, `stock_splits`, `analyst_price_targets`
  use `Numeric(20,6)`; `financial_statements` uses `Numeric(38,6)` for large balance-sheet values.
- **`BackgroundJob` JSON compat** — `JSON().with_variant(JSONB, "postgresql")` so SQLite tests can
  create the table. New models needing test coverage must follow this.
- **`insider_transactions`** — sentinel date `1970-01-01` when yfinance omits `start_date`.
- **`ticker_news.publish_time`** — `DateTime(timezone=True)` in the model, but historically stored
  without timezone info.
- **GICS sectors** — yfinance names differ from the GICS standard: "Financial Services" not
  "Financials", "Consumer Cyclical" not "Consumer Discretionary".
- **Sector ETF mapping** (`SentimentRepository`): XLK=Technology, XLF=Financial Services,
  XLE=Energy, XLP=Consumer Defensive, XLU=Utilities, XLB=Basic Materials, XLI=Industrials,
  XLV=Healthcare, XLY=Consumer Cyclical, XLRE=Real Estate, XLC=Communication Services.
- **`BackgroundJobService` opens its own session** via `database_manager.get_session`, so it never
  sees a session a test is holding. Tests must use the `patched_session_factory` fixture, or mock
  the module-level service instance in `app.services.jobs.scheduler`.
- **The orphan reaper is host-scoped** — `reconcile_orphans` fails any active row whose
  `worker_host != socket.gethostname()`. Two daemons on one database will reap each other's jobs.
  Run exactly one.
- **`macro_news_summaries` / `macro_calibrations` are LLM output**, not scraped facts. They are
  regenerated, not corrected — do not hand-edit rows.

---

## File Locations

```
Models:       api/app/models/{_shared,universe,market_data,macro,jobs}/
__init__:     api/app/models/__init__.py (imports + __all__ — Alembic reads Base.metadata from here)
Repositories: api/app/repositories/{_shared,universe,market_data,macro,jobs}/
Database:     api/app/database.py (DatabaseManager, get_session, pool config)
Config:       api/app/config.py (Settings, DATABASE_URL)
Migrations:   api/alembic/versions/ (40 files, head d1e2f3a4b5c6)
Alembic env:  api/alembic/env.py
Writers:      api/app/services/{market_data,macro,universe}/  ← the only code that writes
Docker:       docker-compose.yml (db on 54320, scheduler on 9000)
```
