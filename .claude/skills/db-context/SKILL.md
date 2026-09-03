---
name: db-context
description: >
  Complete knowledge of the optimizer PostgreSQL database: 57 ingestion tables, schema,
  relationships, live row counts, query patterns, and conventions. Load this skill proactively
  whenever working with database models (packages/portopt-db/src/portopt_db/models/), repositories
  (packages/portopt-db/src/portopt_db/repositories/ + ingestion/app/repositories/jobs/), Alembic
  migrations (packages/portopt-db/alembic/), SQL queries, or any code that reads from or writes to
  the database. Also load when discussing table structure, data contents, schema design, or
  debugging data issues. This skill eliminates the need to read model files for schema questions.
---

# Optimizer Database Reference

PostgreSQL 16 (Alpine) in Docker on host port **54320**.
Connection: `postgresql://postgres:postgres@localhost:54320/optimizer_db`

**Every table here is ingestion.** The database is written by the headless daemon in `ingestion/`
(`app/worker.py` on a schedule, `app/cli.py` by hand) and by nothing else. There is no HTTP API
and no portfolio/optimization/backtest state — migration `d1e2f3a4b5c6` dropped those 17 tables,
and it is one-way (its `downgrade()` raises). If you find a reference to `portfolios`,
`portfolio_snapshots`, `broker_positions`, `optimization_runs`, `backtest_runs`, `factor_scores`,
`risk_limits`, `rebalancing_policies`, `api_keys`, or `regime_states`, it is a leftover — delete it
rather than reviving the table.

**Schema, models, and the single Alembic tree all live in the `portopt-db` package**
(`packages/portopt-db/`), not in `ingestion/`. The daemon imports models from `portopt_db.models`
and repositories from `portopt_db.repositories`; only the `jobs` repository *behavior* still lives
in `ingestion/app/repositories/jobs/`.

Live totals: **57 base tables** (55 documented below + `alembic_version` + `apscheduler_jobs`).
Alembic head `b6c7d8e9f0a1`, 62 migrations. Row counts below are the last verified live snapshot.

- Column-by-column schema → `references/full-schema.md`
- Live volumes, FRED series, indicator lists → `references/data-inventory.md`

---

## 1. Table Catalog (55 documented + 2 infra)

### Core (2)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `exchanges` | Exchange | 14 | name | Exchange dimension; parent of instruments. `t212_id` now legacy-NULL |
| `instruments` | Instrument | 8,898 | (ticker, exchange_id) | Security master; yfinance-sourced. `yfinance_ticker` is the downstream join key |

`instruments` is the head of the pipeline: every other ingestion step iterates it, so a stale
universe silently caps what yfinance fetches. Rebuilt weekly (`universe_build`, Sat 02:00 —
deliberately ahead of `weekly_refetch`). Now built from the yfinance source, so `t212_ticker` and
`exchanges.t212_id` are legacy/NULL. `delisted_at`/`delisting_return` are currently all-NULL
(fresh universe); `delisting_return` default is CRSP-style −0.30 (−1.0 = bankruptcy).

### Per-ticker — Prices & Actions (4, FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `price_history` | PriceHistory | 9,786,726 | (instrument_id, date) | Daily OHLCV + div/split/capital-gains cols, Numeric(20,6). `price_unit` = listing ccy, never converted |
| `dividends` | Dividend | 372,607 | (instrument_id, date) | Cash dividend per share, Numeric(20,6), NOT NULL |
| `stock_splits` | StockSplit | 7,794 | (instrument_id, date) | Split ratio, Numeric(20,6), NOT NULL |
| `options_chain` | OptionContract | 0 | (instrument_id, as_of, contract_symbol) | Weekly option-chain snapshot (weekly_market_wide) |

### Per-ticker — Fundamentals (4, FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `ticker_profiles` | TickerProfile | 8,895 | instrument_id | 1:1 company profile/fundamentals from yf.info (~85 live cols; model maps ~60 — read the model) |
| `ticker_profile_extras` | TickerProfileExtra | 8,895 | instrument_id | 1:1 overflow yf.info fields (short interest, ownership %, governance risk) |
| `financial_statements` | FinancialStatement | 11,253,341 | (instrument_id, statement_type, period_type, period_date, line_item) | EAV; largest table. `statement_type` overloaded (also valuation_measures/eps_trend/eps_revisions/earnings) |
| `sec_filings` | SecFiling | 389,385 | (instrument_id, filing_date, form_type, title) | SEC filing index (US-only) |

### Per-ticker — Estimates (5, FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `earnings_estimate` | EarningsEstimate | 23,656 | (instrument_id, period) | Forward EPS estimates by period label (0q/+1q/0y/+1y) |
| `revenue_estimate` | RevenueEstimate | 30,408 | (instrument_id, period) | Forward revenue estimates by period label |
| `earnings_history` | EarningsHistory | 17,956 | (instrument_id, period_date) | Historical per-quarter EPS surprise |
| `growth_estimates` | GrowthEstimate | 40,203 | (instrument_id, period) | Analyst growth estimates (stock vs index trend) |
| `earnings_dates` | EarningsDate | 106,647 | (instrument_id, earnings_date) | Past/upcoming earnings dates + EPS surprise (per-ticker; distinct from market-wide `earnings_calendar`) |

### Per-ticker — Analyst (3, FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `analyst_actions` | AnalystAction | 419,930 | (instrument_id, action_date, firm, to_grade) | Rating upgrades/downgrades ledger |
| `analyst_price_targets` | AnalystPriceTarget | 7,949 | instrument_id | 1:1 12-mo target aggregates (snapshot only) |
| `analyst_recommendations` | AnalystRecommendation | 17,100 | (instrument_id, period) | Vote counts by trailing period (0m/-1m/…) |

### Per-ticker — Ownership (7, FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `insider_purchases` | InsiderPurchaseSummary | 8,024 | instrument_id | 1:1 6-month insider buy/sell summary |
| `insider_roster` | InsiderRosterHolder | 44,559 | (instrument_id, insider_name) | Point-in-time insider roster snapshot |
| `insider_transactions` | InsiderTransaction | 256,439 | (instrument_id, insider_name, transaction_type, start_date) | Insider trade ledger |
| `institutional_holders` | InstitutionalHolder | 58,417 | (instrument_id, holder_name) | 13F-style holder snapshot |
| `mutual_fund_holders` | MutualFundHolder | 52,398 | (instrument_id, holder_name) | Top mutual-fund holder snapshot |
| `major_holders` | MajorHolders | 7,734 | instrument_id | 1:1 insider/institution % breakdown (fractions 0-1) |
| `shares_outstanding` | SharesOutstanding | 789,921 | (instrument_id, date) | Daily share-count time series |

### Per-ticker — News (1, FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `ticker_news` | TickerNews | 61,898 | (instrument_id, news_uuid) | Per-stock news, optional scraped `full_content` |

### Calendars (4, market-wide, no FK)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `earnings_calendar` | EarningsCalendar | 3,996 | (ticker, event_date) | Market-wide earnings events, consensus vs realized EPS |
| `economic_event_calendar` | EconomicEventCalendar | 1,496 | (event, country, event_date) | Macro-economic calendar; actual/forecast/prior as raw strings |
| `split_calendar` | SplitCalendar | 477 | (ticker, split_date) | Upcoming splits; `ratio` is a String label ('5:1') |
| `ipo_calendar` | IpoCalendar | 11 | (ticker, ipo_date) | Upcoming/priced IPOs (no price/share data) |

Written by the market-wide sweep (`weekly_market_wide`, Saturday). No instrument FK — `ticker` is
free text that need not exist in `instruments`.

### ETF (8, FK → `instruments.id` CASCADE)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `etf_metadata` | ETFMetadata | 839 | instrument_id | 1:1 headline fund metadata (AUM/NAV/expense/category) |
| `etf_asset_classes` | ETFAssetClass | 839 | (instrument_id, as_of) | Asset-class allocation snapshot |
| `etf_holdings` | ETFHolding | 770 | (instrument_id, as_of, holding_symbol) | Top-N constituents |
| `etf_sector_weights` | ETFSectorWeight | 2,200 | (instrument_id, as_of, sector) | Sector allocation weights |
| `etf_equity_holdings` | ETFEquityHoldings | 839 | (instrument_id, as_of) | Equity-sleeve valuation metrics |
| `etf_bond_holdings` | ETFBondHoldings | 774 | (instrument_id, as_of) | Fixed-income duration/maturity/credit |
| `etf_bond_ratings` | ETFBondRating | 6,549 | (instrument_id, as_of, rating) | Credit-rating bucket weights |
| `etf_fund_operations` | ETFFundOperations | 839 | (instrument_id, as_of) | Expense ratio / turnover / net assets |

All ETF-only, point-in-time by `as_of` (except `etf_metadata`, 1:1 headline). Numeric cols read
back as `Decimal`.

### Sector & Market (4, market-wide, no FK)
| Table | Model | Rows | Unique On | Purpose |
|-------|-------|-----:|-----------|---------|
| `sector_snapshots` | SectorSnapshot | 0 | (sector_key, region, as_of) | yf.Sector overview rollups |
| `sector_industries` | SectorIndustry | 0 | (sector_key, region, as_of, industry_key) | Sector→industry taxonomy snapshot |
| `sector_top_companies` | SectorTopCompany | 0 | (sector_key, region, as_of, symbol) | Top constituents per sector |
| `market_summaries` | MarketSummary | 0 | (market, symbol, as_of) | Regional index/quote summaries (8 MARKET_IDENTIFIERS) |

Filled by the weekly market-wide sweep; empty until it first runs.

### Macro (11)
| Table | Model | Rows | Unique On | FK | Purpose |
|-------|-------|-----:|-----------|----|---------|
| `economic_indicators` | EconomicIndicator | 4 | country | — | Il Sole 24 Ore consensus FORECAST snapshot (latest per country) |
| `economic_indicator_observations` | EconomicIndicatorObservation | 4 | (country, date) | — | Il Sole forecast time-series |
| `trading_economics_indicators` | TradingEconomicsIndicator | 138 | (country, indicator_key) | — | Latest scraped TE indicator (real actuals) |
| `trading_economics_observations` | TradingEconomicsObservation | 138 | (country, indicator_key, date) | — | TE indicator time-series |
| `bond_yields` | BondYield | 16 | (country, maturity) | — | Latest yield + day/month/year changes |
| `bond_yield_observations` | BondYieldObservation | 16 | (country, maturity, date) | — | Yield-curve history |
| `fred_observations` | FredObservation | 0 | (series_id, date) | — | FRED series (needs `FRED_API_KEY`; empty until fred step runs) |
| `macro_news` | MacroNews | 36 | news_id | — | Macro articles, optional `full_content` |
| `macro_news_themes` | MacroNewsTheme | 66 | (news_id, theme) | → macro_news.id | Theme tags (junction, clear-then-re-add) |
| `macro_news_summaries` | MacroNewsSummary | 4 | (country, summary_date) | — | **LLM** daily country summaries (BAML `SummarizeCountryNews`) |
| `macro_calibrations` | MacroCalibration | 0 | country | — | **LLM** regime calibration (BAML `ClassifyMacroRegime`); empty until calibrate step |

Il Sole 24 Ore and Trading Economics are **scraped from HTML and take no API key**. FRED needs
`FRED_API_KEY`. The two LLM tables are written by the `summarize` and `calibrate` steps via the
**cloud-only** BAML client (openai|anthropic; Ollama removed). `economic_indicators` are FORECASTS,
`trading_economics_indicators` are realized actuals — do not conflate.

### Operations (2)
| Table | Model | Rows | Unique On | FK | Purpose |
|-------|-------|-----:|-----------|----|---------|
| `background_jobs` | BackgroundJob | 6 | (none) | — | Per-run job state (status/progress/heartbeat/worker id); JSONB `extra`/`result` |
| `background_job_errors` | BackgroundJobError | 12 | (job_id, error_index) | → background_jobs.id | Child: ordered error messages |

`background_jobs` is the **only** place to read job progress — there is no polling endpoint.
`status` is a free-text varchar (active = pending/running; terminal = failed/completed), not a DB
enum. Error strings live in the child table, not a column; `BackgroundJob.errors` is a computed
property and `update(errors=[...])` delete-and-reinserts child rows.

### Infrastructure (2, not in models)
| Table | Purpose |
|-------|---------|
| `alembic_version` | Migration version tracking |
| `apscheduler_jobs` | APScheduler persistent job store (survives restarts → misfired runs replay) |

---

## 2. Relationship Map

```
exchanges 1--* instruments (CASCADE)
    # every per-ticker table: FK instrument_id → instruments.id, ON DELETE CASCADE
    instruments 1--1 ticker_profiles          instruments 1--1 ticker_profile_extras
    instruments 1--* price_history             instruments 1--* dividends
    instruments 1--* stock_splits              instruments 1--* options_chain
    instruments 1--* financial_statements      instruments 1--* sec_filings
    instruments 1--* earnings_estimate         instruments 1--* revenue_estimate
    instruments 1--* earnings_history          instruments 1--* growth_estimates
    instruments 1--* earnings_dates
    instruments 1--* analyst_actions           instruments 1--1 analyst_price_targets
    instruments 1--* analyst_recommendations
    instruments 1--1 insider_purchases         instruments 1--* insider_roster
    instruments 1--* insider_transactions      instruments 1--* institutional_holders
    instruments 1--* mutual_fund_holders       instruments 1--1 major_holders
    instruments 1--* shares_outstanding        instruments 1--* ticker_news
    # ETF (all FK → instruments.id CASCADE)
    instruments 1--1 etf_metadata              instruments 1--* etf_asset_classes
    instruments 1--* etf_holdings              instruments 1--* etf_sector_weights
    instruments 1--* etf_equity_holdings       instruments 1--* etf_bond_holdings
    instruments 1--* etf_bond_ratings          instruments 1--* etf_fund_operations

background_jobs 1--* background_job_errors (CASCADE, delete-orphan)
macro_news      1--* macro_news_themes     (CASCADE, delete-orphan)
```

**No FK at all** (standalone, keyed by natural business columns):
- Calendars: `earnings_calendar`, `economic_event_calendar`, `split_calendar`, `ipo_calendar`
- Sector & Market: `sector_snapshots`, `sector_industries`, `sector_top_companies`, `market_summaries`
- Macro: `economic_indicators`, `economic_indicator_observations`, `trading_economics_indicators`,
  `trading_economics_observations`, `bond_yields`, `bond_yield_observations`, `fred_observations`,
  `macro_news`, `macro_news_summaries`, `macro_calibrations`

Note: `macro_news_themes.news_id` is the UUID FK to `macro_news.id`, distinct from
`macro_news.news_id` (the external String article id used for parent dedup).

---

## 3. Base Model Pattern

Models live in `portopt_db.base` + `portopt_db.models/{universe,market_data,macro,jobs}/`. Import
`portopt_db.models` (not `base`) before `create_all` — its `__init__` registers every table on
`Base.metadata`.

```python
# portopt_db/base.py
class Base(DeclarativeBase):
    type_annotation_map = {datetime: DateTime(timezone=True)}

class BaseModel(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __abstract__ = True   # id: UUID PK (default uuid4); created_at/updated_at: DateTime(tz)
```

Every table gets `id` (UUID PK), `created_at`, `updated_at` (both DateTime TZ). The natural dedup
key is the model's named `UniqueConstraint`, NOT the surrogate `id`.

---

## 4. Design Patterns

### Upsert (ON CONFLICT)
All bulk writes go through `RepositoryBase._upsert()` (`portopt_db.repository`):
```python
stmt = pg_insert(Model).values(rows)
stmt = stmt.on_conflict_do_update(
    constraint="<constraint_name>",       # the name= from the model's UniqueConstraint
    set_={col: stmt.excluded[col] for col in update_columns} | {"updated_at": func.now()},
)
```
`id` and `created_at` are excluded from updates. This is what makes every fetch step re-runnable.
PostgreSQL ON CONFLICT rejects the same key twice in one statement, so several writers de-dupe
in-Python before the upsert (e.g. `earnings_history`, `option_chain`, `etf_holdings`).

### Repositories (package split)
- `RepositoryBase(session)` — session + `_upsert()`; `BaseRepository` — generic CRUD
- Domain repos live in **`portopt_db.repositories/{macro,market_data,universe}/`** +
  `database_admin.py`: `YFinanceRepository`, `ETFMetadataRepository`, `CalendarsRepository`,
  `MarketStructureRepository`, `MarketSummaryRepository`, `MacroRegimeRepository`,
  `MacroSentimentRepository`, `UniverseRepository`, `DatabaseAdminRepository`
- **`BackgroundJobRepository` behavior stays in `ingestion/app/repositories/jobs/`** — it imports
  `RepositoryBase` from `portopt_db.repository` (the model is in `portopt_db.models.jobs`)

### Engine (DbConfig-injected)
- `DatabaseManager` (`portopt_db.engine`) takes a frozen `DbConfig` (`portopt_db.config`: url +
  pool options + `connect_args()`) — no coupling to `app.config`.
- Ingestion's `app/database.py` is a thin singleton: builds a `DbConfig` from `app.config.settings`
  and instantiates `DatabaseManager`.
- **Synchronous** `Session` (not `AsyncSession`); `autoflush=False`, `expire_on_commit=False`.
  There is no request scope — everything opens its own session via `database_manager.get_session()`.

### Child-table properties
`errors` and `themes` are child rows, not JSONB, reconstructed via `@property`:
- `BackgroundJob.errors` → `[message, ...]` ordered by `error_index`
- `MacroNews.themes` → comma-joined string from `theme_entries`

### EAV
`financial_statements` is the one EAV table:
`(instrument_id, statement_type, period_type, period_date, line_item)` → `value` Numeric(38,6).
`statement_type` is overloaded — filter on `statement_type` + `period_type`, not `statement_type`
alone. Largest table (~11.3M rows).

### Time-Series Deduplication
Composite unique constraints make re-fetch upserts safe:
`(series_id, date)` fred · `(country, indicator_key, date)` TE obs ·
`(country, maturity, date)` bond obs · `(country, date)` econ obs ·
`(instrument_id, date)` price_history/dividends/stock_splits/shares_outstanding.

### Survivorship Bias
`instruments.delisted_at` (Date) + `delisting_return` (Float, default −0.30) are set when a ticker
disappears, cleared on re-import. Currently all-NULL (fresh universe). Do not filter delisted rows
blindly — that reintroduces survivorship bias.

---

## 5. Connection & Configuration

```
DbConfig (portopt_db.config): url + pool_size/max_overflow/pool_timeout/pool_recycle + connect_args()
Pre-ping: True (detects stale connections)   Reset on return: rollback
Driver:   psycopg2 with keepalives           Health: cached 30s
Docker:   postgres:16-alpine, container "optimizer_db", volume "postgres_data", host port 54320
```

Overridable via `.env` (`DATABASE_POOL_SIZE`, …); the running container may show larger values.

---

## 6. Migration Conventions

```bash
cd packages/portopt-db && alembic upgrade head      # apply (single migration owner)
cd packages/portopt-db && alembic current           # show version
cd packages/portopt-db && alembic revision --autogenerate -m "add_foo_table"
```

- **62 migrations** in `packages/portopt-db/alembic/versions/`; current head **`b6c7d8e9f0a1`**
- Alembic `env.py` reads `DATABASE_URL` env (fallback ini) and `from portopt_db.models import Base`
- **`d1e2f3a4b5c6` is destructive and one-way** — its `downgrade()` raises rather than recreate the
  17 dropped non-ingestion tables. To go back, restore a pre-upgrade dump
- All instrument FKs use `ondelete="CASCADE"`; timestamps get `server_default=sa.func.now()`
- UUID PKs use `UUID(as_uuid=True)` from `sqlalchemy.dialects.postgresql`

---

## 7. Common Query Patterns

```python
# Lookup by FK
session.execute(select(PriceHistory).where(PriceHistory.instrument_id == iid))

# Date range
select(PriceHistory).where(
    PriceHistory.instrument_id == iid,
    PriceHistory.date >= start, PriceHistory.date <= end,
)

# Staleness check — drives incremental fetch
select(func.max(PriceHistory.date)).where(PriceHistory.instrument_id == iid)

# Every instrument worth fetching
repo.get_instruments_with_yfinance_ticker()   # non-null yfinance_ticker, exchange eager-loaded

# Bulk upsert
repo._upsert(Model, rows, constraint="uq_constraint_name", update_columns=[...])

# Idempotent insert (ignore duplicates)
pg_insert(Model).values(rows).on_conflict_do_nothing(constraint="uq_...")

# Job state — the only way to read progress
select(BackgroundJob).where(BackgroundJob.job_type == "yfinance_fetch").order_by(BackgroundJob.started_at.desc())
```

---

## 8. Adding a New Table

1. **Model**: class in `packages/portopt-db/src/portopt_db/models/<domain>/<file>.py` inheriting `BaseModel`
2. **UniqueConstraint**: in `__table_args__`, explicit `name="uq_<table>_<cols>"` — the upsert path needs it
3. **Indexes**: `Index("ix_<table>_<col>", "<col>")` in `__table_args__`
4. **Register**: import in `portopt_db/models/__init__.py` (Alembic autogenerate reads `Base.metadata`)
5. **Repository**: in `portopt_db/repositories/<domain>/` extending `RepositoryBase` (jobs behavior stays in ingestion)
6. **Migration**: `cd packages/portopt-db && alembic revision --autogenerate` → review before applying
7. **SQLite compat**: if the table needs JSONB, use `JSON().with_variant(JSONB, "postgresql")` or the test suite cannot create it

---

## 9. Gotchas

- **Numeric → Decimal on read** — most price/money/ratio cols are SQL `Numeric`, returned as Python
  `Decimal`, not float; cast before numpy/skfolio math. `price_history`/`dividends`/`stock_splits`
  use Numeric(20,6); `financial_statements` Numeric(38,6). ETF and estimate tables likewise.
- **Some percents are Float fractions (0-1)** — `major_holders`, `institutional_holders.pct_held`,
  `mutual_fund_holders.pct_held`, `ticker_profile_extras` ratios are double precision fractions
  (0.05 = 5%), not Numeric percentages.
- **`price_unit` is the listing currency as-is** — e.g. `GBX` = pence, never converted; prices
  across instruments can be mixed scale/currency. FX/scale normalization is the reader's job.
  `financial_statements.currency_code` is the MAJOR-unit code (GBX→GBP upstream), legacy-nullable.
- **`financial_statements.statement_type` is overloaded** — also carries `valuation_measures`
  (point_in_time), `eps_trend`/`eps_revisions` (estimate), `earnings`. Always filter with
  `period_type`. `line_item` labels are raw yfinance strings — brittle for joins.
- **`earnings_dates` (per-ticker, FK) ≠ `earnings_calendar` (market-wide, no FK)** — likewise
  `stock_splits` (per-ticker Numeric ratio) vs `split_calendar` (market-wide String label ratio).
- **`insider_transactions`** — sentinel date `1970-01-01` when yfinance omits `start_date`;
  `transaction_type` is often free-text prose (yfinance 1.3.0 stopped populating the clean code);
  in-Python dedup on the unique key drops colliding same-day rows (undercounts totals).
- **`ticker_news.publish_time`** — `DateTime(timezone=True)` column but stored tz-stripped (naive
  wall-clock). `macro_news.publish_time` is nullable.
- **1:1 tables declared as lists** — TickerProfile/MajorHolders/etc. are unique on `instrument_id`
  but `Instrument` may declare list relationships; snapshot tables retain no history.
- **`macro_calibrations` sentinel rows** — a fresh row can have `phase=''`, `delta=0/tau=0` before
  BAML runs; two write paths (LLM calibration vs rule-based `regime_classification`) each preserve
  the other's columns.
- **LLM tables are cloud-only output** — `macro_news_summaries`/`macro_calibrations` come from the
  BAML openai|anthropic client (Ollama removed). Regenerated, not corrected — do not hand-edit.
- **Orphan reaper is heartbeat-lease based** — `reap_orphans` fails any active row whose
  `last_heartbeat_at` is NULL/stale (> timeout, default 300s); `worker_pid`/`worker_host` are
  observability-only (the host/PID scope check was removed). Long synchronous steps must run under
  an active heartbeat or get falsely reaped. Still run exactly one daemon per DB.
- **`BackgroundJob` JSON compat** — `extra`/`result` use `JSON().with_variant(JSONB, "postgresql")`
  so SQLite tests can create the table; `update()` merges kwargs into `extra` (read-modify-write).
- **`Column(index=True)` + a separate `op.create_index`** in one migration throws `DuplicateTable`
  on a fresh DB — `index=True` already auto-creates `ix_<table>_<col>`.
- **GICS sectors** — yfinance names differ from GICS: "Financial Services" not "Financials",
  "Consumer Cyclical" not "Consumer Discretionary".
- **Sector ETF mapping** (`MacroSentimentRepository`): XLK=Technology, XLF=Financial Services,
  XLE=Energy, XLP=Consumer Defensive, XLU=Utilities, XLB=Basic Materials, XLI=Industrials,
  XLV=Healthcare, XLY=Consumer Cyclical, XLRE=Real Estate, XLC=Communication Services.

---

## File Locations

```
Models:       packages/portopt-db/src/portopt_db/models/{universe,market_data,macro,jobs}/
__init__:     packages/portopt-db/src/portopt_db/models/__init__.py (registers Base.metadata)
Base/mixins:  packages/portopt-db/src/portopt_db/base.py
Repositories: packages/portopt-db/src/portopt_db/repositories/{macro,market_data,universe}/ + database_admin.py
Jobs repo:    ingestion/app/repositories/jobs/  (behavior only; model in portopt_db.models.jobs)
Engine:       packages/portopt-db/src/portopt_db/engine.py (DatabaseManager, DbConfig-injected)
Config:       packages/portopt-db/src/portopt_db/config.py (DbConfig); ingestion/app/database.py (singleton), ingestion/app/config.py (Settings)
Migrations:   packages/portopt-db/alembic/versions/ (62 files, head b6c7d8e9f0a1)
Alembic env:  packages/portopt-db/alembic/env.py
Writers:      ingestion/app/services/{market_data,macro,universe}/  ← the only code that writes
Docker:       docker-compose.yml (db on 54320, scheduler on 9000)
```
