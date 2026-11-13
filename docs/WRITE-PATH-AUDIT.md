# Write-Path Audit — `ingestion/app`

Gate document for the idempotence-conversion cycle. Every persisted write in
`ingestion/app` is listed below with its target table, natural key, current
mechanism, and a verdict. A conversion issue that adds `ON CONFLICT` to a row
this document marks *needs a unique constraint first* fails at **runtime**,
not at review — that is the failure this audit exists to catch before it
ships.

This issue makes **no behavioural change** to `app/`. It is a document only.

## Method

```
grep -rn "_upsert(" app --include="*.py"
grep -rn "bulk_" app --include="*.py"
grep -rn "session.add(" app --include="*.py"
grep -rn "session.add_all(" app --include="*.py"
```

Run from `ingestion/`, against tracked `*.py` files only, with `__pycache__`
**excluded** — the same run without exclusion returns 73 combined hits
instead of 57; the extra 16 are stale bytecode-cache duplicates of source
lines already counted and add no new sites.

Source-only counts: `_upsert(` = 26, `bulk_` = 27, `session.add(` = 4,
`session.add_all(` = 0. Combined = 57. These reconcile against the guard at
`ingestion/tests/unit/hygiene/test_write_path_audit_reconciles.py`, which
re-runs the same counts on every CI run so this table cannot silently drift
from the tree.

**Updated by issue #17**: row 24 (`upsert_regime_classification`) was
converted from check-then-`session.add` to a single `_upsert` on
`uq_macro_calibration_country`, moving one hit from `session.add(` to
`_upsert(`.

**Updated by issue #18**: row 23 (`macro_news_themes` child rows inside
`upsert_macro_news`) was converted from `theme_entries.clear()` +
per-theme `session.add()` to an `_upsert` on `uq_macro_news_theme` plus a
targeted delete of themes absent from the payload, moving a second hit from
`session.add(` to `_upsert(`.

**Updated by issue #20**: row 29 (`save_exchange`) was converted from
check-then-`session.add` to a single `_upsert` on `uq_exchanges_name`, moving
a third hit from `session.add(` to `_upsert(`. This conversion also required
naming the ORM's previously-unnamed `exchanges.name` unique constraint (see
the former "Constraint landmines" entry below, now resolved) and a migration
normalising any pre-existing differently-named constraint in place.

**Updated by issue #21**: row 30 (`reference_index_seeder.py`) was converted
from `get_instrument_by_yfinance_ticker()`-then-`session.add()` to a single
`_upsert` on `uq_instrument_ticker_exchange` via a new
`YFinanceRepository.upsert_instrument_by_natural_key()`, moving a fourth hit
from `session.add(` to `_upsert(`. The old lookup targeted
`instruments.yfinance_ticker` — a plain, non-unique index (see "Constraint
landmines" below) — so a row present under the true natural key but not
found by that lookup (e.g. a NULL `yfinance_ticker`) could reach `session.add()`
and collide with the real `uq_instrument_ticker_exchange` constraint on flush.

Across all four conversions the combined total (57) and the actual-write-site
total (30) are unchanged — only the per-mechanism split moved.

## Why 57 combined hits is not 57 writes

**None of the 27 `bulk_` hits is a persisted write.** They are:

- `run_bulk_yfinance_fetch` / `run_bulk_macro_fetch` / `run_bulk_fred_fetch` /
  `run_bulk_calibrate` — orchestrator function **definitions**
  (`services/market_data/yfinance_data_service.py:747`,
  `services/macro/macro_regime_service.py:383,460`,
  `services/macro/macro_calibration.py:301`) plus their **imports** and
  `__all__` re-exports in `services/market_data/__init__.py`,
  `services/macro/__init__.py`, and `services/jobs/scheduler.py`, and their
  call sites inside `scheduler.py`.
- `bulk_download` — a yfinance **network** call
  (`services/market_data/yfinance/_facade.py:245`, called at `:320`; its
  protocol declaration is at `yfinance/protocols/interfaces.py:89`; the
  rate-limiter key `"bulk_download"` at `:281` is a string label, not a
  symbol). It fetches from Yahoo Finance — it does not write to the database.
- One `bulk_` hit (`reference_index_seeder.py:64`) is a **code comment**
  referencing the pattern, not a symbol at all.

Of the `_upsert(` hits, 25 are calls to `RepositoryBase._upsert()` and one
(`repositories/_shared/base.py:33`) is the **definition** of that helper —
not itself a write, but retained as a table row below so the row count
reconciles with the stated 26.

Actual persisted-write sites: **25 `_upsert(` calls + 4 `session.add(` calls
+ the 1 `_upsert` definition row (non-write, listed for completeness) = 30
rows**, against 57 combined source-only hits. The gap is entirely the 27
non-write `bulk_` hits.

## `_upsert()` and SQLite: value-idempotence only, not constraint-name safety

`_upsert()` (`repositories/_shared/base.py:33-72`) builds a
`sqlalchemy.dialects.postgresql.insert()` statement and calls
`.on_conflict_do_update(constraint=<name>, ...)`. `tests/unit/test_upsert_updated_at.py`
passes against the project's SQLite test harness, so replay tests for value
idempotence need no PostgreSQL — **but that coverage is narrower than it
looks**. Compiled against SQLite, the named conflict target is silently
dropped:

```
INSERT INTO macro_calibrations (country, id) VALUES (?, ?) ON CONFLICT  DO UPDATE SET ...
```

A deliberately bogus `constraint_name="uq_TOTALLY_BOGUS"` upserts cleanly on
SQLite and produces one row with the second call's values — the SQLite
harness gives **zero** coverage of whether the named constraint actually
exists. A conversion that misnames a constraint is green in the existing
test suite and fails only against real PostgreSQL. The guard at
`test_write_path_audit_reconciles.py` closes this gap mechanically: it
extracts every `constraint_name=` / `constraint=` string literal under
`ingestion/app` and asserts each resolves to a `UniqueConstraint` (or unique
`Index`) actually declared in `app/models/`. All 25 current call-site
literals resolve today.

## Write-path table

Verdict legend: **idempotent** — replay produces one row with the latest
values, safe as-is; **not idempotent** — replay depends on session state or
a preceding read, must convert; **needs a unique constraint first** — no
matching unique constraint exists yet, so `ON CONFLICT` cannot target it
until the constraint (and the matching migration) is added.

| # | Site (file:line) | Table | Natural key | Mechanism | Verdict |
|---|---|---|---|---|---|
| 1 | `repositories/_shared/base.py:33` | — | — | `_upsert()` definition (not a write; listed so the row count reconciles with the stated 26 `_upsert(` hits) | n/a |
| 2 | `repositories/market_data/yfinance_repository.py:223` | `ticker_profiles` | `instrument_id` | `_upsert` on `uq_ticker_profile_instrument` | idempotent |
| 3 | `repositories/market_data/yfinance_repository.py:285` | `price_history` | `(instrument_id, date)` | `_upsert` on `uq_price_history_instrument_date` | idempotent |
| 4 | `repositories/market_data/yfinance_repository.py:351` | `financial_statements` | `(instrument_id, statement_type, period_type, period_date, line_item)` | `_upsert` on `uq_financial_statement_row` | idempotent |
| 5 | `repositories/market_data/yfinance_repository.py:396` | `dividends` | `(instrument_id, date)` | `_upsert` on `uq_dividend_instrument_date` | idempotent |
| 6 | `repositories/market_data/yfinance_repository.py:430` | `stock_splits` | `(instrument_id, date)` | `_upsert` on `uq_stock_split_instrument_date` | idempotent |
| 7 | `repositories/market_data/yfinance_repository.py:467` | `analyst_recommendations` | `(instrument_id, period)` | `_upsert` on `uq_analyst_rec_instrument_period` | idempotent |
| 8 | `repositories/market_data/yfinance_repository.py:498` | `analyst_price_targets` | `instrument_id` | `_upsert` on `uq_analyst_pt_instrument` | idempotent |
| 9 | `repositories/market_data/yfinance_repository.py:534` | `institutional_holders` | `(instrument_id, holder_name)` | `_upsert` on `uq_inst_holder_instrument_name` | idempotent |
| 10 | `repositories/market_data/yfinance_repository.py:574` | `mutual_fund_holders` | `(instrument_id, holder_name)` | `_upsert` on `uq_mutual_fund_holder_instrument_name` | idempotent |
| 11 | `repositories/market_data/yfinance_repository.py:640` | `insider_transactions` | `(instrument_id, insider_name, start_date, transaction_type)` | `_upsert` on `uq_insider_tx_row`, batch pre-deduped at `:623-643` | idempotent (see known defect below) |
| 12 | `repositories/market_data/yfinance_repository.py:739` | `ticker_news` | `(instrument_id, news_uuid)` | `_upsert` on `uq_ticker_news_instrument_uuid` | idempotent |
| 13 | `repositories/macro/macro_regime_repository.py:74` | `economic_indicators` | `country` | `_upsert` on `uq_economic_indicator_country` | idempotent |
| 14 | `repositories/macro/macro_regime_repository.py:140` | `economic_indicator_observations` | `(country, date)` | `_upsert` on `uq_econ_obs_country_date` | idempotent |
| 15 | `repositories/macro/macro_regime_repository.py:205` | `trading_economics_indicators` | `(country, indicator_key)` | `_upsert` on `uq_te_indicator_country_key` | idempotent |
| 16 | `repositories/macro/macro_regime_repository.py:262` | `bond_yields` | `(country, maturity)` | `_upsert` on `uq_bond_yield_country_maturity` | idempotent |
| 17 | `repositories/macro/macro_regime_repository.py:314` | `trading_economics_observations` | `(country, indicator_key, date)` | `_upsert` on `uq_te_obs_country_key_date` | idempotent |
| 18 | `repositories/macro/macro_regime_repository.py:385` | `bond_yield_observations` | `(country, maturity, date)` | `_upsert` on `uq_bond_obs_country_mat_date` | idempotent |
| 19 | `repositories/macro/macro_regime_repository.py:447` | `fred_observations` | `(series_id, date)` | `_upsert` on `uq_fred_observation_series_date` | idempotent |
| 20 | `repositories/macro/macro_regime_repository.py:518` | `macro_news` | `news_id` | `_upsert` on `uq_macro_news_id` (parent row only — child theme rows are row 23 below) | idempotent |
| 21 | `repositories/macro/macro_regime_repository.py:600` | `macro_news_summaries` | `(country, summary_date)` | `_upsert` on `uq_macro_news_summary_country_date` | idempotent |
| 22 | `repositories/macro/macro_regime_repository.py:691` | `macro_calibrations` | `country` | `_upsert` on `uq_macro_calibration_country` (`upsert_macro_calibration`, the BAML-write path) | idempotent |
| 23 | `repositories/macro/macro_regime_repository.py:581` | `macro_news_themes` | `(news_id, theme)` | `_upsert` on `uq_macro_news_theme`, with a targeted delete of themes absent from the incoming payload (`_reconcile_news_themes`) — converted from `theme_entries.clear()` + per-theme `session.add()` by issue #18 | idempotent |
| 24 | `repositories/macro/macro_regime_repository.py:702` | `macro_calibrations` | `country` | `_upsert` on `uq_macro_calibration_country`, `update_columns=["regime_classification", "updated_at"]` (`upsert_regime_classification`, the rule-based-classifier write path, distinct from row 22's BAML path) — converted from check-then-`session.add` by issue #17 | idempotent |
| 25 | `repositories/_shared/base.py:138` | caller-supplied model | none enforced | `BaseRepository.create()` — plain `session.add()`, no conflict handling | n/a — no caller under `app/` currently invokes this generic CRUD method in a production write path; calling it twice with the same logical entity inserts two rows |
| 26 | `repositories/_shared/base.py:145` | caller-supplied model | none enforced | `BaseRepository.create_from_dict()` — plain `session.add()`, no conflict handling | n/a — same as row 25, no current caller under `app/` |
| 27 | `repositories/jobs/background_job_repository.py:88` | `background_jobs` | none (UUID PK per job run) | `session.add()` in `create()` | idempotent by construction — each call represents a new job execution, not an update to an existing natural-keyed entity |
| 28 | `repositories/jobs/background_job_repository.py:182` | `background_job_errors` | `(job_id, error_index)` | delete all rows for `job_id` then `session.add()` per error — clear-and-rebuild inside `_replace_error_entries` | idempotent — full delete-then-insert for the job_id is deterministic (unlike row 23, there is no partial update, and error rows carry no cross-referenced natural key outside their own job) |
| 29 | `repositories/universe/universe_repository.py:24` | `exchanges` | `name` | `_upsert` on `uq_exchanges_name`, `update_columns=["t212_id", "updated_at"]` (`save_exchange`) — converted from check-then-`session.add` by issue #20, after naming the ORM's previously-unnamed unique constraint | idempotent |
| 30 | `services/market_data/reference_index_seeder.py:104` (upsert call) / `repositories/market_data/yfinance_repository.py` (`upsert_instrument_by_natural_key`) | `instruments` | `(ticker, exchange_id)` | `_upsert` on `uq_instrument_ticker_exchange`, `update_columns=["short_name", "yfinance_ticker", "instrument_type", "currency_code", "updated_at"]` — converted from check-then-`session.add` by issue #21 | idempotent |

## Constraint landmines — tables whose natural key lacks a usable unique constraint

- **`exchanges.name`** (row 29) — **resolved by issue #20**.
  `models/universe/universe.py` previously declared `unique=True` at the
  column level, which SQLAlchemy renders as an **unnamed** `UNIQUE (name)`
  constraint, while migration
  `alembic/versions/c3d4e5f6a7b8_align_exchanges_with_orm.py:80`
  conditionally created it as **`uq_exchanges_name`** — the ORM and the
  migration disagreed on the name. `Exchange` now declares
  `UniqueConstraint("name", name="uq_exchanges_name")` in `__table_args__`,
  so `Base.metadata.create_all` (the SQLite test harness's schema source)
  and a PostgreSQL database produce the same constraint name, and migration
  `0287fcf584d0_name_exchanges_name_unique_constraint.py` normalises any
  database whose constraint kept the pre-fix auto-generated name.
- **`instruments.yfinance_ticker`** — `ix_instrument_yfinance_ticker`
  (`models/universe/universe.py:50`) is a plain, **non-unique** `Index`. It
  cannot be an `ON CONFLICT` target. No write path in this audit currently
  targets it as a conflict key (row 30 conflicts on `uq_instrument_ticker_exchange`
  instead), so this is a standing landmine for any *future* write keyed on
  `yfinance_ticker`, not an open defect in an existing conversion.

No other table in the write-path table above is missing a matching unique
constraint — rows 2-29 all target constraints that already exist and are
correctly named in both the ORM and the corresponding migration.

## Known defects — already mitigated, recorded for completeness

- **`insider_transactions` batch collision** (row 11). yfinance can return
  several real transactions sharing `(instrument_id, insider_name,
  start_date, transaction_type)` (e.g. same grant, same day, different
  share counts), which a single multi-row `INSERT ... ON CONFLICT DO UPDATE`
  cannot apply twice to the same conflict key. Mitigated at
  `repositories/market_data/yfinance_repository.py:623-643`: rows are
  deduplicated by that key (last occurrence wins, matching the DB upsert's
  own last-write-wins semantics) before the `_upsert` call at `:640`.
- **`price_history` young-listing skip** (row 3). An insufficient-history
  guard exists to stop a truncated/glitchy re-fetch from clobbering an
  established price series, but for an instrument with no stored prices
  yet a short history is simply a young listing's complete, legitimate
  history. Mitigated at
  `services/market_data/yfinance_data_service.py:317-353`: the guard now
  only discards a short history when prior price data already exists for
  the instrument; a young listing with no existing rows is stored as-is.
- **Per-country news commit** (row 20's caller). `generate_country_summaries`
  in `services/macro/macro_news_summary.py` commits after each country
  (`:418`) so a per-country write does not hold a pooled connection across
  the next country's LLM call. The scheduler-facing wrapper
  `_summarize_country_safe` isolates failures per country: on an exception
  it rolls back (`:429`) and returns `"error"` for that country only,
  logging non-fatally so the job continues with the remaining countries
  rather than aborting the whole run.

## Under-covered research angle

Idempotent upserts (this audit's subject), watermarking, schema drift, and
incremental-vs-backfill semantics for yfinance and FRED are not fully
resolved by this document — this audit is scoped to *write-path* idempotence
only. The remaining angles are restated here as a scoped follow-up rather
than silently dropped:

- **Watermarking** (tracking the last successfully-ingested point per
  series/instrument to drive incremental fetch) is partially covered by the
  existing `get_staleness_info` / `price_max_date` mechanism referenced
  above, but is not audited end-to-end here.
- **Schema drift** (yfinance response shape changes silently breaking a
  write, as already happened once with the `insider_transactions`
  `"Transaction"` column) has no systematic detection in this audit's scope.
- **Incremental vs backfill** semantics for yfinance and FRED are exercised
  by the `mode="incremental"` branch referenced in row 3's mitigation and by
  `run_bulk_*` orchestrators (see "Why 57 combined hits is not 57 writes"
  above), but a full incremental-vs-backfill comparison is out of scope for
  a write-idempotence audit.

**Owner:** ingestion daemon maintainer, per the ownership and conventions
recorded in the repository root `CLAUDE.md` (the `ingestion/` subtree has
no maintainer file of its own; the root file is authoritative for this
whole project, `ingestion/` included) — to be picked up as a follow-up
issue scoped to watermarking/schema-drift/backfill research, separate
from this cycle's idempotence-conversion issues.
