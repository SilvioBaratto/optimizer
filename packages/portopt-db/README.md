# portopt-db

Shared data-access package for the portopt Postgres database: SQLAlchemy `Base` +
models, repositories, the `DatabaseManager` engine/session layer, and the **single**
Alembic migration tree.

Internal workspace package (not published), consumed by `ingestion` and by `fund`.
`portopt-core` (the optimizer library) must **never** depend on this — it stays DB-free
(see [ADR-0002](../../docs/adr/0002-four-package-uv-workspace.md)).

## Consumers and versioning

The package is versioned with the rest of the workspace (`0.1.0`) and resolved through
`[tool.uv.sources]`, so a consumer always builds against the checkout it sits in. What a
consumer may rely on is the surface it imports: the models, the repositories, `RepositoryBase`
and `DatabaseManager`. A change to that surface, or to a table a consumer reads, goes in
`CHANGELOG.md`, and the consumer's tests change in the same pull request.

## Migrations

Migrations run **only** from here. `alembic.ini` lives in this directory, so run
alembic from it, and export `DATABASE_URL` first: `alembic/env.py` reads only that
process variable (never a `.env` file) and refuses to run without it.

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:54320/optimizer_db
cd packages/portopt-db && uv run --package portopt-db alembic upgrade head
```

### Commands

Every command needs the database URL and the package directory. Point `DATABASE_URL` at
a scratch database when you are rehearsing.

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:54320/optimizer_db
cd packages/portopt-db

# Generate a revision: the id is hex, never typed by hand, and the file name is dated
uv run alembic revision -m "short description" --head <current head>

# Validate: models match the migrations, then a window of revisions round-trips
uv run alembic upgrade head
uv run alembic check
uv run alembic downgrade 70393a606218
uv run alembic upgrade head

# Roll back one revision, or to a named one
uv run alembic downgrade -1
```

### Writing a revision

- Generate it with the revision command above. New revision files get a dated prefix from
  `file_template` in `alembic.ini`, so the file names sort in execution order. Existing
  files keep their names.
- A revision is atomic and takes the migration advisory lock (`alembic/env.py`). Keep it
  to one concern.
- It needs a working `downgrade()`. A revision that cannot be undone, because it deletes
  data, instead raises `NotImplementedError` **and** declares a module-level
  `IRREVERSIBLE_REASON` that says what was destroyed and where the restore source is.
  `tests/test_irreversible_migrations_declare_reason.py` fails without the constant.
- Take a database dump before applying a destructive revision. Restoring from it is the
  only way back for the revisions listed in that test.
- Two branches that each add a head are joined by a merge revision (for example
  `70393a606218`). `tests/test_alembic_single_head.py` and
  `tests/test_alembic_chain_integrity.py` keep exactly one head and an unbroken chain.

### Validation before a deploy

The `postgres-integration` CI job builds an empty database and runs the validation
sequence from the commands block (upgrade, check, the downgrade window and back) and then
the integration suite below. A model that changed without a migration fails the check, and
one of the five most recent revisions that cannot be reversed fails the round trip. Run the
same sequence against a scratch database before you merge a revision.

### Rolling back

Stop the daemons first (one daemon per database), then use the downgrade command from the
block above. A revision whose `downgrade()` raises is a one-way door: restore the dump
instead. Rehearse a rollback on a scratch database before doing it on a live one.

## Tests

The default suite runs on in-memory SQLite and needs nothing:

```bash
cd packages/portopt-db && uv run pytest
```

The tests marked `integration` need a real PostgreSQL, because SQLite drops the
`ON CONFLICT ON CONSTRAINT` target and never installs the `human_audit_log` triggers.
They skip unless `PORTOPT_TEST_DATABASE_URL` is set, and they refuse a database whose name
does not end in `_test`:

```bash
docker run -d --rm --name portopt_test_pg -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=portopt_test -p 127.0.0.1:54399:5432 postgres:16
cd packages/portopt-db && \
  PORTOPT_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:54399/portopt_test \
  uv run pytest tests/integration -m integration
```

## Schema

The schema has 68 tables. Most hang off `instruments`; the rest stand alone. The diagram
shows the foreign keys that exist, not every column.

```mermaid
erDiagram
    exchanges ||--o{ instruments : "exchange_id"
    instruments ||--o{ price_history : "instrument_id"
    instruments ||--o{ ticker_profiles : "instrument_id"
    instruments ||--o{ financial_statements : "instrument_id"
    instruments ||--o{ dividends : "instrument_id"
    instruments ||--o{ ticker_news : "instrument_id"
    instruments ||--o{ etf_metadata : "instrument_id"
    background_jobs ||--o{ background_job_errors : "job_id"
    macro_news ||--o{ macro_news_themes : "news_id"
    agent_runs ||--o{ agent_decisions : "run_id"
    paper_orders ||--o{ positions : "paper_order_id"
```

The other 28 per-instrument tables (holders, estimates, ETF holdings, options) reference
`instruments` the same way as `price_history`: all 34 have `ON DELETE CASCADE`, so deleting
an instrument deletes its rows in every one of them. These tables carry no foreign key: `human_audit_log` (append-only, protected by
triggers), `pinned_instruments` (keyed by ticker text), and the macro tables keyed by
country.
