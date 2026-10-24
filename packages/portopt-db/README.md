# portopt-db

Shared data-access package for the portopt Postgres database: SQLAlchemy `Base` +
models, repositories, the `DatabaseManager` engine/session layer, and the **single**
Alembic migration tree.

Internal workspace package (not published). Consumed by `ingestion` (and future siblings).
`portopt-core` (the optimizer library) must **never** depend on this — it stays DB-free.

Migrations run **only** from here. `alembic.ini` lives in this directory, so run
alembic from it, and export `DATABASE_URL` first: `alembic/env.py` reads only that
process variable (never a `.env` file) and refuses to run without it.

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:54320/optimizer_db
cd packages/portopt-db && uv run --package portopt-db alembic upgrade head
```
