# portopt-db

Shared data-access package for the portopt Postgres database: SQLAlchemy `Base` +
models, repositories, the `DatabaseManager` engine/session layer, and the **single**
Alembic migration tree.

Internal workspace package (not published). Consumed by `ingestion` (and future siblings).
`portopt-core` (the optimizer library) must **never** depend on this — it stays DB-free.

Migrations run **only** from here:

```bash
uv run --package portopt-db alembic upgrade head
```
