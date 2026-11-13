# ADR-0005: Keep the LangGraph tables in a dedicated Postgres schema

- Status: Accepted
- Date: 2026-09-16

## Context

The fund bridge persists agent checkpoints and a key-value store with
`langgraph-checkpoint-postgres`. That package has no `schema=` parameter (upstream
issue #7345): its DDL uses unqualified table names that resolve through the session
`search_path`. Alembic owns the `public` schema, and its autogenerate would otherwise see
the six LangGraph tables as unmanaged. The decision is in commit `8c62f4b2` (2026-09-16).

## Decision

The six LangGraph tables live in a dedicated `langgraph` schema:

1. `CREATE SCHEMA IF NOT EXISTS langgraph` runs out-of-band, because `search_path` only
   redirects unqualified DDL and does not create the schema.
2. One psycopg pool forces `search_path=langgraph` and the settings `.setup()` needs
   (autocommit for `CREATE INDEX CONCURRENTLY`, `dict_row`, `prepare_threshold=0`).
3. `PostgresSaver.setup()` and `PostgresStore.setup()` both run on that one pool.

Every process that bootstraps the schema takes the same advisory lock first.

## Alternatives considered

- **Let LangGraph create its tables in `public`.** Rejected in the commit message: the aim
  is to keep them out of Alembic's `public`.
- **Manage the LangGraph tables with Alembic migrations.** Not discussed in the commit, so
  no reason is recorded.

## Consequences

- Alembic owns only the `portopt_db` models; the `langgraph` schema is created at runtime
  by `fund.audit.persistence`, so a fresh database needs the fund worker (or the bootstrap)
  to have run before LangGraph features work.
- Checkpoint, store and Ask-history behaviour is tied to LangGraph's own table layout and
  upgrades; a `deepagents` or `langgraph` bump can change it without an Alembic revision.
- The round trip is covered only by the integration tests marked `integration`.
