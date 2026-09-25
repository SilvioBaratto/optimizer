#!/bin/bash
set -euo pipefail

echo "==> [fund] Waiting for database..."
until pg_isready -h "${DB_HOST:-db}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}" -q; do
    sleep 1
done
echo "==> [fund] Database ready."

# Public schema is owned by the single portopt-db Alembic tree (same as the
# ingestion image). fund reads/writes portfolio_mandates, agent_runs, fund_jobs,
# the MiFID tables — all in `public`. Idempotent (`upgrade head` is a no-op when
# already at head), so bringing up `db + fund` alone (the `optimizer` launcher's
# path) is self-sufficient without the scheduler container.
#
# The dedicated `langgraph` schema is created out-of-band by the worker itself
# (fund.audit.persistence.setup_langgraph → CREATE SCHEMA IF NOT EXISTS), so it is
# NOT handled here.
echo "==> [fund] Running Alembic migrations (portopt-db, the single migration owner)..."
(cd /app/packages/portopt-db && alembic upgrade head)
echo "==> [fund] Migrations complete."

echo "==> [fund] Starting: $*"
exec "$@"
