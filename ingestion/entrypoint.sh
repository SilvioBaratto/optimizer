#!/bin/bash
set -euo pipefail

echo "==> Waiting for database..."
until pg_isready -h "${DB_HOST:-db}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}" -q; do
    sleep 1
done
echo "==> Database ready."

echo "==> Running Alembic migrations (portopt-db, the single migration owner)..."
(cd /app/portopt-db && alembic upgrade head)
echo "==> Migrations complete."

echo "==> Starting application..."
exec "$@"
