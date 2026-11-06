#!/bin/bash
set -euo pipefail

# This script is PID 1 until the final `exec`, and a PID-namespace init discards any
# signal it has no handler for. Without a trap, a `docker stop` during the DB wait or
# the migrations is ignored for the whole grace period and ends in SIGKILL (exit 137).
child=""
# Reap the child before exiting: once PID 1 exits the kernel SIGKILLs the rest of the
# namespace, which would cut short alembic's rollback on SIGTERM.
terminate() {
    if [ -n "$child" ]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit 143
}
trap terminate TERM INT

# Foreground commands defer a trapped signal until they finish; backgrounding and
# `wait`ing lets the trap fire at once.
run() {
    "$@" &
    child=$!
    local status=0
    wait "$child" || status=$?
    child=""
    return "$status"
}

echo "==> Waiting for database..."
wait_limit="${DB_WAIT_TIMEOUT:-120}"
deadline=$((SECONDS + wait_limit))
until run pg_isready -h "${DB_HOST:-db}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}" -q; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "==> Database not ready after ${wait_limit}s; giving up." >&2
        exit 1
    fi
    run sleep 1
done
echo "==> Database ready."

echo "==> Running Alembic migrations (portopt-db, the single migration owner)..."
cd "${MIGRATIONS_DIR:-/app/portopt-db}"
run alembic upgrade head
cd - >/dev/null
echo "==> Migrations complete."

echo "==> Starting application..."
exec "$@"
