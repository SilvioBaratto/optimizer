#!/usr/bin/env bash
#
# Full data rebuild: universe → yfinance (5y) + macro → FRED.
#
# Thin wrapper over the CLI inside the scheduler container. Step order lives in
# app/services/jobs/scheduler.py, not here.
#
# The universe build runs first on purpose: every other step iterates the
# `instruments` table, so a stale universe silently caps what yfinance fetches.
# Expect this to take hours on a cold database.
#
#   ./scheduler/refetch_all.sh                # inside docker compose (default)
#   RUNNER=local ./scheduler/refetch_all.sh   # against the host venv + local DB
#
set -euo pipefail

RUNNER="${RUNNER:-docker}"
SERVICE="${SERVICE:-scheduler}"

case "$RUNNER" in
    docker)
        exec docker compose exec -T "$SERVICE" python -m app.cli refetch-all
        ;;
    local)
        cd "$(dirname "$0")/../ingestion"
        exec python -m app.cli refetch-all
        ;;
    *)
        echo "unknown RUNNER='$RUNNER' (expected 'docker' or 'local')" >&2
        exit 2
        ;;
esac
