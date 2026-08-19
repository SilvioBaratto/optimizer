#!/usr/bin/env bash
#
# Daily ingestion pipeline, run on demand.
#
# Thin wrapper over the CLI inside the scheduler container. Step order and the
# dependency gating (news → summarize → calibrate) live in
# app/services/jobs/scheduler.py, not here — this script only picks the
# execution context.
#
# The scheduler container already runs this pipeline automatically at
# SCHEDULER_DAILY_PIPELINE_CRON (default 07:00 UTC). Use this script to force a
# run now. If the scheduler is mid-run, the CLI is refused per-step with
# JobAlreadyRunningError rather than double-fetching.
#
#   ./scheduler/fetch.sh                # inside docker compose (default)
#   RUNNER=local ./scheduler/fetch.sh   # against the host venv + local DB
#
set -euo pipefail

RUNNER="${RUNNER:-docker}"
SERVICE="${SERVICE:-scheduler}"

case "$RUNNER" in
    docker)
        exec docker compose exec -T "$SERVICE" python -m app.cli daily
        ;;
    local)
        cd "$(dirname "$0")/../ingestion"
        exec python -m app.cli daily
        ;;
    *)
        echo "unknown RUNNER='$RUNNER' (expected 'docker' or 'local')" >&2
        exit 2
        ;;
esac
