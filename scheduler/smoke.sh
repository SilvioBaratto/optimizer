#!/bin/sh
# End-to-end smoke script for the docker-compose stack.
#
# Usage (local):
#   OPTIMIZER_API_URL=http://localhost:8005 ./scheduler/smoke.sh
#
# Usage (CI):
#   invoked by .github/workflows/smoke.yml against a freshly-booted stack.
#
# Verifies:
#   1. GET  /health                        returns 200
#   2. POST /api/v1/optimize                returns 200 with non-empty weights
#   3. POST /api/v1/backtest  → poll /api/v1/backtest/{job_id}
#      until status ∈ {completed, failed}; require completed with non-empty
#      equityCurve
#
# Exits non-zero on any failure. `set -e` is NOT used — we capture per-stage
# exit codes and print a summary so cron/CI can see which stage regressed.

SMOKE_API_URL="${SMOKE_API_URL:-${OPTIMIZER_API_URL:-http://localhost:8005}}"
API_PREFIX="${SMOKE_API_URL}/api/v1"

# Seeded universe + window from api/tests/fixtures/smoke_prices.sql
SMOKE_TICKERS='["SM1","SM2","SM3","SM4","SM5"]'
SMOKE_START='2023-01-02'
SMOKE_END='2023-12-29'

MAX_HEALTH_ATTEMPTS=30
HEALTH_INTERVAL=5
MAX_POLL_SECONDS="${MAX_POLL_SECONDS:-300}"
POLL_INTERVAL=2

echo "=== Smoke test started at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
echo "Target: ${SMOKE_API_URL}"

# ---------------------------------------------------------------------------
# Stage 1 — health check
# ---------------------------------------------------------------------------
echo "--- health: waiting for ${SMOKE_API_URL}/health ---"
i=0
while [ "$i" -lt "$MAX_HEALTH_ATTEMPTS" ]; do
    if curl -sf "${SMOKE_API_URL}/health" > /dev/null 2>&1; then
        echo "health: OK"
        break
    fi
    i=$((i + 1))
    sleep "$HEALTH_INTERVAL"
done
if [ "$i" -ge "$MAX_HEALTH_ATTEMPTS" ]; then
    echo "health: FAILED — API never responded"
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2 — POST /api/v1/optimize (synchronous for small universes)
# ---------------------------------------------------------------------------
echo "--- optimize: POST ${API_PREFIX}/optimize ---"
optimize_body=$(jq -n \
    --argjson tickers "$SMOKE_TICKERS" \
    --arg start "$SMOKE_START" \
    --arg end "$SMOKE_END" \
    '{tickers: $tickers, start_date: $start, end_date: $end, optimizer_type: "hrp", config: {}}')

optimize_resp=$(curl -s -w '\n%{http_code}' -X POST \
    -H 'Content-Type: application/json' \
    -d "$optimize_body" \
    "${API_PREFIX}/optimize")

optimize_code=$(echo "$optimize_resp" | tail -1)
optimize_json=$(echo "$optimize_resp" | sed '$d')

if [ "$optimize_code" != "200" ]; then
    echo "optimize: FAILED — HTTP ${optimize_code}"
    echo "optimize: response=${optimize_json}"
    exit 1
fi

weights_count=$(echo "$optimize_json" | jq '.weights | length')
if [ "$weights_count" -eq 0 ]; then
    echo "optimize: FAILED — weights are empty"
    echo "optimize: response=${optimize_json}"
    exit 1
fi
echo "optimize: OK — ${weights_count} weight(s) returned"

# ---------------------------------------------------------------------------
# Stage 3 — POST /api/v1/backtest → poll /api/v1/backtest/{job_id}
# ---------------------------------------------------------------------------
echo "--- backtest: POST ${API_PREFIX}/backtest ---"
backtest_body=$(jq -n \
    --argjson tickers "$SMOKE_TICKERS" \
    --arg start "$SMOKE_START" \
    --arg end "$SMOKE_END" \
    '{tickers: $tickers, start_date: $start, end_date: $end, pipeline_config: {optimizer_type: "hrp"}}')

backtest_resp=$(curl -s -w '\n%{http_code}' -X POST \
    -H 'Content-Type: application/json' \
    -d "$backtest_body" \
    "${API_PREFIX}/backtest")

backtest_code=$(echo "$backtest_resp" | tail -1)
backtest_json=$(echo "$backtest_resp" | sed '$d')

if [ "$backtest_code" != "202" ]; then
    echo "backtest: FAILED — HTTP ${backtest_code}"
    echo "backtest: response=${backtest_json}"
    exit 1
fi

# Response uses camelCase (jobId / runId). Fall back to snake_case for safety.
job_id=$(echo "$backtest_json" | jq -r '.jobId // .job_id // empty')
if [ -z "$job_id" ]; then
    echo "backtest: FAILED — no jobId returned"
    echo "backtest: response=${backtest_json}"
    exit 1
fi
echo "backtest: queued job_id=${job_id}, polling..."

poll_start=$(date +%s)
while true; do
    sleep "$POLL_INTERVAL"
    elapsed=$(( $(date +%s) - poll_start ))
    if [ "$elapsed" -ge "$MAX_POLL_SECONDS" ]; then
        echo "backtest: TIMEOUT after ${elapsed}s"
        exit 1
    fi

    poll_resp=$(curl -sf "${API_PREFIX}/backtest/${job_id}" 2>/dev/null || echo '{}')
    status=$(echo "$poll_resp" | jq -r '.status // "unknown"')

    case "$status" in
        completed)
            echo "backtest: completed, verifying equity curve..."
            equity_len=$(echo "$poll_resp" | jq '(.result.equityCurve // .result.equity_curve // {}) | length')
            if [ "${equity_len:-0}" -eq 0 ]; then
                echo "backtest: FAILED — equityCurve is empty"
                echo "backtest: response=${poll_resp}"
                exit 1
            fi
            echo "backtest: OK — equityCurve has ${equity_len} point(s)"
            break
            ;;
        failed)
            error=$(echo "$poll_resp" | jq -r '.error // "unknown"')
            echo "backtest: FAILED — ${error}"
            exit 1
            ;;
        *)
            echo "backtest: ${status} (${elapsed}s elapsed)..."
            ;;
    esac
done

echo "=== Smoke test PASSED at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
