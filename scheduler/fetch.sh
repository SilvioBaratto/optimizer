#!/bin/sh
# Daily data fetch script — called by supercronic at 7:00 AM UTC.
# Uses curl + jq to trigger API endpoints and poll for completion.
# No Python required.

set -e

API_URL="${OPTIMIZER_API_URL:-http://api:8000}"
API_PREFIX="${API_URL}/api/v1"
MAX_RETRIES=30
RETRY_INTERVAL=10
POLL_INTERVAL=2

echo "=== Data fetch started at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# ── Wait for API health ──────────────────────────────────────────
echo "Waiting for API at ${API_URL}/health ..."
for i in $(seq 1 "$MAX_RETRIES"); do
    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        echo "API is healthy (attempt ${i}/${MAX_RETRIES})"
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "ERROR: API not healthy after ${MAX_RETRIES} attempts. Aborting."
        exit 1
    fi
    echo "API not ready (attempt ${i}/${MAX_RETRIES}), retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

# ── fire_and_poll ────────────────────────────────────────────────
# Usage: fire_and_poll <label> <endpoint> <json_body>
# POSTs to <endpoint>, extracts job_id, polls <endpoint>/<job_id>
# until status is "completed" or "failed".
fire_and_poll() {
    label="$1"
    endpoint="$2"
    body="$3"

    echo "--- ${label}: starting ---"

    response=$(curl -sf -X POST \
        -H 'Content-Type: application/json' \
        -d "${body}" \
        "${API_PREFIX}${endpoint}")

    job_id=$(echo "$response" | jq -r '.job_id // empty')
    if [ -z "$job_id" ]; then
        echo "${label}: no job_id returned — assuming synchronous completion."
        echo "${label}: response=$(echo "$response" | jq -c '.' 2>/dev/null || echo "$response")"
        return 0
    fi

    echo "${label}: job_id=${job_id}, polling..."
    while true; do
        sleep "$POLL_INTERVAL"
        poll_resp=$(curl -sf "${API_PREFIX}${endpoint}/${job_id}" 2>/dev/null || echo '{}')
        status=$(echo "$poll_resp" | jq -r '.status // "unknown"')
        current=$(echo "$poll_resp" | jq -r '.current // 0')
        total=$(echo "$poll_resp" | jq -r '.total // 0')

        case "$status" in
            completed)
                echo "${label}: completed (${current}/${total})."
                return 0
                ;;
            failed)
                error=$(echo "$poll_resp" | jq -r '.error // "unknown"')
                echo "${label}: FAILED — ${error}"
                return 1
                ;;
            *)
                echo "${label}: ${status} (${current}/${total})..."
                ;;
        esac
    done
}

# ── Trigger fetches (run all, even if one fails) ─────────────────

fire_and_poll "yfinance" "/yfinance-data/fetch" \
    '{"max_workers":4,"period":"5y","mode":"incremental"}'
YFINANCE_EXIT=$?

fire_and_poll "macro" "/macro-data/fetch" \
    '{"countries":null,"include_bonds":true}'
MACRO_EXIT=$?

fire_and_poll "news" "/macro-data/news/fetch" \
    '{"max_articles":30,"fetch_full_content":true}'
NEWS_EXIT=$?

# ── Summary ──────────────────────────────────────────────────────
echo "=== Fetch summary ==="
echo "  yfinance: $([ $YFINANCE_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $YFINANCE_EXIT)")"
echo "  macro:    $([ $MACRO_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $MACRO_EXIT)")"
echo "  news:     $([ $NEWS_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $NEWS_EXIT)")"
echo "=== Data fetch finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Exit with failure if any fetch failed
[ $YFINANCE_EXIT -eq 0 ] && [ $MACRO_EXIT -eq 0 ] && [ $NEWS_EXIT -eq 0 ]
