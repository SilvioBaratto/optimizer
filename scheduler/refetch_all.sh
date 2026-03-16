#!/bin/sh
# Full data re-fetch: universe + yfinance + macro + FRED + news.
# Run from host: OPTIMIZER_API_URL=http://localhost:8005 ./scheduler/refetch_all.sh
# Run from Docker: docker exec optimizer_scheduler /app/refetch_all.sh
#
# NOTE: set -e is intentionally absent. We capture per-step exit codes and
# implement explicit dependency enforcement. set -e would abort the script on
# the first pipeline failure before skip-logic or the summary block can run.

API_URL="${OPTIMIZER_API_URL:-http://api:8000}"
API_PREFIX="${API_URL}/api/v1"
MAX_RETRIES=30
RETRY_INTERVAL=10
POLL_INTERVAL=5
MAX_POLL_SECONDS="${MAX_POLL_SECONDS:-21600}"

echo "=== Full re-fetch started at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

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
fire_and_poll() {
    label="$1"
    endpoint="$2"
    body="$3"

    echo ""
    echo "--- ${label}: starting ---"

    response=$(curl -s -w '\n%{http_code}' -X POST \
        -H 'Content-Type: application/json' \
        -d "${body}" \
        "${API_PREFIX}${endpoint}")

    http_code=$(echo "$response" | tail -1)
    body_resp=$(echo "$response" | sed '$d')

    # 409 = job already running — extract existing job_id and poll it
    if [ "$http_code" = "409" ]; then
        job_id=$(echo "$body_resp" | jq -r '.detail // ""' | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)
        if [ -n "$job_id" ]; then
            echo "${label}: already running (${job_id}), polling existing job..."
        else
            echo "${label}: 409 conflict but no job_id found — skipping."
            return 1
        fi
    elif [ "$http_code" -ge 200 ] 2>/dev/null && [ "$http_code" -lt 300 ] 2>/dev/null; then
        job_id=$(echo "$body_resp" | jq -r '.job_id // .build_id // empty')
        if [ -z "$job_id" ]; then
            echo "${label}: no job_id returned — assuming synchronous completion."
            echo "${label}: response=$(echo "$body_resp" | jq -c '.' 2>/dev/null || echo "$body_resp")"
            return 0
        fi
    else
        echo "${label}: HTTP ${http_code} — failed to start."
        echo "${label}: response=$(echo "$body_resp" | jq -c '.' 2>/dev/null || echo "$body_resp")"
        return 1
    fi

    # Poll with timeout (MAX_POLL_SECONDS, default 21600 = 6 hours).
    echo "${label}: job_id=${job_id}, polling..."
    poll_start=$(date +%s)
    while true; do
        sleep "$POLL_INTERVAL"
        elapsed=$(( $(date +%s) - poll_start ))
        if [ "$elapsed" -ge "$MAX_POLL_SECONDS" ]; then
            echo "${label}: TIMEOUT after ${elapsed}s — aborting poll."
            return 1
        fi
        poll_resp=$(curl -sf "${API_PREFIX}${endpoint}/${job_id}" 2>/dev/null || echo '{}')
        status=$(echo "$poll_resp" | jq -r '.status // "unknown"')
        current=$(echo "$poll_resp" | jq -r '.current // .progress.completed // 0')
        total=$(echo "$poll_resp" | jq -r '.total // .progress.total // 0')

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

# ── Trigger fetches with dependency enforcement ───────────────────
# universe, yfinance, macro, and fred are all independent — always run.

fire_and_poll "universe" "/universe/build" '{}'
UNIVERSE_EXIT=$?

fire_and_poll "yfinance" "/yfinance-data/fetch" \
    '{"max_workers":4,"period":"5y","mode":"incremental"}'
YFINANCE_EXIT=$?

fire_and_poll "macro" "/macro-data/fetch" \
    '{"countries":null,"include_bonds":true}'
MACRO_EXIT=$?

fire_and_poll "fred" "/macro-data/fred/fetch" \
    '{"incremental":false}'
FRED_EXIT=$?

# news depends on yfinance (needs price/ticker data to exist)
if [ $YFINANCE_EXIT -eq 0 ]; then
    fire_and_poll "news" "/macro-data/news/fetch" \
        '{"max_articles":30,"fetch_full_content":true}'
    NEWS_EXIT=$?
else
    echo "--- news: SKIPPED (yfinance failed) ---"
    NEWS_EXIT=1
fi

# summarize depends on news
if [ $NEWS_EXIT -eq 0 ]; then
    fire_and_poll "summarize" "/macro-data/news/summarize" \
        '{"force_refresh":true}'
    SUMMARIZE_EXIT=$?
else
    echo "--- summarize: SKIPPED (news failed/skipped) ---"
    SUMMARIZE_EXIT=1
fi

# calibrate depends on summarize
if [ $SUMMARIZE_EXIT -eq 0 ]; then
    fire_and_poll "calibrate" "/views/macro-calibration/batch" \
        '{"force_refresh":true}'
    CALIBRATE_EXIT=$?
else
    echo "--- calibrate: SKIPPED (summarize failed/skipped) ---"
    CALIBRATE_EXIT=1
fi

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "=== Re-fetch summary ==="
echo "  universe:   $([ $UNIVERSE_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $UNIVERSE_EXIT)")"
echo "  yfinance:   $([ $YFINANCE_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $YFINANCE_EXIT)")"
echo "  macro:      $([ $MACRO_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $MACRO_EXIT)")"
echo "  fred:       $([ $FRED_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $FRED_EXIT)")"
echo "  news:       $([ $NEWS_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $NEWS_EXIT)")"
echo "  summarize:  $([ $SUMMARIZE_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $SUMMARIZE_EXIT)")"
echo "  calibrate:  $([ $CALIBRATE_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $CALIBRATE_EXIT)")"
echo "=== Re-fetch finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Exit with failure if any fetch failed
[ $UNIVERSE_EXIT -eq 0 ] && [ $YFINANCE_EXIT -eq 0 ] && \
[ $MACRO_EXIT -eq 0 ] && [ $FRED_EXIT -eq 0 ] && [ $NEWS_EXIT -eq 0 ] && \
[ $SUMMARIZE_EXIT -eq 0 ] && [ $CALIBRATE_EXIT -eq 0 ]
