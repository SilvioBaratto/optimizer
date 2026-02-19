#!/bin/bash
# Daily data fetch script — called by supercronic at 7:00 AM UTC.
# Waits for API health, then runs yfinance + macro fetches independently.

set -o pipefail

API_URL="${OPTIMIZER_API_URL:-http://api:8000}"
MAX_RETRIES=30
RETRY_INTERVAL=10

echo "=== Data fetch started at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Wait for API to be healthy
echo "Waiting for API at ${API_URL}/health ..."
for i in $(seq 1 "$MAX_RETRIES"); do
    if python -c "import urllib.request; urllib.request.urlopen('${API_URL}/health').read()" 2>/dev/null; then
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

# Run yfinance fetch
echo "--- yfinance fetch ---"
python -m cli --base-url "$API_URL" yfinance fetch
YFINANCE_EXIT=$?

# Run macro fetch (always, even if yfinance failed)
echo "--- macro fetch ---"
python -m cli --base-url "$API_URL" macro fetch
MACRO_EXIT=$?

# Summary
echo "=== Fetch summary ==="
echo "  yfinance: $([ $YFINANCE_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $YFINANCE_EXIT)")"
echo "  macro:    $([ $MACRO_EXIT -eq 0 ] && echo 'OK' || echo "FAILED (exit $MACRO_EXIT)")"
echo "=== Data fetch finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Exit with failure if either fetch failed
[ $YFINANCE_EXIT -eq 0 ] && [ $MACRO_EXIT -eq 0 ]
