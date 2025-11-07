#!/bin/bash
# Entrypoint script for Portfolio Optimizer Docker container
# Starts cron daemon and keeps container running

set -e

# Verify Python packages are installed
echo "Verifying Python dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "ERROR: Python dependencies not installed correctly during build!"
    exit 1
fi
echo "✓ Python dependencies OK"

# Setup cron
echo "Setting up cron..."
mkdir -p /etc/cron.d
if [ -f /app/docker/crontab ]; then
    cp /app/docker/crontab /etc/cron.d/optimizer-cron
    chmod 0644 /etc/cron.d/optimizer-cron
    crontab /etc/cron.d/optimizer-cron
    echo "✓ Crontab configured"
fi
touch /var/log/cron.log

echo "=================================================="
echo "Portfolio Optimizer - Daily Analysis Container"
echo "=================================================="
echo "Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Print environment info
echo "Environment: ${ENVIRONMENT:-development}"
echo "Python version: $(python3 --version)"
echo "Working directory: $(pwd)"
echo ""

# Check if required environment variables are set
if [ -z "$SUPABASE_DB_URL" ]; then
    echo "WARNING: SUPABASE_DB_URL not set. Database operations may fail."
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set. BAML operations may fail."
fi

# Export all environment variables to cron environment
# This ensures cron jobs have access to env vars
printenv | grep -v "no_proxy" >> /etc/environment

echo "=================================================="
echo "Cron Schedule Configuration"
echo "=================================================="
echo "Timezone: Europe/Rome (Italy)"
echo "Daily analysis runs at: 22:00 Italy time"
echo "Execution order:"
echo "  1. Macro regime analysis (src/macro_regime/run_regime_analysis.py)"
echo "  2. Stock signal analysis (src/stock_analyzer/run_signal_analysis.py)"
echo ""
echo "Logs location: /var/log/optimizer/daily_analysis.log"
echo "Cron logs: /var/log/cron.log"
echo "=================================================="
echo ""

# Optional: Run analysis immediately on startup (for testing)
if [ "$RUN_ON_STARTUP" = "true" ]; then
    echo "RUN_ON_STARTUP=true detected. Running analysis now..."
    /app/run_daily_analysis.sh
    echo ""
fi

echo "Starting cron daemon..."
echo "Container is now running. Press Ctrl+C to stop."
echo ""

# Start cron in foreground mode
# This keeps the container running
exec "$@"
