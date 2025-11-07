#!/bin/bash
# Daily Analysis Runner
# Executes macro regime analysis followed by stock signal analysis

set -e  # Exit on error

# Define log file
LOG_FILE="/var/log/optimizer/daily_analysis.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Function to log and display
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

log "========================================"
log "[$DATE] Starting daily analysis"
log "========================================"

# Change to app directory
cd /app

# Step 1: Run macro regime analysis
log "[$DATE] Step 1/2: Running macro regime analysis..."
if python3 src/macro_regime/run_regime_analysis.py 2>&1 | tee -a "$LOG_FILE"; then
    log "[$DATE] ✅ Macro regime analysis completed successfully"
else
    log "[$DATE] ❌ ERROR: Macro regime analysis failed (exit code: $?)"
    log "[$DATE] Aborting daily analysis run"
    exit 1
fi

# Step 2: Run stock signal analysis
log "[$DATE] Step 2/2: Running stock signal analysis..."
if python3 src/stock_analyzer/run_signal_analysis.py 2>&1 | tee -a "$LOG_FILE"; then
    log "[$DATE] ✅ Stock signal analysis completed successfully"
else
    log "[$DATE] ❌ ERROR: Stock signal analysis failed (exit code: $?)"
    log "[$DATE] Daily analysis run completed with errors"
    exit 1
fi

# Success
COMPLETION_DATE=$(date '+%Y-%m-%d %H:%M:%S')
log "[$COMPLETION_DATE] ✅ Daily analysis completed successfully"
log "========================================"
log ""

exit 0
