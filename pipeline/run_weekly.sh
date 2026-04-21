#!/bin/bash
# run_weekly.sh
# Weekly scoring script — run every Monday morning before market open.
# Loads the saved model bundle (fast) and appends latest scores.
#
# To schedule (cron):
#   crontab -e
#   0 7 * * 1 /bin/bash /home/stevej/projects/HMM-RegimeModel/pipeline/run_weekly.sh
#
# To run manually:
#   bash pipeline/run_weekly.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/output/logs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/score_$(date +%Y%m%d_%H%M%S).log"

echo "=== HMM-RegimeModel Weekly Score ===" | tee "$LOG_FILE"
echo "Run date: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"

"$VENV" pipeline/score.py --score-only 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
