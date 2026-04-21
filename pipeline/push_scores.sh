#!/bin/bash
# pipeline/push_scores.sh
# Runs the weekly scoring pipeline and pushes updated scores.csv to GitHub.
# Streamlit app reads scores.csv directly from the repo.
#
# Usage:
#   bash pipeline/push_scores.sh
#
# Cron (every Monday 07:00):
#   crontab -e
#   0 7 * * 1 /bin/bash ~/projects/HMM-RegimeModel/pipeline/push_scores.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/venv/bin/python"

cd "$PROJECT_DIR"

echo "[$(date +%H:%M:%S)] Running pipeline..."
"$VENV" pipeline/score.py --score-only

echo "[$(date +%H:%M:%S)] Pushing scores to GitHub..."
git add output/scores.csv

if git diff --cached --quiet; then
    echo "[$(date +%H:%M:%S)] No changes — scores unchanged since last push"
else
    DATE=$(date +%Y-%m-%d)
    LATEST_REGIME=$(tail -1 output/scores.csv | awk -F',' '{
        state=$2
        if (state==0) print "CRISIS"
        else if (state==1) print "BEAR"
        else print "BULL"
    }')
    LATEST_SCORE=$(tail -1 output/scores.csv | awk -F',' '{printf "%.3f", $6}')
    LATEST_ALERT=$(tail -1 output/scores.csv | awk -F',' '{print ($8==1) ? "ALERT" : "no alert"}')

    git commit -m "scores: $DATE — $LATEST_REGIME, score=$LATEST_SCORE ($LATEST_ALERT)"
    git push
    echo "[$(date +%H:%M:%S)] Done"
fi
