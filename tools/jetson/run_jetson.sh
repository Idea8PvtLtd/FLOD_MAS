#!/usr/bin/env bash
# Start the inspection app on the Jetson.
#
#   tools/jetson/run_jetson.sh            # camera
#   FLOD_VIDEO_FILES=a.mp4:b.mp4 tools/jetson/run_jetson.sh   # recorded videos
#
# Defaults: headless VNC desktop :1, venv ~/.venvs/flod_mas, log in
# ~/flod_mas_data/logs. Override with DISPLAY, FLOD_VENV, FLOD_LOG_DIR.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${FLOD_VENV:-$HOME/.venvs/flod_mas}"
LOG_DIR="${FLOD_LOG_DIR:-$HOME/flod_mas_data/logs}"
export DISPLAY="${DISPLAY:-:1}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/app_$(date +%Y%m%d_%H%M%S).log"
echo "FLOD_MAS on $DISPLAY, log: $LOG"
cd "$REPO"
"$VENV/bin/python" -u main_1366.py --no-startup-window "$@" 2>&1 | tee "$LOG"
