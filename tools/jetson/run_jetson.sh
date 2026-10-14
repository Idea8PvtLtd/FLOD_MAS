#!/usr/bin/env bash
# Start the inspection app on the Jetson's headless VNC desktop.
#
#   tools/jetson/run_jetson.sh
#
# Recorded videos instead of the camera, logging and the GPU are set in this
# Jetson's machine settings (Documents/data files/Files/machine_settings.json:
# video_source.files, logging.enabled, local_inspection.execution_provider).
# Console output (including anything printed before logging starts) also goes
# to ~/flod_mas_data/logs. Override with DISPLAY, FLOD_VENV, FLOD_LOG_DIR.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${FLOD_VENV:-$HOME/.venvs/flod_mas}"
LOG_DIR="${FLOD_LOG_DIR:-$HOME/flod_mas_data/logs}"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/app_$(date +%Y%m%d_%H%M%S).log"
echo "FLOD_MAS on $DISPLAY, console log: $LOG"
cd "$REPO"
"$VENV/bin/python" -u main_1366.py --no-startup-window "$@" 2>&1 | tee "$LOG"
