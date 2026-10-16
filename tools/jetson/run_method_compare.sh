#!/usr/bin/env bash
# Compare every glue-line method on the recordings, in a window on the Jetson's desktop.
#
#   tools/jetson/run_method_compare.sh             prepare once if no run exists, then view
#   tools/jetson/run_method_compare.sh --prepare   always prepare a new run first
#
# Preparing needs the research venv (FLOD_LAB_VENV, default ~/.venvs/flod_lab: the app's
# packages plus PyTorch for SAM and CuPy) and the pipeline-lab checkout ~/flod_lab.
# The window only needs Tk, Pillow, OpenCV and NumPy; it runs with the app's venv.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_VENV="${FLOD_LAB_VENV:-$HOME/.venvs/flod_lab}"
APP_VENV="${FLOD_VENV:-$HOME/.venvs/flod_mas}"
RUNS="$HOME/flod_mas_data/method_compare"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
# torch needs NVIDIA cuDSS (pip package in the lab venv) and the Jetson's own CUDA libraries.
export LD_LIBRARY_PATH="$LAB_VENV/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cd "$REPO"
if [[ "${1:-}" == "--prepare" ]] || ! ls "$RUNS"/*/moments.json >/dev/null 2>&1; then
  mkdir -p "$RUNS"
  "$LAB_VENV/bin/python" -u -m tools.method_compare.prepare 2>&1 | tee "$RUNS/prepare_$(date +%Y%m%d_%H%M%S).log"
fi
exec "$APP_VENV/bin/python" -m tools.method_compare.viewer
