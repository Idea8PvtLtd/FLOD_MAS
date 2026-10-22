#!/usr/bin/env bash
# Glue-width defect research on the Jetson (docs/GLUE_DEFECTS.md).
#
#   tools/jetson/run_glue_defects.sh measure                app glue model: widths and line paths
#   tools/jetson/run_glue_defects.sh visibility             glue-edge sharpness along every line
#   tools/jetson/run_glue_defects.sh sam [--clahe]          SAM 2.1 large on the photo as it is
#   tools/jetson/run_glue_defects.sh labseg MODEL...        pipeline-lab SegFormers (~/flod_lab/trained_models)
#   tools/jetson/run_glue_defects.sh segments --repo DIR    main's graded segments + the two checks
#   tools/jetson/run_glue_defects.sh view                   the window, on the Jetson's desktop
#
# sam and labseg need the research venv (FLOD_LAB_VENV, default ~/.venvs/flod_lab) and the
# pipeline-lab checkout ~/flod_lab; the rest run with the app's venv. Data: --data DIR,
# default ~/flod_mas_data/glue_defects.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_VENV="${FLOD_LAB_VENV:-$HOME/.venvs/flod_lab}"
APP_VENV="${FLOD_VENV:-$HOME/.venvs/flod_mas}"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export HF_HUB_OFFLINE=1
# torch needs NVIDIA cuDSS (pip package in the lab venv) and the Jetson's own CUDA libraries.
export LD_LIBRARY_PATH="$LAB_VENV/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cd "$REPO"
step="${1:?measure | visibility | sam | labseg | segments | view}"
shift
case "$step" in
  measure)    exec "$APP_VENV/bin/python" -u -m tools.glue_defects.measure_app "$@" ;;
  visibility) exec "$APP_VENV/bin/python" -u -m tools.glue_defects.visibility "$@" ;;
  sam)        exec "$LAB_VENV/bin/python" -u -m tools.glue_defects.sam_widths "$@" ;;
  labseg)     exec "$LAB_VENV/bin/python" -u -m tools.glue_defects.labseg_widths "$@" ;;
  segments)   exec "$APP_VENV/bin/python" -u -m tools.glue_defects.segments_check "$@" ;;
  view)       exec "$APP_VENV/bin/python" -m tools.glue_defects.viewer "$@" ;;
  *) echo "unknown step: $step" >&2; exit 2 ;;
esac
