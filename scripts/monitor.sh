#!/usr/bin/env bash
# Launch the live training-monitor Streamlit dashboard.
#
#   scripts/monitor.sh                 start on :8502 (Tailscale-reachable)
#   PORT=8600 scripts/monitor.sh       start on a different port
#   RUNS_DIR=/path scripts/monitor.sh  point at a different runs/ directory
#
# Reads TensorBoard event files under runs/<run_name>/ -- start training
# first (e.g. scripts/train_tmux.sh) so there is something to watch.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
# The dashboard needs streamlit and plotly, which live in .venv-uv, not in
# .venv (that one carries the training stack). Keep them separate: installing
# streamlit into .venv risks pulling protobuf 6, which breaks tensorboard 2.18.
PY="${PY:-$REPO/.venv-uv/bin/python}"
PORT="${PORT:-8502}"
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"

[[ -x "$PY" ]] || { echo "no interpreter at $PY (set PY=...)" >&2; exit 1; }

export MONITOR_RUNS_DIR="$RUNS_DIR"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"

IP=$(tailscale ip -4 2>/dev/null | head -1 || true)
[[ -z "$IP" ]] && IP=$(hostname -I 2>/dev/null | awk '{print $1}')
cat <<EOF

training monitor : http://${IP:-<host>}:$PORT
runs directory   : $RUNS_DIR

stop             : Ctrl-C
EOF

exec "$PY" -m streamlit run "$REPO/monitor_app.py" \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
