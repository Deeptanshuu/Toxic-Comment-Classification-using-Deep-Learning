#!/usr/bin/env bash
# Serve the Streamlit inference demo.
#
#   scripts/serve_app.sh                 port 8501, retrained local checkpoint
#   TOXIC_MODEL_PATH=<path> ...          serve a different checkpoint
#   APP_PORT=8503 scripts/serve_app.sh   different port
#
# Binds 0.0.0.0 so it is reachable over Tailscale at http://<host>:8501

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PORT="${APP_PORT:-8501}"
# The app needs streamlit and plotly, which live in .venv-uv. .venv carries the
# training stack; installing streamlit there risks pulling protobuf 6 and
# breaking tensorboard 2.18.
PY="${PY:-$REPO/.venv-uv/bin/python}"
GPUS="${GPUS:-1}"

[[ -x "$PY" ]] || { echo "no interpreter at $PY (set PY=...)" >&2; exit 1; }
if ss -lnt 2>/dev/null | grep -q ":$PORT "; then
    echo "something is already serving on :$PORT (set APP_PORT=...)" >&2
    exit 1
fi

export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPUS"

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
CKPT="${TOXIC_MODEL_PATH:-weights/toxic_classifier_xlmr_v2/best_model/pytorch_model.bin}"
echo "inference app : http://${IP:-<host>}:$PORT"
echo "checkpoint    : $CKPT"
echo "gpu           : $GPUS"

exec "$PY" -m streamlit run streamlit_app.py \
    --server.port "$PORT" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false
