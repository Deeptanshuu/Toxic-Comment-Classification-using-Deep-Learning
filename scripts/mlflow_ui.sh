#!/usr/bin/env bash
# Serve the MLflow UI over the file-backed store that model/tracking.py writes.
#
#   scripts/mlflow_ui.sh               start on port 5000
#   MLFLOW_PORT=5001 scripts/mlflow_ui.sh
#   Ctrl-C                             stop
#
# Binds to 0.0.0.0 so it is reachable over Tailscale at
# http://<tailscale-ip>:5000

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PORT="${MLFLOW_PORT:-5000}"
PY="${PY:-$REPO/.venv/bin/python}"
URI="${MLFLOW_TRACKING_URI:-file:./mlruns}"

[[ -x "$PY" ]] || { echo "no interpreter at $PY (set PY=...)" >&2; exit 1; }

if ss -lnt 2>/dev/null | grep -q ":$PORT "; then
    echo "something is already serving on :$PORT (set MLFLOW_PORT=...)" >&2
    exit 1
fi

# mlflow >= 3.15 requires this opt-in for the filesystem store.
export MLFLOW_ALLOW_FILE_STORE=true

IP=$(tailscale ip -4 2>/dev/null | head -1 || true)
[[ -z "$IP" ]] && IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "mlflow ui : http://${IP:-<host>}:$PORT   (store: $URI)"

# --allowed-hosts '*': mlflow >= 3 ships a Host-header check that rejects
# anything but localhost even when bound to 0.0.0.0; the Tailscale IP must be
# allowed through. Same open posture as TensorBoard's --bind_all next door.
exec "$PY" -m mlflow ui --backend-store-uri "$URI" \
    --host 0.0.0.0 --port "$PORT" --allowed-hosts '*'
