#!/usr/bin/env bash
# Launch training in tmux with TensorBoard and a GPU monitor alongside.
#
#   scripts/train_tmux.sh              start (session: toxic)
#   tmux attach -t toxic               attach
#   Ctrl-b d                           detach, leaves training running
#   scripts/train_tmux.sh --kill       stop everything
#
# TensorBoard binds to 0.0.0.0 so it is reachable over Tailscale at
# http://<tailscale-ip>:6006

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
SESSION="${SESSION:-toxic}"
TB_PORT="${TB_PORT:-6006}"
PY="${PY:-$REPO/.venv/bin/python}"
# GPU 0 usually has another process on it. Default to GPU 1 alone: with one
# visible device DataParallel never activates, which is what we want here --
# it is memory-imbalanced (3.8x onto device 0) and buys nothing at this batch
# size. Override with GPUS=0,1 to opt in.
GPUS="${GPUS:-1}"

if [[ "${1:-}" == "--kill" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed session $SESSION" || echo "no session $SESSION"
    exit 0
fi

# Reuse an existing session (TensorBoard may already be serving in it) rather
# than colliding on port 6006.
REUSE=0
tmux has-session -t "$SESSION" 2>/dev/null && REUSE=1

[[ -x "$PY" ]] || { echo "no interpreter at $PY (set PY=...)" >&2; exit 1; }

mkdir -p logs runs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_${STAMP}.log"

# Forking dataloader workers after the fast tokenizer has been used warns and can
# deadlock. Must be set before the interpreter starts.
ENVS="export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPUS"

if [[ $REUSE -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n train -c "$REPO"
else
    tmux has-session -t "$SESSION:train" 2>/dev/null || tmux new-window -t "$SESSION" -n train -c "$REPO"
fi
tmux send-keys -t "$SESSION:train" "$ENVS && $PY -m model.train 2>&1 | tee $LOG" C-m

# Only start TensorBoard if nothing is already bound to the port.
if ! ss -lnt 2>/dev/null | grep -q ":$TB_PORT "; then
    tmux has-session -t "$SESSION:board" 2>/dev/null || tmux new-window -t "$SESSION" -n board -c "$REPO"
    tmux send-keys -t "$SESSION:board" \
        "$PY -m tensorboard.main --logdir runs --port $TB_PORT --bind_all --reload_multifile true --samples_per_plugin scalars=100000 --window_title toxic-comment-training" C-m
else
    echo "tensorboard already serving on :$TB_PORT, leaving it alone"
fi

if ! tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx gpu; then
    tmux new-window -t "$SESSION" -n gpu -c "$REPO"
    tmux send-keys -t "$SESSION:gpu" \
        "watch -n 5 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv'" C-m
fi

tmux select-window -t "$SESSION:train"

IP=$(tailscale ip -4 2>/dev/null | head -1 || true)
[[ -z "$IP" ]] && IP=$(hostname -I 2>/dev/null | awk '{print $1}')
cat <<EOF

session   : $SESSION   (train / board / gpu)
log       : $LOG
tensorboard: http://${IP:-<host>}:$TB_PORT

attach    : tmux attach -t $SESSION
detach    : Ctrl-b d
stop      : scripts/train_tmux.sh --kill
EOF
