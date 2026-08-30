#!/usr/bin/env bash
# Fires when the main training run exits. Evaluates it, then starts the ablation.
# Refuses to do either if the run did not actually finish its 6 epochs.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv/bin/python"
LOG="logs/after_training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +%H:%M:%S)] waiting for the training process to exit"
while pgrep -f "python -m model.train" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] trainer exited"
sleep 20

EPOCHS=$("$PY" - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os
ds = [x for x in glob.glob('runs/*') if '_smoke' not in x]
d = sorted(ds, key=os.path.getmtime)[-1]
ea = EventAccumulator(d); ea.Reload()
t = ea.Tags()['scalars']
print(len(ea.Scalars('epoch/val_auc_macro')) if 'epoch/val_auc_macro' in t else 0)
PY
)
echo "[$(date +%H:%M:%S)] completed epochs: $EPOCHS"

BEST="weights/toxic_classifier_xlmr_v2/best_model"
if [[ ! -f "$BEST/pytorch_model.bin" ]]; then
    echo "no best checkpoint at $BEST -- stopping, not starting the ablation"
    exit 1
fi

if [[ "${EPOCHS:-0}" -lt 6 ]]; then
    echo "run finished with only $EPOCHS/6 epochs; evaluating what exists but NOT starting the ablation"
    ABLATE=0
else
    ABLATE=1
fi

echo "[$(date +%H:%M:%S)] evaluating best_model on TEST with thresholds tuned on VAL"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false \
  "$PY" -m model.evaluation.evaluate \
    --model_path "$BEST" \
    --val_file dataset/split/val.csv \
    --test_file dataset/split/test.csv \
    --max_length 512 --batch_size 64 --num_workers 8 \
    --output_dir evaluation_results
echo "[$(date +%H:%M:%S)] evaluation exit: $?"

if [[ "$ABLATE" -eq 1 ]]; then
    echo "[$(date +%H:%M:%S)] starting the ablation control run"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false \
      "$PY" scripts/run_ablation.py
    echo "[$(date +%H:%M:%S)] ablation exit: $?"
fi
echo "[$(date +%H:%M:%S)] done"
