#!/usr/bin/env bash
# Fires when the ablation control run exits. Evaluates it on the same test split
# with thresholds tuned on val, then runs the paired treatment-vs-control
# comparison. This is the step that actually answers the project's central
# question, so it refuses to run on a partial control.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv/bin/python"
LOG="logs/after_ablation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +%H:%M:%S)] waiting for the ablation control run to exit"
while pgrep -f "scripts/run_ablation.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] control exited"
sleep 20

CTRL="weights/toxic_classifier_xlmr_v2_ablation/best_model"
if [[ ! -f "$CTRL/pytorch_model.bin" ]]; then
    echo "no control checkpoint at $CTRL -- cannot compare, stopping"
    exit 1
fi

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
echo "[$(date +%H:%M:%S)] control completed epochs: $EPOCHS"
if [[ "${EPOCHS:-0}" -lt 6 ]]; then
    echo "control only reached $EPOCHS/6 epochs. Comparing a partial control against a"
    echo "full treatment would attribute a training-length difference to the language"
    echo "signal. Refusing to run the comparison."
    exit 1
fi

echo "[$(date +%H:%M:%S)] evaluating the control on TEST, thresholds tuned on VAL"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false \
  "$PY" -m model.evaluation.evaluate \
    --model_path "$CTRL" \
    --val_file dataset/split/val.csv \
    --test_file dataset/split/test.csv \
    --max_length 512 --batch_size 64 --num_workers 8 \
    --output_dir evaluation_results

TREAT="evaluation_results/eval_20260830_072515"
CTRL_EVAL=$(ls -td evaluation_results/eval_2026083*/ | head -1)
CTRL_EVAL="${CTRL_EVAL%/}"
if [[ "$CTRL_EVAL" == "$TREAT" ]]; then
    echo "control evaluation directory not found (still points at the treatment). Stopping."
    exit 1
fi

echo "[$(date +%H:%M:%S)] paired comparison: treatment=$TREAT control=$CTRL_EVAL"
PYTHONPATH="$REPO" "$PY" scripts/compare_ablation.py "$TREAT" "$CTRL_EVAL" \
  | tee "experiments/ablation_result.txt"
echo "[$(date +%H:%M:%S)] done -- result saved to experiments/ablation_result.txt"
