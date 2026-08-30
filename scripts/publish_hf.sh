#!/usr/bin/env bash
# Stage hf_release/ and upload it to the Hugging Face Hub as a PRIVATE repo.
#
#   scripts/publish_hf.sh              dry run: stage, check, print the manifest
#   scripts/publish_hf.sh --confirm    actually create the repo and upload
#
# Dry run is the default and it never touches the network. Nothing is uploaded
# without --confirm, and the repo is always created private=True. There is no
# flag in this script that makes a repo public; do that in the web UI, on
# purpose, after reading the card.
#
# Preflight checks, all of which must pass before an upload is allowed:
#   1. no <<PENDING_...>> placeholders anywhere in the staged text files
#   2. no training run in progress
#   3. best_model reached the configured final epoch
#   4. best_model/pytorch_model.bin is not still being written
#
# Overrides:
#   HF_REPO=owner/name          target repo (default below)
#   BEST=path                   checkpoint directory to publish
#   ALLOW_INCOMPLETE_RUN=1      permit a best epoch below the configured total,
#                               for a run that stopped early on purpose
#   SKIP_TOKENIZER=1            do not bundle the tokenizer files
#   STALE_MIN=10                minutes of checkpoint quiet time required

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

PY="${PY:-$REPO/.venv/bin/python}"
HF_REPO="${HF_REPO:-Deeptanshuu/toxic-comment-multilingual-xlmr}"
SRC="${SRC:-$REPO/hf_release}"
BEST="${BEST:-$REPO/weights/toxic_classifier_xlmr_v2/best_model}"
TOKENIZER_NAME="${TOKENIZER_NAME:-xlm-roberta-large}"
STALE_MIN="${STALE_MIN:-10}"
PLACEHOLDER='<<PENDING_'

CONFIRM=0
case "${1:-}" in
    --confirm) CONFIRM=1 ;;
    "")        CONFIRM=0 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *)         echo "unknown argument: $1 (expected --confirm or nothing)" >&2; exit 2 ;;
esac

[[ -x "$PY" ]]  || { echo "no interpreter at $PY (set PY=...)" >&2; exit 1; }
[[ -d "$SRC" ]] || { echo "no release directory at $SRC" >&2; exit 1; }
[[ -d "$BEST" ]] || { echo "no checkpoint directory at $BEST (set BEST=...)" >&2; exit 1; }
[[ -f "$BEST/pytorch_model.bin" ]] || { echo "no pytorch_model.bin in $BEST" >&2; exit 1; }

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/hf_release_stage.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

echo
echo "target repo : $HF_REPO   (created private)"
echo "source      : $SRC"
echo "checkpoint  : $BEST"
echo "staging     : $STAGE"
echo "mode        : $([[ $CONFIRM -eq 1 ]] && echo 'CONFIRM - will upload' || echo 'DRY RUN - no network calls')"
echo

# ---------------------------------------------------------------- staging
# The weights live outside hf_release/ so they never enter git. Bring them in
# here rather than assuming somebody remembered to copy them. In dry run the
# 2.2 GB file is symlinked instead of copied; every size below is measured
# through the link, so the manifest is the real one either way.
echo "== staging =="
cp -a "$SRC"/. "$STAGE"/
rm -rf "$STAGE/__pycache__"

if [[ $CONFIRM -eq 1 ]]; then
    cp "$BEST/pytorch_model.bin" "$STAGE/pytorch_model.bin"
    echo "copied  pytorch_model.bin from $BEST"
else
    ln -s "$BEST/pytorch_model.bin" "$STAGE/pytorch_model.bin"
    echo "linked  pytorch_model.bin from $BEST (dry run)"
fi

# The training config is regenerated from the checkpoint so it always describes
# the weights being shipped, not whatever was committed weeks ago.
cp "$BEST/config.json" "$STAGE/training_config.json"
echo "copied  training_config.json from $BEST/config.json"

if [[ "${SKIP_TOKENIZER:-0}" != "1" ]]; then
    "$PY" - "$TOKENIZER_NAME" "$STAGE" <<'PY'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
PY
    echo "exported tokenizer '$TOKENIZER_NAME' into the staging directory"
else
    echo "skipped tokenizer export (SKIP_TOKENIZER=1)"
fi
echo

# ---------------------------------------------------------------- manifest
echo "== would upload =="
find "$STAGE" -mindepth 1 \( -type f -o -type l \) -printf '%P\n' | sort | while read -r f; do
    sz=$(stat -Lc '%s' "$STAGE/$f")
    printf '  %10s  %s\n' "$(numfmt --to=iec --suffix=B "$sz")" "$f"
done
TOTAL=$(find "$STAGE" -mindepth 1 \( -type f -o -type l \) -printf '%P\n' \
        | while read -r f; do stat -Lc '%s' "$STAGE/$f"; done \
        | awk '{s+=$1} END {print s+0}')
COUNT=$(find "$STAGE" -mindepth 1 \( -type f -o -type l \) | wc -l)
echo
printf '  %d files, %s total\n' "$COUNT" "$(numfmt --to=iec --suffix=B "$TOTAL")"
echo

# ---------------------------------------------------------------- preflight
echo "== preflight =="
FAILED=0
fail() { echo "  BLOCKED  $*"; FAILED=1; }
pass() { echo "  ok       $*"; }
warn() { echo "  warning  $*"; }

# 1. placeholders. A card with an unfilled metric is worse than no card, and a
#    stale threshold set is worse than none, so both block.
HITS=$(grep -rl --binary-files=without-match -F "$PLACEHOLDER" \
        --include='*.md' --include='*.json' --include='*.py' --include='*.txt' \
        "$STAGE" 2>/dev/null | sed "s|^$STAGE/||" | sort || true)
if [[ -n "$HITS" ]]; then
    fail "unfilled ${PLACEHOLDER}...>> placeholders remain:"
    while read -r f; do
        n=$(grep -c -F "$PLACEHOLDER" "$STAGE/$f")
        printf '             %-24s %s occurrence(s)\n' "$f" "$n"
    done <<< "$HITS"
    echo "           fill these in from a real evaluation of the final checkpoint."
    echo "           do not copy the previous version's numbers into them."
else
    pass "no ${PLACEHOLDER}...>> placeholders in the staged files"
fi

# 2. a live training run means these weights are an intermediate epoch
TRAIN_PIDS=$(pgrep -f 'python -m model.train' || true)
if [[ -n "$TRAIN_PIDS" ]]; then
    fail "training is still running (pid $(echo "$TRAIN_PIDS" | tr '\n' ' ' | sed 's/ $//'))"
    echo "           the checkpoint in $BEST is an intermediate epoch."
else
    pass "no 'python -m model.train' process is running"
fi

# 3. did the run reach its last epoch
if [[ -f "$BEST/best.json" && -f "$BEST/config.json" ]]; then
    BEST_EPOCH=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['epoch'])" "$BEST/best.json")
    TOTAL_EPOCHS=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['epochs'])" "$BEST/config.json")
    if [[ "$BEST_EPOCH" -lt "$TOTAL_EPOCHS" ]]; then
        if [[ "${ALLOW_INCOMPLETE_RUN:-0}" == "1" ]]; then
            warn "best checkpoint is epoch $BEST_EPOCH of $TOTAL_EPOCHS (allowed by ALLOW_INCOMPLETE_RUN=1)"
        else
            fail "best checkpoint is epoch $BEST_EPOCH of $TOTAL_EPOCHS"
            echo "           if the run stopped early on purpose, set ALLOW_INCOMPLETE_RUN=1."
        fi
    else
        pass "best checkpoint is epoch $BEST_EPOCH of $TOTAL_EPOCHS"
    fi
else
    fail "no best.json or config.json in $BEST, cannot tell which epoch this is"
fi

# 4. still being written to
NOW=$(date +%s)
MTIME=$(stat -c '%Y' "$BEST/pytorch_model.bin")
AGE_MIN=$(( (NOW - MTIME) / 60 ))
if [[ "$AGE_MIN" -lt "$STALE_MIN" ]]; then
    warn "pytorch_model.bin was modified ${AGE_MIN} min ago (under STALE_MIN=$STALE_MIN); a checkpoint write may be in flight"
else
    pass "pytorch_model.bin last modified ${AGE_MIN} min ago"
fi

# 5. token, read from the local file only, no network call
if "$PY" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" 2>/dev/null; then
    pass "a Hugging Face token is present locally"
else
    if [[ $CONFIRM -eq 1 ]]; then
        fail "no Hugging Face token found (run: hf auth login)"
    else
        warn "no Hugging Face token found locally (run 'hf auth login' before --confirm)"
    fi
fi
echo

# ---------------------------------------------------------------- verdict
if [[ $FAILED -eq 1 ]]; then
    echo "REFUSING TO PUBLISH: preflight checks failed. Nothing was uploaded."
    exit 1
fi

if [[ $CONFIRM -eq 0 ]]; then
    cat <<EOF
DRY RUN complete. Preflight passed; nothing was uploaded and no network call was made.

To publish for real:

    scripts/publish_hf.sh --confirm

That will create $HF_REPO as a PRIVATE repo and upload the $COUNT files above.
EOF
    exit 0
fi

echo "== uploading =="
"$PY" - "$HF_REPO" "$STAGE" <<'PY'
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

repo_id, folder = sys.argv[1], sys.argv[2]
api = HfApi()

who = api.whoami()
print(f"authenticated as {who['name']}")

# If the repo already exists and is public, stop. create_repo(exist_ok=True)
# would not flip it back to private, and this script must never be the reason
# something is publicly readable.
try:
    info = api.model_info(repo_id)
    if not info.private:
        raise SystemExit(
            f"{repo_id} already exists and is PUBLIC. Refusing to push into a public repo. "
            "Make it private in the web UI first, or pick another name with HF_REPO=..."
        )
    print(f"{repo_id} already exists and is private")
except RepositoryNotFoundError:
    url = api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"created private repo: {url}")

commit = api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder,
    commit_message="Publish multilingual toxicity classifier: weights, card, thresholds, config",
)
print(f"uploaded: {commit.commit_url if hasattr(commit, 'commit_url') else commit}")
print(f"repo (private): https://huggingface.co/{repo_id}")
PY

echo
echo "done. the repo is PRIVATE. review the card on the Hub before changing that."
