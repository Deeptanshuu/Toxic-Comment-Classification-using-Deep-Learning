# Development setup

This is a practical guide to the tools this project runs on: what each one is, why it is
here, and what breaks if you skip the reason and just run the command. If you already know
a tool (say, tmux), skip its explanation and use the command.

All commands below assume your shell is at the repo root.

## Setup with uv

[uv](https://docs.astral.sh/uv/) is a Python package manager and dependency resolver,
written in Rust. It replaces the usual pip + virtualenv combination: instead of manually
creating a virtual environment and then `pip install`-ing into it, `uv sync` does both in
one step, and does them from a **lockfile**.

The lockfile (`uv.lock`) is the point of using uv here. `pyproject.toml` states loose
intent — "I need `transformers==4.48.2`" — but that package itself depends on a tree of
other packages, each with their own version ranges, and resolving that tree can legitimately
produce a different set of exact versions on different days or different machines. `uv.lock`
freezes the *entire resolved tree* — every transitive dependency, pinned to an exact version
and content hash — so `uv sync` installs byte-identical packages everywhere it runs. Without
it, "works on my machine" is not a joke, it is the actual failure mode. `uv.lock` is checked
into git for exactly this reason; `requirements.txt` is a plain-pip-compatible export
generated from it (see [Linting and requirements.txt](#linting-and-requirementstxt) below),
not a second source of truth.

```bash
uv sync --all-extras
```

This creates `.venv/` and installs everything, including the CUDA build of torch
(`torch==2.6.0+cu124`). That specific build matters: PyPI is the default package index —
the server `pip`/`uv` fetch from unless told otherwise — and PyPI's own `torch` wheel for
this version is not the CUDA 12.4 build this project needs. The CUDA build only lives on
PyTorch's own package index, a separate server at `download.pytorch.org`. `pyproject.toml`
tells uv about this explicitly:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu124" }]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

`explicit = true` means this second index is consulted *only* for packages that opt in via
`tool.uv.sources` (here, just `torch`) — everything else still resolves from PyPI as normal.
If you ever install torch by some other route (plain `pip install torch`, a different
resolver, editing this block out), you can silently end up with a CPU-only build that
imports fine and then either fails on `.cuda()` or, worse, "works" by running on the CPU at
a small fraction of the speed. Always check:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.6.0+cu124 True
```

Installing everything (`--all-extras`) is convenient but pulls packages you may not need —
4-bit quantized Mistral for data augmentation, for instance, is a lot of dependency weight
if you are only training. Install narrower with `uv sync --extra serve`, etc.

| Group | Adds | Needed for |
|---|---|---|
| *(core, always installed)* | torch, transformers, scikit-learn, tensorboard, **mlflow**, wandb, langdetect | training, evaluation, `model/predict.py` |
| `serve` | streamlit, gradio, plotly, onnxruntime | `streamlit_app.py`, `app.py`, and `monitor_app.py` (see the venv note below) |
| `augment` | accelerate, bitsandbytes | `augmentation/*.py` (4-bit Mistral-7B generation) |
| `analysis` | seaborn, scipy | `analysis/*.py` plotting scripts |
| `dev` | ruff, pytest | linting, tests — on by default, skip with `--no-dev` |

`mlflow` is a core dependency, not a `serve` extra: the training loop always tries to use it
(see [Monitoring](#monitoring-a-run) below), so it has to be present in any environment that
runs `model/train.py`.

## Environment layout: `.venv` vs `.venv-uv`

There are two virtual environments sitting in this repo, and they are not interchangeable.

- **`.venv`** is the training environment. Every script (`train_tmux.sh`, `monitor.sh`,
  `mlflow_ui.sh`) points its `PY` variable here by default. It has the core dependencies
  plus `mlflow`, and *not* the `serve` extra.
- **`.venv-uv`** was built by an earlier `uv sync --all-extras`, before `mlflow` was added
  as a core dependency. It has `serve` (streamlit, plotly, gradio), `augment`, and
  `analysis` installed, but does not have `mlflow` at all.

This has one concrete, verified consequence worth knowing before you hit it: `scripts/monitor.sh` runs
`streamlit`, but `.venv` — its own default interpreter — does not have streamlit installed. Running
`scripts/monitor.sh` unmodified fails with `ModuleNotFoundError: No module named 'streamlit'`.
Two ways around it:

```bash
PY=.venv-uv/bin/python scripts/monitor.sh   # works today; .venv-uv has streamlit and plotly
# or, to fix it properly and stop needing two environments:
uv sync --extra serve                       # adds streamlit/plotly/etc. to .venv, keeps mlflow
```

`model/tracking.py` is written to degrade gracefully if `mlflow` is missing (it logs one
line and falls back to TensorBoard-only), so running *training* against `.venv-uv` would not
crash — it would just silently drop MLflow logging, which is a much easier mistake to miss
than a `ModuleNotFoundError`. Use `.venv` for training and `scripts/mlflow_ui.sh`.

## Training

```bash
GPUS=1 scripts/train_tmux.sh
```

This script solves one specific problem: training takes hours (roughly half an hour per
epoch, six epochs), and if you launch it in a plain SSH shell, closing your laptop or losing
wifi sends the remote shell a hangup signal that kills every process attached to it —
including your training run, with no checkpoint saved for the epoch in progress. That is
exactly how the April 2025 checkpoint ended up half-trained (see `docs/KNOWN_ISSUES.md`).

[tmux](https://github.com/tmux/tmux/wiki) is a terminal multiplexer: it runs a server process
on the remote machine that owns your shell sessions, independent of any one SSH connection.
"Attaching" opens a window into that server; "detaching" closes the window without touching
what is running inside it. The training process's parent becomes the tmux server, not your
SSH session, so it keeps running whether or not anyone is attached.

`scripts/train_tmux.sh` creates a session named `toxic` with three windows automatically:

| Window | What runs there | Port |
|---|---|---|
| `train` | `python -m model.train`, output tee'd to `logs/train_<timestamp>.log` | — |
| `board` | TensorBoard over `runs/` (skipped if something is already listening on the port) | 6006 |
| `gpu` | `watch nvidia-smi`, refreshed every 5s | — |

In practice you will usually add two more windows by hand for the other monitoring tools
(`tmux new-window -n monitor scripts/monitor.sh`, same for `mlflow_ui.sh`) — that gives the
five-window layout you will see if you attach to a live session.

```bash
tmux attach -t toxic       # attach
# Ctrl-b d                 # detach — the run keeps going
scripts/train_tmux.sh --kill   # stop everything
```

`GPUS=1` sets `CUDA_VISIBLE_DEVICES=1`, hiding every GPU except the second one from the
training process. This machine has two GPUs, and GPU 0 usually has another, unrelated
process holding memory on it — pinning to GPU 1 avoids contending with it. It also has a
second effect: `TrainingConfig.data_parallel` defaults to `True`, but `model/train.py` only
wraps the model in `nn.DataParallel` when `torch.cuda.device_count() > 1`
(`model/train.py:301`). With exactly one GPU visible, that count is 1, so DataParallel never
activates — which is deliberate. DataParallel puts the optimizer state on device 0 only, so
with two GPUs of unequal free memory it produces a lopsided ~3.8x memory imbalance, and at
this batch size it would not have bought any speedup to justify that cost. If you want to
opt into two-GPU DataParallel anyway, `GPUS=0,1`.

## Monitoring a run

| Tool | Started by | URL | Best for |
|---|---|---|---|
| TensorBoard | `train_tmux.sh` (automatic) | `:6006` | live per-step curves — loss, LR, grad norm, GPU memory — at full resolution, zero setup |
| Training dashboard | `scripts/monitor.sh` | `:8502` | a single glance at run health: status (live/stalled/finished), ETA, and computed sanity checks (is the LR schedule actually shaped like warmup-then-decay, are the class weights actually non-uniform) that a raw curve does not tell you by itself |
| MLflow UI | `scripts/mlflow_ui.sh` | `:5000` | comparing *runs against each other* — params, git commit, and the `run.kind=control`/`treatment` tag the ablation below sets, sortable in a table |

Training writes to both TensorBoard and MLflow through one call
(`model/tracking.py`'s `RunTracker`), because they are good at different things and neither
alone covers what this project needs. TensorBoard writes every scalar synchronously and has
no idea that two different directories under `runs/` might be worth comparing to each other.
MLflow's file-backed store (`./mlruns`, no server process needed) does know about runs as
first-class, comparable objects — it also stores parameters, tags, and artifacts, which
TensorBoard has no slot for at all — but writing to it one metric at a time costs roughly
1 ms per call, so `RunTracker` buffers per-step metrics and flushes them in batches, keeping
MLflow's overhead to a fraction of a percent of a training step. Either backend can fail
(the original SafeSummaryWriter design this replaces was built after MLflow's predecessor —
wandb — killed a training run outright on an auth error) without affecting the other or
stopping training; each self-disables after 10 consecutive failures.

**All three bind to `0.0.0.0` with no login.** `0.0.0.0` means "every network interface this
machine has," not just one. The scripts' own comments describe this as being for Tailscale
reachability, and it is reachable there, but that framing undersells it: unlike a
Tailscale-only bind, this is *also* reachable from the plain LAN, or from the public internet
if this machine has a routable IP, by anyone who can reach the host on any interface. There is
no password and, for MLflow specifically, the UI can delete runs and experiments outright.
That is a reasonable trade on a network you fully trust — this is not a defect to fix before
using the tools — but do not assume "bound to `0.0.0.0`" means "only reachable over
Tailscale"; it means the opposite of restricted.

## Running the ablation

An **ablation** is an experiment where you remove one component of a system and rerun
everything else unchanged, to isolate exactly what that component was responsible for. It is
the difference between "the model's output changed when I changed `lang_ids`" (true, and
verified — see `docs/MODEL.md`) and "the model got *better* because of `lang_ids`" (a
completely separate question, only answerable by comparison).

This project's whole premise is that telling the model which language it is reading, and
letting that signal steer attention, beats plain XLM-R fine-tuning. The bias that carries the
language signal used to be a mathematical no-op — it cancelled out under softmax regardless
of `lang_ids` — so historically that hypothesis was never actually tested; there was no
detectable language effect to test. Now that the bias is fixed and demonstrably changes the
model's output, the ablation is the one experiment that tells you whether it changes it *for
the better*. Two runs, identical in every other setting:

```bash
uv run python -m model.train                                # treatment: language conditioning live
TOXIC_DISABLE_LANG_CONDITIONING=1 uv run python -m model.train   # control: language pathway disabled
```

The environment variable is read once in `TrainingConfig.__post_init__`
(`model/training_config.py:451-458`) and forces `disable_lang_conditioning = True`, which
makes `model/language_aware_transformer.py` skip the language bias entirely
(`model/language_aware_transformer.py:252`) — architecturally identical model, just blind to
language. `model/train.py:871-874` tags the MLflow run `run.kind=control` or `=treatment`
accordingly, so the two runs are easy to find and compare in the MLflow UI once both are
done.

If you want to run the control arm inside `train_tmux.sh` rather than in the foreground,
note that the script only forwards four specific environment variables into the `train`
window (`TOKENIZERS_PARALLELISM`, `PYTHONPATH`, `PYTHONUNBUFFERED`, `CUDA_VISIBLE_DEVICES` —
see `scripts/train_tmux.sh:42`); `TOXIC_DISABLE_LANG_CONDITIONING` is not among them, so
setting it in the shell that *launches* the script has no effect on the training process.
Export it inside the `train` window's own shell before starting training there instead.

## Evaluation

```bash
uv run python -m model.evaluation.evaluate
```

The bare command is correct as of this branch — every default below matches what training
actually used. That was not always true (`--test_file` used to default to `val.csv`,
`--max_length` used to default to 128 against training's 512), which is why it is worth
re-checking `--help` after a pull rather than trusting a remembered command.

| Flag | Default | Notes |
|---|---|---|
| `--model_path` | `weights/toxic_classifier_xlm-roberta-large` | directory containing checkpoint subfolders |
| `--checkpoint` | latest under `--model_path` | e.g. `checkpoint_epoch05_20240213` |
| `--val_file` | `dataset/split/val.csv` | per-class thresholds are tuned here |
| `--test_file` | `dataset/split/test.csv` | headline metrics reported here, using thresholds frozen from `--val_file` |
| `--single_split_eval` | off | reproduces the old protocol: tune *and* report on `--test_file` alone (see below) |
| `--dynamic_padding` | off | **do not pass this** — this script's `DataLoader` has no length-aware `collate_fn`, so it crashes unconditionally |
| `--batch_size` | 64 | |
| `--output_dir` | `evaluation_results` | results land under `<output_dir>/eval_<timestamp>/` |
| `--num_workers` | 16 | dataloader workers for evaluation only — a separate setting from training's `num_workers` cap of 8 |
| `--cache_dir` | `cached_data` | tokenized-dataset cache |
| `--force_retokenize` | off | ignore the cache |
| `--prefetch_factor` | 2 | batches prefetched per worker |
| `--max_length` | 512 | must match the `max_length` used in training |
| `--gc_frequency` | 500 | |
| `--label_columns` | all six labels | override to evaluate a subset |

Why tune thresholds on one split and report on another at all: each of the six labels needs
its own probability cutoff — `toxic` might be well served by 0.5, while `threat`, with very
few positive examples, might need a much lower cutoff to get any recall at all. The threshold
search tries many cutoffs and keeps whichever scores highest. If you let it search *and* grade
itself on the same file, some cutoff will fit that file's particular sampling noise better
than the true optimum, and the reported score captures that noise as if it were skill — the
same mechanism as overfitting a model, just applied to one number instead of a few hundred
million. The default behavior avoids this by construction: thresholds come from `--val_file`,
and `--test_file` never influences the number computed from it. The smaller a class's support
(how few positive examples it has), the larger this effect can be, which is exactly why the
rare classes here (`threat`, `severe_toxic`, `identity_hate`) are the ones to be most
skeptical of under the old protocol. `--single_split_eval` exists only to reproduce that old,
biased number for side-by-side comparison — see `docs/KNOWN_ISSUES.md` and `docs/RESULTS.md`
for what that comparison actually shows.

## Known environment hazards

- **protobuf.** `mlflow` and `tensorboard` both depend on `protobuf`, and a bare `mlflow`
  install is happy to pull in `protobuf>=6`, which breaks `tensorboard==2.18` at runtime.
  The resolved version is currently pinned down to `5.29.6` — but only incidentally, via
  `wandb`'s own `protobuf<6` constraint (see the comment in `pyproject.toml` next to the
  `mlflow` dependency). This is fragile: if `wandb` is ever dropped from the dependency list,
  that ceiling disappears with it, and the next `uv lock` could silently pull `protobuf>=6`
  and break TensorBoard logging. If you remove `wandb`, add an explicit `protobuf<6`
  constraint in its place first.
- **`uv.lock` can drift from `pyproject.toml`.** Editing dependencies directly in
  `pyproject.toml` (adding a package, bumping a pin) does not update `uv.lock` by itself.
  `uv run`/`uv sync` will resolve and rewrite the lockfile automatically the next time
  either runs, but if you want to check without installing anything, `uv lock --check` tells
  you whether the lockfile is current; `uv lock` regenerates it in place. Get in the habit of
  running one of these after hand-editing `pyproject.toml`, rather than finding out from a
  confusing resolution error later.

## Linting and requirements.txt

[ruff](https://docs.astral.sh/ruff/) is a linter (and formatter, unused here) written in
Rust that replaces what used to take several separate tools — flake8 for style, isort for
import ordering, pyupgrade for modernizing syntax, flake8-bugbear for common bug patterns —
with one fast binary. `ruff.toml` enables exactly those checks (`E`/`W`/`F`/`I`/`UP`/`B`),
targets Python 3.11, and turns off line-length enforcement (`E501`) as not worth blocking on
here.

```bash
uv run ruff check .
uv export --format requirements-txt --all-extras --no-hashes -o requirements.txt  # regenerate, don't hand-edit
```

`requirements.txt` exists only for tools that understand pip's format but not uv's lockfile;
it is generated output, not something to edit directly — hand edits will just be overwritten
the next time someone regenerates it from `uv.lock`.
