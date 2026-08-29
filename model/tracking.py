# tracking.py
"""Unified experiment tracker: TensorBoard + MLflow behind one interface.

RunTracker is a drop-in replacement for train.py's SafeSummaryWriter (same
constructor, same methods, same never-crash discipline) that additionally
mirrors everything into an MLflow run, giving the project what TensorBoard
alone cannot: per-run params, sortable run comparison, and an artifact store.
TensorBoard keeps the dense per-step curves; MLflow gets the same scalars plus
params/tags/artifacts for comparing runs (e.g. the lang_id ablation).

Design rules, in order of importance:

1. No tracking failure may ever take down a training run. The run that
   produced the current headline checkpoint died on a logging backend auth
   error (docs/KNOWN_ISSUES.md #11). Every public method here is exception
   proof, including __init__, close() and __exit__.
2. The two backends fail independently. Each has its own consecutive-failure
   counter and self-disables after MAX_FAILURES, exactly like the original
   SafeSummaryWriter. A dead MLflow store never stops TensorBoard, and vice
   versa. If mlflow is not installed at all, this degrades to a pure
   TensorBoard writer with a single log line.
3. MLflow must not slow training down. The file-backed store costs ~1.5 ms
   per metric written one at a time (7+ ms per 5-scalar training step), so
   per-step scalars are buffered and written through log_batch in chunks of
   up to 1000. Measured on this box: a 5-scalar add_scalars costs ~0.6 ms
   with buffering, ~0.3% of a ~200 ms training step, so MLflow keeps full
   per-step resolution by default. TensorBoard is written synchronously at
   full resolution as before.

Configuration (all optional):
    MLFLOW_TRACKING_URI            store location, default file:./mlruns
    MLFLOW_EXPERIMENT_NAME         experiment, default toxic-comment-classification
    TRACKING_MLFLOW_MIN_INTERVAL   per-tag throttle in seconds for the MLflow
                                   side only, default 0 (full resolution).
                                   TensorBoard always gets every point.

Tag mapping: TensorBoard tags are used as MLflow keys unchanged whenever
possible. MLflow allows alphanumerics, '_', '-', '.', ' ' and '/' (':' only
outside Windows), so 'train/loss' and 'epoch/val_auc/toxic' pass through
as-is and the UI groups them by prefix just like TensorBoard does. Any other
character becomes '_', path-ambiguous pieces ('..', '//', leading '/') are
flattened, and keys are capped at MLflow's 250-char limit.

The MLflow run name is the basename of the TensorBoard log dir (e.g.
train_20260830_015843), so the two stores cross-reference trivially; the
full log dir is also recorded as the 'tb.log_dir' run tag.
"""

import json
import logging
import os
import re
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Both backends are optional at import time. train.py imports torch anyway,
# but a missing/broken backend package must never make `import model.tracking`
# fatal -- that would defeat the whole point of the class.
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:  # pragma: no cover - torch is always present in training
    SummaryWriter = None
    _TB_IMPORT_ERROR = e
else:
    _TB_IMPORT_ERROR = None

try:
    import mlflow  # noqa: F401 - imported to prove availability
    from mlflow.entities import Metric, Param, RunTag
    from mlflow.tracking import MlflowClient
except Exception as e:
    MlflowClient = None
    _MLFLOW_IMPORT_ERROR = e
else:
    _MLFLOW_IMPORT_ERROR = None

# MLflow's documented limits. Imported when available so a future MLflow that
# raises them is honored automatically; the fallbacks are the mlflow 2.x/3.x
# values and are safe (too small never errors, too large would).
try:
    from mlflow.utils.validation import (
        MAX_ENTITY_KEY_LENGTH,
        MAX_METRICS_PER_BATCH,
        MAX_PARAM_VAL_LENGTH,
        MAX_PARAMS_TAGS_PER_BATCH,
        MAX_TAG_VAL_LENGTH,
    )
except Exception:
    MAX_ENTITY_KEY_LENGTH = 250
    MAX_METRICS_PER_BATCH = 1000
    MAX_PARAM_VAL_LENGTH = 500
    MAX_PARAMS_TAGS_PER_BATCH = 100
    MAX_TAG_VAL_LENGTH = 5000

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT = "toxic-comment-classification"

# Portable MLflow key charset. ':' is legal on Linux but not Windows, so it is
# excluded to keep stores copyable between machines.
_KEY_BAD_CHARS = re.compile(r"[^/\w.\- ]")


def _sanitize_key(key):
    """Map a TensorBoard tag to a valid, stable MLflow key.

    Existing tags (train/loss, epoch/val_auc/toxic, ...) pass through
    unchanged. Disallowed characters become '_'; empty/'.'/'..' path segments
    are dropped or flattened because MLflow rejects keys that do not normalize
    to themselves as a posix path (they map to file paths in the file store).
    """
    key = _KEY_BAD_CHARS.sub("_", str(key))
    parts = [p for p in key.split("/") if p not in ("", ".")]
    key = "/".join("_" if p == ".." else p for p in parts)
    key = key[:MAX_ENTITY_KEY_LENGTH].rstrip("/")
    return key or "_"


def _stringify(value, limit):
    """Param/tag values must be strings within MLflow's length limit."""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except Exception:
            value = str(value)
    return value[:limit]


def _flatten(d, parent=""):
    """Flatten nested dicts to dot-joined keys ('scheduler.warmup_ratio')."""
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _context_tags(log_dir):
    """Tags that make runs comparable later. Every probe fails soft."""
    tags = {"tb.log_dir": str(log_dir)}

    repo = Path(__file__).resolve().parents[1]

    def _git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=5
        ).stdout.strip()

    try:
        sha = _git("rev-parse", "HEAD")
        if sha:
            tags["git.commit"] = sha
            tags["git.dirty"] = str(bool(_git("status", "--porcelain"))).lower()
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        if branch:
            tags["git.branch"] = branch
    except Exception:
        pass

    try:
        tags["sys.hostname"] = socket.gethostname()
    except Exception:
        pass

    try:
        import platform

        tags["sys.python"] = platform.python_version()
    except Exception:
        pass

    try:
        import torch

        tags["sys.torch"] = torch.__version__
        if torch.cuda.is_available():
            tags["sys.gpu_count"] = str(torch.cuda.device_count())
            tags["sys.gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import transformers

        tags["sys.transformers"] = transformers.__version__
    except Exception:
        pass

    return {k: _stringify(v, MAX_TAG_VAL_LENGTH) for k, v in tags.items()}


class RunTracker:
    """Fan-out tracker writing to TensorBoard and MLflow, crash-proof.

    Drop-in replacement for SafeSummaryWriter (same __init__, enabled,
    add_scalar, add_scalars, add_text, flush, close), plus the MLflow-only
    surface: log_params, set_tags, log_artifact, log_metrics, and context
    manager support so the MLflow run is always terminated, FAILED status
    included, even when training raises.
    """

    MAX_FAILURES = 10  # consecutive, per backend, matching SafeSummaryWriter

    def __init__(self, log_dir, experiment_name=None, run_name=None):
        self.log_dir = str(log_dir)
        # TensorBoard state
        self._tb = None
        self._tb_failures = 0
        # MLflow state
        self._ml_client = None
        self._ml_run_id = None
        self._ml_failures = 0
        self._ml_pending = []  # buffered Metric entities, flushed via log_batch
        self._ml_last_sent = {}  # tag -> wall time, for the optional throttle
        try:
            self._ml_min_interval = float(
                os.environ.get("TRACKING_MLFLOW_MIN_INTERVAL", "0") or 0
            )
        except ValueError:
            self._ml_min_interval = 0.0

        # -- TensorBoard, exactly as SafeSummaryWriter did it --
        if SummaryWriter is None:
            logger.warning(
                f"tensorboard unavailable ({_TB_IMPORT_ERROR}); TensorBoard logging disabled"
            )
        else:
            try:
                Path(self.log_dir).mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=self.log_dir)
                logger.info(f"TensorBoard logging to {self.log_dir}")
            except Exception as e:
                logger.warning(
                    f"Could not start TensorBoard writer ({str(e)}); continuing without it"
                )

        # -- MLflow, file-backed, no server required --
        if MlflowClient is None:
            logger.info(
                f"mlflow not installed ({_MLFLOW_IMPORT_ERROR}); "
                "tracking to TensorBoard only"
            )
            return
        try:
            uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
            # mlflow >= 3.15 gates the filesystem store behind an opt-in env
            # var (upstream calls it maintenance mode). A local serverless
            # store is exactly what this project wants, so opt in; setdefault
            # keeps an explicit user 'false' in charge. sqlite:///mlflow.db
            # via MLFLOW_TRACKING_URI remains a drop-in alternative.
            if uri.startswith("file:") or "://" not in uri:
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            experiment = (
                experiment_name
                or os.environ.get("MLFLOW_EXPERIMENT_NAME")
                or DEFAULT_EXPERIMENT
            )
            # Run name = TB run dir name, so runs/train_20260830_015843 and the
            # MLflow run are the same label in both UIs.
            name = run_name or Path(self.log_dir).name

            client = MlflowClient(tracking_uri=uri)
            exp = client.get_experiment_by_name(experiment)
            if exp is not None:
                exp_id = exp.experiment_id
            else:
                try:
                    exp_id = client.create_experiment(experiment)
                except Exception:
                    # Lost a create race with a concurrent process; re-read.
                    exp_id = client.get_experiment_by_name(experiment).experiment_id
            run = client.create_run(
                exp_id, tags=_context_tags(self.log_dir), run_name=name
            )
            self._ml_run_id = run.info.run_id
            self._ml_client = client
            logger.info(
                f"MLflow logging to {uri} experiment='{experiment}' "
                f"run='{name}' id={run.info.run_id}"
            )
        except Exception as e:
            self._ml_client = None
            self._ml_run_id = None
            logger.warning(
                f"Could not start MLflow run ({str(e)}); continuing with TensorBoard only"
            )

    # ------------------------------------------------------------------
    # Backend guards. One counter each: a dying MLflow store must not cost
    # TensorBoard anything, and vice versa.
    # ------------------------------------------------------------------

    @property
    def enabled(self):
        return self._tb is not None or self._ml_client is not None

    @property
    def tb_enabled(self):
        return self._tb is not None

    @property
    def mlflow_enabled(self):
        return self._ml_client is not None

    @property
    def mlflow_run_id(self):
        return self._ml_run_id

    def _tb_guard(self, fn, what):
        if self._tb is None:
            return
        try:
            fn()
            self._tb_failures = 0
        except Exception as e:
            self._tb_failures += 1
            logger.warning(f"TensorBoard {what} failed ({str(e)}); training continues")
            if self._tb_failures >= self.MAX_FAILURES:
                logger.warning(
                    f"Disabling TensorBoard after {self._tb_failures} consecutive failures"
                )
                try:
                    self._tb.close()
                except Exception:
                    pass
                self._tb = None

    def _ml_ok(self):
        self._ml_failures = 0

    def _ml_fail(self, what, exc):
        self._ml_failures += 1
        logger.warning(f"MLflow {what} failed ({str(exc)}); training continues")
        if self._ml_failures >= self.MAX_FAILURES:
            logger.warning(
                f"Disabling MLflow after {self._ml_failures} consecutive failures"
            )
            self._ml_disable(status="FAILED")

    def _ml_disable(self, status):
        client, run_id = self._ml_client, self._ml_run_id
        self._ml_client = None
        self._ml_run_id = None
        self._ml_pending = []
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status=status)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SafeSummaryWriter interface (drop-in)
    # ------------------------------------------------------------------

    def add_scalar(self, tag, value, step):
        if value is None:
            return
        self._tb_guard(
            lambda: self._tb.add_scalar(tag, float(value), step), f"add_scalar({tag})"
        )
        self._ml_buffer_metric(tag, value, step)

    def add_scalars(self, values, step):
        for tag, value in values.items():
            self.add_scalar(tag, value, step)

    def add_text(self, tag, text, step=0):
        self._tb_guard(lambda: self._tb.add_text(tag, text, step), f"add_text({tag})")
        if self._ml_client is None:
            return
        try:
            # Text has no length-limited slot in MLflow, so it becomes a small
            # markdown artifact (renders in the UI's artifact viewer).
            name = _sanitize_key(tag)
            suffix = f"_step{int(step)}" if step else ""
            self._ml_client.log_text(self._ml_run_id, str(text), f"text/{name}{suffix}.md")
            self._ml_ok()
        except Exception as e:
            self._ml_fail(f"add_text({tag})", e)

    def flush(self):
        self._ml_flush_metrics()
        self._tb_guard(lambda: self._tb.flush(), "flush")

    def close(self, status=None):
        """Flush and shut both backends down. Idempotent, never raises."""
        try:
            self._ml_flush_metrics()
            if self._tb is not None:
                try:
                    self._tb.close()
                except Exception as e:
                    logger.warning(f"TensorBoard close failed ({str(e)})")
                self._tb = None
            self._ml_disable(status=status or "FINISHED")
        except Exception as e:  # pragma: no cover - belt and braces
            logger.warning(f"Tracker close failed ({str(e)})")

    # ------------------------------------------------------------------
    # MLflow-only surface
    # ------------------------------------------------------------------

    def log_params(self, params):
        """Log a (possibly nested) dict as MLflow params.

        Nested dicts are flattened with '.', values are stringified (JSON where
        possible) and truncated to MLflow's per-value limit. MLflow rejects
        re-logging a param with a different value; that surfaces as one warning
        here, never an exception.
        """
        if self._ml_client is None:
            return
        try:
            entities = [
                Param(_sanitize_key(k), _stringify(v, MAX_PARAM_VAL_LENGTH))
                for k, v in _flatten(dict(params)).items()
            ]
            for i in range(0, len(entities), MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_client.log_batch(
                    self._ml_run_id, params=entities[i : i + MAX_PARAMS_TAGS_PER_BATCH]
                )
            self._ml_ok()
        except Exception as e:
            self._ml_fail("log_params", e)

    def set_tags(self, tags):
        if self._ml_client is None:
            return
        try:
            entities = [
                RunTag(_sanitize_key(k), _stringify(v, MAX_TAG_VAL_LENGTH))
                for k, v in _flatten(dict(tags)).items()
            ]
            for i in range(0, len(entities), MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_client.log_batch(
                    self._ml_run_id, tags=entities[i : i + MAX_PARAMS_TAGS_PER_BATCH]
                )
            self._ml_ok()
        except Exception as e:
            self._ml_fail("set_tags", e)

    def log_artifact(self, path):
        """Copy a file or directory into the run's artifact store."""
        if self._ml_client is None:
            return
        try:
            p = str(path)
            if os.path.isdir(p):
                base = os.path.basename(os.path.normpath(p))
                self._ml_client.log_artifacts(self._ml_run_id, p, artifact_path=base)
            else:
                self._ml_client.log_artifact(self._ml_run_id, p)
            self._ml_ok()
        except Exception as e:
            self._ml_fail(f"log_artifact({path})", e)

    def log_metrics(self, metrics, step):
        """Convenience alias for add_scalars (goes to both backends)."""
        self.add_scalars(metrics, step)

    # ------------------------------------------------------------------
    # MLflow metric buffering. The file store costs ~1 ms per unbatched
    # metric write; buffering through log_batch brings the per-call cost to
    # ~0.03 ms. Buffer flushes when full, on flush() (train.py calls it every
    # epoch) and on close(), so at most one batch is lost on a hard crash --
    # TensorBoard still has every point.
    # ------------------------------------------------------------------

    def _ml_buffer_metric(self, tag, value, step):
        if self._ml_client is None:
            return
        try:
            now = time.time()
            if self._ml_min_interval > 0:
                last = self._ml_last_sent.get(tag)
                if last is not None and (now - last) < self._ml_min_interval:
                    return
                self._ml_last_sent[tag] = now
            # NB: appending to the buffer is not backend success -- only a
            # real store write may reset the consecutive-failure counter,
            # otherwise a dead store alternating buffer-ok/flush-fail would
            # never trip the disable threshold.
            self._ml_pending.append(
                Metric(
                    key=_sanitize_key(tag),
                    value=float(value),
                    timestamp=int(now * 1000),
                    step=int(step),
                )
            )
        except Exception as e:
            self._ml_fail(f"add_scalar({tag})", e)
            return
        if len(self._ml_pending) >= MAX_METRICS_PER_BATCH:
            self._ml_flush_metrics()

    def _ml_flush_metrics(self):
        if self._ml_client is None:
            self._ml_pending = []
            return
        if not self._ml_pending:
            return
        # Hand the buffer off first: a failing batch is dropped rather than
        # retried forever (a poison payload would otherwise pin the counter).
        pending, self._ml_pending = self._ml_pending, []
        try:
            for i in range(0, len(pending), MAX_METRICS_PER_BATCH):
                self._ml_client.log_batch(
                    self._ml_run_id, metrics=pending[i : i + MAX_METRICS_PER_BATCH]
                )
            self._ml_ok()
        except Exception as e:
            self._ml_fail("log_batch(metrics)", e)

    # ------------------------------------------------------------------
    # Context manager: the MLflow run is always terminated, marked FAILED
    # when the with-block raises. The exception itself always propagates.
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close(status="FAILED" if exc_type is not None else "FINISHED")
        except Exception:  # pragma: no cover - close already never raises
            pass
        return False


# Alias so train.py can adopt this with a single import line that shadows its
# local class: `from model.tracking import RunTracker as SafeSummaryWriter`.
SafeSummaryWriter = RunTracker
