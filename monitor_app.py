"""Live training-monitor dashboard for XLM-RoBERTa toxicity-classifier runs.

Reads TensorBoard scalar event files under runs/<run_name>/ directly (via
tensorboard's EventAccumulator) and renders a Streamlit dashboard styled to
match streamlit_app.py. Does not import streamlit_app.py or anything under
model/ -- this file is self-contained and safe to run without loading the
(2.2 GB) inference model.

Point MONITOR_RUNS_DIR at a different directory to watch runs elsewhere
(scripts/monitor.sh exposes this as RUNS_DIR).
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Import tensorboard's event reader before streamlit. Both bundle their own
# protobuf-generated code, and in this environment (protobuf 5.x + TF 2.18)
# importing streamlit first causes a global protobuf descriptor-pool clash --
# TF's lazy TF-detection then raises AttributeError deep inside
# tensorboard.compat, which tensorboard catches and logs as a traceback
# before falling back to its stub reader. The fallback works either way (it
# never reaches this file's code), but importing tensorboard first avoids
# the clash and the noisy log output entirely.
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402,I001

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

# Page config must be the first Streamlit call.
st.set_page_config(
    page_title="Training Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme -- copied from streamlit_app.py's THEME dict (deliberately not
# imported: that module loads a 2.2 GB model at import time) so the two apps
# read as siblings.
# --------------------------------------------------------------------------
THEME = {
    "primary": "#2D3142",
    "background": "#FFFFFF",
    "surface": "#FFFFFF",
    "text": "#000000",
    "text_secondary": "#FFFFFF",
    "button": "#000000",
    "toxic": "#E53935",
    "non_toxic": "#2E7D32",
    "warning": "#F57C00",
    "info": "#1976D2",
    "sidebar_bg": "#FFFFFF",
    "card_bg": "white",
    "input_bg": "#F8F9FA",
}

RARE_CLASS_COLOR = "#8E24AA"
LIVE_THRESHOLD_SEC = 120
SCALAR_COLUMNS = ["step", "wall_time", "value"]

# (display name, tag, is_rare) -- abundant classes first, then the rare/weak
# ones the user specifically wants visually distinguished.
CLASS_TAGS = [
    ("toxic", "epoch/val_auc/toxic", False),
    ("obscene", "epoch/val_auc/obscene", False),
    ("insult", "epoch/val_auc/insult", False),
    ("severe_toxic", "epoch/val_auc/severe_toxic", True),
    ("threat", "epoch/val_auc/threat", True),
    ("identity_hate", "epoch/val_auc/identity_hate", True),
]


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def inject_css() -> None:
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root, html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
        color: {THEME["text"]};
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
        color: {THEME["text"]};
    }}

    .stApp {{
        background-color: {THEME["background"]};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    section[data-testid="stSidebar"] {{
        background-color: {THEME["sidebar_bg"]};
        color: {THEME["text"]};
    }}

    .main-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: {THEME["text"]};
        margin-bottom: 0.2rem;
        letter-spacing: -0.03em;
    }}

    .subtitle {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 400;
        color: {THEME["text"]};
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }}

    /* Metric cards -- cover both old and new Streamlit testids */
    div[data-testid="stMetric"], div[data-testid="metric-container"] {{
        background-color: {THEME["card_bg"]};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid {hex_to_rgba(THEME["text"], 0.1)};
    }}

    div[data-testid="stMetric"] *, div[data-testid="metric-container"] * {{
        color: {THEME["text"]} !important;
        font-family: 'Space Grotesk', sans-serif;
    }}

    /* Bordered "card" containers used to group panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 12px !important;
        border-color: {hex_to_rgba(THEME["text"], 0.12)} !important;
    }}

    .stButton > button {{
        background-color: {THEME["button"]} !important;
        color: {THEME["text_secondary"]} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        background-color: {hex_to_rgba(THEME["button"], 0.85)} !important;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }}

    .status-pill .dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }}

    .status-live {{
        background: {hex_to_rgba(THEME["non_toxic"], 0.12)};
        color: {THEME["non_toxic"]};
        border: 1px solid {hex_to_rgba(THEME["non_toxic"], 0.4)};
    }}
    .status-live .dot {{
        background: {THEME["non_toxic"]};
        animation: pulse 1.4s infinite;
    }}

    .status-finished {{
        background: {hex_to_rgba(THEME["info"], 0.12)};
        color: {THEME["info"]};
        border: 1px solid {hex_to_rgba(THEME["info"], 0.4)};
    }}
    .status-finished .dot {{ background: {THEME["info"]}; }}

    .status-stalled {{
        background: {hex_to_rgba(THEME["toxic"], 0.12)};
        color: {THEME["toxic"]};
        border: 1px solid {hex_to_rgba(THEME["toxic"], 0.4)};
    }}
    .status-stalled .dot {{ background: {THEME["toxic"]}; }}

    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 {hex_to_rgba(THEME["non_toxic"], 0.5)}; }}
        70% {{ box-shadow: 0 0 0 8px {hex_to_rgba(THEME["non_toxic"], 0)}; }}
        100% {{ box-shadow: 0 0 0 0 {hex_to_rgba(THEME["non_toxic"], 0)}; }}
    }}

    .stat-label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.6;
        margin-bottom: 2px;
    }}
    .stat-value {{
        font-size: 1.35rem;
        font-weight: 600;
        color: {THEME["text"]};
    }}

    .health-ok {{ color: {THEME["non_toxic"]}; font-weight: 600; }}
    .health-warn {{ color: {THEME["toxic"]}; font-weight: 600; }}
</style>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@dataclass
class RunData:
    tags: dict[str, pd.DataFrame]
    error: str | None
    latest_mtime: float
    total_size: int


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCALAR_COLUMNS)


def _find_event_files(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    return sorted(run_dir.glob("events.out.tfevents.*"))


def _event_file_stat(run_dir: Path) -> tuple[float, int, int]:
    """(latest_mtime, total_size, file_count) across a run's event file(s).

    The value is only used to build the st.cache_data key -- it makes the
    cache invalidate exactly when the on-disk data changes, and stay cheap
    otherwise, even while a run is being actively appended to.
    """
    files = _find_event_files(run_dir)
    if not files:
        return (0.0, 0, 0)
    mtime = 0.0
    size = 0
    for f in files:
        try:
            stat = f.stat()
        except OSError:
            continue
        mtime = max(mtime, stat.st_mtime)
        size += stat.st_size
    return (mtime, size, len(files))


@st.cache_data(ttl=5, show_spinner=False)
def _load_run_payload(run_dir_str: str, cache_bust: tuple[float, int, int]) -> dict:
    """Return a plain picklable payload.

    st.cache_data pickles what it caches, and a dataclass defined in a script
    run as __main__ cannot be pickled by reference -- so the cached layer
    returns dicts and DataFrames, and RunData is rebuilt outside the cache.
    """
    size_guidance = {
        "scalars": 0,  # 0 = unbounded; do not reservoir-sample scalars down to 10k
        "tensors": 0,
        "histograms": 1,
        "images": 1,
        "audio": 1,
    }
    tags: dict[str, pd.DataFrame] = {}
    error = None
    try:
        acc = EventAccumulator(run_dir_str, size_guidance=size_guidance)
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            tags[tag] = pd.DataFrame(
                {
                    "step": [e.step for e in events],
                    "wall_time": [e.wall_time for e in events],
                    "value": [e.value for e in events],
                }
            )
    except Exception as exc:  # event files can be mid-write; never crash the page
        error = f"{type(exc).__name__}: {exc}"

    mtime, size, _n_files = cache_bust
    return {"tags": tags, "error": error, "latest_mtime": mtime, "total_size": size}


def load_run_data(run_dir_str: str, cache_bust: tuple[float, int, int]) -> RunData:
    payload = _load_run_payload(run_dir_str, cache_bust)
    return RunData(
        tags=payload["tags"],
        error=payload["error"],
        latest_mtime=payload["latest_mtime"],
        total_size=payload["total_size"],
    )


def get_df(run: RunData, tag: str) -> pd.DataFrame:
    return run.tags.get(tag, _empty_df())


def list_run_dirs(runs_root: Path) -> list[str]:
    if not runs_root.is_dir():
        return []
    dirs = [d for d in runs_root.iterdir() if d.is_dir()]

    def sort_key(d: Path) -> float:
        mtime, _, _ = _event_file_stat(d)
        if mtime > 0:
            return mtime
        try:
            return d.stat().st_mtime
        except OSError:
            return 0.0

    dirs.sort(key=sort_key, reverse=True)
    return [d.name for d in dirs]


# --------------------------------------------------------------------------
# Derived metrics / small numerics
# --------------------------------------------------------------------------
def tb_smooth(values: np.ndarray, weight: float) -> np.ndarray:
    """Debiased EMA smoothing, matching TensorBoard's own scalar-chart smoothing."""
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    last = 0.0
    num_accum = 0
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = out[i - 1] if i > 0 else 0.0
            continue
        num_accum += 1
        last = last * weight + (1 - weight) * v
        debias = 1 - weight**num_accum
        out[i] = last / debias if debias > 0 else v
    return out


def map_epoch_to_step(epoch_df: pd.DataFrame, step_df: pd.DataFrame) -> pd.DataFrame:
    """Attach an estimated global `step` to an epoch-indexed dataframe.

    Epoch-level scalars are logged with step=epoch_number, not the global
    training step, so they can't be placed directly on a per-step x-axis.
    Both series are written at (about) the same instant at epoch end, so
    matching on the nearest-preceding wall_time gives a reliable estimate of
    "the global step this epoch finished on".
    """
    out = epoch_df.copy()
    if out.empty or step_df.empty:
        out["global_step"] = pd.Series(dtype="float64")
        return out
    left = out.sort_values("wall_time")
    right = (
        step_df[["wall_time", "step"]]
        .rename(columns={"step": "global_step"})
        .sort_values("wall_time")
    )
    merged = pd.merge_asof(left, right, on="wall_time", direction="backward")
    merged["global_step"] = merged["global_step"].bfill()
    return merged.sort_values("step")


def check_lr_schedule(lr_df: pd.DataFrame) -> tuple[bool, str]:
    if len(lr_df) < 5:
        return True, "Collecting data..."
    vals = lr_df["value"].to_numpy()
    peak_idx = int(np.argmax(vals))
    tol = 1e-12
    if peak_idx == 0:
        ok = bool(np.all(np.diff(vals) <= tol))
        msg = "Already decaying; monotonic so far." if ok else "LR is not monotonically decreasing -- check the scheduler."
        return ok, msg
    if peak_idx == len(vals) - 1:
        ok = bool(np.all(np.diff(vals) >= -tol))
        msg = "Still warming up; increasing as expected." if ok else "LR is not monotonically increasing during warmup -- check the scheduler."
        return ok, msg
    warmup_ok = bool(np.all(np.diff(vals[: peak_idx + 1]) >= -tol))
    decay_ok = bool(np.all(np.diff(vals[peak_idx:]) <= tol))
    ok = warmup_ok and decay_ok
    msg = (
        "Looks like linear warmup then decay."
        if ok
        else "LR shape does not look like warmup -> cosine decay -- check the scheduler config."
    )
    return ok, msg


def check_class_weights(min_df: pd.DataFrame, max_df: pd.DataFrame) -> tuple[bool | None, str]:
    if min_df.empty or max_df.empty:
        return None, "No class-weight data yet."
    spread = float(max_df["value"].iloc[-1] - min_df["value"].iloc[-1])
    if spread < 1e-6:
        return False, "min == max -- weights look uniform. Check the class-weighting logic."
    return True, f"Weights are non-uniform (latest spread {spread:.3f}) -- looks healthy."


def compute_status(latest_mtime: float, completed_epochs: int, total_epochs: int) -> str:
    if latest_mtime <= 0:
        return "NO DATA"
    age = time.time() - latest_mtime
    if age < LIVE_THRESHOLD_SEC:
        return "LIVE"
    if total_epochs > 0 and completed_epochs >= total_epochs:
        return "FINISHED"
    return "STALLED"


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_eta_clock(seconds_from_now: float) -> str:
    return (datetime.now() + timedelta(seconds=seconds_from_now)).strftime("%H:%M")


def format_lr(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.3e}"


def apply_layout(
    fig: go.Figure,
    *,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 380,
    yaxis_type: str | None = None,
) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=THEME["background"],
        paper_bgcolor=THEME["background"],
        font=dict(family="Space Grotesk, sans-serif", color=THEME["text"]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=xaxis_title, gridcolor=hex_to_rgba(THEME["text"], 0.08)),
        yaxis=dict(title=yaxis_title, gridcolor=hex_to_rgba(THEME["text"], 0.08), type=yaxis_type),
    )


def status_badge_html(status: str) -> str:
    mapping = {
        "LIVE": "status-live",
        "FINISHED": "status-finished",
        "STALLED": "status-stalled",
        "NO DATA": "status-stalled",
    }
    css_class = mapping.get(status, "status-stalled")
    return f'<div class="status-pill {css_class}"><span class="dot"></span>{status}</div>'


def stat_block_html(label: str, value: str) -> str:
    return f'<div class="stat-label">{label}</div><div class="stat-value">{value}</div>'


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def render_status_strip(run: RunData, run_name: str, total_epochs: int) -> None:
    epoch_ref = get_df(run, "epoch/val_auc_macro")
    if epoch_ref.empty:
        epoch_ref = get_df(run, "epoch/train_loss")
    completed_epochs = len(epoch_ref)
    status = compute_status(run.latest_mtime, completed_epochs, total_epochs)

    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div style="font-size:1.4rem; font-weight:700;">{run_name}</div>'
            f"{status_badge_html(status)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        display_epoch = total_epochs if status == "FINISHED" else min(completed_epochs + 1, total_epochs)
        progress = min(completed_epochs / total_epochs, 1.0) if total_epochs else 0.0

        wall_series = [
            get_df(run, t)["wall_time"]
            for t in ("train/loss", "epoch/val_auc_macro", "epoch/train_loss")
            if not get_df(run, t).empty
        ]
        if wall_series:
            start = min(s.min() for s in wall_series)
            end = time.time() if status == "LIVE" else max(s.max() for s in wall_series)
            elapsed = end - start
        else:
            elapsed = None

        epoch_time_df = get_df(run, "epoch/time_sec")
        if not epoch_time_df.empty and total_epochs > completed_epochs:
            avg_epoch_time = float(epoch_time_df["value"].mean())
            remaining = total_epochs - completed_epochs
            eta_seconds = remaining * avg_epoch_time
            if status == "LIVE":
                eta_text = f"{format_duration(eta_seconds)} (~{format_eta_clock(eta_seconds)})"
            else:
                eta_text = format_duration(eta_seconds)
        elif not epoch_time_df.empty:
            eta_text = "complete"
        else:
            eta_text = "estimating..."

        age = None if run.latest_mtime <= 0 else time.time() - run.latest_mtime
        last_seen = "--" if age is None else (f"{int(age)}s ago" if age < 120 else f"{format_duration(age)} ago")

        cols = st.columns(4)
        with cols[0]:
            st.markdown(stat_block_html("EPOCH", f"{display_epoch} / {total_epochs}"), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(stat_block_html("ELAPSED", format_duration(elapsed)), unsafe_allow_html=True)
        with cols[2]:
            st.markdown(stat_block_html("ETA", eta_text), unsafe_allow_html=True)
        with cols[3]:
            st.markdown(stat_block_html("LAST EVENT", last_seen), unsafe_allow_html=True)

        st.progress(progress)


def render_headline_metrics(run: RunData, batch_size: int) -> None:
    with st.container(border=True):
        cols = st.columns(5)

        loss_df = get_df(run, "train/loss")
        with cols[0]:
            if loss_df.empty:
                st.metric("Train loss", "--")
            else:
                current = float(loss_df["value"].iloc[-1])
                delta = float(current - loss_df["value"].iloc[-2]) if len(loss_df) > 1 else None
                st.metric(
                    "Train loss",
                    f"{current:.4f}",
                    delta=None if delta is None else f"{delta:+.4f}",
                    delta_color="inverse",
                )

        val_loss_df = get_df(run, "epoch/val_loss")
        with cols[1]:
            if val_loss_df.empty:
                st.metric("Val loss", "--", help="Logged once per epoch.")
            else:
                current = float(val_loss_df["value"].iloc[-1])
                delta = float(current - val_loss_df["value"].iloc[-2]) if len(val_loss_df) > 1 else None
                st.metric(
                    "Val loss",
                    f"{current:.4f}",
                    delta=None if delta is None else f"{delta:+.4f}",
                    delta_color="inverse",
                    help="Logged once per epoch.",
                )

        auc_df = get_df(run, "epoch/val_auc_macro")
        with cols[2]:
            if auc_df.empty:
                st.metric("Best macro val AUC", "--")
            else:
                best_idx = auc_df["value"].idxmax()
                best_val = float(auc_df.loc[best_idx, "value"])
                best_epoch = int(auc_df.loc[best_idx, "step"])
                st.metric("Best macro val AUC", f"{best_val:.4f}", delta=f"epoch {best_epoch}", delta_color="off")

        lr_df = get_df(run, "train/lr")
        if lr_df.empty:
            lr_df = get_df(run, "epoch/lr")
        with cols[3]:
            st.metric("Learning rate", "--" if lr_df.empty else format_lr(float(lr_df["value"].iloc[-1])))

        batch_time_df = get_df(run, "train/batch_time")
        with cols[4]:
            finite = (
                batch_time_df[np.isfinite(batch_time_df["value"]) & (batch_time_df["value"] > 0)]
                if not batch_time_df.empty
                else batch_time_df
            )
            if finite.empty:
                st.metric("Throughput", "--", help=f"batch_size={batch_size}")
            else:
                recent = finite["value"].tail(50)
                recent_tp = batch_size / recent.mean()
                overall_tp = batch_size / finite["value"].mean()
                delta = recent_tp - overall_tp
                st.metric(
                    "Throughput",
                    f"{recent_tp:.1f}/s",
                    delta=f"{delta:+.1f}/s" if abs(delta) > 0.05 else None,
                    delta_color="normal",
                    help=f"batch_size={batch_size}; recent = last {len(recent)} steps vs run average",
                )


def render_loss_panel(run: RunData, smoothing_on: bool, smoothing_weight: float) -> None:
    with st.container(border=True):
        st.markdown("#### Loss")
        loss_df = get_df(run, "train/loss")
        if loss_df.empty:
            st.info("No per-step training loss logged yet.")
            return
        loss_df = loss_df.sort_values("step")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=loss_df["step"],
                y=loss_df["value"],
                mode="lines",
                line=dict(color=THEME["primary"], width=1),
                opacity=0.25,
                name="train/loss (raw)",
            )
        )
        if smoothing_on:
            smoothed = tb_smooth(loss_df["value"].to_numpy(), smoothing_weight)
            fig.add_trace(
                go.Scatter(
                    x=loss_df["step"],
                    y=smoothed,
                    mode="lines",
                    line=dict(color=THEME["primary"], width=2.5),
                    name=f"train/loss (smoothed {smoothing_weight:.2f})",
                )
            )

        epoch_train_df = get_df(run, "epoch/train_loss")
        if not epoch_train_df.empty:
            mapped = map_epoch_to_step(epoch_train_df, loss_df)
            fig.add_trace(
                go.Scatter(
                    x=mapped["global_step"],
                    y=mapped["value"],
                    mode="lines+markers",
                    line=dict(color=THEME["info"], width=2, dash="dot"),
                    marker=dict(size=8, symbol="diamond"),
                    name="epoch/train_loss",
                )
            )

        epoch_val_df = get_df(run, "epoch/val_loss")
        if not epoch_val_df.empty:
            mapped = map_epoch_to_step(epoch_val_df, loss_df)
            fig.add_trace(
                go.Scatter(
                    x=mapped["global_step"],
                    y=mapped["value"],
                    mode="lines+markers",
                    line=dict(color=THEME["toxic"], width=2, dash="dot"),
                    marker=dict(size=8, symbol="diamond"),
                    name="epoch/val_loss",
                )
            )

        apply_layout(fig, xaxis_title="step", yaxis_title="loss")
        st.plotly_chart(fig, use_container_width=True)
        if epoch_val_df.empty:
            st.caption("Train vs val overlay appears once the first epoch finishes.")


def render_auc_panel(run: RunData) -> None:
    with st.container(border=True):
        st.markdown("#### Per-class validation AUC")
        palette = {
            "toxic": THEME["info"],
            "obscene": THEME["primary"],
            "insult": THEME["non_toxic"],
            "severe_toxic": THEME["toxic"],
            "threat": THEME["warning"],
            "identity_hate": RARE_CLASS_COLOR,
        }
        fig = go.Figure()
        missing = []
        latest_rows = []
        for name, tag, is_rare in CLASS_TAGS:
            df = get_df(run, tag)
            if df.empty:
                missing.append(name)
                continue
            df = df.sort_values("step")
            fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["value"],
                    mode="lines+markers",
                    name=name.replace("_", " "),
                    line=dict(
                        color=palette[name],
                        width=2.5 if is_rare else 2,
                        dash="dash" if is_rare else "solid",
                    ),
                    marker=dict(size=6),
                )
            )
            latest = df.iloc[-1]
            prev_val = float(df.iloc[-2]["value"]) if len(df) > 1 else None
            latest_rows.append(
                {
                    "Class": name.replace("_", " "),
                    "Epoch": int(latest["step"]),
                    "Latest AUC": float(latest["value"]),
                    "Delta vs prev epoch": None if prev_val is None else float(latest["value"]) - prev_val,
                }
            )

        if not latest_rows:
            st.info("No per-class validation AUC logged yet -- wait for the first epoch to complete.")
            return

        apply_layout(fig, xaxis_title="epoch", yaxis_title="AUC")
        st.plotly_chart(fig, use_container_width=True)
        if missing:
            st.caption(f"No data yet for: {', '.join(missing)}.")

        table_df = pd.DataFrame(latest_rows)

        def _delta_style(value: float | None) -> str:
            if value is None or not np.isfinite(value):
                return ""
            if value > 0:
                return f"color: {THEME['non_toxic']}; font-weight: 600;"
            if value < 0:
                return f"color: {THEME['toxic']}; font-weight: 600;"
            return ""

        try:
            styled = table_df.style.format(
                {"Latest AUC": "{:.4f}", "Delta vs prev epoch": "{:+.4f}"}, na_rep="--"
            ).map(_delta_style, subset=["Delta vs prev epoch"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(table_df, use_container_width=True, hide_index=True)


def render_health_panel(run: RunData) -> None:
    with st.container(border=True):
        st.markdown("#### Training health")
        col_lr, col_grad, col_gpu = st.columns(3)

        with col_lr:
            st.markdown("**Learning rate**")
            lr_df = get_df(run, "train/lr")
            if lr_df.empty:
                st.info("No LR data yet.")
            else:
                lr_df = lr_df.sort_values("step")
                ok, msg = check_lr_schedule(lr_df)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=lr_df["step"], y=lr_df["value"], mode="lines", line=dict(color=THEME["info"], width=2))
                )
                apply_layout(fig, xaxis_title="step", yaxis_title="lr", height=280)
                st.plotly_chart(fig, use_container_width=True)
                css_class = "health-ok" if ok else "health-warn"
                st.markdown(f'<span class="{css_class}">{msg}</span>', unsafe_allow_html=True)

        with col_grad:
            st.markdown("**Grad norm**")
            grad_df = get_df(run, "train/grad_norm")
            if grad_df.empty:
                st.info("No grad-norm data yet.")
            else:
                grad_df = grad_df.sort_values("step")
                finite = grad_df[np.isfinite(grad_df["value"]) & (grad_df["value"] > 0)]
                n_bad = len(grad_df) - len(finite)
                if finite.empty:
                    st.warning(
                        f"All {len(grad_df)} grad-norm readings so far are inf/nan "
                        "(fp16 GradScaler overflow). Nothing finite to plot yet."
                    )
                else:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=finite["step"], y=finite["value"], mode="lines", line=dict(color=THEME["primary"], width=1.5)
                        )
                    )
                    apply_layout(fig, xaxis_title="step", yaxis_title="grad norm (log)", height=280, yaxis_type="log")
                    st.plotly_chart(fig, use_container_width=True)
                    if n_bad:
                        st.caption(f"{n_bad} of {len(grad_df)} steps had inf/nan grad norm (fp16 overflow) -- excluded above.")
                    else:
                        st.caption(f"0 of {len(grad_df)} steps had inf/nan grad norm.")

        with col_gpu:
            st.markdown("**GPU memory**")
            gpu_df = get_df(run, "train/gpu_memory_mb")
            if gpu_df.empty:
                st.info("No GPU memory data yet.")
            else:
                gpu_df = gpu_df.sort_values("step")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=gpu_df["step"], y=gpu_df["value"], mode="lines", line=dict(color=THEME["warning"], width=1.5))
                )
                apply_layout(fig, xaxis_title="step", yaxis_title="MB", height=280)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Latest: {gpu_df['value'].iloc[-1]:.0f} MB -- peak: {gpu_df['value'].max():.0f} MB")


def render_class_weight_panel(run: RunData) -> None:
    with st.container(border=True):
        st.markdown("#### Class weights (sanity check)")
        min_df = get_df(run, "train/class_weight_min")
        max_df = get_df(run, "train/class_weight_max")
        mean_df = get_df(run, "train/class_weight_mean")
        if min_df.empty or max_df.empty or mean_df.empty:
            st.info("No class-weight data yet.")
            return

        min_df = min_df.sort_values("step")
        max_df = max_df.sort_values("step")
        mean_df = mean_df.sort_values("step")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=max_df["step"], y=max_df["value"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=min_df["step"],
                y=min_df["value"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=hex_to_rgba(THEME["primary"], 0.12),
                name="min-max range",
            )
        )
        fig.add_trace(
            go.Scatter(x=mean_df["step"], y=mean_df["value"], mode="lines+markers", line=dict(color=THEME["primary"], width=2), name="mean")
        )
        apply_layout(fig, xaxis_title="step", yaxis_title="class weight", height=300)
        st.plotly_chart(fig, use_container_width=True)

        ok, msg = check_class_weights(min_df, max_df)
        if ok is None:
            st.info(msg)
        else:
            css_class = "health-ok" if ok else "health-warn"
            st.markdown(f'<span class="{css_class}">{msg}</span>', unsafe_allow_html=True)


def render_empty_state(runs_root: Path) -> None:
    with st.container(border=True):
        st.markdown("### No training runs found")
        st.write(
            f"Looked in `{runs_root}/` and found no run directories. "
            "Start a run (for example with `scripts/train_tmux.sh`) and this "
            "page will pick it up the next time you interact with a control "
            "or click Refresh now."
        )


def render_waiting_state(run_name: str, run_dir: Path) -> None:
    with st.container(border=True):
        st.markdown(f"### {run_name}")
        st.write(
            "This run directory exists but no scalar data could be read from "
            "its event file yet. This is normal for the first few seconds "
            "after training starts, or while the first epoch is still "
            "running -- the panels below will populate as steps are logged."
        )
        st.caption(f"Watching: `{run_dir}`")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def render_dashboard_body(
    runs_root: Path,
    run_name: str,
    smoothing_on: bool,
    smoothing_weight: float,
    batch_size: int,
    total_epochs: int,
) -> None:
    run_dir = runs_root / run_name
    cache_bust = _event_file_stat(run_dir)
    run = load_run_data(str(run_dir), cache_bust)

    st.caption(f"Checked {datetime.now().strftime('%H:%M:%S')}")

    if run.error:
        st.error(f"Could not read event file(s) for this run: {run.error}")
        return

    if not run.tags:
        render_waiting_state(run_name, run_dir)
        return

    render_status_strip(run, run_name, total_epochs)
    render_headline_metrics(run, batch_size)
    render_loss_panel(run, smoothing_on, smoothing_weight)
    render_auc_panel(run)
    render_health_panel(run)
    render_class_weight_panel(run)


def build_dashboard_fragment(run_every: int | None):
    """Return a fragment whose identity is stable across reruns.

    Decorating a fresh closure on every script run gives the fragment a new id
    each time, so the previous fragment's auto-refresh timer keeps firing
    against an id that no longer exists ("fragment ... does not exist anymore")
    and the page never actually refreshes. Memoizing per interval in session
    state keeps one fragment object alive for the whole session.
    """
    key = f"_dashboard_fragment_{run_every}"
    if key not in st.session_state:

        @st.fragment(run_every=run_every)
        def _render(**kwargs) -> None:
            render_dashboard_body(**kwargs)

        st.session_state[key] = _render
    return st.session_state[key]


def main() -> None:
    inject_css()
    st.markdown('<div class="main-title">Training Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">XLM-RoBERTa toxicity classifier -- live view over TensorBoard event files</div>',
        unsafe_allow_html=True,
    )

    runs_root = Path(os.environ.get("MONITOR_RUNS_DIR", "runs"))

    with st.sidebar:
        st.markdown('<div style="font-size:1.3rem; font-weight:700;">Controls</div>', unsafe_allow_html=True)
        run_names = list_run_dirs(runs_root)
        if run_names:
            selected_run = st.selectbox("Run", run_names, index=0, help="Directories under runs/, newest first.")
        else:
            selected_run = None
            st.warning(f"No runs found in `{runs_root}/`.")

        st.divider()
        auto_refresh = st.toggle("Auto-refresh", value=True)
        refresh_interval = st.number_input(
            "Refresh interval (sec)", min_value=3, max_value=300, value=15, step=1, disabled=not auto_refresh
        )
        if st.button("Refresh now", use_container_width=True):
            st.toast("Refreshed")

        st.divider()
        smoothing_on = st.checkbox("Smooth loss curve", value=True)
        smoothing_weight = st.slider(
            "Smoothing factor", min_value=0.0, max_value=0.99, value=0.6, step=0.01, disabled=not smoothing_on
        )

        st.divider()
        batch_size = st.number_input(
            "Batch size", min_value=1, value=128, step=1, help="Used to convert train/batch_time into samples/sec."
        )
        total_epochs = st.number_input("Total epochs (planned)", min_value=1, value=6, step=1)

        st.divider()
        st.caption(f"Runs directory: `{runs_root}`")

    if not run_names or selected_run is None:
        render_empty_state(runs_root)
        return

    dashboard = build_dashboard_fragment(int(refresh_interval) if auto_refresh else None)
    dashboard(
        runs_root=runs_root,
        run_name=selected_run,
        smoothing_on=smoothing_on,
        smoothing_weight=smoothing_weight,
        batch_size=int(batch_size),
        total_epochs=int(total_epochs),
    )


main()
