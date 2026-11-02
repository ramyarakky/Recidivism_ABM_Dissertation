"""
sensitivity_oat.py  —  One-At-a-Time (OAT) Sensitivity Analysis
================================================================
Recidivism ABM — PhD Dissertation Tool

CHANGES FROM PREVIOUS VERSION
------------------------------
  1. BASELINE_CAL updated to post-bug-fix recalibration values:
       Risk_Effect_Decay_After_3Y            0.90  (was 1.0)
       Risk_Effect_Decay_After_6Y            0.70  (was 1.0)
       Supervision_Monitoring_Decay_After_3Y 0.35  (was 1.0)
       Supervision_Monitoring_Decay_After_6Y 0.20  (was 1.0)

  2. PARAM_DEFS baselines for RED3, RED6, SMD3, SMD6 updated to match.

  3. Tornado charts revised — dots removed, pill labels added:
       - ax.scatter() dot calls removed entirely
       - Plain yticklabels replaced with coloured rounded-rectangle pills
         drawn via ax.text() + bbox, using get_yaxis_transform() so
         x is in axes-fraction space and y is in data space
       - fig.subplots_adjust(left=0.32) gives pills space outside the axes
       - Legend split: bar-direction (lower right) + group key (upper right)
         via _tornado_legends() helper — no more mixed patch legend
       - Same treatment applied to the tornado panel in plot_full_report()
         with gs left=0.18 to accommodate pills in the composite figure

PURPOSE
-------
Tests how sensitive the ABM's 3/6/9-year cumulative rearrest rates are
to perturbations of each parameter individually, holding all others fixed
at their calibrated (baseline) values.

PERTURBATION DESIGN
-------------------
  9 levels per parameter:  -40%, -30%, -20%, -10%, 0% (baseline),
                           +10%, +20%, +30%, +40%
  For each level: N_REPS x N_SEEDS = 100 runs
  Total simulations: ~9,900  (11 params x 9 levels x 100 runs)

CHARTS PRODUCED
---------------
  1. oat_tornado_3yr.png
  2. oat_tornado_6yr.png
  3. oat_tornado_9yr.png
  4. oat_lines.png
  5. oat_heatmap.png
  6. oat_group_summary.png
  7. oat_full_report.png

USAGE
-----
  python sensitivity_oat.py
  python sensitivity_oat.py --cores 8
  python sensitivity_oat.py --replot
  python sensitivity_oat.py --rerun
"""

import os
import sys
import json
import warnings
import multiprocessing
warnings.filterwarnings("ignore")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter, FuncFormatter
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats as scipy_stats
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_global_calibration_params,
)

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "initial_agents":        1000,
    "monthly_intake":        10,
    "warmup_months":         144,
    "study_months":          108,
    "bias_factor":           0.0,
    "enable_peer_influence": True,
    "output_directory":      "oat_sensitivity_output",
    "N_REPS":                10,
    "SEEDS":                 [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "bjs_targets":           {3: 0.68, 6: 0.79, 9: 0.83},
}

# ── Confirmed baseline calibration parameters (post bug-fix recalibration) ────
# Decay bug: original code compounded both decay multipliers for agents past
# year 6 (two if-blocks both fired). Fixed to elif. All four decay params
# re-swept on corrected code. See risk_config.py for full calibration notes.
BASELINE_CAL = {
    "Risk_Contrast_Strength":                0.00,   # unidentified — boundary result
    "Supervision_Monitoring_Intensity":      0.35,   # identified — U-shaped minimum
    "Risk_Effect_Decay_After_3Y":            0.90,   # identified — interior minimum
    "Risk_Effect_Decay_After_6Y":            0.70,   # identified — interior minimum
    "Supervision_Monitoring_Decay_After_3Y": 0.35,   # boundary result
    "Supervision_Monitoring_Decay_After_6Y": 0.20,   # boundary result
}

# ── Perturbation levels ───────────────────────────────────────────────────────
PERTURBATIONS  = [-0.40, -0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30, +0.40]
PERTURB_LABELS = ["-40%", "-30%", "-20%", "-10%", "Baseline",
                  "+10%", "+20%", "+30%", "+40%"]

# ── Parameter group definitions ───────────────────────────────────────────────
PARAM_DEFS = [
    # ── Calibration parameters ──────────────────────────────────────────────
    {
        "key":         "Supervision_Monitoring_Intensity",
        "label":       "Supervision\nMonitoring Intensity",
        "short":       "SMI",
        "group":       "Calibration",
        "baseline":    0.35,
        "mode":        "multiply",
        "group_color": "#2E75B6",
    },
    {
        "key":         "Risk_Contrast_Strength",
        "label":       "Risk Contrast\nStrength",
        "short":       "RCS",
        "group":       "Calibration",
        "baseline":    0.0,
        "mode":        "add_fixed",
        "step":        0.1,
        "group_color": "#2E75B6",
    },
    {
        "key":         "Risk_Effect_Decay_After_3Y",
        "label":       "Risk Effect\nDecay After 3Y",
        "short":       "RED3",
        "group":       "Calibration",
        "baseline":    0.90,        # updated from 1.0
        "mode":        "multiply",
        "group_color": "#2E75B6",
    },
    {
        "key":         "Risk_Effect_Decay_After_6Y",
        "label":       "Risk Effect\nDecay After 6Y",
        "short":       "RED6",
        "group":       "Calibration",
        "baseline":    0.70,        # updated from 1.0
        "mode":        "multiply",
        "group_color": "#2E75B6",
    },
    {
        "key":         "Supervision_Monitoring_Decay_After_3Y",
        "label":       "Supervision Decay\nAfter 3Y",
        "short":       "SMD3",
        "group":       "Calibration",
        "baseline":    0.35,        # updated from 1.0
        "mode":        "multiply",
        "group_color": "#2E75B6",
    },
    {
        "key":         "Supervision_Monitoring_Decay_After_6Y",
        "label":       "Supervision Decay\nAfter 6Y",
        "short":       "SMD6",
        "group":       "Calibration",
        "baseline":    0.20,        # updated from 1.0
        "mode":        "multiply",
        "group_color": "#2E75B6",
    },
    # ── Risk weight parameters ───────────────────────────────────────────────
    {
        "key":         "Age_at_Release",
        "label":       "Age at\nRelease Weight",
        "short":       "Age",
        "group":       "Risk Weight",
        "baseline":    0.73,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
    },
    {
        "key":         "Prior_Revocations_Supervision",
        "label":       "Prior Revocations\nSupervision Weight",
        "short":       "Revocation",
        "group":       "Risk Weight",
        "baseline":    0.4338,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
    },
    {
        "key":         "Gang_Affiliated",
        "label":       "Gang Affiliated\nWeight",
        "short":       "Gang",
        "group":       "Risk Weight",
        "baseline":    0.84,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
    },
    {
        "key":         "Percent_Days_Employed",
        "label":       "Percent Days\nEmployed Weight",
        "short":       "Employment",
        "group":       "Risk Weight",
        "baseline":    -0.95,       # negative: employment reduces rearrest risk
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
    },
    {
        "key":         "Program_Attendances",
        "label":       "Program\nAttendances Weight",
        "short":       "Program",
        "group":       "Risk Weight",
        "baseline":    -0.38,       # negative: attendance reduces rearrest risk
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
    },
]

# ── Colours ───────────────────────────────────────────────────────────────────
COLORS = {
    "bjs":        "#1A3D5C",
    "baseline":   "#555555",
    "up":         "#C00000",
    "down":       "#2E75B6",
    "grid":       "#E8E8E8",
    "cal":        "#2E75B6",
    "structural": "#70AD47",
    "risk":       "#ED7D31",
    "3yr":        "#2E75B6",
    "6yr":        "#70AD47",
    "9yr":        "#ED7D31",
}

GROUP_COLORS = {
    "Calibration": "#2E75B6",
    "Risk Weight": "#ED7D31",
}


# =============================================================================
# PERTURBATION VALUE CALCULATOR
# =============================================================================

def get_perturbed_values(pdef: dict) -> list:
    mode = pdef["mode"]
    base = pdef["baseline"]
    vals = []
    for pct in PERTURBATIONS:
        if mode == "multiply":
            v = base * (1.0 + pct)
            v = max(0.0, min(1.0, v))
        elif mode == "add_fixed":
            step  = pdef.get("step", 0.1)
            level = round(pct / 0.10)
            v     = base + level * step
            v     = max(0.0, v)
        elif mode in ("structural", "risk_weight"):
            v = base * (1.0 + pct)
            if mode == "structural" and pdef["key"] in (
                    "initial_agents", "warmup_months", "monthly_intake"):
                v = max(1, int(round(v)))
        vals.append(v)
    return vals


# =============================================================================
# WORKER
# =============================================================================

def _worker(task: tuple) -> tuple:
    (payload_json, seed) = task
    payload  = json.loads(payload_json)
    task_id  = payload.pop("__task_id__", 0.0)
    warmup   = int(payload.pop("__warmup__",  CONFIG["warmup_months"]))
    study    = int(payload.pop("__study__",   CONFIG["study_months"]))
    agents   = int(payload.pop("__agents__",  CONFIG["initial_agents"]))
    intake   = int(payload.pop("__intake__",  CONFIG["monthly_intake"]))
    rw_ovr   = payload.pop("__risk_weights__", {})

    cal = get_global_calibration_params()
    cal.update(BASELINE_CAL)
    for k, v in payload.items():
        if not k.startswith("__"):
            cal[k] = v

    rw = get_flat_risk_weights()
    for k, v in rw_ovr.items():
        if k in rw:
            rw[k] = v

    try:
        model = RecidivismModel(
            initial_agents=agents,
            bias_factor=CONFIG["bias_factor"],
            monthly_intake=intake,
            warmup_months=warmup,
            study_months=study,
            enable_peer_influence=CONFIG["enable_peer_influence"],
            seed=seed,
        )
        model.calibration = cal
        if hasattr(model, "apply_calibration"):
            model.apply_calibration(model.calibration)

        for _ in range(warmup + study):
            if not model.running:
                break
            model.step()

        rates = {}
        for yrs in [3, 6, 9]:
            r = model.calculate_flag_rate(f"rearrest_{yrs}_yrs")
            rates[yrs] = r if r is not None else 0.0

        return (task_id, seed, rates[3], rates[6], rates[9])

    except Exception as e:
        print(f"  Worker error (tid={task_id:.4f}, seed={seed}): {e}", flush=True)
        return (task_id, seed, 0.0, 0.0, 0.0)


# =============================================================================
# RESULTS_RUN1 CLEANUP
# =============================================================================

_CLEANUP_EVERY_N = 50


def _cleanup_result_run1() -> int:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_script_dir)
    candidates  = [
        os.path.join(_parent_dir, "Results_Run1"),
        os.path.join(_script_dir, "Results_Run1"),
        os.path.join(os.getcwd(), "Results_Run1"),
    ]
    target = next((p for p in candidates if os.path.isdir(p)), None)
    if target is None:
        return 0
    try:
        all_files = os.listdir(target)
        if not all_files:
            return 0
        deleted = 0
        for fname in all_files:
            fpath = os.path.join(target, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    deleted += 1
            except OSError:
                pass
        print(f"  Cleaned {deleted} files from Results_Run1/", flush=True)
        return deleted
    except OSError:
        return 0


# =============================================================================
# PARALLEL BATCH RUNNER
# =============================================================================

def _run_batch(overrides_list: list, n_workers: int, desc: str = "Running") -> dict:
    seeds  = CONFIG["SEEDS"]
    n_reps = CONFIG["N_REPS"]

    tasks = []
    for task_id, payload in overrides_list:
        d = dict(payload)
        d["__task_id__"] = float(task_id)
        pj = json.dumps(d)
        for rep in range(n_reps):
            for seed in seeds:
                tasks.append((pj, seed + rep * 10_000))

    raw        = {float(tid): {3: [], 6: [], 9: []} for tid, _ in overrides_list}
    _completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc=f"  {desc}", ncols=82) as bar:
            for fut in as_completed(futures):
                try:
                    tid, _, r3, r6, r9 = fut.result()
                    key = min(raw.keys(), key=lambda k: abs(k - tid))
                    raw[key][3].append(r3)
                    raw[key][6].append(r6)
                    raw[key][9].append(r9)
                except Exception as e:
                    print(f"\n  Future error: {e}", flush=True)
                finally:
                    bar.update(1)
                    _completed += 1
                    if _completed % _CLEANUP_EVERY_N == 0:
                        _cleanup_result_run1()

    return raw


def _aggregate(bucket: dict) -> dict:
    out = {}
    for yrs, vals in bucket.items():
        arr  = np.array(vals) if vals else np.array([0.0])
        n    = len(arr)
        mean = float(arr.mean())
        std  = float(arr.std(ddof=min(1, n - 1)))
        sem  = std / np.sqrt(n) if n > 1 else std
        tc   = float(scipy_stats.t.ppf(0.975, df=max(1, n - 1)))
        out[yrs]           = mean
        out[f"std_{yrs}"]  = std
        out[f"ci95_{yrs}"] = tc * sem
        out[f"n_{yrs}"]    = n
    return out


# =============================================================================
# OAT SWEEP ENGINE
# =============================================================================

def run_oat_sweep(pdef: dict, n_workers: int) -> dict:
    mode   = pdef["mode"]
    key    = pdef["key"]
    values = get_perturbed_values(pdef)

    overrides_list = []
    for i, v in enumerate(values):
        payload = {}
        if mode in ("multiply", "add_fixed"):
            payload[key]          = float(v)
            payload["__warmup__"] = CONFIG["warmup_months"]
            payload["__study__"]  = CONFIG["study_months"]
            payload["__agents__"] = CONFIG["initial_agents"]
            payload["__intake__"] = CONFIG["monthly_intake"]
        elif mode == "structural":
            payload["__warmup__"] = int(v) if key == "warmup_months"  else CONFIG["warmup_months"]
            payload["__study__"]  = CONFIG["study_months"]
            payload["__agents__"] = int(v) if key == "initial_agents" else CONFIG["initial_agents"]
            payload["__intake__"] = int(v) if key == "monthly_intake" else CONFIG["monthly_intake"]
        elif mode == "risk_weight":
            payload["__warmup__"]       = CONFIG["warmup_months"]
            payload["__study__"]        = CONFIG["study_months"]
            payload["__agents__"]       = CONFIG["initial_agents"]
            payload["__intake__"]       = CONFIG["monthly_intake"]
            payload["__risk_weights__"] = {key: float(v)}
        overrides_list.append((float(i), payload))

    raw = _run_batch(overrides_list, n_workers, desc=f"{pdef['short']:6s} OAT")

    rows = []
    for i, v in enumerate(values):
        agg = _aggregate(raw[float(i)])
        rows.append({
            "level_idx": i,
            "pct":       PERTURBATIONS[i],
            "label":     PERTURB_LABELS[i],
            "value":     v,
            "mean_3yr":  agg[3],
            "mean_6yr":  agg[6],
            "mean_9yr":  agg[9],
            "ci95_3yr":  agg["ci95_3"],
            "ci95_6yr":  agg["ci95_6"],
            "ci95_9yr":  agg["ci95_9"],
        })

    df           = pd.DataFrame(rows)
    baseline_row = df[df["pct"] == 0.0].iloc[0]

    return {
        "key":      key,
        "label":    pdef["label"],
        "short":    pdef["short"],
        "group":    pdef["group"],
        "color":    pdef["group_color"],
        "df":       df,
        "baseline": {
            3: float(baseline_row["mean_3yr"]),
            6: float(baseline_row["mean_6yr"]),
            9: float(baseline_row["mean_9yr"]),
        },
    }


def sensitivity_magnitude(result: dict, window: int, pct: float = 0.40) -> float:
    df  = result["df"]
    pos = df[df["pct"] ==  pct][f"mean_{window}yr"].values
    neg = df[df["pct"] == -pct][f"mean_{window}yr"].values
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return abs(float(pos[0]) - float(neg[0])) / 2.0


# =============================================================================
# CHART HELPERS
# =============================================================================

def _style_ax(ax, xlabel="", ylabel="", title=""):
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _draw_pill_labels(ax, rows, fontsize=9):
    """
    Draw coloured rounded-rectangle pill labels for each tornado row.

    Replaces the plain ax.set_yticklabels() text and the ax.scatter() dots.
    Each pill background is filled with the parameter's group colour so group
    identity is encoded directly on the label rather than as a separate dot.

    Uses ax.get_yaxis_transform() so x is in axes-fraction space (allowing
    pills to sit outside the data area) and y remains in data-coordinate space.

    Must be called after barh() so that the axes y-limits are finalised.
    fig.subplots_adjust(left=0.32) should be set before calling this so the
    pills have room to the left of the axes boundary.
    """
    ax.set_yticklabels([""] * len(rows))   # blank default labels

    for i, row in enumerate(rows):
        ax.text(
            -0.003,               # axes-fraction x: just left of the zero-line
            i,                    # data-coordinate y: row centre
            row["full_label"],
            va="center",
            ha="right",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=row["color"],   # group colour fills the pill
                edgecolor="none",
                alpha=0.93,
            ),
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            zorder=6,
        )


def _tornado_legends(ax):
    """
    Add two non-overlapping legends to a tornado axes:
      lower right — perturbation direction (+40% red, -40% blue)
      upper right — parameter group colour key (Calibration / Risk Weight)

    Keeping them separate removes the need to decode a mixed-category legend.
    """
    bar_patches = [
        mpatches.Patch(color=COLORS["up"],   alpha=0.85, label="+40% perturbation"),
        mpatches.Patch(color=COLORS["down"], alpha=0.85, label="-40% perturbation"),
    ]
    group_patches = [
        mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
    ]

    # Draw bar-direction legend first; keep it with add_artist before adding second
    leg1 = ax.legend(
        handles=bar_patches,
        fontsize=8.5,
        loc="lower right",
        framealpha=0.9,
        title="Perturbation direction",
        title_fontsize=8,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=group_patches,
        fontsize=8.5,
        loc="upper right",
        framealpha=0.9,
        title="Parameter group",
        title_fontsize=8,
    )


# =============================================================================
# CHART 1 — Tornado Charts
# =============================================================================

def plot_tornado(all_results: list, window: int, outdir: str) -> str:
    """
    Horizontal bar tornado chart.

    Visual encoding:
      Bars      — red = positive change, blue = negative change
      Pill label — background colour encodes group (blue=Calibration,
                   amber=Risk Weight); white bold text on the pill
      Legends   — perturbation direction (lower right) + group (upper right)

    No dots are used. Group identity is read directly from the label pill.
    """
    targets = CONFIG["bjs_targets"]

    rows = []
    for r in all_results:
        df    = r["df"]
        pos40 = df[df["pct"] ==  0.40][f"mean_{window}yr"].values
        neg40 = df[df["pct"] == -0.40][f"mean_{window}yr"].values
        pos20 = df[df["pct"] ==  0.20][f"mean_{window}yr"].values
        neg20 = df[df["pct"] == -0.20][f"mean_{window}yr"].values
        base  = r["baseline"][window]
        if len(pos40) == 0 or len(neg40) == 0:
            continue
        delta_pos = float(pos40[0]) - base
        delta_neg = float(neg40[0]) - base
        mag40     = abs(delta_pos - delta_neg) / 2.0
        mag20     = (abs(float(pos20[0]) - float(neg20[0])) / 2.0
                     if len(pos20) > 0 and len(neg20) > 0 else 0.0)
        rows.append({
            "full_label": r["label"].replace("\n", " "),
            "group":      r["group"],
            "color":      r["color"],
            "delta_pos":  delta_pos,
            "delta_neg":  delta_neg,
            "mag40":      mag40,
            "mag20":      mag20,
        })

    rows.sort(key=lambda x: x["mag40"])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(14, max(5, n * 0.62 + 1.5)))
    fig.patch.set_facecolor("#FAFAFA")
    fig.subplots_adjust(left=0.32)   # space for pill labels outside axes

    ypos = np.arange(n)

    for i, row in enumerate(rows):
        c_pos = COLORS["up"]   if row["delta_pos"] >= 0 else COLORS["down"]
        c_neg = COLORS["down"] if row["delta_neg"] <= 0 else COLORS["up"]
        ax.barh(i, row["delta_pos"], left=0,
                color=c_pos, alpha=0.85, height=0.55, zorder=3)
        ax.barh(i, row["delta_neg"], left=0,
                color=c_neg, alpha=0.85, height=0.55, zorder=3)

        max_ext = max(abs(row["delta_pos"]), abs(row["delta_neg"]))
        ax.text(
            max_ext + 0.0005, i,
            f"+-40%:{row['mag40']:.3f}  +-20%:{row['mag20']:.3f}",
            va="center", fontsize=7.5, color="#444444",
        )

    ax.set_yticks(ypos)
    _draw_pill_labels(ax, rows, fontsize=9)   # coloured pills, no dots

    ax.axvline(0, color="#333333", linewidth=1.2, zorder=4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1%}"))
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _tornado_legends(ax)   # two separate legends

    ax.set_title(
        f"Tornado Chart — {window}-Year Rearrest Rate Sensitivity\n"
        f"Baseline = {targets[window]:.0%}  |  "
        f"Bars = change from +-40% perturbation  |  "
        f"Label colour = parameter group",
        fontsize=12, fontweight="bold", pad=10, color="#1A3D5C",
    )
    ax.set_xlabel("Change in Rearrest Rate from Baseline", fontsize=10)

    plt.tight_layout()
    path = os.path.join(outdir, f"oat_tornado_{window}yr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Tornado {window}yr -> {path}")
    return path


# =============================================================================
# CHART 2 — Line plots per parameter
# =============================================================================

def plot_lines(all_results: list, outdir: str) -> str:
    n_params   = len(all_results)
    n_cols     = 4
    n_rows     = int(np.ceil(n_params / n_cols))
    fig, axes  = plt.subplots(n_rows, n_cols,
                               figsize=(n_cols * 4.2, n_rows * 3.5))
    fig.patch.set_facecolor("#FAFAFA")
    axes_flat  = axes.flatten()
    targets    = CONFIG["bjs_targets"]
    x          = np.array(PERTURBATIONS) * 100
    win_colors = {3: COLORS["3yr"], 6: COLORS["6yr"], 9: COLORS["9yr"]}

    for idx, r in enumerate(all_results):
        ax = axes_flat[idx]
        df = r["df"]
        for win in [3, 6, 9]:
            means = df[f"mean_{win}yr"].values
            cis   = df[f"ci95_{win}yr"].values
            clr   = win_colors[win]
            ax.plot(x, means, color=clr, linewidth=2.0,
                    label=f"{win}yr", zorder=3)
            ax.fill_between(x, means - cis, means + cis,
                            color=clr, alpha=0.15, zorder=2)
            ax.axhline(targets[win], color=clr, linewidth=0.8,
                       linestyle=":", alpha=0.6)
        ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--")
        ax.axvspan(x[0] - 3, x[0] - 1, color=r["color"], alpha=0.6)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlim(x[0] - 3, x[-1] + 3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v):+d}%" for v in x], fontsize=7.5)
        ax.tick_params(labelsize=8)
        ax.set_title(r["label"].replace("\n", " "), fontsize=9,
                     fontweight="bold", color="#1A3D5C", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color=COLORS["grid"], linewidth=0.4, linestyle="--")

    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=COLORS["3yr"], lw=2, label="3-Year Rate"),
        Line2D([0], [0], color=COLORS["6yr"], lw=2, label="6-Year Rate"),
        Line2D([0], [0], color=COLORS["9yr"], lw=2, label="9-Year Rate"),
        Line2D([0], [0], color="#888888", lw=1, linestyle="--",
               label="Baseline (0%)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "OAT Sensitivity — Response Curves Per Parameter\n"
        "Shaded bands = 95% CI across 100 runs  |  Dotted lines = BJS targets",
        fontsize=13, fontweight="bold", y=1.01, color="#1A3D5C",
    )
    plt.tight_layout()
    path = os.path.join(outdir, "oat_lines.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Line plots -> {path}")
    return path


# =============================================================================
# CHART 3 — Sensitivity Heatmap
# =============================================================================

def plot_heatmap(all_results: list, outdir: str) -> str:
    windows = [3, 6, 9]
    labels  = [r["short"] for r in all_results]
    groups  = [r["group"] for r in all_results]

    matrix = np.zeros((len(all_results), 3))
    for i, r in enumerate(all_results):
        for j, w in enumerate(windows):
            matrix[i, j] = sensitivity_magnitude(r, w, pct=0.40)

    fig, ax = plt.subplots(figsize=(7, max(5, len(all_results) * 0.52 + 1.5)))
    fig.patch.set_facecolor("#FAFAFA")
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=matrix.max() * 1.1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["3-Year", "6-Year", "9-Year"],
                       fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(all_results)):
        for j in range(3):
            val = matrix[i, j]
            txt_color = "white" if val > matrix.max() * 0.6 else "#222222"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="bold")

    prev_grp = None
    for i, grp in enumerate(groups):
        if grp != prev_grp:
            if i > 0:
                ax.axhline(i - 0.5, color="white", linewidth=2.5)
            ax.text(-0.6, i, grp, ha="right", va="center",
                    fontsize=8, color=GROUP_COLORS[grp],
                    fontweight="bold", clip_on=False)
            prev_grp = grp

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(
        "Sensitivity Magnitude\n(pp change per +-40% perturbation)",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(
        "Sensitivity Heatmap — All Parameters x All Windows\n"
        "Values = |mean(+40%) - mean(-40%)| / 2",
        fontsize=11, fontweight="bold", pad=10, color="#1A3D5C",
    )
    plt.tight_layout()
    path = os.path.join(outdir, "oat_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Heatmap -> {path}")
    return path


# =============================================================================
# CHART 4 — Group Summary
# =============================================================================

def plot_group_summary(all_results: list, outdir: str) -> str:
    windows = [3, 6, 9]
    groups  = list(GROUP_COLORS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, win in zip(axes, windows):
        group_means, group_errs = [], []
        for grp in groups:
            mags = [sensitivity_magnitude(r, win, 0.40)
                    for r in all_results if r["group"] == grp]
            group_means.append(np.mean(mags) if mags else 0.0)
            group_errs.append(np.std(mags) if len(mags) > 1 else 0.0)

        colors = [GROUP_COLORS[g] for g in groups]
        xpos   = np.arange(len(groups))
        ax.bar(xpos, group_means, color=colors, edgecolor="white",
               width=0.55, zorder=3)
        ax.errorbar(xpos, group_means, yerr=group_errs, fmt="none",
                    color="#333333", elinewidth=1.8, capsize=6,
                    capthick=1.8, zorder=4)
        for xi, (m, e) in enumerate(zip(group_means, group_errs)):
            ax.text(xi, m + e + 0.0003, f"{m:.4f}",
                    ha="center", fontsize=9, fontweight="bold",
                    color="#333333")
        ax.set_xticks(xpos)
        ax.set_xticklabels(groups, fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        _style_ax(ax, "", "Avg Sensitivity (pp)",
                  f"{win}-Year Rate Sensitivity by Group")

    fig.suptitle(
        "Group-Level Sensitivity Summary\n"
        "Average |+-40% response| per parameter group",
        fontsize=13, fontweight="bold", y=1.02, color="#1A3D5C",
    )
    plt.tight_layout()
    path = os.path.join(outdir, "oat_group_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Group summary -> {path}")
    return path


# =============================================================================
# CHART 5 — Full Dissertation Report
# =============================================================================

def plot_full_report(all_results: list, outdir: str) -> str:
    """
    Dissertation composite. Tornado panel uses pill labels (no dots),
    matching the standalone plot_tornado style exactly.
    gs left=0.18 reserves space for the pills in the tight composite layout.
    """
    windows = [3, 6, 9]
    targets = CONFIG["bjs_targets"]

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(
        2, 3, hspace=0.45, wspace=0.40,
        left=0.18, right=0.97, top=0.91, bottom=0.06,
    )

    # ── A: Tornado 9yr ────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])

    rows = []
    for r in all_results:
        df    = r["df"]
        pos40 = df[df["pct"] ==  0.40]["mean_9yr"].values
        neg40 = df[df["pct"] == -0.40]["mean_9yr"].values
        pos20 = df[df["pct"] ==  0.20]["mean_9yr"].values
        neg20 = df[df["pct"] == -0.20]["mean_9yr"].values
        base  = r["baseline"][9]
        if len(pos40) == 0 or len(neg40) == 0:
            continue
        delta_pos = float(pos40[0]) - base
        delta_neg = float(neg40[0]) - base
        mag40     = abs(delta_pos - delta_neg) / 2.0
        mag20     = (abs(float(pos20[0]) - float(neg20[0])) / 2.0
                     if len(pos20) > 0 and len(neg20) > 0 else 0.0)
        rows.append({
            "full_label": r["label"].replace("\n", " "),
            "group":      r["group"],
            "color":      r["color"],
            "delta_pos":  delta_pos,
            "delta_neg":  delta_neg,
            "mag40":      mag40,
            "mag20":      mag20,
        })
    rows.sort(key=lambda x: x["mag40"])

    for i, row in enumerate(rows):
        c_pos = COLORS["up"]   if row["delta_pos"] >= 0 else COLORS["down"]
        c_neg = COLORS["down"] if row["delta_neg"] <= 0 else COLORS["up"]
        ax_a.barh(i, row["delta_pos"], color=c_pos, alpha=0.85,
                  height=0.55, zorder=3)
        ax_a.barh(i, row["delta_neg"], color=c_neg, alpha=0.85,
                  height=0.55, zorder=3)
        ax_a.text(
            max(abs(row["delta_pos"]), abs(row["delta_neg"])) + 0.0003, i,
            f"+-40%:{row['mag40']:.3f}  +-20%:{row['mag20']:.3f}",
            va="center", fontsize=7, color="#444444",
        )

    ax_a.set_yticks(range(len(rows)))
    _draw_pill_labels(ax_a, rows, fontsize=8)
    ax_a.axvline(0, color="#333333", linewidth=1.2, zorder=4)
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1%}"))
    ax_a.grid(True, color=COLORS["grid"], linewidth=0.4,
              linestyle="--", zorder=0)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    _tornado_legends(ax_a)
    _style_ax(
        ax_a,
        "Change in 9-Year Rate from Baseline", "",
        f"A — Tornado: 9-Year Sensitivity  (baseline={targets[9]:.0%})",
    )

    # ── B: Heatmap ────────────────────────────────────────────────────────────
    ax_b   = fig.add_subplot(gs[0, 2])
    labels = [r["short"] for r in all_results]
    matrix = np.zeros((len(all_results), 3))
    for i, r in enumerate(all_results):
        for j, w in enumerate(windows):
            matrix[i, j] = sensitivity_magnitude(r, w, 0.40)

    im = ax_b.imshow(matrix, aspect="auto", cmap="YlOrRd",
                     vmin=0, vmax=max(matrix.max() * 1.1, 0.001))
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["3yr", "6yr", "9yr"],
                         fontsize=10, fontweight="bold")
    ax_b.set_yticks(range(len(labels)))
    ax_b.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(all_results)):
        for j in range(3):
            val = matrix[i, j]
            tc  = "white" if val > matrix.max() * 0.6 else "#222222"
            ax_b.text(j, i, f"{val:.3f}", ha="center", va="center",
                      fontsize=7.5, color=tc, fontweight="bold")
    plt.colorbar(im, ax=ax_b, shrink=0.7,
                 pad=0.02).ax.tick_params(labelsize=7)
    _style_ax(ax_b, "", "", "B — Heatmap: All Params x Windows")

    # ── C / D / E: Group summaries ─────────────────────────────────────────────
    for col_i, win in enumerate(windows):
        ax    = fig.add_subplot(gs[1, col_i])
        grps  = list(GROUP_COLORS.keys())
        means, errs = [], []
        for grp in grps:
            mags = [sensitivity_magnitude(r, win, 0.40)
                    for r in all_results if r["group"] == grp]
            means.append(np.mean(mags) if mags else 0.0)
            errs.append(np.std(mags) if len(mags) > 1 else 0.0)
        clrs = [GROUP_COLORS[g] for g in grps]
        xp   = np.arange(len(grps))
        ax.bar(xp, means, color=clrs, edgecolor="white", width=0.55, zorder=3)
        ax.errorbar(xp, means, yerr=errs, fmt="none", color="#333333",
                    elinewidth=1.5, capsize=5, zorder=4)
        for xi, (m, e) in enumerate(zip(means, errs)):
            ax.text(xi, m + e + 0.0001, f"{m:.4f}",
                    ha="center", fontsize=8, color="#333333")
        ax.set_xticks(xp)
        ax.set_xticklabels(grps, fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        label_letter = chr(67 + col_i)
        _style_ax(ax, "", "Avg Sensitivity",
                  f"{label_letter} — {win}-Year: Group Sensitivity")

    fig.suptitle(
        "OAT Sensitivity Analysis — Recidivism ABM\n"
        "One-At-a-Time perturbation +-10/20/30/40%  |  n=100 runs per condition",
        fontsize=14, fontweight="bold", y=0.975, color="#1A3D5C",
    )
    path = os.path.join(outdir, "oat_full_report.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Full report -> {path}")
    return path


# =============================================================================
# CORE DETECTION
# =============================================================================

def detect_workers() -> tuple:
    if _PSUTIL_AVAILABLE:
        physical = psutil.cpu_count(logical=False)
        logical  = psutil.cpu_count(logical=True)
        if physical and physical > 0:
            n = max(1, physical - 1)
            return n, f"psutil: {physical} physical -> {n} workers"
        n = max(1, (logical or 2) // 2 - 1)
        return n, f"psutil fallback: {n} workers"
    logical = multiprocessing.cpu_count()
    n = max(1, logical // 2 - 1)
    return n, f"logical/2: {logical} -> {n} workers"


# =============================================================================
# MAIN
# =============================================================================

def main(n_workers: int, force_rerun: bool = False):
    outdir   = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)

    n_total  = CONFIG["N_REPS"] * len(CONFIG["SEEDS"])
    n_params = len(PARAM_DEFS)
    n_levels = len(PERTURBATIONS)
    est_sims = n_params * n_levels * n_total

    print("=" * 70)
    print("  RECIDIVISM ABM — OAT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"  Workers        : {n_workers}")
    print(f"  Parameters     : {n_params}  (6 calibration + 5 risk weights)")
    print(f"  Levels / param : {n_levels}  ({', '.join(PERTURB_LABELS)})")
    print(f"  Runs / level   : {n_total}  "
          f"({CONFIG['N_REPS']} reps x {len(CONFIG['SEEDS'])} seeds)")
    print(f"  Est. sims      : ~{est_sims:,}")
    print(f"  Output dir     : {outdir}")
    print(f"  Checkpoints    : {'disabled (--rerun)' if force_rerun else 'enabled'}")
    print()
    print("  Baseline calibration:")
    for k, v in BASELINE_CAL.items():
        print(f"    {k:<45} {v}")
    print("=" * 70)

    rw_base = get_flat_risk_weights()
    for pdef in PARAM_DEFS:
        if pdef["mode"] == "risk_weight" and pdef["baseline"] is None:
            key = pdef["key"]
            pdef["baseline"] = float(rw_base.get(key, 1.0))
            print(f"  Risk weight '{key}' baseline: {pdef['baseline']}")

    all_results = []
    for pdef in PARAM_DEFS:
        cp = os.path.join(outdir, f"oat_{pdef['key']}.json")
        if not force_rerun and os.path.exists(cp):
            print(f"  > {pdef['short']:8s} — loading checkpoint")
            with open(cp) as f:
                saved = json.load(f)
            saved["df"] = pd.DataFrame(saved["df_records"])
            del saved["df_records"]
            all_results.append(saved)
            continue

        print(f"\n  > {pdef['short']:8s} — "
              f"{pdef['label'].replace(chr(10), ' ')}  "
              f"(group: {pdef['group']})")
        result = run_oat_sweep(pdef, n_workers)
        all_results.append(result)

        save = {k: v for k, v in result.items() if k != "df"}
        save["df_records"] = result["df"].to_dict(orient="records")
        with open(cp, "w") as f:
            json.dump(save, f, indent=2)

    print(f"\n{'='*70}")
    print("  OAT SENSITIVITY SUMMARY  (magnitude at +-20% perturbation)")
    print(f"{'='*70}")
    print(f"  {'Param':<10} {'Group':<14} {'3yr sens':>10} "
          f"{'6yr sens':>10} {'9yr sens':>10}")
    print(f"  {'-'*54}")
    for r in sorted(all_results,
                    key=lambda x: sensitivity_magnitude(x, 9, 0.20),
                    reverse=True):
        s3 = sensitivity_magnitude(r, 3, 0.20)
        s6 = sensitivity_magnitude(r, 6, 0.20)
        s9 = sensitivity_magnitude(r, 9, 0.20)
        print(f"  {r['short']:<10} {r['group']:<14} "
              f"{s3:>9.4f}  {s6:>9.4f}  {s9:>9.4f}")

    print(f"\n{'─'*70}")
    print("  Generating charts...")
    print(f"{'─'*70}")

    for win in [3, 6, 9]:
        plot_tornado(all_results, win, outdir)
    plot_lines(all_results, outdir)
    plot_heatmap(all_results, outdir)
    plot_group_summary(all_results, outdir)
    plot_full_report(all_results, outdir)

    summary = {
        "n_params":      n_params,
        "n_levels":      n_levels,
        "n_runs":        n_total,
        "perturbations": PERTURBATIONS,
        "bjs_targets":   CONFIG["bjs_targets"],
        "baseline_cal":  BASELINE_CAL,
        "results": [
            {
                "key":             r["key"],
                "short":           r["short"],
                "group":           r["group"],
                "sensitivity_3yr": sensitivity_magnitude(r, 3, 0.20),
                "sensitivity_6yr": sensitivity_magnitude(r, 6, 0.20),
                "sensitivity_9yr": sensitivity_magnitude(r, 9, 0.20),
                "baseline_3yr":    r["baseline"][3],
                "baseline_6yr":    r["baseline"][6],
                "baseline_9yr":    r["baseline"][9],
            }
            for r in all_results
        ],
    }
    json_path = os.path.join(outdir, "oat_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary JSON -> {json_path}")
    print(f"  Charts -> {outdir}/oat_*.png")
    return summary


# =============================================================================
# REPLOT FROM CHECKPOINTS
# =============================================================================

def replot_from_checkpoints(outdir: str = None):
    if outdir is None:
        outdir = CONFIG["output_directory"]

    rw_base = get_flat_risk_weights()
    for pdef in PARAM_DEFS:
        if pdef["mode"] == "risk_weight" and pdef["baseline"] is None:
            pdef["baseline"] = float(rw_base.get(pdef["key"], 1.0))

    all_results = []
    for pdef in PARAM_DEFS:
        cp = os.path.join(outdir, f"oat_{pdef['key']}.json")
        if not os.path.exists(cp):
            print(f"  Checkpoint not found for {pdef['short']} — skipping")
            continue
        with open(cp) as f:
            saved = json.load(f)
        saved["df"] = pd.DataFrame(saved["df_records"])
        del saved["df_records"]
        all_results.append(saved)

    if not all_results:
        print("No checkpoints found. Run main() first.")
        return

    for win in [3, 6, 9]:
        plot_tornado(all_results, win, outdir)
    plot_lines(all_results, outdir)
    plot_heatmap(all_results, outdir)
    plot_group_summary(all_results, outdir)
    plot_full_report(all_results, outdir)
    print("All charts regenerated.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = sys.argv[1:]

    if "--replot" in args:
        replot_from_checkpoints()
        sys.exit(0)

    force_rerun = "--rerun" in args

    if "--cores" in args:
        idx = args.index("--cores")
        try:
            n_workers = int(args[idx + 1])
            print(f"\n  Manual override: {n_workers} workers")
        except (IndexError, ValueError):
            print("Usage: python sensitivity_oat.py [--cores N] [--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  Core auto-detection: {reason}")

    print(f"  Starting OAT sensitivity analysis with {n_workers} workers...\n")
    main(n_workers=n_workers, force_rerun=force_rerun)