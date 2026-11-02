"""
sensitivity_oat.py  —  One-At-a-Time (OAT) Sensitivity Analysis
================================================================
Recidivism ABM — PhD Dissertation Tool

PURPOSE
-------
Tests how sensitive the ABM's 3/6/9-year cumulative rearrest rates are
to perturbations of each parameter individually, holding all others fixed
at their calibrated (baseline) values.

Three parameter groups are analysed:
  1. CALIBRATION  — 6 global calibration parameters
  2. RISK WEIGHTS — Age_at_Release, Prior_Revocations_Supervision,
                    Gang_Affiliated, Percent_Days_Employed, Program_Attendances

PERTURBATION DESIGN
-------------------
  9 levels per parameter:  -40%, -30%, -20%, -10%, 0% (baseline),
                           +10%, +20%, +30%, +40%
  For each level: N_REPS × N_SEEDS = 100 runs
  Total simulations: ~9,900  (11 params × 9 levels × 100 runs)

CHARTS PRODUCED
---------------
  1. oat_tornado_3yr.png     — Tornado chart ranked by 3yr sensitivity
  2. oat_tornado_6yr.png     — Tornado chart ranked by 6yr sensitivity
  3. oat_tornado_9yr.png     — Tornado chart ranked by 9yr sensitivity
  4. oat_lines.png           — Response curves per parameter (all windows)
  5. oat_heatmap.png         — Sensitivity heatmap (params × windows)
  6. oat_group_summary.png   — Group-level sensitivity comparison
  7. oat_full_report.png     — Dissertation-ready composite figure

USAGE
-----
  python sensitivity_oat.py
  python sensitivity_oat.py --cores 8
  python sensitivity_oat.py --replot        # regenerate charts only
  python sensitivity_oat.py --rerun         # ignore checkpoints

DEPENDENCIES
------------
  pip install matplotlib numpy pandas tqdm scipy psutil
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

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
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

# ─── Confirmed baseline calibration parameters ────────────────────────────────
BASELINE_CAL = {
    "Risk_Contrast_Strength":                0.0,
    "Risk_Effect_Decay_After_3Y":            0.9000,
    "Risk_Effect_Decay_After_6Y":            0.7000,
    "Supervision_Monitoring_Decay_After_3Y": 0.3500,
    "Supervision_Monitoring_Decay_After_6Y": 0.2000,
    "Supervision_Monitoring_Intensity":      0.35,
}

# ─── Perturbation levels ──────────────────────────────────────────────────────
PERTURBATIONS = [-0.40, -0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30, +0.40]
PERTURB_LABELS = ["-40%", "-30%", "-20%", "-10%", "Baseline",
                  "+10%", "+20%", "+30%", "+40%"]

# ─── Parameter group definitions ─────────────────────────────────────────────
# Each entry: (display_name, group, baseline_value, perturb_mode)
# perturb_mode: "multiply" (relative %) or "add" (absolute offset)
# For Risk_Contrast_Strength baseline=0.0: use additive ±0.1 steps instead
PARAM_DEFS = [
    # ── Calibration parameters ──
    {
        "key":          "Supervision_Monitoring_Intensity",
        "label":        "Supervision\nMonitoring Intensity",
        "short":        "SMI",
        "group":        "Calibration",
        "baseline":     0.35,
        "mode":         "multiply",
        "group_color":  "#2E75B6",
    },
    {
        "key":          "Risk_Contrast_Strength",
        "label":        "Risk Contrast\nStrength",
        "short":        "RCS",
        "group":        "Calibration",
        "baseline":     0.0,
        "mode":         "add_fixed",   # additive: ±0.1, ±0.2, ±0.3 from 0.0
        "step":         0.1,
        "group_color":  "#2E75B6",
    },
    {
        "key":          "Risk_Effect_Decay_After_3Y",
        "label":        "Risk Effect\nDecay After 3Y",
        "short":        "RED3",
        "group":        "Calibration",
        "baseline":     0.9,
        "mode":         "multiply",
        "group_color":  "#2E75B6",
    },
    {
        "key":          "Risk_Effect_Decay_After_6Y",
        "label":        "Risk Effect\nDecay After 6Y",
        "short":        "RED6",
        "group":        "Calibration",
        "baseline":     0.7,
        "mode":         "multiply",
        "group_color":  "#2E75B6",
    },
    {
        "key":          "Supervision_Monitoring_Decay_After_3Y",
        "label":        "Supervision Decay\nAfter 3Y",
        "short":        "SMD3",
        "group":        "Calibration",
        "baseline":     0.35,
        "mode":         "multiply",
        "group_color":  "#2E75B6",
    },
    {
        "key":          "Supervision_Monitoring_Decay_After_6Y",
        "label":        "Supervision Decay\nAfter 6Y",
        "short":        "SMD6",
        "group":        "Calibration",
        "baseline":     0.2,
        "mode":         "multiply",
        "group_color":  "#2E75B6",
    },
    # ── Risk weight parameters (keys match get_flat_risk_weights exactly) ──
    {
        "key":          "Age_at_Release",
        "label":        "Age at\nRelease Weight",
        "short":        "Age",
        "group":        "Risk Weight",
        "baseline":     0.73,   # NIJ logistic regression coefficient
        "mode":         "risk_weight",
        "group_color":  "#ED7D31",
    },
    {
        "key":          "Prior_Revocations_Supervision",
        "label":        "Prior Revocations\nSupervision Weight",
        "short":        "Revocation",
        "group":        "Risk Weight",
        "baseline":     0.4338,
        "mode":         "risk_weight",
        "group_color":  "#ED7D31",
    },
    {
        "key":          "Gang_Affiliated",
        "label":        "Gang Affiliated\nWeight",
        "short":        "Gang",
        "group":        "Risk Weight",
        "baseline":     0.84,
        "mode":         "risk_weight",
        "group_color":  "#ED7D31",
    },
    {
        "key":          "Percent_Days_Employed",
        "label":        "Percent Days\nEmployed Weight",
        "short":        "Employment",
        "group":        "Risk Weight",
        "baseline":     -0.95,  # negative: employment reduces rearrest risk
        "mode":         "risk_weight",
        "group_color":  "#ED7D31",
    },
    {
        "key":          "Program_Attendances",
        "label":        "Program\nAttendances Weight",
        "short":        "Program",
        "group":        "Risk Weight",
        "baseline":     -0.38,  # negative: attendance reduces rearrest risk
        "mode":         "risk_weight",
        "group_color":  "#ED7D31",
    },
]

# ─── Colors ───────────────────────────────────────────────────────────────────
COLORS = {
    "bjs":         "#1A3D5C",
    "baseline":    "#555555",
    "up":          "#C00000",
    "down":        "#2E75B6",
    "grid":        "#E8E8E8",
    "cal":         "#2E75B6",
    "structural":  "#70AD47",
    "risk":        "#ED7D31",
    "3yr":         "#2E75B6",
    "6yr":         "#70AD47",
    "9yr":         "#ED7D31",
}

GROUP_COLORS = {
    "Calibration":  "#2E75B6",
    "Risk Weight":  "#ED7D31",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PERTURBATION VALUE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def get_perturbed_values(pdef: dict) -> list:
    """
    Return list of 7 actual parameter values for the 7 perturbation levels.
    """
    mode = pdef["mode"]
    base = pdef["baseline"]
    vals = []

    for pct in PERTURBATIONS:
        if mode == "multiply":
            v = base * (1.0 + pct)
            # Clamp to [0, 1] only for calibration params (always positive)
            v = max(0.0, min(1.0, v))
        elif mode == "add_fixed":
            step  = pdef.get("step", 0.1)
            level = round(pct / 0.10)   # -4,-3,-2,-1,0,1,2,3,4
            v     = base + level * step
            v     = max(0.0, v)
        elif mode in ("structural", "risk_weight"):
            v = base * (1.0 + pct)
            if mode == "structural" and pdef["key"] in ("initial_agents",
                                                         "warmup_months",
                                                         "monthly_intake"):
                v = max(1, int(round(v)))
            # Risk weights: no clamping — negative weights (e.g. employment,
            # program attendance) must remain negative; ±40% scales them correctly
        vals.append(v)
    return vals


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER  —  module-level for pickling
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(task: tuple) -> tuple:
    """
    Run one ABM simulation.
    task_payload encodes: calibration overrides + structural overrides +
                          risk weight overrides + task metadata.
    """
    (payload_json, seed) = task

    payload   = json.loads(payload_json)
    task_id   = payload.pop("__task_id__", 0.0)
    warmup    = int(payload.pop("__warmup__",  CONFIG["warmup_months"]))
    study     = int(payload.pop("__study__",   CONFIG["study_months"]))
    agents    = int(payload.pop("__agents__",  CONFIG["initial_agents"]))
    intake    = int(payload.pop("__intake__",  CONFIG["monthly_intake"]))
    rw_ovr    = payload.pop("__risk_weights__", {})

    # Build calibration dict
    cal = get_global_calibration_params()
    cal.update(BASELINE_CAL)
    # Apply any calibration overrides from payload
    for k, v in payload.items():
        if not k.startswith("__"):
            cal[k] = v

    # Build risk weights
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
        print(f"  ⚠️  Worker error (tid={task_id:.4f}, seed={seed}): {e}", flush=True)
        return (task_id, seed, 0.0, 0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT_RUN1 CSV CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

_CLEANUP_EVERY_N   = 50    # delete every N completed simulations


def _cleanup_result_run1() -> int:
    """
    Delete ALL files in Results_Run1/. Called every _CLEANUP_EVERY_N
    completed simulations. Skips subdirectories.
    """
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _parent_dir  = os.path.dirname(_script_dir)

    candidates = [
        os.path.join(_parent_dir,  "Results_Run1"),   # one level up from script ✓
        os.path.join(_script_dir,  "Results_Run1"),   # same dir as script
        os.path.join(os.getcwd(),  "Results_Run1"),   # cwd
    ]
    target = next((p for p in candidates if os.path.isdir(p)), None)

    if target is None:
        print("  ⚠️  Results_Run1 not found. Checked:", flush=True)
        for p in candidates:
            print(f"       {p}", flush=True)
        return 0

    try:
        all_files = os.listdir(target)
        count     = len(all_files)
        print(f"  🔍  Result_Run1/ found {count} files at {target}", flush=True)
        if count == 0:
            print(f"  ✅  Result_Run1/ is already empty — nothing to delete",
                  flush=True)
            return 0
        deleted = 0
        failed  = 0
        for fname in all_files:
            fpath = os.path.join(target, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    deleted += 1
            except OSError as e:
                failed += 1
                print(f"  ⚠️  Could not delete {fname}: {e}", flush=True)
        if failed == 0:
            print(f"  ✅  Deleted all {deleted} files from Result_Run1/",
                  flush=True)
        elif deleted > 0:
            print(f"  ⚠️  Partially deleted: {deleted} removed, "
                  f"{failed} failed out of {count} total", flush=True)
        else:
            print(f"  ❌  Deletion failed — 0 of {count} files removed",
                  flush=True)
        return deleted
    except OSError as e:
        print(f"  ⚠️  Result_Run1 cleanup error: {e}", flush=True)
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_batch(overrides_list: list, n_workers: int, desc: str = "Running") -> dict:
    """
    overrides_list: list of (task_id, payload_dict) pairs
    Returns dict: task_id → {3: [...], 6: [...], 9: [...]}
    """
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

    raw = {float(tid): {3: [], 6: [], 9: []} for tid, _ in overrides_list}
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
                    print(f"\n  ⚠️  Future error: {e}", flush=True)
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


# ═══════════════════════════════════════════════════════════════════════════════
# OAT SWEEP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_oat_sweep(pdef: dict, n_workers: int) -> dict:
    """
    Run OAT sweep for one parameter.
    Returns dict with perturbed values, means, CIs for each window.
    """
    mode   = pdef["mode"]
    key    = pdef["key"]
    values = get_perturbed_values(pdef)

    overrides_list = []
    for i, v in enumerate(values):
        payload = {}

        if mode in ("multiply", "add_fixed"):
            # Calibration parameter — goes into cal overrides
            payload[key] = float(v)
            payload["__warmup__"] = CONFIG["warmup_months"]
            payload["__study__"]  = CONFIG["study_months"]
            payload["__agents__"] = CONFIG["initial_agents"]
            payload["__intake__"] = CONFIG["monthly_intake"]

        elif mode == "structural":
            payload["__warmup__"] = (int(v) if key == "warmup_months"
                                     else CONFIG["warmup_months"])
            payload["__study__"]  = CONFIG["study_months"]
            payload["__agents__"] = (int(v) if key == "initial_agents"
                                     else CONFIG["initial_agents"])
            payload["__intake__"] = (int(v) if key == "monthly_intake"
                                     else CONFIG["monthly_intake"])

        elif mode == "risk_weight":
            payload["__warmup__"]       = CONFIG["warmup_months"]
            payload["__study__"]        = CONFIG["study_months"]
            payload["__agents__"]       = CONFIG["initial_agents"]
            payload["__intake__"]       = CONFIG["monthly_intake"]
            payload["__risk_weights__"] = {key: float(v)}

        overrides_list.append((float(i), payload))

    raw = _run_batch(overrides_list, n_workers,
                     desc=f"{pdef['short']:6s} OAT")

    rows = []
    for i, v in enumerate(values):
        agg = _aggregate(raw[float(i)])
        rows.append({
            "level_idx":   i,
            "pct":         PERTURBATIONS[i],
            "label":       PERTURB_LABELS[i],
            "value":       v,
            "mean_3yr":    agg[3],
            "mean_6yr":    agg[6],
            "mean_9yr":    agg[9],
            "ci95_3yr":    agg["ci95_3"],
            "ci95_6yr":    agg["ci95_6"],
            "ci95_9yr":    agg["ci95_9"],
        })

    df = pd.DataFrame(rows)
    baseline_row = df[df["pct"] == 0.0].iloc[0]

    return {
        "key":     key,
        "label":   pdef["label"],
        "short":   pdef["short"],
        "group":   pdef["group"],
        "color":   pdef["group_color"],
        "df":      df,
        "baseline": {
            3: float(baseline_row["mean_3yr"]),
            6: float(baseline_row["mean_6yr"]),
            9: float(baseline_row["mean_9yr"]),
        },
    }


def sensitivity_magnitude(result: dict, window: int, pct: float = 0.40) -> float:
    """
    Return |mean(+pct) - mean(-pct)| / 2  for a given window.
    Default uses ±40% — the outer perturbation level.
    """
    df  = result["df"]
    pos = df[df["pct"] == pct][f"mean_{window}yr"].values
    neg = df[df["pct"] == -pct][f"mean_{window}yr"].values
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return abs(float(pos[0]) - float(neg[0])) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, xlabel="", ylabel="", title=""):
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Tornado Charts (one per output window)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tornado(all_results: list, window: int, outdir: str) -> str:
    """
    Horizontal bar tornado chart — all params ranked by sensitivity magnitude
    at ±40% perturbation for the given output window.
    Annotates both ±20% and ±40% sensitivity for comparison.
    """
    targets = CONFIG["bjs_targets"]

    rows = []
    for r in all_results:
        df   = r["df"]
        pos40 = df[df["pct"] ==  0.40][f"mean_{window}yr"].values
        neg40 = df[df["pct"] == -0.40][f"mean_{window}yr"].values
        pos20 = df[df["pct"] ==  0.20][f"mean_{window}yr"].values
        neg20 = df[df["pct"] == -0.20][f"mean_{window}yr"].values
        base  = r["baseline"][window]
        if len(pos40) == 0 or len(neg40) == 0:
            continue
        delta_pos = float(pos40[0]) - base
        delta_neg = float(neg40[0]) - base
        mag40 = abs(delta_pos - delta_neg) / 2.0
        mag20 = (abs(float(pos20[0]) - float(neg20[0])) / 2.0
                 if len(pos20) > 0 and len(neg20) > 0 else 0.0)
        rows.append({
            "label":     r["short"],
            "full_label": r["label"].replace("\n", " "),
            "group":     r["group"],
            "color":     r["color"],
            "delta_pos": delta_pos,
            "delta_neg": delta_neg,
            "mag40":     mag40,
            "mag20":     mag20,
        })

    rows.sort(key=lambda x: x["mag40"])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.58 + 1.5)))
    fig.patch.set_facecolor("#FAFAFA")

    ypos = np.arange(n)
    base_val = targets[window]

    for i, row in enumerate(rows):
        c_pos = COLORS["up"]   if row["delta_pos"] >= 0 else COLORS["down"]
        c_neg = COLORS["down"] if row["delta_neg"] <= 0 else COLORS["up"]
        ax.barh(i, row["delta_pos"], left=0,
                color=c_pos, alpha=0.85, height=0.55, zorder=3)
        ax.barh(i, row["delta_neg"], left=0,
                color=c_neg, alpha=0.85, height=0.55, zorder=3)
        ax.scatter(-0.002, i, s=80, color=row["color"], zorder=5, clip_on=False)

        max_ext = max(abs(row["delta_pos"]), abs(row["delta_neg"]))
        ax.text(max_ext + 0.0005, i,
                f"±40%:{row['mag40']:.3f}  ±20%:{row['mag20']:.3f}",
                va="center", fontsize=7.5, color="#444444")

    ax.set_yticks(ypos)
    ax.set_yticklabels([r["full_label"] for r in rows], fontsize=9)
    ax.axvline(0, color="#333333", linewidth=1.2, zorder=4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1%}"))

    patches = [
        mpatches.Patch(color=COLORS["up"],   alpha=0.85, label="+40% perturbation"),
        mpatches.Patch(color=COLORS["down"], alpha=0.85, label="−40% perturbation"),
    ] + [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, fontsize=8.5, loc="lower right", framealpha=0.9)

    ax.set_title(
        f"Tornado Chart — {window}-Year Rearrest Rate Sensitivity\n"
        f"Baseline = {base_val:.0%}  |  Bars show change from ±40% parameter perturbation",
        fontsize=12, fontweight="bold", pad=10, color="#1A3D5C",
    )
    ax.set_xlabel("Change in Rearrest Rate from Baseline", fontsize=10)

    plt.tight_layout()
    path = os.path.join(outdir, f"oat_tornado_{window}yr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    📊 Tornado {window}yr → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Line plots per parameter
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lines(all_results: list, outdir: str) -> str:
    """
    One subplot per parameter showing response across all 7 levels,
    all three output windows overlaid.
    """
    n_params = len(all_results)
    n_cols   = 4
    n_rows   = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.2, n_rows * 3.5))
    fig.patch.set_facecolor("#FAFAFA")
    axes_flat = axes.flatten()

    targets = CONFIG["bjs_targets"]
    x       = np.array(PERTURBATIONS) * 100  # convert to % for x-axis

    win_colors = {3: COLORS["3yr"], 6: COLORS["6yr"], 9: COLORS["9yr"]}

    for idx, r in enumerate(all_results):
        ax  = axes_flat[idx]
        df  = r["df"]

        for win in [3, 6, 9]:
            means = df[f"mean_{win}yr"].values
            cis   = df[f"ci95_{win}yr"].values
            clr   = win_colors[win]
            ax.plot(x, means, color=clr, linewidth=2.0,
                    label=f"{win}yr", zorder=3)
            ax.fill_between(x, means - cis, means + cis,
                            color=clr, alpha=0.15, zorder=2)
            # BJS reference line
            ax.axhline(targets[win], color=clr, linewidth=0.8,
                       linestyle=":", alpha=0.6)

        # Baseline vertical
        ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--")

        # Group colour bar on left
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

    # Hide unused subplots
    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
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
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    📊 Line plots → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Sensitivity Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(all_results: list, outdir: str) -> str:
    """
    Heatmap: rows = parameters, columns = output windows,
    cell = sensitivity magnitude at ±20%.
    """
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
    ax.set_xticklabels(["3-Year", "6-Year", "9-Year"], fontsize=11,
                       fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Cell annotations
    for i in range(len(all_results)):
        for j in range(3):
            val = matrix[i, j]
            txt_color = "white" if val > matrix.max() * 0.6 else "#222222"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="bold")

    # Group separators and labels
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
    cbar.set_label("Sensitivity Magnitude\n(pp change per ±40% perturbation)",
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Sensitivity Heatmap — All Parameters × All Windows\n"
        "Values = |mean(+40%) − mean(−40%)| / 2",
        fontsize=11, fontweight="bold", pad=10, color="#1A3D5C",
    )

    plt.tight_layout()
    path = os.path.join(outdir, "oat_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    📊 Heatmap → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Group Summary
# ═══════════════════════════════════════════════════════════════════════════════

def plot_group_summary(all_results: list, outdir: str) -> str:
    """
    Bar chart: average sensitivity magnitude per group per window.
    """
    windows = [3, 6, 9]
    groups  = list(GROUP_COLORS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, win in zip(axes, windows):
        group_means = []
        group_errs  = []
        for grp in groups:
            mags = [sensitivity_magnitude(r, win, 0.40)
                    for r in all_results if r["group"] == grp]
            group_means.append(np.mean(mags) if mags else 0.0)
            group_errs.append(np.std(mags)   if len(mags) > 1 else 0.0)

        colors = [GROUP_COLORS[g] for g in groups]
        xpos   = np.arange(len(groups))
        bars   = ax.bar(xpos, group_means, color=colors, edgecolor="white",
                        width=0.55, zorder=3)
        ax.errorbar(xpos, group_means, yerr=group_errs,
                    fmt="none", color="#333333",
                    elinewidth=1.8, capsize=6, capthick=1.8, zorder=4)

        for xi, (m, e) in enumerate(zip(group_means, group_errs)):
            ax.text(xi, m + e + 0.0003, f"{m:.4f}",
                    ha="center", fontsize=9, fontweight="bold", color="#333333")

        ax.set_xticks(xpos)
        ax.set_xticklabels(groups, fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        _style_ax(ax, "", "Avg Sensitivity (pp)",
                  f"{win}-Year Rate Sensitivity by Group")

    fig.suptitle(
        "Group-Level Sensitivity Summary\n"
        "Average |±40% response| per parameter group",
        fontsize=13, fontweight="bold", y=1.02, color="#1A3D5C",
    )
    plt.tight_layout()
    path = os.path.join(outdir, "oat_group_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    📊 Group summary → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Full Dissertation Report
# ═══════════════════════════════════════════════════════════════════════════════

def plot_full_report(all_results: list, outdir: str) -> str:
    """
    Dissertation-ready composite: Tornado (9yr) | Heatmap | Group summary.
    """
    windows = [3, 6, 9]
    targets = CONFIG["bjs_targets"]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.40,
                            left=0.07, right=0.97, top=0.91, bottom=0.06)

    # ── A: Tornado 9yr ───────────────────────────────────────────────────────
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
        mag40 = abs(delta_pos - delta_neg) / 2.0
        mag20 = (abs(float(pos20[0]) - float(neg20[0])) / 2.0
                 if len(pos20) > 0 and len(neg20) > 0 else 0.0)
        rows.append({
            "label":     r["short"],
            "group":     r["group"],
            "color":     r["color"],
            "delta_pos": delta_pos,
            "delta_neg": delta_neg,
            "mag40":     mag40,
            "mag20":     mag20,
        })
    rows.sort(key=lambda x: x["mag40"])

    for i, row in enumerate(rows):
        c_pos = COLORS["up"]   if row["delta_pos"] >= 0 else COLORS["down"]
        c_neg = COLORS["down"] if row["delta_neg"] <= 0 else COLORS["up"]
        ax_a.barh(i, row["delta_pos"], color=c_pos, alpha=0.85,
                  height=0.55, zorder=3)
        ax_a.barh(i, row["delta_neg"], color=c_neg, alpha=0.85,
                  height=0.55, zorder=3)
        ax_a.scatter(-0.0015, i, s=70, color=row["color"],
                     zorder=5, clip_on=False)
        ax_a.text(max(abs(row["delta_pos"]), abs(row["delta_neg"])) + 0.0003,
                  i, f"±40%:{row['mag40']:.3f}  ±20%:{row['mag20']:.3f}",
                  va="center", fontsize=7, color="#444444")

    ax_a.set_yticks(range(len(rows)))
    ax_a.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax_a.axvline(0, color="#333333", linewidth=1.2, zorder=4)
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1%}"))
    patches_a = [
        mpatches.Patch(color=COLORS["up"],   alpha=0.85, label="+40% perturb"),
        mpatches.Patch(color=COLORS["down"], alpha=0.85, label="−40% perturb"),
    ] + [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax_a.legend(handles=patches_a, fontsize=8, loc="lower right")
    _style_ax(ax_a, "Change in 9-Year Rate from Baseline", "",
              f"A — Tornado: 9-Year Sensitivity  (baseline={targets[9]:.0%})")

    # ── B: Heatmap ────────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])

    labels = [r["short"] for r in all_results]
    matrix = np.zeros((len(all_results), 3))
    for i, r in enumerate(all_results):
        for j, w in enumerate(windows):
            matrix[i, j] = sensitivity_magnitude(r, w, 0.40)

    im = ax_b.imshow(matrix, aspect="auto", cmap="YlOrRd",
                     vmin=0, vmax=max(matrix.max() * 1.1, 0.001))
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["3yr", "6yr", "9yr"], fontsize=10, fontweight="bold")
    ax_b.set_yticks(range(len(labels)))
    ax_b.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(all_results)):
        for j in range(3):
            val = matrix[i, j]
            tc  = "white" if val > matrix.max() * 0.6 else "#222222"
            ax_b.text(j, i, f"{val:.3f}", ha="center", va="center",
                      fontsize=7.5, color=tc, fontweight="bold")
    plt.colorbar(im, ax=ax_b, shrink=0.7, pad=0.02).ax.tick_params(labelsize=7)
    _style_ax(ax_b, "", "", "B — Heatmap: All Params × Windows")

    # ── C/D/E: Group summaries ─────────────────────────────────────────────────
    for col_i, win in enumerate(windows):
        ax = fig.add_subplot(gs[1, col_i])
        grps  = list(GROUP_COLORS.keys())
        means = []
        errs  = []
        for grp in grps:
            mags = [sensitivity_magnitude(r, win, 0.40)
                    for r in all_results if r["group"] == grp]
            means.append(np.mean(mags) if mags else 0.0)
            errs.append(np.std(mags)   if len(mags) > 1 else 0.0)

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
        label = chr(67 + col_i)
        _style_ax(ax, "", "Avg Sensitivity",
                  f"{label} — {win}-Year: Group Sensitivity")

    fig.suptitle(
        "OAT Sensitivity Analysis — Recidivism ABM\n"
        "One-At-a-Time perturbation ±10/20/30/40%  |  n=100 runs per condition",
        fontsize=14, fontweight="bold", y=0.975, color="#1A3D5C",
    )

    path = os.path.join(outdir, "oat_full_report.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    📊 Full report → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_workers() -> tuple:
    if _PSUTIL_AVAILABLE:
        physical = psutil.cpu_count(logical=False)
        logical  = psutil.cpu_count(logical=True)
        if physical and physical > 0:
            n = max(1, physical - 1)
            return n, f"psutil: {physical} physical → {n} workers"
        n = max(1, (logical or 2) // 2 - 1)
        return n, f"psutil fallback: {n} workers"
    logical = multiprocessing.cpu_count()
    n = max(1, logical // 2 - 1)
    return n, f"logical/2: {logical} → {n} workers"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(n_workers: int, force_rerun: bool = False):
    outdir = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)

    n_total  = CONFIG["N_REPS"] * len(CONFIG["SEEDS"])
    n_params = len(PARAM_DEFS)
    n_levels = len(PERTURBATIONS)
    est_sims = n_params * n_levels * n_total

    print("=" * 70)
    print("  RECIDIVISM ABM — OAT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"  Workers        : {n_workers}")
    print(f"  Parameters     : {n_params}  "
          f"(6 calibration + 5 risk weights)")
    print(f"  Levels / param : {n_levels}  ({', '.join(PERTURB_LABELS)})")
    print(f"  Runs / level   : {n_total}  "
          f"({CONFIG['N_REPS']} reps × {len(CONFIG['SEEDS'])} seeds)")
    print(f"  Est. sims      : ~{est_sims:,}")
    print(f"  Output dir     : {outdir}")
    print(f"  Checkpoints    : {'disabled (--rerun)' if force_rerun else 'enabled'}")
    print("=" * 70)

    # ── Resolve risk weight baselines ─────────────────────────────────────────
    # Baselines are hardcoded in PARAM_DEFS from risk_config.py NIJ coefficients.
    # get_flat_risk_weights() used as fallback only for any None entries.
    rw_base = get_flat_risk_weights()
    for pdef in PARAM_DEFS:
        if pdef["mode"] == "risk_weight" and pdef["baseline"] is None:
            key = pdef["key"]
            if key in rw_base:
                pdef["baseline"] = float(rw_base[key])
                print(f"  ℹ️  Risk weight '{key}' baseline resolved from "
                      f"get_flat_risk_weights(): {pdef['baseline']}")
            else:
                pdef["baseline"] = 1.0
                print(f"  ⚠️  Risk weight key '{key}' not found — using 1.0")

    # ── Run or load each parameter sweep ─────────────────────────────────────
    all_results = []
    for pdef in PARAM_DEFS:
        cp = os.path.join(outdir, f"oat_{pdef['key']}.json")

        if not force_rerun and os.path.exists(cp):
            print(f"  ▶ {pdef['short']:8s} — loading checkpoint")
            with open(cp) as f:
                saved = json.load(f)
            saved["df"] = pd.DataFrame(saved["df_records"])
            del saved["df_records"]
            all_results.append(saved)
            continue

        print(f"\n  ▶ {pdef['short']:8s} — {pdef['label'].replace(chr(10), ' ')}  "
              f"(group: {pdef['group']})")
        result = run_oat_sweep(pdef, n_workers)
        all_results.append(result)

        # Checkpoint
        save = {k: v for k, v in result.items() if k != "df"}
        save["df_records"] = result["df"].to_dict(orient="records")
        with open(cp, "w") as f:
            json.dump(save, f, indent=2)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  OAT SENSITIVITY SUMMARY  (magnitude at ±20% perturbation)")
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

    # ── Generate charts ───────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Generating charts...")
    print(f"{'─'*70}")

    for win in [3, 6, 9]:
        plot_tornado(all_results, win, outdir)
    plot_lines(all_results, outdir)
    plot_heatmap(all_results, outdir)
    plot_group_summary(all_results, outdir)
    plot_full_report(all_results, outdir)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "n_params":   n_params,
        "n_levels":   n_levels,
        "n_runs":     n_total,
        "perturbations": PERTURBATIONS,
        "bjs_targets": CONFIG["bjs_targets"],
        "results": [
            {
                "key":   r["key"],
                "short": r["short"],
                "group": r["group"],
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

    print(f"\n  ✅  Summary JSON → {json_path}")
    print(f"  ✅  Charts → {outdir}/oat_*.png")
    print(f"\n  Charts produced:")
    for name in ["oat_tornado_3yr.png", "oat_tornado_6yr.png",
                 "oat_tornado_9yr.png", "oat_lines.png",
                 "oat_heatmap.png", "oat_group_summary.png",
                 "oat_full_report.png"]:
        print(f"    {name}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# REPLOT FROM CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

def replot_from_checkpoints(outdir: str = None):
    if outdir is None:
        outdir = CONFIG["output_directory"]

    # Resolve risk weight baselines first
    rw_base = get_flat_risk_weights()
    for pdef in PARAM_DEFS:
        if pdef["mode"] == "risk_weight" and pdef["baseline"] is None:
            pdef["baseline"] = float(rw_base.get(pdef["key"], 1.0))

    all_results = []
    for pdef in PARAM_DEFS:
        cp = os.path.join(outdir, f"oat_{pdef['key']}.json")
        if not os.path.exists(cp):
            print(f"  ⚠️  Checkpoint not found for {pdef['short']} — skipping")
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
    print("✅ All charts regenerated.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

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
            print(f"\n  ── Manual override: {n_workers} workers")
        except (IndexError, ValueError):
            print("Usage: python sensitivity_oat.py [--cores N] [--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  ── Core auto-detection: {reason}")

    print(f"  ── Starting OAT sensitivity analysis with {n_workers} workers...\n")
    main(n_workers=n_workers, force_rerun=force_rerun)