#!/usr/bin/env python3
"""
sensitivity_oat.py  —  One-At-a-Time (OAT) Sensitivity Analysis
================================================================
Recidivism ABM — PhD Dissertation Tool

BASELINE CALIBRATION (Final validated values)
----------------------------------------------
  Supervision_Monitoring_Intensity      1.050   identified — U-shaped minimum
  Risk_Effect_Decay_After_3Y            0.867   identified — interior minimum
  Risk_Effect_Decay_After_6Y            0.800   identified — interior minimum
  Supervision_Monitoring_Decay_After_3Y 0.933   identified — interior minimum
  Supervision_Monitoring_Decay_After_6Y 0.300   fixed — flat OAT surface
  Risk_Contrast_Strength                1.000   dual-constraint identified

PERTURBATION DESIGN
-------------------
  9 levels per parameter: -40%, -30%, -20%, -10%, 0% (baseline),
                          +10%, +20%, +30%, +40%
  For each level: N_REPS x N_SEEDS = 100 runs
  Total simulations: ~9,900

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
  python sensitivity_oat.py --cores 60
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
    "output_directory":      "oat_sensitivity_output_0414",
    "N_REPS":                10,
    "SEEDS":                 [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "bjs_targets":           {3: 0.68, 6: 0.79, 9: 0.83},
}

# ── Final validated calibration baseline ─────────────────────────────────────
'''
BASELINE_CAL = {
    "Supervision_Monitoring_Intensity":      1.050,
    "Risk_Effect_Decay_After_3Y":            0.867,
    "Risk_Effect_Decay_After_6Y":            0.800,
    "Supervision_Monitoring_Decay_After_3Y": 0.933,
    "Supervision_Monitoring_Decay_After_6Y": 0.300,
    "Risk_Contrast_Strength":                1.000,
}
'''
BASELINE_CAL = get_global_calibration_params()

# ── Perturbation levels ───────────────────────────────────────────────────────
PERTURBATIONS  = [-0.40, -0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30, +0.40]
PERTURB_LABELS = ["-40%", "-30%", "-20%", "-10%", "Baseline",
                  "+10%", "+20%", "+30%", "+40%"]

# ── Parameter definitions ─────────────────────────────────────────────────────
PARAM_DEFS = [
    # ── Stage 1 calibration parameters ──────────────────────────────────────
    {
        "key":         "Supervision_Monitoring_Intensity",
        "label":       "Supervision\nMonitoring Intensity (α)",
        "short":       "SMI (α)",
        "group":       "Stage 1 — Calibrated",
        "baseline":    1.050,
        "mode":        "multiply",
        "group_color": "#2E75B6",
        "description": "Extra detection chance\nwhile under supervision",
    },
    {
        "key":         "Risk_Effect_Decay_After_3Y",
        "label":       "Risk Effect\nDecay 3-6yr (δr3)",
        "short":       "RED3 (δr3)",
        "group":       "Stage 1 — Calibrated",
        "baseline":    0.867,
        "mode":        "multiply",
        "group_color": "#2E75B6",
        "description": "How much risk matters\nafter 3 years free",
    },
    {
        "key":         "Risk_Effect_Decay_After_6Y",
        "label":       "Risk Effect\nDecay 6-9yr (δr6)",
        "short":       "RED6 (δr6)",
        "group":       "Stage 1 — Calibrated",
        "baseline":    0.800,
        "mode":        "multiply",
        "group_color": "#2E75B6",
        "description": "Further risk reduction\nafter 6 years free",
    },
    {
        "key":         "Supervision_Monitoring_Decay_After_3Y",
        "label":       "Supervision Decay\n3-6yr (δs3)",
        "short":       "SMD3 (δs3)",
        "group":       "Stage 1 — Calibrated",
        "baseline":    0.933,
        "mode":        "multiply",
        "group_color": "#2E75B6",
        "description": "Reduces supervision\nintensity over time",
    },
    {
        "key":         "Supervision_Monitoring_Decay_After_6Y",
        "label":       "Supervision Decay\n6-9yr (δs6)",
        "short":       "SMD6 (δs6)*",
        "group":       "Stage 1 — Fixed",
        "baseline":    0.300,
        "mode":        "multiply",
        "group_color": "#D4AC0D",
        "description": "Strong supervision\nreduction after 6 years\n(*fixed — flat surface)",
    },
    # ── Stage 2 parameter ────────────────────────────────────────────────────
    {
        "key":         "Risk_Contrast_Strength",
        "label":       "Risk Contrast\nStrength (γ)",
        "short":       "RCS (γ)",
        "group":       "Stage 2 — Calibrated",
        "baseline":    1.000,
        "mode":        "add_fixed",
        "step":        0.25,
        "group_color": "#1E8449",
        "description": "How much risk score\nchanges arrest probability",
    },
    # ── Key risk weight parameters ───────────────────────────────────────────
    {
        "key":         "Age_at_Release",
        "label":       "Age at Release\nWeight",
        "short":       "Age",
        "group":       "Risk Weights",
        "baseline":    0.73,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
        "description": "Age effect on\nrearrest probability",
    },
    {
        "key":         "Prior_Revocations_Supervision",
        "label":       "Prior Revocations\nWeight",
        "short":       "Revocation",
        "group":       "Risk Weights",
        "baseline":    0.4338,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
        "description": "Prior revocation\nhistory effect",
    },
    {
        "key":         "Gang_Affiliated",
        "label":       "Gang Affiliation\nWeight",
        "short":       "Gang",
        "group":       "Risk Weights",
        "baseline":    0.84,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
        "description": "Gang membership\neffect on risk",
    },
    {
        "key":         "Percent_Days_Employed",
        "label":       "Employment\nWeight",
        "short":       "Employment",
        "group":       "Risk Weights",
        "baseline":    -0.95,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
        "description": "Employment reduces\nrearrest risk (negative)",
    },
    {
        "key":         "Program_Attendances",
        "label":       "Program Attendance\nWeight",
        "short":       "Program",
        "group":       "Risk Weights",
        "baseline":    -0.38,
        "mode":        "risk_weight",
        "group_color": "#ED7D31",
        "description": "Programme attendance\nreduces risk (negative)",
    },
]

# ── Colours ───────────────────────────────────────────────────────────────────
COLORS = {
    "bjs":      "#1A3D5C",
    "baseline": "#555555",
    "up":       "#C00000",
    "down":     "#2E75B6",
    "grid":     "#E8E8E8",
    "bg":       "#FAFBFC",
    "3yr":      "#2E75B6",
    "6yr":      "#70AD47",
    "9yr":      "#ED7D31",
}

GROUP_COLORS = {
    "Stage 1 — Calibrated": "#2E75B6",
    "Stage 1 — Fixed":      "#D4AC0D",
    "Stage 2 — Calibrated": "#1E8449",
    "Risk Weights":         "#ED7D31",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    COLORS["bg"],
    "figure.facecolor":  COLORS["bg"],
    "grid.color":        COLORS["grid"],
    "grid.linewidth":    0.55,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titleweight":  "bold",
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})


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
            v = max(0.0, v)
        elif mode == "add_fixed":
            step  = pdef.get("step", 0.25)
            level = round(pct / 0.10)
            v     = base + level * step
            v     = max(0.0, v)
        elif mode == "risk_weight":
            v = base * (1.0 + pct)
        vals.append(v)
    return vals


# =============================================================================
# WORKER
# =============================================================================

def _worker(task: tuple) -> tuple:
    payload_json, seed = task
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
            initial_agents        = agents,
            bias_factor           = CONFIG["bias_factor"],
            monthly_intake        = intake,
            warmup_months         = warmup,
            study_months          = study,
            enable_peer_influence = CONFIG["enable_peer_influence"],
            seed                  = seed,
            weights               = rw,
            calibration_params    = cal,
        )
        model.export_csv = False

        while model.running:
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
# PARALLEL BATCH RUNNER
# =============================================================================

def _run_batch(overrides_list: list, n_workers: int, desc: str = "Running") -> dict:
    seeds  = CONFIG["SEEDS"]
    n_reps = CONFIG["N_REPS"]

    tasks = []
    for task_id, payload in overrides_list:
        d  = dict(payload)
        d["__task_id__"] = float(task_id)
        pj = json.dumps(d)
        for rep in range(n_reps):
            for seed in seeds:
                tasks.append((pj, seed + rep * 10_000))

    raw = {float(tid): {3: [], 6: [], 9: []} for tid, _ in overrides_list}

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
        payload = {
            "__warmup__": CONFIG["warmup_months"],
            "__study__":  CONFIG["study_months"],
            "__agents__": CONFIG["initial_agents"],
            "__intake__": CONFIG["monthly_intake"],
        }
        if mode in ("multiply", "add_fixed"):
            payload[key] = float(v)
        elif mode == "risk_weight":
            payload["__risk_weights__"] = {key: float(v)}
        overrides_list.append((float(i), payload))

    raw = _run_batch(overrides_list, n_workers,
                     desc=f"{pdef['short'][:18]:18s} OAT")

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
    if title:  ax.set_title(title, fontsize=11, fontweight="bold",
                             pad=8, color="#1A3D5C")
    ax.grid(True, axis="both", zorder=0)
    ax.tick_params(labelsize=9)


def _draw_pill_labels(ax, rows, fontsize=9):
    """Coloured pill labels replacing plain yticklabels."""
    ax.set_yticklabels([""] * len(rows))
    for i, row in enumerate(rows):
        ax.text(
            -0.003, i,
            f"  {row['short']}  ",
            va="center", ha="right",
            fontsize=fontsize,
            color="white", fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=row["color"],
                edgecolor="none", alpha=0.92,
            ),
            transform=ax.get_yaxis_transform(),
            clip_on=False, zorder=6,
        )


def _tornado_legends(ax):
    bar_patches = [
        mpatches.Patch(color=COLORS["up"],   alpha=0.85,
                       label="+40% perturbation"),
        mpatches.Patch(color=COLORS["down"], alpha=0.85,
                       label="-40% perturbation"),
    ]
    grp_patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in GROUP_COLORS.items()
    ]
    leg1 = ax.legend(handles=bar_patches, fontsize=8.5,
                     loc="lower right", framealpha=0.9,
                     title="Perturbation direction", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=grp_patches, fontsize=8.5,
              loc="upper right", framealpha=0.9,
              title="Parameter group", title_fontsize=8)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"    -> {os.path.basename(path)}")


# =============================================================================
# CHART 1 — Tornado Charts (per window)
# =============================================================================

def plot_tornado(all_results: list, window: int, outdir: str) -> str:
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
        dp   = float(pos40[0]) - base
        dn   = float(neg40[0]) - base
        m40  = abs(dp - dn) / 2.0
        m20  = (abs(float(pos20[0]) - float(neg20[0])) / 2.0
                if len(pos20) > 0 and len(neg20) > 0 else 0.0)
        rows.append({
            "short":      r["short"],
            "full_label": r["label"].replace("\n", " "),
            "group":      r["group"],
            "color":      r["color"],
            "delta_pos":  dp,
            "delta_neg":  dn,
            "mag40":      m40,
            "mag20":      m20,
        })

    rows.sort(key=lambda x: x["mag40"])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(16, max(6, n * 0.68 + 2.0)))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.subplots_adjust(left=0.30)

    for i, row in enumerate(rows):
        c_pos = COLORS["up"]   if row["delta_pos"] >= 0 else COLORS["down"]
        c_neg = COLORS["down"] if row["delta_neg"] <= 0 else COLORS["up"]
        ax.barh(i, row["delta_pos"], color=c_pos, alpha=0.85,
                height=0.58, zorder=3,
                label="+40%" if i == 0 else "")
        ax.barh(i, row["delta_neg"], color=c_neg, alpha=0.85,
                height=0.58, zorder=3)

        ext = max(abs(row["delta_pos"]), abs(row["delta_neg"]))
        ax.text(ext + 0.0008, i,
                f"±40%: {row['mag40']:.4f}   ±20%: {row['mag20']:.4f}",
                va="center", fontsize=8, color="#444444")

    ax.set_yticks(range(n))
    _draw_pill_labels(ax, rows, fontsize=9)
    ax.axvline(0, color="#333333", linewidth=1.3, zorder=4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1%}"))
    _tornado_legends(ax)

    bjs_t = targets[window]
    win_color = COLORS[f"{window}yr"]
    ax.set_title(
        f"Tornado Chart — {window}-Year Rearrest Rate Sensitivity  "
        f"(BJS target = {bjs_t:.0%})\n"
        f"Bars = change from ±40% perturbation | "
        f"Sorted by magnitude | Label colour = parameter group",
        fontsize=12, fontweight="bold", pad=12, color="#1A3D5C",
    )
    ax.set_xlabel(
        f"Change in {window}-Year Cumulative Rearrest Rate from Calibrated Baseline",
        fontsize=10,
    )

    # BJS annotation
    ax.text(0.01, 0.02,
            f"Calibrated baseline: {bjs_t:.0%} (BJS NCJ 250975)",
            transform=ax.transAxes, fontsize=8.5, color="#1A3D5C",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(outdir, f"oat_tornado_{window}yr.png")
    _save(fig, path)
    return path


# =============================================================================
# CHART 2 — Response Curves (line plots per parameter)
# =============================================================================

def plot_lines(all_results: list, outdir: str) -> str:
    n_params  = len(all_results)
    n_cols    = 4
    n_rows    = int(np.ceil(n_params / n_cols))
    targets   = CONFIG["bjs_targets"]
    x         = np.array(PERTURBATIONS) * 100
    win_cfg   = [(3, COLORS["3yr"], "o"), (6, COLORS["6yr"], "s"),
                 (9, COLORS["9yr"], "^")]

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.5, n_rows * 3.8))
    fig.patch.set_facecolor(COLORS["bg"])
    axes_flat = axes.flatten()

    for idx, r in enumerate(all_results):
        ax = axes_flat[idx]
        df = r["df"]

        for win, clr, mkr in win_cfg:
            means = df[f"mean_{win}yr"].values
            cis   = df[f"ci95_{win}yr"].values
            ax.plot(x, means, color=clr, linewidth=2.2,
                    marker=mkr, markersize=5,
                    label=f"{win}yr", zorder=3)
            ax.fill_between(x, means - cis, means + cis,
                            color=clr, alpha=0.12, zorder=2)
            ax.axhline(targets[win], color=clr, linewidth=0.9,
                       linestyle=":", alpha=0.7)

        ax.axvline(0, color="#888888", linewidth=1.2,
                   linestyle="--", zorder=2)

        # Colour strip on left = group identity
        ax.axvspan(x[0]-5, x[0]-1, color=r["color"], alpha=0.7,
                   clip_on=False)

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlim(x[0]-4, x[-1]+4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v):+d}%" for v in x], fontsize=7.5,
                           rotation=45, ha="right")
        ax.set_title(
            r["label"].replace("\n", " "),
            fontsize=9.5, fontweight="bold", color="#1A3D5C", pad=5,
        )

        # Baseline value annotation
        base_3 = r["baseline"][3]
        ax.annotate(f"Base: {base_3:.1%}",
                    xy=(0, base_3),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=7.5, color="#555555",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8))

        ax.grid(True, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0],[0], color=COLORS["3yr"], lw=2, marker="o",
               markersize=5, label="3-Year Rate"),
        Line2D([0],[0], color=COLORS["6yr"], lw=2, marker="s",
               markersize=5, label="6-Year Rate"),
        Line2D([0],[0], color=COLORS["9yr"], lw=2, marker="^",
               markersize=5, label="9-Year Rate"),
        Line2D([0],[0], color="#888888", lw=1.2, linestyle="--",
               label="Baseline (0% perturbation)"),
        Line2D([0],[0], color="#888888", lw=1.2,
               linestyle=(0, (1, 1)),
               label="BJS targets (dotted)"),
    ]

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "OAT Sensitivity — Response Curves Per Parameter\n"
        "Each panel: rearrest rate vs perturbation level  |  "
        "Shaded band = 95% CI (n=100 runs)  |  "
        "Dotted lines = BJS targets  |  "
        "Left colour strip = parameter group",
        fontsize=13, fontweight="bold", y=1.01, color="#1A3D5C",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(outdir, "oat_lines.png")
    _save(fig, path)
    return path


# =============================================================================
# CHART 3 — Sensitivity Heatmap
# =============================================================================

def plot_heatmap(all_results: list, outdir: str) -> str:
    windows = [3, 6, 9]
    labels  = [r["short"] for r in all_results]
    groups  = [r["group"] for r in all_results]
    colors  = [r["color"] for r in all_results]

    matrix = np.zeros((len(all_results), 3))
    for i, r in enumerate(all_results):
        for j, w in enumerate(windows):
            matrix[i, j] = sensitivity_magnitude(r, w, pct=0.40)

    fig, ax = plt.subplots(figsize=(9, max(6, len(all_results)*0.58+2)))
    fig.patch.set_facecolor(COLORS["bg"])

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=max(matrix.max()*1.05, 0.001))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["3-Year", "6-Year", "9-Year"],
                       fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([""] * len(labels))

    # Coloured pill labels on y-axis
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax.text(-0.003, i, f"  {lbl}  ",
                va="center", ha="right", fontsize=9,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col,
                          edgecolor="none", alpha=0.92),
                transform=ax.get_yaxis_transform(),
                clip_on=False, zorder=6)

    # Cell values
    for i in range(len(all_results)):
        for j in range(3):
            val = matrix[i, j]
            tc  = "white" if val > matrix.max() * 0.62 else "#222222"
            ax.text(j, i, f"{val:.4f}",
                    ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")

    # Group dividers
    prev_grp = None
    for i, grp in enumerate(groups):
        if grp != prev_grp and i > 0:
            ax.axhline(i - 0.5, color="white", linewidth=2.5)
        prev_grp = grp

    cbar = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label(
        "Sensitivity Magnitude\n|mean(+40%) − mean(−40%)| / 2",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Sensitivity Heatmap — All Parameters × All Windows\n"
        "Deeper red = larger effect on rearrest rate | "
        "Values = magnitude at ±40% perturbation",
        fontsize=11, fontweight="bold", pad=12, color="#1A3D5C",
    )

    # Group legend on right
    grp_patches = [mpatches.Patch(color=c, label=g)
                   for g, c in GROUP_COLORS.items()]
    ax.legend(handles=grp_patches, fontsize=8, loc="lower right",
              framealpha=0.9, title="Parameter group", title_fontsize=8,
              bbox_to_anchor=(1.35, 0))

    fig.subplots_adjust(left=0.28)
    plt.tight_layout()
    path = os.path.join(outdir, "oat_heatmap.png")
    _save(fig, path)
    return path


# =============================================================================
# CHART 4 — Group Summary
# =============================================================================

def plot_group_summary(all_results: list, outdir: str) -> str:
    windows = [3, 6, 9]
    groups  = list(GROUP_COLORS.keys())
    targets = CONFIG["bjs_targets"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    for ax, win in zip(axes, windows):
        group_means, group_errs, group_ns = [], [], []
        for grp in groups:
            mags = [sensitivity_magnitude(r, win, 0.40)
                    for r in all_results if r["group"] == grp]
            group_means.append(np.mean(mags) if mags else 0.0)
            group_errs.append(np.std(mags) if len(mags) > 1 else 0.0)
            group_ns.append(len(mags))

        clrs = [GROUP_COLORS[g] for g in groups]
        xpos = np.arange(len(groups))

        bars = ax.bar(xpos, group_means, color=clrs,
                      edgecolor="white", width=0.58,
                      zorder=3, alpha=0.88)
        ax.errorbar(xpos, group_means, yerr=group_errs,
                    fmt="none", color="#333333",
                    elinewidth=2.0, capsize=7, capthick=1.8, zorder=4)

        for xi, (m, e, n) in enumerate(
                zip(group_means, group_errs, group_ns)):
            ax.text(xi, m + e + 0.0003,
                    f"{m:.4f}\n(n={n} params)",
                    ha="center", fontsize=8.5,
                    fontweight="bold", color="#333333")

        # BJS target reference line
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")

        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [g.replace(" — ", "\n") for g in groups],
            fontsize=8.5,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylabel("Mean Sensitivity Magnitude (pp)", fontsize=10)
        win_clr = COLORS[f"{win}yr"]
        ax.set_title(
            f"{win}-Year Rate Sensitivity by Parameter Group\n"
            f"BJS target = {targets[win]:.0%}",
            fontsize=11, fontweight="bold", color="#1A3D5C",
        )
        ax.grid(True, axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Group-Level Sensitivity Summary\n"
        "Mean |±40% response| per parameter group  |  "
        "Error bars = 1 SD across parameters within group",
        fontsize=13, fontweight="bold", y=1.02, color="#1A3D5C",
    )
    plt.tight_layout()
    path = os.path.join(outdir, "oat_group_summary.png")
    _save(fig, path)
    return path


# =============================================================================
# CHART 5 — Full Dissertation Report (composite)
# =============================================================================

def plot_full_report(all_results: list, outdir: str) -> str:
    windows = [3, 6, 9]
    targets = CONFIG["bjs_targets"]

    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(COLORS["bg"])

    gs = gridspec.GridSpec(
        2, 3, hspace=0.48, wspace=0.38,
        left=0.20, right=0.97, top=0.90, bottom=0.06,
    )

    # ── Panel A: Tornado (9yr) ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])

    rows_t = []
    for r in all_results:
        df    = r["df"]
        pos40 = df[df["pct"] ==  0.40]["mean_9yr"].values
        neg40 = df[df["pct"] == -0.40]["mean_9yr"].values
        pos20 = df[df["pct"] ==  0.20]["mean_9yr"].values
        neg20 = df[df["pct"] == -0.20]["mean_9yr"].values
        base  = r["baseline"][9]
        if len(pos40) == 0 or len(neg40) == 0:
            continue
        dp  = float(pos40[0]) - base
        dn  = float(neg40[0]) - base
        m40 = abs(dp - dn) / 2.0
        m20 = (abs(float(pos20[0]) - float(neg20[0])) / 2.0
               if len(pos20) > 0 and len(neg20) > 0 else 0.0)
        rows_t.append({
            "short":      r["short"],
            "full_label": r["label"].replace("\n", " "),
            "group":      r["group"],
            "color":      r["color"],
            "delta_pos":  dp,
            "delta_neg":  dn,
            "mag40":      m40,
            "mag20":      m20,
        })

    rows_t.sort(key=lambda x: x["mag40"])

    for i, row in enumerate(rows_t):
        c_pos = COLORS["up"]   if row["delta_pos"] >= 0 else COLORS["down"]
        c_neg = COLORS["down"] if row["delta_neg"] <= 0 else COLORS["up"]
        ax_a.barh(i, row["delta_pos"], color=c_pos,
                  alpha=0.85, height=0.58, zorder=3)
        ax_a.barh(i, row["delta_neg"], color=c_neg,
                  alpha=0.85, height=0.58, zorder=3)
        ext = max(abs(row["delta_pos"]), abs(row["delta_neg"]))
        ax_a.text(ext + 0.0005, i,
                  f"±40%:{row['mag40']:.4f}  ±20%:{row['mag20']:.4f}",
                  va="center", fontsize=7.5, color="#444444")

    ax_a.set_yticks(range(len(rows_t)))
    _draw_pill_labels(ax_a, rows_t, fontsize=8.5)
    ax_a.axvline(0, color="#333333", linewidth=1.3, zorder=4)
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1%}"))
    ax_a.grid(True, zorder=0)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    _tornado_legends(ax_a)
    ax_a.set_title(
        f"A — Tornado: 9-Year Rate Sensitivity  "
        f"(BJS target = {targets[9]:.0%})\n"
        "Sorted by magnitude | ±40% perturbation of each parameter",
        fontsize=11, fontweight="bold", color="#1A3D5C", pad=10,
    )
    ax_a.set_xlabel("Change in 9-Year Rate from Calibrated Baseline",
                    fontsize=10)

    # ── Panel B: Heatmap ──────────────────────────────────────────────────────
    ax_b   = fig.add_subplot(gs[0, 2])
    labels = [r["short"] for r in all_results]
    clrs_h = [r["color"] for r in all_results]
    matrix = np.zeros((len(all_results), 3))
    for i, r in enumerate(all_results):
        for j, w in enumerate(windows):
            matrix[i, j] = sensitivity_magnitude(r, w, 0.40)

    im = ax_b.imshow(matrix, aspect="auto", cmap="YlOrRd",
                     vmin=0, vmax=max(matrix.max()*1.05, 0.001))
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["3yr", "6yr", "9yr"],
                         fontsize=10, fontweight="bold")
    ax_b.set_yticks(range(len(labels)))
    ax_b.set_yticklabels([""] * len(labels))

    for i, (lbl, col) in enumerate(zip(labels, clrs_h)):
        ax_b.text(-0.003, i, f"  {lbl}  ",
                  va="center", ha="right", fontsize=8,
                  color="white", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25",
                            facecolor=col, edgecolor="none", alpha=0.92),
                  transform=ax_b.get_yaxis_transform(),
                  clip_on=False, zorder=6)

    for i in range(len(all_results)):
        for j in range(3):
            val = matrix[i, j]
            tc  = "white" if val > matrix.max()*0.62 else "#222222"
            ax_b.text(j, i, f"{val:.3f}",
                      ha="center", va="center",
                      fontsize=8, color=tc, fontweight="bold")

    plt.colorbar(im, ax=ax_b, shrink=0.72,
                 pad=0.02).ax.tick_params(labelsize=7)
    ax_b.set_title("B — Heatmap: All Params × All Windows\n"
                   "Values = magnitude at ±40%",
                   fontsize=10, fontweight="bold",
                   color="#1A3D5C", pad=8)

    # ── Panels C/D/E: Group summaries ─────────────────────────────────────────
    letters = ["C", "D", "E"]
    for col_i, (win, letter) in enumerate(zip(windows, letters)):
        ax = fig.add_subplot(gs[1, col_i])
        grps = list(GROUP_COLORS.keys())
        means, errs = [], []
        for grp in grps:
            mags = [sensitivity_magnitude(r, win, 0.40)
                    for r in all_results if r["group"] == grp]
            means.append(np.mean(mags) if mags else 0.0)
            errs.append(np.std(mags) if len(mags) > 1 else 0.0)

        clrs = [GROUP_COLORS[g] for g in grps]
        xp   = np.arange(len(grps))
        ax.bar(xp, means, color=clrs, edgecolor="white",
               width=0.58, zorder=3, alpha=0.88)
        ax.errorbar(xp, means, yerr=errs, fmt="none",
                    color="#333333", elinewidth=1.5,
                    capsize=5, zorder=4)
        for xi, (m, e) in enumerate(zip(means, errs)):
            ax.text(xi, m + e + 0.0001, f"{m:.4f}",
                    ha="center", fontsize=8, color="#333333",
                    fontweight="bold")
        ax.set_xticks(xp)
        ax.set_xticklabels(
            [g.replace(" — ", "\n") for g in grps], fontsize=7.5,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylabel("Mean Sensitivity", fontsize=9)
        ax.set_title(
            f"{letter} — {win}-Year Group Sensitivity\n"
            f"BJS target = {targets[win]:.0%}",
            fontsize=10, fontweight="bold", color="#1A3D5C",
        )
        ax.grid(True, axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "OAT Sensitivity Analysis — Recidivism ABM\n"
        "One-At-a-Time perturbation ±10/20/30/40%  |  "
        "n=100 runs per condition  |  "
        "Calibrated baseline: α=1.050, δr3=0.867, δr6=0.800, "
        "δs3=0.933, δs6=0.300, γ=1.000",
        fontsize=13, fontweight="bold", y=0.975, color="#1A3D5C",
    )

    path = os.path.join(outdir, "oat_full_report.png")
    _save(fig, path)
    return path


# =============================================================================
# MAIN
# =============================================================================

def main(n_workers: int, force_rerun: bool = False):
    outdir   = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)

    n_total  = CONFIG["N_REPS"] * len(CONFIG["SEEDS"])
    n_params = len(PARAM_DEFS)
    n_levels = len(PERTURBATIONS)

    print("=" * 70)
    print("  RECIDIVISM ABM — OAT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"  Workers        : {n_workers}")
    print(f"  Parameters     : {n_params}  "
          f"(6 calibration + 5 risk weights)")
    print(f"  Levels/param   : {n_levels}  "
          f"({', '.join(PERTURB_LABELS)})")
    print(f"  Runs/level     : {n_total}  "
          f"({CONFIG['N_REPS']} reps × {len(CONFIG['SEEDS'])} seeds)")
    print(f"  Est. sims      : ~{n_params*n_levels*n_total:,}")
    print(f"  Output         : {outdir}")
    print()
    print("  Final calibrated baseline:")
    for k, v in BASELINE_CAL.items():
        print(f"    {k:<48} {v}")
    print("=" * 70)

    all_results = []
    for pdef in PARAM_DEFS:
        cp = os.path.join(outdir, f"oat_{pdef['key']}.json")
        if not force_rerun and os.path.exists(cp):
            print(f"  > {pdef['short']:<20} loading checkpoint")
            with open(cp) as f:
                saved = json.load(f)
            saved["df"] = pd.DataFrame(saved["df_records"])
            del saved["df_records"]
            # JSON converts int keys to strings — restore int keys
            if "baseline" in saved:
                saved["baseline"] = {
                    int(k): v for k, v in saved["baseline"].items()
                }
            all_results.append(saved)
            continue

        print(f"\n  > {pdef['short']:<20} "
              f"{pdef['label'].replace(chr(10), ' ')}  "
              f"[{pdef['group']}]")
        result = run_oat_sweep(pdef, n_workers)
        all_results.append(result)

        save = {k: v for k, v in result.items() if k != "df"}
        save["df_records"] = result["df"].to_dict(orient="records")
        with open(cp, "w") as f:
            json.dump(save, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print("  OAT SENSITIVITY SUMMARY  (magnitude at ±40% perturbation)")
    print(f"{'='*70}")
    print(f"  {'Parameter':<22} {'Group':<24} "
          f"{'3yr':>8} {'6yr':>8} {'9yr':>8}")
    print(f"  {'-'*70}")
    for r in sorted(all_results,
                    key=lambda x: sensitivity_magnitude(x, 9, 0.40),
                    reverse=True):
        s3 = sensitivity_magnitude(r, 3, 0.40)
        s6 = sensitivity_magnitude(r, 6, 0.40)
        s9 = sensitivity_magnitude(r, 9, 0.40)
        print(f"  {r['short']:<22} {r['group']:<24} "
              f"{s3:>8.4f} {s6:>8.4f} {s9:>8.4f}")

    print(f"\n{'─'*70}")
    print("  Generating dissertation charts...")

    for win in [3, 6, 9]:
        plot_tornado(all_results, win, outdir)
    plot_lines(all_results, outdir)
    plot_heatmap(all_results, outdir)
    plot_group_summary(all_results, outdir)
    plot_full_report(all_results, outdir)

    # Save summary JSON
    summary = {
        "baseline_cal":  BASELINE_CAL,
        "bjs_targets":   CONFIG["bjs_targets"],
        "n_runs":        n_total,
        "perturbations": PERTURBATIONS,
        "results": [
            {
                "key":             r["key"],
                "short":           r["short"],
                "group":           r["group"],
                "sensitivity_3yr": sensitivity_magnitude(r, 3, 0.40),
                "sensitivity_6yr": sensitivity_magnitude(r, 6, 0.40),
                "sensitivity_9yr": sensitivity_magnitude(r, 9, 0.40),
                "baseline_3yr":    r["baseline"][3],
                "baseline_6yr":    r["baseline"][6],
                "baseline_9yr":    r["baseline"][9],
            }
            for r in all_results
        ],
    }
    with open(os.path.join(outdir, "oat_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Charts -> {outdir}/oat_*.png")
    print(f"  Summary -> {outdir}/oat_summary.json")
    return summary


# =============================================================================
# REPLOT FROM CHECKPOINTS
# =============================================================================
def replot_from_checkpoints(outdir: str = None):
    if outdir is None:
        outdir = CONFIG["output_directory"]
    all_results = []
    for pdef in PARAM_DEFS:
        cp = os.path.join(outdir, f"oat_{pdef['key']}.json")
        if not os.path.exists(cp):
            print(f"  Checkpoint not found: {pdef['short']}")
            continue
        with open(cp) as f:
            saved = json.load(f)
        saved["df"] = pd.DataFrame(saved["df_records"])
        del saved["df_records"]

        # JSON converts int keys to strings — restore int keys
        if "baseline" in saved:
            saved["baseline"] = {
                int(k): v for k, v in saved["baseline"].items()
            }

        all_results.append(saved)

    if not all_results:
        print("No checkpoints found. Run without --replot first.")
        return

    for win in [3, 6, 9]:
        plot_tornado(all_results, win, outdir)
    plot_lines(all_results, outdir)
    plot_heatmap(all_results, outdir)
    plot_group_summary(all_results, outdir)
    plot_full_report(all_results, outdir)
    print("All charts regenerated.")



# =============================================================================
# CORE DETECTION + ENTRY POINT
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
        except (IndexError, ValueError):
            print("Usage: python sensitivity_oat.py "
                  "[--cores N] [--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  Core detection: {reason}")

    print(f"  Starting with {n_workers} workers...\n")
    main(n_workers=n_workers, force_rerun=force_rerun)