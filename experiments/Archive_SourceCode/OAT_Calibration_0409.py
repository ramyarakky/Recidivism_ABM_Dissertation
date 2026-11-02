"""
calibrate.py  —  Sequential One-At-A-Time (OAT) Calibration  [PARALLEL]
=========================================================================
Recidivism ABM — PhD Dissertation Tool

CALIBRATION ORDER (6 parameters)
-----------------
Stage 1 — Aggregate BJS targets (68% / 79% / 83%)
  Step 1: Supervision_Monitoring_Intensity       → drives 3yr gap (primary lever)
  Step 2: Risk_Effect_Decay_After_3Y             → shapes hazard curve years 3–6
  Step 3: Risk_Effect_Decay_After_6Y             → shapes hazard curve years 6–9
  Step 4: Supervision_Monitoring_Decay_After_3Y  → supervision boost decay 3–6yr
  Step 5: Supervision_Monitoring_Decay_After_6Y  → supervision boost decay 6–9yr (fixed)

Stage 2 — PCRA tier-stratified targets (4 tiers × 3 windows)
  Step 6: Risk_Contrast_Strength                 → tier spread (γ)

CHART FIXES APPLIED
--------------------
  1. plot_recidivism_comparison — y-axis starts at 0.40, ±2pp shaded band added
  2. plot_final_summary         — right panel replaced with parameter table
  3. plot_per_param_comparison_summary — figure size and font sizes increased
  4. plot_calibration_convergence — NEW: OAT sequence visualisation
  5. plot_seed_variability       — NEW: box plots across seeds at final params
  6. _worker                     — export_csv, _export_warmup_csv, _silent all disabled
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
from matplotlib.ticker import PercentFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_global_calibration_params,
    get_uncalibrated_params,
    get_flat_risk_weights,
    get_stage2_sweep_params,
    _deep_merge,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "initial_agents":        1000,
    "warmup_months":         144,
    "study_months":          108,
    "monthly_intake":        10,
    "bias_factor":           0.0,
    "enable_peer_influence": True,
    "mode":                  "realistic",
    "output_directory":      "calibration_output_0330",
    "N_REPS":                10,
    "SEEDS":                 [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "bjs_targets":           {3: 0.68, 6: 0.79, 9: 0.83},
}

PARAM_RANGES = {
    # α needs a lower ceiling — your model now has less headroom to fill
    "Supervision_Monitoring_Intensity": np.linspace(0.80, 1.30, 11),
    
    # Decay params — desistance means values < 1.0, sweep lower
    "Risk_Effect_Decay_After_3Y":            np.linspace(0.20, 1.00, 13),
    "Risk_Effect_Decay_After_6Y":            np.linspace(0.10, 1.00, 13),
    "Supervision_Monitoring_Decay_After_3Y": np.linspace(0.20, 1.00, 13),
    "Supervision_Monitoring_Decay_After_6Y": np.linspace(0.10, 0.60, 6),
    
    # γ — extend ceiling since tier distribution is now wider
   "Risk_Contrast_Strength": np.linspace(0.5, 3.0, 11),
}

PARAM_PRIMARY_WINDOW = {
    "Supervision_Monitoring_Intensity":      [3, 6, 9],
    "Risk_Effect_Decay_After_3Y":            [6],
    "Risk_Effect_Decay_After_6Y":            [9],
    "Supervision_Monitoring_Decay_After_3Y": [6],
    "Supervision_Monitoring_Decay_After_6Y": [9],
    "Risk_Contrast_Strength":                [3, 6, 9],
}

PARAM_LABELS = {
    "Supervision_Monitoring_Intensity":      "Supervision Monitoring Intensity (α)",
    "Risk_Effect_Decay_After_3Y":            "Risk Effect Decay 3–6 yr (δᵣ₃)",
    "Risk_Effect_Decay_After_6Y":            "Risk Effect Decay 6–9 yr (δᵣ₆)",
    "Supervision_Monitoring_Decay_After_3Y": "Supv. Monitoring Decay 3–6 yr (δₛ₃)",
    "Supervision_Monitoring_Decay_After_6Y": "Supv. Monitoring Decay 6–9 yr (δₛ₆)",
    "Risk_Contrast_Strength":                "Risk Contrast Strength (γ)  [Stage 2 — PCRA tiers]",
}

PARAM_GROUNDING = {
    "Supervision_Monitoring_Intensity":
        "Langan & Levin (2002): 30% rearrest\nwithin 6 months of supervision",
    "Risk_Effect_Decay_After_3Y":
        "Sampson & Laub (2003);\nKurlychek et al. (2006)",
    "Risk_Effect_Decay_After_6Y":
        "Sampson & Laub (2003);\nBushway et al. (2011)",
    "Supervision_Monitoring_Decay_After_3Y":
        "Petersilia (2003): declining\nmonitoring intensity over time",
    "Supervision_Monitoring_Decay_After_6Y":
        "Fixed: flat OAT surface;\n<15% agents supervised past yr 6",
    "Risk_Contrast_Strength":
        "PCRA Table 6, Federal\nProbation (Sept 2023)",
}

CALIBRATION_ORDER = [
    "Supervision_Monitoring_Intensity",
    "Risk_Effect_Decay_After_3Y",
    "Risk_Effect_Decay_After_6Y",
    "Supervision_Monitoring_Decay_After_3Y",
    "Supervision_Monitoring_Decay_After_6Y",
    "Risk_Contrast_Strength",
]

PARAM_SHORT_NAMES = [
    "Supv. Intensity (α)",
    "Risk Decay 3–6yr (δᵣ₃)",
    "Risk Decay 6–9yr (δᵣ₆)",
    "Supv. Decay 3–6yr (δₛ₃)",
    "Supv. Decay 6–9yr (δₛ₆)",
    "Risk Contrast (γ)",
]

PARAM_SYMBOLS = ["α", "δᵣ₃", "δᵣ₆", "δₛ₃", "δₛ₆", "γ"]
PARAM_STAGES  = ["1", "1", "1", "1", "1*", "2"]
PARAM_WINDOWS = ["3yr", "6yr", "9yr", "6yr", "9yr", "PCRA"]

assert len(CALIBRATION_ORDER) == len(PARAM_SHORT_NAMES)

STAGE2_PARAMS = {"Risk_Contrast_Strength"}

COLORS = {
    "bjs":        "#1A3D5C",
    "baseline":   "#AAAAAA",
    "calibrated": "#E05C2A",
    "3yr":        "#2E75B6",
    "6yr":        "#70AD47",
    "9yr":        "#ED7D31",
    "loss_all":   "#1A3D5C",
    "loss_3yr":   "#2E75B6",
    "loss_6yr":   "#70AD47",
    "loss_9yr":   "#ED7D31",
    "optimal":    "#C00000",
    "grid":       "#E0E0E0",
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_baseline(res: dict) -> dict:
    out = {}
    for k, v in res.items():
        try:
            out[int(k)] = v
        except (ValueError, TypeError):
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(task: tuple) -> tuple:
    (cal_json, seed, warmup, study, agents,
     intake, mode, bias, peer, outdir) = task

    calibration_overrides = json.loads(cal_json)
    param_value = calibration_overrides.get("__task_value__", 0.0)

    cal = _deep_merge(
        get_uncalibrated_params(),
        {k: v for k, v in calibration_overrides.items() if not k.startswith("__")},
    )

    try:
        model = RecidivismModel(
            initial_agents=agents,
            bias_factor=bias,
            monthly_intake=intake,
            warmup_months=warmup,
            study_months=study,
            enable_peer_influence=peer,
            weights=get_flat_risk_weights(),
            seed=seed,
            mode=mode,
            output_directory=os.path.join(outdir, "tmp"),
            calibration_params=cal,
        )

        # FIX: disable ALL file I/O in parallel workers
        model.export_csv         = False
        #model._export_warmup_csv = False
        #model._silent            = True

        for _ in range(warmup + study):
            if not model.running:
                break
            model.step()

        rates = {}
        for yrs in [3, 6, 9]:
            r = model.calculate_flag_rate(f"rearrest_{yrs}_yrs")
            rates[yrs] = r if r is not None else 0.0

        try:
            gamma_loss = model.compute_gamma_loss()
            if gamma_loss is None or (isinstance(gamma_loss, float) and np.isnan(gamma_loss)):
                gamma_loss = 999.0
        except Exception:
            gamma_loss = 999.0

        return (param_value, seed, rates[3], rates[6], rates[9], gamma_loss)

    except Exception as e:
        print(f"  ⚠️  Worker error (value={param_value:.4f}, seed={seed}): {e}", flush=True)
        return (param_value, seed, 0.0, 0.0, 0.0, 999.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL SWEEP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_tasks(param_name: str, locked_params: dict) -> list:
    values = PARAM_RANGES[param_name]
    seeds  = CONFIG["SEEDS"]
    n_reps = CONFIG["N_REPS"]
    warmup = CONFIG["warmup_months"]
    study  = CONFIG["study_months"]
    agents = CONFIG["initial_agents"]
    intake = CONFIG["monthly_intake"]
    mode   = CONFIG["mode"]
    bias   = CONFIG["bias_factor"]
    peer   = CONFIG["enable_peer_influence"]
    outdir = CONFIG["output_directory"]

    tasks = []
    for v in values:
        overrides = dict(locked_params)
        overrides[param_name]       = float(v)
        overrides["__task_value__"] = float(v)
        cal_json = json.dumps(overrides)
        for rep in range(n_reps):
            for seed in seeds:
                effective_seed = seed + rep * 10_000
                tasks.append((cal_json, effective_seed, warmup, study, agents,
                              intake, mode, bias, peer, outdir))
    return tasks


def sweep_parameter_parallel(param_name: str, locked_params: dict,
                               n_workers: int) -> pd.DataFrame:
    targets  = CONFIG["bjs_targets"]
    values   = PARAM_RANGES[param_name]
    seeds    = CONFIG["SEEDS"]
    n_reps   = CONFIG["N_REPS"]
    primary  = PARAM_PRIMARY_WINDOW.get(param_name, [3, 6, 9])
    n_tasks  = len(values) * n_reps * len(seeds)

    stage_label = "Stage 2 — PCRA tier loss" if param_name in STAGE2_PARAMS \
                  else "Stage 1 — BJS aggregate loss"
    print(f"\n  Sweeping  {PARAM_LABELS[param_name]}")
    print(f"  {stage_label}")
    print(f"  {len(values)} values × {n_reps} reps × {len(seeds)} seeds = "
          f"{n_tasks} tasks  →  {n_workers} workers")

    tasks = _build_tasks(param_name, locked_params)

    raw_by_value: dict = {
        float(v): {3: [], 6: [], 9: [], "gamma": []}
        for v in values
    }

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=n_tasks, desc=f"  {param_name[:38]}", ncols=82) as bar:
            for fut in as_completed(futures):
                try:
                    pv, seed, r3, r6, r9, gl = fut.result()
                    key = min(raw_by_value.keys(), key=lambda k: abs(k - pv))
                    raw_by_value[key][3].append(r3)
                    raw_by_value[key][6].append(r6)
                    raw_by_value[key][9].append(r9)
                    raw_by_value[key]["gamma"].append(gl)
                except Exception as e:
                    print(f"\n  ⚠️  Future error: {e}", flush=True)
                finally:
                    bar.update(1)

    rows = []
    for v in sorted(raw_by_value.keys()):
        bucket   = raw_by_value[v]
        for yrs in [3, 6, 9]:
            if not bucket[yrs]:
                bucket[yrs] = [0.0]
        if not bucket["gamma"]:
            bucket["gamma"] = [999.0]

        arr3     = np.array(bucket[3])
        arr6     = np.array(bucket[6])
        arr9     = np.array(bucket[9])
        arr_gam  = np.array([g for g in bucket["gamma"] if g < 998.0] or [999.0])
        n_actual = len(arr3)

        mean3, mean6, mean9 = arr3.mean(), arr6.mean(), arr9.mean()
        std3  = float(arr3.std(ddof=min(1, n_actual - 1)))
        std6  = float(arr6.std(ddof=min(1, n_actual - 1)))
        std9  = float(arr9.std(ddof=min(1, n_actual - 1)))
        mae3  = abs(mean3 - targets[3])
        mae6  = abs(mean6 - targets[6])
        mae9  = abs(mean9 - targets[9])

        row = {
            "value":          v,
            "n_runs":         n_actual,
            "rate_3yr":       mean3,
            "rate_6yr":       mean6,
            "rate_9yr":       mean9,
            "std_3yr":        std3,
            "std_6yr":        std6,
            "std_9yr":        std9,
            "mae_3yr":        mae3,
            "mae_6yr":        mae6,
            "mae_9yr":        mae9,
            "mae_all":        (mae3 + mae6 + mae9) / 3,
            "gamma_loss":     float(arr_gam.mean()),
            "gamma_loss_std": float(arr_gam.std(ddof=min(1, len(arr_gam) - 1))),
            "mae_primary":    float(np.mean([
                abs([mean3, mean6, mean9][[3, 6, 9].index(w)] - targets[w])
                for w in primary
            ])),
        }
        for i in range(n_actual):
            row[f"run{i:03d}_3yr"] = bucket[3][i]
            row[f"run{i:03d}_6yr"] = bucket[6][i]
            row[f"run{i:03d}_9yr"] = bucket[9][i]
        rows.append(row)

    return pd.DataFrame(rows)


def run_replicated_parallel(calibration_overrides: dict, n_workers: int) -> dict:
    seeds  = CONFIG["SEEDS"]
    n_reps = CONFIG["N_REPS"]
    warmup = CONFIG["warmup_months"]
    study  = CONFIG["study_months"]
    agents = CONFIG["initial_agents"]
    intake = CONFIG["monthly_intake"]
    mode   = CONFIG["mode"]
    bias   = CONFIG["bias_factor"]
    peer   = CONFIG["enable_peer_influence"]
    outdir = CONFIG["output_directory"]

    overrides = dict(calibration_overrides)
    overrides["__task_value__"] = 0.0
    cal_json = json.dumps(overrides)

    tasks = []
    for rep in range(n_reps):
        for seed in seeds:
            effective_seed = seed + rep * 10_000
            tasks.append((cal_json, effective_seed, warmup, study, agents,
                           intake, mode, bias, peer, outdir))

    raw = {3: [], 6: [], 9: [], "gamma": []}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in futures:
            try:
                _, _, r3, r6, r9, gl = fut.result()
                raw[3].append(r3)
                raw[6].append(r6)
                raw[9].append(r9)
                raw["gamma"].append(gl)
            except Exception as e:
                print(f"\n  ⚠️  Future error: {e}", flush=True)

    out = {}
    for yrs in [3, 6, 9]:
        arr = np.array(raw[yrs])
        out[yrs]          = float(arr.mean())
        out[f"std_{yrs}"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[f"all_{yrs}"] = arr.tolist()
        out[f"n_{yrs}"]   = len(tasks)

    gam = np.array([g for g in raw["gamma"] if g < 998.0] or [999.0])
    out["gamma_loss"]     = float(gam.mean())
    out["gamma_loss_std"] = float(gam.std(ddof=1)) if len(gam) > 1 else 0.0

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL VALUE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _select_optimal(df: pd.DataFrame, param_name: str) -> float:
    if param_name in STAGE2_PARAMS:
        loss_col = "gamma_loss"
        if loss_col not in df.columns or df[loss_col].min() >= 998.0:
            print(f"  ⚠️  gamma_loss not available — falling back to mae_all")
            loss_col = "mae_all"
    else:
        loss_col = "mae_primary"
    best_idx = df[loss_col].idxmin()
    return float(df.loc[best_idx, "value"])


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_loss_curves(df: pd.DataFrame, param_name: str,
                     optimal_value: float, outdir: str):
    n_seeds   = len(CONFIG["SEEDS"])
    is_stage2 = param_name in STAGE2_PARAMS
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Loss Curves — {PARAM_LABELS[param_name]}\n"
        f"(mean ± 1 SD  |  {n_seeds} seeds per value)\n"
        + ("Stage 2 — primary loss: PCRA tier MAE (gamma_loss)"
           if is_stage2 else
           "Stage 1 — primary loss: BJS aggregate MAE"),
        fontsize=13, fontweight="bold", y=0.98
    )
    label = PARAM_LABELS[param_name]
    x     = df["value"].values

    if is_stage2 and "gamma_loss" in df.columns:
        primary_col     = "gamma_loss"
        primary_std_col = "gamma_loss_std"
        primary_title   = "PCRA Tier MAE — gamma_loss (4 tiers × 3 windows)"
        primary_ylabel  = "Mean Absolute Error (tier-stratified)"
    else:
        primary_col     = "mae_all"
        primary_std_col = None
        primary_title   = "Combined MAE (3yr + 6yr + 9yr)"
        primary_ylabel  = "Mean Absolute Error"

    panels = [
        (primary_col, primary_std_col, COLORS["loss_all"], primary_title,                    primary_ylabel),
        ("mae_3yr",   "std_3yr",       COLORS["loss_3yr"], "MAE vs. BJS 3-Year Target (68%)", "Absolute Error"),
        ("mae_6yr",   "std_6yr",       COLORS["loss_6yr"], "MAE vs. BJS 6-Year Target (79%)", "Absolute Error"),
        ("mae_9yr",   "std_9yr",       COLORS["loss_9yr"], "MAE vs. BJS 9-Year Target (83%)", "Absolute Error"),
    ]
    for ax, (col, std_col, color, title, ylabel) in zip(axes.flat, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        y = df[col].values
        ax.plot(x, y, color=color, linewidth=2.2, marker="o", markersize=4,
                label="Mean loss", zorder=3)
        ax.fill_between(x, y, alpha=0.10, color=color)
        if std_col and std_col in df.columns:
            sd = df[std_col].values
            ax.fill_between(x, y - sd, y + sd, alpha=0.18, color=color,
                            label="±1 SD")
        ax.axvline(optimal_value, color=COLORS["optimal"], linewidth=1.8,
                   linestyle="--", label=f"Optimal = {optimal_value:.3f}")
        opt_idx = df["value"].sub(optimal_value).abs().idxmin()
        opt_y   = float(df.loc[opt_idx, col])
        ax.annotate(f"{opt_y:.4f}", xy=(optimal_value, opt_y),
                    xytext=(12, -18), textcoords="offset points",
                    fontsize=8.5, color=COLORS["optimal"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["optimal"], lw=1.2))
        ax.legend(fontsize=8)
        _style_ax(ax, label, ylabel, title)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, f"loss_{param_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    📊 Loss chart → {path}")


def plot_recidivism_comparison(df: pd.DataFrame, param_name: str,
                                optimal_value: float,
                                baseline_rates: dict, outdir: str):
    """
    FIX 1: y-axis starts at 0.40 — zooms into relevant range
    FIX 2: ±2pp shaded band around BJS target added
    FIX 3: legend position set to 'best'
    """
    n_seeds = len(CONFIG["SEEDS"])
    targets = CONFIG["bjs_targets"]
    label   = PARAM_LABELS[param_name]
    x       = df["value"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        f"Recidivism Rate Comparison — {PARAM_LABELS[param_name]}\n"
        f"(mean ± 1 SD  |  {n_seeds} seeds per value)",
        fontsize=13, fontweight="bold", y=1.03
    )
    for ax, (yrs, col, std_col, color, title) in zip(axes, [
        (3, "rate_3yr", "std_3yr", COLORS["3yr"], "3-Year Cumulative Rearrest"),
        (6, "rate_6yr", "std_6yr", COLORS["6yr"], "6-Year Cumulative Rearrest"),
        (9, "rate_9yr", "std_9yr", COLORS["9yr"], "9-Year Cumulative Rearrest"),
    ]):
        y     = df[col].values
        bjs_t = targets[yrs]
        base  = baseline_rates.get(yrs, 0.0)

        # FIX 2: ±2pp shaded acceptable band
        ax.axhspan(bjs_t - 0.02, bjs_t + 0.02, alpha=0.08, color=color,
                   label="±2pp acceptable band")

        ax.plot(x, y, color=color, linewidth=2.4, marker="o", markersize=4,
                label="ABM Mean", zorder=3)
        if std_col in df.columns:
            sd = df[std_col].values
            ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1),
                            alpha=0.20, color=color, label="±1 SD")

        ax.axhline(bjs_t, color=COLORS["bjs"], linewidth=2.0, linestyle="-",
                   label=f"BJS Target ({bjs_t:.0%})", zorder=4)
        ax.axhline(base, color=COLORS["baseline"], linewidth=1.6, linestyle=":",
                   label=f"Baseline ({base:.1%})", zorder=2)
        ax.axvline(optimal_value, color=COLORS["optimal"], linewidth=1.8,
                   linestyle="--", label=f"Optimal = {optimal_value:.3f}", zorder=5)

        opt_idx = df["value"].sub(optimal_value).abs().idxmin()
        opt_y   = float(df.loc[opt_idx, col])
        opt_sd  = float(df.loc[opt_idx, std_col]) if std_col in df.columns else 0.0
        ax.annotate(f"{opt_y:.1%} ± {opt_sd:.3f}", xy=(optimal_value, opt_y),
                    xytext=(10, 12), textcoords="offset points",
                    fontsize=8.5, color=COLORS["optimal"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS["optimal"], lw=1.2))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        # FIX 1: zoom y-axis into relevant range
        y_min = max(0.35, min(y.min(), base) - 0.05)
        y_max = min(1.02, max(y.max(), bjs_t, base) + 0.06)
        ax.set_ylim(y_min, y_max)

        # FIX 3: best legend position
        ax.legend(fontsize=8, loc="best")
        _style_ax(ax, label, "Cumulative Rearrest Rate", title)

    plt.tight_layout()
    path = os.path.join(outdir, f"recidivism_{param_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    📊 Recidivism chart → {path}")

def plot_final_summary(calibrated_params: dict, baseline_rates: dict,
                       calibrated_rates: dict, outdir: str):
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    labels  = ["3-Year", "6-Year", "9-Year"]

    # WIDER figure and right panel gets more space
    fig = plt.figure(figsize=(20, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.8], wspace=0.25)

    # ── Left panel: grouped bar ───────────────────────────────────────────
    ax1    = fig.add_subplot(gs[0])
    x      = np.arange(len(windows))
    w      = 0.25
    b_vals = [baseline_rates.get(yr, 0) for yr in windows]
    c_vals = [calibrated_rates.get(yr, 0) for yr in windows]
    t_vals = [targets[yr] for yr in windows]

    ax1.bar(x - w, b_vals, w, label="Baseline ABM",
            color=COLORS["baseline"], edgecolor="white")
    ax1.bar(x,     c_vals, w, label="Calibrated ABM",
            color=COLORS["calibrated"], edgecolor="white")
    ax1.bar(x + w, t_vals, w, label="BJS Target",
            color=COLORS["bjs"], edgecolor="white")
    for i, (bv, cv, tv) in enumerate(zip(b_vals, c_vals, t_vals)):
        ax1.text(i - w, bv + 0.005, f"{bv:.1%}",
                 ha="center", va="bottom", fontsize=8.5)
        ax1.text(i,     cv + 0.005, f"{cv:.1%}",
                 ha="center", va="bottom", fontsize=8.5,
                 color=COLORS["calibrated"], fontweight="bold")
        ax1.text(i + w, tv + 0.005, f"{tv:.1%}",
                 ha="center", va="bottom", fontsize=8.5,
                 color=COLORS["bjs"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    _style_ax(ax1, "Follow-up Window", "Cumulative Rearrest Rate",
              "Final Calibration — Baseline vs. Calibrated vs. BJS Targets")

    # ── Right panel: parameter table ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    # Shortened column headers to save space
    col_labels = ["Parameter", "Sym", "Value", "Stg", "Win", "Empirical Grounding"]

    # Shortened grounding text so it fits in the column
    GROUNDING_SHORT = {
        "Supervision_Monitoring_Intensity":
            "Langan & Levin (2002): 30% rearrest within 6mo",
        "Risk_Effect_Decay_After_3Y":
            "Sampson & Laub (2003); Kurlychek et al. (2006)",
        "Risk_Effect_Decay_After_6Y":
            "Sampson & Laub (2003); Bushway et al. (2011)",
        "Supervision_Monitoring_Decay_After_3Y":
            "Petersilia (2003): declining monitoring intensity",
        "Supervision_Monitoring_Decay_After_6Y":
            "Fixed: flat OAT surface; <15% agents supervised past yr 6",
        "Risk_Contrast_Strength":
            "PCRA Table 6, Federal Probation (Sept 2023)",
    }

    table_data      = []
    cell_colors_tbl = []

    for i, param in enumerate(CALIBRATION_ORDER):
        val    = calibrated_params.get(param, 0.0)
        symbol = PARAM_SYMBOLS[i]
        stage  = PARAM_STAGES[i]
        window = PARAM_WINDOWS[i]
        ground = GROUNDING_SHORT.get(param, "")
        table_data.append([PARAM_SHORT_NAMES[i], symbol, f"{val:.3f}",
                           stage, window, ground])
        row_color   = "#EAF4FB" if stage == "2" else "#F8F8F8"
        fixed_color = "#FFF3CD" if "*" in stage else row_color
        cell_colors_tbl.append([fixed_color] * len(col_labels))

    tbl = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colors_tbl,
        colColours=["#1A3D5C"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.4)

    # Explicit column widths — Grounding gets the most space
    col_widths = [0.26, 0.08, 0.10, 0.07, 0.08, 0.41]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_width(col_widths[col])
        if col == 5:  # Grounding — left align, slightly smaller font
            cell.get_text().set_ha("left")
            cell.get_text().set_fontsize(8.5)
        if row == 0:  # Header row
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")

    ax2.set_title("Calibrated Parameter Values", fontsize=12,
                  fontweight="bold", pad=12)
    ax2.text(0.5, -0.03,
             "Stage 1* = fixed (flat OAT surface)   Stage 2 = PCRA tier loss",
             transform=ax2.transAxes, ha="center", fontsize=8.5,
             color="#555555", style="italic")

    plt.suptitle(
        "Recidivism ABM — Calibration Summary\n"
        f"({CONFIG['N_REPS']} reps × {len(CONFIG['SEEDS'])} seeds"
        f"  |  Option B dual pipeline)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(outdir, "FINAL_calibration_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Final summary → {path}")



def plot_per_param_comparison_summary(all_results: dict, calibrated_params: dict,
                                       baseline_rates: dict, outdir: str):
    """
    FIX: Larger figure (was 16×3.2*n), larger fonts, ±2pp bands, y-axis at 0.40.
    """
    targets  = CONFIG["bjs_targets"]
    n_params = len(CALIBRATION_ORDER)

    fig, axes = plt.subplots(n_params, 3, figsize=(18, 4.2 * n_params))
    fig.suptitle("Per-Parameter Calibration Overview — All Parameters",
                 fontsize=15, fontweight="bold", y=1.01)

    for row, param in enumerate(CALIBRATION_ORDER):
        if param not in all_results:
            continue
        df  = all_results[param]["df"]
        opt = all_results[param]["optimal"]
        x   = df["value"].values

        for col, (yrs, y_col, std_col, color) in enumerate([
            (3, "rate_3yr", "std_3yr", COLORS["3yr"]),
            (6, "rate_6yr", "std_6yr", COLORS["6yr"]),
            (9, "rate_9yr", "std_9yr", COLORS["9yr"]),
        ]):
            ax    = axes[row, col]
            y     = df[y_col].values
            bjs_t = targets[yrs]

            ax.axhspan(bjs_t - 0.02, bjs_t + 0.02, alpha=0.08, color=color)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4)
            if std_col in df.columns:
                sd = df[std_col].values
                ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1),
                                alpha=0.15, color=color)
            ax.axhline(bjs_t, color=COLORS["bjs"], linewidth=1.8,
                       linestyle="-", label=f"BJS {bjs_t:.0%}")
            ax.axhline(baseline_rates.get(yrs, 0), color=COLORS["baseline"],
                       linewidth=1.4, linestyle=":", label="Baseline")
            ax.axvline(opt, color=COLORS["optimal"], linewidth=1.6,
                       linestyle="--", label=f"Opt={opt:.3f}")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

            y_min = max(0.35, min(y.min(), baseline_rates.get(yrs, 0)) - 0.05)
            ax.set_ylim(y_min, min(1.02, max(y.max(), bjs_t) + 0.06))

            ax.grid(True, color=COLORS["grid"], linewidth=0.5, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=9)

            if col == 0:
                label_text = PARAM_LABELS[param]
                if param in STAGE2_PARAMS:
                    label_text = "★ " + label_text
                ax.set_ylabel(label_text, fontsize=10, labelpad=4)
            if row == 0:
                ax.set_title(f"{yrs}-Year Rearrest Rate", fontsize=11,
                             fontweight="bold")
            if row == n_params - 1:
                ax.legend(fontsize=8, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    path = os.path.join(outdir, "overview_all_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Overview chart → {path}")


def plot_calibration_convergence(all_results: dict, calibrated_params: dict,
                                  outdir: str):
    """
    NEW: OAT calibration sequence visualisation.
    Shows identified optimal value at each calibration step,
    normalised within sweep range. Demonstrates sequential conditioning.
    Standard ABM calibration figure — Windrum et al. (2007).
    """
    fig, (ax_norm, ax_raw) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "OAT Calibration — Parameter Identification Sequence\n"
        "Each parameter identified conditional on all previously locked parameters",
        fontsize=13, fontweight="bold", y=1.02
    )

    steps       = list(range(1, len(CALIBRATION_ORDER) + 1))
    norm_vals   = []
    raw_vals    = []
    step_labels = []

    for i, param in enumerate(CALIBRATION_ORDER):
        val  = calibrated_params.get(param, 0.0)
        rng  = PARAM_RANGES[param]
        norm = (val - rng.min()) / max(rng.max() - rng.min(), 1e-6)
        norm_vals.append(norm)
        raw_vals.append(val)
        step_labels.append(f"Step {i+1}\n{PARAM_SYMBOLS[i]}")

    stage_colors = [
        COLORS["calibrated"] if CALIBRATION_ORDER[i] not in STAGE2_PARAMS
        else COLORS["bjs"]
        for i in range(len(CALIBRATION_ORDER))
    ]

    # Normalised panel
    ax_norm.plot(steps, norm_vals, "o-", color="#555555", linewidth=1.5,
                 markersize=2, alpha=0.4, zorder=2)
    for i, (step, nv, rv, color) in enumerate(zip(steps, norm_vals,
                                                    raw_vals, stage_colors)):
        ax_norm.scatter(step, nv, s=160, color=color, zorder=5,
                        edgecolors="white", linewidths=1.5)
        ax_norm.annotate(
            f"{PARAM_SYMBOLS[i]}={rv:.3f}",
            (step, nv),
            textcoords="offset points",
            xytext=(0, 14), ha="center", fontsize=9,
            color=color, fontweight="bold"
        )

    ax_norm.axhline(0.5, color=COLORS["grid"], linewidth=1, linestyle="--")
    ax_norm.set_xticks(steps)
    ax_norm.set_xticklabels(step_labels, fontsize=9)
    ax_norm.set_ylim(-0.12, 1.18)
    ax_norm.set_ylabel("Normalised value\n(0 = min of range, 1 = max of range)", fontsize=10)
    ax_norm.set_title("A — Normalised Parameter Values", fontsize=11, fontweight="bold")
    ax_norm.grid(True, alpha=0.3, linestyle="--")
    ax_norm.spines["top"].set_visible(False)
    ax_norm.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["calibrated"], label="Stage 1 — BJS aggregate loss"),
        Patch(facecolor=COLORS["bjs"],        label="Stage 2 — PCRA tier loss"),
    ]
    ax_norm.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # Raw values horizontal bar
    y_pos  = np.arange(len(CALIBRATION_ORDER))
    colors = stage_colors
    bars   = ax_raw.barh(y_pos, raw_vals, color=colors, height=0.55,
                          edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, raw_vals):
        ax_raw.text(val + max(raw_vals) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left",
                    fontsize=9, fontweight="bold")

    ax_raw.set_yticks(y_pos)
    ax_raw.set_yticklabels(PARAM_SHORT_NAMES, fontsize=9)
    ax_raw.set_xlim(0, max(raw_vals) * 1.25 + 0.2)
    ax_raw.set_xlabel("Calibrated value", fontsize=10)
    ax_raw.set_title("B — Raw Calibrated Parameter Values",
                     fontsize=11, fontweight="bold")
    ax_raw.grid(True, axis="x", color=COLORS["grid"],
                linewidth=0.5, linestyle="--")
    ax_raw.spines["top"].set_visible(False)
    ax_raw.spines["right"].set_visible(False)
    ax_raw.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(outdir, "calibration_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Convergence chart → {path}")


def plot_seed_variability(final_res: dict, outdir: str):
    """
    NEW: Box plots of rearrest rates across seeds at final calibrated params.
    Directly supports dissertation claim about Monte Carlo uncertainty.
    Windrum et al. (2007): MAE < 0.05 acceptable threshold shown.
    """
    targets = CONFIG["bjs_targets"]
    n_seeds = len(CONFIG["SEEDS"])
    n_reps  = CONFIG["N_REPS"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "Calibrated Model — Distribution Across Seeds\n"
        f"n = 20 seeds × 10 reps = 200 runs  |  " 
        "Windrum et al. (2007): MAE < 0.05 acceptable",
        fontsize=12, fontweight="bold", y=1.02
    )

    colors = [COLORS["3yr"], COLORS["6yr"], COLORS["9yr"]]
    for ax, yrs, color in zip(axes, [3, 6, 9], colors):
        data   = final_res.get(f"all_{yrs}", [])
        if not data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        data   = np.array(data)
        target = targets[yrs]
        mean   = data.mean()
        mae    = abs(mean - target)

        ax.boxplot(
            data, vert=True, patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.35, linewidth=1.5),
            medianprops=dict(color="#1A1A1A", linewidth=2.5),
            whiskerprops=dict(linewidth=1.8, linestyle="--"),
            capprops=dict(linewidth=2.0),
            flierprops=dict(marker="o", markerfacecolor=color,
                            markersize=4, alpha=0.5),
        )

        ax.axhline(target, color=COLORS["bjs"], linewidth=2.2,
                   linestyle="--", label=f"BJS target {target:.0%}", zorder=4)
        ax.axhline(mean, color=color, linewidth=1.8,
                   linestyle=":", label=f"Mean {mean:.1%}", zorder=3)
        ax.axhspan(target - 0.02, target + 0.02, alpha=0.07, color=COLORS["bjs"])

        flag = "✅" if mae < 0.02 else ("⚠️" if mae < 0.05 else "❌")
        ax.text(1.35, mean,
                f"MAE = {mae:.4f} {flag}\nn = {len(data)}",
                fontsize=9, color=color, fontweight="bold",
                va="center", ha="left")

        ci = 1.96 * data.std() / np.sqrt(len(data))
        ax.text(0.5, 0.04,
                f"95% CI: [{mean-ci:.3f}, {mean+ci:.3f}]",
                transform=ax.transAxes, ha="center", fontsize=8.5,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.8))

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        y_min = max(0.35, data.min() - 0.03)
        y_max = min(1.02, max(data.max(), target) + 0.05)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_title(f"{yrs}-Year Rearrest Rate", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Rearrest Rate", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(outdir, "seed_variability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Seed variability → {path}")


def plot_three_way_comparison(baseline_rates, baseline_std,
                               calibrated_rates, calibrated_std, outdir):
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    xlabels = ["3-Year", "6-Year", "9-Year"]
    n_seeds = len(CONFIG["SEEDS"])

    b_vals  = [baseline_rates.get(w, 0.0)   for w in windows]
    c_vals  = [calibrated_rates.get(w, 0.0) for w in windows]
    t_vals  = [targets[w]                   for w in windows]
    b_errs  = [baseline_std.get(w, 0.0)     for w in windows]
    c_errs  = [calibrated_std.get(w, 0.0)   for w in windows]
    b_mae   = [abs(b - t) for b, t in zip(b_vals, t_vals)]
    c_mae   = [abs(c - t) for c, t in zip(c_vals, t_vals)]
    pct_imp = [((bm - cm) / bm * 100) if bm > 0 else 0.0
               for bm, cm in zip(b_mae, c_mae)]

    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.38,
                            left=0.08, right=0.97, top=0.91, bottom=0.07)

    ax_bar = fig.add_subplot(gs[0, 0])
    x = np.arange(len(windows)); w = 0.24
    bars_b = ax_bar.bar(x - w, b_vals, w, color=COLORS["baseline"],
                        edgecolor="#888888", linewidth=0.8,
                        label="Baseline ABM", zorder=3)
    bars_c = ax_bar.bar(x,     c_vals, w, color=COLORS["calibrated"],
                        edgecolor="white",   linewidth=0.8,
                        label="Calibrated ABM", zorder=3)
    bars_t = ax_bar.bar(x + w, t_vals, w, color=COLORS["bjs"],
                        edgecolor="white",   linewidth=0.8,
                        label="BJS Empirical Target", zorder=3)
    for bars, vals, clr in [(bars_b, b_vals, "#555555"),
                             (bars_c, c_vals, COLORS["calibrated"]),
                             (bars_t, t_vals, COLORS["bjs"])]:
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.1%}", ha="center", va="bottom",
                        fontsize=8.5, color=clr, fontweight="bold")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(xlabels, fontsize=10)
    ax_bar.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_bar.set_ylim(0, 1.08)
    ax_bar.legend(fontsize=9, loc="upper left", framealpha=0.9)
    _style_ax(ax_bar, "Follow-up Window", "Cumulative Rearrest Rate",
              "A — Recidivism Rate: Baseline vs. Calibrated vs. BJS Target")

    ax_dot = fig.add_subplot(gs[0, 1])
    offsets = [-0.18, 0.0, 0.18]
    x_pos   = np.arange(len(windows))
    ax_dot.scatter(x_pos + offsets[2], t_vals, marker="D", s=90,
                   color=COLORS["bjs"], zorder=5, label="BJS Target")
    ax_dot.errorbar(x_pos + offsets[1], c_vals, yerr=c_errs, fmt="o",
                    markersize=9, color=COLORS["calibrated"],
                    ecolor=COLORS["calibrated"], elinewidth=2,
                    capsize=5, capthick=2, zorder=4,
                    label=f"Calibrated ABM (±1 SD, n={n_seeds})")
    ax_dot.errorbar(x_pos + offsets[0], b_vals, yerr=b_errs, fmt="s",
                    markersize=8, color=COLORS["baseline"],
                    ecolor="#999999", elinewidth=1.8,
                    capsize=5, capthick=2, zorder=3,
                    label=f"Baseline ABM (±1 SD, n={n_seeds})")
    for xi, bv, cv in zip(x_pos, b_vals, c_vals):
        ax_dot.annotate("", xy=(xi + offsets[1], cv),
                        xytext=(xi + offsets[0], bv),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC",
                                        lw=1.4, mutation_scale=10))
    ax_dot.set_xticks(x_pos); ax_dot.set_xticklabels(xlabels, fontsize=10)
    ax_dot.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_dot.set_ylim(0, 1.08)
    ax_dot.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    _style_ax(ax_dot, "Follow-up Window", "Cumulative Rearrest Rate",
              "B — Uncertainty View: ±1 SD Across Seeds")

    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")
    col_labels = ["Window", "Baseline\nMAE", "Calibrated\nMAE",
                  "MAE\nReduction", "Baseline\nRate", "Calibrated\nRate", "BJS\nTarget"]
    table_data, cell_colors = [], []
    for i, (w_yr, xl) in enumerate(zip(windows, xlabels)):
        bm, cm, imp = b_mae[i], c_mae[i], pct_imp[i]
        flag = "✅" if cm < 0.02 else ("⚠️" if cm < 0.05 else "❌")
        table_data.append([xl, f"{bm:.4f}", f"{cm:.4f}  {flag}", f"−{imp:.1f}%",
                           f"{b_vals[i]:.1%}", f"{c_vals[i]:.1%}", f"{t_vals[i]:.1%}"])
        mae_c = "#D6EAD6" if cm < 0.02 else ("#FFF3CD" if cm < 0.05 else "#FADBD8")
        imp_c = "#D6EAD6" if imp > 20  else ("#FFF3CD" if imp > 0   else "#FADBD8")
        cell_colors.append(["#F0F0F0","#F8F8F8", mae_c, imp_c,
                             "#F8F8F8","#EAF4FB","#EAF4FB"])
    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                        cellColours=cell_colors,
                        colColours=["#1A3D5C"] * len(col_labels),
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.0, 2.1)
    for j in range(len(col_labels)):
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax_tbl.set_title("C — MAE Summary Table", fontsize=11,
                     fontweight="bold", pad=10, loc="left")

    ax_lol = fig.add_subplot(gs[1, 1])
    colors_lol = ["#27AE60" if p > 20 else ("#F39C12" if p > 0 else "#E74C3C")
                  for p in pct_imp]
    y_pos = np.arange(len(windows))
    ax_lol.barh(y_pos, pct_imp, height=0.08,
                color=[c + "55" for c in colors_lol], zorder=2)
    ax_lol.scatter(pct_imp, y_pos, s=180, color=colors_lol,
                   zorder=4, edgecolors="white", linewidths=1.5)
    ax_lol.axvline(0, color="#CCCCCC", linewidth=1.2, zorder=1)
    for yi, pi, clr in zip(y_pos, pct_imp, colors_lol):
        offset = 1.5 if pi >= 0 else -1.5
        ax_lol.text(pi + offset, yi, f"{pi:+.1f}%", va="center",
                    ha="left" if pi >= 0 else "right",
                    fontsize=10, color=clr, fontweight="bold")
    ax_lol.set_yticks(y_pos); ax_lol.set_yticklabels(xlabels, fontsize=10)
    ax_lol.set_xlabel("MAE Reduction (%)", fontsize=10, labelpad=6)
    ax_lol.set_xlim(min(pct_imp) - 15, max(pct_imp) + 20)
    ax_lol.set_title("D — % MAE Improvement: Baseline → Calibrated",
                     fontsize=11, fontweight="bold", pad=8)
    ax_lol.grid(True, axis="x", color=COLORS["grid"], linewidth=0.6, linestyle="--")
    ax_lol.spines["top"].set_visible(False)
    ax_lol.spines["right"].set_visible(False)
    ax_lol.tick_params(labelsize=9)

    fig.suptitle(
        "Recidivism ABM — Three-Way Comparison: Baseline vs. Calibrated vs. BJS Targets\n"
        f"BJS targets: 3yr={targets[3]:.0%}  6yr={targets[6]:.0%}  9yr={targets[9]:.0%}"
        f"   |   {n_seeds} seeds per condition",
        fontsize=13, fontweight="bold", y=0.975, color="#1A3D5C"
    )
    path = os.path.join(outdir, "THREE_WAY_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✅ Three-way comparison chart → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CALIBRATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main(n_workers: int, force_rerun: bool = False):
    outdir = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "tmp"), exist_ok=True)

    targets        = CONFIG["bjs_targets"]
    n_seeds        = len(CONFIG["SEEDS"])
    n_reps         = CONFIG["N_REPS"]
    runs_per_value = n_reps * n_seeds

    stage1_params = [p for p in CALIBRATION_ORDER if p not in STAGE2_PARAMS]
    stage2_params = [p for p in CALIBRATION_ORDER if p in STAGE2_PARAMS]

    print("=" * 70)
    print("  RECIDIVISM ABM — PARALLEL OAT CALIBRATION  (6 parameters)")
    print("=" * 70)
    print(f"  Workers       : {n_workers}")
    print(f"  Agents        : {CONFIG['initial_agents']}")
    print(f"  Warmup/Study  : {CONFIG['warmup_months']} / {CONFIG['study_months']} months")
    print(f"  Runs/value    : {n_reps} reps × {n_seeds} seeds = {runs_per_value}")
    print(f"  Stage 1 params: {len(stage1_params)}  (BJS aggregate loss)")
    print(f"  Stage 2 params: {len(stage2_params)}  (PCRA tier loss)")
    print(f"  BJS targets   : 3yr={targets[3]:.0%}  6yr={targets[6]:.0%}  9yr={targets[9]:.0%}")
    print(f"  Output        : {outdir}")
    print(f"  Checkpoint    : {'disabled (--rerun)' if force_rerun else 'enabled'}")
    print("=" * 70)

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_json = os.path.join(outdir, "baseline.json")
    if not force_rerun and os.path.exists(baseline_json):
        print(f"\n▶ Step 0: Loading baseline from checkpoint...")
        with open(baseline_json) as f:
            baseline_res = _normalise_baseline(json.load(f))
    else:
        print(f"\n▶ Step 0: Running baseline ({runs_per_value} runs)...")
        #_default_cal = get_global_calibration_params()
        #baseline_overrides = {k: _default_cal[k] for k in CALIBRATION_ORDER
        #                      if k in _default_cal}
        baseline_overrides = get_uncalibrated_params()
        baseline_res = run_replicated_parallel(baseline_overrides, n_workers)
        baseline_res = _normalise_baseline(baseline_res)
        with open(baseline_json, "w") as f:
            json.dump(baseline_res, f, indent=2)

    baseline_rates = {3: baseline_res[3], 6: baseline_res[6], 9: baseline_res[9]}
    print(f"  Baseline: 3yr={baseline_rates[3]:.1%} ±{baseline_res.get('std_3', 0):.3f}  "
          f"6yr={baseline_rates[6]:.1%} ±{baseline_res.get('std_6', 0):.3f}  "
          f"9yr={baseline_rates[9]:.1%} ±{baseline_res.get('std_9', 0):.3f}")

    locked_params = {k: get_uncalibrated_params()[k]
                    for k in CALIBRATION_ORDER
                    if k in get_uncalibrated_params()}

    calibrated  = {}
    all_results = {}

    # ── Sequential OAT sweep ─────────────────────────────────────────────────
    for step, param in enumerate(CALIBRATION_ORDER, start=1):
        is_stage2 = param in STAGE2_PARAMS
        stage_tag = "Stage 2 — PCRA tier loss" if is_stage2 \
                    else "Stage 1 — BJS aggregate loss"

        print(f"\n{'─'*70}")
        print(f"▶ Step {step}/{len(CALIBRATION_ORDER)}: {PARAM_LABELS[param]}")
        print(f"  [{stage_tag}]")
        if calibrated:
            print("  Locked: " + "  ".join(
                f"{k}={v:.4f}" for k, v in calibrated.items()))
        print(f"{'─'*70}")

        csv_path = os.path.join(outdir, f"sweep_{param}.csv")
        if not force_rerun and os.path.exists(csv_path):
            print(f"  ✔ Checkpoint found — loading {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            df = sweep_parameter_parallel(param, locked_params, n_workers)
            df.to_csv(csv_path, index=False)
            print(f"  Sweep data → {csv_path}")

        best_value = _select_optimal(df, param)
        best_idx   = df["value"].sub(best_value).abs().idxmin()

        if is_stage2 and "gamma_loss" in df.columns:
            best_loss  = float(df.loc[best_idx, "gamma_loss"])
            loss_label = "gamma_loss (tier MAE)"
        else:
            primary       = PARAM_PRIMARY_WINDOW.get(param, [3, 6, 9])
            mae_cols      = [f"mae_{y}yr" for y in primary
                             if f"mae_{y}yr" in df.columns]
            df["mae_primary"] = df[mae_cols].mean(axis=1)
            best_loss  = float(df.loc[best_idx, "mae_primary"])
            loss_label = "mae_primary"

        print(f"\n  ✅  {param} = {best_value:.4f}  ({loss_label} = {best_loss:.4f})")
        print(f"      At optimum: "
              f"3yr={df.loc[best_idx,'rate_3yr']:.1%} "
              f"±{df.loc[best_idx,'std_3yr']:.3f}  "
              f"6yr={df.loc[best_idx,'rate_6yr']:.1%} "
              f"±{df.loc[best_idx,'std_6yr']:.3f}  "
              f"9yr={df.loc[best_idx,'rate_9yr']:.1%} "
              f"±{df.loc[best_idx,'std_9yr']:.3f}")

        calibrated[param]    = best_value
        locked_params[param] = best_value
        all_results[param]   = {"df": df, "optimal": best_value}

        plot_loss_curves(df, param, best_value, outdir)
        plot_recidivism_comparison(df, param, best_value, baseline_rates, outdir)

    # ── Final validation ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"▶ Final validation ({runs_per_value} runs)...")
    final_res        = run_replicated_parallel(locked_params, n_workers)
    final_res        = _normalise_baseline(final_res)
    calibrated_rates = {3: final_res[3], 6: final_res[6], 9: final_res[9]}
    print(f"  Calibrated: 3yr={calibrated_rates[3]:.1%} "
          f"±{final_res.get('std_3',0):.3f}  "
          f"6yr={calibrated_rates[6]:.1%} ±{final_res.get('std_6',0):.3f}  "
          f"9yr={calibrated_rates[9]:.1%} ±{final_res.get('std_9',0):.3f}")

    # ── Dissertation charts ───────────────────────────────────────────────────
    plot_final_summary(locked_params, baseline_rates, calibrated_rates, outdir)
    plot_per_param_comparison_summary(all_results, locked_params, baseline_rates, outdir)
    plot_three_way_comparison(
        baseline_rates   = baseline_rates,
        baseline_std     = {3: baseline_res.get("std_3", 0),
                             6: baseline_res.get("std_6", 0),
                             9: baseline_res.get("std_9", 0)},
        calibrated_rates = calibrated_rates,
        calibrated_std   = {3: final_res.get("std_3", 0),
                             6: final_res.get("std_6", 0),
                             9: final_res.get("std_9", 0)},
        outdir           = outdir,
    )
    plot_calibration_convergence(all_results, locked_params, outdir)
    plot_seed_variability(final_res, outdir)

    # ── Save ─────────────────────────────────────────────────────────────────
    recommended = {
        "calibrated_params":  locked_params,
        "baseline_rates":     baseline_rates,
        "baseline_std":       {3: baseline_res.get("std_3", 0),
                                6: baseline_res.get("std_6", 0),
                                9: baseline_res.get("std_9", 0)},
        "calibrated_rates":   calibrated_rates,
        "calibrated_std":     {3: final_res.get("std_3", 0),
                                6: final_res.get("std_6", 0),
                                9: final_res.get("std_9", 0)},
        "final_validation":   {f"all_{y}": final_res.get(f"all_{y}", [])
                               for y in [3, 6, 9]},
        "bjs_targets":        targets,
        "seeds_used":         CONFIG["SEEDS"],
        "n_workers":          n_workers,
        "final_mae": {
            "3yr": abs(calibrated_rates[3] - targets[3]),
            "6yr": abs(calibrated_rates[6] - targets[6]),
            "9yr": abs(calibrated_rates[9] - targets[9]),
        },
        "config": CONFIG,
    }
    json_path = os.path.join(outdir, "recommended_params.json")
    with open(json_path, "w") as f:
        json.dump(recommended, f, indent=2)

    print(f"\n{'='*70}")
    print("  CALIBRATION COMPLETE")
    print(f"{'='*70}")
    for p in CALIBRATION_ORDER:
        stage = " [Stage 2]" if p in STAGE2_PARAMS else " [Stage 1]"
        print(f"  {p:<50}  {locked_params[p]:.4f}{stage}")
    print(f"{'─'*70}")
    for yrs in [3, 6, 9]:
        err  = abs(calibrated_rates[yrs] - targets[yrs])
        flag = "✅" if err < 0.02 else ("⚠️ " if err < 0.05 else "❌")
        print(f"  {flag} {yrs}-year: ABM={calibrated_rates[yrs]:.1%} "
              f"±{final_res.get(f'std_{yrs}', 0):.3f}  "
              f"BJS={targets[yrs]:.1%}  MAE={err:.4f}")
    print(f"{'='*70}")
    print(f"\n  Recommended params → {json_path}")
    print("\n  ── risk_config.py update block ──")
    for k, v in locked_params.items():
        print(f'    "{k}": {v:.4f},')

    print(f"\n  📊 Charts generated in {outdir}/:")
    for fname in [
        "FINAL_calibration_summary.png",
        "THREE_WAY_comparison.png",
        "overview_all_params.png",
        "calibration_convergence.png",
        "seed_variability.png",
    ] + [f"loss_{p}.png" for p in CALIBRATION_ORDER] \
      + [f"recidivism_{p}.png" for p in CALIBRATION_ORDER]:
        fp = os.path.join(outdir, fname)
        print(f"    {'✅' if os.path.exists(fp) else '❌'} {fname}")

    return recommended


# ═══════════════════════════════════════════════════════════════════════════════
# REPLOT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def replot_from_csv(outdir: str = None):
    if outdir is None:
        outdir = CONFIG["output_directory"]
    json_path = os.path.join(outdir, "recommended_params.json")
    if not os.path.exists(json_path):
        print("No recommended_params.json found. Run calibration first.")
        return
    with open(json_path) as f:
        rec = json.load(f)
    calibrated_params = rec["calibrated_params"]
    baseline_rates    = {int(k): v for k, v in rec["baseline_rates"].items()}
    calibrated_rates  = {int(k): v for k, v in rec["calibrated_rates"].items()}

    all_results = {}
    for param in CALIBRATION_ORDER:
        csv_path = os.path.join(outdir, f"sweep_{param}.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {param} — CSV not found")
            continue
        df  = pd.read_csv(csv_path)
        opt = calibrated_params.get(param, _select_optimal(df, param))
        all_results[param] = {"df": df, "optimal": opt}
        plot_loss_curves(df, param, opt, outdir)
        plot_recidivism_comparison(df, param, opt, baseline_rates, outdir)

    plot_final_summary(calibrated_params, baseline_rates, calibrated_rates, outdir)
    plot_per_param_comparison_summary(all_results, calibrated_params, baseline_rates, outdir)
    plot_calibration_convergence(all_results, calibrated_params, outdir)

    b_std = {int(k): v for k, v in rec.get("baseline_std",   {}).items()}
    c_std = {int(k): v for k, v in rec.get("calibrated_std", {}).items()}
    for w in [3, 6, 9]:
        b_std.setdefault(w, 0.0)
        c_std.setdefault(w, 0.0)
    plot_three_way_comparison(baseline_rates, b_std, calibrated_rates, c_std, outdir)

    final_val = rec.get("final_validation", {})
    if final_val:
        plot_seed_variability(final_val, outdir)
    else:
        print("  Skipping seed_variability — re-run calibration to generate")

    print("✅ All charts regenerated.")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_workers() -> tuple:
    if _PSUTIL_AVAILABLE:
        physical = psutil.cpu_count(logical=False)
        logical  = psutil.cpu_count(logical=True)
        if physical and physical > 0:
            n_workers = max(1, physical - 1)
            return n_workers, (f"psutil: {physical} physical cores → {n_workers} workers")
        n_workers = max(1, (logical or 2) // 2 - 1)
        return n_workers, (f"psutil: {logical} logical → {n_workers} workers (conservative)")
    logical   = multiprocessing.cpu_count()
    n_workers = max(1, logical // 2 - 1)
    return n_workers, (f"{logical} logical CPUs → {n_workers} workers")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = sys.argv[1:]

    if "--replot" in args:
        replot_from_csv()
        sys.exit(0)

    force_rerun = "--rerun" in args

    if "--cores" in args:
        idx = args.index("--cores")
        try:
            n_workers = int(args[idx + 1])
            print(f"\n  ── Manual override → {n_workers} workers")
        except (IndexError, ValueError):
            print("Usage: python calibrate.py [--cores N] [--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  ── Core detection: {reason}")

    print(f"  ── Starting calibration with {n_workers} workers...\n")
    main(n_workers=n_workers, force_rerun=force_rerun)