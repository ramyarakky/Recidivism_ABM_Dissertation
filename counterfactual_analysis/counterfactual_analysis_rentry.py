#!/usr/bin/env python3
"""
counterfactual_analysis.py
===============================================================
System-Level Reentry Environment Counterfactual
for the Recidivism Agent-Based Model.

Purpose
-------
This experiment evaluates how aggregate rearrest changes when the calibrated
quarterly rearrest hazard is proportionally increased or reduced.

Implementation
--------------
The canonical model stores `reentry_hazard_cf`; agents.py applies:

    q_cf = q_baseline * (1 - reentry_hazard_cf)

Interpretation:
    -0.80 -> 1.80x calibrated quarterly hazard (worse reentry environment)
     0.00 -> 1.00x calibrated quarterly hazard (calibrated baseline)
    +0.80 -> 0.20x calibrated quarterly hazard (improved reentry environment)

This is a system-level sensitivity/bounding experiment, not an estimate of the
causal effect of one specific policy.

Experimental design
-------------------
For EACH counterfactual level:
    10 runs x 20 seeds = 200 matched replications

The exact same run/seed combinations are reused at every intervention level,
allowing paired comparisons to the 0% baseline.

Default sweep:
    -80% to +80% in 10 percentage-point increments (17 levels)
    17 x 200 = 3,400 ABM simulations

Outputs
-------
counterfactual_output/
    model_results/
        hazard_raw.csv
        hazard_summary.csv
        hazard_paired_seed_raw.csv
        hazard_paired_effects.csv
        hazard_dissertation_table.csv
        hazard_mechanism_summary.csv
        hazard_run_metadata.json
        hazard_incremental_rearrest_by_period.csv
    charts/
        CF1_hazard_rearrest_rates.png
        CF2_hazard_change_from_baseline.png
        CF3_hazard_mechanism_check.png
        CF4_hazard_effect_by_horizon.png
        CF5_hazard_selected_scenarios.png
        CF6_hazard_incremental_rearrest_by_period.png

Run
---
python counterfactual_analysis.py --workers 120
python counterfactual_analysis.py --replot

The script expects to live in:
    recidivism_abm/counterfactual_analysis/counterfactual_analysis.py
and imports the canonical model from:
    recidivism_abm/model/recidivism_model.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# Prevent nested BLAS threading when many ABM workers are used.
for _env in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env, "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None
    SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ================================================================
# PROJECT ROOT / IMPORTS
# ================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_global_calibration_params,
    get_peer_influence_config,
)


# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    "initial_agents": 1000,
    "monthly_intake": 10,
    "warmup_months": 144,
    "study_months": 108,
    "bias_factor": 1,
    "enable_peer_influence": True,
    "mode": "realistic",

    # Dissertation replication design.
    "N_RUNS": 10,
    "SEEDS_PER_RUN": 20,
    "BASE_SEED": 1000,

    "output_directory": "counterfactual_output",
    "results_subdir": "model_results",
    "charts_subdir": "charts",

    # Contextual empirical anchors; plotted only as references.
    "bjs_targets": {3: 0.684, 6: 0.794, 9: 0.834},
}

WINDOWS = (3, 6, 9)

CF_LEVELS = np.array(
    [-0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10,
      0.00,
      0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80],
    dtype=float,
)

# Selected scenarios used in the committee-friendly summary figure/table.
SELECTED_LEVELS = (-0.80, -0.50, -0.20, 0.00, 0.20, 0.50, 0.80)

DPI = 350
TITLE_FS = 15
LABEL_FS = 12
TICK_FS = 10
LEGEND_FS = 10
NOTE_FS = 9


# ================================================================
# GENERAL HELPERS
# ================================================================

def ensure_output_dirs():
    outdir = Path(CONFIG["output_directory"])
    results_dir = outdir / CONFIG["results_subdir"]
    charts_dir = outdir / CONFIG["charts_subdir"]

    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return outdir, results_dir, charts_dir


def detect_workers(requested: int | None = None) -> int:
    if requested is not None:
        return max(1, int(requested))

    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return max(1, int(slurm) - 2)

    if PSUTIL_AVAILABLE:
        physical = psutil.cpu_count(logical=False)
        if physical:
            return max(1, physical - 1)

    return max(1, multiprocessing.cpu_count() - 1)


def effective_seed(run_id: int, seed_id: int) -> int:
    return int(CONFIG["BASE_SEED"] + run_id * 10_000 + seed_id)


def matched_seed_grid() -> List[Tuple[int, int, int]]:
    return [
        (run_id, seed_id, effective_seed(run_id, seed_id))
        for run_id in range(1, CONFIG["N_RUNS"] + 1)
        for seed_id in range(1, CONFIG["SEEDS_PER_RUN"] + 1)
    ]


def t_critical_95(n: int) -> float:
    if n <= 1:
        return 0.0
    if SCIPY_AVAILABLE:
        return float(scipy_stats.t.ppf(0.975, df=n - 1))
    return 1.96


def mean_ci95(values) -> Tuple[float, float, float, float]:
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, 0.0, mean, mean

    sd = float(np.std(x, ddof=1))
    sem = sd / math.sqrt(len(x))
    half = t_critical_95(len(x)) * sem
    return mean, sd, mean - half, mean + half


def environment_label(level: float) -> str:
    if np.isclose(level, 0.0):
        return "Calibrated baseline"
    if level > 0:
        return f"{level:+.0%} hazard reduction"
    return f"{abs(level):.0%} hazard increase"


def style_axis(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=8)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=18)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def dynamic_limits(values, minimum_padding=2.0, frac=0.08):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = max(hi - lo, 1.0)
    pad = max(minimum_padding, frac * span)
    return lo - pad, hi + pad


# ================================================================
# MODEL BUILD
# ================================================================

def validate_model_interface():
    """Fail early if the canonical model does not expose the new parameter."""
    signature = inspect.signature(RecidivismModel.__init__).parameters
    if "reentry_hazard_cf" not in signature:
        raise RuntimeError(
            "RecidivismModel.__init__ does not contain `reentry_hazard_cf`. "
            "Add `reentry_hazard_cf=0.0` to recidivism_model.py and store it as "
            "`self.reentry_hazard_cf` before running this experiment."
        )


def build_model(seed: int, reentry_hazard_cf: float = 0.0) -> RecidivismModel:
    calibration = get_global_calibration_params()
    weights = get_flat_risk_weights()
    peer_config = get_peer_influence_config()

    kwargs = {
        "initial_agents": CONFIG["initial_agents"],
        "bias_factor": CONFIG["bias_factor"],
        "monthly_intake": CONFIG["monthly_intake"],
        "warmup_months": CONFIG["warmup_months"],
        "study_months": CONFIG["study_months"],
        "enable_peer_influence": CONFIG["enable_peer_influence"],
        "weights": weights,
        "peer_config": peer_config,
        "seed": seed,
        "mode": CONFIG["mode"],
        "calibration_params": calibration,
        "reentry_hazard_cf": float(reentry_hazard_cf),
    }

    signature = inspect.signature(RecidivismModel.__init__).parameters
    if "output_directory" in signature:
        kwargs["output_directory"] = str(Path(CONFIG["output_directory"]) / "tmp")

    model = RecidivismModel(**kwargs)

    # Explicit assignment also makes the experiment transparent during debugging.
    model.reentry_hazard_cf = float(reentry_hazard_cf)

    if hasattr(model, "export_csv"):
        model.export_csv = False

    return model


# ================================================================
# RESULT EXTRACTION
# ================================================================

def _numeric_or_nan(v):
    if isinstance(v, (int, float, bool, np.integer, np.floating)):
        return float(v)
    return np.nan


def extract_results(
    model,
    run_id: int,
    seed_id: int,
    seed: int,
    cf_level: float,
) -> Dict:
    eligible = [
        a for a in model.schedule.agents
        if getattr(a, "study_eligible_agent", False)
    ]

    n = len(eligible)
    if n == 0:
        raise RuntimeError(
            f"No study-eligible agents returned "
            f"(run={run_id}, seed={seed}, level={cf_level:+.2f})."
        )

    hazard_cf = float(getattr(model, "reentry_hazard_cf", 0.0))

    row = {
        "experiment": "hazard",
        "cf_level": float(cf_level),
        "cf_percent": float(cf_level * 100.0),
        "scenario_label": environment_label(cf_level),
        "hazard_multiplier": float(1.0 - hazard_cf),
        "run_id": int(run_id),
        "seed_id": int(seed_id),
        "seed": int(seed),
        "n_eligible": int(n),
    }

    for years in WINDOWS:
        flag = f"rearrest_{years}_yrs"
        count = sum(bool(getattr(a, flag, False)) for a in eligible)
        row[f"rearrest_{years}yr_count"] = int(count)
        row[f"rearrest_{years}yr"] = float(count / n)

    # Mechanism diagnostics. These are not expected to drive the intervention;
    # they help show that the experiment acts through the hazard multiplier.
    employed = np.asarray([
        float(getattr(a, "Percent_Days_Employed", np.nan)) for a in eligible
    ], dtype=float)
    attendance = np.asarray([
        float(getattr(a, "Program_Attendances", np.nan)) for a in eligible
    ], dtype=float)
    technical = np.asarray([
        _numeric_or_nan(getattr(a, "Violations_Technical", np.nan))
        for a in eligible
    ], dtype=float)

    row["mean_percent_days_employed"] = (
        float(np.nanmean(employed)) if np.isfinite(employed).any() else np.nan
    )
    row["mean_program_attendances"] = (
        float(np.nanmean(attendance)) if np.isfinite(attendance).any() else np.nan
    )
    row["mean_technical_violations"] = (
        float(np.nanmean(technical)) if np.isfinite(technical).any() else np.nan
    )

    return row


# ================================================================
# PARALLEL EXECUTION
# ================================================================

def worker(task: Tuple) -> Dict:
    run_id, seed_id, seed, cf_level = task
    model = build_model(seed=seed, reentry_hazard_cf=float(cf_level))

    while model.running:
        model.step()

    return extract_results(
        model=model,
        run_id=run_id,
        seed_id=seed_id,
        seed=seed,
        cf_level=float(cf_level),
    )


def build_tasks(cf_level: float) -> List[Tuple]:
    return [
        (run_id, seed_id, seed, float(cf_level))
        for run_id, seed_id, seed in matched_seed_grid()
    ]


def run_task_batch(tasks: List[Tuple], n_workers: int, description: str) -> List[Dict]:
    results: List[Dict] = []
    max_workers = min(n_workers, len(tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        iterator = as_completed(futures)

        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc=description, leave=False)

        for future in iterator:
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                run_id, seed_id, seed, level = task
                print(
                    f"\nWARNING: failed task level={level:+.0%}, "
                    f"run={run_id}, seed_id={seed_id}, seed={seed}: {exc}",
                    flush=True,
                )

    return results


def run_experiment(n_workers: int, results_dir: Path) -> pd.DataFrame:
    all_rows: List[Dict] = []
    tasks_per_level = CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"]
    total_tasks = tasks_per_level * len(CF_LEVELS)

    print("\n" + "=" * 82)
    print("SYSTEM-LEVEL REENTRY ENVIRONMENT COUNTERFACTUAL")
    print("=" * 82)
    print(f"Replication design : {CONFIG['N_RUNS']} runs x {CONFIG['SEEDS_PER_RUN']} seeds")
    print(f"Replications/level : {tasks_per_level}")
    print(f"Counterfactuals    : {len(CF_LEVELS)} levels ({CF_LEVELS[0]:+.0%} to {CF_LEVELS[-1]:+.0%})")
    print(f"Total simulations  : {total_tasks:,}")
    print(f"Workers            : {n_workers}")
    print(f"Warm-up / Study    : {CONFIG['warmup_months']} / {CONFIG['study_months']} months")
    print("Interpretation     : positive values represent increasingly favorable reentry conditions")
    print("=" * 82)

    for level_idx, cf_level in enumerate(CF_LEVELS, start=1):
        tasks = build_tasks(float(cf_level))
        multiplier = 1.0 - float(cf_level)

        print(
            f"\n[{level_idx:02d}/{len(CF_LEVELS):02d}] "
            f"{environment_label(float(cf_level))} "
            f"(technical hazard multiplier={multiplier:.2f}x)"
        )

        level_results = run_task_batch(
            tasks,
            n_workers,
            f"  scenario {cf_level:+.0%}",
        )

        if len(level_results) != len(tasks):
            print(
                f"  WARNING: {len(level_results)} / {len(tasks)} "
                "replications completed successfully."
            )

        all_rows.extend(level_results)

        checkpoint = results_dir / "hazard_raw_checkpoint.csv"
        pd.DataFrame(all_rows).to_csv(checkpoint, index=False)
        print(f"  Checkpoint -> {checkpoint}")

    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise RuntimeError("No counterfactual results were produced.")

    raw.sort_values(["cf_level", "run_id", "seed_id"], inplace=True)
    raw.reset_index(drop=True, inplace=True)

    raw_path = results_dir / "hazard_raw.csv"
    raw.to_csv(raw_path, index=False)

    checkpoint = results_dir / "hazard_raw_checkpoint.csv"
    if checkpoint.exists():
        checkpoint.unlink()

    print(f"\nRaw results -> {raw_path}")
    return raw


# ================================================================
# SUMMARIES / PAIRED EFFECTS
# ================================================================

def summarize_experiment(raw: pd.DataFrame, results_dir: Path):
    expected_pairs = CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"]

    # Absolute rearrest rates and mechanism summaries.
    summary_rows = []
    for level, group in raw.groupby("cf_level", sort=True):
        row = {
            "cf_level": float(level),
            "cf_percent": float(level * 100.0),
            "scenario_label": environment_label(float(level)),
            "hazard_multiplier": float(1.0 - level),
            "n_replications": int(len(group)),
        }

        for years in WINDOWS:
            mean, sd, lo, hi = mean_ci95(group[f"rearrest_{years}yr"])
            row[f"rearrest_{years}yr_mean"] = mean
            row[f"rearrest_{years}yr_sd"] = sd
            row[f"rearrest_{years}yr_ci95_low"] = lo
            row[f"rearrest_{years}yr_ci95_high"] = hi

        for metric in (
            "mean_percent_days_employed",
            "mean_program_attendances",
            "mean_technical_violations",
        ):
            mean, sd, lo, hi = mean_ci95(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("cf_level").reset_index(drop=True)

    # Matched baseline comparison using the same run_id + seed_id + seed.
    baseline = raw[np.isclose(raw["cf_level"], 0.0)].copy()
    if baseline.empty:
        raise RuntimeError("The 0% calibrated baseline is missing.")
    if len(baseline) != expected_pairs:
        print(
            f"WARNING: expected {expected_pairs} baseline replications but found {len(baseline)}."
        )

    baseline_cols = {
        f"rearrest_{years}yr": f"baseline_rearrest_{years}yr"
        for years in WINDOWS
    }
    baseline_small = baseline[
        ["run_id", "seed_id", "seed"] + list(baseline_cols.keys())
    ].rename(columns=baseline_cols)

    paired_raw = raw.merge(
        baseline_small,
        on=["run_id", "seed_id", "seed"],
        how="left",
        validate="many_to_one",
    )

    for years in WINDOWS:
        paired_raw[f"delta_{years}yr"] = (
            paired_raw[f"rearrest_{years}yr"]
            - paired_raw[f"baseline_rearrest_{years}yr"]
        )
        paired_raw[f"delta_{years}yr_pp"] = paired_raw[f"delta_{years}yr"] * 100.0

    paired_rows = []
    for level, group in paired_raw.groupby("cf_level", sort=True):
        row = {
            "cf_level": float(level),
            "cf_percent": float(level * 100.0),
            "scenario_label": environment_label(float(level)),
            "hazard_multiplier": float(1.0 - level),
            "n_matched_pairs": int(len(group)),
        }

        for years in WINDOWS:
            mean, sd, lo, hi = mean_ci95(group[f"delta_{years}yr_pp"])
            row[f"delta_{years}yr_pp_mean"] = mean
            row[f"delta_{years}yr_pp_sd"] = sd
            row[f"delta_{years}yr_pp_ci95_low"] = lo
            row[f"delta_{years}yr_pp_ci95_high"] = hi

        paired_rows.append(row)

    paired = pd.DataFrame(paired_rows).sort_values("cf_level").reset_index(drop=True)

    # Mechanism summary: intended multiplier plus diagnostics.
    mechanism_cols = [
        "cf_level", "cf_percent", "scenario_label", "hazard_multiplier", "n_replications",
        "mean_percent_days_employed_mean", "mean_percent_days_employed_sd",
        "mean_program_attendances_mean", "mean_program_attendances_sd",
        "mean_technical_violations_mean", "mean_technical_violations_sd",
    ]
    mechanism = summary[[c for c in mechanism_cols if c in summary.columns]].copy()

    # Committee/dissertation table merges absolute rates with paired differences.
    table = summary.merge(
        paired,
        on=["cf_level", "cf_percent", "scenario_label", "hazard_multiplier"],
        how="left",
        validate="one_to_one",
    )

    dissertation_cols = [
        "scenario_label", "cf_percent", "hazard_multiplier", "n_replications",
        "rearrest_3yr_mean", "rearrest_6yr_mean", "rearrest_9yr_mean",
        "delta_3yr_pp_mean", "delta_3yr_pp_ci95_low", "delta_3yr_pp_ci95_high",
        "delta_6yr_pp_mean", "delta_6yr_pp_ci95_low", "delta_6yr_pp_ci95_high",
        "delta_9yr_pp_mean", "delta_9yr_pp_ci95_low", "delta_9yr_pp_ci95_high",
    ]
    dissertation_table = table[[c for c in dissertation_cols if c in table.columns]].copy()

    summary.to_csv(results_dir / "hazard_summary.csv", index=False)
    paired_raw.to_csv(results_dir / "hazard_paired_seed_raw.csv", index=False)
    paired.to_csv(results_dir / "hazard_paired_effects.csv", index=False)
    mechanism.to_csv(results_dir / "hazard_mechanism_summary.csv", index=False)
    dissertation_table.to_csv(results_dir / "hazard_dissertation_table.csv", index=False)

    return summary, paired, mechanism, dissertation_table


def build_incremental_rearrest_table(summary: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Decompose cumulative rearrest into 0-3, 3-6, and 6-9 year intervals."""
    interval = summary.copy()
    interval["rearrest_0_3"] = interval["rearrest_3yr_mean"]
    interval["rearrest_3_6"] = interval["rearrest_6yr_mean"] - interval["rearrest_3yr_mean"]
    interval["rearrest_6_9"] = interval["rearrest_9yr_mean"] - interval["rearrest_6yr_mean"]
    interval["rearrest_0_3_pp"] = interval["rearrest_0_3"] * 100.0
    interval["rearrest_3_6_pp"] = interval["rearrest_3_6"] * 100.0
    interval["rearrest_6_9_pp"] = interval["rearrest_6_9"] * 100.0
    out_cols = [
        "cf_level", "cf_percent", "scenario_label", "hazard_multiplier", "n_replications",
        "rearrest_0_3", "rearrest_3_6", "rearrest_6_9",
        "rearrest_0_3_pp", "rearrest_3_6_pp", "rearrest_6_9_pp",
    ]
    interval_table = interval[[c for c in out_cols if c in interval.columns]].copy()
    path = results_dir / "hazard_incremental_rearrest_by_period.csv"
    interval_table.to_csv(path, index=False)
    print(f"Incremental rearrest table -> {path}")
    return interval_table


# ================================================================
# DISSERTATION FIGURES
# ================================================================

def plot_rearrest_rates(summary: pd.DataFrame, charts_dir: Path):
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = summary["cf_percent"].to_numpy(dtype=float)

    for years in WINDOWS:
        y = summary[f"rearrest_{years}yr_mean"].to_numpy(dtype=float)
        lo = summary[f"rearrest_{years}yr_ci95_low"].to_numpy(dtype=float)
        hi = summary[f"rearrest_{years}yr_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])

        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", markersize=5.3, linewidth=2.0, capsize=3,
            label=f"{years}-Year Rearrest",
        )

    ax.axvline(0.0, linewidth=1.15, linestyle="--", color="0.35", alpha=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(float(np.min(x)) - 5, float(np.max(x)) + 5)
    ax.set_ylim(0.0, min(1.0, max(0.10, float(summary[[f"rearrest_{y}yr_ci95_high" for y in WINDOWS]].max().max()) + 0.05)))

    style_axis(
        ax,
        xlabel="Reentry Environment Scenario (% Change in Calibrated Hazard)",
        ylabel="Cumulative rearrest rate",
        title="System-Level Reentry Environment Counterfactual:\nRearrest Rates",
    )
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best")
    ax.text(
        0.02, 0.985,
        "Adverse Reentry Environment",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    ax.text(
        0.98, 0.985,
        "Favorable Reentry Environment",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    fig.text(
        0.5, 0.015,
        "Negative values represent increasingly adverse reentry conditions; positive values represent "
        "increasingly favorable reentry conditions. Scenario percentages are operationalized as "
        "proportional changes in the calibrated quarterly rearrest hazard. "
        f"Means across {CONFIG['N_RUNS'] * CONFIG['SEEDS_PER_RUN']} matched-seed ABM replications "
        "per level; error bars show 95% confidence intervals.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = charts_dir / "CF1_hazard_rearrest_rates.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_change_from_baseline(paired: pd.DataFrame, charts_dir: Path):
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = paired["cf_percent"].to_numpy(dtype=float)

    limits = []
    for years in WINDOWS:
        y = paired[f"delta_{years}yr_pp_mean"].to_numpy(dtype=float)
        lo = paired[f"delta_{years}yr_pp_ci95_low"].to_numpy(dtype=float)
        hi = paired[f"delta_{years}yr_pp_ci95_high"].to_numpy(dtype=float)
        limits.extend(lo.tolist())
        limits.extend(hi.tolist())

        ax.errorbar(
            x, y, yerr=np.vstack([y - lo, hi - y]),
            marker="o", markersize=5.5, linewidth=2.0, capsize=3,
            label=f"{years}-Year",
        )

    ax.axhline(0.0, linewidth=1.2, linestyle="--", color="0.35", alpha=0.9)
    ax.axvline(0.0, linewidth=1.0, linestyle=":", color="0.45", alpha=0.8)
    ylo, yhi = dynamic_limits(limits)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(float(np.min(x)) - 5, float(np.max(x)) + 5)

    style_axis(
        ax,
        xlabel="Reentry Environment Scenario (% Change in Calibrated Hazard)",
        ylabel="Change from calibrated baseline (percentage points)",
        title="System-Level Reentry Environment Counterfactual:\nChange from Calibrated Baseline",
    )
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best")

    ax.text(
        0.02, 0.985,
        "Adverse Reentry Environment",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.98, 0.985,
        "Favorable Reentry Environment",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    fig.text(
        0.5, 0.015,
        "Each point is a matched-seed mean difference relative to the calibrated baseline. "
        "Positive scenario values represent increasingly favorable reentry conditions; "
        "error bars show 95% confidence intervals.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = charts_dir / "CF2_hazard_change_from_baseline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mechanism_check(summary: pd.DataFrame, charts_dir: Path):
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = summary["cf_percent"].to_numpy(dtype=float)
    y = summary["hazard_multiplier"].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", markersize=6, linewidth=2.2)
    ax.axvline(0.0, linewidth=1.0, linestyle="--", color="0.35", alpha=0.8)
    ax.axhline(1.0, linewidth=1.0, linestyle=":", color="0.45", alpha=0.8)

    style_axis(
        ax,
        xlabel="Counterfactual Scenario (% Change in Calibrated Hazard)",
        ylabel="Multiplier on calibrated quarterly rearrest hazard",
        title="Mechanism Check:\nCounterfactual Scaling of the Calibrated Hazard",
    )

    ax.text(
        0.02, 0.04,
        "Baseline = 1.00x calibrated hazard\n50% favorable scenario = 0.50x hazard\n"
        "50% adverse scenario = 1.50x hazard",
        transform=ax.transAxes,
        fontsize=NOTE_FS,
        va="bottom",
    )

    fig.text(
        0.5, 0.015,
        "This mechanism-check figure retains the technical hazard terminology to verify how the reentry scenarios "
        "are operationalized; the baseline condition exactly preserves the calibrated hazard.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = charts_dir / "CF3_hazard_mechanism_check.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_by_horizon(paired: pd.DataFrame, charts_dir: Path):
    selected = paired[
        paired["cf_level"].apply(lambda v: any(np.isclose(v, s) for s in (-0.80, -0.50, -0.20, 0.20, 0.50, 0.80)))
    ].copy()

    fig, ax = plt.subplots(figsize=(12.2, 7.3))
    x = np.arange(len(WINDOWS), dtype=float)
    levels = sorted(selected["cf_level"].unique())
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(levels))

    for offset, level in zip(offsets, levels):
        row = selected[np.isclose(selected["cf_level"], level)].iloc[0]
        vals = [row[f"delta_{years}yr_pp_mean"] for years in WINDOWS]
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=environment_label(float(level)),
        )

    ax.axhline(0.0, linewidth=1.2, linestyle="--", color="0.35", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{years}-Year" for years in WINDOWS], fontsize=TICK_FS)

    style_axis(
        ax,
        xlabel="Follow-up horizon",
        ylabel="Change from baseline (percentage points)",
        title="Rearrest Effect by Follow-Up Horizon\nand Reentry Environment Scenario",
    )
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="best")

    fig.text(
        0.5, 0.015,
        "Selected adverse and favorable reentry scenarios show the magnitude and persistence of matched changes "
        "across 3-, 6-, and 9-year outcomes.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = charts_dir / "CF4_hazard_effect_by_horizon.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selected_scenarios(summary: pd.DataFrame, charts_dir: Path):
    selected = summary[
        summary["cf_level"].apply(lambda v: any(np.isclose(v, s) for s in SELECTED_LEVELS))
    ].copy().sort_values("cf_level")

    labels = [environment_label(float(v)) for v in selected["cf_level"]]
    x = np.arange(len(selected), dtype=float)
    width = 0.23

    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    for i, years in enumerate(WINDOWS):
        vals = selected[f"rearrest_{years}yr_mean"].to_numpy(dtype=float) * 100.0
        ax.bar(x + (i - 1) * width, vals, width=width, label=f"{years}-Year")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 100)

    style_axis(
        ax,
        xlabel="Reentry Environment Scenario",
        ylabel="Rearrest rate (%)",
        title="Selected Reentry Environment Scenarios:\nAggregate Rearrest Outcomes",
    )
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best")

    fig.text(
        0.5, 0.015,
        "Selected scenarios are shown for communication clarity; statistical analyses use all 17 counterfactual levels.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    path = charts_dir / "CF5_hazard_selected_scenarios.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_incremental_rearrest_by_period(summary: pd.DataFrame, charts_dir: Path):
    """CF6: Stacked decomposition of 9-year cumulative rearrest by follow-up period."""
    selected_levels = (0.00, 0.20, 0.50, 0.80)
    selected = summary[
        summary["cf_level"].apply(lambda v: any(np.isclose(v, s) for s in selected_levels))
    ].copy().sort_values("cf_level")

    # Mutually exclusive contributions to the 9-year cumulative rearrest rate.
    selected["rearrest_0_3_pp"] = selected["rearrest_3yr_mean"] * 100.0
    selected["rearrest_3_6_pp"] = (
        selected["rearrest_6yr_mean"] - selected["rearrest_3yr_mean"]
    ) * 100.0
    selected["rearrest_6_9_pp"] = (
        selected["rearrest_9yr_mean"] - selected["rearrest_6yr_mean"]
    ) * 100.0

    labels = []
    for v in selected["cf_level"]:
        if np.isclose(v, 0.0):
            labels.append("Calibrated\nBaseline")
        elif np.isclose(v, 0.20):
            labels.append("Moderately Favorable\nReentry Scenario\n(20%)")
        elif np.isclose(v, 0.50):
            labels.append("Strongly Favorable\nReentry Scenario\n(50%)")
        elif np.isclose(v, 0.80):
            labels.append("Very Strongly Favorable\nReentry Scenario\n(80%)")
        else:
            labels.append(environment_label(float(v)))
    x = np.arange(len(selected), dtype=float)

    early = selected["rearrest_0_3_pp"].to_numpy(dtype=float)
    middle = selected["rearrest_3_6_pp"].to_numpy(dtype=float)
    late = selected["rearrest_6_9_pp"].to_numpy(dtype=float)
    totals = early + middle + late

    fig, ax = plt.subplots(figsize=(11.8, 7.4))

    # Stacked bars show how the 9-year cumulative total is composed.
    ax.bar(x, early, width=0.62, label="Rearrest by Year 3")
    ax.bar(x, middle, width=0.62, bottom=early, label="Additional rearrest: Years 3–6")
    ax.bar(x, late, width=0.62, bottom=early + middle, label="Additional rearrest: Years 6–9")

    # Add the 9-year total above each bar for direct interpretation.
    for xpos, total in zip(x, totals):
        ax.text(
            xpos, total + 1.2, f"{total:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)

    style_axis(
        ax,
        xlabel="Reentry Environment Scenario",
        ylabel="Cumulative Rearrest Rate (%)",
        title="Decomposition of 9-Year Rearrest\nby Follow-Up Period",
    )

    ymax = float(np.nanmax(totals)) if len(totals) else 100.0
    ax.set_ylim(0.0, min(100.0, max(10.0, ymax + 8.0)))
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best")

    fig.text(
        0.5, 0.015,
        "Each stacked bar equals the 9-year cumulative rearrest rate. Scenario percentages represent "
        "proportional reductions in the calibrated quarterly rearrest hazard used to operationalize "
        "increasingly favorable system-level reentry conditions.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = charts_dir / "CF6_hazard_incremental_rearrest_by_period.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def generate_charts(summary: pd.DataFrame, paired: pd.DataFrame, charts_dir: Path):
    print("\nGenerating dissertation-ready figures...")
    paths = [
        plot_rearrest_rates(summary, charts_dir),
        plot_change_from_baseline(paired, charts_dir),
        plot_mechanism_check(summary, charts_dir),
        plot_effect_by_horizon(paired, charts_dir),
        plot_selected_scenarios(summary, charts_dir),
        plot_incremental_rearrest_by_period(summary, charts_dir),
    ]
    for path in paths:
        print(f"  Chart -> {path}")
    return paths


# ================================================================
# CONSOLE / DISSERTATION SUMMARY
# ================================================================

def print_summary_table(summary: pd.DataFrame, paired: pd.DataFrame):
    selected_summary = summary[
        summary["cf_level"].apply(lambda v: any(np.isclose(v, s) for s in SELECTED_LEVELS))
    ].copy()
    selected_paired = paired[
        paired["cf_level"].apply(lambda v: any(np.isclose(v, s) for s in SELECTED_LEVELS))
    ].copy()

    display = selected_summary.merge(
        selected_paired[[
            "cf_level", "delta_3yr_pp_mean", "delta_6yr_pp_mean", "delta_9yr_pp_mean"
        ]],
        on="cf_level",
        how="left",
    )

    display = display[[
        "scenario_label", "hazard_multiplier",
        "rearrest_3yr_mean", "rearrest_6yr_mean", "rearrest_9yr_mean",
        "delta_3yr_pp_mean", "delta_6yr_pp_mean", "delta_9yr_pp_mean",
    ]].copy()

    for col in ("rearrest_3yr_mean", "rearrest_6yr_mean", "rearrest_9yr_mean"):
        display[col] = display[col].map(lambda x: f"{x:.2%}")
    for col in ("delta_3yr_pp_mean", "delta_6yr_pp_mean", "delta_9yr_pp_mean"):
        display[col] = display[col].map(lambda x: f"{x:+.2f}")
    display["hazard_multiplier"] = display["hazard_multiplier"].map(lambda x: f"{x:.2f}x")

    print("\n" + "=" * 118)
    print("SELECTED SYSTEM-LEVEL REENTRY ENVIRONMENT RESULTS")
    print("=" * 118)
    print(display.to_string(index=False))
    print("\nRearrest columns are absolute rates; delta columns are percentage-point changes from matched baseline.")

    for target in (0.50, 0.80):
        row = paired[np.isclose(paired["cf_level"], target)]
        if not row.empty:
            r = row.iloc[0]
            print(f"\nEffect for {environment_label(target)} vs calibrated baseline:")
            for years in WINDOWS:
                print(
                    f"  {years}-year: {r[f'delta_{years}yr_pp_mean']:+.2f} pp "
                    f"(95% CI {r[f'delta_{years}yr_pp_ci95_low']:+.2f}, "
                    f"{r[f'delta_{years}yr_pp_ci95_high']:+.2f})"
                )

    print("=" * 118)


# ================================================================
# METADATA / REPLOT
# ================================================================

def save_metadata(results_dir: Path, n_workers: int, elapsed_seconds: float):
    metadata = {
        "experiment": "system_level_reentry_environment",
        "mechanism": "quarterly_rearrest_hazard_scaling",
        "formula": "q_cf = q_baseline * (1 - reentry_hazard_cf)",
        "interpretation": (
            "Positive counterfactual values represent increasingly favorable reentry conditions and "
            "negative values increasingly adverse conditions. These scenarios are operationalized "
            "through proportional changes in the calibrated quarterly rearrest hazard. This is a "
            "system-level sensitivity/bounding experiment, not a causal estimate for one policy."
        ),
        "initial_agents": CONFIG["initial_agents"],
        "monthly_intake": CONFIG["monthly_intake"],
        "warmup_months": CONFIG["warmup_months"],
        "study_months": CONFIG["study_months"],
        "bias_factor": CONFIG["bias_factor"],
        "enable_peer_influence": CONFIG["enable_peer_influence"],
        "n_runs": CONFIG["N_RUNS"],
        "seeds_per_run": CONFIG["SEEDS_PER_RUN"],
        "replications_per_level": CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"],
        "counterfactual_levels": [float(x) for x in CF_LEVELS],
        "hazard_multipliers": [float(1.0 - x) for x in CF_LEVELS],
        "n_levels": len(CF_LEVELS),
        "total_simulations": CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"] * len(CF_LEVELS),
        "matched_seed_design": True,
        "bjs_targets": CONFIG["bjs_targets"],
        "workers": int(n_workers),
        "elapsed_seconds": float(elapsed_seconds),
    }

    path = results_dir / "hazard_run_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata -> {path}")


def replot(results_dir: Path, charts_dir: Path):
    raw_path = results_dir / "hazard_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot replot: {raw_path} does not exist.")

    raw = pd.read_csv(raw_path)
    summary, paired, _, _ = summarize_experiment(raw, results_dir)
    build_incremental_rearrest_table(summary, results_dir)
    generate_charts(summary, paired, charts_dir)
    print_summary_table(summary, paired)


# ================================================================
# MAIN
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the dissertation-ready System-Level Reentry Environment "
            "counterfactual for the Recidivism ABM."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Recreate summaries and figures from hazard_raw.csv without rerunning the ABM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir, results_dir, charts_dir = ensure_output_dirs()
    workers = detect_workers(args.workers)

    print("\n" + "=" * 82)
    print("RECIDIVISM ABM — SYSTEM-LEVEL REENTRY ENVIRONMENT COUNTERFACTUAL")
    print("=" * 82)
    print(f"Output root      : {outdir.resolve()}")
    print(f"Model results    : {results_dir.resolve()}")
    print(f"Charts           : {charts_dir.resolve()}")
    print(f"Workers          : {workers}")
    print(f"Replication      : {CONFIG['N_RUNS']} runs x {CONFIG['SEEDS_PER_RUN']} seeds")
    print(f"Levels           : {len(CF_LEVELS)} ({CF_LEVELS[0]:+.0%} to {CF_LEVELS[-1]:+.0%})")
    print(f"Total tasks      : {CONFIG['N_RUNS'] * CONFIG['SEEDS_PER_RUN'] * len(CF_LEVELS):,}")
    print("=" * 82)

    if args.replot:
        replot(results_dir, charts_dir)
        print("\nReplot complete.")
        return

    validate_model_interface()

    start = time.time()
    raw = run_experiment(workers, results_dir)
    summary, paired, _, _ = summarize_experiment(raw, results_dir)
    build_incremental_rearrest_table(summary, results_dir)
    generate_charts(summary, paired, charts_dir)
    print_summary_table(summary, paired)

    elapsed = time.time() - start
    save_metadata(results_dir, workers, elapsed)

    print("\n" + "=" * 82)
    print("COUNTERFACTUAL ANALYSIS COMPLETE")
    print("=" * 82)
    print(f"Raw replications       -> {results_dir / 'hazard_raw.csv'}")
    print(f"Absolute-rate summary  -> {results_dir / 'hazard_summary.csv'}")
    print(f"Paired effects         -> {results_dir / 'hazard_paired_effects.csv'}")
    print(f"Dissertation table     -> {results_dir / 'hazard_dissertation_table.csv'}")
    print(f"Primary outcome figure -> {charts_dir / 'CF2_hazard_change_from_baseline.png'}")
    print(f"Committee summary      -> {charts_dir / 'CF5_hazard_selected_scenarios.png'}")
    print(f"Incremental-period fig -> {charts_dir / 'CF6_hazard_incremental_rearrest_by_period.png'}")
    print(f"Incremental-period CSV -> {results_dir / 'hazard_incremental_rearrest_by_period.csv'}")
    print("=" * 82)


if __name__ == "__main__":
    main()
