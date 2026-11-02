#!/usr/bin/env python3
"""
counterfactual_analysis.py
===============================================================
Dissertation-ready counterfactual experiment runner
for the Recidivism Agent-Based Model.

Experimental design
-------------------
For EACH intervention level:
    10 runs × 20 seeds = 200 model replications

The SAME matched seeds are reused across all intervention levels.

Default intervention sweep:
    -80% to +80% in 10-percentage-point increments


Experiments:
    employment : changes post-release employment opportunity
    program    : changes program attendance opportunity
    both       : runs both sweeps separately

Outputs:
    counterfactual_output/
        model_results/
            employment_raw.csv
        employment_summary.csv
        employment_paired_effects.csv
        employment_mechanism_summary.csv
        charts/
            CF1_employment_rearrest_rates.png
        CF2_employment_change_from_baseline.png
        CF3_employment_mechanism_check.png
        CF4_employment_effect_by_horizon.png

        program_raw.csv
        program_summary.csv
        program_paired_effects.csv
        program_mechanism_summary.csv
        CF1_program_rearrest_rates.png
        CF2_program_change_from_baseline.png
        CF3_program_mechanism_check.png
        CF4_program_effect_by_horizon.png

        counterfactual_run_metadata.json

Run:
    python counterfactual_analysis.py
    python counterfactual_analysis.py --experiment employment
    python counterfactual_analysis.py --experiment program
    python counterfactual_analysis.py --experiment both
    python counterfactual_analysis.py --workers 120
    python counterfactual_analysis.py --replot

Notes
-----
1. The full 144-month warm-up + 108-month study are run for every
   simulation, exactly as in the validation pipeline.
2. The intervention logic itself should be gated inside Person so the
   intervention is active only after warm-up.
3. Outcomes are calculated only from study_eligible_agent records.
4. Counterfactual factors are attached to the model object before
   model.step() begins. This works whether or not RecidivismModel.__init__
   explicitly declares employment_cf/program_cf.
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

    # Requested validation-style replication design
    "N_RUNS": 5,
    "SEEDS_PER_RUN": 10,
    "BASE_SEED": 1000,

    "output_directory": "counterfactual_output",
    "results_subdir": "model_results",
    "charts_subdir": "charts",

    # Contextual BJS reference values
    "bjs_targets": {3: 0.684, 6: 0.794, 9: 0.834},
}

WINDOWS = (3, 6, 9)

CF_LEVELS = np.array(
    [-0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10,
      0.00,
      0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80],
    dtype=float,
)


DPI = 350
TITLE_FS = 15
LABEL_FS = 12
TICK_FS = 10
LEGEND_FS = 10
NOTE_FS = 9


# ================================================================
# HELPERS
# ================================================================

def ensure_output_dirs():
    """
    Keep model/tabular outputs separate from dissertation figures.

    counterfactual_output/
        model_results/
        charts/
    """
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


def intervention_label(level: float) -> str:
    return "Baseline" if np.isclose(level, 0.0) else f"{level:+.0%}"


def style_axis(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=8)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ================================================================
# MODEL BUILD
# ================================================================

def build_model(
    seed: int,
    employment_cf: float,
    program_cf: float,
) -> RecidivismModel:
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
    }

    signature = inspect.signature(RecidivismModel.__init__).parameters

    if "employment_cf" in signature:
        kwargs["employment_cf"] = float(employment_cf)
    if "program_cf" in signature:
        kwargs["program_cf"] = float(program_cf)
    if "output_directory" in signature:
        kwargs["output_directory"] = str(Path(CONFIG["output_directory"]) / "tmp")

    model = RecidivismModel(**kwargs)

    # Compatibility with model versions that do not expose these in __init__.
    model.employment_cf = float(employment_cf)
    model.program_cf = float(program_cf)

    if hasattr(model, "export_csv"):
        model.export_csv = False

    return model


# ================================================================
# RESULT EXTRACTION
# ================================================================

def extract_results(
    model, run_id, seed_id, seed, experiment, cf_level
) -> Dict:
    eligible = [
        a for a in model.schedule.agents
        if getattr(a, "study_eligible_agent", False)
    ]

    n = len(eligible)
    if n == 0:
        raise RuntimeError(
            f"No study-eligible agents returned (run={run_id}, seed={seed}, level={cf_level:+.2f})"
        )

    row = {
        "experiment": experiment,
        "cf_level": float(cf_level),
        "cf_percent": float(cf_level * 100.0),
        "run_id": int(run_id),
        "seed_id": int(seed_id),
        "seed": int(seed),
        "n_eligible": int(n),
    }

    for years in WINDOWS:
        flag = f"rearrest_{years}_yrs"
        count = sum(1 for a in eligible if bool(getattr(a, flag, False)))
        row[f"rearrest_{years}yr_count"] = int(count)
        row[f"rearrest_{years}yr"] = count / n

    employed = np.array([
        float(getattr(a, "Percent_Days_Employed", np.nan)) for a in eligible
    ], dtype=float)
    attendance = np.array([
        float(getattr(a, "Program_Attendances", np.nan)) for a in eligible
    ], dtype=float)
    absences = np.array([
        float(getattr(a, "Program_UnexcusedAbsences", np.nan)) for a in eligible
    ], dtype=float)

    def numeric_or_nan(v):
        return float(v) if isinstance(v, (int, float, bool, np.integer, np.floating)) else np.nan

    technical = np.array([
        numeric_or_nan(getattr(a, "Violations_Technical", np.nan)) for a in eligible
    ], dtype=float)

    row["mean_percent_days_employed"] = (
        float(np.nanmean(employed)) if np.isfinite(employed).any() else np.nan
    )
    row["mean_program_attendances"] = (
        float(np.nanmean(attendance)) if np.isfinite(attendance).any() else np.nan
    )
    row["mean_program_unexcused_absences"] = (
        float(np.nanmean(absences)) if np.isfinite(absences).any() else np.nan
    )
    row["mean_technical_violations"] = (
        float(np.nanmean(technical)) if np.isfinite(technical).any() else np.nan
    )

    return row


# ================================================================
# WORKER / TASKS
# ================================================================

def worker(task: Tuple) -> Dict:
    run_id, seed_id, seed, experiment, cf_level = task

    if experiment == "employment":
        employment_cf, program_cf = cf_level, 0.0
    elif experiment == "program":
        employment_cf, program_cf = 0.0, cf_level
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    model = build_model(seed, employment_cf, program_cf)

    # Same execution pattern used by validation scripts.
    while model.running:
        model.step()

    return extract_results(
        model, run_id, seed_id, seed, experiment, cf_level
    )


def build_tasks(
    experiment: str,
    cf_level: float,
) -> List[Tuple]:
    return [
        (
            run_id, seed_id, seed, experiment, float(cf_level)
        )
        for run_id, seed_id, seed in matched_seed_grid()
    ]


def run_task_batch(tasks: List[Tuple], n_workers: int, description: str) -> List[Dict]:
    results = []
    max_workers = min(n_workers, len(tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc=description, ncols=88)
            if tqdm is not None
            else as_completed(futures)
        )

        for future in iterator:
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                run_id, seed_id, seed, exp, level = task
                print(
                    f"\nWARNING: failed task experiment={exp}, level={level:+.0%}, "
                    f"run={run_id}, seed_id={seed_id}, seed={seed}: {exc}",
                    flush=True,
                )

    return results


# ================================================================
# RUN EXPERIMENT
# ================================================================

def run_experiment(
    experiment: str,
    n_workers: int,
    results_dir: Path,
) -> pd.DataFrame:

    all_rows = []
    tasks_per_level = CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"]
    total_levels = len(CF_LEVELS)
    total_tasks = tasks_per_level * total_levels

    print("\n" + "=" * 78)
    print(f"COUNTERFACTUAL EXPERIMENT: {experiment.upper()}")
    print("=" * 78)
    print(f"Replication design : {CONFIG['N_RUNS']} runs × {CONFIG['SEEDS_PER_RUN']} seeds")
    print(f"CF levels          : {total_levels}")
    print(f"Tasks per level    : {tasks_per_level}")
    print(f"Total simulations  : {total_tasks:,}")
    print(f"Workers            : {n_workers}")
    print(f"Warm-up / Study    : {CONFIG['warmup_months']} / {CONFIG['study_months']} months")
    print("=" * 78)

    for level_idx, cf_level in enumerate(CF_LEVELS, start=1):
        tasks = build_tasks(experiment, float(cf_level))

        print(
            f"\n[{level_idx:02d}/{total_levels:02d}] "
            f"{experiment.capitalize()} {intervention_label(cf_level)}"
        )

        level_results = run_task_batch(
            tasks,
            n_workers,
            f"  {experiment} {cf_level:+.0%}",
        )

        if len(level_results) != len(tasks):
            print(
                f"  WARNING: received {len(level_results)} / {len(tasks)} "
                f"successful replications."
            )

        all_rows.extend(level_results)

        checkpoint_path = results_dir / f"{experiment}_raw_checkpoint.csv"
        pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
        print(f"  Checkpoint -> {checkpoint_path}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError(f"No results returned for {experiment} experiment.")

    df.sort_values(
        ["cf_level", "run_id", "seed_id"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    raw_path = results_dir / f"{experiment}_raw.csv"
    df.to_csv(raw_path, index=False)

    checkpoint_path = results_dir / f"{experiment}_raw_checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nRaw results -> {raw_path}")
    return df


# ================================================================
# SUMMARIES / MATCHED BASELINE EFFECTS
# ================================================================

def summarize_experiment(
    raw: pd.DataFrame,
    experiment: str,
    results_dir: Path,
):
    summary_rows = []

    for level, group in raw.groupby("cf_level", sort=True):
        row = {
            "experiment": experiment,
            "cf_level": float(level),
            "cf_percent": float(level * 100.0),
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
            "mean_program_unexcused_absences",
            "mean_technical_violations",
        ):
            mean, sd, lo, hi = mean_ci95(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("cf_level")

    baseline = raw[np.isclose(raw["cf_level"], 0.0)].copy()
    if baseline.empty:
        raise RuntimeError(f"Baseline level cf_level=0 is missing for {experiment}.")

    baseline_cols = {
        f"rearrest_{years}yr": f"baseline_rearrest_{years}yr"
        for years in WINDOWS
    }

    baseline_small = baseline[
        ["run_id", "seed_id", "seed"] + list(baseline_cols.keys())
    ].rename(columns=baseline_cols)

    paired = raw.merge(
        baseline_small,
        on=["run_id", "seed_id", "seed"],
        how="left",
        validate="many_to_one",
    )

    for years in WINDOWS:
        paired[f"delta_{years}yr"] = (
            paired[f"rearrest_{years}yr"] - paired[f"baseline_rearrest_{years}yr"]
        )
        paired[f"delta_{years}yr_pp"] = paired[f"delta_{years}yr"] * 100.0

    paired_rows = []
    for level, group in paired.groupby("cf_level", sort=True):
        row = {
            "experiment": experiment,
            "cf_level": float(level),
            "cf_percent": float(level * 100.0),
            "n_matched_pairs": int(len(group)),
        }

        for years in WINDOWS:
            mean, sd, lo, hi = mean_ci95(group[f"delta_{years}yr_pp"])
            row[f"delta_{years}yr_pp_mean"] = mean
            row[f"delta_{years}yr_pp_sd"] = sd
            row[f"delta_{years}yr_pp_ci95_low"] = lo
            row[f"delta_{years}yr_pp_ci95_high"] = hi

        paired_rows.append(row)

    paired_summary = pd.DataFrame(paired_rows).sort_values("cf_level")

    mechanism_cols = [
        "experiment", "cf_level", "cf_percent", "n_replications",
        "mean_percent_days_employed_mean",
        "mean_percent_days_employed_sd",
        "mean_percent_days_employed_ci95_low",
        "mean_percent_days_employed_ci95_high",
        "mean_program_attendances_mean",
        "mean_program_attendances_sd",
        "mean_program_attendances_ci95_low",
        "mean_program_attendances_ci95_high",
        "mean_program_unexcused_absences_mean",
        "mean_program_unexcused_absences_sd",
        "mean_technical_violations_mean",
        "mean_technical_violations_sd",
    ]
    mechanism = summary[[c for c in mechanism_cols if c in summary.columns]].copy()

    summary.to_csv(results_dir / f"{experiment}_summary.csv", index=False)
    paired.to_csv(results_dir / f"{experiment}_paired_seed_raw.csv", index=False)
    paired_summary.to_csv(results_dir / f"{experiment}_paired_effects.csv", index=False)
    mechanism.to_csv(results_dir / f"{experiment}_mechanism_summary.csv", index=False)

    return summary, paired_summary, mechanism


# ================================================================
# DISSERTATION FIGURES
# ================================================================



def plot_rearrest_rates(summary, experiment, charts_dir):
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = summary["cf_percent"].to_numpy()

    for years in WINDOWS:
        y = summary[f"rearrest_{years}yr_mean"].to_numpy()
        lo = summary[f"rearrest_{years}yr_ci95_low"].to_numpy()
        hi = summary[f"rearrest_{years}yr_ci95_high"].to_numpy()
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(
            x, y, yerr=yerr, marker="o", markersize=5,
            linewidth=2.1, capsize=3, label=f"{years}-Year Rearrest"
        )

    ax.axvline(0, linewidth=1.4, linestyle="--", alpha=0.7, label="Calibrated Baseline")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    style_axis(
        ax,
        xlabel=f"{experiment.capitalize()} counterfactual change (%)",
        ylabel="Rearrest rate",
        title=f"Counterfactual Effect of {experiment.capitalize()} on Rearrest",
    )
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best")

    fig.text(
        0.5, 0.015,
        f"Means across {CONFIG['N_RUNS'] * CONFIG['SEEDS_PER_RUN']} matched-seed ABM replications per intervention level; error bars show 95% confidence intervals.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = charts_dir / f"CF1_{experiment}_rearrest_rates.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_change_from_baseline(paired, experiment, charts_dir):
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = paired["cf_percent"].to_numpy()

    for years in WINDOWS:
        y = paired[f"delta_{years}yr_pp_mean"].to_numpy()
        lo = paired[f"delta_{years}yr_pp_ci95_low"].to_numpy()
        hi = paired[f"delta_{years}yr_pp_ci95_high"].to_numpy()
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(
            x, y, yerr=yerr, marker="o", markersize=5,
            linewidth=2.1, capsize=3, label=f"{years}-Year"
        )

    ax.axhline(0, linewidth=1.4, linestyle="--", alpha=0.75)
    ax.axvline(0, linewidth=1.0, linestyle=":", alpha=0.55)

    style_axis(
        ax,
        xlabel=f"{experiment.capitalize()} counterfactual change (%)",
        ylabel="Change from baseline (percentage points)",
        title=f"Change in Rearrest Relative to the Calibrated Baseline: {experiment.capitalize()}",
    )
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best")

    fig.text(
        0.5, 0.015,
        "Each estimate is paired to the identical run and random seed at the 0% intervention baseline; negative values indicate lower rearrest.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = charts_dir / f"CF2_{experiment}_change_from_baseline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mechanism_check(summary, experiment, charts_dir):
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = summary["cf_percent"].to_numpy()

    if experiment == "employment":
        y = summary["mean_percent_days_employed_mean"].to_numpy()
        ylabel = "Mean percent days employed"
        title = "Mechanism Check: Employment Counterfactual Changes Post-Release Employment"
    else:
        y = summary["mean_program_attendances_mean"].to_numpy()
        ylabel = "Mean program attendances"
        title = "Mechanism Check: Program Counterfactual Changes Program Participation"

    ax.plot(x, y, marker="o", markersize=6, linewidth=2.2)
    ax.axvline(0, linewidth=1.4, linestyle="--", alpha=0.7)

    style_axis(
        ax,
        xlabel=f"{experiment.capitalize()} counterfactual change (%)",
        ylabel=ylabel,
        title=title,
    )

    fig.text(
        0.5, 0.015,
        f"Mechanism outcomes are averaged across study-eligible agents within each of {CONFIG['N_RUNS'] * CONFIG['SEEDS_PER_RUN']} model replications per intervention level.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = charts_dir / f"CF3_{experiment}_mechanism_check.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_by_horizon(paired, experiment, charts_dir):
    selected = paired[paired["cf_level"].isin([-0.50, -0.20, 0.20, 0.50])].copy()

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = np.arange(len(WINDOWS), dtype=float)
    width = 0.18
    levels = sorted(selected["cf_level"].unique())
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(levels))

    for offset, level in zip(offsets, levels):
        row = selected[np.isclose(selected["cf_level"], level)].iloc[0]
        vals = [row[f"delta_{years}yr_pp_mean"] for years in WINDOWS]
        ax.bar(x + offset, vals, width=width, label=intervention_label(level))

    ax.axhline(0, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{years}-Year" for years in WINDOWS], fontsize=TICK_FS)

    style_axis(
        ax,
        xlabel="Follow-up horizon",
        ylabel="Change from baseline (percentage points)",
        title=f"Counterfactual Effect by Follow-Up Horizon: {experiment.capitalize()}",
    )
    ax.legend(title="Intervention change", fontsize=LEGEND_FS, title_fontsize=LEGEND_FS, frameon=False)

    fig.text(
        0.5, 0.015,
        "Bars show matched-seed mean differences relative to the calibrated 0% intervention condition.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = charts_dir / f"CF4_{experiment}_effect_by_horizon.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_charts(summary, paired, experiment, charts_dir):
    print(f"\nGenerating dissertation-ready {experiment} figures...")
    paths = [
        plot_rearrest_rates(summary, experiment, charts_dir),
        plot_change_from_baseline(paired, experiment, charts_dir),
        plot_mechanism_check(summary, experiment, charts_dir),
        plot_effect_by_horizon(paired, experiment, charts_dir),
    ]
    for path in paths:
        print(f"  Chart -> {path}")


# ================================================================
# CONSOLE SUMMARY
# ================================================================

def print_summary_table(summary, paired, experiment):
    print("\n" + "=" * 90)
    print(f"{experiment.upper()} COUNTERFACTUAL SUMMARY ({CONFIG['N_RUNS'] * CONFIG['SEEDS_PER_RUN']} REPLICATIONS PER LEVEL)")
    print("=" * 90)

    display = summary[[
        "cf_percent", "n_replications",
        "rearrest_3yr_mean", "rearrest_6yr_mean", "rearrest_9yr_mean",
    ]].copy()

    display["cf_percent"] = display["cf_percent"].map(lambda x: f"{x:+.0f}%")
    for c in ("rearrest_3yr_mean", "rearrest_6yr_mean", "rearrest_9yr_mean"):
        display[c] = display[c].map(lambda x: f"{x:.2%}")

    print(display.to_string(index=False))

    positive = paired[np.isclose(paired["cf_level"], 0.50)]
    if not positive.empty:
        r = positive.iloc[0]
        print("\nEffect at +50% intervention vs baseline:")
        for years in WINDOWS:
            print(
                f"  {years}-year: {r[f'delta_{years}yr_pp_mean']:+.2f} pp "
                f"(95% CI {r[f'delta_{years}yr_pp_ci95_low']:+.2f}, "
                f"{r[f'delta_{years}yr_pp_ci95_high']:+.2f})"
            )

    print("=" * 90)


# ================================================================
# REPLOT / METADATA
# ================================================================

def load_existing_and_replot(
    experiment: str,
    results_dir: Path,
    charts_dir: Path,
):
    raw_path = (
        results_dir /
        f"{experiment}_raw.csv"
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Cannot replot: {raw_path} does not exist."
        )

    raw = pd.read_csv(raw_path)

    summary, paired, _ = summarize_experiment(
        raw,
        experiment,
        results_dir,
    )

    generate_charts(
        summary,
        paired,
        experiment,
        charts_dir,
    )

    print_summary_table(
        summary,
        paired,
        experiment,
    )


def save_metadata(
                        results_dir: Path,
                        experiments: List[str],
                        n_workers: int,
                        elapsed_seconds=None,
                    ):
    metadata = {
        "experiments": experiments,
        "initial_agents": CONFIG["initial_agents"],
        "monthly_intake": CONFIG["monthly_intake"],
        "warmup_months": CONFIG["warmup_months"],
        "study_months": CONFIG["study_months"],
        "bias_factor": CONFIG["bias_factor"],
        "enable_peer_influence": CONFIG["enable_peer_influence"],
        "n_runs": CONFIG["N_RUNS"],
        "seeds_per_run": CONFIG["SEEDS_PER_RUN"],
        "replications_per_level": CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"],
        "cf_levels": [float(x) for x in CF_LEVELS],
        "n_cf_levels": len(CF_LEVELS),
        "simulations_per_experiment": {
            "employment": CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"] * len(CF_LEVELS),
            "program": CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"] * len(CF_LEVELS),
        },
        "workers": int(n_workers),
        "bjs_targets": CONFIG["bjs_targets"],
        "matched_seed_design": True,
        "elapsed_seconds": elapsed_seconds,
    }

    with open(results_dir / "counterfactual_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ================================================================
# MAIN
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dissertation-ready Recidivism ABM counterfactual experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=("employment", "program", "both"),
        default="both",
        help="Counterfactual experiment to run. Default: both.",
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
        help="Regenerate summaries/figures from existing raw CSVs without rerunning ABM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir, results_dir, charts_dir = ensure_output_dirs()
    workers = detect_workers(args.workers)

    experiments = ["employment", "program"] if args.experiment == "both" else [args.experiment]

    print("\n" + "=" * 78)
    print("RECIDIVISM ABM — COUNTERFACTUAL ANALYSIS")
    print("=" * 78)
    print(f"Output root      : {outdir.resolve()}")
    print(f"Model results    : {results_dir.resolve()}")
    print(f"Charts           : {charts_dir.resolve()}")
    print(f"Workers          : {workers}")
    print(f"Replication      : {CONFIG['N_RUNS']} runs × {CONFIG['SEEDS_PER_RUN']} seeds")
    print(f"Tasks per level  : {CONFIG['N_RUNS'] * CONFIG['SEEDS_PER_RUN']}")
    print(f"Levels           : {len(CF_LEVELS)} ({CF_LEVELS[0]:+.0%} to {CF_LEVELS[-1]:+.0%})")
    print(
        f"Per experiment   : "
        f"{CONFIG['N_RUNS'] * CONFIG['SEEDS_PER_RUN'] * len(CF_LEVELS):,} simulations"
    )
    print(f"Experiments      : {', '.join(experiments)}")
    print("=" * 78)

    start = time.time()

    if args.replot:
        for experiment in experiments:
            load_existing_and_replot(
                                    experiment,
                                    results_dir,
                                    charts_dir,
                                )
        save_metadata(results_dir, experiments, workers, elapsed_seconds=0.0)
        print("\nReplot complete.")
        return

    for experiment in experiments:
        raw = run_experiment(
                            experiment,
                            workers,
                            results_dir,
                        )
        summary, paired, _ = summarize_experiment(
                            raw,
                            experiment,
                            results_dir,
                        )
        generate_charts(summary, paired, experiment, charts_dir)
        print_summary_table(summary, paired, experiment)

    elapsed = time.time() - start
    save_metadata(results_dir, experiments, workers, elapsed)

    print("\n" + "=" * 78)
    print("COUNTERFACTUAL ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Model results -> {results_dir.resolve()}")
    print(f"Charts        -> {charts_dir.resolve()}")
    print("Primary dissertation figure: CF2_*_change_from_baseline.png")
    print("Primary dissertation table : *_paired_effects.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
