#!/usr/bin/env python3
# recidivism_multirun_convergence_parallel.py

import os, sys, time, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import cpu_count
from joblib import Parallel, delayed

# ─────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_peer_influence_config,
    get_uncalibrated_params,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_DIR = "Recidivism_MultiRun_Convergence"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MASTER_SEED = 424242   # single global seed for reproducibility
AGENT_SIZES = [100,250,500,1000,1500,2000,2500,3000]
N_RUNS = 10
SEEDS_PER_RUN = 20

# Baseline population size for intake scaling (1000-agent run uses intake=10).
# All other sizes scale proportionally so intake/population ratio stays constant.
BASELINE_AGENTS = 1000
BASELINE_INTAKE = 10

BASE_PARAMS = {
    "warmup_months": 144,
    "study_months": 108,
    "bias_factor": 0,
    "enable_peer_influence": True
}

BJS_TARGETS = {3: 0.68, 6: 0.79, 9: 0.83}

REQUESTED_CORES = 128
N_CORES = min(REQUESTED_CORES, cpu_count())

RUN_LEVEL_CSV = os.path.join(OUTPUT_DIR, "run_level_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "recidivism_summary.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "status.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ─────────────────────────────────────────────
# Single model run
# ─────────────────────────────────────────────
def run_single(seed, agent_size):
    # Scale intake proportionally so intake/population ratio stays constant
    # across all agent sizes (convergence test requires only N to vary).
    monthly_intake = max(1, round(BASELINE_INTAKE * agent_size / BASELINE_AGENTS))

    model = RecidivismModel(
        initial_agents=agent_size,
        seed=seed,
        monthly_intake=monthly_intake,
        calibration_params=get_uncalibrated_params(),
        **BASE_PARAMS
    )
    while model.running:
        model.step()

    return {
        3: model.calculate_flag_rate("rearrest_3_yrs"),
        6: model.calculate_flag_rate("rearrest_6_yrs"),
        9: model.calculate_flag_rate("rearrest_9_yrs"),
    }

# ─────────────────────────────────────────────
# One run block (multiple seeds)
# ─────────────────────────────────────────────
def run_seed_block(seeds, agent_size):
    results = [run_single(s, agent_size) for s in seeds]
    return {y: float(np.mean([r[y] for r in results])) for y in (3, 6, 9)}

# ─────────────────────────────────────────────
# One parallel task
# ─────────────────────────────────────────────
def run_task(agent_size, run_id):
    # Create a unique RNG per (agent_size, run_id)
    rng = np.random.default_rng(
        MASTER_SEED + agent_size * 10_000 + run_id
    )

    # Draw truly random, non-overlapping seeds
    seeds = rng.integers(
        low=1,
        high=2_000_000_000,
        size=SEEDS_PER_RUN,
        dtype=np.int64
    ).tolist()

    t0 = time.time()
    try:
        block = run_seed_block(seeds, agent_size)
        status, err = "ok", ""
    except Exception as e:
        block = {3: np.nan, 6: np.nan, 9: np.nan}
        status, err = "fail", repr(e)

    return {
        "agent_size": agent_size,
        "run_id": run_id,
        "seeds": ",".join(map(str, seeds)),   # 🔍 logged for audit
        "recid_3": block[3],
        "recid_6": block[6],
        "recid_9": block[9],
        "status": status,
        "elapsed_sec": round(time.time() - t0, 2),
        "error": err
    }


# ─────────────────────────────────────────────
# t-critical (two-sided 95%) — avoids scipy dependency
# ─────────────────────────────────────────────
_T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,
        8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,14:2.145,19:2.093,
        24:2.064,29:2.045,39:2.023,49:2.010,59:2.000,99:1.984}

def _t95(n: int) -> float:
    df = max(1, n - 1)
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


# ─────────────────────────────────────────────
# Parallel experiment
# ─────────────────────────────────────────────
def run_experiment_parallel():
    tasks = [(a, r) for a in AGENT_SIZES for r in range(N_RUNS)]

    print(f"🧵 Using {N_CORES} cores")
    logging.info(f"Starting experiment with {len(tasks)} tasks")

    records = Parallel(
        n_jobs=N_CORES,
        backend="loky",
        verbose=10
    )(delayed(run_task)(a, r) for (a, r) in tasks)

    df_runs = pd.DataFrame(records)
    df_runs.to_csv(RUN_LEVEL_CSV, index=False)

    # summary
    rows = []
    for agent_size in AGENT_SIZES:
        sub = df_runs[(df_runs.agent_size == agent_size) & (df_runs.status == "ok")]
        for y, col in [(3, "recid_3"), (6, "recid_6"), (9, "recid_9")]:
            vals = sub[col].dropna().values
            rows.append({
                "agent_size": agent_size,
                "year": y,
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "ci": _t95(len(vals)) * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals)
            })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    return df_runs, df_summary

# ─────────────────────────────────────────────
# Line-chart aggregation
# ─────────────────────────────────────────────
def aggregate_runs_for_lines(df_runs):
    rows = []
    for agent_size in sorted(df_runs.agent_size.unique()):
        sub = df_runs[(df_runs.agent_size == agent_size) & (df_runs.status == "ok")]
        for y, col in [(3, "recid_3"), (6, "recid_6"), (9, "recid_9")]:
            vals = sub[col].values
            rows.append({
                "agent_size": agent_size,
                "year": y,
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Plotting (Mean ± 95% CI error bars)
# Uses df_summary which already contains: mean, ci, n
# ─────────────────────────────────────────────
def plot_line_by_year_ci(df_summary):
    """
    One plot per year (3/6/9). Forces ALL agent sizes to appear on x-axis.
    """
    # Use all sizes (across all years) so ticks don't disappear if a year is missing a size
    all_sizes = np.sort(df_summary["agent_size"].unique())

    for y in (3, 6, 9):
        sub = df_summary[df_summary["year"] == y].sort_values("agent_size")
        if sub.empty:
            continue

        x = np.asarray(sub["agent_size"], dtype=float)
        mean = np.asarray(sub["mean"], dtype=float) * 100
        ci = np.asarray(sub["ci"], dtype=float) * 100

        plt.figure(figsize=(10, 5))
        plt.errorbar(x, mean, yerr=ci, fmt="o-", capsize=5, label=f"{y}-Year Mean ±95% CI")

        # BJS reference line
        plt.axhline(BJS_TARGETS[y] * 100, linestyle="--", linewidth=1, label="BJS Target")

        # ⭐ Force all agent sizes to show (not just the ones in this subset)
        plt.xticks(all_sizes, rotation=45)

        plt.xlabel("Agent Size")
        plt.ylabel(f"{y}-Year Recidivism Rate (%)")
        plt.title(f"{y}-Year Recidivism Rate vs Agent Size (±95% CI)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"recidivism_{y}yr_vs_agent_size_ci.png"), dpi=300)
        plt.close()


def plot_consolidated_lines_ci(df_summary):
    """
    Single plot with 3 lines (3/6/9 year). Forces ALL agent sizes to appear on x-axis.
    """
    plt.figure(figsize=(11, 6))

    # ⭐ Use ALL sizes across all years for x ticks
    all_sizes = np.sort(df_summary["agent_size"].unique())

    for y in (3, 6, 9):
        sub = df_summary[df_summary["year"] == y].sort_values("agent_size")
        if sub.empty:
            continue

        x = np.asarray(sub["agent_size"], dtype=float)
        mean = np.asarray(sub["mean"], dtype=float) * 100
        ci = np.asarray(sub["ci"], dtype=float) * 100

        plt.errorbar(
            x, mean, yerr=ci,
            fmt="o-", capsize=4,
            label=f"{y}-Year (Mean ±95% CI)"
        )

        # BJS reference line for that horizon
        plt.axhline(BJS_TARGETS[y] * 100, linestyle="--", linewidth=1, alpha=0.7)

    # ⭐ Force all agent sizes to show
    plt.xticks(all_sizes, rotation=45)

    plt.xlabel("Agent Size")
    plt.ylabel("Recidivism Rate (%)")
    plt.title("Recidivism Rate vs Agent Size (Mean ±95% CI)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "recidivism_consolidated_lines_ci.png"), dpi=300)
    plt.close()


def plot_variance_vs_agent_size_all_years(df_runs):
    """
    Variance–N convergence diagnostic for 3-, 6-, and 9-year recidivism.
    Shows how variance of estimated recidivism rates across runs decreases with agent size.
    """
    plt.figure(figsize=(10, 6))

    # ⭐ Force all sizes to show on x-axis
    all_sizes = np.sort(df_runs["agent_size"].unique())

    for year, col in [(3, "recid_3"), (6, "recid_6"), (9, "recid_9")]:
        rows = []

        for agent_size in all_sizes:
            sub = df_runs[
                (df_runs["agent_size"] == agent_size) &
                (df_runs["status"] == "ok")
            ][col].dropna()

            var = float(np.var(sub, ddof=1)) if len(sub) > 1 else 0.0

            rows.append({"agent_size": agent_size, "variance": var})

        df_var = pd.DataFrame(rows)

        plt.plot(
            np.asarray(df_var["agent_size"], dtype=float),
            np.asarray(df_var["variance"], dtype=float),
            marker="o",
            label=f"{year}-Year"
        )

    plt.xticks(all_sizes, rotation=45)
    plt.xlabel("Agent Size (N)")
    plt.ylabel("Variance of Estimated Recidivism Rate")
    plt.title("Variance–N Convergence Diagnostic (3, 6, 9 Years)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "variance_vs_agent_size_all_years.png"), dpi=300)
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_runs, df_summary = run_experiment_parallel()
    df_line = aggregate_runs_for_lines(df_runs)

    #plot_line_by_year(df_line)
    #plot_consolidated_lines(df_line)

    # CI errorbar plots (matches your sample chart style)
    plot_line_by_year_ci(df_summary)
    plot_consolidated_lines_ci(df_summary)
    plot_variance_vs_agent_size_all_years(df_runs)


    print("\n✅ CI error-bar multi-run recidivism analysis complete")
    print(f"📁 Outputs saved in {OUTPUT_DIR}/")

