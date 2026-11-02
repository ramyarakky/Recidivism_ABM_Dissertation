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
    residence  : changes post-release residential-instability transition probability
                 (+ values reduce instability; - values increase instability)
    both       : runs employment + program sweeps separately
    all        : runs employment + program + residence + supervision

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
    "N_RUNS": 10,
    "SEEDS_PER_RUN": 20,
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


# ================================================================
# RISK-TARGETED SUPERVISION COUNTERFACTUAL
# ================================================================
# Raw policy weights. These are normalized against the PCRA-tier
# distribution measured from the matched calibrated baseline so the
# population-weighted mean multiplier remains 1.0.
SUPERVISION_SCENARIOS = {
    "baseline": {
        "Low": 1.00,
        "LowModerate": 1.00,
        "Moderate": 1.00,
        "High": 1.00,
    },
    "mild_targeting": {
        "Low": 0.80,
        "LowModerate": 0.90,
        "Moderate": 1.10,
        "High": 1.30,
    },
    "moderate_targeting": {
        "Low": 0.60,
        "LowModerate": 0.80,
        "Moderate": 1.10,
        "High": 1.40,
    },
    "strong_targeting": {
        "Low": 0.40,
        "LowModerate": 0.65,
        "Moderate": 1.15,
        "High": 1.60,
    },
}


def normalize_smi_multipliers(raw_multipliers, tier_shares):
    """Normalize tier multipliers to a population-weighted mean of 1.0."""
    weighted_mean = sum(
        float(tier_shares[tier]) * float(raw_multipliers[tier])
        for tier in raw_multipliers
    )
    if weighted_mean <= 0:
        raise ValueError("Weighted SMI multiplier must be > 0.")

    return {
        tier: float(raw_multipliers[tier]) / weighted_mean
        for tier in raw_multipliers
    }


DPI = 350
TITLE_FS = 15
LABEL_FS = 12
TICK_FS = 10
LEGEND_FS = 10
NOTE_FS = 9

# 95% CI presentation settings.  Multi-horizon point estimates are deliberately
# offset within each intervention/scenario so confidence intervals remain
# visually distinct.  The offset is graphical only; nominal x values do not change.
CI_CAPSIZE = 6
CI_ELINEWIDTH = 2.2
CI_MARKERSIZE = 7.0
CI_MARKEREDGEWIDTH = 1.0
CONTINUOUS_CI_XOFFSETS = {3: -1.8, 6: 0.0, 9: 1.8}
CATEGORICAL_CI_XOFFSETS = {3: -0.22, 6: 0.0, 9: 0.22}


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
    smi_tier_multipliers=None,
    residence_cf: float = 0.0,
) -> RecidivismModel:
    calibration = get_global_calibration_params()
    calibrated_smi = float( calibration["Supervision_Monitoring_Intensity"])
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
    model.calibrated_global_smi = calibrated_smi
    if not np.isclose(calibrated_smi, 1.10):
        print(
            f"WARNING: calibrated global SMI is {calibrated_smi:.4f}, "
            "not the expected 1.10."
        )

    # Compatibility with model versions that do not expose these in __init__.
    model.employment_cf = float(employment_cf)
    model.program_cf = float(program_cf)
    # Residential-stability counterfactual. Positive values reduce the
    # probability of an unstable residential transition; negative values increase it.
    model.residence_cf = float(residence_cf)

    # Risk-targeted supervision counterfactual.
    # Baseline defaults to 1.0 for every tier, leaving calibrated SMI unchanged.
    model.smi_tier_multipliers = (
        dict(smi_tier_multipliers)
        if smi_tier_multipliers is not None
        else {
            "Low": 1.0,
            "LowModerate": 1.0,
            "Moderate": 1.0,
            "High": 1.0,
        }
    )

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
    residence_changes = np.array([
        float(getattr(a, "Residence_Changes", np.nan)) for a in eligible
    ], dtype=float)
    residence_changes_study = np.array([
        float(getattr(a, "residence_changes_during_study", np.nan)) for a in eligible
    ], dtype=float)
    residence_move_opportunities = np.array([
        float(getattr(a, "residence_move_opportunities", np.nan)) for a in eligible
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
    row["mean_residence_changes"] = (
        float(np.nanmean(residence_changes)) if np.isfinite(residence_changes).any() else np.nan
    )
    row["mean_residence_changes_during_study"] = (
        float(np.nanmean(residence_changes_study)) if np.isfinite(residence_changes_study).any() else np.nan
    )
    row["mean_residence_move_opportunities"] = (
        float(np.nanmean(residence_move_opportunities)) if np.isfinite(residence_move_opportunities).any() else np.nan
    )

    return row



# ================================================================
# SUPERVISION COUNTERFACTUAL HELPERS
# ================================================================

def extract_tier_counts(model) -> Dict[str, int]:
    eligible = [
        a for a in model.schedule.agents
        if getattr(a, "study_eligible_agent", False)
    ]
    counts = {"Low": 0, "LowModerate": 0, "Moderate": 0, "High": 0}
    for a in eligible:
        tier = a.get_pcra_tier()
        if tier in counts:
            counts[tier] += 1
    return counts


def pooled_tier_shares(baseline_rows: List[Dict]) -> Dict[str, float]:
    counts = {"Low": 0, "LowModerate": 0, "Moderate": 0, "High": 0}
    for row in baseline_rows:
        for tier in counts:
            counts[tier] += int(row.get(f"tier_count_{tier}", 0))

    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("Could not derive PCRA tier shares from baseline simulations.")

    return {tier: counts[tier] / total for tier in counts}


def extract_supervision_results(
    model, run_id, seed_id, seed, scenario_name, multipliers
) -> Dict:
    eligible = [
        a for a in model.schedule.agents
        if getattr(a, "study_eligible_agent", False)
    ]
    n = len(eligible)
    if n == 0:
        raise RuntimeError(
            f"No study-eligible agents returned "
            f"(run={run_id}, seed={seed}, scenario={scenario_name})"
        )

    row = {
        "experiment": "supervision",
        "scenario": scenario_name,
        "run_id": int(run_id),
        "seed_id": int(seed_id),
        "seed": int(seed),
        "n_eligible": int(n),
    }

    global_smi = float( getattr(model, "calibrated_global_smi", 1.10))

    row["calibrated_global_smi"] = global_smi

    for tier in ("Low", "LowModerate", "Moderate", "High"):

        multiplier = float(multipliers[tier])

        row[f"smi_multiplier_{tier}"] = multiplier

        row[f"implied_smi_{tier}"] = (
            global_smi * multiplier
        )

    # Rearrest outcomes
    for years in WINDOWS:
        flag = f"rearrest_{years}_yrs"
        row[f"rearrest_{years}yr"] = (
            sum(bool(getattr(a, flag, False)) for a in eligible) / n
        )

    # PCRA tier counts + tier-specific supervision/rearrest outcomes.
    # These diagnostics let us verify that the intervention truly redistributes
    # monitoring burden across risk tiers rather than only changing the aggregate.
    tiers = ("Low", "LowModerate", "Moderate", "High")
    tier_members = {tier: [] for tier in tiers}
    for a in eligible:
        tier = a.get_pcra_tier()
        if tier in tier_members:
            tier_members[tier].append(a)

    for tier in tiers:
        members = tier_members[tier]
        row[f"tier_count_{tier}"] = int(len(members))

        # Tier-specific rearrest rates.
        for years in WINDOWS:
            flag = f"rearrest_{years}_yrs"
            row[f"tier_rearrest_{years}yr_{tier}"] = (
                sum(bool(getattr(a, flag, False)) for a in members) / len(members)
                if members else np.nan
            )

        # Tier-specific realized supervision burden.
        tier_exposure = np.asarray([
            float(getattr(a, "cumulative_supervision_exposure", 0.0))
            for a in members
        ], dtype=float)
        tier_events = np.asarray([
            float(getattr(a, "supervision_exposure_events", 0.0))
            for a in members
        ], dtype=float)
        tier_technical = np.asarray([
            float(getattr(a, "Violations_Technical", 0.0))
            if isinstance(
                getattr(a, "Violations_Technical", 0.0),
                (int, float, bool, np.integer, np.floating)
            )
            else np.nan
            for a in members
        ], dtype=float)

        row[f"tier_supervision_exposure_{tier}"] = (
            float(np.nanmean(tier_exposure))
            if tier_exposure.size and np.isfinite(tier_exposure).any()
            else np.nan
        )
        row[f"tier_supervision_exposure_events_{tier}"] = (
            float(np.nanmean(tier_events))
            if tier_events.size and np.isfinite(tier_events).any()
            else np.nan
        )
        row[f"tier_technical_violations_{tier}"] = (
            float(np.nanmean(tier_technical))
            if tier_technical.size and np.isfinite(tier_technical).any()
            else np.nan
        )

        # Mean effective monitoring intensity per realized monitoring opportunity.
        # This is the ratio of pooled cumulative exposure to pooled exposure events
        # within the tier, avoiding instability from agent-level 0/0 ratios.
        exposure_sum = float(np.nansum(tier_exposure)) if tier_exposure.size else 0.0
        events_sum = float(np.nansum(tier_events)) if tier_events.size else 0.0
        row[f"tier_mean_effective_monitoring_intensity_{tier}"] = (
            exposure_sum / events_sum if events_sum > 0.0 else np.nan
        )

    # Aggregate surveillance burden / technical violations.
    exposure = np.asarray([
        float(getattr(a, "cumulative_supervision_exposure", 0.0))
        for a in eligible
    ], dtype=float)
    events = np.asarray([
        float(getattr(a, "supervision_exposure_events", 0.0))
        for a in eligible
    ], dtype=float)
    technical = np.asarray([
        float(getattr(a, "Violations_Technical", 0.0))
        if isinstance(getattr(a, "Violations_Technical", 0.0),
                      (int, float, bool, np.integer, np.floating))
        else np.nan
        for a in eligible
    ], dtype=float)

    row["mean_supervision_exposure"] = float(np.nanmean(exposure))
    row["mean_supervision_exposure_events"] = float(np.nanmean(events))
    row["mean_technical_violations"] = float(np.nanmean(technical))
    total_events = float(np.nansum(events))
    row["mean_effective_monitoring_intensity"] = (
        float(np.nansum(exposure)) / total_events
        if total_events > 0.0 else np.nan
    )

    # Fairness outcomes: rate gaps in percentage points.
    def subgroup_rate(attr, value, years):
        members = [a for a in eligible if getattr(a, attr, None) == value]
        if not members:
            return np.nan
        flag = f"rearrest_{years}_yrs"
        return sum(bool(getattr(a, flag, False)) for a in members) / len(members)

    for years in WINDOWS:
        black = subgroup_rate("Race", "Black", years)
        white = subgroup_rate("Race", "White", years)
        female = subgroup_rate("Gender", "Female", years)
        male = subgroup_rate("Gender", "Male", years)

        row[f"black_white_gap_{years}yr_pp"] = (
            (black - white) * 100.0
            if np.isfinite(black) and np.isfinite(white) else np.nan
        )
        row[f"female_male_gap_{years}yr_pp"] = (
            (female - male) * 100.0
            if np.isfinite(female) and np.isfinite(male) else np.nan
        )

    return row


def supervision_worker(task: Tuple) -> Dict:
    run_id, seed_id, seed, scenario_name, multipliers = task
    model = build_model(
        seed=seed,
        employment_cf=0.0,
        program_cf=0.0,
        smi_tier_multipliers=multipliers,
    )

    while model.running:
        model.step()

    return extract_supervision_results(
        model, run_id, seed_id, seed, scenario_name, multipliers
    )


def run_supervision_task_batch(tasks, n_workers, description):
    results = []
    max_workers = min(n_workers, len(tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(supervision_worker, task): task for task in tasks}
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
                run_id, seed_id, seed, scenario_name, _ = task
                print(
                    f"\nWARNING: failed supervision task "
                    f"scenario={scenario_name}, run={run_id}, "
                    f"seed_id={seed_id}, seed={seed}: {exc}",
                    flush=True,
                )
    return results


def run_supervision_experiment(n_workers: int, results_dir: Path) -> pd.DataFrame:
    """Run baseline first, derive PCRA tier shares, then run normalized scenarios."""
    seed_grid = matched_seed_grid()
    baseline_mult = SUPERVISION_SCENARIOS["baseline"]

    print("\n" + "=" * 78)
    print("COUNTERFACTUAL EXPERIMENT: RISK-TARGETED SUPERVISION")
    print("=" * 78)
    print(f"Scenarios         : {len(SUPERVISION_SCENARIOS)}")
    print(f"Replications      : {len(seed_grid)} per scenario")
    print("Normalization     : pooled PCRA-tier shares from matched baseline")
    calibration = get_global_calibration_params()
    global_smi = float(calibration["Supervision_Monitoring_Intensity"])

    print(f"Calibrated SMI    : {global_smi:.4f}")
    print(
        "Interpretation    : tier multipliers are applied "
        "relative to calibrated SMI"
    )
    print("=" * 78)

    # 1) Matched calibrated baseline
    baseline_tasks = [
        (run_id, seed_id, seed, "baseline", baseline_mult)
        for run_id, seed_id, seed in seed_grid
    ]
    baseline_rows = run_supervision_task_batch(
        baseline_tasks, n_workers, "  supervision baseline"
    )

    tier_shares = pooled_tier_shares(baseline_rows)
    print("\nBaseline PCRA tier shares:")
    for tier, share in tier_shares.items():
        print(f"  {tier:12s}: {share:.3%}")

    all_rows = list(baseline_rows)

    # 2) Resource-neutral targeted scenarios
    for scenario_name, raw_mult in SUPERVISION_SCENARIOS.items():
        if scenario_name == "baseline":
            continue

        normalized = normalize_smi_multipliers(raw_mult, tier_shares)
        weighted_mean = sum(tier_shares[t] * normalized[t] for t in normalized)

        print(f"\n{scenario_name}:")
        print(
            "  normalized multipliers: "
            + ", ".join(f"{t}={normalized[t]:.3f}" for t in normalized)
        )
        print(f"  weighted mean: {weighted_mean:.6f}")

        tasks = [
            (run_id, seed_id, seed, scenario_name, normalized)
            for run_id, seed_id, seed in seed_grid
        ]
        all_rows.extend(
            run_supervision_task_batch(
                tasks, n_workers, f"  supervision {scenario_name}"
            )
        )

    raw = pd.DataFrame(all_rows).sort_values(
        ["scenario", "run_id", "seed_id"]
    ).reset_index(drop=True)

    raw_path = results_dir / "supervision_raw.csv"
    raw.to_csv(raw_path, index=False)

    tier_share_path = results_dir / "supervision_baseline_tier_shares.json"
    with open(tier_share_path, "w", encoding="utf-8") as f:
        json.dump(tier_shares, f, indent=2)

    print(f"\nRaw supervision results -> {raw_path}")
    print(f"Baseline tier shares     -> {tier_share_path}")
    return raw


def summarize_supervision(raw: pd.DataFrame, results_dir: Path):
    baseline = raw[raw["scenario"] == "baseline"].copy()
    if baseline.empty:
        raise RuntimeError("Supervision baseline is missing.")

    tiers = ("Low", "LowModerate", "Moderate", "High")
    key = ["run_id", "seed_id", "seed"]

    # Every metric listed here is paired to the identical baseline run/seed.
    baseline_metrics = {}

    for years in WINDOWS:
        baseline_metrics[f"rearrest_{years}yr"] = f"baseline_rearrest_{years}yr"
        baseline_metrics[f"black_white_gap_{years}yr_pp"] = (
            f"baseline_black_white_gap_{years}yr_pp"
        )
        baseline_metrics[f"female_male_gap_{years}yr_pp"] = (
            f"baseline_female_male_gap_{years}yr_pp"
        )

    for metric in (
        "mean_supervision_exposure",
        "mean_supervision_exposure_events",
        "mean_effective_monitoring_intensity",
        "mean_technical_violations",
    ):
        baseline_metrics[metric] = f"baseline_{metric}"

    for tier in tiers:
        for years in WINDOWS:
            metric = f"tier_rearrest_{years}yr_{tier}"
            baseline_metrics[metric] = f"baseline_{metric}"

        for metric in (
            f"tier_supervision_exposure_{tier}",
            f"tier_supervision_exposure_events_{tier}",
            f"tier_mean_effective_monitoring_intensity_{tier}",
            f"tier_technical_violations_{tier}",
        ):
            baseline_metrics[metric] = f"baseline_{metric}"

    missing = [c for c in baseline_metrics if c not in raw.columns]
    if missing:
        raise RuntimeError(
            "Supervision raw results are missing tier-specific columns. "
            "Rerun --experiment supervision with the updated script. "
            f"First missing columns: {missing[:8]}"
        )

    baseline_small = baseline[key + list(baseline_metrics)].rename(
        columns=baseline_metrics
    )
    paired = raw.merge(
        baseline_small, on=key, how="left", validate="many_to_one"
    )

    # Aggregate matched-baseline effects.
    for years in WINDOWS:
        paired[f"delta_rearrest_{years}yr_pp"] = (
            paired[f"rearrest_{years}yr"]
            - paired[f"baseline_rearrest_{years}yr"]
        ) * 100.0
        paired[f"delta_black_white_gap_{years}yr_pp"] = (
            paired[f"black_white_gap_{years}yr_pp"]
            - paired[f"baseline_black_white_gap_{years}yr_pp"]
        )
        paired[f"delta_female_male_gap_{years}yr_pp"] = (
            paired[f"female_male_gap_{years}yr_pp"]
            - paired[f"baseline_female_male_gap_{years}yr_pp"]
        )

    for metric in (
        "mean_supervision_exposure",
        "mean_supervision_exposure_events",
        "mean_effective_monitoring_intensity",
        "mean_technical_violations",
    ):
        paired[f"delta_{metric}"] = (
            paired[metric] - paired[f"baseline_{metric}"]
        )

    # Backward-compatible aliases used by the existing console summary/output naming.
    paired["delta_supervision_exposure"] = paired["delta_mean_supervision_exposure"]
    paired["delta_supervision_exposure_events"] = paired["delta_mean_supervision_exposure_events"]
    paired["delta_effective_monitoring_intensity"] = paired["delta_mean_effective_monitoring_intensity"]
    paired["delta_technical_violations"] = paired["delta_mean_technical_violations"]

    # Tier-specific matched-baseline effects.
    for tier in tiers:
        for years in WINDOWS:
            metric = f"tier_rearrest_{years}yr_{tier}"
            paired[f"delta_tier_rearrest_{years}yr_{tier}_pp"] = (
                paired[metric] - paired[f"baseline_{metric}"]
            ) * 100.0

        for metric in (
            f"tier_supervision_exposure_{tier}",
            f"tier_supervision_exposure_events_{tier}",
            f"tier_mean_effective_monitoring_intensity_{tier}",
            f"tier_technical_violations_{tier}",
        ):
            paired[f"delta_{metric}"] = (
                paired[metric] - paired[f"baseline_{metric}"]
            )

    # Build dissertation-ready scenario summary with 95% CIs.
    metrics = [
        "mean_supervision_exposure",
        "mean_supervision_exposure_events",
        "mean_effective_monitoring_intensity",
        "mean_technical_violations",
        "delta_mean_supervision_exposure",
        "delta_mean_supervision_exposure_events",
        "delta_mean_effective_monitoring_intensity",
        "delta_mean_technical_violations",
        # Backward-compatible concise aliases
        "delta_supervision_exposure",
        "delta_supervision_exposure_events",
        "delta_effective_monitoring_intensity",
        "delta_technical_violations",
    ]

    for years in WINDOWS:
        metrics.extend([
            f"rearrest_{years}yr",
            f"delta_rearrest_{years}yr_pp",
            f"black_white_gap_{years}yr_pp",
            f"delta_black_white_gap_{years}yr_pp",
            f"female_male_gap_{years}yr_pp",
            f"delta_female_male_gap_{years}yr_pp",
        ])

    for tier in tiers:
        for years in WINDOWS:
            metrics.extend([
                f"tier_rearrest_{years}yr_{tier}",
                f"delta_tier_rearrest_{years}yr_{tier}_pp",
            ])

        metrics.extend([
            f"tier_supervision_exposure_{tier}",
            f"delta_tier_supervision_exposure_{tier}",
            f"tier_supervision_exposure_events_{tier}",
            f"delta_tier_supervision_exposure_events_{tier}",
            f"tier_mean_effective_monitoring_intensity_{tier}",
            f"delta_tier_mean_effective_monitoring_intensity_{tier}",
            f"tier_technical_violations_{tier}",
            f"delta_tier_technical_violations_{tier}",
        ])

    summary_rows = []
    for scenario, group in paired.groupby("scenario", sort=False):
        row = {
            "experiment": "supervision",
            "scenario": scenario,
            "n_matched_pairs": int(len(group)),
        }

        for metric in metrics:
            mean, sd, lo, hi = mean_ci95(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        for tier in tiers:
            row[f"smi_multiplier_{tier}"] = float(
                group[f"smi_multiplier_{tier}"].iloc[0]
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    paired.to_csv(
        results_dir / "supervision_paired_seed_raw.csv", index=False
    )
    summary.to_csv(
        results_dir / "supervision_summary.csv", index=False
    )

    # Compact tier-only table for dissertation analysis and charting.
    tier_rows = []
    for scenario, group in paired.groupby("scenario", sort=False):
        for tier in tiers:
            tier_row = {
                "experiment": "supervision",
                "scenario": scenario,
                "tier": tier,
                "n_matched_pairs": int(len(group)),
                "smi_multiplier": float(
                    group[f"smi_multiplier_{tier}"].iloc[0]
                ),
            }

            tier_metrics = [
                f"tier_supervision_exposure_{tier}",
                f"delta_tier_supervision_exposure_{tier}",
                f"tier_supervision_exposure_events_{tier}",
                f"delta_tier_supervision_exposure_events_{tier}",
                f"tier_mean_effective_monitoring_intensity_{tier}",
                f"delta_tier_mean_effective_monitoring_intensity_{tier}",
                f"tier_technical_violations_{tier}",
                f"delta_tier_technical_violations_{tier}",
            ]
            for years in WINDOWS:
                tier_metrics.extend([
                    f"tier_rearrest_{years}yr_{tier}",
                    f"delta_tier_rearrest_{years}yr_{tier}_pp",
                ])

            for metric in tier_metrics:
                mean, sd, lo, hi = mean_ci95(group[metric])
                clean = metric.replace(f"_{tier}", "")
                tier_row[f"{clean}_mean"] = mean
                tier_row[f"{clean}_sd"] = sd
                tier_row[f"{clean}_ci95_low"] = lo
                tier_row[f"{clean}_ci95_high"] = hi

            tier_rows.append(tier_row)

    tier_summary = pd.DataFrame(tier_rows)
    tier_summary.to_csv(
        results_dir / "supervision_tier_summary.csv", index=False
    )

    return summary, paired


# ================================================================
# DISSERTATION-READY SUPERVISION FIGURES
# ================================================================

SUPERVISION_SCENARIO_ORDER = [
    "baseline",
    "mild_targeting",
    "moderate_targeting",
    "strong_targeting",
]

SUPERVISION_SCENARIO_LABELS = {
    "baseline": "Baseline",
    "mild_targeting": "Mild",
    "moderate_targeting": "Moderate",
    "strong_targeting": "Strong",
}

TIER_ORDER = ["Low", "LowModerate", "Moderate", "High"]
TIER_LABELS = {
    "Low": "Low",
    "LowModerate": "Low-Moderate",
    "Moderate": "Moderate",
    "High": "High",
}


def _ordered_supervision_summary(summary: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(SUPERVISION_SCENARIO_ORDER)}
    out = summary.copy()
    out["_order"] = out["scenario"].map(order)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _ordered_tier_summary(tier_summary: pd.DataFrame) -> pd.DataFrame:
    scenario_order = {name: i for i, name in enumerate(SUPERVISION_SCENARIO_ORDER)}
    tier_order = {name: i for i, name in enumerate(TIER_ORDER)}
    out = tier_summary.copy()
    out["_scenario_order"] = out["scenario"].map(scenario_order)
    out["_tier_order"] = out["tier"].map(tier_order)
    return out.sort_values(
        ["_scenario_order", "_tier_order"]
    ).drop(columns=["_scenario_order", "_tier_order"]).reset_index(drop=True)





def _add_zero_reference(ax):
    """Neutral zero reference line for matched-baseline effect figures."""
    ax.axhline(
        0.0,
        linewidth=1.15,
        linestyle="--",
        color="0.35",
        alpha=0.9,
        zorder=0,
    )


def _annotate_effects(ax, xs, ys, fmt="{:+.2f}", y_offset=6):
    """Add compact mean-effect labels to point estimates."""
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            va = "bottom" if y >= 0 else "top"
            offset = y_offset if y >= 0 else -y_offset
            ax.annotate(
                fmt.format(y),
                (x, y),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=NOTE_FS,
            )

def plot_supervision_rearrest_rates(summary: pd.DataFrame, charts_dir: Path):
    """Absolute aggregate 3/6/9-year rearrest rates for supervision scenarios."""
    s = _ordered_supervision_summary(summary).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    x = np.arange(len(s), dtype=float)
    for years in WINDOWS:
        y = s[f"rearrest_{years}yr_mean"].to_numpy(dtype=float) * 100.0
        lo = s[f"rearrest_{years}yr_ci95_low"].to_numpy(dtype=float) * 100.0
        hi = s[f"rearrest_{years}yr_ci95_high"].to_numpy(dtype=float) * 100.0
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(
            x + CATEGORICAL_CI_XOFFSETS[years], y, yerr=yerr,
            fmt="o", linestyle="none", markersize=CI_MARKERSIZE,
            markeredgewidth=CI_MARKEREDGEWIDTH, capsize=CI_CAPSIZE,
            capthick=CI_ELINEWIDTH, elinewidth=CI_ELINEWIDTH,
            label=f"{years}-Year", zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SUPERVISION_SCENARIO_LABELS[v] for v in s["scenario"]],
        fontsize=TICK_FS,
    )
    style_axis(
        ax,
        xlabel="Risk-targeted supervision scenario",
        ylabel="Aggregate rearrest rate (%)",
        title="Aggregate Rearrest Rates Under Risk-Targeted Supervision",
    )
    ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=3, loc="best")

    n = int(s["n_matched_pairs"].iloc[0])
    fig.text(
        0.5, 0.015,
        f"Aggregate rates across {n} matched ABM replications per scenario; "
        "error bars are 95% confidence intervals. Horizon-specific points are horizontally offset "
        "within each scenario only to prevent CI overlap. Baseline is the calibrated supervision condition.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    path = charts_dir / "CF_S0_supervision_rearrest_rates.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_supervision_rearrest_change(summary: pd.DataFrame, charts_dir: Path):
    """Primary aggregate outcome figure: point estimates + 95% CIs."""
    s = _ordered_supervision_summary(summary)
    s = s[s["scenario"] != "baseline"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    x = np.arange(len(s), dtype=float)
    for years in WINDOWS:
        y = s[f"delta_rearrest_{years}yr_pp_mean"].to_numpy(dtype=float)
        lo = s[f"delta_rearrest_{years}yr_pp_ci95_low"].to_numpy(dtype=float)
        hi = s[f"delta_rearrest_{years}yr_pp_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        xs = x + CATEGORICAL_CI_XOFFSETS[years]

        ax.errorbar(
            xs, y, yerr=yerr,
            fmt="o", linestyle="none",
            markersize=CI_MARKERSIZE, markeredgewidth=CI_MARKEREDGEWIDTH,
            capsize=CI_CAPSIZE, capthick=CI_ELINEWIDTH,
            elinewidth=CI_ELINEWIDTH, label=f"{years}-Year", zorder=3,
        )
        _annotate_effects(ax, xs, y, fmt="{:+.2f}", y_offset=7)

    _add_zero_reference(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SUPERVISION_SCENARIO_LABELS[v] for v in s["scenario"]],
        fontsize=TICK_FS,
    )
    style_axis(
        ax,
        xlabel="Risk-targeted supervision scenario",
        ylabel="Change in rearrest from baseline (percentage points)",
        title="Aggregate Rearrest Effects of Risk-Targeted Supervision",
    )
    ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=3, loc="lower left")

    n_pairs = int(s["n_matched_pairs"].iloc[0])

    fig.text(
        0.5, 0.015,
        f"Matched-seed mean differences across {n_pairs} replications per scenario; "
        "error bars are 95% confidence intervals. Horizon-specific points are horizontally offset "
        "within each scenario only to prevent CI overlap. "
        "Negative values indicate lower recorded rearrest than the calibrated baseline.",
        ha="center",
        fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    path = charts_dir / "CF_S1_supervision_rearrest_change.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_supervision_realized_burden(summary: pd.DataFrame, charts_dir: Path):
    """Absolute realized cumulative supervision exposure by scenario."""
    s = _ordered_supervision_summary(summary)

    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    labels = [SUPERVISION_SCENARIO_LABELS[v] for v in s["scenario"]]
    y = s["mean_supervision_exposure_mean"].to_numpy(dtype=float)
    lo = s["mean_supervision_exposure_ci95_low"].to_numpy(dtype=float)
    hi = s["mean_supervision_exposure_ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])

    x = np.arange(len(s), dtype=float)
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o", linestyle="none",
        markersize=8, capsize=5, elinewidth=1.8,
    )
    _annotate_effects(ax, x, y, fmt="{:.2f}", y_offset=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    style_axis(
        ax,
        xlabel="Risk-targeted supervision scenario",
        ylabel="Mean cumulative supervision exposure",
        title="Realized Supervision Burden by Targeting Scenario",
    )

    baseline_value = float(
        s.loc[s["scenario"] == "baseline", "mean_supervision_exposure_mean"].iloc[0]
    )
    fig.text(
        0.5, 0.015,
        f"SMI multipliers are normalized to a population-weighted mean of 1.0, but realized exposure remains dynamic. "
        f"Baseline mean exposure = {baseline_value:.2f}; error bars indicate 95% confidence intervals.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    path = charts_dir / "CF_S2_supervision_realized_burden.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_supervision_tier_exposure_change(tier_summary: pd.DataFrame, charts_dir: Path):
    """Mechanism figure: realized exposure redistribution by PCRA tier."""
    ts = _ordered_tier_summary(tier_summary)
    ts = ts[ts["scenario"] != "baseline"].copy()

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    x = np.arange(len(TIER_ORDER), dtype=float)
    offsets = {
        "mild_targeting": -0.18,
        "moderate_targeting": 0.0,
        "strong_targeting": 0.18,
    }

    for scenario in ("mild_targeting", "moderate_targeting", "strong_targeting"):
        g = ts[ts["scenario"] == scenario].set_index("tier").reindex(TIER_ORDER)
        y = g["delta_tier_supervision_exposure_mean"].to_numpy(dtype=float)
        lo = g["delta_tier_supervision_exposure_ci95_low"].to_numpy(dtype=float)
        hi = g["delta_tier_supervision_exposure_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        xs = x + offsets[scenario]

        ax.errorbar(
            xs, y, yerr=yerr,
            fmt="o", linestyle="none",
            markersize=7, capsize=4, elinewidth=1.8,
            label=SUPERVISION_SCENARIO_LABELS[scenario],
        )

    _add_zero_reference(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], fontsize=TICK_FS)
    style_axis(
        ax,
        xlabel="PCRA risk tier",
        ylabel="Change in cumulative supervision exposure from baseline",
        title="Change in Supervision Exposure by PCRA Risk Tier",
    )
    ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=3, loc="upper left")

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.text(
        0.02, (0 - ymin) / span + 0.025,
        "Increased exposure ↑",
        transform=ax.transAxes, fontsize=NOTE_FS, va="bottom"
    )
    ax.text(
        0.02, (0 - ymin) / span - 0.025,
        "Reduced exposure ↓",
        transform=ax.transAxes, fontsize=NOTE_FS, va="top"
    )

    fig.text(
        0.5, 0.015,
        "Matched-seed changes relative to the calibrated baseline within each PCRA tier; "
        "error bars indicate 95% confidence intervals.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    path = charts_dir / "CF_S3_supervision_tier_exposure_change.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_supervision_tier_rearrest_change(
    tier_summary: pd.DataFrame,
    charts_dir: Path,
    years: int = 3,
):
    """Tier-specific rearrest response: point estimates + 95% CIs."""
    ts = _ordered_tier_summary(tier_summary)
    ts = ts[ts["scenario"] != "baseline"].copy()

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    x = np.arange(len(TIER_ORDER), dtype=float)
    offsets = {
        "mild_targeting": -0.18,
        "moderate_targeting": 0.0,
        "strong_targeting": 0.18,
    }

    for scenario in ("mild_targeting", "moderate_targeting", "strong_targeting"):
        g = ts[ts["scenario"] == scenario].set_index("tier").reindex(TIER_ORDER)
        metric = f"delta_tier_rearrest_{years}yr_pp"
        y = g[f"{metric}_mean"].to_numpy(dtype=float)
        lo = g[f"{metric}_ci95_low"].to_numpy(dtype=float)
        hi = g[f"{metric}_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        xs = x + offsets[scenario]

        ax.errorbar(
            xs, y, yerr=yerr,
            fmt="o", linestyle="none",
            markersize=7, capsize=4, elinewidth=1.8,
            label=SUPERVISION_SCENARIO_LABELS[scenario],
        )

        # Label the Strong scenario to keep the figure readable while surfacing
        # the most policy-relevant effect magnitudes.
        if scenario == "strong_targeting":
            _annotate_effects(ax, xs, y, fmt="{:+.1f}", y_offset=8)

    _add_zero_reference(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], fontsize=TICK_FS)

    style_axis(
        ax,
        xlabel="PCRA risk tier",
        ylabel=f"Change in {years}-year rearrest from baseline (percentage points)",
        title=f"Change in {years}-Year Rearrest by PCRA Risk Tier",
    )
    ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=3, loc="upper left")

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.text(
        0.02, (0 - ymin) / span + 0.025,
        "Higher recorded rearrest ↑",
        transform=ax.transAxes, fontsize=NOTE_FS, va="bottom"
    )
    ax.text(
        0.02, (0 - ymin) / span - 0.025,
        "Lower recorded rearrest ↓",
        transform=ax.transAxes, fontsize=NOTE_FS, va="top"
    )

    fig.text(
        0.5, 0.015,
        "Matched-seed changes relative to each tier's calibrated baseline; error bars indicate 95% confidence intervals. "
        "Strong-scenario means are labeled for readability.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    path = charts_dir / f"CF_S4_supervision_tier_rearrest_{years}yr.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_supervision_fairness_change(summary: pd.DataFrame, charts_dir: Path):
    """Demographic disparity figure with neutral interpretation."""
    s = _ordered_supervision_summary(summary)
    s = s[s["scenario"] != "baseline"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    x = np.arange(len(s), dtype=float)

    metrics = [
        ("delta_black_white_gap_3yr_pp", "Black–White gap"),
        ("delta_female_male_gap_3yr_pp", "Female–Male gap"),
    ]
    offsets = [-0.10, 0.10]

    for (metric, label), offset in zip(metrics, offsets):
        y = s[f"{metric}_mean"].to_numpy(dtype=float)
        lo = s[f"{metric}_ci95_low"].to_numpy(dtype=float)
        hi = s[f"{metric}_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])

        ax.errorbar(
            x + offset, y, yerr=yerr,
            fmt="o", linestyle="none",
            markersize=7, capsize=4, elinewidth=1.8,
            label=label,
        )

    _add_zero_reference(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SUPERVISION_SCENARIO_LABELS[v] for v in s["scenario"]],
        fontsize=TICK_FS,
    )

    style_axis(
        ax,
        xlabel="Risk-targeted supervision scenario",
        ylabel="Change in 3-year demographic gap (percentage points)",
        title="Demographic Disparity Effects of Risk-Targeted Supervision",
    )
    ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=2, loc="upper right")

    ax.text(
        0.02, 0.96,
        "All 95% CIs include zero",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=NOTE_FS,
        fontweight="bold",
    )

    fig.text(
        0.5, 0.015,
        "Points are matched-seed mean changes in 3-year Black–White and Female–Male rearrest gaps relative to baseline; "
        "error bars indicate 95% confidence intervals.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    path = charts_dir / "CF_S5_supervision_demographic_disparity_change.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_supervision_intensity_event_decomposition(
    tier_summary: pd.DataFrame,
    charts_dir: Path,
):
    """Decompose cumulative burden into intensity per event and monitoring opportunities."""
    ts = _ordered_tier_summary(tier_summary)
    ts = ts[ts["scenario"] != "baseline"].copy()

    # Two separate figures are used because intensity and event counts have
    # different units/scales. This avoids dual-axis interpretation problems.
    paths = []
    x = np.arange(len(TIER_ORDER), dtype=float)
    offsets = {
        "mild_targeting": -0.18,
        "moderate_targeting": 0.0,
        "strong_targeting": 0.18,
    }

    specs = [
        (
            "delta_tier_mean_effective_monitoring_intensity",
            "Change in effective monitoring intensity per event",
            "Change in Monitoring Intensity per Supervision Event",
            "CF_S6a_supervision_intensity_per_event.png",
        ),
        (
            "delta_tier_supervision_exposure_events",
            "Change in mean supervision exposure events",
            "Change in Supervision Monitoring Opportunities",
            "CF_S6b_supervision_exposure_events.png",
        ),
    ]

    for metric, ylabel, title, filename in specs:
        fig, ax = plt.subplots(figsize=(11.2, 7.0))

        for scenario in ("mild_targeting", "moderate_targeting", "strong_targeting"):
            g = ts[ts["scenario"] == scenario].set_index("tier").reindex(TIER_ORDER)
            y = g[f"{metric}_mean"].to_numpy(dtype=float)
            lo = g[f"{metric}_ci95_low"].to_numpy(dtype=float)
            hi = g[f"{metric}_ci95_high"].to_numpy(dtype=float)
            yerr = np.vstack([y - lo, hi - y])

            ax.errorbar(
                x + offsets[scenario],
                y,
                yerr=yerr,
                fmt="o", linestyle="none",
                markersize=7, capsize=4, elinewidth=1.8,
                label=SUPERVISION_SCENARIO_LABELS[scenario],
            )

        _add_zero_reference(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], fontsize=TICK_FS)

        style_axis(
            ax,
            xlabel="PCRA risk tier",
            ylabel=ylabel,
            title=title,
        )
        ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=3, loc="upper left")

        fig.text(
            0.5, 0.015,
            "Matched-seed changes relative to the calibrated baseline within each PCRA tier; "
            "error bars indicate 95% confidence intervals.",
            ha="center", fontsize=NOTE_FS,
        )
        fig.tight_layout(rect=[0, 0.055, 1, 1])

        path = charts_dir / filename
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def generate_supervision_charts(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    results_dir: Path,
    charts_dir: Path,
):
    """Generate all dissertation-ready risk-targeted supervision figures."""
    tier_path = results_dir / "supervision_tier_summary.csv"
    if not tier_path.exists():
        raise FileNotFoundError(
            f"Missing {tier_path}. Rerun the supervision experiment with tier outputs enabled."
        )
    tier_summary = pd.read_csv(tier_path)

    print("\nGenerating dissertation-ready supervision figures...")
    paths = [
        # Outcome view 1: absolute aggregate rates (substantive magnitude)
        plot_supervision_rearrest_rates(summary, charts_dir),
        # Outcome view 2: matched change from calibrated baseline (effect + uncertainty)
        plot_supervision_rearrest_change(summary, charts_dir),
        plot_supervision_realized_burden(summary, charts_dir),
        plot_supervision_tier_exposure_change(tier_summary, charts_dir),
        plot_supervision_tier_rearrest_change(tier_summary, charts_dir, years=3),
        plot_supervision_fairness_change(summary, charts_dir),
    ]
    paths.extend(
        plot_supervision_intensity_event_decomposition(
            tier_summary, charts_dir
        )
    )
    for path in paths:
        print(f"  Chart -> {path}")
    return paths


# ================================================================
# WORKER / TASKS
# ================================================================

def worker(task: Tuple) -> Dict:
    run_id, seed_id, seed, experiment, cf_level = task

    if experiment == "program":
        employment_cf, program_cf, residence_cf = 0.0, cf_level, 0.0
    elif experiment == "employment":
        employment_cf, program_cf, residence_cf = cf_level, 0.0, 0.0
    elif experiment == "residence":
        employment_cf, program_cf, residence_cf = 0.0, 0.0, cf_level
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    model = build_model(seed, employment_cf, program_cf, residence_cf=residence_cf)

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

        # Older raw CSVs may not contain mechanism columns added in later
        # versions of the counterfactual script. Summarize only metrics that
        # are actually present so --replot remains backward compatible.
        for metric in (
            "mean_percent_days_employed",
            "mean_program_attendances",
            "mean_program_unexcused_absences",
            "mean_technical_violations",
            "mean_residence_changes",
            "mean_residence_changes_during_study",
            "mean_residence_move_opportunities",
        ):
            if metric not in group.columns:
                continue
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
    # For the housing-stability experiment, pair the realized mechanism to the
    # identical baseline seed as well as the rearrest outcomes.
    if experiment == "residence":
        for metric in (
            "mean_residence_changes",
            "mean_residence_changes_during_study",
            "mean_residence_move_opportunities",
        ):
            if metric in raw.columns:
                baseline_cols[metric] = f"baseline_{metric}"

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

    if experiment == "residence":
        for metric in (
            "mean_residence_changes",
            "mean_residence_changes_during_study",
            "mean_residence_move_opportunities",
        ):
            baseline_metric = f"baseline_{metric}"
            if metric in paired.columns and baseline_metric in paired.columns:
                paired[f"delta_{metric}"] = paired[metric] - paired[baseline_metric]

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

        if experiment == "residence":
            for metric in (
                "delta_mean_residence_changes",
                "delta_mean_residence_changes_during_study",
                "delta_mean_residence_move_opportunities",
            ):
                if metric not in group.columns:
                    continue
                mean, sd, lo, hi = mean_ci95(group[metric])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi

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
        "mean_residence_changes_mean",
        "mean_residence_changes_sd",
        "mean_residence_changes_ci95_low",
        "mean_residence_changes_ci95_high",
        "mean_residence_changes_during_study_mean",
        "mean_residence_changes_during_study_sd",
        "mean_residence_changes_during_study_ci95_low",
        "mean_residence_changes_during_study_ci95_high",
        "mean_residence_move_opportunities_mean",
    ]
    mechanism = summary[[c for c in mechanism_cols if c in summary.columns]].copy()

    summary.to_csv(results_dir / f"{experiment}_summary.csv", index=False)
    paired.to_csv(results_dir / f"{experiment}_paired_seed_raw.csv", index=False)
    paired_summary.to_csv(results_dir / f"{experiment}_paired_effects.csv", index=False)
    mechanism.to_csv(results_dir / f"{experiment}_mechanism_summary.csv", index=False)

    if experiment == "residence":
        # Compact dissertation table: effect estimate [95% CI] plus mechanism response.
        table = paired_summary[["cf_percent", "n_matched_pairs"]].copy()
        table.rename(columns={"cf_percent": "housing_stability_change_percent"}, inplace=True)
        for years in WINDOWS:
            m = paired_summary[f"delta_{years}yr_pp_mean"]
            lo = paired_summary[f"delta_{years}yr_pp_ci95_low"]
            hi = paired_summary[f"delta_{years}yr_pp_ci95_high"]
            table[f"delta_rearrest_{years}yr_pp_95ci"] = [
                f"{a:+.2f} [{b:+.2f}, {c:+.2f}]" for a, b, c in zip(m, lo, hi)
            ]
        m = paired_summary["delta_mean_residence_changes_during_study_mean"]
        lo = paired_summary["delta_mean_residence_changes_during_study_ci95_low"]
        hi = paired_summary["delta_mean_residence_changes_during_study_ci95_high"]
        table["delta_unstable_transitions_95ci"] = [
            f"{a:+.3f} [{b:+.3f}, {c:+.3f}]" for a, b, c in zip(m, lo, hi)
        ]
        table.to_csv(results_dir / "residence_dissertation_table.csv", index=False)

    return summary, paired_summary, mechanism


# ================================================================
# DISSERTATION FIGURES
# ================================================================



def plot_rearrest_rates(summary, experiment, charts_dir):
    """Aggregate 3/6/9-year rearrest rates with clearly separated 95% CIs."""
    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    x = summary["cf_percent"].to_numpy(dtype=float)

    for years in WINDOWS:
        y = summary[f"rearrest_{years}yr_mean"].to_numpy(dtype=float)
        lo = summary[f"rearrest_{years}yr_ci95_low"].to_numpy(dtype=float)
        hi = summary[f"rearrest_{years}yr_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        xs = x + CONTINUOUS_CI_XOFFSETS[years]

        # Horizontal offsets separate 3/6/9-year CIs at the same nominal policy level.
        ax.errorbar(
            xs, y, yerr=yerr,
            fmt="o-", markersize=CI_MARKERSIZE, linewidth=1.6,
            markeredgewidth=CI_MARKEREDGEWIDTH, capsize=CI_CAPSIZE,
            capthick=CI_ELINEWIDTH, elinewidth=CI_ELINEWIDTH,
            label=f"{years}-Year Rearrest", zorder=3,
        )

    ax.axvline(0, linewidth=1.2, linestyle=":", color="0.45", alpha=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(x)
    ax.set_xlim(float(np.min(x)) - 6, float(np.max(x)) + 6)

    if experiment == "employment":
        xlabel = "Employment intervention (%)"
        title = "Aggregate Rearrest Rates Under Employment Counterfactuals"
    elif experiment == "program":
        xlabel = "Program-participation intervention (%)"
        title = "Aggregate Rearrest Rates Under Program Counterfactuals"
    elif experiment == "residence":
        xlabel = "Housing-stability intervention (%)"
        title = "Aggregate Rearrest Rates Under Housing-Stability Counterfactuals"
    else:
        xlabel = f"{experiment.capitalize()} intervention (%)"
        title = f"Aggregate Rearrest Rates Under {experiment.capitalize()} Counterfactuals"

    style_axis(
        ax, xlabel=xlabel, ylabel="Rearrest rate", title=title,
    )
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="best", ncol=1)

    n_rep = int(summary["n_replications"].iloc[0]) if "n_replications" in summary.columns else (
        CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"]
    )
    fig.text(
        0.5, 0.015,
        f"Means across {n_rep} matched-seed ABM replications per intervention level; "
        "error bars are 95% confidence intervals. Points are slightly offset horizontally "
        "within each intervention level only to prevent CI overlap.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    path = charts_dir / f"CF1_{experiment}_rearrest_rates.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_change_from_baseline(paired, experiment, charts_dir):
    """
    Dissertation/defense-ready matched-baseline rearrest figure.

    Employment and program use the SAME y-axis limits so their effects can be
    compared directly when the two figures are placed side-by-side.
    """
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    x = paired["cf_percent"].to_numpy(dtype=float)

    for years in WINDOWS:
        y = paired[f"delta_{years}yr_pp_mean"].to_numpy(dtype=float)
        lo = paired[f"delta_{years}yr_pp_ci95_low"].to_numpy(dtype=float)
        hi = paired[f"delta_{years}yr_pp_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])

        xs = x + CONTINUOUS_CI_XOFFSETS[years]
        ax.errorbar(
            xs, y, yerr=yerr,
            fmt="o-", markersize=CI_MARKERSIZE, linewidth=1.6,
            markeredgewidth=CI_MARKEREDGEWIDTH, capsize=CI_CAPSIZE,
            capthick=CI_ELINEWIDTH, elinewidth=CI_ELINEWIDTH,
            label=f"{years}-Year", zorder=3,
        )

    # Neutral reference lines so they do not visually compete with data series.
    ax.axhline(
        0.0,
        linewidth=1.15,
        linestyle="--",
        color="0.35",
        alpha=0.9,
        zorder=0,
    )
    ax.axvline(
        0.0,
        linewidth=1.0,
        linestyle=":",
        color="0.45",
        alpha=0.75,
        zorder=0,
    )

    # IMPORTANT: identical scale for Employment and Program figures.
    # This makes side-by-side visual comparison valid on the dissertation slide.
    ax.set_ylim(-1.0, 0.8)
    ax.set_yticks(np.arange(-1.0, 0.81, 0.2))
    ax.set_xlim(float(np.min(x)) - 8, float(np.max(x)) + 8)

    if experiment == "employment":
        title = "Employment Opportunity: Change in Rearrest"
        xlabel = "Employment counterfactual change (%)"
    elif experiment == "program":
        title = "Program Opportunity: Change in Rearrest"
        xlabel = "Program counterfactual change (%)"
    elif experiment == "residence":
        title = "Housing Stability: Change in Rearrest"
        xlabel = "Housing-stability intervention (%)"
    else:
        title = f"{experiment.capitalize()}: Change in Rearrest"
        xlabel = f"{experiment.capitalize()} counterfactual change (%)"

    style_axis(
        ax,
        xlabel=xlabel,
        ylabel="Change in rearrest from baseline (percentage points)",
        title=title,
    )

    # Keep legend placement identical for the two slide figures.
    ax.legend(
        fontsize=LEGEND_FS,
        frameon=False,
        loc="upper right",
        ncol=1,
    )

    n_pairs = int(paired["n_matched_pairs"].iloc[0]) if "n_matched_pairs" in paired.columns else (
        CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"]
    )
    fig.text(
        0.5,
        0.015,
        f"Matched-seed mean differences across {n_pairs} replications per intervention level; "
        "error bars are 95% confidence intervals. Points are slightly offset horizontally "
        "within each intervention level only to prevent CI overlap.",
        ha="center",
        fontsize=NOTE_FS,
    )

    fig.tight_layout(rect=[0, 0.055, 1, 1])

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
    elif experiment == "program":
        y = summary["mean_program_attendances_mean"].to_numpy()
        ylabel = "Mean program attendances"
        title = "Mechanism Check: Program Counterfactual Changes Program Participation"
    else:
        y = summary["mean_residence_changes_during_study_mean"].to_numpy()
        ylabel = "Mean unstable residential transitions during study"
        title = "Mechanism Check: Housing-Stability Intervention Changes Residential Instability"

    ax.plot(x, y, marker="o", markersize=6, linewidth=2.2)
    ax.axvline(0, linewidth=1.4, linestyle="--", alpha=0.7)

    style_axis(
        ax,
        xlabel=f"{experiment.capitalize()} intervention (%)",
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



# ================================================================
# ADDITIONAL RESIDENCE-SPECIFIC DISSERTATION FIGURES
# ================================================================

def plot_residence_realized_instability_change(paired: pd.DataFrame, charts_dir: Path):
    """
    Residence Figure R1: realized mechanism response relative to matched baseline.

    Uses residence_paired_effects.csv-compatible columns produced by
    summarize_experiment(). Negative values mean fewer unstable residential
    transitions during the study period.
    """
    required = [
        "cf_percent", "delta_mean_residence_changes_during_study_mean",
        "delta_mean_residence_changes_during_study_ci95_low",
        "delta_mean_residence_changes_during_study_ci95_high",
    ]
    missing = [c for c in required if c not in paired.columns]
    if missing:
        raise RuntimeError(f"Residence paired-effects CSV is missing columns: {missing}")

    s = paired.sort_values("cf_percent").copy()
    x = s["cf_percent"].to_numpy(dtype=float)
    y = s["delta_mean_residence_changes_during_study_mean"].to_numpy(dtype=float)
    lo = s["delta_mean_residence_changes_during_study_ci95_low"].to_numpy(dtype=float)
    hi = s["delta_mean_residence_changes_during_study_ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    ax.errorbar(
        x, y, yerr=yerr, marker="o", markersize=CI_MARKERSIZE,
        linewidth=1.8, markeredgewidth=CI_MARKEREDGEWIDTH,
        capsize=CI_CAPSIZE, capthick=CI_ELINEWIDTH,
        elinewidth=CI_ELINEWIDTH, zorder=3
    )
    _add_zero_reference(ax)
    ax.axvline(0.0, linewidth=1.0, linestyle=":", color="0.45", alpha=0.75)
    style_axis(
        ax,
        xlabel="Housing-stability counterfactual change (%)",
        ylabel="Change in unstable residential transitions per agent",
        title="Realized Change in Residential Instability",
    )
    n_pairs = int(s["n_matched_pairs"].iloc[0])
    fig.text(
        0.5, 0.015,
        f"Matched-seed mean differences across {n_pairs} replications per intervention level; "
        "error bars indicate 95% confidence intervals. Positive policy values represent stronger housing stability; "
        "negative outcome values therefore indicate fewer unstable residential transitions.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    path = charts_dir / "CF_R1_residence_realized_instability_change.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residence_mechanism_outcome_link(paired: pd.DataFrame, charts_dir: Path):
    """
    Residence Figure R2: mechanism-to-outcome response.

    X is the matched-seed mean change in realized unstable transitions; Y is
    matched-seed rearrest change. This separates whether the intervention
    successfully changes housing instability from whether rearrest responds.
    """
    s = paired.sort_values("cf_percent").copy()
    x = s["delta_mean_residence_changes_during_study_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    for years in WINDOWS:
        y = s[f"delta_{years}yr_pp_mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", markersize=5.5, linewidth=1.8, label=f"{years}-Year")

    _add_zero_reference(ax)
    ax.axvline(0.0, linewidth=1.0, linestyle=":", color="0.45", alpha=0.75)
    style_axis(
        ax,
        xlabel="Realized change in unstable residential transitions per agent",
        ylabel="Change in rearrest from baseline (percentage points)",
        title="Rearrest Response to Realized Residential Instability",
    )
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="best")

    # Label policy endpoints and baseline without cluttering every point.
    for idx in (0, len(s) // 2, len(s) - 1):
        ax.annotate(
            f"{s.iloc[idx]['cf_percent']:+.0f}%",
            (x[idx], s.iloc[idx]["delta_3yr_pp_mean"]),
            xytext=(5, 5), textcoords="offset points", fontsize=NOTE_FS,
        )

    fig.text(
        0.5, 0.015,
        "Each point is an intervention level summarized across matched seeds. This figure is descriptive: "
        "it evaluates whether rearrest changes systematically as the residential-instability mechanism changes and "
        "should not be interpreted as a causal regression.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    path = charts_dir / "CF_R2_residence_mechanism_outcome_link.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residence_selected_policy_effects(paired: pd.DataFrame, charts_dir: Path):
    """
    Residence Figure R3: compact policy-relevant forest plot with 95% CIs.

    Shows representative instability-increase and stability-improvement levels
    across all three follow-up horizons.
    """
    selected_levels = (-0.80, -0.40, 0.40, 0.80)
    s = paired[paired["cf_level"].apply(lambda v: any(np.isclose(v, z) for z in selected_levels))].copy()
    if s.empty:
        raise RuntimeError("No selected residence policy levels found for CF_R3.")
    s["_order"] = s["cf_level"].apply(lambda v: min(range(len(selected_levels)), key=lambda i: abs(v-selected_levels[i])))
    s = s.sort_values("_order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    x = np.arange(len(s), dtype=float)
    for years in WINDOWS:
        y = s[f"delta_{years}yr_pp_mean"].to_numpy(dtype=float)
        lo = s[f"delta_{years}yr_pp_ci95_low"].to_numpy(dtype=float)
        hi = s[f"delta_{years}yr_pp_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(
            x + CATEGORICAL_CI_XOFFSETS[years], y, yerr=yerr,
            fmt="o", linestyle="none", markersize=CI_MARKERSIZE,
            markeredgewidth=CI_MARKEREDGEWIDTH, capsize=CI_CAPSIZE,
            capthick=CI_ELINEWIDTH, elinewidth=CI_ELINEWIDTH,
            label=f"{years}-Year", zorder=3
        )

    _add_zero_reference(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:+.0%}" for v in s["cf_level"]], fontsize=TICK_FS)
    style_axis(
        ax,
        xlabel="Housing-stability counterfactual change",
        ylabel="Change in rearrest from baseline (percentage points)",
        title="Rearrest Effects at Selected Housing-Stability Scenarios",
    )
    ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=3, loc="best")
    n_pairs = int(s["n_matched_pairs"].iloc[0])
    fig.text(
        0.5, 0.015,
        f"Representative policy levels from the full −80% to +80% sweep; {n_pairs} matched replications per level. "
        "Points are mean changes from the calibrated 0% condition; error bars indicate 95% confidence intervals.",
        ha="center", fontsize=NOTE_FS,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = charts_dir / "CF_R3_residence_selected_policy_effects.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def generate_charts(summary, paired, experiment, charts_dir):
    print(f"\nGenerating dissertation-ready {experiment} figures...")

    # These two outcome charts are available for every experiment and are
    # intentionally always generated.
    paths = [
        plot_rearrest_rates(summary, experiment, charts_dir),
        plot_change_from_baseline(paired, experiment, charts_dir),
        plot_effect_by_horizon(paired, experiment, charts_dir),
    ]

    # Mechanism charts require mechanism columns. Older *_raw.csv files may
    # predate those fields, so skip them rather than failing the entire replot.
    mechanism_col = {
        "employment": "mean_percent_days_employed_mean",
        "program": "mean_program_attendances_mean",
        "residence": "mean_residence_changes_during_study_mean",
    }.get(experiment)
    if mechanism_col and mechanism_col in summary.columns:
        paths.append(plot_mechanism_check(summary, experiment, charts_dir))
    elif mechanism_col:
        print(f"  NOTE: Skipping mechanism chart; existing raw CSV does not contain {mechanism_col.replace('_mean','')}.")

    if experiment == "residence":
        residence_required = "delta_mean_residence_changes_during_study_mean"
        if residence_required in paired.columns:
            paths.extend([
                plot_residence_realized_instability_change(paired, charts_dir),
                plot_residence_mechanism_outcome_link(paired, charts_dir),
                plot_residence_selected_policy_effects(paired, charts_dir),
            ])
        else:
            print(
                "  NOTE: Skipping residence mechanism-specific figures because the "
                "existing residence_raw.csv predates the residence mechanism fields. "
                "The aggregate rearrest-rate and matched-change charts were still generated. "
                "Rerun the residence experiment once to populate those mechanism fields."
            )

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
            "supervision": CONFIG["N_RUNS"] * CONFIG["SEEDS_PER_RUN"] * len(SUPERVISION_SCENARIOS),
        },
        "supervision_scenarios": SUPERVISION_SCENARIOS,
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
        choices=("employment", "program", "residence", "supervision", "both", "all"),
        default="both",
        help="Run employment, program, residence, supervision, both (employment+program), or all.",
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

    if args.experiment == "both":
        experiments = ["employment", "program"]
    elif args.experiment == "all":
        experiments = ["employment", "program", "residence", "supervision"]
    else:
        experiments = [args.experiment]

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
            if experiment == "supervision":
                raw_path = results_dir / "supervision_raw.csv"
                if not raw_path.exists():
                    raise FileNotFoundError(f"Cannot replot: {raw_path} does not exist.")
                raw = pd.read_csv(raw_path)
                summary, paired = summarize_supervision(raw, results_dir)
                generate_supervision_charts(
                    summary, paired, results_dir, charts_dir
                )
                print("Supervision summaries and charts regenerated from existing raw CSV.")
            else:
                load_existing_and_replot(
                    experiment,
                    results_dir,
                    charts_dir,
                )
        save_metadata(results_dir, experiments, workers, elapsed_seconds=0.0)
        print("\nReplot complete.")
        return

    for experiment in experiments:
        if experiment == "supervision":
            raw = run_supervision_experiment(workers, results_dir)
            summary, paired = summarize_supervision(raw, results_dir)
            generate_supervision_charts(
                summary, paired, results_dir, charts_dir
            )
            print("\nSUPERVISION COUNTERFACTUAL SUMMARY")
            print(summary[
                [
                    "scenario",
                    "delta_rearrest_3yr_pp_mean",
                    "delta_rearrest_6yr_pp_mean",
                    "delta_rearrest_9yr_pp_mean",
                    "delta_supervision_exposure_mean",
                    "delta_black_white_gap_3yr_pp_mean",
                    "delta_female_male_gap_3yr_pp_mean",
                ]
            ].to_string(index=False))
        else:
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
    print("Primary employment/program figure: CF2_*_change_from_baseline.png")
    print("Primary residence figure         : CF2_residence_change_from_baseline.png")
    print("Residence mechanism figure       : CF3_residence_mechanism_check.png")
    print("Residence realized mechanism      : CF_R1_residence_realized_instability_change.png")
    print("Residence mechanism→outcome       : CF_R2_residence_mechanism_outcome_link.png")
    print("Residence selected policy effects : CF_R3_residence_selected_policy_effects.png")
    print("Primary supervision figure       : CF_S1_supervision_rearrest_change.png")
    print("Primary supervision mechanism    : CF_S3_supervision_tier_exposure_change.png")
    print("Primary supervision table        : supervision_tier_summary.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
