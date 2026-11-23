#!/usr/bin/env python3
"""
Generate a dissertation-ready ~50,000-agent BIASED ML synthetic cohort from multiple
independent realizations of the calibrated Recidivism ABM.


Design
* Keeps the calibrated ABM configuration fixed across independent simulation seeds.
* Runs each simulation in a separate output folder.
* Captures agent attributes at two points:
    - Start of Study: Agent characteristics used as predictors.
    - End of Study: Rearrest outcomes observed during the 9-year follow-up.
* Joins the start-of-study and end-of-study datasets using the agent ID
  within each simulation.
* Uses only start-of-study attributes as predictors and end-of-study
  rearrest outcomes as target labels to prevent data leakage.
* Pools study-eligible agent records across independent simulations
  to generate the target synthetic cohort of approximately 50,000 agents.
* Preserves simulation_id and simulation_seed for independent
  simulation-based training, testing, and cross-validation.
* Saves individual simulation outputs, combined datasets, summaries,
  and charts in separate folders.


IMPORTANT
---------
This script is for ML synthetic-data generation, not the 10 x 20 ABM validation
experiment. Keep the existing validation script/results separate.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Project-root imports: preserve the convention used in the existing experiment scripts.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_global_calibration_params


# -----------------------------------------------------------------------------
# Dissertation experiment configuration
# -----------------------------------------------------------------------------
TARGET_AGENTS = 50_000
N_SEEDS = 6                 # Run at least five independent realizations.
BASE_SEED = 21_000
SEED_STEP = 1_000

BASE_PARAMS = {
    "initial_agents": 3000,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 1,
    "enable_peer_influence": True,
}

BJS_TARGETS = {
    "rearrest_3_yrs": 0.684,
    "rearrest_6_yrs": 0.794,
    "rearrest_9_yrs": 0.834,
}

OUTCOME_COLS = list(BJS_TARGETS.keys())

# Separate artifact folders by purpose.
ROOT_OUTPUT_DIR = Path("ml_synthetic_50k_multiseed_bias_DLF")
RUNS_DIR = ROOT_OUTPUT_DIR / "01_seed_model_runs"
DATA_DIR = ROOT_OUTPUT_DIR / "02_combined_dataset"
SUMMARY_DIR = ROOT_OUTPUT_DIR / "03_summaries"
CHART_DIR = ROOT_OUTPUT_DIR / "04_charts"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in [ROOT_OUTPUT_DIR, RUNS_DIR, DATA_DIR, SUMMARY_DIR, CHART_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_seed_list(n_seeds: int, base_seed: int, seed_step: int) -> List[int]:
    return [base_seed + i * seed_step for i in range(n_seeds)]


def normalize_binary(series: pd.Series) -> pd.Series:
    """Convert common bool/0-1/string outcome representations to numeric 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    mapping = {
        "true": 1, "false": 0, "yes": 1, "no": 0,
        "rearrested": 1, "not rearrested": 0, "1": 1, "0": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


# Exclude outcomes and any attributes explicitly describing post-entry events.
# The downstream ML training script must additionally use a reviewed feature allowlist.
EXCLUDE_ENTRY_COLUMNS = {
    "recidivated_agent", "recidivated", "rearrest_month", "rearrest_year",
    "rearrestmonth", "rearrest_within_3yr", "reincarceration_within_3yr",
    "rearrest_quarterly", "exited_due_to_recidivism",
    "committed_new_offense_this_month", "reset_done",
}


def find_study_snapshots(run_dir: Path, seed: int, started_at: float) -> Tuple[Path, Path]:
    """Require both snapshots from this seed's current simulation."""
    entry = run_dir / f"{seed}_study_entry_cohort.csv"
    ends = sorted(run_dir.glob(f"{seed}_study_cohort_*.csv"),
                  key=lambda p: p.stat().st_mtime)
    if not entry.is_file() or entry.stat().st_mtime < started_at - 2:
        raise FileNotFoundError(f"Missing current study-entry snapshot: {entry}")
    ends = [p for p in ends if p.stat().st_mtime >= started_at - 2]
    if len(ends) != 1:
        raise RuntimeError(f"Expected one current end snapshot for seed {seed}, got {len(ends)}")
    return entry, ends[0]


def build_leakage_free_cohort(entry_path: Path, end_path: Path) -> pd.DataFrame:
    """Join entry-time predictors with final labels, never final attributes."""
    entry = pd.read_csv(entry_path)
    end = pd.read_csv(end_path)
    for name, df in (("entry", entry), ("end", end)):
        if df.empty or "ID" not in df.columns:
            raise ValueError(f"{name} snapshot empty or missing ID")
        if df["ID"].isna().any() or df["ID"].duplicated().any():
            raise ValueError(f"{name} snapshot has null or duplicate IDs")
    if set(entry["ID"]) != set(end["ID"]):
        raise ValueError("Entry and end snapshots have different agent IDs")
    missing = sorted(set(OUTCOME_COLS) - set(end.columns))
    if missing:
        raise ValueError(f"End snapshot missing outcomes: {missing}")

    # Only snapshot-entry attributes are candidate predictors. Explicitly exclude
    # any pre-existing outcome flags and event timestamps from that snapshot.
    predictors = [
        c for c in entry.columns
        if c != "ID"
        and c.lower() not in EXCLUDE_ENTRY_COLUMNS
        and not c.lower().startswith(("rearrest_", "reincarceration_"))
        and c not in OUTCOME_COLS
    ]
    result = entry[["ID"] + predictors].merge(
        end[["ID"] + OUTCOME_COLS], on="ID", how="inner",
        validate="one_to_one", sort=False
    )
    if len(result) != len(entry):
        raise ValueError("Study-cohort size changed during snapshot join")
    for col in OUTCOME_COLS:
        y = normalize_binary(result[col])
        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError(f"Missing or invalid final labels: {col}")
        result[col] = y.astype("int8")
    # Monotonicity of cumulative rearrest labels is required.
    if ((result[OUTCOME_COLS[0]] > result[OUTCOME_COLS[1]]) |
        (result[OUTCOME_COLS[1]] > result[OUTCOME_COLS[2]])).any():
        raise ValueError("Cumulative rearrest labels are not monotone")
    return result


def run_one_seed(simulation_id: int, seed: int, calibration_params: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Run one calibrated ABM realization and return its exported study cohort."""
    set_seed(seed)

    run_dir = RUNS_DIR / f"simulation_{simulation_id:02d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale snapshots to prevent accidental reuse after a failed run.
    for old in run_dir.glob(f"{seed}_study_entry_cohort.csv"):
        old.unlink()
    for old in run_dir.glob(f"{seed}_study_cohort_*.csv"):
        old.unlink()

    params = dict(BASE_PARAMS)
    params.update({
        "seed": seed,
        "calibration_params": calibration_params,
        "output_directory": str(run_dir),
    })

    print("\n" + "=" * 78)
    print(f"SIMULATION {simulation_id:02d} | seed={seed}")
    print(f"Output: {run_dir}")
    print("=" * 78)

    started_at = time.time()
    model = RecidivismModel(**params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()

    elapsed = time.time() - started_at
    # The model advances current_month to max_months + 1 on its last step,
    # then performs the final export on the following step.
    if model.current_month != model.max_months + 1:
        raise RuntimeError(f"Seed {seed} stopped early at month {model.current_month}")
    entry_path, end_path = find_study_snapshots(run_dir, seed, started_at)
    cohort = build_leakage_free_cohort(entry_path, end_path)
    cohort_path = end_path

    # Preserve provenance for seed-aware train/test splitting and CV.
    cohort.insert(0, "simulation_id", simulation_id)
    cohort.insert(1, "simulation_seed", seed)
    cohort.insert(2, "source_row_id", np.arange(1, len(cohort) + 1))

    # Unique cross-simulation record key without overwriting the model's own ID.
    cohort.insert(
        0,
        "synthetic_record_id",
        [f"SIM{simulation_id:02d}_S{seed}_R{i:06d}" for i in range(1, len(cohort) + 1)],
    )

    summary = {
        "simulation_id": simulation_id,
        "seed": seed,
        "n_study_agents": int(len(cohort)),
        "runtime_sec": float(elapsed),
        "source_csv": str(cohort_path),
        "entry_csv": str(entry_path),
    }
    for col in OUTCOME_COLS:
        y = normalize_binary(cohort[col])
        summary[f"{col}_rate"] = float(y.mean())
        summary[f"{col}_n_nonmissing"] = int(y.notna().sum())

    print(
        f"Completed seed {seed}: N={len(cohort):,} | "
        f"3Y={summary['rearrest_3_yrs_rate']:.3f} | "
        f"6Y={summary['rearrest_6_yrs_rate']:.3f} | "
        f"9Y={summary['rearrest_9_yrs_rate']:.3f} | "
        f"{elapsed/60:.1f} min"
    )
    return cohort, summary


def assemble_target_cohort(seed_cohorts: List[pd.DataFrame], target_agents: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate records across every run so no independent seed is dropped."""
    sizes = [len(df) for df in seed_cohorts]
    if sum(sizes) < target_agents:
        raise ValueError("Insufficient agents to reach target")
    # Proportional allocations with deterministic largest-remainder rounding.
    fractions = [target_agents * n / sum(sizes) for n in sizes]
    allocations = [int(math.floor(x)) for x in fractions]
    leftover = target_agents - sum(allocations)
    order = sorted(range(len(sizes)), key=lambda i: (-(fractions[i] - allocations[i]), i))
    for i in order[:leftover]:
        allocations[i] += 1
    if any(n == 0 for n in allocations):
        raise ValueError("Target too small to retain all independent simulations")
    parts, rows = [], []
    for df, take_n in zip(seed_cohorts, allocations):
        seed = int(df["simulation_seed"].iloc[0])
        sim_id = int(df["simulation_id"].iloc[0])
        selected = df.sample(n=take_n, random_state=seed).copy() if take_n < len(df) else df.copy()
        parts.append(selected)
        rows.append({"simulation_id": sim_id, "seed": seed,
                     "available_agents": len(df), "selected_agents": take_n,
                     "partially_sampled_to_hit_target": take_n < len(df)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows)


def combined_outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, target in BJS_TARGETS.items():
        y = normalize_binary(df[col])
        rate = float(y.mean())
        rows.append({
            "horizon": col.replace("rearrest_", "").replace("_yrs", "-Year"),
            "outcome_column": col,
            "n_nonmissing": int(y.notna().sum()),
            "synthetic_rate": rate,
            "bjs_target": target,
            "difference_pp": (rate - target) * 100,
        })
    return pd.DataFrame(rows)


def demographic_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["Race", "Gender", "pcra_tier", "offense"]:
        if col not in df.columns:
            continue
        counts = df[col].fillna("Missing").astype(str).value_counts(dropna=False)
        for category, n in counts.items():
            rows.append({
                "variable": col,
                "category": category,
                "n": int(n),
                "percent": float(n / len(df) * 100),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Dissertation-ready charts
# -----------------------------------------------------------------------------
def save_figure(fig: go.Figure, stem: str) -> None:
    html_path = CHART_DIR / f"{stem}.html"
    png_path = CHART_DIR / f"{stem}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), width=1800, height=900, scale=2)
        print(f"Chart -> {png_path}")
    except Exception as exc:
        print(f"PNG skipped ({exc}). Interactive chart -> {html_path}")


def base_layout(title: str, subtitle: str = "") -> Dict:
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += f"<br><sup>{subtitle}</sup>"
    return dict(
        template="plotly_white",
        width=1800,
        height=900,
        title=dict(
            text=full_title,
            x=0.5,
            xanchor="center",
            y=0.975,
            yanchor="top",
            font=dict(size=26),
            pad=dict(t=4, b=4),
        ),
        font=dict(family="Arial", size=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=105, r=70, t=105, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="center", x=0.5),
    )


def chart_seed_cohort_sizes(seed_summary: pd.DataFrame) -> None:
    fig = go.Figure(go.Bar(
        x=[f"Seed {s}" for s in seed_summary["seed"]],
        y=seed_summary["n_study_agents"],
        text=[f"{n:,}" for n in seed_summary["n_study_agents"]],
        textposition="outside",
        name="Study-eligible agents",
    ))
    fig.update_layout(**base_layout(
        "Study-Eligible Synthetic Agents by ABM Simulation Seed",
        "Each seed represents an independent realization of the fixed calibrated ABM",
    ))
    fig.update_yaxes(title="Study-eligible agents", range=[0, float(seed_summary["n_study_agents"].max()) * 1.10], gridcolor="#E6E6E6")
    fig.update_xaxes(title="Simulation seed")
    save_figure(fig, "chart01_agents_generated_by_seed")


def chart_seed_rearrest_rates(seed_summary: pd.DataFrame) -> None:
    fig = go.Figure()
    specs = [
        ("rearrest_3_yrs_rate", "3-Year", BJS_TARGETS["rearrest_3_yrs"]),
        ("rearrest_6_yrs_rate", "6-Year", BJS_TARGETS["rearrest_6_yrs"]),
        ("rearrest_9_yrs_rate", "9-Year", BJS_TARGETS["rearrest_9_yrs"]),
    ]
    x = [f"S{i}" for i in seed_summary["simulation_id"]]
    for col, label, _ in specs:
        fig.add_trace(go.Scatter(x=x, y=seed_summary[col], mode="lines+markers", name=label))
    fig.update_layout(**base_layout(
        "Rearrest Rates Across Independent Synthetic Cohorts",
        "Seed-level stability check for the ML data-generating process",
    ))
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=0.925, xanchor="center", x=0.5))
    fig.update_yaxes(title="Cumulative rearrest rate", tickformat=".0%", gridcolor="#E6E6E6")
    fig.update_xaxes(title="Independent ABM realization")
    save_figure(fig, "chart02_rearrest_rates_across_seeds")


def chart_combined_vs_bjs(outcomes: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=outcomes["horizon"], y=outcomes["synthetic_rate"],
        name="Combined synthetic cohort",
        text=[f"{v:.1%}" for v in outcomes["synthetic_rate"]], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=outcomes["horizon"], y=outcomes["bjs_target"],
        name="BJS benchmark",
        text=[f"{v:.1%}" for v in outcomes["bjs_target"]], textposition="outside",
    ))
    fig.update_layout(**base_layout(
        "Combined Synthetic Cohort vs BJS Rearrest Benchmarks",
        "Descriptive validation of the pooled ML synthetic dataset",
    ), barmode="group")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=0.925, xanchor="center", x=0.5))
    fig.update_yaxes(title="Cumulative rearrest rate", tickformat=".0%", range=[0, 1.0], gridcolor="#E6E6E6")
    fig.update_xaxes(title="Rearrest horizon")
    save_figure(fig, "chart03_combined_rearrest_vs_bjs")


def chart_seed_contribution(allocation: pd.DataFrame) -> None:
    fig = go.Figure(go.Bar(
        x=[f"S{i}" for i in allocation["simulation_id"]],
        y=allocation["selected_agents"],
        text=[f"{n:,}" for n in allocation["selected_agents"]],
        textposition="outside",
        name="Agents retained",
    ))
    fig.update_layout(**base_layout(
        "ABM Simulation Contributions to the Final ML Cohort",
        "Provenance of the approximately 50,000-agent synthetic dataset",
    ))
    fig.update_yaxes(title="Agents retained in combined cohort", range=[0, float(allocation["selected_agents"].max()) * 1.10], gridcolor="#E6E6E6")
    fig.update_xaxes(title="Independent ABM realization")
    save_figure(fig, "chart04_seed_contribution_to_final_cohort")


def chart_demographic_distribution(demo: pd.DataFrame, variable: str, stem: str) -> None:
    d = demo[demo["variable"] == variable].sort_values("percent", ascending=False)
    if d.empty:
        return
    fig = go.Figure(go.Bar(
        x=d["category"], y=d["percent"],
        text=[f"{v:.1f}%" for v in d["percent"]], textposition="outside",
        name=variable,
    ))
    fig.update_layout(**base_layout(
        f"{variable} Distribution in Combined Synthetic ML Cohort",
        f"N = {int(d['n'].sum()):,} synthetic agents",
    ))
    fig.update_yaxes(title="Percent of combined cohort", ticksuffix="%", range=[0, float(d["percent"].max()) * 1.15], gridcolor="#E6E6E6")
    fig.update_xaxes(title=variable)
    save_figure(fig, stem)


def generate_charts(seed_summary: pd.DataFrame, allocation: pd.DataFrame,
                    outcomes: pd.DataFrame, demo: pd.DataFrame) -> None:
    chart_seed_cohort_sizes(seed_summary)
    chart_seed_rearrest_rates(seed_summary)
    chart_combined_vs_bjs(outcomes)
    chart_seed_contribution(allocation)
    chart_demographic_distribution(demo, "Race", "chart05_race_distribution")
    chart_demographic_distribution(demo, "Gender", "chart06_gender_distribution")
    chart_demographic_distribution(demo, "pcra_tier", "chart07_pcra_tier_distribution")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ~50K multiseed calibrated ABM synthetic ML cohort")
    parser.add_argument("--target-agents", type=int, default=TARGET_AGENTS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--keep-all-generated", action="store_true",
                        help="Also save one uncapped CSV containing every generated seed cohort.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the script output root before starting.")
    parser.add_argument("--replot", action="store_true",
                        help="Regenerate charts only from existing dataset/summaries; do not rerun the ABM.")
    return parser.parse_args()


def replot_existing_outputs() -> None:
    """Regenerate dissertation charts from existing outputs without rerunning the ABM."""
    ensure_dirs()

    seed_summary_path = SUMMARY_DIR / "seed_generation_summary.csv"
    allocation_path = SUMMARY_DIR / "final_cohort_seed_allocation.csv"
    outcome_path = SUMMARY_DIR / "combined_outcome_summary.csv"
    demo_path = SUMMARY_DIR / "combined_demographic_summary.csv"

    required = [seed_summary_path, allocation_path, outcome_path, demo_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot replot because required summary files are missing:\n  - "
            + "\n  - ".join(missing)
            + "\nRun the generator once without --replot first."
        )

    seed_summary_df = pd.read_csv(seed_summary_path)
    allocation_df = pd.read_csv(allocation_path)
    outcome_df = pd.read_csv(outcome_path)
    demo_df = pd.read_csv(demo_path)

    print("\nREPLOT MODE — existing ABM results will be reused")
    print(f"Summaries : {SUMMARY_DIR}/")
    print(f"Charts    : {CHART_DIR}/")
    generate_charts(seed_summary_df, allocation_df, outcome_df, demo_df)
    print("\n✅ Replot complete. No ABM simulations were rerun.")


def main() -> None:
    args = parse_args()

    global ROOT_OUTPUT_DIR, RUNS_DIR, DATA_DIR, SUMMARY_DIR, CHART_DIR

    if args.replot:
        if args.overwrite:
            raise ValueError("Do not combine --replot with --overwrite; --overwrite would delete the existing results needed for replotting.")
        replot_existing_outputs()
        return
    if args.overwrite and ROOT_OUTPUT_DIR.exists():
        shutil.rmtree(ROOT_OUTPUT_DIR)
    ensure_dirs()

    #calibration_params = get_global_calibration_params()
    calibration_params = get_global_calibration_params(overrides={
    "bias_factor": 1,
     })
    if args.n_seeds < 5:
        raise ValueError("Use at least five independent seeds for leave-one-simulation-out evaluation")
    if args.target_agents < args.n_seeds:
        raise ValueError("Target agent count must be >= number of seeds")
    seeds = build_seed_list(args.n_seeds, args.base_seed, args.seed_step)

    print("\nMULTI-SEED CALIBRATED ABM SYNTHETIC DATA GENERATION")
    print(f"Target final ML cohort : {args.target_agents:,}")
    print(f"Independent seeds      : {seeds}")
    print(f"Initial agents / seed  : {BASE_PARAMS['initial_agents']:,}")
    print("Calibration            : get_global_calibration_params() [FIXED]")
    print("Bias configuration     : factor=1.0, scope=supervision_only")
    print(f"Root output            : {ROOT_OUTPUT_DIR}/")

    seed_cohorts: List[pd.DataFrame] = []
    seed_summaries: List[Dict] = []
    generated_total = 0

    for simulation_id, seed in enumerate(seeds, start=1):
        cohort, summary = run_one_seed(simulation_id, seed, calibration_params)
        seed_cohorts.append(cohort)
        seed_summaries.append(summary)
        generated_total += len(cohort)


    if generated_total < args.target_agents:
        raise RuntimeError(
            f"Only {generated_total:,} study agents were generated from {len(seed_cohorts)} seeds, "
            f"below target {args.target_agents:,}. Increase --n-seeds and rerun."
        )

    seed_summary_df = pd.DataFrame(seed_summaries)
    combined_df, allocation_df = assemble_target_cohort(seed_cohorts, args.target_agents)

    # Reorder deterministically for reproducibility without destroying provenance.
    combined_df = combined_df.sort_values(
        ["simulation_id", "source_row_id"], kind="stable"
    ).reset_index(drop=True)

    # Save core ML dataset.
    final_csv = DATA_DIR / f"synthetic_ml_cohort_multiseed_N{len(combined_df)}.csv"
    combined_df.to_csv(final_csv, index=False)
    # Candidate columns were measured at entry, but require domain review before ML.
    provenance = {"synthetic_record_id", "simulation_id", "simulation_seed", "source_row_id", "ID"}
    candidate_features = [c for c in combined_df.columns if c not in provenance and c not in OUTCOME_COLS]
    with open(DATA_DIR / "entry_feature_candidates_REVIEW_REQUIRED.json", "w") as f:
        json.dump({"candidate_study_entry_features": candidate_features,
                   "warning": "Review and approve an explicit feature allowlist in the ML training script; do not automatically train on all columns."}, f, indent=2)

    if args.keep_all_generated:
        all_generated = pd.concat(seed_cohorts, ignore_index=True)
        all_generated.to_csv(DATA_DIR / f"all_generated_before_cap_N{len(all_generated)}.csv", index=False)

    # Summaries.
    outcome_df = combined_outcome_summary(combined_df)
    demo_df = demographic_summary(combined_df)

    seed_summary_df.to_csv(SUMMARY_DIR / "seed_generation_summary.csv", index=False)
    allocation_df.to_csv(SUMMARY_DIR / "final_cohort_seed_allocation.csv", index=False)
    outcome_df.to_csv(SUMMARY_DIR / "combined_outcome_summary.csv", index=False)
    demo_df.to_csv(SUMMARY_DIR / "combined_demographic_summary.csv", index=False)

    metadata = {
        "purpose": "ML synthetic-data generation from multiple independent calibrated ABM realizations",
        "target_agents": int(args.target_agents),
        "final_agents": int(len(combined_df)),
        "total_agents_generated_before_cap": int(generated_total),
        "independent_seeds_used": [int(x) for x in seed_summary_df["seed"].tolist()],
        "n_independent_seeds_used": int(len(seed_summary_df)),
        "base_model_params": BASE_PARAMS,
        "bias_configuration": {"bias_factor": 1.0, "bias_scope": "supervision_only"},
        "calibration_source": "get_global_calibration_params()",
        "bjs_targets_descriptive_reference": BJS_TARGETS,
        "provenance_columns": ["synthetic_record_id", "simulation_id", "simulation_seed", "source_row_id"],
        "recommended_ml_validation": (
            "Use simulation_seed/simulation_id for group-aware train/test splitting and, where practical, group-aware CV. "
            "Do not include provenance columns as predictive features. Review entry feature candidates before training."
        ),
        "final_dataset": str(final_csv),
    }
    with open(SUMMARY_DIR / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    generate_charts(seed_summary_df, allocation_df, outcome_df, demo_df)

    print("\n" + "=" * 78)
    print("FINAL SYNTHETIC ML COHORT")
    print("=" * 78)
    print(f"Final N                 : {len(combined_df):,}")
    print(f"Independent seeds used : {len(seed_summary_df)}")
    print(f"Dataset                 : {final_csv}")
    print(f"Seed model runs         : {RUNS_DIR}/")
    print(f"Summaries               : {SUMMARY_DIR}/")
    print(f"Charts                  : {CHART_DIR}/")
    print("\nRearrest outcomes in final combined cohort:")
    print(outcome_df[["horizon", "synthetic_rate", "bjs_target", "difference_pp"]].to_string(index=False))
    print("\nNOTE: Review entry_feature_candidates_REVIEW_REQUIRED.json before training; exclude provenance fields.")


if __name__ == "__main__":
    main()
