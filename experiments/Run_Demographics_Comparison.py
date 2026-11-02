# demographics_comparison_flagrate_ci_parallel.py
import os, datetime, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_peer_influence_config, get_flat_risk_weights, get_global_calibration_params

# -----------------------
# CONFIG
# -----------------------
windows = ['rearrest_3_yrs', 'rearrest_6_yrs', 'rearrest_9_yrs']
TARGET_YEARS = [3, 6, 9]
PEER_CONFIG = get_peer_influence_config()
BJS_BASELINE = {
    3: {"Male": 0.68, "Female": 0.58, "White": 0.66, "Black": 0.72, "Hispanic": 0.68},
    6: {"Male": 0.79, "Female": 0.70, "White": 0.77, "Black": 0.82, "Hispanic": 0.79},
    9: {"Male": 0.84, "Female": 0.77, "White": 0.81, "Black": 0.87, "Hispanic": 0.84}
}

N_SEEDS = 10   # distinct random seeds
N_REPS  = 10   # repetitions per seed  →  100 total runs per window

BASE_DIR   = os.path.dirname(__file__)
CHARTS_DIR = os.path.join(BASE_DIR, "Demographic_Charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# -----------------------
# MODEL RUNNER
# Fixed: top-level function avoids joblib lambda-capture bug
# -----------------------
def run_model(seed, agents=1000, warmup_months=144, study_months=108):
    model = RecidivismModel(
        initial_agents=agents,
        bias_factor=0,
        monthly_intake=10,
        warmup_months=warmup_months,
        study_months=study_months,
        enable_peer_influence=True,
        weights=get_flat_risk_weights(),
        peer_config=PEER_CONFIG,
        seed=seed
    )
    while model.running:
        model.step()
    return model

def run_one(seed_rep_tuple, agents, warmup, study):
    """Top-level callable for joblib — avoids lambda closure capture bug."""
    seed, rep = seed_rep_tuple
    # Offset seed by rep so each repetition is genuinely independent
    effective_seed = seed * 1000 + rep
    model = run_model(effective_seed, agents, warmup, study)
    return model

# -----------------------
# DEMOGRAPHIC AGGREGATION
# -----------------------
def recid_by_group(model, flag_name):
    eligible_agents = [a for a in model.schedule.agents if getattr(a, "study_eligible_agent", False)]
    groups = {"Male": [], "Female": [], "White": [], "Black": [], "Hispanic": []}
    totals = {g: 0 for g in groups}
    for a in eligible_agents:
        if getattr(a, "Gender", None) in ["Male", "Female"]:
            totals[a.Gender] += 1
            if getattr(a, flag_name, False):
                groups[a.Gender].append(a)
        if getattr(a, "Race", None) in ["White", "Black", "Hispanic"]:
            totals[a.Race] += 1
            if getattr(a, flag_name, False):
                groups[a.Race].append(a)
    return {g: len(groups[g]) / max(totals[g], 1) for g in groups}

# -----------------------
# COMPARISON
# -----------------------
def compare_to_bjs(sim_rates, bjs_rates):
    diffs = {g: sim_rates.get(g, 0) - bjs_rates.get(g, 0) for g in bjs_rates}
    rmse = np.sqrt(np.mean([v**2 for v in diffs.values()]))
    return diffs, rmse

# -----------------------
# PLOTTING FUNCTIONS
# -----------------------
def plot_group_comparison(avg_rates, ci95, bjs_rates, year, timestamp_tag):
    groups = list(bjs_rates.keys())
    avg_vals = [avg_rates.get(g, 0) for g in groups]
    ci_vals  = [ci95.get(g, 0) for g in groups]
    bjs_vals = [bjs_rates.get(g, 0) for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(groups))
    ax.bar(x - 0.25, bjs_vals, width=0.4, label='BJS 2018')
    ax.bar(x + 0.25, avg_vals, width=0.4, yerr=ci_vals, capsize=5,
           label=f'Simulation (avg ± 95% CI, n={N_SEEDS * N_REPS})')
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel('Recidivism Rate')
    ax.set_ylim(0, 1)
    ax.set_title(f'Recidivism Comparison by Group — {year} Years  (n={N_SEEDS}×{N_REPS}={N_SEEDS*N_REPS} runs)')
    ax.legend()
    chart_file = os.path.join(CHARTS_DIR, f"{timestamp_tag}_demographics_{year}yr.png")
    fig.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return chart_file

def plot_differences(diffs, year, timestamp_tag):
    groups = list(diffs.keys())
    vals   = [diffs[g] for g in groups]
    colors = ['green' if abs(v) < 0.05 else 'orange' if abs(v) < 0.08 else 'red' for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(groups, vals, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline( 0.05, color='orange', linewidth=0.6, linestyle='--', label='±5pp threshold')
    ax.axhline(-0.05, color='orange', linewidth=0.6, linestyle='--')
    ax.set_ylabel('ΔRecid (Simulation − BJS)')
    ax.set_title(f'Difference vs BJS — {year} Years')
    ax.legend(fontsize=8)
    chart_file = os.path.join(CHARTS_DIR, f"{timestamp_tag}_differences_{year}yr.png")
    fig.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return chart_file

def plot_trends(all_avg_rates, all_ci95, timestamp_tag):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'Male':'steelblue','Female':'tomato','White':'seagreen',
              'Black':'darkorange','Hispanic':'mediumpurple'}
    for g in all_avg_rates[3].keys():
        vals = [all_avg_rates[y][g] for y in TARGET_YEARS]
        errs = [all_ci95[y][g]      for y in TARGET_YEARS]
        bjs  = [BJS_BASELINE[y][g]  for y in TARGET_YEARS]
        ax.plot(TARGET_YEARS, vals, marker='o', label=f'{g} sim', color=colors.get(g))
        ax.fill_between(TARGET_YEARS,
                        [v - e for v, e in zip(vals, errs)],
                        [v + e for v, e in zip(vals, errs)],
                        alpha=0.15, color=colors.get(g))
        ax.plot(TARGET_YEARS, bjs, marker='x', linestyle='--',
                color=colors.get(g), alpha=0.6, label=f'{g} BJS')
    ax.set_xlabel('Years After Release')
    ax.set_ylabel('Recidivism Rate')
    ax.set_title(f'Simulation vs BJS Trends by Group  (n={N_SEEDS}×{N_REPS}={N_SEEDS*N_REPS})')
    ax.legend(fontsize=7, ncol=2)
    chart_file = os.path.join(CHARTS_DIR, f"{timestamp_tag}_trends.png")
    fig.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return chart_file

def plot_rmse_trend(rmse_by_year, timestamp_tag):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(TARGET_YEARS, [rmse_by_year[y] for y in TARGET_YEARS], marker='s', color='red')
    ax.set_xlabel('Years After Release')
    ax.set_ylabel('RMSE vs BJS')
    ax.set_title('RMSE Trend Across Windows')
    chart_file = os.path.join(CHARTS_DIR, f"{timestamp_tag}_rmse_trend.png")
    fig.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return chart_file

# -----------------------
# TOP-LEVEL JOBLIB HELPER
# Must be at module level — joblib cannot pickle lambdas or closures.
# -----------------------
def _run_and_collect(seed_rep_tuple, flag_name, agents, warmup, study):
    model = run_one(seed_rep_tuple, agents, warmup, study)
    return recid_by_group(model, flag_name)

# -----------------------
# MAIN SCRIPT
# -----------------------
def main():
    agents      = 1500
    warmup      = 144
    study       = 108
    timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build full job list: all (seed, rep) combos
    job_list = [(s, r) for s in range(10, 10 + N_SEEDS) for r in range(N_REPS)]
    total_runs = len(job_list)
    print(f"Total runs: {N_SEEDS} seeds × {N_REPS} reps = {total_runs} per window")

    results      = []
    all_avg_rates = {}
    all_ci95      = {}
    rmse_by_year  = {}

    for y, flag in zip(TARGET_YEARS, windows):
        print(f"\n=== {y}-year window ({total_runs} parallel runs) ===")

        per_run_rates = Parallel(n_jobs=-1, verbose=5)(
            delayed(_run_and_collect)(sr, flag, agents, warmup, study)
            for sr in job_list
        )

        groups = list(BJS_BASELINE[y].keys())
        avg_rates = {g: np.mean([r[g] for r in per_run_rates]) for g in groups}
        ci95      = {g: 1.96 * np.std([r[g] for r in per_run_rates], ddof=1) / np.sqrt(total_runs)
                     for g in groups}
        diffs, rmse = compare_to_bjs(avg_rates, BJS_BASELINE[y])

        print(f"  RMSE={rmse:.4f}")
        for g in groups:
            print(f"  {g:10s}  sim={avg_rates[g]:.4f} ±{ci95[g]:.4f}  bjs={BJS_BASELINE[y][g]:.2f}"
                  f"  diff={diffs[g]:+.4f}")

        plot_group_comparison(avg_rates, ci95, BJS_BASELINE[y], y, timestamp_tag)
        plot_differences(diffs, y, timestamp_tag)

        row = {"year": y, "rmse": rmse}
        for g in groups:
            row[f"sim_{g}_avg"] = avg_rates[g]
            row[f"sim_{g}_ci95"] = ci95[g]
            row[f"bjs_{g}"]     = BJS_BASELINE[y][g]
            row[f"diff_{g}"]    = diffs[g]
        results.append(row)

        all_avg_rates[y] = avg_rates
        all_ci95[y]      = ci95
        rmse_by_year[y]  = rmse

    plot_trends(all_avg_rates, all_ci95, timestamp_tag)
    plot_rmse_trend(rmse_by_year, timestamp_tag)

    df = pd.DataFrame(results)
    csv_file = os.path.join(CHARTS_DIR, f"{timestamp_tag}_demographics_comparison_avg.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved -> {csv_file}")
    print("\n=== BASELINE_OFFSETS for phase2_bias.py ===")
    print("BASELINE_OFFSETS = {")
    for _, row in df.iterrows():
        yr = int(row['year'])
        print(f"    {yr}: {{\"Male\": {row['diff_Male']:+.4f}, \"Female\": {row['diff_Female']:+.4f},"
              f" \"White\": {row['diff_White']:+.4f},")
        print(f"        \"Black\": {row['diff_Black']:+.4f}, \"Hispanic\": {row['diff_Hispanic']:+.4f}}},")
    print("}")

if __name__ == "__main__":
    main()