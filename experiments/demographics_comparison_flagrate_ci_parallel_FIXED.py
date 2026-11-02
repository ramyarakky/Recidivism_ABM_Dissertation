import os, datetime, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

# Ensure imports work relative to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_peer_influence_config,
    get_flat_risk_weights
)

# -----------------------
# CONFIG
# -----------------------
WINDOWS = ['rearrest_3_yrs', 'rearrest_6_yrs', 'rearrest_9_yrs']
TARGET_YEARS = [3, 6, 9]
PEER_CONFIG = get_peer_influence_config()

BJS_BASELINE = {
    3: {"Male": 0.68, "Female": 0.58, "White": 0.66, "Black": 0.72, "Hispanic": 0.68},
    6: {"Male": 0.79, "Female": 0.70, "White": 0.77, "Black": 0.82, "Hispanic": 0.79},
    9: {"Male": 0.84, "Female": 0.77, "White": 0.81, "Black": 0.87, "Hispanic": 0.84},
}

BASE_DIR = os.path.dirname(__file__)
CHARTS_DIR = os.path.join(BASE_DIR, "Charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# -----------------------
# MODEL RUNNER
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

# -----------------------
# CLEAN GROUP RECIDIVISM
# -----------------------
def recid_by_group_clean(model, flag_name):
    """
    Computes recidivism rates separately by gender and race
    using mutually exclusive denominators.
    """
    eligible = [
        a for a in model.schedule.agents
        if getattr(a, "study_eligible_agent", False)
    ]

    groups = {
        "Male": [],
        "Female": [],
        "White": [],
        "Black": [],
        "Hispanic": []
    }

    for a in eligible:
        if a.Gender in ["Male", "Female"]:
            groups[a.Gender].append(a)
        if a.Race in ["White", "Black", "Hispanic"]:
            groups[a.Race].append(a)

    rates = {}
    for g, members in groups.items():
        if len(members) == 0:
            rates[g] = np.nan
        else:
            rates[g] = np.mean([getattr(a, flag_name, False) for a in members])

    return rates

# -----------------------
# CLEAN RMSE (NO OVERLAP)
# -----------------------
def compare_to_bjs_clean(sim_rates, bjs_rates):
    gender_groups = ["Male", "Female"]
    race_groups = ["White", "Black", "Hispanic"]

    diffs = {}
    gender_sq_err = []
    race_sq_err = []

    for g in bjs_rates:
        diff = sim_rates[g] - bjs_rates[g]
        diffs[g] = diff

        if g in gender_groups:
            gender_sq_err.append(diff ** 2)
        if g in race_groups:
            race_sq_err.append(diff ** 2)

    rmse_gender = np.sqrt(np.mean(gender_sq_err))
    rmse_race = np.sqrt(np.mean(race_sq_err))

    return diffs, rmse_gender, rmse_race

# -----------------------
# PLOTS
# -----------------------
def plot_group_comparison(avg, ci, bjs, year, tag):
    groups = list(bjs.keys())
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.25, [bjs[g] for g in groups], width=0.4, label="BJS")
    ax.bar(
        x + 0.25,
        [avg[g] for g in groups],
        width=0.4,
        yerr=[ci[g] for g in groups],
        capsize=5,
        label="Simulation (avg ± 95% CI)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recidivism Rate")
    ax.set_title(f"Recidivism by Group — {year} Years")
    ax.legend()

    file = os.path.join(CHARTS_DIR, f"{tag}_group_compare_{year}yr.png")
    fig.savefig(file, dpi=150)
    plt.close(fig)

def plot_rmse(rmse_gender, rmse_race, year, tag):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Gender RMSE", "Race RMSE"], [rmse_gender, rmse_race])
    ax.set_ylabel("RMSE vs BJS")
    ax.set_title(f"Model Fit by Demographic Dimension — {year} Years")

    file = os.path.join(CHARTS_DIR, f"{tag}_rmse_{year}yr.png")
    fig.savefig(file, dpi=150)
    plt.close(fig)

def plot_trends(all_rates, tag):
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in all_rates[3]:
        ax.plot(
            TARGET_YEARS,
            [all_rates[y][g] for y in TARGET_YEARS],
            marker="o",
            label=g
        )

    ax.set_xlabel("Years After Release")
    ax.set_ylabel("Recidivism Rate")
    ax.set_title("Simulated Recidivism Trends by Group")
    ax.legend()

    file = os.path.join(CHARTS_DIR, f"{tag}_trends.png")
    fig.savefig(file, dpi=150)
    plt.close(fig)

# -----------------------
# MAIN
# -----------------------
def main():
    seeds = list(range(10, 15))
    agents = 1000
    warmup, study = 144, 108
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    all_avg_rates = {}

    for year, flag in zip(TARGET_YEARS, WINDOWS):
        print(f"Running {year}-year window...")

        per_seed = Parallel(n_jobs=-1)(
            delayed(lambda s: recid_by_group_clean(
                run_model(s, agents, warmup, study),
                flag
            ))(s)
            for s in tqdm(seeds)
        )

        avg = {g: np.nanmean([r[g] for r in per_seed]) for g in BJS_BASELINE[year]}
        ci = {
            g: 1.96 * np.nanstd([r[g] for r in per_seed]) / np.sqrt(len(seeds))
            for g in BJS_BASELINE[year]
        }

        diffs, rmse_gender, rmse_race = compare_to_bjs_clean(avg, BJS_BASELINE[year])

        plot_group_comparison(avg, ci, BJS_BASELINE[year], year, tag)
        plot_rmse(rmse_gender, rmse_race, year, tag)

        row = {"year": year, "rmse_gender": rmse_gender, "rmse_race": rmse_race}
        for g in avg:
            row[f"sim_{g}_avg"] = avg[g]
            row[f"sim_{g}_ci95"] = ci[g]
            row[f"bjs_{g}"] = BJS_BASELINE[year][g]
            row[f"diff_{g}"] = diffs[g]

        rows.append(row)
        all_avg_rates[year] = avg

    plot_trends(all_avg_rates, tag)

    df = pd.DataFrame(rows)
    csv = os.path.join(CHARTS_DIR, f"{tag}_demographics_fixed.csv")
    df.to_csv(csv, index=False)
    print(f"Saved results → {csv}")

if __name__ == "__main__":
    main()
