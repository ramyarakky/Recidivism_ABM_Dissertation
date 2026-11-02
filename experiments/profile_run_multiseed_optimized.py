import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

# Ensure project-root imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from recidivism_abm_phase2.model.recidivism_model import RecidivismModel

# ─────────────────────────────────────────────
# Simulation Parameters
# ─────────────────────────────────────────────
MASTER_SEED = 424242
base_params = {
    "initial_agents": 1500,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0,
    "enable_peer_influence": True
}

# ─────────────────────────────────────────────
# Fast deterministic unique seed generation
# ─────────────────────────────────────────────
def generate_seeds(master_seed: int, n_runs: int, n_seeds: int):
    """
    Deterministically generate a (n_runs x n_seeds) matrix of unique seeds
    using a master RNG. Memory-safe and reproducible.
    """
    rng = np.random.default_rng(master_seed)

    total = n_runs * n_seeds

    # Draw unique seeds directly (no giant arrays)
    seeds = rng.choice(
        np.iinfo(np.int32).max,
        size=total,
        replace=False
    ).astype(np.int64)

    return seeds.reshape(n_runs, n_seeds)


def export_seed_schedule(seeds_matrix: np.ndarray, master_seed: int, filename: str = None) -> pd.DataFrame:
    """
    Export the full run × seed schedule to CSV for reproducibility.
    """
    if filename is None:
        filename = f"simulation_seeds_master_{master_seed}.csv"

    n_runs, n_seeds = seeds_matrix.shape

    df_seeds = pd.DataFrame({
        "run_id": np.repeat(np.arange(1, n_runs + 1), n_seeds),
        "seed": seeds_matrix.reshape(-1).astype(np.int64),
        "master_seed": master_seed
    })

    df_seeds.to_csv(filename, index=False)
    print(f"\n📁 Seed schedule exported to: {filename}")
    return df_seeds

# ─────────────────────────────────────────────
# Run Model for a Single Job (WORKER)
# ─────────────────────────────────────────────
def run_model_for_seed(job):
    """
    job = (run_id, seed)
    Using a single argument reduces pickling overhead vs multiple args.
    """
    run_id, seed = job
    start_time = time.perf_counter()

    params = base_params.copy()
    params["seed"] = int(seed)

    model = RecidivismModel(**params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()
    tech_rates = compute_tech_violation_rates(model)
    # Set cutoff month for evaluation consistency
    model.current_month = model.warmup_months + 108

    elapsed = time.perf_counter() - start_time

    return {
        "run_id": int(run_id),
        "seed": int(seed),
        "rate_3yr": model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6yr": model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9yr": model.calculate_flag_rate("rearrest_9_yrs"),
        "runtime_sec": elapsed,
        # ---- new: tech violation rates ----
        "tech_male": tech_rates["tech_by_gender"]["Male"],
        "tech_female": tech_rates["tech_by_gender"]["Female"],
        "tech_white": tech_rates["tech_by_race"]["White"],
        "tech_black": tech_rates["tech_by_race"]["Black"],
        "tech_hispanic": tech_rates["tech_by_race"]["Hispanic"],
        "tech_other": tech_rates["tech_by_race"]["Other"]
    }

# ─────────────────────────────────────────────
# Single-pool simulation runner (FAST + progress bar)
# ─────────────────────────────────────────────
def run_simulation(master_seed: int, n_runs: int, n_seeds: int, max_workers=None, verbose=True):
    """
    Run all simulations using ONE ProcessPool (fast).
    Shows a low-overhead tqdm progress bar.
    """
    seeds_matrix = generate_seeds(master_seed, n_runs, n_seeds)

    # Export seed schedule once
    export_seed_schedule(seeds_matrix, master_seed)

    # Build ALL jobs up-front: (run_id, seed)
    jobs = [
        (run_id, int(seed))
        for run_id in range(1, n_runs + 1)
        for seed in seeds_matrix[run_id - 1]
    ]
    total_jobs = len(jobs)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    # chunksize improves throughput (fewer IPC roundtrips)
    chunksize = max(1, total_jobs // (max_workers * 4))

    if verbose:
        print(f"\n🚀 Running {total_jobs} simulations | workers={max_workers} | chunksize={chunksize}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(run_model_for_seed, jobs, chunksize=chunksize)

        # Low-overhead progress bar (limits redraw frequency)
        results = list(
            tqdm(
                results_iter,
                total=total_jobs,
                desc="Simulations",
                miniters=max(1, total_jobs // 100),  # ~100 UI updates max
                smoothing=0.1
            )
        )

    df = pd.DataFrame(results)
    df["master_seed"] = master_seed

    if verbose:
        runtime_summary = (
            df.groupby("run_id")["runtime_sec"]
              .mean()
              .rename("avg_runtime_sec")
              .reset_index()
        )
        print("\n⏱ Avg runtime per run:")
        print(runtime_summary)

    return df

def compute_tech_violation_rates(model) -> dict:
    """
    Computes 'ever detected technical violation' rates by Gender and Race
    using agent attributes at end of run.

    Assumes agents have:
      - Gender, Race
      - Violations_Technical (bool or int)
      - study_eligible_agent (optional)
    """
    agents = list(getattr(model.schedule, "agents", []))

    # Optional: restrict to study-eligible cohort if you use that concept
    eligible = [a for a in agents if getattr(a, "study_eligible_agent", True)]
    if not eligible:
        eligible = agents

    def rate(group_agents):
        if not group_agents:
            return None
        # If Violations_Technical is boolean each quarter, you need an "ever" counter.
        # This fallback treats current value as indicator (works if you store "ever" or count).
        flagged = sum(1 for a in group_agents if bool(getattr(a, "Violations_Technical", False)))
        return flagged / len(group_agents)

    # By gender
    by_gender = {}
    for g in ["Male", "Female"]:
        grp = [a for a in eligible if getattr(a, "Gender", None) == g]
        by_gender[g] = rate(grp)

    # By race
    races = ["White", "Black", "Hispanic", "Other"]
    by_race = {}
    for r in races:
        grp = [a for a in eligible if getattr(a, "Race", None) == r]
        by_race[r] = rate(grp)

    # Gender x Race (optional)
    by_gender_race = {}
    for g in ["Male", "Female"]:
        for r in ["White", "Black", "Hispanic", "Other"]:
            grp = [a for a in eligible if getattr(a, "Gender", None) == g and getattr(a, "Race", None) == r]
            by_gender_race[(g, r)] = rate(grp)

    return {
        "tech_by_gender": by_gender,
        "tech_by_race": by_race,
        "tech_by_gender_race": by_gender_race,
        "n_eligible": len(eligible)
    }

def summarize_tech_violation_rates(df: pd.DataFrame):
    summary = {
        "Gender": {
            "Male": df["tech_male"].mean(),
            "Female": df["tech_female"].mean()
        },
        "Race": {
            "White": df["tech_white"].mean(),
            "Black": df["tech_black"].mean(),
            "Hispanic": df["tech_hispanic"].mean(),
            "Other": df["tech_other"].mean()
        }
    }
    return summary

def plot_tech_violation_by_gender(df):
    rates = summarize_tech_violation_rates(df)["Gender"]

    fig = go.Figure()

    fig.add_bar(
        x=list(rates.keys()),
        y=list(rates.values()),
        text=[f"{v:.1%}" for v in rates.values()],
        textposition="outside",
        marker_color=["#4C78A8", "#F58518"],
        name="Technical Violation Rate"
    )

    fig.update_layout(
        title="Detected Technical Violation Rate by Gender",
        yaxis_title="Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white"
    )
    try:
        fig.write_image("Technical_Violation_Rates_ByGender.png")
        print("\n📁 Chart saved as 'Technical_Violation_Rates_ByGender.png'")
    except Exception:
        print("⚠️ Install kaleido to enable PNG export")
    fig.show()

def plot_tech_violation_by_race(df):
    rates = summarize_tech_violation_rates(df)["Race"]

    fig = go.Figure()

    fig.add_bar(
        x=list(rates.keys()),
        y=list(rates.values()),
        text=[f"{v:.1%}" for v in rates.values()],
        textposition="outside",
        marker_color=["#9E9E9E", "#E45756", "#72B7B2", "#BAB0AC"],
        name="Technical Violation Rate"
    )

    fig.update_layout(
        title="Detected Technical Violation Rate by Race",
        yaxis_title="Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white"
    )
    try:
        fig.write_image("Technical_Violation_Rates_ByRace.png")
        print("\n📁 Chart saved as 'Technical_Violation_Rates_ByRace.png'")
    except Exception:
        print("⚠️ Install kaleido to enable PNG export")
    fig.show()


# ─────────────────────────────────────────────
# Aggregation + Plot
# ─────────────────────────────────────────────
def plot_clustered_bars_with_ci(df):
    labels = ["3-Year", "6-Year", "9-Year"]
    model_means = np.array([df["rate_3yr"].mean(), df["rate_6yr"].mean(), df["rate_9yr"].mean()])
    bjs = np.array([0.68, 0.79, 0.83])

    n = len(df)
    model_std = np.array([df["rate_3yr"].std(ddof=1), df["rate_6yr"].std(ddof=1), df["rate_9yr"].std(ddof=1)])
    ci95 = 1.96 * (model_std / np.sqrt(n))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels, y=model_means,
        name="Model mean (±95% CI)",
        marker_color="#4C78A8",
        error_y=dict(type="data", array=ci95, thickness=2),
        text=[f"{v:.0%}" for v in model_means],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=labels, y=bjs,
        name="BJS observed",
        marker_color="#9E9E9E",
        text=[f"{v:.0%}" for v in bjs],
        textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        title="Recidivism ABM Model Mean vs BJS Observed Recidivism (±95% CI on Model)",
        yaxis_title="Recidivism rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        bargap=0.35
    )
    try:
        fig.write_image("recidivism_vs_bjs_targets.png")
        print("\n📁 Chart saved as 'recidivism_vs_bjs_targets.png'")
    except Exception:
        print("⚠️ Install kaleido to enable PNG export")
    fig.show()

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    n_seeds =20
    n_runs = 10

    # For CPU-heavy ABM, using more workers than cores rarely helps
    max_workers = os.cpu_count() or 1

    df_results = run_simulation(
        master_seed=MASTER_SEED,
        n_runs=n_runs,
        n_seeds=n_seeds,
        max_workers=max_workers,
        verbose=True
    )
    plot_tech_violation_by_gender(df_results)
    plot_tech_violation_by_race(df_results)

    print("\n✅ Total simulations:", len(df_results))
    print(df_results.head())

    #plot_clustered_bars_with_ci(df_results)
