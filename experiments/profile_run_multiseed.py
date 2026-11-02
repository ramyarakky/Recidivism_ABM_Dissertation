import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import plotly.graph_objects as go
# Ensure project-root imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from recidivism_abm.model.recidivism_model import RecidivismModel
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Simulation Parameters 
# ─────────────────────────────────────────────
MASTER_SEED = 424242
base_params = {
    "initial_agents": 1000,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0,    
    "enable_peer_influence": True
}

def generate_seeds(master_seed: int, n_runs: int, n_seeds: int):
    """
    Deterministically generate a (n_runs x n_seeds) matrix of unique integer seeds
    from a single master seed.
    """
    rng = np.random.default_rng(master_seed)

    # Generate many candidate seeds; enforce uniqueness
    total = n_runs * n_seeds
    seeds = rng.integers(1, 2**31 - 1, size=total, dtype=np.int64)

    # Ensure uniqueness (extremely likely already, but we enforce it)
    seen = set()
    unique = []
    for s in seeds:
        s_int = int(s)
        if s_int not in seen:
            seen.add(s_int)
            unique.append(s_int)
        if len(unique) == total:
            break

    # If somehow not enough unique seeds (very unlikely), keep drawing
    while len(unique) < total:
        s_int = int(rng.integers(1, 2**31 - 1))
        if s_int not in seen:
            seen.add(s_int)
            unique.append(s_int)

    # Reshape into runs
    return np.array(unique, dtype=np.int64).reshape(n_runs, n_seeds)

# ─────────────────────────────────────────────
# Run Model for a Single Seed (WORKER)
# ─────────────────────────────────────────────
def run_model_for_seed(seed):
    start_time = time.perf_counter()

    params = base_params.copy()
    params["seed"] = seed

    model = RecidivismModel(**params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()

    # Set cutoff month for evaluation consistency
    model.current_month = model.warmup_months + 108

    elapsed = time.perf_counter() - start_time

    return {
        "seed": seed,
        "rate_3yr": model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6yr": model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9yr": model.calculate_flag_rate("rearrest_9_yrs"),
        "runtime_sec": elapsed
    }

# ─────────────────────────────────────────────
# Run Multiple Seeds in Parallel
# ─────────────────────────────────────────────
def run_multiple_seeds_parallel(seed_list, max_workers=None):
    print("🚀 Running simulations in parallel...\n")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_model_for_seed, seed): seed
            for seed in seed_list
        }

        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                results.append(result)

                print(
                    f"✅ Seed {seed} completed | "
                    f"3y={result['rate_3yr']:.3f}, "
                    f"6y={result['rate_6yr']:.3f}, "
                    f"9y={result['rate_9yr']:.3f} | "
                    f"time={result['runtime_sec']:.2f}s"
                )

            except Exception as e:
                print(f"❌ Seed {seed} failed:", e)

    return pd.DataFrame(results)

def export_seed_schedule(seeds_matrix, master_seed, filename=None):
    """
    Export the full run × seed schedule to CSV for reproducibility.
    """
    records = []

    n_runs, n_seeds = seeds_matrix.shape

    for run_id in range(n_runs):
        for seed in seeds_matrix[run_id]:
            records.append({
                "run_id": run_id + 1,
                "seed": int(seed),
                "master_seed": master_seed
            })

    df_seeds = pd.DataFrame(records)

    if filename is None:
        filename = f"simulation_seeds_master_{master_seed}.csv"

    df_seeds.to_csv(filename, index=False)
    print(f"\n📁 Seed schedule exported to: {filename}")

    return df_seeds


def run_simulation(master_seed: int, n_runs: int, n_seeds: int, max_workers=None):
    """
    Run the model n_runs times with n_seeds per run.
    Seeds are generated deterministically from master_seed.
    """
    all_runs = []

    seeds_matrix = generate_seeds(master_seed, n_runs, n_seeds)

    # 🔹 EXPORT SEED SCHEDULE (ONCE)
    export_seed_schedule(seeds_matrix, master_seed)

    for run_id in range(n_runs):
        run_seeds = [int(s) for s in seeds_matrix[run_id]]

        print(
            f"\n🧪 Starting run {run_id + 1}/{n_runs} "
            f"| seeds: {run_seeds}"
        )

        df_run = run_multiple_seeds_parallel(
            run_seeds,
            max_workers=max_workers
        )

        df_run["run_id"] = run_id + 1
        df_run["master_seed"] = master_seed

        all_runs.append(df_run)

    return pd.concat(all_runs, ignore_index=True)




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
        name="Model mean",
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

    fig.show()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    n_seeds = 10
    n_runs = 10

    max_workers = min(os.cpu_count(), n_seeds)

    df_results = run_simulation(
        master_seed=MASTER_SEED,
        n_runs=n_runs,
        n_seeds=n_seeds,
        max_workers=max_workers
    )

    print("\n✅ Total simulations:", len(df_results))
    print(df_results.head())

    plot_clustered_bars_with_ci(df_results)

