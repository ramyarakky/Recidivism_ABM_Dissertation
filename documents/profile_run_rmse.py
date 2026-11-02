from model.RecidivismModel import RecidivismModel
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─────────────────────────────────────────────
# Simulation Parameters (excluding seed)
# ─────────────────────────────────────────────
base_params = {
    "initial_agents": 5000,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0.03,
    "enable_peer_influence": True,
    "mode": "realistic"
}

# ─────────────────────────────────────────────
# RMSE Calculation
# ─────────────────────────────────────────────
def compute_rmse(observed, targets):
    errors = [(observed[i] - targets[i])**2 for i in range(len(targets))]
    return np.sqrt(np.mean(errors))

# ─────────────────────────────────────────────
# Run Model for a Single Seed
# ─────────────────────────────────────────────
def run_model_for_seed(seed):
    params = base_params.copy()
    params["seed"] = seed
    print("Model Params:", params)

    model = RecidivismModel(**params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()

    model.current_month = model.warmup_months + 108
    agents = model.schedule.agents

    subgroups = {
        "Male": [a for a in agents if a.Gender == "Male"],
        "Female": [a for a in agents if a.Gender == "Female"],
        "Black": [a for a in agents if a.Race == "Black"],
        "White": [a for a in agents if a.Race == "White"],
        "Hispanic": [a for a in agents if a.Race == "Hispanic"],
        "Other": [a for a in agents if a.Race == "Other"]
    }

    def group_rate(group, attr):
        return np.mean([getattr(a, attr) for a in group]) if group else np.nan

    return {
        "seed": seed,
        "rate_3yr": model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6yr": model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9yr": model.calculate_flag_rate("rearrest_9_yrs"),
        **{f"{g}_3yr": group_rate(a, "rearrest_3_yrs") for g, a in subgroups.items()},
        **{f"{g}_5yr": group_rate(a, "rearrest_6_yrs") for g, a in subgroups.items()},
        **{f"{g}_9yr": group_rate(a, "rearrest_9_yrs") for g, a in subgroups.items()}
    }


# ─────────────────────────────────────────────
# Run Multiple Seeds and Aggregate Results
# ─────────────────────────────────────────────
def run_multiple_seeds(seed_list):
    print("🚀 Running simulations across multiple seeds...\n")

    results = []
    max_workers = min(len(seed_list), os.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_model_for_seed, seed): seed for seed in seed_list}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✅ Completed seed {seed}")
            except Exception as e:
                print(f"❌ Seed {seed} failed: {e}")

    df = pd.DataFrame(results)
    # ─────────────────────────────────────────────
    # 📁 Export Subgroup Breakdown to CSV
    # ─────────────────────────────────────────────
    df.to_csv("recidivism_subgroups_by_seed.csv", index=False)
    print("📁 Subgroup CSV saved as 'recidivism_subgroups_by_seed.csv'")

    time_windows = ["3-Year", "6-Year", "9-Year"]
    bjs_targets = [0.68, 0.77, 0.83]

    # Compute means
    avg_rates = [df["rate_3yr"].mean(), df["rate_6yr"].mean(), df["rate_9yr"].mean()]

    # Compute standard deviations
    stds = [df["rate_3yr"].std(), df["rate_6yr"].std(), df["rate_9yr"].std()]
    n = len(seed_list)
    ci_bounds = {
        "lower": [avg_rates[i] - 1.96 * stds[i] / np.sqrt(n) for i in range(3)],
        "upper": [avg_rates[i] + 1.96 * stds[i] / np.sqrt(n) for i in range(3)]
    }

    # Compute RMSE
    rmse = compute_rmse(avg_rates, bjs_targets)
    subgroup_rmse = {}
    for group in ["Black", "White", "Hispanic", "Other", "Male", "Female"]:
        rates = [df[f"{group}_3yr"].mean(), df[f"{group}_5yr"].mean(), df[f"{group}_9yr"].mean()]
        subgroup_rmse[group] = compute_rmse(rates, bjs_targets)

    print("\n📐 Subgroup RMSEs vs BJS Targets:")
    for group, value in subgroup_rmse.items():
        print(f"{group}: RMSE = {value:.4f}")
    print("\n📊 Subgroup 3-Year Confidence Intervals:")
    for group in ["Black", "White", "Hispanic", "Other", "Male", "Female"]:
        col = f"{group}_3yr"
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            ci = 1.96 * std / np.sqrt(len(df))
            print(f"{group}: {mean:.3f} ± {ci:.3f}")

    print("\n📊 Average Recidivism Rates Across Seeds:")
    for i, label in enumerate(time_windows):
        print(f"{label}: {avg_rates[i]:.3f} ± {1.96 * stds[i] / np.sqrt(n):.3f}")
    print(f"\n📐 RMSE vs BJS Targets: {rmse:.4f}")

    # ─────────────────────────────────────────────
    # Line Chart Comparing BJS vs Observed Rates
    # ─────────────────────────────────────────────
    fig = go.Figure()

    # BJS Target Line
    fig.add_trace(go.Scatter(
        x=time_windows,
        y=bjs_targets,
        mode="lines+markers+text",
        name="BJS Target",
        marker=dict(color="red", size=10),
        line=dict(color="red", dash="dash"),
        text=[f"{r*100:.1f}%" for r in bjs_targets],
        textposition="top center"
    ))

    # Observed Line with Confidence Intervals
    fig.add_trace(go.Scatter(
        x=time_windows,
        y=avg_rates,
        mode="lines+markers+text",
        name="Observed Average",
        marker=dict(color="blue", size=10),
        line=dict(color="blue"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[ci_bounds["upper"][i] - avg_rates[i] for i in range(3)],
            arrayminus=[avg_rates[i] - ci_bounds["lower"][i] for i in range(3)],
            thickness=1.5,
            width=6,
            color="blue"
        ),
        text=[f"{avg_rates[i]*100:.1f}% ± {(ci_bounds['upper'][i] - avg_rates[i])*100:.1f}%" for i in range(3)],
        textposition="bottom center"
    ))

    fig.update_layout(
        title=f"📈 Recidivism Rates: BJS vs Observed (RMSE = {rmse:.4f})",
        yaxis_title="Recidivism Rate",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        showlegend=True
    )

    try:
        fig.write_image("recidivism_line_chart_with_ci.png")
        print("📁 Chart saved as 'recidivism_line_chart_with_ci.png'")
    except Exception as e:
        print("⚠️ Could not save chart. Make sure 'kaleido' is installed.")
        print("Error:", e)

    fig.show()
    fig_subgroup = go.Figure()

    for group, color in zip(["Black", "White", "Hispanic", "Other", "Male", "Female"],
                            ["black", "gray", "orange", "green", "blue", "pink"]):
        col = f"{group}_3yr"
        if col in df.columns:
            fig_subgroup.add_trace(go.Scatter(
                x=df["seed"],
                y=df[col],
                mode="lines+markers",
                name=group,
                line=dict(color=color),
                marker=dict(size=8)
            ))

    fig_subgroup.update_layout(
        title="📊 3-Year Recidivism Rates by Subgroup Across Seeds",
        xaxis_title="Seed",
        yaxis_title="Recidivism Rate",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        showlegend=True
    )

    try:
        fig_subgroup.write_image("subgroup_trends_3yr.png")
        print("📁 Chart saved as 'subgroup_trends_3yr.png'")
    except Exception as e:
        print("⚠️ Could not save subgroup chart. Make sure 'kaleido' is installed.")
        print("Error:", e)

    fig_subgroup.show()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    seed_list = [100, 101, 102, 103, 104] 
    run_multiple_seeds(seed_list)
