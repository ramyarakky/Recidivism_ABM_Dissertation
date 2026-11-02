from model.RecidivismModel import RecidivismModel
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Simulation Parameters (excluding seed)
# ─────────────────────────────────────────────
base_params = {
    "initial_agents": 1200,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0.03,
    "enable_peer_influence": True
   # "weights": None,
   # "peer_config": None
}

# ─────────────────────────────────────────────
# Run Model for a Single Seed
# ─────────────────────────────────────────────
def run_model_for_seed(seed):
    params = base_params.copy()
    params["seed"] = seed
    print("Model Params:",params)

    model = RecidivismModel(**params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()
    # Manually set current_month to cutoff for evaluation
    model.current_month = model.warmup_months + 108  

    return {
        "seed": seed,
        "rate_3yr": model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6yr": model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9yr": model.calculate_flag_rate("rearrest_9_yrs")
    }

# ─────────────────────────────────────────────
# Run Multiple Seeds and Aggregate Results
# ─────────────────────────────────────────────
def run_multiple_seeds(seed_list):
    print("🚀 Running simulations across multiple seeds...\n")
    results = []

    for seed in seed_list:
        print(f"🔁 Running seed {seed}...")
        result = run_model_for_seed(seed)
        results.append(result)

    df = pd.DataFrame(results)
    avg_3yr = df["rate_3yr"].mean()
    avg_6yr = df["rate_6yr"].mean()
    avg_9yr = df["rate_9yr"].mean()

    print("\n📊 Average Recidivism Rates Across Seeds:")
    print(f"3-Year: {avg_3yr:.3f}")
    print(f"5-Year: {avg_6yr:.3f}")
    print(f"9-Year: {avg_9yr:.3f}")

    # ─────────────────────────────────────────────
    # Create Plot Comparing Averages to BJS Targets
    # ─────────────────────────────────────────────
    fig = go.Figure()

    # Bar chart for average observed rates
    fig.add_trace(go.Bar(
        x=["3-Year", "6-Year", "9-Year"],
        y=[avg_3yr, avg_6yr, avg_9yr],
        name="Average Observed",
        marker_color="blue"
    ))

    # BJS target rates
    bjs_targets = [0.68, 0.79, 0.83]

    # Scatter markers for targets
    fig.add_trace(go.Scatter(
        x=["3-Year", "6-Year", "9-Year"],
        y=bjs_targets,
        mode="markers+text",
        marker=dict(color="red", size=12, symbol="line-ns"),
        text=[
            f"Target: {bjs_targets[0]*100:.1f}%",
            f"Target: {bjs_targets[1]*100:.1f}%",
            f"Target: {bjs_targets[2]*100:.1f}%"
        ],
        textposition="top center",
        name="BJS Target"
    ))

    fig.update_layout(
        title="📈 Average Recidivism Rates vs BJS Targets",
        yaxis_title="Recidivism Rate",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        showlegend=True
    )
    
        # Save the plot locally as a PNG
    try:
        fig.write_image("recidivism_vs_bjs_targets.png")
        print("📁 Chart saved as 'recidivism_vs_bjs_targets.png'")
    except Exception as e:
        print("⚠️ Could not save chart. Make sure 'kaleido' is installed.")
        print("Error:", e)

    fig.show()

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    seed_list = [100, 101, 102, 103, 104]  # You can expand this list
    run_multiple_seeds(seed_list)
