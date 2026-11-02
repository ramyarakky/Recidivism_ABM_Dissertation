from model.RecidivismModel import RecidivismModel
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Simulation Parameters
# ─────────────────────────────────────────────
model_params = {
    "initial_agents": 250,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0.03,
    "enable_peer_influence": True,
    "weights": None,
    "peer_config": None,
    "seed": 40
}

# ─────────────────────────────────────────────
# Run Model Headlessly
# ─────────────────────────────────────────────
def run_headless():
    print("🚀 Starting RecidivismModel simulation...\n")

    model = RecidivismModel(**model_params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()
        print(f"📅 Month {model.current_month} | 👥 Active agents: {len(model.schedule.agents)}")

    print("\n🏁 Simulation complete. Exporting data...")

    # ─────────────────────────────────────────────
    # Collect and Export Data
    # ─────────────────────────────────────────────
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    model_data.index = model_data.index + 1  # Shift to start at Month 1
    model_data.to_csv("model_data.csv", index_label="Months")
    agent_data.to_csv("agent_data.csv")

    # ─────────────────────────────────────────────
    # Use Recidivism Window Function for 3, 6, and 9 Years
    # ─────────────────────────────────────────────
    rate_3yr = model.percent_study_recidivists_in_window(3)
    rate_6yr = model.percent_study_recidivists_in_window(6)
    rate_9yr = model.percent_study_recidivists_in_window(9)

    print(f"📊 Final 3-Year Recidivism Rate: {rate_3yr:.3f}")
    print(f"📊 Final 6-Year Recidivism Rate: {rate_6yr:.3f}")
    print(f"📊 Final 9-Year Recidivism Rate: {rate_9yr:.3f}")

    # ─────────────────────────────────────────────
    # Create Static Plot for All Three Windows
    # ─────────────────────────────────────────────
    fig = go.Figure()

    # Bar chart for observed rates
    fig.add_trace(go.Bar(
        x=["3-Year", "6-Year", "9-Year"],
        y=[rate_3yr, rate_6yr, rate_9yr],
        name="Observed",
        marker_color="blue"
    ))

    # BJS target rates (converted to decimals)
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
        name="Target"
    ))

    fig.update_layout(
        title="📈 Final Recidivism Rates vs BJS Targets",
        yaxis_title="Recidivism Rate",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        showlegend=True
    )

    fig.show()

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_headless()
