import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.RecidivismModel import RecidivismModel

def detect_stable_warmup(model_class, max_warmup=300, step_size=12, threshold=0.01, seeds=[42, 43, 44, 45, 46]):
    """
    Runs the model across multiple seeds and warm-up durations.
    Computes average month-to-month change in recidivism rate over last 12 months.
    Detects earliest stable warm-up and plots annotated chart.
    """
    warmup_values = list(range(step_size, max_warmup + step_size, step_size))
    stability_by_warmup = []

    for warmup_months in warmup_values:
        deltas = []
        for seed in seeds:
            model = model_class(
                initial_agents=100,
                bias_factor=0.05,
                monthly_intake=10,
                warmup_months=warmup_months,
                study_months=0,
                enable_peer_influence=True,
                weights=None,
                peer_config=None,
                seed=seed
            )
            while model.running:
                model.step()

            rates = model.monthly_recidivism_rate[-12:]
            if len(rates) < 2:
                continue
            delta = np.mean([abs(rates[i] - rates[i-1]) for i in range(1, len(rates))])
            deltas.append(delta)

        avg_delta = np.mean(deltas) if deltas else None
        stability_by_warmup.append((warmup_months, avg_delta))
        print(f"Warmup={warmup_months} → Avg ΔRecidivism={avg_delta:.4f}")

    # ─── Detect Optimal Warm-up ───
    optimal_warmup = None
    for warmup, delta in stability_by_warmup:
        if delta is not None and delta < threshold and optimal_warmup is None:
            optimal_warmup = warmup
            print(f"✅ Stability achieved at {warmup} months")

    if optimal_warmup is None:
        optimal_warmup = max_warmup

    # ─── Plot Stability Chart ───
    warmup_vals, avg_deltas = zip(*stability_by_warmup)
    plt.figure(figsize=(10, 6))
    plt.plot(warmup_vals, avg_deltas, marker='o', label='Avg ΔRecidivism Rate (multi-seed)')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Stability Threshold ({threshold})')
    plt.axvline(x=optimal_warmup, color='g', linestyle='--', label=f'Optimal Warm-up ({optimal_warmup} months)')
    plt.annotate(f'Optimal = {optimal_warmup} months',
                 xy=(optimal_warmup, avg_deltas[warmup_vals.index(optimal_warmup)]),
                 xytext=(optimal_warmup + 10, threshold + 0.01),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=10, color='green')
    plt.xlabel("Warm-up Months")
    plt.ylabel("Avg ΔRecidivism Rate (last 12 months)")
    plt.title("Model Stability(Monthly Recidivism Rate) vs Warm-up Duration (Multi-Seed Average)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Charts/recidivism_stability_chart.png")
    plt.show()

    return optimal_warmup

# ─── Example Usage ───
if __name__ == "__main__":
    optimal_warmup = detect_stable_warmup(RecidivismModel)
    print(f"📌 Using {optimal_warmup} months as warm-up for study phase.")

    model = RecidivismModel(
        initial_agents=100,
        bias_factor=0.05,
        monthly_intake=10,
        warmup_months=optimal_warmup,
        study_months=120,
        enable_peer_influence=True,
        weights=None,
        peer_config=None,
        seed=42
    )

    while model.running:
        model.step()

    print(f"✅ Study complete. Recidivism rate: {model.percent_study_recidivists_in_window(3):.4f}")
