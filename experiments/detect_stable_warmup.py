import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.RecidivismModel import RecidivismModel

def detect_stable_warmup(model_class, max_warmup=300, step_size=12, threshold=0.0085, uncertainty_threshold=0.005, seeds=[42, 43, 44, 45, 46]):
    """
    Runs the model across multiple seeds and warm-up durations.
    Computes mean and std deviation of month-to-month change in recidivism rate over last 12 months.
    Statistically detects earliest stable warm-up and plots annotated chart.
    """
    warmup_values = list(range(step_size, max_warmup + step_size, step_size))
    mean_deltas = []
    std_deltas = []

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

        mean = np.mean(deltas) if deltas else None
        std = np.std(deltas) if deltas else None
        mean_deltas.append(mean)
        std_deltas.append(std)
        print(f"Warmup={warmup_months} → Mean Δ={mean:.4f}, Std={std:.4f}")

    # ─── Identify Optimal Warm-up ───
    optimal_warmup = None
    for i, (mean, std) in enumerate(zip(mean_deltas, std_deltas)):
        if mean is not None and mean < threshold and std < uncertainty_threshold:
            optimal_warmup = warmup_values[i]
            print(f"✅ Stability achieved at {optimal_warmup} months (mean={mean:.4f}, std={std:.4f})")
            break
    if optimal_warmup is None:
        optimal_warmup = max_warmup

    # ─── Plot with Error Bars ───
    plt.figure(figsize=(10, 6))
    plt.errorbar(warmup_values, mean_deltas, yerr=std_deltas, fmt='o-', capsize=5, label='Mean ± Std ΔRecidivism')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Mean Threshold ({threshold})')
    plt.axvline(x=optimal_warmup, color='g', linestyle='--', label=f'Optimal Warm-up ({optimal_warmup} months)')
    plt.annotate(f'Optimal = {optimal_warmup} months',
                 xy=(optimal_warmup, mean_deltas[warmup_values.index(optimal_warmup)]),
                 xytext=(optimal_warmup + 10, threshold + 0.01),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=10, color='green')
    plt.xlabel("Warm-up Months")
    plt.ylabel("Mean ΔRecidivism Rate (last 12 months)")
    plt.title("Model Stability (Monthly Recidivism Rate) vs Warm-up Duration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Charts", exist_ok=True)
    plt.savefig("Charts/warmup_stability_uncertainty.png")
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
        study_months=108,
        enable_peer_influence=True,
        weights=None,
        peer_config=None,
        seed=42
    )

    while model.running:
        model.step()

    print(f"✅ Study complete. Recidivism rate: {model.percent_study_recidivists_in_window(3):.4f}")
