import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.RecidivismModel import RecidivismModel

# ─── Parameters ───
seeds = [41, 42, 43, 44, 45, 46 , 47, 48, 49, 50]
warmup_months = 72
study_months = 108  # 9 years
windows = ['rearrest_3_yrs', 'rearrest_6_yrs', 'rearrest_9_yrs']
results = {w: [] for w in windows}

# ─── Run Simulations ───
for seed in seeds:
    model = RecidivismModel(
        initial_agents=250,
        bias_factor=0.05,
        monthly_intake=10,
        warmup_months=warmup_months,
        study_months=study_months,
        enable_peer_influence=True,
        weights=None,  # Replace with your tuned weights
        peer_config=None,
        seed=seed
    )
    while model.running:
        model.step()

    for w in windows:
        rate = model.calculate_flag_rate(w)
        results[w].append(rate)

# ─── Plot Results ───
labels = [f"{w}-Year" for w in windows]
x = np.arange(len(labels))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))
for i, seed in enumerate(seeds):
    rates = [results[w][i] for w in windows]
    ax.bar(x + i * bar_width, rates, width=bar_width, label=f"Seed {seed}")

# ─── Add Benchmarks and Averages ───
benchmarks = [0.68, 0.79, 0.83]
avg_rates = [np.mean(results[w]) for w in windows]
ax.plot(x + 2 * bar_width, avg_rates, 'k--o', label="Mean Across Seeds")
ax.plot(x + 2 * bar_width, benchmarks, 'r--x', label="BJS Benchmark")

# ─── Final Touches ───
ax.set_xticks(x + 2 * bar_width)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.set_ylabel("Recidivism Rate")
ax.set_title("Multi-Seed Recidivism Validation vs BJS Benchmarks")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("Charts/multi_seed_validation.png")
plt.show()
