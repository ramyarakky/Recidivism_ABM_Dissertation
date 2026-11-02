#!/usr/bin/env python3
"""
Run Recidivism ABM for:
- 10 runs (replicates)
- 20 seeds per run
Then compute overall means + 95% CI for 3y/6y/9y rearrest rates.

Key ideas
---------
- Each (run_id, seed) is an independent simulation.
- We aggregate all 10*20 = 200 results (or also show per-run means).
- 95% CI uses t-distribution (works well for finite n).
"""

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objects as go

# Ensure project-root imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel  


# ─────────────────────────────────────────────
# Simulation Parameters (excluding seed)
# ─────────────────────────────────────────────
BASE_PARAMS = {
    "initial_agents": 1500,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0,
    "enable_peer_influence": True,
}

# BJS targets for plotting reference
BJS_TARGETS = {"3-Year": 0.68, "6-Year": 0.79, "9-Year": 0.83}


# ─────────────────────────────────────────────
# Utilities: confidence intervals (no SciPy needed)
# ─────────────────────────────────────────────
# Good approximations for t critical values (two-sided) at 95% for common df.
# For df > 120, 1.98 is already very close to 1.96.
_T95_LOOKUP = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    24: 2.064, 29: 2.045, 39: 2.023, 49: 2.010, 59: 2.000,
    79: 1.990, 99: 1.984, 120: 1.980,
}


def t_critical_95(df: int) -> float:
    """Return an approximate two-sided 95% t critical value for given degrees of freedom."""
    if df <= 1:
        return _T95_LOOKUP.get(1, 12.706)
    # Find nearest key >= df, else use ~1.96-1.98 for large df
    keys = sorted(_T95_LOOKUP.keys())
    for k in keys:
        if df <= k:
            return _T95_LOOKUP[k]
    return 1.96  # large df approximation


def mean_ci95(values: List[float]) -> Tuple[float, float, float]:
    """
    Compute mean and 95% CI (t-based): mean ± t * (sd / sqrt(n)).
    Returns (mean, lower, upper).
    """
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        m = float(vals[0])
        return (m, m, m)

    m = float(sum(vals) / n)
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    tcrit = t_critical_95(n - 1)
    half = tcrit * se
    return (m, m - half, m + half)


# ─────────────────────────────────────────────
# Worker: run model for a single (run_id, seed)
# ─────────────────────────────────────────────
def run_model_for_seed(task: Dict) -> Dict:
    """
    Runs one simulation for the given seed and returns rearrest rates.
    task expects:
      - run_id
      - seed
      - base_seed (optional, for reproducible derived seeds)
    """
    run_id = int(task["run_id"])
    seed = int(task["seed"])

    # Optional: stabilize Python + numpy RNGs if your model uses them implicitly
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    start_time = time.perf_counter()

    params = dict(BASE_PARAMS)
    params["seed"] = seed

    model = RecidivismModel(**params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()

    # Evaluate consistently at end of study
    model.current_month = model.warmup_months + model.study_months

    elapsed = time.perf_counter() - start_time

    return {
        "run_id": run_id,
        "seed": seed,
        "rate_3yr": model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6yr": model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9yr": model.calculate_flag_rate("rearrest_9_yrs"),
        "runtime_sec": elapsed,
    }


# ─────────────────────────────────────────────
# Run many seeds across many runs in parallel
# ─────────────────────────────────────────────
def build_tasks(n_runs: int, seeds_per_run: int, base_seed: int = 1000) -> List[Dict]:
    """
    Build a deterministic list of (run_id, seed) tasks.
    Each run gets its own block of seeds: base_seed + run_id*10000 + i
    """
    tasks: List[Dict] = []
    for run_id in range(1, n_runs + 1):
        for i in range(seeds_per_run):
            seed = base_seed + (run_id * 10_000) + i
            tasks.append({"run_id": run_id, "seed": seed})
    return tasks


def run_tasks_parallel(tasks: List[Dict], max_workers: int = None) -> pd.DataFrame:
    print(f"🚀 Running {len(tasks)} simulations in parallel...\n")
    results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_model_for_seed, t): t for t in tasks}

        for future in as_completed(futures):
            t = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(
                    f"✅ Run {r['run_id']:02d} | Seed {r['seed']} | "
                    f"3y={r['rate_3yr']:.3f}, 6y={r['rate_6yr']:.3f}, 9y={r['rate_9yr']:.3f} | "
                    f"time={r['runtime_sec']:.2f}s"
                )
            except Exception as e:
                print(f"❌ Failed run_id={t['run_id']} seed={t['seed']}: {e}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Summaries: per-run + overall mean and CI
# ─────────────────────────────────────────────
def summarize_with_ci(df: pd.DataFrame) -> Dict:
    """
    Returns dict with:
      - overall mean + CI for 3y/6y/9y using all sims
      - per-run means + CI (across seeds within each run)
    """
    out: Dict = {}

    # Overall across all simulations (n = n_runs * seeds_per_run)
    overall = {}
    for col, label in [("rate_3yr", "3-Year"), ("rate_6yr", "6-Year"), ("rate_9yr", "9-Year")]:
        m, lo, hi = mean_ci95(df[col].tolist())
        overall[label] = {"mean": m, "ci_low": lo, "ci_high": hi, "n": int(df[col].notna().sum())}
    out["overall"] = overall

    # Per-run aggregation (each run aggregates its 20 seeds)
    per_run_rows = []
    for run_id, g in df.groupby("run_id"):
        row = {"run_id": int(run_id), "n": int(g["rate_3yr"].notna().sum())}
        for col, label in [("rate_3yr", "3y"), ("rate_6yr", "6y"), ("rate_9yr", "9y")]:
            m, lo, hi = mean_ci95(g[col].tolist())
            row[f"{label}_mean"] = m
            row[f"{label}_ci_low"] = lo
            row[f"{label}_ci_high"] = hi
        per_run_rows.append(row)

    out["per_run_df"] = pd.DataFrame(per_run_rows).sort_values("run_id")
    return out


# ─────────────────────────────────────────────
# Plot: average with CI + BJS targets
# ─────────────────────────────────────────────

# Expects a dict like:
# overall_stats["3-Year"] = {"mean": 0.666, "ci_low": 0.650, "ci_high": 0.680, "n": 200}
# and BJS_TARGETS = {"3-Year": 0.68, "6-Year": 0.79, "9-Year": 0.83}

def plot_overall_with_ci_grouped_bars(
    overall_stats: Dict,
    out_png: str = "recidivism_vs_bjs_grouped.png",
    title: str = "ABM vs BJS Recidivism",
    show_ci: bool = True,
):
    x = ["3-Year", "6-Year", "9-Year"]

    # --- ABM stats ---
    means = [overall_stats[k]["mean"] for k in x]
    lows  = [overall_stats[k]["ci_low"] for k in x]
    highs = [overall_stats[k]["ci_high"] for k in x]

    # Error bars (ABM only)
    err_plus  = [h - m for m, h in zip(means, highs)]
    err_minus = [m - l for m, l in zip(means, lows)]

    # --- BJS targets ---
    bjs = [BJS_TARGETS[k] for k in x]

    # Bar labels
    abm_text = [f"{m*100:.1f}%" for m in means]
    bjs_text = [f"{v*100:.1f}%" for v in bjs]

    fig = go.Figure()

    # --- ABM bars (blue) ---
    fig.add_trace(go.Bar(
        x=x,
        y=means,
        name="ABM mean",
        marker_color="#2F6FB2",  # same academic blue
        text=abm_text,
        textposition="outside",
        cliponaxis=False,
        error_y=dict(
            type="data",
            symmetric=False,
            array=err_plus if show_ci else None,
            arrayminus=err_minus if show_ci else None,
            color="#2F6FB2",
            thickness=1.5
        ) if show_ci else None,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "ABM mean: %{y:.3f}<br>"
            + ("95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>" if show_ci else "")
            + "n = %{customdata[2]}<extra></extra>"
        ),
        customdata=[[lo, hi, overall_stats[k].get("n", None)] for k, lo, hi in zip(x, lows, highs)],
    ))

    # --- BJS bars (gray) ---
    fig.add_trace(go.Bar(
        x=x,
        y=bjs,
        name="BJS",
        marker_color="#A6A6A6",  # neutral gray like your screenshot
        text=bjs_text,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>BJS: %{y:.3f}<extra></extra>",
    ))

    # Title (optionally include subtitle line like your example)
    final_title = title

    # --- Layout to match your screenshot style ---
    y_max = max(max(means), max(bjs))
    # ── Annotate 95% CI on ABM bars (subtle, non-cluttered) ──
    for xi, m, lo, hi in zip(x, means, lows, highs):
        fig.add_annotation(
            x=xi,
            y=m,
            text=f"[{lo*100:.1f}%, {hi*100:.1f}%]",
            showarrow=False,
            yshift=-18,                     # below the % label
            font=dict(
                size=11,
                color="#1F3A5F"             # same navy as axes
            ),
            align="center"
        )

    fig.update_layout(
        title=final_title,
        barmode="group",
        bargap=0.35,
        bargroupgap=0.10,
        template="plotly_white",
        font=dict(color="#1F3A5F"),
        yaxis_title="Rearrest rate",
        yaxis=dict(
            tickformat=".0%",
            range=[0, max(1.0, y_max * 1.15)],
            showgrid=True,
            gridcolor="#D6E4F0",
            linecolor="#1F3A5F",
            zeroline=False,
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#1F3A5F",
        ),
        legend=dict(
            x=1.02, y=1.0,
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(t=70, b=45, l=70, r=120),
    )

    # Optional: export
    try:
        fig.write_image(out_png, scale=2)
        print(f"\n📁 Chart saved as '{out_png}'")
    except Exception:
        print("⚠️ Install kaleido to enable PNG export: pip install -U kaleido")

    fig.show()




def print_summary(overall_stats: Dict):
    print("\n📊 Overall Mean Recidivism Rates (All sims) with 95% CI:")
    for label in ["3-Year", "6-Year", "9-Year"]:
        m = overall_stats[label]["mean"]
        lo = overall_stats[label]["ci_low"]
        hi = overall_stats[label]["ci_high"]
        n = overall_stats[label]["n"]
        print(f"{label}: mean={m:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]  (n={n})")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    N_RUNS = 10
    SEEDS_PER_RUN = 20
    BASE_SEED = 1000

    tasks = build_tasks(n_runs=N_RUNS, seeds_per_run=SEEDS_PER_RUN, base_seed=BASE_SEED)

    df_results = run_tasks_parallel(
        tasks,
        max_workers=min(len(tasks), os.cpu_count() or 1)
    )

    # Persist raw results
    df_results.to_csv("recidivism_seed_runs_results.csv", index=False)
    print("\n📁 Raw results saved to 'recidivism_seed_runs_results.csv'")

    summary = summarize_with_ci(df_results)
    print_summary(summary["overall"])

    # Optional: per-run table
    per_run_df = summary["per_run_df"]
    per_run_df.to_csv("recidivism_per_run_summary.csv", index=False)
    print("📁 Per-run summary saved to 'recidivism_per_run_summary.csv'")

    #plot_overall_with_ci(summary["overall"])
    plot_overall_with_ci_grouped_bars(
        overall_stats = summary["overall"],
        out_png="abm_vs_bjs.png",
        title="ABM vs BJS Recidivism",
        show_ci=True  # set True if you want ABM CI whiskers
    )
