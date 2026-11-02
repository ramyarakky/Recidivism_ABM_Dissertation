"""
plot_pcra_tier_charts.py
========================
Generates PCRA tier-stratified rearrest rate charts directly from
ABM simulation results — matching Federal Probation 87(2) Figure 1 style.

OUTPUT FILES
------------
    baseline_output/pcra_tier_3yr.png        ← 3-year window
    baseline_output/pcra_tier_6yr.png        ← 6-year window
    baseline_output/pcra_tier_9yr.png        ← 9-year window
    baseline_output/pcra_tier_combined.png   ← all three side by side
    baseline_output/pcra_tier_stats.csv      ← rearrest rates per tier
    baseline_output/pcra_tier_weights.csv    ← tier distribution (for pcra_to_bjs.py)

HOW TO INTEGRATE INTO YOUR EXISTING BASELINE RUNNER
----------------------------------------------------
Option A — add to plot_all() in your runner:

    from plot_pcra_tier_charts import (
        run_model_for_seed_with_agents,
        plot_pcra_tier_breakdown,
        compute_tier_distribution,
    )
    # Use run_model_for_seed_with_agents instead of run_model_for_seed
    # Then at the end of plot_all():
    plot_pcra_tier_breakdown(results, out_dir=OUT_DIR,
                             n_runs=N_RUNS, seeds_per_run=SEEDS_PER_RUN)
    tier_weights = compute_tier_distribution(results, out_dir=OUT_DIR)

Option B — run standalone:
    python plot_pcra_tier_charts.py

PCRA EMPIRICAL TARGETS
----------------------
  3-year : Federal Probation 87(2), Table 6 — direct values
  6-year : Extrapolated from BJS 6yr aggregate (79%), maintaining tier ratios
  9-year : Extrapolated from BJS 9yr aggregate (83%), maintaining tier ratios
"""

import os
import sys
import math
import random
import time
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_global_calibration_params, get_uncalibrated_params
from recidivism_abm.config.pcra_to_bjs import get_pcra_tier_targets

# ─────────────────────────────────────────────────────────────────────────────
# PCRA EMPIRICAL TARGETS
# Federal Probation 87(2), Table 6 (3yr direct; 6yr/9yr extrapolated)
PCRA_TARGETS = get_pcra_tier_targets()

TIER_ORDER  = ["Low", "LowModerate", "Moderate", "High"]
TIER_LABELS = {
    "Low":         "Low\nRisk",
    "LowModerate": "Low-Moderate\nRisk",
    "Moderate":    "Moderate\nRisk",
    "High":        "High\nRisk",
}

# Progressively darker gray per tier — matching the paper exactly
TIER_GRAYS = {
    "Low":         "#CECECE",
    "LowModerate": "#B5B5B5",
    "Moderate":    "#9C9C9C",
    "High":        "#838383",
}


# ─────────────────────────────────────────────────────────────────────────────
# BASE PARAMS  — mirror your existing runner
# ─────────────────────────────────────────────────────────────────────────────
BASE_PARAMS = {
    "initial_agents":        1000,
    "warmup_months":         144,
    "study_months":          108,
    "monthly_intake":        10,
    "bias_factor":           0,
    "enable_peer_influence": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED WORKER
# Returns aggregate rates + per-agent (tier, rearrest flags)
# ─────────────────────────────────────────────────────────────────────────────

def run_model_for_seed_with_agents(task: Dict) -> Dict:
    """
    Same as run_model_for_seed() but also collects per-agent PCRA tier
    and rearrest flags for every study-eligible agent.

    Returns
    -------
    dict: run_id, seed, rate_3yr, rate_6yr, rate_9yr, runtime_sec,
          agent_records — list of {tier, r3, r6, r9}
    """
    run_id = int(task["run_id"])
    seed   = int(task["seed"])

    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass

    t0     = time.perf_counter()
    params = dict(BASE_PARAMS)
    params["seed"]               = seed
    params["calibration_params"] = task["calibration_params"]

    model = RecidivismModel(**params)
    model.export_csv = False
    model.max_months = model.warmup_months + model.study_months

    try:
        while model.running:
            model.step()
    except Exception as e:
        print(f"⚠️  Worker error run={run_id} seed={seed}: {e}")

    elapsed = time.perf_counter() - t0

    def safe_rate(flag):
        r = model.calculate_flag_rate(flag)
        return r if r is not None else 0.0

    # ── Per-agent tier records ────────────────────────────────────────────────
    agent_records = []
    for agent in model.schedule.agents:
        if not getattr(agent, "study_eligible_agent", False):
            continue
        try:
            tier = agent.get_pcra_tier()
        except Exception:
            tier = "Unknown"
        agent_records.append({
            "tier": tier,
            "r3":   int(bool(getattr(agent, "rearrest_3_yrs", False))),
            "r6":   int(bool(getattr(agent, "rearrest_6_yrs", False))),
            "r9":   int(bool(getattr(agent, "rearrest_9_yrs", False))),
        })

    return {
        "run_id":        run_id,
        "seed":          seed,
        "rate_3yr":      safe_rate("rearrest_3_yrs"),
        "rate_6yr":      safe_rate("rearrest_6_yrs"),
        "rate_9yr":      safe_rate("rearrest_9_yrs"),
        "runtime_sec":   elapsed,
        "agent_records": agent_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TIER DISTRIBUTION
# Computes the proportion of study-eligible agents in each PCRA tier.
# This is the empirical TIER_WEIGHTS needed by pcra_to_bjs.py Step 4.
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier_distribution(
    results: List[Dict],
    out_dir: str = None,
) -> Dict[str, float]:
    """
    Compute the actual PCRA tier distribution from ABM study-period agents.
    Counts every study-eligible agent across all runs and seeds.

    Parameters
    ----------
    results : list of dicts returned by run_model_for_seed_with_agents
    out_dir : if provided, saves tier_weights.csv and prints paste-ready block

    Returns
    -------
    dict: {tier: proportion}
        e.g. {"Low": 0.163, "LowModerate": 0.192, "Moderate": 0.246, "High": 0.399}
        Proportions sum to 1.0 (rounded to 3dp, may differ by ≤0.001 due to rounding).

    Usage
    -----
    After running plot_pcra_tier_charts.py, paste the printed TIER_WEIGHTS
    block directly into pcra_to_bjs.py to replace the undocumented defaults.
    """
    tier_counts = {t: 0 for t in TIER_ORDER}
    unknown     = 0
    total       = 0

    for res in results:
        for rec in res.get("agent_records", []):
            t = rec["tier"]
            if t in tier_counts:
                tier_counts[t] += 1
                total += 1
            else:
                unknown += 1

    if total == 0:
        print("⚠️  No agent records found — did you use run_model_for_seed_with_agents?")
        return {t: 0.0 for t in TIER_ORDER}

    weights = {t: round(tier_counts[t] / total, 3) for t in TIER_ORDER}

    # ── Print diagnostic ──────────────────────────────────────────────────────
    n_runs = len(results)
    print(f"\n── ABM Study-Period Tier Distribution ──────────────────────────────")
    print(f"   Total agents : {total:,}  across {n_runs} runs")
    if unknown > 0:
        print(f"   Unknown tier : {unknown:,} agents excluded")
    print(f"   {'Tier':15s}  {'Count':>8s}  {'Proportion':>10s}")
    print("   " + "─" * 38)
    for tier in TIER_ORDER:
        print(f"   {tier:15s}  {tier_counts[tier]:>8,}  {weights[tier]:>10.3f}")
    print(f"   {'TOTAL':15s}  {total:>8,}  {'1.000':>10s}")

    # ── PCRA reference for comparison ─────────────────────────────────────────
    pcra_ref = {"Low": 0.292, "LowModerate": 0.356, "Moderate": 0.239, "High": 0.096}
    print(f"\n   Comparison — PCRA federal supervision population")
    print(f"   (Johnson 2023, Federal Probation 87(2), Table 1):")
    print(f"   {'Tier':15s}  {'ABM':>8s}  {'PCRA ref':>10s}  {'Diff':>8s}")
    print("   " + "─" * 46)
    for tier in TIER_ORDER:
        diff = weights[tier] - pcra_ref[tier]
        print(f"   {tier:15s}  {weights[tier]:>8.3f}  "
              f"{pcra_ref[tier]:>10.3f}  {diff:>+8.3f}")

    # ── Paste-ready block for pcra_to_bjs.py ─────────────────────────────────
    print(f"\n   Paste into pcra_to_bjs.py as TIER_WEIGHTS:")
    print(f"   # Source: compute_tier_distribution(), {n_runs} runs, "
          f"{total:,} agents")
    print(f"   TIER_WEIGHTS = {{")
    for tier in TIER_ORDER:
        print(f'       "{tier}": {weights[tier]},')
    print(f"   }}")

    # ── Save to CSV if out_dir provided ───────────────────────────────────────
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        rows = [
            {"tier": t, "count": tier_counts[t],
             "proportion": weights[t], "pcra_ref": pcra_ref[t],
             "diff_from_pcra": round(weights[t] - pcra_ref[t], 3)}
            for t in TIER_ORDER
        ]
        csv_path = os.path.join(out_dir, "pcra_tier_weights.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n   📁 Tier weights → {csv_path}")

        # Also save as JSON for easy import into pcra_to_bjs.py
        json_path = os.path.join(out_dir, "pcra_tier_weights.json")
        with open(json_path, "w") as f:
            json.dump({
                "tier_weights":    weights,
                "tier_counts":     tier_counts,
                "total_agents":    total,
                "n_runs":          n_runs,
                "pcra_reference":  pcra_ref,
            }, f, indent=2)
        print(f"   📁 Tier weights JSON → {json_path}")

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# TIER STATS  —  run-level means → grand mean + 95% CI
# ─────────────────────────────────────────────────────────────────────────────

def _mean_ci95(values: List[float]) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        m = float(values[0])
        return m, m, m
    m   = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    se  = math.sqrt(var) / math.sqrt(n)
    t   = 2.045 if n >= 30 else (2.228 if n >= 10 else 2.776)
    return m, m - t * se, m + t * se


def compute_tier_stats(results: List[Dict]) -> pd.DataFrame:
    """
    Aggregates per-agent records across all runs.
    CI is computed over run-level means (correct sampling unit).

    Returns DataFrame with columns:
        tier, n_agents,
        rate_3yr, ci_lo_3, ci_hi_3,
        rate_6yr, ci_lo_6, ci_hi_6,
        rate_9yr, ci_lo_9, ci_hi_9
    """
    run_means: Dict[str, Dict[int, List[float]]] = {
        t: {3: [], 6: [], 9: []} for t in TIER_ORDER
    }
    tier_n: Dict[str, int] = {t: 0 for t in TIER_ORDER}

    for res in results:
        records = res.get("agent_records", [])
        if not records:
            continue

        counts = {t: {"n": 0, "r3": 0, "r6": 0, "r9": 0} for t in TIER_ORDER}
        for rec in records:
            t = rec["tier"]
            if t not in counts:
                continue
            counts[t]["n"]  += 1
            counts[t]["r3"] += rec["r3"]
            counts[t]["r6"] += rec["r6"]
            counts[t]["r9"] += rec["r9"]

        for t in TIER_ORDER:
            n = counts[t]["n"]
            if n == 0:
                continue
            tier_n[t] += n
            run_means[t][3].append(counts[t]["r3"] / n)
            run_means[t][6].append(counts[t]["r6"] / n)
            run_means[t][9].append(counts[t]["r9"] / n)

    rows = []
    for t in TIER_ORDER:
        row = {"tier": t, "n_agents": tier_n[t]}
        for w in [3, 6, 9]:
            vals = run_means[t][w]
            if not vals:
                row[f"rate_{w}yr"] = float("nan")
                row[f"ci_lo_{w}"]  = float("nan")
                row[f"ci_hi_{w}"]  = float("nan")
            else:
                m, lo, hi = _mean_ci95(vals)
                row[f"rate_{w}yr"] = m
                row[f"ci_lo_{w}"]  = lo
                row[f"ci_hi_{w}"]  = hi
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE WINDOW CHART  —  paper style
# ─────────────────────────────────────────────────────────────────────────────

def _plot_one_window(
    ax: plt.Axes,
    window: int,
    df: pd.DataFrame,
    n_runs: int,
    seeds_per_run: int,
) -> None:
    """
    Draw one bar chart on ax matching Federal Probation Figure 1:
      - Gray bars (progressively darker Low → High)
      - 95% CI error bars
      - Bold rate label above each bar
      - Diamond markers for PCRA empirical targets
      - No y-axis, single bottom spine
    """
    x      = np.arange(len(TIER_ORDER))
    rates  = [df.loc[df.tier == t, f"rate_{window}yr"].values[0] for t in TIER_ORDER]
    ci_lo  = [df.loc[df.tier == t, f"ci_lo_{window}"].values[0]  for t in TIER_ORDER]
    ci_hi  = [df.loc[df.tier == t, f"ci_hi_{window}"].values[0]  for t in TIER_ORDER]
    pcra   = [PCRA_TARGETS[t][window] for t in TIER_ORDER]

    err_lo = [r - lo if not math.isnan(r) else 0 for r, lo in zip(rates, ci_lo)]
    err_hi = [hi - r  if not math.isnan(r) else 0 for r, hi in zip(rates, ci_hi)]

    bars = ax.bar(
        x, rates,
        width=0.55,
        color=[TIER_GRAYS[t] for t in TIER_ORDER],
        edgecolor="none",
        zorder=3,
        yerr=[err_lo, err_hi],
        error_kw=dict(
            ecolor="#444444", elinewidth=1.6,
            capsize=6, capthick=1.6, zorder=5,
        ),
    )

    for bar, rate, hi in zip(bars, rates, ci_hi):
        if math.isnan(rate):
            continue
        label_y = (hi if not math.isnan(hi) else rate) + 0.012
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{rate * 100:.1f}%",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color="#222222",
        )

    ax.scatter(
        x, pcra,
        marker="D", s=60,
        color="#1A1A1A",
        zorder=6,
        label="PCRA empirical\n(Fed. Prob. 87(2))",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TIER_LABELS[t] for t in TIER_ORDER],
        fontsize=11, color="#333333",
    )
    ax.tick_params(axis="x", length=0, pad=10)

    ax.yaxis.set_visible(False)
    valid_rates = [r for r in rates if not math.isnan(r)]
    valid_ci_hi = [h for h in ci_hi if not math.isnan(h)]
    top = max(valid_ci_hi + valid_rates + pcra, default=0.8)
    ax.set_ylim(0, top * 1.38)

    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(1.2)

    ax.set_title(
        f"PCRA Risk Category\n{window}-Year Cumulative Rearrest Rate",
        fontsize=14, fontweight="bold",
        color="#1A1A1A", pad=16,
    )

    ax.legend(
        fontsize=8.5, loc="upper left",
        frameon=False, handletextpad=0.4,
    )

    ax.text(
        0.5, -0.17,
        f"ABM: {n_runs} runs × {seeds_per_run} seeds  |  "
        "Error bars = 95% CI  |  ◆ = PCRA empirical target",
        transform=ax.transAxes,
        ha="center", fontsize=8, color="#999999", style="italic",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLOT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def plot_pcra_tier_breakdown(
    results: List[Dict],
    out_dir: str = "baseline_output",
    n_runs: int = 10,
    seeds_per_run: int = 20,
    out_png: str = None,
) -> None:
    """
    Generates four chart files from ABM simulation results:
        pcra_tier_3yr.png
        pcra_tier_6yr.png
        pcra_tier_9yr.png
        pcra_tier_combined.png
        pcra_tier_stats.csv
    """
    if out_png is not None and out_dir == "baseline_output":
        out_dir = os.path.dirname(out_png) or "baseline_output"
    os.makedirs(out_dir, exist_ok=True)

    df = compute_tier_stats(results)
    csv_path = os.path.join(out_dir, "pcra_tier_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n   📁 Tier stats → {csv_path}")
    print(df.to_string(index=False))

    for window in [3, 6, 9]:
        fig, ax = plt.subplots(figsize=(6.5, 5.6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        _plot_one_window(ax, window, df, n_runs, seeds_per_run)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        path = os.path.join(out_dir, f"pcra_tier_{window}yr.png")
        plt.savefig(path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"   📁 {path}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    fig.patch.set_facecolor("white")
    for ax, window in zip(axes, [3, 6, 9]):
        ax.set_facecolor("white")
        _plot_one_window(ax, window, df, n_runs, seeds_per_run)
    fig.suptitle(
        "PCRA Risk Category — Cumulative Rearrest Rate by Follow-up Window\n"
        "ABM Simulation Results (bars ± 95% CI)  vs  "
        "Federal Probation 87(2) Empirical Targets (◆)",
        fontsize=13, fontweight="bold",
        color="#1A2B3C", y=1.04,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    combined = os.path.join(out_dir, "pcra_tier_combined.png")
    plt.savefig(combined, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"   📁 {combined}")


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_and_plot_tier_analysis(
    n_runs: int = 10,
    seeds_per_run: int = 20,
    base_seed: int = 1000,
    out_dir: str = "baseline_output",
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Builds tasks, runs simulations in parallel, generates all tier charts,
    and computes the empirical tier distribution for pcra_to_bjs.py.

    Returns the tier stats DataFrame.
    Saves pcra_tier_weights.csv and pcra_tier_weights.json to out_dir.
    """
    #cal = get_uncalibrated_params()
    cal = get_global_calibration_params()

    tasks: List[Dict] = []
    for run_id in range(1, n_runs + 1):
        for i in range(seeds_per_run):
            seed = base_seed + run_id * 10_000 + i
            tasks.append({
                "run_id":             run_id,
                "seed":               seed,
                "calibration_params": cal,
            })

    if max_workers is None:
        logical     = os.cpu_count() or 4
        max_workers = max(1, logical // 2 - 1)

    n_total = len(tasks)
    print(f"\n🚀 PCRA tier analysis — {n_total} simulations  ({max_workers} workers)")
    print(f"   {n_runs} runs × {seeds_per_run} seeds = {n_total} total\n")

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_model_for_seed_with_agents, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(
                    f"  ✅ Run {r['run_id']:02d} | Seed {r['seed']:6d} | "
                    f"3y={r['rate_3yr']:.3f}  6y={r['rate_6yr']:.3f}  "
                    f"9y={r['rate_9yr']:.3f} | "
                    f"{len(r['agent_records'])} agents | "
                    f"{r['runtime_sec']:.1f}s"
                )
            except Exception as e:
                print(f"  ❌ run={t['run_id']} seed={t['seed']}: {e}")

    # ── Charts ────────────────────────────────────────────────────────────────
    plot_pcra_tier_breakdown(
        results,
        out_dir=out_dir,
        n_runs=n_runs,
        seeds_per_run=seeds_per_run,
    )

    # ── Tier distribution — paste output into pcra_to_bjs.py ─────────────────
    compute_tier_distribution(results, out_dir=out_dir)

    return compute_tier_stats(results)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    df_tier = run_and_plot_tier_analysis(
        n_runs        = 10,
        seeds_per_run = 20,
        base_seed     = 1000,
        out_dir       = "baseline_output",
    )

    print("\n📊 Final Tier Stats:")
    print(df_tier.to_string(index=False))
    print("\n✅ Done — outputs saved to baseline_output/")
    print("   pcra_tier_3yr.png")
    print("   pcra_tier_6yr.png")
    print("   pcra_tier_9yr.png")
    print("   pcra_tier_combined.png")
    print("   pcra_tier_stats.csv")
    print("   pcra_tier_weights.csv   ← paste TIER_WEIGHTS from this into pcra_to_bjs.py")
    print("   pcra_tier_weights.json  ← machine-readable version")