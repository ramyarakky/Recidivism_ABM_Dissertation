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
import json
import pandas as pd
import plotly.graph_objects as go

# Ensure project-root imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_uncalibrated_params


# ─────────────────────────────────────────────
# Simulation Parameters (excluding seed)
# ─────────────────────────────────────────────
BASE_PARAMS = {
    "initial_agents": 1000,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0,
    "enable_peer_influence": True,
}

# BJS targets for plotting reference
BJS_TARGETS = {"3-Year": 0.684, "6-Year": 0.794, "9-Year": 0.834}


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
# ─────────────────────────────────────────────────────────────────────────────
# Worker Cap
# ─────────────────────────────────────────────────────────────────────────────

def detect_safe_workers(n_tasks: int) -> int:
    logical  = os.cpu_count() or 4
    physical = max(1, logical // 2)
    safe     = max(1, physical - 2)
    return min(n_tasks, safe)
    
def run_model_for_seed(task: Dict) -> Dict:
    """
    Runs one simulation for the given seed and returns rearrest rates.
    task expects:
      - run_id
      - seed
      - calibration_params (dict): pass get_uncalibrated_params() or get_global_calibration_params()
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
    params["calibration_params"] = task["calibration_params"]

    model = RecidivismModel(**params)
    model.export_csv=False

    model.max_months = model.warmup_months + model.study_months

    try:
        while model.running:
            model.step()
    except Exception as e:
        print(f"⚠️  Model error run_id={run_id} seed={seed}: {e}")

    # Evaluate consistently at end of study
    #model.current_month = model.warmup_months + model.study_months

    elapsed = time.perf_counter() - start_time

    # Safe rate extraction — returns 0.0 if flag rate is None
    def safe_rate(flag_name):
        r = model.calculate_flag_rate(flag_name)
        return r if r is not None else 0.0

    r3 = safe_rate("rearrest_3_yrs")
    r6 = safe_rate("rearrest_6_yrs")
    r9 = safe_rate("rearrest_9_yrs")

    return {
        "run_id": run_id,
        "seed": seed,
        "rate_3yr": r3,
        "rate_6yr": r6,
        "rate_9yr": r9,
        "runtime_sec": elapsed,
    }


# ─────────────────────────────────────────────
# Run many seeds across many runs in parallel
# ─────────────────────────────────────────────
def build_tasks(n_runs: int, seeds_per_run: int, base_seed: int = 1000,
                calibration_params: Dict = None) -> List[Dict]:
    """
    Build a deterministic list of (run_id, seed) tasks.
    Each run gets its own block of seeds: base_seed + run_id*10000 + i

    calibration_params: pass get_uncalibrated_params() or get_global_calibration_params().
    Defaults to calibrated if not specified.
    """
    cal = calibration_params if calibration_params is not None else get_uncalibrated_params() #get_global_calibration_params()
    tasks: List[Dict] = []
    for run_id in range(1, n_runs + 1):
        for i in range(seeds_per_run):
            seed = base_seed + (run_id * 10_000) + i
            tasks.append({"run_id": run_id, "seed": seed, "calibration_params": cal})
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
):  # kept for backward-compat — calls plot_calibration_bars internally
    plot_calibration_bars(overall_stats, out_png)


# ─────────────────────────────────────────────────────────────────────────────
# 5-Chart Visualization Suite
# ─────────────────────────────────────────────────────────────────────────────

_C_ABM  = "#2F6FB2"
_C_BJS  = "#E07B39"
_C_GRID = "#E8EDF2"
_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Georgia, serif", color="#1A2B3C", size=13),
    paper_bgcolor="#FAFBFC",
    plot_bgcolor="#FAFBFC",
)
_WINDOW_STYLES = [
    ("rate_3yr", "3-Year", "#2F6FB2"),
    ("rate_6yr", "6-Year", "#27AE60"),
    ("rate_9yr", "9-Year", "#8E44AD"),
]


def _save(fig: go.Figure, path: str) -> None:
    try:
        fig.write_image(path, scale=2)
        print(f"   📁 {path}")
    except Exception:
        print(f"   ⚠️  kaleido not installed — skipping PNG: {path}")
    fig.show()


def plot_calibration_bars(
    overall_stats: Dict,
    out_png: str = "chart1_calibration.png",
) -> None:
    """Chart 1 — Grouped bars: ABM mean ±95% CI vs BJS target, Δ annotated."""
    windows = ["3-Year", "6-Year", "9-Year"]
    means   = [overall_stats[w]["mean"]    for w in windows]
    lows    = [overall_stats[w]["ci_low"]  for w in windows]
    highs   = [overall_stats[w]["ci_high"] for w in windows]
    bjs     = [BJS_TARGETS[w]             for w in windows]
    ns      = [overall_stats[w]["n"]       for w in windows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ABM mean", x=windows, y=means,
        marker_color=_C_ABM, marker_line_width=0,
        error_y=dict(type="data", symmetric=False,
                     array=[h-m for m,h in zip(means,highs)],
                     arrayminus=[m-l for m,l in zip(means,lows)],
                     color=_C_ABM, thickness=2, width=6),
        text=[f"{m*100:.1f}%" for m in means], textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>ABM: %{y:.3f}<br>"
            "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
            "n=%{customdata[2]}<extra></extra>"),
        customdata=[[lo,hi,n] for lo,hi,n in zip(lows,highs,ns)],
    ))
    fig.add_trace(go.Bar(
        name="BJS target", x=windows, y=bjs,
        marker_color=_C_BJS, marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in bjs], textposition="outside",
        hovertemplate="<b>%{x}</b><br>BJS: %{y:.3f}<extra></extra>",
    ))
    for w, m, b in zip(windows, means, bjs):
        diff   = m - b
        colour = "#C0392B" if abs(diff) > 0.03 else "#27AE60"
        fig.add_annotation(x=w, y=max(m,b)+0.07,
            text=f"Δ {diff:+.3f}", showarrow=False,
            font=dict(size=11, color=colour, family="Georgia, serif"))
    fig.update_layout(**_LAYOUT,
        title=dict(text="<b>Chart 1 — Recidivism ABM Results vs BJS Targets</b>",
                   font=dict(size=16), x=0.5),
        barmode="group", bargap=0.30, bargroupgap=0.08,
        yaxis=dict(title="Cumulative rearrest rate", tickformat=".0%",
                   range=[0, 1.05], gridcolor=_C_GRID, zeroline=False),
        xaxis=dict(showgrid=False),
        legend=dict(x=0.82, y=0.98),
        margin=dict(t=80, b=50, l=70, r=40),
    )
    _save(fig, out_png)


def plot_mae_bars(
    overall_stats: Dict,
    out_png: str = "chart2_mae.png",
) -> None:
    """Chart 2 — Horizontal MAE bars with colour-coded verdict tiers."""
    windows = ["3-Year", "6-Year", "9-Year"]
    maes    = [abs(overall_stats[w]["mean"] - BJS_TARGETS[w]) for w in windows]
    colours, verdicts = [], []
    for e in maes:
        if   e < 0.02: colours.append("#27AE60"); verdicts.append("Excellent  (<2%)")
        elif e < 0.05: colours.append("#F39C12"); verdicts.append("Acceptable (2–5%)")
        elif e < 0.10: colours.append("#E67E22"); verdicts.append("Marginal   (5–10%)")
        else:          colours.append("#C0392B"); verdicts.append("Poor       (>10%)")

    fig = go.Figure(go.Bar(
        x=maes, y=windows, orientation="h",
        marker_color=colours, marker_line_width=0,
        text=[f"{e*100:.2f}%  — {v}" for e,v in zip(maes,verdicts)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>MAE: %{x:.4f}<extra></extra>",
    ))
    for thresh, label, col in [(0.02,"Excellent","#27AE60"),(0.05,"Acceptable","#F39C12")]:
        fig.add_vline(x=thresh, line_dash="dot", line_color=col, line_width=1.5,
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(size=10, color=col))
    fig.update_layout(**_LAYOUT,
        title=dict(text="<b>Chart 2 — Mean Absolute Error vs BJS Benchmarks</b>",
                   font=dict(size=16), x=0.5),
        xaxis=dict(title="|ABM − BJS|", tickformat=".1%",
                   range=[0, max(maes)*1.6+0.01],
                   gridcolor=_C_GRID, zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(t=80, b=50, l=90, r=220),
    )
    _save(fig, out_png)


def plot_per_run_variance(
    df: pd.DataFrame,
    overall_stats: Dict,
    out_png: str = "chart3_variance.png",
) -> None:
    """Chart 3 — Strip plot: one dot per seed, mean ± CI band, BJS dotted line."""
    offsets = {"rate_3yr": -0.25, "rate_6yr": 0.0, "rate_9yr": 0.25}
    fig     = go.Figure()
    for col, label, colour in _WINDOW_STYLES:
        xs = [r["run_id"] + offsets[col] for _, r in df.iterrows()]
        fig.add_trace(go.Scatter(
            x=xs, y=df[col].tolist(), mode="markers", name=label,
            marker=dict(color=colour, size=5, opacity=0.55),
            hovertemplate=f"<b>{label}</b><br>Run %{{x:.0f}}<br>Rate: %{{y:.3f}}<extra></extra>",
        ))
        run_ids = sorted(df["run_id"].unique())
        x0, x1 = min(run_ids)-0.5, max(run_ids)+0.5
        m, lo, hi = (overall_stats[label]["mean"],
                     overall_stats[label]["ci_low"],
                     overall_stats[label]["ci_high"])
        fig.add_shape(type="line",  x0=x0, x1=x1, y0=m,  y1=m,
                      line=dict(color=colour, width=2))
        fig.add_shape(type="rect",  x0=x0, x1=x1, y0=lo, y1=hi,
                      fillcolor=colour, opacity=0.10, line_width=0)
        fig.add_shape(type="line",  x0=x0, x1=x1,
                      y0=BJS_TARGETS[label], y1=BJS_TARGETS[label],
                      line=dict(color=colour, width=1.5, dash="dot"))
    fig.update_layout(**_LAYOUT,
        title=dict(
            text="<b>Chart 3 — Simulation Variance: Each Dot = One Seed</b><br>"
                 "<sup>Solid = overall mean | Band = 95% CI | Dotted = BJS target</sup>",
            font=dict(size=15), x=0.5),
        xaxis=dict(title="Run ID", dtick=1, showgrid=False),
        yaxis=dict(title="Cumulative rearrest rate", tickformat=".0%",
                   range=[0.40, 1.0], gridcolor=_C_GRID, zeroline=False),
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=100, b=60, l=70, r=40),
    )
    _save(fig, out_png)


def plot_per_run_convergence(
    per_run_df: pd.DataFrame,
    overall_stats: Dict,
    out_png: str = "chart4_convergence.png",
) -> None:
    """Chart 4 — Per-run mean ± 95% CI connected lines (stability check)."""
    fig     = go.Figure()
    run_ids = per_run_df["run_id"].tolist()
    for prefix, label, colour in [("3y","3-Year",_C_ABM),
                                   ("6y","6-Year","#27AE60"),
                                   ("9y","9-Year","#8E44AD")]:
        means = per_run_df[f"{prefix}_mean"].tolist()
        lows  = per_run_df[f"{prefix}_ci_low"].tolist()
        highs = per_run_df[f"{prefix}_ci_high"].tolist()
        fig.add_trace(go.Scatter(
            x=run_ids, y=means, mode="lines+markers", name=label,
            line=dict(color=colour, width=2), marker=dict(color=colour, size=7),
            error_y=dict(type="data", symmetric=False,
                         array=[h-m for m,h in zip(means,highs)],
                         arrayminus=[m-l for m,l in zip(means,lows)],
                         color=colour, thickness=1.2, width=4),
            hovertemplate=(
                f"<b>{label} — Run %{{x}}</b><br>Mean: %{{y:.3f}}<br>"
                "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"),
            customdata=[[lo,hi] for lo,hi in zip(lows,highs)],
        ))
        bjs = BJS_TARGETS[label]
        fig.add_shape(type="line",
            x0=min(run_ids)-0.4, x1=max(run_ids)+0.4, y0=bjs, y1=bjs,
            line=dict(color=colour, width=1.2, dash="dot"))
        fig.add_annotation(x=max(run_ids)+0.5, y=bjs,
            text=f"BJS {bjs:.0%}", showarrow=False,
            font=dict(size=10, color=colour), xanchor="left")
    fig.update_layout(**_LAYOUT,
        title=dict(
            text="<b>Chart 4 — Per-Run Mean ± 95% CI (Stability Check)</b><br>"
                 "<sup>Flat lines = stable calibration | Dotted = BJS targets</sup>",
            font=dict(size=15), x=0.5),
        xaxis=dict(title="Run ID", dtick=1, showgrid=False),
        yaxis=dict(title="Cumulative rearrest rate", tickformat=".0%",
                   range=[0.50, 0.95], gridcolor=_C_GRID, zeroline=False),
        legend=dict(x=0.01, y=0.25),
        margin=dict(t=100, b=60, l=70, r=100),
    )
    _save(fig, out_png)


def plot_distribution(
    df: pd.DataFrame,
    overall_stats: Dict,
    out_png: str = "chart5_distribution.png",
) -> None:
    """Chart 5 — Violin + box showing full distribution of seed-level rates."""
    fig = go.Figure()
    for col, label, colour in _WINDOW_STYLES:
        vals = df[col].dropna().tolist()
        fig.add_trace(go.Violin(
            x=[label]*len(vals), y=vals, name=label,
            box_visible=True, meanline_visible=True,
            fillcolor=colour, opacity=0.35, line_color=colour,
            points="all", pointpos=0,
            marker=dict(size=4, opacity=0.5),
            hovertemplate="<b>%{x}</b><br>Rate: %{y:.3f}<extra></extra>",
        ))
        bjs = BJS_TARGETS[label]
        fig.add_annotation(x=label, y=bjs,
            text=f"◀ BJS {bjs:.0%}", showarrow=False,
            xshift=60, font=dict(size=10, color=colour))
    fig.update_layout(**_LAYOUT,
        title=dict(
            text="<b>Chart 5 — Distribution of Seed-Level Rearrest Rates</b><br>"
                 "<sup>Each point = one seed | Box = IQR | Centre line = mean</sup>",
            font=dict(size=15), x=0.5),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Cumulative rearrest rate", tickformat=".0%",
                   gridcolor=_C_GRID, zeroline=False),
        violinmode="overlay", showlegend=False,
        margin=dict(t=100, b=60, l=70, r=110),
    )
    _save(fig, out_png)


def plot_all(
    df_results: pd.DataFrame,
    overall_stats: Dict,
    per_run_df: pd.DataFrame,
    n_runs: int,
    seeds_per_run: int,
    out_dir: str = "baseline_output",
) -> None:
    """Render all 5 charts. Call after run_tasks_parallel() completes."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📊 Generating 5 charts ({n_runs} runs × {seeds_per_run} seeds) …")
    plot_calibration_bars(overall_stats,                os.path.join(out_dir, "chart1_calibration.png"))
    plot_mae_bars(overall_stats,                        os.path.join(out_dir, "chart2_mae.png"))
    plot_per_run_variance(df_results, overall_stats,    os.path.join(out_dir, "chart3_variance.png"))
    plot_per_run_convergence(per_run_df, overall_stats, os.path.join(out_dir, "chart4_convergence.png"))
    plot_distribution(df_results, overall_stats,        os.path.join(out_dir, "chart5_distribution.png"))
    print(f"\n✅ All 5 charts saved to {out_dir}/")


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
    N_RUNS        = 10
    SEEDS_PER_RUN = 20
    BASE_SEED     = 1000

    # ── Choose calibration mode ──────────────────────────────────────────────
    # CALIBRATED   = get_global_calibration_params()  — OAT-tuned values
    # UNCALIBRATED = get_uncalibrated_params()         — neutral baseline
    #CAL_PARAMS = get_global_calibration_params()
    CAL_PARAMS = get_uncalibrated_params()
    # ────────────────────────────────────────────────────────────────────────

    # Smoke test before full run — uncomment these two lines:
    # N_RUNS = 3 ; SEEDS_PER_RUN = 5

    tasks     = build_tasks(n_runs=N_RUNS, seeds_per_run=SEEDS_PER_RUN,
                            base_seed=BASE_SEED, calibration_params=CAL_PARAMS)
    n_workers = detect_safe_workers(len(tasks))

    print(f"💻 CPU cores (logical): {os.cpu_count()} | Safe workers: {n_workers}")
    print(f"📋 Tasks: {len(tasks)}  ({N_RUNS} runs × {SEEDS_PER_RUN} seeds)\n")

    OUT_DIR = "baseline_uncalibrated_output"
    os.makedirs(OUT_DIR, exist_ok=True)

    df_results = run_tasks_parallel(tasks, max_workers=n_workers)

    df_results.to_csv(os.path.join(OUT_DIR, "recidivism_seed_runs_results.csv"), index=False)
    print(f"\n📁 Raw results → {OUT_DIR}/recidivism_seed_runs_results.csv")

    summary    = summarize_with_ci(df_results)
    print_summary(summary["overall"])

    per_run_df = summary["per_run_df"]
    per_run_df.to_csv(os.path.join(OUT_DIR, "recidivism_per_run_summary.csv"), index=False)
    print(f"📁 Per-run summary → {OUT_DIR}/recidivism_per_run_summary.csv")

    plot_all(
        df_results    = df_results,
        overall_stats = summary["overall"],
        per_run_df    = per_run_df,
        n_runs        = N_RUNS,
        seeds_per_run = SEEDS_PER_RUN,
        out_dir       = OUT_DIR,
    )
   

    json_path = "calibration_output_0330/recommended_params.json"
    with open(json_path) as f:
        rec = json.load(f)

    # Store actual per-seed results — not placeholders
    rec["final_validation"] = {
        "all_3": df_results["rate_3yr"].dropna().tolist(),
        "all_6": df_results["rate_6yr"].dropna().tolist(),
        "all_9": df_results["rate_9yr"].dropna().tolist(),
    }
    rec["calibrated_rates"] = {
        "3": float(df_results["rate_3yr"].mean()),
        "6": float(df_results["rate_6yr"].mean()),
        "9": float(df_results["rate_9yr"].mean()),
    }
    rec["calibrated_std"] = {
        "3": float(df_results["rate_3yr"].std()),
        "6": float(df_results["rate_6yr"].std()),
        "9": float(df_results["rate_9yr"].std()),
    }

    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"✅ Updated recommended_params.json with {len(df_results)} real seed results")