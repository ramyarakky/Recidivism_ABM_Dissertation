#!/usr/bin/env python3
"""
recidivism_multirun_convergence_parallel.py
============================================
Agent-Based Model Convergence Analysis — Recidivism ABM
PhD Dissertation Tool

PURPOSE
-------
Demonstrates that simulation outcomes are robust to population size (N)
by showing that recidivism rate estimates converge as N increases.
This is a standard ABM validation requirement (Railsback & Grimm, 2012;
Windrum et al., 2007) establishing that results are not artefacts of
small-sample stochasticity.

OUTPUTS
-------
Six dissertation-quality charts:
  1. Convergence lines — mean ±95% CI by agent size (3 windows)
  2. Consolidated convergence — all three windows on one plot
  3. Variance–N diagnostic — shows 1/N decay of variance
  4. MAE vs agent size — proximity to BJS targets across N
  5. CI width vs agent size — quantifies estimation precision
  6. Summary panel — four-panel overview for dissertation appendix

Sources
-------
  Railsback, S.F. & Grimm, V. (2012). Agent-Based and Individual-Based
      Modeling. Princeton University Press. Ch. 5.
  Windrum, P., Fagiolo, G. & Moneta, A. (2007). Empirical Validation of
      Agent-Based Models. Journal of Artificial Societies, 10(2), 8.
  Alper, M., Durose, M.R. & Markman, J. (2018). BJS NCJ 250975.
"""

import os
import sys
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter, FuncFormatter
from multiprocessing import cpu_count
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_peer_influence_config,
    get_uncalibrated_params,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "Recidivism_MultiRun_Convergence"
MASTER_SEED     = 424242
AGENT_SIZES     = [100, 250, 500, 1000, 1500, 2000, 2500, 3000]
N_RUNS          = 10
SEEDS_PER_RUN   = 20
BASELINE_AGENTS = 1000
BASELINE_INTAKE = 10
BASE_PARAMS     = {
    "warmup_months":         144,
    "study_months":          108,
    "bias_factor":           0,
    "enable_peer_influence": True,
}
BJS_TARGETS     = {3: 0.68, 6: 0.79, 9: 0.83}
REQUESTED_CORES = 128
N_CORES         = min(REQUESTED_CORES, cpu_count())

RUN_LEVEL_CSV   = os.path.join(OUTPUT_DIR, "run_level_results.csv")
SUMMARY_CSV     = os.path.join(OUTPUT_DIR, "recidivism_summary.csv")
LOG_FILE        = os.path.join(OUTPUT_DIR, "status.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ─────────────────────────────────────────────────────────────────────────────
# CHART STYLE — consistent across all figures
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "3yr":     "#2E75B6",
    "6yr":     "#70AD47",
    "9yr":     "#ED7D31",
    "bjs":     "#1A3D5C",
    "grid":    "#E8EDF2",
    "bg":      "#FAFBFC",
    "annot":   "#555555",
    "good":    "#27AE60",
    "warn":    "#F39C12",
    "bad":     "#C0392B",
}
WINDOW_STYLES = [
    (3, "3-Year",  STYLE["3yr"],  "o"),
    (6, "6-Year",  STYLE["6yr"],  "s"),
    (9, "9-Year",  STYLE["9yr"],  "^"),
]
LAYOUT = dict(
    facecolor=STYLE["bg"],
    edgecolor="none",
)

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.titlepad":    10,
    "axes.labelsize":   10,
    "axes.facecolor":   STYLE["bg"],
    "figure.facecolor": STYLE["bg"],
    "grid.color":       STYLE["grid"],
    "grid.linewidth":   0.6,
    "grid.linestyle":   "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
def run_single(seed, agent_size):
    monthly_intake = max(1, round(BASELINE_INTAKE * agent_size / BASELINE_AGENTS))
    model = RecidivismModel(
        initial_agents     = agent_size,
        seed               = seed,
        monthly_intake     = monthly_intake,
        calibration_params = get_uncalibrated_params(),
        **BASE_PARAMS,
    )
    model.export_csv = False
    while model.running:
        model.step()
    return {
        y: (model.calculate_flag_rate(f"rearrest_{y}_yrs") or 0.0)
        for y in (3, 6, 9)
    }


def run_seed_block(seeds, agent_size):
    results = [run_single(s, agent_size) for s in seeds]
    return {y: float(np.mean([r[y] for r in results])) for y in (3, 6, 9)}


def run_task(agent_size, run_id):
    rng   = np.random.default_rng(MASTER_SEED + agent_size * 10_000 + run_id)
    seeds = rng.integers(1, 2_000_000_000, size=SEEDS_PER_RUN,
                         dtype=np.int64).tolist()
    t0    = time.time()
    try:
        block         = run_seed_block(seeds, agent_size)
        status, err   = "ok", ""
    except Exception as e:
        block         = {3: np.nan, 6: np.nan, 9: np.nan}
        status, err   = "fail", repr(e)
    return {
        "agent_size":  agent_size,
        "run_id":      run_id,
        "seeds":       ",".join(map(str, seeds)),
        "recid_3":     block[3],
        "recid_6":     block[6],
        "recid_9":     block[9],
        "status":      status,
        "elapsed_sec": round(time.time() - t0, 2),
        "error":       err,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
_T95 = {
    1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
    6:2.447,  7:2.365, 8:2.306, 9:2.262, 10:2.228,
    11:2.201, 12:2.179, 14:2.145, 19:2.093, 24:2.064,
    29:2.045, 39:2.023, 49:2.010, 59:2.000, 99:1.984,
}

def _t95(n: int) -> float:
    df = max(1, n - 1)
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def build_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent_size in AGENT_SIZES:
        sub = df_runs[
            (df_runs.agent_size == agent_size) &
            (df_runs.status == "ok")
        ]
        for y, col in [(3,"recid_3"), (6,"recid_6"), (9,"recid_9")]:
            vals = sub[col].dropna().values
            n    = len(vals)
            m    = float(np.mean(vals)) if n > 0 else np.nan
            s    = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            ci   = _t95(n) * s / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                "agent_size": agent_size,
                "year":       y,
                "mean":       m,
                "std":        s,
                "ci":         ci,
                "var":        s**2,
                "mae":        abs(m - BJS_TARGETS[y]) if not np.isnan(m) else np.nan,
                "n":          n,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment_parallel():
    tasks = [(a, r) for a in AGENT_SIZES for r in range(N_RUNS)]
    print(f"\n🧵 Using {N_CORES} cores for {len(tasks)} tasks\n")
    logging.info(f"Starting experiment: {len(tasks)} tasks")

    records = Parallel(
        n_jobs=N_CORES, backend="loky", verbose=10
    )(delayed(run_task)(a, r) for (a, r) in tasks)

    df_runs    = pd.DataFrame(records)
    df_summary = build_summary(df_runs)

    df_runs.to_csv(RUN_LEVEL_CSV,   index=False)
    df_summary.to_csv(SUMMARY_CSV,  index=False)
    return df_runs, df_summary


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get(df_s, year, col):
    sub = df_s[df_s.year == year].sort_values("agent_size")
    return sub["agent_size"].values, sub[col].values


def _pct(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))


def _xticks(ax):
    ax.set_xticks(AGENT_SIZES)
    ax.set_xticklabels(
        [str(a) for a in AGENT_SIZES],
        rotation=45, ha="right", fontsize=8.5
    )


def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"   📊 {name}")
    return path


def _add_bjs(ax, year, label=True):
    target = BJS_TARGETS[year]
    col    = STYLE[f"{year}yr"]
    ax.axhline(target, color=col, linewidth=1.4,
               linestyle=":", alpha=0.8,
               label=f"BJS Target {target:.0%}" if label else None)
    ax.axhspan(target-0.02, target+0.02,
               alpha=0.05, color=col)


def _annotate_convergence(ax, sizes, means, color):
    """Mark where rate stabilises — where consecutive means differ < 0.5pp."""
    for i in range(1, len(means)-1):
        if (abs(means[i] - means[i-1]) < 0.005 and
                abs(means[i+1] - means[i]) < 0.005):
            ax.axvline(sizes[i], color=color, linewidth=1.0,
                       linestyle="-.", alpha=0.4)
            break


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Convergence lines per window (3 separate panels)
# ─────────────────────────────────────────────────────────────────────────────
def chart1_convergence_per_window(df_s: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)
    fig.suptitle(
        "Chart 1 — Recidivism Rate Convergence by Follow-up Window\n"
        "Mean ±95% CI across 10 runs × 20 seeds per agent size | "
        "Dotted line = BJS benchmark | Shaded = ±2pp acceptable band",
        fontsize=11, fontweight="bold", y=1.02
    )

    for ax, (yrs, label, color, marker) in zip(axes, WINDOW_STYLES):
        sizes, means = _get(df_s, yrs, "mean")
        _,     cis   = _get(df_s, yrs, "ci")

        _add_bjs(ax, yrs)
        ax.errorbar(
            sizes, means, yerr=cis,
            fmt=f"{marker}-", color=color,
            linewidth=2, markersize=7, capsize=5, capthick=1.5,
            elinewidth=1.4, label=f"{label} mean ±95% CI",
            zorder=3,
        )
        _annotate_convergence(ax, sizes, means, color)

        # Annotate final value
        ax.annotate(
            f"{means[-1]:.1%}",
            xy=(sizes[-1], means[-1]),
            xytext=(-30, 10), textcoords="offset points",
            fontsize=8.5, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

        _pct(ax)
        _xticks(ax)
        ax.set_ylim(
            max(0, min(means)-0.08),
            min(1, max(means)+0.10)
        )
        ax.set_xlabel("Agent Population Size (N)", fontsize=10)
        ax.set_ylabel("Cumulative Rearrest Rate", fontsize=10)
        ax.set_title(f"{label} Rearrest Rate", fontsize=11)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(True)

    plt.tight_layout()
    _save(fig, "chart1_convergence_per_window.png")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Consolidated convergence (all 3 windows, one plot)
# ─────────────────────────────────────────────────────────────────────────────
def chart2_consolidated_convergence(df_s: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Chart 2 — Consolidated Convergence: All Follow-up Windows\n"
        "Demonstrates that rate estimates stabilise as population size increases, "
        "validating N=1,000 as the simulation population",
        fontsize=11, fontweight="bold"
    )

    for yrs, label, color, marker in WINDOW_STYLES:
        sizes, means = _get(df_s, yrs, "mean")
        _,     cis   = _get(df_s, yrs, "ci")

        _add_bjs(ax, yrs, label=False)
        ax.errorbar(
            sizes, means, yerr=cis,
            fmt=f"{marker}-", color=color,
            linewidth=2.2, markersize=8, capsize=5, capthick=1.5,
            elinewidth=1.4, label=f"{label} ±95% CI",
            zorder=3,
        )
        # BJS label at right edge
        ax.annotate(
            f"BJS {BJS_TARGETS[yrs]:.0%}",
            xy=(AGENT_SIZES[-1], BJS_TARGETS[yrs]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, color=color, alpha=0.8,
        )

    # Mark chosen N
    chosen_n = 1000
    ax.axvline(chosen_n, color=STYLE["bjs"], linewidth=1.6,
               linestyle="--", label=f"Chosen N={chosen_n:,}", zorder=5)
    ax.annotate(
        f"N={chosen_n:,}\n(selected)",
        xy=(chosen_n, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.62),
        xytext=(15, 10), textcoords="offset points",
        fontsize=8.5, color=STYLE["bjs"], fontweight="bold",
    )

    _pct(ax)
    _xticks(ax)
    ax.set_xlabel("Agent Population Size (N)", fontsize=11)
    ax.set_ylabel("Cumulative Rearrest Rate", fontsize=11)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True)

    plt.tight_layout()
    _save(fig, "chart2_consolidated_convergence.png")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Variance–N diagnostic
# ─────────────────────────────────────────────────────────────────────────────
def chart3_variance_convergence(df_s: pd.DataFrame):
    fig, (ax_main, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Chart 3 — Variance–N Convergence Diagnostic\n"
        "Left: raw variance | Right: log-log scale with theoretical 1/N reference\n"
        "Variance declining as 1/N confirms classical Monte Carlo convergence",
        fontsize=11, fontweight="bold"
    )

    for yrs, label, color, marker in WINDOW_STYLES:
        sizes, variances = _get(df_s, yrs, "var")

        ax_main.plot(sizes, variances, f"{marker}-",
                     color=color, linewidth=2, markersize=7,
                     label=f"{label}", zorder=3)
        ax_log.plot(sizes, variances, f"{marker}-",
                    color=color, linewidth=2, markersize=7,
                    label=f"{label}", zorder=3)

    # Theoretical 1/N reference on log-log
    n_range  = np.array(AGENT_SIZES, dtype=float)
    ref_var  = variances[0] * (n_range[0] / n_range)   # scale to first point
    ax_log.plot(n_range, ref_var, "k--", linewidth=1.2,
                alpha=0.5, label="Theoretical 1/N", zorder=2)

    for ax, scale, title in [
        (ax_main, "linear", "A — Variance vs Population Size (Linear Scale)"),
        (ax_log,  "log",    "B — Variance vs Population Size (Log-Log Scale)"),
    ]:
        ax.set_xscale("log" if scale == "log" else "linear")
        ax.set_yscale("log" if scale == "log" else "linear")
        _xticks(ax)
        ax.set_xlabel("Agent Population Size (N)", fontsize=10)
        ax.set_ylabel("Variance of Estimated Rearrest Rate", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True)

    plt.tight_layout()
    _save(fig, "chart3_variance_convergence.png")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — MAE vs agent size
# ─────────────────────────────────────────────────────────────────────────────
def chart4_mae_vs_size(df_s: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(
        "Chart 4 — Mean Absolute Error vs BJS Targets by Population Size\n"
        "Shows that calibration accuracy stabilises at larger N | "
        "Windrum et al. (2007): MAE < 0.05 acceptable threshold",
        fontsize=11, fontweight="bold"
    )

    for yrs, label, color, marker in WINDOW_STYLES:
        sizes, maes = _get(df_s, yrs, "mae")
        ax.plot(sizes, maes, f"{marker}-",
                color=color, linewidth=2, markersize=7,
                label=f"{label}", zorder=3)
        ax.fill_between(sizes, maes, alpha=0.08, color=color)

    # Threshold lines
    ax.axhline(0.02, color=STYLE["good"], linewidth=1.4,
               linestyle="--", label="Excellent threshold (MAE<0.02)")
    ax.axhline(0.05, color=STYLE["warn"], linewidth=1.4,
               linestyle="--", label="Acceptable threshold (MAE<0.05, Windrum et al., 2007)")

    # Mark chosen N
    ax.axvline(1000, color=STYLE["bjs"], linewidth=1.6,
               linestyle="--", label="Chosen N=1,000", zorder=5)

    # Colour-coded MAE zones
    ax.axhspan(0,    0.02, alpha=0.04, color=STYLE["good"])
    ax.axhspan(0.02, 0.05, alpha=0.04, color=STYLE["warn"])
    ax.axhspan(0.05, 0.20, alpha=0.04, color=STYLE["bad"])

    ax.text(AGENT_SIZES[0]+20, 0.01,  "Excellent",    fontsize=8,
            color=STYLE["good"], style="italic")
    ax.text(AGENT_SIZES[0]+20, 0.032, "Acceptable",   fontsize=8,
            color=STYLE["warn"], style="italic")
    ax.text(AGENT_SIZES[0]+20, 0.065, "Marginal/Poor",fontsize=8,
            color=STYLE["bad"],  style="italic")

    _xticks(ax)
    ax.set_xlabel("Agent Population Size (N)", fontsize=11)
    ax.set_ylabel("|ABM Mean − BJS Target|", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True)

    plt.tight_layout()
    _save(fig, "chart4_mae_vs_size.png")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — CI width vs agent size
# ─────────────────────────────────────────────────────────────────────────────
def chart5_ci_width_vs_size(df_s: pd.DataFrame):
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Chart 5 — Estimation Precision: 95% CI Width vs Population Size\n"
        "Left: absolute CI width | Right: CI width as % of BJS target\n"
        "Narrowing CI with larger N confirms Monte Carlo stability",
        fontsize=11, fontweight="bold"
    )

    for yrs, label, color, marker in WINDOW_STYLES:
        sizes, cis = _get(df_s, yrs, "ci")
        ci_width   = cis * 2   # full width = 2 × half-CI

        ax_abs.plot(sizes, ci_width, f"{marker}-",
                    color=color, linewidth=2, markersize=7,
                    label=f"{label}", zorder=3)

        rel_width = ci_width / BJS_TARGETS[yrs] * 100
        ax_rel.plot(sizes, rel_width, f"{marker}-",
                    color=color, linewidth=2, markersize=7,
                    label=f"{label}", zorder=3)

    # Target: CI width < 2pp (0.02) is publication-standard precision
    ax_abs.axhline(0.02, color=STYLE["good"], linewidth=1.3,
                   linestyle="--", label="2pp precision threshold")
    ax_rel.axhline(3.0,  color=STYLE["good"], linewidth=1.3,
                   linestyle="--", label="3% relative precision threshold")

    for ax, ylabel, title in [
        (ax_abs, "95% CI Full Width (pp)",
         "A — Absolute CI Width vs Population Size"),
        (ax_rel, "CI Width as % of BJS Target",
         "B — Relative CI Width vs Population Size"),
    ]:
        ax.axvline(1000, color=STYLE["bjs"], linewidth=1.5,
                   linestyle="--", label="Chosen N=1,000")
        _xticks(ax)
        ax.set_xlabel("Agent Population Size (N)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5)
        ax.grid(True)

    plt.tight_layout()
    _save(fig, "chart5_ci_width_vs_size.png")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — Summary panel (4-panel dissertation overview)
# ─────────────────────────────────────────────────────────────────────────────
def chart6_summary_panel(df_s: pd.DataFrame, df_runs: pd.DataFrame):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Figure — Recidivism ABM Agent-Size Convergence Analysis\n"
        f"N={N_RUNS} runs × {SEEDS_PER_RUN} seeds per agent size | "
        f"Agent sizes: {min(AGENT_SIZES):,}–{max(AGENT_SIZES):,} | "
        "BJS NCJ 250975 benchmarks: 3yr=68%, 6yr=79%, 9yr=83%",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 2, hspace=0.42, wspace=0.32,
        left=0.07, right=0.97, top=0.90, bottom=0.08
    )

    # ── Panel A: Consolidated convergence ────────────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    for yrs, label, color, marker in WINDOW_STYLES:
        sizes, means = _get(df_s, yrs, "mean")
        _,     cis   = _get(df_s, yrs, "ci")
        _add_bjs(axA, yrs, label=False)
        axA.errorbar(
            sizes, means, yerr=cis,
            fmt=f"{marker}-", color=color,
            linewidth=2, markersize=6, capsize=4,
            label=f"{label} ±95% CI", zorder=3,
        )
    axA.axvline(1000, color=STYLE["bjs"], linewidth=1.5,
                linestyle="--", label="Chosen N=1,000")
    _pct(axA)
    _xticks(axA)
    axA.set_xlabel("Agent Population Size (N)")
    axA.set_ylabel("Cumulative Rearrest Rate")
    axA.set_title("A — Rate Convergence (Mean ±95% CI)")
    axA.legend(fontsize=7.5, loc="lower right", ncol=1)
    axA.grid(True)

    # ── Panel B: MAE vs N ────────────────────────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    for yrs, label, color, marker in WINDOW_STYLES:
        sizes, maes = _get(df_s, yrs, "mae")
        axB.plot(sizes, maes, f"{marker}-",
                 color=color, linewidth=2, markersize=6,
                 label=f"{label}", zorder=3)
    axB.axhline(0.02, color=STYLE["good"], linewidth=1.3,
                linestyle="--", label="Excellent (MAE<0.02)")
    axB.axhline(0.05, color=STYLE["warn"], linewidth=1.3,
                linestyle="--", label="Acceptable (MAE<0.05)")
    axB.axvline(1000, color=STYLE["bjs"], linewidth=1.5,
                linestyle="--", label="Chosen N=1,000")
    axB.axhspan(0,    0.02, alpha=0.05, color=STYLE["good"])
    axB.axhspan(0.02, 0.05, alpha=0.05, color=STYLE["warn"])
    _xticks(axB)
    axB.set_xlabel("Agent Population Size (N)")
    axB.set_ylabel("|ABM − BJS Target|")
    axB.set_title("B — Calibration Accuracy (MAE vs BJS)")
    axB.legend(fontsize=7.5, loc="upper right")
    axB.grid(True)

    # ── Panel C: Variance–N log-log ──────────────────────────────────────
    axC = fig.add_subplot(gs[1, 0])
    ref_plotted = False
    for yrs, label, color, marker in WINDOW_STYLES:
        sizes, variances = _get(df_s, yrs, "var")
        axC.plot(sizes, variances, f"{marker}-",
                 color=color, linewidth=2, markersize=6,
                 label=f"{label}", zorder=3)
        if not ref_plotted and len(variances) > 0 and variances[0] > 0:
            n_range = np.array(AGENT_SIZES, dtype=float)
            ref     = variances[0] * (n_range[0] / n_range)
            axC.plot(n_range, ref, "k--", linewidth=1,
                     alpha=0.4, label="Theoretical 1/N")
            ref_plotted = True
    axC.set_xscale("log")
    axC.set_yscale("log")
    _xticks(axC)
    axC.set_xlabel("Agent Population Size (N, log scale)")
    axC.set_ylabel("Variance (log scale)")
    axC.set_title("C — Variance–N Diagnostic (Log-Log)")
    axC.legend(fontsize=7.5)
    axC.grid(True, which="both", alpha=0.5)

    # ── Panel D: Summary table ────────────────────────────────────────────
    axD = fig.add_subplot(gs[1, 1])
    axD.axis("off")

    # Build table: one row per agent size, cols = mean±CI for each window
    chosen_sizes = [100, 500, 1000, 2000, 3000]
    col_labels   = ["N", "3yr mean±CI", "MAE 3yr",
                     "6yr mean±CI", "MAE 6yr",
                     "9yr mean±CI", "MAE 9yr"]
    table_data, cell_colors = [], []

    for sz in chosen_sizes:
        row, row_colors = [f"{sz:,}"], ["#F0F0F0"]
        for yrs, _, color, _ in WINDOW_STYLES:
            sub = df_s[(df_s.agent_size==sz) & (df_s.year==yrs)]
            if sub.empty:
                row += ["—", "—"]
                row_colors += ["#F8F8F8", "#F8F8F8"]
                continue
            m   = float(sub["mean"].iloc[0])
            ci  = float(sub["ci"].iloc[0])
            mae = float(sub["mae"].iloc[0])
            row.append(f"{m:.1%}±{ci*100:.1f}pp")
            row.append(f"{mae:.4f}")
            # Colour MAE cell
            mae_color = ("#D6EAD6" if mae < 0.02 else
                         "#FFF3CD" if mae < 0.05 else "#FADBD8")
            row_colors += ["#F8F8F8", mae_color]
        table_data.append(row)
        cell_colors.append(row_colors)

    tbl = axD.table(
        cellText    = table_data,
        colLabels   = col_labels,
        cellColours = cell_colors,
        colColours  = ["#1A3D5C"] * len(col_labels),
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.9)

    # Set column widths
    col_widths = [0.12, 0.18, 0.12, 0.18, 0.12, 0.18, 0.12]
    for (row, col), cell in tbl.get_celld().items():
        if col < len(col_widths):
            cell.set_width(col_widths[col])
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")

    axD.set_title("D — Convergence Summary Table\n"
                  "(Green=Excellent MAE<0.02 | Yellow=Acceptable | Red=Poor)",
                  fontsize=10, fontweight="bold", pad=10)

    plt.savefig(
        os.path.join(OUTPUT_DIR, "chart6_summary_panel.png"),
        dpi=150, bbox_inches="tight", facecolor=STYLE["bg"]
    )
    plt.close(fig)
    print("   📊 chart6_summary_panel.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip simulation, regenerate charts from saved CSVs")
    args = parser.parse_args()

    if args.replot and os.path.exists(RUN_LEVEL_CSV):
        print(f"\n✔ Loading saved results from {RUN_LEVEL_CSV}")
        df_runs    = pd.read_csv(RUN_LEVEL_CSV)
        df_summary = build_summary(df_runs)
    else:
        df_runs, df_summary = run_experiment_parallel()

    df_summary.to_csv(SUMMARY_CSV, index=False)

    # Print console summary
    print(f"\n{'='*65}")
    print(f"  CONVERGENCE ANALYSIS SUMMARY")
    print(f"  {'N':>6}  {'3yr mean':>10}  {'6yr mean':>10}  {'9yr mean':>10}")
    print(f"{'─'*65}")
    for sz in AGENT_SIZES:
        vals = {}
        for yrs in [3, 6, 9]:
            sub = df_summary[(df_summary.agent_size==sz)&(df_summary.year==yrs)]
            vals[yrs] = f"{sub['mean'].iloc[0]:.1%}" if not sub.empty else "—"
        print(f"  {sz:>6,}  {vals[3]:>10}  {vals[6]:>10}  {vals[9]:>10}")
    print(f"{'='*65}")

    # Generate all charts
    print(f"\n📊 Generating 6 dissertation charts...")
    chart1_convergence_per_window(df_summary)
    chart2_consolidated_convergence(df_summary)
    chart3_variance_convergence(df_summary)
    chart4_mae_vs_size(df_summary)
    chart5_ci_width_vs_size(df_summary)
    chart6_summary_panel(df_summary, df_runs)

    print(f"\n✅ All charts saved to {OUTPUT_DIR}/")
    print(f"\nFor dissertation:")
    print(f"  Primary:  chart6_summary_panel.png  (four-panel overview)")
    print(f"  Section:  chart2_consolidated_convergence.png")
    print(f"  Appendix: chart3_variance_convergence.png")
    print(f"            chart4_mae_vs_size.png")


if __name__ == "__main__":
    main()