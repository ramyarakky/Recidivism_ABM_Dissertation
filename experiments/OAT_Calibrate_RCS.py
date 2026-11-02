#!/usr/bin/env python3
"""
OAT_Calibration_RCS.py
=============
Targeted dual-constrained sweep of Risk_Contrast_Strength (γ) only.
All other parameters locked at final calibrated values.

IDENTIFICATION STRATEGY
------------------------
γ is identified via dual constraint:
  1. Aggregate BJS fit: all three windows within ±2pp of BJS targets
  2. Tier spread: minimise PCRA tier MAE within the admissible range

This resolves the structural incompatibility between PCRA tier targets
and the BJS aggregate when 39.9% of agents are in the High-risk tier —
no single γ value can simultaneously satisfy both constraints at full
precision, so the admissible range is identified first, then the value
within that range minimising tier MAE is selected.

OUTPUTS
-------
  tune_gamma_output/gamma_sweep.csv
  tune_gamma_output/gamma_chart1_aggregate_rates.png
  tune_gamma_output/gamma_chart2_dual_constraint.png
  tune_gamma_output/gamma_chart3_admissible_range.png
  tune_gamma_output/gamma_chart4_key_values.png
  tune_gamma_output/gamma_chart5_tradeoff.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import PercentFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_flat_risk_weights

# ─────────────────────────────────────────────────────────────────────────────
# LOCKED PARAMETERS — all five Stage 1 values + bias off
# ─────────────────────────────────────────────────────────────────────────────
LOCKED = {
    "Supervision_Monitoring_Intensity":      1.050,
    "Risk_Effect_Decay_After_3Y":            0.8667,
    "Risk_Effect_Decay_After_6Y":            0.8000,
    "Supervision_Monitoring_Decay_After_3Y": 0.9333,
    "Supervision_Monitoring_Decay_After_6Y": 0.3000,
    "bias_factor":         0.0,
    "bias_scope":          "supervision_only",
    "tech_detect_base":    0.30,
    "group_bias_gender":   {"Male": 0.0, "Female": 0.0},
    "group_bias_race":     {"White": 0.0, "Black": 0.0,
                            "Hispanic": 0.0, "Other": 0.0},
}

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GAMMA_VALUES = np.linspace(0.0, 3.0, 13)   # 0.00, 0.25, 0.50 ... 3.00
SEEDS        = [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087]
N_REPS       = 5
BJS          = {3: 0.68, 6: 0.79, 9: 0.83}
AGG_THRESHOLD = 0.02   # ±2pp acceptable MAE on each window

MODEL_CONFIG = {
    "initial_agents":        1000,
    "warmup_months":         144,
    "study_months":          108,
    "monthly_intake":        10,
    "bias_factor":           0.0,
    "enable_peer_influence": True,
    "mode":                  "realistic",
    "output_directory":      "OAT_Calibrate_RCS",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHART STYLE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "3yr":      "#2E75B6",
    "6yr":      "#70AD47",
    "9yr":      "#ED7D31",
    "tier_mae": "#7030A0",
    "bjs":      "#1A3D5C",
    "admit":    "#E8F5E9",
    "optimal":  "#C00000",
    "grid":     "#E0E0E0",
}


# ─────────────────────────────────────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────────────────────────────────────
def _worker(task):
    gamma, seed = task
    cal = dict(LOCKED)
    cal["Risk_Contrast_Strength"] = float(gamma)

    try:
        model = RecidivismModel(
            initial_agents        = MODEL_CONFIG["initial_agents"],
            bias_factor           = MODEL_CONFIG["bias_factor"],
            monthly_intake        = MODEL_CONFIG["monthly_intake"],
            warmup_months         = MODEL_CONFIG["warmup_months"],
            study_months          = MODEL_CONFIG["study_months"],
            enable_peer_influence = MODEL_CONFIG["enable_peer_influence"],
            weights               = get_flat_risk_weights(),
            seed                  = seed,
            mode                  = MODEL_CONFIG["mode"],
            output_directory      = MODEL_CONFIG["output_directory"],
            calibration_params    = cal,
        )
        model.export_csv = False

        while model.running:
            model.step()

        rates = {}
        for yrs in [3, 6, 9]:
            r = model.calculate_flag_rate(f"rearrest_{yrs}_yrs")
            rates[yrs] = r if r is not None else 0.0

        try:
            gamma_loss = model.compute_gamma_loss()
            if gamma_loss is None or np.isnan(gamma_loss):
                gamma_loss = 999.0
        except Exception:
            gamma_loss = 999.0

        return (gamma, seed, rates[3], rates[6], rates[9], gamma_loss)

    except Exception as e:
        print(f"  ⚠️  Worker error γ={gamma:.3f} seed={seed}: {e}")
        return (gamma, seed, 0.0, 0.0, 0.0, 999.0)


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_sweep(n_workers: int) -> list:
    tasks = []
    for gamma in GAMMA_VALUES:
        for rep in range(N_REPS):
            for seed in SEEDS:
                tasks.append((float(gamma), seed + rep * 10_000))

    raw = {float(g): {"r3": [], "r6": [], "r9": [], "gl": []}
           for g in GAMMA_VALUES}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="  γ sweep", ncols=70) as bar:
            for fut in as_completed(futures):
                try:
                    g, seed, r3, r6, r9, gl = fut.result()
                    key = min(raw.keys(), key=lambda k: abs(k - g))
                    raw[key]["r3"].append(r3)
                    raw[key]["r6"].append(r6)
                    raw[key]["r9"].append(r9)
                    raw[key]["gl"].append(gl)
                except Exception as e:
                    print(f"\n  ⚠️  {e}")
                finally:
                    bar.update(1)

    rows = []
    for g in sorted(raw.keys()):
        r3 = np.mean(raw[g]["r3"]) if raw[g]["r3"] else 0.0
        r6 = np.mean(raw[g]["r6"]) if raw[g]["r6"] else 0.0
        r9 = np.mean(raw[g]["r9"]) if raw[g]["r9"] else 0.0
        gl_vals = [x for x in raw[g]["gl"] if x < 998.0]
        gl = np.mean(gl_vals) if gl_vals else 999.0

        m3 = abs(r3 - BJS[3])
        m6 = abs(r6 - BJS[6])
        m9 = abs(r9 - BJS[9])
        agg_ok = m3 < AGG_THRESHOLD and m6 < AGG_THRESHOLD and m9 < AGG_THRESHOLD

        rows.append({
            "gamma":      g,
            "r3":         r3,
            "r6":         r6,
            "r9":         r9,
            "gamma_loss": gl,
            "mae_3yr":    m3,
            "mae_6yr":    m6,
            "mae_9yr":    m9,
            "agg_ok":     agg_ok,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE + IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def identify_gamma(rows: list) -> float:
    print(f"\n{'='*75}")
    print(f"  {'γ':>6}  {'3yr':>7}  {'6yr':>7}  {'9yr':>7}  "
          f"{'tier_MAE':>9}  {'mae_3yr':>8}  verdict")
    print(f"{'─'*75}")

    best_gamma = None
    best_loss  = 999.0

    for r in rows:
        verdict = ""
        if r["agg_ok"]:
            if r["gamma_loss"] < best_loss:
                best_loss  = r["gamma_loss"]
                best_gamma = r["gamma"]
                verdict = "✅ AGG+TIER"
            else:
                verdict = "✅ AGG"
        else:
            verdict = "❌ overshoot"

        print(f"  {r['gamma']:>6.3f}  {r['r3']:>7.1%}  {r['r6']:>7.1%}  "
              f"{r['r9']:>7.1%}  {r['gamma_loss']:>9.4f}  "
              f"{r['mae_3yr']:>8.4f}  {verdict}")

    print(f"{'='*75}")

    if best_gamma is not None:
        opt = next(r for r in rows if r["gamma"] == best_gamma)
        print(f"\n  ✅ IDENTIFIED: γ = {best_gamma:.4f}  "
              f"(tier MAE = {best_loss:.4f})")
        print(f"     3yr={opt['r3']:.1%}  6yr={opt['r6']:.1%}  "
              f"9yr={opt['r9']:.1%}")
    else:
        print(f"\n  ⚠️  No γ satisfies ±2pp aggregate constraint")
        best_row   = min(rows, key=lambda r: r["mae_3yr"]+r["mae_6yr"]+r["mae_9yr"])
        best_gamma = best_row["gamma"]
        print(f"     Closest aggregate fit: γ={best_gamma:.3f}  "
              f"3yr={best_row['r3']:.1%}  6yr={best_row['r6']:.1%}  "
              f"9yr={best_row['r9']:.1%}")

    return best_gamma


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _admit_bounds(rows):
    admit = [r["gamma"] for r in rows if r["agg_ok"]]
    return (min(admit), max(admit)) if admit else (0, 0)


def _save(fig, path, label):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 {label} → {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Aggregate rates across γ
# ─────────────────────────────────────────────────────────────────────────────
def chart1_aggregate_rates(rows, best_gamma, outdir):
    gammas = [r["gamma"] for r in rows]
    r3s    = [r["r3"]    for r in rows]
    r6s    = [r["r6"]    for r in rows]
    r9s    = [r["r9"]    for r in rows]
    amin, amax = _admit_bounds(rows)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(
        "Chart 1 — Risk Contrast Strength (γ): Effect on Aggregate Rearrest Rates\n"
        "Calibration target: all windows within ±2pp of BJS benchmarks",
        fontsize=12, fontweight="bold")

    ax.axvspan(amin, amax, alpha=0.08, color="#27AE60",
               label=f"Admissible γ∈[{amin:.2f},{amax:.2f}]")

    ax.plot(gammas, r3s, "o-", color=COLORS["3yr"], linewidth=2,
            markersize=6, label="ABM 3-Year", zorder=3)
    ax.plot(gammas, r6s, "s-", color=COLORS["6yr"], linewidth=2,
            markersize=6, label="ABM 6-Year", zorder=3)
    ax.plot(gammas, r9s, "^-", color=COLORS["9yr"], linewidth=2,
            markersize=6, label="ABM 9-Year", zorder=3)

    for yrs, col in [(3, COLORS["3yr"]), (6, COLORS["6yr"]),
                     (9, COLORS["9yr"])]:
        ax.axhline(BJS[yrs], color=col, linewidth=1.5,
                   linestyle=":", alpha=0.7)
        ax.axhspan(BJS[yrs]-0.02, BJS[yrs]+0.02, alpha=0.04, color=col)

    ax.axvline(best_gamma, color=COLORS["optimal"], linewidth=1.8,
               linestyle="--", label=f"Selected γ={best_gamma:.3f}", zorder=5)

    opt = next(r for r in rows if r["gamma"] == best_gamma)
    ax.annotate(
        f"Selected: γ={best_gamma:.2f}\n"
        f"3yr={opt['r3']:.1%}  6yr={opt['r6']:.1%}  9yr={opt['r9']:.1%}",
        xy=(best_gamma, opt["r3"]),
        xytext=(best_gamma + 0.25, opt["r3"] - 0.04),
        fontsize=8.5, color=COLORS["optimal"],
        arrowprops=dict(arrowstyle="->", color=COLORS["optimal"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["optimal"], alpha=0.9))

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(min(gammas)-0.1, max(gammas)+0.1)
    ax.set_ylim(0.60, 0.92)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    _style_ax(ax, "Risk Contrast Strength (γ)",
              "Cumulative Rearrest Rate",
              "Aggregate Rearrest Rates by γ Value")

    plt.tight_layout()
    _save(fig, os.path.join(outdir, "gamma_chart1_aggregate_rates.png"),
          "Chart 1")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Dual constraint: tier MAE + aggregate MAE
# ─────────────────────────────────────────────────────────────────────────────
def chart2_dual_constraint(rows, best_gamma, outdir):
    gammas   = [r["gamma"]      for r in rows]
    tier_mae = [r["gamma_loss"] for r in rows]
    mae3     = [r["mae_3yr"]    for r in rows]
    mae6     = [r["mae_6yr"]    for r in rows]
    mae9     = [r["mae_9yr"]    for r in rows]
    amin, amax = _admit_bounds(rows)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Chart 2 — Dual-Constraint γ Identification\n"
        "Left: PCRA Tier MAE (tier spread fit) | "
        "Right: BJS Aggregate MAE (population fit)",
        fontsize=12, fontweight="bold")

    for ax in (axL, axR):
        ax.axvspan(amin, amax, alpha=0.08, color="#27AE60",
                   label=f"Admissible γ∈[{amin:.2f},{amax:.2f}]")
        ax.axvline(best_gamma, color=COLORS["optimal"], linewidth=1.8,
                   linestyle="--", label=f"Selected γ={best_gamma:.3f}",
                   zorder=5)

    # Left panel — tier MAE
    axL.plot(gammas, tier_mae, "o-", color=COLORS["tier_mae"],
             linewidth=2.2, markersize=7, label="Tier MAE", zorder=3)
    axL.fill_between(gammas, tier_mae, alpha=0.10, color=COLORS["tier_mae"])
    axL.legend(fontsize=9)
    _style_ax(axL, "Risk Contrast Strength (γ)",
              "Mean Absolute Error (tier-stratified)",
              "A — PCRA Tier MAE vs γ")

    # Right panel — aggregate MAE per window
    axR.plot(gammas, mae3, "o-", color=COLORS["3yr"],
             linewidth=2, markersize=6, label="MAE 3-Year")
    axR.plot(gammas, mae6, "s-", color=COLORS["6yr"],
             linewidth=2, markersize=6, label="MAE 6-Year")
    axR.plot(gammas, mae9, "^-", color=COLORS["9yr"],
             linewidth=2, markersize=6, label="MAE 9-Year")
    axR.axhline(AGG_THRESHOLD, color=COLORS["optimal"], linewidth=1.5,
                linestyle="--", label=f"±2pp threshold (MAE={AGG_THRESHOLD})")
    axR.fill_between(gammas, [0]*len(gammas),
                     [AGG_THRESHOLD]*len(gammas),
                     alpha=0.06, color="#27AE60", label="Acceptable zone")
    axR.legend(fontsize=9, loc="upper left")
    _style_ax(axR, "Risk Contrast Strength (γ)",
              "Mean Absolute Error vs BJS Target",
              "B — Aggregate MAE vs γ")

    plt.tight_layout()
    _save(fig, os.path.join(outdir, "gamma_chart2_dual_constraint.png"),
          "Chart 2")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Admissible range with verdict colour coding
# ─────────────────────────────────────────────────────────────────────────────
def chart3_admissible_range(rows, best_gamma, outdir):
    gammas   = [r["gamma"]      for r in rows]
    tier_mae = [r["gamma_loss"] for r in rows]
    amin, amax = _admit_bounds(rows)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(
        "Chart 3 — Admissible Range for γ Identification\n"
        "Green ● = satisfies ±2pp aggregate constraint | "
        "Red ✕ = aggregate overshoot",
        fontsize=12, fontweight="bold")

    ax.plot(gammas, tier_mae, "-", color="#AAAAAA",
            linewidth=1.2, zorder=2, alpha=0.5)

    for r in rows:
        color  = "#27AE60" if r["agg_ok"] else "#C0392B"
        marker = "o"       if r["agg_ok"] else "X"
        ax.scatter(r["gamma"], r["gamma_loss"],
                   color=color, s=130, marker=marker,
                   zorder=4, edgecolors="white", linewidths=1.2)

    if amin != amax:
        ax.axvspan(amin, amax, alpha=0.10, color="#27AE60",
                   label=f"Admissible: γ∈[{amin:.2f},{amax:.2f}]")

    ax.axvline(best_gamma, color=COLORS["optimal"], linewidth=1.8,
               linestyle="--", zorder=5)

    opt_mae = next(r["gamma_loss"] for r in rows if r["gamma"] == best_gamma)
    ax.annotate(
        f"Selected γ={best_gamma:.2f}\nTier MAE={opt_mae:.4f}",
        xy=(best_gamma, opt_mae),
        xytext=(best_gamma + 0.2, opt_mae + 0.008),
        fontsize=9, color=COLORS["optimal"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["optimal"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["optimal"], alpha=0.9))

    legend_elements = [
        mlines.Line2D([0],[0], marker="o", color="w",
                      markerfacecolor="#27AE60", markersize=10,
                      label="Admissible (aggregate ±2pp OK)"),
        mlines.Line2D([0],[0], marker="X", color="w",
                      markerfacecolor="#C0392B", markersize=10,
                      label="Inadmissible (aggregate overshoot)"),
        mlines.Line2D([0],[0], color=COLORS["optimal"],
                      linewidth=2, linestyle="--",
                      label=f"Selected γ={best_gamma:.2f}"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    _style_ax(ax, "Risk Contrast Strength (γ)",
              "PCRA Tier MAE",
              "Admissible γ Range — Dual Constraint Identification")

    plt.tight_layout()
    _save(fig, os.path.join(outdir, "gamma_chart3_admissible_range.png"),
          "Chart 3")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Aggregate rates at key γ values
# ─────────────────────────────────────────────────────────────────────────────
def chart4_key_values(rows, best_gamma, outdir):
    amin, amax = _admit_bounds(rows)
    key_gammas = [0.0, best_gamma, amax]
    key_labels = [
        "γ=0.00\n(no risk contrast)",
        f"γ={best_gamma:.2f}\n(selected)",
        f"γ={amax:.2f}\n(max admissible)",
    ]
    windows    = [3, 6, 9]
    win_labels = ["3-Year", "6-Year", "9-Year"]
    win_colors = [COLORS["3yr"], COLORS["6yr"], COLORS["9yr"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Chart 4 — Rearrest Rates at Key γ Values vs BJS Targets\n"
        "Illustrates effect of risk contrast on aggregate outcomes",
        fontsize=12, fontweight="bold")

    for ax, gval, glabel in zip(axes, key_gammas, key_labels):
        row      = min(rows, key=lambda r: abs(r["gamma"] - gval))
        abm_vals = [row["r3"], row["r6"], row["r9"]]
        bjs_vals = [BJS[w] for w in windows]
        x = np.arange(len(windows))
        w = 0.35

        bars_abm = ax.bar(x - w/2, abm_vals, w,
                          color=win_colors, alpha=0.85,
                          label="ABM", edgecolor="white")
        bars_bjs = ax.bar(x + w/2, bjs_vals, w,
                          color=COLORS["bjs"], alpha=0.70,
                          label="BJS Target", edgecolor="white")

        for bar, val in zip(bars_abm, abm_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        for bar, val in zip(bars_bjs, bjs_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.0%}", ha="center", va="bottom",
                    fontsize=8.5, color=COLORS["bjs"])

        ax.set_xticks(x)
        ax.set_xticklabels(win_labels, fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.02)
        ax.set_title(glabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, color=COLORS["grid"], linewidth=0.6,
                linestyle="--", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, os.path.join(outdir, "gamma_chart4_key_values.png"),
          "Chart 4")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Tradeoff: tier fit vs aggregate fit
# ─────────────────────────────────────────────────────────────────────────────
def chart5_tradeoff(rows, best_gamma, outdir):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        "Chart 5 — Calibration Tradeoff: Tier Differentiation vs Aggregate Accuracy\n"
        "Each point = one γ value | Ideal = bottom-left | "
        "Selected = best balance within admissible zone",
        fontsize=12, fontweight="bold")

    for r in rows:
        color  = "#27AE60" if r["agg_ok"] else "#C0392B"
        marker = "o"       if r["agg_ok"] else "X"
        ax.scatter(r["gamma_loss"], r["mae_3yr"],
                   color=color, s=130, marker=marker,
                   zorder=4, edgecolors="white", linewidths=1.2)
        ax.annotate(f"γ={r['gamma']:.2f}",
                    (r["gamma_loss"], r["mae_3yr"]),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color="#555555")

    ax.axhline(AGG_THRESHOLD, color=COLORS["optimal"], linewidth=1.3,
               linestyle="--", alpha=0.8,
               label=f"±2pp aggregate threshold (MAE={AGG_THRESHOLD})")
    ax.axhspan(0, AGG_THRESHOLD, alpha=0.05, color="#27AE60")

    opt = next(r for r in rows if r["gamma"] == best_gamma)
    ax.scatter(opt["gamma_loss"], opt["mae_3yr"],
               color=COLORS["optimal"], s=250, marker="*",
               zorder=6, label=f"Selected γ={best_gamma:.2f}",
               edgecolors="white", linewidths=1.5)

    legend_elements = [
        mlines.Line2D([0],[0], marker="o", color="w",
                      markerfacecolor="#27AE60", markersize=10,
                      label="Admissible (aggregate OK)"),
        mlines.Line2D([0],[0], marker="X", color="w",
                      markerfacecolor="#C0392B", markersize=10,
                      label="Inadmissible (overshoot)"),
        mlines.Line2D([0],[0], marker="*", color="w",
                      markerfacecolor=COLORS["optimal"], markersize=14,
                      label=f"Selected γ={best_gamma:.2f}"),
        mlines.Line2D([0],[0], color=COLORS["optimal"],
                      linewidth=1.5, linestyle="--",
                      label=f"±2pp aggregate threshold"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    _style_ax(ax,
              "PCRA Tier MAE (lower = better tier differentiation)",
              "3-Year Aggregate MAE (lower = better BJS fit)",
              "Tradeoff: Tier Differentiation vs Aggregate Accuracy")

    plt.tight_layout()
    _save(fig, os.path.join(outdir, "gamma_chart5_tradeoff.png"),
          "Chart 5")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT ALL CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_gamma_results(rows: list, best_gamma: float, outdir: str):
    print(f"\n📊 Generating 5 dissertation charts...")
    chart1_aggregate_rates(rows, best_gamma, outdir)
    chart2_dual_constraint(rows, best_gamma, outdir)
    chart3_admissible_range(rows, best_gamma, outdir)
    chart4_key_values(rows, best_gamma, outdir)
    chart5_tradeoff(rows, best_gamma, outdir)
    print(f"\n   ✅ All 5 charts saved to {outdir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores",   type=int, default=None)
    parser.add_argument("--replot",  action="store_true",
                        help="Skip sweep, regenerate charts from saved CSV")
    args = parser.parse_args()

    outdir = MODEL_CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "gamma_sweep.csv")

    n_tasks   = len(GAMMA_VALUES) * N_REPS * len(SEEDS)
    n_workers = args.cores or min(60, n_tasks)

    print(f"\n{'='*60}")
    print(f"  γ TARGETED SWEEP  —  Dual Constraint Identification")
    print(f"  {len(GAMMA_VALUES)} values × {N_REPS} reps × {len(SEEDS)} seeds"
          f" = {n_tasks} tasks  →  {n_workers} workers")
    print(f"  α locked at {LOCKED['Supervision_Monitoring_Intensity']}")
    print(f"  Aggregate threshold: ±{AGG_THRESHOLD*100:.0f}pp on all windows")
    print(f"{'='*60}\n")

    # ── Load from CSV or run sweep ────────────────────────────────────────
    if args.replot and os.path.exists(csv_path):
        print(f"  ✔ Loading existing sweep from {csv_path}")
        df   = pd.read_csv(csv_path)
        rows = df.to_dict("records")
    else:
        rows = run_sweep(n_workers)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n  Results → {csv_path}")

    # ── Identify optimal γ ────────────────────────────────────────────────
    best_gamma = identify_gamma(rows)

    # ── Charts ────────────────────────────────────────────────────────────
    plot_gamma_results(rows, best_gamma, outdir)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION: set Risk_Contrast_Strength = {best_gamma:.4f}")
    print(f"  Update get_global_calibration_params() in risk_config.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()