# scenario_sim_cal_stress.py
# ============================================================
# Recidivism ABM — Global Parameter Stress Test
#
# Runs one scenario per global calibration parameter at its
# LOW, BASELINE, and HIGH stress bound (one parameter varied
# at a time, all others held at baseline). NIJ risk weights
# are never modified — they are fixed empirical coefficients.
#
# Parameters tested
# -----------------
#   SMI  — Supervision_Monitoring_Intensity        0.00 / 0.35 / 1.00
#   RCS  — Risk_Contrast_Strength                  0.00 / 0.00 / 4.00
#   RED3 — Risk_Effect_Decay_After_3Y              0.45 / 0.90 / 1.00
#   RED6 — Risk_Effect_Decay_After_6Y              0.35 / 0.70 / 1.00
#   SMD3 — Supervision_Monitoring_Decay_After_3Y   0.10 / 0.35 / 0.70
#   SMD6 — Supervision_Monitoring_Decay_After_6Y   0.05 / 0.20 / 0.40
#
# Outputs
# -------
#   stress_results.csv       — all rates for every scenario/window
#   stress_overview.png      — horizontal bars: all params LOW/BASE/HIGH
#   stress_heatmap.png       — gap-to-baseline heatmap across windows
#   stress_param_{key}.png   — per-parameter deep-dive (3 windows)
#
# Usage
# -----
#   python scenario_sim_cal_stress.py
#   python scenario_sim_cal_stress.py --reps 10 --cores 8
# ============================================================

import os, sys, datetime, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm_phase2.model.recidivism_model import RecidivismModel
from recidivism_abm_phase2.config.risk_config import (
    get_peer_influence_config,
    get_flat_risk_weights,
    get_global_calibration_params,
)

# ─────────────────────────────────────────────────────────────
# RUN CONFIG
# ─────────────────────────────────────────────────────────────
N_REPS     = 10
BASE_SEEDS = list(range(10, 10 + N_REPS))

NOMINAL_RUN = dict(
    initial_agents=1000, monthly_intake=10,
    warmup_months=144,   study_months=108,
    bias_factor=0,       enable_peer_influence=True,
)

WIN_FLAGS = {3: "rearrest_3_yrs", 6: "rearrest_6_yrs", 9: "rearrest_9_yrs"}
WINDOWS   = [3, 6, 9]
WIN_COLS  = {3: "#2166ac", 6: "#1a9641", 9: "#d95f02"}
WIN_LBLS  = {3: "3-Year",  6: "6-Year",  9: "9-Year"}
BJS_OVERALL = {3: 0.68, 6: 0.79, 9: 0.83}

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "SL_stress_output", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# GLOBAL PARAMETER DEFINITIONS
# ─────────────────────────────────────────────────────────────
GLOBAL_PARAMS = {
    "SMI": {
        "key":      "Supervision_Monitoring_Intensity",
        "label":    "Supervision Monitoring Intensity",
        "short":    "SMI",
        "baseline": 0.35,
        "low":      0.00,
        "high":     1.00,
        "color":    "#c0392b",
        "finding":  "DOMINANT — only parameter with meaningful output sensitivity (+-4 to 6pp)",
    },
    "RCS": {
        "key":      "Risk_Contrast_Strength",
        "label":    "Risk Contrast Strength",
        "short":    "RCS",
        "baseline": 0.00,
        "low":      0.00,
        "high":     4.00,
        "color":    "#8e44ad",
        "finding":  "ANOMALOUS — HIGH bound suppresses rates ~10pp; fixed at 0.0",
    },
    "RED3": {
        "key":      "Risk_Effect_Decay_After_3Y",
        "label":    "Risk Effect Decay After 3Y",
        "short":    "RED3",
        "baseline": 0.90,
        "low":      0.45,
        "high":     1.00,
        "color":    "#2980b9",
        "finding":  "INERT — <=0.2pp deviation at any bound across all windows",
    },
    "RED6": {
        "key":      "Risk_Effect_Decay_After_6Y",
        "label":    "Risk Effect Decay After 6Y",
        "short":    "RED6",
        "baseline": 0.70,
        "low":      0.35,
        "high":     1.00,
        "color":    "#16a085",
        "finding":  "INERT — <=0.2pp deviation at any bound across all windows",
    },
    "SMD3": {
        "key":      "Supervision_Monitoring_Decay_After_3Y",
        "label":    "Supervision Monitoring Decay After 3Y",
        "short":    "SMD3",
        "baseline": 0.35,
        "low":      0.10,
        "high":     0.70,
        "color":    "#e67e22",
        "finding":  "INERT — boundary result; sensitivity check recommended pre Phase 2",
    },
    "SMD6": {
        "key":      "Supervision_Monitoring_Decay_After_6Y",
        "label":    "Supervision Monitoring Decay After 6Y",
        "short":    "SMD6",
        "baseline": 0.20,
        "low":      0.05,
        "high":     0.40,
        "color":    "#27ae60",
        "finding":  "INERT — boundary result; sensitivity check recommended pre Phase 2",
    },
}

# ─────────────────────────────────────────────────────────────
# BUILD SCENARIO LIST
# One scenario per parameter x level (LOW / BASELINE / HIGH).
# All other parameters held at their calibrated baseline values.
# ─────────────────────────────────────────────────────────────
def _build_scenarios():
    base = {meta["key"]: meta["baseline"] for meta in GLOBAL_PARAMS.values()}
    scenarios = {}

    scenarios["baseline"] = {
        **base,
        "_label": "Baseline", "_short": "BASE",
        "_param": None, "_level": "baseline", "_color": "#555555",
    }

    for pname, meta in GLOBAL_PARAMS.items():
        for level, val in [("low", meta["low"]), ("high", meta["high"])]:
            sname = f"{pname}_{level}"
            overrides = {**base, meta["key"]: val}
            overrides["_label"]  = f"{meta['short']} {level.upper()} ({val})"
            overrides["_short"]  = f"{meta['short']}_{level.upper()}"
            overrides["_param"]  = pname
            overrides["_level"]  = level
            overrides["_color"]  = meta["color"]
            scenarios[sname] = overrides

    return scenarios

SCENARIOS = _build_scenarios()
_META_KEYS = {"_label", "_short", "_param", "_level", "_color"}

# ─────────────────────────────────────────────────────────────
# MODEL RUNNER
# ─────────────────────────────────────────────────────────────
def _run(args):
    seed, sname, overrides = args
    try:
        cal = get_global_calibration_params()
        for k, v in overrides.items():
            if k not in _META_KEYS:
                cal[k] = v

        model = RecidivismModel(
            initial_agents        = NOMINAL_RUN["initial_agents"],
            bias_factor           = NOMINAL_RUN["bias_factor"],
            monthly_intake        = NOMINAL_RUN["monthly_intake"],
            warmup_months         = NOMINAL_RUN["warmup_months"],
            study_months          = NOMINAL_RUN["study_months"],
            enable_peer_influence = NOMINAL_RUN["enable_peer_influence"],
            weights               = get_flat_risk_weights(),
            peer_config           = get_peer_influence_config(True),
            calibration_params    = cal,
            seed                  = seed,
        )

        while model.running:
            model.step()

        rates = {}
        for yrs, flag in WIN_FLAGS.items():
            eligible = [a for a in model.schedule.agents
                        if getattr(a, "study_eligible_agent", False)]
            rates[yrs] = (
                sum(getattr(a, flag, False) for a in eligible) / len(eligible)
                if eligible else float("nan")
            )

        return {"scenario": sname, "seed": seed,
                "rate_3": rates[3], "rate_6": rates[6], "rate_9": rates[9]}, None

    except Exception as e:
        import traceback
        return None, f"[{sname} seed={seed}] {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────
def aggregate(raw):
    rows = []
    for sname in SCENARIOS:
        runs = [r for r, _ in raw if r and r["scenario"] == sname]
        for yrs in WINDOWS:
            vals = [r[f"rate_{yrs}"] for r in runs if np.isfinite(r[f"rate_{yrs}"])]
            rows.append({
                "scenario": sname,
                "param":    SCENARIOS[sname]["_param"],
                "level":    SCENARIOS[sname]["_level"],
                "label":    SCENARIOS[sname]["_label"],
                "window":   yrs,
                "mean":     float(np.mean(vals)) if vals else float("nan"),
                "std":      float(np.std(vals))  if len(vals) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# CHART 1 — STRESS OVERVIEW
# Grouped horizontal bars: one group per parameter,
# three bars per group (LOW / BASELINE / HIGH),
# three panels (one per window).
# ─────────────────────────────────────────────────────────────
def plot_stress_overview(df):
    param_keys = list(GLOBAL_PARAMS.keys())
    n_params   = len(param_keys)
    levels     = ["low", "baseline", "high"]
    level_cols = {"low": "#27ae60", "baseline": "#555555", "high": "#c0392b"}
    level_lbls = {"low": "LOW bound", "baseline": "Baseline", "high": "HIGH bound"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 9), facecolor="white")
    fig.suptitle(
        "Global Parameter Stress Test — Rearrest Rate Change by Parameter & Level\n"
        "One parameter varied at a time; all others held at calibrated baseline",
        fontsize=13, fontweight="bold", color="#1a3d5c", y=1.00,
    )

    bar_h   = 0.22
    offsets = {"low": -0.24, "baseline": 0.0, "high": 0.24}

    for wi, w in enumerate(WINDOWS):
        ax  = axes[wi]
        bjs = BJS_OVERALL[w]
        ax.set_facecolor("white")

        row_b     = df[(df["scenario"] == "baseline") & (df["window"] == w)]
        base_rate = float(row_b["mean"].values[0]) if not row_b.empty else float("nan")

        for pi, pname in enumerate(param_keys):
            meta  = GLOBAL_PARAMS[pname]
            y_ctr = n_params - 1 - pi

            if pi % 2 == 0:
                ax.axhspan(y_ctr - 0.45, y_ctr + 0.45,
                           color="#f5f5f5", zorder=0, linewidth=0)

            for level in levels:
                sname = "baseline" if level == "baseline" else f"{pname}_{level}"
                row   = df[(df["scenario"] == sname) & (df["window"] == w)]
                if row.empty:
                    continue
                rate   = float(row["mean"].values[0])
                gap_pp = (rate - base_rate) * 100
                yy     = y_ctr + offsets[level]
                col    = level_cols[level]

                ax.barh(yy, gap_pp, left=0, height=bar_h,
                        color=col, alpha=0.82, linewidth=0,
                        align="center", zorder=3)

                sign  = "+" if gap_pp >= 0 else ""
                x_lbl = gap_pp + (0.12 if gap_pp >= 0 else -0.12)
                ha    = "left" if gap_pp >= 0 else "right"
                if abs(gap_pp) < 0.15:
                    x_lbl = 0.15; ha = "left"
                ax.text(x_lbl, yy,
                        f"{sign}{gap_pp:.1f}pp  ({rate:.3f})",
                        va="center", ha=ha, fontsize=7,
                        color="#222222", fontfamily="monospace")

        ax.axvline(0, color="#333333", linewidth=1.5, zorder=5)
        ax.axvline((bjs - base_rate) * 100, color="#e74c3c",
                   linewidth=1.2, linestyle=":", zorder=4)

        ax.set_yticks(range(n_params))
        ax.set_yticklabels(
            [GLOBAL_PARAMS[p]["label"] for p in reversed(param_keys)],
            fontsize=9)
        for tick, pname in zip(ax.get_yticklabels(), reversed(param_keys)):
            tick.set_color(GLOBAL_PARAMS[pname]["color"])
            tick.set_fontweight("bold")

        ax.set_ylim(-0.65, n_params - 0.35)

        all_gaps = []
        for pname in param_keys:
            for level in ["low", "high"]:
                sname = f"{pname}_{level}"
                row   = df[(df["scenario"] == sname) & (df["window"] == w)]
                if not row.empty:
                    all_gaps.append(
                        (float(row["mean"].values[0]) - base_rate) * 100)
        xmax = max((abs(g) for g in all_gaps if np.isfinite(g)), default=1.0)
        xmax = max(xmax * 1.5, 1.5)
        ax.set_xlim(-xmax, xmax)

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:+.1f}pp"))
        ax.tick_params(axis="x", labelsize=8, colors="#555555")
        ax.set_xlabel("Change from baseline (pp)", fontsize=8.5, color="#555555")
        ax.set_title(f"{WIN_LBLS[w]} Window  (BJS {bjs:.0%})",
                     fontsize=11, fontweight="bold", color="#1a3d5c", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(axis="x", color="#eeeeee", linewidth=0.6, zorder=0)

    patches = [mpatches.Patch(color=level_cols[l], label=level_lbls[l])
               for l in levels]
    patches += [plt.Line2D([0],[0], color="#e74c3c", linewidth=1.5,
                            linestyle=":", label="BJS target")]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9.5, framealpha=0.95, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, "stress_overview.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────
# CHART 2 — GAP HEATMAP
# ─────────────────────────────────────────────────────────────
def plot_heatmap(df):
    import matplotlib.colors as mcolors

    oat_scenarios = [s for s in SCENARIOS if s != "baseline"]
    n_rows = len(oat_scenarios)
    n_cols = len(WINDOWS)

    base_rates = {}
    for w in WINDOWS:
        row = df[(df["scenario"] == "baseline") & (df["window"] == w)]
        base_rates[w] = float(row["mean"].values[0]) if not row.empty else float("nan")

    gap_matrix  = np.zeros((n_rows, n_cols))
    rate_matrix = np.zeros((n_rows, n_cols))
    for ri, sname in enumerate(oat_scenarios):
        for ci, w in enumerate(WINDOWS):
            row  = df[(df["scenario"] == sname) & (df["window"] == w)]
            rate = float(row["mean"].values[0]) if not row.empty else float("nan")
            gap_matrix[ri, ci]  = (rate - base_rates[w]) * 100
            rate_matrix[ri, ci] = rate

    finite = gap_matrix[np.isfinite(gap_matrix)]
    vmax   = max(abs(finite.min()), abs(finite.max()), 0.5)
    cmap   = mcolors.LinearSegmentedColormap.from_list(
        "gap", ["#1a9641", "#f7f7f7", "#c0392b"])

    row_labels = [SCENARIOS[s]["_label"] for s in oat_scenarios]
    row_colors = [SCENARIOS[s]["_color"] for s in oat_scenarios]

    fig, ax = plt.subplots(figsize=(11, max(7, n_rows * 0.42 + 2)),
                           facecolor="white")
    fig.suptitle(
        "Global Parameter Stress Test — Gap from Baseline Heatmap\n"
        "Cell value = pp change from baseline  |  Green = decrease  |  Red = increase",
        fontsize=12, fontweight="bold", color="#1a3d5c", y=1.01,
    )

    im = ax.imshow(gap_matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    for ri in range(n_rows):
        for ci in range(n_cols):
            gap  = gap_matrix[ri, ci]
            rate = rate_matrix[ri, ci]
            sign = "+" if gap >= 0 else ""
            tc   = "white" if abs(gap) > vmax * 0.55 else "#1a1a1a"
            ax.text(ci, ri, f"{sign}{gap:.1f}pp\n({rate:.3f})",
                    ha="center", va="center", fontsize=8.5,
                    color=tc, fontweight="bold", fontfamily="monospace")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [f"{WIN_LBLS[w]}\n(BJS {BJS_OVERALL[w]:.0%})" for w in WINDOWS],
        fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    for tick, col in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(col)
        tick.set_fontweight("bold")

    for i in range(1, len(GLOBAL_PARAMS)):
        ax.axhline(i * 2 - 0.5, color="white", linewidth=2.0, zorder=5)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([i * 2 + 0.5 for i in range(len(GLOBAL_PARAMS))])
    ax2.set_yticklabels(
        [GLOBAL_PARAMS[p]["short"] for p in GLOBAL_PARAMS],
        fontsize=9, fontweight="bold")
    for tick, pname in zip(ax2.get_yticklabels(), GLOBAL_PARAMS):
        tick.set_color(GLOBAL_PARAMS[pname]["color"])
    ax2.tick_params(length=0)
    ax2.spines[:].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.12)
    cbar.set_label("Gap from baseline (pp)", fontsize=9)
    ax.spines[:].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "stress_heatmap.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────
# CHART 3 — PER-PARAMETER DEEP DIVE
# ─────────────────────────────────────────────────────────────
def plot_per_param(df):
    level_cols = {"low": "#27ae60", "baseline": "#555555", "high": "#c0392b"}
    level_lbls = {"low": "LOW",     "baseline": "BASELINE","high": "HIGH"}

    for pname, meta in GLOBAL_PARAMS.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                 sharey=False, facecolor="white")
        fig.suptitle(
            f"Parameter Deep Dive — {meta['label']}  ({meta['short']})\n"
            f"LOW={meta['low']}   BASELINE={meta['baseline']}   "
            f"HIGH={meta['high']}   |   {meta['finding']}",
            fontsize=11, fontweight="bold", color="#1a3d5c", y=1.02,
        )

        for wi, w in enumerate(WINDOWS):
            ax  = axes[wi]
            bjs = BJS_OVERALL[w]
            ax.set_facecolor("#fafafa")

            scenarios_for_param = [
                ("low",      f"{pname}_low",  meta["low"]),
                ("baseline", "baseline",       meta["baseline"]),
                ("high",     f"{pname}_high",  meta["high"]),
            ]

            rates  = []
            stds   = []
            colors = []
            for level, sname, val in scenarios_for_param:
                row = df[(df["scenario"] == sname) & (df["window"] == w)]
                r   = float(row["mean"].values[0]) if not row.empty else float("nan")
                s   = float(row["std"].values[0])  if not row.empty else 0.0
                rates.append(r)
                stds.append(s)
                colors.append(level_cols[level])

            x_pos = np.array([0, 1, 2])
            bars  = ax.bar(x_pos, rates, 0.55, color=colors, alpha=0.85,
                           yerr=stds, capsize=5,
                           error_kw={"linewidth": 1.2}, zorder=3)

            base_rate = rates[1]
            for xi, (bar, rate, level) in enumerate(
                    zip(bars, rates, ["low","baseline","high"])):
                if not np.isfinite(rate):
                    continue
                gap   = (rate - base_rate) * 100
                sign  = "+" if gap >= 0 else ""
                g_lbl = "" if level == "baseline" else f"\n{sign}{gap:.1f}pp"
                ax.text(bar.get_x() + bar.get_width()/2,
                        rate + max(stds) * 0.5 + 0.006,
                        f"{rate:.3f}{g_lbl}",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color=level_cols[level],
                        fontfamily="monospace")

            ax.axhline(bjs, color="#e74c3c", linewidth=1.6,
                       linestyle="--", zorder=4)
            ax.text(2.42, bjs, f"BJS\n{bjs:.0%}",
                    va="center", ha="left", fontsize=8,
                    color="#e74c3c", fontweight="bold")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [f"{level_lbls[l]}\n({v})"
                 for (l, _, v) in scenarios_for_param],
                fontsize=9)
            finite_rates = [r for r in rates if np.isfinite(r)]
            ymin = max(0, min(finite_rates) - 0.06)
            ymax = min(1, max(finite_rates) + 0.08)
            ax.set_ylim(ymin, ymax)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(f"{WIN_LBLS[w]} Window",
                         fontsize=10, fontweight="bold",
                         color=WIN_COLS[w], pad=6)
            ax.set_ylabel("Cumulative rearrest rate" if wi == 0 else "",
                          fontsize=8.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")
            ax.grid(axis="y", color="#eeeeee", linewidth=0.6, zorder=0)

        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"stress_param_{pname}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────
def print_summary(df):
    base = df[df["scenario"] == "baseline"].set_index("window")

    print("\n" + "=" * 72)
    print("  GLOBAL PARAMETER STRESS TEST — RESULTS")
    print("=" * 72)
    print("\n  Calibrated baseline:")
    for pname, meta in GLOBAL_PARAMS.items():
        print(f"    {meta['short']:<6} = {meta['baseline']}")
    print()
    print(f"  {'Scenario':<28} {'3yr':>7}  {'6yr':>7}  {'9yr':>7}  "
          f"{'vs base 3yr':>12}")
    print(f"  {'-' * 65}")

    prev_param = None
    for sname, sc_meta in SCENARIOS.items():
        p = sc_meta["_param"]
        if p != prev_param:
            label = GLOBAL_PARAMS[p]["label"] if p else "Baseline"
            print(f"\n  -- {label} --")
            prev_param = p

        sub  = df[df["scenario"] == sname].set_index("window")
        vals = [sub.loc[w, "mean"] if w in sub.index else float("nan")
                for w in WINDOWS]
        b3   = base.loc[3, "mean"]
        vs_b = (vals[0] - b3) / b3 * 100 if np.isfinite(vals[0]) else float("nan")
        vs_s = f"{vs_b:+.1f}%" if np.isfinite(vs_b) and sname != "baseline" else "--"

        print(f"  {sc_meta['_label']:<28}  "
              f"{vals[0]:>6.3f}  {vals[1]:>6.3f}  {vals[2]:>6.3f}  {vs_s:>12}")

    print(f"\n  Output: {OUT_DIR}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    global N_REPS, BASE_SEEDS

    parser = argparse.ArgumentParser(
        description="Recidivism ABM — Global Parameter Stress Test")
    parser.add_argument("--reps",  type=int, default=N_REPS,
                        help="Replications per scenario (default 5)")
    parser.add_argument("--cores", type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    args = parser.parse_args()

    N_REPS     = args.reps
    BASE_SEEDS = list(range(10, 10 + N_REPS))

    n_scenarios = len(SCENARIOS)
    print(f"\nRecidivism ABM — Global Parameter Stress Test")
    print(f"  Timestamp  : {TIMESTAMP}")
    print(f"  Output     : {OUT_DIR}")
    print(f"  Scenarios  : {n_scenarios}  "
          f"(1 baseline + {len(GLOBAL_PARAMS)} params x 2 bounds = "
          f"{1 + len(GLOBAL_PARAMS)*2})")
    print(f"  Reps/scen  : {N_REPS}")
    print(f"  Total runs : {n_scenarios * N_REPS}")
    print(f"  Parameters : {', '.join(GLOBAL_PARAMS.keys())}\n")

    jobs = [
        (s * 1000 + i, sname, SCENARIOS[sname])
        for sname in SCENARIOS
        for i, s in enumerate(BASE_SEEDS)
    ]

    raw = Parallel(n_jobs=args.cores)(
        delayed(_run)(j) for j in jobs
    )

    errors = [e for _, e in raw if e]
    if errors:
        print(f"\n  {len(errors)} run(s) failed:")
        for e in errors[:5]:
            print(f"    {e[:300]}")

    df = aggregate(raw)

    csv_path = os.path.join(OUT_DIR, "stress_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")

    print("\n  Generating charts...")
    plot_stress_overview(df)
    plot_heatmap(df)
    plot_per_param(df)

    print_summary(df)


if __name__ == "__main__":
    main()