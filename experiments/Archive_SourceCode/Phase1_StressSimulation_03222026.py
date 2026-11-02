# scenario_sim.py
# ============================================================
# Recidivism ABM — Scenario Simulation
#
# Runs three scenarios side-by-side without modifying risk_config:
#
#   baseline      — calibrated settings (monitoring=0.35, contrast=0.0)
#   max_decrease  — minimum achievable rates (monitoring=0.0, contrast=0.0)
#   max_increase  — maximum achievable rates (monitoring=1.0, contrast=2.0)
#
# Outputs
# -------
#   scenario_results.csv   — rearrest rates by scenario / window / group
#   scenario_compare.png   — side-by-side bar chart across all three
#   scenario_groups.png    — subgroup breakdown per scenario
#
# Usage
# -----
#   python scenario_sim.py
#   python scenario_sim.py --reps 10 --cores 8
#   python scenario_sim.py --scenario max_increase
# ============================================================

import os, sys, datetime, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_peer_influence_config,
    get_flat_risk_weights,
    get_global_calibration_params,
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────
N_REPS     = 5
BASE_SEEDS = list(range(10, 10 + N_REPS))

NOMINAL_RUN = dict(
    initial_agents=1500, monthly_intake=10,
    warmup_months=144,   study_months=108,
    bias_factor=0,       enable_peer_influence=True,
)

GROUPS    = ["Male", "Female", "White", "Black", "Hispanic"]
WIN_FLAGS = {3: "rearrest_3_yrs", 6: "rearrest_6_yrs", 9: "rearrest_9_yrs"}

BJS = {
    3: {"Male":0.68,"Female":0.58,"White":0.66,"Black":0.72,"Hispanic":0.68},
    6: {"Male":0.79,"Female":0.70,"White":0.77,"Black":0.82,"Hispanic":0.79},
    9: {"Male":0.84,"Female":0.77,"White":0.81,"Black":0.87,"Hispanic":0.84},
}

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "CL_StressTest_Global_RiskWeights", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SCENARIOS
# Defined entirely inline — risk_config.py is never modified.
#
# Each scenario has three levers working together:
#
#   1. Risk weights (via weight_overrides)
#      Asymmetric changes shift the population mean risk score.
#      Uniform scaling cancels out in normalisation — only
#      asymmetric changes (criminogenic up, protective unchanged,
#      or vice versa) move the population mean risk_01 score.
#
#   2. Risk_Contrast_Strength
#      Amplifies the separation between low and high risk agents.
#      Must be > 0 for weight changes to have any population-level
#      effect on overall rearrest rates.
#
#   3. Supervision_Monitoring_Intensity
#      Adds a log-odds uplift for supervised agents only.
#      Effect is proportional to the fraction of agents on supervision.
#
# Analytical targets (synthetic population, sup_frac=0.35):
#   increase_25pct  →  ~+24.8%  (crim ×1.5, protect ×0.5, contrast=1.7)
#   decrease_25pct  →  ~-25.1%  (protect ×1.5, crim ×0.5, contrast=2.0)
# ─────────────────────────────────────────────────────────────
SCENARIOS = {
    "baseline": {
        # Calibrated values — reproduces BJS NCJ 252248 targets
        # 3yr ~68%  6yr ~79%  9yr ~84%
        "Supervision_Monitoring_Intensity": 0.35,
        "Risk_Contrast_Strength":           0.0,
        "weight_overrides": {},
        "label": "Baseline",
        "color": "#2980b9",
        "expected_change": "0%",
    },
    "max_increase": {
        # Analytical max before saturation (rate > 0.95): ~+55%
        # Three levers working together:
        #   1. Criminogenic weights × 3.0  — gang, priors, violations, MH
        #      amplified; shifts population mean risk score upward
        #   2. Risk_Contrast_Strength = 3.0 — amplifies the spread between
        #      low and high risk agents so the shifted mean drives rates up
        #   3. Supervision_Monitoring_Intensity = 0.35 — kept at nominal;
        #      monitoring already captured in baseline, no extra push needed
        #   4. Protective weights × 0.25 — employment and programme
        #      participation effects heavily suppressed
        "Supervision_Monitoring_Intensity": 0.35,
        "Risk_Contrast_Strength":           3.0,
        "weight_overrides": {
            # Criminogenic × 3.0
            "Gang_Affiliated":                  2.520,   # 0.84   × 3.0
            "Prior_Revocations_Supervision":    1.3014,  # 0.4338 × 3.0
            "Prior_Conviction_Episodes_Misd":   1.170,   # 0.39   × 3.0
            "Prior_Conviction_Episodes_Felony": 0.726,   # 0.242  × 3.0
            "Violations_Technical":             0.630,   # 0.21   × 3.0
            "Condition_MH_SA":                  1.050,   # 0.35   × 3.0
            # Protective × 0.25
            "Percent_Days_Employed":           -0.2375,  # -0.95  × 0.25
            "Program_Attendances":             -0.095,   # -0.38  × 0.25
        },
        "label": "Max Increase",
        "color": "#c0392b",
        "expected_change": "~+55%",
    },
    "max_decrease": {
        # Analytical max before saturation (rate < 0.15): ~-75%
        # Three levers working together:
        #   1. Protective weights × 3.0  — employment and programme
        #      effects massively amplified; shifts mean risk score down
        #   2. Risk_Contrast_Strength = 4.0 — amplifies low-risk agents'
        #      advantage so the shifted mean drives overall rates down
        #   3. Supervision_Monitoring_Intensity = 0.0 — monitoring removed
        #      so supervised agents get no detection uplift
        #   4. Criminogenic weights × 0.25 — gang, priors, violations
        #      effects heavily suppressed
        "Supervision_Monitoring_Intensity": 0.0,
        "Risk_Contrast_Strength":           4.0,
        "weight_overrides": {
            # Protective × 3.0
            "Percent_Days_Employed":           -2.850,   # -0.95  × 3.0
            "Program_Attendances":             -1.140,   # -0.38  × 3.0
            "Condition_Cog_Ed":               -0.04470,  # -0.0149× 3.0
            # Criminogenic × 0.25
            "Gang_Affiliated":                  0.210,   # 0.84   × 0.25
            "Prior_Revocations_Supervision":    0.10845, # 0.4338 × 0.25
            "Prior_Conviction_Episodes_Misd":   0.0975,  # 0.39   × 0.25
            "Prior_Conviction_Episodes_Felony": 0.0605,  # 0.242  × 0.25
            "Violations_Technical":             0.0525,  # 0.21   × 0.25
            "Condition_MH_SA":                  0.0875,  # 0.35   × 0.25
        },
        "label": "Max Decrease",
        "color": "#27ae60",
        "expected_change": "~-75%",
    },
}

# ─────────────────────────────────────────────────────────────
# RESULTS_RUN1 ROLLING CLEANUP
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# RESULTS_RUN1 ROLLING CLEANUP
# Purges CSVs from Results_Run1/ every CLEAN_EVERY_N completed
# simulations. Counter lives next to Results_Run1/ (not inside the
# timestamped OUT_DIR) so it persists correctly across runs and the
# threshold accumulates globally, not per-session.
# Uses fcntl file locking so parallel joblib workers don't double-fire.
# ─────────────────────────────────────────────────────────────
RESULTS_DIR        = os.path.join(os.path.dirname(__file__), "Results_Run1")
CLEAN_EVERY_N      = 10
_COUNTER_FILE      = os.path.join(os.path.dirname(__file__), ".sim_counter")
_COUNTER_LOCK_FILE = os.path.join(os.path.dirname(__file__), ".sim_counter.lock")


def _increment_and_maybe_clean():
    """Increment shared sim counter; purge Results_Run1 every CLEAN_EVERY_N runs."""
    import fcntl
    lock = open(_COUNTER_LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        count = 0
        if os.path.exists(_COUNTER_FILE):
            try:
                count = int(open(_COUNTER_FILE).read().strip())
            except (ValueError, OSError):
                count = 0
        count += 1
        with open(_COUNTER_FILE, "w") as f:
            f.write(str(count))
        should_clean = (count % CLEAN_EVERY_N == 0)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    if should_clean:
        _clean_results_dir(count)


def _clean_results_dir(at_count):
    """Delete all CSVs from Results_Run1/ to recover disk space."""
    if not os.path.isdir(RESULTS_DIR):
        return
    removed = freed = 0
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            freed += os.path.getsize(fpath)
            os.remove(fpath)
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"  [Results_Run1 cleanup @ sim #{at_count}] "
              f"removed {removed} CSV(s), freed {freed / 1024:.1f} KB")


# ─────────────────────────────────────────────────────────────
# MODEL RUNNER (top-level for joblib)
# ─────────────────────────────────────────────────────────────
def _run_scenario(args):
    """Run one model instance under a given scenario. Returns collected rates."""
    seed, scenario_name, scenario_overrides = args
    try:
        # ── Calibration overrides (contrast, monitoring, bias, etc.) ──
        _skip = ("label", "color", "expected_change", "weight_overrides")
        cal   = get_global_calibration_params()
        for k, v in scenario_overrides.items():
            if k not in _skip:
                cal[k] = v

        # ── Risk weight overrides ──────────────────────────────────────
        # Start from flat NIJ baseline weights.
        baseline_weights = get_flat_risk_weights()

        # Apply scenario-specific asymmetric changes (criminogenic up /
        # protective down, or vice versa) to shift the population mean.
        w_over  = scenario_overrides.get("weight_overrides", {})
        weights = {**baseline_weights, **w_over}

        # ── Inject baseline weights into calibration ───────────────────
        # compute_reference_risk_stats() in recidivism_model.py uses
        # baseline_risk_weights (if present) to compute risk_midpoint and
        # risk_spread during warmup.  Without this, scenario weights scale
        # both the raw scores AND the normalisation reference together,
        # making every z-score identical to baseline → zero effect on rates.
        cal["baseline_risk_weights"] = baseline_weights

        model = RecidivismModel(
            initial_agents        = NOMINAL_RUN["initial_agents"],
            bias_factor           = NOMINAL_RUN["bias_factor"],
            monthly_intake        = NOMINAL_RUN["monthly_intake"],
            warmup_months         = NOMINAL_RUN["warmup_months"],
            study_months          = NOMINAL_RUN["study_months"],
            enable_peer_influence = NOMINAL_RUN["enable_peer_influence"],
            weights               = weights,
            peer_config           = get_peer_influence_config(True),
            seed                  = seed,
        )
        model.apply_calibration(cal)

        while model.running:
            model.step()

        _increment_and_maybe_clean()

        # ── Collect overall rates ──
        overall = {}
        for yrs, flag in WIN_FLAGS.items():
            eligible = [a for a in model.schedule.agents
                        if getattr(a, "study_eligible_agent", False)]
            overall[yrs] = (
                sum(getattr(a, flag, False) for a in eligible) / len(eligible)
                if eligible else float("nan")
            )

        # ── Collect subgroup rates ──
        group_rates = {}
        for yrs, flag in WIN_FLAGS.items():
            eligible = [a for a in model.schedule.agents
                        if getattr(a, "study_eligible_agent", False)]
            hits = {g: 0 for g in GROUPS}
            tots = {g: 0 for g in GROUPS}
            for a in eligible:
                g = getattr(a, "Gender", None)
                r = getattr(a, "Race",   None)
                v = int(getattr(a, flag, False))
                if g in hits: hits[g] += v; tots[g] += 1
                if r in hits: hits[r] += v; tots[r] += 1
            group_rates[yrs] = {g: hits[g] / max(tots[g], 1) for g in GROUPS}

        return {
            "scenario": scenario_name,
            "seed":     seed,
            "overall":  overall,
            "groups":   group_rates,
        }, None

    except Exception as e:
        import traceback
        return None, traceback.format_exc()


# ─────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────
def aggregate(raw_results):
    """
    From a list of raw run dicts, build two DataFrames:
      overall_df  — mean/std of overall rates per scenario × window
      groups_df   — mean/std of subgroup rates per scenario × window × group
    """
    overall_rows = []
    group_rows   = []

    for r in raw_results:
        sname = r["scenario"]
        for yrs in (3, 6, 9):
            overall_rows.append({
                "scenario": sname,
                "window":   yrs,
                "rate":     r["overall"][yrs],
            })
            for grp in GROUPS:
                group_rows.append({
                    "scenario": sname,
                    "window":   yrs,
                    "group":    grp,
                    "rate":     r["groups"][yrs][grp],
                })

    overall_df = (pd.DataFrame(overall_rows)
                  .groupby(["scenario","window"])["rate"]
                  .agg(mean="mean", std="std")
                  .reset_index())

    groups_df = (pd.DataFrame(group_rows)
                 .groupby(["scenario","window","group"])["rate"]
                 .agg(mean="mean", std="std")
                 .reset_index())

    return overall_df, groups_df


# ─────────────────────────────────────────────────────────────
# STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────
STYLE = {
    "bg":        "#f8f9fa",
    "grid":      "#e0e0e0",
    "annotate":  "#2c3e50",
    "bjs_color": "#e74c3c",
}

def _apply_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent clean styling to an axes."""
    ax.set_facecolor(STYLE["bg"])
    ax.grid(axis="y", color=STYLE["grid"], linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STYLE["grid"])
    ax.spines["bottom"].set_color(STYLE["grid"])
    if title:  ax.set_title(title,  fontsize=10, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9,  labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9,  labelpad=6)
    ax.tick_params(labelsize=8)


def _annotate_bars(ax, bars, offset=0.012):
    """Print value labels centred on top of each bar."""
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h): continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"{h:.1%}",
            ha="center", va="bottom",
            fontsize=7, color=STYLE["annotate"], fontweight="bold",
        )


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
def plot_overall(overall_df, baseline_rates):
    """
    Four-panel figure:
      A. Grouped bar  — absolute rearrest rates per scenario x window
      B. Horizontal bar — % change from baseline per window
      C. Line         — rearrest trajectory across windows per scenario
      D. Bar          — gap to BJS target per scenario x window
    """
    windows   = [3, 6, 9]
    scenarios = list(SCENARIOS.keys())
    n_scen    = len(scenarios)
    n_win     = len(windows)
    width     = 0.22
    x         = np.arange(n_win)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Recidivism ABM — Scenario Comparison",
        fontsize=14, fontweight="bold", y=0.98,
    )
    axes = fig.subplot_mosaic(
        [["A", "A", "B"],
         ["C", "C", "D"]],
        gridspec_kw={"hspace": 0.42, "wspace": 0.32},
    )

    # ── A: Absolute rates grouped bar ──────────────────────────
    ax = axes["A"]
    for i, sname in enumerate(scenarios):
        scen   = SCENARIOS[sname]
        sub    = overall_df[overall_df["scenario"] == sname].set_index("window")
        means  = [sub.loc[w, "mean"] for w in windows]
        stds   = [sub.loc[w, "std"]  for w in windows]
        offset = (i - n_scen / 2 + 0.5) * width
        bars   = ax.bar(x + offset, means, width, yerr=stds,
                        label=sname.replace("_", " ").title(),
                        color=scen["color"], alpha=0.88,
                        capsize=4, error_kw={"linewidth": 1.2},
                        zorder=3)
        _annotate_bars(ax, bars)

    # BJS shaded band (min-max across subgroups) + mean dashed line
    for wi, yrs in enumerate(windows):
        bjs_vals = list(BJS[yrs].values())
        ax.fill_between(
            [wi - 0.42, wi + 0.42],
            [min(bjs_vals)] * 2, [max(bjs_vals)] * 2,
            color=STYLE["bjs_color"], alpha=0.12, zorder=1,
        )
        ax.hlines(np.mean(bjs_vals), wi - 0.42, wi + 0.42,
                  colors=STYLE["bjs_color"], linewidths=1.4,
                  linestyles="--", zorder=2,
                  label="BJS target" if wi == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(["3-Year Window", "6-Year Window", "9-Year Window"])
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    _apply_style(ax,
                 title="A.  Rearrest Rate by Scenario & Follow-up Window",
                 ylabel="Cumulative rearrest rate")

    # ── B: % change horizontal bars ────────────────────────────
    ax    = axes["B"]
    ypos  = np.arange(n_win)
    h     = 0.28
    non_base = [s for s in scenarios if s != "baseline"]
    for ci, sname in enumerate(non_base):
        sub_s = overall_df[overall_df["scenario"] == sname].set_index("window")
        sub_b = overall_df[overall_df["scenario"] == "baseline"].set_index("window")
        pcts  = [(sub_s.loc[w, "mean"] - sub_b.loc[w, "mean"])
                 / sub_b.loc[w, "mean"] * 100
                 for w in windows]
        offset = (ci - len(non_base) / 2 + 0.5) * h
        bars   = ax.barh(ypos + offset, pcts, h,
                         color=SCENARIOS[sname]["color"],
                         alpha=0.88,
                         label=sname.replace("_", " ").title(),
                         zorder=3)
        for bar, pct in zip(bars, pcts):
            ax.text(pct + (0.5 if pct >= 0 else -0.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:+.1f}%", va="center",
                    ha="left" if pct >= 0 else "right",
                    fontsize=7, fontweight="bold",
                    color=STYLE["annotate"])

    ax.axvline(0, color="#2c3e50", linewidth=1.0, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["3-Year", "6-Year", "9-Year"])
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_style(ax,
                 title="B.  % Change from Baseline",
                 xlabel="Change from baseline (%)")
    ax.grid(axis="x", color=STYLE["grid"], linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    # ── C: Trajectory lines across windows ─────────────────────
    ax = axes["C"]
    for sname in scenarios:
        scen  = SCENARIOS[sname]
        sub   = overall_df[overall_df["scenario"] == sname].set_index("window")
        m_arr = np.array([sub.loc[w, "mean"] for w in windows])
        s_arr = np.array([sub.loc[w, "std"]  for w in windows])
        lw    = 2.5 if sname == "baseline" else 2.0
        ls    = "--" if sname == "baseline" else "-"
        ax.plot(windows, m_arr, marker="o", markersize=7,
                color=scen["color"], linewidth=lw, linestyle=ls,
                label=sname.replace("_", " ").title(), zorder=3)
        ax.fill_between(windows, m_arr - s_arr, m_arr + s_arr,
                        color=scen["color"], alpha=0.10, zorder=2)
        for w, m in zip(windows, m_arr):
            ax.text(w, m + 0.016, f"{m:.1%}",
                    ha="center", fontsize=7,
                    color=scen["color"], fontweight="bold")

    bjs_means = [np.mean(list(BJS[y].values())) for y in windows]
    ax.plot(windows, bjs_means, marker="D", markersize=5,
            color=STYLE["bjs_color"], linewidth=1.4,
            linestyle=":", label="BJS Target", zorder=4)

    ax.set_xticks(windows)
    ax.set_xticklabels(["3-Year", "6-Year", "9-Year"])
    ax.set_ylim(0.40, 1.06)
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_style(ax,
                 title="C.  Rearrest Rate Trajectory Across Follow-up Windows",
                 xlabel="Follow-up window (years)",
                 ylabel="Cumulative rearrest rate")

    # ── D: Gap to BJS target ───────────────────────────────────
    ax     = axes["D"]
    gap_x  = np.arange(n_scen)
    gap_w  = 0.22
    for wi, yrs in enumerate(windows):
        bjs_mean = np.mean(list(BJS[yrs].values()))
        gaps = []
        cols = []
        for sname in scenarios:
            sub  = overall_df[overall_df["scenario"] == sname].set_index("window")
            gaps.append(sub.loc[yrs, "mean"] - bjs_mean)
            cols.append(SCENARIOS[sname]["color"])
        offset = (wi - n_win / 2 + 0.5) * gap_w
        bars   = ax.bar(gap_x + offset, gaps, gap_w,
                        color=cols, alpha=0.80, zorder=3,
                        label=f"{yrs}-Year")
        for bar, gap in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    gap + (0.006 if gap >= 0 else -0.006),
                    f"{gap:+.3f}",
                    ha="center",
                    va="bottom" if gap >= 0 else "top",
                    fontsize=6.5, fontweight="bold",
                    color=STYLE["annotate"])

    ax.axhline(0, color=STYLE["bjs_color"], linewidth=1.2,
               linestyle="--", zorder=4, label="BJS (gap=0)")
    xlabels = [s.replace("_", "\n").title() for s in scenarios]
    ax.set_xticks(gap_x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.legend(fontsize=7, framealpha=0.9)
    _apply_style(ax,
                 title="D.  Gap to BJS Benchmark\n(+ above target, - below)",
                 ylabel="Rate gap vs BJS mean")

    fig.savefig(os.path.join(OUT_DIR, "scenario_compare.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {os.path.join(OUT_DIR, 'scenario_compare.png')}")


def plot_groups(groups_df):
    """
    Two charts:
      1. Grouped bars per subgroup, one panel per window — all 3 scenarios overlaid
      2. Heatmap — subgroup x scenario gap to BJS target at 3yr
    """
    windows   = [3, 6, 9]
    scenarios = list(SCENARIOS.keys())
    n_grp     = len(GROUPS)
    x         = np.arange(n_grp)
    width     = 0.22
    n_scen    = len(scenarios)

    # ── Chart 1: Grouped bars per window ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Recidivism ABM — Subgroup Rearrest Rates by Scenario",
                 fontsize=13, fontweight="bold", y=1.01)

    for ci, yrs in enumerate(windows):
        ax = axes[ci]
        for si, sname in enumerate(scenarios):
            scen  = SCENARIOS[sname]
            sub   = groups_df[(groups_df["scenario"] == sname) &
                              (groups_df["window"]   == yrs)].set_index("group")
            means  = [sub.loc[g, "mean"] for g in GROUPS]
            stds   = [sub.loc[g, "std"]  for g in GROUPS]
            offset = (si - n_scen / 2 + 0.5) * width
            bars   = ax.bar(x + offset, means, width, yerr=stds,
                            color=scen["color"], alpha=0.85,
                            capsize=3, zorder=3,
                            label=sname.replace("_", " ").title()
                            if ci == 0 else "")
            _annotate_bars(ax, bars, offset=0.008)

        # BJS dots per group
        bjs_v = [BJS[yrs].get(g, float("nan")) for g in GROUPS]
        ax.scatter(x, bjs_v, color=STYLE["bjs_color"], zorder=5,
                   s=55, marker="D",
                   label="BJS target" if ci == 0 else "")
        for xi, bv in zip(x, bjs_v):
            ax.text(xi, bv + 0.022, f"{bv:.0%}",
                    ha="center", fontsize=6.5,
                    color=STYLE["bjs_color"], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(GROUPS, fontsize=9)
        ax.set_ylim(0, 1.14)
        _apply_style(ax,
                     title=f"{yrs}-Year Follow-up Window",
                     ylabel="Rearrest rate" if ci == 0 else None)
        if ci == 0:
            ax.legend(fontsize=8, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scenario_groups.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {os.path.join(OUT_DIR, 'scenario_groups.png')}")

    # ── Chart 2: Gap-to-BJS heatmap (3yr) ─────────────────────
    import matplotlib.colors as mcolors

    gap_data = np.zeros((len(scenarios), n_grp))
    for si, sname in enumerate(scenarios):
        sub = groups_df[(groups_df["scenario"] == sname) &
                        (groups_df["window"]   == 3)].set_index("group")
        for gi, g in enumerate(GROUPS):
            gap_data[si, gi] = sub.loc[g, "mean"] - BJS[3][g]

    vmax = max(abs(gap_data.min()), abs(gap_data.max()))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gap", ["#27ae60", "white", "#c0392b"])

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    im = ax.imshow(gap_data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_grp))
    ax.set_xticklabels(GROUPS, fontsize=10)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=10)

    for si in range(len(scenarios)):
        for gi in range(n_grp):
            gap = gap_data[si, gi]
            ax.text(gi, si, f"{gap:+.3f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if abs(gap) > vmax * 0.5 else "#2c3e50")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Rate gap vs BJS target  (+ above, - below)", fontsize=8)
    ax.set_title(
        "Subgroup Gap to BJS Benchmark — 3-Year Window\n"
        "(green = below BJS, red = above BJS)",
        fontsize=11, fontweight="bold", pad=10)
    ax.spines[:].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scenario_gap_heatmap.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {os.path.join(OUT_DIR, 'scenario_gap_heatmap.png')}")




def plot_scenario_params(overall_df):
    """
    One chart per non-baseline scenario.
    Layout: top — results summary boxes | bottom-left — global params table
                                        | bottom-right — weight changes bar chart
    """
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    BASE_W = {
        "Gang_Affiliated":                   0.84,
        "Prior_Revocations_Supervision":     0.4338,
        "Prior_Conviction_Episodes_Misd":    0.39,
        "Prior_Conviction_Episodes_Felony":  0.242,
        "Violations_Technical":              0.21,
        "Condition_MH_SA":                   0.35,
        "Percent_Days_Employed":            -0.95,
        "Program_Attendances":              -0.38,
        "Condition_Cog_Ed":                -0.0149,
        "Age_at_Release":                    0.73,
        "Dependents":                        0.1605,
        "Residence_Changes":                 0.211,
        "Supervision_Risk_Score":            0.037,
        "Prior_Conviction_Episodes_Violent": 0.086,
        "Prior_Conviction_Episodes_Property":0.027,
        "Prior_Conviction_Episodes_Drug":    0.026,
        "Condition_Other":                   0.103,
        "Program_UnexcusedAbsences":         0.11,
    }
    BASE_PARAMS = {
        "Supervision Monitoring Intensity": 0.35,
        "Risk Contrast Strength":           0.0,
    }
    PARAM_KEYS = {
        "Supervision Monitoring Intensity": "Supervision_Monitoring_Intensity",
        "Risk Contrast Strength":           "Risk_Contrast_Strength",
    }

    SHORT = {
        "Gang_Affiliated":                   "Gang Affiliated",
        "Prior_Revocations_Supervision":     "Prior Supervision Revocations",
        "Prior_Conviction_Episodes_Misd":    "Prior Convictions — Misdemeanour",
        "Prior_Conviction_Episodes_Felony":  "Prior Convictions — Felony",
        "Prior_Conviction_Episodes_Violent": "Prior Convictions — Violent",
        "Prior_Conviction_Episodes_Property":"Prior Convictions — Property",
        "Prior_Conviction_Episodes_Drug":    "Prior Convictions — Drug",
        "Violations_Technical":              "Technical Violations",
        "Condition_MH_SA":                   "Mental Health / Substance Abuse",
        "Condition_Cog_Ed":                  "Cognitive / Education Condition",
        "Condition_Other":                   "Other Condition",
        "Percent_Days_Employed":             "Employment Rate",
        "Program_Attendances":               "Programme Attendances",
        "Program_UnexcusedAbsences":         "Programme Unexcused Absences",
        "Age_at_Release":                    "Age at Release",
        "Dependents":                        "Number of Dependents",
        "Residence_Changes":                 "Residence Changes",
        "Supervision_Risk_Score":            "Supervision Risk Score",
    }

    non_base = [s for s in SCENARIOS if s != "baseline"]

    for sname in non_base:
        scen   = SCENARIOS[sname]
        w_over = scen.get("weight_overrides", {})
        color  = scen["color"]

        # ── Changed weights ──────────────────────────────────────
        changed = {}
        for k, base_v in BASE_W.items():
            scenario_v = w_over.get(k, base_v)
            if abs(scenario_v - base_v) > 1e-6:
                pct = (scenario_v - base_v) / (abs(base_v) + 1e-9) * 100
                changed[k] = {"base": base_v, "scenario": scenario_v, "pct": pct}

        # ── Global param changes ─────────────────────────────────
        param_rows = []
        for label, base_v in BASE_PARAMS.items():
            key        = PARAM_KEYS[label]
            scenario_v = scen.get(key, base_v)
            is_changed = abs(scenario_v - base_v) > 1e-6
            if base_v == 0 and scenario_v != 0:
                chg_str = f"{scenario_v:.3f} (new)"
                bg      = "#fde8e8" if scenario_v > 0 else "#e8f8e8"
            elif is_changed:
                pct     = (scenario_v - base_v) / (abs(base_v) + 1e-9) * 100
                chg_str = f"{pct:+.0f}%"
                bg      = "#fde8e8" if pct > 0 else "#e8f8e8"
            else:
                chg_str = "unchanged"
                bg      = "#f8f9fa"
            param_rows.append({
                "label": label, "base": base_v,
                "value": scenario_v, "chg": chg_str, "bg": bg,
            })

        # ── Actual simulation results ────────────────────────────
        result_pcts = {}
        result_rates = {}
        if overall_df is not None:
            sub_s = overall_df[overall_df["scenario"] == sname].set_index("window")
            sub_b = overall_df[overall_df["scenario"] == "baseline"].set_index("window")
            for w in [3, 6, 9]:
                if w in sub_s.index and w in sub_b.index:
                    rm = sub_s.loc[w, "mean"]
                    bm = sub_b.loc[w, "mean"]
                    result_pcts[w]  = (rm - bm) / bm * 100
                    result_rates[w] = rm

        # ── Figure layout ────────────────────────────────────────
        n_w      = len(changed)
        fig_h    = max(10, 4 + n_w * 0.52)
        fig      = plt.figure(figsize=(20, fig_h))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Parameter Breakdown — {sname.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold", y=0.99,
        )

        gs = GridSpec(
            2, 2,
            figure=fig,
            height_ratios=[1, max(3, n_w * 0.55)],
            width_ratios=[1, 2],
            hspace=0.38, wspace=0.38,
        )

        # ── TOP ROW: results summary boxes ──────────────────────
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.axis("off")
        ax_top.set_facecolor("white")

        if result_pcts:
            box_w = 0.18
            starts = [0.08, 0.30, 0.52]
            for i, (w, pct) in enumerate(result_pcts.items()):
                rate = result_rates[w]
                bkg  = "#fde8e8" if pct > 0 else "#e8f8e8"
                txt_col = "#c0392b" if pct > 0 else "#27ae60"
                rect = mpatches.FancyBboxPatch(
                    (starts[i], 0.05), box_w, 0.82,
                    boxstyle="round,pad=0.02",
                    facecolor=bkg, edgecolor=txt_col,
                    linewidth=1.5, transform=ax_top.transAxes, zorder=2,
                )
                ax_top.add_patch(rect)
                ax_top.text(starts[i] + box_w/2, 0.72, f"{w}-Year Window",
                            ha="center", va="center", fontsize=10,
                            fontweight="bold", color="#2c3e50",
                            transform=ax_top.transAxes)
                ax_top.text(starts[i] + box_w/2, 0.46, f"{pct:+.1f}%",
                            ha="center", va="center", fontsize=18,
                            fontweight="bold", color=txt_col,
                            transform=ax_top.transAxes)
                ax_top.text(starts[i] + box_w/2, 0.22,
                            f"Rate: {rate:.3f}",
                            ha="center", va="center", fontsize=9,
                            color="#555", transform=ax_top.transAxes)
            ax_top.text(0.78, 0.50,
                        "vs Baseline\n(actual simulation)",
                        ha="center", va="center", fontsize=9,
                        color="#7f8c8d", style="italic",
                        transform=ax_top.transAxes)
        else:
            ax_top.text(0.5, 0.5, "Run with --scenario all to see results",
                        ha="center", va="center", fontsize=11,
                        color="#7f8c8d", transform=ax_top.transAxes)

        # ── BOTTOM LEFT: global params table ────────────────────
        ax_l = fig.add_subplot(gs[1, 0])
        ax_l.axis("off")
        ax_l.set_facecolor("white")

        col_labels = ["Parameter", "Baseline", "Scenario", "Change"]
        t_data  = [[r["label"], f"{r['base']:.3f}", f"{r['value']:.3f}", r["chg"]]
                   for r in param_rows]
        t_color = [[r["bg"]] * 4 for r in param_rows]

        tbl = ax_l.table(
            cellText    = t_data,
            colLabels   = col_labels,
            cellLoc     = "center",
            loc         = "center",
            cellColours = t_color,
            bbox        = [0.02, 0.2, 0.96, 0.65],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
            if c == 0 and r > 0:
                cell.set_text_props(ha="left")
        tbl.auto_set_column_width([0, 1, 2, 3])

        ax_l.set_title("Global Parameters", fontsize=11,
                       fontweight="bold", pad=8, loc="left")

        # ── BOTTOM RIGHT: weight changes bar chart ───────────────
        ax_r = fig.add_subplot(gs[1, 1])
        ax_r.set_facecolor(STYLE["bg"])

        if changed:
            # Sort: protective changes first (most negative pct), then criminogenic
            sorted_items = sorted(changed.items(), key=lambda x: x[1]["pct"])
            labels  = [SHORT.get(k, k)  for k, _ in sorted_items]
            pcts    = [v["pct"]         for _, v in sorted_items]
            base_vs = [v["base"]        for _, v in sorted_items]
            scen_vs = [v["scenario"]    for _, v in sorted_items]
            keys    = [k                for k, _ in sorted_items]

            bar_cols = []
            for k, p in zip(keys, pcts):
                bv = BASE_W[k]
                if bv >= 0:
                    bar_cols.append("#c0392b" if p > 0 else "#27ae60")
                else:
                    bar_cols.append("#27ae60" if p < 0 else "#c0392b")

            y    = np.arange(len(labels))
            bars = ax_r.barh(y, pcts, color=bar_cols, alpha=0.85,
                             height=0.60, zorder=3)
            ax_r.axvline(0, color="#2c3e50", linewidth=1.5, zorder=4)

            # Clamp x-axis to ±max(|pct|) + 20% padding
            max_abs = max(abs(p) for p in pcts)
            pad     = max_abs * 0.22
            ax_r.set_xlim(-max_abs - pad, max_abs + pad)

            for bar, p, bv, sv in zip(bars, pcts, base_vs, scen_vs):
                # Always place label outside bar end for legibility
                bar_end = p
                if p >= 0:
                    x_txt = bar_end + max_abs * 0.02
                    ha    = "left"
                else:
                    x_txt = bar_end - max_abs * 0.02
                    ha    = "right"
                ax_r.text(
                    x_txt,
                    bar.get_y() + bar.get_height() / 2,
                    f"{p:+.0f}%   {bv:.3f} → {sv:.3f}",
                    va="center", ha=ha, fontsize=8.5,
                    color=STYLE["annotate"], fontweight="bold",
                )

            ax_r.set_yticks(y)
            ax_r.set_yticklabels(labels, fontsize=9.5)
            ax_r.set_xlabel("% change from baseline NIJ weight", fontsize=9.5, labelpad=8)
            ax_r.grid(axis="x", color=STYLE["grid"], linewidth=0.8, zorder=0)
            ax_r.spines["top"].set_visible(False)
            ax_r.spines["right"].set_visible(False)
            ax_r.spines["left"].set_color(STYLE["grid"])
            ax_r.tick_params(axis="y", labelsize=9.5)

            red_p   = mpatches.Patch(color="#c0392b", alpha=0.85,
                                     label="Criminogenic amplified  /  Protective weakened")
            green_p = mpatches.Patch(color="#27ae60", alpha=0.85,
                                     label="Protective amplified  /  Criminogenic weakened")
            ax_r.legend(handles=[red_p, green_p], fontsize=8.5,
                        loc="lower right", framealpha=0.95,
                        edgecolor="#cccccc")
        else:
            ax_r.text(0.5, 0.5, "No weight changes in this scenario",
                      ha="center", va="center", fontsize=11,
                      color="#7f8c8d", transform=ax_r.transAxes)
            ax_r.axis("off")

        ax_r.set_title("Risk Weight Changes  (baseline → scenario value)",
                       fontsize=11, fontweight="bold", pad=8, loc="left")

        fname = f"scenario_params_{sname}.png"
        fig.savefig(os.path.join(OUT_DIR, fname),
                    dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  -> {os.path.join(OUT_DIR, fname)}")



# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────
def print_summary(overall_df):
    print("\n" + "="*65)
    print("SCENARIO SIMULATION — RESULTS SUMMARY")
    print("="*65)
    print(f"  {'Scenario':<16} {'Window':>7}  {'Rate':>7}  {'vs Baseline':>12}  {'vs BJS':>8}")
    print("-"*65)

    base = overall_df[overall_df["scenario"] == "baseline"].set_index("window")

    for sname in SCENARIOS:
        sub = overall_df[overall_df["scenario"] == sname].set_index("window")
        for yrs in (3, 6, 9):
            rate     = sub.loc[yrs, "mean"]
            std      = sub.loc[yrs, "std"]
            base_r   = base.loc[yrs, "mean"]
            bjs_r    = np.mean(list(BJS[yrs].values()))
            vs_base  = (rate - base_r) / base_r * 100
            vs_bjs   = rate - bjs_r
            b_str    = f"{vs_base:+.1f}%" if sname != "baseline" else "  —"
            print(f"  {sname:<16} {yrs:>4}yr  {rate:>6.3f}  "
                  f"{b_str:>12}  {vs_bjs:>+7.3f}")
        print()

    print(f"  Output: {OUT_DIR}")
    print("="*65)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    global N_REPS, BASE_SEEDS

    parser = argparse.ArgumentParser(
        description="Recidivism ABM — Scenario Simulation")
    parser.add_argument("--scenario", default="all",
                        choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--reps",  type=int, default=N_REPS)
    args = parser.parse_args()

    N_REPS     = args.reps
    BASE_SEEDS = list(range(10, 10 + N_REPS))

    scenarios_to_run = (list(SCENARIOS.keys()) if args.scenario == "all"
                        else [args.scenario])

    print(f"\nRecidivism ABM — Scenario Simulation")
    print(f"Timestamp:  {TIMESTAMP}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Scenarios:  {scenarios_to_run}")
    print(f"Reps: {N_REPS}  |  Cores: {args.cores}\n")

    for sname in scenarios_to_run:
        scen = SCENARIOS[sname]
        print(f"Running: {sname}  (expected {scen['expected_change']})")
        print(f"  Supervision_Monitoring_Intensity = "
              f"{scen['Supervision_Monitoring_Intensity']}")
        print(f"  Risk_Contrast_Strength           = "
              f"{scen['Risk_Contrast_Strength']}")

    # Build jobs across all scenarios × seeds
    jobs = [
        (s * 1000 + i, sname, SCENARIOS[sname])
        for sname in scenarios_to_run
        for i, s in enumerate(BASE_SEEDS)
    ]

    raw = Parallel(n_jobs=args.cores)(
        delayed(_run_scenario)(j) for j in jobs
    )

    results    = [r for r, e in raw if r]
    errors     = [e for r, e in raw if e]
    if errors:
        print(f"\n  ⚠️  {len(errors)} run(s) failed:")
        for e in errors[:3]:
            print(f"    {e[:200]}")

    overall_df, groups_df = aggregate(results)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "scenario_results.csv")
    overall_df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")

    baseline_rates = (overall_df[overall_df["scenario"] == "baseline"]
                      .set_index("window")["mean"].to_dict()
                      if "baseline" in scenarios_to_run else {})

    # Only plot if more than one scenario was run
    if len(scenarios_to_run) > 1 and "baseline" in scenarios_to_run:
        plot_overall(overall_df, baseline_rates)
        plot_groups(groups_df)

    # Parameter breakdown — always generated for any non-baseline scenario
    plot_scenario_params(overall_df if len(scenarios_to_run) > 1 else None)

    print_summary(overall_df)


if __name__ == "__main__":
    main()