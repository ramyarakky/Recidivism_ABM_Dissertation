# cal_stress_test.py
# ============================================================
# Recidivism ABM — Calibration Parameter Stress Test
#
# What it does
# ------------
# Runs the model 15 times: once at baseline, then with each
# calibration parameter pushed to its LOW and HIGH stress bound
# (one at a time), then with all six pushed simultaneously.
#
# Why calibration-only
# --------------------
# NIJ risk weights are published empirical coefficients — not
# free parameters. Only the six values we calibrated are tested.
#
# Outputs (each fits one A3 landscape page)
# ------------------------------------------
#   stress_results.csv      — raw rates for all scenarios/windows
#   stress_3yr.png          — absolute rates, 3-year window
#   stress_6yr.png          — absolute rates, 6-year window
#   stress_9yr.png          — absolute rates, 9-year window
#   stress_summary.png      — all three windows side by side
#
# Each chart shows for every scenario:
#   • All six parameter values (changed one highlighted in bold)
#   • Inline rate bar with BJS target tick
#   • Gap to BJS benchmark (+/- percentage points)
#
# Usage
# -----
#   python cal_stress_test.py
#   python cal_stress_test.py --reps 10 --cores 8
# ============================================================

import os, sys, argparse, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm_phase2.model.recidivism_model import RecidivismModel
from recidivism_abm_phase2.config.risk_config import (
    get_peer_influence_config,
    get_flat_risk_weights,
    get_global_calibration_params,
)

# =============================================================================
# CALIBRATED BASELINE VALUES  (loaded live from risk_config)
# =============================================================================
_CAL = get_global_calibration_params()

SMI  = float(_CAL["Supervision_Monitoring_Intensity"])        # 0.35
RCS  = float(_CAL["Risk_Contrast_Strength"])                  # 0.00
RED3 = float(_CAL["Risk_Effect_Decay_After_3Y"])              # 0.90
RED6 = float(_CAL["Risk_Effect_Decay_After_6Y"])              # 0.70
SMD3 = float(_CAL["Supervision_Monitoring_Decay_After_3Y"])   # 0.35
SMD6 = float(_CAL["Supervision_Monitoring_Decay_After_6Y"])   # 0.20

# Parameter names and their baseline values — used for changed-param detection
_PKEYS    = ["SMI", "RCS", "RED3", "RED6", "SMD3", "SMD6"]
_PBASE    = [SMI, RCS, RED3, RED6, SMD3, SMD6]

# =============================================================================
# STRESS SCENARIOS
#
# Flat list — each row is one scenario.
# Columns:  label, SMI, RCS, RED3, RED6, SMD3, SMD6
#
# Stress bounds:
#   SMI  : 0.00 (no monitoring)      → 1.00 (maximum monitoring)
#   RCS  : 0.00 (no amplification)   → 4.00 (strong amplification)
#   RED3 : 0.45 (fast desistance)    → 1.00 (no desistance)
#   RED6 : 0.35 (fast desistance)    → 1.00 (no desistance)
#   SMD3 : 0.10 (fades fast)         → 0.70 (persists)
#   SMD6 : 0.05 (fades fast)         → 0.40 (persists)
# =============================================================================

SCENARIOS = [
    #  label                          SMI   RCS   RED3  RED6  SMD3  SMD6
    ("Baseline",                      SMI,  RCS,  RED3, RED6, SMD3, SMD6),

    # ── LOW bounds — one parameter at a time ────────────────────────────────
    ("SMI Low  (0.00)",               0.00, RCS,  RED3, RED6, SMD3, SMD6),
    ("RCS Low  (0.00)",               SMI,  0.00, RED3, RED6, SMD3, SMD6),
    ("RED3 Low (0.45)",               SMI,  RCS,  0.45, RED6, SMD3, SMD6),
    ("RED6 Low (0.35)",               SMI,  RCS,  RED3, 0.35, SMD3, SMD6),
    ("SMD3 Low (0.10)",               SMI,  RCS,  RED3, RED6, 0.10, SMD6),
    ("SMD6 Low (0.05)",               SMI,  RCS,  RED3, RED6, SMD3, 0.05),

    # ── HIGH bounds — one parameter at a time ───────────────────────────────
    ("SMI High  (1.00)",              1.00, RCS,  RED3, RED6, SMD3, SMD6),
    ("RCS High  (4.00)",              SMI,  4.00, RED3, RED6, SMD3, SMD6),
    ("RED3 High (1.00)",              SMI,  RCS,  1.00, RED6, SMD3, SMD6),
    ("RED6 High (1.00)",              SMI,  RCS,  RED3, 1.00, SMD3, SMD6),
    ("SMD3 High (0.70)",              SMI,  RCS,  RED3, RED6, 0.70, SMD6),
    ("SMD6 High (0.40)",              SMI,  RCS,  RED3, RED6, SMD3, 0.40),

    # ── Combined ─────────────────────────────────────────────────────────────
    ("All LOW  (best case)",          0.00, 0.00, 0.45, 0.35, 0.10, 0.05),
    ("All HIGH (worst case)",         1.00, 4.00, 1.00, 1.00, 0.70, 0.40),
]

# Colour scheme
def _colour(label):
    if "Baseline"  in label: return "#555555"
    if "All LOW"   in label: return "#1a7a4a"
    if "All HIGH"  in label: return "#a61c00"
    if " Low"      in label: return "#27ae60"
    if " High"     in label: return "#c0392b"
    return "#888888"

COLOURS = [_colour(s[0]) for s in SCENARIOS]

# =============================================================================
# RUN CONFIG
# =============================================================================
N_REPS    = 5
SEEDS     = [42, 137, 251, 389, 503]
WIN_FLAGS = {3: "rearrest_3_yrs", 6: "rearrest_6_yrs", 9: "rearrest_9_yrs"}
BJS_MEAN  = {3: 0.68, 6: 0.79, 9: 0.83}

RUN_CFG = dict(
    initial_agents=1000, monthly_intake=10,
    warmup_months=144,   study_months=108,
    bias_factor=0,       enable_peer_influence=True,
)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "CL_stress_output", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# MODEL RUNNER
# =============================================================================
def _run(args):
    label, smi, rcs, red3, red6, smd3, smd6, seed = args
    try:
        cal = get_global_calibration_params()
        cal.update({
            "Supervision_Monitoring_Intensity":      smi,
            "Risk_Contrast_Strength":                rcs,
            "Risk_Effect_Decay_After_3Y":            red3,
            "Risk_Effect_Decay_After_6Y":            red6,
            "Supervision_Monitoring_Decay_After_3Y": smd3,
            "Supervision_Monitoring_Decay_After_6Y": smd6,
        })
        model = RecidivismModel(
            initial_agents        = RUN_CFG["initial_agents"],
            bias_factor           = RUN_CFG["bias_factor"],
            monthly_intake        = RUN_CFG["monthly_intake"],
            warmup_months         = RUN_CFG["warmup_months"],
            study_months          = RUN_CFG["study_months"],
            enable_peer_influence = RUN_CFG["enable_peer_influence"],
            weights               = get_flat_risk_weights(),
            peer_config           = get_peer_influence_config(True),
            calibration_params    = cal,   # ← injected before create_agent()
            seed                  = seed,
        )
        #model.apply_calibration(cal)
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
        return {"label": label, "seed": seed,
                "rate_3": rates[3], "rate_6": rates[6], "rate_9": rates[9]}
    except Exception as e:
        print(f"  Error [{label} seed={seed}]: {e}")
        return None

# =============================================================================
# AGGREGATE
# =============================================================================
def aggregate(raw):
    rows = []
    for sc in SCENARIOS:
        label = sc[0]
        runs  = [r for r in raw if r and r["label"] == label]
        for yrs in (3, 6, 9):
            vals = [r[f"rate_{yrs}"] for r in runs
                    if r and np.isfinite(r[f"rate_{yrs}"])]
            rows.append({
                "scenario": label,
                "window":   yrs,
                "mean":     float(np.mean(vals)) if vals else float("nan"),
                "std":      float(np.std(vals))  if len(vals) > 1 else 0.0,
            })
    return pd.DataFrame(rows)

# =============================================================================
# SHARED CHART HELPERS
# =============================================================================

# Group separator positions (between rows 0–1, 7–8, 13–14 in SCENARIOS list)
_SEP_ROWS   = [0.5, 6.5, 12.5]
_GROUP_INFO = [
    (0,    0,    "Baseline",   "#555555"),
    (1,    6,    "LOW bounds", "#27ae60"),
    (7,    12,   "HIGH bounds","#c0392b"),
    (13,   14,   "Combined",   "#555555"),
]


def _param_display(sc_tuple):
    """
    Returns a list of (text, is_changed) tuples for the six parameters.
    Used to render changed params in a different style.
    """
    result = []
    for i, key in enumerate(_PKEYS):
        val     = sc_tuple[i + 1]
        changed = abs(val - _PBASE[i]) > 1e-6
        result.append((f"{key}={val:.2f}", changed))
    return result


def _draw_bar_cell(ax, y_row, rate, bjs, color,
                   x_start=0.62, width=0.22, height=0.50):
    """
    Draw a filled horizontal bar at data-space row y_row.
    The bar spans [x_start, x_start+width] in axes-x space.
    Fill represents the rate. Red tick marks the BJS target.
    """
    # Background track
    ax.barh(y_row, width, left=x_start, height=height,
            color="#eeeeee", zorder=2, linewidth=0,
            align="center")
    # Rate fill (capped at full width)
    fill = min(rate, 1.0) * width
    ax.barh(y_row, fill, left=x_start, height=height,
            color=color, alpha=0.80, zorder=3, linewidth=0,
            align="center")
    # BJS tick
    bjs_x = x_start + bjs * width
    ax.plot([bjs_x, bjs_x],
            [y_row - height * 0.55, y_row + height * 0.55],
            color="#e74c3c", linewidth=2.0, zorder=4, solid_capstyle="butt")


def _single_window_page(df, window_yrs):
    """
    One full page for one output window.
    Rows = scenarios (top = Baseline, bottom = All HIGH).
    Columns:
        Col A (left):   colour dot + scenario label
        Col B:          all six parameter values, changed one in bold
        Col C:          inline rate bar with BJS tick
        Col D:          rate value (numeric)
        Col E:          gap vs BJS (+/- pp)
    """
    n      = len(SCENARIOS)
    labels = [s[0] for s in SCENARIOS]
    bjs    = BJS_MEAN[window_yrs]
    sub    = df[df["window"] == window_yrs].set_index("scenario")

    # A4 landscape: 11.69 x 8.27 in  — tall enough for 15 rows
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n + 0.8)
    ax.axis("off")

    # ── Title block ───────────────────────────────────────────────────────────
    fig.text(0.50, 0.97,
             f"Calibration Stress Test — {window_yrs}-Year Rearrest Rate",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color="#1a3d5c")
    fig.text(0.50, 0.94,
             f"BJS benchmark: {bjs:.0%}  |  "
             f"Baseline: SMI={SMI:.2f}  RCS={RCS:.2f}  RED3={RED3:.2f}  "
             f"RED6={RED6:.2f}  SMD3={SMD3:.2f}  SMD6={SMD6:.2f}  |  "
             "Bold = changed parameter",
             ha="center", va="top", fontsize=8.5, color="#555555")

    # ── Column headers ────────────────────────────────────────────────────────
    header_y = n + 0.35
    ax.text(0.01, header_y, "Scenario",
            fontsize=9, fontweight="bold", color="#333333", va="center")
    ax.text(0.24, header_y, "Parameter values  (bold = changed from baseline)",
            fontsize=9, fontweight="bold", color="#333333", va="center")
    ax.text(0.64, header_y, f"Rate (BJS={bjs:.0%} = red tick)",
            fontsize=9, fontweight="bold", color="#333333", va="center")
    ax.text(0.88, header_y, "Rate",
            fontsize=9, fontweight="bold", color="#333333", va="center")
    ax.text(0.94, header_y, "Gap vs BJS",
            fontsize=9, fontweight="bold", color="#333333", va="center")
    ax.axhline(n + 0.10, color="#cccccc", linewidth=0.8)

    # ── Group separators and labels ───────────────────────────────────────────
    for sep in _SEP_ROWS:
        ax.axhline(sep, color="#e8e8e8", linewidth=1.2, linestyle="--")

    group_labels = {
        0:  ("Baseline",    "#555555"),
        1:  ("LOW bounds",  "#27ae60"),
        7:  ("HIGH bounds", "#c0392b"),
        13: ("Combined",    "#444444"),
    }

    # ── Draw each scenario row ────────────────────────────────────────────────
    for i, sc in enumerate(SCENARIOS):
        lbl   = sc[0]
        color = COLOURS[i]
        # Rows drawn top-to-bottom: row index i maps to y = (n-1-i)
        y = (n - 1 - i)

        # Alternate row background
        if i % 2 == 0:
            ax.barh(y, 1.0, left=0, height=0.90,
                    color="#f9f9f9", zorder=0, linewidth=0, align="center")

        # Group label on left margin at first row of each group
        if i in group_labels:
            grp_lbl, grp_col = group_labels[i]
            ax.text(-0.005, y, grp_lbl,
                    fontsize=7.5, fontweight="bold", color=grp_col,
                    va="center", ha="right", clip_on=False)

        # Col A — colour dot + scenario name
        ax.plot(0.015, y, "s", color=color, markersize=7, zorder=5)
        ax.text(0.030, y, lbl,
                fontsize=8.5, fontweight="bold", color=color,
                va="center", ha="left")

        # Col B — parameter values, changed one in bold + red
        param_tokens = _param_display(sc)
        x_cur = 0.24
        for token_txt, is_changed in param_tokens:
            fc = "#c0392b" if is_changed else "#aaaaaa"
            fw = "bold"   if is_changed else "normal"
            ax.text(x_cur, y, token_txt,
                    fontsize=7.5, color=fc, fontweight=fw,
                    va="center", ha="left",
                    fontfamily="monospace")
            x_cur += 0.063   # fixed-width spacing

        # Col C — inline rate bar
        rate = sub.loc[lbl, "mean"] if lbl in sub.index else float("nan")
        std  = sub.loc[lbl, "std"]  if lbl in sub.index else 0.0
        if np.isfinite(rate):
            _draw_bar_cell(ax, y, rate, bjs, color,
                           x_start=0.62, width=0.22, height=0.52)

            # Col D — numeric rate
            ax.text(0.86, y, f"{rate:.3f}",
                    fontsize=8.5, fontweight="bold", color=color,
                    va="center", ha="left", fontfamily="monospace")

            # Col E — gap vs BJS
            gap     = (rate - bjs) * 100
            gap_str = f"{gap:+.1f} pp"
            gap_col = "#c0392b" if gap > 0.5 else ("#27ae60" if gap < -0.5 else "#888888")
            ax.text(0.945, y, gap_str,
                    fontsize=8.5, fontweight="bold", color=gap_col,
                    va="center", ha="left")

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color="#555555", label="Baseline"),
        mpatches.Patch(color="#27ae60", label="LOW bound (OAT)"),
        mpatches.Patch(color="#c0392b", label="HIGH bound (OAT)"),
        mpatches.Patch(color="#1a7a4a", label="All LOW — best case"),
        mpatches.Patch(color="#a61c00", label="All HIGH — worst case"),
        mpatches.Patch(color="#e74c3c", label="BJS target (red tick)"),
    ]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=6,
              fontsize=8, framealpha=0.9)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.06)
    path = os.path.join(OUT_DIR, f"stress_{window_yrs}yr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def _summary_page(df):
    """
    Summary page — all three windows side by side.
    Rows = scenarios.  Three bar columns: 3yr | 6yr | 9yr.
    Each cell shows a mini bar with BJS tick + numeric rate + gap badge.
    """
    n      = len(SCENARIOS)
    labels = [s[0] for s in SCENARIOS]
    wins   = [3, 6, 9]
    wcolors = {3: "#2E75B6", 6: "#70AD47", 9: "#ED7D31"}

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n + 1.0)
    ax.axis("off")

    # Title
    fig.text(0.50, 0.97,
             "Calibration Stress Test — Summary: All Three Windows",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color="#1a3d5c")
    fig.text(0.50, 0.94,
             f"Baseline: SMI={SMI:.2f}  RCS={RCS:.2f}  RED3={RED3:.2f}  "
             f"RED6={RED6:.2f}  SMD3={SMD3:.2f}  SMD6={SMD6:.2f}  |  "
             "Bold = changed parameter  |  Red tick = BJS target",
             ha="center", va="top", fontsize=8.5, color="#555555")

    # Column layout
    # Scenario label: 0.00 – 0.19
    # Changed params: 0.20 – 0.31
    # 3yr block:      0.32 – 0.52
    # 6yr block:      0.53 – 0.73
    # 9yr block:      0.74 – 0.94
    win_x  = {3: 0.32, 6: 0.53, 9: 0.74}
    bar_w  = 0.14   # width of mini bar within each window block

    # Column headers
    header_y = n + 0.50
    ax.text(0.01,  header_y, "Scenario",
            fontsize=9, fontweight="bold", color="#333333", va="center")
    ax.text(0.20,  header_y, "Changed param",
            fontsize=9, fontweight="bold", color="#333333", va="center")
    for w in wins:
        bjs = BJS_MEAN[w]
        ax.text(win_x[w] + bar_w / 2, header_y + 0.25,
                f"{w}-Year  (BJS {bjs:.0%})",
                fontsize=9, fontweight="bold",
                color=wcolors[w], va="center", ha="center")
        ax.text(win_x[w] + bar_w + 0.003, header_y,
                "Rate",
                fontsize=8, color="#555555", va="center")
        ax.text(win_x[w] + bar_w + 0.04,  header_y,
                "Gap",
                fontsize=8, color="#555555", va="center")

    ax.axhline(n + 0.22, color="#cccccc", linewidth=0.8)

    # Group separators
    for sep in _SEP_ROWS:
        ax.axhline(sep, color="#e8e8e8", linewidth=1.2, linestyle="--")

    group_labels = {
        0:  ("Baseline",    "#555555"),
        1:  ("LOW",         "#27ae60"),
        7:  ("HIGH",        "#c0392b"),
        13: ("Combined",    "#444444"),
    }

    for i, sc in enumerate(SCENARIOS):
        lbl   = sc[0]
        color = COLOURS[i]
        y     = n - 1 - i

        # Alternate row shade
        if i % 2 == 0:
            ax.barh(y, 1.0, left=0, height=0.88,
                    color="#f9f9f9", zorder=0, linewidth=0, align="center")

        # Group label
        if i in group_labels:
            grp_lbl, grp_col = group_labels[i]
            ax.text(-0.005, y, grp_lbl,
                    fontsize=7.5, fontweight="bold", color=grp_col,
                    va="center", ha="right", clip_on=False)

        # Scenario name
        ax.plot(0.015, y, "s", color=color, markersize=6, zorder=5)
        ax.text(0.030, y, lbl,
                fontsize=8, fontweight="bold", color=color,
                va="center", ha="left")

        # Changed params (compact)
        changed = [f"{_PKEYS[j]}={sc[j+1]:.2f}"
                   for j in range(6) if abs(sc[j+1] - _PBASE[j]) > 1e-6]
        chg_str = "  ".join(changed) if changed else "—"
        ax.text(0.20, y, chg_str,
                fontsize=7, color="#c0392b", fontweight="bold",
                va="center", ha="left", fontfamily="monospace")

        # Three window bars
        for w in wins:
            bjs    = BJS_MEAN[w]
            sub_w  = df[df["window"] == w].set_index("scenario")
            rate   = sub_w.loc[lbl, "mean"] if lbl in sub_w.index else float("nan")
            xstart = win_x[w]
            wc     = wcolors[w]

            if np.isfinite(rate):
                _draw_bar_cell(ax, y, rate, bjs, wc,
                               x_start=xstart, width=bar_w, height=0.46)
                # Numeric rate
                ax.text(xstart + bar_w + 0.005, y,
                        f"{rate:.3f}",
                        fontsize=7.5, fontweight="bold", color=wc,
                        va="center", ha="left", fontfamily="monospace")
                # Gap badge
                gap     = (rate - bjs) * 100
                gap_str = f"{gap:+.1f}pp"
                gap_col = "#c0392b" if gap > 0.5 else ("#27ae60" if gap < -0.5 else "#888888")
                ax.text(xstart + bar_w + 0.043, y,
                        gap_str,
                        fontsize=7.5, fontweight="bold", color=gap_col,
                        va="center", ha="left")

    # Legend
    patches = [
        mpatches.Patch(color="#555555", label="Baseline"),
        mpatches.Patch(color="#27ae60", label="LOW bound (OAT)"),
        mpatches.Patch(color="#c0392b", label="HIGH bound (OAT)"),
        mpatches.Patch(color="#1a7a4a", label="All LOW — best case"),
        mpatches.Patch(color="#a61c00", label="All HIGH — worst case"),
        mpatches.Patch(color="#e74c3c", label="BJS target (red tick)"),
    ]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=6,
              fontsize=8, framealpha=0.9)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.06)
    path = os.path.join(OUT_DIR, "stress_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
def print_summary(df):
    base = df[df["scenario"] == "Baseline"].set_index("window")

    print("\n" + "=" * 70)
    print("  CALIBRATION STRESS TEST — RESULTS")
    print("=" * 70)
    print(f"\n  Baseline (from risk_config.py):")
    print(f"    SMI={SMI:.4f}  RCS={RCS:.4f}  RED3={RED3:.4f}  "
          f"RED6={RED6:.4f}  SMD3={SMD3:.4f}  SMD6={SMD6:.4f}")
    print()
    print(f"  {'Scenario':<26}  {'3yr':>7}  {'6yr':>7}  {'9yr':>7}  "
          f"{'vs base 3yr':>12}")
    print(f"  {'-'*62}")

    prev_group = None
    for sc in SCENARIOS:
        lbl = sc[0]
        if " Low"  in lbl and prev_group != "low":
            print(f"  {'— LOW bounds ':—<62}"); prev_group = "low"
        elif " High" in lbl and prev_group != "high":
            print(f"  {'— HIGH bounds ':—<62}"); prev_group = "high"
        elif "All"  in lbl and prev_group != "combined":
            print(f"  {'— Combined ':—<62}"); prev_group = "combined"
        elif lbl == "Baseline":
            prev_group = "baseline"

        sub   = df[df["scenario"] == lbl].set_index("window")
        rates = [sub.loc[y, "mean"] if y in sub.index else float("nan")
                 for y in [3, 6, 9]]
        b3    = base.loc[3, "mean"]
        vs_b  = (((rates[0] - b3) / b3 * 100)
                 if b3 and np.isfinite(rates[0]) else float("nan"))
        vs_s  = f"{vs_b:+.1f}%" if np.isfinite(vs_b) and lbl != "Baseline" else "—"
        print(f"  {lbl:<26}  {rates[0]:>6.3f}  {rates[1]:>6.3f}  "
              f"{rates[2]:>6.3f}  {vs_s:>12}")

    print(f"\n  Output: {OUT_DIR}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================
def main():
    global N_REPS, SEEDS

    parser = argparse.ArgumentParser(
        description="Recidivism ABM — Calibration Stress Test")
    parser.add_argument("--reps",  type=int, default=N_REPS,
                        help="Replications per scenario (default 5)")
    parser.add_argument("--cores", type=int, default=-1,
                        help="Parallel workers  (-1 = all cores)")
    args = parser.parse_args()

    N_REPS = args.reps
    SEEDS  = list(range(42, 42 + N_REPS))

    print(f"\nCalibration Stress Test")
    print(f"  Scenarios  : {len(SCENARIOS)}  "
          f"(1 baseline + 12 OAT + 2 combined)")
    print(f"  Reps/scen  : {N_REPS}")
    print(f"  Total runs : {len(SCENARIOS) * N_REPS}")
    print(f"  Output     : {OUT_DIR}\n")

    jobs = [(*sc, seed) for sc in SCENARIOS for seed in SEEDS]
    raw  = Parallel(n_jobs=args.cores)(delayed(_run)(j) for j in jobs)

    df = aggregate([r for r in raw if r])

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "stress_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")

    # Four charts — each one page
    print("\n  Generating charts...")
    _single_window_page(df, 3)
    _single_window_page(df, 6)
    _single_window_page(df, 9)
    _summary_page(df)

    print_summary(df)


if __name__ == "__main__":
    main()