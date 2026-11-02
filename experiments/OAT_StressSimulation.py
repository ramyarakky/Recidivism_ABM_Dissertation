#!/usr/bin/env python3
"""
cal_stress_test.py  —  Recidivism ABM Calibration Stress Test
==============================================================
BASELINE drawn from get_global_calibration_params() at import time.
All SCENARIOS and PARAM_DEFS baseline values are derived from that
snapshot — no hardcoded values that can drift from risk_config.py.

Stage 1 — Supervision parameters
  SMI  (α)   Supervision Monitoring Intensity
  SMD3 (δs3) Supervision Monitoring Decay After 3Y
  SMD6 (δs6) Supervision Monitoring Decay After 6Y

Stage 2
  RCS  (γ)   Risk Contrast Strength

Stage 3 — Offense hazard shifts (log-odds, nested dict)
  OFF_V      Violent
  OFF_D      Drug
  OFF_P      Property
  OFF_O      Other(PublicOrder)

BJS-anchored (not stressed — fixed identification constraints)
  RED1 (δr1) Risk Effect Decay After 1Y
  RED3 (δr3) Risk Effect Decay After 3Y
  RED6 (δr6) Risk Effect Decay After 6Y

USAGE
  python cal_stress_test.py
  python cal_stress_test.py --reps 10 --cores 60
  python cal_stress_test.py --replot PATH/stress_results.csv
"""

import os, sys, argparse, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from joblib import Parallel, delayed
import scipy.stats as st

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_global_calibration_params,
)

# =============================================================================
# BASELINE — single source of truth
# =============================================================================
_CAL = get_global_calibration_params()

SMI  = float(_CAL["Supervision_Monitoring_Intensity"])
SMD3 = float(_CAL["Supervision_Monitoring_Decay_After_3Y"])
SMD6 = float(_CAL["Supervision_Monitoring_Decay_After_6Y"])
RCS  = float(_CAL["Risk_Contrast_Strength"])

# Offense hazard shifts (Stage 3, nested dict)
_OHS  = _CAL["offense_hazard_shift"]
OFF_V = float(_OHS["Violent"])
OFF_D = float(_OHS["Drug"])
OFF_P = float(_OHS["Property"])
OFF_O = float(_OHS["Other(PublicOrder)"])

# BJS-anchored decays — carried through for reference but NOT stressed
RED1 = float(_CAL["Risk_Effect_Decay_After_1Y"])
RED3 = float(_CAL["Risk_Effect_Decay_After_3Y"])
RED6 = float(_CAL["Risk_Effect_Decay_After_6Y"])

# Ordered key / baseline lists used for "what changed" display
_PKEYS = ["SMI", "SMD3", "SMD6", "RCS",
          "OFF_V", "OFF_D", "OFF_P", "OFF_O"]
_PBASE = [SMI,   SMD3,   SMD6,   RCS,
          OFF_V, OFF_D,  OFF_P,  OFF_O]

BJS      = {3: 0.684, 6: 0.794, 9: 0.834}
PASS_PP  = 5.0
WARN_PP  = 10.0

# =============================================================================
# SCENARIOS
# Each tuple: (label, SMI, SMD3, SMD6, RCS, OFF_V, OFF_D, OFF_P, OFF_O)
# =============================================================================
SCENARIOS = [
    # ── Baseline ─────────────────────────────────────────────────────────────
    ("Baseline",
     SMI,  SMD3,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O),

    # ── Stage 1 LOW ──────────────────────────────────────────────────────────
    ("SMI Low  (α=0.00)",
     0.00,  SMD3,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O),
    ("SMD3 Low (δs3=0.50)",
     SMI,   0.50,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O),
    ("SMD6 Low (δs6=0.10)",
     SMI,   SMD3,  0.10,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O),

    # ── Stage 1 HIGH ─────────────────────────────────────────────────────────
    ("SMI High (α=2.00)",
     2.00,  SMD3,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O),
    ("SMD3 High (δs3=1.00)",
     SMI,   1.00,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O),
    ("SMD6 High (δs6=0.80)",
     SMI,   SMD3,  0.80,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O),

    # ── Stage 2 LOW / HIGH ───────────────────────────────────────────────────
    ("RCS Low  (γ=0.00)",
     SMI,   SMD3,  SMD6,  0.00,
     OFF_V, OFF_D, OFF_P, OFF_O),
    ("RCS High (γ=3.00)",
     SMI,   SMD3,  SMD6,  3.00,
     OFF_V, OFF_D, OFF_P, OFF_O),

    # ── Stage 3 — Offense hazard shifts LOW (−0.40 from calibrated) ─────────
    ("OFF Violent Low",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V - 0.40, OFF_D, OFF_P, OFF_O),
    ("OFF Drug Low",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V, OFF_D - 0.40, OFF_P, OFF_O),
    ("OFF Property Low",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P - 0.40, OFF_O),
    ("OFF Other Low",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O - 0.40),

    # ── Stage 3 — Offense hazard shifts HIGH (+0.40 from calibrated) ────────
    ("OFF Violent High",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V + 0.40, OFF_D, OFF_P, OFF_O),
    ("OFF Drug High",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V, OFF_D + 0.40, OFF_P, OFF_O),
    ("OFF Property High",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P + 0.40, OFF_O),
    ("OFF Other High",
     SMI,   SMD3,  SMD6,  RCS,
     OFF_V, OFF_D, OFF_P, OFF_O + 0.40),

    # ── Joint stress ─────────────────────────────────────────────────────────
    ("All LOW  (best case)",
     0.00,  0.50,  0.10,  0.00,
     OFF_V - 0.40, OFF_D - 0.40, OFF_P - 0.40, OFF_O - 0.40),
    ("All HIGH (worst case)",
     2.00,  1.00,  0.80,  3.00,
     OFF_V + 0.40, OFF_D + 0.40, OFF_P + 0.40, OFF_O + 0.40),
]

# Plain-English descriptions
SC_DESC = {
    "Baseline":              "All parameters at calibrated values",
    "SMI Low  (α=0.00)":    "No supervision monitoring (structural zero)",
    "SMD3 Low (δs3=0.50)":  "Rapid supervision intensity decay 3–6yr",
    "SMD6 Low (δs6=0.10)":  "Extreme supervision decay after year 6",
    "SMI High (α=2.00)":    "Extreme surveillance (2× calibrated)",
    "SMD3 High (δs3=1.00)": "No supervision decay 3–6yr (persists)",
    "SMD6 High (δs6=0.80)": "Minimal supervision decay after year 6",
    "RCS Low  (γ=0.00)":    "No risk differentiation across tiers",
    "RCS High (γ=3.00)":    "Maximum risk contrast (3× calibrated)",
    "OFF Violent Low":       f"Violent shift −0.40 (from {OFF_V:+.2f})",
    "OFF Drug Low":          f"Drug shift −0.40 (from {OFF_D:+.2f})",
    "OFF Property Low":      f"Property shift −0.40 (from {OFF_P:+.2f})",
    "OFF Other Low":         f"Other shift −0.40 (from {OFF_O:+.2f})",
    "OFF Violent High":      f"Violent shift +0.40 (from {OFF_V:+.2f})",
    "OFF Drug High":         f"Drug shift +0.40 (from {OFF_D:+.2f})",
    "OFF Property High":     f"Property shift +0.40 (from {OFF_P:+.2f})",
    "OFF Other High":        f"Other shift +0.40 (from {OFF_O:+.2f})",
    "All LOW  (best case)":  "All parameters at minimum stress bounds",
    "All HIGH (worst case)": "All parameters at maximum stress bounds",
}

# =============================================================================
# STYLE
# =============================================================================
C_BG      = "#FFFFFF"
C_HDR     = "#1A3D5C"
C_PASS    = "#1E7A34"
C_WARN    = "#B7770D"
C_FAIL    = "#C0392B"
C_BASE    = "#1A3D5C"
C_LOW     = "#1565C0"
C_HIGH    = "#C62828"
C_JOINT_L = "#0D47A1"
C_JOINT_H = "#6D1A1A"
C_OFF_L   = "#6A0572"   # Stage 3 LOW (purple)
C_OFF_H   = "#9C27B0"   # Stage 3 HIGH (lighter purple)
C_ALT     = "#F0F4F8"
C_GRID    = "#E0E6EE"
WIN_C     = {3: "#2E75B6", 6: "#38863A", 9: "#C07020"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    "#FAFBFC",
    "figure.facecolor":  C_BG,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     10,
})


def _sc_color(label):
    if "Baseline"    in label: return C_BASE
    if "All LOW"     in label: return C_JOINT_L
    if "All HIGH"    in label: return C_JOINT_H
    if "OFF" in label and "Low"  in label: return C_OFF_L
    if "OFF" in label and "High" in label: return C_OFF_H
    if "Low"         in label: return C_LOW
    if "High"        in label: return C_HIGH
    return "#555555"


SC_COLORS = [_sc_color(s[0]) for s in SCENARIOS]


def _verdict(gap_pp):
    a = abs(gap_pp)
    if a <= PASS_PP: return "ROBUST",      C_PASS, "#D6EAD6"
    if a <= WARN_PP: return "ACCEPTABLE",  C_WARN, "#FFF3CD"
    return               "IMPLAUSIBLE", C_FAIL, "#FADBD8"


def _changed(sc):
    """Return a compact string showing which params differ from baseline."""
    # sc = (label, SMI, SMD3, SMD6, RCS, OFF_V, OFF_D, OFF_P, OFF_O)
    parts = [f"{_PKEYS[i]}={sc[i+1]:.2f}"
             for i in range(len(_PKEYS))
             if abs(sc[i+1] - _PBASE[i]) > 1e-6]
    return "  ".join(parts) if parts else "—"


# =============================================================================
# RUN CONFIG
# =============================================================================
N_REPS = 10
SEEDS  = [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087,
          1201, 1319, 1433, 1549, 1663, 1777, 1889, 2003, 2111, 2221]

RUN_CFG = dict(initial_agents=1000, monthly_intake=10,
               warmup_months=144, study_months=108,
               bias_factor=0, enable_peer_influence=True)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = os.path.join(os.path.dirname(__file__),
                          "OAT_StressTest_Output", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# MODEL RUNNER
# =============================================================================
def _run(args):
    """
    args = (label, SMI, SMD3, SMD6, RCS, OFF_V, OFF_D, OFF_P, OFF_O, seed)

    Starts from get_global_calibration_params() so BJS-anchored decays
    (RED1/RED3/RED6) and any other non-swept params stay at calibrated values.
    Offense hazard shifts are merged key-by-key into the nested dict so that
    only the perturbed offense changes; the other three stay calibrated.
    """
    (label, smi, smd3, smd6, rcs,
     off_v, off_d, off_p, off_o, seed) = args
    try:
        cal = get_global_calibration_params()

        # Stage 1 / Stage 2 — flat top-level keys
        cal["Supervision_Monitoring_Intensity"]      = smi
        cal["Supervision_Monitoring_Decay_After_3Y"] = smd3
        cal["Supervision_Monitoring_Decay_After_6Y"] = smd6
        cal["Risk_Contrast_Strength"]                = rcs

        # Stage 3 — merge into nested dict key-by-key
        ohs = dict(cal.get("offense_hazard_shift", {}))
        ohs["Violent"]           = off_v
        ohs["Drug"]              = off_d
        ohs["Property"]          = off_p
        ohs["Other(PublicOrder)"] = off_o
        cal["offense_hazard_shift"] = ohs

        model = RecidivismModel(
            initial_agents        = RUN_CFG["initial_agents"],
            bias_factor           = RUN_CFG["bias_factor"],
            monthly_intake        = RUN_CFG["monthly_intake"],
            warmup_months         = RUN_CFG["warmup_months"],
            study_months          = RUN_CFG["study_months"],
            enable_peer_influence = RUN_CFG["enable_peer_influence"],
            weights               = get_flat_risk_weights(),
            calibration_params    = cal,
            seed                  = seed,
        )
        model.export_csv = False
        while model.running:
            model.step()

        # Aggregate rearrest rates
        rates = {}
        for yrs in [3, 6, 9]:
            r = model.calculate_flag_rate(f"rearrest_{yrs}_yrs")
            rates[yrs] = r if r is not None else float("nan")

        # Tier spread at 3yr (γ health check)
        tier_low = tier_high = float("nan")
        try:
            eligible = [a for a in model.schedule.agents
                        if getattr(a, "study_eligible_agent", False)]
            if eligible:
                low_agents  = [a for a in eligible
                               if a.get_pcra_tier() == "Low"]
                high_agents = [a for a in eligible
                               if a.get_pcra_tier() == "High"]
                if low_agents:
                    tier_low = (sum(1 for a in low_agents
                                   if getattr(a, "rearrest_3_yrs", False))
                                / len(low_agents))
                if high_agents:
                    tier_high = (sum(1 for a in high_agents
                                    if getattr(a, "rearrest_3_yrs", False))
                                 / len(high_agents))
        except Exception as tier_err:
            print(f"  Tier calc skipped [{label} seed={seed}]: {tier_err}")

        return {
            "label":           label,
            "seed":            seed,
            "rate_3":          rates[3],
            "rate_6":          rates[6],
            "rate_9":          rates[9],
            "tier_low_3yr":    tier_low,
            "tier_high_3yr":   tier_high,
            "tier_spread_3yr": (tier_high - tier_low
                                if np.isfinite(tier_low)
                                and np.isfinite(tier_high)
                                else float("nan")),
        }
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
            n    = len(vals)
            mean = float(np.mean(vals)) if vals else float("nan")
            std  = float(np.std(vals))  if n > 1 else 0.0
            ci95 = (st.t.ppf(0.975, df=max(1, n-1)) * std / np.sqrt(n)
                    if n > 1 else 0.0)
            rows.append({"scenario": label, "window": yrs,
                         "mean": mean, "std": std, "ci95": ci95, "n": n})

        for tier in ["tier_low_3yr", "tier_high_3yr", "tier_spread_3yr"]:
            vals = [r[tier] for r in runs
                    if r and tier in r and np.isfinite(r[tier])]
            n    = len(vals)
            mean = float(np.mean(vals)) if vals else float("nan")
            std  = float(np.std(vals))  if n > 1 else 0.0
            ci95 = (st.t.ppf(0.975, df=max(1, n-1)) * std / np.sqrt(n)
                    if n > 1 else 0.0)
            rows.append({"scenario": label, "window": tier,
                         "mean": mean, "std": std, "ci95": ci95, "n": n})

    return pd.DataFrame(rows)


# =============================================================================
# CHART HELPERS
# =============================================================================
def _save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  -> {fname}")
    return path


def _leg_patches():
    return [
        mpatches.Patch(facecolor="#D6EAD6", edgecolor=C_PASS, lw=1.2,
                       label=f"ROBUST  (|gap| ≤ {PASS_PP:.0f}pp)"),
        mpatches.Patch(facecolor="#FFF3CD", edgecolor=C_WARN, lw=1.2,
                       label=f"ACCEPTABLE  ({PASS_PP:.0f}–{WARN_PP:.0f}pp)"),
        mpatches.Patch(facecolor="#FADBD8", edgecolor=C_FAIL, lw=1.2,
                       label=f"IMPLAUSIBLE  (> {WARN_PP:.0f}pp)"),
        mpatches.Patch(color=C_BASE,    label="Baseline"),
        mpatches.Patch(color=C_LOW,     label="Stage 1/2 LOW"),
        mpatches.Patch(color=C_HIGH,    label="Stage 1/2 HIGH"),
        mpatches.Patch(color=C_OFF_L,   label="Stage 3 offense LOW"),
        mpatches.Patch(color=C_OFF_H,   label="Stage 3 offense HIGH"),
        mpatches.Patch(color=C_JOINT_L, label="All LOW (best case)"),
        mpatches.Patch(color=C_JOINT_H, label="All HIGH (worst case)"),
    ]


# =============================================================================
# CHART 1 — Single window: absolute rates + gap panel
# =============================================================================
def chart_single_window(df, window_yrs):
    n   = len(SCENARIOS)
    bjs = BJS[window_yrs]
    sub = df[df["window"] == window_yrs].set_index("scenario")

    labels = [s[0] for s in SCENARIOS]
    means  = [sub.loc[l, "mean"] if l in sub.index else np.nan
              for l in labels]
    cis    = [sub.loc[l, "ci95"] if l in sub.index else 0.0
              for l in labels]
    gaps   = [(m - bjs) * 100 if np.isfinite(m) else np.nan
              for m in means]

    fig = plt.figure(figsize=(22, max(11, n * 0.55 + 3)))
    fig.patch.set_facecolor(C_BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1.0],
                             wspace=0.08,
                             left=0.24, right=0.97,
                             top=0.91, bottom=0.10)

    fig.suptitle(
        f"Calibration Stress Test — {window_yrs}-Year Rearrest Rate\n"
        f"Each scenario perturbs one parameter group to its structural extreme  |  "
        f"BJS benchmark = {bjs:.0%}  |  "
        f"Robust = |gap| ≤ {PASS_PP:.0f}pp  |  "
        f"n = {N_REPS} reps × 20 seeds",
        fontsize=12, fontweight="bold", color=C_HDR, y=0.97,
    )

    # ── Panel A: absolute rates ───────────────────────────────────────────────
    axA = fig.add_subplot(gs[0])
    axA.set_facecolor("#FAFBFC")
    axA.axvspan(bjs - PASS_PP/100, bjs + PASS_PP/100,
                alpha=0.10, color=C_PASS, zorder=0)
    axA.axvspan(bjs - WARN_PP/100, bjs - PASS_PP/100,
                alpha=0.06, color=C_WARN, zorder=0)
    axA.axvspan(bjs + PASS_PP/100, bjs + WARN_PP/100,
                alpha=0.06, color=C_WARN, zorder=0)
    axA.axvline(bjs, color="#E53935", linewidth=2.2, zorder=4,
                label=f"BJS benchmark ({bjs:.0%})")

    for i, (lbl, m, ci) in enumerate(zip(labels, means, cis)):
        if not np.isfinite(m): continue
        y = n - 1 - i
        axA.barh(y, m, height=0.62, color=_sc_color(lbl), alpha=0.85,
                 zorder=3, xerr=ci,
                 error_kw={"elinewidth": 1.4, "ecolor": "#444", "capsize": 4})
        gap = (m - bjs) * 100
        _, vc, _ = _verdict(gap)
        axA.text(m + ci + 0.003, y, f"{m:.3f}",
                 va="center", fontsize=8.5, color=_sc_color(lbl),
                 fontweight="bold")
        axA.text(0.985, y, f"{gap:+.1f}pp",
                 va="center", ha="right", fontsize=8,
                 color=vc, fontweight="bold",
                 transform=axA.get_yaxis_transform())

    axA.set_yticks(range(n))
    axA.set_yticklabels([s[0] for s in reversed(SCENARIOS)],
                        fontsize=8.5, fontweight="bold")
    for tick, sc in zip(axA.get_yticklabels(), reversed(SCENARIOS)):
        tick.set_color(_sc_color(sc[0]))

    axA.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axA.set_xlabel(f"{window_yrs}-Year Cumulative Rearrest Rate", fontsize=11)
    axA.set_title(f"A — Rearrest Rates with 95% CI  (BJS = {bjs:.0%})",
                  fontsize=11, fontweight="bold", color=C_HDR)
    axA.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
    axA.grid(True, axis="x", zorder=1)

    # ── Panel B: gap bars ─────────────────────────────────────────────────────
    axB = fig.add_subplot(gs[1])
    axB.set_facecolor("#FAFBFC")
    axB.axvspan(-PASS_PP, PASS_PP, alpha=0.10, color=C_PASS, zorder=0)
    axB.axvspan(-WARN_PP, -PASS_PP, alpha=0.06, color=C_WARN, zorder=0)
    axB.axvspan(PASS_PP, WARN_PP, alpha=0.06, color=C_WARN, zorder=0)
    axB.axvline(0, color="#E53935", lw=2.0, zorder=4)

    for i, (lbl, gap) in enumerate(zip(labels, gaps)):
        if not np.isfinite(gap): continue
        y = n - 1 - i
        _, vc, _ = _verdict(gap)
        axB.barh(y, gap, height=0.62,
                 color=_sc_color(lbl), alpha=0.85, zorder=3)
        axB.text(gap + (0.3 if gap >= 0 else -0.3), y,
                 f"{gap:+.1f}pp",
                 va="center", ha="left" if gap >= 0 else "right",
                 fontsize=8.5, color=vc, fontweight="bold")

    axB.set_yticks(range(n))
    axB.set_yticklabels([])
    axB.set_xlabel("Gap vs BJS Benchmark (percentage points)", fontsize=11)
    axB.set_title("B — Gap to BJS Target (pp)",
                  fontsize=11, fontweight="bold", color=C_HDR)
    axB.grid(True, axis="x", zorder=1)

    fig.legend(handles=_leg_patches(), loc="lower center",
               ncol=5, fontsize=8.5, framealpha=0.95,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

    _save(fig, f"stress_{window_yrs}yr.png")


# =============================================================================
# CHART 2 — Summary: all three windows
# =============================================================================
def chart_summary(df):
    n      = len(SCENARIOS)
    labels = [s[0] for s in SCENARIOS]

    fig = plt.figure(figsize=(26, max(12, n * 0.55 + 4)))
    fig.patch.set_facecolor(C_BG)
    gs  = gridspec.GridSpec(1, 3, wspace=0.06,
                             left=0.18, right=0.97,
                             top=0.91, bottom=0.11)

    fig.suptitle(
        "Calibration Stress Test — Deviation from BJS Benchmarks\n"
        f"Green = Robust (|gap| ≤ {PASS_PP:.0f}pp)  |  "
        f"Amber = Acceptable ({PASS_PP:.0f}–{WARN_PP:.0f}pp)  |  "
        "Red = Implausible  |  "
        f"n = {N_REPS} reps × 20 seeds",
        fontsize=12, fontweight="bold", color=C_HDR, y=0.97,
    )

    for col_i, window_yrs in enumerate([3, 6, 9]):
        ax  = fig.add_subplot(gs[0, col_i])
        bjs = BJS[window_yrs]
        sub = df[df["window"] == window_yrs].set_index("scenario")

        gaps = [(sub.loc[l, "mean"] - bjs) * 100
                if l in sub.index and np.isfinite(sub.loc[l, "mean"])
                else np.nan for l in labels]

        ax.set_facecolor("#FAFBFC")
        ax.axvspan(-PASS_PP, PASS_PP, alpha=0.12, color=C_PASS, zorder=0)
        ax.axvspan(-WARN_PP, -PASS_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvspan(PASS_PP, WARN_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvline(0, color="#E53935", lw=2.0, zorder=4)

        for i, (lbl, gap) in enumerate(zip(labels, gaps)):
            if not np.isfinite(gap): continue
            y = n - 1 - i
            _, vc, _ = _verdict(gap)
            ax.barh(y, gap, height=0.65,
                    color=_sc_color(lbl), alpha=0.85, zorder=3)
            if abs(gap) > 2:
                ax.text(gap / 2, y, f"{gap:+.1f}pp",
                        va="center", ha="center",
                        fontsize=7.5, color="white", fontweight="bold")
            else:
                offset = 0.3 if gap >= 0 else -0.3
                ax.text(gap + offset, y, f"{gap:+.1f}pp",
                        va="center",
                        ha="left" if gap >= 0 else "right",
                        fontsize=7, color=vc, fontweight="bold")

        ax.set_yticks(range(n))
        if col_i == 0:
            ax.set_yticklabels([s[0] for s in reversed(SCENARIOS)],
                               fontsize=8, fontweight="bold")
            for tick, sc in zip(ax.get_yticklabels(),
                                reversed(SCENARIOS)):
                tick.set_color(_sc_color(sc[0]))
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Gap vs BJS Benchmark (pp)", fontsize=10)
        ax.set_title(
            f"{window_yrs}-Year Window\nBJS benchmark = {bjs:.0%}",
            fontsize=11, fontweight="bold", color=WIN_C[window_yrs], pad=8,
        )
        ax.grid(True, axis="x", zorder=1)

    fig.legend(handles=_leg_patches(), loc="lower center",
               ncol=5, fontsize=8.5, framealpha=0.95,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

    _save(fig, "stress_summary.png")


# =============================================================================
# CHART 3 — Verdict scorecard  (HTML-table style, matching reference image)
# =============================================================================
def chart_scorecard(df):
    """
    Renders a clean, HTML-table-style scorecard using matplotlib.
    Layout mirrors the reference image:
      - Dark navy title bar
      - Italic subtitle row
      - Merged column-group headers (3yr / 6yr / 9yr / Overall / Tier)
      - Column header row with light-grey background
      - Alternating row striping
      - Compact ROBUST / ACCEPT. / IMPLAUS. verdict badges (filled cells)
      - Coloured scenario labels (blue=LOW, red=HIGH, purple=offense)
      - Bottom legend strip
    """
    N = len(SCENARIOS)

    # ── Column definitions: (header, param_label, relative_width) ─────────────
    # Columns: Scenario | Parameter | Description |
    #          Rate/Gap/Verdict ×3 | Overall | Tier Spread
    COL_SPEC = [
        ("Scenario",    "",  2.20),
        ("Parameter",   "",  0.90),
        ("Description", "",  2.00),
        # 3yr
        ("Rate (%)",    "3", 0.62),
        ("Gap",         "3", 0.58),
        ("Verdict",     "3", 0.82),
        # 6yr
        ("Rate (%)",    "6", 0.62),
        ("Gap",         "6", 0.58),
        ("Verdict",     "6", 0.82),
        # 9yr
        ("Rate (%)",    "9", 0.62),
        ("Gap",         "9", 0.58),
        ("Verdict",     "9", 0.82),
        # summary
        ("Overall",     "",  0.75),
        ("Tier Spread\n(pp)", "", 0.72),
    ]
    NCOLS = len(COL_SPEC)

    # Normalise widths to [0,1] and compute x positions
    total_w = sum(rw for _, _, rw in COL_SPEC)
    col_frac = [rw / total_w for _, _, rw in COL_SPEC]
    col_x    = []
    cx = 0.0
    for f in col_frac:
        col_x.append(cx)
        cx += f

    # ── Row heights (figure-fraction units) ──────────────────────────────────
    TITLE_H  = 0.048   # dark navy title bar
    SUB_H    = 0.028   # italic subtitle
    GRP_H    = 0.030   # window group headers (3yr / 6yr / 9yr)
    COL_H    = 0.036   # column headers
    ROW_H    = 0.038   # data rows
    LEG_H    = 0.038   # legend strip at bottom

    FIG_H_IN = (TITLE_H + SUB_H + GRP_H + COL_H
                + ROW_H * N + LEG_H) / 0.014   # approx inch conversion
    FIG_H_IN = max(14.0, min(FIG_H_IN, 32.0))
    FIG_W_IN = 28.0

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    fig.patch.set_facecolor("#F5F6F8")

    # Convert all heights to figure-fraction
    tot_h = TITLE_H + SUB_H + GRP_H + COL_H + ROW_H * N + LEG_H

    def frac(h): return h / tot_h   # relative → figure fraction

    # y positions (bottom-up)
    y_leg   = 0.0
    y_data0 = frac(LEG_H)                                     # bottom of first data row
    y_col   = y_data0 + frac(ROW_H) * N                       # column headers
    y_grp   = y_col + frac(COL_H)                             # window group headers
    y_sub   = y_grp + frac(GRP_H)                             # subtitle
    y_title = y_sub + frac(SUB_H)                             # title bar

    # ── Helper: draw one cell ─────────────────────────────────────────────────
    LEFT_PAD  = 0.012   # left margin (figure fraction)
    RIGHT_PAD = 0.005
    USABLE    = 1.0 - LEFT_PAD - RIGHT_PAD

    def _cx(col_idx):
        return LEFT_PAD + col_x[col_idx] * USABLE

    def _cw(col_idx):
        return col_frac[col_idx] * USABLE

    def draw_cell(col_idx, row_y, row_h,
                  text, bg="#FFFFFF", fg="#1A2B3C",
                  fs=8.5, bold=False, italic=False,
                  ha="center", va="center",
                  border_color="#D0D5DD", border_lw=0.5,
                  pad_left=0.04):
        x = _cx(col_idx)
        w = _cw(col_idx)
        ax = fig.add_axes([x, row_y, w, row_h])
        ax.set_facecolor(bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(border_lw)
            spine.set_edgecolor(border_color)
        tx = pad_left if ha == "left" else (1 - pad_left if ha == "right" else 0.5)
        style = "italic" if italic else "normal"
        weight = "bold" if bold else "normal"
        ax.text(tx, 0.5, str(text),
                ha=ha, va=va, fontsize=fs,
                fontweight=weight, fontstyle=style,
                color=fg, clip_on=False,
                linespacing=1.25, wrap=False)

    def draw_span_cell(x_start, x_end, row_y, row_h,
                       text, bg, fg="white", fs=9, bold=True):
        """Span across multiple column positions."""
        w = x_end - x_start
        ax = fig.add_axes([x_start, row_y, w, row_h])
        ax.set_facecolor(bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("white")
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=fg, clip_on=False)

    # ── TITLE BAR ─────────────────────────────────────────────────────────────
    draw_span_cell(LEFT_PAD, 1.0 - RIGHT_PAD,
                   y_title, frac(TITLE_H),
                   "Calibration Stress Test — Verdict Scorecard",
                   bg=C_HDR, fg="white", fs=14)

    # ── SUBTITLE ──────────────────────────────────────────────────────────────
    sub_ax = fig.add_axes([LEFT_PAD, y_sub, USABLE, frac(SUB_H)])
    sub_ax.set_facecolor("#FFFFFF")
    sub_ax.axis("off")
    sub_ax.text(0.5, 0.5,
                f"Robust = |gap| ≤ {PASS_PP:.0f} pp  │  "
                f"Acceptable = {PASS_PP:.0f}–{WARN_PP:.0f} pp  │  "
                f"Implausible > {WARN_PP:.0f} pp  │  "
                "BJS benchmarks: 3yr = 68%, 6yr = 79%, 9yr = 83%  │  "
                f"n = {N_REPS} reps × 20 seeds",
                ha="center", va="center",
                fontsize=8.5, fontstyle="italic", color="#444444")

    # ── WINDOW GROUP HEADERS ──────────────────────────────────────────────────
    # Blank spans for Scenario / Parameter / Description
    draw_span_cell(_cx(0), _cx(3), y_grp, frac(GRP_H),
                   "", bg="#E8ECF0", fg="#1A2B3C", fs=8)

    # 3yr span: cols 3-5
    draw_span_cell(_cx(3), _cx(6), y_grp, frac(GRP_H),
                   f"3-Year Window  (BJS = {BJS[3]:.0%})",
                   bg="#D6E4F0", fg="#1A3D5C", fs=9)
    # 6yr span: cols 6-8
    draw_span_cell(_cx(6), _cx(9), y_grp, frac(GRP_H),
                   f"6-Year Window  (BJS = {BJS[6]:.0%})",
                   bg="#D5EDD5", fg="#1A3D1A", fs=9)
    # 9yr span: cols 9-11
    draw_span_cell(_cx(9), _cx(12), y_grp, frac(GRP_H),
                   f"9-Year Window  (BJS = {BJS[9]:.0%})",
                   bg="#FDEBD0", fg="#5D3A00", fs=9)
    # Overall + Tier Spread spans
    draw_span_cell(_cx(12), _cx(13), y_grp, frac(GRP_H),
                   "", bg="#E8ECF0", fg="#1A2B3C", fs=8)
    draw_span_cell(_cx(13), 1.0 - RIGHT_PAD, y_grp, frac(GRP_H),
                   "† Tier Spread", bg="#EDE7F6", fg="#4A148C", fs=8.5)

    # ── COLUMN HEADERS ────────────────────────────────────────────────────────
    col_hdrs = ["Scenario", "Parameter", "Description",
                "Rate (%)", "Gap", "Verdict",
                "Rate (%)", "Gap", "Verdict",
                "Rate (%)", "Gap", "Verdict",
                "Overall", "Tier\nSpread"]
    for ci, hdr in enumerate(col_hdrs):
        draw_cell(ci, y_col, frac(COL_H), hdr,
                  bg="#E8ECF0", fg="#1A2B3C",
                  fs=8, bold=True,
                  border_color="#B0B8C4", border_lw=0.7)

    # ── Compact parameter display for each scenario ───────────────────────────
    def _param_label(sc):
        """Short italic parameter label like 'α = 0.00' or 'γ = 3.00'."""
        lbl = sc[0]
        if "Baseline" in lbl: return "—"
        chg = _changed(sc)   # e.g. "SMI=0.00"
        if chg == "—": return "—"
        # Map internal keys to greek symbols
        sym = {"SMI": "α", "SMD3": "δs3", "SMD6": "δs6",
               "RCS": "γ", "OFF_V": "Viol.", "OFF_D": "Drug",
               "OFF_P": "Prop.", "OFF_O": "Other"}
        parts = chg.split("  ")
        out = []
        for p in parts:
            k, v = p.split("=")
            out.append(f"{sym.get(k, k)} = {float(v):.2f}")
        return "  ".join(out)

    # ── DATA ROWS ─────────────────────────────────────────────────────────────
    # Alternating stripe colours
    STRIPE_A = "#FFFFFF"
    STRIPE_B = "#F5F7FA"

    for ri, sc in enumerate(SCENARIOS):
        lbl   = sc[0]
        row_y = y_data0 + frac(ROW_H) * (N - 1 - ri)
        bg    = STRIPE_A if ri % 2 == 0 else STRIPE_B
        sc_c  = _sc_color(lbl)
        rh    = frac(ROW_H)

        # Gather data
        rates, gaps, verdicts = {}, {}, []
        for w in [3, 6, 9]:
            sub = df[(df["scenario"] == lbl) & (df["window"] == w)]
            if sub.empty or not np.isfinite(sub["mean"].iloc[0]):
                rates[w] = np.nan; gaps[w] = np.nan
                verdicts.append(("—", "#888888", bg))
            else:
                m = sub["mean"].iloc[0]; g = (m - BJS[w]) * 100
                v, vc, vbg = _verdict(g)
                rates[w] = m; gaps[w] = g
                verdicts.append((v, vc, vbg))

        max_gap = max((abs(gaps[w]) for w in [3, 6, 9]
                       if np.isfinite(gaps[w])), default=0)
        if   max_gap <= PASS_PP: ov, ovc, obg = "ROBUST",     C_PASS, "#D6EAD6"
        elif max_gap <= WARN_PP: ov, ovc, obg = "ACCEPT.",    C_WARN, "#FFF3CD"
        else:                    ov, ovc, obg = "IMPLAUS.",   C_FAIL, "#FADBD8"

        # Col 0 — Scenario (bold, coloured)
        draw_cell(0, row_y, rh, lbl,
                  bg=bg, fg=sc_c, fs=8.5, bold=True, ha="left")
        # Col 1 — Parameter (italic, grey)
        draw_cell(1, row_y, rh, _param_label(sc),
                  bg=bg, fg="#555555", fs=8, italic=True)
        # Col 2 — Description
        draw_cell(2, row_y, rh, SC_DESC.get(lbl, ""),
                  bg=bg, fg="#333333", fs=7.8, ha="left")

        # Cols 3-11 — Rate / Gap / Verdict per window
        win_colors = [WIN_C[3], WIN_C[6], WIN_C[9]]
        ci = 3
        for wi, w in enumerate([3, 6, 9]):
            r          = rates[w]
            g          = gaps[w]
            v, vc, vbg = verdicts[wi]
            wc         = win_colors[wi]

            # Rate
            draw_cell(ci, row_y, rh,
                      f"{r*100:.1f}%" if np.isfinite(r) else "—",
                      bg=bg, fg=wc, fs=8.5, bold=True)
            # Gap
            if np.isfinite(g):
                gc = (C_FAIL if abs(g) > WARN_PP else
                      C_WARN if abs(g) > PASS_PP else
                      "#1E7A34" if g <= 0 else "#555555")
                draw_cell(ci + 1, row_y, rh, f"{g:+.1f}pp",
                          bg=bg, fg=gc, fs=8.5, bold=(abs(g) > PASS_PP))
            else:
                draw_cell(ci + 1, row_y, rh, "—", bg=bg, fg="#888")
            # Verdict badge (filled cell)
            short_v = v[:6] if v != "—" else "—"   # ROBUST / ACCEPT. / IMPLAUS.
            draw_cell(ci + 2, row_y, rh, short_v,
                      bg=vbg, fg=vc, fs=8, bold=True,
                      border_color=vc, border_lw=0.8)
            ci += 3

        # Col 12 — Overall badge
        draw_cell(12, row_y, rh, ov,
                  bg=obg, fg=ovc, fs=8.5, bold=True,
                  border_color=ovc, border_lw=0.9)

        # Col 13 — Tier spread
        tier_sub = df[(df["scenario"] == lbl) &
                      (df["window"] == "tier_spread_3yr")]
        if not tier_sub.empty and np.isfinite(tier_sub["mean"].iloc[0]):
            spread = tier_sub["mean"].iloc[0] * 100
            s_col  = (C_PASS if spread > 10 else
                      C_WARN if spread > 5  else C_FAIL)
            s_bg   = ("#D6EAD6" if spread > 10 else
                      "#FFF3CD" if spread > 5  else "#FADBD8")
            s_txt  = f"{spread:.1f}"
        else:
            s_col, s_bg, s_txt = "#888888", bg, "—"
        draw_cell(13, row_y, rh, s_txt,
                  bg=s_bg, fg=s_col, fs=8.5, bold=True)

    # ── LEGEND STRIP ──────────────────────────────────────────────────────────
    leg_ax = fig.add_axes([LEFT_PAD, y_leg, USABLE, frac(LEG_H)])
    leg_ax.set_facecolor("#F0F2F5")
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")
    for spine in leg_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#D0D5DD")
        spine.set_linewidth(0.6)

    # Coloured squares + labels
    items = [
        ("#D6EAD6", C_PASS, f"ROBUST (|gap| ≤ {PASS_PP:.0f} pp)"),
        ("#FFF3CD", C_WARN, f"ACCEPTABLE ({PASS_PP:.0f}–{WARN_PP:.0f} pp)"),
        ("#FADBD8", C_FAIL, f"IMPLAUSIBLE (> {WARN_PP:.0f} pp)"),
        (C_HDR,    C_HDR,  "Baseline"),
        (C_LOW,    C_LOW,  "Stage 1/2 LOW"),
        (C_HIGH,   C_HIGH, "Stage 1/2 HIGH"),
        (C_OFF_L,  C_OFF_L,"Stage 3 offense LOW"),
        (C_OFF_H,  C_OFF_H,"Stage 3 offense HIGH"),
        (C_JOINT_L,C_JOINT_L,"All LOW (joint min.)"),
        (C_JOINT_H,C_JOINT_H,"All HIGH (joint max.)"),
    ]
    n_items = len(items)
    sq = 0.025    # square size (axes fraction)
    gap_x = 1.0 / n_items
    for ii, (bg_i, fg_i, lbl_i) in enumerate(items):
        x0 = ii * gap_x + 0.005
        # coloured square
        from matplotlib.patches import FancyBboxPatch
        sq_ax = leg_ax.inset_axes([x0, 0.20, sq * 0.6, 0.55],
                                   transform=leg_ax.transAxes)
        sq_ax.set_facecolor(bg_i)
        sq_ax.set_xlim(0,1); sq_ax.set_ylim(0,1)
        sq_ax.axis("off")
        for sp in sq_ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(fg_i); sp.set_linewidth(1.2)
        leg_ax.text(x0 + sq * 0.72, 0.50, lbl_i,
                    ha="left", va="center",
                    fontsize=7.2, color="#333333",
                    transform=leg_ax.transAxes)

    leg_ax.text(0.995, 0.50,
                "† Tier spread > 10 pp = healthy risk differentiation",
                ha="right", va="center",
                fontsize=7.2, fontstyle="italic", color="#555555",
                transform=leg_ax.transAxes)

    path = os.path.join(OUT_DIR, "stress_scorecard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F5F6F8")
    plt.close(fig)
    print("  -> stress_scorecard.png")


# =============================================================================
# CHART 4 — Deviation bars (3 windows side by side)
# =============================================================================
def chart_deviation(df):
    labels = [s[0] for s in SCENARIOS]
    n      = len(labels)

    fig, axes = plt.subplots(1, 3,
                              figsize=(24, max(11, n * 0.55 + 3)),
                              sharey=True)
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(
        "Calibration Stress Test — Gap to BJS Benchmark by Scenario\n"
        f"Green = Robust (|gap| ≤ {PASS_PP:.0f}pp)  |  "
        f"Amber = Acceptable ({PASS_PP:.0f}–{WARN_PP:.0f}pp)  |  "
        "Red = Implausible",
        fontsize=12, fontweight="bold", color=C_HDR, y=0.97,
    )

    for ax, window_yrs in zip(axes, [3, 6, 9]):
        bjs = BJS[window_yrs]
        sub = df[df["window"] == window_yrs].set_index("scenario")

        gaps = [(sub.loc[l, "mean"] - bjs) * 100
                if l in sub.index and np.isfinite(sub.loc[l, "mean"])
                else np.nan for l in labels]

        ax.set_facecolor("#FAFBFC")
        ax.axvspan(-PASS_PP, PASS_PP, alpha=0.12, color=C_PASS, zorder=0)
        ax.axvspan(-WARN_PP, -PASS_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvspan(PASS_PP, WARN_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvline(0, color="#E53935", lw=2.0, zorder=4)

        for i, (lbl, gap) in enumerate(
                zip(reversed(labels), reversed(gaps))):
            if not np.isfinite(gap): continue
            y = i
            _, vc, _ = _verdict(gap)
            ax.barh(y, gap, height=0.68,
                    color=_sc_color(lbl), alpha=0.88, zorder=3)
            if abs(gap) > 3:
                ax.text(gap / 2, y, f"{gap:+.1f}pp",
                        va="center", ha="center",
                        fontsize=7.5, color="white", fontweight="bold")
            else:
                offset = 0.3 if gap >= 0 else -0.3
                ax.text(gap + offset, y, f"{gap:+.1f}pp",
                        va="center",
                        ha="left" if gap >= 0 else "right",
                        fontsize=7, color=vc, fontweight="bold")

        ax.set_yticks(range(n))
        if window_yrs == 3:
            ax.set_yticklabels(list(reversed(labels)),
                               fontsize=8.5, fontweight="bold")
            for tick, sc in zip(ax.get_yticklabels(),
                                list(reversed(SCENARIOS))):
                tick.set_color(_sc_color(sc[0]))
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Gap vs BJS Benchmark (pp)", fontsize=11)
        ax.set_title(
            f"{window_yrs}-Year Window\nBJS benchmark = {bjs:.0%}",
            fontsize=11, fontweight="bold", color=WIN_C[window_yrs], pad=10,
        )
        ax.grid(True, axis="x", zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(handles=_leg_patches(), loc="lower center",
               ncol=5, fontsize=8.5, framealpha=0.95,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    _save(fig, "stress_deviation.png")


# =============================================================================
# CHART 5 — Disaggregated SMI + tier spread check
# =============================================================================
def chart_disaggregated_smi(df):
    scenarios_of_interest = [
        "Baseline",
        "SMI Low  (α=0.00)",
        "SMI High (α=2.00)",
        "RCS Low  (γ=0.00)",
        "RCS High (γ=3.00)",
        "All LOW  (best case)",
        "All HIGH (worst case)",
    ]
    # Keep only those that actually exist in df
    scenarios_of_interest = [s for s in scenarios_of_interest
                              if s in df["scenario"].values]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(
        "Disaggregated Stress Check — Tier-Level Rearrest Rates\n"
        "Verifies SMI and RCS operate on correct agents and windows",
        fontsize=13, fontweight="bold", color=C_HDR, y=0.98,
    )

    ax = axes[0]
    ax.set_facecolor("#FAFBFC")
    ax.set_title("A — Low vs High Tier 3yr Rate by Scenario",
                 fontsize=11, fontweight="bold", color=C_HDR)

    x     = np.arange(len(scenarios_of_interest))
    width = 0.30
    low_means, high_means = [], []
    for sc in scenarios_of_interest:
        low_sub  = df[(df["scenario"] == sc) & (df["window"] == "tier_low_3yr")]
        high_sub = df[(df["scenario"] == sc) & (df["window"] == "tier_high_3yr")]
        low_means.append(low_sub["mean"].iloc[0]
                         if not low_sub.empty else np.nan)
        high_means.append(high_sub["mean"].iloc[0]
                          if not high_sub.empty else np.nan)

    bars1 = ax.bar(x - width/2, low_means,  width, label="Low Tier",
                   color=C_LOW,  alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width/2, high_means, width, label="High Tier",
                   color=C_HIGH, alpha=0.85, zorder=3)

    if low_means and np.isfinite(low_means[0]):
        ax.axhline(low_means[0],  color=C_LOW,  lw=1.2, ls="--",
                   alpha=0.5, label="Baseline Low")
    if high_means and np.isfinite(high_means[0]):
        ax.axhline(high_means[0], color=C_HIGH, lw=1.2, ls="--",
                   alpha=0.5, label="Baseline High")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                    color=C_LOW if bar in bars1 else C_HIGH)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_of_interest, rotation=15,
                       ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("3-Year Rearrest Rate", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", zorder=0, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B — tier spread
    ax2 = axes[1]
    ax2.set_facecolor("#FAFBFC")
    ax2.set_title("B — Tier Spread (High − Low) by Scenario\n"
                  "Collapse toward 0 = γ differentiation lost",
                  fontsize=11, fontweight="bold", color=C_HDR)

    spreads, colors = [], []
    for sc in scenarios_of_interest:
        sp_sub = df[(df["scenario"] == sc) &
                    (df["window"] == "tier_spread_3yr")]
        sp = sp_sub["mean"].iloc[0] if not sp_sub.empty else np.nan
        spreads.append(sp * 100 if np.isfinite(sp) else np.nan)
        colors.append(_sc_color(sc))

    bars3 = ax2.bar(x, spreads, width=0.5,
                    color=colors, alpha=0.85, zorder=3)
    ax2.axhline(10, color=C_PASS, lw=1.8, ls="--",
                label="Healthy threshold (10pp)", zorder=4)
    ax2.axhline(5,  color=C_WARN, lw=1.2, ls=":",
                label="Minimum threshold (5pp)",  zorder=4)

    for bar, sp in zip(bars3, spreads):
        if np.isfinite(sp):
            col = (C_PASS if sp > 10 else C_WARN if sp > 5 else C_FAIL)
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     sp + 0.3, f"{sp:.1f}pp",
                     ha="center", va="bottom",
                     fontsize=9, color=col, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios_of_interest, rotation=15,
                        ha="right", fontsize=9)
    ax2.set_ylabel("Tier Spread (pp)", fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, axis="y", zorder=0, alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "stress_disaggregated_smi.png")


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
def print_summary(df):
    print("\n" + "=" * 78)
    print("  CALIBRATION STRESS TEST — RESULTS SUMMARY")
    print("=" * 78)
    print(f"\n  Calibrated baseline (from get_global_calibration_params()):")
    for k, v in zip(_PKEYS, _PBASE):
        print(f"    {k:<8} = {v:+.4f}")
    print(f"    RED1 = {RED1:.4f}  RED3 = {RED3:.4f}  "
          f"RED6 = {RED6:.4f}  [BJS-anchored, not stressed]")
    print()
    print(f"  {'Scenario':<30}  "
          f"{'3yr':>6}  {'6yr':>6}  {'9yr':>6}  "
          f"{'Gap3':>8}  {'Gap6':>8}  {'Gap9':>8}  Verdict")
    print(f"  {'-'*84}")
    for sc in SCENARIOS:
        lbl = sc[0]
        sub = df[df["scenario"] == lbl].set_index("window")
        r   = {y: sub.loc[y, "mean"] if y in sub.index else np.nan
               for y in [3, 6, 9]}
        g   = {y: (r[y] - BJS[y]) * 100 for y in [3, 6, 9]}
        v   = max(abs(g[y]) for y in [3, 6, 9] if np.isfinite(g[y]))
        verdict = ("ROBUST" if v <= PASS_PP else
                   "ACCEPTABLE" if v <= WARN_PP else "IMPLAUSIBLE")
        print(f"  {lbl:<30}  "
              f"{r[3]:>6.3f}  {r[6]:>6.3f}  {r[9]:>6.3f}  "
              f"{g[3]:>+7.1f}pp  {g[6]:>+7.1f}pp  {g[9]:>+7.1f}pp  "
              f"{verdict}")
    print(f"\n  Output: {OUT_DIR}")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================
def main():
    global N_REPS, SEEDS, OUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--reps",   type=int, default=N_REPS)
    parser.add_argument("--cores",  type=int, default=-1)
    parser.add_argument("--replot", type=str, default=None,
                        metavar="CSV",
                        help="Replot from saved CSV, skip simulation")
    args = parser.parse_args()

    if args.replot:
        OUT_DIR = os.path.dirname(os.path.abspath(args.replot))
        df      = pd.read_csv(args.replot)
        print(f"\n  Replotting from {args.replot}")
    else:
        N_REPS = args.reps
        SEEDS  = [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087,
                  1201, 1319, 1433, 1549, 1663, 1777, 1889, 2003, 2111, 2221]

        print(f"\n  Calibration Stress Test")
        print(f"  Scenarios   : {len(SCENARIOS)}")
        print(f"  Reps×Seeds  : {N_REPS} × {len(SEEDS)} = "
              f"{N_REPS * len(SEEDS)} runs per scenario")
        print(f"  Total runs  : {len(SCENARIOS) * N_REPS * len(SEEDS)}")
        print(f"  Output      : {OUT_DIR}\n")
        print(f"  Baseline (from get_global_calibration_params()):")
        for k, v in zip(_PKEYS, _PBASE):
            print(f"    {k:<8} = {v:+.4f}")
        print(f"    RED1={RED1:.4f}  RED3={RED3:.4f}  "
              f"RED6={RED6:.4f}  [not stressed]\n")

        # Each job tuple matches _run signature
        jobs = [(*sc, seed)
                for sc in SCENARIOS
                for _ in range(N_REPS)
                for seed in SEEDS]
        raw  = Parallel(n_jobs=args.cores)(
            delayed(_run)(j) for j in jobs)

        df = aggregate([r for r in raw if r])
        df.to_csv(os.path.join(OUT_DIR, "stress_results.csv"), index=False)

    print("\n  Generating charts...")
    for w in [3, 6, 9]:
        chart_single_window(df, w)
    chart_summary(df)
    chart_scorecard(df)
    chart_deviation(df)
    chart_disaggregated_smi(df)
    print_summary(df)


if __name__ == "__main__":
    main()