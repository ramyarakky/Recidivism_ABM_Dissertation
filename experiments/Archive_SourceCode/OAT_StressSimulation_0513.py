#!/usr/bin/env python3
"""
cal_stress_test.py  —  Recidivism ABM Calibration Stress Test
==============================================================
FINAL CALIBRATED BASELINE
  SMI (a)    = 1.050   Supervision Monitoring Intensity
  RCS (g)    = 1.000   Risk Contrast Strength
  RED3 (dr3) = 0.867   Risk Effect Decay After 3Y
  RED6 (dr6) = 0.800   Risk Effect Decay After 6Y
  SMD3 (ds3) = 0.933   Supervision Monitoring Decay After 3Y
  SMD6 (ds6) = 0.300   Supervision Monitoring Decay After 6Y [fixed]

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
# BASELINE + SCENARIOS
# =============================================================================
SMI=1.050; RCS=1.000; RED3=0.867; RED6=0.800; SMD3=0.933; SMD6=0.300

_PKEYS = ["SMI","RCS","RED3","RED6","SMD3","SMD6"]
_PBASE = [SMI,  RCS,  RED3,  RED6,  SMD3,  SMD6]

SCENARIOS = [
    # (label,                         SMI,  RCS,  RED3, RED6, SMD3, SMD6)
    ("Baseline",                      SMI,  RCS,  RED3, RED6, SMD3, SMD6),
    # LOW bounds
    ("SMI Low  (a=0.00)",            0.00,  RCS,  RED3, RED6, SMD3, SMD6),
    ("RCS Low  (g=0.00)",             SMI, 0.00,  RED3, RED6, SMD3, SMD6),
    ("RED3 Low (dr3=0.50)",           SMI,  RCS,  0.50, RED6, SMD3, SMD6),
    ("RED6 Low (dr6=0.40)",           SMI,  RCS,  RED3, 0.40, SMD3, SMD6),
    ("SMD3 Low (ds3=0.50)",           SMI,  RCS,  RED3, RED6, 0.50, SMD6),
    ("SMD6 Low (ds6=0.10)",           SMI,  RCS,  RED3, RED6, SMD3, 0.10),
    # HIGH bounds
    ("SMI High (a=2.00)",            2.00,  RCS,  RED3, RED6, SMD3, SMD6),
    ("RCS High (g=3.00)",             SMI, 3.00,  RED3, RED6, SMD3, SMD6),
    ("RED3 High (dr3=1.00)",          SMI,  RCS,  1.00, RED6, SMD3, SMD6),
    ("RED6 High (dr6=1.00)",          SMI,  RCS,  RED3, 1.00, SMD3, SMD6),
    ("SMD3 High (ds3=1.00)",          SMI,  RCS,  RED3, RED6, 1.00, SMD6),
    ("SMD6 High (ds6=0.60)",          SMI,  RCS,  RED3, RED6, SMD3, 0.60),
    # Joint
    ("All LOW  (best case)",         0.00, 0.00,  0.50, 0.40, 0.50, 0.10),
    ("All HIGH (worst case)",        2.00, 3.00,  1.00, 1.00, 1.00, 0.60),
]

PASS_PP = 5.0
WARN_PP = 10.0
BJS     = {3: 0.68, 6: 0.79, 9: 0.83}

# Plain-English descriptions for each scenario
SC_DESC = {
    "Baseline":              "All parameters at calibrated values",
    "SMI Low  (a=0.00)":    "No supervision monitoring (structural zero)",
    "RCS Low  (g=0.00)":    "No risk differentiation across tiers",
    "RED3 Low (dr3=0.50)":  "Very rapid desistance after year 3",
    "RED6 Low (dr6=0.40)":  "Very rapid desistance after year 6",
    "SMD3 Low (ds3=0.50)":  "Rapid supervision intensity decay 3-6yr",
    "SMD6 Low (ds6=0.10)":  "Extreme supervision decay after year 6",
    "SMI High (a=2.00)":    "Extreme surveillance (2x calibrated)",
    "RCS High (g=3.00)":    "Maximum risk contrast (3x calibrated)",
    "RED3 High (dr3=1.00)": "No desistance effect after year 3",
    "RED6 High (dr6=1.00)": "No desistance effect after year 6",
    "SMD3 High (ds3=1.00)": "No supervision decay 3-6yr (persists)",
    "SMD6 High (ds6=0.60)": "Minimal supervision decay after year 6",
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
    if "Baseline"  in label: return C_BASE
    if "All LOW"   in label: return C_JOINT_L
    if "All HIGH"  in label: return C_JOINT_H
    if "Low"       in label: return C_LOW
    if "High"      in label: return C_HIGH
    return "#555555"

SC_COLORS = [_sc_color(s[0]) for s in SCENARIOS]

def _verdict(gap_pp):
    a = abs(gap_pp)
    if a <= PASS_PP:  return "ROBUST",      C_PASS, "#D6EAD6"
    if a <= WARN_PP:  return "ACCEPTABLE",  C_WARN, "#FFF3CD"
    return "IMPLAUSIBLE", C_FAIL, "#FADBD8"

def _changed(sc):
    parts = [f"{_PKEYS[i]}={sc[i+1]:.2f}"
             for i in range(6) if abs(sc[i+1]-_PBASE[i])>1e-6]
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
                          "oat_stress_output_0416", TIMESTAMP)
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
            initial_agents=RUN_CFG["initial_agents"],
            bias_factor=RUN_CFG["bias_factor"],
            monthly_intake=RUN_CFG["monthly_intake"],
            warmup_months=RUN_CFG["warmup_months"],
            study_months=RUN_CFG["study_months"],
            enable_peer_influence=RUN_CFG["enable_peer_influence"],
            weights=get_flat_risk_weights(),
            calibration_params=cal,
            seed=seed,
        )
        model.export_csv = False
        while model.running:
            model.step()

        # ── Aggregate rates (unchanged) ───────────────────────────────────────
        rates = {}
        for yrs in [3, 6, 9]:
            r = model.calculate_flag_rate(f"rearrest_{yrs}_yrs")
            rates[yrs] = r if r is not None else float("nan")

        # ── Tier rates at 3yr — computed directly from agents ─────────────────
        tier_low  = float("nan")
        tier_high = float("nan")
        try:
            eligible = [
                a for a in model.schedule.agents
                if getattr(a, "study_eligible_agent", False)
            ]
            if eligible:
                low_agents  = [a for a in eligible
                               if a.get_pcra_tier() == "Low" ]
                high_agents = [a for a in eligible
                               if a.get_pcra_tier() == "High"]
                if low_agents:
                    tier_low = sum(
                        1 for a in low_agents
                        if getattr(a, "rearrest_3_yrs", False)
                    ) / len(low_agents)
                if high_agents:
                    tier_high = sum(
                        1 for a in high_agents
                        if getattr(a, "rearrest_3_yrs", False)
                    ) / len(high_agents)
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
            "tier_spread_3yr": (tier_high - tier_low)
                               if np.isfinite(tier_low)
                               and np.isfinite(tier_high)
                               else float("nan"),
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

        # ── Aggregate rearrest rates (existing) ───────────────────────────────
        for yrs in (3, 6, 9):
            vals = [r[f"rate_{yrs}"] for r in runs
                    if r and np.isfinite(r[f"rate_{yrs}"])]
            n    = len(vals)
            mean = float(np.mean(vals)) if vals else float("nan")
            std  = float(np.std(vals))  if n > 1 else 0.0
            ci95 = (st.t.ppf(0.975, df=max(1,n-1)) * std / np.sqrt(n)
                    if n > 1 else 0.0)
            rows.append({"scenario":label,"window":yrs,
                         "mean":mean,"std":std,"ci95":ci95,"n":n})

        # ── Tier spread (new) ─────────────────────────────────────────────────
        # tier_spread = High tier 3yr rate − Low tier 3yr rate
        # A healthy model should maintain spread > 0.10 (10pp minimum)
        # Collapse toward zero means gamma has lost tier differentiation
        for tier in ["tier_low_3yr", "tier_high_3yr", "tier_spread_3yr"]:
            vals = [r[tier] for r in runs
                    if r and tier in r and np.isfinite(r[tier])]
            n    = len(vals)
            mean = float(np.mean(vals)) if vals else float("nan")
            std  = float(np.std(vals))  if n > 1 else 0.0
            ci95 = (st.t.ppf(0.975, df=max(1,n-1)) * std / np.sqrt(n)
                    if n > 1 else 0.0)
            rows.append({"scenario":label,"window":tier,
                         "mean":mean,"std":std,"ci95":ci95,"n":n})

    return pd.DataFrame(rows)

def _changed_params(sc_tuple):
    parts = [f"{_PKEYS[i]}={sc_tuple[i+1]:.2f}"
             for i in range(6) if abs(sc_tuple[i+1]-_PBASE[i])>1e-6]
    return "  ".join(parts) if parts else "—"

def _save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  -> {fname}")
    return path

def _leg_patches():
    return [
        mpatches.Patch(facecolor="#D6EAD6", edgecolor=C_PASS, lw=1.2,
                       label=f"ROBUST  (|gap| <= {PASS_PP:.0f}pp)"),
        mpatches.Patch(facecolor="#FFF3CD", edgecolor=C_WARN, lw=1.2,
                       label=f"ACCEPTABLE  ({PASS_PP:.0f}-{WARN_PP:.0f}pp)"),
        mpatches.Patch(facecolor="#FADBD8", edgecolor=C_FAIL, lw=1.2,
                       label=f"IMPLAUSIBLE  (> {WARN_PP:.0f}pp)"),
        mpatches.Patch(color=C_BASE,    label="Baseline"),
        mpatches.Patch(color=C_LOW,     label="LOW bound"),
        mpatches.Patch(color=C_HIGH,    label="HIGH bound"),
        mpatches.Patch(color=C_JOINT_L, label="All LOW (best case)"),
        mpatches.Patch(color=C_JOINT_H, label="All HIGH (worst case)"),
        mpatches.Patch(facecolor="#D6EAD6", edgecolor=C_PASS, lw=1.2,
                       label="Tier spread > 10pp  (\u03b3 differentiation healthy)"),
        mpatches.Patch(facecolor="#FADBD8", edgecolor=C_FAIL, lw=1.2,
                       label="Tier spread < 5pp   (\u03b3 differentiation collapsed)"),
    ]

# =============================================================================
# CHART 1 — Single window: absolute rates + gap panel
# =============================================================================
def chart_single_window(df, window_yrs):
    n   = len(SCENARIOS)
    bjs = BJS[window_yrs]
    sub = df[df["window"]==window_yrs].set_index("scenario")
    wc  = WIN_C[window_yrs]

    labels = [s[0] for s in SCENARIOS]
    means  = [sub.loc[l,"mean"] if l in sub.index else np.nan for l in labels]
    cis    = [sub.loc[l,"ci95"] if l in sub.index else 0.0   for l in labels]
    gaps   = [(m-bjs)*100 if np.isfinite(m) else np.nan for m in means]

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(C_BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.6,1.0],
                             wspace=0.08,
                             left=0.22, right=0.97,
                             top=0.91, bottom=0.10)

    fig.suptitle(
        f"Calibration Stress Test — {window_yrs}-Year Rearrest Rate\n"
        f"Each scenario perturbs one parameter to its structural extreme  |  "
        f"BJS benchmark = {bjs:.0%}  |  "
        f"Robust = |gap| <= {PASS_PP:.0f}pp  |  "
        f"Acceptable = {PASS_PP:.0f}-{WARN_PP:.0f}pp  |  "
        f"n = {N_REPS} reps x 20 seeds",
        fontsize=12, fontweight="bold", color=C_HDR, y=0.97,
    )

    # ── Panel A: absolute rates ───────────────────────────────────────────────
    axA = fig.add_subplot(gs[0])
    axA.set_facecolor("#FAFBFC")

    axA.axvspan(bjs-PASS_PP/100, bjs+PASS_PP/100,
                alpha=0.10, color=C_PASS, zorder=0,
                label=f"Robust zone (+/-{PASS_PP:.0f}pp)")
    axA.axvspan(bjs-WARN_PP/100, bjs-PASS_PP/100,
                alpha=0.06, color=C_WARN, zorder=0)
    axA.axvspan(bjs+PASS_PP/100, bjs+WARN_PP/100,
                alpha=0.06, color=C_WARN, zorder=0,
                label=f"Acceptable zone (+/-{WARN_PP:.0f}pp)")
    axA.axvline(bjs, color="#E53935", linewidth=2.2, zorder=4,
                label=f"BJS benchmark ({bjs:.0%})")

    # Group section backgrounds
    for y_lo, y_hi, bg in [
        (n-1.0, n+0.5, "#EEF4FB"),   # Baseline
        (n-7.5, n-1.0, "#EEF4FF"),   # LOW
        (n-13.5,n-7.5, "#FFF0EE"),   # HIGH
        (-0.5,  n-13.5,"#F5F5F5"),   # Joint
    ]:
        axA.axhspan(y_lo, y_hi, alpha=0.35, color=bg, zorder=0)

    for i, (lbl, m, ci) in enumerate(zip(labels, means, cis)):
        if not np.isfinite(m): continue
        color = SC_COLORS[i]
        y     = n - 1 - i
        axA.barh(y, m, height=0.62, color=color, alpha=0.85,
                 zorder=3, left=0,
                 xerr=ci,
                 error_kw={"elinewidth":1.4,"ecolor":"#444","capsize":4})
        gap      = (m - bjs)*100
        v,vc,vbg = _verdict(gap)
        axA.text(m + ci + 0.003, y, f"{m:.3f}",
                 va="center", fontsize=8.5, color=color, fontweight="bold")
        axA.text(0.985, y, f"{gap:+.1f}pp",
                 va="center", ha="right", fontsize=8,
                 color=vc, fontweight="bold",
                 transform=axA.get_yaxis_transform())

    axA.set_yticks(range(n))
    axA.set_yticklabels([s[0] for s in reversed(SCENARIOS)],
                         fontsize=9, fontweight="bold")
    for tick, sc in zip(axA.get_yticklabels(), reversed(SCENARIOS)):
        tick.set_color(_sc_color(sc[0]))

    for sep in [n-1.5, n-7.5, n-13.5]:
        axA.axhline(sep, color="#AAAAAA", lw=1.2, linestyle="--", zorder=2)

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
    axB.axvspan(-WARN_PP,-PASS_PP, alpha=0.06, color=C_WARN, zorder=0)
    axB.axvspan( PASS_PP, WARN_PP, alpha=0.06, color=C_WARN, zorder=0)
    axB.axvline(0, color="#E53935", lw=2.0, zorder=4)

    # Zone text labels
    for txt, xp in [("Robust", 0), ("Accept.", PASS_PP+2.5),
                    ("Accept.", -PASS_PP-2.5)]:
        axB.text(xp, n-0.3, txt, ha="center", fontsize=7,
                 color="#666666", style="italic")

    for i, (lbl, gap) in enumerate(zip(labels, gaps)):
        if not np.isfinite(gap): continue
        y        = n - 1 - i
        v,vc,vbg = _verdict(gap)
        axB.barh(y, gap, height=0.62,
                 color=_sc_color(lbl), alpha=0.85, zorder=3)
        axB.text(gap+(0.3 if gap>=0 else -0.3), y,
                 f"{gap:+.1f}pp",
                 va="center",
                 ha="left" if gap>=0 else "right",
                 fontsize=8.5, color=vc, fontweight="bold")

    for sep in [n-1.5, n-7.5, n-13.5]:
        axB.axhline(sep, color="#AAAAAA", lw=1.2, linestyle="--", zorder=2)

    axB.set_yticks(range(n))
    axB.set_yticklabels([])
    axB.set_xlabel("Gap vs BJS Benchmark (percentage points)", fontsize=11)
    axB.set_title("B — Gap to BJS Target (pp)",
                  fontsize=11, fontweight="bold", color=C_HDR)
    axB.grid(True, axis="x", zorder=1)

    # Shared legend
    fig.legend(handles=_leg_patches(), loc="lower center",
               ncol=4, fontsize=9, framealpha=0.95,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

    _save(fig, f"stress_{window_yrs}yr.png")

# =============================================================================
# CHART 2 — Summary: all three windows
# =============================================================================
def chart_summary(df):
    n      = len(SCENARIOS)
    labels = [s[0] for s in SCENARIOS]

    fig = plt.figure(figsize=(24, 12))
    fig.patch.set_facecolor(C_BG)

    gs = gridspec.GridSpec(1, 3, wspace=0.06,
                            left=0.18, right=0.97,
                            top=0.91, bottom=0.11)

    fig.suptitle(
        "Calibration Stress Test — Deviation from BJS Benchmarks\n"
        "Each bar shows how far the stressed model rate falls from the BJS target  |  "
        f"Green = Robust (|gap| <= {PASS_PP:.0f}pp)  |  "
        f"Amber = Acceptable ({PASS_PP:.0f}-{WARN_PP:.0f}pp)  |  "
        "Red = Implausible  |  "
        f"n = {N_REPS} reps x 20 seeds",
        fontsize=12, fontweight="bold", color=C_HDR, y=0.97,
    )

    for col_i, window_yrs in enumerate([3, 6, 9]):
        ax  = fig.add_subplot(gs[0, col_i])
        bjs = BJS[window_yrs]
        wc  = WIN_C[window_yrs]
        sub = df[df["window"]==window_yrs].set_index("scenario")

        gaps = [(sub.loc[l,"mean"]-bjs)*100
                if l in sub.index and np.isfinite(sub.loc[l,"mean"])
                else np.nan for l in labels]

        ax.set_facecolor("#FAFBFC")
        ax.axvspan(-PASS_PP, PASS_PP, alpha=0.12, color=C_PASS, zorder=0)
        ax.axvspan(-WARN_PP,-PASS_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvspan( PASS_PP, WARN_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvline(0, color="#E53935", lw=2.0, zorder=4,
                   label=f"BJS {bjs:.0%}")

        # Group section backgrounds
        for y_lo, y_hi, bg in [
            (n-1.0, n+0.5, "#EEF4FB"),
            (n-7.5, n-1.0, "#EEF4FF"),
            (n-13.5,n-7.5, "#FFF0EE"),
            (-0.5,  n-13.5,"#F5F5F5"),
        ]:
            ax.axhspan(y_lo, y_hi, alpha=0.4, color=bg, zorder=0)

        for i, (lbl, gap) in enumerate(zip(labels, gaps)):
            if not np.isfinite(gap): continue
            y        = n - 1 - i
            v,vc,vbg = _verdict(gap)
            ax.barh(y, gap, height=0.65,
                    color=_sc_color(lbl), alpha=0.85, zorder=3)
            # Value label — inside bar if space, outside if not
            if abs(gap) > 2:
                ax.text(gap/2, y, f"{gap:+.1f}pp",
                        va="center", ha="center",
                        fontsize=8, color="white", fontweight="bold")
            else:
                offset = 0.3 if gap >= 0 else -0.3
                ax.text(gap+offset, y, f"{gap:+.1f}pp",
                        va="center",
                        ha="left" if gap>=0 else "right",
                        fontsize=7.5, color=vc, fontweight="bold")

        for sep in [n-1.5, n-7.5, n-13.5]:
            ax.axhline(sep, color="#888888", lw=1.0, linestyle="--", zorder=2)

        ax.set_yticks(range(n))
        if col_i == 0:
            ax.set_yticklabels([s[0] for s in reversed(SCENARIOS)],
                               fontsize=8.5, fontweight="bold")
            for tick, sc in zip(ax.get_yticklabels(), reversed(SCENARIOS)):
                tick.set_color(_sc_color(sc[0]))
            # Group labels on left margin
            for y_ctr, txt, col in [
                (n-0.5,  "Baseline",    C_BASE),
                (n-4.0,  "LOW bounds",  C_LOW),
                (n-10.5, "HIGH bounds", C_HIGH),
                (n-13.5, "Joint",       "#444444"),
            ]:
                ax.text(-0.01, y_ctr, txt,
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center",
                        fontsize=7.5, fontweight="bold",
                        color=col, rotation=0)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Gap vs BJS Benchmark (pp)", fontsize=10)
        ax.set_title(
            f"{window_yrs}-Year Window\nBJS benchmark = {bjs:.0%}",
            fontsize=11, fontweight="bold", color=wc, pad=8,
        )
        ax.grid(True, axis="x", zorder=1)

    fig.legend(handles=_leg_patches(), loc="lower center",
               ncol=4, fontsize=9, framealpha=0.95,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

    _save(fig, "stress_summary.png")

# =============================================================================
# CHART 3 — Verdict scorecard  (per-cell axes)
# =============================================================================
def chart_scorecard(df):
    N       = len(SCENARIOS)
    FW, FH  = 26, 15

    # Vertical fractions
    Y_FOOT  = 0.000
    Y_LEG_T = 0.060
    Y_SUB   = 0.910
    Y_TTL   = 0.968
    HDR_H   = 0.062
    ROW_H   = (Y_SUB - 0.008 - Y_LEG_T - HDR_H) / N

    # REPLACE WITH THIS:
    LEFT   = 0.030   # left margin (room for section badges)
    RIGHT  = 0.995   # right edge
    USABLE = RIGHT - LEFT

    COL_SPEC = [
        ("Scenario",            2.20),
        ("What Changed",        0.88),
        ("Description",         1.90),
        ("3yr Rate",            0.60),
        ("Gap\n3yr",            0.65),
        ("Verdict\n3yr",        0.90),
        ("6yr Rate",            0.60),
        ("Gap\n6yr",            0.65),
        ("Verdict\n6yr",        0.90),
        ("9yr Rate",            0.60),
        ("Gap\n9yr",            0.65),
        ("Verdict\n9yr",        0.90),
        ("Overall",             0.80),
        ("Tier Spread\n(High-Low 3yr)", 0.85),   # NEW
    ]

    total_w = sum(rw for _, rw in COL_SPEC)
    col_x, col_w = [], []
    x = LEFT
    for _, rw in COL_SPEC:
        w = rw / total_w * USABLE
        col_x.append(x)
        col_w.append(w)
        x += w

    # Build COLS as (header, x, width) to keep rest of code unchanged
    COLS = [(h, col_x[i], col_w[i]) for i, (h, _) in enumerate(COL_SPEC)]

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor(C_BG)

    fig.text(0.50, Y_TTL,
             "Calibration Stress Test — Verdict Scorecard",
             ha="center", va="bottom",
             fontsize=16, fontweight="bold", color=C_HDR)
    fig.text(0.50, Y_SUB,
             f"Robust = |gap| <= {PASS_PP:.0f}pp  |  "
             f"Acceptable = {PASS_PP:.0f}-{WARN_PP:.0f}pp  |  "
             f"Implausible = > {WARN_PP:.0f}pp  |  "
             "BJS benchmarks: 3yr=68%  6yr=79%  9yr=83%  |  "
             f"n = {N_REPS} reps x 20 seeds per scenario",
             ha="center", va="bottom",
             fontsize=10, color="#444444", style="italic")

    def cell(x, y, w, h, text,
             bg=C_BG, fg="#1A2B3C",
             fs=9, bold=False, ha="center", pad=0.006):
        ac = fig.add_axes([x, y, w, h])
        ac.set_facecolor(bg)
        ac.set_xlim(0, 1); ac.set_ylim(0, 1)
        ac.axis("off")
        for sp in ac.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_edgecolor("#C0C8D0")
        tx = pad if ha == "left" else 0.5
        ac.text(tx, 0.5, str(text),
                ha=ha, va="center",
                fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=fg, clip_on=False, linespacing=1.3)

    # ── Window group headers above col triplets ───────────────────────────────
    y_grp_hdr = Y_LEG_T + ROW_H * N + HDR_H
    grp_h     = 0.025
    for label, x1, x2, wc in [
        ("3-Year Window",  COLS[3][1],  COLS[5][1]+COLS[5][2],  WIN_C[3]),
        ("6-Year Window",  COLS[6][1],  COLS[8][1]+COLS[8][2],  WIN_C[6]),
        ("9-Year Window",  COLS[9][1],  COLS[11][1]+COLS[11][2],WIN_C[9]),
    ]:
        ax_g = fig.add_axes([x1, y_grp_hdr, x2-x1, grp_h])
        ax_g.set_facecolor(wc)
        ax_g.axis("off")
        ax_g.text(0.5, 0.5, label,
                  ha="center", va="center",
                  fontsize=9, fontweight="bold", color="white")

    # Tier spread group header — same y-level as window headers
    ax_ts = fig.add_axes([COLS[13][1], y_grp_hdr,
                           COLS[13][2], grp_h])
    ax_ts.set_facecolor("#6A0572")
    ax_ts.axis("off")
    ax_ts.text(0.5, 0.5, "\u03b3 Tier Check",
               ha="center", va="center",
               fontsize=9, fontweight="bold", color="white")

    # ── Header row ────────────────────────────────────────────────────────────
    y_hdr = Y_LEG_T + ROW_H * N
    for hdr, cx, cw in COLS:
        cell(cx, y_hdr, cw, HDR_H, hdr,
             bg=C_HDR, fg="white", fs=8.5, bold=True)
    # NOTE: ax_ts has been removed from here — it is now above in the group headers block

    # ── Section divider bars ──────────────────────────────────────────────────
    SEC_DIVIDERS = {
        0:  ("Baseline",    C_BASE),
        1:  ("LOW Bounds",  C_LOW),
        7:  ("HIGH Bounds", C_HIGH),
        13: ("Joint Stress","#444444"),
    }

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ri, sc in enumerate(SCENARIOS):
        lbl  = sc[0]
        y    = y_hdr - (ri+1)*ROW_H
        bg   = C_ALT if ri % 2 == 0 else C_BG
        sc_c = _sc_color(lbl)

        # Section divider
        if ri in SEC_DIVIDERS:
            sec_lbl, sec_col = SEC_DIVIDERS[ri]
            # Left-margin badge
            ax_b = fig.add_axes([0.0, y+ROW_H*0.1,
                                  COLS[0][1]-0.002, ROW_H*0.8])
            ax_b.set_facecolor(sec_col)
            ax_b.axis("off")
            ax_b.text(0.5, 0.5, sec_lbl,
                      ha="center", va="center",
                      fontsize=6.5, fontweight="bold",
                      color="white", rotation=90)
            # Top divider line
            ax_d = fig.add_axes([COLS[0][1], y+ROW_H*0.96,
                                  COLS[-1][1]+COLS[-1][2]-COLS[0][1], 0.003])
            ax_d.set_facecolor(sec_col)
            ax_d.axis("off")

        # Gather rates / gaps / verdicts
        rates, gaps, verdicts = {}, {}, []
        for w in [3, 6, 9]:
            sub = df[(df["scenario"]==lbl) & (df["window"]==w)]
            if sub.empty or not np.isfinite(sub["mean"].iloc[0]):
                rates[w]=np.nan; gaps[w]=np.nan
                verdicts.append(("—","#888888",bg))
            else:
                m=sub["mean"].iloc[0]; g=(m-BJS[w])*100
                v,vc,vbg=_verdict(g)
                rates[w]=m; gaps[w]=g
                verdicts.append((v,vc,vbg))

        max_gap = max((abs(gaps[w]) for w in [3,6,9]
                       if np.isfinite(gaps[w])), default=0)
        if   max_gap<=PASS_PP: ov,ovc,obg="ROBUST",     C_PASS,"#D6EAD6"
        elif max_gap<=WARN_PP: ov,ovc,obg="ACCEPTABLE", C_WARN,"#FFF3CD"
        else:                  ov,ovc,obg="IMPLAUSIBLE",C_FAIL, "#FADBD8"

        chg  = _changed(sc)
        desc = SC_DESC.get(lbl, "")

        # Col 0 — Scenario
        cell(COLS[0][1],y,COLS[0][2],ROW_H,
             f"  {lbl}",bg=bg,fg=sc_c,fs=8.5,bold=True,ha="left")
        # Col 1 — What changed
        fc = "#B71C1C" if chg!="—" else "#888888"
        cell(COLS[1][1],y,COLS[1][2],ROW_H,
             chg,bg=bg,fg=fc,fs=7.5,bold=(chg!="—"))
        # Col 2 — Description
        cell(COLS[2][1],y,COLS[2][2],ROW_H,
             desc,bg=bg,fg="#333333",fs=7.5,ha="left")

        # Cols 3-11 — rate/gap/verdict per window
        ci = 3
        for wi, w in enumerate([3,6,9]):
            r        = rates[w]
            g        = gaps[w]
            v,vc,vbg = verdicts[wi]
            wc       = [WIN_C[3],WIN_C[6],WIN_C[9]][wi]

            cell(COLS[ci][1],y,COLS[ci][2],ROW_H,
                 f"{r:.3f}" if np.isfinite(r) else "—",
                 bg=bg,fg=wc,fs=8.5,bold=True)

            if np.isfinite(g):
                gc=(C_FAIL if g>PASS_PP else C_PASS if g<-PASS_PP else "#555")
                cell(COLS[ci+1][1],y,COLS[ci+1][2],ROW_H,
                     f"{g:+.1f}pp",bg=bg,fg=gc,fs=8.5,bold=True)
            else:
                cell(COLS[ci+1][1],y,COLS[ci+1][2],ROW_H,
                     "—",bg=bg,fg="#888",fs=8.5)

            cell(COLS[ci+2][1],y,COLS[ci+2][2],ROW_H,
                 v,bg=vbg,fg=vc,fs=8,bold=True)
            ci += 3

        # Col 12 — Overall
        cell(COLS[12][1],y,COLS[12][2],ROW_H,
             ov,bg=obg,fg=ovc,fs=9,bold=True)

        # Col 13 — Tier Spread (γ health check)  ← ADD THIS BLOCK
        tier_sub = df[(df["scenario"]==lbl) &
                      (df["window"]=="tier_spread_3yr")]
        if not tier_sub.empty and np.isfinite(tier_sub["mean"].iloc[0]):
            spread    = tier_sub["mean"].iloc[0] * 100   # convert to pp
            s_col = (C_PASS if spread > 10 else
                     C_WARN if spread > 5  else C_FAIL)
            s_bg  = ("#D6EAD6" if spread > 10 else
                     "#FFF3CD" if spread > 5  else "#FADBD8")
            s_txt = f"{spread:.1f}pp"
        else:
            s_col, s_bg, s_txt = "#888888", C_BG, "—"

        cell(COLS[13][1],y,COLS[13][2],ROW_H,
             s_txt,bg=s_bg,fg=s_col,fs=9,bold=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    fig.legend(handles=_leg_patches(),
               loc="lower center", ncol=4,
               fontsize=9.5, framealpha=0.95,
               edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, Y_FOOT),
               bbox_transform=fig.transFigure)

    path = os.path.join(OUT_DIR, "stress_scorecard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  -> stress_scorecard.png")

# =============================================================================
# CHART 4 — Deviation bars (3 windows side by side)
# =============================================================================
def chart_deviation(df):
    labels = [s[0] for s in SCENARIOS]
    n      = len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(22, 11), sharey=True)
    fig.patch.set_facecolor(C_BG)

    fig.suptitle(
        "Calibration Stress Test — Gap to BJS Benchmark by Scenario\n"
        "Bars show how far each stressed scenario deviates from BJS published targets  |  "
        f"Green shading = Robust (|gap| <= {PASS_PP:.0f}pp)  |  "
        f"Amber = Acceptable ({PASS_PP:.0f}-{WARN_PP:.0f}pp)  |  "
        "Red bars = Implausible (> 10pp)",
        fontsize=12, fontweight="bold", color=C_HDR, y=0.97,
    )

    for ax, window_yrs in zip(axes, [3, 6, 9]):
        bjs = BJS[window_yrs]
        wc  = WIN_C[window_yrs]
        sub = df[df["window"]==window_yrs].set_index("scenario")

        gaps = [(sub.loc[l,"mean"]-bjs)*100
                if l in sub.index and np.isfinite(sub.loc[l,"mean"])
                else np.nan for l in labels]

        ax.set_facecolor("#FAFBFC")
        ax.axvspan(-PASS_PP, PASS_PP, alpha=0.12, color=C_PASS, zorder=0)
        ax.axvspan(-WARN_PP,-PASS_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvspan( PASS_PP, WARN_PP, alpha=0.07, color=C_WARN, zorder=0)
        ax.axvline(0, color="#E53935", lw=2.0, zorder=4)

        # Zone labels inside chart
        for txt, xp, col in [
            ("ROBUST",     0,            "#1E7A34"),
            ("ACCEPT.",    PASS_PP+2.5,  "#B7770D"),
            ("ACCEPT.",   -PASS_PP-2.5,  "#B7770D"),
            ("IMPLAUS.",   WARN_PP+1.5,  "#C0392B"),
            ("IMPLAUS.",  -WARN_PP-1.5,  "#C0392B"),
        ]:
            if abs(xp) <= 25:
                ax.text(xp, n-0.2, txt, ha="center",
                        fontsize=6.5, color=col, style="italic",
                        fontweight="bold")

        for i, (lbl, gap) in enumerate(zip(reversed(labels), reversed(gaps))):
            if not np.isfinite(gap): continue
            y        = i
            v,vc,vbg = _verdict(gap)
            bar_color = _sc_color(lbl)
            ax.barh(y, gap, height=0.68,
                    color=bar_color, alpha=0.88, zorder=3)
            # Value label
            if abs(gap) > 3:
                ax.text(gap/2, y, f"{gap:+.1f}pp",
                        va="center", ha="center",
                        fontsize=8, color="white", fontweight="bold")
            else:
                offset = 0.3 if gap>=0 else -0.3
                ax.text(gap+offset, y, f"{gap:+.1f}pp",
                        va="center",
                        ha="left" if gap>=0 else "right",
                        fontsize=7.5, color=vc, fontweight="bold")

        for sep in [n-1.5, n-7.5, n-13.5]:
            ax.axhline(sep, color="#888888", lw=1.0, linestyle="--", zorder=2)

        ax.set_yticks(range(n))
        if window_yrs == 3:
            ax.set_yticklabels(list(reversed(labels)),
                               fontsize=9, fontweight="bold")
            for tick, sc in zip(ax.get_yticklabels(),
                                list(reversed(SCENARIOS))):
                tick.set_color(_sc_color(sc[0]))
            # Group labels
            for y_ctr, txt, col in [
                (n-0.5,  "Baseline",    C_BASE),
                (n-4.0,  "LOW bounds",  C_LOW),
                (n-10.5, "HIGH bounds", C_HIGH),
                (n-13.5, "Joint",       "#444444"),
            ]:
                ax.text(-0.01, y_ctr, txt,
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center",
                        fontsize=7.5, fontweight="bold", color=col)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Gap vs BJS Benchmark (pp)", fontsize=11)
        ax.set_title(
            f"{window_yrs}-Year Window\n"
            f"BJS benchmark = {bjs:.0%}",
            fontsize=11, fontweight="bold", color=wc, pad=10,
        )
        ax.grid(True, axis="x", zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(handles=_leg_patches(), loc="lower center",
               ncol=4, fontsize=9, framealpha=0.95,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    _save(fig, "stress_deviation.png")

def chart_disaggregated_smi(df):
    """
    Shows rearrest rates split by supervision status proxy —
    comparing Baseline vs SMI Low across tier groups to check
    whether SMI=0 affects all agents or only supervised ones.
    """
    scenarios_of_interest = [
        "Baseline",
        "SMI Low  (a=0.00)",
        "SMI High (a=2.00)",
        "All LOW  (best case)",
        "All HIGH (worst case)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(
        "Disaggregated Stress Check — Tier-Level Rearrest Rates by Scenario\n"
        "Checks whether SMI sensitivity operates uniformly or is tier-specific",
        fontsize=13, fontweight="bold", color=C_HDR, y=0.98
    )

    # ── Panel A: Tier rates side by side ──────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#FAFBFC")
    ax.set_title("A — Low vs High Tier 3yr Rearrest Rate by Scenario",
                 fontsize=11, fontweight="bold", color=C_HDR)

    x     = np.arange(len(scenarios_of_interest))
    width = 0.30

    low_means, high_means = [], []
    for sc in scenarios_of_interest:
        low_sub  = df[(df["scenario"]==sc) & (df["window"]=="tier_low_3yr")]
        high_sub = df[(df["scenario"]==sc) & (df["window"]=="tier_high_3yr")]
        low_means.append(
            low_sub["mean"].iloc[0]  if not low_sub.empty  else np.nan)
        high_means.append(
            high_sub["mean"].iloc[0] if not high_sub.empty else np.nan)

    bars1 = ax.bar(x - width/2, low_means,  width, label="Low Tier",
                   color=C_LOW,  alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width/2, high_means, width, label="High Tier",
                   color=C_HIGH, alpha=0.85, zorder=3)

    # Baseline reference lines
    bl_low  = low_means[0]
    bl_high = high_means[0]
    ax.axhline(bl_low,  color=C_LOW,  linewidth=1.2,
               linestyle="--", alpha=0.5, label="Baseline Low tier")
    ax.axhline(bl_high, color=C_HIGH, linewidth=1.2,
               linestyle="--", alpha=0.5, label="Baseline High tier")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8, color=C_LOW, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8, color=C_HIGH, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_of_interest, rotation=15,
                       ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("3-Year Rearrest Rate", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", zorder=0, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: Tier SPREAD — does SMI collapse differentiation? ─────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FAFBFC")
    ax2.set_title("B — Tier Spread (High − Low) by Scenario\n"
                  "Collapse toward 0 = SMI destroying risk differentiation",
                  fontsize=11, fontweight="bold", color=C_HDR)

    spreads, colors = [], []
    for sc in scenarios_of_interest:
        sp_sub = df[(df["scenario"]==sc) & (df["window"]=="tier_spread_3yr")]
        sp = sp_sub["mean"].iloc[0] if not sp_sub.empty else np.nan
        spreads.append(sp * 100 if np.isfinite(sp) else np.nan)
        colors.append(_sc_color(sc))

    bars3 = ax2.bar(x, spreads, width=0.5,
                    color=colors, alpha=0.85, zorder=3)

    # 10pp threshold line
    ax2.axhline(10, color=C_PASS, linewidth=1.8, linestyle="--",
                label="Healthy threshold (10 pp)", zorder=4)
    ax2.axhline(5,  color=C_WARN, linewidth=1.2, linestyle=":",
                label="Minimum threshold (5 pp)",  zorder=4)
    ax2.axhline(0,  color="#999", linewidth=0.8, zorder=2)

    for bar, sp in zip(bars3, spreads):
        if np.isfinite(sp):
            col = (C_PASS if sp > 10 else C_WARN if sp > 5 else C_FAIL)
            ax2.text(bar.get_x() + bar.get_width()/2,
                     sp + (0.3 if sp >= 0 else -1.0),
                     f"{sp:.1f}pp",
                     ha="center", va="bottom" if sp >= 0 else "top",
                     fontsize=9, color=col, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios_of_interest, rotation=15,
                        ha="right", fontsize=9)
    ax2.set_ylabel("Tier Spread (percentage points)", fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, axis="y", zorder=0, alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Annotation box explaining what to look for
    ax2.text(0.98, 0.97,
             "Key question:\nDoes SMI Low collapse\nthe spread toward 0?\n"
             "If yes → SMI is a global\nmechanism affecting all tiers.\n"
             "If no → SMI only affects\naggregate rate via composition.",
             transform=ax2.transAxes,
             ha="right", va="top", fontsize=8, color="#333",
             style="italic",
             bbox=dict(fc="#fffff0", ec="#ccc", boxstyle="round,pad=0.4"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "stress_disaggregated_smi.png")
    print("\n  Disaggregated SMI check saved.")
# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
def print_summary(df):
    print("\n" + "="*75)
    print("  CALIBRATION STRESS TEST — RESULTS SUMMARY")
    print("="*75)
    print(f"\n  Calibrated baseline:")
    for k, v in zip(_PKEYS, _PBASE):
        print(f"    {k} = {v:.3f}")
    print()
    print(f"  {'Scenario':<28}  "
          f"{'3yr':>6}  {'6yr':>6}  {'9yr':>6}  "
          f"{'Gap3':>8}  {'Gap6':>8}  {'Gap9':>8}  Verdict")
    print(f"  {'-'*82}")
    for sc in SCENARIOS:
        lbl = sc[0]
        sub = df[df["scenario"]==lbl].set_index("window")
        r   = {y: sub.loc[y,"mean"] if y in sub.index else np.nan
               for y in [3,6,9]}
        g   = {y: (r[y]-BJS[y])*100 for y in [3,6,9]}
        v   = max(abs(g[y]) for y in [3,6,9] if np.isfinite(g[y]))
        verdict = ("ROBUST" if v<=PASS_PP else
                   "ACCEPTABLE" if v<=WARN_PP else "IMPLAUSIBLE")
        print(f"  {lbl:<28}  "
              f"{r[3]:>6.3f}  {r[6]:>6.3f}  {r[9]:>6.3f}  "
              f"{g[3]:>+7.1f}pp  {g[6]:>+7.1f}pp  {g[9]:>+7.1f}pp  "
              f"{verdict}")
    print(f"\n  Output: {OUT_DIR}")
    print("="*75)

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
        SEEDS  = [42,137,251,389,503,617,743,863,971,1087,
                  1201,1319,1433,1549,1663,1777,1889,2003,2111,2221]

        print(f"\n  Calibration Stress Test")
        print(f"  Scenarios  : {len(SCENARIOS)}")
        print(f"  Reps x Seeds: {N_REPS} x {len(SEEDS)} = "
              f"{N_REPS*len(SEEDS)} runs per scenario")
        print(f"  Total runs : {len(SCENARIOS)*N_REPS*len(SEEDS)}")
        print(f"  Output     : {OUT_DIR}\n")
        print(f"  Baseline: SMI={SMI} RCS={RCS} RED3={RED3} "
              f"RED6={RED6} SMD3={SMD3} SMD6={SMD6}\n")

        jobs = [(*sc, seed)
                for sc in SCENARIOS
                for rep in range(N_REPS)
                for seed in SEEDS]
        raw  = Parallel(n_jobs=args.cores)(
            delayed(_run)(j) for j in jobs)

        df = aggregate([r for r in raw if r])
        df.to_csv(os.path.join(OUT_DIR, "stress_results.csv"),
                  index=False)

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