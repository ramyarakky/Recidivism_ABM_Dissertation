#!/usr/bin/env python3
"""
recidivism_validation.py
========================
Validation suite for Recidivism ABM.

Charts generated
----------------
1. Cumulative rearrest rate  — Years 1–9 (aggregate)    [Δ labels ALL years]
2. Cumulative rearrest rate  — 3yr / 6yr / 9yr bar chart vs BJS
3. Cumulative rearrest rate  — Years 1–9 by offense type [Δ labels ALL years]
4. Cumulative rearrest rate  — 3yr / 6yr / 9yr by offense type
5. Non-cumulative (annual first-arrest %) — Years 1–9 aggregate [Δ labels ALL years]

BJS benchmarks (Alper, Durose & Markman, 2018. NCJ 250975)
-----------------------------------------------------------
Aggregate cumulative:
    Year 1: 43.9%  Year 2: 60.1%  Year 3: 68.4%  Year 4: 73.5%
    Year 5: 77.0%  Year 6: 79.4%  Year 7: 81.1%  Year 8: 82.4%  Year 9: 83.4%

By offense type (Table 7):
    Violent:      38.9 / 54.2 / 62.2 / 67.6 / 71.6 / 74.2 / 76.1 / 77.7 / 78.7
    Property:     50.8 / 67.1 / 75.0 / 79.6 / 82.4 / 84.4 / 85.8 / 86.9 / 87.8
    Drug:         42.8 / 59.9 / 68.6 / 73.9 / 77.5 / 79.8 / 81.5 / 82.7 / 83.8
    Public order: 40.5 / 55.9 / 65.0 / 70.2 / 74.1 / 76.9 / 79.2 / 80.6 / 81.9
"""

import os
import sys
import time
import math
import random
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_uncalibrated_params,
    get_global_calibration_params,
)

# =============================================================================
# BJS BENCHMARKS  (Alper et al., 2018, NCJ 250975)
# =============================================================================
BJS_YEARS = list(range(1, 10))
ANCHOR_YEARS = [3, 6, 9]

# Table 7 — cumulative % arrested by year since release
BJS_CUMULATIVE_ALL = [43.9, 60.1, 68.4, 73.5, 77.0, 79.4, 81.1, 82.4, 83.4]

BJS_BY_OFFENSE = {
    "Violent":      [38.9, 54.2, 62.2, 67.6, 71.6, 74.2, 76.1, 77.7, 78.7],
    "Property":     [50.8, 67.1, 75.0, 79.6, 82.4, 84.4, 85.8, 86.9, 87.8],
    "Drug":         [42.8, 59.9, 68.6, 73.9, 77.5, 79.8, 81.5, 82.7, 83.8],
    "Public order": [40.5, 55.9, 65.0, 70.2, 74.1, 76.9, 79.2, 80.6, 81.9],
}

# Non-cumulative: fraction with FIRST arrest in each year
# Derived from BJS cumulative: diff[0]=BJS[0], diff[i]=BJS[i]-BJS[i-1]
BJS_FIRST_ARREST = (
    [BJS_CUMULATIVE_ALL[0]]
    + [BJS_CUMULATIVE_ALL[i] - BJS_CUMULATIVE_ALL[i-1]
       for i in range(1, 9)]
)

# Compact targets for bar chart
BJS_TARGETS = {3: 68.4, 6: 79.4, 9: 83.4}

# =============================================================================
# ABM OFFENSE MAPPING  →  BJS offense groups
# =============================================================================
OFFENSE_MAP = {
    "Violent/Non-Sex":      "Violent",
    "Violent-SexOffender":  "Violent",
    "Drug":                 "Drug",
    "Property":             "Property",
    "Other(PublicOrder)":   "Public order",
}
OFFENSE_GROUPS  = ["Violent", "Drug", "Property", "Public order"]
OFFENSE_COLOURS = {
    "Violent":      "#C0392B",
    "Drug":         "#8E44AD",
    "Property":     "#2980B9",
    "Public order": "#27AE60",
}

# =============================================================================
# SIMULATION CONFIG
# =============================================================================
BASE_PARAMS = {
    "initial_agents":       1000,
    "warmup_months":        144,
    "study_months":         108,
    "monthly_intake":       10,
    "bias_factor":          0,
    "enable_peer_influence": True,
}

# =============================================================================
# UTILITIES
# =============================================================================
_T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,
        9:2.262,10:2.228,15:2.131,20:2.086,30:2.042,60:2.000,120:1.980}

def _t95(df):
    for k in sorted(_T95):
        if df <= k: return _T95[k]
    return 1.96

def mean_ci(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    n = len(v)
    if n == 0: return float("nan"), float("nan"), float("nan")
    if n == 1: m=float(v[0]); return m, m, m
    m = sum(v)/n
    sd = math.sqrt(sum((x-m)**2 for x in v)/(n-1))
    h = _t95(n-1)*sd/math.sqrt(n)
    return m, m-h, m+h

def detect_workers(n_tasks):
    logical  = os.cpu_count() or 4
    physical = max(1, logical//2)
    return min(n_tasks, max(1, physical-2))

# =============================================================================
# GAP-LABEL HELPERS  (shared across all charts)
# =============================================================================
# Dissertation-standard thresholds: ±2pp robust, ±5pp acceptable, else off-target
GAP_GOOD    = 2.0
GAP_WARN    = 5.0
COLOUR_GOOD = "#27AE60"   # green  — within 2pp
COLOUR_WARN = "#F39C12"   # amber  — within 5pp
COLOUR_BAD  = "#C0392B"   # red    — exceeds 5pp

def gap_colour(diff: float) -> str:
    """Return the colour for a gap (in pp) using the 2pp / 5pp thresholds."""
    a = abs(diff)
    if a <= GAP_GOOD: return COLOUR_GOOD
    if a <= GAP_WARN: return COLOUR_WARN
    return COLOUR_BAD

def _add_gap_label(fig, x, y, diff, row=None, col=None, fontsize=9,
                   prefix="Δ"):
    """Add a single Δ annotation to the figure with threshold-based colour."""
    kw = {}
    if row is not None: kw["row"] = row
    if col is not None: kw["col"] = col
    fig.add_annotation(
        x=x, y=y,
        text=f"{prefix} {diff:+.1f}pp",
        showarrow=False,
        font=dict(size=fontsize, color=gap_colour(diff),
                  family="Georgia, serif"),
        **kw,
    )

def _add_year_by_year_gaps(fig, years, abm_vals, bjs_vals, y_offset=4.0,
                            row=None, col=None, fontsize=9, prefix="Δ"):
    """
    Add Δ labels for EVERY year in the series.
    - Positions each label above the higher of (ABM, BJS) for that year.
    - Alternates vertical offset slightly for odd/even years to reduce overlap.
    """
    for i, yr in enumerate(years):
        diff   = abm_vals[i] - bjs_vals[i]
        anchor = max(abm_vals[i], bjs_vals[i])
        # Alternate tiny vertical stagger so adjacent labels don't collide
        stagger = 0.0 if i % 2 == 0 else 1.8
        _add_gap_label(fig, x=yr, y=anchor + y_offset + stagger,
                       diff=diff, row=row, col=col,
                       fontsize=fontsize, prefix=prefix)

def _gap_legend_annotation(fig, x=0.99, y=-0.12, xref="paper", yref="paper"):
    """
    Add a small legend explaining the Δ colour thresholds.
    Placed below the chart so it doesn't interfere with data.
    """
    fig.add_annotation(
        x=x, y=y, xref=xref, yref=yref,
        text=(f"<span style='color:{COLOUR_GOOD}'>● within ±2pp (robust)</span>  "
              f"<span style='color:{COLOUR_WARN}'>● within ±5pp (acceptable)</span>  "
              f"<span style='color:{COLOUR_BAD}'>● beyond ±5pp (off-target)</span>"),
        showarrow=False, align="right",
        font=dict(size=10, family="Georgia, serif"),
    )

# =============================================================================
# WORKER — extracts year-by-year and offense-stratified data
# =============================================================================
def run_worker(task):
    run_id = int(task["run_id"])
    seed   = int(task["seed"])
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception: pass

    params = dict(BASE_PARAMS)
    params["seed"] = seed
    params["calibration_params"] = task["calibration_params"]

    model = RecidivismModel(**params)
    #model.export_csv = False
    model.max_months = model.warmup_months + model.study_months

    try:
        while model.running:
            model.step()
    except Exception as e:
        print(f"  Worker error run={run_id} seed={seed}: {e}")

    # ── Collect per-agent data ────────────────────────────────────────────────
    eligible = [a for a in model.schedule.agents
                if getattr(a, "study_eligible_agent", False)]
    n_total  = len(eligible)
    if n_total == 0:
        return {"run_id": run_id, "seed": seed, "n": 0,
                "cum_all": [0]*9, "first_all": [0]*9,
                "cum_offense": {g: [0]*9 for g in OFFENSE_GROUPS},
                "n_offense": {g: 0 for g in OFFENSE_GROUPS}}

    # Year of first rearrest for each agent (community-months-based)
    # community_months_at_risk is the months at risk when rearrest occurred
    # Year = ceil(community_months / 12) capped at 9
    def get_rearrest_year(agent):
        if not getattr(agent, "recidivated_agent", False):
            return None
        cm = getattr(agent, "community_months_at_risk", 0)
        if cm <= 0:
            # Fallback: use rearrest_month relative to warmup end
            rm = getattr(agent, "rearrest_month", None)
            if rm is None: return None
            cm = rm - model.warmup_months
        yr = math.ceil(cm / 12)
        return max(1, min(9, yr))

    # Build cumulative and first-arrest arrays
    # cum[y-1] = fraction arrested by end of year y
    # first[y-1] = fraction with FIRST arrest in year y
    cum_counts  = [0]*9   # cumulative by year y (index y-1)
    first_counts = [0]*9  # first-arrest in year y

    offense_counts = {g: [0]*9 for g in OFFENSE_GROUPS}
    offense_totals = {g: 0     for g in OFFENSE_GROUPS}

    for agent in eligible:
        yr = get_rearrest_year(agent)
        offense_raw = getattr(agent, "offense", "Other(PublicOrder)")
        og = OFFENSE_MAP.get(offense_raw, "Public order")
        offense_totals[og] += 1

        if yr is not None:
            # first-arrest count
            first_counts[yr-1] += 1
            # cumulative: all years >= yr get a count
            for y in range(yr, 10):
                cum_counts[y-1] += 1
            # offense-stratified cumulative
            for y in range(yr, 10):
                offense_counts[og][y-1] += 1

    cum_all   = [c/n_total*100 for c in cum_counts]
    first_all = [c/n_total*100 for c in first_counts]
    cum_off   = {g: [offense_counts[g][i]/offense_totals[g]*100
                     if offense_totals[g] > 0 else 0.0
                     for i in range(9)]
                 for g in OFFENSE_GROUPS}

    return {
        "run_id":     run_id,
        "seed":       seed,
        "n":          n_total,
        "cum_all":    cum_all,
        "first_all":  first_all,
        "cum_offense": cum_off,
        "n_offense":  offense_totals,
    }

# =============================================================================
# TASK BUILDER
# =============================================================================
def build_tasks(n_runs, seeds_per_run, base_seed=1000, cal=None):
    if cal is None: cal = get_global_calibration_params()
    tasks = []
    for run_id in range(1, n_runs+1):
        for i in range(seeds_per_run):
            tasks.append({"run_id": run_id,
                          "seed": base_seed + run_id*10_000 + i,
                          "calibration_params": cal})
    return tasks

# =============================================================================
# RUN PARALLEL
# =============================================================================
def run_parallel(tasks, max_workers=None):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                c = r["cum_all"]
                print(f"  Run {r['run_id']:02d} seed {r['seed']} | "
                      f"3yr={c[2]:.1f}% 6yr={c[5]:.1f}% 9yr={c[8]:.1f}%")
            except Exception as e:
                print(f"  FAILED run={t['run_id']} seed={t['seed']}: {e}")
    return results

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
def aggregate(results):
    """
    Returns:
        cum_mean / cum_lo / cum_hi     : lists of 9 floats (aggregate cumulative)
        first_mean / first_lo / first_hi : lists of 9 floats (non-cumulative)
        off_mean / off_lo / off_hi     : dict[offense] -> list of 9 floats
    """
    cum_mean, cum_lo, cum_hi = [], [], []
    for y in range(9):
        vals = [r["cum_all"][y] for r in results]
        m, lo, hi = mean_ci(vals)
        cum_mean.append(m); cum_lo.append(lo); cum_hi.append(hi)

    first_mean, first_lo, first_hi = [], [], []
    for y in range(9):
        vals = [r["first_all"][y] for r in results]
        m, lo, hi = mean_ci(vals)
        first_mean.append(m); first_lo.append(lo); first_hi.append(hi)

    off_mean = {g: [] for g in OFFENSE_GROUPS}
    off_lo   = {g: [] for g in OFFENSE_GROUPS}
    off_hi   = {g: [] for g in OFFENSE_GROUPS}
    for g in OFFENSE_GROUPS:
        for y in range(9):
            vals = [r["cum_offense"][g][y] for r in results]
            m, lo, hi = mean_ci(vals)
            off_mean[g].append(m); off_lo[g].append(lo); off_hi[g].append(hi)

    return (cum_mean, cum_lo, cum_hi,
            first_mean, first_lo, first_hi,
            off_mean, off_lo, off_hi)

# =============================================================================
# STYLE HELPERS
# =============================================================================
_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Georgia, serif", color="#1A2B3C", size=13),
    paper_bgcolor="#FAFBFC",
    plot_bgcolor="#FAFBFC",
)
_C_ABM  = "#2F6FB2"
_C_BJS  = "#E07B39"
_C_GRID = "#E8EDF2"

def _save(fig, path):
    try:
        fig.write_image(path, scale=2)
        print(f"  Saved -> {path}")
    except Exception:
        print(f"  kaleido not available, skipping PNG: {path}")
    try:
        fig.write_html(path.replace(".png", ".html"))
    except Exception:
        pass

def _add_bjs_line(fig, y_vals, label, colour, row=None, col=None):
    kw = {}
    if row: kw["row"] = row
    if col: kw["col"] = col
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=y_vals, mode="lines+markers",
        name=label, line=dict(color=colour, width=2, dash="dot"),
        marker=dict(symbol="diamond", size=7, color=colour),
        hovertemplate=f"<b>BJS {label}</b><br>Year %{{x}}<br>%{{y:.1f}}%<extra></extra>",
    ), **kw)

# =============================================================================
# CHART 1 — Cumulative rearrest Years 1–9 (aggregate)  [Δ labels for ALL years]
# =============================================================================
def chart1_cumulative_by_year(cum_mean, cum_lo, cum_hi, out_dir):
    fig = go.Figure()

    # ABM line with CI band
    fig.add_trace(go.Scatter(
        x=BJS_YEARS+BJS_YEARS[::-1],
        y=cum_hi+cum_lo[::-1],
        fill="toself", fillcolor=f"rgba(47,111,178,0.15)",
        line_color="rgba(0,0,0,0)", showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=cum_mean, mode="lines+markers",
        name="ABM calibrated",
        line=dict(color=_C_ABM, width=3),
        marker=dict(size=8, color=_C_ABM),
        error_y=dict(type="data", symmetric=False,
                     array=[h-m for m,h in zip(cum_mean,cum_hi)],
                     arrayminus=[m-l for m,l in zip(cum_mean,cum_lo)],
                     color=_C_ABM, thickness=1.5, width=5),
        hovertemplate="<b>ABM Year %{x}</b><br>%{y:.1f}%<extra></extra>",
    ))

    # BJS line
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=BJS_CUMULATIVE_ALL,
        mode="lines+markers", name="BJS NCJ 250975",
        line=dict(color=_C_BJS, width=2.5, dash="dot"),
        marker=dict(symbol="diamond", size=8, color=_C_BJS),
        hovertemplate="<b>BJS Year %{x}</b><br>%{y:.1f}%<extra></extra>",
    ))

    # Δ annotations for EVERY year (was: only 3, 6, 9)
    _add_year_by_year_gaps(fig, BJS_YEARS, cum_mean, BJS_CUMULATIVE_ALL,
                           y_offset=4.0, fontsize=10)

    fig.update_layout(**_LAYOUT,
        title=dict(
            text="<b>Chart 1 — Cumulative Rearrest Rate: Years 1–9</b><br>"
                 "<sup>ABM calibrated vs BJS NCJ 250975 (Alper et al., 2018)  |  "
                 "Shaded band = 95% CI across seeds  |  Δ = ABM − BJS (pp) at each year</sup>",
            font=dict(size=15), x=0.5),
        xaxis=dict(title="Years since release", tickvals=BJS_YEARS,
                   showgrid=False, range=[0.5, 9.5]),
        yaxis=dict(title="Cumulative rearrest rate (%)",
                   range=[30, 100], gridcolor=_C_GRID, zeroline=False),
        legend=dict(x=0.05, y=0.97),
        margin=dict(t=100, b=90, l=70, r=40),
    )
    _gap_legend_annotation(fig)
    _save(fig, os.path.join(out_dir, "chart1_cumulative_by_year.png"))

# =============================================================================
# CHART 2 — Cumulative rearrest at 3yr / 6yr / 9yr (bar chart)
# =============================================================================
def chart2_cumulative_bar(cum_mean, cum_lo, cum_hi, out_dir):
    windows  = ["3-Year", "6-Year", "9-Year"]
    y_idx    = [2, 5, 8]
    abm_vals = [cum_mean[i] for i in y_idx]
    abm_lo   = [cum_lo[i]   for i in y_idx]
    abm_hi   = [cum_hi[i]   for i in y_idx]
    bjs_vals = [BJS_TARGETS[k] for k in [3,6,9]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ABM calibrated", x=windows, y=abm_vals,
        marker_color=_C_ABM, marker_line_width=0,
        error_y=dict(type="data", symmetric=False,
                     array=[h-m for m,h in zip(abm_vals,abm_hi)],
                     arrayminus=[m-l for m,l in zip(abm_vals,abm_lo)],
                     color=_C_ABM, thickness=2, width=6),
        text=[f"{v:.1f}%" for v in abm_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="BJS NCJ 250975", x=windows, y=bjs_vals,
        marker_color=_C_BJS, marker_line_width=0,
        text=[f"{v:.1f}%" for v in bjs_vals], textposition="outside",
    ))
    # Δ labels above each window — using shared helper for consistent colouring
    for w, av, bv in zip(windows, abm_vals, bjs_vals):
        diff = av - bv
        _add_gap_label(fig, x=w, y=max(av,bv)+8, diff=diff, fontsize=11)

    fig.update_layout(**_LAYOUT,
        title=dict(
            text="<b>Chart 2 — Cumulative Rearrest Rate at 3, 6 and 9 Years</b><br>"
                 "<sup>ABM calibrated vs BJS NCJ 250975 (Alper et al., 2018)  |  "
                 "Error bars = 95% CI  |  Δ = ABM − BJS (pp)</sup>",
            font=dict(size=15), x=0.5),
        barmode="group", bargap=0.30, bargroupgap=0.08,
        yaxis=dict(title="Cumulative rearrest rate (%)",
                   range=[0, 105], gridcolor=_C_GRID, zeroline=False),
        xaxis=dict(showgrid=False),
        legend=dict(x=0.05, y=0.97),
        margin=dict(t=100, b=90, l=70, r=40),
    )
    _gap_legend_annotation(fig)
    _save(fig, os.path.join(out_dir, "chart2_cumulative_bar.png"))

"""
Drop-in replacement for chart3_cumulative_by_offense() in recidivism_validation.py

Design:
  - Landscape 4-across layout (1 row × 4 columns instead of 2×2)
  - ABM vs BJS trajectories with CI band
  - Δ labels at Years 3, 6, 9 ONLY (the dissertation anchor windows)
  - Summary box in corner: Mean Δ across all 9 years + worst year
  - No residual strip — streamlined for slides/print

Replace the existing chart3_cumulative_by_offense() in recidivism_validation.py
with the function below. The _subplot_idx helper is no longer needed.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Years where Δ labels are shown — matches Stage 1 BJS calibration anchors
ANCHOR_YEARS = [3, 6, 9]


def chart3_cumulative_by_offense(off_mean, off_lo, off_hi, out_dir):
    """
    Landscape 4-panel chart: one offense per column.
    Δ labels shown only at dissertation anchor windows (years 3, 6, 9),
    keeping trajectories readable while preserving the key calibration
    reference points.
    """
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"<b>{g}</b>" for g in OFFENSE_GROUPS],
        shared_yaxes=True,
        horizontal_spacing=0.035,
    )

    for col_idx, g in enumerate(OFFENSE_GROUPS, start=1):
        colour  = OFFENSE_COLOURS[g]
        abm_m   = off_mean[g]
        abm_h   = off_hi[g]
        abm_l   = off_lo[g]
        bjs_off = BJS_BY_OFFENSE[g]
        diffs   = [abm_m[i] - bjs_off[i] for i in range(9)]

        # ── CI band ──────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=BJS_YEARS + BJS_YEARS[::-1],
            y=abm_h + abm_l[::-1],
            fill="toself",
            fillcolor=(f"rgba({int(colour[1:3],16)},"
                       f"{int(colour[3:5],16)},"
                       f"{int(colour[5:7],16)},0.15)"),
            line_color="rgba(0,0,0,0)",
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col_idx)

        # ── ABM trajectory ───────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=abm_m, mode="lines+markers",
            name=f"ABM — {g}", legendgroup=g,
            line=dict(color=colour, width=2.8),
            marker=dict(size=7, color=colour),
            hovertemplate=(f"<b>ABM {g} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}%<extra></extra>"),
        ), row=1, col=col_idx)

        # ── BJS trajectory ───────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=bjs_off, mode="lines+markers",
            name=f"BJS — {g}", legendgroup=f"bjs_{g}",
            line=dict(color=colour, width=1.8, dash="dot"),
            marker=dict(symbol="diamond", size=6, color=colour),
            hovertemplate=(f"<b>BJS {g} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}%<extra></extra>"),
        ), row=1, col=col_idx)

        # ── Δ labels at anchor years only (3, 6, 9) ─────────────────────────
        for anchor_yr in ANCHOR_YEARS:
            i    = anchor_yr - 1
            diff = diffs[i]
            # Anchor label slightly above whichever line is higher that year
            anchor_y = max(abm_m[i], bjs_off[i]) + 4
            _add_gap_label(
                fig, x=anchor_yr, y=anchor_y, diff=diff,
                row=1, col=col_idx, fontsize=9,
            )

        # ── Corner summary box ───────────────────────────────────────────────
        # Shows aggregate fit quality across ALL 9 years, not just the anchors.
        # This way the reader gets the 'headline number' without needing every
        # label drawn on the curve.
        mean_gap  = sum(diffs) / len(diffs)
        worst_idx = max(range(9), key=lambda j: abs(diffs[j]))
        worst_yr  = BJS_YEARS[worst_idx]
        worst_gap = diffs[worst_idx]

        # xref/yref for per-subplot domain positioning
        axis_suffix = "" if col_idx == 1 else str(col_idx)
        fig.add_annotation(
            x=0.97, y=0.08,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=(f"<b>Mean Δ:</b> {mean_gap:+.1f}pp<br>"
                  f"<b>Worst:</b> Yr {worst_yr} ({worst_gap:+.1f}pp)"),
            showarrow=False, align="right",
            font=dict(size=9, family="Georgia, serif", color="#333333"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CCCCCC", borderwidth=1,
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    fig.update_xaxes(
        title_text="Years since release",
        tickvals=BJS_YEARS, showgrid=False, range=[0.5, 9.5],
    )
    fig.update_yaxes(
        range=[25, 105], gridcolor=_C_GRID, zeroline=False,
    )
    # Only the leftmost panel gets the y-axis title (shared_yaxes hides ticks
    # on cols 2-4, but we still want the title on col 1)
    fig.update_yaxes(title_text="Cumulative rearrest rate (%)",
                     row=1, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Chart 3 — Cumulative Rearrest Rate by Offense Type: "
                  "Years 1–9</b><br>"
                  "<sup>ABM (solid) vs BJS NCJ 250975 (dotted)  |  "
                  "95% CI band  |  Δ labels at anchor years (3, 6, 9)  |  "
                  "Alper et al. (2018), Table 7</sup>"),
            font=dict(size=14), x=0.5),
        height=440,
        width=1600,
        legend=dict(x=1.01, y=0.99, tracegroupgap=10, font=dict(size=10)),
        margin=dict(t=95, b=80, l=70, r=180),
    )

    _gap_legend_annotation(fig, x=0.5, y=-0.18)
    _save(fig, os.path.join(out_dir, "chart3_cumulative_by_offense.png"))

def _subplot_idx(row, col, n_cols):
    """
    Compute the subplot axis suffix number used by plotly for annotation xref/yref.
    For a subplot at (row, col) in a grid with n_cols columns,
    the plotly axis index is (row-1)*n_cols + col.
    Axis 1 is the default (no suffix); axes 2+ get suffixes like 'x2', 'y2'.
    """
    idx = (row - 1) * n_cols + col
    return "" if idx == 1 else str(idx)
# =============================================================================
# CHART 4 — Cumulative rearrest at 3yr / 6yr / 9yr by offense type
# =============================================================================
def chart4_bar_by_offense(off_mean, off_lo, off_hi, out_dir):
    windows = [3, 6, 9]
    y_idx   = [2, 5, 8]
    x_lbls  = ["3-Year", "6-Year", "9-Year"]

    fig = go.Figure()
    bw  = 0.15
    x_base = [1, 2, 3]

    for i, g in enumerate(OFFENSE_GROUPS):
        colour  = OFFENSE_COLOURS[g]
        abm_m   = [off_mean[g][j] for j in y_idx]
        abm_h   = [off_hi[g][j]   for j in y_idx]
        abm_l   = [off_lo[g][j]   for j in y_idx]
        bjs_v   = [BJS_BY_OFFENSE[g][j] for j in y_idx]
        offset  = (i - 1.5) * bw * 2.2
        xs      = [x + offset for x in x_base]

        # ABM bars
        fig.add_trace(go.Bar(
            name=f"ABM — {g}", x=xs, y=abm_m,
            width=bw*1.8, marker_color=colour, marker_line_width=0,
            legendgroup=g,
            error_y=dict(type="data", symmetric=False,
                         array=[h-m for m,h in zip(abm_m,abm_h)],
                         arrayminus=[m-l for m,l in zip(abm_m,abm_l)],
                         color=colour, thickness=1.5, width=4),
            hovertemplate=f"<b>ABM {g}</b><br>%{{y:.1f}}%<extra></extra>",
        ))
        # BJS markers
        fig.add_trace(go.Scatter(
            name=f"BJS — {g}", x=xs, y=bjs_v,
            mode="markers", legendgroup=f"bjs_{g}",
            marker=dict(symbol="diamond", size=10, color=colour,
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>BJS {g}</b><br>%{{y:.1f}}%<extra></extra>",
        ))

        # Δ labels above each bar, colour-coded by threshold
        for x_pos, av, bv in zip(xs, abm_m, bjs_v):
            diff = av - bv
            _add_gap_label(fig, x=x_pos, y=max(av, bv) + 4.5,
                           diff=diff, fontsize=8)

    fig.update_layout(**_LAYOUT,
        title=dict(
            text="<b>Chart 4 — Cumulative Rearrest at 3, 6 and 9 Years by Offense Type</b><br>"
                 "<sup>Bars = ABM  |  Diamond markers = BJS targets  |  "
                 "Error bars = 95% CI  |  Δ = ABM − BJS (pp)  |  "
                 "Alper et al. (2018), NCJ 250975 Table 7</sup>",
            font=dict(size=14), x=0.5),
        barmode="overlay",
        xaxis=dict(tickvals=x_base, ticktext=x_lbls,
                   showgrid=False, range=[0.4, 3.6]),
        yaxis=dict(title="Cumulative rearrest rate (%)",
                   range=[0, 110], gridcolor=_C_GRID, zeroline=False),
        legend=dict(x=1.01, y=0.99),
        margin=dict(t=110, b=100, l=70, r=180),
        height=560,
    )
    _gap_legend_annotation(fig, x=0.5, y=-0.10)
    _save(fig, os.path.join(out_dir, "chart4_cumulative_bar_by_offense.png"))

# =============================================================================
# CHART 5 — Non-cumulative annual first-arrest %  Years 1–9  [Δ labels ALL years]
# =============================================================================
def chart5_noncumulative(first_mean, first_lo, first_hi, out_dir):
    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=BJS_YEARS+BJS_YEARS[::-1],
        y=first_hi+first_lo[::-1],
        fill="toself", fillcolor="rgba(47,111,178,0.15)",
        line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
    ))

    # ABM bars
    fig.add_trace(go.Bar(
        name="ABM calibrated", x=BJS_YEARS, y=first_mean,
        marker_color=_C_ABM, marker_line_width=0, opacity=0.85,
        error_y=dict(type="data", symmetric=False,
                     array=[h-m for m,h in zip(first_mean,first_hi)],
                     arrayminus=[m-l for m,l in zip(first_mean,first_lo)],
                     color=_C_ABM, thickness=1.5, width=5),
        hovertemplate="<b>ABM Year %{x}</b><br>%{y:.1f}% first arrest<extra></extra>",
    ))

    # BJS line overlay
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=BJS_FIRST_ARREST,
        mode="lines+markers", name="BJS NCJ 250975",
        line=dict(color=_C_BJS, width=2.5, dash="dot"),
        marker=dict(symbol="diamond", size=8, color=_C_BJS),
        hovertemplate="<b>BJS Year %{x}</b><br>%{y:.1f}% first arrest<extra></extra>",
    ))

    # Δ labels for EVERY year (new — wasn't present before)
    _add_year_by_year_gaps(fig, BJS_YEARS, first_mean, BJS_FIRST_ARREST,
                           y_offset=2.5, fontsize=10)

    # Desistance annotation (positioned above the label band)
    y_ann = max(max(first_mean), max(BJS_FIRST_ARREST)) + 10
    fig.add_annotation(x=5.5, y=y_ann,
        text="Desistance: first-arrest rate declines after Year 1<br>"
             "(Sampson & Laub, 2003; Kurlychek et al., 2006)",
        showarrow=False,
        font=dict(size=10, color="#555555", family="Georgia, serif"),
        align="center",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    y_max = max(first_mean + BJS_FIRST_ARREST) + 16
    fig.update_layout(**_LAYOUT,
        title=dict(
            text="<b>Chart 5 — Annual First-Arrest Rate: Years 1–9 (Non-Cumulative)</b><br>"
                 "<sup>Percentage of released prisoners whose FIRST rearrest occurred in each year  |  "
                 "BJS NCJ 250975, Alper et al. (2018)  |  Δ = ABM − BJS (pp) at each year</sup>",
            font=dict(size=14), x=0.5),
        xaxis=dict(title="Year since release", tickvals=BJS_YEARS,
                   showgrid=False),
        yaxis=dict(title="First-arrest rate in year (%)",
                   range=[0, y_max],
                   gridcolor=_C_GRID, zeroline=False),
        legend=dict(x=0.65, y=0.97),
        margin=dict(t=110, b=90, l=70, r=40),
    )
    _gap_legend_annotation(fig)
    _save(fig, os.path.join(out_dir, "chart5_noncumulative_first_arrest.png"))

# =============================================================================
# PRINT SUMMARY
# =============================================================================
def print_summary(cum_mean, first_mean, off_mean, results):
    n = len(results)
    print(f"\n{'='*65}")
    print(f"  VALIDATION SUMMARY  ({n} simulation runs)")
    print(f"{'='*65}")
    print(f"  {'Year':>4}  {'ABM cum%':>9}  {'BJS cum%':>9}  {'Δpp':>6}  "
          f"{'ABM 1st%':>9}  {'BJS 1st%':>9}")
    print(f"  {'-'*60}")
    for i, yr in enumerate(BJS_YEARS):
        diff = cum_mean[i] - BJS_CUMULATIVE_ALL[i]
        flag = "✅" if abs(diff) <= GAP_GOOD else ("⚠️ " if abs(diff) <= GAP_WARN else "❌")
        print(f"  {yr:>4}  {cum_mean[i]:>8.1f}%  "
              f"{BJS_CUMULATIVE_ALL[i]:>8.1f}%  "
              f"{diff:>+5.1f}  {flag}  "
              f"{first_mean[i]:>8.1f}%  "
              f"{BJS_FIRST_ARREST[i]:>8.1f}%")
    print(f"\n  Offense-stratified at 3yr / 6yr / 9yr:")
    print(f"  {'Offense':>14}  {'ABM 3yr':>8}  {'BJS 3yr':>8}  "
          f"{'ABM 9yr':>8}  {'BJS 9yr':>8}")
    print(f"  {'-'*55}")
    for g in OFFENSE_GROUPS:
        print(f"  {g:>14}  {off_mean[g][2]:>7.1f}%  "
              f"{BJS_BY_OFFENSE[g][2]:>7.1f}%  "
              f"{off_mean[g][8]:>7.1f}%  "
              f"{BJS_BY_OFFENSE[g][8]:>7.1f}%")
    print(f"{'='*65}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    N_RUNS        = 10
    SEEDS_PER_RUN = 20
    BASE_SEED     = 1000
    OUT_DIR       = "validation_output"

    CAL = get_global_calibration_params()
    # CAL = get_uncalibrated_params()   # uncomment for baseline run

    tasks     = build_tasks(N_RUNS, SEEDS_PER_RUN, BASE_SEED, CAL)
    n_workers = detect_workers(len(tasks))

    print(f"  CPU logical: {os.cpu_count()} | workers: {n_workers}")
    print(f"  Tasks: {len(tasks)}  ({N_RUNS} runs × {SEEDS_PER_RUN} seeds)")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Run simulations ───────────────────────────────────────────────────────
    results = run_parallel(tasks, n_workers)

    # ── Save raw results ──────────────────────────────────────────────────────
    rows = []
    for r in results:
        row = {"run_id": r["run_id"], "seed": r["seed"], "n": r["n"]}
        for i, yr in enumerate(BJS_YEARS):
            row[f"cum_yr{yr}"]   = r["cum_all"][i]
            row[f"first_yr{yr}"] = r["first_all"][i]
            for g in OFFENSE_GROUPS:
                row[f"cum_{g.replace(' ','_')}_yr{yr}"] = r["cum_offense"][g][i]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "validation_raw.csv"), index=False)
    print(f"  Raw results -> {OUT_DIR}/validation_raw.csv")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    (cum_mean, cum_lo, cum_hi,
     first_mean, first_lo, first_hi,
     off_mean, off_lo, off_hi) = aggregate(results)

    print_summary(cum_mean, first_mean, off_mean, results)

    # ── Generate all 5 charts ─────────────────────────────────────────────────
    print("  Generating 5 validation charts...")
    chart1_cumulative_by_year(cum_mean, cum_lo, cum_hi, OUT_DIR)
    chart2_cumulative_bar(cum_mean, cum_lo, cum_hi, OUT_DIR)
    chart3_cumulative_by_offense(off_mean, off_lo, off_hi, OUT_DIR)
    chart4_bar_by_offense(off_mean, off_lo, off_hi, OUT_DIR)
    chart5_noncumulative(first_mean, first_lo, first_hi, OUT_DIR)
    print(f"  All charts saved to {OUT_DIR}/")