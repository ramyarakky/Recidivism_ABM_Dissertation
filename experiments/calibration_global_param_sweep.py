#!/usr/bin/env python3
"""
Standalone script — produces ONE chart:
  ABM Uncalibrated  vs  ABM Calibrated  vs  BJS Target
  for 3-Year, 6-Year, 9-Year cumulative rearrest windows.

Usage:
  python plot_calibration_comparison.py

Outputs:
  chart6_calibration_comparison.png
"""

import os
import sys
import time
import math
import random
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_global_calibration_params,
    get_peer_influence_config,
    get_flat_risk_weights,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS        = 10
SEEDS_PER_RUN = 20
BASE_SEED     = 1000

# Smoke test — uncomment for a quick check (3×5 = 15 sims per mode):
# N_RUNS = 3; SEEDS_PER_RUN = 5

BJS_TARGETS = {"3-Year": 0.68, "6-Year": 0.79, "9-Year": 0.83}

BASE_PARAMS = {
    "initial_agents":        1000,
    "warmup_months":         144,
    "study_months":          108,
    "monthly_intake":        10,
    "bias_factor":           0,
    "enable_peer_influence": True,
}

# Overrides passed into get_global_calibration_params()
CALIBRATED_OVERRIDES = {
    "Supervision_Monitoring_Intensity":      0.35,  # IDENTIFIED
    "Risk_Contrast_Strength":                0.0,   # UNIDENTIFIED — boundary optimum
    "Risk_Effect_Decay_After_3Y":            1.0,   # UNIDENTIFIED — flat surface
    "Risk_Effect_Decay_After_6Y":            1.0,   # UNIDENTIFIED — flat surface
    "Supervision_Monitoring_Decay_After_3Y": 1.0,   # UNIDENTIFIED — flat surface
    "Supervision_Monitoring_Decay_After_6Y": 1.0,   # UNIDENTIFIED — flat surface
}

UNCALIBRATED_OVERRIDES = {
    "Supervision_Monitoring_Intensity":      0,  # same — already identified
    "Risk_Contrast_Strength":                0,   # pre-calibration default
    "Risk_Effect_Decay_After_3Y":            0,   # pre-calibration default
    "Risk_Effect_Decay_After_6Y":            0,   # pre-calibration default
    "Supervision_Monitoring_Decay_After_3Y": 0,  # pre-calibration default
    "Supervision_Monitoring_Decay_After_6Y": 0,  # pre-calibration default
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
_T95 = {
    1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365,
    8:2.306, 9:2.262, 10:2.228, 15:2.131, 20:2.086, 30:2.042,
    40:2.021, 60:2.000, 80:1.990, 120:1.980,
}

def _t95(df: int) -> float:
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96

def mean_ci95(values: List[float]) -> Tuple[float, float, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0: return (float("nan"),) * 3
    if n == 1: return (vals[0], vals[0], vals[0])
    m   = sum(vals) / n
    sd  = math.sqrt(sum((x - m)**2 for x in vals) / (n - 1))
    h   = _t95(n - 1) * sd / math.sqrt(n)
    return (m, m - h, m + h)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation worker
# ─────────────────────────────────────────────────────────────────────────────
def _run_one(task: Dict) -> Dict:
    seed     = int(task["seed"])
    run_id   = int(task["run_id"])
    cal_mode = task["calibration_mode"]

    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass

    t0 = time.perf_counter()

    # Build calibration dict via risk_config, then apply overrides
    overrides = CALIBRATED_OVERRIDES if cal_mode == "calibrated" else UNCALIBRATED_OVERRIDES
    calibration = get_global_calibration_params()
    calibration.update(overrides)
    calibration["peer_influence"] = get_peer_influence_config(enable_peer_influence=True)
    calibration["risk_weights"]   = get_flat_risk_weights()

    # Construct model with base params only — no calibration knobs in constructor
    params = {**BASE_PARAMS, "seed": seed}
    model  = RecidivismModel(**params)

    # Attach calibration AFTER construction — same pattern as calibrate.py
    model.calibration = calibration

    if not hasattr(model, "np_rng"):
        import numpy as np
        model.np_rng = np.random.default_rng(seed)

    model.max_months = model.warmup_months + model.study_months
    while model.running:
        model.step()
    model.current_month = model.warmup_months + model.study_months

    return {
        "run_id":           run_id,
        "seed":             seed,
        "calibration_mode": cal_mode,
        "rate_3yr":         model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6yr":         model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9yr":         model.calculate_flag_rate("rearrest_9_yrs"),
        "runtime_sec":      time.perf_counter() - t0,
    }


def _build_tasks(cal_mode: str) -> List[Dict]:
    tasks = []
    for run_id in range(1, N_RUNS + 1):
        for i in range(SEEDS_PER_RUN):
            tasks.append({
                "run_id":           run_id,
                "seed":             BASE_SEED + run_id * 10_000 + i,
                "calibration_mode": cal_mode,
            })
    return tasks


def _run_parallel(tasks: List[Dict]) -> pd.DataFrame:
    logical  = os.cpu_count() or 4
    workers  = max(1, min(len(tasks), max(1, logical // 2) - 2))
    mode     = tasks[0]["calibration_mode"].upper()
    print(f"\n🚀 [{mode}] Running {len(tasks)} sims on {workers} workers …")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                rows.append(r)
                print(
                    f"  ✅ [{mode}] Run {r['run_id']:02d} | Seed {r['seed']} | "
                    f"3y={r['rate_3yr']:.3f} 6y={r['rate_6yr']:.3f} "
                    f"9y={r['rate_9yr']:.3f} | {r['runtime_sec']:.1f}s"
                )
            except Exception as e:
                print(f"  ❌ run={t['run_id']} seed={t['seed']}: {e}")
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame) -> Dict:
    out = {}
    for col, label in [("rate_3yr","3-Year"),("rate_6yr","6-Year"),("rate_9yr","9-Year")]:
        m, lo, hi = mean_ci95(df[col].tolist())
        out[label] = {"mean": m, "ci_low": lo, "ci_high": hi, "n": int(df[col].notna().sum())}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(
    cal_stats:   Dict,
    uncal_stats: Dict,
    out_png:     str = "chart6_calibration_comparison.png",
) -> None:
    windows     = ["3-Year", "6-Year", "9-Year"]
    uncal_means = [uncal_stats[w]["mean"]   for w in windows]
    cal_means   = [cal_stats[w]["mean"]     for w in windows]
    cal_lows    = [cal_stats[w]["ci_low"]   for w in windows]
    cal_highs   = [cal_stats[w]["ci_high"]  for w in windows]
    bjs         = [BJS_TARGETS[w]           for w in windows]

    C_UNCAL = "#95A5A6"   # grey
    C_CAL   = "#2F6FB2"   # blue
    C_BJS   = "#E07B39"   # orange
    C_GRID  = "#E8EDF2"

    fig = go.Figure()

    # ── Uncalibrated ──────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        name="ABM Uncalibrated",
        x=windows, y=uncal_means,
        marker_color=C_UNCAL, marker_line_width=0,
        text=[f"{m*100:.1f}%" for m in uncal_means],
        textposition="outside",
        textfont=dict(size=13, color=C_UNCAL),
        hovertemplate="<b>%{x}</b><br>Uncalibrated ABM: %{y:.3f}<extra></extra>",
    ))

    # ── Calibrated ± 95% CI ───────────────────────────────────────────────
    fig.add_trace(go.Bar(
        name="ABM Calibrated",
        x=windows, y=cal_means,
        marker_color=C_CAL, marker_line_width=0,
        error_y=dict(
            type="data", symmetric=False,
            array     =[h - m for m, h in zip(cal_means, cal_highs)],
            arrayminus=[m - l for m, l in zip(cal_means, cal_lows)],
            color="#1A4A80", thickness=2.5, width=8,
        ),
        text=[f"{m*100:.1f}%" for m in cal_means],
        textposition="outside",
        textfont=dict(size=13, color=C_CAL),
        hovertemplate=(
            "<b>%{x}</b><br>Calibrated ABM: %{y:.3f}<br>"
            "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
            "<extra></extra>"
        ),
        customdata=[[lo, hi] for lo, hi in zip(cal_lows, cal_highs)],
    ))

    # ── BJS Target ────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        name="BJS Target (NCJ 252248)",
        x=windows, y=bjs,
        marker_color=C_BJS, marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in bjs],
        textposition="outside",
        textfont=dict(size=13, color=C_BJS),
        hovertemplate="<b>%{x}</b><br>BJS Target: %{y:.3f}<extra></extra>",
    ))

    # ── MAE annotations ───────────────────────────────────────────────────
    y_top = max(max(uncal_means), max(cal_means), max(bjs))
    for w, um, cm, b in zip(windows, uncal_means, cal_means, bjs):
        mae_before  = abs(um - b)
        mae_after   = abs(cm - b)
        improvement = mae_before - mae_after
        c = "#27AE60" if improvement >= 0 else "#C0392B"
        fig.add_annotation(
            x=w, y=y_top + 0.07,
            text=(
                f"MAE: <b>{mae_before*100:.1f}%</b> → <b>{mae_after*100:.1f}%</b><br>"
                f"<span style='color:{c}'>↓ {abs(improvement)*100:.1f}pp</span>"
            ),
            showarrow=False,
            font=dict(size=11, family="Georgia, serif", color="#1A2B3C"),
            align="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Georgia, serif", color="#1A2B3C", size=13),
        paper_bgcolor="#FAFBFC",
        plot_bgcolor="#FAFBFC",
        title=dict(
            text=(
                "<b>ABM Uncalibrated vs ABM Calibrated vs BJS Target</b><br>"
                f"<sup>{N_RUNS} runs × {SEEDS_PER_RUN} seeds = "
                f"{N_RUNS*SEEDS_PER_RUN} simulations per mode | "
                "Calibrated error bars = 95% CI | BJS NCJ 252248 (2018)</sup>"
            ),
            font=dict(size=16),
            x=0.5,
        ),
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        yaxis=dict(
            title="Cumulative Rearrest Rate",
            tickformat=".0%",
            range=[0.55, 1.02],
            gridcolor=C_GRID,
            zeroline=False,
        ),
        xaxis=dict(
            title="Follow-up Window",
            showgrid=False,
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        margin=dict(t=120, b=70, l=80, r=40),
    )

    try:
        fig.write_image(out_png, scale=2)
        print(f"\n📁 Chart saved → {out_png}")
    except Exception:
        print("⚠️  Install kaleido for PNG export: pip install -U kaleido")
    fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Calibrated run ───────────────────────────────────────────────────
    df_cal      = _run_parallel(_build_tasks("calibrated"))
    cal_stats   = _summarise(df_cal)
    df_cal.to_csv("results_calibrated.csv", index=False)

    print("\n📊 CALIBRATED results:")
    for w in ["3-Year","6-Year","9-Year"]:
        s = cal_stats[w]
        print(f"  {w}: {s['mean']:.3f}  CI=[{s['ci_low']:.3f},{s['ci_high']:.3f}]  n={s['n']}")

    # ── Uncalibrated run ─────────────────────────────────────────────────
    df_uncal    = _run_parallel(_build_tasks("uncalibrated"))
    uncal_stats = _summarise(df_uncal)
    df_uncal.to_csv("results_uncalibrated.csv", index=False)

    print("\n📊 UNCALIBRATED results:")
    for w in ["3-Year","6-Year","9-Year"]:
        s = uncal_stats[w]
        print(f"  {w}: {s['mean']:.3f}  CI=[{s['ci_low']:.3f},{s['ci_high']:.3f}]  n={s['n']}")

    # ── Chart ─────────────────────────────────────────────────────────────
    plot_comparison(cal_stats, uncal_stats)

    print("\n✅ Done.")