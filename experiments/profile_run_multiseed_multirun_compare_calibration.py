#!/usr/bin/env python3
"""
comparison_chart.py

Produces ONE chart: ABM Uncalibrated vs ABM Calibrated vs BJS Target
for 3-Year, 6-Year, 9-Year cumulative rearrest windows.

Usage:
    python comparison_chart.py

Outputs:
    calibration_comparison.png / .html
    results_calibrated.csv
    results_uncalibrated.csv
"""

import os, sys, time, math
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_global_calibration_params,
    get_peer_influence_config,
    get_flat_risk_weights,
)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS        = 10
SEEDS_PER_RUN = 20
MASTER_SEED   = 424242

# Smoke test — uncomment:
# N_RUNS = 3; SEEDS_PER_RUN = 5

BJS = {"3y": 0.68, "6y": 0.79, "9y": 0.83}

OUT_DIR = "comparison_output"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_PARAMS = {
    "initial_agents":        1000,
    "warmup_months":         144,
    "study_months":          108,
    "monthly_intake":        10,
    "bias_factor":           0,
    "enable_peer_influence": True,
}

# Post-calibration confirmed values
CALIBRATED = {
    "Supervision_Monitoring_Intensity":      0.35,
    "Risk_Contrast_Strength":                0.0,
    "Risk_Effect_Decay_After_3Y":            1.0,
    "Risk_Effect_Decay_After_6Y":            1.0,
    "Supervision_Monitoring_Decay_After_3Y": 1.0,
    "Supervision_Monitoring_Decay_After_6Y": 1.0,
}

# Pre-calibration defaults
UNCALIBRATED = {
    "Supervision_Monitoring_Intensity":      0,
    "Risk_Contrast_Strength":                0,
    "Risk_Effect_Decay_After_3Y":            0,
    "Risk_Effect_Decay_After_6Y":            0,
    "Supervision_Monitoring_Decay_After_3Y": 0,
    "Supervision_Monitoring_Decay_After_6Y": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Worker  (mirrors calibrate.py run_one_sim_job exactly)
# ─────────────────────────────────────────────────────────────────────────────
def run_one_sim_job(job: Dict[str, Any]) -> Dict[str, Any]:
    seed        = int(job["seed"])
    calibration = dict(job["calibration"])
    base_params = dict(job["base_params"])

    t0     = time.perf_counter()
    params = {**base_params, "seed": seed}

    model = RecidivismModel(**params)
    model.max_months  = model.warmup_months + model.study_months
    model.calibration = calibration                        # attach after construction

    if not hasattr(model, "np_rng"):
        model.np_rng = np.random.default_rng(seed)

    while model.running:
        model.step()

    model.current_month = model.warmup_months + model.study_months

    return {
        "label":       job["label"],
        "run_id":      job["run_id"],
        "seed":        seed,
        "rate_3y":     model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6y":     model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9y":     model.calculate_flag_rate("rearrest_9_yrs"),
        "runtime_sec": time.perf_counter() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_calibration(overrides: Dict) -> Dict:
    cal = get_global_calibration_params()
    cal.update(overrides)
    cal["peer_influence"] = get_peer_influence_config(enable_peer_influence=True)
    cal["risk_weights"]   = get_flat_risk_weights()
    return cal


def make_jobs(label: str, overrides: Dict, seeds: np.ndarray) -> List[Dict]:
    cal  = make_calibration(overrides)
    jobs = []
    for run_id in range(seeds.shape[0]):
        for j in range(seeds.shape[1]):
            jobs.append({
                "label":       label,
                "run_id":      run_id + 1,
                "seed":        int(seeds[run_id, j]),
                "calibration": cal,
                "base_params": BASE_PARAMS,
            })
    return jobs


def run_jobs(jobs: List[Dict], desc: str) -> pd.DataFrame:
    workers = max(1, min((os.cpu_count() or 4) // 2 - 1, len(jobs)))
    rows    = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(run_one_sim_job, jobs, chunksize=2),
                      total=len(jobs), desc=desc):
            rows.append(r)
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> Dict:
    n = len(df)
    out = {}
    for k in ["3y", "6y", "9y"]:
        vals = df[f"rate_{k}"]
        m    = float(vals.mean())
        ci   = 1.96 * float(vals.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
        out[k] = {"mean": m, "ci": ci}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────
def make_chart(cal: Dict, uncal: Dict, out_png: str = None):
    if out_png is None:
        out_png = os.path.join(OUT_DIR, "calibration_comparison.png")
    labels = ["3-Year", "6-Year", "9-Year"]
    keys   = ["3y", "6y", "9y"]

    fig = go.Figure()

    def bar(name, colour, means, cis):
        fig.add_trace(go.Bar(
            name=name, x=labels, y=means,
            marker_color=colour, marker_line_width=0,
            error_y=dict(type="data", array=cis, thickness=2, width=6, color=colour),
            text=[f"{m*100:.1f}%" for m in means],
            textposition="outside",
            textfont=dict(size=12, color=colour),
        ))

    bar("ABM Uncalibrated",       "#95A5A6",
        [uncal[k]["mean"] for k in keys],
        [uncal[k]["ci"]   for k in keys])

    bar("ABM Calibrated",         "#2F6FB2",
        [cal[k]["mean"]   for k in keys],
        [cal[k]["ci"]     for k in keys])

    bar("BJS Target (NCJ 252248)","#E07B39",
        [BJS[k]           for k in keys],
        [0, 0, 0])

    # MAE annotations
    y_top = max(uncal["9y"]["mean"], cal["9y"]["mean"], BJS["9y"])
    for lbl, k in zip(labels, keys):
        before = abs(uncal[k]["mean"] - BJS[k])
        after  = abs(cal[k]["mean"]   - BJS[k])
        delta  = before - after
        colour = "#27AE60" if delta >= 0 else "#C0392B"
        fig.add_annotation(
            x=lbl, y=y_top + 0.065,
            text=f"MAE {before*100:.1f}% → {after*100:.1f}%<br><b>↓{abs(delta)*100:.1f}pp</b>",
            showarrow=False,
            font=dict(size=10, color=colour, family="Georgia, serif"),
            align="center",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd", borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Georgia, serif", color="#1A2B3C", size=13),
        paper_bgcolor="#FAFBFC", plot_bgcolor="#FAFBFC",
        title=dict(
            text=(
                "<b>ABM Uncalibrated vs ABM Calibrated vs BJS Target</b><br>"
                f"<sup>{N_RUNS} runs × {SEEDS_PER_RUN} seeds per mode | "
                "Error bars = 95% CI | BJS NCJ 252248 (2018)</sup>"
            ),
            font=dict(size=16), x=0.5,
        ),
        barmode="group", bargap=0.22, bargroupgap=0.06,
        yaxis=dict(title="Cumulative Rearrest Rate", tickformat=".0%",
                   range=[0.55, 1.05], gridcolor="#E8EDF2", zeroline=False),
        xaxis=dict(title="Follow-up Window", showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ddd", borderwidth=1),
        margin=dict(t=120, b=70, l=80, r=40),
    )

    try:
        fig.write_image(out_png, scale=2)
        print(f"📁 PNG  → {out_png}")
    except Exception:
        print("⚠️  PNG needs kaleido: pip install -U kaleido")

    html = out_png.replace(".png", ".html")
    fig.write_html(html)
    print(f"📁 HTML → {html}")
    fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Same seeds for both modes — apples-to-apples comparison
    rng   = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(1, 2**31 - 1, size=(N_RUNS, SEEDS_PER_RUN), dtype=np.int64)

    print("── CALIBRATED ──────────────────────────────────────")
    df_cal = run_jobs(make_jobs("calibrated", CALIBRATED, seeds), "calibrated")
    df_cal.to_csv(os.path.join(OUT_DIR, "results_calibrated.csv"), index=False)
    cal_stats = stats(df_cal)

    print("── UNCALIBRATED ────────────────────────────────────")
    df_uncal = run_jobs(make_jobs("uncalibrated", UNCALIBRATED, seeds), "uncalibrated")
    df_uncal.to_csv(os.path.join(OUT_DIR, "results_uncalibrated.csv"), index=False)
    uncal_stats = stats(df_uncal)

    print("\n── RESULTS ─────────────────────────────────────────")
    for k, lbl in [("3y","3-Year"),("6y","6-Year"),("9y","9-Year")]:
        print(f"  {lbl}  uncal={uncal_stats[k]['mean']:.3f}  "
              f"cal={cal_stats[k]['mean']:.3f}  bjs={BJS[k]:.3f}")

    make_chart(cal_stats, uncal_stats)
    print("\n✅ Done.")