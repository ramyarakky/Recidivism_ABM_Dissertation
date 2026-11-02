#!/usr/bin/env python3
# sensitivity_oat.py  (optimized parallel + status + chunking + crash-safe saves)

import os, sys, time, math, json
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# Ensure project-root imports work
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_global_calibration_params,
    get_peer_influence_config,
    get_flat_risk_weights,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR = "Sensitivity_OAT_Weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_SEED = 424242

BASE_PARAMS = {
    "initial_agents": 1000,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0.0,
    "enable_peer_influence": True,
}

BJS_TARGETS = {"3y": 0.68, "6y": 0.79, "9y": 0.83}

N_RUNS  = 10
N_SEEDS = 20

# Use most cores but keep headroom on desktops; override by env if needed.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(min(DEFAULT_WORKERS, 24))))

# Chunking reduces overhead: jobs are grouped per worker pull.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8"))

# Periodic autosave to CSV every N completed jobs
AUTOSAVE_EVERY = int(os.getenv("AUTOSAVE_EVERY", "200"))

# ─────────────────────────────────────────────
# Parameter grids (OAT)
# ─────────────────────────────────────────────
'''
PARAM_GRIDS: Dict[str, List[float]] = {
    "National_Rearrest_Calibration_Factor": [0.95, 1.00, 1.08, 1.10, 1.15],
    "Risk_Contrast_Strength":              [1.0, 1.5, 2.0, 2.5, 3.0],
    "Supervision_Monitoring_Intensity":    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
    "Risk_Effect_Decay_After_3Y":          [0.50, 0.60, 0.70, 0.80, 0.90],
    "Risk_Effect_Decay_After_6Y":          [0.60, 0.70, 0.80, 0.90, 0.95],
    "Supervision_Monitoring_Decay_After_3Y":[0.50, 0.60, 0.70, 0.80, 0.90],
    "Supervision_Monitoring_Decay_After_6Y":[0.60, 0.70, 0.80, 0.90, 0.95],
}
'''

RISK_WEIGHT_BASELINES = {
    "Age_at_Release": 0.73,
    "Gang_Affiliated": 0.84,
    "Prior_Revocations_Supervision": 0.4338,
    "Condition_MH_SA": 0.35,
    "Program_Attendances": -0.38,
    "Program_UnexcusedAbsences": 0.11   
}

def oat_span_around(b: float) -> List[float]:
    """
    Default grid around baseline:
      - positives: 50%, 75%, 100%, 125%, 150% of baseline
      - negatives: same scaling on magnitude, preserving sign
    """
    mults = [0.5, 0.75, 1.0, 1.25, 1.5]
    if b >= 0:
        return [round(b*m, 6) for m in mults]
    else:
        return [round(-abs(b)*m, 6) for m in mults]

# Build sensitivity grids for nested risk weights
WEIGHT_GRIDS = {
    f"risk_weights.{k}": oat_span_around(v)
    for k, v in RISK_WEIGHT_BASELINES.items()
}

PARAM_GRIDS = {
    # Global level (odds multiplier): show under/over calibration clearly
    "National_Rearrest_Calibration_Factor": [0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50],

    # Risk separation: show weak/medium/strong heterogeneity
    "Risk_Contrast_Strength": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],

    # Monitoring: include “nearly none” through “very intensive”
    "Supervision_Monitoring_Intensity": [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60],

    # Risk decay: allow stronger damping and near-none
    "Risk_Effect_Decay_After_3Y": [0.30, 0.45, 0.60, 0.75, 0.90, 1.00],
    "Risk_Effect_Decay_After_6Y": [0.30, 0.45, 0.60, 0.75, 0.90, 1.00],

    # Monitoring decay: show strong fade vs persistent effect
    "Supervision_Monitoring_Decay_After_3Y": [0.30, 0.45, 0.60, 0.75, 0.90, 1.00],
    "Supervision_Monitoring_Decay_After_6Y": [0.30, 0.45, 0.60, 0.75, 0.90, 1.00],
    
    # ─── Individual risk factor weights (NEW) ───
    **WEIGHT_GRIDS,
}


# ─────────────────────────────────────────────
# Baseline calibration from risk_config.py
# ─────────────────────────────────────────────
BASE_CAL = get_global_calibration_params()
BASE_CAL["peer_influence"] = get_peer_influence_config(
    enable_peer_influence=BASE_PARAMS.get("enable_peer_influence", True)
)
BASE_CAL["risk_weights"] = get_flat_risk_weights()

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_seeds(master_seed: int, n_settings: int, n_runs: int, n_seeds: int) -> np.ndarray:
    rng = np.random.default_rng(master_seed)
    total = n_settings * n_runs * n_seeds
    seeds = rng.integers(1, 2**31 - 1, size=total, dtype=np.int64)
    return seeds.reshape(n_settings, n_runs, n_seeds)

def mean_ci(vals: np.ndarray) -> Tuple[float, float, float]:
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(np.mean(vals))
    if n == 1:
        return (m, m, m)
    s = float(np.std(vals, ddof=1))
    half = 1.96 * (s / math.sqrt(n))
    return (m, m - half, m + half)

def weighted_loss(mean_3y: float, mean_6y: float, mean_9y: float) -> float:
    w = {"3y": 2.0, "6y": 1.0, "9y": 1.0}
    loss = 0.0
    loss += w["3y"] * (mean_3y - BJS_TARGETS["3y"])**2
    loss += w["6y"] * (mean_6y - BJS_TARGETS["6y"])**2
    loss += w["9y"] * (mean_9y - BJS_TARGETS["9y"])**2
    return float(loss)

def set_nested(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set nested dict value using dot-path, e.g.:
      set_nested(cal, "risk_weights.Age_at_Release", 0.9)
    """
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

# ─────────────────────────────────────────────
# Worker (picklable)
# NOTE: keep payload small; do not send huge objects repeatedly.
# ─────────────────────────────────────────────
def run_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(job["seed"])
    run_id = int(job["run_id"])
    param_name = str(job["param_name"])
    param_value = float(job["param_value"])

    # deep copy so nested dicts aren't shared across jobs
    calibration = json.loads(json.dumps(job["calibration"]))
    base_params = dict(job["base_params"])

    t0 = time.perf_counter()
    try:
        params = dict(base_params)
        params["seed"] = seed

        # >>> IMPORTANT: pull weights + peer config from calibration <<<
        weights = calibration.get("risk_weights", None)
        peer_config = calibration.get("peer_influence", None)

        # construct model with swept weights
        model = RecidivismModel(**params, weights=weights, peer_config=peer_config)
        model.apply_calibration(calibration)
        model.max_months = model.warmup_months + model.study_months

        # keep this if you use other knobs in calibration elsewhere
        model.calibration = calibration

        if not hasattr(model, "np_rng"):
            model.np_rng = np.random.default_rng(seed)

        while model.running:
            model.step()

        model.current_month = model.warmup_months + model.study_months

        return {
            "param_name": param_name,
            "param_value": param_value,
            "run_id": run_id,
            "seed": seed,
            "rate_3y": model.calculate_flag_rate("rearrest_3_yrs"),
            "rate_6y": model.calculate_flag_rate("rearrest_6_yrs"),
            "rate_9y": model.calculate_flag_rate("rearrest_9_yrs"),
            "runtime_sec": round(time.perf_counter() - t0, 3),
            "status": "ok",
            "error": "",
        }

    except Exception as e:
        return {
            "param_name": param_name,
            "param_value": param_value,
            "run_id": run_id,
            "seed": seed,
            "rate_3y": float("nan"),
            "rate_6y": float("nan"),
            "rate_9y": float("nan"),
            "runtime_sec": round(time.perf_counter() - t0, 3),
            "status": "fail",
            "error": repr(e),
        }


'''
def build_oat_settings(base_cal: Dict[str, Any], grids: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    settings = []
    for pname, values in grids.items():
        for v in values:
            cal = dict(base_cal)
            cal[pname] = float(v)
            settings.append({"param_name": pname, "param_value": float(v), "calibration": cal})
    return settings
'''

def build_oat_settings(base_cal: Dict[str, Any], grids: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    settings = []
    for pname, values in grids.items():
        for v in values:
            cal = json.loads(json.dumps(base_cal))  # deep copy, safe for nested dicts
            if "." in pname:
                set_nested(cal, pname, float(v))
            else:
                cal[pname] = float(v)
            settings.append({"param_name": pname, "param_value": float(v), "calibration": cal})
    return settings


def estimate_eta(start_time: float, done: int, total: int) -> str:
    if done <= 0:
        return "ETA: --"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    return f"ETA: {int(remaining//60)}m {int(remaining%60)}s"

def save_status(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

# ─────────────────────────────────────────────
# Main runner (optimized)
# - uses executor.map with chunksize (less overhead)
# - autosaves partial results
# - status.json updates continuously
# ─────────────────────────────────────────────
def run_sensitivity_oat(
    base_params: Dict[str, Any],
    base_cal: Dict[str, Any],
    param_grids: Dict[str, List[float]],
    n_runs: int = 10,
    n_seeds: int = 10,
    max_workers: int = 8,
    chunksize: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    settings = build_oat_settings(base_cal, param_grids)
    seeds = generate_seeds(MASTER_SEED, n_settings=len(settings), n_runs=n_runs, n_seeds=n_seeds)

    jobs: List[Dict[str, Any]] = []
    for setting_id, s in enumerate(settings):
        for run_id in range(n_runs):
            for j in range(n_seeds):
                jobs.append({
                    "seed": int(seeds[setting_id, run_id, j]),
                    "run_id": run_id + 1,
                    "param_name": s["param_name"],
                    "param_value": s["param_value"],
                    "calibration": s["calibration"],
                    "base_params": base_params,
                })

    total = len(jobs)
    start = time.time()

    raw_csv = os.path.join(OUTPUT_DIR, "sensitivity_raw.csv")
    status_json = os.path.join(OUTPUT_DIR, "status.json")

    # Initialize status
    save_status(status_json, {
        "started_at": now_str(),
        "total_jobs": total,
        "completed": 0,
        "ok": 0,
        "fail": 0,
        "max_workers": max_workers,
        "chunksize": chunksize,
        "note": "updates as jobs finish; raw CSV autosaved periodically",
    })

    rows: List[Dict[str, Any]] = []
    ok_ct = 0
    fail_ct = 0

    print(f"▶ {now_str()} | Starting OAT sensitivity")
    print(f"   workers={max_workers}, chunksize={chunksize}, jobs={total}")
    print(f"   autosave_every={AUTOSAVE_EVERY} → {raw_csv}")
    print(f"   status → {status_json}")

    # Use map() with chunksize for better throughput than submit() per job
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(run_one_job, jobs, chunksize=chunksize)

        pbar = tqdm(total=total, desc="Running OAT sims", smoothing=0.05)
        completed = 0

        for out in it:
            rows.append(out)
            completed += 1
            pbar.update(1)

            if out.get("status") == "ok":
                ok_ct += 1
            else:
                fail_ct += 1

            # Update tqdm postfix with live stats
            if completed % 10 == 0 or completed == total:
                pbar.set_postfix_str(
                    f"ok={ok_ct} fail={fail_ct} | {estimate_eta(start, completed, total)}"
                )

            # Autosave + status updates
            if completed % AUTOSAVE_EVERY == 0 or completed == total:
                df_partial = pd.DataFrame(rows)
                df_partial.to_csv(raw_csv, index=False)

                save_status(status_json, {
                    "started_at": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
                    "last_update": now_str(),
                    "total_jobs": total,
                    "completed": completed,
                    "ok": ok_ct,
                    "fail": fail_ct,
                    "max_workers": max_workers,
                    "chunksize": chunksize,
                    "elapsed_sec": round(time.time() - start, 2),
                    "eta": estimate_eta(start, completed, total),
                    "raw_csv": raw_csv,
                })

        pbar.close()

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(raw_csv, index=False)

    # Summaries
    ok = df_raw[df_raw["status"] == "ok"].copy()

    summary_rows = []
    for (pname, pval), sub in ok.groupby(["param_name", "param_value"], sort=True):
        v3 = sub["rate_3y"].astype(float).to_numpy()
        v6 = sub["rate_6y"].astype(float).to_numpy()
        v9 = sub["rate_9y"].astype(float).to_numpy()

        m3, lo3, hi3 = mean_ci(v3)
        m6, lo6, hi6 = mean_ci(v6)
        m9, lo9, hi9 = mean_ci(v9)

        summary_rows.append({
            "param_name": pname,
            "param_value": float(pval),
            "n_sims": int(len(sub)),
            "mean_3y": m3, "ci_low_3y": lo3, "ci_high_3y": hi3,
            "mean_6y": m6, "ci_low_6y": lo6, "ci_high_6y": hi6,
            "mean_9y": m9, "ci_low_9y": lo9, "ci_high_9y": hi9,
            "loss": weighted_loss(m3, m6, m9),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(["param_name", "param_value"]).reset_index(drop=True)

    eff_rows = []
    for pname, sub in df_summary.groupby("param_name", sort=True):
        eff_rows.append({
            "param_name": pname,
            "span_3y": float(sub["mean_3y"].max() - sub["mean_3y"].min()),
            "span_6y": float(sub["mean_6y"].max() - sub["mean_6y"].min()),
            "span_9y": float(sub["mean_9y"].max() - sub["mean_9y"].min()),
        })
    df_effect = pd.DataFrame(eff_rows).sort_values("span_3y", ascending=False).reset_index(drop=True)

    summary_csv = os.path.join(OUTPUT_DIR, "sensitivity_summary.csv")
    effect_csv = os.path.join(OUTPUT_DIR, "sensitivity_effect_span.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_effect.to_csv(effect_csv, index=False)

    # Final status
    save_status(status_json, {
        "started_at": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": now_str(),
        "total_jobs": total,
        "completed": total,
        "ok": int(ok_ct),
        "fail": int(fail_ct),
        "max_workers": max_workers,
        "chunksize": chunksize,
        "elapsed_sec": round(time.time() - start, 2),
        "raw_csv": raw_csv,
        "summary_csv": summary_csv,
        "effect_csv": effect_csv,
    })

    print("\n✅ Saved:")
    print(" -", raw_csv)
    print(" -", summary_csv)
    print(" -", effect_csv)
    print(" -", status_json)

    return df_raw, df_summary, df_effect

# ─────────────────────────────────────────────
# Optional Plotly helper
# ─────────────────────────────────────────────

def split_params(df_summary: pd.DataFrame):
    df_weights = df_summary[df_summary["param_name"].str.startswith("risk_weights.")]
    df_calib   = df_summary[~df_summary["param_name"].str.startswith("risk_weights.")]
    return df_calib, df_weights

def save_png(fig, path, width=1200, height=700, scale=2):
    fig.write_image(
        path,
        width=width,
        height=height,
        scale=scale,
        engine="kaleido"
    )

def plot_oat_curves(df_summary: pd.DataFrame):
    import plotly.graph_objects as go

    for pname, sub in df_summary.groupby("param_name", sort=True):
        sub = sub.sort_values("param_value")
        x = sub["param_value"].to_numpy()

        fig = go.Figure()
        for k, label in [("3y", "3-Year"), ("6y", "6-Year"), ("9y", "9-Year")]:
            fig.add_trace(go.Scatter(
                x=x, y=sub[f"mean_{k}"],
                mode="lines+markers",
                name=f"ABM {label}"
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([sub[f"ci_high_{k}"], sub[f"ci_low_{k}"][::-1]]),
                fill="toself",
                opacity=0.15,
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=x, y=[BJS_TARGETS[k]] * len(x),
                mode="lines",
                line=dict(dash="dash", width=1),
                name=f"BJS {label}"
            ))

        fig.update_layout(
            title=f"OAT Sensitivity: {pname}",
            xaxis_title=pname,
            yaxis_title="Rearrest rate",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        )

        out_png = os.path.join(OUTPUT_DIR, f"oat_{pname}.png")
        save_png(fig, out_png)
        print("📈 Saved PNG:", out_png)

def plot_oat_delta_from_targets(df_summary: pd.DataFrame):
    import plotly.graph_objects as go

    df_calib, df_weights = split_params(df_summary)

    # ─────────────────────────────────────────
    # 1) CALIBRATION PARAM PLOTS
    # ─────────────────────────────────────────
    for pname, sub in df_calib.groupby("param_name", sort=True):
        sub = sub.sort_values("param_value")
        x = sub["param_value"].to_numpy()

        fig = go.Figure()
        for k, label in [("3y", "3-Year"), ("6y", "6-Year"), ("9y", "9-Year")]:
            delta = (sub[f"mean_{k}"] - BJS_TARGETS[k]).to_numpy()
            fig.add_trace(go.Scatter(
                x=x, y=delta,
                mode="lines+markers",
                name=f"Δ vs BJS {label}"
            ))

        fig.add_hline(y=0, line_dash="dash")

        fig.update_layout(
            title=f"Calibration Sensitivity: {pname}",
            xaxis_title=pname,
            yaxis_title="ABM − BJS",
            template="plotly_white",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        )

        out_png = os.path.join(OUTPUT_DIR, f"CALIB_delta_{pname}.png")
        save_png(fig, out_png)
        print("📉 Saved CALIB PNG:", out_png)

    # ─────────────────────────────────────────
    # 2) RISK WEIGHT PLOTS
    # ─────────────────────────────────────────
    for pname, sub in df_weights.groupby("param_name", sort=True):
        sub = sub.sort_values("param_value")
        x = sub["param_value"].to_numpy()

        fig = go.Figure()
        for k, label in [("3y", "3-Year"), ("6y", "6-Year"), ("9y", "9-Year")]:
            delta = (sub[f"mean_{k}"] - BJS_TARGETS[k]).to_numpy()
            fig.add_trace(go.Scatter(
                x=x, y=delta,
                mode="lines+markers",
                name=f"Δ vs BJS {label}"
            ))

        fig.add_hline(y=0, line_dash="dash")

        # prettier titles for dissertation
        clean_name = pname.replace("risk_weights.", "")

        fig.update_layout(
            title=f"Risk Factor Sensitivity: {clean_name}",
            xaxis_title=f"Weight: {clean_name}",
            yaxis_title="ABM − BJS",
            template="plotly_white",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        )

        out_png = os.path.join(OUTPUT_DIR, f"WEIGHT_delta_{clean_name}.png")
        save_png(fig, out_png)
        print("📉 Saved WEIGHT PNG:", out_png)


def plot_oat_loss(df_summary: pd.DataFrame):
    import plotly.graph_objects as go

    for pname, sub in df_summary.groupby("param_name", sort=True):
        sub = sub.sort_values("param_value")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub["param_value"],
            y=sub["loss"],
            mode="lines+markers",
            name="Weighted loss"
        ))

        fig.update_layout(
            title=f"OAT Fit: {pname}",
            xaxis_title=pname,
            yaxis_title="Weighted loss (↓ better)",
            template="plotly_white",
        )

        out_png = os.path.join(OUTPUT_DIR, f"oat_loss_{pname}.png")
        save_png(fig, out_png)
        print("📉 Saved PNG:", out_png)

def plot_tornado_effects(df_effect: pd.DataFrame):
    import plotly.graph_objects as go

    df = df_effect.sort_values("span_3y", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["param_name"], x=df["span_3y"],
        orientation="h", name="3-Year"
    ))
    fig.add_trace(go.Bar(
        y=df["param_name"], x=df["span_6y"],
        orientation="h", name="6-Year"
    ))
    fig.add_trace(go.Bar(
        y=df["param_name"], x=df["span_9y"],
        orientation="h", name="9-Year"
    ))

    fig.update_layout(
        barmode="group",
        title="OAT Sensitivity: Effect Span by Parameter",
        xaxis_title="Span of mean rearrest rate",
        yaxis_title="Parameter",
        template="plotly_white",
        margin=dict(l=200),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )

    out_png = os.path.join(OUTPUT_DIR, "oat_tornado_effects.png")
    save_png(fig, out_png)
    print("🌪️ Saved PNG:", out_png)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("▶ Running OAT sensitivity analysis")
    print(f"   N_RUNS={N_RUNS}, N_SEEDS={N_SEEDS}")
    print(f"   MAX_WORKERS={MAX_WORKERS} (override with env MAX_WORKERS)")
    print(f"   CHUNK_SIZE={CHUNK_SIZE} (override with env CHUNK_SIZE)")
    print(f"   Output dir: {OUTPUT_DIR}")

    df_raw, df_summary, df_effect = run_sensitivity_oat(
        base_params=BASE_PARAMS,
        base_cal=BASE_CAL,
        param_grids=PARAM_GRIDS,
        n_runs=N_RUNS,
        n_seeds=N_SEEDS,
        max_workers=MAX_WORKERS,
        chunksize=CHUNK_SIZE,
    )

    plot_oat_curves(df_summary)
    plot_oat_delta_from_targets(df_summary)
    plot_oat_loss(df_summary)
    plot_tornado_effects(df_effect)

    print("\n✅ Done.")
