#!/usr/bin/env python3
# stress_test_global_params.py
# Extreme-scenario ("stress test") validation for GLOBAL parameters in Recidivism ABM
# - matches your RecidivismModel usage pattern from sensitivity_oat.py
# - saves charts as PNG

import os, sys, time, math, json
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
OUTPUT_DIR = "StressTest_GlobalParams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_SEED = 424242

BASE_PARAMS = {
    "initial_agents": 1000,
    "warmup_months": 144,
    "study_months": 108,   # 9 years
    "monthly_intake": 10,
    "bias_factor": 0.0,
    "enable_peer_influence": True,
}

BJS_TARGETS = {"3y": 0.68, "6y": 0.79, "9y": 0.83}

# Choose stability vs runtime (start here; bump later if needed)
N_RUNS  = 5
N_SEEDS = 20

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(min(DEFAULT_WORKERS, 24))))
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "8"))

# ─────────────────────────────────────────────
# Baseline calibration dict (same pattern as your OAT script)
# ─────────────────────────────────────────────
BASE_CAL = get_global_calibration_params()
BASE_CAL["peer_influence"] = get_peer_influence_config(
    enable_peer_influence=BASE_PARAMS.get("enable_peer_influence", True)
)
BASE_CAL["risk_weights"] = get_flat_risk_weights()

# If you want, overwrite BASE_CAL here with your calibrated values:
# BASE_CAL.update({
#     "Risk_Contrast_Strength": 1.25,
#     "Supervision_Monitoring_Intensity": 0.40,
#     "Risk_Effect_Decay_After_3Y": 0.30,
#     "Risk_Effect_Decay_After_6Y": 0.15,
#     "Supervision_Monitoring_Decay_After_3Y": 1.00,
#     "Supervision_Monitoring_Decay_After_6Y": 0.85,
# })

# ─────────────────────────────────────────────
# ~10 extreme scenarios (plus baseline)
# Each scenario only overrides GLOBAL calibration keys (not BASE_PARAMS)
# ─────────────────────────────────────────────
STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "BASELINE": {},

    # Supervision strength extremes
    "NO_supervision_effect": {"Supervision_Monitoring_Intensity": 0.0},
    "VERY_strong_supervision": {"Supervision_Monitoring_Intensity": 0.8},

    # Risk heterogeneity extremes
    "NO_risk_heterogeneity": {"Risk_Contrast_Strength": 0.0},
    "VERY_strong_risk_heterogeneity": {"Risk_Contrast_Strength": 2.5},

    # Risk decay (desistance) extremes
    "NO_risk_decay": {"Risk_Effect_Decay_After_3Y": 1.0, "Risk_Effect_Decay_After_6Y": 1.0},
    "MAX_risk_decay": {"Risk_Effect_Decay_After_3Y": 0.0, "Risk_Effect_Decay_After_6Y": 0.0},

    # Supervision persistence extremes
    "NO_supervision_persistence": {"Supervision_Monitoring_Decay_After_3Y": 0.0, "Supervision_Monitoring_Decay_After_6Y": 0.0},
    "FULL_supervision_persistence": {"Supervision_Monitoring_Decay_After_3Y": 1.0, "Supervision_Monitoring_Decay_After_6Y": 1.0},

    # Combined “worst/best” stress cases
    "WORST_case_high_risk_no_supervision": {
        "Risk_Contrast_Strength": 2.5,
        "Supervision_Monitoring_Intensity": 0.0,
        "Risk_Effect_Decay_After_3Y": 1.0,
        "Risk_Effect_Decay_After_6Y": 1.0,
        "Supervision_Monitoring_Decay_After_3Y": 0.0,
        "Supervision_Monitoring_Decay_After_6Y": 0.0,
    },
    "BEST_case_low_risk_strong_supervision_desistance": {
        "Risk_Contrast_Strength": 0.5,
        "Supervision_Monitoring_Intensity": 0.8,
        "Risk_Effect_Decay_After_3Y": 0.0,
        "Risk_Effect_Decay_After_6Y": 0.0,
        "Supervision_Monitoring_Decay_After_3Y": 1.0,
        "Supervision_Monitoring_Decay_After_6Y": 1.0,
    },
}

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

def estimate_eta(start_time: float, done: int, total: int) -> str:
    if done <= 0:
        return "ETA: --"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    return f"ETA: {int(remaining//60)}m {int(remaining%60)}s"

# ─────────────────────────────────────────────
# Worker (picklable)
# Mirrors your run_one_job pattern
# ─────────────────────────────────────────────
def run_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(job["seed"])
    run_id = int(job["run_id"])
    scenario = str(job["scenario"])

    calibration = dict(job["calibration"])
    base_params = dict(job["base_params"])

    t0 = time.perf_counter()
    try:
        params = dict(base_params)
        params["seed"] = seed

        model = RecidivismModel(**params)
        model.max_months = model.warmup_months + model.study_months

        model.calibration = calibration

        if not hasattr(model, "np_rng"):
            model.np_rng = np.random.default_rng(seed)

        while model.running:
            model.step()

        model.current_month = model.warmup_months + model.study_months

        return {
            "scenario": scenario,
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
            "scenario": scenario,
            "run_id": run_id,
            "seed": seed,
            "rate_3y": float("nan"),
            "rate_6y": float("nan"),
            "rate_9y": float("nan"),
            "runtime_sec": round(time.perf_counter() - t0, 3),
            "status": "fail",
            "error": repr(e),
        }

# ─────────────────────────────────────────────
# Build stress settings (baseline + extremes)
# ─────────────────────────────────────────────
def build_stress_settings(base_cal: Dict[str, Any], scenarios: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    settings = []
    for sname, overrides in scenarios.items():
        cal = dict(base_cal)
        for k, v in overrides.items():
            cal[k] = float(v)
        settings.append({"scenario": sname, "calibration": cal})
    return settings

# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────
def run_stress_tests(
    base_params: Dict[str, Any],
    base_cal: Dict[str, Any],
    scenarios: Dict[str, Dict[str, float]],
    n_runs: int,
    n_seeds: int,
    max_workers: int,
    chunksize: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    settings = build_stress_settings(base_cal, scenarios)
    seeds = generate_seeds(MASTER_SEED, n_settings=len(settings), n_runs=n_runs, n_seeds=n_seeds)

    jobs: List[Dict[str, Any]] = []
    for setting_id, s in enumerate(settings):
        for run_id in range(n_runs):
            for j in range(n_seeds):
                jobs.append({
                    "seed": int(seeds[setting_id, run_id, j]),
                    "run_id": run_id + 1,
                    "scenario": s["scenario"],
                    "calibration": s["calibration"],
                    "base_params": base_params,
                })

    total = len(jobs)
    start = time.time()

    raw_csv = os.path.join(OUTPUT_DIR, "stress_raw.csv")
    status_json = os.path.join(OUTPUT_DIR, "status.json")

    # Status init
    with open(status_json, "w", encoding="utf-8") as f:
        json.dump({
            "started_at": now_str(),
            "total_jobs": total,
            "completed": 0,
            "ok": 0,
            "fail": 0,
            "max_workers": max_workers,
            "chunksize": chunksize,
        }, f, indent=2)

    rows: List[Dict[str, Any]] = []
    ok_ct, fail_ct = 0, 0

    print(f"▶ {now_str()} | Starting STRESS tests")
    print(f"   scenarios={len(settings)}, runs={n_runs}, seeds={n_seeds}, jobs={total}")
    print(f"   workers={max_workers}, chunksize={chunksize}")
    print(f"   output={OUTPUT_DIR}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(run_one_job, jobs, chunksize=chunksize)
        pbar = tqdm(total=total, desc="Running stress sims", smoothing=0.05)

        completed = 0
        for out in it:
            rows.append(out)
            completed += 1
            pbar.update(1)

            if out.get("status") == "ok":
                ok_ct += 1
            else:
                fail_ct += 1

            if completed % 50 == 0 or completed == total:
                pbar.set_postfix_str(f"ok={ok_ct} fail={fail_ct} | {estimate_eta(start, completed, total)}")

            # Periodic autosave
            if completed % 200 == 0 or completed == total:
                pd.DataFrame(rows).to_csv(raw_csv, index=False)
                with open(status_json, "w", encoding="utf-8") as f:
                    json.dump({
                        "started_at": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
                        "last_update": now_str(),
                        "total_jobs": total,
                        "completed": completed,
                        "ok": ok_ct,
                        "fail": fail_ct,
                        "elapsed_sec": round(time.time() - start, 2),
                        "eta": estimate_eta(start, completed, total),
                        "raw_csv": raw_csv,
                    }, f, indent=2)

        pbar.close()

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(raw_csv, index=False)

    ok = df_raw[df_raw["status"] == "ok"].copy()

    # Summary per scenario
    summary_rows = []
    for sname, sub in ok.groupby(["scenario"], sort=True):
        v3 = sub["rate_3y"].astype(float).to_numpy()
        v6 = sub["rate_6y"].astype(float).to_numpy()
        v9 = sub["rate_9y"].astype(float).to_numpy()

        m3, lo3, hi3 = mean_ci(v3)
        m6, lo6, hi6 = mean_ci(v6)
        m9, lo9, hi9 = mean_ci(v9)

        summary_rows.append({
            "scenario": sname,
            "n_sims": int(len(sub)),
            "mean_3y": m3, "ci_low_3y": lo3, "ci_high_3y": hi3,
            "mean_6y": m6, "ci_low_6y": lo6, "ci_high_6y": hi6,
            "mean_9y": m9, "ci_low_9y": lo9, "ci_high_9y": hi9,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("mean_9y", ascending=False).reset_index(drop=True)

    summary_csv = os.path.join(OUTPUT_DIR, "stress_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    print("\n✅ Saved:")
    print(" -", raw_csv)
    print(" -", summary_csv)
    print(" -", status_json)

    return df_raw, df_summary

# ─────────────────────────────────────────────
# Charts (matplotlib only; no explicit colors)
# ─────────────────────────────────────────────
def plot_bars(df_summary: pd.DataFrame, horizon: str):
    mean_col = f"mean_{horizon}"
    lo_col   = f"ci_low_{horizon}"
    hi_col   = f"ci_high_{horizon}"

    s = df_summary.sort_values(mean_col, ascending=False).copy()
    s["err_low"]  = s[mean_col] - s[lo_col]
    s["err_high"] = s[hi_col] - s[mean_col]

    x = np.arange(len(s))
    y = s[mean_col].values
    yerr = np.vstack([s["err_low"].values, s["err_high"].values])

    plt.figure()
    plt.bar(x, y, yerr=yerr, capsize=3)
    plt.xticks(x, s["scenario"].values, rotation=60, ha="right")
    plt.ylabel(f"Rearrest rate ({horizon})")
    plt.title(f"Stress test: cumulative rearrest at {horizon} (mean ± 95% CI)")
    plt.ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"stress_bar_{horizon}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("📊 Saved:", out)

def plot_delta_from_bjs(df_summary: pd.DataFrame):
    s = df_summary.copy()
    s["d3"] = s["mean_3y"] - BJS_TARGETS["3y"]
    s["d6"] = s["mean_6y"] - BJS_TARGETS["6y"]
    s["d9"] = s["mean_9y"] - BJS_TARGETS["9y"]

    s = s.sort_values("d9", ascending=False)

    x = np.arange(len(s))

    plt.figure()
    plt.bar(x - 0.25, s["d3"].values, width=0.25, label="Δ 3y")
    plt.bar(x,        s["d6"].values, width=0.25, label="Δ 6y")
    plt.bar(x + 0.25, s["d9"].values, width=0.25, label="Δ 9y")

    plt.axhline(0, linewidth=1)
    plt.xticks(x, s["scenario"].values, rotation=60, ha="right")
    plt.ylabel("ABM − BJS")
    plt.title("Stress test: deviation from BJS targets")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "stress_delta_vs_bjs.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("📉 Saved:", out)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("▶ Running global-parameter stress tests")
    print(f"   scenarios={len(STRESS_SCENARIOS)} (includes baseline)")
    print(f"   N_RUNS={N_RUNS}, N_SEEDS={N_SEEDS}")
    print(f"   MAX_WORKERS={MAX_WORKERS} (override with env MAX_WORKERS)")
    print(f"   CHUNK_SIZE={CHUNK_SIZE} (override with env CHUNK_SIZE)")
    print(f"   Output dir: {OUTPUT_DIR}")

    df_raw, df_summary = run_stress_tests(
        base_params=BASE_PARAMS,
        base_cal=BASE_CAL,
        scenarios=STRESS_SCENARIOS,
        n_runs=N_RUNS,
        n_seeds=N_SEEDS,
        max_workers=MAX_WORKERS,
        chunksize=CHUNK_SIZE,
    )

    # Charts saved as PNG
    plot_bars(df_summary, "3y")
    plot_bars(df_summary, "6y")
    plot_bars(df_summary, "9y")
    plot_delta_from_bjs(df_summary)

    print("\n✅ Done.")
