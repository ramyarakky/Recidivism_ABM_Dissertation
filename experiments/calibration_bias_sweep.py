import os, sys, time, math
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

# Ensure project-root imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from recidivism_abm.model.recidivism_model import RecidivismModel


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MASTER_SEED = 424242

BASE_PARAMS = {
    "initial_agents": 1500,
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": 0,  # keep here if your model __init__ expects it; calibration overrides below are used anyway
    "enable_peer_influence": True
}

# BJS targets you showed
BJS_GENDER = {
    "3y": {"Male": 0.68, "Female": 0.58},
    "6y": {"Male": 0.79, "Female": 0.70},
    "9y": {"Male": 0.84, "Female": 0.77},
}
BJS_RACE = {
    "3y": {"White": 0.66, "Black": 0.72, "Hispanic": 0.68},
    "6y": {"White": 0.77, "Black": 0.82, "Hispanic": 0.79},
    "9y": {"White": 0.81, "Black": 0.87, "Hispanic": 0.84},
}

# Your bias maps (fixed during sweep)
BASE_CALIBRATION = {
    "group_bias_gender": {"Male": 0.25, "Female": -0.25},
    "group_bias_race": {"White": -0.10, "Black": 0.15, "Hispanic": 0.05, "Other": 0.00},

    # we sweep this:
    "bias_factor": 0.0,

    # apply bias ONLY to tech detection
    "bias_on_trial": 0.0,
    "bias_on_tech_detect": 1.0,
    "tech_detect_base": 0.35,

    # optional: if you later add escalation-to-trial from tech flags
    # "p_trial_base": 0.05,
    # "p_trial_tech_boost": 0.20,
    # "p_trial_prior_boost": 0.10,
}


# ─────────────────────────────────────────────
# JOB SPEC
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class Job:
    bias_factor: float
    seed: int
    run_id: int
    job_id: int


# ─────────────────────────────────────────────
# GROUP METRICS (agent-level aggregation)
# ─────────────────────────────────────────────
def _safe_rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom > 0 else float("nan")


def _filter_agents(model) -> List[Any]:
    agents = list(getattr(model.schedule, "agents", []))
    # Prefer study_eligible cohort if present
    eligible = [a for a in agents if getattr(a, "study_eligible_agent", True)]
    return eligible if eligible else agents


def _rate_by_group(agents: List[Any], group_attr: str, group_values: List[str], flag_attr: str) -> Dict[str, float]:
    out = {}
    for gv in group_values:
        grp = [a for a in agents if getattr(a, group_attr, None) == gv]
        denom = len(grp)
        numer = sum(1 for a in grp if bool(getattr(a, flag_attr, False)))
        out[gv] = _safe_rate(numer, denom)
    return out


def _collect_metrics(model) -> Dict[str, float]:
    """
    Collects:
      - rearrest rates by gender and race (3/6/9)
      - tech violation detected rates by gender and race
    """
    agents = _filter_agents(model)

    # rearrest flags (match your flag names)
    rearrest_flags = {
        "3y": "rearrest_3_yrs",
        "6y": "rearrest_6_yrs",
        "9y": "rearrest_9_yrs",
    }

    genders = ["Male", "Female"]
    races = ["White", "Black", "Hispanic", "Other"]

    metrics = {}

    # Rearrest by gender/race for each horizon
    for h, flag in rearrest_flags.items():
        g_rates = _rate_by_group(agents, "Gender", genders, flag)
        r_rates = _rate_by_group(agents, "Race", races, flag)

        for g in genders:
            metrics[f"rearrest_{h}_gender_{g}"] = g_rates.get(g, float("nan"))
        for r in races:
            metrics[f"rearrest_{h}_race_{r}"] = r_rates.get(r, float("nan"))

    # Technical violation detected:
    # Prefer "Ever_Technical_Violation" if you add it; else fallback to current "Violations_Technical"
    tech_flag = "Ever_Technical_Violation"
    if not any(hasattr(a, tech_flag) for a in agents):
        tech_flag = "Violations_Technical"

    tech_g = _rate_by_group(agents, "Gender", genders, tech_flag)
    tech_r = _rate_by_group(agents, "Race", races, tech_flag)

    metrics["tech_gender_Male"] = tech_g.get("Male", float("nan"))
    metrics["tech_gender_Female"] = tech_g.get("Female", float("nan"))

    metrics["tech_race_White"] = tech_r.get("White", float("nan"))
    metrics["tech_race_Black"] = tech_r.get("Black", float("nan"))
    metrics["tech_race_Hispanic"] = tech_r.get("Hispanic", float("nan"))
    metrics["tech_race_Other"] = tech_r.get("Other", float("nan"))

    metrics["n_agents_eval"] = len(agents)
    metrics["tech_flag_used"] = tech_flag  # to confirm which one was used

    return metrics


# ─────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────
def run_job(job: Job, base_params: Dict[str, Any], base_cal: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Build per-job calibration
    cal = dict(base_cal)
    cal["bias_factor"] = float(job.bias_factor)

    params = dict(base_params)
    params["seed"] = int(job.seed)

    model = RecidivismModel(**params)

    # Ensure model has a deterministic numpy RNG (if your model already sets this, it’s fine)
    if not hasattr(model, "np_rng"):
        model.np_rng = np.random.default_rng(int(job.seed))

    # Attach calibration dict where your agent code reads it
    model.calibration = cal

    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()

    # Align for consistent evaluation (your prior approach)
    model.current_month = model.warmup_months + 108

    metrics = _collect_metrics(model)

    metrics.update({
        "job_id": job.job_id,
        "run_id": job.run_id,
        "seed": job.seed,
        "bias_factor": float(job.bias_factor),
        "runtime_sec": time.perf_counter() - t0,
    })
    return metrics


# ─────────────────────────────────────────────
# JOB GENERATION (optimized + deterministic)
# ─────────────────────────────────────────────
def generate_jobs(
    master_seed: int,
    bias_factors: List[float],
    n_runs: int,
    n_seeds: int
) -> List[Job]:
    rng = np.random.default_rng(master_seed)
    total = len(bias_factors) * n_runs * n_seeds

    # generate unique seeds
    seeds = rng.integers(1, 2**31 - 1, size=total, dtype=np.int64)
    seeds = [int(s) for s in seeds]

    jobs = []
    k = 0
    job_id = 0
    for bf in bias_factors:
        for run_id in range(1, n_runs + 1):
            for _ in range(n_seeds):
                jobs.append(Job(bias_factor=float(bf), seed=seeds[k], run_id=run_id, job_id=job_id))
                k += 1
                job_id += 1
    return jobs


# ─────────────────────────────────────────────
# SWEEP RUNNER (single pool, low overhead)
# ─────────────────────────────────────────────
def run_sweep_fast(
    master_seed: int,
    bias_factors: List[float],
    n_runs: int,
    n_seeds: int,
    max_workers: int,
    chunksize: int = 2
) -> pd.DataFrame:
    jobs = generate_jobs(master_seed, bias_factors, n_runs, n_seeds)

    worker = lambda j: run_job(j, BASE_PARAMS, BASE_CALIBRATION)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(worker, jobs, chunksize=chunksize)
        for row in tqdm(it, total=len(jobs), desc="Calibration sweep", miniters=1):
            results.append(row)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# LOSS / SELECTION
# ─────────────────────────────────────────────
def compute_loss(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    summary_df has one row per bias_factor with mean subgroup rates.
    Adds loss columns vs BJS targets.
    """
    def sq(x): return (x * x)

    losses = []
    for _, r in summary_df.iterrows():
        bf = float(r["bias_factor"])

        # weights
        w = {"3y": 2.0, "6y": 1.0, "9y": 1.0}

        loss_gender = 0.0
        loss_race = 0.0

        for h in ["3y", "6y", "9y"]:
            # gender
            for g in ["Male", "Female"]:
                pred = float(r[f"rearrest_{h}_gender_{g}"])
                tgt = float(BJS_GENDER[h][g])
                loss_gender += w[h] * sq(pred - tgt)

            # race (ignore Other for loss)
            for rc in ["White", "Black", "Hispanic"]:
                pred = float(r[f"rearrest_{h}_race_{rc}"])
                tgt = float(BJS_RACE[h][rc])
                loss_race += w[h] * sq(pred - tgt)

        losses.append({
            "bias_factor": bf,
            "loss_gender": loss_gender,
            "loss_race": loss_race,
            "loss_total": loss_gender + loss_race,
        })

    loss_df = pd.DataFrame(losses).sort_values("loss_total").reset_index(drop=True)
    return loss_df


def summarize_by_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean subgroup metrics per bias_factor.
    """
    metric_cols = [c for c in df.columns if c.startswith(("rearrest_", "tech_"))]
    out = df.groupby("bias_factor")[metric_cols].mean().reset_index()
    return out.sort_values("bias_factor").reset_index(drop=True)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_tech_sensitivity(summary: pd.DataFrame):
    # Gender
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=summary["bias_factor"], y=summary["tech_gender_Male"],
                               mode="lines+markers", name="Male"))
    fig_g.add_trace(go.Scatter(x=summary["bias_factor"], y=summary["tech_gender_Female"],
                               mode="lines+markers", name="Female"))
    fig_g.update_layout(
        title="Detected Technical Violation Rate vs bias_factor — Gender",
        xaxis_title="bias_factor",
        yaxis_title="Detected tech violation rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white"
    )
    fig_g.show()

    # Race
    fig_r = go.Figure()
    for col, label in [
        ("tech_race_White", "White"),
        ("tech_race_Black", "Black"),
        ("tech_race_Hispanic", "Hispanic"),
        ("tech_race_Other", "Other"),
    ]:
        fig_r.add_trace(go.Scatter(x=summary["bias_factor"], y=summary[col],
                                   mode="lines+markers", name=label))
    fig_r.update_layout(
        title="Detected Technical Violation Rate vs bias_factor — Race",
        xaxis_title="bias_factor",
        yaxis_title="Detected tech violation rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white"
    )
    fig_r.show()


def plot_rearrest_sensitivity_gender(summary: pd.DataFrame):
    for h in ["3y", "6y", "9y"]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary["bias_factor"], y=summary[f"rearrest_{h}_gender_Male"],
                                 mode="lines+markers", name="ABM Male"))
        fig.add_trace(go.Scatter(x=summary["bias_factor"], y=summary[f"rearrest_{h}_gender_Female"],
                                 mode="lines+markers", name="ABM Female"))

        # BJS targets as horizontal lines (two traces)
        fig.add_trace(go.Scatter(x=summary["bias_factor"], y=[BJS_GENDER[h]["Male"]]*len(summary),
                                 mode="lines", name=f"BJS Male ({h})"))
        fig.add_trace(go.Scatter(x=summary["bias_factor"], y=[BJS_GENDER[h]["Female"]]*len(summary),
                                 mode="lines", name=f"BJS Female ({h})"))

        fig.update_layout(
            title=f"Rearrest Rate vs bias_factor — Gender ({h})",
            xaxis_title="bias_factor",
            yaxis_title="Rearrest rate",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            template="plotly_white"
        )
        fig.show()


def plot_rearrest_sensitivity_race(summary: pd.DataFrame):
    for h in ["3y", "6y", "9y"]:
        fig = go.Figure()
        for rc in ["White", "Black", "Hispanic"]:
            fig.add_trace(go.Scatter(x=summary["bias_factor"], y=summary[f"rearrest_{h}_race_{rc}"],
                                     mode="lines+markers", name=f"ABM {rc}"))
            fig.add_trace(go.Scatter(x=summary["bias_factor"], y=[BJS_RACE[h][rc]]*len(summary),
                                     mode="lines", name=f"BJS {rc} ({h})"))

        fig.update_layout(
            title=f"Rearrest Rate vs bias_factor — Race ({h})",
            xaxis_title="bias_factor",
            yaxis_title="Rearrest rate",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            template="plotly_white"
        )
        fig.show()


def plot_loss(loss_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=loss_df["bias_factor"], y=loss_df["loss_total"],
                             mode="lines+markers", name="Total loss"))
    fig.add_trace(go.Scatter(x=loss_df["bias_factor"], y=loss_df["loss_gender"],
                             mode="lines+markers", name="Gender loss"))
    fig.add_trace(go.Scatter(x=loss_df["bias_factor"], y=loss_df["loss_race"],
                             mode="lines+markers", name="Race loss"))
    fig.update_layout(
        title="Calibration Loss vs bias_factor (lower is better)",
        xaxis_title="bias_factor",
        yaxis_title="Loss (weighted MSE)",
        template="plotly_white"
    )
    fig.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Sweep values (adjust as needed)
    bias_grid = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    # Start smaller for speed; increase once you see separation
    n_runs = 5
    n_seeds = 10

    max_workers = min(os.cpu_count() or 4, n_seeds)  # keep worker count sane
    chunksize = 2

    print(f"Running sweep with {len(bias_grid)} bias factors × {n_runs} runs × {n_seeds} seeds "
          f"= {len(bias_grid)*n_runs*n_seeds} simulations")

    df = run_sweep_fast(
        master_seed=MASTER_SEED,
        bias_factors=bias_grid,
        n_runs=n_runs,
        n_seeds=n_seeds,
        max_workers=max_workers,
        chunksize=chunksize,
    )

    # Summaries
    summary = summarize_by_bias(df)
    loss_df = compute_loss(summary)

    print("\nTech flag used (should be Ever_Technical_Violation if you added it):")
    print(df["tech_flag_used"].value_counts(dropna=False))

    print("\nTop 5 bias_factor by total loss:")
    print(loss_df.head(5))

    # Charts
    plot_tech_sensitivity(summary)
    plot_rearrest_sensitivity_gender(summary)
    plot_rearrest_sensitivity_race(summary)
    plot_loss(loss_df)

    # Export for your records
    df.to_csv("calibration_sweep_raw.csv", index=False)
    summary.to_csv("calibration_sweep_summary.csv", index=False)
    loss_df.to_csv("calibration_sweep_loss.csv", index=False)

    print("\nSaved:")
    print(" - calibration_sweep_raw.csv")
    print(" - calibration_sweep_summary.csv")
    print(" - calibration_sweep_loss.csv")
