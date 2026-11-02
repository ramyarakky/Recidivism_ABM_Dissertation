"""
OAT_Calibrate_BJS_PCRA.py   —  Joint OAT Calibration  [PARALLEL]
====================================================
Recidivism ABM — PhD Dissertation Tool  [v5 — Stage 3 offense shifts added]

CALIBRATION TARGETS
-------------------
Stage 1 — BJS Aggregate (Alper et al., 2018, NCJ 250975):
  3yr = 68.4%,  6yr = 79.4%,  9yr = 83.4%

Stage 2 — PCRA Tier-Stratified (Johnson, 2023, Federal Probation 87(2)):
  Low:         3yr=46.2%  6yr=61.4%  9yr=67.6%
  LowModerate: 3yr=72.0%  6yr=84.3%  9yr=88.8%
  Moderate:    3yr=84.5%  6yr=92.1%  9yr=94.6%
  High:        3yr=91.0%  6yr=95.0%  9yr=95.0%

Stage 3 — BJS Per-Offense (Alper et al., 2018, NCJ 250975, Table 7):
  Violent, Drug, Property, Other(PublicOrder) at 3/6/9 years.
  Calibrates offense_hazard_shift dict (4 keys).

CALIBRATION ORDER (11 parameters, 3 fixed)
-------------------------------------------
Step 1  : Risk_Contrast_Strength                 (gamma)  [Stage 2]
Step 2  : Supervision_Monitoring_Intensity       (alpha)  [Stage 1]
Step 3  : Risk_Effect_Decay_After_1Y             (dr1)    [Stage 1] FIXED=0.524
Step 4  : Risk_Effect_Decay_After_3Y             (dr3)    [Stage 1] FIXED=0.500
Step 5  : Risk_Effect_Decay_After_6Y             (dr6)    [Stage 1] FIXED=0.508
Step 6  : Supervision_Monitoring_Decay_After_3Y  (ds3)    [Stage 1]
Step 7  : Supervision_Monitoring_Decay_After_6Y  (ds6)    [Stage 1]
Step 8  : offense_hazard_shift.Violent           (ov)     [Stage 3]
Step 9  : offense_hazard_shift.Drug              (od)     [Stage 3]
Step 10 : offense_hazard_shift.Property          (op)     [Stage 3]
Step 11 : offense_hazard_shift.Other(PublicOrder)(oo)    [Stage 3]

VALIDATION ANCHORS
------------------
Year 1 is collected and reported as a diagnostic validation metric only.
It is NOT a calibration target.
"""

import os, sys, json, warnings, multiprocessing
warnings.filterwarnings("ignore")

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
from matplotlib.lines  import Line2D
from matplotlib.patches import Patch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_uncalibrated_params,
    get_flat_risk_weights,
    _deep_merge,
)

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "initial_agents":        3000,
    "warmup_months":         144,
    "study_months":          108,
    "monthly_intake":        10,
    "bias_factor":           0.0,
    "enable_peer_influence": True,
    "mode":                  "realistic",
    "output_directory":      "OAT_Calibrate_BJS_PCRA_Offense_Output",
    "N_REPS":                10,
    "SEEDS":                 [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "bjs_targets":           {3: 0.684, 6: 0.794, 9: 0.834},
}

BJS_DIAGNOSTIC = {1: 0.439}

PCRA_TARGETS = {
    "Low":         {3: 0.462, 6: 0.614, 9: 0.676},
    "LowModerate": {3: 0.720, 6: 0.843, 9: 0.888},
    "Moderate":    {3: 0.845, 6: 0.921, 9: 0.946},
    "High":        {3: 0.910, 6: 0.950, 9: 0.950},
}
TIERS        = ["Low", "LowModerate", "Moderate", "High"]
TIERS_LB     = ["Low", "Low-Moderate", "Moderate", "High"]
WINDOWS      = [3, 6, 9]
WIN_LB       = ["3-Year", "6-Year", "9-Year"]

PCRA_COUNTS  = {"Low": 138230, "LowModerate": 169153, "Moderate": 113487, "High": 45467}
PCRA_N_TOTAL = 475528
PCRA_WEIGHTS = [0.291, 0.356, 0.239, 0.096]

# STAGE 3 — BJS Per-Offense targets (NCJ 250975 Table 7)
BJS_OFFENSE_TARGETS = {
    "Violent":             {3: 0.622, 6: 0.742, 9: 0.787},   
	"Property":            {3: 0.75, 6: 0.844, 9: 0.878},
    "Drug":                {3: 0.686, 6: 0.798, 9: 0.838},
    "Other(PublicOrder)":  {3: 0.65, 6: 0.769, 9: 0.819},
}
OFFENSES      = ["Violent", "Drug", "Property", "Other(PublicOrder)"]
OFFENSES_LB   = ["Violent", "Drug", "Property", "Other/PublicOrder"]

# =============================================================================
# CALIBRATION STEPS
# =============================================================================
CALIBRATION_STEPS = [
    {
        "param":   "Risk_Contrast_Strength",
        "values":  np.linspace(0.75, 1.50, 16),
        "loss":    "gamma",
        "primary": [3, 6, 9],
        "csv":     "sweep_gamma.csv",
        "symbol":  "gamma",
        "label":   "Risk Contrast Strength (gamma)",
        "short":   "gamma -- PCRA tier spread",
        "note":    "Admissible range 0.75-1.50. Source: Johnson (2023)",
        "stage":   "2",
    },
    {
        "param":   "Supervision_Monitoring_Intensity",
        "values":  np.linspace(1.00, 1.20, 11),
        "loss":    "aggregate",
        "primary": [3, 6, 9],
        "csv":     "sweep_alpha.csv",
        "symbol":  "alpha",
        "label":   "Supervision Monitoring Intensity (alpha)",
        "short":   "alpha - Supervision intensity",
        "note":    "Admissible range 0.90-1.30. Source: Langan & Levin (2002)",
        "stage":   "1",
    },
    {
        "param":   "Risk_Effect_Decay_After_1Y",
        "values":  np.array([0.524]),
        "loss":    "aggregate",
        "primary": [3],
        "csv":     "sweep_decay1y.csv",
        "symbol":  "dr1",
        "label":   "Risk Effect Decay -- Years 1-3 (dr1) [FIXED=0.524]",
        "short":   "dr1 - Desistance years 1-3 [fixed]",
        "note":    "Fixed at BJS anchor q_23/q_1 = 0.524.",
        "stage":   "1*",
    },
    {
        "param":   "Risk_Effect_Decay_After_3Y",
        "values":  np.array([0.50]),
        "loss":    "aggregate",
        "primary": [6],
        "csv":     "sweep_decay3y.csv",
        "symbol":  "dr3",
        "label":   "Risk Effect Decay -- Years 3-6 (dr3) [FIXED=0.50]",
        "short":   "dr3 - Desistance years 3-6 [fixed]",
        "note":    "Fixed at BJS anchor q2/q1=0.50.",
        "stage":   "1*",
    },
    {
        "param":   "Risk_Effect_Decay_After_6Y",
        "values":  np.array([0.508]),
        "loss":    "aggregate",
        "primary": [9],
        "csv":     "sweep_decay6y.csv",
        "symbol":  "dr6",
        "label":   "Risk Effect Decay -- Years 6-9 (dr6) [FIXED=0.508]",
        "short":   "dr6 - Desistance years 6-9 [fixed]",
        "note":    "Fixed at BJS anchor q3/q2=0.508.",
        "stage":   "1*",
    },
    {
        "param":   "Supervision_Monitoring_Decay_After_3Y",
        "values":  np.linspace(0.60, 0.99, 13),
        "loss":    "aggregate",
        "primary": [6],
        "csv":     "sweep_smi_decay3y.csv",
        "symbol":  "ds3",
        "label":   "Supervision Monitoring Decay -- Years 3-6 (ds3)",
        "short":   "ds3 - Supervision intensity decay 3-6yr",
        "note":    "Admissible range 0.60-0.99. Source: Petersilia (2003)",
        "stage":   "1",
    },
    {
        "param":   "Supervision_Monitoring_Decay_After_6Y",
        "values":  np.linspace(0.20, 0.70, 11),
        "loss":    "aggregate",
        "primary": [9],
        "csv":     "sweep_smi_decay6y.csv",
        "symbol":  "ds6",
        "label":   "Supervision Monitoring Decay -- Years 6-9 (ds6)",
        "short":   "ds6 - Supervision intensity decay 6-9yr",
        "note":    "Admissible range 0.20-0.70. Source: Petersilia (2003)",
        "stage":   "1",
    },
    # ── STAGE 3 — Offense-specific hazard shifts ─────────────────────────────
    # Calibrated against BJS NCJ 250975 Table 7 per-offense rearrest targets.
    # Sweep runs AFTER Stage 1 (aggregate) and Stage 2 (tier) are identified,
    # so γ, α, and decay parameters are locked before offense shifts are swept.
    #
    # Baseline gaps (20 seeds, post Stage 1+2, all offense shifts at 0.0):
    #   Offense             Yr 3     Yr 6     Yr 9    Pattern
    #   Violent             +8.6pp   +4.4pp   +0.8pp  Shape drift, Yr 9 at target
    #   Drug                -0.1pp   -1.3pp   -3.5pp  Near target across all windows
    #   Property            -8.9pp   -6.6pp   -8.2pp  Level offset (~-8pp constant)
    #   Other(PublicOrder)  +8.1pp   +6.6pp   +3.4pp  Shape drift, Yr 9 near target
    #
    # Sweep ranges calibrated to span the likely optimum for each offense
    # given these baseline gaps and the observed shift-to-rate sensitivity
    # (+0.5 shift ≈ +10 pp in rate, from smoke-test validation).
    {
        "param":   "offense_hazard_shift.Violent",
        "values": np.linspace(-0.40, +0.05, 10),
        "loss":    "offense",
        "primary": [3, 6, 9],
        "csv":     "sweep_oshift_violent.csv",
        "symbol":  "ov",
        "label":   "Offense hazard shift -- Violent (ov)",
        "short":   "ov -- Violent offense shift",
        "note":    "Narrow confirmation sweep. Baseline near target. "
                   "Source: Alper et al. (2018), NCJ 250975 Table 7.",
        "stage":   "3",
    },
    {
        "param":   "offense_hazard_shift.Drug",
        "values": np.linspace(-0.15, +0.20, 8),
        "loss":    "offense",
        "primary": [3, 6, 9],
        "csv":     "sweep_oshift_drug.csv",
        "symbol":  "od",
        "label":   "Offense hazard shift -- Drug (od)",
        "short":   "od -- Drug offense shift",
        "note":    "Positive shift to close -10.2pp gap. "
                   "Source: Alper et al. (2018); Durose et al. (2014).",
        "stage":   "3",
    },
    {
        "param":   "offense_hazard_shift.Property",
        "values": np.linspace(+0.20, +0.80, 13),
        "loss":    "offense",
        "primary": [3, 6, 9],
        "csv":     "sweep_oshift_property.csv",
        "symbol":  "op",
        "label":   "Offense hazard shift -- Property (op)",
        "short":   "op -- Property offense shift",
        "note":    "Large positive shift to close -16.5pp gap. "
                   "Source: Alper et al. (2018), NCJ 250975 Table 7.",
        "stage":   "3",
    },
    {
        "param":   "offense_hazard_shift.Other(PublicOrder)",
        "values": np.linspace(-0.40, +0.05, 10),
        "loss":    "offense",
        "primary": [3, 6, 9],
        "csv":     "sweep_oshift_pubord.csv",
        "symbol":  "oo",
        "label":   "Offense hazard shift -- Other/PublicOrder (oo)",
        "short":   "oo -- Other/PublicOrder shift",
        "note":    "Narrow confirmation sweep. Baseline near target. "
                   "Source: Alper et al. (2018), NCJ 250975 Table 7.",
        "stage":   "3",
    },
]
CALIBRATION_ORDER = [s["param"] for s in CALIBRATION_STEPS]

# =============================================================================
# COLOUR PALETTE
# =============================================================================
C = {
    "bjs":        "#1A3D5C",
    "baseline":   "#888888",
    "calibrated": "#D05A28",
    "band":       "#DDEEFF",
    "3yr":        "#2166AC",
    "6yr":        "#4DAC26",
    "9yr":        "#E08030",
    "optimal":    "#CC0000",
    "grid":       "#DDDDDD",
    "Low":        "#2166AC",
    "LowModerate":"#74ADD1",
    "Moderate":   "#F4A582",
    "High":       "#D6604D",
    "target":     "#1A3D5C",
    "good":       "#276419",
    "warn":       "#B8860B",
    "bad":        "#CC4400",
    # STAGE 3 — per-offense palette
    "Violent":              "#C0392B",
    "Drug":                 "#8E44AD",
    "Property":             "#2980B9",
    "Other(PublicOrder)":   "#27AE60",
}

def _gap_color(gap):
    if abs(gap) <= 0.02: return C["good"]
    if abs(gap) <= 0.05: return C["warn"]
    return C["bad"]

def _gap_label(gap):
    if abs(gap) <= 0.02: return "(within 2pp)"
    if abs(gap) <= 0.05: return "(within 5pp)"
    return "(exceeds 5pp)"

def _style(ax, xlabel="", ylabel="", title="", fs=11):
    ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    ax.set_title(title, fontsize=fs, fontweight="bold", pad=9)
    ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

def _pct(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.))

def _bjs_band(ax, yrs):
    t = CONFIG["bjs_targets"][yrs]
    ax.axhline(t, color=C["bjs"], linewidth=2., linestyle="-",
               label=f"BJS target ({t:.1%})", zorder=4)


# =============================================================================
# STAGE 3 — parameter path helper
# =============================================================================
def _wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n <= 0:
        return (None, None)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half   = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def _apply_param(ov, param, value):
    """
    Apply a possibly-dotted parameter path to the calibration dict.
    Handles flat keys like "Risk_Contrast_Strength" and nested keys like
    "offense_hazard_shift.Property".
    """
    if "." not in param:
        ov[param] = float(value)
        return
    outer, inner = param.split(".", 1)
    if outer not in ov or not isinstance(ov[outer], dict):
        ov[outer] = {}
    else:
        # Deep copy to prevent mutating sibling parameter locked state
        ov[outer] = dict(ov[outer])
    ov[outer][inner] = float(value)


def _get_param(d, param):
    """Read a possibly-dotted parameter path from a dict."""
    if "." not in param:
        return d.get(param)
    outer, inner = param.split(".", 1)
    if outer not in d or not isinstance(d[outer], dict):
        return None
    return d[outer].get(inner)


# =============================================================================
# WORKER
# STAGE 3 — now also returns per-offense 3/6/9 rates and model offense loss.
# =============================================================================
def _worker(task):
    (cal_json, seed, warmup, study, agents,
     intake, mode, bias, peer, outdir) = task

    overrides   = json.loads(cal_json)
    param_value = overrides.get("__task_value__", 0.0)
    cal = _deep_merge(
        get_uncalibrated_params(),
        {k: v for k, v in overrides.items() if not k.startswith("__")},
    )

    try:
        model = RecidivismModel(
            initial_agents=agents, bias_factor=bias,
            monthly_intake=intake, warmup_months=warmup,
            study_months=study, enable_peer_influence=peer,
            weights=get_flat_risk_weights(), seed=seed, mode=mode,
            output_directory=os.path.join(outdir, "tmp"),
            calibration_params=cal,
        )
        model.export_csv = False
        for _ in range(warmup + study):
            if not model.running: break
            model.step()

        eligible  = [a for a in model.schedule.agents
                     if getattr(a, "study_eligible_agent", False)]
        n_elig    = len(eligible)

        counts = {}
        for yrs in [1, 3, 6, 9]:
            counts[yrs] = sum(1 for a in eligible
                            if getattr(a, f"rearrest_{yrs}_yrs", False))

        # Aggregate rearrest rates (year 1 is diagnostic-only)
        rates = {}
        for yrs in [1, 3, 6, 9]:
            r = model.calculate_flag_rate(f"rearrest_{yrs}_yrs")
            rates[yrs] = r if r is not None else 0.0

        # Per-tier 3yr rates
        tier_r = {}
        for tier in TIERS:
            agents_t = [a for a in eligible if a.get_pcra_tier() == tier]
            nt = len(agents_t)
            tier_r[tier] = (
                sum(1 for a in agents_t if getattr(a, "rearrest_3_yrs", False)) / nt
                if nt > 0 else 0.0
            )

        # Tier composition
        tier_share = {}
        for tier in TIERS:
            agents_t = [a for a in eligible if a.get_pcra_tier() == tier]
            tier_share[tier] = len(agents_t) / n_elig if n_elig > 0 else 0.0

        # STAGE 3 — Per-offense rates at 3/6/9 years
        offense_rates = {}
        try:
            for yrs in [3, 6, 9]:
                rates_by_off = model.calculate_flag_rate_by_offense(yrs)
                for off in OFFENSES:
                    offense_rates[(off, yrs)] = rates_by_off.get(off, 0.0)
        except Exception:
            for yrs in [3, 6, 9]:
                for off in OFFENSES:
                    offense_rates[(off, yrs)] = 0.0

        # Stage 2 gamma loss
        try:
            gl = model.compute_gamma_loss()
            if gl is None or (isinstance(gl, float) and np.isnan(gl)):
                gl = 999.0
        except Exception:
            gl = 999.0

        # STAGE 3 — offense loss
        try:
            ol = model.compute_offense_loss()
            if ol is None or (isinstance(ol, float) and np.isnan(ol)):
                ol = 999.0
        except Exception:
            ol = 999.0

        return {
            "param_value": param_value,
            "seed":        seed,
            "rates":       rates,                   # {1, 3, 6, 9}
            "tier_r":      tier_r,                  # {tier: rate_3yr}
            "tier_share":  tier_share,              # {tier: share}
            "offense":     offense_rates,           # {(off, yrs): rate}
            "gamma_loss":  gl,
            "offense_loss": ol,
            "n_elig": n_elig,
            "counts": counts,
        }

    except Exception as e:
        print(f"  Worker error (value={param_value:.3f}, seed={seed}): {e}", flush=True)
        return {
                "param_value": param_value,
                "seed":        seed,
                "rates":       {1: 0.0, 3: 0.0, 6: 0.0, 9: 0.0},
                "counts":      {1: 0, 3: 0, 6: 0, 9: 0},      # NEW
                "n_elig":      0,                              # NEW
                "tier_r":      {t: 0.0 for t in TIERS},
                "tier_share":  {t: 0.25 for t in TIERS},
                "offense":     {(off, yrs): 0.0 for off in OFFENSES for yrs in [3, 6, 9]},
                "gamma_loss":  999.0,
                "offense_loss": 999.0,
            }


# =============================================================================
# TASK BUILDER
# STAGE 3 — uses _apply_param for dotted paths
# =============================================================================
def _build_tasks(param, values, locked):
    tasks = []
    for v in values:
        ov = json.loads(json.dumps(locked))   # deep copy
        _apply_param(ov, param, v)
        ov["__task_value__"] = float(v)
        cal_json = json.dumps(ov)
        for rep in range(CONFIG["N_REPS"]):
            for seed in CONFIG["SEEDS"]:
                tasks.append((
                    cal_json, seed + rep * 10_000,
                    CONFIG["warmup_months"], CONFIG["study_months"],
                    CONFIG["initial_agents"], CONFIG["monthly_intake"],
                    CONFIG["mode"], CONFIG["bias_factor"],
                    CONFIG["enable_peer_influence"], CONFIG["output_directory"]
                ))
    return tasks


# =============================================================================
# SWEEP ENGINE
# STAGE 3 — aggregates per-offense rates and offense_loss into rows
# =============================================================================
def sweep_parallel(step, locked, n_workers):
    param   = step["param"]
    values  = step["values"]
    targets = CONFIG["bjs_targets"]
    primary = step["primary"]
    n_tasks = len(values) * CONFIG["N_REPS"] * len(CONFIG["SEEDS"])

    assert all(w in targets for w in primary), (
        f"Step {step['symbol']} has primary window(s) outside calibration "
        f"targets {list(targets)}: {primary}"
    )

    print(f"\n  Sweeping {step['symbol']} ({step['label']})")
    print(f"  {len(values)} values x {CONFIG['N_REPS']} reps x "
          f"{len(CONFIG['SEEDS'])} seeds = {n_tasks} tasks")

    tasks = _build_tasks(param, values, locked)

    def _empty_bucket():
        b = {
            1: [], 3: [], 6: [], 9: [], "gamma": [], "offense_loss": [],
            "Low": [], "LowModerate": [], "Moderate": [], "High": [],
            "share_Low": [], "share_LowModerate": [],
            "share_Moderate": [], "share_High": [],
        }
        # STAGE 3 — per-offense rate buckets
        for off in OFFENSES:
            for yrs in [3, 6, 9]:
                b[f"off_{off}_{yrs}"] = []
        return b

    bucket = {float(v): _empty_bucket() for v in values}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=n_tasks, desc=f"  {step['symbol']:6s}", ncols=80) as bar:
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    key = min(bucket.keys(), key=lambda k: abs(k - r["param_value"]))
                    b = bucket[key]
                    for y in [1, 3, 6, 9]:
                        b[y].append(r["rates"][y])
                    for t in TIERS:
                        b[t].append(r["tier_r"][t])
                        b[f"share_{t}"].append(r["tier_share"][t])
                    b["gamma"].append(r["gamma_loss"])
                    b["offense_loss"].append(r["offense_loss"])
                    for off in OFFENSES:
                        for yrs in [3, 6, 9]:
                            b[f"off_{off}_{yrs}"].append(r["offense"][(off, yrs)])
                except Exception as e:
                    print(f"\n  Future error: {e}", flush=True)
                finally:
                    bar.update(1)

    rows = []
    for v in sorted(bucket.keys()):
        b = bucket[v]
        for yrs in [1, 3, 6, 9]:
            if not b[yrs]: b[yrs] = [0.]
        if not b["gamma"]: b["gamma"] = [999.]
        if not b["offense_loss"]: b["offense_loss"] = [999.]

        a1 = np.array(b[1])
        a3 = np.array(b[3]); a6 = np.array(b[6]); a9 = np.array(b[9])
        n  = len(a3)
        m1 = a1.mean()
        m3, m6, m9 = a3.mean(), a6.mean(), a9.mean()
        gam = np.array([g for g in b["gamma"] if g < 998.] or [999.])
        ol  = np.array([o for o in b["offense_loss"] if o < 998.] or [999.])
        tier_means = {t: np.mean(b[t]) if b[t] else 0. for t in TIERS}

        row = {
            "value":          v, "n_runs": n,
            "rate_1yr":       m1,
            "std_1yr":        float(a1.std(ddof=min(1, n-1))),
            "rate_3yr":       m3, "rate_6yr": m6, "rate_9yr": m9,
            "std_3yr":        float(a3.std(ddof=min(1, n-1))),
            "std_6yr":        float(a6.std(ddof=min(1, n-1))),
            "std_9yr":        float(a9.std(ddof=min(1, n-1))),
            "mae_3yr":        abs(m3-targets[3]),
            "mae_6yr":        abs(m6-targets[6]),
            "mae_9yr":        abs(m9-targets[9]),
            "mae_all":        (abs(m3-targets[3])+abs(m6-targets[6])+abs(m9-targets[9]))/3,
            "gamma_loss":     float(gam.mean()),
            "offense_loss":   float(ol.mean()),
            "tier_mae_3yr":   float(np.mean([
                abs(tier_means[t]-PCRA_TARGETS[t][3]) for t in TIERS])),
        }
        rate_map = {3: m3, 6: m6, 9: m9}
        row["mae_primary"] = float(np.mean([
            abs(rate_map[w] - targets[w]) for w in primary
        ]))
        for t in TIERS:
            row[f"rate_3yr_{t}"] = tier_means[t]
            row[f"share_{t}"]    = float(np.mean(b[f"share_{t}"])) if b[f"share_{t}"] else 0.25

        # STAGE 3 — per-offense means on the row
        for off in OFFENSES:
            for yrs in [3, 6, 9]:
                vals = b[f"off_{off}_{yrs}"]
                row[f"off_{off}_{yrs}yr"] = float(np.mean(vals)) if vals else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# OPTIMAL VALUE SELECTION
# STAGE 3 — offense loss selection added
# =============================================================================
def _select_optimal(df, step):
    if step["loss"] == "gamma":
        col = "gamma_loss"
    elif step["loss"] == "offense":
        col = "offense_loss"
    else:
        col = "mae_primary"
    if col not in df.columns or df[col].min() >= 998.:
        col = "mae_all"
    return float(df.loc[df[col].idxmin(), "value"])


# =============================================================================
# REPLICATION RUN
# STAGE 3 — per-offense rates added to raw, aggregate, and per-seed CSV
# =============================================================================
def run_replicated(overrides, n_workers, label="run"):
    ov = dict(overrides); ov["__task_value__"] = 0.
    cal_json = json.dumps(ov)

    task_meta = []
    tasks     = []
    for rep in range(CONFIG["N_REPS"]):
        for seed in CONFIG["SEEDS"]:
            t = (
                cal_json, seed + rep * 10_000,
                CONFIG["warmup_months"], CONFIG["study_months"],
                CONFIG["initial_agents"], CONFIG["monthly_intake"],
                CONFIG["mode"], CONFIG["bias_factor"],
                CONFIG["enable_peer_influence"], CONFIG["output_directory"]
            )
            tasks.append(t)
            task_meta.append((rep, seed))

    raw = {1: [], 3: [], 6: [], 9: [],
           "gamma": [], "offense_loss": [],
           "Low": [], "LowModerate": [], "Moderate": [], "High": [],
           "share_Low": [], "share_LowModerate": [],
           "share_Moderate": [], "share_High": [],
           "n_elig": [],          # NEW
           "counts_per_yr": []   # NEW — list of dicts {1: k, 3: k, 6: k, 9: k}
           }
    for off in OFFENSES:
        for yrs in [3, 6, 9]:
            raw[f"off_{off}_{yrs}"] = []

    per_seed_rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_meta = {
            pool.submit(_worker, t): meta
            for t, meta in zip(tasks, task_meta)
        }
        for fut, (rep, seed) in future_to_meta.items():
            try:
                r = fut.result()
                for y in [1, 3, 6, 9]:
                    raw[y].append(r["rates"][y])
                raw["gamma"].append(r["gamma_loss"])
                raw["offense_loss"].append(r["offense_loss"])
                for t in TIERS:
                    raw[t].append(r["tier_r"][t])
                    raw[f"share_{t}"].append(r["tier_share"][t])
                for off in OFFENSES:
                    for yrs in [3, 6, 9]:
                        raw[f"off_{off}_{yrs}"].append(r["offense"][(off, yrs)])
                        
                raw["n_elig"].append(r.get("n_elig", 0))
                raw["counts_per_yr"].append(r.get("counts", {1: 0, 3: 0, 6: 0, 9: 0}))

                n_elig = r.get("n_elig", 0)
                counts = r.get("counts", {1: 0, 3: 0, 6: 0, 9: 0})

                row = {
                    "label":        label,
                    "rep":          rep,
                    "seed":         seed,
                    "actual_seed":  seed + rep * 10_000,
                    "n_elig":       n_elig,
                    "rate_1yr":     r["rates"][1],
                    "rate_3yr":     r["rates"][3],
                    "rate_6yr":     r["rates"][6],
                    "rate_9yr":     r["rates"][9],
                    "gamma_loss":   r["gamma_loss"] if r["gamma_loss"] < 998. else None,
                    "offense_loss": r["offense_loss"] if r["offense_loss"] < 998. else None,
                }

                # Per-run 95% Wilson CI on each rate (within-run binomial sampling uncertainty)
                for yrs in [1, 3, 6, 9]:
                    lo, hi = _wilson_ci(counts.get(yrs, 0), n_elig)
                    row[f"ci95_lo_{yrs}yr"] = lo
                    row[f"ci95_hi_{yrs}yr"] = hi
                    row[f"ci95_halfwidth_{yrs}yr"] = ((hi - lo) / 2.0) if (lo is not None and hi is not None) else None

                for t in TIERS:
                    row[f"tier_3yr_{t}"] = r["tier_r"][t]
                    row[f"share_{t}"]    = r["tier_share"][t]
                for off in OFFENSES:
                    for yrs in [3, 6, 9]:
                        row[f"off_{off}_{yrs}yr"] = r["offense"][(off, yrs)]
                per_seed_rows.append(row)

            except Exception as e:
                print(f"  {e}", flush=True)

    if per_seed_rows:
        csv_path = os.path.join(
            CONFIG["output_directory"], f"{label}_per_seed.csv"
        )
        pd.DataFrame(per_seed_rows).to_csv(csv_path, index=False)
        print(f"  Per-seed CSV ({len(per_seed_rows)} rows) -> {csv_path}")

    out = {}
    for yrs in [1, 3, 6, 9]:
        arr = np.array(raw[yrs])
        out[yrs]            = float(arr.mean())
        out[f"std_{yrs}yr"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.
        out[f"all_{yrs}"]   = arr.tolist()
    gam = np.array([g for g in raw["gamma"] if g < 998.] or [999.])
    ol  = np.array([o for o in raw["offense_loss"] if o < 998.] or [999.])
    out["gamma_loss"]   = float(gam.mean())
    out["offense_loss"] = float(ol.mean())
    for t in TIERS:
        arr   = np.array(raw[t])
        arr_s = np.array(raw[f"share_{t}"])
        out[f"tier_3yr_{t}"]   = float(arr.mean())   if len(arr)   else 0.
        out[f"tier_share_{t}"] = float(arr_s.mean()) if len(arr_s) else 0.
    # STAGE 3 — aggregate per-offense means
    for off in OFFENSES:
        for yrs in [3, 6, 9]:
            arr = np.array(raw[f"off_{off}_{yrs}"])
            out[f"off_{off}_{yrs}yr"] = float(arr.mean()) if len(arr) else 0.

    out["n_elig_per_run"] = list(raw["n_elig"])
    for yrs in [1, 3, 6, 9]:
        out[f"counts_{yrs}_per_run"] = [c.get(yrs, 0) for c in raw["counts_per_yr"]]
    return out


def _norm(res):
    out = {}
    for k, v in res.items():
        try: out[int(k)] = v
        except: out[k] = v
    return out


# =============================================================================
# SWEEP CHARTS — dissertation-level titles, metrics, non-overlapping layout
# =============================================================================
def plot_sweep(df, step, opt, baseline, outdir):
    """
    2×2 panel sweep chart. Panel A = primary loss curve; Panels B/C/D =
    ABM rate vs BJS target at 3/6/9 years, with optimum marker.
    """
    is_gamma   = step["loss"] == "gamma"
    is_offense = step["loss"] == "offense"
    x = df["value"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    if is_gamma:
        stage_label = "Stage 2 — PCRA tier-stratified loss"
        loss_description = ("Mean |ABM tier rate − PCRA target| "
                            "across 4 tiers × 3 windows (12 cells)")
    elif is_offense:
        stage_label = "Stage 3 — BJS per-offense loss"
        loss_description = ("Mean |ABM offense rate − BJS target| "
                            "across 4 offenses × 3 windows (12 cells)")
    else:
        stage_label = "Stage 1 — BJS aggregate loss"
        loss_description = ("Mean |ABM rate − BJS target| "
                            "across 3-, 6-, and 9-year windows")

    # ── Title block ────────────────────────────────────────────────────────
    fig.suptitle(
        f"OAT Parameter Sweep: {step['label']}\n"
        f"Identified optimum: {step['symbol']} = {opt:.3f}   |   {stage_label}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.945, loss_description,
        ha="center", fontsize=9.5, color="#444444", style="italic",
    )

    # ── Panel A — primary loss curve ──────────────────────────────────────
    ax = axes[0, 0]
    if is_gamma:
        loss_col = "gamma_loss"
        ylab = "PCRA tier MAE (rate units, 0 – 1)"
    elif is_offense:
        loss_col = "offense_loss"
        ylab = "BJS offense MAE (rate units, 0 – 1)"
    else:
        loss_col = "mae_primary"
        ylab = "BJS aggregate MAE (rate units, 0 – 1)"

    y = df[loss_col].values
    ax.plot(x, y, color=C["calibrated"], linewidth=2., marker="o", markersize=5,
            label="Mean loss across seeds", zorder=3)
    ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
               label=f"Identified optimum ({step['symbol']} = {opt:.3f})",
               zorder=5)
    opt_y = float(df.loc[df["value"].sub(opt).abs().idxmin(), loss_col])

    ax.annotate(
        f"Loss at optimum: {opt_y:.4f}",
        xy=(opt, opt_y), xytext=(14, 18),
        textcoords="offset points", fontsize=9,
        color=C["optimal"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=C["optimal"], alpha=0.95, linewidth=1.0),
        arrowprops=dict(arrowstyle="->", color=C["optimal"], lw=1.2),
    )

    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    _style(ax, step["label"], ylab, "Panel A — Primary Loss Function")

    # ── Panels B/C/D — per-window rate curves ─────────────────────────────
    for ax, (yrs, col, std_col, colour, panel) in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        [(3, "rate_3yr", "std_3yr", C["3yr"], "B"),
         (6, "rate_6yr", "std_6yr", C["6yr"], "C"),
         (9, "rate_9yr", "std_9yr", C["9yr"], "D")]
    ):
        y      = df[col].values
        sd     = df[std_col].values if std_col in df.columns else np.zeros_like(y)
        base   = baseline.get(yrs, 0.)
        target = CONFIG["bjs_targets"][yrs]

        _bjs_band(ax, yrs)
        ax.axhline(base, color=C["baseline"], linewidth=1.5, linestyle=":",
                   label=f"Uncalibrated baseline ({base:.1%})", zorder=2)
        ax.plot(x, y, color=colour, linewidth=2., marker="o", markersize=5,
                label="ABM mean across seeds", zorder=3)
        ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1),
                        color=colour, alpha=0.15, label="± 1 SD across seeds")
        ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
                   label=f"Optimum ({step['symbol']} = {opt:.3f})", zorder=5)

        opt_y  = float(df.loc[df["value"].sub(opt).abs().idxmin(), col])
        opt_sd = (float(df.loc[df["value"].sub(opt).abs().idxmin(), std_col])
                  if std_col in df.columns else 0.)
        gap_pp = (opt_y - target) * 100

        # Stats box — bottom-right axes coords, monospaced for alignment
        stats_text = (
            f"ABM at optimum: {opt_y:.1%} (SD {opt_sd*100:.2f} pp)\n"
            f"BJS target:     {target:.1%}\n"
            f"Δ:              {gap_pp:+.2f} pp"
        )
        ax.text(0.98, 0.04, stats_text,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, family="monospace",
                color=colour, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=colour, alpha=0.95, linewidth=1.0))

        _pct(ax)
        ax.set_ylim(max(0.35, min(y.min(), base) - 0.05),
                    min(1.02, max(y.max(), target) + 0.10))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        _style(ax, step["label"], "Cumulative rearrest rate",
               f"Panel {panel} — {yrs}-Year Rearrest Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, f"sweep_{step['csv'].replace('.csv', '')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Sweep chart -> {path}")


def plot_tier_sweep(df, step, opt, outdir):
    """
    1×4 landscape sweep chart showing 3-year rate by PCRA tier.
    """
    if not all(f"rate_3yr_{t}" in df.columns for t in TIERS):
        return
    x = df["value"].values

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.8), sharey=False)
    fig.suptitle(
        f"PCRA Tier-Stratified Response to {step['label']}\n"
        f"Identified optimum: {step['symbol']} = {opt:.3f}",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.text(
        0.5, 0.945,
        "3-Year cumulative rearrest rate by PCRA risk tier  |  "
        "Source: Johnson (2023), Federal Probation 87(2), Table 6",
        ha="center", fontsize=9.5, color="#444444", style="italic",
    )

    panel_letters = ["A", "B", "C", "D"]
    for ax, tier, tl, letter in zip(axes, TIERS, TIERS_LB, panel_letters):
        col    = f"rate_3yr_{tier}"
        y      = df[col].values
        target = PCRA_TARGETS[tier][3]
        colour = C[tier]

        ax.plot(x, y, color=colour, linewidth=2.5, marker="o",
                markersize=6, label="Simulated rate", zorder=3)
        ax.axhline(target, color=C["bjs"], linewidth=2., linestyle="--",
                   label=f"PCRA target ({target:.1%})", zorder=4)
        ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
                   label=f"γ = {opt:.3f}", zorder=5)

        opt_y = float(df.loc[df["value"].sub(opt).abs().idxmin(), col])
        gap_pp = (opt_y - target) * 100

        stats_text = (f"Simulated: {opt_y:.1%}\n"
                      f"Target:    {target:.1%}\n"
                      f"Δ:         {gap_pp:+.2f} pp")
        ax.text(0.98, 0.04, stats_text,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, family="monospace",
                color=C["optimal"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=C["optimal"], alpha=0.95, linewidth=1.0))

        _pct(ax)
        y_hi = max(y.max(), target) + 0.12
        y_lo = min(y.min(), target) - 0.05
        ax.set_ylim(max(0, y_lo), min(1.02, y_hi))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        _style(ax, f"{step['symbol']}  ({step['label']})",
               "3-Year rearrest rate",
               f"Panel {letter} — {tl}")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(outdir, "sweep_gamma_tiers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Tier sweep chart -> {path}")


# =============================================================================
# STAGE 3 — per-offense sweep chart
# =============================================================================
def plot_offense_sweep(df, step, opt, outdir):
    """
    For a Stage 3 sweep, plots all four offense rates at 3yr against their
    BJS targets. The offense being swept shows the strongest response;
    others should stay roughly flat.
    """
    col_test = any(f"off_{off}_3yr" in df.columns for off in OFFENSES)
    if not col_test:
        return

    x = df["value"].values
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.2), sharey=False)

    offense_name = step["param"].split(".", 1)[1] if "." in step["param"] else None
    target_lb = (OFFENSES_LB[OFFENSES.index(offense_name)]
                 if offense_name in OFFENSES else offense_name)

    fig.suptitle(
        f"Stage 3 Per-Offense Response: {step['label']}\n"
        f"Sweep of {step['symbol']} targeting {target_lb}  |  "
        f"Identified optimum: {step['symbol']} = {opt:.3f}",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.text(
        0.5, 0.945,
        "3-Year cumulative rearrest rate by offense category  |  "
        "Target offense highlighted in bold; others plotted as controls  |  "
        "Source: Alper et al. (2018), BJS NCJ 250975 Table 7",
        ha="center", fontsize=9, color="#444444", style="italic",
    )

    panel_letters = ["A", "B", "C", "D"]
    for ax, off, off_lb, letter in zip(axes, OFFENSES, OFFENSES_LB, panel_letters):
        col = f"off_{off}_3yr"
        if col not in df.columns:
            continue
        y      = df[col].values
        target = BJS_OFFENSE_TARGETS[off][3]
        colour = C[off]
        is_target_offense = (off == offense_name)

        ax.plot(x, y, color=colour,
                linewidth=2.8 if is_target_offense else 1.6,
                marker="o", markersize=7 if is_target_offense else 4,
                label="Simulated rate (swept)" if is_target_offense
                      else "Simulated rate (control)",
                zorder=3)
        ax.axhline(target, color=C["bjs"], linewidth=2., linestyle="--",
                   label=f"BJS target ({target:.1%})", zorder=4)
        ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
                   label=f"Optimum = {opt:.3f}", zorder=5)

        opt_y = float(df.loc[df["value"].sub(opt).abs().idxmin(), col])
        gap_pp = (opt_y - target) * 100

        stats_text = (f"Simulated: {opt_y:.1%}\n"
                      f"Target:    {target:.1%}\n"
                      f"Δ:         {gap_pp:+.2f} pp")
        ax.text(0.98, 0.04, stats_text,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, family="monospace",
                color=C["optimal"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C["optimal"], alpha=0.95, linewidth=1.0))

        _pct(ax)
        y_hi = max(y.max(), target) + 0.12
        y_lo = min(y.min(), target) - 0.05
        ax.set_ylim(max(0, y_lo), min(1.02, y_hi))
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
        title_suffix = "  (target)" if is_target_offense else "  (control)"
        _style(ax,
               f"{step['symbol']}  ({step['short']})",
               "3-Year rearrest rate",
               f"Panel {letter} — {off_lb}{title_suffix}",
               fs=10)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    base = step["csv"].replace(".csv", "")
    path = os.path.join(outdir, f"{base}_offenses.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Offense sweep chart -> {path}")


# =============================================================================
# CALIBRATION SUMMARY CHARTS
# =============================================================================
def plot_final_summary(cal_params, baseline, calibrated, outdir):
    """
    Two-panel calibration summary: bar comparison (Panel A) and full
    parameter table with stage colour coding (Panel B).
    """
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]

    baseline   = {int(k): v for k, v in baseline.items()
                  if str(k).strip().isdigit()}
    calibrated = {int(k): v for k, v in calibrated.items()
                  if str(k).strip().isdigit()}

    FIXED_PARAMS = {
        "Risk_Effect_Decay_After_1Y",
        "Risk_Effect_Decay_After_3Y",
        "Risk_Effect_Decay_After_6Y",
    }

    fig = plt.figure(figsize=(20, 9))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.9], wspace=0.28,
                             left=0.05, right=0.97, top=0.87, bottom=0.08)

    # ── Panel A — bar comparison ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(3); w = 0.24
    b_vals = [baseline.get(yr, 0)   for yr in windows]
    c_vals = [calibrated.get(yr, 0) for yr in windows]
    t_vals = [targets[yr]           for yr in windows]

    b1 = ax1.bar(x-w, b_vals, w, color=C["baseline"],   edgecolor="white",
                 label="Uncalibrated baseline")
    b2 = ax1.bar(x,   c_vals, w, color=C["calibrated"], edgecolor="white",
                 label="Calibrated model")
    b3 = ax1.bar(x+w, t_vals, w, color=C["bjs"],        edgecolor="white",
                 label="BJS empirical target")
    for bars, vals, clr in [(b1, b_vals, "#555"),
                             (b2, c_vals, C["calibrated"]),
                             (b3, t_vals, C["bjs"])]:
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.012,
                     f"{val:.1%}", ha="center", va="bottom",
                     fontsize=9, color=clr, fontweight="bold")

    # MAE summary box
    mae_lines = []
    for yr, cv, tv in zip(windows, c_vals, t_vals):
        mae_lines.append(f"{yr}-year MAE:  {abs(cv-tv):.4f}")
    ax1.text(0.98, 0.97, "\n".join(mae_lines),
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=8.5, family="monospace", color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.95, linewidth=1.0))

    ax1.set_xticks(x); ax1.set_xticklabels(WIN_LB, fontsize=11)
    _pct(ax1); ax1.set_ylim(0, 1.10)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.95)
    _style(ax1, "Follow-up window", "Cumulative rearrest rate",
           "Panel A — Baseline vs. Calibrated vs. BJS Target")

    # ── Panel B — parameter table ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")

    COL_LABELS = ["Parameter", "Symbol", "Value", "Stage"]
    COL_WIDTHS = [0.36,         0.10,     0.11,    0.09]
    N_COLS     = len(COL_LABELS)

    rows, cell_colors = [], []
    for s in CALIBRATION_STEPS:
        val      = _get_param(cal_params, s["param"]) or 0.0
        is_fixed = s["param"] in FIXED_PARAMS
        stage_label = s["stage"] + ("*" if is_fixed else "")
        param_label = s["short"] + (" [fixed]" if is_fixed else "")

        rows.append([param_label, s["symbol"], f"{val:.4f}", stage_label])

        if   is_fixed:                   clr = "#FFF8E1"
        elif s["stage"].startswith("3"): clr = "#F0EAF8"
        elif "2" in s["stage"]:          clr = "#EAF4FB"
        else:                            clr = "#F8F8F8"
        cell_colors.append([clr] * N_COLS)

    tbl = ax2.table(
        cellText=rows, colLabels=COL_LABELS,
        cellColours=cell_colors, colColours=["#1A3D5C"] * N_COLS,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        if c < N_COLS:
            cell.set_width(COL_WIDTHS[c])
        if r == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")

    ax2.set_title(
        "Panel B — Calibrated Parameter Values\n"
        "Stage 1 BJS (white)  ·  Stage 2 PCRA (blue)  ·  "
        "Stage 3 Offense (purple)  ·  BJS-anchored fixed (yellow)",
        fontsize=11, fontweight="bold", pad=14,
    )

    # ── Figure title + subtitle ────────────────────────────────────────────
    fig.suptitle(
        "Recidivism ABM — Three-Stage OAT Calibration Summary",
        fontsize=14, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.925,
        f"Calibrated against BJS NCJ 250975 (aggregate + offense) "
        f"and PCRA Table 6 (tier-stratified)  |  "
        f"{CONFIG['N_REPS'] * len(CONFIG['SEEDS'])} simulation runs per sweep point",
        ha="center", fontsize=10, color="#444444", style="italic",
    )

    path = os.path.join(outdir, "FINAL_calibration_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  FINAL summary -> {path}")


def plot_three_way(baseline, baseline_std, calibrated, calibrated_std, outdir):
    """
    Four-panel three-way validation: baseline vs calibrated vs BJS target.
      Panel A: bar comparison
      Panel B: uncertainty across seeds (error bars)
      Panel C: MAE summary table
      Panel D: MAE improvement lollipop
    """
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    b_vals = [baseline.get(w, 0.) for w in windows]
    c_vals = [calibrated.get(w, 0.) for w in windows]
    t_vals = [targets[w] for w in windows]
    b_errs = [baseline_std.get(w, 0.) for w in windows]
    c_errs = [calibrated_std.get(w, 0.) for w in windows]
    b_mae  = [abs(b - t) for b, t in zip(b_vals, t_vals)]
    c_mae  = [abs(c - t) for c, t in zip(c_vals, t_vals)]
    imp    = [((bm - cm) / bm * 100) if bm > 0 else 0.
              for bm, cm in zip(b_mae, c_mae)]

    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.32,
                            left=0.07, right=0.97, top=0.90, bottom=0.07)

    # ── Panel A — bar comparison ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(3); w = 0.24
    bars_specs = [
        (b_vals, C["baseline"], "Uncalibrated baseline", "#555"),
        (c_vals, C["calibrated"], "Calibrated model", C["calibrated"]),
        (t_vals, C["bjs"], "BJS empirical target", C["bjs"]),
    ]
    offsets = [-w, 0, w]
    for (vals, bar_clr, lbl, text_clr), off in zip(bars_specs, offsets):
        bars = ax1.bar(x + off, vals, w, color=bar_clr,
                       edgecolor="white", linewidth=0.8, label=lbl)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.012,
                     f"{val:.1%}", ha="center", va="bottom",
                     fontsize=8.5, color=text_clr, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(WIN_LB, fontsize=10)
    _pct(ax1); ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.92)
    _style(ax1, "Follow-up window", "Cumulative rearrest rate",
           "Panel A — Rate Comparison")

    # ── Panel B — uncertainty ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    xp = np.arange(3)
    ax2.scatter(xp + 0.18, t_vals, marker="D", s=90, color=C["bjs"],
                zorder=5, label="BJS empirical target")
    ax2.errorbar(xp, c_vals, yerr=c_errs, fmt="o", markersize=9,
                 color=C["calibrated"], ecolor=C["calibrated"],
                 elinewidth=2, capsize=5, capthick=2, zorder=4,
                 label=f"Calibrated model (±1 SD, n = {len(CONFIG['SEEDS'])})")
    ax2.errorbar(xp - 0.18, b_vals, yerr=b_errs, fmt="s", markersize=8,
                 color=C["baseline"], ecolor="#999", elinewidth=1.8,
                 capsize=5, capthick=2, zorder=3,
                 label=f"Uncalibrated baseline (±1 SD, n = {len(CONFIG['SEEDS'])})")
    for xi, bv, cv in zip(xp, b_vals, c_vals):
        ax2.annotate("", xy=(xi, cv), xytext=(xi - 0.18, bv),
                     arrowprops=dict(arrowstyle="-|>", color="#BBBBBB",
                                     lw=1.4, mutation_scale=10))
    ax2.set_xticks(xp); ax2.set_xticklabels(WIN_LB, fontsize=10)
    _pct(ax2); ax2.set_ylim(0, 1.12)
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    _style(ax2, "Follow-up window", "Cumulative rearrest rate",
           "Panel B — Uncertainty Across Seeds")

    # ── Panel C — MAE table ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0]); ax3.axis("off")
    tbl_data, tbl_cols = [], []
    for i, (xl, bm, cm, pi) in enumerate(zip(WIN_LB, b_mae, c_mae, imp)):
        flag = ("within 2pp" if cm < 0.02 else
                ("within 5pp" if cm < 0.05 else "exceeds 5pp"))
        tbl_data.append([
            xl, f"{bm:.4f}", f"{cm:.4f} ({flag})", f"{pi:+.1f}%",
            f"{b_vals[i]:.1%}", f"{c_vals[i]:.1%}", f"{t_vals[i]:.1%}",
        ])
        mae_c = ("#D6EAD6" if cm < 0.02 else
                 ("#FFF3CD" if cm < 0.05 else "#FADBD8"))
        imp_c = ("#D6EAD6" if pi > 20 else
                 ("#FFF3CD" if pi > 0 else "#FADBD8"))
        tbl_cols.append(["#F0F0F0", "#F8F8F8", mae_c, imp_c,
                         "#F8F8F8", "#EAF4FB", "#EAF4FB"])
    tbl = ax3.table(
        cellText=tbl_data,
        colLabels=["Window", "Baseline MAE", "Calibrated MAE",
                   "MAE Improvement", "Baseline Rate",
                   "Calibrated Rate", "BJS Target"],
        cellColours=tbl_cols, colColours=["#1A3D5C"] * 7,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1., 2.1)
    for j in range(7):
        tbl[0, j].get_text().set_color("white")
        tbl[0, j].get_text().set_fontweight("bold")
    ax3.set_title("Panel C — MAE Summary Table", fontsize=11,
                  fontweight="bold", pad=10, loc="left")

    # ── Panel D — MAE improvement lollipop ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    clrs = ["#27AE60" if p > 20 else ("#F39C12" if p > 0 else "#E74C3C")
            for p in imp]
    yp = np.arange(3)
    ax4.barh(yp, imp, height=0.08, color=[c + "55" for c in clrs], zorder=2)
    ax4.scatter(imp, yp, s=180, color=clrs, zorder=4,
                edgecolors="white", linewidths=1.5)
    ax4.axvline(0, color="#BBBBBB", linewidth=1.2, zorder=1)
    for yi, pi, clr in zip(yp, imp, clrs):
        pad = 1.5 if pi >= 0 else -1.5
        ha  = "left" if pi >= 0 else "right"
        ax4.text(pi + pad, yi, f"{pi:+.1f}%",
                 va="center", ha=ha,
                 fontsize=10, color=clr, fontweight="bold")
    ax4.set_yticks(yp); ax4.set_yticklabels(WIN_LB, fontsize=10)
    span = max(25, max(abs(min(imp)), abs(max(imp))) + 15)
    ax4.set_xlim(-span, span)
    _style(ax4, "MAE reduction (%, positive = improvement)",
           "", "Panel D — MAE Improvement: Baseline → Calibrated")

    # ── Figure title + subtitle ────────────────────────────────────────────
    fig.suptitle(
        "Recidivism ABM — Three-Way Validation:\n"
        "Baseline vs. Calibrated vs. BJS Empirical Targets",
        fontsize=14, fontweight="bold", y=0.97, color="#1A1A1A",
    )
    fig.text(
        0.5, 0.928,
        f"BJS targets: 3yr = {targets[3]:.1%}  |  6yr = {targets[6]:.1%}  |  "
        f"9yr = {targets[9]:.1%}  |  "
        f"{len(CONFIG['SEEDS'])} seeds × {CONFIG['N_REPS']} reps  |  "
        f"Source: Alper et al. (2018), BJS NCJ 250975",
        ha="center", fontsize=10, color="#444444", style="italic",
    )

    path = os.path.join(outdir, "THREE_WAY_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  THREE_WAY comparison -> {path}")


# =============================================================================
# STAGE 3 — per-offense validation chart
# =============================================================================
def plot_offense_validation(baseline_res, final_res, outdir):
    """
    Four-panel validation chart — one panel per offense — showing
    baseline vs calibrated vs BJS target at 3/6/9 years.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    fig.suptitle(
        "Stage 3 Validation — Per-Offense Rearrest Rates:\n"
        "Baseline vs. Calibrated vs. BJS Empirical Target",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.945,
        "Cumulative rearrest rate at 3-, 6-, and 9-year follow-up windows  |  "
        "Source: Alper, Durose & Markman (2018), BJS NCJ 250975, Table 7",
        ha="center", fontsize=10, color="#444444", style="italic",
    )

    panel_letters = ["A", "B", "C", "D"]
    for ax, off, off_lb, letter in zip(axes.flat, OFFENSES,
                                         OFFENSES_LB, panel_letters):
        x = np.arange(3); w = 0.25
        b_vals = [baseline_res.get(f"off_{off}_{y}yr", 0.) for y in WINDOWS]
        c_vals = [final_res.get(f"off_{off}_{y}yr", 0.)   for y in WINDOWS]
        t_vals = [BJS_OFFENSE_TARGETS[off][y] for y in WINDOWS]

        ax.bar(x - w, b_vals, w, color=C["baseline"], edgecolor="white",
               label="Uncalibrated baseline", alpha=0.85)
        ax.bar(x, c_vals, w, color=C[off], edgecolor="white",
               label="Calibrated", alpha=0.92)
        ax.bar(x + w, t_vals, w, color=C["bjs"], edgecolor="white",
               label="BJS target", alpha=0.88)

        for xi, bv, cv, tv in zip(x, b_vals, c_vals, t_vals):
            gap = cv - tv
            y_label = max(bv, cv, tv) + 0.025
            ax.text(xi, y_label, f"Δ {gap*100:+.1f}pp",
                    ha="center", fontsize=9,
                    color=_gap_color(gap), fontweight="bold")

        mae_cal = np.mean([abs(c - t) for c, t in zip(c_vals, t_vals)])
        mae_base = np.mean([abs(b - t) for b, t in zip(b_vals, t_vals)])
        reduction = ((mae_base - mae_cal) / mae_base * 100
                     if mae_base > 0 else 0)
        ax.text(0.98, 0.04,
                f"Mean |Δ| across windows:\n"
                f"Baseline:   {mae_base*100:.2f} pp\n"
                f"Calibrated: {mae_cal*100:.2f} pp\n"
                f"Reduction:  {reduction:+.1f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, family="monospace", color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.93, linewidth=1.0))

        ax.set_xticks(x); ax.set_xticklabels(WIN_LB, fontsize=10)
        _pct(ax); ax.set_ylim(0, 1.12)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
        _style(ax, "Follow-up window", "Cumulative rearrest rate",
               f"Panel {letter} — {off_lb}")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(outdir, "STAGE3_offense_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Stage 3 offense validation -> {path}")


def plot_seed_distribution(final_res, outdir, per_seed_csv=None):
    """
    Seed-level boxplot distribution per follow-up window.
    """
    targets = CONFIG["bjs_targets"]
    df_seeds = None
    if per_seed_csv and os.path.exists(per_seed_csv):
        df_seeds = pd.read_csv(per_seed_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    fig.suptitle(
        "Calibrated Model — Per-Seed Distribution of Rearrest Rates\n"
        "Box and Whisker Summary Across Independent Simulation Runs",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.text(
        0.5, 0.93,
        f"Boxplot of {len(CONFIG['SEEDS']) * CONFIG['N_REPS']} simulation runs  |  "
        "Box spans IQR (25th–75th percentile)  |  "
        "Whiskers extend to 1.5 × IQR  |  "
        "BJS empirical target shown as dashed navy line",
        ha="center", fontsize=9, color="#444444", style="italic",
    )

    panel_letters = ["A", "B", "C"]
    for ax, yrs, colour, letter in zip(axes, [3, 6, 9],
                                         [C["3yr"], C["6yr"], C["9yr"]],
                                         panel_letters):
        data = np.array(final_res.get(f"all_{yrs}", []))
        target = targets[yrs]

        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            _style(ax, "", "Cumulative rearrest rate",
                   f"Panel {letter} — {yrs}-Year Rearrest Rate")
            continue

        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        widths=0.6,
                        boxprops=dict(facecolor=colour, alpha=0.35,
                                      edgecolor=colour, linewidth=1.5),
                        medianprops=dict(color="#111", linewidth=2.5),
                        whiskerprops=dict(color=colour, linewidth=1.2),
                        capprops=dict(color=colour, linewidth=1.2),
                        flierprops=dict(marker="o", markerfacecolor=colour,
                                         markeredgecolor="white",
                                         markersize=5, alpha=0.7))
        ax.axhline(target, color=C["bjs"], linewidth=2.0, linestyle="--",
                   label=f"BJS target ({target:.1%})", zorder=2)
        ax.axhline(data.mean(), color="#222222", linewidth=1.4, linestyle=":",
                   label=f"Sample mean ({data.mean():.1%})", zorder=2)

        # Stats summary box
        gap_pp = (data.mean() - target) * 100
        stats_text = (f"n = {len(data)}\n"
                      f"Mean = {data.mean():.1%}\n"
                      f"SD   = {data.std(ddof=1)*100:.2f} pp\n"
                      f"Δ    = {gap_pp:+.2f} pp")
        ax.text(0.03, 0.97, stats_text,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, family="monospace", color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.93, linewidth=1.0))

        _pct(ax)
        ax.set_xticks([])
        ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
        _style(ax, "", "Cumulative rearrest rate",
               f"Panel {letter} — {yrs}-Year Rearrest Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    path = os.path.join(outdir, "seed_stability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed stability -> {path}")


def plot_seed_mcse(final_res, outdir):
    """
    MCSE error-bar chart showing 95% CI of the mean against BJS target.
    """
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    colours = [C["3yr"], C["6yr"], C["9yr"]]

    means, cihalf, ns = [], [], []
    for y in windows:
        arr = np.array(final_res.get(f"all_{y}", []), dtype=float)
        n = len(arr)
        if n < 2:
            means.append(float(final_res.get(y, 0.)))
            cihalf.append(0.); ns.append(n); continue
        means.append(arr.mean())
        cihalf.append(1.96 * arr.std(ddof=1) / np.sqrt(n))
        ns.append(n)
    t_vals = [targets[y] for y in windows]

    fig, ax = plt.subplots(figsize=(13, 7))

    x = np.arange(3)
    ax.scatter(x + 0.12, t_vals, marker="D", s=130, color=C["bjs"],
               zorder=5, label="BJS empirical target", edgecolors="white",
               linewidths=1.2)
    for xi, mu, ci, colour in zip(x - 0.12, means, cihalf, colours):
        ax.errorbar(xi, mu, yerr=ci, fmt="o", markersize=14,
                    color=colour, ecolor=colour,
                    elinewidth=3, capsize=8, capthick=2.5, zorder=6,
                    markeredgecolor="white", markeredgewidth=1.2)

    # One legend handle representing all three calibrated windows
    ax.errorbar([], [], yerr=0.01, fmt="o", markersize=11,
                color=C["calibrated"], ecolor=C["calibrated"],
                elinewidth=3, capsize=8, capthick=2.5,
                label=f"Calibrated model (±95% CI, n = {ns[0]} seeds)")

    # Metrics table, top-right axes coords, legend goes bottom-left
    lines = []
    for y, mu, ci, tv in zip(windows, means, cihalf, t_vals):
        gap_pp = (mu - tv) * 100
        lines.append(
            f"{y}yr:  mean = {mu*100:5.1f}%   "
            f"target = {tv*100:5.1f}%   "
            f"Δ = {gap_pp:+5.2f} pp   "
            f"95% CI = ±{ci*100:.2f} pp"
        )
    ax.text(0.98, 0.97, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.95, linewidth=1.0))

    ax.set_xticks(x); ax.set_xticklabels(WIN_LB, fontsize=11)
    _pct(ax)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.93)
    _style(ax, "Follow-up window", "Cumulative rearrest rate", "")

    fig.suptitle(
        "Seed-Averaged Precision vs. BJS Empirical Target\n"
        "Calibrated Model: 95% Confidence Interval of the Mean",
        fontsize=13, fontweight="bold", y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(outdir, "seed_mcse_errorbar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed MCSE error-bar -> {path}")


def plot_seed_convergence(final_res, outdir):
    """
    Dissertation-level seed convergence chart showing the 95% CI of the running
    mean shrinking as seeds are added, against the BJS empirical target.
    """
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    colours = [C["3yr"], C["6yr"], C["9yr"]]
    panel_letters = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4))

    fig.suptitle(
        "Calibrated Model — Convergence of the Mean Estimate Across Independent Simulation Seeds",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.text(
        0.5, 0.93,
        "Cumulative rearrest rate at 3, 6, and 9 years post-release  |  "
        "Ribbon = 95% confidence interval of the running mean across seeds  |  "
        "Narrows as the mean is estimated more precisely with more seeds |  "
        "Dashed line = BJS NCJ 250975 empirical target  |  "
        "Dotted line = calibrated model mean at n = N",
        ha="center", fontsize=9, color="#444444", style="italic", wrap=True,
    )

    for ax, y, colour, letter in zip(axes, windows, colours, panel_letters):
        arr = np.array(final_res.get(f"all_{y}", []), dtype=float)
        N = len(arr)

        if N < 2:
            ax.text(0.5, 0.5, "Insufficient data (need ≥ 2 seeds)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#666666")
            _style(ax,
                   xlabel="Number of seeds included in running mean (n)",
                   ylabel="Cumulative rearrest rate",
                   title=f"Panel {letter} — {y}-Year Rearrest Rate",
                   fs=11)
            continue

        full_mean = arr.mean()
        target    = targets[y]
        gap_pp    = (full_mean - target) * 100
        ns        = np.arange(1, N + 1)

        with np.errstate(invalid="ignore", divide="ignore"):
            running_se = np.array([
                arr[:k].std(ddof=1) / np.sqrt(k) if k >= 2 else np.nan
                for k in ns
            ])
        ci95 = 1.96 * running_se
        ci_final = ci95[-1] if not np.isnan(ci95[-1]) else 0.0
        ci_final_pp = ci_final * 100

        ax.fill_between(ns, full_mean - ci95, full_mean + ci95,
                        color=colour, alpha=0.28, zorder=2,
                        label="95% CI around running mean")
        ax.axhline(target, color=C["bjs"], linewidth=1.8, linestyle="--",
                   zorder=3,
                   label=f"BJS empirical target ({target:.1%})")
        ax.axhline(full_mean, color="#222222", linewidth=1.2, linestyle=":",
                   zorder=4,
                   label=f"Calibrated mean at n = {N} ({full_mean:.1%})")

        # Stats box — side selected to not collide with curves
        box_position = "upper right" if target < full_mean else "lower right"
        if box_position == "upper right":
            box_xy = (0.97, 0.97); box_va = "top"
        else:
            box_xy = (0.97, 0.03); box_va = "bottom"

        stats_text = (
            f"Mean    = {full_mean:.1%}\n"
            f"BJS     = {target:.1%}\n"
            f"Δ       = {gap_pp:+.1f} pp\n"
            f"95% CI  = ±{ci_final_pp:.2f} pp at n = {N}"
        )
        ax.text(box_xy[0], box_xy[1], stats_text,
                transform=ax.transAxes, ha="right", va=box_va,
                fontsize=8.5, fontweight="bold",
                color=colour, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=colour, alpha=0.95, linewidth=1.2))

        ribbon_lo = float(np.nanmin(full_mean - ci95))
        ribbon_hi = float(np.nanmax(full_mean + ci95))
        y_lo = min(target, full_mean, ribbon_lo) - 0.012
        y_hi = max(target, full_mean, ribbon_hi) + 0.012
        ax.set_xlim(1, N)
        ax.set_ylim(y_lo, y_hi)
        _pct(ax)

        legend_loc = ("lower right" if box_position == "upper right"
                      else "upper right")
        ax.legend(fontsize=8.5, loc=legend_loc, framealpha=0.95)

        _style(ax,
               xlabel="Number of seeds included in running mean (n)",
               ylabel="Cumulative rearrest rate",
               title=f"Panel {letter} — {y}-Year Rearrest Rate",
               fs=11)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(outdir, "seed_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed convergence -> {path}")

def plot_seed_strip_with_ci(final_res, outdir):
    """
    Two-layer uncertainty chart: per-run 95% Wilson CI (within-run binomial
    sampling uncertainty) + across-seed 95% CI of the mean (between-run
    stochastic variation), shown side-by-side at 3/6/9 years.

    Reads from final_res (the dict returned by run_replicated). Requires:
      - final_res[f"all_{y}"]                : per-run rates (length N)
      - final_res["n_elig_per_run"]          : per-run agent counts (length N)
      - final_res[f"counts_{y}_per_run"]     : per-run rearrest counts (length N)

    If counts/n_elig are unavailable, the chart is skipped cleanly.
    """
    targets       = CONFIG["bjs_targets"]
    windows       = [3, 6, 9]
    colours       = [C["3yr"], C["6yr"], C["9yr"]]
    panel_letters = ["A", "B", "C"]

    n_elig_arr = np.array(final_res.get("n_elig_per_run", []), dtype=float)
    if len(n_elig_arr) < 2 or not np.any(n_elig_arr > 0):
        print("  [skip] plot_seed_strip_with_ci: no per-run n_elig in final_res")
        return

    # Pre-compute per-panel data
    panel_data = {}
    for y in windows:
        rates = np.array(final_res.get(f"all_{y}", []), dtype=float)
        counts = np.array(final_res.get(f"counts_{y}_per_run", []), dtype=float)
        N = len(rates)
        if N < 2 or len(counts) != N or len(n_elig_arr) != N:
            print(f"  [skip] plot_seed_strip_with_ci: misaligned arrays at {y}yr")
            return

        ci_lo = np.empty(N)
        ci_hi = np.empty(N)
        for i in range(N):
            lo, hi = _wilson_ci(int(counts[i]), int(n_elig_arr[i]))
            ci_lo[i] = lo if lo is not None else np.nan
            ci_hi[i] = hi if hi is not None else np.nan

        mu = rates.mean()
        sd = rates.std(ddof=1)
        se = sd / np.sqrt(N)
        band_lo = mu - 1.96 * se
        band_hi = mu + 1.96 * se

        panel_data[y] = dict(
            rates=rates, ci_lo=ci_lo, ci_hi=ci_hi,
            mu=mu, sd=sd, se=se,
            band_lo=band_lo, band_hi=band_hi,
            target=targets[y], N=N,
        )

    y_lo_global = (min(min(np.nanmin(d["ci_lo"]), d["target"])
                       for d in panel_data.values()) - 0.012)
    y_hi_global = (max(max(np.nanmax(d["ci_hi"]), d["target"])
                       for d in panel_data.values()) + 0.012)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0))

    fig.suptitle(
        "Calibrated Recidivism ABM — Two-Layer Uncertainty View\n"
        "Per-Run 95% Binomial CI (within-run) + Across-Seed 95% CI of the Mean (between-run)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    mean_n = float(np.mean(n_elig_arr))
    fig.text(
        0.5, 0.915,
        f"Each point = one simulation run (N = {panel_data[3]['N']}).  "
        f"Vertical bar = within-run 95% Wilson CI on n≈{int(mean_n)} agents/run.",
        ha="center", fontsize=9, color="#444444", style="italic",
    )
    fig.text(
        0.5, 0.890,
        "Solid line = across-seed sample mean.  Grey band = across-seed 95% CI of the mean.  "
        "Dashed navy line = BJS NCJ 250975 empirical target.",
        ha="center", fontsize=9, color="#444444", style="italic",
    )

    for ax, y, colour, letter in zip(axes, windows, colours, panel_letters):
        d = panel_data[y]
        x = np.arange(1, d["N"] + 1)

        # Across-seed CI band + reference lines
        ax.axhspan(d["band_lo"], d["band_hi"],
                   color="#888888", alpha=0.22, zorder=1)
        ax.axhline(d["mu"], color="#222222",
                   linewidth=1.4, linestyle="-", zorder=4)
        ax.axhline(d["target"], color=C["bjs"],
                   linewidth=1.6, linestyle="--", zorder=4)

        # Per-run error bars (within-run Wilson CI)
        yerr_lo = d["rates"] - d["ci_lo"]
        yerr_hi = d["ci_hi"] - d["rates"]
        ax.errorbar(
            x, d["rates"],
            yerr=[yerr_lo, yerr_hi],
            fmt="o", markersize=4.5,
            color=colour, ecolor=colour,
            elinewidth=0.9, capsize=2,
            alpha=0.75,
            markeredgecolor="white", markeredgewidth=0.4,
            zorder=5,
        )

        ax.set_xlim(0, d["N"] + 1)
        ax.set_ylim(y_lo_global, y_hi_global)
        _pct(ax)

        within_hw_pp  = float(np.nanmean(d["ci_hi"] - d["ci_lo"]) / 2 * 100)
        between_hw_pp = 1.96 * d["se"] * 100
        gap_pp        = (d["mu"] - d["target"]) * 100

        legend_handles = [
            Line2D([0], [0], color=colour, marker="o", linestyle="-",
                   markersize=5, linewidth=0.9,
                   label="Per-run rate ± 95% Wilson CI"),
            Patch(facecolor="#888888", alpha=0.40,
                  label=f"Across-seed 95% CI: ±{between_hw_pp:.2f} pp "
                        f"(n={d['N']} seeds)"),
            Line2D([0], [0], color="#222222", linewidth=1.4, linestyle="-",
                   label=f"Sample mean = {d['mu']:.1%}"),
            Line2D([0], [0], color=C["bjs"], linewidth=1.6, linestyle="--",
                   label=f"BJS target = {d['target']:.1%}  (Δ = {gap_pp:+.2f} pp)"),
            Line2D([0], [0], color="none", marker="", linestyle="",
                   label=f"Mean within-run CI half-width: ±{within_hw_pp:.2f} pp"),
        ]

        legend_loc = "upper right" if y == 3 else "lower right"
        ax.legend(handles=legend_handles, fontsize=8.2,
                  loc=legend_loc, framealpha=0.95,
                  handlelength=2.0, handletextpad=0.6)

        _style(ax,
               xlabel="Seed run index (1 … N)",
               ylabel="Cumulative rearrest rate per simulation run",
               title=f"Panel {letter} — {y}-Year Rearrest Rate",
               fs=11)

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    path = os.path.join(outdir, "seed_strip_with_within_run_ci.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed strip with within-run CI -> {path}")

def plot_seed_strip(final_res, outdir):
    """
    Dissertation-level per-seed strip plot showing the raw cumulative rearrest
    rate from each individual simulation run, against the BJS empirical target.

    Polished version with:
      - Single unified legend per panel (geometry + robustness stats merged)
      - Smart legend placement (Panel A upper-right, others lower-right)
        to avoid overlapping data on the shared y-axis
      - Two-line subtitle for readability
      - Fully-shared y-axis range across all three panels for fair
        cross-panel comparison
    """
    targets       = CONFIG["bjs_targets"]
    windows       = [3, 6, 9]
    cmaps         = ["Blues", "Greens", "Oranges"]
    accents       = [C["3yr"], C["6yr"], C["9yr"]]
    panel_letters = ["A", "B", "C"]

    # ─── Pre-pass: compute single shared y-axis range across all panels ──
    all_lo, all_hi = [], []
    for y in windows:
        arr = np.array(final_res.get(f"all_{y}", []), dtype=float)
        if len(arr) < 2:
            continue
        target = targets[y]
        all_lo.append(min(arr.min(), target))
        all_hi.append(max(arr.max(), target))

    if all_lo:
        y_lo = min(all_lo) - 0.010
        y_hi = max(all_hi) + 0.010
    else:
        y_lo, y_hi = 0.40, 0.85

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))

    fig.suptitle(
        "Per-Seed Cumulative Rearrest Rate from the Calibrated Model\n"
        "Raw Simulation Output Across Independent Random Seeds",
        fontsize=13, fontweight="bold", y=1.00,
    )

    fig.text(
        0.5, 0.945,
        "Each point = one full simulation run (144-month warmup + 108-month study).  "
        "Point colour intensity scales with |deviation from sample mean|.",
        ha="center", fontsize=9, color="#444444", style="italic",
    )
    fig.text(
        0.5, 0.918,
        "Solid line = sample mean.  Grey band = 95% CI of the mean.  "
        "Dashed line = BJS NCJ 250975 empirical target.",
        ha="center", fontsize=9, color="#444444", style="italic",
    )

    for ax, y, cmap_name, accent, letter in zip(
        axes, windows, cmaps, accents, panel_letters
    ):
        arr = np.array(final_res.get(f"all_{y}", []), dtype=float)
        N = len(arr)

        if N < 2:
            ax.text(0.5, 0.5, "Insufficient data (need ≥ 2 seeds)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#666666")
            ax.set_xlim(0, 1)
            ax.set_ylim(y_lo, y_hi)
            _pct(ax)
            _style(ax,
                   xlabel="Seed run index (1 … N)",
                   ylabel="Cumulative rearrest rate per simulation run",
                   title=f"Panel {letter} — {y}-Year Rearrest Rate",
                   fs=11)
            continue

        mu      = arr.mean()
        sd      = arr.std(ddof=1)
        mcse    = sd / np.sqrt(N)
        target  = targets[y]
        x       = np.arange(1, N + 1)
        abs_dev = np.abs(arr - mu)
        max_dev = max(abs_dev.max(), 1e-9)
        ci      = 1.96 * mcse
        ci_lo   = mu - ci
        ci_hi   = mu + ci

        seeds_above_bjs  = int((arr > target).sum())
        seeds_within_1pp = int((abs_dev <= 0.01).sum())

        # ── Reference geometry ────────────────────────────────────────────
        ax.axhspan(ci_lo, ci_hi, color="#888888", alpha=0.22, zorder=1)
        ax.axhline(mu,     color="#222222", linewidth=1.4, linestyle="-",  zorder=3)
        ax.axhline(target, color=C["bjs"],  linewidth=1.6, linestyle="--", zorder=3)

        # ── Per-seed dots, intensity scaled by |deviation| ───────────────
        norm        = mcolors.Normalize(vmin=0., vmax=max_dev)
        cmap        = plt.get_cmap(cmap_name)
        dot_colours = cmap(0.55 + 0.45 * norm(abs_dev))
        ax.scatter(x, arr, c=dot_colours, s=32,
                   edgecolors="white", linewidths=0.5, zorder=5)

        # ── Shared y-axis range across all panels ────────────────────────
        ax.set_xlim(0, N + 1)
        ax.set_ylim(y_lo, y_hi)
        _pct(ax)

        # ── Unified legend: geometry rows + spacer + robustness rows ─────
        empty = Line2D([0], [0], color="none", marker="", linestyle="")

        legend_handles = [
            Line2D([0], [0], color="#222222", linewidth=1.4, linestyle="-",
                   label=f"Sample mean = {mu:.1%}  (SD = {sd*100:.2f} pp)"),
            Patch(facecolor="#888888", alpha=0.40,
                  label=(f"95% CI of mean = [{ci_lo:.1%}, {ci_hi:.1%}]  "
                         f"(max |dev| = {max_dev*100:.2f} pp)")),
            Line2D([0], [0], color=C["bjs"], linewidth=1.6, linestyle="--",
                   label=f"BJS target = {target:.1%}"),
            empty,
            Line2D([0], [0], color="none", marker="", linestyle="",
                   label=f"Seeds above BJS: {seeds_above_bjs}/{N}"),
            Line2D([0], [0], color="none", marker="", linestyle="",
                   label=f"Seeds within ±1 pp of mean: {seeds_within_1pp}/{N}"),
        ]

        # Panel A's data clusters in the lower half; its legend goes
        # upper-right. Panels B and C have data in the middle, so
        # lower-right is clear.
        legend_loc = "upper right" if y == 3 else "lower right"
        ax.legend(handles=legend_handles, fontsize=8.5,
                  loc=legend_loc, framealpha=0.95,
                  handlelength=2.0, handletextpad=0.6)

        _style(ax,
               xlabel="Seed run index (1 … N)",
               ylabel="Cumulative rearrest rate per simulation run",
               title=f"Panel {letter} — {y}-Year Rearrest Rate",
               fs=11)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = os.path.join(outdir, "seed_strip.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed strip -> {path}")


# =============================================================================
# TIER VALIDATION CHARTS
# =============================================================================
def plot_tier_before_after(baseline_tier, calibrated_tier, cal_params, outdir):
    """
    Two-panel before/after showing 3yr rates by tier against PCRA target.
    """
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    fig.suptitle(
        "PCRA Risk-Tier-Stratified Rearrest Rates:\n"
        "Before vs. After Stage 2 Calibration",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.text(
        0.5, 0.93,
        f"3-Year cumulative rearrest rate by PCRA risk tier  |  "
        f"Calibrated γ = {gamma_val:.3f}  |  "
        f"Source: Johnson (2023), Federal Probation 87(2), Table 6",
        ha="center", fontsize=10, color="#444444", style="italic",
    )

    x = np.arange(4); w = 0.28
    panel_specs = [
        (axes[0], baseline_tier, "A", "Before Calibration (γ = 0.0)"),
        (axes[1], calibrated_tier, "B",
         f"After Calibration (γ = {gamma_val:.3f})"),
    ]

    for ax, data, letter, heading in panel_specs:
        sim = [data.get(k, 0.) for k in TIERS]
        tgt = [PCRA_TARGETS[k][3] for k in TIERS]
        clrs = [C[k] for k in TIERS]

        bars1 = ax.bar(x - w/2, sim, w, color=clrs, edgecolor="white",
                       linewidth=0.8, label="ABM simulation")
        bars2 = ax.bar(x + w/2, tgt, w, color=C["target"], alpha=0.75,
                       edgecolor="white", linewidth=0.8,
                       label="PCRA empirical target")

        # Value labels above bars
        for bar, val in zip(bars1, sim):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=8.5, color="#333333", fontweight="bold")
        for bar, val in zip(bars2, tgt):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=8.5, color=C["bjs"], fontweight="bold")

        # MAE summary
        mae = np.mean([abs(s - t) for s, t in zip(sim, tgt)])
        ax.text(0.98, 0.97, f"Mean |Δ|: {mae*100:.2f} pp",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace", color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.93, linewidth=1.0))

        ax.set_xticks(x); ax.set_xticklabels(TIERS_LB, fontsize=10)
        _pct(ax); ax.set_ylim(0, 1.18)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
        _style(ax, "PCRA risk tier", "3-Year rearrest rate",
               f"Panel {letter} — {heading}")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    path = os.path.join(outdir, "tier_chart1_before_after_3yr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 1 -> {path}")


def plot_tier_trajectories(baseline_tier, calibrated_tier, cal_params, outdir):
    """
    2×2 panels: per-tier rearrest trajectories across 3/6/9 years,
    comparing before/after calibration against PCRA target curve.
    """
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875
    bjs = CONFIG["bjs_targets"]
    r6 = bjs[6] / bjs[3]
    r9 = bjs[9] / bjs[3]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

    fig.suptitle(
        "Cumulative Rearrest Rate Trajectories by PCRA Risk Tier\n"
        "Baseline vs. Calibrated ABM vs. PCRA Empirical Target",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.955,
        f"Windows: 3, 6, 9 years post-release  |  "
        f"Calibrated γ = {gamma_val:.3f}  |  "
        f"Source: Johnson (2023), Federal Probation 87(2), Table 6",
        ha="center", fontsize=9.5, color="#444444", style="italic",
    )

    panel_letters = ["A", "B", "C", "D"]
    for ax, tk, tl, letter in zip(axes.flat, TIERS, TIERS_LB, panel_letters):
        clr = C[tk]
        x   = WINDOWS
        tgt = [PCRA_TARGETS[tk][w] for w in WINDOWS]
        b3  = baseline_tier.get(tk, 0.)
        c3  = calibrated_tier.get(tk, 0.)

        uc  = [b3, min(b3 * r6, 1.), min(b3 * r9, 1.)]
        cal = [c3, min(c3 * r6, 1.), min(c3 * r9, 1.)]

        ax.plot(x, tgt, color=C["target"], linewidth=2.5, marker="D",
                markersize=9, label="PCRA empirical target", zorder=4)
        ax.plot(x, uc, color=C["baseline"], linewidth=2.0, linestyle="--",
                marker="s", markersize=7,
                label="Uncalibrated baseline", zorder=3)
        ax.plot(x, cal, color=clr, linewidth=2.8, marker="o", markersize=8,
                label=f"Calibrated (γ = {gamma_val:.3f})", zorder=5)

        # Per-window Δ labels beside calibrated points
        for xi, cv, tv in zip(x, cal, tgt):
            gap_pp = (cv - tv) * 100
            ax.annotate(f"Δ {gap_pp:+.1f}pp",
                        xy=(xi, cv), xytext=(8, 10),
                        textcoords="offset points",
                        fontsize=8, fontweight="bold",
                        color=_gap_color(cv - tv))

        # Cumulative MAE summary
        mae = np.mean([abs(c - t) for c, t in zip(cal, tgt)])
        ax.text(0.03, 0.97,
                f"Mean |Δ| (3/6/9 yr):\n{mae*100:.2f} pp",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, family="monospace", color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.93, linewidth=1.0))

        _pct(ax)
        ax.set_xticks(WINDOWS)
        ax.set_xticklabels(WIN_LB, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
        _style(ax, "Follow-up window", "Cumulative rearrest rate",
               f"Panel {letter} — {tl}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "tier_chart2_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 2 -> {path}")


def plot_tier_gap_heatmap(baseline_tier, calibrated_tier, cal_params, outdir):
    """
    Two-panel heatmap showing simulated-minus-target gaps by tier × window.
    """
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875
    bjs = CONFIG["bjs_targets"]
    r6 = bjs[6] / bjs[3]
    r9 = bjs[9] / bjs[3]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    fig.suptitle(
        "Calibration Gap Heatmap: ABM Simulated − PCRA Empirical Target\n"
        "PCRA Risk Tier × Follow-up Window",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.text(
        0.5, 0.935,
        f"Cell value = ABM rate − PCRA target (in rate units)  |  "
        f"Green = ABM below target  ·  Red = ABM above target  |  "
        f"Calibrated γ = {gamma_val:.3f}",
        ha="center", fontsize=9.5, color="#444444", style="italic",
    )

    norm = mcolors.TwoSlopeNorm(vmin=-0.28, vcenter=0., vmax=0.28)

    panel_specs = [
        (axes[0], baseline_tier, "A", "Before Calibration (γ = 0.0)"),
        (axes[1], calibrated_tier, "B",
         f"After Calibration (γ = {gamma_val:.3f})"),
    ]

    for ax, tier_data, letter, heading in panel_specs:
        matrix = np.array([
            [tier_data.get(tk, 0.) - PCRA_TARGETS[tk][3],
             min(tier_data.get(tk, 0.) * r6, 1.) - PCRA_TARGETS[tk][6],
             min(tier_data.get(tk, 0.) * r9, 1.) - PCRA_TARGETS[tk][9]]
            for tk in TIERS
        ])
        im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")

        # Cell value annotations (in pp for readability)
        for i in range(4):
            for j in range(3):
                gap_pp = matrix[i, j] * 100
                text_clr = "white" if abs(matrix[i, j]) > 0.15 else "#222222"
                ax.text(j, i, f"{gap_pp:+.1f}",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_clr)

        ax.set_xticks(range(3)); ax.set_xticklabels(WIN_LB, fontsize=10)
        ax.set_yticks(range(4)); ax.set_yticklabels(TIERS_LB, fontsize=10)
        ax.set_title(f"Panel {letter} — {heading}",
                     fontsize=11, fontweight="bold", pad=9)

        # Mean absolute gap annotation
        mean_abs = np.mean(np.abs(matrix)) * 100
        ax.text(0.98, -0.22,
                f"Mean |Δ| across 12 cells: {mean_abs:.2f} pp",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace", color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.93, linewidth=1.0))

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Gap (rate units)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    path = os.path.join(outdir, "tier_chart3_gap_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 3 -> {path}")


def plot_tier_dashboard(baseline_rates, baseline_tier,
                         calibrated_rates, calibrated_tier,
                         cal_params, outdir):
    """
    Single-figure tier dashboard: aggregate rate + per-tier gap summary,
    laid out as a compact reference card.
    """
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875
    bjs = CONFIG["bjs_targets"]

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[1, 1.3],
                             hspace=0.35, wspace=0.28,
                             left=0.06, right=0.97, top=0.88, bottom=0.07)

    # ── Panel A — aggregate rate comparison ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(3); w = 0.26
    b_vals = [baseline_rates.get(y, 0.) for y in WINDOWS]
    c_vals = [calibrated_rates.get(y, 0.) for y in WINDOWS]
    t_vals = [bjs[y] for y in WINDOWS]

    bars_b = ax1.bar(x - w, b_vals, w, color=C["baseline"], edgecolor="white",
                     label="Baseline")
    bars_c = ax1.bar(x,     c_vals, w, color=C["calibrated"], edgecolor="white",
                     label="Calibrated")
    bars_t = ax1.bar(x + w, t_vals, w, color=C["bjs"], edgecolor="white",
                     label="BJS target")
    for bars, vals, clr in [(bars_b, b_vals, "#555"),
                             (bars_c, c_vals, C["calibrated"]),
                             (bars_t, t_vals, C["bjs"])]:
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.012,
                     f"{val:.1%}", ha="center", va="bottom",
                     fontsize=8, color=clr, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(WIN_LB, fontsize=10)
    _pct(ax1); ax1.set_ylim(0, 1.10)
    ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    _style(ax1, "Follow-up window", "Cumulative rearrest rate",
           "Panel A — Aggregate Rates")

    # ── Panel B — per-tier 3yr rates ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(4); w = 0.28
    tier_b = [baseline_tier.get(k, 0.) for k in TIERS]
    tier_c = [calibrated_tier.get(k, 0.) for k in TIERS]
    tier_t = [PCRA_TARGETS[k][3] for k in TIERS]

    ax2.bar(x - w, tier_b, w, color=C["baseline"], edgecolor="white",
            label="Baseline")
    ax2.bar(x,     tier_c, w, color=[C[k] for k in TIERS], edgecolor="white",
            label="Calibrated")
    ax2.bar(x + w, tier_t, w, color=C["bjs"], alpha=0.8, edgecolor="white",
            label="PCRA target")
    ax2.set_xticks(x); ax2.set_xticklabels(TIERS_LB, fontsize=10)
    _pct(ax2); ax2.set_ylim(0, 1.10)
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    _style(ax2, "PCRA risk tier", "3-Year rearrest rate",
           "Panel B — PCRA Tier Rates (3-Year)")

    # ── Panel C — summary metrics table ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :]); ax3.axis("off")

    # Build summary table
    agg_mae  = np.mean([abs(c - t) for c, t in zip(c_vals, t_vals)])
    tier_mae = np.mean([abs(c - t) for c, t in zip(tier_c, tier_t)])

    summary_rows = []
    for yr, cv, tv in zip(WINDOWS, c_vals, t_vals):
        flag = ("within 2pp" if abs(cv - tv) < 0.02 else
                ("within 5pp" if abs(cv - tv) < 0.05 else "exceeds 5pp"))
        summary_rows.append([
            f"Aggregate · {yr}-year",
            f"{cv:.1%}", f"{tv:.1%}",
            f"{(cv-tv)*100:+.2f} pp", flag,
        ])
    for tk, tl, cv, tv in zip(TIERS, TIERS_LB, tier_c, tier_t):
        flag = ("within 2pp" if abs(cv - tv) < 0.02 else
                ("within 5pp" if abs(cv - tv) < 0.05 else "exceeds 5pp"))
        summary_rows.append([
            f"PCRA · {tl} (3-year)",
            f"{cv:.1%}", f"{tv:.1%}",
            f"{(cv-tv)*100:+.2f} pp", flag,
        ])

    cell_colors = []
    for row in summary_rows:
        gap_pp = float(row[3].replace(" pp", ""))
        if abs(gap_pp) < 2:    gap_c = "#D6EAD6"
        elif abs(gap_pp) < 5:  gap_c = "#FFF3CD"
        else:                   gap_c = "#FADBD8"
        cell_colors.append(["#F0F0F0", "#F8F8F8", "#F8F8F8", gap_c, gap_c])

    tbl = ax3.table(
        cellText=summary_rows,
        colLabels=["Metric", "Calibrated", "Target", "Δ (pp)", "Status"],
        cellColours=cell_colors,
        colColours=["#1A3D5C"] * 5,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.7)
    for j in range(5):
        tbl[0, j].get_text().set_color("white")
        tbl[0, j].get_text().set_fontweight("bold")

    ax3.set_title(
        f"Panel C — Calibration Summary  |  "
        f"Aggregate MAE: {agg_mae*100:.2f} pp  |  "
        f"PCRA Tier MAE: {tier_mae*100:.2f} pp",
        fontsize=11, fontweight="bold", pad=10, loc="left",
    )

    # ── Figure title + subtitle ────────────────────────────────────────────
    fig.suptitle(
        "Stage 1 + Stage 2 Calibration Dashboard\n"
        "BJS Aggregate + PCRA Tier-Stratified Validation",
        fontsize=14, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.925,
        f"Calibrated γ = {gamma_val:.3f}  |  "
        f"{CONFIG['N_REPS'] * len(CONFIG['SEEDS'])} simulation runs  |  "
        f"Sources: Alper et al. (2018), BJS NCJ 250975  ·  "
        f"Johnson (2023), Federal Probation 87(2)",
        ha="center", fontsize=9.5, color="#444444", style="italic",
    )

    path = os.path.join(outdir, "tier_chart4_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 4 -> {path}")


PCRA_REF_PCT = [29.1, 35.6, 23.9, 9.6]
PCRA_REF_TOTAL = 475528

def plot_tier_composition(baseline_res, final_res, outdir):
    """
    Two-panel donut comparison of PCRA tier composition: ABM vs PCRA reference.
    """
    colours = [C["Low"], C["LowModerate"], C["Moderate"], C["High"]]

    abm_pct = [final_res.get(f"tier_share_{t}", 0.) * 100 for t in TIERS]
    if sum(abm_pct) < 0.5:
        abm_pct = [25.0, 25.0, 25.0, 25.0]
    total = sum(abm_pct)
    if total > 0:
        abm_pct = [v / total * 100 for v in abm_pct]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    fig.suptitle(
        "PCRA Risk Tier Composition:\n"
        "Calibrated ABM Study Population vs. Federal PCRA Reference Distribution",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.text(
        0.5, 0.925,
        f"ABM: {CONFIG['N_REPS'] * len(CONFIG['SEEDS'])} simulation runs  |  "
        f"PCRA reference: N = {PCRA_REF_TOTAL:,} federal offenders  |  "
        f"Source: Johnson (2023), Federal Probation 87(2)",
        ha="center", fontsize=10, color="#444444", style="italic",
    )

    for ax, pct, title_letter, title_text in [
        (axes[0], abm_pct,       "A", "Calibrated ABM Study Population"),
        (axes[1], PCRA_REF_PCT,  "B", "PCRA Federal Reference"),
    ]:
        wedges, texts, autotexts = ax.pie(
            pct,
            labels=TIERS_LB,
            colors=colours,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.78,
            labeldistance=1.08,
            wedgeprops=dict(edgecolor="white", linewidth=2, width=0.45),
            textprops=dict(fontsize=10),
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_color("white")
            at.set_fontsize(10)

        ax.set_title(f"Panel {title_letter} — {title_text}",
                     fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(outdir, "tier_composition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier composition -> {path}")

def plot_equifinality_single_panel(baseline_tier, calibrated_tier,
                                    cal_params, outdir,
                                    filename="equifinality.png"):
    """
    Single-panel grouped bar chart: Uncalibrated vs Calibrated vs PCRA target.
    One group of 3 bars per PCRA risk tier (Low, Low-Moderate, Moderate, High).

    Parameters
    ----------
    baseline_tier   : {tier_key: 3yr_rate}  — uncalibrated run
                      keys: 'Low', 'LowModerate', 'Moderate', 'High'
    calibrated_tier : {tier_key: 3yr_rate}  — final calibrated run
    cal_params      : locked parameter dict (used to read γ for the label)
    outdir          : output directory (must already exist)
    filename        : output filename inside outdir
    """
    # Uses the module-level C, TIERS, TIERS_LB, PCRA_TARGETS already defined
    # in OAT_Calibrate_BJS_PCRA.py — no private copies needed.

    gamma_val   = cal_params.get("Risk_Contrast_Strength", 1.0)
    gamma_label = f"γ = {gamma_val:.3f}"

    uncal_vals = [baseline_tier.get(t, 0.0)   for t in TIERS]
    cal_vals   = [calibrated_tier.get(t, 0.0) for t in TIERS]
    tgt_vals   = [PCRA_TARGETS[t][3]           for t in TIERS]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#FAFAFA")

    x       = np.arange(len(TIERS))
    w       = 0.24
    offsets = [-w, 0.0, w]

    bar_specs = [
        (offsets[0], uncal_vals, C["baseline"],   "Uncalibrated γ = 0.0"),
        (offsets[1], cal_vals,   C["calibrated"], f"Calibrated {gamma_label}"),
        #(offsets[2], tgt_vals,   C["bjs"],        "PCRA empirical target"),
        (offsets[2], tgt_vals,   C["bjs"],        "PCRA empirical target"),
        #(offsets[2], tgt_vals,   C["bjs"],        "_nolegend_"),
    ]

    for off, vals, colour, label in bar_specs:
        bars = ax.bar(x + off, vals, width=w,
                      color=colour, edgecolor="white", linewidth=0.8,
                      label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.010,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=colour)

    # MAE summary box (top-right)
    #mae_uncal  = float(np.mean([abs(u - t) for u, t in zip(uncal_vals, tgt_vals)]))
    #mae_cal    = float(np.mean([abs(c - t) for c, t in zip(cal_vals,   tgt_vals)]))
    #reduction  = ((mae_uncal - mae_cal) / mae_uncal * 100) if mae_uncal > 0 else 0.0
    #ax.text(0.98, 0.97,
    #        f"Mean |Δ| uncalibrated : {mae_uncal * 100:.2f} pp\n"
    #        f"Mean |Δ| calibrated   : {mae_cal   * 100:.2f} pp\n"
    #        f"MAE reduction         : {reduction:+.1f}%",
    #        transform=ax.transAxes, ha="right", va="top",
    #        fontsize=9, family="monospace", color="#333333",
    #        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
    #                  edgecolor="#CCCCCC", alpha=0.95, linewidth=1.0))

    # Per-tier Δ label (calibrated vs target)
    for i, (cv, tv) in enumerate(zip(cal_vals, tgt_vals)):
        gap_pp = (cv - tv) * 100
        clr    = "#27AE60" if abs(gap_pp) <= 2 else (
                 "#F39C12" if abs(gap_pp) <= 5 else "#E74C3C")
        ax.text(x[i], max(cv, tv) + 0.052,
                f"Δ {gap_pp:+.1f} pp",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=clr)

    ax.set_xticks(x)
    ax.set_xticklabels(TIERS_LB, fontsize=11)
    ax.set_xlabel("PCRA risk tier", fontsize=11, labelpad=6)
    ax.set_ylabel("3-Year cumulative rearrest rate (%)", fontsize=11, labelpad=6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.14)

    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
            linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    ax.legend(loc="upper left", fontsize=10,
              frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    ax.set_title( 
                "Before vs. After Stage 2 Calibration:\n"
                "PCRA Risk-Tier-Stratified Rearrest Rates" ,
                 fontsize=13, fontweight="bold", pad=14)
    #fig.text(0.5, 0.01,
    #         f"3-Year cumulative rearrest rate by PCRA risk tier  |  "
    #         f"Calibrated {gamma_label}  |  "
    #         f"Source: Johnson (2023), Federal Probation 87(2), Table 6",
    #         ha="center", va="bottom", fontsize=9,
    #         color="#555555", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(outdir, filename)
    #plt.savefig(path, dpi=150, bbox_inches="tight",
    #            facecolor=fig.get_facecolor())
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close()
    print(f"  Equifinality single-panel chart -> {path}")

def plot_equifinality_three_windows(baseline_tier, calibrated_tier,
                                     cal_params, outdir,
                                     filename="equifinality_3windows.png"):
    """
    Three-panel grouped bar chart: Uncalibrated vs Calibrated vs PCRA target
    across 3-year, 6-year, and 9-year windows.

    Parameters
    ----------
    baseline_tier   : {tier_key: {3: rate, 6: rate, 9: rate}}
                      Uncalibrated run results per tier per window.
                      Also accepts {tier_key: scalar} for 3yr-only fallback.
    calibrated_tier : {tier_key: {3: rate, 6: rate, 9: rate}}
                      Final calibrated run results per tier per window.
    cal_params      : dict
                      Locked parameter dict — used to read γ for the label.
    outdir          : str
                      Output directory (must already exist).
    filename        : str
                      Output filename inside outdir.

    Dissertation use
    ----------------
    Extends plot_equifinality_single_panel() to confirm that γ identifies
    tier spread at all three follow-up windows (3/6/9yr), not just 3yr.
    Source: Johnson (2023), Federal Probation 87(2), Table 6.
    """
    gamma_val   = cal_params.get("Risk_Contrast_Strength", 1.0)
    gamma_label = f"γ = {gamma_val:.3f}"
    windows     = [3, 6, 9]
    window_lbls = ["3-Year", "6-Year", "9-Year"]

    # ── Helper: extract rate for a tier and window ────────────────────────────
    def _get(data, tier, yr):
        val = data.get(tier, {})
        if isinstance(val, dict):
            return float(val.get(yr, 0.0))
        # scalar fallback — only valid for 3yr
        return float(val) if yr == 3 else 0.0

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        wspace=0.06,
        left=0.06, right=0.97,
        top=0.82, bottom=0.14,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    x       = np.arange(len(TIERS))
    w       = 0.24
    offsets = [-w, 0.0, w]

    legend_handles = []
    legend_labels  = []

    for col, (ax, yr, wlbl) in enumerate(zip(axes, windows, window_lbls)):

        uncal_vals = [_get(baseline_tier,   t, yr) for t in TIERS]
        cal_vals   = [_get(calibrated_tier, t, yr) for t in TIERS]
        tgt_vals   = [PCRA_TARGETS[t][yr]          for t in TIERS]

        bar_specs = [
            (offsets[0], uncal_vals, C["baseline"],   "Uncalibrated  γ = 0.0"),
            (offsets[1], cal_vals,   C["calibrated"], f"Calibrated  {gamma_label}"),
            (offsets[2], tgt_vals,   C["bjs"],        "PCRA empirical target"),
        ]

        for off, vals, colour, label in bar_specs:
            bars = ax.bar(
                x + off, vals,
                width=w,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                label=label,
                zorder=3,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.013,
                    f"{val:.1%}",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold",
                    color=colour,
                )

        # Per-tier Δ annotation (calibrated vs PCRA target)
        for i, (cv, tv) in enumerate(zip(cal_vals, tgt_vals)):
            gap_pp = (cv - tv) * 100
            clr = (
                "#27AE60" if abs(gap_pp) <= 2 else
                "#F39C12" if abs(gap_pp) <= 5 else
                "#E74C3C"
            )
            ax.text(
                x[i] + offsets[1],
                max(cv, tv) + 0.072,
                f"Δ {gap_pp:+.1f} pp",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color=clr,
            )

        # MAE summary box — top-right of each panel
        mae_cal  = float(np.mean([abs(c - t) for c, t in zip(cal_vals,   tgt_vals)]))
        mae_base = float(np.mean([abs(u - t) for u, t in zip(uncal_vals, tgt_vals)]))
        ax.text(
            0.98, 0.97,
            f"Calibrated MAE: {mae_cal  * 100:.2f} pp\n"
            f"Baseline MAE:   {mae_base * 100:.2f} pp",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.93, linewidth=1.0),
        )

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels(TIERS_LB, fontsize=9.5, rotation=15, ha="right")
        ax.set_xlim(-0.55, len(TIERS) - 0.45)
        ax.set_ylim(0, 1.24)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="y", color=C["grid"],
                linewidth=0.6, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(wlbl, fontsize=12, fontweight="bold", pad=8)

        if col == 0:
            ax.set_ylabel(
                "Cumulative rearrest rate (%)",
                fontsize=10, labelpad=6,
            )
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    # ── Shared figure title ───────────────────────────────────────────────────
    fig.suptitle(
        "Before vs. After Stage 2 Calibration:\n"
        "PCRA Risk-Tier-Stratified Rearrest Rates  |  "
        "3-Year, 6-Year, 9-Year Windows",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Shared legend ─────────────────────────────────────────────────────────
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True, framealpha=0.92,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, 0.01),
    )

    # ── Delta colour key ──────────────────────────────────────────────────────
    #fig.text(
    #    0.97, 0.02,
    #    "Δ colour:  ■ ≤2pp (green)   ■ ≤5pp (amber)   ■ >5pp (red)",
    #    ha="right", va="bottom",
    #    fontsize=8, color="#555555", style="italic",
    #)

    # ── Source footnote ───────────────────────────────────────────────────────
    #fig.text(
    #    0.03, 0.02,
    #    f"Source: Johnson (2023), Federal Probation 87(2), Table 6  |  "
    #    f"Calibrated {gamma_label}  |  BJS NCJ 250975",
    #    ha="left", va="bottom",
    #    fontsize=8, color="#555555", style="italic",
    #)

    path = os.path.join(outdir, filename)
    plt.savefig(
        path, dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    plt.close()
    print(f"  Equifinality three-window chart -> {path}")
    return path


def plot_threeway_rate_comparison(baseline_rates, calibrated_rates, outdir,
                                   filename="CalibrationSummary.png"):
    """
    Single-panel grouped bar chart: Uncalibrated vs Calibrated vs BJS target.
    One group of 3 bars per follow-up window (3-Year, 6-Year, 9-Year).

    Parameters
    ----------
    baseline_rates   : {3: rate, 6: rate, 9: rate}  — uncalibrated run
    calibrated_rates : {3: rate, 6: rate, 9: rate}  — final calibrated run
    outdir           : output directory — pass `outdir` from main()
    filename         : output filename inside outdir
    """
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]

    b_vals = [baseline_rates.get(w, 0.)   for w in windows]
    c_vals = [calibrated_rates.get(w, 0.) for w in windows]
    t_vals = [targets[w]                  for w in windows]

    mae_uncal = float(np.mean([abs(b - t) for b, t in zip(b_vals, t_vals)]))
    mae_cal   = float(np.mean([abs(c - t) for c, t in zip(c_vals, t_vals)]))
    reduction = ((mae_uncal - mae_cal) / mae_uncal * 100) if mae_uncal > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    #fig.patch.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x       = np.arange(len(windows))
    w       = 0.24
    offsets = [-w, 0.0, w]

    bar_specs = [
        (offsets[0], b_vals, C["baseline"],   "Uncalibrated baseline"),
        (offsets[1], c_vals, C["calibrated"], "Calibrated model"),
        (offsets[2], t_vals, C["bjs"],        "BJS empirical target"),
    ]

    for off, vals, colour, label in bar_specs:
        bars = ax.bar(x + off, vals, width=w,
                      color=colour, edgecolor="white", linewidth=0.8,
                      label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.1%}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=colour)

    # MAE annotation box — bottom right, clear of bars
    #ax.text(0.98, 0.97,
    #        f"Mean |Δ| uncalibrated : {mae_uncal * 100:.2f} pp\n"
    #        f"Mean |Δ| calibrated   : {mae_cal   * 100:.2f} pp\n"
    #        f"MAE reduction         : {reduction:+.1f}%",
    #        transform=ax.transAxes, ha="right", va="top",
    #        fontsize=9, family="monospace", color="#333333",
    #        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
    #                  edgecolor="#CCCCCC", alpha=0.95, linewidth=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(WIN_LB, fontsize=11)
    ax.set_xlabel("Follow-up window", fontsize=11, labelpad=6)
    ax.set_ylabel("Cumulative rearrest rate", fontsize=11, labelpad=6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
            linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    # Legend placed outside above the chart, centred, horizontal
    ax.legend(
        loc="upper left",
        fontsize=10,
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
    )

    ax.set_title(
        "Three-Stage Calibration Across 3, 6, and 9 Year Follow-Up Windows",
        fontsize=13, fontweight="bold", pad=14,
    )
    #fig.text(
    #    0.5, 0.01,
    #    f"BJS targets: 3yr = {targets[3]:.1%}  |  6yr = {targets[6]:.1%}  |  "
    #    f"9yr = {targets[9]:.1%}  |  "
    #    "Source: Alper et al. (2018), BJS NCJ 250975",
    #    ha="center", va="bottom", fontsize=9,
    #    color="#555555", style="italic",
    #)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close()
    print(f"  Three-way rate comparison -> {path}")

def plot_all_tier_charts(baseline_res, calibrated_res, calibrated_rates,
                          cal_params, outdir):
    print("\n  Generating dissertation tier calibration charts...")
    baseline_tier   = {t: baseline_res.get(f"tier_3yr_{t}", 0.) for t in TIERS}
    calibrated_tier = {t: calibrated_res.get(f"tier_3yr_{t}", 0.) for t in TIERS}

    plot_tier_before_after(baseline_tier, calibrated_tier, cal_params, outdir)
    plot_tier_trajectories(baseline_tier, calibrated_tier, cal_params, outdir)
    plot_tier_gap_heatmap(baseline_tier, calibrated_tier, cal_params, outdir)
    plot_tier_dashboard(
        baseline_rates={3: baseline_res.get(3, 0.),
                        6: baseline_res.get(6, 0.),
                        9: baseline_res.get(9, 0.)},
        baseline_tier=baseline_tier,
        calibrated_rates=calibrated_rates,
        calibrated_tier=calibrated_tier,
        cal_params=cal_params, outdir=outdir,
    )
    plot_tier_composition(baseline_res, calibrated_res, outdir)
    
    print("  All tier charts saved.")


# =============================================================================
# MAIN
# STAGE 3 — locked dict handles nested offense_hazard_shift keys
# =============================================================================
def main(n_workers, force_rerun=False):
    outdir = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "tmp"), exist_ok=True)

    targets      = CONFIG["bjs_targets"]
    runs_per_val = CONFIG["N_REPS"] * len(CONFIG["SEEDS"])

    print("="*68)
    print("  RECIDIVISM ABM -- OAT CALIBRATION  (11 parameters, 3 fixed)")
    print("="*68)
    print(f"  Workers  : {n_workers}")
    print(f"  Agents   : {CONFIG['initial_agents']}")
    print(f"  Warmup / Study : {CONFIG['warmup_months']} / {CONFIG['study_months']} months")
    print(f"  Runs/value : {CONFIG['N_REPS']} reps x {len(CONFIG['SEEDS'])} seeds = {runs_per_val}")
    print(f"  BJS targets : 3yr={targets[3]:.1%}  6yr={targets[6]:.1%}  9yr={targets[9]:.1%}")
    print(f"  Output : {outdir}")
    print(f"  Order  : Stage 2 gamma -> Stage 1 (alpha, ds3, ds6) -> Stage 3 (ov, od, op, oo)")
    print("="*68)

    bj = os.path.join(outdir, "baseline.json")
    if not force_rerun and os.path.exists(bj):
        print("\n  Step 0: Loading baseline from checkpoint...")
        with open(bj) as f: baseline_res = _norm(json.load(f))
    else:
        print(f"\n  Step 0: Baseline ({runs_per_val} runs)...")
        baseline_res = _norm(run_replicated(get_uncalibrated_params(), n_workers, label="baseline"))
        with open(bj, "w") as f: json.dump(baseline_res, f, indent=2)

    baseline_rates      = {3: baseline_res[3], 6: baseline_res[6], 9: baseline_res[9]}
    baseline_diagnostic = {1: baseline_res.get(1, 0.)}
    print(f"  Baseline: 3yr={baseline_rates[3]:.1%}  "
          f"6yr={baseline_rates[6]:.1%}  9yr={baseline_rates[9]:.1%}")

    # STAGE 3 — build locked from uncalibrated params with nested support
    uncalibrated = get_uncalibrated_params()
    locked = {}
    for s in CALIBRATION_STEPS:
        param = s["param"]
        if "." in param:
            outer, inner = param.split(".", 1)
            if outer in uncalibrated and isinstance(uncalibrated[outer], dict):
                locked.setdefault(outer, {})
                locked[outer][inner] = uncalibrated[outer].get(inner, 0.0)
        elif param in uncalibrated:
            locked[param] = uncalibrated[param]

    calibrated = {}

    for idx, step in enumerate(CALIBRATION_STEPS, 1):
        param    = step["param"]
        csv_path = os.path.join(outdir, step["csv"])

        print(f"\n{'--'*34}")
        print(f"  Step {idx}/{len(CALIBRATION_STEPS)}: {step['label']}")
        locked_str_parts = []
        for s in CALIBRATION_STEPS[:idx-1]:
            if s["param"] in calibrated:
                locked_str_parts.append(f"{s['symbol']}={calibrated[s['param']]:.4f}")
        if locked_str_parts:
            print(f"  Locked: {'  '.join(locked_str_parts)}")
        print(f"{'--'*34}")

        if len(step["values"]) == 1:
            opt = float(step["values"][0])
            print(f"  {step['symbol']} = {opt:.4f}  [fixed -- not swept]")
            calibrated[param] = opt
            _apply_param(locked, param, opt)
            continue

        if not force_rerun and os.path.exists(csv_path):
            print(f"  Checkpoint -- loading {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            df = sweep_parallel(step, locked, n_workers)
            df.to_csv(csv_path, index=False)

        opt  = _select_optimal(df, step)
        best = df.loc[df["value"].sub(opt).abs().idxmin()]
        if step["loss"] == "gamma":
            loss_col = "gamma_loss"
        elif step["loss"] == "offense":
            loss_col = "offense_loss"
        else:
            loss_col = "mae_primary"
        loss_val = float(best.get(loss_col, best.get("mae_all", 0.)))
        print(f"  {step['symbol']} = {opt:.4f}  (loss={loss_val:.4f})")
        print(f"  3yr={best['rate_3yr']:.1%}  6yr={best['rate_6yr']:.1%}  9yr={best['rate_9yr']:.1%}")

        calibrated[param] = opt
        _apply_param(locked, param, opt)

        plot_sweep(df, step, opt, baseline_rates, outdir)
        if step["loss"] == "gamma":
            plot_tier_sweep(df, step, opt, outdir)
        if step["loss"] == "offense":
            plot_offense_sweep(df, step, opt, outdir)

    print(f"\n{'='*68}")
    print(f"  Final validation ({runs_per_val} runs)...")
    final_res = _norm(run_replicated(locked, n_workers, label="final"))
    cal_rates      = {3: final_res[3], 6: final_res[6], 9: final_res[9]}
    cal_diagnostic = {1: final_res.get(1, 0.)}
    print(f"  Calibrated: 3yr={cal_rates[3]:.1%}  "
          f"6yr={cal_rates[6]:.1%}  9yr={cal_rates[9]:.1%}")

    # STAGE 3 — per-offense summary
    print("\n  Per-offense rates (calibrated):")
    print(f"  {'Offense':22s}  {'3yr':>8s}  {'6yr':>8s}  {'9yr':>8s}")
    for off in OFFENSES:
        r3 = final_res.get(f"off_{off}_3yr", 0.)
        r6 = final_res.get(f"off_{off}_6yr", 0.)
        r9 = final_res.get(f"off_{off}_9yr", 0.)
        t3 = BJS_OFFENSE_TARGETS[off][3]
        print(f"  {off:22s}  {r3:.1%} (Δ{(r3-t3)*100:+.1f}pp)  "
              f"{r6:.1%}  {r9:.1%}")

    plot_final_summary(locked, baseline_rates, cal_rates, outdir)
    plot_three_way(
        baseline_rates,
        {3: baseline_res.get("std_3yr",0.), 6: baseline_res.get("std_6yr",0.),
         9: baseline_res.get("std_9yr",0.)},
        cal_rates,
        {3: final_res.get("std_3yr",0.), 6: final_res.get("std_6yr",0.),
         9: final_res.get("std_9yr",0.)},
        outdir,
    )
    plot_threeway_rate_comparison(
        baseline_rates   = baseline_rates,
        calibrated_rates = cal_rates,
        outdir           = outdir,
    )
    plot_equifinality_single_panel(                                   
        baseline_tier   = {t: baseline_res.get(f"tier_3yr_{t}", 0.) for t in TIERS},
        calibrated_tier = {t: final_res.get(f"tier_3yr_{t}",   0.) for t in TIERS},
        cal_params      = locked,
        outdir          = outdir,
    )
    plot_equifinality_three_windows(
        baseline_tier={
            t: {
                3: baseline_res.get(f"tier_3yr_{t}", 0.),
                6: baseline_res.get(f"tier_6yr_{t}",
                   baseline_res.get(f"tier_3yr_{t}", 0.)),
                9: baseline_res.get(f"tier_9yr_{t}",
                   baseline_res.get(f"tier_3yr_{t}", 0.)),
            }
            for t in TIERS
        },
        calibrated_tier={
            t: {
                3: final_res.get(f"tier_3yr_{t}", 0.),
                6: final_res.get(f"tier_6yr_{t}",
                   final_res.get(f"tier_3yr_{t}", 0.)),
                9: final_res.get(f"tier_9yr_{t}",
                   final_res.get(f"tier_3yr_{t}", 0.)),
            }
            for t in TIERS
        },
        cal_params=locked,
        outdir=outdir,
    )
    # STAGE 3 — per-offense validation chart
    plot_offense_validation(baseline_res, final_res, outdir)

    plot_seed_mcse(final_res, outdir)
    plot_seed_convergence(final_res, outdir)
    plot_seed_strip(final_res, outdir)
    plot_seed_strip_with_ci(final_res, outdir)
    plot_all_tier_charts(baseline_res, final_res, cal_rates, locked, outdir)

    diag_mae_1yr = abs(cal_diagnostic[1] - BJS_DIAGNOSTIC[1])

    recommended = {
        "calibrated_params": locked,
        "baseline_rates":    baseline_rates,
        "baseline_std":      {3: baseline_res.get("std_3yr",0.),
                               6: baseline_res.get("std_6yr",0.),
                               9: baseline_res.get("std_9yr",0.)},
        "baseline_tier":     {t: baseline_res.get(f"tier_3yr_{t}", 0.)  for t in TIERS},
        "baseline_share":    {t: baseline_res.get(f"tier_share_{t}", 0.) for t in TIERS},
        "calibrated_rates":  cal_rates,
        "calibrated_std":    {3: final_res.get("std_3yr",0.),
                               6: final_res.get("std_6yr",0.),
                               9: final_res.get("std_9yr",0.)},
        "final_validation":  {f"all_{y}": final_res.get(f"all_{y}", [])
                               for y in [3, 6, 9]},
        "calibrated_tier":   {t: final_res.get(f"tier_3yr_{t}", 0.)    for t in TIERS},
        "calibrated_share":  {t: final_res.get(f"tier_share_{t}", 0.)  for t in TIERS},
        # STAGE 3 — save per-offense baseline and calibrated rates
        "baseline_offense": {
            off: {y: baseline_res.get(f"off_{off}_{y}yr", 0.) for y in WINDOWS}
            for off in OFFENSES
        },
        "calibrated_offense": {
            off: {y: final_res.get(f"off_{off}_{y}yr", 0.) for y in WINDOWS}
            for off in OFFENSES
        },
        "bjs_offense_targets": BJS_OFFENSE_TARGETS,
        "offense_loss_final": final_res.get("offense_loss", 0.),
        "bjs_targets":       targets,
        "pcra_targets":      PCRA_TARGETS,
        "seeds_used":        CONFIG["SEEDS"],
        "n_workers":         n_workers,
        "final_mae":         {str(y): abs(cal_rates[y]-targets[y])
                               for y in [3, 6, 9]},
        "diagnostic": {
            "bjs_reference_1yr": BJS_DIAGNOSTIC[1],
            "baseline_rate_1yr": baseline_diagnostic[1],
            "calibrated_rate_1yr": cal_diagnostic[1],
            "calibrated_mae_1yr": diag_mae_1yr,
        },
        "config":            CONFIG,
    }
    jp = os.path.join(outdir, "recommended_params.json")
    with open(jp, "w") as f: json.dump(recommended, f, indent=2)

    print(f"\n{'='*68}")
    print("  CALIBRATION COMPLETE")
    print(f"{'='*68}")
    for s in CALIBRATION_STEPS:
        fixed_tag = " [fixed]" if len(s["values"])==1 else ""
        val = _get_param(locked, s["param"]) or 0.0
        print(f"  {s['param']:<50}  {val:.4f}  [{s['stage']}]{fixed_tag}")
    print(f"{'--'*34}")
    for yrs in [3, 6, 9]:
        err  = abs(cal_rates[yrs]-targets[yrs])
        flag = "OK" if err<0.02 else ("OK (5pp)" if err<0.05 else "EXCEEDS 5pp")
        print(f"  {yrs}-year: ABM={cal_rates[yrs]:.1%}  BJS={targets[yrs]:.1%}  "
              f"MAE={err:.4f}  {flag}")
    print(f"  1-year: ABM={cal_diagnostic[1]:.1%}  BJS={BJS_DIAGNOSTIC[1]:.1%}  "
          f"MAE={diag_mae_1yr:.4f}  [diagnostic]")

    # STAGE 3 — final offense report
    print(f"{'--'*34}")
    print(f"  STAGE 3 — PER-OFFENSE VALIDATION")
    for off in OFFENSES:
        r3 = final_res.get(f"off_{off}_3yr", 0.)
        t3 = BJS_OFFENSE_TARGETS[off][3]
        err = abs(r3 - t3)
        flag = "OK" if err<0.02 else ("OK (5pp)" if err<0.05 else "EXCEEDS 5pp")
        print(f"  {off:22s}  ABM 3yr={r3:.1%}  target={t3:.1%}  "
              f"MAE={err:.4f}  {flag}")

    print(f"{'='*68}")
    print(f"  Recommended params -> {jp}")
    return recommended


# =============================================================================
# REPLOT
# =============================================================================
def replot(outdir=None):
    if outdir is None: outdir = CONFIG["output_directory"]
    jp = os.path.join(outdir, "recommended_params.json")
    if not os.path.exists(jp):
        print("No recommended_params.json -- run calibration first."); return
    with open(jp) as f: rec = json.load(f)
    cal = rec["calibrated_params"]
    b_r = {int(k):v for k,v in rec["baseline_rates"].items()}
    c_r = {int(k):v for k,v in rec["calibrated_rates"].items()}
    fv  = rec.get("final_validation", {})

    for step in CALIBRATION_STEPS:
        if len(step["values"])==1: continue
        csv_path = os.path.join(outdir, step["csv"])
        if not os.path.exists(csv_path):
            print(f"  Skipping {step['symbol']} -- CSV not found"); continue
        df  = pd.read_csv(csv_path)
        opt_v = _get_param(cal, step["param"])
        opt   = opt_v if opt_v is not None else _select_optimal(df, step)
        plot_sweep(df, step, opt, b_r, outdir)
        if step["loss"]=="gamma":   plot_tier_sweep(df, step, opt, outdir)
        if step["loss"]=="offense": plot_offense_sweep(df, step, opt, outdir)

    plot_final_summary(cal, b_r, c_r, outdir)
    b_std = {int(k):v for k,v in rec.get("baseline_std",{}).items()}
    c_std = {int(k):v for k,v in rec.get("calibrated_std",{}).items()}
    for w in [3,6,9]: b_std.setdefault(w,0.); c_std.setdefault(w,0.)
    plot_three_way(b_r, b_std, c_r, c_std, outdir)

    # ── NEW: two clean summary charts ────────────────────────────────────────
    plot_threeway_rate_comparison(          # takes only baseline_rates, cal_rates, outdir
        baseline_rates   = b_r,
        calibrated_rates = c_r,
        outdir           = outdir,
    )

    # STAGE 3 — replot offense validation if data present
    if "calibrated_offense" in rec:
        base_off = rec.get("baseline_offense", {})
        cal_off  = rec["calibrated_offense"]
        base_res = {f"off_{off}_{y}yr": base_off.get(off, {}).get(str(y), 0.)
                    for off in OFFENSES for y in WINDOWS}
        cal_res  = {f"off_{off}_{y}yr": cal_off.get(off, {}).get(str(y), 0.)
                    for off in OFFENSES for y in WINDOWS}
        plot_offense_validation(base_res, cal_res, outdir)

    bj_path = os.path.join(outdir, "baseline.json")
    if os.path.exists(bj_path):
        with open(bj_path) as f: bres = json.load(f)
        b_tier = {t: bres.get(f"tier_3yr_{t}", 0.) for t in TIERS}
    else:
        b_tier = {t: rec.get("baseline_tier",{}).get(t,0.) for t in TIERS}

    c_tier = {t: rec.get("calibrated_tier",{}).get(t, 0.) for t in TIERS}

    if any(v>0 for v in c_tier.values()):
        plot_tier_before_after(b_tier, c_tier, cal, outdir)
        plot_tier_trajectories(b_tier, c_tier, cal, outdir)
        plot_tier_gap_heatmap(b_tier, c_tier, cal, outdir)
        plot_tier_dashboard(b_r, b_tier, c_r, c_tier, cal, outdir)
        plot_equifinality_single_panel(     # ── NEW ──
            baseline_tier   = b_tier,
            calibrated_tier = c_tier,
            cal_params      = cal,
            outdir          = outdir,
        )
        plot_equifinality_three_windows(
            baseline_tier={
                t: {
                    3: b_tier.get(t, 0.),
                    6: rec.get("baseline_tier_6yr", {}).get(t,
                       b_tier.get(t, 0.)),
                    9: rec.get("baseline_tier_9yr", {}).get(t,
                       b_tier.get(t, 0.)),
                }
                for t in TIERS
            },
            calibrated_tier={
                t: {
                    3: c_tier.get(t, 0.),
                    6: rec.get("calibrated_tier_6yr", {}).get(t,
                       c_tier.get(t, 0.)),
                    9: rec.get("calibrated_tier_9yr", {}).get(t,
                       c_tier.get(t, 0.)),
                }
                for t in TIERS
            },
            cal_params=cal,
            outdir=outdir,
        )

    b_share_fixed = {f"tier_share_{t}": rec.get("baseline_share", {}).get(t, 0.) for t in TIERS}
    c_share_fixed = {f"tier_share_{t}": rec.get("calibrated_share", {}).get(t, 0.) for t in TIERS}
    plot_tier_composition(b_share_fixed, c_share_fixed, outdir)

    if fv:
        plot_seed_mcse(fv, outdir)
        plot_seed_convergence(fv, outdir)
        plot_seed_strip(fv, outdir)
        plot_seed_strip_with_ci(fv, outdir)     # ── NEW: was missing from replot ──

    print("All charts regenerated.")


# =============================================================================
# CORE DETECTION & ENTRY POINT
# =============================================================================
def detect_workers():
    if _PSUTIL:
        ph = psutil.cpu_count(logical=False)
        lg = psutil.cpu_count(logical=True)
        if ph and ph > 0:
            n = max(1, ph-1)
            return n, f"psutil: {ph} physical -> {n} workers"
        n = max(1, (lg or 2)//2-1)
        return n, f"psutil: {lg} logical -> {n} workers"
    lg = multiprocessing.cpu_count()
    n  = max(1, lg//2-1)
    return n, f"{lg} logical -> {n} workers"


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = sys.argv[1:]
    if "--replot" in args:
        replot(); sys.exit(0)
    force = "--rerun" in args
    if "--cores" in args:
        idx = args.index("--cores")
        try:    n_workers = int(args[idx+1])
        except: print("Usage: [--cores N] [--rerun] [--replot]"); sys.exit(1)
        print(f"  Manual: {n_workers} workers")
    else:
        n_workers, reason = detect_workers()
        print(f"  Core detection: {reason}")
    print(f"  Starting with {n_workers} workers...\n")
    main(n_workers=n_workers, force_rerun=force)