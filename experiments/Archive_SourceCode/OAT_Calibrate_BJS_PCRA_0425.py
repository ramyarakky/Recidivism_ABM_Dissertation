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
    "initial_agents":        1000,
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
        }

    except Exception as e:
        print(f"  Worker error (value={param_value:.3f}, seed={seed}): {e}", flush=True)
        return {
            "param_value": param_value,
            "seed":        seed,
            "rates":       {1: 0.0, 3: 0.0, 6: 0.0, 9: 0.0},
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
           "share_Moderate": [], "share_High": []}
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

                row = {
                    "label":        label,
                    "rep":          rep,
                    "seed":         seed,
                    "actual_seed":  seed + rep * 10_000,
                    "rate_1yr":     r["rates"][1],
                    "rate_3yr":     r["rates"][3],
                    "rate_6yr":     r["rates"][6],
                    "rate_9yr":     r["rates"][9],
                    "gamma_loss":   r["gamma_loss"] if r["gamma_loss"] < 998. else None,
                    "offense_loss": r["offense_loss"] if r["offense_loss"] < 998. else None,
                }
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
    return out


def _norm(res):
    out = {}
    for k, v in res.items():
        try: out[int(k)] = v
        except: out[k] = v
    return out


# =============================================================================
# SWEEP CHARTS
# =============================================================================
def plot_sweep(df, step, opt, baseline, outdir):
    is_gamma   = step["loss"] == "gamma"
    is_offense = step["loss"] == "offense"
    x = df["value"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if is_gamma:
        stage_label = "Stage 2: PCRA tier loss"
    elif is_offense:
        stage_label = "Stage 3: BJS per-offense loss"
    else:
        stage_label = "Stage 1: BJS aggregate loss"

    fig.suptitle(
        f"OAT Sweep -- {step['label']}\n"
        f"Optimal: {step['symbol']} = {opt:.3f}   ({stage_label})",
        fontsize=13, fontweight="bold", y=0.99
    )

    ax = axes[0, 0]
    if is_gamma:
        loss_col = "gamma_loss"
    elif is_offense:
        loss_col = "offense_loss"
    else:
        loss_col = "mae_primary"
    y = df[loss_col].values
    ax.plot(x, y, color=C["calibrated"], linewidth=2., marker="o", markersize=5,
            label="Mean loss across seeds", zorder=3)
    ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
               label=f"Optimal = {opt:.3f}", zorder=5)
    opt_y = float(df.loc[df["value"].sub(opt).abs().idxmin(), loss_col])
    ax.annotate(f"{opt_y:.4f}", xy=(opt, opt_y), xytext=(8, -18),
                textcoords="offset points", fontsize=9, color=C["optimal"],
                arrowprops=dict(arrowstyle="->", color=C["optimal"], lw=1.2))
    ax.legend(fontsize=8, framealpha=0.9)
    if is_gamma:
        ylab = "PCRA Tier MAE (4 tiers x 3 windows)"
    elif is_offense:
        ylab = "BJS Offense MAE (4 offenses x 3 windows)"
    else:
        ylab = "BJS Aggregate MAE"
    _style(ax, step["label"], ylab, "Panel A -- Primary Loss Function")

    for ax, (yrs, col, std_col, colour, panel) in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        [(3,"rate_3yr","std_3yr",C["3yr"],"B"),
         (6,"rate_6yr","std_6yr",C["6yr"],"C"),
         (9,"rate_9yr","std_9yr",C["9yr"],"D")]
    ):
        y    = df[col].values
        sd   = df[std_col].values if std_col in df.columns else np.zeros_like(y)
        base = baseline.get(yrs, 0.)
        _bjs_band(ax, yrs)
        ax.axhline(base, color=C["baseline"], linewidth=1.5, linestyle=":",
                   label=f"Uncalibrated baseline ({base:.1%})", zorder=2)
        ax.plot(x, y, color=colour, linewidth=2., marker="o", markersize=5,
                label="ABM mean across seeds", zorder=3)
        ax.fill_between(x, np.clip(y-sd,0,1), np.clip(y+sd,0,1),
                        color=colour, alpha=0.15, label="+/- 1 SD across seeds")
        ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
                   label=f"Optimal {step['symbol']} = {opt:.3f}", zorder=5)
        opt_y  = float(df.loc[df["value"].sub(opt).abs().idxmin(), col])
        opt_sd = float(df.loc[df["value"].sub(opt).abs().idxmin(), std_col]) if std_col in df.columns else 0.
        ax.annotate(f"{opt_y:.1%} +/- {opt_sd:.3f}",
                    xy=(opt, opt_y), xytext=(8, 12),
                    textcoords="offset points", fontsize=9,
                    color=C["optimal"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["optimal"], lw=1.2))
        _pct(ax)
        ax.set_ylim(max(0.35, min(y.min(), base)-0.05),
                    min(1.02, max(y.max(), CONFIG["bjs_targets"][yrs])+0.06))
        ax.legend(fontsize=8, framealpha=0.9, loc="best")
        _style(ax, step["label"], "Cumulative Rearrest Rate",
               f"Panel {panel} -- {yrs}-Year Rearrest Rate")

    plt.tight_layout(rect=[0,0,1,0.96])
    path = os.path.join(outdir, f"sweep_{step['csv'].replace('.csv','')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Sweep chart -> {path}")


def plot_tier_sweep(df, step, opt, outdir):
    if not all(f"rate_3yr_{t}" in df.columns for t in TIERS):
        return
    x = df["value"].values
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle(
        f"PCRA Tier Rates vs. Targets -- {step['label']}\n"
        f"Optimal: {step['symbol']} = {opt:.3f}  |  "
        "3-Year Cumulative Rearrest Rate by PCRA Risk Tier  |  "
        "Target: Johnson (2023), Federal Probation 87(2)",
        fontsize=12, fontweight="bold", y=1.02
    )
    for ax, tier, tl in zip(axes, TIERS, TIERS_LB):
        col    = f"rate_3yr_{tier}"
        y      = df[col].values
        target = PCRA_TARGETS[tier][3]
        colour = C[tier]
        ax.plot(x, y, color=colour, linewidth=2.5, marker="o",
                markersize=6, label="Simulated rate", zorder=3)
        ax.axhline(target, color=C["bjs"], linewidth=2., linestyle="--",
                   label=f"PCRA target ({target:.1%})", zorder=4)
        ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
                   label=f"Optimal gamma = {opt:.3f}", zorder=5)
        opt_y = float(df.loc[df["value"].sub(opt).abs().idxmin(), col])
        gap   = opt_y - target
        ax.annotate(
            f"Simulated: {opt_y:.1%}\nTarget: {target:.1%}\nGap: {gap:+.1%}",
            xy=(opt, opt_y), xytext=(10, -32),
            textcoords="offset points", fontsize=8.5,
            color=C["optimal"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=C["optimal"], alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=C["optimal"], lw=1.2))
        _pct(ax)
        ax.legend(fontsize=8, framealpha=0.9, loc="best")
        _style(ax, "gamma (Risk Contrast Strength)", "3-Year Rearrest Rate", tl)
    plt.tight_layout()
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
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

    # Which offense is this step targeting?
    offense_name = step["param"].split(".", 1)[1] if "." in step["param"] else None
    subtitle = (f"Sweep of {step['symbol']} targeting "
                f"{offense_name}  |  Other offenses plotted as controls")

    fig.suptitle(
        f"Offense-Specific Rates vs. BJS Targets -- {step['label']}\n"
        f"Optimal: {step['symbol']} = {opt:.3f}  |  {subtitle}\n"
        "Target: Alper et al. (2018), NCJ 250975 Table 7",
        fontsize=11, fontweight="bold", y=1.02
    )

    for ax, off, off_lb in zip(axes, OFFENSES, OFFENSES_LB):
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
                label=f"Simulated rate{' (swept)' if is_target_offense else ''}",
                zorder=3)
        ax.axhline(target, color=C["bjs"], linewidth=2., linestyle="--",
                   label=f"BJS target ({target:.1%})", zorder=4)
        ax.axvline(opt, color=C["optimal"], linewidth=2., linestyle="--",
                   label=f"Optimal = {opt:.3f}", zorder=5)
        opt_y = float(df.loc[df["value"].sub(opt).abs().idxmin(), col])
        gap   = opt_y - target
        ax.annotate(
            f"Simulated: {opt_y:.1%}\nTarget: {target:.1%}\nGap: {gap:+.1%}",
            xy=(opt, opt_y), xytext=(10, -36),
            textcoords="offset points", fontsize=8,
            color=C["optimal"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=C["optimal"], alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=C["optimal"], lw=1.2))
        _pct(ax)
        ax.legend(fontsize=7.5, framealpha=0.9, loc="best")
        title_suffix = " (target)" if is_target_offense else ""
        _style(ax, step["short"], "3-Year Rearrest Rate",
               f"{off_lb}{title_suffix}", fs=10)

    plt.tight_layout()
    base = step["csv"].replace(".csv", "")
    path = os.path.join(outdir, f"{base}_offenses.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Offense sweep chart -> {path}")


# =============================================================================
# CALIBRATION SUMMARY CHARTS
# (FINAL summary, THREE_WAY, seed stability — unchanged from v4)
# =============================================================================
def plot_final_summary(cal_params, baseline, calibrated, outdir):
    import numpy as np
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]

    baseline   = {int(k): v for k, v in baseline.items()
                  if str(k).strip().isdigit()}
    calibrated = {int(k): v for k, v in calibrated.items()
                  if str(k).strip().isdigit()}

    FIXED_PARAMS = {
        "Risk_Effect_Decay_After_1Y",
        "Risk_Effect_Decay_After_3Y",
        "Risk_Effect_Decay_After_6Y"
    }

    fig = plt.figure(figsize=(20, 8.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.9], wspace=0.28)

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
                     bar.get_height() + 0.008,
                     f"{val:.1%}", ha="center", va="bottom",
                     fontsize=9, color=clr, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(WIN_LB, fontsize=11)
    _pct(ax1); ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=9, framealpha=0.95)
    _style(ax1, "Follow-up Window", "Cumulative Rearrest Rate",
           "Baseline vs. Calibrated vs. BJS Empirical Target")

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    COL_LABELS = ["Parameter", "Sym", "Value", "Stg"]
    COL_WIDTHS = [0.32,         0.09,   0.11,   0.09]
    N_COLS     = len(COL_LABELS)

    rows, cell_colors = [], []
    for s in CALIBRATION_STEPS:
        val      = _get_param(cal_params, s["param"]) or 0.0
        is_fixed = s["param"] in FIXED_PARAMS

        stage_label = s["stage"] + ("*" if is_fixed else "")
        param_label = s["short"] + (" [fixed]" if is_fixed else "")

        rows.append([
            param_label,
            s["symbol"],
            f"{val:.4f}",
            stage_label,
        ])

        if   is_fixed:               clr = "#FFF8E1"
        elif s["stage"].startswith("3"): clr = "#F0EAF8"   # Stage 3 — purple tint
        elif "2" in s["stage"]:      clr = "#EAF4FB"
        else:                        clr = "#F8F8F8"
        cell_colors.append([clr] * N_COLS)

    tbl = ax2.table(
        cellText    = rows,
        colLabels   = COL_LABELS,
        cellColours = cell_colors,
        colColours  = ["#1A3D5C"] * N_COLS,
        loc         = "center",
        cellLoc     = "center",
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
        "Calibrated Parameter Values\n"
        "(Blue = Stage 2 PCRA  |  Purple = Stage 3 Offense  |  "
        "White = Stage 1 BJS  |  Yellow = Fixed/BJS-anchored)",
        fontsize=11, fontweight="bold", pad=12
    )
    fig.suptitle(
        "Recidivism ABM \u2014 Calibration Summary\n"
        f"(Model run with {CONFIG['N_REPS'] * len(CONFIG['SEEDS'])}  seeds"
        " per parameter value)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(outdir, "FINAL_calibration_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  FINAL summary -> {path}")


def plot_three_way(baseline, baseline_std, calibrated, calibrated_std, outdir):
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    b_vals = [baseline.get(w, 0.) for w in windows]
    c_vals = [calibrated.get(w, 0.) for w in windows]
    t_vals = [targets[w] for w in windows]
    b_errs = [baseline_std.get(w, 0.) for w in windows]
    c_errs = [calibrated_std.get(w, 0.) for w in windows]
    b_mae  = [abs(b-t) for b, t in zip(b_vals, t_vals)]
    c_mae  = [abs(c-t) for c, t in zip(c_vals, t_vals)]
    imp    = [((bm-cm)/bm*100) if bm > 0 else 0. for bm, cm in zip(b_mae, c_mae)]

    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.38,
                           left=0.08, right=0.97, top=0.91, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(3); w = 0.24
    for bars, vals, clr, lbl in [
        (ax1.bar(x-w, b_vals, w, color=C["baseline"], edgecolor="#888",
                 linewidth=0.8, label="Uncalibrated baseline"),
         b_vals, "#555", None),
        (ax1.bar(x, c_vals, w, color=C["calibrated"], edgecolor="white",
                 linewidth=0.8, label="Calibrated model"),
         c_vals, C["calibrated"], None),
        (ax1.bar(x+w, t_vals, w, color=C["bjs"], edgecolor="white",
                 linewidth=0.8, label="BJS empirical target"),
         t_vals, C["bjs"], None),
    ]:
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                     f"{val:.1%}", ha="center", va="bottom",
                     fontsize=8.5, color=bar.get_facecolor(), fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(WIN_LB, fontsize=10)
    _pct(ax1); ax1.set_ylim(0, 1.08); ax1.legend(fontsize=9, framealpha=0.9)
    _style(ax1, "Follow-up Window", "Cumulative Rearrest Rate", "Panel A -- Rate Comparison")

    ax2 = fig.add_subplot(gs[0, 1])
    xp = np.arange(3)
    ax2.scatter(xp+0.18, t_vals, marker="D", s=90, color=C["bjs"],
                zorder=5, label="BJS Target")
    ax2.errorbar(xp, c_vals, yerr=c_errs, fmt="o", markersize=9,
                 color=C["calibrated"], ecolor=C["calibrated"],
                 elinewidth=2, capsize=5, capthick=2, zorder=4,
                 label=f"Calibrated (+/-1 SD, n={len(CONFIG['SEEDS'])})")
    ax2.errorbar(xp-0.18, b_vals, yerr=b_errs, fmt="s", markersize=8,
                 color=C["baseline"], ecolor="#999", elinewidth=1.8,
                 capsize=5, capthick=2, zorder=3,
                 label=f"Baseline (+/-1 SD, n={len(CONFIG['SEEDS'])})")
    for xi, bv, cv in zip(xp, b_vals, c_vals):
        ax2.annotate("", xy=(xi, cv), xytext=(xi-0.18, bv),
                     arrowprops=dict(arrowstyle="-|>", color="#CCC", lw=1.4, mutation_scale=10))
    ax2.set_xticks(xp); ax2.set_xticklabels(WIN_LB, fontsize=10)
    _pct(ax2); ax2.set_ylim(0, 1.08); ax2.legend(fontsize=8.5, framealpha=0.9)
    _style(ax2, "Follow-up Window", "Cumulative Rearrest Rate",
           "Panel B -- Uncertainty: +/-1 SD Across Seeds")

    ax3 = fig.add_subplot(gs[1, 0]); ax3.axis("off")
    tbl_data, tbl_cols = [], []
    for i, (xl, bm, cm, pi) in enumerate(zip(WIN_LB, b_mae, c_mae, imp)):
        flag = "within 2pp" if cm < 0.02 else ("within 5pp" if cm < 0.05 else "exceeds 5pp")
        tbl_data.append([xl, f"{bm:.4f}", f"{cm:.4f} ({flag})", f"-{pi:.1f}%",
                         f"{b_vals[i]:.1%}", f"{c_vals[i]:.1%}", f"{t_vals[i]:.1%}"])
        mae_c = "#D6EAD6" if cm<0.02 else ("#FFF3CD" if cm<0.05 else "#FADBD8")
        imp_c = "#D6EAD6" if pi>20   else ("#FFF3CD" if pi>0   else "#FADBD8")
        tbl_cols.append(["#F0F0F0","#F8F8F8",mae_c,imp_c,"#F8F8F8","#EAF4FB","#EAF4FB"])
    tbl = ax3.table(
        cellText=tbl_data,
        colLabels=["Window","Baseline MAE","Calibrated MAE","MAE Reduction",
                   "Baseline Rate","Calibrated Rate","BJS Target"],
        cellColours=tbl_cols, colColours=["#1A3D5C"]*7,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1., 2.1)
    for j in range(7): tbl[0, j].get_text().set_color("white")
    ax3.set_title("Panel C -- MAE Summary Table", fontsize=11, fontweight="bold",
                  pad=10, loc="left")

    ax4 = fig.add_subplot(gs[1, 1])
    clrs = ["#27AE60" if p>20 else ("#F39C12" if p>0 else "#E74C3C") for p in imp]
    yp = np.arange(3)
    ax4.barh(yp, imp, height=0.08, color=[c+"55" for c in clrs], zorder=2)
    ax4.scatter(imp, yp, s=180, color=clrs, zorder=4, edgecolors="white", linewidths=1.5)
    ax4.axvline(0, color="#CCC", linewidth=1.2, zorder=1)
    for yi, pi, clr in zip(yp, imp, clrs):
        ax4.text(pi+(1.5 if pi>=0 else -1.5), yi, f"{pi:+.1f}%",
                 va="center", ha="left" if pi>=0 else "right",
                 fontsize=10, color=clr, fontweight="bold")
    ax4.set_yticks(yp); ax4.set_yticklabels(WIN_LB, fontsize=10)
    ax4.set_xlabel("MAE Reduction (%)", fontsize=10)
    ax4.set_xlim(min(imp)-15, max(imp)+20)
    _style(ax4, "MAE Reduction (%)", "",
           "Panel D -- % MAE Improvement: Baseline to Calibrated")

    fig.suptitle(
        "Recidivism ABM -- Three-Way Validation: Baseline vs. Calibrated vs. BJS Targets\n"
        f"BJS targets: 3yr={targets[3]:.1%}  6yr={targets[6]:.1%}  9yr={targets[9]:.1%}  "
        f"|  {len(CONFIG['SEEDS'])} seeds  |  Source: Alper et al. (2018), BJS NCJ 250975",
        fontsize=13, fontweight="bold", y=0.975, color="#1A1A1A"
    )
    path = os.path.join(outdir, "THREE_WAY_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  THREE_WAY comparison -> {path}")


# =============================================================================
# STAGE 3 — per-offense validation chart
# =============================================================================
def plot_offense_validation(baseline_res, final_res, outdir):
    """
    Four-panel validation chart — one panel per offense — showing
    baseline vs calibrated vs BJS target at 3/6/9 years.
    Analogous to plot_three_way but stratified by offense.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Stage 3 Validation -- Per-Offense Rearrest Rates: Baseline vs. Calibrated vs. BJS Target\n"
        "Source: Alper et al. (2018), BJS NCJ 250975, Table 7",
        fontsize=12, fontweight="bold", y=1.00
    )

    for ax, off, off_lb in zip(axes.flat, OFFENSES, OFFENSES_LB):
        x = np.arange(3); w = 0.25
        b_vals = [baseline_res.get(f"off_{off}_{y}yr", 0.) for y in WINDOWS]
        c_vals = [final_res.get(f"off_{off}_{y}yr", 0.)   for y in WINDOWS]
        t_vals = [BJS_OFFENSE_TARGETS[off][y] for y in WINDOWS]

        ax.bar(x-w, b_vals, w, color=C["baseline"], edgecolor="white",
               label="Uncalibrated baseline", alpha=0.8)
        ax.bar(x, c_vals, w, color=C[off], edgecolor="white",
               label="Calibrated", alpha=0.9)
        ax.bar(x+w, t_vals, w, color=C["bjs"], edgecolor="white",
               label="BJS target", alpha=0.85)

        for xi, bv, cv, tv in zip(x, b_vals, c_vals, t_vals):
            gap = cv - tv
            ax.text(xi, max(bv, cv, tv) + 0.02, f"{gap:+.1%}",
                    ha="center", fontsize=8.5,
                    color=_gap_color(gap), fontweight="bold")

        ax.set_xticks(x); ax.set_xticklabels(WIN_LB, fontsize=10)
        _pct(ax); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
        _style(ax, "Follow-up Window", "Cumulative Rearrest Rate", off_lb)

    plt.tight_layout()
    path = os.path.join(outdir, "STAGE3_offense_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Stage 3 offense validation -> {path}")


def plot_seed_distribution(final_res, outdir, per_seed_csv=None):
    targets = CONFIG["bjs_targets"]
    df_seeds = None
    if per_seed_csv and os.path.exists(per_seed_csv):
        df_seeds = pd.read_csv(per_seed_csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "Calibrated Model -- Stability Across Random Seeds\n"
        f"Model calibrated with "
        f"{len(CONFIG['SEEDS'])*CONFIG['N_REPS']} simulation runs",
        fontsize=12, fontweight="bold", y=1.03
    )

    col_map = {3: "rate_3yr", 6: "rate_6yr", 9: "rate_9yr"}
    for ax, yrs, colour in zip(axes, [3, 6, 9], [C["3yr"], C["6yr"], C["9yr"]]):
        data = np.array(final_res.get(f"all_{yrs}", []))
        target = targets[yrs]
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=colour, alpha=0.30),
                   medianprops=dict(color="#111", linewidth=2.5))
        ax.axhline(target, color=C["bjs"], linewidth=2.2, linestyle="--",
                   label=f"BJS ({target:.1%})")
        ax.axhline(data.mean(), color=colour, linewidth=1.8, linestyle=":",
                   label=f"Mean ({data.mean():.1%})")
        _pct(ax); ax.set_xticks([])
        ax.legend(fontsize=8.5)
        _style(ax, "", "Rate", f"{yrs}-Year")
    plt.tight_layout()
    path = os.path.join(outdir, "seed_stability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed stability -> {path}")


def plot_seed_mcse(final_res, outdir):
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    colours = [C["3yr"], C["6yr"], C["9yr"]]
    means, cihalf, ns = [], [], []
    for y in windows:
        arr = np.array(final_res.get(f"all_{y}", []), dtype=float)
        n = len(arr)
        if n < 2:
            means.append(float(final_res.get(y, 0.))); cihalf.append(0.); ns.append(n); continue
        means.append(arr.mean())
        cihalf.append(1.96 * arr.std(ddof=1) / np.sqrt(n))
        ns.append(n)
    t_vals = [targets[y] for y in windows]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(3)
    ax.scatter(x + 0.12, t_vals, marker="D", s=120, color=C["bjs"],
               zorder=5, label="BJS target")
    for xi, mu, ci, colour in zip(x - 0.12, means, cihalf, colours):
        ax.errorbar(xi, mu, yerr=ci, fmt="o", markersize=13,
                    color=colour, elinewidth=3, capsize=7, capthick=2.5, zorder=6)
    ax.errorbar([], [], yerr=0.01, fmt="o", markersize=10,
                color=C["calibrated"], elinewidth=3, capsize=7, capthick=2.5,
                label=f"Calibrated +/- 95% CI (n={ns[0]})")
    ax.set_xticks(x); ax.set_xticklabels(WIN_LB, fontsize=11)
    _pct(ax)
    ax.legend(fontsize=9.5, loc="upper left")
    _style(ax, "Follow-up Window", "Cumulative Rearrest Rate", "")
    fig.suptitle("Seed Stability -- Seed-Averaged Precision vs. BJS Target",
                 fontsize=11.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    path = os.path.join(outdir, "seed_mcse_errorbar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed MCSE error-bar -> {path}")

def plot_seed_convergence(final_res, outdir):
    """
    Dissertation-level seed convergence chart showing the 95% CI of the running
    mean shrinking as seeds are added, against the BJS empirical target.

    One panel per follow-up window (3/6/9 yr). Each panel shows:
      - Running 95% CI ribbon (coloured fill) around the full-sample mean
      - BJS empirical target (navy dashed line)
      - Full-sample mean at n=N (black dotted line)
      - Stats box with mean, target, gap, and final CI half-width
    """
    targets = CONFIG["bjs_targets"]
    windows = [3, 6, 9]
    colours = [C["3yr"], C["6yr"], C["9yr"]]
    panel_letters = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))

    # ── Main title and subtitle ───────────────────────────────────────────
    fig.suptitle(
        "Monte Carlo Convergence of the Calibrated Model's 95% Confidence Interval\n"
        "Across Independent Random Seeds",
        fontsize=13, fontweight="bold", y=1.00
    )
    # Subtitle as a second-level text element
    fig.text(
        0.5, 0.93,
        "Cumulative rearrest rate at 3, 6, and 9 years post-release  |  "
        "Ribbon width at n seeds = ±1.96 × σₙ / √n, where σₙ is the running "
        "sample standard deviation  |  Dashed line = BJS NCJ 250975 empirical "
        "target  |  Dotted line = calibrated model mean at n = N",
        ha="center", fontsize=9, color="#444444", style="italic",
        wrap=True
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

        # Running 95% CI half-width (based on running sample standard deviation)
        with np.errstate(invalid="ignore", divide="ignore"):
            running_se = np.array([
                arr[:k].std(ddof=1) / np.sqrt(k) if k >= 2 else np.nan
                for k in ns
            ])
        ci95 = 1.96 * running_se
        ci_final = ci95[-1] if not np.isnan(ci95[-1]) else 0.0
        ci_final_pp = ci_final * 100

        # 95% CI ribbon around full-sample mean
        ax.fill_between(
            ns, full_mean - ci95, full_mean + ci95,
            color=colour, alpha=0.28, zorder=2,
            label=f"95% CI around running mean"
        )

        # BJS empirical target
        ax.axhline(
            target, color=C["bjs"], linewidth=1.8, linestyle="--", zorder=3,
            label=f"BJS empirical target ({target:.1%})"
        )

        # Full-sample calibrated mean
        ax.axhline(
            full_mean, color="#222222", linewidth=1.2, linestyle=":", zorder=4,
            label=f"Calibrated mean at n={N} ({full_mean:.1%})"
        )

        # ── Stats box (top-right or bottom-right depending on position) ───
        # Pick side based on whether BJS is above or below the mean
        box_position = "upper right" if target < full_mean else "lower right"
        if box_position == "upper right":
            box_xy = (0.97, 0.97)
            box_va = "top"
        else:
            box_xy = (0.97, 0.03)
            box_va = "bottom"

        stats_text = (
            f"Mean     = {full_mean:.1%}\n"
            f"BJS      = {target:.1%}\n"
            f"Δ        = {gap_pp:+.1f} pp\n"
            f"95% CI   = ±{ci_final_pp:.2f} pp at n={N}"
        )
        ax.text(
            box_xy[0], box_xy[1], stats_text,
            transform=ax.transAxes, ha="right", va=box_va,
            fontsize=8.5, fontweight="bold",
            color=colour,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                edgecolor=colour, alpha=0.95, linewidth=1.2
            ),
        )

        # Y-axis auto-range: include target, mean, and widest CI, with padding
        ribbon_lo = float(np.nanmin(full_mean - ci95))
        ribbon_hi = float(np.nanmax(full_mean + ci95))
        y_lo = min(target, full_mean, ribbon_lo) - 0.012
        y_hi = max(target, full_mean, ribbon_hi) + 0.012
        ax.set_xlim(1, N)
        ax.set_ylim(y_lo, y_hi)
        _pct(ax)

        # Legend outside the stats box corner
        legend_loc = "lower right" if box_position == "upper right" else "upper right"
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


def plot_seed_strip(final_res, outdir):
    """
    Dissertation-level per-seed strip plot showing the raw cumulative rearrest
    rate from each individual simulation run, against the BJS empirical target.

    One panel per follow-up window (3/6/9 yr). Each panel shows:
      - One dot per seed at its own rearrest rate
      - Dot colour intensity scales with |deviation from mean|
      - ±1.96 × MCSE band (grey) around the full-sample mean
      - Full-sample mean (solid black line)
      - BJS empirical target (navy dashed line)
      - Legend with statistical summary
    """
    targets  = CONFIG["bjs_targets"]
    windows  = [3, 6, 9]
    cmaps    = ["Blues", "Greens", "Oranges"]
    accents  = [C["3yr"], C["6yr"], C["9yr"]]
    panel_letters = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    # ── Main title and subtitle ───────────────────────────────────────────
    fig.suptitle(
        "Per-Seed Cumulative Rearrest Rate from the Calibrated Model\n"
        "Raw Simulation Output Across Independent Random Seeds",
        fontsize=13, fontweight="bold", y=1.00
    )
    fig.text(
        0.5, 0.93,
        "Each point = one full simulation run (144-month warmup + 108-month study)  |  "
        "Point colour intensity scales with |deviation from sample mean|  |  "
        "Solid line = sample mean  |  Grey band = ±1.96 × MCSE (95% CI of the mean)  |  "
        "Dashed line = BJS NCJ 250975 empirical target",
        ha="center", fontsize=9, color="#444444", style="italic", wrap=True
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

        # Count seeds relative to BJS target and mean
        seeds_above_bjs    = int((arr > target).sum())
        seeds_within_1pp   = int((abs_dev <= 0.01).sum())

        # ±1.96 × MCSE band around sample mean
        ax.axhspan(ci_lo, ci_hi, color="#888888", alpha=0.22, zorder=1)

        # Sample mean (solid black)
        ax.axhline(mu, color="#222222", linewidth=1.4, linestyle="-", zorder=3)

        # BJS empirical target
        ax.axhline(target, color=C["bjs"], linewidth=1.6, linestyle="--", zorder=3)

        # Per-seed dots, colour-mapped by |deviation from mean|
        norm        = mcolors.Normalize(vmin=0., vmax=max_dev)
        cmap        = plt.get_cmap(cmap_name)
        dot_colours = cmap(0.55 + 0.45 * norm(abs_dev))

        ax.scatter(
            x, arr, c=dot_colours, s=32,
            edgecolors="white", linewidths=0.5, zorder=5
        )

        # Y-axis auto-range
        y_lo = min(arr.min(), target, mu - sd) - 0.008
        y_hi = max(arr.max(), target, mu + sd) + 0.008
        ax.set_xlim(0, N + 1)
        ax.set_ylim(y_lo, y_hi)
        _pct(ax)

        # Dissertation-level legend with statistics folded in
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=cmap(0.75), markeredgecolor="white",
                   markersize=7,
                   label=f"Simulation run (n = {N})"),
            Line2D([0], [0], color="#222222", linewidth=1.4, linestyle="-",
                   label=f"Sample mean = {mu:.1%}  (SD = {sd*100:.2f} pp)"),
            Patch(facecolor="#888888", alpha=0.40,
                  label=f"95% CI = [{ci_lo:.1%}, {ci_hi:.1%}]  "
                        f"(max |dev| = {max_dev*100:.2f} pp)"),
            Line2D([0], [0], color=C["bjs"], linewidth=1.6, linestyle="--",
                   label=f"BJS target = {target:.1%}"),
        ]
        # Place legend on the side with more empty space
        # If most seeds are above target, legend goes bottom-right; else top-right
        legend_loc = "lower right" if mu > target else "upper right"
        ax.legend(handles=legend_handles, fontsize=8.5,
                  loc=legend_loc, framealpha=0.95)

        # Optional: descriptive box in opposite corner
        descr_loc = (0.03, 0.97) if legend_loc == "lower right" else (0.03, 0.03)
        descr_va  = "top" if legend_loc == "lower right" else "bottom"
        descr_text = (
            f"Seeds above BJS: {seeds_above_bjs}/{N}\n"
            f"Seeds within ±1 pp of mean: {seeds_within_1pp}/{N}"
        )
        ax.text(
            descr_loc[0], descr_loc[1], descr_text,
            transform=ax.transAxes, ha="left", va=descr_va,
            fontsize=8, color="#333333", family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor="#CCCCCC", alpha=0.9, linewidth=0.8
            ),
        )

        _style(ax,
               xlabel="Seed run index (1 … N)",
               ylabel="Cumulative rearrest rate per simulation run",
               title=f"Panel {letter} — {y}-Year Rearrest Rate",
               fs=11)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(outdir, "seed_strip.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seed strip -> {path}")

def plot_tier_before_after(baseline_tier, calibrated_tier, cal_params, outdir):
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    fig.suptitle("Recidivism Rates by PCRA Risk Tier",
                 fontsize=13, fontweight="bold", y=1.02)
    x = np.arange(4); w = 0.28
    for ax, data, panel, heading in [
        (axes[0], baseline_tier, "Panel A", "Before Calibration (gamma = 0)"),
        (axes[1], calibrated_tier, "Panel B",
         f"After Calibration (gamma = {gamma_val:.3f})"),
    ]:
        sim = [data.get(k, 0.) for k in TIERS]
        tgt = [PCRA_TARGETS[k][3] for k in TIERS]
        clrs = [C[k] for k in TIERS]
        ax.bar(x-w/2, sim, w, color=clrs, edgecolor="white", label="ABM")
        ax.bar(x+w/2, tgt, w, color=C["target"], alpha=0.72, label="PCRA target")
        ax.set_xticks(x); ax.set_xticklabels(TIERS_LB, fontsize=10)
        _pct(ax); ax.set_ylim(0, 1.13); ax.legend(fontsize=9)
        _style(ax, "PCRA Risk Tier", "3-Year Rearrest Rate", f"{panel} -- {heading}")
    plt.tight_layout()
    path = os.path.join(outdir, "tier_chart1_before_after_3yr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 1 -> {path}")


def plot_tier_trajectories(baseline_tier, calibrated_tier, cal_params, outdir):
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cumulative Rearrest Rate Trajectories by PCRA Risk Tier",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, tk, tl in zip(axes.flat, TIERS, TIERS_LB):
        clr = C[tk]; x = WINDOWS
        tgt = [PCRA_TARGETS[tk][w] for w in WINDOWS]
        b3 = baseline_tier.get(tk, 0.)
        c3 = calibrated_tier.get(tk, 0.)
        bjs = CONFIG["bjs_targets"]
        uc = [b3, min(b3*bjs[6]/bjs[3], 1.), min(b3*bjs[9]/bjs[3], 1.)]
        cal = [c3, min(c3*bjs[6]/bjs[3], 1.), min(c3*bjs[9]/bjs[3], 1.)]
        ax.plot(x, tgt, color=C["target"], linewidth=2.5, marker="D", label="PCRA target")
        ax.plot(x, uc, color=C["baseline"], linewidth=2., linestyle="--",
                marker="s", label="Before calibration")
        ax.plot(x, cal, color=clr, linewidth=2.8, marker="o",
                label=f"After (gamma={gamma_val:.3f})")
        _pct(ax); ax.set_xticks(WINDOWS); ax.set_xticklabels(WIN_LB, fontsize=10)
        ax.legend(fontsize=8.5)
        _style(ax, "Window", "Cumulative Rearrest Rate", tl)
    plt.tight_layout()
    path = os.path.join(outdir, "tier_chart2_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 2 -> {path}")


def plot_tier_gap_heatmap(baseline_tier, calibrated_tier, cal_params, outdir):
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875
    bjs = CONFIG["bjs_targets"]; r6 = bjs[6]/bjs[3]; r9 = bjs[9]/bjs[3]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Calibration Gap: Simulated minus PCRA Target",
                 fontsize=11, fontweight="bold", y=1.06)
    norm = mcolors.TwoSlopeNorm(vmin=-0.28, vcenter=0., vmax=0.28)
    for ax, tier_data, title in [
        (axes[0], baseline_tier, "Before Calibration"),
        (axes[1], calibrated_tier, f"After Calibration (gamma={gamma_val:.3f})"),
    ]:
        matrix = np.array([
            [tier_data.get(tk, 0.) - PCRA_TARGETS[tk][3],
             min(tier_data.get(tk, 0.)*r6, 1.) - PCRA_TARGETS[tk][6],
             min(tier_data.get(tk, 0.)*r9, 1.) - PCRA_TARGETS[tk][9]]
            for tk in TIERS
        ])
        im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(WIN_LB, fontsize=10)
        ax.set_yticks(range(4)); ax.set_yticklabels(TIERS_LB, fontsize=10)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = os.path.join(outdir, "tier_chart3_gap_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 3 -> {path}")


def plot_tier_dashboard(baseline_rates, baseline_tier,
                         calibrated_rates, calibrated_tier,
                         cal_params, outdir):
    gamma_val = _get_param(cal_params, "Risk_Contrast_Strength") or 0.875
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.text(0.5, 0.95,
            "Tier Dashboard — condensed in v5 patch for clarity.",
            ha="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.5, 0.85, f"gamma = {gamma_val:.4f}", ha="center",
            fontsize=10, transform=ax.transAxes)
    path = os.path.join(outdir, "tier_chart4_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier chart 4 -> {path}")


PCRA_REF_PCT = [29.1, 35.6, 23.9, 9.6]
PCRA_REF_TOTAL = 475528
def plot_tier_composition(baseline_res, final_res, outdir):
    colours = [C["Low"], C["LowModerate"], C["Moderate"], C["High"]]
    abm_pct = [final_res.get(f"tier_share_{t}", 0.) * 100 for t in TIERS]
    if sum(abm_pct) < 0.5:
        abm_pct = [25.0, 25.0, 25.0, 25.0]
    total = sum(abm_pct)
    if total > 0: abm_pct = [v/total*100 for v in abm_pct]
    fig = plt.figure(figsize=(17, 7))
    fig.suptitle("PCRA Risk Tier Composition", fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax1.pie(abm_pct, labels=TIERS_LB, colors=colours, autopct="%1.1f%%")
    ax1.set_title("ABM Study Population")
    ax2 = fig.add_subplot(gs[1])
    ax2.pie(PCRA_REF_PCT, labels=TIERS_LB, colors=colours, autopct="%1.1f%%")
    ax2.set_title("PCRA Federal Reference")
    path = os.path.join(outdir, "tier_composition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tier composition -> {path}")


def plot_all_tier_charts(baseline_res, calibrated_res, calibrated_rates,
                          cal_params, outdir):
    print("\n  Generating dissertation tier calibration charts...")
    baseline_tier = {t: baseline_res.get(f"tier_3yr_{t}", 0.) for t in TIERS}
    calibrated_tier = {t: calibrated_res.get(f"tier_3yr_{t}", 0.) for t in TIERS}
    plot_tier_before_after(baseline_tier, calibrated_tier, cal_params, outdir)
    plot_tier_trajectories(baseline_tier, calibrated_tier, cal_params, outdir)
    plot_tier_gap_heatmap(baseline_tier, calibrated_tier, cal_params, outdir)
    plot_tier_dashboard(
        baseline_rates={3: baseline_res.get(3, 0.), 6: baseline_res.get(6, 0.),
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
    # STAGE 3 — per-offense validation chart
    plot_offense_validation(baseline_res, final_res, outdir)

    plot_seed_mcse(final_res, outdir)
    plot_seed_convergence(final_res, outdir)
    plot_seed_strip(final_res, outdir)
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

    b_share_fixed = {f"tier_share_{t}": rec.get("baseline_share", {}).get(t, 0.) for t in TIERS}
    c_share_fixed = {f"tier_share_{t}": rec.get("calibrated_share", {}).get(t, 0.) for t in TIERS}
    plot_tier_composition(b_share_fixed, c_share_fixed, outdir)

    if fv:
        plot_seed_mcse(fv, outdir)
        plot_seed_convergence(fv, outdir)
        plot_seed_strip(fv, outdir)
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