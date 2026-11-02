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
6. Desistance survival curve  — % NOT yet arrested by year, mirrors BJS Fig. 4
7. Annual first-arrest rate by Gender  — mirrors BJS Fig. 5 / Table 3
8. Cumulative rearrest rate by Gender  — derived running sum of Chart 7 hazards
9. Annual first-arrest rate by AGE AT RELEASE  — BJS Table 3 age rows
10. Cumulative rearrest rate by AGE AT RELEASE
11. Annual first-arrest rate by RACE  — BJS Table 3 race rows
12. Cumulative rearrest rate by RACE

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

YEAR-OF-FIRST-ARREST DERIVATION
-------------------------------
Primary:    agent.rearrest_year   — direct, already in [1, 9]
Fallback 1: agent.rearrest_month  — absolute sim month
                                    year = ceil((rearrest_month - warmup) / 12)

USAGE
-----
Full run (simulate + plot):
    python recidivism_validation.py

Replot only (load cached results, regenerate charts):
    python recidivism_validation.py --replot
"""

import os
import sys
import time
import math
import random
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "..")))

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
BJS_FIRST_ARREST = (
    [BJS_CUMULATIVE_ALL[0]]
    + [BJS_CUMULATIVE_ALL[i] - BJS_CUMULATIVE_ALL[i-1]
       for i in range(1, 9)]
)

BJS_TARGETS = {3: 68.4, 6: 79.4, 9: 83.4}

# =============================================================================
# BJS TABLE 3 — non-cumulative year-of-first-arrest by Gender (Alper et al. 2018)
# Source values supplied as fractions; converted to percent for chart parity.
# https://bjs.ojp.gov/content/pub/pdf/18upr9yfup0514.pdf
# =============================================================================
Gender_GROUPS = ["Male", "Female"]

_BJS_TABLE3_RAW = {
    "total_9yr": {"Male": 0.842, "Female": 0.768},
    1: {"Male": 0.449, "Female": 0.351},
    2: {"Male": 0.163, "Female": 0.157},
    3: {"Male": 0.083, "Female": 0.085},
    4: {"Male": 0.051, "Female": 0.055},
    5: {"Male": 0.034, "Female": 0.042},
    6: {"Male": 0.023, "Female": 0.025},
    7: {"Male": 0.017, "Female": 0.022},
    8: {"Male": 0.013, "Female": 0.017},
    9: {"Male": 0.009, "Female": 0.014},
}
# Per-year first-arrest %, indexed by Gender → list[year-1 .. year-9]
BJS_FIRST_BY_Gender = {
    s: [_BJS_TABLE3_RAW[y][s] * 100 for y in range(1, 10)]
    for s in Gender_GROUPS
}
# Cumulative 9-year first-arrest % by Gender (from the "total_9yr" row)
BJS_CUM9_BY_Gender = {
    s: _BJS_TABLE3_RAW["total_9yr"][s] * 100
    for s in Gender_GROUPS
}

# Map raw ABM Gender strings → BJS Gender groups (case-insensitive, defensive)
def _normalize_Gender(raw):
    if raw is None: return None
    s = str(raw).strip().lower()
    if s in ("m", "male", "man"):    return "Male"
    if s in ("f", "female", "woman"): return "Female"
    return None

# =============================================================================
# BJS TABLE 3 — AGE AT RELEASE
# Five non-overlapping buckets.  The table also gives a "25–39" combined row
# but we use the three sub-buckets (25–29 / 30–34 / 35–39) so groups don't
# double-count.
# =============================================================================
AGE_GROUPS = ["24 or younger", "25-29", "30-34", "35-39", "40 or older"]

_BJS_TABLE3_AGE_RAW = {
    "total_9yr": {
        "24 or younger": 0.901, "25-29": 0.870, "30-34": 0.843,
        "35-39": 0.843,        "40 or older": 0.765,
    },
    1: {"24 or younger": 0.518, "25-29": 0.459, "30-34": 0.439, "35-39": 0.446, "40 or older": 0.378},
    2: {"24 or younger": 0.170, "25-29": 0.168, "30-34": 0.165, "35-39": 0.168, "40 or older": 0.151},
    3: {"24 or younger": 0.077, "25-29": 0.088, "30-34": 0.082, "35-39": 0.087, "40 or older": 0.081},
    4: {"24 or younger": 0.048, "25-29": 0.055, "30-34": 0.053, "35-39": 0.049, "40 or older": 0.051},
    5: {"24 or younger": 0.034, "25-29": 0.038, "30-34": 0.035, "35-39": 0.034, "40 or older": 0.035},
    6: {"24 or younger": 0.020, "25-29": 0.025, "30-34": 0.023, "35-39": 0.022, "40 or older": 0.025},
    7: {"24 or younger": 0.017, "25-29": 0.015, "30-34": 0.019, "35-39": 0.017, "40 or older": 0.019},
    8: {"24 or younger": 0.010, "25-29": 0.013, "30-34": 0.017, "35-39": 0.012, "40 or older": 0.014},
    9: {"24 or younger": 0.007, "25-29": 0.009, "30-34": 0.010, "35-39": 0.009, "40 or older": 0.012},
}
BJS_FIRST_BY_AGE = {
    g: [_BJS_TABLE3_AGE_RAW[y][g] * 100 for y in range(1, 10)]
    for g in AGE_GROUPS
}
BJS_CUM9_BY_AGE = {
    g: _BJS_TABLE3_AGE_RAW["total_9yr"][g] * 100 for g in AGE_GROUPS
}

# Map an ABM agent's numeric/string age at release → BJS bucket
def _normalize_age(raw):
    """Accepts an integer/float age in years OR a string already matching a
    bucket name. Returns the BJS group label, or None if unparseable."""
    if raw is None: return None
    # Already a valid group label?
    if isinstance(raw, str) and raw.strip() in AGE_GROUPS:
        return raw.strip()
    try:
        a = float(raw)
    except (TypeError, ValueError):
        return None
    if a < 25: return "24 or younger"
    if a < 30: return "25-29"
    if a < 35: return "30-34"
    if a < 40: return "35-39"
    return "40 or older"

# =============================================================================
# BJS TABLE 3 — RACE/ETHNICITY
# Four non-overlapping buckets as published by BJS.
# =============================================================================
RACE_GROUPS = ["White", "Black", "Hispanic", "Other"]

_BJS_TABLE3_RACE_RAW = {
    "total_9yr": {"White": 0.809, "Black": 0.869, "Hispanic": 0.813, "Other": 0.824},
    1: {"White": 0.402, "Black": 0.460, "Hispanic": 0.473, "Other": 0.441},
    2: {"White": 0.158, "Black": 0.174, "Hispanic": 0.143, "Other": 0.164},
    3: {"White": 0.084, "Black": 0.086, "Hispanic": 0.072, "Other": 0.088},
    4: {"White": 0.052, "Black": 0.055, "Hispanic": 0.042, "Other": 0.045},
    5: {"White": 0.038, "Black": 0.034, "Hispanic": 0.031, "Other": 0.030},
    6: {"White": 0.026, "Black": 0.023, "Hispanic": 0.022, "Other": 0.013},
    7: {"White": 0.022, "Black": 0.016, "Hispanic": 0.009, "Other": 0.021},
    8: {"White": 0.015, "Black": 0.012, "Hispanic": 0.013, "Other": 0.013},
    9: {"White": 0.012, "Black": 0.009, "Hispanic": 0.007, "Other": 0.011},
}
BJS_FIRST_BY_RACE = {
    g: [_BJS_TABLE3_RACE_RAW[y][g] * 100 for y in range(1, 10)]
    for g in RACE_GROUPS
}
BJS_CUM9_BY_RACE = {
    g: _BJS_TABLE3_RACE_RAW["total_9yr"][g] * 100 for g in RACE_GROUPS
}

def _normalize_race(raw):
    """Maps common race/ethnicity strings to BJS buckets.  Hispanic takes
    priority over racial labels per BJS convention."""
    if raw is None: return None
    s = str(raw).strip().lower()
    # Hispanic ethnicity (overrides race per BJS reporting convention)
    if s in ("hispanic", "latino", "latina", "latinx", "h"):
        return "Hispanic"
    if s in ("white", "caucasian", "w", "non-hispanic white", "nh white"):
        return "White"
    if s in ("black", "african american", "afro-american", "b", "non-hispanic black", "nh black"):
        return "Black"
    # Anything else (Asian, Native American, Pacific Islander, multi-racial, etc.)
    if s in ("asian", "native", "native american", "american indian", "pacific islander",
             "multi", "multi-racial", "multiracial", "two or more", "other", "o"):
        return "Other"
    return None

# =============================================================================
# ABM OFFENSE MAPPING  →  BJS offense groups
# =============================================================================
OFFENSE_MAP = {
    "Violent":      "Violent",
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
    "initial_agents":       3000,
    "warmup_months":        144,
    "study_months":         108,
    "monthly_intake":       10,
    "bias_factor":          0,
    "enable_peer_influence": True,
}

# Cache filename for replot mode
RESULTS_CACHE = "aggregated_results.json"

# =============================================================================
# UTILITIES
# =============================================================================
_T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,
        9:2.262,10:2.228,15:2.131,20:2.086,30:2.042,60:2.000,120:1.980}

def _t95(df):
    for k in sorted(_T95):
        if df <= k: return _T95[k]
    return 1.96

def mean_ci_sd(vals):
    """Return (mean, lo_95ci, hi_95ci, sd) for a list of values."""
    v = [x for x in vals if x is not None and not math.isnan(x)]
    n = len(v)
    if n == 0: return float("nan"), float("nan"), float("nan"), float("nan")
    if n == 1:
        m = float(v[0])
        return m, m, m, 0.0
    m = sum(v)/n
    sd = math.sqrt(sum((x-m)**2 for x in v)/(n-1))
    h = _t95(n-1)*sd/math.sqrt(n)
    return m, m-h, m+h, sd

# Backward-compat shim
def mean_ci(vals):
    m, lo, hi, _ = mean_ci_sd(vals)
    return m, lo, hi

def detect_workers(n_tasks):
    logical  = os.cpu_count() or 4
    physical = max(1, logical//2)
    return min(n_tasks, max(1, physical-2))

# =============================================================================
# GAP-LABEL HELPERS  (shared across all charts)
# =============================================================================
GAP_GOOD    = 2.0
GAP_WARN    = 5.0
COLOUR_GOOD = "#27AE60"
COLOUR_WARN = "#F39C12"
COLOUR_BAD  = "#C0392B"

def gap_colour(diff: float) -> str:
    a = abs(diff)
    if a <= GAP_GOOD: return COLOUR_GOOD
    if a <= GAP_WARN: return COLOUR_WARN
    return COLOUR_BAD

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
                "cum_offense":  {g: [0]*9 for g in OFFENSE_GROUPS},
                "n_offense":    {g: 0 for g in OFFENSE_GROUPS},
                "first_by_Gender": {s: [0]*9 for s in Gender_GROUPS},
                "n_by_Gender":     {s: 0 for s in Gender_GROUPS},
                "first_by_age": {g: [0]*9 for g in AGE_GROUPS},
                "n_by_age":     {g: 0 for g in AGE_GROUPS},
                "first_by_race":{g: [0]*9 for g in RACE_GROUPS},
                "n_by_race":    {g: 0 for g in RACE_GROUPS}}

    # ── Year of first rearrest for each agent ────────────────────────────────
    # Primary:    agent.rearrest_year — set directly by the agent as
    #             ceil(community_months_at_risk / 12), clipped to [1, 9].
    #             Guaranteed consistent with rearrest_{1,3,6,9}_yrs flags
    #             (both derive from the same formula in the agent).
    # Fallback:   community_months_at_risk or rearrest_month - warmup_months.
    #             Used only if rearrest_year is missing / invalid (defensive
    #             guard against older runs or serialization issues).
    def get_rearrest_year(agent):
        if not getattr(agent, "recidivated_agent", False):
            return None

        # Primary: direct attribute (already in [1, 9] by construction)
        ry = getattr(agent, "rearrest_year", None)
        if ry is not None:
            try:
                y = int(ry)
                if 1 <= y <= 9:
                    return y
            except (TypeError, ValueError):
                pass

        # Fallback: re-derive from community_months_at_risk
        cm = getattr(agent, "community_months_at_risk", 0)
        if cm <= 0:
            rm = getattr(agent, "rearrest_month", None)
            if rm is None:
                return None
            cm = rm - model.warmup_months

        if cm <= 0:
            return None

        return max(1, min(9, math.ceil(cm / 12)))

    # Build cumulative and first-arrest arrays
    cum_counts   = [0]*9
    first_counts = [0]*9

    offense_counts = {g: [0]*9 for g in OFFENSE_GROUPS}
    offense_totals = {g: 0     for g in OFFENSE_GROUPS}

    # Per-Gender first-arrest tracking (for Chart 7 — BJS Table 3 parity)
    Gender_first_counts = {s: [0]*9 for s in Gender_GROUPS}
    Gender_totals       = {s: 0     for s in Gender_GROUPS}

    # Per-age-group first-arrest tracking (BJS Table 3, age-at-release rows)
    age_first_counts = {g: [0]*9 for g in AGE_GROUPS}
    age_totals       = {g: 0     for g in AGE_GROUPS}

    # Per-race first-arrest tracking (BJS Table 3, race rows)
    race_first_counts = {g: [0]*9 for g in RACE_GROUPS}
    race_totals       = {g: 0     for g in RACE_GROUPS}

    for agent in eligible:
        yr = get_rearrest_year(agent)
        offense_raw = getattr(agent, "offense", "Other(PublicOrder)")
        og = OFFENSE_MAP.get(offense_raw, "Public order")
        offense_totals[og] += 1

        # Sex: try common attribute names; if absent or unmappable, agent
        # is excluded from Gender-stratified counts (but still counted in totals).
        Gender_raw = (getattr(agent, "Gender", None)
                   or getattr(agent, "gender", None)
                   or getattr(agent, "agent_Gender", None))
        Gender_norm = _normalize_Gender(Gender_raw)
        if Gender_norm is not None:
            Gender_totals[Gender_norm] += 1

        # Age at release: try several common attribute names.  Falls back to
        # generic 'age' only if no release-specific attribute is present (the
        # plain 'age' may have aged with the simulation; the release-time
        # snapshot is what BJS Table 3 stratifies on).
        age_raw = (getattr(agent, "Age_at_Release", None)
                   or getattr(agent, "release_age", None)
                   or getattr(agent, "age_release", None)
                   or getattr(agent, "age", None))
        age_norm = _normalize_age(age_raw)
        if age_norm is not None:
            age_totals[age_norm] += 1

        # Race / ethnicity
        race_raw = (getattr(agent, "race", None)
                    or getattr(agent, "ethnicity", None)
                    or getattr(agent, "race_ethnicity", None))
        race_norm = _normalize_race(race_raw)
        if race_norm is not None:
            race_totals[race_norm] += 1

        if yr is not None:
            first_counts[yr-1] += 1
            for y in range(yr, 10):
                cum_counts[y-1] += 1
            for y in range(yr, 10):
                offense_counts[og][y-1] += 1
            if Gender_norm is not None:
                Gender_first_counts[Gender_norm][yr-1] += 1
            if age_norm is not None:
                age_first_counts[age_norm][yr-1] += 1
            if race_norm is not None:
                race_first_counts[race_norm][yr-1] += 1

    cum_all   = [c/n_total*100 for c in cum_counts]
    first_all = [c/n_total*100 for c in first_counts]
    cum_off   = {g: [offense_counts[g][i]/offense_totals[g]*100
                     if offense_totals[g] > 0 else 0.0
                     for i in range(9)]
                 for g in OFFENSE_GROUPS}

    # Per-Gender annual first-arrest % (denominator = agents of that Gender)
    first_by_Gender = {
        s: [Gender_first_counts[s][i] / Gender_totals[s] * 100 if Gender_totals[s] > 0 else 0.0
            for i in range(9)]
        for s in Gender_GROUPS
    }
    # Per-age-bucket annual first-arrest %
    first_by_age = {
        g: [age_first_counts[g][i] / age_totals[g] * 100 if age_totals[g] > 0 else 0.0
            for i in range(9)]
        for g in AGE_GROUPS
    }
    # Per-race annual first-arrest %
    first_by_race = {
        g: [race_first_counts[g][i] / race_totals[g] * 100 if race_totals[g] > 0 else 0.0
            for i in range(9)]
        for g in RACE_GROUPS
    }

    return {
        "run_id":        run_id,
        "seed":          seed,
        "n":             n_total,
        "cum_all":       cum_all,
        "first_all":     first_all,
        "cum_offense":   cum_off,
        "n_offense":     offense_totals,
        "first_by_Gender":  first_by_Gender,
        "n_by_Gender":      Gender_totals,
        "first_by_age":  first_by_age,
        "n_by_age":      age_totals,
        "first_by_race": first_by_race,
        "n_by_race":     race_totals,
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
# AGGREGATE RESULTS  (now also computes per-year SD)
# =============================================================================
def _aggregate_grouped(results, key, groups):
    """Aggregate a per-run, per-group, per-year first-arrest matrix into
    mean / 95% CI / SD arrays.  `key` names the worker dict field and `groups`
    is the canonical group list. Defensive against missing keys (older caches).
    """
    out_mean = {g: [] for g in groups}
    out_lo   = {g: [] for g in groups}
    out_hi   = {g: [] for g in groups}
    out_sd   = {g: [] for g in groups}
    for g in groups:
        for y in range(9):
            vals = [r.get(key, {}).get(g, [0]*9)[y] for r in results]
            m, lo, hi, sd = mean_ci_sd(vals)
            out_mean[g].append(m); out_lo[g].append(lo)
            out_hi[g].append(hi); out_sd[g].append(sd)
    return out_mean, out_lo, out_hi, out_sd


def aggregate(results):
    cum_mean, cum_lo, cum_hi, cum_sd = [], [], [], []
    for y in range(9):
        vals = [r["cum_all"][y] for r in results]
        m, lo, hi, sd = mean_ci_sd(vals)
        cum_mean.append(m); cum_lo.append(lo); cum_hi.append(hi); cum_sd.append(sd)

    first_mean, first_lo, first_hi, first_sd = [], [], [], []
    for y in range(9):
        vals = [r["first_all"][y] for r in results]
        m, lo, hi, sd = mean_ci_sd(vals)
        first_mean.append(m); first_lo.append(lo); first_hi.append(hi); first_sd.append(sd)

    off_mean = {g: [] for g in OFFENSE_GROUPS}
    off_lo   = {g: [] for g in OFFENSE_GROUPS}
    off_hi   = {g: [] for g in OFFENSE_GROUPS}
    off_sd   = {g: [] for g in OFFENSE_GROUPS}
    for g in OFFENSE_GROUPS:
        for y in range(9):
            vals = [r["cum_offense"][g][y] for r in results]
            m, lo, hi, sd = mean_ci_sd(vals)
            off_mean[g].append(m); off_lo[g].append(lo); off_hi[g].append(hi); off_sd[g].append(sd)

    # Per-stratifier annual first-arrest %.  Defensive: workers from older builds
    # may not have populated these fields; fall back to empty distributions.
    Gender_mean,  Gender_lo,  Gender_hi,  Gender_sd  = _aggregate_grouped(results, "first_by_Gender",  Gender_GROUPS)
    age_mean,  age_lo,  age_hi,  age_sd  = _aggregate_grouped(results, "first_by_age",  AGE_GROUPS)
    race_mean, race_lo, race_hi, race_sd = _aggregate_grouped(results, "first_by_race", RACE_GROUPS)

    return (cum_mean, cum_lo, cum_hi, cum_sd,
            first_mean, first_lo, first_hi, first_sd,
            off_mean, off_lo, off_hi, off_sd,
            Gender_mean, Gender_lo, Gender_hi, Gender_sd,
            age_mean, age_lo, age_hi, age_sd,
            race_mean, race_lo, race_hi, race_sd)

# =============================================================================
# CACHE I/O — for replot-only mode
# =============================================================================
def save_aggregated(path, agg, n_runs):
    """Persist aggregated arrays to JSON so charts can be regenerated
    without re-running the simulation."""
    (cum_mean, cum_lo, cum_hi, cum_sd,
     first_mean, first_lo, first_hi, first_sd,
     off_mean, off_lo, off_hi, off_sd,
     Gender_mean, Gender_lo, Gender_hi, Gender_sd,
     age_mean, age_lo, age_hi, age_sd,
     race_mean, race_lo, race_hi, race_sd) = agg
    payload = {
        "n_runs":     n_runs,
        "cum_mean":   cum_mean, "cum_lo": cum_lo, "cum_hi": cum_hi, "cum_sd": cum_sd,
        "first_mean": first_mean, "first_lo": first_lo, "first_hi": first_hi, "first_sd": first_sd,
        "off_mean":   off_mean, "off_lo": off_lo, "off_hi": off_hi, "off_sd": off_sd,
        "Gender_mean":   Gender_mean, "Gender_lo": Gender_lo, "Gender_hi": Gender_hi, "Gender_sd": Gender_sd,
        "age_mean":   age_mean, "age_lo": age_lo, "age_hi": age_hi, "age_sd": age_sd,
        "race_mean":  race_mean, "race_lo": race_lo, "race_hi": race_hi, "race_sd": race_sd,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Aggregated results -> {path}")

def load_aggregated(path):
    """Inverse of save_aggregated. Returns (agg_tuple, n_runs).
    Backwards-compatible: stratifier arrays default to zeros if the cache
    predates a particular chart, so older caches still replot the charts
    that don't depend on the missing data."""
    with open(path, "r") as f:
        p = json.load(f)
    _empty_Gender  = {s: [0.0]*9 for s in Gender_GROUPS}
    _empty_age  = {g: [0.0]*9 for g in AGE_GROUPS}
    _empty_race = {g: [0.0]*9 for g in RACE_GROUPS}
    agg = (p["cum_mean"], p["cum_lo"], p["cum_hi"], p["cum_sd"],
           p["first_mean"], p["first_lo"], p["first_hi"], p["first_sd"],
           p["off_mean"],   p["off_lo"],   p["off_hi"],   p["off_sd"],
           p.get("Gender_mean",  _empty_Gender),  p.get("Gender_lo",  _empty_Gender),
           p.get("Gender_hi",    _empty_Gender),  p.get("Gender_sd",  _empty_Gender),
           p.get("age_mean",  _empty_age),  p.get("age_lo",  _empty_age),
           p.get("age_hi",    _empty_age),  p.get("age_sd",  _empty_age),
           p.get("race_mean", _empty_race), p.get("race_lo", _empty_race),
           p.get("race_hi",   _empty_race), p.get("race_sd", _empty_race))
    return agg, p.get("n_runs", -1)

# =============================================================================
# STYLE — modern academic sans-serif stack
# =============================================================================
_FONT_FAMILY = "Source Sans Pro, Helvetica, Arial, sans-serif"

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=_FONT_FAMILY, color="#1A2B3C", size=12),
    paper_bgcolor="#FAFBFC",
    plot_bgcolor="#FAFBFC",
)
_C_ABM  = "#2F6FB2"
_C_BJS  = "#E07B39"
_C_GRID = "#E8EDF2"

# Distinct color for σ (run-to-run SD) visualization — purple-grey,
# distinguishable from ABM blue, BJS orange, and the four offense colors.
_C_SD       = "#7A3F99"
_C_SD_FILL  = "rgba(122, 63, 153, 0.22)"
_C_SD_LINE  = "rgba(122, 63, 153, 0.95)"

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
# GAP-LABEL HELPERS  (font family unified)
# =============================================================================
def _add_gap_label(fig, x, y, diff, row=None, col=None, fontsize=9,
                   prefix="Δ"):
    kw = {}
    if row is not None: kw["row"] = row
    if col is not None: kw["col"] = col
    fig.add_annotation(
        x=x, y=y,
        text=f"{prefix} {diff:+.1f}pp",
        showarrow=False,
        font=dict(size=fontsize, color=gap_colour(diff), family=_FONT_FAMILY),
        **kw,
    )

def _add_year_by_year_gaps(fig, years, abm_vals, bjs_vals, y_offset=4.0,
                            row=None, col=None, fontsize=9, prefix="Δ"):
    """Alternate above/below to prevent vertical stacking near convergence."""
    for i, yr in enumerate(years):
        diff = abm_vals[i] - bjs_vals[i]
        if i % 2 == 0:
            anchor = max(abm_vals[i], bjs_vals[i]) + y_offset
        else:
            anchor = min(abm_vals[i], bjs_vals[i]) - y_offset - 1.5
        _add_gap_label(fig, x=yr, y=anchor,
                       diff=diff, row=row, col=col,
                       fontsize=fontsize, prefix=prefix)

def _gap_legend_annotation(fig, x=0.5, y=-0.22, xref="paper", yref="paper"):
    """Horizontal legend strip below the x-axis explaining gap-color thresholds.
    Default y=-0.22 is below the trace legend (which sits around y=-0.14),
    keeping them from overlapping."""
    fig.add_annotation(
        x=x, y=y, xref=xref, yref=yref,
        text=(f"<span style='color:{COLOUR_GOOD}'>● within ±2pp (robust)</span>  "
              f"<span style='color:{COLOUR_WARN}'>● within ±5pp (acceptable)</span>  "
              f"<span style='color:{COLOUR_BAD}'>● beyond ±5pp (off-target)</span>"),
        showarrow=False, align="center",
        font=dict(size=10, family=_FONT_FAMILY),
    )

def _sd_summary_text(sd_array, label="ABM SD across runs"):
    """Compact one-liner summarizing per-year σ for ABM results."""
    mean_sd = sum(sd_array) / len(sd_array)
    max_sd  = max(sd_array)
    max_yr  = sd_array.index(max_sd) + 1
    return (f"<b>{label}:</b> mean σ = {mean_sd:.2f} pp "
            f"(max {max_sd:.2f} pp at Yr {max_yr})")


# =============================================================================
# CHART 1 — Cumulative rearrest Years 1-9 (aggregate)
# =============================================================================
def chart1_cumulative_by_year(cum_mean, cum_lo, cum_hi, cum_sd, out_dir):
    gaps = [cum_mean[i] - BJS_CUMULATIVE_ALL[i] for i in range(9)]
    mean_abs_gap = sum(abs(g) for g in gaps) / 9
    max_gap_idx = max(range(9), key=lambda i: abs(gaps[i]))
    max_gap_yr = BJS_YEARS[max_gap_idx]
    max_gap_val = gaps[max_gap_idx]

    fig = go.Figure()

    # ── ±1σ ribbon (run-to-run spread) ───────────────────────────────────
    # Drawn first so the (narrower) 95% CI band of the mean overlays it where
    # they intersect. CI uses σ/√n so it is always narrower than ±σ for n>1.
    sd_hi = [m + s for m, s in zip(cum_mean, cum_sd)]
    sd_lo = [m - s for m, s in zip(cum_mean, cum_sd)]
    fig.add_trace(go.Scatter(
        x=BJS_YEARS + BJS_YEARS[::-1],
        y=sd_hi + sd_lo[::-1],
        fill="toself", fillcolor=_C_SD_FILL,
        line=dict(color=_C_SD_LINE, width=1, dash="dot"),
        name="ABM ±1σ across runs", legendgroup="sd",
        hoverinfo="skip",
    ))

    # ── 95% CI band of the ABM mean ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=BJS_YEARS + BJS_YEARS[::-1],
        y=cum_hi + cum_lo[::-1],
        fill="toself", fillcolor="rgba(47,111,178,0.15)",
        line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=cum_mean, mode="lines+markers",
        name="Calibrated ABM",
        line=dict(color=_C_ABM, width=3),
        marker=dict(size=8, color=_C_ABM),
        error_y=dict(type="data", symmetric=False,
                     array=[h-m for m,h in zip(cum_mean, cum_hi)],
                     arrayminus=[m-l for m,l in zip(cum_mean, cum_lo)],
                     color=_C_ABM, thickness=1.5, width=5),
        customdata=cum_sd,
        hovertemplate=("<b>ABM Year %{x}</b><br>%{y:.1f}%"
                       "<br>σ across runs: %{customdata:.2f} pp<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=BJS_CUMULATIVE_ALL,
        mode="lines+markers", name="BJS empirical",
        line=dict(color=_C_BJS, width=2.5, dash="dot"),
        marker=dict(symbol="diamond", size=8, color=_C_BJS),
        hovertemplate="<b>BJS Year %{x}</b><br>%{y:.1f}%<extra></extra>",
    ))

    _add_year_by_year_gaps(fig, BJS_YEARS, cum_mean, BJS_CUMULATIVE_ALL,
                           y_offset=3.5, fontsize=9)

    # Summary box: bottom-right in paper coords so it doesn't collide with data
    fig.add_annotation(
        x=0.98, y=0.05, xref="paper", yref="paper",
        text=(f"<b>Mean |Δ| across 9 years:</b> {mean_abs_gap:.2f} pp<br>"
              f"<b>Largest gap:</b> Year {max_gap_yr} ({max_gap_val:+.1f} pp)<br>"
              f"{_sd_summary_text(cum_sd)}"),
        showarrow=False, align="right", xanchor="right", yanchor="bottom",
        font=dict(size=10, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Aggregate Cumulative Rearrest Rate by Follow-up Year</b><br>"
                  "<sup>Calibrated recidivism ABM vs. BJS NCJ 250975 empirical targets  |  "
                  "Δ (pp) = ABM − BJS at each year</sup><br>"
                  "<sup>Shaded blue band = 95% CI  |  "
                  "<span style='color:#7A3F99'>Purple band = ±1σ across runs</span></sup>"),
            font=dict(size=15, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        xaxis=dict(title="Years since release",
                   tickvals=BJS_YEARS, showgrid=False, range=[0.5, 9.5]),
        yaxis=dict(title="Cumulative rearrest rate (%)",
                   range=[28, 102], gridcolor=_C_GRID, zeroline=False),
        legend=dict(x=0.04, y=0.97,
                    font=dict(size=9, family=_FONT_FAMILY),
                    itemsizing="constant",
                    itemwidth=30,
                    tracegroupgap=2,
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#CCCCCC", borderwidth=1),
        height=520,
        margin=dict(t=80, b=90, l=70, r=40),
    )
    _gap_legend_annotation(fig, x=0.5, y=-0.18)
    _save(fig, os.path.join(out_dir, "chart1_cumulative_by_year.png"))


# =============================================================================
# CHART 2 — Cumulative rearrest at 3yr / 6yr / 9yr (bar chart)
# =============================================================================
def chart2_cumulative_bar(cum_mean, cum_lo, cum_hi, cum_sd, out_dir):
    windows  = ["3-Year", "6-Year", "9-Year"]
    y_idx    = [2, 5, 8]
    abm_vals = [cum_mean[i] for i in y_idx]
    abm_lo   = [cum_lo[i]   for i in y_idx]
    abm_hi   = [cum_hi[i]   for i in y_idx]
    abm_sd   = [cum_sd[i]   for i in y_idx]
    bjs_vals = [BJS_TARGETS[k] for k in [3, 6, 9]]

    gaps = [abm_vals[i] - bjs_vals[i] for i in range(3)]
    mae = sum(abs(g) for g in gaps) / 3

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Calibrated ABM (95% CI)", x=windows, y=abm_vals,
        marker_color=_C_ABM, marker_line_width=0,
        error_y=dict(type="data", symmetric=False,
                     array=[h-m for m,h in zip(abm_vals, abm_hi)],
                     arrayminus=[m-l for m,l in zip(abm_vals, abm_lo)],
                     color=_C_ABM, thickness=2, width=6),
        text=[f"{v:.1f}%" for v in abm_vals],
        textposition="outside",
        textfont=dict(size=11, family=_FONT_FAMILY),
        customdata=abm_sd,
        hovertemplate=("<b>ABM %{x}</b><br>%{y:.1f}%"
                       "<br>σ across runs: %{customdata:.2f} pp<extra></extra>"),
    ))
    fig.add_trace(go.Bar(
        name="BJS empirical target", x=windows, y=bjs_vals,
        marker_color=_C_BJS, marker_line_width=0,
        text=[f"{v:.1f}%" for v in bjs_vals], textposition="outside",
        textfont=dict(size=11, family=_FONT_FAMILY),
    ))

    # ── Purple ±σ whiskers on each ABM bar ───────────────────────────────────
    # In barmode="group" with 2 traces, the ABM bar sits LEFT of category center.
    # Use a negative offset so the σ whisker lands on the ABM bar's right edge,
    # not in the gap between bars.
    fig.add_trace(go.Bar(
        x=windows, y=abm_vals,
        width=0.001, offset=-0.18,
        marker=dict(color="rgba(0,0,0,0)"),
        error_y=dict(type="data", array=abm_sd,
                     color=_C_SD_LINE, thickness=3, width=12),
        showlegend=False,
        customdata=abm_sd,
        hovertemplate=("<b>ABM %{x} σ</b><br>"
                       "σ across runs: %{customdata:.2f} pp<extra></extra>"),
    ))
    # Phantom Scatter trace gives σ whisker a proper line-style legend swatch
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=_C_SD_LINE, width=3),
        name="ABM ±1σ across runs", legendgroup="sd",
        hoverinfo="skip", showlegend=True,
    ))

    # Δ labels well above the bar value labels (which sit ~3pp above bar tops)
    for w, av, bv in zip(windows, abm_vals, bjs_vals):
        diff = av - bv
        label_y = max(av, bv) + 22
        _add_gap_label(fig, x=w, y=label_y, diff=diff, fontsize=11)

    # Summary: in the bottom margin strip so it doesn't overlap any bar.
    # Sits above the gap-color legend with a clean vertical gap.
    fig.add_annotation(
        x=0.5, y=-0.18, xref="paper", yref="paper",
        text=(f"<b>Mean Absolute Error:</b> {mae:.2f} pp  |  "
              f"{_sd_summary_text(abm_sd, label='ABM SD at anchors')}  |  "
              f"<span style='color:#555'>Per-window σ: 3y={abm_sd[0]:.2f}, "
              f"6y={abm_sd[1]:.2f}, 9y={abm_sd[2]:.2f} pp</span><br>"
              f"<i>Anchor windows used for Stage 1 calibration</i>"),
        showarrow=False, align="center", xanchor="center", yanchor="top",
        font=dict(size=10, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Cumulative Rearrest Rate at BJS Anchor Windows "
                  "(3-, 6-, and 9-Year Follow-up)</b><br>"
                  "<sup>Calibrated recidivism ABM vs. BJS NCJ 250975  |  "
                  "Δ (pp) = ABM − BJS</sup>"),
            font=dict(size=15, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        barmode="group", bargap=0.30, bargroupgap=0.08,
        yaxis=dict(title="Cumulative rearrest rate (%)",
                   range=[0, 122], gridcolor=_C_GRID, zeroline=False),
        xaxis=dict(title="Follow-up window", showgrid=False),
        legend=dict(orientation="h", x=0.5, y=1.04,
                    xanchor="center", yanchor="bottom",
                    font=dict(size=10, family=_FONT_FAMILY),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#CCCCCC", borderwidth=1),
        height=540,
        margin=dict(t=110, b=160, l=70, r=40),
    )
    _gap_legend_annotation(fig, x=0.5, y=-0.34)
    _save(fig, os.path.join(out_dir, "chart2_cumulative_bar.png"))


# =============================================================================
# CHART 3 — Cumulative rearrest Years 1-9 by offense type (4-panel)
#           - per-subplot MAE legend boxes REMOVED
#           - σ summary added
#           - bottom legend separated from gap legend (no overlap)
# =============================================================================
def chart3_cumulative_by_offense(off_mean, off_lo, off_hi, off_sd, out_dir):
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"<b>{g}</b>" for g in OFFENSE_GROUPS],
        shared_yaxes=True,
        horizontal_spacing=0.03,
    )

    for col_idx, g in enumerate(OFFENSE_GROUPS, start=1):
        colour  = OFFENSE_COLOURS[g]
        abm_m   = off_mean[g]
        abm_h   = off_hi[g]
        abm_l   = off_lo[g]
        abm_sd  = off_sd[g]
        bjs_off = BJS_BY_OFFENSE[g]
        diffs   = [abm_m[i] - bjs_off[i] for i in range(9)]

        # 95% CI band
        fig.add_trace(go.Scatter(
            x=BJS_YEARS + BJS_YEARS[::-1],
            y=abm_h + abm_l[::-1],
            fill="toself",
            fillcolor=(f"rgba({int(colour[1:3],16)},"
                       f"{int(colour[3:5],16)},"
                       f"{int(colour[5:7],16)},0.18)"),
            line_color="rgba(0,0,0,0)",
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col_idx)

        # ABM — solid line
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=abm_m, mode="lines+markers",
            name=g, legendgroup=g, showlegend=False,
            line=dict(color=colour, width=2.5),        # 3.0 -> 2.5
            marker=dict(size=7, color=colour),          # 9 -> 7
            customdata=abm_sd,
            hovertemplate=(f"<b>ABM {g} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}%<br>σ: %{{customdata:.2f}} pp"
                           f"<extra></extra>"),
        ), row=1, col=col_idx)

        # BJS — dashed line, open diamond
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=bjs_off, mode="lines+markers",
            name=f"BJS — {g}", legendgroup=g, showlegend=False,
            line=dict(color=colour, width=1.8, dash="dash"),  # 2.2 -> 1.8
            marker=dict(symbol="diamond-open", size=7,         # 9 -> 7
                        color=colour, line=dict(width=1.5)),   # 2.0 -> 1.5
            hovertemplate=(f"<b>BJS {g} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}%<extra></extra>"),
        ), row=1, col=col_idx)

        # Delta labels — tighter vertical offsets to avoid overlap at reduced height
        for j, anchor_yr in enumerate(ANCHOR_YEARS):
            i    = anchor_yr - 1
            diff = diffs[i]
            anchor_y = (max(abm_m[i], bjs_off[i]) + 2.5 if j % 2 == 0   # 4.5 -> 2.5
                        else min(abm_m[i], bjs_off[i]) - 3.5)             # 6.0 -> 3.5
            _add_gap_label(
                fig, x=anchor_yr, y=anchor_y, diff=diff,
                row=1, col=col_idx, fontsize=11,        # 14 -> 11
            )

    fig.update_xaxes(
        title_text="Years since release",
        title_font=dict(size=13, family=_FONT_FAMILY),  # 16 -> 13
        tickfont=dict(size=11, family=_FONT_FAMILY),    # 14 -> 11
        tickvals=BJS_YEARS,
        showgrid=False,
        range=[0.5, 9.5],
    )
    fig.update_yaxes(
        range=[23, 108],
        gridcolor=_C_GRID,
        zeroline=False,
        tickfont=dict(size=11, family=_FONT_FAMILY),    # 14 -> 11
    )
    fig.update_yaxes(
        title_text="Cumulative rearrest rate (%)",
        title_font=dict(size=13, family=_FONT_FAMILY),  # 16 -> 13
        row=1, col=1,
    )

    # Subplot titles scaled to match height
    for ann in fig.layout.annotations:
        ann.font = dict(size=14, family=_FONT_FAMILY, color="#1A2B3C")  # 18 -> 14

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Stage 3 — Offense-Stratified Calibration: "
                  "ABM vs. BJS Empirical Targets</b><br>"),
            font=dict(size=15, family=_FONT_FAMILY),    # 20 -> 15
            x=0.5, xanchor="center",
        ),
        width=980,
        height=420,
        margin=dict(t=100, b=50, l=90, r=60),   # r: 20 -> 60 to stop right truncation
    )

    _save(fig, os.path.join(out_dir, "chart3_cumulative_by_offense.png"))


# =============================================================================
# CHART 4 — Cumulative rearrest at 3yr / 6yr / 9yr by offense type
# =============================================================================
def chart4_bar_by_offense(off_mean, off_lo, off_hi, off_sd, out_dir):
    windows = [3, 6, 9]
    y_idx   = [2, 5, 8]
    x_lbls  = ["3-Year", "6-Year", "9-Year"]

    grid_gaps = []
    for g in OFFENSE_GROUPS:
        for j in y_idx:
            grid_gaps.append(off_mean[g][j] - BJS_BY_OFFENSE[g][j])
    overall_mae = sum(abs(d) for d in grid_gaps) / len(grid_gaps)

    # Mean σ across all (offense × anchor) cells
    all_sd = [off_sd[g][j] for g in OFFENSE_GROUPS for j in y_idx]
    overall_mean_sd = sum(all_sd) / len(all_sd)

    fig = go.Figure()
    bw  = 0.15
    x_base = [1, 2, 3]

    # Single legend entry for "ABM ±1σ across runs"
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=_C_SD_LINE, width=2.8),
        name="ABM ±1σ across runs", legendgroup="sd",
        hoverinfo="skip", showlegend=True,
    ))

    # First pass: draw bars + diamonds + σ whiskers; collect per-cell info
    # so the second pass can place Δ labels with rank-aware staggering.
    cells = []  # list of (window_idx, offense_idx, x_pos, av, bv, diff, top)
    for i, g in enumerate(OFFENSE_GROUPS):
        colour  = OFFENSE_COLOURS[g]
        abm_m   = [off_mean[g][j] for j in y_idx]
        abm_h   = [off_hi[g][j]   for j in y_idx]
        abm_l   = [off_lo[g][j]   for j in y_idx]
        abm_sd  = [off_sd[g][j]   for j in y_idx]
        bjs_v   = [BJS_BY_OFFENSE[g][j] for j in y_idx]
        offset  = (i - 1.5) * bw * 2.2
        xs      = [x + offset for x in x_base]

        fig.add_trace(go.Bar(
            name=f"ABM — {g}", x=xs, y=abm_m,
            width=bw*1.8, marker_color=colour, marker_line_width=0,
            legendgroup=g,
            error_y=dict(type="data", symmetric=False,
                         array=[h-m for m,h in zip(abm_m, abm_h)],
                         arrayminus=[m-l for m,l in zip(abm_m, abm_l)],
                         color=colour, thickness=1.5, width=4),
            customdata=abm_sd,
            hovertemplate=(f"<b>ABM {g}</b><br>%{{y:.1f}}%"
                           f"<br>σ across runs: %{{customdata:.2f}} pp<extra></extra>"),
        ))

        # Purple σ whisker offset slightly right of the ABM bar center
        sigma_x = [x + bw * 0.55 for x in xs]
        fig.add_trace(go.Scatter(
            x=sigma_x, y=abm_m,
            mode="markers",
            marker=dict(symbol="line-ew", size=8,
                        color=_C_SD_LINE, line=dict(width=2, color=_C_SD_LINE)),
            error_y=dict(type="data", array=abm_sd,
                         color=_C_SD_LINE, thickness=2.2, width=5),
            legendgroup="sd", showlegend=False,
            customdata=abm_sd,
            hovertemplate=(f"<b>{g} σ</b><br>"
                           f"σ across runs: %{{customdata:.2f}} pp<extra></extra>"),
        ))

        fig.add_trace(go.Scatter(
            name=f"BJS — {g}", x=xs, y=bjs_v,
            mode="markers", legendgroup=f"bjs_{g}",
            marker=dict(symbol="diamond", size=10, color=colour,
                        line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>BJS {g}</b><br>%{{y:.1f}}%<extra></extra>",
        ))

        for k, (x_pos, av, bv) in enumerate(zip(xs, abm_m, bjs_v)):
            cells.append((k, i, x_pos, av, bv, av - bv, max(av, bv)))

    # Second pass: stagger Δ labels by rank-by-height within each window so
    # each window's four labels sit at four distinct y-levels and never stack.
    # Thin guide lines connect each label down to its bar so attribution is clear.
    BASE_OFFSET = 4.0    # pp above the tallest bar top in the window
    STEP        = 6.5    # pp between successive label rows (bigger = clearer)
    for window_idx in range(3):
        in_win = [c for c in cells if c[0] == window_idx]
        # Sort by bar top: shortest bar gets the lowest label slot, etc.
        in_win.sort(key=lambda c: c[6])
        win_max_top = max(c[6] for c in in_win)
        for slot, (_, i_off, x_pos, av, bv, diff, top) in enumerate(in_win):
            label_y = win_max_top + BASE_OFFSET + slot * STEP
            # Thin grey guide line from label down to the bar top
            fig.add_shape(
                type="line",
                x0=x_pos, x1=x_pos,
                y0=top + 0.5, y1=label_y - 0.8,
                line=dict(color="#BBBBBB", width=0.6, dash="dot"),
                layer="below",
            )
            _add_gap_label(fig, x=x_pos, y=label_y, diff=diff, fontsize=8)

    # Summary banner above the plot area — sits in the top margin between title and bars
    fig.add_annotation(
        x=0.5, y=1.02, xref="paper", yref="paper",
        text=(f"<b>Overall MAE (4 offenses × 3 windows):</b> {overall_mae:.2f} pp  |  "
              f"<b>Mean ABM σ across cells:</b> {overall_mean_sd:.2f} pp  |  "
              f"<b>Stage 3 calibration target:</b> BJS NCJ 250975 Table 7"),
        showarrow=False, align="center", xanchor="center", yanchor="bottom",
        font=dict(size=10, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Offense-Stratified Cumulative Rearrest at BJS Anchor Windows</b><br>"
                  "<sup>Calibrated ABM bars vs. BJS empirical diamonds  |  "
                  "Error bars = 95% CI  |  Δ (pp) = ABM − BJS  |  "
                  "Alper et al. (2018), NCJ 250975 Table 7</sup>"),
            font=dict(size=14, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        barmode="overlay",
        xaxis=dict(title="Follow-up window",
                   tickvals=x_base, ticktext=x_lbls,
                   showgrid=False, range=[0.4, 3.6]),
        yaxis=dict(title="Cumulative rearrest rate (%)",
                   range=[0, 125], gridcolor=_C_GRID, zeroline=False),
        # Right-side legend (no x-axis collision possible)
        legend=dict(x=1.02, y=0.99, font=dict(size=9, family=_FONT_FAMILY),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#CCCCCC", borderwidth=1),
        width=1200,
        height=580,
        margin=dict(t=110, b=100, l=70, r=170),
    )
    # Gap legend just below the x-axis title
    _gap_legend_annotation(fig, x=0.5, y=-0.18)
    _save(fig, os.path.join(out_dir, "chart4_cumulative_bar_by_offense.png"))


# =============================================================================
# CHART 5 — Non-cumulative annual first-arrest %  Years 1-9
#           - legend top-RIGHT
#           - σ summary added
# =============================================================================
def chart5_noncumulative(first_mean, first_lo, first_hi, first_sd, out_dir):
    yr1_abm = first_mean[0]
    yr9_abm = first_mean[8]
    yr1_bjs = BJS_FIRST_ARREST[0]
    yr9_bjs = BJS_FIRST_ARREST[8]
    decline_abm = yr1_abm - yr9_abm
    decline_bjs = yr1_bjs - yr9_bjs

    fig = go.Figure()

    # ── 95% CI band of the ABM mean ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=BJS_YEARS + BJS_YEARS[::-1],
        y=first_hi + first_lo[::-1],
        fill="toself", fillcolor="rgba(47,111,178,0.15)",
        line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        name="Calibrated ABM", x=BJS_YEARS, y=first_mean,
        marker_color=_C_ABM, marker_line_width=0, opacity=0.85,
        error_y=dict(type="data", symmetric=False,
                     array=[h-m for m,h in zip(first_mean, first_hi)],
                     arrayminus=[m-l for m,l in zip(first_mean, first_lo)],
                     color=_C_ABM, thickness=1.5, width=5),
        customdata=first_sd,
        hovertemplate=("<b>ABM Year %{x}</b><br>%{y:.1f}% first arrest"
                       "<br>σ across runs: %{customdata:.2f} pp<extra></extra>"),
    ))

    # ── Purple ±σ whiskers on each ABM bar ───────────────────────────────────
    # Per-bar whiskers (rather than a ribbon) read more clearly for bar charts
    # because each σ is anchored to a discrete bar value.
    fig.add_trace(go.Bar(
        x=BJS_YEARS, y=first_mean,
        width=0.001, offset=0.10,
        marker=dict(color="rgba(0,0,0,0)"),
        error_y=dict(type="data", array=first_sd,
                     color=_C_SD_LINE, thickness=2.8, width=8),
        showlegend=False,
        customdata=first_sd,
        hovertemplate=("<b>ABM Yr %{x} σ</b><br>"
                       "σ across runs: %{customdata:.2f} pp<extra></extra>"),
    ))
    # Phantom Scatter trace for legend swatch
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=_C_SD_LINE, width=3),
        name="ABM ±1σ across runs", legendgroup="sd",
        hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=BJS_FIRST_ARREST,
        mode="lines+markers", name="BJS empirical (Alper et al., 2018)",
        line=dict(color=_C_BJS, width=2.5, dash="dot"),
        marker=dict(symbol="diamond", size=8, color=_C_BJS),
        hovertemplate="<b>BJS Year %{x}</b><br>%{y:.1f}% first arrest<extra></extra>",
    ))

    # Δ labels alternating above/below
    for i, yr in enumerate(BJS_YEARS):
        diff = first_mean[i] - BJS_FIRST_ARREST[i]
        if i % 2 == 0:
            anchor = max(first_mean[i], BJS_FIRST_ARREST[i]) + 2.2
        else:
            anchor = min(first_mean[i], BJS_FIRST_ARREST[i]) - 3.5
        _add_gap_label(fig, x=yr, y=anchor, diff=diff, fontsize=9)

    # Desistance summary: middle-right, where bars are tiny and won't be obscured
    fig.add_annotation(
        x=0.98, y=0.55, xref="paper", yref="paper",
        text=(f"<b>Desistance decline (Yr 1 → Yr 9):</b><br>"
              f"ABM: {decline_abm:.1f} pp  |  BJS: {decline_bjs:.1f} pp<br>"
              f"<i>Consistent with age-graded desistance<br>"
              f"(Sampson &amp; Laub, 2003; Kurlychek et al., 2006)</i><br>"
              f"{_sd_summary_text(first_sd)}"),
        showarrow=False, align="right", xanchor="right", yanchor="top",
        font=dict(size=9.5, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    y_max = max(first_mean + BJS_FIRST_ARREST) + 14

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Annual First-Arrest Rate: Hazard Distribution "
                  "Across 9-Year Follow-up</b><br>"
                  "<sup>Percentage of released prisoners whose first rearrest occurred in each year  |  "
                  "Blue whiskers = 95% CI  |  "
                  "<span style='color:#7A3F99'>Purple whiskers = ±1σ across runs</span></sup>"),
            font=dict(size=14, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        xaxis=dict(title="Year since release",
                   tickvals=BJS_YEARS, showgrid=False),
        yaxis=dict(title="Share of cohort with first rearrest in year (%)",
                   range=[-2, y_max], gridcolor=_C_GRID, zeroline=False),
        # Legend in TOP-RIGHT
        legend=dict(x=0.98, y=0.97,
                    xanchor="right", yanchor="top",
                    font=dict(size=9, family=_FONT_FAMILY),
                    itemsizing="constant", itemwidth=30, tracegroupgap=2,
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="#CCCCCC", borderwidth=1),
        height=520,
        margin=dict(t=80, b=90, l=70, r=40),
    )
    _gap_legend_annotation(fig, x=0.5, y=-0.18)
    _save(fig, os.path.join(out_dir, "chart5_noncumulative_first_arrest.png"))


# =============================================================================
# CHART 6 — Desistance survival curve  (% NOT yet arrested by year)
#
# Survival = 100 − cumulative arrest rate.  This is mathematically a flip of
# Chart 1 but is the canonical view used by Alper, Durose & Markman (2018,
# NCJ 250975, Figure 4).  Surfacing it directly mirrors the BJS report and
# makes the calibration story easy to communicate at a glance: the closer
# the ABM curve hugs the BJS curve (and stays inside the ±2pp green band),
# the better the desistance shape is reproduced.
# =============================================================================
_C_BJS_GREEN      = "#1F8B4C"                    # BJS green (matches Fig. 4 styling)
_C_BJS_GREEN_FILL = "rgba(31, 139, 76, 0.13)"    # ±2pp acceptance band
_C_ABM_SD_FILL    = "rgba(47, 111, 178, 0.18)"   # ABM σ band (light blue)

def chart6_desistance_survival(cum_mean, cum_sd, out_dir):
    # Survival = 100 − cumulative
    abm_surv     = [100 - m for m in cum_mean]
    bjs_surv     = [100 - v for v in BJS_CUMULATIVE_ALL]
    # σ on the survival scale is identical to σ on the arrest scale
    # (a constant offset doesn't change spread)
    abm_surv_hi  = [s + sd for s, sd in zip(abm_surv, cum_sd)]
    abm_surv_lo  = [s - sd for s, sd in zip(abm_surv, cum_sd)]
    # BJS ±2pp acceptance band (matches the green-shaded zone in the example)
    bjs_band_hi  = [v + 2.0 for v in bjs_surv]
    bjs_band_lo  = [v - 2.0 for v in bjs_surv]

    # Δ = ABM − BJS on the survival scale (positive = ABM understates rearrest)
    deltas       = [abm_surv[i] - bjs_surv[i] for i in range(9)]
    mean_abs_gap = sum(abs(d) for d in deltas) / 9
    max_idx      = max(range(9), key=lambda i: abs(deltas[i]))

    # Ordinal x-tick labels: 1st, 2nd, 3rd, 4th, …, 9th
    def _ordinal(n):
        if 10 <= n % 100 <= 20: suffix = "th"
        else: suffix = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    x_labels = [_ordinal(y) for y in BJS_YEARS]

    fig = go.Figure()

    # ── BJS ±2pp target band (green, drawn first, lowest layer) ─────────────
    fig.add_trace(go.Scatter(
        x=BJS_YEARS + BJS_YEARS[::-1],
        y=bjs_band_hi + bjs_band_lo[::-1],
        fill="toself", fillcolor=_C_BJS_GREEN_FILL,
        line=dict(color="rgba(0,0,0,0)"),
        name="BJS ±2pp target band",
        hoverinfo="skip",
    ))

    # ── ABM ±1σ band (light blue) ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=BJS_YEARS + BJS_YEARS[::-1],
        y=abm_surv_hi + abm_surv_lo[::-1],
        fill="toself", fillcolor=_C_ABM_SD_FILL,
        line=dict(color="rgba(0,0,0,0)"),
        name="ABM ±1 SD across seeds",
        hoverinfo="skip",
    ))

    # ── BJS empirical (green, diamond markers) ─────────────────────────────
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=bjs_surv, mode="lines+markers",
        name="BJS NCJ 250975 (Fig. 4)",
        line=dict(color=_C_BJS_GREEN, width=3),
        marker=dict(symbol="diamond", size=11, color=_C_BJS_GREEN,
                    line=dict(color="white", width=1)),
        hovertemplate="<b>BJS Year %{x}</b><br>%{y:.1f}% not yet arrested<extra></extra>",
    ))

    # ── ABM calibrated (blue, circle markers) ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=BJS_YEARS, y=abm_surv, mode="lines+markers",
        name="ABM calibrated",
        line=dict(color=_C_ABM, width=2.8),
        marker=dict(size=9, color=_C_ABM,
                    line=dict(color="white", width=1)),
        customdata=cum_sd,
        hovertemplate=("<b>ABM Year %{x}</b><br>%{y:.1f}% not yet arrested"
                       "<br>σ across runs: %{customdata:.2f} pp<extra></extra>"),
    ))

    # ── Δ labels above the higher of the two curves at each year ───────────
    # Color follows the existing gap_colour() thresholds for consistency with
    # the rest of the chart suite.
    for i, yr in enumerate(BJS_YEARS):
        d = deltas[i]
        anchor_y = max(abm_surv[i], bjs_surv[i]) + 3.5
        fig.add_annotation(
            x=yr, y=anchor_y,
            text=f"<b>Δ {d:+.1f}%</b>",
            showarrow=False,
            font=dict(size=10, family=_FONT_FAMILY, color=gap_colour(d)),
        )

    # ── Title + layout ─────────────────────────────────────────────────────
    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Desistance Survival Curve: ABM vs BJS NCJ 250975 (Figure 4)</b><br>"
                  "<sup>Percent of released prisoners NOT yet arrested at each year  |  "
                  "Source: Alper, Durose &amp; Markman (2018)</sup><br>"
                  f"<sup>Mean |Δ| across 9 years = {mean_abs_gap:.2f} pp  |  "
                  f"Largest gap: Year {BJS_YEARS[max_idx]} ({deltas[max_idx]:+.1f} pp)  |  "
                  f"<span style='color:{_C_SD}'>{_sd_summary_text(cum_sd, label='ABM SD across runs')}</span></sup>"),
            font=dict(size=15, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        xaxis=dict(
            title="Year after release",
            tickvals=BJS_YEARS, ticktext=x_labels,
            showgrid=False, range=[0.5, 9.5],
        ),
        yaxis=dict(
            title="Percent of released prisoners not arrested since release",
            range=[0, 75],
            gridcolor=_C_GRID, zeroline=False,
            tickvals=list(range(0, 80, 10)),
            ticktext=[f"{v}%" for v in range(0, 80, 10)],
        ),
        legend=dict(
            x=0.98, y=0.97, xanchor="right", yanchor="top",
            font=dict(size=10, family=_FONT_FAMILY),
            itemsizing="constant", itemwidth=30, tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CCCCCC", borderwidth=1,
        ),
        height=560,
        margin=dict(t=110, b=70, l=80, r=40),
    )
    _save(fig, os.path.join(out_dir, "chart6_desistance_survival.png"))


# =============================================================================
# CHART 7 — Annual first-arrest rate by Gender (mirrors BJS Fig. 5)
#
# Two ABM curves (Male, Female) overlaid on the BJS Table 3 curves, with
# ±1σ shading on the ABM lines so seed-level variability is visible.  Styling
# echoes the BJS report: green for Male, grey for Female, ordinal x-tick
# labels (1st…9th), inline text callouts at the steepest part of each curve.
# =============================================================================
_C_BJS_MALE   = "#1F8B4C"      # green (matches BJS Fig. 5)
_C_BJS_FEMALE = "#7F8C8D"      # cool grey
_C_ABM_MALE   = "#2F6FB2"      # ABM blue (consistent with rest of suite)
_C_ABM_FEMALE = "#9B59B6"      # purple, distinct but harmonious

_C_ABM_MALE_FILL   = "rgba(47, 111, 178, 0.18)"
_C_ABM_FEMALE_FILL = "rgba(155,  89, 182, 0.18)"

_Gender_PALETTE = {
    "Male":   {"bjs": _C_BJS_MALE,   "abm": _C_ABM_MALE,   "fill": _C_ABM_MALE_FILL},
    "Female": {"bjs": _C_BJS_FEMALE, "abm": _C_ABM_FEMALE, "fill": _C_ABM_FEMALE_FILL},
}

def chart7_first_arrest_by_Gender(Gender_mean, Gender_sd, out_dir):
    # Defensive: if every value is zero (older cache, or sim agents lacked a
    # Gender attribute), explain in a single annotation instead of a blank chart.
    has_data = any(any(v > 0 for v in Gender_mean[s]) for s in Gender_GROUPS)

    # Ordinal labels matching BJS Fig. 5 styling
    def _ordinal(n):
        if 10 <= n % 100 <= 20: suffix = "th"
        else: suffix = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    x_labels = [_ordinal(y) for y in BJS_YEARS]

    fig = go.Figure()

    if not has_data:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=("<b>No Gender-stratified data available.</b><br>"
                  "<sup>Worker did not capture per-agent Gender (attribute "
                  "<code>Gender</code>/<code>gender</code> missing) or replot cache "
                  "predates Chart 7.  Re-run the simulation to populate.</sup>"),
            showarrow=False, align="center",
            font=dict(size=12, family=_FONT_FAMILY, color="#888888"),
        )
        fig.update_layout(**_LAYOUT,
            title=dict(text="<b>Chart 7 — Annual First-Arrest Rate by Sex</b>",
                       font=dict(size=15, family=_FONT_FAMILY),
                       x=0.5, xanchor="center"),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=400, margin=dict(t=60, b=40, l=40, r=40),
        )
        _save(fig, os.path.join(out_dir, "chart7_first_arrest_by_Gender.png"))
        return

    # Overall MAE per Gender (across the 9 yearly hazards)
    mae_by_Gender = {}
    for s in Gender_GROUPS:
        diffs = [Gender_mean[s][i] - BJS_FIRST_BY_Gender[s][i] for i in range(9)]
        mae_by_Gender[s] = sum(abs(d) for d in diffs) / 9

    # ── Plot order: BJS first (so ABM lines draw on top), with ABM σ ribbons ─
    for s in Gender_GROUPS:
        c = _Gender_PALETTE[s]

        # ABM ±1σ ribbon
        m  = Gender_mean[s]; sd = Gender_sd[s]
        hi = [a + b for a, b in zip(m, sd)]
        lo = [max(0.0, a - b) for a, b in zip(m, sd)]
        fig.add_trace(go.Scatter(
            x=BJS_YEARS + BJS_YEARS[::-1],
            y=hi + lo[::-1],
            fill="toself", fillcolor=c["fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name=f"ABM {s} ±1σ",
            legendgroup=f"abm_{s}", showlegend=True,
            hoverinfo="skip",
        ))

        # BJS line (thick, BJS color, dotted to match other charts in suite)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=BJS_FIRST_BY_Gender[s],
            mode="lines+markers",
            name=f"BJS {s} (Table 3)",
            line=dict(color=c["bjs"], width=3, dash="dot"),
            marker=dict(symbol="diamond", size=8, color=c["bjs"]),
            legendgroup=f"bjs_{s}",
            hovertemplate=(f"<b>BJS {s} Year %{{x}}</b><br>"
                           f"%{{y:.2f}}% first arrest<extra></extra>"),
        ))

        # ABM line (solid, ABM color)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=Gender_mean[s],
            mode="lines+markers",
            name=f"ABM {s}",
            line=dict(color=c["abm"], width=2.8),
            marker=dict(size=8, color=c["abm"],
                        line=dict(color="white", width=1)),
            legendgroup=f"abm_{s}",
            customdata=Gender_sd[s],
            hovertemplate=(f"<b>ABM {s} Year %{{x}}</b><br>"
                           f"%{{y:.2f}}% first arrest"
                           f"<br>σ across runs: %{{customdata:.2f}} pp<extra></extra>"),
        ))

    # ── Inline callout labels mirroring BJS Fig. 5 ("Male" / "Female" tags) ─
    # Placed near the Year 2 dip where the curves diverge most clearly,
    # offset so they don't collide with Δ labels at Year 1.
    callouts = [
        # (label, x, y_anchor_year_index, dy)
        ("Male",   1.6, BJS_FIRST_BY_Gender["Male"][1],   8.0),   # above year 2
        ("Female", 1.6, BJS_FIRST_BY_Gender["Female"][1], -5.0),  # below year 2
    ]
    for label, x_pos, y_anchor, dy in callouts:
        c = _Gender_PALETTE[label]
        fig.add_annotation(
            x=x_pos, y=y_anchor + dy,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=13, family=_FONT_FAMILY, color=c["bjs"]),
            xanchor="left",
        )

    # ── Δ labels at every year (ABM − BJS), color-coded by gap_colour ──────
    for s in Gender_GROUPS:
        for i, yr in enumerate(BJS_YEARS):
            d = Gender_mean[s][i] - BJS_FIRST_BY_Gender[s][i]
            # Stagger: Male labels above the higher of the two; Female below
            if s == "Male":
                anchor_y = max(Gender_mean[s][i], BJS_FIRST_BY_Gender[s][i]) + 1.6
            else:
                anchor_y = min(Gender_mean[s][i], BJS_FIRST_BY_Gender[s][i]) - 2.0
            _add_gap_label(fig, x=yr, y=anchor_y, diff=d, fontsize=8)

    # ── Summary box ─────────────────────────────────────────────────────────
    fig.add_annotation(
        x=0.98, y=0.97, xref="paper", yref="paper",
        text=(f"<b>Mean |Δ| across 9 years</b><br>"
              f"Male: {mae_by_Gender['Male']:.2f} pp  |  "
              f"Female: {mae_by_Gender['Female']:.2f} pp<br>"
              f"<b>9-year cumulative (BJS Table 3):</b><br>"
              f"Male: {BJS_CUM9_BY_Gender['Male']:.1f}%  |  "
              f"Female: {BJS_CUM9_BY_Gender['Female']:.1f}%<br>"
              f"<span style='color:{_C_SD}'>"
              f"<b>ABM σ across runs (mean):</b> "
              f"M={sum(Gender_sd['Male'])/9:.2f}, "
              f"F={sum(Gender_sd['Female'])/9:.2f} pp</span>"),
        showarrow=False, align="right", xanchor="right", yanchor="top",
        font=dict(size=10, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    # y-range: a touch above the highest Year-1 hazard, with breathing room
    y_top = max(max(BJS_FIRST_BY_Gender[s][0] for s in Gender_GROUPS),
                max(Gender_mean[s][0] + Gender_sd[s][0] for s in Gender_GROUPS))

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Annual First-Arrest Rate by Sex: ABM vs BJS NCJ 250975 (Figure 5)</b><br>"
                  "<sup>Percent of released prisoners whose first rearrest occurred in each year, by Gender  |  "
                  "Source: Alper, Durose &amp; Markman (2018), Table 3</sup>"),
            font=dict(size=15, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        xaxis=dict(
            title="Year of first arrest",
            tickvals=BJS_YEARS, ticktext=x_labels,
            showgrid=False, range=[0.5, 9.5],
        ),
        yaxis=dict(
            title="Percent of released prisoners",
            range=[-2, y_top + 8],
            gridcolor=_C_GRID, zeroline=False,
            ticksuffix="%",
        ),
        legend=dict(
            x=0.98, y=0.55, xanchor="right", yanchor="top",
            font=dict(size=9, family=_FONT_FAMILY),
            itemsizing="constant", itemwidth=30, tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CCCCCC", borderwidth=1,
        ),
        height=560,
        margin=dict(t=90, b=70, l=80, r=40),
    )
    _save(fig, os.path.join(out_dir, "chart7_first_arrest_by_Gender.png"))


# =============================================================================
# CHART 8 — Cumulative rearrest rate by Gender (Years 1-9)
#
# Companion to Chart 7: where Chart 7 shows per-year first-arrest hazards, this
# chart shows the running cumulative arrest curve by Gender.  Mathematically,
# cumulative_y = Σ first_arrest_1..y, so it is derivable from the same data
# the worker already collects — no extra simulation work is required.
#
# This view is the natural way to read "what fraction of men/women have been
# rearrested by year N", and lets the validator confirm that the simulated
# 9-year cumulative endpoint matches the BJS Table 3 totals
# (Male 84.2%, Female 76.8%).
# =============================================================================
def chart8_cumulative_by_Gender(Gender_mean, Gender_sd, out_dir):
    # Defensive: if every value is zero, render an explainer like Chart 7 does
    has_data = any(any(v > 0 for v in Gender_mean[s]) for s in Gender_GROUPS)

    def _ordinal(n):
        if 10 <= n % 100 <= 20: suffix = "th"
        else: suffix = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    x_labels = [_ordinal(y) for y in BJS_YEARS]

    fig = go.Figure()

    if not has_data:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=("<b>No Gender-stratified data available.</b><br>"
                  "<sup>Worker did not capture per-agent Gender (attribute "
                  "<code>Gender</code>/<code>gender</code> missing) or replot cache "
                  "predates Chart 8.  Re-run the simulation to populate.</sup>"),
            showarrow=False, align="center",
            font=dict(size=12, family=_FONT_FAMILY, color="#888888"),
        )
        fig.update_layout(**_LAYOUT,
            title=dict(text="<b>Chart 8 — Cumulative Rearrest Rate by Sex</b>",
                       font=dict(size=15, family=_FONT_FAMILY),
                       x=0.5, xanchor="center"),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=400, margin=dict(t=60, b=40, l=40, r=40),
        )
        _save(fig, os.path.join(out_dir, "chart8_cumulative_by_Gender.png"))
        return

    # ── Derive cumulative arrays from per-year first-arrest hazards ──────────
    # ABM cumulative = running sum of ABM first-arrest hazards
    # BJS cumulative = running sum of BJS first-arrest hazards (Table 3)
    abm_cum = {}
    bjs_cum = {}
    abm_cum_sd = {}  # cumulative-scale SD bound, see note below
    for s in Gender_GROUPS:
        m  = Gender_mean[s]
        sd = Gender_sd[s]
        bjs = BJS_FIRST_BY_Gender[s]
        abm_cum[s] = [sum(m[:i+1])   for i in range(9)]
        bjs_cum[s] = [sum(bjs[:i+1]) for i in range(9)]
        # Conservative cumulative SD ≈ sqrt(Σ σᵢ²).  This treats the per-year
        # hazards as independent, which slightly overstates spread (years are
        # actually weakly negatively correlated within a cohort), but it is the
        # only honest reconstruction available from already-aggregated data and
        # gives the right qualitative picture of run-to-run variability.
        abm_cum_sd[s] = [math.sqrt(sum(x*x for x in sd[:i+1])) for i in range(9)]

    # Per-year and 9-year endpoint diagnostics
    diffs       = {s: [abm_cum[s][i] - bjs_cum[s][i] for i in range(9)] for s in Gender_GROUPS}
    mae_by_Gender  = {s: sum(abs(d) for d in diffs[s]) / 9 for s in Gender_GROUPS}
    endpoint_d  = {s: abm_cum[s][8] - bjs_cum[s][8] for s in Gender_GROUPS}

    # ── Plot order: σ ribbon → BJS dotted → ABM solid ────────────────────────
    for s in Gender_GROUPS:
        c   = _Gender_PALETTE[s]
        m   = abm_cum[s]
        sd  = abm_cum_sd[s]
        bjs = bjs_cum[s]
        hi  = [a + b for a, b in zip(m, sd)]
        lo  = [max(0.0, a - b) for a, b in zip(m, sd)]

        # ABM ±1σ ribbon (cumulative scale)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS + BJS_YEARS[::-1],
            y=hi + lo[::-1],
            fill="toself", fillcolor=c["fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name=f"ABM {s} ±1σ",
            legendgroup=f"sd_{s}",
            hoverinfo="skip",
        ))

        # BJS curve (dotted, diamond markers)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=bjs, mode="lines+markers",
            name=f"BJS {s} (Table 3)",
            legendgroup=f"bjs_{s}",
            line=dict(color=c["bjs"], width=2.6, dash="dot"),
            marker=dict(symbol="diamond", size=10, color=c["bjs"],
                        line=dict(color="white", width=1)),
            hovertemplate=(f"<b>BJS {s} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}% cumulative<extra></extra>"),
        ))

        # ABM curve (solid, circle markers)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=m, mode="lines+markers",
            name=f"ABM {s}",
            legendgroup=f"abm_{s}",
            line=dict(color=c["abm"], width=2.8),
            marker=dict(size=8, color=c["abm"],
                        line=dict(color="white", width=1)),
            customdata=sd,
            hovertemplate=(f"<b>ABM {s} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}% cumulative"
                           f"<br>cumulative σ ≈ %{{customdata:.2f}} pp<extra></extra>"),
        ))

    # ── Δ labels at anchor years (3, 6, 9) only — full-year labels overlap ──
    # Stagger Male above curves, Female below curves so they don't collide.
    for anchor_yr in ANCHOR_YEARS:
        i = anchor_yr - 1
        # Male: above the higher of (ABM, BJS) at that year
        m_y = max(abm_cum["Male"][i], bjs_cum["Male"][i]) + 3.0
        _add_gap_label(fig, x=anchor_yr, y=m_y, diff=diffs["Male"][i], fontsize=9,
                       prefix="M Δ")
        # Female: below the lower of (ABM, BJS) at that year
        f_y = min(abm_cum["Female"][i], bjs_cum["Female"][i]) - 4.0
        _add_gap_label(fig, x=anchor_yr, y=f_y, diff=diffs["Female"][i], fontsize=9,
                       prefix="F Δ")

    # ── Summary box: bottom-right where the curves haven't reached ───────────
    fig.add_annotation(
        x=0.98, y=0.05, xref="paper", yref="paper",
        text=(f"<b>Mean |Δ| across 9 yrs:</b>  "
              f"Male: {mae_by_Gender['Male']:.2f} pp  |  "
              f"Female: {mae_by_Gender['Female']:.2f} pp<br>"
              f"<b>9-yr endpoint Δ (ABM − BJS):</b>  "
              f"Male: {endpoint_d['Male']:+.2f} pp  |  "
              f"Female: {endpoint_d['Female']:+.2f} pp<br>"
              f"<b>BJS 9-yr cumulative (Table 3):</b>  "
              f"Male: {BJS_CUM9_BY_Gender['Male']:.1f}%  |  "
              f"Female: {BJS_CUM9_BY_Gender['Female']:.1f}%<br>"
              f"<span style='color:{_C_SD}'>"
              f"<i>Cumulative σ shown as √(Σσ²) bound</i></span>"),
        showarrow=False, align="right", xanchor="right", yanchor="bottom",
        font=dict(size=10, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    # y-axis: lower bound near the smallest first-year value to remove dead space
    y_min = min(min(bjs_cum["Female"]), min(abm_cum["Female"])) - 5
    y_top = max(max(bjs_cum["Male"]), max(abm_cum["Male"][i] + abm_cum_sd["Male"][i]
                                          for i in range(9)))

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=("<b>Cumulative Rearrest Rate by Sex: ABM vs BJS NCJ 250975 (Table 3)</b><br>"
                  "<sup>Running cumulative arrest fraction by year and Gender  |  "
                  "Source: Alper, Durose &amp; Markman (2018)</sup>"),
            font=dict(size=15, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        xaxis=dict(
            title="Years since release",
            tickvals=BJS_YEARS, ticktext=x_labels,
            showgrid=False, range=[0.5, 9.5],
        ),
        yaxis=dict(
            title="Cumulative rearrest rate",
            range=[max(0, y_min), y_top + 5],
            gridcolor=_C_GRID, zeroline=False,
            ticksuffix="%",
        ),
        legend=dict(
            x=0.02, y=0.97, xanchor="left", yanchor="top",
            font=dict(size=9, family=_FONT_FAMILY),
            itemsizing="constant", itemwidth=30, tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CCCCCC", borderwidth=1,
        ),
        height=560,
        margin=dict(t=90, b=70, l=80, r=40),
    )
    _save(fig, os.path.join(out_dir, "chart8_cumulative_by_Gender.png"))


# =============================================================================
# COLOR PALETTES — Age and Race stratifications
# Distinct, colorblind-friendlier mappings; BJS reference uses a darker shade
# of the same hue as the ABM curve so paired comparisons stay grouped.
# =============================================================================
_AGE_PALETTE = {
    # younger → blue/teal/green/amber/red as age increases (intuitive ramp)
    "24 or younger": {"abm": "#1F77B4", "bjs": "#0C4A78", "fill": "rgba( 31, 119, 180, 0.16)"},
    "25-29":         {"abm": "#17A2A2", "bjs": "#0B6868", "fill": "rgba( 23, 162, 162, 0.16)"},
    "30-34":         {"abm": "#2CA02C", "bjs": "#175C17", "fill": "rgba( 44, 160,  44, 0.16)"},
    "35-39":         {"abm": "#E29A1A", "bjs": "#8B5A0A", "fill": "rgba(226, 154,  26, 0.16)"},
    "40 or older":   {"abm": "#C0392B", "bjs": "#7B2418", "fill": "rgba(192,  57,  43, 0.16)"},
}
_RACE_PALETTE = {
    "White":    {"abm": "#4C72B0", "bjs": "#1F3D6E", "fill": "rgba( 76, 114, 176, 0.16)"},
    "Black":    {"abm": "#55A868", "bjs": "#2A6839", "fill": "rgba( 85, 168, 104, 0.16)"},
    "Hispanic": {"abm": "#C44E52", "bjs": "#7A2A2D", "fill": "rgba(196,  78,  82, 0.16)"},
    "Other":    {"abm": "#8172B2", "bjs": "#4D4475", "fill": "rgba(129, 114, 178, 0.16)"},
}


# =============================================================================
# GENERIC STRATIFIER PLOTTERS — used by Charts 9–12 (age, race)
# Mirrors the structure of charts 7 & 8 but accepts the stratifier metadata
# as parameters, so we do not duplicate ~200 lines of layout code per chart.
# =============================================================================
def _ordinal(n):
    if 10 <= n % 100 <= 20: suffix = "th"
    else: suffix = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _no_data_chart(title, png_name, out_dir, attr_hint):
    """Render a single 'no data' explainer when a stratifier is unpopulated."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=(f"<b>No {attr_hint}-stratified data available.</b><br>"
              f"<sup>Worker did not capture per-agent {attr_hint} or replot cache "
              f"predates this chart.  Re-run the simulation to populate.</sup>"),
        showarrow=False, align="center",
        font=dict(size=12, family=_FONT_FAMILY, color="#888888"),
    )
    fig.update_layout(**_LAYOUT,
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=15, family=_FONT_FAMILY),
                   x=0.5, xanchor="center"),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=400, margin=dict(t=60, b=40, l=40, r=40),
    )
    _save(fig, os.path.join(out_dir, png_name))


def _chart_first_arrest_by_group(group_mean, group_sd, *, groups, palette,
                                 bjs_first, bjs_cum9,
                                 stratifier_label, png_name, out_dir,
                                 fig_title_suffix):
    """Generic version of Chart 7: per-year first-arrest hazards stratified by
    a categorical attribute (Gender / age / race).  Renders one curve per group
    plus its BJS Table 3 reference, σ ribbons, and Δ labels at anchor years."""
    has_data = any(any(v > 0 for v in group_mean[g]) for g in groups)
    if not has_data:
        _no_data_chart(f"Chart — Annual First-Arrest Rate by {stratifier_label}",
                       png_name, out_dir, stratifier_label.lower())
        return

    x_labels = [_ordinal(y) for y in BJS_YEARS]

    # MAE per group, for the summary panel
    mae = {g: sum(abs(group_mean[g][i] - bjs_first[g][i]) for i in range(9)) / 9
           for g in groups}

    fig = go.Figure()
    for g in groups:
        c   = palette[g]
        m   = group_mean[g]; sd = group_sd[g]
        hi  = [a + b for a, b in zip(m, sd)]
        lo  = [max(0.0, a - b) for a, b in zip(m, sd)]

        # ABM ±1σ ribbon — kept off the legend; the band itself is self-explanatory
        # and a "<group> ±1σ" entry per group would double the legend's height.
        fig.add_trace(go.Scatter(
            x=BJS_YEARS + BJS_YEARS[::-1], y=hi + lo[::-1],
            fill="toself", fillcolor=c["fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name=f"ABM {g} ±1σ",
            legendgroup=f"sd_{g}", showlegend=False,
            hoverinfo="skip",
        ))
        # BJS reference (dotted, diamonds)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=bjs_first[g], mode="lines+markers",
            name=f"BJS {g}",
            legendgroup=f"bjs_{g}",
            line=dict(color=c["bjs"], width=2.4, dash="dot"),
            marker=dict(symbol="diamond", size=8, color=c["bjs"],
                        line=dict(color="white", width=1)),
            hovertemplate=(f"<b>BJS {g} Yr %{{x}}</b><br>"
                           f"%{{y:.2f}}%<extra></extra>"),
        ))
        # ABM mean (solid, circles)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=m, mode="lines+markers",
            name=f"ABM {g}",
            legendgroup=f"abm_{g}",
            line=dict(color=c["abm"], width=2.6),
            marker=dict(size=7, color=c["abm"],
                        line=dict(color="white", width=1)),
            customdata=sd,
            hovertemplate=(f"<b>ABM {g} Yr %{{x}}</b><br>"
                           f"%{{y:.2f}}%<br>σ across runs: %{{customdata:.2f}} pp"
                           f"<extra></extra>"),
        ))

    # ── Summary panel: in the BOTTOM margin strip so it never overlaps curves ─
    # One row per metric (MAE / cumulative / σ), values comma-separated.
    mae_line  = "  |  ".join(f"{g}: {mae[g]:.2f}pp" for g in groups)
    cum9_line = "  |  ".join(f"{g}: {bjs_cum9[g]:.1f}%" for g in groups)
    sd_line   = "  |  ".join(f"{g}: {sum(group_sd[g])/9:.2f}" for g in groups)
    fig.add_annotation(
        x=0.5, y=-0.22, xref="paper", yref="paper",
        text=(f"<b>Mean |Δ| across 9 yrs:</b>  {mae_line}<br>"
              f"<b>BJS 9-yr cumulative (Table 3):</b>  {cum9_line}<br>"
              f"<span style='color:{_C_SD}'>"
              f"<b>ABM σ across runs (mean):</b>  {sd_line} pp</span>"),
        showarrow=False, align="center", xanchor="center", yanchor="top",
        font=dict(size=9.5, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    # y-axis: a touch above the highest first-year value
    y_top = max(max(bjs_first[g][0] for g in groups),
                max(group_mean[g][0] + group_sd[g][0] for g in groups))

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=(f"<b>Annual First-Arrest Rate by {stratifier_label}: "
                  f"ABM vs BJS NCJ 250975 (Table 3)</b><br>"
                  f"<sup>{fig_title_suffix}  |  "
                  f"Source: Alper, Durose &amp; Markman (2018)</sup>"),
            font=dict(size=15, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        xaxis=dict(title="Year of first arrest",
                   tickvals=BJS_YEARS, ticktext=x_labels,
                   showgrid=False, range=[0.5, 9.5]),
        yaxis=dict(title="Percent of released prisoners",
                   range=[-2, y_top + 4],
                   gridcolor=_C_GRID, zeroline=False, ticksuffix="%"),
        legend=dict(
            x=0.98, y=0.97, xanchor="right", yanchor="top",
            font=dict(size=9, family=_FONT_FAMILY),
            itemsizing="constant", itemwidth=30, tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CCCCCC", borderwidth=1,
        ),
        height=600, margin=dict(t=90, b=140, l=80, r=40),
    )
    _save(fig, os.path.join(out_dir, png_name))


def _chart_cumulative_by_group(group_mean, group_sd, *, groups, palette,
                               bjs_first, bjs_cum9,
                               stratifier_label, png_name, out_dir):
    """Generic version of Chart 8: cumulative arrest curve stratified by
    category.  Cumulative arrays are derived from the per-year hazards via
    running sum; cumulative σ uses the conservative √(Σσ²) bound (called out
    in the summary footnote)."""
    has_data = any(any(v > 0 for v in group_mean[g]) for g in groups)
    if not has_data:
        _no_data_chart(f"Chart — Cumulative Rearrest Rate by {stratifier_label}",
                       png_name, out_dir, stratifier_label.lower())
        return

    x_labels = [_ordinal(y) for y in BJS_YEARS]

    # Derive cumulative arrays
    abm_cum    = {g: [sum(group_mean[g][:i+1]) for i in range(9)] for g in groups}
    bjs_cum    = {g: [sum(bjs_first[g][:i+1])  for i in range(9)] for g in groups}
    abm_cum_sd = {g: [math.sqrt(sum(x*x for x in group_sd[g][:i+1])) for i in range(9)]
                  for g in groups}

    diffs      = {g: [abm_cum[g][i] - bjs_cum[g][i] for i in range(9)] for g in groups}
    mae_cum    = {g: sum(abs(d) for d in diffs[g]) / 9 for g in groups}
    endpoint_d = {g: abm_cum[g][8] - bjs_cum[g][8] for g in groups}

    fig = go.Figure()
    for g in groups:
        c   = palette[g]
        m   = abm_cum[g]; sd = abm_cum_sd[g]
        bjs = bjs_cum[g]
        hi  = [a + b for a, b in zip(m, sd)]
        lo  = [max(0.0, a - b) for a, b in zip(m, sd)]

        # σ ribbon — kept off the legend (otherwise the legend doubles in height)
        fig.add_trace(go.Scatter(
            x=BJS_YEARS + BJS_YEARS[::-1], y=hi + lo[::-1],
            fill="toself", fillcolor=c["fill"],
            line=dict(color="rgba(0,0,0,0)"),
            name=f"ABM {g} ±1σ",
            legendgroup=f"sd_{g}", showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=bjs, mode="lines+markers",
            name=f"BJS {g}",
            legendgroup=f"bjs_{g}",
            line=dict(color=c["bjs"], width=2.4, dash="dot"),
            marker=dict(symbol="diamond", size=8, color=c["bjs"],
                        line=dict(color="white", width=1)),
            hovertemplate=(f"<b>BJS {g} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}% cumulative<extra></extra>"),
        ))
        fig.add_trace(go.Scatter(
            x=BJS_YEARS, y=m, mode="lines+markers",
            name=f"ABM {g}",
            legendgroup=f"abm_{g}",
            line=dict(color=c["abm"], width=2.6),
            marker=dict(size=7, color=c["abm"],
                        line=dict(color="white", width=1)),
            customdata=sd,
            hovertemplate=(f"<b>ABM {g} Yr %{{x}}</b><br>"
                           f"%{{y:.1f}}% cumulative"
                           f"<br>cumulative σ ≈ %{{customdata:.2f}} pp"
                           f"<extra></extra>"),
        ))

    # NOTE: per-curve endpoint Δ labels removed — with 4-5 curves stacking in a
    # narrow horizontal range at year 9, they overlap badly.  The summary panel
    # below the plot carries the same numeric information in a cleaner layout.

    # ── Summary panel: in the BOTTOM margin strip ───────────────────────────
    mae_line      = "  |  ".join(f"{g}: {mae_cum[g]:.2f}pp" for g in groups)
    endpoint_line = "  |  ".join(f"{g}: {endpoint_d[g]:+.2f}pp" for g in groups)
    cum9_line     = "  |  ".join(f"{g}: {bjs_cum9[g]:.1f}%" for g in groups)
    fig.add_annotation(
        x=0.5, y=-0.22, xref="paper", yref="paper",
        text=(f"<b>Mean |Δ| across 9 yrs:</b>  {mae_line}<br>"
              f"<b>9-yr endpoint Δ (ABM − BJS):</b>  {endpoint_line}<br>"
              f"<b>BJS 9-yr cumulative (Table 3):</b>  {cum9_line}<br>"
              f"<span style='color:{_C_SD}'>"
              f"<i>Cumulative σ shown as √(Σσ²) bound across years</i></span>"),
        showarrow=False, align="center", xanchor="center", yanchor="top",
        font=dict(size=9.5, family=_FONT_FAMILY, color="#333333"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CCCCCC", borderwidth=1,
    )

    # y-axis: tight to the data range (no dead space at the bottom)
    y_min = min(min(bjs_cum[g][0] for g in groups),
                min(abm_cum[g][0] for g in groups)) - 5
    y_top = max(max(bjs_cum[g][8] for g in groups),
                max(abm_cum[g][8] + abm_cum_sd[g][8] for g in groups))

    fig.update_layout(**_LAYOUT,
        title=dict(
            text=(f"<b>Cumulative Rearrest Rate by {stratifier_label}: "
                  f"ABM vs BJS NCJ 250975 (Table 3)</b><br>"
                  f"<sup>Running cumulative arrest fraction by year and {stratifier_label.lower()}  |  "
                  f"Source: Alper, Durose &amp; Markman (2018)</sup>"),
            font=dict(size=15, family=_FONT_FAMILY),
            x=0.5, xanchor="center"),
        xaxis=dict(title="Years since release",
                   tickvals=BJS_YEARS, ticktext=x_labels,
                   showgrid=False, range=[0.5, 9.5]),
        yaxis=dict(title="Cumulative rearrest rate",
                   range=[max(0, y_min), y_top + 5],
                   gridcolor=_C_GRID, zeroline=False, ticksuffix="%"),
        legend=dict(
            x=0.02, y=0.97, xanchor="left", yanchor="top",
            font=dict(size=9, family=_FONT_FAMILY),
            itemsizing="constant", itemwidth=30, tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CCCCCC", borderwidth=1,
        ),
        height=600, margin=dict(t=90, b=140, l=80, r=40),
    )
    _save(fig, os.path.join(out_dir, png_name))


# =============================================================================
# CHARTS 9 / 10 — Age-at-release stratification
# =============================================================================
def chart9_first_arrest_by_age(age_mean, age_sd, out_dir):
    _chart_first_arrest_by_group(
        age_mean, age_sd,
        groups=AGE_GROUPS, palette=_AGE_PALETTE,
        bjs_first=BJS_FIRST_BY_AGE, bjs_cum9=BJS_CUM9_BY_AGE,
        stratifier_label="Age at Release",
        png_name="chart9_first_arrest_by_age.png",
        out_dir=out_dir,
        fig_title_suffix="Five non-overlapping age buckets at time of release",
    )

def chart10_cumulative_by_age(age_mean, age_sd, out_dir):
    _chart_cumulative_by_group(
        age_mean, age_sd,
        groups=AGE_GROUPS, palette=_AGE_PALETTE,
        bjs_first=BJS_FIRST_BY_AGE, bjs_cum9=BJS_CUM9_BY_AGE,
        stratifier_label="Age at Release",
        png_name="chart10_cumulative_by_age.png",
        out_dir=out_dir,
    )

# =============================================================================
# CHARTS 11 / 12 — Race stratification
# =============================================================================
def chart11_first_arrest_by_race(race_mean, race_sd, out_dir):
    _chart_first_arrest_by_group(
        race_mean, race_sd,
        groups=RACE_GROUPS, palette=_RACE_PALETTE,
        bjs_first=BJS_FIRST_BY_RACE, bjs_cum9=BJS_CUM9_BY_RACE,
        stratifier_label="Race",
        png_name="chart11_first_arrest_by_race.png",
        out_dir=out_dir,
        fig_title_suffix="Hispanic ethnicity supersedes racial label per BJS reporting",
    )

def chart12_cumulative_by_race(race_mean, race_sd, out_dir):
    _chart_cumulative_by_group(
        race_mean, race_sd,
        groups=RACE_GROUPS, palette=_RACE_PALETTE,
        bjs_first=BJS_FIRST_BY_RACE, bjs_cum9=BJS_CUM9_BY_RACE,
        stratifier_label="Race",
        png_name="chart12_cumulative_by_race.png",
        out_dir=out_dir,
    )


# =============================================================================
# RENDER — single entry point used by both full-run and replot modes
# =============================================================================
def render_all_charts(agg, out_dir):
    (cum_mean, cum_lo, cum_hi, cum_sd,
     first_mean, first_lo, first_hi, first_sd,
     off_mean, off_lo, off_hi, off_sd,
     Gender_mean, Gender_lo, Gender_hi, Gender_sd,
     age_mean, age_lo, age_hi, age_sd,
     race_mean, race_lo, race_hi, race_sd) = agg

    print("  Generating 12 validation charts...")
    chart1_cumulative_by_year(cum_mean, cum_lo, cum_hi, cum_sd, out_dir)
    chart2_cumulative_bar(cum_mean, cum_lo, cum_hi, cum_sd, out_dir)
    chart3_cumulative_by_offense(off_mean, off_lo, off_hi, off_sd, out_dir)
    chart4_bar_by_offense(off_mean, off_lo, off_hi, off_sd, out_dir)
    chart5_noncumulative(first_mean, first_lo, first_hi, first_sd, out_dir)
    chart6_desistance_survival(cum_mean, cum_sd, out_dir)
    chart7_first_arrest_by_Gender(Gender_mean, Gender_sd, out_dir)
    chart8_cumulative_by_Gender(Gender_mean, Gender_sd, out_dir)
    chart9_first_arrest_by_age(age_mean, age_sd, out_dir)
    chart10_cumulative_by_age(age_mean, age_sd, out_dir)
    chart11_first_arrest_by_race(race_mean, race_sd, out_dir)
    chart12_cumulative_by_race(race_mean, race_sd, out_dir)
    print(f"  All charts saved to {out_dir}/")


# =============================================================================
# PRINT SUMMARY
# =============================================================================
def print_summary(cum_mean, cum_sd, first_mean, off_mean, n_runs):
    print(f"\n{'='*72}")
    print(f"  VALIDATION SUMMARY  ({n_runs} simulation runs)")
    print(f"{'='*72}")
    print(f"  {'Year':>4}  {'ABM cum%':>9}  {'BJS cum%':>9}  {'Δpp':>6}  "
          f"{'σ pp':>6}  {'ABM 1st%':>9}  {'BJS 1st%':>9}")
    print(f"  {'-'*68}")
    for i, yr in enumerate(BJS_YEARS):
        diff = cum_mean[i] - BJS_CUMULATIVE_ALL[i]
        flag = "✅" if abs(diff) <= GAP_GOOD else ("⚠️ " if abs(diff) <= GAP_WARN else "❌")
        print(f"  {yr:>4}  {cum_mean[i]:>8.1f}%  "
              f"{BJS_CUMULATIVE_ALL[i]:>8.1f}%  "
              f"{diff:>+5.1f}  {flag} "
              f"{cum_sd[i]:>5.2f}  "
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
    print(f"{'='*72}\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Recidivism ABM validation suite")
    parser.add_argument("--replot", action="store_true",
                        help="Skip simulation; reload aggregated results from "
                             "the cache JSON and regenerate charts only.")
    parser.add_argument("--out-dir", default="validation_output_calibrated",
                        help="Output directory (default: validation_output)")
    args = parser.parse_args()

    OUT_DIR     = args.out_dir
    cache_path  = os.path.join(OUT_DIR, RESULTS_CACHE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── REPLOT-ONLY MODE ──────────────────────────────────────────────────────
    if args.replot:
        if not os.path.exists(cache_path):
            print(f"  ERROR: cache not found at {cache_path}")
            print(f"  Run without --replot first to generate it.")
            sys.exit(1)
        print(f"  [REPLOT MODE] loading {cache_path}")
        agg, n_runs = load_aggregated(cache_path)
        (cum_mean, _, _, cum_sd,
         first_mean, _, _, _,
         off_mean, _, _, _,
         _, _, _, _,            # Gender_*
         _, _, _, _,            # age_*
         _, _, _, _) = agg      # race_*
        print_summary(cum_mean, cum_sd, first_mean, off_mean, n_runs)
        render_all_charts(agg, OUT_DIR)
        return

    # ── FULL RUN ──────────────────────────────────────────────────────────────
    N_RUNS        = 10
    SEEDS_PER_RUN = 20
    BASE_SEED     = 1000

    CAL = get_global_calibration_params()
    # CAL = get_uncalibrated_params()

    tasks     = build_tasks(N_RUNS, SEEDS_PER_RUN, BASE_SEED, CAL)
    n_workers = detect_workers(len(tasks))

    print(f"  CPU logical: {os.cpu_count()} | workers: {n_workers}")
    print(f"  Tasks: {len(tasks)}  ({N_RUNS} runs × {SEEDS_PER_RUN} seeds)")

    results = run_parallel(tasks, n_workers)

    # Per-seed raw CSV (preserves existing behaviour, adds stratifier columns)
    def _safe_col(s):
        # Keep CSV column names well-behaved (no spaces / special chars)
        return s.replace(" ", "_").replace("-", "_")
    rows = []
    for r in results:
        row = {"run_id": r["run_id"], "seed": r["seed"], "n": r["n"]}
        for i, yr in enumerate(BJS_YEARS):
            row[f"cum_yr{yr}"]   = r["cum_all"][i]
            row[f"first_yr{yr}"] = r["first_all"][i]
            for g in OFFENSE_GROUPS:
                row[f"cum_{_safe_col(g)}_yr{yr}"] = r["cum_offense"][g][i]
            for s in Gender_GROUPS:
                row[f"first_Gender_{_safe_col(s)}_yr{yr}"] = r.get("first_by_Gender", {}).get(s, [0]*9)[i]
            for g in AGE_GROUPS:
                row[f"first_age_{_safe_col(g)}_yr{yr}"] = r.get("first_by_age", {}).get(g, [0]*9)[i]
            for g in RACE_GROUPS:
                row[f"first_race_{_safe_col(g)}_yr{yr}"] = r.get("first_by_race", {}).get(g, [0]*9)[i]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "validation_raw.csv"), index=False)
    print(f"  Raw results -> {OUT_DIR}/validation_raw.csv")

    # Aggregate + cache for replot mode
    agg = aggregate(results)
    save_aggregated(cache_path, agg, n_runs=len(results))

    (cum_mean, _, _, cum_sd,
     first_mean, _, _, _,
     off_mean, _, _, _,
     _, _, _, _,                # Gender_*
     _, _, _, _,                # age_*
     _, _, _, _) = agg          # race_*

    print_summary(cum_mean, cum_sd, first_mean, off_mean, n_runs=len(results))
    render_all_charts(agg, OUT_DIR)


if __name__ == "__main__":
    main()