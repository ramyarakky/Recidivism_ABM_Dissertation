"""
recidivism_validation_demographics.py 
=================================
End-to-end driver:
  1. Runs the calibrated ABM across multiple seeds (parallel)
  2. Extracts per-agent rearrest data from study-eligible agents
  3. Derives year-of-first-arrest from community_months_at_risk
     (matches the semantic used by rearrest_{1,3,6,9}_yrs flags in the agent,
      ensuring internal consistency across all validation charts)
  4. Computes year-by-year rates by Gender (Male / Female) and Race
  5. Produces a two-panel small-multiples line chart:
       - Male panel   (ABM vs BJS)
       - Female panel (ABM vs BJS)
  6. Produces a three-panel small-multiples line chart:
       - White panel    (ABM vs BJS)
       - Black panel    (ABM vs BJS)
       - Hispanic panel (ABM vs BJS)
     Note: "Other" race category is excluded; BJS cell sizes for Other
     are small and the category is heterogeneous.
  7. Produces a grouped bar chart comparing ABM vs BJS at 3/6/9 yr
     windows by Gender.
  8. Produces a grouped bar chart comparing ABM vs BJS at 3/6/9 yr
     windows by Race (White, Black, Hispanic).
  9. Produces a Gender Bias Calibration Effect chart using paired runs:
       - Fair baseline: bias_factor = 0
       - Calibrated bias: bias_factor = 1
     The chart uses direct rearrest_{3,6,9}_yrs flags and compares
     absolute BJS fit plus Male–Female disparity recovery.

Bar chart note
--------------
The bar charts use cumulative rearrest rates at the 3/6/9-yr anchor
windows — the same windows used for Stage 1 calibration and the main
validation suite — rather than the per-year non-cumulative rates shown
in the line charts. Cumulative rates are derived by summing per-year
non-cumulative rates over the relevant window.

Year-of-first-arrest semantic
-----------------------------
community_months_at_risk counts months actually spent in Free or Supervision
since the most recent reset. It is reset to 0 at study start by
reset_agent_for_study() in recidivism_model.py. No warmup contamination.

For an agent rearrested during the study:
  year_of_first_arrest = ceil(community_months_at_risk / 12), clipped to [1, 9]

This is the SAME semantic used by the agent's rearrest_{1,3,6,9}_yrs flags,
so this driver's output is internally consistent with Charts 2 and 4 of
recidivism_validation.py (the 3/6/9yr anchor bars).

Calibration parameters
-----------------------
All calibration parameters are drawn from
recidivism_abm.config.risk_config.get_global_calibration_params() rather than
defined locally. This guarantees that the driver always uses the current
production-calibrated values and stays in sync with any future Stage 1/2/3
updates without manual copying.

Output:
  - validation_output_demographics/chart_gender_small_multiples.png
  - validation_output_demographics/chart_gender_bar.png
  - validation_output_demographics/chart_race_small_multiples.png
  - validation_output_demographics/chart_race_bar.png
  - validation_output_demographics/agent_cohort_pooled.csv
  - validation_output_demographics/gender_rearrest_by_year.csv
  - validation_output_demographics/race_rearrest_by_year.csv
  - chart_gender_bias_calibration_effect.png
  - gender_bias_calibration_effect.csv
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_peer_influence_config,
    get_global_calibration_params,
)



# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "initial_agents":         1500,
    "monthly_intake":         10,
    "warmup_months":          144,
    "study_months":           108,
    "bias_factor":            1,
    "enable_peer_influence":  True,
    "output_dir":             "validation_output_demographics_bias_calibrated",
    "seeds":                  [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "n_reps":                 1,
}

# Resolved once at import time so the console header can print the values
# and so that all worker processes share an identical parameter snapshot.
_CALIBRATION_PARAMS = get_global_calibration_params(
    overrides={
        "bias_factor": CONFIG["bias_factor"],
        "bias_scope": "supervision_only",
    }
)

# Calibration anchor windows used for bar charts
BAR_WINDOWS = [3, 6, 9]

# ── BJS Table 3 (non-cumulative year-of-first-arrest) by sex ─────────────────
# https://bjs.ojp.gov/content/pub/pdf/18upr9yfup0514.pdf
BJS_TABLE3 = {
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

# ── BJS Table 3 (non-cumulative year-of-first-arrest) by race ────────────────
# Source: Alper, Durose & Markman (2018), NCJ 250975, Table 3
# "Other" is retained in the raw dict for completeness but excluded from
# charts and CSV output — cell sizes are small and the category is
# heterogeneous across states.
_BJS_TABLE3_RACE_RAW = {
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

# Races included in charts and CSV (Other excluded by design)
RACE_GROUPS = ["White", "Black", "Hispanic"]

# Derive 9-yr totals by summing the per-year non-cumulative rates
BJS_TABLE3_RACE = dict(_BJS_TABLE3_RACE_RAW)
BJS_TABLE3_RACE["total_9yr"] = {
    race: sum(_BJS_TABLE3_RACE_RAW[y][race] for y in range(1, 10))
    for race in ("White", "Black", "Hispanic", "Other")
}

# Pre-compute BJS cumulative rates at 3/6/9 yr for bar chart reference lines.
# Cumulative rate at window W = sum of per-year non-cumulative rates 1..W.
BJS_CUMULATIVE_GENDER = {
    w: {grp: sum(BJS_TABLE3[y][grp] for y in range(1, w + 1))
        for grp in ("Male", "Female")}
    for w in BAR_WINDOWS
}
BJS_CUMULATIVE_RACE = {
    w: {grp: sum(_BJS_TABLE3_RACE_RAW[y][grp] for y in range(1, w + 1))
        for grp in ("White", "Black", "Hispanic")}
    for w in BAR_WINDOWS
}


PALETTE = {
    # ── Gender panels ─────────────────────────────────────────────────────────
    "Male": {
        "bjs":  "#0D47A1",   # deep navy blue (BJS reference)
        "abm":  "#42A5F5",   # bright sky blue (ABM simulation)
        "fill": "#E3F2FD",   # pale blue (tolerance band)
    },
    "Female": {
        "bjs":  "#B71C1C",   # deep crimson (BJS reference)
        "abm":  "#EF5350",   # bright coral (ABM simulation)
        "fill": "#FFEBEE",   # pale rose (tolerance band)
    },
    # ── Race panels ───────────────────────────────────────────────────────────
    "White": {
        "bjs":  "#1B5E20",   # deep forest green (BJS reference)
        "abm":  "#66BB6A",   # medium green (ABM simulation)
        "fill": "#E8F5E9",   # pale green (tolerance band)
    },
    "Black": {
        "bjs":  "#4A148C",   # deep purple (BJS reference)
        "abm":  "#AB47BC",   # medium purple (ABM simulation)
        "fill": "#F3E5F5",   # pale lavender (tolerance band)
    },
    "Hispanic": {
        "bjs":  "#E65100",   # deep orange (BJS reference)
        "abm":  "#FFA726",   # amber (ABM simulation)
        "fill": "#FFF3E0",   # pale peach (tolerance band)
    },
}

C = {
    "good":  "#2E7D32",  # green  — within ±2pp
    "warn":  "#F57C00",  # amber  — within ±5pp
    "bad":   "#C62828",  # red    — exceeds ±5pp
    "grid":  "#E0E0E0",
}


def _gap_color(gap_pp):
    if abs(gap_pp) <= 2.0: return C["good"]
    if abs(gap_pp) <= 5.0: return C["warn"]
    return C["bad"]


# =============================================================================
# MODEL RUNNER (parallel worker)
# =============================================================================
def _run_one(args):
    if len(args) == 3:
        seed, rep, bias_factor = args
    else:
        seed, rep = args
        bias_factor = CONFIG["bias_factor"]

    effective_seed = seed * 1000 + rep

    try:
        model = RecidivismModel(
            initial_agents        = CONFIG["initial_agents"],
            bias_factor           = bias_factor,
            monthly_intake        = CONFIG["monthly_intake"],
            warmup_months         = CONFIG["warmup_months"],
            study_months          = CONFIG["study_months"],
            enable_peer_influence = CONFIG["enable_peer_influence"],
            weights               = get_flat_risk_weights(),
            peer_config           = get_peer_influence_config(),
            calibration_params    = _CALIBRATION_PARAMS,
            seed                  = effective_seed,
        )
        model.export_csv = False

        while model.running:
            model.step()

        rows = []
        for a in model.schedule.agents:
            if not getattr(a, "study_eligible_agent", False):
                continue
            rows.append({
                "seed":                     effective_seed,
                "agent_id":                 getattr(a, "unique_id", None),
                "Gender":                   getattr(a, "Gender", None),
                "Race":                     getattr(a, "Race", None),
                "Age_at_Release":           getattr(a, "Age_at_Release", None),
                "recidivated_agent":        bool(getattr(a, "recidivated_agent", False)),
                "community_months_at_risk": getattr(a, "community_months_at_risk", 0),
                "rearrest_month":           getattr(a, "rearrest_month", None),
                "rearrest_year":            getattr(a, "rearrest_year", None),
                "rearrest_1_yrs":           bool(getattr(a, "rearrest_1_yrs", False)),
                "rearrest_3_yrs":           bool(getattr(a, "rearrest_3_yrs", False)),
                "rearrest_6_yrs":           bool(getattr(a, "rearrest_6_yrs", False)),
                "rearrest_9_yrs":           bool(getattr(a, "rearrest_9_yrs", False)),
            })
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  Worker error (seed={effective_seed}): {e}", flush=True)
        return pd.DataFrame()


def run_all_seeds_parallel(n_workers, bias_factor=None, label=None):
    if bias_factor is None:
        bias_factor = CONFIG["bias_factor"]

    jobs = [(s, r, bias_factor)
            for s in CONFIG["seeds"]
            for r in range(CONFIG["n_reps"])]

    condition = label or f"bias_factor={bias_factor:g}"
    print(f"  Running {len(jobs)} simulations [{condition}] "
          f"({len(CONFIG['seeds'])} seeds × {CONFIG['n_reps']} reps) "
          f"with {n_workers} workers...")

    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        with tqdm(total=len(jobs), desc="  Simulating", ncols=80) as bar:
            for fut in as_completed(futures):
                try:
                    df = fut.result()
                    if len(df) > 0:
                        frames.append(df)
                except Exception as e:
                    print(f"\n  Future error: {e}", flush=True)
                finally:
                    bar.update(1)

    if not frames:
        raise RuntimeError("No simulation data returned")
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# YEAR-OF-FIRST-ARREST — use community_months_at_risk
# =============================================================================
def derive_first_arrest_year(df, warmup_months=144):
    """
    Derive year-of-first-arrest from community_months_at_risk.

    Primary:    community_months_at_risk — months spent in Free/Supervision
                when first rearrest occurred. Reset to 0 at study start by
                reset_agent_for_study(). Matches the semantic used by
                rearrest_{1,3,6,9}_yrs flags in the agent.

                year = ceil(community_months_at_risk / 12), clipped to [1, 9]

    Fallback 1: rearrest_month - warmup_months — calendar months since study
                start. Used if community_months_at_risk is missing/zero.

    Fallback 2: rearrest_year attribute — direct [1, 9] value, accepted only
                if already in valid range (defensive guard).
    """
    df = df.copy()
    n = len(df)

    rearrested = df["recidivated_agent"].astype(bool).values

    year = np.full(n, np.nan)
    source = "community_months_at_risk (primary)"

    cm = pd.to_numeric(df["community_months_at_risk"], errors="coerce").values
    valid_cm = rearrested & ~np.isnan(cm) & (cm > 0)
    year[valid_cm] = np.ceil(cm[valid_cm] / 12.0)

    missing_mask = rearrested & np.isnan(year)
    if missing_mask.any() and "rearrest_month" in df.columns:
        print(f"  {int(missing_mask.sum()):,} rearrested agents missing "
              "community_months_at_risk — deriving from rearrest_month")
        rm = pd.to_numeric(df["rearrest_month"], errors="coerce").values
        rm_fallback = rm - warmup_months
        use_rm = missing_mask & ~np.isnan(rm) & (rm_fallback > 0)
        year[use_rm] = np.ceil(rm_fallback[use_rm] / 12.0)

    missing_mask = rearrested & np.isnan(year)
    if missing_mask.any() and "rearrest_year" in df.columns:
        print(f"  {int(missing_mask.sum()):,} rearrested agents still missing "
              "— trying rearrest_year attribute")
        ry = pd.to_numeric(df["rearrest_year"], errors="coerce").values
        use_ry = missing_mask & ~np.isnan(ry) & (ry >= 1) & (ry <= 9)
        year[use_ry] = ry[use_ry]

    year = np.where(
        (~np.isnan(year)) & (year >= 1) & (year <= 9),
        np.clip(year, 1, 9),
        np.nan
    )
    df["year_of_first_arrest"] = year

    n_rearrested = int(rearrested.sum())
    n_assigned   = int((~np.isnan(year)).sum())
    print(f"  Source                 : {source}")
    print(f"  Agents total           : {n:,}")
    print(f"  Agents rearrested      : {n_rearrested:,} ({n_rearrested/n:.1%})")
    print(f"  Assigned year-of-first : {n_assigned:,} ({n_assigned/n:.1%})")

    if n_assigned > 0:
        valid_cm_at_rearrest = cm[valid_cm]
        if len(valid_cm_at_rearrest) > 0:
            print(f"  community_months range : "
                  f"min={int(valid_cm_at_rearrest.min())}, "
                  f"median={int(np.median(valid_cm_at_rearrest))}, "
                  f"max={int(valid_cm_at_rearrest.max())}")

        valid_years = year[~np.isnan(year)].astype(int)
        year_hist = np.bincount(valid_years, minlength=10)[1:10]
        total = year_hist.sum()
        print(f"  Year distribution:")
        for y in range(1, 10):
            pct = year_hist[y-1] / total if total > 0 else 0
            bar = "\u2588" * int(pct * 40)
            print(f"    Year {y}: {year_hist[y-1]:>6,} ({pct:>5.1%})  {bar}")

    return df


# =============================================================================
# RATE COMPUTATION HELPERS
# =============================================================================
def compute_rates_by_sex(df):
    """Per-year non-cumulative first-arrest rate for Male and Female."""
    out = {}
    for sex in ["Male", "Female"]:
        subset = df[df["Gender"] == sex]
        n = len(subset)
        out[sex] = {"n": n}
        for y in range(1, 10):
            out[sex][y] = ((subset["year_of_first_arrest"] == y).sum() / n
                           if n > 0 else 0.0)
        out[sex]["total_9yr"] = sum(out[sex][y] for y in range(1, 10))
    return out


def compute_rates_by_race(df):
    """
    Per-year non-cumulative first-arrest rate for White, Black, Hispanic.

    "Other" agents are present in the ABM but excluded from charts and CSV
    output to match the BJS decision to exclude the heterogeneous Other cell.
    """
    out = {}
    for race in RACE_GROUPS:
        subset = df[df["Race"] == race]
        n = len(subset)
        out[race] = {"n": n}
        for y in range(1, 10):
            out[race][y] = ((subset["year_of_first_arrest"] == y).sum() / n
                            if n > 0 else 0.0)
        out[race]["total_9yr"] = sum(out[race][y] for y in range(1, 10))
    return out


def _cumulative(rates_dict, group, window):
    """Sum per-year rates 1..window to produce a cumulative rate."""
    return sum(rates_dict[group][y] for y in range(1, window + 1))



# =============================================================================
# GENDER BIAS CALIBRATION EFFECT — PAIRED FAIR vs CALIBRATED RUNS
# =============================================================================
def compute_direct_cumulative_gender_rates(df):
    """Use the direct agent rearrest_3/6/9_yrs flags used by calibration."""
    out = {}
    for sex in ("Male", "Female"):
        subset = df[df["Gender"] == sex]
        n = len(subset)
        out[sex] = {"n": n}
        for w in BAR_WINDOWS:
            col = f"rearrest_{w}_yrs"
            out[sex][w] = float(subset[col].mean()) if n > 0 else 0.0
    return out


def compute_gender_bias_effect_table(fair_rates, calibrated_rates):
    rows = []
    for w in BAR_WINDOWS:
        for sex in ("Male", "Female"):
            bjs = BJS_CUMULATIVE_GENDER[w][sex]
            before = fair_rates[sex][w]
            after = calibrated_rates[sex][w]
            err_before = abs(before - bjs) * 100.0
            err_after = abs(after - bjs) * 100.0
            rows.append({
                "Gender": sex,
                "Window": f"{w}-Year",
                "BJS": bjs,
                "Fair_Baseline": before,
                "Bias_Calibrated": after,
                "Error_Before_pp": err_before,
                "Error_After_pp": err_after,
                "Improvement_pp": err_before - err_after,
            })
    return pd.DataFrame(rows)


def _gender_gap(rates, w):
    return (rates["Male"][w] - rates["Female"][w]) * 100.0


def plot_gender_bias_calibration_effect(fair_rates, calibrated_rates, output_path):
    windows = BAR_WINDOWS
    labels = [f"{w}-Year" for w in windows]
    x = np.arange(len(windows), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("white")

    # Panel A: Female fit before vs after
    ax = axes[0]
    width = 0.24
    bjs_f = [BJS_CUMULATIVE_GENDER[w]["Female"] * 100.0 for w in windows]
    fair_f = [fair_rates["Female"][w] * 100.0 for w in windows]
    cal_f = [calibrated_rates["Female"][w] * 100.0 for w in windows]

    ax.bar(x - width, bjs_f, width, label="BJS Female")
    ax.bar(x, fair_f, width, label="Fair baseline (bias=0)")
    ax.bar(x + width, cal_f, width, label="Bias calibrated (bias=1)")

    for i, w in enumerate(windows):
        before_err = abs(fair_rates["Female"][w] - BJS_CUMULATIVE_GENDER[w]["Female"]) * 100.0
        after_err = abs(calibrated_rates["Female"][w] - BJS_CUMULATIVE_GENDER[w]["Female"]) * 100.0
        improvement = before_err - after_err
        y = max(bjs_f[i], fair_f[i], cal_f[i]) + 1.0
        ax.text(x[i], y, f"Error change: {improvement:+.1f} pp",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cumulative rearrest rate (%)")
    ax.set_title("Female Rearrest Fit to BJS\nBefore vs After Gender Bias Calibration",
                 fontweight="bold")
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.95)

    # Panel B: gender-gap recovery
    ax = axes[1]
    bjs_gap = [
        (BJS_CUMULATIVE_GENDER[w]["Male"] - BJS_CUMULATIVE_GENDER[w]["Female"]) * 100.0
        for w in windows
    ]
    fair_gap = [_gender_gap(fair_rates, w) for w in windows]
    cal_gap = [_gender_gap(calibrated_rates, w) for w in windows]

    ax.plot(x, bjs_gap, marker="D", linewidth=2.5, label="BJS Male–Female gap")
    ax.plot(x, fair_gap, marker="o", linewidth=2.2, linestyle="--",
            label="Fair baseline gap")
    ax.plot(x, cal_gap, marker="o", linewidth=2.5,
            label="Bias-calibrated gap")

    for i, w in enumerate(windows):
        before_gap_err = abs(fair_gap[i] - bjs_gap[i])
        after_gap_err = abs(cal_gap[i] - bjs_gap[i])
        improvement = before_gap_err - after_gap_err
        y = max(bjs_gap[i], fair_gap[i], cal_gap[i]) + 0.7
        ax.text(x[i], y, f"Gap-error change: {improvement:+.1f} pp",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Male − Female rearrest gap (percentage points)")
    ax.set_title("Recovery of the BJS Gender Disparity", fontweight="bold")
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.95)

    female_mae_before = np.mean([
        abs(fair_rates["Female"][w] - BJS_CUMULATIVE_GENDER[w]["Female"]) * 100.0
        for w in windows
    ])
    female_mae_after = np.mean([
        abs(calibrated_rates["Female"][w] - BJS_CUMULATIVE_GENDER[w]["Female"]) * 100.0
        for w in windows
    ])
    gap_mae_before = np.mean([
        abs(fair_gap[i] - bjs_gap[i]) for i in range(len(windows))
    ])
    gap_mae_after = np.mean([
        abs(cal_gap[i] - bjs_gap[i]) for i in range(len(windows))
    ])

    fig.suptitle(
        "Gender Bias Calibration Effect — Fair Baseline vs Calibrated Supervision Bias\n"
        f"Female MAE: {female_mae_before:.2f} → {female_mae_after:.2f} pp   |   "
        f"Gender-gap MAE: {gap_mae_before:.2f} → {gap_mae_after:.2f} pp",
        fontsize=14, fontweight="bold", y=0.99
    )
    fig.text(
        0.5, 0.015,
        "Paired comparison uses identical seeds and global calibration; only bias_factor changes from 0 to 1. "
        "Positive error-change values indicate improved BJS fit.",
        ha="center", va="bottom", fontsize=9
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.91])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Gender bias calibration effect chart -> {output_path}")


def save_gender_bias_effect_csv(effect_df, output_dir):
    path = os.path.join(output_dir, "gender_bias_calibration_effect.csv")
    effect_df.to_csv(path, index=False)
    print(f"  Gender bias calibration effect CSV -> {path}")


# =============================================================================
# LINE CHARTS (small-multiples)
# =============================================================================
def plot_small_multiples(abm_rates, output_path):
    """Two-panel line chart: Male | Female."""
    years = list(range(1, 10))
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.18,
                          left=0.07, right=0.97, top=0.87, bottom=0.11)

    y_max = max(BJS_TABLE3[1]["Male"], BJS_TABLE3[1]["Female"],
                abm_rates["Male"][1], abm_rates["Female"][1])
    y_max_top = max(0.55, y_max * 1.15)

    for col, sex in enumerate(["Male", "Female"]):
        ax = fig.add_subplot(gs[0, col])
        pal = PALETTE[sex]
        ax.set_facecolor("#FAFAFA")

        bjs_vals = [BJS_TABLE3[y][sex] for y in years]
        abm_vals = [abm_rates[sex][y]  for y in years]

        ax.fill_between(years,
                        [max(0, v - 0.02) for v in bjs_vals],
                        [v + 0.02         for v in bjs_vals],
                        color=pal["fill"], alpha=0.55, zorder=1,
                        label="BJS ±2pp tolerance")
        ax.plot(years, bjs_vals, color=pal["bjs"], linewidth=3.5,
                marker="D", markersize=11,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"BJS — {sex}", zorder=5)
        ax.plot(years, abm_vals, color=pal["abm"], linewidth=2.8,
                marker="o", markersize=10,
                markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=pal["abm"],
                label=f"ABM — {sex}", zorder=6)

        for y, bv, av in zip(years, bjs_vals, abm_vals):
            ax.plot([y, y], [bv, av], color="#BBBBBB", linewidth=1.0,
                    linestyle=":", zorder=3, alpha=0.7)
            gap = (av - bv) * 100
            ax.annotate(f"\u0394{gap:+.1f}",
                        xy=(y, max(bv, av)), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold",
                        color=_gap_color(gap))

        ord_labels = {1: "1st", 2: "2nd", 3: "3rd"}
        ax.set_xticks(years)
        ax.set_xticklabels([ord_labels.get(y, f"{y}th") for y in years],
                           fontsize=10)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0, y_max_top)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlabel("Year of first arrest", fontsize=11, labelpad=8)
        if col == 0:
            ax.set_ylabel("Percent of released prisoners", fontsize=11, labelpad=8)
        ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
                linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=10, framealpha=0.95, loc="upper right")

        t_bjs = BJS_TABLE3["total_9yr"][sex]
        t_abm = abm_rates[sex]["total_9yr"]
        mean_abs_gap = np.mean([abs(abm_rates[sex][y] - BJS_TABLE3[y][sex]) * 100
                                for y in years])
        ax.set_title(
            f"{sex}  (n = {abm_rates[sex]['n']:,})\n"
            f"9-yr total: BJS {t_bjs:.1%} vs ABM {t_abm:.1%}  "
            f"(\u0394 {(t_abm-t_bjs)*100:+.1f}pp)  |  "
            f"Mean |\u0394|/year: {mean_abs_gap:.1f}pp",
            fontsize=11, fontweight="bold", pad=12, color=pal["bjs"])

    fig.suptitle(
        "Study-Period First-Arrest Rate: ABM vs BJS Table 3 — Sex Comparison\n"
        "Male (left) and Female (right) panels  |  "
        "Alper, Durose & Markman (2018), NCJ 250975",
        fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart -> {output_path}")


def plot_race_small_multiples(abm_rates, output_path):
    """Three-panel line chart: White | Black | Hispanic."""
    years = list(range(1, 10))
    fig = plt.figure(figsize=(21, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.18,
                          left=0.06, right=0.98, top=0.87, bottom=0.11)

    y_max = max(max(BJS_TABLE3_RACE[1][r] for r in RACE_GROUPS),
                max(abm_rates[r][1]       for r in RACE_GROUPS))
    y_max_top = max(0.55, y_max * 1.15)
    ord_labels = {1: "1st", 2: "2nd", 3: "3rd"}

    for col, race in enumerate(RACE_GROUPS):
        ax = fig.add_subplot(gs[0, col])
        pal = PALETTE[race]
        ax.set_facecolor("#FAFAFA")

        bjs_vals = [BJS_TABLE3_RACE[y][race] for y in years]
        abm_vals = [abm_rates[race][y]        for y in years]

        ax.fill_between(years,
                        [max(0, v - 0.02) for v in bjs_vals],
                        [v + 0.02         for v in bjs_vals],
                        color=pal["fill"], alpha=0.55, zorder=1,
                        label="BJS ±2pp tolerance")
        ax.plot(years, bjs_vals, color=pal["bjs"], linewidth=3.5,
                marker="D", markersize=11,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"BJS — {race}", zorder=5)
        ax.plot(years, abm_vals, color=pal["abm"], linewidth=2.8,
                marker="o", markersize=10,
                markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=pal["abm"],
                label=f"ABM — {race}", zorder=6)

        for y, bv, av in zip(years, bjs_vals, abm_vals):
            ax.plot([y, y], [bv, av], color="#BBBBBB", linewidth=1.0,
                    linestyle=":", zorder=3, alpha=0.7)
            gap = (av - bv) * 100
            ax.annotate(f"\u0394{gap:+.1f}",
                        xy=(y, max(bv, av)), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold",
                        color=_gap_color(gap))

        ax.set_xticks(years)
        ax.set_xticklabels([ord_labels.get(y, f"{y}th") for y in years],
                           fontsize=10)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0, y_max_top)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlabel("Year of first arrest", fontsize=11, labelpad=8)
        if col == 0:
            ax.set_ylabel("Percent of released prisoners", fontsize=11, labelpad=8)
        ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
                linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=10, framealpha=0.95, loc="upper right")

        t_bjs = BJS_TABLE3_RACE["total_9yr"][race]
        t_abm = abm_rates[race]["total_9yr"]
        mean_abs_gap = np.mean([
            abs(abm_rates[race][y] - BJS_TABLE3_RACE[y][race]) * 100
            for y in years])
        ax.set_title(
            f"{race}  (n = {abm_rates[race]['n']:,})\n"
            f"9-yr total: BJS {t_bjs:.1%} vs ABM {t_abm:.1%}  "
            f"(\u0394 {(t_abm-t_bjs)*100:+.1f}pp)  |  "
            f"Mean |\u0394|/year: {mean_abs_gap:.1f}pp",
            fontsize=11, fontweight="bold", pad=12, color=pal["bjs"])

    fig.suptitle(
        "Study-Period First-Arrest Rate: ABM vs BJS Table 3 — Race Comparison\n"
        "White (left) | Black (centre) | Hispanic (right)  |  "
        "Alper, Durose & Markman (2018), NCJ 250975",
        fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart -> {output_path}")


# =============================================================================
# BAR CHART — GENDER
# =============================================================================
def plot_gender_bar(abm_rates, output_path):
    """
    Grouped bar chart: cumulative rearrest rate at 3, 6, 9 years by Gender.

    Layout
    ------
    Three window groups along the x-axis (3yr, 6yr, 9yr). Within each
    group, four bars in order: BJS Male | ABM Male | BJS Female | ABM Female.
    Gap annotations (Δ pp) are placed above each ABM bar, coloured by the
    same ±2/±5 pp traffic-light scheme used in the line charts.
    A horizontal dashed reference line marks the BJS value for each group
    to make the ABM over/under-shoot immediately visible.
    """
    groups  = ["Male", "Female"]
    windows = BAR_WINDOWS           # [3, 6, 9]
    n_win   = len(windows)
    n_grp   = len(groups)

    # ── Build value arrays ──────────────────────────────────────────────────
    # bjs_vals[w][g], abm_vals[w][g]
    bjs_vals = {
        w: {g: BJS_CUMULATIVE_GENDER[w][g] for g in groups}
        for w in windows
    }
    abm_vals = {
        w: {g: _cumulative(abm_rates, g, w) for g in groups}
        for w in windows
    }

    # ── Layout geometry ─────────────────────────────────────────────────────
    # Each window cluster occupies a width of 1.0.
    # Within a cluster: BJS Male, ABM Male, [gap], BJS Female, ABM Female
    bar_w    = 0.17   # individual bar width
    gap_w    = 0.08   # gap between gender sub-groups within a cluster
    # Offsets of the four bars relative to the cluster centre
    offsets = [-1.5*bar_w - gap_w/2,   # BJS Male
               -0.5*bar_w - gap_w/2,   # ABM Male
                0.5*bar_w + gap_w/2,   # BJS Female
                1.5*bar_w + gap_w/2]   # ABM Female

    x_centres = np.arange(n_win, dtype=float)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor("#FAFAFA")

    y_max_data = max(bjs_vals[w][g] for w in windows for g in groups)
    ax.set_ylim(0, min(1.0, y_max_data * 1.22))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    # Draw bars
    for wi, w in enumerate(windows):
        for gi, g in enumerate(groups):
            bv = bjs_vals[w][g]
            av = abm_vals[w][g]
            pal = PALETTE[g]
            x_base = x_centres[wi]

            # BJS bar — solid, full colour
            bjs_x = x_base + offsets[gi * 2]
            ax.bar(bjs_x, bv, width=bar_w,
                   color=pal["bjs"], alpha=0.90, zorder=3,
                   label=f"BJS {g}" if wi == 0 else "_")

            # ABM bar — hatched, lighter colour
            abm_x = x_base + offsets[gi * 2 + 1]
            ax.bar(abm_x, av, width=bar_w,
                   color=pal["abm"], alpha=0.90, hatch="///",
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"ABM {g}" if wi == 0 else "_")

            # Δ annotation above the ABM bar
            gap = (av - bv) * 100
            ax.text(abm_x, av + 0.005,
                    f"Δ{gap:+.1f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color=_gap_color(gap), zorder=5)

    # BJS reference dashed lines per window (one per gender, drawn at BJS level)
    for wi, w in enumerate(windows):
        for g in groups:
            bv = bjs_vals[w][g]
            pal = PALETTE[g]
            gi = groups.index(g)
            x_l = x_centres[wi] + offsets[gi * 2] - bar_w / 2
            x_r = x_centres[wi] + offsets[gi * 2 + 1] + bar_w / 2
            ax.hlines(bv, x_l, x_r,
                      colors=pal["bjs"], linewidths=1.4,
                      linestyles="--", alpha=0.7, zorder=4)

    # ── Axes dressing ────────────────────────────────────────────────────────
    ax.set_xticks(x_centres)
    ax.set_xticklabels([f"{w}-Year" for w in windows], fontsize=13)
    ax.set_xlim(-0.55, n_win - 0.45)
    ax.set_ylabel("Cumulative rearrest rate", fontsize=12, labelpad=8)
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
            linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # Sub-group labels (Male / Female) below the x-axis
    for wi, w in enumerate(windows):
        for gi, g in enumerate(groups):
            mid_x = x_centres[wi] + (offsets[gi*2] + offsets[gi*2+1]) / 2
            ax.text(mid_x, -ax.get_ylim()[1] * 0.045,
                    g, ha="center", va="top",
                    fontsize=8.5, color=PALETTE[g]["bjs"],
                    fontweight="bold",
                    transform=ax.get_xaxis_transform())

    # Legend: deduplicated (BJS/ABM × Male/Female = 4 entries)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, framealpha=0.95,
              ncol=2, loc="upper left")

    ax.set_title(
        "Cumulative Rearrest Rate at 3, 6, 9 Years by Gender: ABM vs BJS\n"
        "Solid = BJS reference  |  Hatched = ABM simulation  |  "
        "Δ = ABM − BJS (pp)  |  Alper, Durose & Markman (2018), NCJ 250975",
        fontsize=12, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart -> {output_path}")


# =============================================================================
# BAR CHART — RACE
# =============================================================================
def plot_race_bar(abm_rates, output_path):
    """
    Grouped bar chart: cumulative rearrest rate at 3, 6, 9 years by Race.

    Layout
    ------
    Three window groups along the x-axis (3yr, 6yr, 9yr). Within each
    group, six bars in order:
      BJS White | ABM White | BJS Black | ABM Black | BJS Hispanic | ABM Hispanic
    Gap annotations (Δ pp) are placed above each ABM bar.
    """
    windows = BAR_WINDOWS   # [3, 6, 9]
    n_win   = len(windows)

    bjs_vals = {
        w: {r: BJS_CUMULATIVE_RACE[w][r] for r in RACE_GROUPS}
        for w in windows
    }
    abm_vals = {
        w: {r: _cumulative(abm_rates, r, w) for r in RACE_GROUPS}
        for w in windows
    }

    # ── Layout geometry ─────────────────────────────────────────────────────
    bar_w  = 0.12
    gap_w  = 0.07   # gap between race sub-groups
    n_race = len(RACE_GROUPS)

    # Build offsets: [BJS_W, ABM_W, BJS_B, ABM_B, BJS_H, ABM_H]
    # Total cluster width = n_race * 2 * bar_w + (n_race - 1) * gap_w
    # Centre each pair, then space race groups by (2*bar_w + gap_w)
    race_slot_w = 2 * bar_w + gap_w
    cluster_w   = n_race * race_slot_w - gap_w
    offsets = []
    for ri in range(n_race):
        centre = -cluster_w / 2 + ri * race_slot_w + bar_w / 2
        offsets.extend([centre - bar_w / 2, centre + bar_w / 2])
    # offsets: [BJS_r0, ABM_r0, BJS_r1, ABM_r1, BJS_r2, ABM_r2]

    x_centres = np.arange(n_win, dtype=float)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor("#FAFAFA")

    y_max_data = max(bjs_vals[w][r] for w in windows for r in RACE_GROUPS)
    ax.set_ylim(0, min(1.0, y_max_data * 1.22))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    for wi, w in enumerate(windows):
        for ri, race in enumerate(RACE_GROUPS):
            bv  = bjs_vals[w][race]
            av  = abm_vals[w][race]
            pal = PALETTE[race]
            x_base = x_centres[wi]

            bjs_x = x_base + offsets[ri * 2]
            abm_x = x_base + offsets[ri * 2 + 1]

            ax.bar(bjs_x, bv, width=bar_w,
                   color=pal["bjs"], alpha=0.90, zorder=3,
                   label=f"BJS {race}" if wi == 0 else "_")
            ax.bar(abm_x, av, width=bar_w,
                   color=pal["abm"], alpha=0.90, hatch="///",
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"ABM {race}" if wi == 0 else "_")

            gap = (av - bv) * 100
            ax.text(abm_x, av + 0.004,
                    f"Δ{gap:+.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold",
                    color=_gap_color(gap), zorder=5)

            # BJS reference dashed line spanning both bars
            x_l = bjs_x - bar_w / 2
            x_r = abm_x + bar_w / 2
            ax.hlines(bv, x_l, x_r,
                      colors=pal["bjs"], linewidths=1.4,
                      linestyles="--", alpha=0.7, zorder=4)

    # ── Axes dressing ────────────────────────────────────────────────────────
    ax.set_xticks(x_centres)
    ax.set_xticklabels([f"{w}-Year" for w in windows], fontsize=13)
    ax.set_xlim(-0.60, n_win - 0.40)
    ax.set_ylabel("Cumulative rearrest rate", fontsize=12, labelpad=8)
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
            linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # Race sub-group labels below x-axis
    for wi, w in enumerate(windows):
        for ri, race in enumerate(RACE_GROUPS):
            mid_x = x_centres[wi] + (offsets[ri*2] + offsets[ri*2+1]) / 2
            ax.text(mid_x, -ax.get_ylim()[1] * 0.045,
                    race, ha="center", va="top",
                    fontsize=8, color=PALETTE[race]["bjs"],
                    fontweight="bold",
                    transform=ax.get_xaxis_transform())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, framealpha=0.95,
              ncol=3, loc="upper left")

    ax.set_title(
        "Cumulative Rearrest Rate at 3, 6, 9 Years by Race: ABM vs BJS\n"
        "Solid = BJS reference  |  Hatched = ABM simulation  |  "
        "Δ = ABM − BJS (pp)  |  Alper, Durose & Markman (2018), NCJ 250975",
        fontsize=12, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart -> {output_path}")


# =============================================================================
# CSV EXPORT
# =============================================================================
def save_gender_csv(rates, output_dir):
    rows = []
    for sex in ["Male", "Female"]:
        for y in range(1, 10):
            rows.append({
                "Gender":   sex,
                "year":     y,
                "abm_rate": rates[sex][y],
                "bjs_rate": BJS_TABLE3[y][sex],
                "gap_pp":   (rates[sex][y] - BJS_TABLE3[y][sex]) * 100,
                "n":        rates[sex]["n"],
            })
        rows.append({
            "Gender":   sex,
            "year":     "Total 9yr",
            "abm_rate": rates[sex]["total_9yr"],
            "bjs_rate": BJS_TABLE3["total_9yr"][sex],
            "gap_pp":   (rates[sex]["total_9yr"] -
                         BJS_TABLE3["total_9yr"][sex]) * 100,
            "n":        rates[sex]["n"],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "gender_rearrest_by_year.csv")
    df.to_csv(path, index=False)
    print(f"  CSV  -> {path}")


def save_race_csv(rates, output_dir):
    rows = []
    for race in RACE_GROUPS:
        for y in range(1, 10):
            rows.append({
                "Race":     race,
                "year":     y,
                "abm_rate": rates[race][y],
                "bjs_rate": BJS_TABLE3_RACE[y][race],
                "gap_pp":   (rates[race][y] - BJS_TABLE3_RACE[y][race]) * 100,
                "n":        rates[race]["n"],
            })
        rows.append({
            "Race":     race,
            "year":     "Total 9yr",
            "abm_rate": rates[race]["total_9yr"],
            "bjs_rate": BJS_TABLE3_RACE["total_9yr"][race],
            "gap_pp":   (rates[race]["total_9yr"] -
                         BJS_TABLE3_RACE["total_9yr"][race]) * 100,
            "n":        rates[race]["n"],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "race_rearrest_by_year.csv")
    df.to_csv(path, index=False)
    print(f"  CSV  -> {path}")


# =============================================================================
# CORE DETECTION
# =============================================================================
def detect_workers():
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return max(1, physical - 1)
    except ImportError:
        pass
    import multiprocessing
    return max(1, multiprocessing.cpu_count() // 2 - 1)


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    n_workers = detect_workers()

    _HEADER_KEYS = (
        "Risk_Contrast_Strength",
        "Supervision_Monitoring_Intensity",
        "Risk_Effect_Decay_After_1Y",
        "Risk_Effect_Decay_After_3Y",
        "Risk_Effect_Decay_After_6Y",
        "Supervision_Monitoring_Decay_After_3Y",
        "Supervision_Monitoring_Decay_After_6Y",
        "offense_hazard_shift",
    )

    print("="*76)
    print("  END-TO-END DRIVER — Gender & Race x Year-of-First-Arrest Validation")
    print("  Derivation source: community_months_at_risk")
    print("  (matches semantic of rearrest_{1,3,6,9}_yrs flags in agent)")
    print("="*76)
    print(f"  Seeds       : {len(CONFIG['seeds'])}")
    print(f"  Reps/seed   : {CONFIG['n_reps']}")
    print(f"  Total runs  : {len(CONFIG['seeds']) * CONFIG['n_reps']}")
    print(f"  Agents/run  : {CONFIG['initial_agents']}")
    print(f"  Warmup/Study: {CONFIG['warmup_months']} / {CONFIG['study_months']} months")
    print(f"  Workers     : {n_workers}")
    print(f"  Output dir  : {CONFIG['output_dir']}")
    print()
    print("  Calibration parameters (from get_global_calibration_params()):")

    for k in _HEADER_KEYS:
        v = _CALIBRATION_PARAMS.get(k)

        if isinstance(v, dict):
            print(f"    {k}:")
            for sub_k, sub_v in v.items():
                print(f"      {sub_k:<30s} {sub_v}")
        else:
            tag = "  [BJS-anchored]" if "Risk_Effect_Decay" in k else ""
            print(f"    {k:<45s} {v}{tag}")


    # ─────────────────────────────────────────────────────────────────────
    # Phase 2 bias / override parameters
    # ─────────────────────────────────────────────────────────────────────

    print("\n  Phase 2 Bias / Override Parameters:")

    bias_factor = float(_CALIBRATION_PARAMS.get("bias_factor", 0.0))
    bias_scope = _CALIBRATION_PARAMS.get("bias_scope", "supervision_only")

    print(f"    {'bias_factor':<45s} {bias_factor}")
    print(f"    {'bias_scope':<45s} {bias_scope}")


    # Race bias
    race_bias = _CALIBRATION_PARAMS.get("group_bias_race", {})

    print("    group_bias_race:")
    for group, beta in race_bias.items():
        effective_beta = bias_factor * float(beta)
        odds_ratio = math.exp(effective_beta)

        print(
            f"      {group:<12s}"
            f" beta={beta:>7.3f}"
            f"   effective_beta={effective_beta:>7.3f}"
            f"   OR={odds_ratio:>6.3f}x"
        )


    # Gender bias
    gender_bias = _CALIBRATION_PARAMS.get("group_bias_gender", {})

    print("    group_bias_gender:")
    for group, beta in gender_bias.items():
        effective_beta = bias_factor * float(beta)
        odds_ratio = math.exp(effective_beta)

        print(
            f"      {group:<12s}"
            f" beta={beta:>7.3f}"
            f"   effective_beta={effective_beta:>7.3f}"
            f"   OR={odds_ratio:>6.3f}x"
        )

    print("=" * 76)

    # ── Step 1: Run the bias-calibrated model ───────────────────────────────
    print("\nSTEP 1: Running bias-calibrated model across seeds...")
    df = run_all_seeds_parallel(
        n_workers,
        bias_factor=CONFIG["bias_factor"],
        label="Bias calibrated"
    )

    pool_path = os.path.join(CONFIG["output_dir"], "agent_cohort_pooled.csv")
    df.to_csv(pool_path, index=False)
    print(f"  Pooled agent data -> {pool_path}")
    print(f"  Study-eligible agents: {len(df):,}")

    gender_counts = df["Gender"].value_counts()
    print(f"\n  Sex distribution:")
    for sex in ["Male", "Female"]:
        n = gender_counts.get(sex, 0)
        print(f"    {sex:<8}: {n:>7,} ({n/len(df):.1%})")

    race_counts = df["Race"].value_counts()
    print(f"\n  Race distribution (ABM agents):")
    for race in ["White", "Black", "Hispanic", "Other"]:
        n = race_counts.get(race, 0)
        suffix = "  [excluded from race charts]" if race == "Other" else ""
        print(f"    {race:<10}: {n:>7,} ({n/len(df):.1%}){suffix}")


    # ── Step 1b: Paired fair-baseline run for Gender Bias Calibration Effect ──
    print("\nSTEP 1b: Running paired FAIR baseline (bias_factor=0)...")
    df_fair = run_all_seeds_parallel(
        n_workers,
        bias_factor=0.0,
        label="Fair baseline"
    )

    fair_pool_path = os.path.join(
        CONFIG["output_dir"], "agent_cohort_pooled_fair_baseline.csv"
    )
    df_fair.to_csv(fair_pool_path, index=False)
    print(f"  Fair-baseline pooled agent data -> {fair_pool_path}")

    rates_gender_direct_cal = compute_direct_cumulative_gender_rates(df)
    rates_gender_direct_fair = compute_direct_cumulative_gender_rates(df_fair)

    effect_df = compute_gender_bias_effect_table(
        rates_gender_direct_fair,
        rates_gender_direct_cal
    )

    print("\n  GENDER BIAS CALIBRATION EFFECT (direct 3/6/9-year flags)")
    print("  " + "-" * 96)
    print(effect_df.to_string(
        index=False,
        formatters={
            "BJS": lambda x: f"{x:.1%}",
            "Fair_Baseline": lambda x: f"{x:.1%}",
            "Bias_Calibrated": lambda x: f"{x:.1%}",
            "Error_Before_pp": lambda x: f"{x:.2f}",
            "Error_After_pp": lambda x: f"{x:.2f}",
            "Improvement_pp": lambda x: f"{x:+.2f}",
        }
    ))

    plot_gender_bias_calibration_effect(
        rates_gender_direct_fair,
        rates_gender_direct_cal,
        os.path.join(CONFIG["output_dir"],
                     "chart_gender_bias_calibration_effect.png")
    )
    save_gender_bias_effect_csv(effect_df, CONFIG["output_dir"])

    # ── Step 2: Derive year-of-first-arrest ─────────────────────────────────
    print("\nSTEP 2: Deriving year-of-first-arrest from community_months_at_risk...")
    df = derive_first_arrest_year(df, warmup_months=CONFIG["warmup_months"])

    # ── Step 3: Compute rates ────────────────────────────────────────────────
    print("\nSTEP 3: Computing year-by-year rates by Gender...")
    rates_gender = compute_rates_by_sex(df)

    print(f"\n  YEAR-BY-YEAR RATES (Gender):")
    print(f"    {'Year':<8}{'BJS Male':>10}{'ABM Male':>10}"
          f"{'BJS Female':>12}{'ABM Female':>12}{'delta M':>10}{'delta F':>10}")
    print("    " + "-"*72)
    for y in range(1, 10):
        bjs_m, abm_m = BJS_TABLE3[y]["Male"],   rates_gender["Male"][y]
        bjs_f, abm_f = BJS_TABLE3[y]["Female"], rates_gender["Female"][y]
        print(f"    Year {y:<3}{bjs_m:>10.1%}{abm_m:>10.1%}"
              f"{bjs_f:>12.1%}{abm_f:>12.1%}"
              f"{(abm_m-bjs_m)*100:>+9.1f}pp"
              f"{(abm_f-bjs_f)*100:>+9.1f}pp")
    print(f"\n    {'Total':<8}{BJS_TABLE3['total_9yr']['Male']:>10.1%}"
          f"{rates_gender['Male']['total_9yr']:>10.1%}"
          f"{BJS_TABLE3['total_9yr']['Female']:>12.1%}"
          f"{rates_gender['Female']['total_9yr']:>12.1%}")

    print("\nSTEP 3b: Computing year-by-year rates by Race...")
    rates_race = compute_rates_by_race(df)

    print(f"\n  YEAR-BY-YEAR RATES (Race):")
    hdr = (f"    {'Year':<8}"
           + "".join(f"{'BJS '+r:>11}{'ABM '+r:>11}" for r in RACE_GROUPS))
    print(hdr)
    print("    " + "-"*80)
    for y in range(1, 10):
        row = f"    Year {y:<3}"
        for race in RACE_GROUPS:
            row += f"{BJS_TABLE3_RACE[y][race]:>11.1%}{rates_race[race][y]:>11.1%}"
        print(row)
    total_row = f"    {'Total':<8}"
    for race in RACE_GROUPS:
        total_row += (f"{BJS_TABLE3_RACE['total_9yr'][race]:>11.1%}"
                      f"{rates_race[race]['total_9yr']:>11.1%}")
    print(f"\n{total_row}")

    # ── Step 4: Gender line chart ─────────────────────────────────────────────
    print("\nSTEP 4: Generating gender small-multiples line chart...")
    plot_small_multiples(rates_gender,
                         os.path.join(CONFIG["output_dir"],
                                      "chart_gender_small_multiples.png"))

    # ── Step 5: Gender bar chart ──────────────────────────────────────────────
    print("\nSTEP 5: Generating gender bar chart (cumulative 3/6/9yr)...")
    plot_gender_bar(rates_gender,
                    os.path.join(CONFIG["output_dir"], "chart_gender_bar.png"))

    # ── Step 6: Gender CSV ────────────────────────────────────────────────────
    print("\nSTEP 6: Saving gender CSV...")
    save_gender_csv(rates_gender, CONFIG["output_dir"])

    # ── Step 7: Race line chart ───────────────────────────────────────────────
    print("\nSTEP 7: Generating race small-multiples line chart...")
    plot_race_small_multiples(rates_race,
                              os.path.join(CONFIG["output_dir"],
                                           "chart_race_small_multiples.png"))

    # ── Step 8: Race bar chart ────────────────────────────────────────────────
    print("\nSTEP 8: Generating race bar chart (cumulative 3/6/9yr)...")
    plot_race_bar(rates_race,
                  os.path.join(CONFIG["output_dir"], "chart_race_bar.png"))

    # ── Step 9: Race CSV ──────────────────────────────────────────────────────
    print("\nSTEP 9: Saving race CSV...")
    save_race_csv(rates_race, CONFIG["output_dir"])

    print("\n" + "="*76)
    print("  COMPLETE")
    print("="*76)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()