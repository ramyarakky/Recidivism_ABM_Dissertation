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
  5. Produces a two-panel small-multiples chart:
       - Male panel   (ABM vs BJS)
       - Female panel (ABM vs BJS)
  6. Produces a three-panel small-multiples chart:
       - White panel    (ABM vs BJS)
       - Black panel    (ABM vs BJS)
       - Hispanic panel (ABM vs BJS)
     Note: "Other" race category is excluded; BJS cell sizes for Other
     are small and the category is heterogeneous.

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
  - validation_output_demographics/chart_race_small_multiples.png
  - validation_output_demographics/agent_cohort_pooled.csv
  - validation_output_demographics/gender_rearrest_by_year.csv
  - validation_output_demographics/race_rearrest_by_year.csv
"""

import os
import sys
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

# Resolved once at import time so the console header can print the values
# and so that all worker processes share an identical parameter snapshot.
_CALIBRATION_PARAMS = get_global_calibration_params()


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "initial_agents":         1500,
    "monthly_intake":         10,
    "warmup_months":          144,
    "study_months":           108,
    "bias_factor":            0.0,
    "enable_peer_influence":  True,
    "output_dir":             "validation_output_demographics",
    "seeds":                  [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "n_reps":                 1,
}

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
    "good":         "#2E7D32",  # green — within ±2pp
    "warn":         "#F57C00",  # amber — within ±5pp
    "bad":          "#C62828",  # red — exceeds ±5pp
    "grid":         "#E0E0E0",
}


def _gap_color(gap_pp):
    if abs(gap_pp) <= 2.0: return C["good"]
    if abs(gap_pp) <= 5.0: return C["warn"]
    return C["bad"]


# =============================================================================
# MODEL RUNNER (parallel worker)
# =============================================================================
def _run_one(args):
    seed, rep = args
    effective_seed = seed * 1000 + rep

    try:
        model = RecidivismModel(
            initial_agents        = CONFIG["initial_agents"],
            bias_factor           = CONFIG["bias_factor"],
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

        # Extract per-agent study-period data
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


def run_all_seeds_parallel(n_workers):
    jobs = [(s, r) for s in CONFIG["seeds"] for r in range(CONFIG["n_reps"])]
    print(f"  Running {len(jobs)} simulations "
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

    Under absorbing semantics (first rearrest ends the agent's study
    participation), the primary and Fallback 1 produce identical values for
    agents with no mid-study incarceration. They diverge only for agents who
    were revoked-and-reincarcerated before their first study-period rearrest,
    where community_months_at_risk < (rearrest_month - warmup_months).
    """
    df = df.copy()
    n = len(df)

    rearrested = df["recidivated_agent"].astype(bool).values

    # Primary source: community_months_at_risk
    year = np.full(n, np.nan)
    source = "community_months_at_risk (primary)"

    cm = pd.to_numeric(df["community_months_at_risk"],
                        errors="coerce").values
    valid_cm = rearrested & ~np.isnan(cm) & (cm > 0)
    year[valid_cm] = np.ceil(cm[valid_cm] / 12.0)

    # Fallback 1: rearrest_month - warmup_months
    # For any rearrested agents where community_months_at_risk is unusable
    missing_mask = rearrested & np.isnan(year)
    if missing_mask.any() and "rearrest_month" in df.columns:
        print(f"  {int(missing_mask.sum()):,} rearrested agents missing "
              "community_months_at_risk — deriving from rearrest_month")
        rm = pd.to_numeric(df["rearrest_month"],
                            errors="coerce").values
        rm_fallback = rm - warmup_months
        use_rm = missing_mask & ~np.isnan(rm) & (rm_fallback > 0)
        year[use_rm] = np.ceil(rm_fallback[use_rm] / 12.0)

    # Fallback 2: rearrest_year attribute (only if still valid [1,9])
    missing_mask = rearrested & np.isnan(year)
    if missing_mask.any() and "rearrest_year" in df.columns:
        print(f"  {int(missing_mask.sum()):,} rearrested agents still missing "
              "— trying rearrest_year attribute")
        ry = pd.to_numeric(df["rearrest_year"],
                            errors="coerce").values
        use_ry = missing_mask & ~np.isnan(ry) & (ry >= 1) & (ry <= 9)
        year[use_ry] = ry[use_ry]

    # Final clipping to valid range [1, 9]
    year = np.where(
        (~np.isnan(year)) & (year >= 1) & (year <= 9),
        np.clip(year, 1, 9),
        np.nan
    )
    df["year_of_first_arrest"] = year

    # Diagnostics
    n_rearrested = int(rearrested.sum())
    n_assigned   = int((~np.isnan(year)).sum())
    print(f"  Source                 : {source}")
    print(f"  Agents total           : {n:,}")
    print(f"  Agents rearrested      : {n_rearrested:,} "
          f"({n_rearrested/n:.1%})")
    print(f"  Assigned year-of-first : {n_assigned:,} "
          f"({n_assigned/n:.1%})")

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
            print(f"    Year {y}: {year_hist[y-1]:>6,} "
                  f"({pct:>5.1%})  {bar}")

    return df


# =============================================================================
# YEAR-BY-YEAR RATE COMPUTATION BY SEX
# =============================================================================
def compute_rates_by_sex(df):
    """Compute per-year first-arrest rate for Male and Female."""
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


# =============================================================================
# SMALL-MULTIPLES CHART
# =============================================================================
def plot_small_multiples(abm_rates, output_path):
    years = list(range(1, 10))

    # Two-panel layout: Male (left) and Female (right), side by side.
    # No gradient panel — per-sex panels are the focus.
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(
        1, 2,
        wspace=0.18,
        left=0.07, right=0.97, top=0.87, bottom=0.11,
    )

    y_max = max(
        BJS_TABLE3[1]["Male"], BJS_TABLE3[1]["Female"],
        abm_rates["Male"][1], abm_rates["Female"][1],
    )
    y_max_top = max(0.55, y_max * 1.15)

    # Per-sex panels (Male on left, Female on right)
    for col, sex in enumerate(["Male", "Female"]):
        ax = fig.add_subplot(gs[0, col])
        pal = PALETTE[sex]
        ax.set_facecolor("#FAFAFA")

        bjs_vals = [BJS_TABLE3[y][sex] for y in years]
        abm_vals = [abm_rates[sex][y]  for y in years]

        ax.fill_between(
            years,
            [max(0, v - 0.02) for v in bjs_vals],
            [v + 0.02 for v in bjs_vals],
            color=pal["fill"], alpha=0.55, zorder=1,
            label="BJS \u00b12pp tolerance",
        )

        ax.plot(
            years, bjs_vals,
            color=pal["bjs"], linewidth=3.5,
            marker="D", markersize=11,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"BJS \u2014 {sex}", zorder=5,
        )

        ax.plot(
            years, abm_vals,
            color=pal["abm"], linewidth=2.8,
            marker="o", markersize=10,
            markerfacecolor="white", markeredgewidth=2.5,
            markeredgecolor=pal["abm"],
            label=f"ABM \u2014 {sex}", zorder=6,
        )

        for y, bv, av in zip(years, bjs_vals, abm_vals):
            ax.plot([y, y], [bv, av],
                    color="#BBBBBB", linewidth=1.0,
                    linestyle=":", zorder=3, alpha=0.7)

        for y, bv, av in zip(years, bjs_vals, abm_vals):
            gap = (av - bv) * 100
            clr = _gap_color(gap)
            ax.annotate(
                f"\u0394{gap:+.1f}",
                xy=(y, max(bv, av)),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", color=clr,
            )

        ord_labels = {1: "1st", 2: "2nd", 3: "3rd"}
        ax.set_xticks(years)
        ax.set_xticklabels([ord_labels.get(y, f"{y}th") for y in years],
                            fontsize=10)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0, y_max_top)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        ax.set_xlabel("Year of first arrest", fontsize=11, labelpad=8)
        if col == 0:
            ax.set_ylabel("Percent of released prisoners",
                          fontsize=11, labelpad=8)

        ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
                linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=10, framealpha=0.95, loc="upper right")

        t_bjs = BJS_TABLE3["total_9yr"][sex]
        t_abm = abm_rates[sex]["total_9yr"]
        n_agents = abm_rates[sex]["n"]
        mean_abs_gap = np.mean([
            abs(abm_rates[sex][y] - BJS_TABLE3[y][sex]) * 100
            for y in years
        ])
        gap_9yr = (t_abm - t_bjs) * 100

        title_text = (
            f"{sex}  (n = {n_agents:,})\n"
            f"9-yr total: BJS {t_bjs:.1%} vs ABM {t_abm:.1%}  "
            f"(\u0394 {gap_9yr:+.1f}pp)  |  "
            f"Mean |\u0394|/year: {mean_abs_gap:.1f}pp"
        )
        ax.set_title(title_text, fontsize=11, fontweight="bold",
                     pad=12, color=pal["bjs"])

    fig.suptitle(
        "Study-Period First-Arrest Rate: ABM vs BJS Table 3 \u2014 Sex Comparison\n"
        "Male (left) and Female (right) panels  |  "
        "Alper, Durose & Markman (2018), NCJ 250975",
        fontsize=13, fontweight="bold", y=0.98,
    )

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


# =============================================================================
# YEAR-BY-YEAR RATE COMPUTATION BY RACE
# =============================================================================
def compute_rates_by_race(df):
    """
    Compute per-year first-arrest rate for White, Black, and Hispanic.

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


# =============================================================================
# RACE SMALL-MULTIPLES CHART
# =============================================================================
def plot_race_small_multiples(abm_rates, output_path):
    """Three-panel chart: White | Black | Hispanic  (ABM vs BJS)."""
    years = list(range(1, 10))

    fig = plt.figure(figsize=(21, 7))
    gs = fig.add_gridspec(
        1, 3,
        wspace=0.18,
        left=0.06, right=0.98, top=0.87, bottom=0.11,
    )

    y_max = max(
        max(BJS_TABLE3_RACE[1][r] for r in RACE_GROUPS),
        max(abm_rates[r][1]       for r in RACE_GROUPS),
    )
    y_max_top = max(0.55, y_max * 1.15)

    ord_labels = {1: "1st", 2: "2nd", 3: "3rd"}

    for col, race in enumerate(RACE_GROUPS):
        ax = fig.add_subplot(gs[0, col])
        pal = PALETTE[race]
        ax.set_facecolor("#FAFAFA")

        bjs_vals = [BJS_TABLE3_RACE[y][race] for y in years]
        abm_vals = [abm_rates[race][y]        for y in years]

        # ±2pp tolerance band
        ax.fill_between(
            years,
            [max(0, v - 0.02) for v in bjs_vals],
            [v + 0.02          for v in bjs_vals],
            color=pal["fill"], alpha=0.55, zorder=1,
            label="BJS \u00b12pp tolerance",
        )

        # BJS reference line
        ax.plot(
            years, bjs_vals,
            color=pal["bjs"], linewidth=3.5,
            marker="D", markersize=11,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"BJS \u2014 {race}", zorder=5,
        )

        # ABM simulation line
        ax.plot(
            years, abm_vals,
            color=pal["abm"], linewidth=2.8,
            marker="o", markersize=10,
            markerfacecolor="white", markeredgewidth=2.5,
            markeredgecolor=pal["abm"],
            label=f"ABM \u2014 {race}", zorder=6,
        )

        # Vertical connectors between BJS and ABM points
        for y, bv, av in zip(years, bjs_vals, abm_vals):
            ax.plot([y, y], [bv, av],
                    color="#BBBBBB", linewidth=1.0,
                    linestyle=":", zorder=3, alpha=0.7)

        # Δ annotations
        for y, bv, av in zip(years, bjs_vals, abm_vals):
            gap = (av - bv) * 100
            ax.annotate(
                f"\u0394{gap:+.1f}",
                xy=(y, max(bv, av)),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold",
                color=_gap_color(gap),
            )

        ax.set_xticks(years)
        ax.set_xticklabels([ord_labels.get(y, f"{y}th") for y in years],
                           fontsize=10)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0, y_max_top)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        ax.set_xlabel("Year of first arrest", fontsize=11, labelpad=8)
        if col == 0:
            ax.set_ylabel("Percent of released prisoners",
                          fontsize=11, labelpad=8)

        ax.grid(True, axis="y", color=C["grid"], linewidth=0.6,
                linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=10, framealpha=0.95, loc="upper right")

        t_bjs = BJS_TABLE3_RACE["total_9yr"][race]
        t_abm = abm_rates[race]["total_9yr"]
        n_agents = abm_rates[race]["n"]
        mean_abs_gap = np.mean([
            abs(abm_rates[race][y] - BJS_TABLE3_RACE[y][race]) * 100
            for y in years
        ])
        gap_9yr = (t_abm - t_bjs) * 100

        title_text = (
            f"{race}  (n = {n_agents:,})\n"
            f"9-yr total: BJS {t_bjs:.1%} vs ABM {t_abm:.1%}  "
            f"(\u0394 {gap_9yr:+.1f}pp)  |  "
            f"Mean |\u0394|/year: {mean_abs_gap:.1f}pp"
        )
        ax.set_title(title_text, fontsize=11, fontweight="bold",
                     pad=12, color=pal["bjs"])

    fig.suptitle(
        "Study-Period First-Arrest Rate: ABM vs BJS Table 3 \u2014 Race Comparison\n"
        "White (left) | Black (centre) | Hispanic (right)  |  "
        "Alper, Durose & Markman (2018), NCJ 250975",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart -> {output_path}")


# =============================================================================
# RACE CSV EXPORT
# =============================================================================
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

    # Stage 1/2/3 keys to surface in the console header.
    # Bias and group_bias keys are omitted — they are all zero at the
    # fair-baseline and add noise to the header without diagnostic value.
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
    print("  END-TO-END DRIVER \u2014 Gender & Race x Year-of-First-Arrest Validation")
    print("  Derivation source: community_months_at_risk")
    print("  (matches semantic of rearrest_{1,3,6,9}_yrs flags in agent)")
    print("="*76)
    print(f"  Seeds       : {len(CONFIG['seeds'])}")
    print(f"  Reps/seed   : {CONFIG['n_reps']}")
    print(f"  Total runs  : {len(CONFIG['seeds']) * CONFIG['n_reps']}")
    print(f"  Agents/run  : {CONFIG['initial_agents']}")
    print(f"  Warmup/Study: {CONFIG['warmup_months']} / "
          f"{CONFIG['study_months']} months")
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
    print("="*76)

    # ── Step 1: Run the model ────────────────────────────────────────────────
    print("\nSTEP 1: Running calibrated model across seeds...")
    df = run_all_seeds_parallel(n_workers)

    pool_path = os.path.join(CONFIG["output_dir"], "agent_cohort_pooled.csv")
    df.to_csv(pool_path, index=False)
    print(f"  Pooled agent data -> {pool_path}")
    print(f"  Study-eligible agents: {len(df):,}")

    # Quick sex distribution check
    gender_counts = df["Gender"].value_counts()
    print(f"\n  Sex distribution:")
    for sex in ["Male", "Female"]:
        n = gender_counts.get(sex, 0)
        print(f"    {sex:<8}: {n:>7,} ({n/len(df):.1%})")

    # Quick race distribution check
    race_counts = df["Race"].value_counts()
    print(f"\n  Race distribution (ABM agents):")
    for race in ["White", "Black", "Hispanic", "Other"]:
        n = race_counts.get(race, 0)
        print(f"    {race:<10}: {n:>7,} ({n/len(df):.1%})"
              + ("  [excluded from race chart]" if race == "Other" else ""))

    # ── Step 2: Derive year-of-first-arrest (study period) ──────────────────
    print("\nSTEP 2: Deriving year-of-first-arrest from "
          "community_months_at_risk...")
    df = derive_first_arrest_year(df, warmup_months=CONFIG["warmup_months"])

    # ── Step 3: Compute year-by-year rates by sex ──────────────────────────
    print("\nSTEP 3: Computing year-by-year rates by Gender...")
    rates_gender = compute_rates_by_sex(df)

    # Console summary — Gender
    print(f"\n  YEAR-BY-YEAR RATES (Gender):")
    print(f"    {'Year':<8}{'BJS Male':>10}{'ABM Male':>10}"
          f"{'BJS Female':>12}{'ABM Female':>12}{'delta M':>10}{'delta F':>10}")
    print("    " + "-"*72)
    for y in range(1, 10):
        bjs_m = BJS_TABLE3[y]["Male"]
        abm_m = rates_gender["Male"][y]
        bjs_f = BJS_TABLE3[y]["Female"]
        abm_f = rates_gender["Female"][y]
        d_m = (abm_m - bjs_m) * 100
        d_f = (abm_f - bjs_f) * 100
        print(f"    Year {y:<3}{bjs_m:>10.1%}{abm_m:>10.1%}"
              f"{bjs_f:>12.1%}{abm_f:>12.1%}"
              f"{d_m:>+9.1f}pp{d_f:>+9.1f}pp")
    print(f"\n    {'Total':<8}{BJS_TABLE3['total_9yr']['Male']:>10.1%}"
          f"{rates_gender['Male']['total_9yr']:>10.1%}"
          f"{BJS_TABLE3['total_9yr']['Female']:>12.1%}"
          f"{rates_gender['Female']['total_9yr']:>12.1%}")

    # ── Step 3b: Compute year-by-year rates by race ─────────────────────────
    print("\nSTEP 3b: Computing year-by-year rates by Race...")
    rates_race = compute_rates_by_race(df)

    # Console summary — Race
    print(f"\n  YEAR-BY-YEAR RATES (Race):")
    hdr = (f"    {'Year':<8}"
           + "".join(f"{'BJS '+r:>11}{'ABM '+r:>11}" for r in RACE_GROUPS))
    print(hdr)
    print("    " + "-"*80)
    for y in range(1, 10):
        row = f"    Year {y:<3}"
        for race in RACE_GROUPS:
            bv = BJS_TABLE3_RACE[y][race]
            av = rates_race[race][y]
            row += f"{bv:>11.1%}{av:>11.1%}"
        print(row)
    total_row = f"    {'Total':<8}"
    for race in RACE_GROUPS:
        total_row += (f"{BJS_TABLE3_RACE['total_9yr'][race]:>11.1%}"
                      f"{rates_race[race]['total_9yr']:>11.1%}")
    print(f"\n{total_row}")

    # ── Step 4: Generate gender chart ──────────────────────────────────────
    print("\nSTEP 4: Generating gender small-multiples chart...")
    chart_path = os.path.join(CONFIG["output_dir"],
                              "chart_gender_small_multiples.png")
    plot_small_multiples(rates_gender, chart_path)

    # ── Step 5: Save gender CSV ─────────────────────────────────────────────
    print("\nSTEP 5: Saving gender CSV...")
    save_gender_csv(rates_gender, CONFIG["output_dir"])

    # ── Step 6: Generate race chart ─────────────────────────────────────────
    print("\nSTEP 6: Generating race small-multiples chart...")
    race_chart_path = os.path.join(CONFIG["output_dir"],
                                   "chart_race_small_multiples.png")
    plot_race_small_multiples(rates_race, race_chart_path)

    # ── Step 7: Save race CSV ───────────────────────────────────────────────
    print("\nSTEP 7: Saving race CSV...")
    save_race_csv(rates_race, CONFIG["output_dir"])

    print("\n" + "="*76)
    print("  COMPLETE")
    print("="*76)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()