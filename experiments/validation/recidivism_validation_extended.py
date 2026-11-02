"""
recidivism_validation_extended.py
=================================
Additional BJS NCJ 250975 validations to complement the primary
aggregate + offense-stratified charts (Charts 1-5).

Produces three additional dissertation charts:
  Chart 6  - Desistance (no-arrest) group validation
  Chart 7  - Sex-stratified rearrest validation (male / female)
  Chart 8  - Age-at-release stratified rearrest validation

All validations use BJS Alper, Durose & Markman (2018, NCJ 250975),
Tables 3 and 5.

Expected inputs:
  - study_cohort CSV(s) from run_replicated(label="final")
    with columns: Gender, Age_at_Release,
    rearrest_3_yrs, rearrest_6_yrs, rearrest_9_yrs

Expected outputs:
  - chart6_desistance.png / .html
  - chart7_sex_stratified.png / .html
  - chart8_age_stratified.png / .html
  - validation_extended_raw.csv  (all three validation tables merged)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "input_glob":  "Results_Run1/*_study_cohort_*.csv",
    "output_dir":  "validation_output",
}

# BJS Alper et al. (2018), NCJ 250975, Table 3 - 9-year cumulative rearrest
# by demographic group
BJS_TARGETS = {
    "aggregate":       0.834,   # Table 2, year 9
    "desistance_9yr":  0.166,   # 1 - aggregate  (17% never arrested in 9 years)

    # Sex (Table 3)
    "sex": {
        "Male":   {3: 0.697, 6: 0.806, 9: 0.843},
        "Female": {3: 0.592, 6: 0.719, 9: 0.770},
    },

    # Age at release (Table 3) - 9-year cumulative
    "age_at_release": {
        "24 or younger": {3: 0.768, 6: 0.876, 9: 0.904},
        "25-29":         {3: 0.711, 6: 0.826, 9: 0.864},
        "30-39":         {3: 0.675, 6: 0.788, 9: 0.827},
        "40 or older":   {3: 0.603, 6: 0.706, 9: 0.752},
    },
}


# =============================================================================
# COLOR PALETTE (matches your existing validation script)
# =============================================================================
C = {
    "abm":        "#1f77b4",
    "bjs":        "#D05A28",
    "good":       "#276419",
    "warn":       "#B8860B",
    "bad":        "#CC4400",
    "grid":       "#DDDDDD",
    "Male":       "#2166AC",
    "Female":     "#D6604D",
    "24 or younger": "#CC4400",
    "25-29":         "#F4A582",
    "30-39":         "#74ADD1",
    "40 or older":   "#2166AC",
}


def _gap_color(gap):
    if abs(gap) <= 0.02: return C["good"]
    if abs(gap) <= 0.05: return C["warn"]
    return C["bad"]


def _gap_flag(gap):
    if abs(gap) <= 0.02: return "within 2pp"
    if abs(gap) <= 0.05: return "within 5pp"
    return "exceeds 5pp"


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


# =============================================================================
# DATA LOADING
# =============================================================================
def load_study_cohorts(input_glob):
    """
    Load all study-cohort CSVs from the final calibrated runs.

    One CSV per seed/replication. Concatenated into a single DataFrame
    so per-seed variance is pooled (weighted by agent count).

    Returns
    -------
    df : pd.DataFrame with at least Gender, Age_at_Release, rearrest_*_yrs
    """
    files = glob.glob(input_glob)
    if not files:
        raise FileNotFoundError(
            f"No study-cohort CSVs found matching {input_glob}. "
            "Run the main calibration with export_csv=True first."
        )

    print(f"  Loading {len(files)} study-cohort CSVs...")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_file"] = os.path.basename(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(out):,} total agent records")
    return out


def categorize_age(age):
    """Bin Age_at_Release into BJS Table 3 groups."""
    if age <= 24:      return "24 or younger"
    if age <= 29:      return "25-29"
    if age <= 39:      return "30-39"
    return "40 or older"


# =============================================================================
# CHART 6: DESISTANCE VALIDATION
# =============================================================================
def plot_desistance(df, outdir):
    """
    Validates the 'never arrested during 9-year follow-up' group.
    BJS reports 17.0% of releasees were never arrested (NCJ 250975).
    """

    # Each agent's "never arrested in 9 years" flag
    # is simply: NOT rearrest_9_yrs
    n_total = len(df)
    n_desist = (~df["rearrest_9_yrs"].astype(bool)).sum()
    pct_abm = n_desist / n_total

    pct_bjs = BJS_TARGETS["desistance_9yr"]
    gap = pct_abm - pct_bjs
    flag = _gap_flag(gap)
    color = _gap_color(gap)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(2)
    vals = [pct_abm, pct_bjs]
    colors = [C["abm"], C["bjs"]]
    labels = ["ABM calibrated", "BJS NCJ 250975"]

    bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=bar.get_facecolor())

    # Gap annotation between bars
    ax.annotate(
        f"Δ = {gap:+.1%}\n({flag})",
        xy=(0.5, max(vals) + 0.035), ha="center", fontsize=11,
        color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=color, alpha=0.95)
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    _pct(ax)
    ax.set_ylim(0, max(vals) + 0.10)
    _style(
        ax, "", "% never arrested during 9-year follow-up",
        "Chart 6 — Desistance (no-arrest) group validation\n"
        f"n = {n_total:,} study-eligible agents  |  "
        "BJS NCJ 250975 Figure 1: 17% of releasees were never arrested",
        fs=12
    )

    plt.tight_layout()
    path_png = os.path.join(outdir, "chart6_desistance.png")
    plt.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart 6 (desistance) -> {path_png}")

    return {
        "group":    "All releasees",
        "target":   pct_bjs,
        "abm":      pct_abm,
        "gap_pp":   gap,
        "flag":     flag,
        "n":        n_total,
    }


# =============================================================================
# CHART 7: SEX-STRATIFIED VALIDATION
# =============================================================================
def plot_sex_stratified(df, outdir):
    """
    Validates rearrest rates by sex against BJS Table 3.
    Two-panel chart: panel A trajectory, panel B 9-yr bar summary.
    """
    if "Gender" not in df.columns:
        print("    WARNING: Gender column not found, skipping sex validation")
        return []

    targets = BJS_TARGETS["sex"]
    groups = ["Male", "Female"]
    windows = [3, 6, 9]

    results = []
    for sex in groups:
        subset = df[df["Gender"] == sex]
        n_sex = len(subset)
        for w in windows:
            col = f"rearrest_{w}_yrs"
            abm_rate = subset[col].astype(bool).mean() if n_sex > 0 else 0.0
            bjs_rate = targets[sex][w]
            gap = abm_rate - bjs_rate
            results.append({
                "group":  sex,
                "window": w,
                "abm":    abm_rate,
                "target": bjs_rate,
                "gap_pp": gap,
                "flag":   _gap_flag(gap),
                "n":      n_sex,
            })

    # ── TWO-PANEL CHART ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Trajectory by sex
    for sex in groups:
        abm_vals = [r["abm"] for r in results if r["group"] == sex]
        bjs_vals = [targets[sex][w] for w in windows]
        clr = C[sex]
        ax1.plot(windows, abm_vals, color=clr, marker="o", markersize=9,
                 linewidth=2.5, label=f"ABM — {sex}", zorder=4)
        ax1.plot(windows, bjs_vals, color=clr, marker="D", markersize=9,
                 linewidth=2.0, linestyle=":", label=f"BJS — {sex}",
                 alpha=0.8, zorder=3)
        # Annotate each year with Δ
        for w, a, b in zip(windows, abm_vals, bjs_vals):
            g = a - b
            ax1.annotate(
                f"Δ{g:+.1%}",
                xy=(w, a), xytext=(0, -18 if sex == "Female" else 12),
                textcoords="offset points", ha="center", fontsize=8,
                color=_gap_color(g), fontweight="bold",
            )
    ax1.set_xticks(windows)
    ax1.set_xticklabels([f"{w}-Year" for w in windows], fontsize=10)
    ax1.set_ylim(0.5, 0.95)
    _pct(ax1)
    ax1.legend(fontsize=9, framealpha=0.95, loc="lower right")
    _style(ax1, "Follow-up Window", "Cumulative Rearrest Rate",
           "Panel A — Sex-Stratified Trajectory", fs=11)

    # Panel B: 9-year bar comparison
    x = np.arange(2)
    w_bar = 0.35
    abm_9yr = [r["abm"]    for r in results if r["window"] == 9]
    bjs_9yr = [r["target"] for r in results if r["window"] == 9]

    b1 = ax2.bar(x - w_bar/2, abm_9yr, w_bar, color=[C[g] for g in groups],
                 edgecolor="white", linewidth=1, label="ABM calibrated")
    b2 = ax2.bar(x + w_bar/2, bjs_9yr, w_bar, color="#888888", alpha=0.75,
                 edgecolor="white", linewidth=1, label="BJS target")

    for bar, val in zip(b1, abm_9yr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=bar.get_facecolor())
    for bar, val in zip(b2, bjs_9yr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=10, color="#555")

    # Gap annotations
    for i, (a, b) in enumerate(zip(abm_9yr, bjs_9yr)):
        g = a - b
        ax2.annotate(
            f"Δ{g:+.1%}\n({_gap_flag(g)})",
            xy=(x[i], max(a, b) + 0.04), ha="center", fontsize=9,
            color=_gap_color(g), fontweight="bold"
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=11)
    _pct(ax2)
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=9, framealpha=0.95, loc="upper right")
    _style(ax2, "", "9-Year Cumulative Rearrest Rate",
           "Panel B — 9-Year Rearrest by Sex", fs=11)

    fig.suptitle(
        "Chart 7 — Sex-Stratified Rearrest Validation\n"
        f"ABM vs BJS NCJ 250975 Table 3  |  "
        f"n = {len(df):,} agents  |  "
        "Source: Alper, Durose & Markman (2018)",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    path_png = os.path.join(outdir, "chart7_sex_stratified.png")
    plt.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart 7 (sex-stratified) -> {path_png}")
    return results


# =============================================================================
# CHART 8: AGE-AT-RELEASE STRATIFIED VALIDATION
# =============================================================================
def plot_age_stratified(df, outdir):
    """
    Validates rearrest rates by age at release against BJS Table 3.
    Four-panel chart: one per age group.
    """
    if "Age_at_Release" not in df.columns:
        print("    WARNING: Age_at_Release column not found, skipping age validation")
        return []

    # Apply age binning
    df = df.copy()
    df["age_group"] = df["Age_at_Release"].apply(categorize_age)

    targets = BJS_TARGETS["age_at_release"]
    groups = ["24 or younger", "25-29", "30-39", "40 or older"]
    windows = [3, 6, 9]

    results = []
    for age_grp in groups:
        subset = df[df["age_group"] == age_grp]
        n_age = len(subset)
        for w in windows:
            col = f"rearrest_{w}_yrs"
            abm_rate = subset[col].astype(bool).mean() if n_age > 0 else 0.0
            bjs_rate = targets[age_grp][w]
            gap = abm_rate - bjs_rate
            results.append({
                "group":  age_grp,
                "window": w,
                "abm":    abm_rate,
                "target": bjs_rate,
                "gap_pp": gap,
                "flag":   _gap_flag(gap),
                "n":      n_age,
            })

    # ── FOUR-PANEL CHART (one per age group) ───────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)

    for ax, age_grp in zip(axes, groups):
        abm_vals = [r["abm"]    for r in results if r["group"] == age_grp]
        bjs_vals = [r["target"] for r in results if r["group"] == age_grp]
        n_age = next(r["n"] for r in results if r["group"] == age_grp)
        clr = C[age_grp]

        # Plot trajectories
        ax.fill_between(windows,
                        [b - 0.02 for b in bjs_vals],
                        [b + 0.02 for b in bjs_vals],
                        color="#1A3D5C", alpha=0.09, zorder=1,
                        label="BJS target ±2pp band")
        ax.plot(windows, bjs_vals, color="#1A3D5C", linewidth=2.0,
                linestyle=":", marker="D", markersize=9, zorder=3,
                label="BJS NCJ 250975")
        ax.plot(windows, abm_vals, color=clr, linewidth=2.8,
                marker="o", markersize=10, zorder=4,
                label="ABM calibrated")

        # Annotate ABM points
        for w, a, b in zip(windows, abm_vals, bjs_vals):
            g = a - b
            ax.annotate(
                f"{a:.1%}\nΔ{g:+.1%}",
                xy=(w, a), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=9,
                fontweight="bold", color=_gap_color(g),
            )

        # Summary box
        max_gap = max(results, key=lambda r: abs(r["gap_pp"]) if r["group"] == age_grp else -1)
        relevant_gaps = [r["gap_pp"] for r in results if r["group"] == age_grp]
        mean_abs_gap = np.mean([abs(g) for g in relevant_gaps])
        ax.text(0.5, 0.04,
                f"Mean |Δ|: {mean_abs_gap:.1%}\nn = {n_age:,}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C["grid"], alpha=0.95))

        ax.set_xticks(windows)
        ax.set_xticklabels([f"{w}Y" for w in windows], fontsize=10)
        ax.set_xlim(2.4, 9.6)
        ax.set_ylim(0.50, 1.0)
        _pct(ax)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
        _style(ax, "Follow-up Window", "Cumulative Rearrest Rate",
               f"{age_grp}", fs=11)

    fig.suptitle(
        "Chart 8 — Age-at-Release Stratified Rearrest Validation\n"
        f"ABM vs BJS NCJ 250975 Table 3  |  "
        f"n = {len(df):,} agents  |  "
        "Source: Alper, Durose & Markman (2018)",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    path_png = os.path.join(outdir, "chart8_age_stratified.png")
    plt.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart 8 (age-stratified) -> {path_png}")
    return results


# =============================================================================
# SUMMARY TABLE
# =============================================================================
def save_summary_table(desist_row, sex_rows, age_rows, outdir):
    """Write all three validation tables to a combined CSV."""
    rows = []

    # Desistance (one row)
    rows.append({
        "dimension":  "Desistance",
        "group":      "All releasees",
        "window":     9,
        "abm":        desist_row["abm"],
        "target":     desist_row["target"],
        "gap_pp":     desist_row["gap_pp"],
        "flag":       desist_row["flag"],
        "n":          desist_row["n"],
    })

    # Sex (6 rows: Male/Female × 3 windows)
    for r in sex_rows:
        rows.append({
            "dimension":  "Sex",
            "group":      r["group"],
            "window":     r["window"],
            "abm":        r["abm"],
            "target":     r["target"],
            "gap_pp":     r["gap_pp"],
            "flag":       r["flag"],
            "n":          r["n"],
        })

    # Age at release (12 rows: 4 age groups × 3 windows)
    for r in age_rows:
        rows.append({
            "dimension":  "Age at release",
            "group":      r["group"],
            "window":     r["window"],
            "abm":        r["abm"],
            "target":     r["target"],
            "gap_pp":     r["gap_pp"],
            "flag":       r["flag"],
            "n":          r["n"],
        })

    df_out = pd.DataFrame(rows)
    path = os.path.join(outdir, "validation_extended_raw.csv")
    df_out.to_csv(path, index=False)
    print(f"\n  Summary table -> {path}")
    return df_out


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("="*68)
    print("  EXTENDED BJS VALIDATION — Desistance, Sex, Age at Release")
    print("="*68)
    print(f"  Input glob : {CONFIG['input_glob']}")
    print(f"  Output dir : {CONFIG['output_dir']}")
    print("="*68)

    df = load_study_cohorts(CONFIG["input_glob"])

    # Verify required columns
    required = ["Gender", "Age_at_Release", "rearrest_3_yrs",
                "rearrest_6_yrs", "rearrest_9_yrs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  WARNING: Missing columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        print("  Some validations may be skipped.\n")

    print("\n  Generating extended validation charts...")

    desist_row = plot_desistance(df, CONFIG["output_dir"])
    sex_rows   = plot_sex_stratified(df, CONFIG["output_dir"])
    age_rows   = plot_age_stratified(df, CONFIG["output_dir"])

    summary_df = save_summary_table(desist_row, sex_rows, age_rows,
                                     CONFIG["output_dir"])

    # ── Print summary to console ───────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*68}")
    print(f"\n  Desistance (no-arrest in 9 years):")
    print(f"    ABM: {desist_row['abm']:.1%}  "
          f"BJS: {desist_row['target']:.1%}  "
          f"Δ: {desist_row['gap_pp']:+.1%}  [{desist_row['flag']}]")

    print(f"\n  Sex-stratified (9-year cumulative):")
    for sex in ["Male", "Female"]:
        r = next(r for r in sex_rows if r["group"] == sex and r["window"] == 9)
        print(f"    {sex:10s}  ABM: {r['abm']:.1%}  "
              f"BJS: {r['target']:.1%}  "
              f"Δ: {r['gap_pp']:+.1%}  [{r['flag']}]")

    print(f"\n  Age-at-release (9-year cumulative):")
    for age_grp in ["24 or younger", "25-29", "30-39", "40 or older"]:
        r = next(r for r in age_rows if r["group"] == age_grp and r["window"] == 9)
        print(f"    {age_grp:15s}  ABM: {r['abm']:.1%}  "
              f"BJS: {r['target']:.1%}  "
              f"Δ: {r['gap_pp']:+.1%}  [{r['flag']}]")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()