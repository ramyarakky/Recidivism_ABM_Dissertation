"""
chart9_desistance_curve.py
==========================
Replicates BJS NCJ 250975 Figure 4:
"Percent of prisoners released in 30 states in 2005 who were
 not arrested since release, by year following release"

Input: validation_raw.csv with columns year, abm_rate, bjs_rate, abm_std
Output: chart9_desistance_curve.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


CONFIG = {
    "validation_csv": "validation_raw.csv",
    "output_dir":     "validation_output",
}

# BJS Figure 4 / Table 2 values: % NOT arrested at each year
BJS_DESISTANCE = {
    1: 1.000 - 0.439,
    2: 1.000 - 0.601,
    3: 1.000 - 0.684,
    4: 1.000 - 0.736,
    5: 1.000 - 0.770,
    6: 1.000 - 0.794,
    7: 1.000 - 0.812,
    8: 1.000 - 0.824,
    9: 1.000 - 0.834,
}


C = {
    "abm":    "#1f77b4",
    "bjs":    "#2CA02C",
    "grid":   "#DDDDDD",
    "good":   "#276419",
    "warn":   "#B8860B",
    "bad":    "#CC4400",
    "band":   "#E8F4E8",
}


def _gap_color(gap_pp):
    if abs(gap_pp) <= 0.02: return C["good"]
    if abs(gap_pp) <= 0.05: return C["warn"]
    return C["bad"]


def _gap_flag(gap_pp):
    if abs(gap_pp) <= 0.02: return "robust"
    if abs(gap_pp) <= 0.05: return "acceptable"
    return "off-target"


def load_cumulative_rates(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def compute_desistance_from_cumulative(df):
    df = df.copy()
    df["abm_desist"]     = 1.0 - df["abm_rate"]
    df["bjs_desist"]     = 1.0 - df["bjs_rate"]
    df["abm_desist_std"] = df.get("abm_std", 0.0)
    df["gap_pp"]         = df["abm_desist"] - df["bjs_desist"]
    return df


def plot_desistance_curve(df, outdir):
    """Single-panel desistance survival curve (BJS Figure 4 replication)."""
    fig, ax = plt.subplots(figsize=(11, 7))

    years = df["year"].values
    abm   = df["abm_desist"].values
    bjs   = df["bjs_desist"].values
    gap   = df["gap_pp"].values
    sd    = df.get("abm_desist_std", pd.Series(np.zeros(len(df)))).values

    # BJS ±2pp target band
    ax.fill_between(years, bjs - 0.02, bjs + 0.02,
                    color=C["band"], alpha=0.6, zorder=1,
                    label="BJS ±2pp target band")

    # BJS line — thick green like Figure 4
    ax.plot(years, bjs, color=C["bjs"], linewidth=3.0, marker="D",
            markersize=10, zorder=5, label="BJS NCJ 250975 (Fig. 4)")

    # ABM calibrated
    ax.plot(years, abm, color=C["abm"], linewidth=2.5, marker="o",
            markersize=9, zorder=4, label="ABM calibrated")

    # ±1 SD ribbon if available
    if sd.sum() > 0:
        ax.fill_between(years, np.clip(abm - sd, 0, 1), np.clip(abm + sd, 0, 1),
                        color=C["abm"], alpha=0.15, zorder=3,
                        label="ABM ±1 SD across seeds")

    # Year-by-year Δ labels
    for yr, a, g in zip(years, abm, gap):
        clr = _gap_color(g)
        ax.annotate(
            f"Δ{g:+.1%}",
            xy=(yr, a), xytext=(0, 14),
            textcoords="offset points", ha="center",
            fontsize=9, color=clr, fontweight="bold",
        )

    # Axis styling — matches BJS Figure 4 format
    ord_labels = {1: "1st", 2: "2nd", 3: "3rd"}
    ax.set_xticks(years)
    ax.set_xticklabels([ord_labels.get(y, f"{y}th") for y in years], fontsize=11)
    ax.set_xlabel("Year after release", fontsize=12, labelpad=10)
    ax.set_ylabel("Percent of released prisoners not arrested since release",
                  fontsize=12, labelpad=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.))
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0.5, 9.5)

    ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, framealpha=0.95, loc="upper right")

    # Title
    fig.suptitle(
        "Chart 9 — Desistance Survival Curve: ABM vs BJS NCJ 250975 (Figure 4)\n"
        "Percent of released prisoners NOT yet arrested at each year  |  "
        "Source: Alper, Durose & Markman (2018)",
        fontsize=13, fontweight="bold", y=0.99
    )

    plt.tight_layout()
    path_png = os.path.join(outdir, "chart9_desistance_curve.png")
    plt.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart 9 (desistance curve) -> {path_png}")

    # Print summary to console
    print(f"\n  Desistance curve summary:")
    print(f"  {'Year':<6}{'ABM':>8}{'BJS':>8}{'Δ pp':>10}{'Flag':>15}")
    print(f"  {'-'*6}{'-'*8}{'-'*8}{'-'*10}{'-'*15}")
    for yr, a, b, g in zip(years, abm, bjs, gap):
        flag = _gap_flag(g)
        print(f"  {yr:<6}{a:>7.1%}{b:>8.1%}{g*100:>+9.1f}pp{flag:>15}")
    mean_abs_gap_pp = np.abs(gap * 100).mean()
    print(f"  {'-'*6}{'-'*8}{'-'*8}{'-'*10}{'-'*15}")
    print(f"  Mean |Δ|: {mean_abs_gap_pp:.2f}pp")


def build_example_df():
    """Placeholder data — replace with load from actual validation CSV."""
    years     = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    abm_rates = [0.418, 0.588, 0.691, 0.742, 0.774, 0.793, 0.802, 0.807, 0.810]
    bjs_rates = [0.439, 0.601, 0.684, 0.736, 0.770, 0.794, 0.812, 0.824, 0.834]
    return pd.DataFrame({
        "year":      years,
        "abm_rate":  abm_rates,
        "bjs_rate":  bjs_rates,
        "abm_std":   [0.005] * 9,
    })


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("="*68)
    print("  CHART 9 — DESISTANCE SURVIVAL CURVE (BJS Figure 4 replication)")
    print("="*68)

    if os.path.exists(CONFIG["validation_csv"]):
        print(f"  Loading {CONFIG['validation_csv']}...")
        df = load_cumulative_rates(CONFIG["validation_csv"])
    else:
        print(f"  {CONFIG['validation_csv']} not found — using example data")
        df = build_example_df()

    df = compute_desistance_from_cumulative(df)
    plot_desistance_curve(df, CONFIG["output_dir"])
    print("="*68)


if __name__ == "__main__":
    main()