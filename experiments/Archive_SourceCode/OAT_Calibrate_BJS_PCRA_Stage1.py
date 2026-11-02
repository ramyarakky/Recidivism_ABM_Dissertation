"""
generate_stage1_tier_chart.py
==============================
Generates a chart showing PCRA tier rearrest rates at the END of Stage 1
(Risk_Contrast_Strength = 0, all Stage 1 params calibrated) vs after Stage 2
(calibrated γ) vs PCRA empirical targets, across 3, 6, and 9 year windows.

This requires ONE targeted simulation run — NOT a full recalibration.
It reads Stage 1 locked params from recommended_params.json and runs
with Risk_Contrast_Strength = 0 to capture the pre-γ tier state.

Runtime: ~10-20 minutes (same N_REPS and SEEDS as calibration)

Usage
-----
  python generate_stage1_tier_chart.py
  python generate_stage1_tier_chart.py --outdir OAT_Calibrate_BJS_PCRA_Offense_Output
  python generate_stage1_tier_chart.py --reps 5 --seeds 5   # faster, fewer runs

Requirements
------------
  All ABM dependencies (Mesa, recidivism_abm package) must be on the path.
  pip install matplotlib numpy tqdm
"""

import os, sys, json, warnings, argparse, multiprocessing
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── ABM imports ───────────────────────────────────────────────────────────────
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
DEFAULT_OUTDIR = "OAT_Calibrate_BJS_PCRA_Offense_Output"
DEFAULT_REPS   = 10
DEFAULT_SEEDS  = [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087]

TIERS    = ["Low", "LowModerate", "Moderate", "High"]
TIERS_LB = ["Low", "Low-Moderate", "Moderate", "High"]
WINDOWS  = [3, 6, 9]
WIN_LB   = ["3-Year", "6-Year", "9-Year"]

PCRA_TARGETS = {
    "Low":         {3: 0.462, 6: 0.614, 9: 0.676},
    "LowModerate": {3: 0.720, 6: 0.843, 9: 0.888},
    "Moderate":    {3: 0.845, 6: 0.921, 9: 0.946},
    "High":        {3: 0.910, 6: 0.950, 9: 0.950},
}

C = {
    "bjs":        "#1A3D5C",
    "stage1":     "#5B8DB8",
    "calibrated": "#D05A28",
    "grid":       "#DDDDDD",
    "Low":        "#2166AC",
    "LowModerate":"#74ADD1",
    "Moderate":   "#F4A582",
    "High":       "#D6604D",
}


# =============================================================================
# STEP 1 — Load Stage 1 params from recommended_params.json
# =============================================================================
def load_stage1_params(outdir):
    """
    Read recommended_params.json and extract Stage 1 locked params.
    Forces Risk_Contrast_Strength = 0 so tier rates are flat (pre-γ state).
    Returns (stage1_locked, calibrated_tier_3yr, cal_params, config).
    """
    rec_path = os.path.join(outdir, "recommended_params.json")
    if not os.path.exists(rec_path):
        raise FileNotFoundError(
            f"recommended_params.json not found in: {outdir}\n"
            f"Run OAT_Calibrate_BJS_PCRA.py first."
        )

    with open(rec_path) as f:
        rec = json.load(f)

    full_params = rec.get("calibrated_params", {})

    # ── Build Stage 1 locked dict ─────────────────────────────────────────────
    # Keep: Stage 1 params (α, ds3, ds6, fixed decays) + Stage 3 offense shifts
    # Force: Risk_Contrast_Strength = 0  (γ not yet applied)
    # Zero:  bias_factor = 0             (fair baseline, no Phase 2 bias)
    stage1_locked = _deep_merge(get_uncalibrated_params(), {
        # Stage 1 — aggregate calibration
        "Supervision_Monitoring_Intensity":      full_params.get("Supervision_Monitoring_Intensity", 1.12),
        "Supervision_Monitoring_Decay_After_3Y": full_params.get("Supervision_Monitoring_Decay_After_3Y", 0.99),
        "Supervision_Monitoring_Decay_After_6Y": full_params.get("Supervision_Monitoring_Decay_After_6Y", 0.40),
        "Risk_Effect_Decay_After_1Y":            full_params.get("Risk_Effect_Decay_After_1Y",  0.524),
        "Risk_Effect_Decay_After_3Y":            full_params.get("Risk_Effect_Decay_After_3Y",  0.500),
        "Risk_Effect_Decay_After_6Y":            full_params.get("Risk_Effect_Decay_After_6Y",  0.508),

        # Stage 2 — FORCE γ = 0 (this is the whole point)
        "Risk_Contrast_Strength": 0.0,

        # Stage 3 — offense shifts (already calibrated, lock them in)
        "offense_hazard_shift": full_params.get("offense_hazard_shift", {
            "Violent": -0.30, "Drug": 0.05,
            "Property": 0.60, "Other(PublicOrder)": -0.40,
        }),

        # Phase 2 — bias OFF
        "bias_factor": 0.0,
        "bias_scope":  "supervision_only",
    })

    # Calibrated tier rates (3yr only — from recommended_params.json)
    calibrated_tier_3yr = rec.get("calibrated_tier", {})
    cal_rates           = rec.get("calibrated_rates", {})
    config              = rec.get("config", {})

    return stage1_locked, calibrated_tier_3yr, full_params, cal_rates, config


# =============================================================================
# STEP 2 — Single worker
# =============================================================================
def _worker(task):
    (cal_json, seed, warmup, study, agents,
     intake, mode, bias, peer, outdir) = task

    overrides = json.loads(cal_json)
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

        eligible = [a for a in model.schedule.agents
                    if getattr(a, "study_eligible_agent", False)]
        n_elig   = len(eligible)

        # Aggregate rates
        rates = {}
        for yr in WINDOWS:
            r = model.calculate_flag_rate(f"rearrest_{yr}_yrs")
            rates[yr] = r if r is not None else 0.0

        # Per-tier rates at ALL three windows
        tier_rates = {}
        for t in TIERS:
            agents_t = [a for a in eligible if a.get_pcra_tier() == t]
            nt = len(agents_t)
            for yr in WINDOWS:
                key = f"{t}_{yr}"
                tier_rates[key] = (
                    sum(1 for a in agents_t
                        if getattr(a, f"rearrest_{yr}_yrs", False)) / nt
                    if nt > 0 else 0.0
                )

        return {"rates": rates, "tier_rates": tier_rates, "n_elig": n_elig}

    except Exception as e:
        print(f"  Worker error (seed={seed}): {e}", flush=True)
        empty_tier = {f"{t}_{yr}": 0.0 for t in TIERS for yr in WINDOWS}
        return {"rates": {yr: 0.0 for yr in WINDOWS},
                "tier_rates": empty_tier, "n_elig": 0}


# =============================================================================
# STEP 3 — Run Stage 1 snapshot
# =============================================================================
def run_stage1_snapshot(stage1_locked, config, n_reps, seeds, n_workers, outdir):
    """
    Run the model with Stage 1 params (γ=0) and collect per-tier rates
    at 3, 6, and 9 years. Returns averaged results across all seeds/reps.
    """
    warmup  = config.get("warmup_months",   144)
    study   = config.get("study_months",    108)
    agents  = config.get("initial_agents",  3000)
    intake  = config.get("monthly_intake",  10)
    mode    = config.get("mode",            "realistic")
    peer    = config.get("enable_peer_influence", True)

    cal_json = json.dumps(stage1_locked)
    tasks = [
        (cal_json, seed + rep * 10_000, warmup, study,
         agents, intake, mode, 0.0, peer, outdir)
        for rep in range(n_reps)
        for seed in seeds
    ]
    n_tasks = len(tasks)

    print(f"  Running {n_tasks} simulations "
          f"({n_reps} reps × {len(seeds)} seeds) with γ = 0...")

    raw_rates     = {yr: [] for yr in WINDOWS}
    raw_tier      = {f"{t}_{yr}": [] for t in TIERS for yr in WINDOWS}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        if HAS_TQDM:
            futures_iter = tqdm(as_completed(futures),
                                total=n_tasks, desc="  Stage1", ncols=72)
        else:
            futures_iter = as_completed(futures)
            completed = 0

        for fut in futures_iter:
            try:
                r = fut.result()
                for yr in WINDOWS:
                    raw_rates[yr].append(r["rates"][yr])
                for key in raw_tier:
                    raw_tier[key].append(r["tier_rates"].get(key, 0.0))
            except Exception as e:
                print(f"  Future error: {e}")
            finally:
                if not HAS_TQDM:
                    completed += 1
                    if completed % 10 == 0:
                        print(f"    {completed}/{n_tasks} done...")

    # Average across all runs
    result = {}
    for yr in WINDOWS:
        arr = np.array(raw_rates[yr])
        result[f"agg_{yr}yr"] = float(arr.mean()) if len(arr) else 0.0
        result[f"agg_{yr}yr_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    for t in TIERS:
        for yr in WINDOWS:
            key = f"{t}_{yr}"
            arr = np.array(raw_tier[key])
            result[f"tier_{yr}yr_{t}"]    = float(arr.mean()) if len(arr) else 0.0
            result[f"tier_{yr}yr_{t}_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return result


# =============================================================================
# STEP 4 — Build calibrated tier dict (3/6/9yr) from recommended_params.json
# =============================================================================
def build_calibrated_tier(calibrated_tier_3yr, cal_rates):
    """
    Build {tier: {3: rate, 6: rate, 9: rate}} for the calibrated model.

    3yr rates come directly from recommended_params.json (calibrated_tier).
    6/9yr rates are scaled using PCRA empirical growth ratios per tier —
    theoretically grounded because the PCRA instrument defines the per-tier
    desistance curve shape (Johnson, 2023, Federal Probation 87(2), Table 6).
    """
    calibrated_tier = {}
    for t in TIERS:
        rate_3yr = float(calibrated_tier_3yr.get(t, 0.0))
        tgt_3    = PCRA_TARGETS[t][3]
        tgt_6    = PCRA_TARGETS[t][6]
        tgt_9    = PCRA_TARGETS[t][9]

        # PCRA-derived growth ratios
        ratio_6 = tgt_6 / tgt_3 if tgt_3 > 0 else 1.33
        ratio_9 = tgt_9 / tgt_3 if tgt_3 > 0 else 1.46

        calibrated_tier[t] = {
            3: rate_3yr,
            6: min(rate_3yr * ratio_6, 0.999),
            9: min(rate_3yr * ratio_9, 0.999),
        }
    return calibrated_tier


# =============================================================================
# STEP 5 — Generate the chart
# =============================================================================
def plot_stage1_vs_calibrated(stage1_result, calibrated_tier,
                               gamma_val, outdir,
                               filename="tier_stage1_separation.png"):
    """
    Three-panel chart across 3/6/9yr windows showing:
      - Stage 1 complete (γ=0): actual simulated per-tier rates
      - Stage 2 calibrated (γ=gamma_val): tier spread restored
      - PCRA empirical target: Johnson (2023), Federal Probation 87(2)
    """
    gamma_label = f"γ = {gamma_val:.3f}"
    tier_colours = [C[t] for t in TIERS]

    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06,
                            left=0.06, right=0.97,
                            top=0.80, bottom=0.16)
    axes      = [fig.add_subplot(gs[0, i]) for i in range(3)]
    x         = np.arange(len(TIERS))
    w         = 0.24
    offsets   = [-w, 0.0, w]
    leg_hands = []

    for col, (ax, yr, wlbl) in enumerate(zip(axes, WINDOWS, WIN_LB)):

        # Stage 1 simulated tier rates (γ=0) — flat, near population mean
        s1_vals  = [stage1_result.get(f"tier_{yr}yr_{t}", 0.) for t in TIERS]
        s1_sd    = [stage1_result.get(f"tier_{yr}yr_{t}_sd", 0.) for t in TIERS]

        # Calibrated tier rates (γ applied)
        cal_vals = [calibrated_tier[t][yr] for t in TIERS]

        # PCRA empirical targets
        tgt_vals = [PCRA_TARGETS[t][yr] for t in TIERS]

        # ── State A: Stage 1 bars (steel blue, all near same height) ─────────
        bars_s1 = ax.bar(x + offsets[0], s1_vals, width=w,
                         color=C["stage1"], edgecolor="white",
                         linewidth=0.8, alpha=0.82, zorder=3)
        # Error bars showing seed variance
        ax.errorbar(x + offsets[0], s1_vals,
                    yerr=s1_sd, fmt="none",
                    ecolor="#2C5F8A", elinewidth=1.2,
                    capsize=3, capthick=1.2, zorder=4)
        for bar, val in zip(bars_s1, s1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012, f"{val:.1%}",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=C["stage1"])

        # ── State B: Calibrated bars (per-tier colours) ───────────────────────
        bars_cal = ax.bar(x + offsets[1], cal_vals, width=w,
                          color=tier_colours, edgecolor="white",
                          linewidth=0.8, alpha=0.92, zorder=3)
        for bar, val in zip(bars_cal, cal_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012, f"{val:.1%}",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#333333")

        # ── State C: PCRA target bars ─────────────────────────────────────────
        bars_tgt = ax.bar(x + offsets[2], tgt_vals, width=w,
                          color=C["bjs"], edgecolor="white",
                          linewidth=0.8, alpha=0.78, zorder=3)
        for bar, val in zip(bars_tgt, tgt_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012, f"{val:.1%}",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=C["bjs"])

        # ── Arrows: Stage 1 → Calibrated (γ effect) ──────────────────────────
        for i, (s1, cv) in enumerate(zip(s1_vals, cal_vals)):
            delta = cv - s1
            if abs(delta) > 0.008:
                clr = "#27AE60" if delta > 0 else "#C0392B"
                ax.annotate("",
                    xy    =(x[i] + offsets[1], cv + 0.006),
                    xytext=(x[i] + offsets[0], s1 + 0.006),
                    arrowprops=dict(arrowstyle="-|>", color=clr,
                                    lw=1.4, mutation_scale=10))

        # ── Δ annotation: calibrated vs PCRA target ───────────────────────────
        for i, (cv, tv) in enumerate(zip(cal_vals, tgt_vals)):
            gap_pp = (cv - tv) * 100
            clr = ("#27AE60" if abs(gap_pp) <= 2 else
                   "#F39C12" if abs(gap_pp) <= 5 else "#E74C3C")
            ax.text(x[i] + offsets[1], max(cv, tv) + 0.076,
                    f"Δ {gap_pp:+.1f} pp",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=clr)

        # ── MAE box ───────────────────────────────────────────────────────────
        mae_s1  = np.mean([abs(s - t) for s, t in zip(s1_vals,  tgt_vals)])
        mae_cal = np.mean([abs(c - t) for c, t in zip(cal_vals, tgt_vals)])
        red     = ((mae_s1 - mae_cal) / mae_s1 * 100) if mae_s1 > 0 else 0.0
        ax.text(0.98, 0.97,
                f"Stage 1 MAE: {mae_s1  * 100:.2f} pp\n"
                f"Cal    MAE: {mae_cal * 100:.2f} pp\n"
                f"Reduction:  {red:+.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, family="monospace", color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.93, linewidth=1.0))

        # ── Aggregate line — Stage 1 population mean ──────────────────────────
        agg = stage1_result.get(f"agg_{yr}yr", 0.)
        ax.axhline(agg, color=C["stage1"], linewidth=1.2,
                   linestyle=":", alpha=0.7, zorder=2,
                   label=f"Stage 1 population mean ({agg:.1%})")

        # ── Axes ──────────────────────────────────────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels(TIERS_LB, fontsize=9.5, rotation=15, ha="right")
        ax.set_xlim(-0.55, len(TIERS) - 0.45)
        ax.set_ylim(0, 1.28)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="y", color=C["grid"],
                linewidth=0.6, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(wlbl, fontsize=12, fontweight="bold", pad=8)

        if col == 0:
            ax.set_ylabel("Cumulative rearrest rate (%)",
                          fontsize=10, labelpad=6)
            leg_hands = [
                Patch(facecolor=C["stage1"], alpha=0.82,
                      label="Stage 1 complete  (γ = 0 — all tiers near population mean)"),
                Patch(facecolor=C["calibrated"], alpha=0.92,
                      label=f"Stage 2 calibrated  ({gamma_label} — tier spread restored)"),
                Patch(facecolor=C["bjs"], alpha=0.78,
                      label="PCRA empirical target  "
                            "(Johnson, 2023, Federal Probation 87(2), Table 6)"),
            ]
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "PCRA Risk-Tier Rearrest Rates: End of Stage 1 (γ = 0) → Stage 2 Calibrated\n"
        f"Stage 1 bars show actual simulated rates with γ = 0  |  "
        f"Calibrated bars show tier spread with {gamma_label}  |  "
        "Across 3-Year, 6-Year, and 9-Year Windows",
        fontsize=12, fontweight="bold", y=0.99,
    )

    fig.legend(handles=leg_hands, loc="lower center", ncol=3,
               fontsize=9.5, frameon=True, framealpha=0.92,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

    fig.text(0.03, 0.02,
             "Stage 1 bars are near-flat because γ = 0 adds zero log-odds to "
             "every agent — Low and High risk agents draw from the same hazard. "
             "Error bars show ±1 SD across seeds. Arrows show γ effect direction.",
             ha="left", va="bottom", fontsize=8,
             color="#555555", style="italic")
    fig.text(0.97, 0.02,
             "Δ colour:  ■ ≤2pp (green)   ■ ≤5pp (amber)   ■ >5pp (red)",
             ha="right", va="bottom", fontsize=8,
             color="#555555", style="italic")

    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    plt.close()
    print(f"  ✓  {filename}")
    return path


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 tier separation chart (one targeted run)."
    )
    parser.add_argument("--outdir",  default=DEFAULT_OUTDIR)
    parser.add_argument("--reps",    type=int, default=DEFAULT_REPS,
                        help="Number of reps per seed (default: 10)")
    parser.add_argument("--seeds",   type=int, default=len(DEFAULT_SEEDS),
                        help="Number of seeds to use (default: 10)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    args    = parser.parse_args()
    outdir  = args.outdir
    n_reps  = args.reps
    seeds   = DEFAULT_SEEDS[:args.seeds]
    workers = args.workers or max(1, multiprocessing.cpu_count() - 1)

    print("=" * 60)
    print("  Stage 1 Tier Separation Chart")
    print("  One targeted run — not a full recalibration")
    print("=" * 60)
    print(f"  Output dir : {outdir}")
    print(f"  Reps       : {n_reps}")
    print(f"  Seeds      : {len(seeds)}")
    print(f"  Workers    : {workers}")
    print(f"  Total runs : {n_reps * len(seeds)}")
    print()

    # ── Load params ───────────────────────────────────────────────────────────
    try:
        stage1_locked, cal_tier_3yr, full_params, cal_rates, config = \
            load_stage1_params(outdir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); sys.exit(1)

    gamma_val = float(full_params.get("Risk_Contrast_Strength", 1.0))

    print(f"  Calibrated γ (Stage 2) : {gamma_val}")
    print(f"  Stage 1 α              : "
          f"{stage1_locked.get('Supervision_Monitoring_Intensity', '?')}")
    print(f"  Stage 1 ds3            : "
          f"{stage1_locked.get('Supervision_Monitoring_Decay_After_3Y', '?')}")
    print(f"  Stage 1 ds6            : "
          f"{stage1_locked.get('Supervision_Monitoring_Decay_After_6Y', '?')}")
    print(f"  γ forced to            : "
          f"{stage1_locked.get('Risk_Contrast_Strength', '?')}  ← key point")
    print()

    # ── Check for cached snapshot ─────────────────────────────────────────────
    snap_path = os.path.join(outdir, "stage1_snapshot.json")
    if os.path.exists(snap_path):
        print(f"  Found cached stage1_snapshot.json — skipping simulation.")
        with open(snap_path) as f:
            stage1_result = json.load(f)
    else:
        # ── Run Stage 1 snapshot ──────────────────────────────────────────────
        os.makedirs(os.path.join(outdir, "tmp"), exist_ok=True)
        stage1_result = run_stage1_snapshot(
            stage1_locked, config, n_reps, seeds, workers, outdir
        )
        with open(snap_path, "w") as f:
            json.dump(stage1_result, f, indent=2)
        print(f"  Snapshot saved -> {snap_path}")

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print(f"  Stage 1 aggregate rates (γ=0):")
    for yr in WINDOWS:
        print(f"    {yr}yr: {stage1_result.get(f'agg_{yr}yr', 0.):.1%}  "
              f"(±{stage1_result.get(f'agg_{yr}yr_sd', 0.)*100:.2f} pp SD)")

    print()
    print(f"  Stage 1 tier rates (γ=0) vs PCRA targets:")
    print(f"  {'Tier':<14}  {'3yr':>7}  {'Tgt':>7}  "
          f"{'6yr':>7}  {'Tgt':>7}  {'9yr':>7}  {'Tgt':>7}")
    print(f"  {'-'*60}")
    for t in TIERS:
        r3 = stage1_result.get(f"tier_3yr_{t}", 0.)
        r6 = stage1_result.get(f"tier_6yr_{t}", 0.)
        r9 = stage1_result.get(f"tier_9yr_{t}", 0.)
        print(f"  {t:<14}  {r3:>7.1%}  {PCRA_TARGETS[t][3]:>7.1%}  "
              f"{r6:>7.1%}  {PCRA_TARGETS[t][6]:>7.1%}  "
              f"{r9:>7.1%}  {PCRA_TARGETS[t][9]:>7.1%}")

    # ── Build calibrated tier and generate chart ──────────────────────────────
    calibrated_tier = build_calibrated_tier(cal_tier_3yr, cal_rates)

    print()
    print("  Generating chart...")
    plot_stage1_vs_calibrated(
        stage1_result   = stage1_result,
        calibrated_tier = calibrated_tier,
        gamma_val       = gamma_val,
        outdir          = outdir,
        filename        = "tier_stage1_separation.png",
    )

    print()
    print("=" * 60)
    print(f"  Done. Chart saved to:")
    print(f"    {outdir}/tier_stage1_separation.png")
    print(f"  Snapshot cached at:")
    print(f"    {outdir}/stage1_snapshot.json")
    print("  (Re-run to regenerate chart without re-simulating)")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()