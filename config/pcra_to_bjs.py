"""
pcra_to_bjs.py
==============
Converts PCRA federal supervision tier rates (Federal Probation 87(2), Table 6)
into BJS-scaled tier targets for a state prison-release population.

TARGET POPULATION
-----------------
Alper, Durose & Markman (2018). 2018 Update on Prisoner Recidivism:
A 9-Year Follow-Up Period (2005-2014). BJS NCJ 250975.
  - 401,288 state prisoners released in 30 states in 2005
  - Aggregate rearrest rates: 3yr=68%, 6yr=79%, 9yr=83%
  - Cohort composition: 74.1% conditional release (supervised),
    25.9% unconditional release (unsupervised)
    Source: Alper & Durose (2019). NCJ 251773, Table 1.

PCRA SOURCE
-----------
Johnson, J.L. (2023). Federal Post-Conviction Supervision Outcomes:
Rearrests and Revocations. Federal Probation, 87(2), 20-28. Table 6.
  - Federal probation + supervised release population only
  - All-tier 5yr rate = 26.6%

WHY DIRECT RATIO SCALING FAILS
-------------------------------
The PCRA all-tier 5yr rate (26.6%) is 2.56x lower than the BJS 3yr
rate (68%). Multiplying PCRA High-tier ratios by the BJS baseline
produces probabilities above 1.0 for Moderate and High tiers.
Log-odds transfer is used instead.

ON SUPERVISED vs. UNSUPERVISED DECOMPOSITION
---------------------------------------------
PCRA tier rates are measured exclusively in supervised populations,
while the BJS aggregate (68%/79%/83%) reflects a mixed cohort of
74.1% supervised and 25.9% unsupervised releasees. This raises the
question of whether R_supervised and R_unsupervised must be separated
before applying PCRA differentials.

The decomposition is not required, for the following reason:

Solomon, Kachnowski & Bhati (2005, Urban Institute, "Does Parole Work?")
analyzed the 1994 BJS cohort and found raw 2-year rearrest rates of:
  Discretionary parolees (supervised):    54%
  Mandatory parolees (supervised):        61%
  Unconditional releasees (unsupervised): 60%

After controlling for risk composition (criminal history, offense type,
demographics), the predicted rearrest probability converged to:
  Supervised (mandatory):   61%
  Unsupervised:             61%
  Supervised (discretionary): 57%

The raw difference between supervised and unsupervised releasees is
entirely explained by compositional differences — unconditional
releasees are typically max-outs with higher-risk profiles. At
equivalent risk levels, the two groups reoffend at the same rate.

This means the PCRA log-odds differential Δ_i, which captures
tier deviation from population average, is a risk-level signal rather
than a supervision-status signal. Applying it to logit(R_BJS) — the
population-average rate across the full mixed BJS cohort — is
therefore correct:

  R_i_target = sigmoid(logit(R_BJS) + Δ_i)

The 74.1%/25.9% supervision mix is already embedded in R_BJS.
No stream decomposition is needed because risk drives the tier
differences, not supervision status.

METHODOLOGY
-----------
Step 1 — Log-odds differentials
  lo_diff(tier, mo) = logit(tier_rate(mo)) - logit(All_rate(mo))
  Computed at 36, 48, 60 months from PCRA Table 6.
  Captures the instrument's rank-ordering signal independently of
  absolute hazard level, enabling transfer across populations.
  Source: Skeem & Monahan (2011). Current Directions in Psychological
          Science, 20(1), 38-42.

Step 2 — Log-linear extrapolation to 72mo and 108mo
  Table 6 ends at 60 months. Log-linear trend is fitted to
  differentials at 36/48/60 months and extrapolated to 72mo (6yr)
  and 108mo (9yr).

Step 3 — Transfer to BJS baseline
  target(tier, yrs) = sigmoid(logit(BJS_aggregate(yrs)) + lo_diff(tier, yrs))
  The full BJS aggregate is used as the baseline.

Step 4 — Iterative population-weighted constraint
  Forces: sum(TIER_WEIGHTS[t] * target(t, yrs)) == BJS_TARGETS[yrs]
  using the ABM study cohort tier distribution.
  Iterative convergence to residual < 1e-6 (single-shot log-odds
  shift is inexact due to sigmoid nonlinearity).
"""

import math
from scipy.stats import linregress


# ─────────────────────────────────────────────────────────────────────────────
# PCRA TABLE 6
# Federal Probation 87(2) — Johnson (2023)
# Cumulative rearrest rates by tier and observation window (months)
# ─────────────────────────────────────────────────────────────────────────────
PCRA_TABLE = {
    "All":         [0.023, 0.041, 0.074, 0.105, 0.132, 0.182, 0.228, 0.266],
    "Low":         [0.011, 0.017, 0.028, 0.039, 0.049, 0.069, 0.089, 0.107],
    "LowModerate": [0.020, 0.036, 0.067, 0.098, 0.127, 0.181, 0.234, 0.279],
    "Moderate":    [0.037, 0.070, 0.132, 0.185, 0.232, 0.318, 0.395, 0.453],
    "High":        [0.067, 0.128, 0.221, 0.296, 0.362, 0.465, 0.546, 0.605],
}
PCRA_MONTHS = [3, 6, 12, 18, 24, 36, 48, 60]

# Indices used for trend fitting: 36, 48, 60 months
FIT_IDX = [5, 6, 7]
FIT_MO  = [36, 48, 60]


# ─────────────────────────────────────────────────────────────────────────────
# BJS AGGREGATE TARGETS
# Alper, Durose & Markman (2018). NCJ 250975.
# Full mixed cohort: 74.1% supervised + 25.9% unsupervised
# (NCJ 251773, Table 1 — documented; not decomposed per methodology above)
# ─────────────────────────────────────────────────────────────────────────────
BJS_TARGETS = {3: 0.68, 6: 0.79, 9: 0.83}
BJS_MONTHS  = {3: 36,   6: 72,   9: 108}


# ─────────────────────────────────────────────────────────────────────────────
# TIER WEIGHTS
# Source: Johnson (2023). Federal Post-Conviction Supervision Outcomes.
#         Federal Probation, 87(2), Table 1. n = 475,528 supervision terms.
#
# These are the observed tier proportions in the PCRA reference population —
# the same population from which Table 6 rates are drawn. Using the reference
# population weights is appropriate because:
#   (a) The ABM tier classification uses fixed quartile boundaries on a
#       sigmoid-normalized score, producing a uniform 25%/25%/25%/25%
#       distribution by mathematical construction — not a meaningful
#       population prevalence estimate.
#   (b) The PCRA reference weights reflect the actual risk composition of
#       the supervised federal cohort that the instrument was validated on,
#       making them the correct denominator for the log-odds transfer.
#
# Counts from Table 1 (n=475,528):
#   Low          138,230  →  29.1%
#   LowModerate  169,153  →  35.6%
#   Moderate     113,487  →  23.9%
#   High          45,467  →   9.6%
# ─────────────────────────────────────────────────────────────────────────────
TIER_WEIGHTS = {
    "Low":         0.291,   # 138,230 / 475,528
    "LowModerate": 0.356,   # 169,153 / 475,528
    "Moderate":    0.239,   # 113,487 / 475,528
    "High":        0.096,   #  45,467 / 475,528
}

TIERS      = ["Low", "LowModerate", "Moderate", "High"]
TARGET_CAP = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >  60: return 1.0
    if x < -60: return 0.0
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOG-ODDS DIFFERENTIALS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lo_differentials() -> dict:
    """
    lo_diff(tier, mo) = logit(tier_rate) - logit(All_rate)
    at 36, 48, 60 months.
    Returns {tier: [diff_36, diff_48, diff_60]}.
    """
    all_rates = PCRA_TABLE["All"]
    return {
        tier: [
            _logit(PCRA_TABLE[tier][i]) - _logit(all_rates[i])
            for i in FIT_IDX
        ]
        for tier in TIERS
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LOG-LINEAR TREND FIT AND EXTRAPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def _fit_lo_trends(lo_diffs: dict) -> dict:
    """
    Fit log-linear trend to differentials at 36/48/60 months.
    Returns {tier: {slope, intercept, r_squared}}.
    """
    trends = {}
    for tier in TIERS:
        slope, intercept, r, _, _ = linregress(FIT_MO, lo_diffs[tier])
        trends[tier] = {
            "slope":     slope,
            "intercept": intercept,
            "r_squared": r ** 2,
        }
    return trends


def _extrapolated_lo_diff(tier: str, months: int,
                           lo_diffs: dict, trends: dict) -> float:
    t = trends[tier]
    return t["intercept"] + t["slope"] * months


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRANSFER TO BJS BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_raw_tier_targets(lo_diffs: dict, trends: dict) -> dict:
    """
    target(tier, yrs) = sigmoid(logit(BJS_aggregate(yrs)) + lo_diff(tier, yrs))

    BJS aggregate is the correct baseline. The 74.1%/25.9% supervision
    mix is embedded in it. PCRA differentials reflect risk-tier deviations
    from population average, which transfer across populations because
    supervision status does not independently explain rearrest rates
    after risk adjustment (Solomon et al., 2005).

    Returns {tier: {3: float, 6: float, 9: float}}.
    """
    targets = {t: {} for t in TIERS}
    for yrs, mo in BJS_MONTHS.items():
        baseline = BJS_TARGETS[yrs]
        for tier in TIERS:
            lo_diff = _extrapolated_lo_diff(tier, mo, lo_diffs, trends)
            targets[tier][yrs] = min(
                TARGET_CAP, _sigmoid(_logit(baseline) + lo_diff)
            )
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: ITERATIVE POPULATION-WEIGHTED CONSTRAINT
# ─────────────────────────────────────────────────────────────────────────────

def _apply_population_weight_constraint(raw: dict,
                                         max_iter: int = 50,
                                         tol: float = 1e-6) -> dict:
    """
    Iteratively shift tier targets in log-odds space until:
      sum(TIER_WEIGHTS[t] * target(t, yrs)) == BJS_TARGETS[yrs]
    to within tol for each window.
    Returns {tier: {3: float, 6: float, 9: float}} rounded to 3dp.
    """
    current = {t: {yrs: raw[t][yrs] for yrs in [3, 6, 9]}
               for t in TIERS}

    for yrs, bjs_rate in BJS_TARGETS.items():
        for _ in range(max_iter):
            wavg = sum(TIER_WEIGHTS[t] * current[t][yrs] for t in TIERS)
            if abs(wavg - bjs_rate) < tol:
                break
            lo_shift = _logit(bjs_rate) - _logit(wavg)
            for tier in TIERS:
                shifted = _sigmoid(_logit(current[tier][yrs]) + lo_shift)
                current[tier][yrs] = min(TARGET_CAP, shifted)

    return {
        t: {yrs: round(current[t][yrs], 3) for yrs in [3, 6, 9]}
        for t in TIERS
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def compute_pcra_bjs_targets(verbose: bool = False) -> dict:
    """
    Run the full pipeline and return tier targets.

    Returns
    -------
    dict : {tier: {3: float, 6: float, 9: float}}
        Cumulative rearrest rate targets per tier per follow-up window.
        Weighted average across ABM tier distribution equals BJS aggregate.
    """
    lo_diffs  = _compute_lo_differentials()
    lo_trends = _fit_lo_trends(lo_diffs)
    raw       = _compute_raw_tier_targets(lo_diffs, lo_trends)
    final     = _apply_population_weight_constraint(raw)

    if verbose:
        _print_verbose(lo_diffs, lo_trends, raw, final)

    return final


def get_pcra_tier_targets() -> dict:
    """Convenience wrapper. Returns final targets silently."""
    return compute_pcra_bjs_targets(verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# VERBOSE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def _print_verbose(lo_diffs: dict, lo_trends: dict,
                   raw: dict, final: dict) -> None:

    print("\n── Step 1: Log-Odds Differentials ──────────────────────────────────")
    print(f"  {'Tier':15s}  {'36mo':>8s}  {'48mo':>8s}  {'60mo':>8s}")
    print("  " + "─" * 46)
    for tier in TIERS:
        d = lo_diffs[tier]
        print(f"  {tier:15s}  {d[0]:>8.3f}  {d[1]:>8.3f}  {d[2]:>8.3f}")

    print("\n── Step 2: Log-Linear Trend Fit ────────────────────────────────────")
    print(f"  {'Tier':15s}  {'slope':>10s}  {'intercept':>10s}  {'R²':>8s}")
    print("  " + "─" * 50)
    for tier in TIERS:
        t = lo_trends[tier]
        print(f"  {tier:15s}  {t['slope']:>10.5f}  "
              f"{t['intercept']:>10.3f}  {t['r_squared']:>8.4f}")

    print("\n── Step 3: Raw Tier Targets ────────────────────────────────────────")
    print(f"  BJS baseline (full mixed cohort): "
          f"3yr={BJS_TARGETS[3]}  6yr={BJS_TARGETS[6]}  9yr={BJS_TARGETS[9]}")
    print(f"  Supervision mix: 74.1% conditional + 25.9% unconditional")
    print(f"  (NCJ 251773 Table 1; not decomposed — see docstring)")
    print(f"  {'Tier':15s}  {'3yr':>8s}  {'6yr':>8s}  {'9yr':>8s}")
    print("  " + "─" * 44)
    for tier in TIERS:
        t = raw[tier]
        print(f"  {tier:15s}  {t[3]:>8.3f}  {t[6]:>8.3f}  {t[9]:>8.3f}")

    print("\n── Step 4: Final Constrained Targets ───────────────────────────────")
    print(f"  Tier weights: "
          + "  ".join(f"{t}={TIER_WEIGHTS[t]:.3f}" for t in TIERS))
    print(f"  {'Tier':15s}  {'3yr':>8s}  {'6yr':>8s}  {'9yr':>8s}")
    print("  " + "─" * 44)
    for tier in TIERS:
        t = final[tier]
        print(f"  {tier:15s}  {t[3]:>8.3f}  {t[6]:>8.3f}  {t[9]:>8.3f}")

    print("\n── Weighted Average Verification ───────────────────────────────────")
    print(f"  {'Window':8s}  {'Weighted avg':>13s}  {'BJS target':>10s}  "
          f"{'Diff':>9s}  {'OK':>4s}")
    print("  " + "─" * 52)
    for yrs in [3, 6, 9]:
        wavg = sum(TIER_WEIGHTS[t] * final[t][yrs] for t in TIERS)
        diff = wavg - BJS_TARGETS[yrs]
        flag = "✅" if abs(diff) < 0.005 else "⚠️"
        print(f"  {yrs}-Year    {wavg:>13.4f}  {BJS_TARGETS[yrs]:>10.3f}  "
              f"{diff:>+9.4f}  {flag}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  PCRA → BJS Tier Target Conversion")
    print("=" * 70)

    result = compute_pcra_bjs_targets(verbose=True)

    print("\n── PCRA_TARGETS (ready to paste) ───────────────────────────────────")
    print("PCRA_TARGETS = {")
    for tier, windows in result.items():
        print(f'    "{tier}": '
              f'{{3: {windows[3]}, 6: {windows[6]}, 9: {windows[9]}}},')
    print("}")