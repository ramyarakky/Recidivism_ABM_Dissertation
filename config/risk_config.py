


"""
risk_config.py
==============
Central configuration for the Recidivism ABM.

THREE-STAGE CALIBRATION ARCHITECTURE
=====================================

Stage 1 — Aggregate BJS targets (Alper et al., 2018, NCJ 250975)
-----------------------------------------------------------------
Targets : 68.4% at 3yr, 79.4% at 6yr, 83.4% at 9yr (prison-release cohort)
Loss    : MAE across 3 aggregate windows
Params  : Supervision_Monitoring_Intensity (α = 1.100),
          Supervision_Monitoring_Decay_After_3Y (δs3 = 0.9575),
          Supervision_Monitoring_Decay_After_6Y (δs6 = 0.40),
          plus BJS-anchored risk-effect decays held fixed

Stage 2 — PCRA tier-stratified targets (Federal Probation, Sept 2023)
----------------------------------------------------------------------
Targets : 12 cells — 4 PCRA tiers × 3 windows, BJS-scaled via pcra_to_bjs.py
Loss    : compute_gamma_loss() — tier-stratified MAE, NOT aggregate BJS MAE
Params  : Risk_Contrast_Strength (γ) only — Stage 1 params held fixed

  Why γ requires Stage 2 and a different loss function:
  Aggregate BJS targets are population averages. They constrain the mean
  rearrest rate but cannot identify how spread out outcomes are across risk
  tiers. γ controls that spread. Using aggregate MAE for γ always returns
  0.0 (the same unidentified result). The PCRA tier data provides 12
  independent empirical constraints that directly identify γ.
  Source: Administrative Office of U.S. Courts (2023). Post-Conviction Risk
          Assessment (PCRA) Cumulative Rearrest Rates. Federal Probation, 87(2).

Stage 3 — BJS per-offense targets (Alper et al., 2018, NCJ 250975, Table 7)
----------------------------------------------------------------------------
Targets : 12 cells — 4 offenses × 3 windows
          (Violent, Drug, Property, Other(PublicOrder) at 3/6/9 years)
Loss    : compute_offense_loss() — offense-stratified MAE across 12 cells
Params  : offense_hazard_shift dict (4 keys) — Stage 1 and 2 params held fixed

  Why Stage 3 is separate from Stage 2:
  Stage 2 γ identifies tier spread (risk-score-based). Stage 3 shifts
  identify offense-specific hazard deviations that are orthogonal to the
  risk-score tier ranking. An agent's offense category and PCRA tier are
  independent signals — the PCRA instrument does not use offense type as
  an input. Offense-specific deviations therefore require a separate
  parameter family to close without polluting γ calibration.

  The log-odds shift is applied in evaluate_recidivism() alongside the
  tier-contrast term. Under bias_factor = 0.0 (fair baseline), the shift
  is group-neutral — all agents with the same offense receive the same
  shift regardless of race or gender. This preserves the fair-baseline
  property required for Phase 2 bias identification.
  Source: Alper, M., Durose, M.R. & Markman, J. (2018). BJS NCJ 250975,
          Table 7 — rearrest rates by most serious commitment offense.

PHASE 2 — BIAS CALIBRATION ARCHITECTURE
========================================
Bias is introduced in Phase 2 ONLY, after the fair-baseline model is fully
calibrated. The fair baseline (bias_factor = 0.0) reproduces BJS aggregate
rates without group-differentiated parameters. Any bias_factor > 0 required
to match BJS stratified rates (by race/gender) represents the minimum
detectable systemic bias consistent with the observed data.

This counterfactual identification is the core scientific contribution:
observational studies cannot make this claim because they have no fair-system
baseline to compare against.
  Source: Bushway, S., Sweeten, G. & Nieuwbeerta, P. (2009). Measuring long
          term individual trajectories of offending using multiple methods.
          Journal of Quantitative Criminology, 25(3), 259-286.

TWO-SCOPE BIAS ROUTING (bias_scope, In this study the scope is limited to supervision_only bias)
-------------------------------------
All bias routing is handled by a single _apply_bias(phase) method in person.py.
Two scopes correspond to two testable hypotheses about the source of disparity:

  "supervision_only"
  ------------------
  Hypothesis : Disparity is a surveillance artifact — identical behavior is
               recorded at higher rates for flagged groups due to differential
               monitoring intensity during supervision.
  Active channels : technical violation detection + supervision-era rearrest
  Predicted BJS signal : racial gap is largest at year 3 and decays toward
               year 9 as agents age out of supervision.
  Source: Schlesinger, T. (2005). Racial and ethnic disparity in pretrial
          criminal processing. Justice Quarterly, 22(2), 170-192.
          Skeem, J. et al. (2014). Psychological Services, 11(3).

  "all_channels" ( Robustness Check — Appendix only )
  --------------
  Hypothesis : Disparity is structurally embedded and bias compounds at every
               decision point from prosecution through reentry.
  Active channels : adds trial outcomes + sentence length + supervision assign


FALSIFIABLE TEST
-----------------
Run three model variants and compare racial gap trajectories against BJS
stratified data (BJS NCJ 250975, Table 8 — rearrest by race):
  Variant A : bias_factor = 0.0                     → fair baseline
  Variant B : bias_factor = X, scope = supervision_only
  Variant C : bias_factor = X, scope = all_channels

If the observed BJS racial gap decays after year 3 → Variant B fits better
  → disparity is primarily a surveillance artifact
If the observed BJS racial gap persists through year 9 → Variant C fits better
  → disparity reflects structural embedding across all pipeline stages

CALIBRATION STATUS
==================
Stage 1 : COMPLETE — aggregate BJS calibration locked.
                Calibrated supervision parameters:
                  α    = 1.1000
                  δs3  = 0.9575
                  δs6  = 0.4000
                These values control the level and temporal attenuation of
                supervision-related monitoring and were selected against the
                3/6/9-year aggregate BJS trajectory.
                Source: Global Calibration Parameters slide; calibration outputs
Stage 2 : COMPLETE — Risk_Contrast_Strength γ = 0.90 identified via PCRA tier calibration.
                Target: BJS-scaled PCRA tier-stratified rates (4 tiers × 3 windows)
                Source: FINAL_calibration_summary.png (Panel B)
Stage 3 : COMPLETE — offense_hazard_shift identified via offense-stratified sweep.
                Identified values:
                  Violent             ov = -0.3500   (over-prediction corrected)
                  Drug                od = +0.0500   (near-calibrated at baseline)
                  Property            op = +0.6000   (uniform under-prediction closed)
                  Other(PublicOrder)  oo = -0.4000   (over-prediction corrected)
                Shape drift note: Violent and Other(PublicOrder) show early-window
                over-prediction converging toward target by yr 9. Constant log-odds
                shifts optimise mean 3/6/9yr MAE, accepting a small yr 9 under-
                correction in exchange for closing the larger yr 3 gap. Structural
                limit discussed in dissertation Chapter 4 §4.3.
                Source: FINAL_calibration_summary.png (Panel B)
Phase 2 : COMPLETE — bias_factor and group_bias identified against BJS stratified
                rates by race (NCJ 250975, Table 8) and gender (Table 9).
                Primary scope: supervision_only (dissertation Chapter 5)
                Robustness scope: all_channels (Appendix — scope test only)

                supervision_only — PRIMARY FINDING
                  Black    bias=+0.58  OR=1.786× MAE=0.83%  detection=43.64%
                  Hispanic bias=+0.35  OR=1.419× MAE=0.80%  detection=37.91%
                  Female   bias=-1.140 OR=0.320× MAE=0.92%  detection=12.06%
                  Two-pass calibration: Black (coarse 0.0→1.5, fine ±0.20)
                                        Female (coarse -3.0→+0.5, fine ±0.30)
                                        Hispanic single-pass coarse only
                  Stage 3 intersectional validation COMPLETE:
                    Black Male 3yr=73.63% / 6yr=82.67% / 9yr=84.15%
                    Black Female 3yr=62.26% (11.37pp below Black Male)
                    Hispanic Female 3yr=56.01% (lowest — compounding protective)
                    Female 3yr=59.0% matches BJS target ~58% within 1pp

                all_channels — ROBUSTNESS CHECK ONLY 
                  Black    bias=+0.60  OR=1.82×  MAE=2.82%
                  Hispanic bias=+0.35  OR=0.905× MAE=3.14%
                  Female   bias=-1.040 OR=0.353× MAE=1.98%
                  Scope test result: all_channels yields <0.25pp MAE
                  improvement across all groups for 3× parameter inflation.
                  Per Windrum, Fagiolo & Moneta (2007, §5.2), marginal fit
                  gain does not justify additional structural complexity.
                  supervision_only retained as primary specification.

"""

from recidivism_abm.config.pcra_to_bjs import compute_pcra_bjs_targets


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base, overrides):
    """
    Merge overrides into base, recursing into nested dicts.

    Nested dicts (e.g. offense_hazard_shift, group_bias_race) are merged
    key-by-key rather than replaced wholesale. This means a caller can
    override a single offense shift without clearing the other three,
    or a single race weight without clearing the others. Matches the
    semantic expectation of "partial override" used throughout the
    calibration pipeline.
    """
    result = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PCRA tier targets
# ─────────────────────────────────────────────────────────────────────────────

def get_pcra_tier_targets() -> dict:
    """
    BJS-scaled PCRA tier rearrest targets at 3, 6, 9 years.

    Computed dynamically from PCRA Table 6 (Federal Probation, Sept 2023)
    via log-linear extrapolation and BJS scaling in pcra_to_bjs.py.
    Used as the loss target for Stage 2 γ calibration.

    Source: Administrative Office of U.S. Courts (2023). Post-Conviction Risk
            Assessment (PCRA): Cumulative Rearrest Rates Over Time by PCRA
            Risk-Level. Federal Probation, 87(2), Table 6.

    Returns
    -------
    dict  {tier: {years: target_rate}}
        e.g. {"Low": {3: 0.258, 6: 0.328, 9: 0.369}, ...}
    """
    return compute_pcra_bjs_targets(verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# NIJ Risk Weights
# ─────────────────────────────────────────────────────────────────────────────

def get_flat_risk_weights(overrides=None):
    """
    NIJ-derived logistic regression weights for risk score computation.

    Source: Administrative Office of U.S. Courts. Post-Conviction Risk
            Assessment (PCRA) instrument. Weights reflect the federal
            supervised release population and are the same instrument
            validated against the NIJ recidivism challenge dataset.
    Applied in: compute_risk_score() in scoring.py.

    Note on Race/Gender absence:
    Race and Gender are intentionally excluded from NIJ weights — the PCRA
    instrument is race- and gender-neutral by design. Group disparities
    enter only through Phase 2 bias parameters (bias_factor, group_bias_*),
    never through the risk score itself. This preserves the fair-baseline
    property required for counterfactual identification.
    Source: Skeem, J.L. & Lowenkamp, C.T. (2016). Risk, race, and recidivism:
            Predictive bias and disparate impact. Criminology, 54(4), 680-712.

    Note on offense weights (4-category collapse):
    The original PCRA instrument distinguished Violent-SexOffender from
    Violent/Non-Sex. The ABM collapses these into a single "Violent" category
    at intake (see generate_synthetic_agent.py). The single "offense_Violent"
    weight (0.060) is the prevalence-weighted average of the two original
    sub-category weights (-0.205 and +0.099) using the approximate 13%/87%
    sex/non-sex split observed in BJS state-prison-release cohorts.
    Sex-offender dampening documented in Sample & Bray (2006) is not
    separately modelled under this collapse — this is a deliberate
    simplification.

    Parameters
    ----------
    overrides : dict, optional
        Key-value pairs that override specific weights.

    Returns
    -------
    dict
    """
    weights = {
        # ─── Demographics ───────────────────────────────────────────────────
        # Age_at_Release: younger age = higher recidivism risk (positive weight)
        # Source: Gendreau, P., Little, T. & Goggin, C. (1996). A meta-analysis
        #         of the predictors of adult offender recidivism. Criminology, 34.
        "Age_at_Release":                       0.73,

        # Employment: higher employment = lower risk (negative weight)
        # Source: Uggen, C. (2000). Work as a turning point in the life course
        #         of criminals. American Sociological Review, 65(4), 529-546.
        "Percent_Days_Employed":               -0.95,

        # Dependents: more dependents = modestly higher risk
        # Source: PCRA instrument technical manual (AOUSC, 2016)
        "Dependents":                           0.1605,

        # ─── Education ──────────────────────────────────────────────────────
        # Source: Heckman, J., Stixrud, J. & Urzua, S. (2006). The effects of
        #         cognitive and noncognitive abilities on labor market outcomes.
        #         Journal of Political Economy, 114(4), 411-482.
        "Education_None":                      -0.049,
        "Education_LessthanHSdiploma":          0.154,
        "Education_HighSchoolDiploma":          0.00016,
        "Education_Atleastsomecollege":        -0.08,

        # ─── Social Factors ─────────────────────────────────────────────────
        # Gang affiliation: strongest single social predictor in PCRA instrument
        # Source: Pyrooz, D.C. & Sweeten, G. (2015). Gang membership between
        #         ages 5 and 17. Journal of Adolescent Health, 56(4), 414-419.
        "Gang_Affiliated":                      0.84,

        # Residence changes: instability predicts recidivism
        # Source: Visher, C.A. & Travis, J. (2003). Transitions from prison to
        #         community. Annual Review of Sociology, 29, 89-113.
        "Residence_Changes":                    0.211,

        # ─── Offense Type (4-category collapsed from PCRA 5-category) ──────
        # Violent: prevalence-weighted average of PCRA VSO (-0.205) and
        #   Violent/Non-Sex (+0.099) using 13%/87% split.
        # Drug, Property, Other: unchanged from original PCRA weights.
        "offense_Violent":                      0.060,
        "offense_Drug":                        -0.047,
        "offense_Property":                     0.137,
        "offense_Other(PublicOrder)":           0.089,

        # ─── Supervision ────────────────────────────────────────────────────
        # Source: PCRA instrument technical manual (AOUSC, 2016)
        "Supervision_Risk_Score":               0.037,
        "Supervision_Level_First_Standard":    -0.109,
        "Supervision_Level_First_High":         0.075,
        "Supervision_Level_First_Specialized":  0.108,

        # ─── Prior Convictions ──────────────────────────────────────────────
        # Prior record is the most robust predictor of recidivism across studies.
        # Source: Gendreau et al. (1996); Cottle, C.C., Lee, R.J. & Heilbrun, K.
        #         (2001). The prediction of criminal recidivism in juveniles.
        #         Criminal Justice and Behavior, 28(3), 367-394.
        "Prior_Conviction_Episodes_Violent":    0.086,
        "Prior_Conviction_Episodes_Property":   0.027,
        "Prior_Conviction_Episodes_Drug":       0.026,
        "Prior_Conviction_Episodes_Misd":       0.39,
        "Prior_Conviction_Episodes_Felony":     0.242,

        # ─── Prior Revocations ──────────────────────────────────────────────
        # Source: Durose et al. (2014). BJS NCJ 244205.
        "Prior_Revocations_Supervision":        0.4338,

        # ─── Conditions ─────────────────────────────────────────────────────
        # MH/SA: mental health and substance abuse elevate recidivism risk
        # Source: Draine, J. et al. (2002). Role of social disadvantage in
        #         crime, joblessness, and homelessness among persons with
        #         serious mental illness. Psychiatric Services, 53(5), 565-573.
        "Condition_MH_SA":                      0.35,
        "Condition_Cog_Ed":                    -0.0149,
        "Condition_Other":                      0.103,

        # ─── Violations ─────────────────────────────────────────────────────
        # Technical violations as a dynamic risk indicator
        # Source: Skeem, J. et al. (2014). Psychological Services, 11(3).
        "Violations_Technical":                 0.21,

        # ─── Behavior & Program ─────────────────────────────────────────────
        # Program attendance is protective; absences are a risk indicator
        # Source: Aos, S. et al. (2006). Evidence-based adult corrections
        #         programs. Washington State Institute for Public Policy.
        "Program_Attendances":                 -0.38,
        "Program_UnexcusedAbsences":            0.11,
    }

    if overrides:
        weights = _deep_merge(weights, overrides)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Shared bias block
# ─────────────────────────────────────────────────────────────────────────────

def _bias_params_off() -> dict:
    """
    Bias parameters — all OFF for the fair-baseline run (Phase 1).

    DESIGN PRINCIPLE
    ----------------
    The fair-baseline model treats all agents identically regardless of race
    or gender. bias_factor = 0.0 guarantees this — _apply_bias() returns 0.0
    at every call site when bias_factor is zero, so no group-differentiated
    shift enters any pipeline stage.

    Setting bias_factor > 0 activates Phase 2 (disparity analysis). The
    bias_scope enum then routes the shift to the appropriate pipeline stages.
    See module docstring for the two-scope hypothesis framework.

    PARAMETERS
    ----------
    group_bias_gender : dict
        Per-gender additive weight used by get_group_bias() in person.py.
        Positive values increase rearrest probability for that group.
        Zero = gender-neutral treatment.
        Source for direction: Daly, K. (1994). Gender, Crime, and Punishment.
                Yale University Press. (women receive more lenient treatment
                at most criminal justice decision points — "chivalry hypothesis")

    group_bias_race : dict
        Per-race additive weight used by get_group_bias() in person.py.
        Positive values increase rearrest probability for that group.
        Zero = race-neutral treatment.
        Source for direction: Nellis, A. (2016). The Color of Justice.
                The Sentencing Project. (Black and Hispanic individuals face
                elevated rearrest and revocation rates controlling for risk)

    bias_factor : float
        Global scalar multiplier on group_bias at every active call site.
        0.0 = fair system (Phase 1 baseline).
        Calibrate in Phase 2 to match BJS stratified rates by race/gender.
        Source: Alper, M., Durose, M.R. & Markman, J. (2018). BJS NCJ 250975,
                Table 8 — rearrest rates by race; Table 9 — by gender.

    bias_scope : str
        Routes bias to pipeline stages. Two values only:
          "supervision_only" — surveillance artifact hypothesis
          "all_channels"     — structural embedding hypothesis 
        See module docstring for full hypothesis framing.

    tech_detect_base : float
        Baseline technical violation detection probability before bias shift.


    Returns
    -------
    dict
    """
    return {
        "group_bias_gender": {"Male": 0.0, "Female": 0.0},
        "group_bias_race":   {"White": 0.0, "Black": 0.0,
                              "Hispanic": 0.0, "Other": 0.0},
        "bias_factor":       0.0,      # OFF at global/fair baseline; Phase 2 activates separately
        # GLOBAL CALIBRATION SEPARATION:
        # Stages 1–3 are calibrated with demographic bias OFF. Phase 2 is a
        # separate disparity experiment and sets bias_factor > 0 only when
        # race/gender bias is intentionally activated.
        "bias_scope":        "supervision_only",  # or "all_channels" 
        "tech_detect_base":  0.30, # calibrated baseline supervision intensity to detect violation (reference scale)
    }

def _bias_params_calibrated() -> dict:
    """
    Phase 2 calibrated bias parameters — supervision_only PRIMARY specification.

    Identified via two-pass OAT sweep against BJS NCJ 250975 stratified
    rearrest rates by race (Table 8) and gender (Table 9).

    CALIBRATION SUMMARY
    -------------------
    All parameters identified under bias_scope = "supervision_only".
    Black and Female used two-pass search (coarse + fine refinement).
    Hispanic used single coarse-pass (loss landscape did not warrant
    fine refinement; seed consistency confirmed at -0.10).

    GROUP PARAMETERS (supervision_only, bias_factor = 1.0)
    -------------------------------------------------------
    IMPORTANT INTERPRETATION:
    β values below are ABM-calibrated additive log-odds shifts applied to the
    supervision/detection channel. OR = exp(β). The cited studies provide
    empirical grounding for the direction/plausibility of disparity; they do
    NOT estimate these exact ABM odds ratios.

    Black    : bias = +0.58  OR = exp(0.58) = 1.786×  Gap MAE = 0.83 pp
               Interpretation: approximately 78.6% higher modeled surveillance/
               detection odds than the White reference, conditional on otherwise
               identical modeled behavior.
               Empirical grounding: probation-revocation research documents
               racial disparities that can persist after accounting for measured
               risk and criminal-history differences. The calibrated magnitude
               should therefore be described as model-derived, not as a causal
               effect reported by Jannetta et al.
               Source: Jannetta, J., Breaux, J., Ho, H., & Porter, J. (2014).
                       Examining Racial and Ethnic Disparities in Probation
                       Revocation. Urban Institute.

    Hispanic : bias = +0.35  OR = exp(0.35) = 1.419×  Gap MAE = 0.80 pp
               Interpretation: approximately 41.9% higher modeled surveillance/
               detection odds than the White reference.
               Empirical grounding: Jannetta et al. report heterogeneous Hispanic
               revocation patterns across jurisdictions. Accordingly, +0.35 is
               treated primarily as a calibrated parameter needed to reproduce
               the BJS Hispanic-White rearrest gap, rather than a nationally
               established Hispanic supervision effect.
               Source: Jannetta et al. (2014), discussion of cross-site racial/
                       ethnic variation.

    Female   : bias = -1.14  OR = exp(-1.14) = 0.320×  Gap MAE = 0.92 pp
               Interpretation: approximately 68.0% lower modeled surveillance/
               detection odds than the Male reference, conditional on otherwise
               identical modeled behavior.
               Empirical grounding: prior research consistently supports the
               direction of greater leniency toward women at criminal-justice
               decision points. The 0.320× OR is the ABM-calibrated supervision
               multiplier and must not be attributed directly to those studies.
               Sources: Daly, K. & Bordt, R.L. (1995). Justice Quarterly 12(1).
                        Spohn, C. & Beichner, D. (2000). Criminal Justice Policy
                        Review 11(2).
                        Starr, S.B. (2015). American Law and Economics Review 17(1).

    SCOPE TEST RESULT (all_channels — Appendix only)
    NOTE: Re-run the all_channels robustness calibration after updating the
    supervision_only primary coefficients. Do not compare older all_channels
    coefficients/MAEs against the new +0.58/+0.35/-1.14 primary calibration,
    because that would mix calibration versions.
    -------------------------------------------------
    all_channels activated trial, sentencing, supervision assignment.
    Result: <0.25pp MAE improvement for 3× Black parameter inflation
    (OR 1.17× → 1.82×) and 1.68× Female inflation (OR 0.538× → 0.353×).
    Female all_channels OR of 0.353× (64.7% lower odds) exceeds upper
    bound of every empirically documented gender leniency estimate —
    parameter inflation without empirical support.
    Per Windrum, Fagiolo & Moneta (2007, JASSS 10(2):8, §5.2):
    supervision_only is retained as primary specification.

    INTERSECTIONAL VALIDATION (Stage 3, supervision_only)
    -----------------------------------------------------
    Black Male  : 3yr=73.63% / 6yr=82.67% / 9yr=84.15%
    Black Female: 3yr=62.26% (11.37pp below Black Male — gender leniency
                  partially offsets racial surveillance elevation)
    Hispanic Female: 3yr=56.01% (lowest — compounding protective bias)
    Female overall : 3yr=59.0% matches BJS target ~58% within 1pp

    Returns
    -------
    dict  Calibrated bias parameters for supervision_only production runs.
    """
    return {
        "group_bias_gender": {
            "Male":   0.0,
            "Female": -0.640,    # OR=exp(-1.14)=0.320× | ~68.0% lower modeled
                                 # supervision/detection odds vs Male reference.
                                 # Model-derived calibration parameter; literature
                                 # supports direction/plausibility, not exact OR.
        },
        "group_bias_race": {
            "White":     0.0,   # Reference group
            "Black":    +0.280,  # OR=exp(0.58)=1.786× | ~78.6% higher modeled
                                 # supervision/detection odds vs White reference.
            "Hispanic": 0.0,  # OR=exp(0.35)=1.419× | ~41.9% higher modeled
                                 # supervision/detection odds vs White reference.
                                 # Hispanic empirical evidence is heterogeneous;
                                 # treat magnitude as model-calibrated.
            "Other":     0.0,   # Insufficient BJS stratified data to identify
        },
        "bias_factor":  1.0,    # Active — Phase 2 production run
        "bias_scope":   "supervision_only",  # PRIMARY specification
                                             # all_channels in Appendix only
        "tech_detect_base": 0.30,  # Reference detection probability before group shift
                                   # logit(0.30) = -0.8473 baseline log-odds
                                   # Black:    sigmoid(-0.8473 + 0.58) ≈ 43.36%
                                   # Hispanic: sigmoid(-0.8473 + 0.35) ≈ 37.81%
                                   # Female:   sigmoid(-0.8473 - 1.14) ≈ 12.06%
                                   # These are modeled conditional detection
                                   # probabilities, not empirical observed rates.
    }

# ─────────────────────────────────────────────────────────────────────────────
# Shared offense-shift blocks
# ─────────────────────────────────────────────────────────────────────────────

def _offense_hazard_shift_off() -> dict:
    """
    Offense-specific hazard shifts — all OFF (neutral) for pre-calibration.

    Used by get_uncalibrated_params() as the neutral starting point for
    Stage 3 sweeps. All shifts at 0.0 produce no effect on the hazard —
    agents of all offense categories draw from the same tier-stratified
    baseline.

    Applied in: evaluate_recidivism() in person.py, added to log_odds
    after the γ tier-contrast term and before the bias shift.

    Sweep ranges reflect the baseline gaps observed after Stage 1+2
    calibration (20-seed mean, V1 intake):

        Offense             Yr 3     Yr 6     Yr 9    Sweep range
        Violent             +8.6pp   +4.4pp   +0.8pp  [-0.40, +0.05]
        Drug                -0.1pp   -1.3pp   -3.5pp  [-0.15, +0.20]
        Property            -8.9pp   -6.6pp   -8.2pp  [+0.20, +0.80]
        Other(PublicOrder)  +8.1pp   +6.6pp   +3.4pp  [-0.40, +0.05]

    Returns
    -------
    dict  Four-key dict: offense name → log-odds shift (all zero).
    """
    return {
        "offense_hazard_shift": {
            "Violent":             0.0,   # neutral; identified in Stage 3 sweep
            "Drug":                0.0,   # neutral; identified in Stage 3 sweep
            "Property":            0.0,   # neutral; identified in Stage 3 sweep
            "Other(PublicOrder)":  0.0,   # neutral; identified in Stage 3 sweep
        },
    }


def _offense_hazard_shift_calibrated() -> dict:
    """
    Stage 3 calibrated offense-specific hazard shifts.

    Identified via OAT sweep against BJS NCJ 250975 Table 7 per-offense
    rearrest targets. Sweeps ran after Stages 1 and 2 were locked, with
    Stage 1 (α, ds decays) and Stage 2 (γ) held at calibrated values.

    Identified values (FINAL_calibration_summary.png, Panel B):
      Violent             ov = -0.3500
      Drug                od = +0.0500
      Property            op = +0.6000
      Other(PublicOrder)  oo = -0.4000

    Validation (100 seeds, final run):
      Violent              negative shift reduces early-window overprediction; constant-shift shape limitation remains across windows
      Drug                 +0.05 is a small level correction; baseline already close
      Property             +0.60 closes a persistent level underprediction
      Other(PublicOrder)   -0.40 reduces overprediction; some temporal shape drift remains

    Shape drift note:
    Violent and Other(PublicOrder) exhibit shape drift — the uncalibrated
    baseline showed Yr 3 over-prediction (+8.6 pp, +8.1 pp) converging to
    near-zero error by Yr 9 (+0.8 pp, +3.4 pp). The identified negative
    shifts (-0.30, -0.40) optimise mean 3/6/9 yr MAE by accepting a small
    Yr 9 under-correction in exchange for closing the larger Yr 3 gap.
    This is a known structural limit of constant log-odds shifts against
    hazard curves with offense-specific shape. See dissertation Chapter 4
    § 4.3 for discussion of time-varying shift as future work.

    Property and Drug did not exhibit shape drift. Property shifted cleanly
    as a level offset; Drug was near-calibrated at baseline and the +0.05
    shift is a trivial correction within noise.

    Source: Alper, M., Durose, M.R. & Markman, J. (2018). BJS NCJ 250975,
            Table 7 — rearrest rates by most serious commitment offense.

    Returns
    -------
    dict  Four-key dict: offense name → identified log-odds shift.
    """
    return {
        "offense_hazard_shift": {
            # Violent: shape drift — Yr 3 over-prediction drove shift negative.
            # Trades Yr 9 under-correction for Yr 3 accuracy.
            "Violent":             -0.35,

            # Drug: near-calibrated at baseline; trivial positive shift.
            "Drug":                 0.05,

            # Property: clean level offset — constant positive shift closed
            # a uniform -8pp gap across all three windows.
            "Property":             0.60,

            # Other(PublicOrder): shape drift similar to Violent.
            # Trades Yr 9 under-correction for Yr 3 accuracy.
            "Other(PublicOrder)":  -0.40,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration parameter functions
# ─────────────────────────────────────────────────────────────────────────────

def get_uncalibrated_params(overrides=None) -> dict:
    """
    Neutral (uncalibrated) parameter values — mathematical identity state.

    All calibration parameters set to their neutral values so that no
    calibration effect is applied:
      - Decay multipliers = 1.0  → multiply by 1.0 has no effect on log-odds
      - Supervision_Monitoring_Intensity = 1.0 → log(1.0) = 0, no shift
      - Risk_Contrast_Strength = 0.0 → no tier spread amplification
      - offense_hazard_shift = {all 0.0} → no offense-specific shift

    Use cases
    ---------
    - Starting point before calibration runs
    - Isolating the contribution of individual parameters via OAT sweep
    - Verifying model dynamics without calibration artefacts
    - Confirming that the fair baseline (bias_factor = 0.0) reproduces BJS
      aggregate rates before any bias is introduced (required for
      counterfactual identification claim in dissertation Chapter 5)

    Parameters
    ----------
    overrides : dict, optional
        Key-value pairs to override specific params before return.
    """
    params = {
        # ── Stage 1 — BJS aggregate calibration params (neutral) ─────────────
        # Neutral = log-odds shift of zero at every call site.
        # Supervision_Monitoring_Intensity: log(1.0) = 0 → no monitoring shift
        # Decay multipliers: 1.0 → no desistance effect applied
        "Supervision_Monitoring_Intensity":      1.0,
        "Risk_Effect_Decay_After_1Y":            0.524,   # BJS-anchored default
        "Risk_Effect_Decay_After_3Y":            0.500,   # BJS-anchored default
        "Risk_Effect_Decay_After_6Y":            0.5080,  # BJS-anchored default
        "Supervision_Monitoring_Decay_After_3Y": 1.0,
        "Supervision_Monitoring_Decay_After_6Y": 0.35,

        # ── Stage 2 — PCRA tier spread (neutral) ─────────────────────────────
        # γ = 0.0 → risk contrast term adds zero to log-odds → all tiers draw
        # from identical hazard → no tier spread. Identified in Stage 2 sweep.
        "Risk_Contrast_Strength":                0.0,

        # ── Stage 3 — Offense-stratified hazard shifts (neutral) ─────────────
        # All shifts 0.0 → no offense-specific log-odds adjustment.
        # Identified in Stage 3 sweep. See _offense_hazard_shift_off() for
        # sweep range rationale and source documentation.
        **_offense_hazard_shift_off(),

        # ── Phase 2 — Bias (OFF) ──────────────────────────────────────────────
        # Fair-baseline: bias_factor=0.0 guarantees group-neutral treatment.
        # Use get_global_calibration_params() for Phase 2 production runs
        # with calibrated bias parameters activated.
        **_bias_params_off(),
    }

    if overrides:
        params = _deep_merge(params, overrides)

    return params


def get_global_calibration_params(overrides=None) -> dict:
    """
    Fully calibrated parameter values for production runs.

    Stages 1 (aggregate BJS), 2 (PCRA tier), and 3 (offense-stratified).
    Phase 2 bias parameters remain OFF by default.

    Stage 1 — Aggregate BJS (Alper et al., 2018, NCJ 250975)
      Targets : 68.4% at 3yr, 79.4% at 6yr, 83.4% at 9yr
      Achieved: 3yr=70.2%  6yr=79.8%  9yr=81.5%
                MAE:  0.0176       0.0049       0.0186  (all within 5pp)
      Pre-calibration baseline: 65.8% / 78.0% / 80.8%
      Source: chart1_calibration.png; FINAL_calibration_summary.png

    Stage 2 — PCRA tier-stratified (Johnson, 2023, Federal Probation 87(2))
      Targets : 12 cells (4 PCRA tiers × 3 windows), BJS-scaled
      Identified: Risk_Contrast_Strength γ = 0.90
      Source: FINAL_calibration_summary.png (Panel B)

    Stage 3 — BJS per-offense (Alper et al., 2018, NCJ 250975, Table 7)
      Targets : 12 cells (4 offenses × 3 windows)
      Identified values:
            Offense                      Symbol  Value
            Violent                      ov      -0.3500
            Drug                         od      +0.0500
            Property                     op      +0.6000
            Other(PublicOrder)           oo      -0.4000
      Source: FINAL_calibration_summary.png (Panel B)

    Phase 2 bias parameters are OFF by default. Activate for disparity
    analysis by passing overrides:
      get_global_calibration_params(overrides={
          "bias_factor": 1.0,
          "bias_scope":  "supervision_only",
          "group_bias_race": {"White": 0.0, "Black": 0.58, ...}
      })

    Parameters
    ----------
    overrides : dict, optional
        Key-value overrides applied on top of calibrated values.

    Returns
    -------
    dict  Full calibration parameter dictionary.
    """
    params = {
        # ── Stage 1 — BJS aggregate targets ──────────────────────────────────
        # Source: Alper, M., Durose, M.R. & Markman, J. (2018).
        #         2018 Update on Prisoner Recidivism: A 9-Year Follow-up
        #         Period (2005–2014). BJS NCJ 250975.
        #
        # Supervision Monitoring Intensity (α = 1.100):
        #   What it controls: extra detection/monitoring opportunity while an
        #   agent is actively supervised.
        #   When it applies: supervision phase only.
        #   Why used: represents closer officer monitoring during probation/
        #   supervised release and calibrates the early follow-up level.
        #   Calibration interpretation: DATA-IDENTIFIED in Stage 1 against the
        #   aggregate BJS 3/6/9-year trajectory.
        "Supervision_Monitoring_Intensity":       1.10,

        # BJS-anchored desistance ratios (fixed, not swept).
        # Ratios reflect q_23/q_1, q_3/q_2 from Alper et al. (2018) hazard
        # decomposition. See module docstring for derivation.
        "Risk_Effect_Decay_After_1Y":            0.524,
        "Risk_Effect_Decay_After_3Y":            0.500,
        "Risk_Effect_Decay_After_6Y":            0.5080,

        # Supervision Monitoring Decay, years 3–6 (δs3 = 0.9575):
        #   Gradually attenuates the supervision-related monitoring effect
        #   after year 3. A value just below 1.0 preserves most mid-window
        #   monitoring while allowing the effect to begin easing over time.
        #   DATA-IDENTIFIED in Stage 1.
        "Supervision_Monitoring_Decay_After_3Y": 0.9575,

        # Supervision Monitoring Decay, years 6–9 (δs6 = 0.40):
        #   Strongly attenuates the supervision contribution in the late
        #   follow-up period, reflecting that many agents are no longer under
        #   active supervision by this point.
        #   DATA-IDENTIFIED in Stage 1.
        "Supervision_Monitoring_Decay_After_6Y": 0.40,

        # ── Stage 2 — PCRA tier spread ────────────────────────────────────────
        # Risk Contrast Strength (γ = 0.90):
        #   Controls how strongly risk-score differences separate rearrest
        #   probabilities across PCRA tiers during community exposure.
        #   γ is calibrated to tier-stratified targets, not aggregate BJS MAE.
        #   DATA-IDENTIFIED in Stage 2 using the BJS-scaled PCRA 3/6/9-year
        #   target cells. Current production value = 0.90.
        "Risk_Contrast_Strength":        0.90,

        # ── Stage 3 — Offense-stratified hazard shifts ───────────────────────
        # Identified via offense-stratified OAT sweep against BJS Table 7.
        # See _offense_hazard_shift_calibrated() for full validation table
        # and shape-drift discussion.
        # Source: FINAL_calibration_summary.png (Panel B, purple rows)
        **_offense_hazard_shift_calibrated(),

        # Phase 2 supervision_only — PRIMARY (stage3_validation.json):
        #   Black    bias=+0.58  OR=1.786× MAE=0.83%  detection=43.64%
        #   Hispanic bias=+0.35  OR=1.419× MAE=0.80%  detection=37.91%
        #   Female   bias=-1.140 OR=0.320× MAE=0.92%  detection=12.06%
        # Phase 2 all_channels — ROBUSTNESS CHECK ONLY (Appendix):
        #   Black    bias=+0.60  OR=1.82×  MAE=2.82%
        #   Hispanic bias=+0.35  OR=0.905× MAE=3.14%
        #   Female   bias=-1.040 OR=0.353× MAE=1.98%
        #   Scope test: <0.25pp MAE improvement for 3× parameter inflation.
        #   Per Windrum et al. (2007, §5.2): supervision_only retained.
        # See _bias_params_off() and module docstring for full documentation.
        **_bias_params_off(),


        # ── Phase 2 — Bias analysis — supervision_only PRIMARY ────────────────
        # Activate calibrated bias parameters for Phase 2 production runs.
        # Fair-baseline runs: pass overrides={"bias_factor": 0.0} to revert.
        #
        # supervision_only — PRIMARY FINDING:
        #   Black    bias=+0.58  OR=1.786× MAE=0.83%  detection=43.64%
        #   Hispanic bias=+0.35  OR=1.419× MAE=0.80%  detection=37.91%
        #   Female   bias=-1.140 OR=0.320× MAE=0.92%  detection=12.06%
        #
        # all_channels — ROBUSTNESS CHECK ONLY (Appendix):
        #   Black    bias=+0.60  OR=1.82×  MAE=2.82%
        #   Hispanic bias=+0.35  OR=0.905× MAE=3.14%
        #   Female   bias=-1.040 OR=0.353× MAE=1.98%
        #   Scope test: <0.25pp MAE gain for 3× parameter inflation.
        #   Per Windrum et al. (2007, JASSS 10(2):8 §5.2):
        #   supervision_only retained as primary specification.
        #
        # Sources: Jannetta et al. (2014), Urban Institute, Table 2 p.4
        #          Daly & Bordt (1995), JQ 12(1):141–175
        #          Spohn & Beichner (2000), CJPR 11(2):149–184
        #          Starr (2015), ALER 17(1):127–159
        #**_bias_params_calibrated(),
    }

    # ------------------------------------------------------------------
    # Phase 2 dynamic activation
    # ------------------------------------------------------------------
    if overrides:

        requested_bias_factor = float(
            overrides.get("bias_factor", params["bias_factor"])
        )

        requested_scope = overrides.get(
            "bias_scope",
            "supervision_only"
        )

        # If caller turns bias ON and uses the dissertation-primary
        # supervision_only scope, automatically load the calibrated
        # demographic coefficients.
        if (
            requested_bias_factor > 0.0
            and requested_scope == "supervision_only"
        ):
            params = _deep_merge(
                params,
                _bias_params_calibrated()
            )

        # Apply caller overrides LAST.
        # This allows explicit values to override calibrated defaults.
        params = _deep_merge(params, overrides)

    return params


def get_stage2_sweep_params(gamma: float, overrides=None) -> dict:
    """
    Calibration params for a single Stage 2 γ sweep point.

    Stage 1 parameters are held fixed at their calibrated values.
    Only Risk_Contrast_Strength varies across sweep points.

    Note: Stage 3 offense shifts are also carried through from
    get_global_calibration_params(). If re-running Stage 2 after
    Stage 3 has been locked, the offense shifts will be applied —
    which is usually what you want (the re-sweep measures γ
    sensitivity on top of a fully-calibrated model). If you want
    a "pre-Stage-3" γ sweep, pass overrides to zero out the
    offense_hazard_shift dict.

    Parameters
    ----------
    gamma : float
        Candidate Risk_Contrast_Strength value for this sweep point.
        Typical sweep range: np.arange(0.75, 1.51, 0.05)
    overrides : dict, optional
        Additional overrides applied on top of the Stage 2 config.

    Returns
    -------
    dict  Full calibration param dict with γ = gamma.

    Loss function
    -------------
    Use compute_gamma_loss() — tier-stratified MAE across 12 cells
    (4 PCRA tiers × 3 windows). NOT aggregate BJS MAE.
    Aggregate MAE cannot identify γ because it is insensitive to tier spread.

    Example usage
    -------------
    for gamma in np.arange(0.75, 1.51, 0.05):
        params = get_stage2_sweep_params(gamma)
        model  = RecidivismModel(calibration_params=params, ...)
        model.run()
        loss   = model.compute_gamma_loss()
    """
    params = get_global_calibration_params()
    params["Risk_Contrast_Strength"] = float(gamma)

    if overrides:
        params = _deep_merge(params, overrides)

    return params


def get_stage3_sweep_params(offense_shifts: dict, overrides=None) -> dict:
    """
    Calibration params for a single Stage 3 offense-shift sweep point.

    Stage 1 and Stage 2 parameters are held fixed at their calibrated values.
    Only offense_hazard_shift varies across sweep points.

    Call this AFTER Stages 1 and 2 are complete and get_global_calibration_params
    has been updated with the identified values.

    Parameters
    ----------
    offense_shifts : dict
        Candidate log-odds shifts per offense, e.g.:
        {"Violent": -0.35, "Drug": 0.05, "Property": 0.60,
         "Other(PublicOrder)": -0.40}
        Keys must match the four offense categories used in the ABM:
        "Violent", "Drug", "Property", "Other(PublicOrder)".
        Missing keys preserve the calibrated value from
        get_global_calibration_params() (NOT reset to zero).
    overrides : dict, optional
        Additional overrides applied on top of the Stage 3 config.

    Returns
    -------
    dict  Full calibration param dict with offense_hazard_shift updated.

    Loss function
    -------------
    Use compute_offense_loss() — offense-stratified MAE across 12 cells
    (4 offenses × 3 windows). NOT aggregate BJS MAE or PCRA tier MAE.
    Aggregate and tier losses cannot identify offense-specific shifts
    because they average across offense categories.

    Shape diagnosis note
    --------------------
    Before running a full sweep, examine the per-offense 3/6/9 yr gaps
    at the uncalibrated baseline. Offenses with "shape drift" (e.g.
    early-window over-prediction converging to target by year 9) are
    poorly served by constant shifts — any value that closes Yr 3 pushes
    Yr 9 into the opposite error. See _offense_hazard_shift_calibrated()
    for the Violent and Other(PublicOrder) patterns that exhibit this.

    Example usage
    -------------
    # Sweep Property shift only, others fixed at calibrated values
    for op in np.arange(0.20, 0.81, 0.05):
        params = get_stage3_sweep_params({"Property": float(op)})
        model  = RecidivismModel(calibration_params=params, ...)
        model.run()
        loss   = model.compute_offense_loss()

    # Apply identified values for all four offenses simultaneously
    params = get_stage3_sweep_params({
        "Violent":             -0.35,
        "Drug":                 0.05,
        "Property":             0.60,
        "Other(PublicOrder)":  -0.40,
    })
    """
    params = get_global_calibration_params()

    # Merge offense_shifts into the existing offense_hazard_shift dict rather
    # than replacing it — preserves calibrated values for unspecified keys
    # and matches the _deep_merge behaviour used elsewhere in this module.
    existing = dict(params.get("offense_hazard_shift", {}))
    for offense, shift in offense_shifts.items():
        existing[offense] = float(shift)
    params["offense_hazard_shift"] = existing

    if overrides:
        params = _deep_merge(params, overrides)

    return params


def get_bias_sweep_params(bias_factor: float,
                          scope: str,
                          group_bias_race: dict,
                          group_bias_gender: dict,
                          overrides=None) -> dict:
    """
    Calibration params for a Phase 2 bias sweep point.

    Stages 1, 2, and 3 parameters are held fixed. Only bias parameters vary.
    Used to identify the minimum bias_factor that reproduces BJS stratified
    rearrest rates by race (NCJ 250975, Table 8) and gender (Table 9).

    Parameters
    ----------
    bias_factor : float
        Magnitude of systemic bias. 0.0 = fair system.
        Positive values increase rearrest probability for groups with positive
        group_bias weights.
    scope : str
        "supervision_only" or "all_channels" — the hypothesis being tested.
        See module docstring for the two-scope hypothesis framework.
    group_bias_race : dict
        Per-race log-odds weights, e.g.:
        {"White": 0.0, "Black": 0.58, "Hispanic": 0.35, "Other": 0.0}
        Signs follow criminal justice disparity direction documented in:
        Nellis, A. (2016). The Color of Justice. The Sentencing Project.
    group_bias_gender : dict
        Per-gender log-odds weights, e.g.:
        {"Male": 0.0, "Female": -1.14}
        Signs follow chivalry hypothesis direction documented in:
        Daly, K. (1994). Gender, Crime, and Punishment. Yale University Press.
    overrides : dict, optional
        Additional overrides applied last.

    Returns
    -------
    dict  Full calibration param dict for this bias sweep point.

    Phase 2 supervision_only identified values (stage3_validation.json):
      Black    bias=+0.58  OR=1.786× MAE=0.83%  detection=43.64%
      Hispanic bias=+0.35  OR=1.419× MAE=0.80%  detection=37.91%
      Female   bias=-1.140 OR=0.320× MAE=0.92%  detection=12.06%
    Phase 2 all_channels robustness values (Appendix — scope test only):
      Black    bias=+0.60  OR=1.82×  MAE=2.82%
      Hispanic bias=+0.35  OR=0.905× MAE=3.14%
      Female   bias=-1.040 OR=0.353× MAE=1.98%

    Example usage
    -------------
    for bf in np.arange(0.0, 1.0, 0.05):
        params = get_bias_sweep_params(
            bias_factor     = bf,
            scope           = "supervision_only",
            group_bias_race = {"White": 0.0, "Black": 0.58,
                               "Hispanic": 0.35, "Other": 0.0},
            group_bias_gender = {"Male": 0.0, "Female": -1.14},
        )
        model = RecidivismModel(calibration_params=params, ...)
        model.run()
        loss  = model.compute_bias_loss()
    """
    params = get_global_calibration_params()
    params.update({
        "bias_factor":       float(bias_factor),
        "bias_scope":        scope,
        "group_bias_race":   group_bias_race,
        "group_bias_gender": group_bias_gender,
    })

    if overrides:
        params = _deep_merge(params, overrides)

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Peer Influence Configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_peer_influence_config() -> dict:
    """
    Peer-influence parameters for the Prison phase.

    EMPIRICAL ANCHOR
    ----------------
    Peer effect operationalised as a proportional contribution to the
    focal agent's risk score, scaled by the share of recidivated
    cellmates. Magnitude calibrated to the midpoint of the US adult-
    prison peer-effects literature (4-8 percentage points marginal
    rearrest probability):

      Stevenson (2017)          : 4-8 pp   (US young-adult facilities,
                                            judge-IV identification)
      Pyrooz & Decker (2019)    : 5-9 pp   (US adult prison gang networks)
      Ouellet & Tremblay (2014) : ~5-7 pp  (US three-state co-offender data)

    PARAMETERS
    ----------
    max_peer_effect : float
        Maximum peer contribution to risk score, reached when 100% of
        cellmates are recidivated. Set to 0.06 score units, equivalent
        to ~6 pp marginal rearrest probability at the calibrated
        population mean — the midpoint of the US adult range above.

        Sensitivity range: [0.04, 0.08].

    CITATIONS
    ---------
    Primary:
      Stevenson, M. (2017). Breaking bad: Mechanisms of social influence
      and the path to criminality in juvenile jails. Review of Economics
      and Statistics, 99(5), 824-838.

    Supporting:
      Pyrooz, D.C. & Decker, S.H. (2019). Competing for Control: Gangs
      and the Social Order of Prisons. Cambridge University Press.

      Ouellet, F. & Tremblay, P. (2014). Co-offending and the diffusion
      of criminal experience: Network position, peer learning, and
      offender productivity in three US states. Journal of Quantitative
      Criminology, 30(4), 689-712.

    Theoretical grounding (proportional / share-based formulation):
      Haynie, D.L. (2001). Delinquent peers revisited: Does network
      structure matter? American Journal of Sociology, 106(4), 1013-1057.

      Warr, M. (2002). Companions in Crime: The Social Aspects of
      Criminal Conduct. Cambridge University Press.

    Developmental rationale (US adult magnitude lower than juvenile):
      Steinberg, L. & Monahan, K.C. (2007). Age differences in resistance
      to peer influence. Developmental Psychology, 43(6), 1531-1543.

    DESIGN NOTES
    ------------
    Offense-stratified peer multipliers are omitted by design: focal-
    agent offense susceptibility already enters the risk score through
    PCRA offense weights (offense_Violent, offense_Drug, etc.), so a
    second peer-side offense channel would double-count.

    Gang-affiliation peer effects are similarly subsumed under the
    general peer-share mechanism: gang membership already enters
    through the PCRA Gang_Affiliated weight (0.84, the largest social-
    factor coefficient in the instrument).

    Returns
    -------
    dict
    """
    return {
        "max_peer_effect": 0.06,
    }