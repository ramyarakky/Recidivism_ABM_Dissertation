from mesa import Agent
import random
from recidivism_abm.scoring import compute_risk_score, normalize_score
import math 
import numpy as np
import traceback

class Person(Agent):
    def __init__(self, unique_id, model, agent_data):
        super().__init__(unique_id, model)
        self.agent_data = {}
        self.unique_id = unique_id
        self.model = model
        #self.rand_val= self.model.np_rng.random()
        self.exited_due_to_recidivism = False
        self.is_supervised = False
        # Load synthetic attributes
        for key, value in agent_data.items():
            setattr(self, key, value)
        # 🖨️ Print all initialized attributes
        #print(f"🔍 Agent {self.unique_id} initialized with:")
        #for attr, val in vars(self).items():
        #   print(f"   {attr}: {val}")

        # Simulation flags
        if self.model.current_month <= self.model.warmup_months:
            self.justice_state = "Trial"
        else:
            #print('Study Period')
            self.justice_state = "Free"
            
        # ── Trial wait threshold (pre-trial detention / case processing time) ──────
        # BJS "Felony Defendants in Large Urban Counties" (NCJ 243777, 2009, Table 21)
        # reports median time from arrest to adjudication:
        #   Violent felonies  : median ~9 months  (range 4-18+)
        #   Drug felonies     : median ~6 months  (range 3-12)
        #   Property felonies : median ~5 months  (range 2-10)
        # The original uniform draw of 1-4 months was 2-3x too short for felony-level
        # offenses. Offense-stratified medians with +-2 month jitter produce realistic
        # variation. Public-order / misdemeanour cases process faster (~3 months).
        # Source: Cohen, T.H. & Kyckelhahn, T. (2010). BJS NCJ 228944.
        _offense = getattr(self, "offense", "Other(PublicOrder)")
        _wait_medians = {
            "Violent":             9,
            "Drug":                6,
            "Property":            5,
            "Other(PublicOrder)":  3,
        }
        _median = _wait_medians.get(_offense, 6)
        self.trial_wait_threshold = max(1, _median + int(model.np_rng.integers(-2, 3)))
        self.raw_score = compute_risk_score(self,weights=self.model.weights) or 0.0
        #print('Raw score:',self.raw_score)
        self.months_in_state = 0
        self.time_served_months = 0
        self.sentence_length_months = 0
        self.dynamic_risk_score = 0.0
        #self.reincarceration_within_3yr = 0
        self.committed_new_offense_this_month = False
        self.recidivated_agent = False
        self.study_eligible_agent = False
        self.free_at_warmup_end = False
        self.free_start_month = 0
        self.MonthsCrimeFree = 0
        self.supervision_start_month=0
        self.agent_entry_month = self.model.current_month
        self.entry_month = self.model.current_month  # overwritten by model after init
        self.cohort = "warmup" if self.model.current_month <= self.model.warmup_months else "study"
        self.rearrest_month = None
        self.release_month = None
        self.recidivism_month = None
        self.justice_state_changed = False
        self.normalized_score =0.0
        # Optional flags for analysis
        self.quarters_at_risk = 0
        self.rearrest_1_yrs = False
        self.rearrest_2_yrs = False
        self.rearrest_3_yrs = False
        self.rearrest_4_yrs = False
        self.rearrest_5_yrs = False
        self.rearrest_6_yrs = False
        self.rearrest_7_yrs = False
        self.rearrest_8_yrs = False
        self.rearrest_9_yrs = False
        self.community_months_at_risk = 0
        self.peer_influence_boost = 0.0   # set in Prison phase; zero elsewhere
        self.rearrest_quarterly = False

        # Counterfactual monitoring-burden metrics
        self.cumulative_supervision_exposure = 0.0
        self.supervision_exposure_events = 0
        self.residence_changes_during_study = 0
        self.residence_move_opportunities = 0

        # -------------------------------------------------------
        # Dynamic Reentry Adjustment diagnostics
        # -------------------------------------------------------
        self.dynamic_reentry_adjustment_value = 0.0
        self.dra_employment_effect = 0.0
        self.dra_program_effect = 0.0
        

    def initialize_attributes_from_data(self):
        """No-op: agent_data is applied in __init__ via setattr loop."""
        pass

    def display_all_attributes(self):
        print(f"\n🧠 Agent {self.unique_id} Full Attribute Snapshot:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")
            
    def get_group_bias(self):
        """
        Returns additive bias score based on group membership.
        Uses model-level calibration dict so you can tune without changing agent code.
        """
        cal = getattr(self.model, "calibration", {}) or {}

        gender_map = (cal.get("group_bias_gender", {}) or {})
        race_map   = (cal.get("group_bias_race", {}) or {})

        g = getattr(self, "Gender", None)  # "Male"/"Female"
        r = getattr(self, "Race", None)    # "White"/"Black"/"Hispanic"/"Other"

        return float(gender_map.get(g, 0.0)) + float(race_map.get(r, 0.0))

    def _apply_bias(self, phase: str) -> float:
        """
        Returns the log-odds bias shift for the current agent and phase.
        Single entry point — no per-pathway multipliers needed.

        bias_scope controls which pipeline stages are active:

        "supervision_only" → differential surveillance only
                Channels: technical violation detection + supervision-era rearrest
                Hypothesis: disparity is a surveillance artifact — same behavior,
                            more monitoring = more recorded arrests for flagged groups
                BJS signal: racial gap decays after supervision ends (years 3→9)

        "all_channels"     → structural bias across all decision points
                Channels: adds trial outcomes + sentence length + supervision assignment
                Hypothesis: disparity is structurally embedded — bias compounds
                            at every stage from prosecution through reentry
                BJS signal: racial gap persists through year 9

        Parameters
        ----------
        phase : str
            The pipeline stage requesting a bias shift. Must match a key in
            ACTIVE_PHASES for the current scope to return a non-zero value.

        Returns
        -------
        float
            Log-odds shift to add at the calling site. Returns 0.0 if:
            - bias_factor == 0.0  (fair-baseline run)
            - phase is not active under current scope
        """
        cal = getattr(self.model, "calibration", {}) or {}
        bias_factor = float(cal.get("bias_factor", 0.0))
        if bias_factor == 0.0:
            return 0.0

        scope = cal.get("bias_scope", "supervision_only")
        group_bias = self.get_group_bias()

        ACTIVE_PHASES = {
            "supervision_only": {
                "detect",                # technical violation detection
                "supervision_rearrest",  # rearrest draw during supervision
            },
            "all_channels": {
                "detect",                # technical violation detection
                "supervision_rearrest",  # rearrest draw during supervision
                "trial",                 # release/diversion probabilities
                "sentence",              # sentence length multiplier
                "supervision_assign",    # post-prison supervision assignment
            },
        }

        if phase not in ACTIVE_PHASES.get(scope, set()):
            return 0.0

        return bias_factor * group_bias

    # ─────────────────────────────────────────────
    # Absolute Risk Normalization (0–1 Scale)
    # ─────────────────────────────────────────────
    
    def normalize_risk_absolute(self):
        """
        Absolute 0–1 risk using MODEL warm-up reference stats.
        Safe against overflow.
        """
        raw = compute_risk_score(self, weights=self.model.weights) or 0.0

        midpoint = float(getattr(self.model, "risk_midpoint", 0.0))
        spread = float(getattr(self.model, "risk_spread", 1.0))

        # Safety: prevent division explosion
        spread = max(1e-3, spread)  # use 1e-3 instead of 1e-6 for extra stability

        z = (raw - midpoint) / spread
        normalized_risk =  self.safe_sigmoid(z)
        self.dynamic_risk_score = normalized_risk
        return normalized_risk

    def get_pcra_tier(self) -> str:
        if (hasattr(self, "_cached_tier")
                and hasattr(self, "_cached_tier_month")
                and self._cached_tier_month == self.model.current_month):
            return self._cached_tier

        # Use frozen study-entry score if available — prevents score drift
        # during study period from re-inverting tier boundaries
        if hasattr(self, "frozen_risk_score"):
            raw = self.frozen_risk_score
            z   = (raw - self.model.risk_midpoint) / max(self.model.risk_spread, 1e-6)
            # Inline sigmoid — same as normalize_risk_absolute() but uses frozen raw
            if z > 60:    r = 1.0
            elif z < -60: r = 0.0
            elif z >= 0:  r = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                r  = ez / (1.0 + ez)
        else:
            # Fallback to live score before study entry (warmup period)
            r = self.normalize_risk_absolute()

        # Use population-derived boundaries
        b_low    = getattr(self.model, "tier_boundary_low",    0.25)
        b_lowmod = getattr(self.model, "tier_boundary_lowmod", 0.50)
        b_mod    = getattr(self.model, "tier_boundary_mod",    0.75)

        if r < b_low:        tier = "Low"
        elif r < b_lowmod:   tier = "LowModerate"
        elif r < b_mod:      tier = "Moderate"
        else:                tier = "High"

        self._cached_tier       = tier
        self._cached_tier_month = self.model.current_month
        return tier

    def _compute_peer_influence(self) -> float:
        """
        Share-based peer-influence boost for the Prison phase.

        The boost is proportional to the share of recidivated cellmates,
        scaled by max_peer_effect from the peer-influence config:

            boost = max_peer_effect × (recidivated_cellmates / total_cellmates)

        Anchored to US adult-prison peer-effects literature (4-8 pp marginal
        rearrest probability across studies). Sources:
        Stevenson, M. (2017). Breaking bad. Review of Economics and
            Statistics, 99(5), 824-838.
        Pyrooz, D.C. & Decker, S.H. (2019). Competing for Control.
            Cambridge University Press.
        Ouellet, F. & Tremblay, P. (2014). Co-offending and the diffusion
            of criminal experience. Journal of Quantitative Criminology,
            30(4), 689-712.

        Theoretical grounding (share-based formulation):
        Haynie, D.L. (2001). Delinquent peers revisited. American Journal
            of Sociology, 106(4), 1013-1057.
        Warr, M. (2002). Companions in Crime. Cambridge University Press.

        See get_peer_influence_config() for full citation chain.

        Returns
        -------
        float
            Peer-influence boost in [0, max_peer_effect]. Returns 0.0 when
            the agent has no cellmates.
        """
        cellmates = [
            m for m in self.model.grid.get_cell_list_contents([self.pos])
            if m is not self
        ]
        if not cellmates:
            return 0.0

        recidivated = sum(
            1 for m in cellmates if getattr(m, "recidivated_agent", False)
        )
        recid_share = recidivated / len(cellmates)

        return self.model.peer_influence_weights["max_peer_effect"] * recid_share

    def compute_adjusted_risk_score(self, phase: str) -> float:
        """
        Compute the normalised dynamic risk score for a given justice phase.

        Adds peer-influence boost during the Prison phase if enabled.
        Returns a score bounded in [0, 1].

        Peer-influence formulation
        --------------------------
        Share-based proportional contribution anchored to US adult-prison
        peer-effects literature (4-8 pp range). See _compute_peer_influence()
        and get_peer_influence_config() for citations and rationale.
        """
        # ─── Peer Influence (Prison Phase Only) ─────────────────────────
        if phase == "Prison" and self.model.enable_peer_influence:
            self.peer_influence_boost = self._compute_peer_influence()
        else:
            self.peer_influence_boost = 0.0

        # ─── Compute Final Score ────────────────────────────────────────
        score = self.normalize_risk_absolute() + self.peer_influence_boost
        return max(0.0, min(score, 1.0))


    def assign_sentence_length(self):
        """
        Assigns prison sentence length (in months) based on offense type and normalised risk score.

        Sentence ranges are anchored to:
          - BJS "Felony Sentences in State Courts" series (Rosenmerkel et al., 2009,
            BJS NCJ 226846, Table 2.5): mean time-to-serve by offense category.
          - US Sentencing Commission (USSC) Annual Report (2022), Table 8:
            average sentence length by primary offense type in federal courts.
          - BJS "Time Served in State Prison" (Bhati, 2010, NCJ 228827):
            actual time served (not imposed sentence) by offense type.

        Ranges reflect the lower end of actual time served (min) to the upper
        end of the imposed sentence distribution (max):
          Violent-SexOffender : 48-120 mo  (USSC mean ~102; BJS state mean ~88)
          Violent/Non-Sex     : 36-96 mo   (BJS state mean ~63 mo time-served)
          Drug                : 18-60 mo   (BJS state mean ~24 mo time-served)
          Property            : 6-48 mo    (BJS state mean ~18 mo time-served)
          Other(PublicOrder)  : 6-60 mo    (broad range; includes driving, weapons)

        The 0.25 risk dampener on the scaling factor prevents extreme outliers
        for low-risk agents while preserving sentence differentiation. This is a
        modelling choice, not a directly empirical parameter.
        """

        # ─── 1. Normalize risk score for prison phase ───
        risk = self.normalize_risk_absolute()
        # ─── 2. Base sentence ranges by offense type ─────────────────────────
        ranges = {
                    "Violent":             (42, 108),   # blended: SexOffender (48-120) + Non-Sex (36-96)
                    "Drug":                (18, 60),
                    "Property":            (6, 48),
                    "Other(PublicOrder)":  (6, 60)
                }

        min_months, max_months = ranges.get(self.offense, (6, 120))
        sentence = min_months  # safe default if TypeError fires below

        try:
            # ─── 3. Risk-scaled sentence within empirical range ───────────────────
            # The 0.25 scalar caps the risk contribution to 25% of the sentence
            # range, so even the highest-risk agent receives at most:
            #   min + 0.25 * (max - min)
            # This keeps simulated sentences within the lower quarter of each
            # offense range — consistent with BJS median time-served data
            # (Bhati, 2010, NCJ 228827) sitting well below the statutory maximum.
            #
            # STRUCTURAL SIMPLIFICATION: The 0.25 value is not directly estimated
            # from data. It was chosen so that median simulated sentences match
            # published BJS time-served medians. It is acknowledged as a structural
            # simplification in the dissertation (Chapter 4, §4.3) and does not
            # affect the aggregate recidivism calibration targets, which are
            # validated independently of sentence length.
            scaled = risk * 0.25 #0.50
            sentence = min_months + (max_months - min_months) * scaled
            #print("Sentence b4",sentence)
            # ─── 4. Round to nearest 3-month block ───
            sentence = 3 * round(sentence / 3)
            sentence = max(min_months, sentence)

            # ── Phase 2 sentence length bias ("all_channels" scope only) ─────────────
            # _apply_bias("sentence") returns a log-odds group shift converted here
            # to a multiplicative sentence scaling factor: sentence *= (1 + shift).
            # Under "supervision_only" scope this returns 0.0 — sentence length
            # remains group-neutral (identical treatment regardless of race/gender).
            # Empirical basis for direction: Black defendants receive sentences
            # approximately 19% longer than white defendants with equivalent
            # criminal history and offense severity (USSC, 2017 — Demographic
            # Differences in Federal Sentencing, Table 3).
            # Source: U.S. Sentencing Commission (2017). Demographic Differences
            #         in Federal Sentencing Practices. Washington, D.C.

            sentence_bias = self._apply_bias("sentence")
            if sentence_bias != 0.0:
                sentence *= (1.0 + sentence_bias)
            sentence = max(min_months, sentence)

            #print("Sentence after",sentence)
        except TypeError as e:
            print(f"Agent {self.unique_id} failed multiplication during assign_sentence_length: {e}")


        # ─── 5. Final assignment ───
        self.sentence_length_months = int(sentence)
        self.Prison_Years = self.sentence_length_months
        return self.Prison_Years

    def derive_supervision_duration(self):
        """
        Calculates supervision duration (months) based on offense type, risk score,
        and prior conviction history. Also assigns supervision level, risk code, and
        a 1-10 supervision risk score.

        Supervision duration anchors:
        - US Parole Commission guidelines and state parole board data compiled in:
            Petersilia, J. (2003). When Prisoners Come Home. Oxford University Press.
            Ch. 3 (average parole terms 12-48 months by offense severity).
        - BJS "Probation and Parole in the United States" annual series (Kaeble, 2021,
            NCJ 256094): median supervision terms by offense type.
        - Violent-SexOffender: High supervision, 48-mo bonus reflects mandatory
            sex-offender registration and extended parole terms in most US states.
            Source: Sample, L.L. & Bray, T.M. (2006). Are sex offenders dangerous?
            Criminology & Public Policy, 5(1), 59-82.

        Prior conviction boost (2*felony + misdemeanour):
        - Reflects guidelines-based supervision extensions for repeat offenders
            consistent with USPC and state parole board practices. Multipliers
            are a structural simplification; exact coefficients vary by jurisdiction.

        Differentiated supervision cap (post-fix):
        - Violent and sex offenders are capped at 60 months rather than 48,
            reflecting mandatory sex-offender registration requirements and extended
            parole terms documented in most US states (Sample & Bray, 2006).
        - All other offense types retain the empirically validated 48-month ceiling
            (Petersilia, 2003; Kaeble, 2021).
        """

        # ─── Supervision Profile by Offense ───────────────────────────────────────
        # Each offense type maps to a supervision level, risk code, numeric risk score
        # (1-10), and a bonus duration added on top of the risk-scaled base.
        # Levels follow APPA supervision classification (Standard / High / Specialized).
        # Source: American Probation and Parole Association (APPA) Standards (2010).
        #
        # bonus values anchor the offense-specific floor:
        #   Violent-SexOffender : +48 mo  mandatory extended supervision / SO registration
        #   Violent/Non-Sex     : +36 mo  high-risk violent offense premium
        #   Drug                : +48 mo  treatment supervision requirements
        #   Property            : +24 mo  moderate supervision floor
        #   Other(PublicOrder)  : +12 mo  minimum supervision floor
        offense = getattr(self, "offense", "Other(PublicOrder)")
        profile_map = {
            "Violent":             ("High",        "VIO", 9,  40),   # blended
            "Drug":                ("Specialized", "DRG", 8,  48),
            "Property":            ("Standard",    "PRP", 5,  24),
            "Other(PublicOrder)":  ("Standard",    "PUB", 3,  12)
        }
        
        
        level, code, risk_score, bonus = profile_map.get(offense, ("Standard", "UNK", 1, 0))

        # ─── Assign Supervision Attributes to Agent ───────────────────────────────
        # These are written once at trial sentencing and persist through the
        # supervision phase. Supervision_Risk_Score feeds into compute_risk_score()
        # via the NIJ weight "Supervision_Risk_Score": 0.037.
        self.Supervision_Level_First = level
        self.Supervision_Risk_Code   = code
        self.Supervision_Risk_Score  = max(1, min(risk_score, 10))  # clamp to 1–10

        # ─── Safe Default Duration ────────────────────────────────────────────────
        # Initialised before the try block so the return value is always defined
        # even if a TypeError fires during the risk-scaled calculation below.
        # Value = offense-based bonus + 12-month floor, consistent with BJS minimum
        # supervision terms (Kaeble, 2021, NCJ 256094).
        duration = max(3, bonus + 12)

        try:
            # ─── Risk-Scaled Base Duration ────────────────────────────────────────
            # base = 6 + 30*risk: minimum 6 months supervision for all releasees;
            # risk-scaled extension up to 36 additional months for highest-risk agents.
            # Reflects BJS median parole terms of 12-24 months (Kaeble, 2021).
            # risk is normalised to [0, 1] via normalize_risk_absolute(), so the
            # maximum contribution from this term is 36 months at risk = 1.0.
            risk = self.compute_adjusted_risk_score("Supervision") or 0.0
            if risk is None or risk_score is None:
                raise ValueError(
                    f"Agent {self.unique_id} missing values: risk={risk}, "
                    f"risk_score={risk_score}"
                )

            base = 6 + 30 * risk   # 6–36 months risk-scaled

            # ─── Offense Severity Boost ───────────────────────────────────────────
            # 1.2 × risk_score adds up to +12 months for the highest-severity offense
            # (risk_score = 10). Reflects that high-severity offenses carry longer
            # supervision terms independent of individual risk score.
            severity_boost = 1.2 * risk_score   # up to +12 months

            # ─── Prior Conviction Boost ───────────────────────────────────────────
            # Guidelines-based extensions for repeat offenders are standard practice
            # (USPC; state parole board guidelines), though no single national benchmark
            # quantifies the per-conviction increment directly.
            #
            # STRUCTURAL SIMPLIFICATION: 2 months per felony episode and 1 month per
            # misdemeanour episode produce durations within the empirically validated
            # 3–48 month range (Petersilia, 2003; Kaeble, 2021) across the full range
            # of conviction histories in the synthetic population. Acknowledged as a
            # structural simplification in the dissertation (Chapter 4, §4.3).
            # Sensitivity is bounded by the hard cap below — no agent exceeds it
            # regardless of prior history depth.
            felony      = getattr(self, "Prior_Conviction_Episodes_Felony", 0)
            misd        = getattr(self, "Prior_Conviction_Episodes_Misd",   0)
            prior_boost = 2 * felony + misd

            # ─── Compute Raw Duration ─────────────────────────────────────────────
            # Sum all components: risk-scaled base + offense severity + offense bonus
            # + prior conviction extension. This is the unclamped supervision term.
            duration = base + severity_boost + bonus + prior_boost

            # ─── Differentiated Duration Cap ──────────────────────────────────────
            # Violent offenders: 60-month ceiling.
            #   Sex offender registration requirements and extended parole conditions
            #   in most US states routinely exceed the standard 48-month cap.
            #   Source: Sample, L.L. & Bray, T.M. (2006). Criminology & Public Policy.
            # All other offenses: 48-month ceiling.
            #   Consistent with Petersilia (2003) and BJS median parole terms.
            #   The cap prevents unrealistically long supervision for agents with
            #   extensive prior histories while keeping outcomes within documented ranges.
            max_duration = 60 if offense == "Violent" else 48
            duration = max(3, min(duration, max_duration))

            # ─── Round to Nearest 3-Month Block ──────────────────────────────────
            # Supervision terms are administered in quarterly review cycles.
            # Rounding prevents fractional months and aligns with how supervision
            # terms are reported in BJS administrative data.
            duration = 3 * round(duration / 3)

        except TypeError as e:
            # If any component is None or non-numeric, fall back to the offense-based
            # default computed before the try block. This preserves simulation
            # continuity without crashing the agent.
            print(f"Agent {self.unique_id} failed multiplication during "
                f"derive_supervision_duration: {e}")
            duration = max(3, bonus + 12)

        return duration
    
    # ─────────────────────────────────────────────
    # Math helpers (kept as instance methods for readability)
    # ─────────────────────────────────────────────
    def safe_sigmoid(self,x: float):
        """
        Numerically-stable sigmoid with clamping to prevent exp overflow.
        """
        # Clamp to a safe range. Beyond ~±60 sigmoid is effectively 0/1 anyway.
        if x > 60:
            return 1.0
        if x < -60:
            return 0.0

        if x >= 0:
            ez = math.exp(-x)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(x)
            return ez / (1.0 + ez)

    def logit(self, p: float):
        """
        Convert probability to log-odds (with safe clamping).
        Prevents math errors when p is 0 or 1.
        """
        p = max(1e-6, min(1.0 - 1e-6, p))
        return math.log(p / (1.0 - p))

    # ─────────────────────────────────────────────
    # BJS-calibrated baseline quarterly rearrest hazard
    # ─────────────────────────────────────────────
    '''
    def _bjs_baseline_quarterly_prob(self, quarters_since_study: int) -> float:
        """
        Baseline quarterly rearrest probability calibrated to Bureau of Justice Statistics (BJS)
        cumulative rearrest rates for released prisoners.

        Targets (cohort-average cumulative rearrest):
            - 3 years  ≈ 0.68
            - 6 years  ≈ 0.79
            - 9 years  ≈ 0.83

        Method:
        --------
        BJS reports cumulative rearrest probabilities at multi-year horizons, not
        per-period hazards. To simulate *when* rearrest occurs, we convert each
        cumulative target into an equivalent constant quarterly hazard over a
        3-year (12-quarter) block.

        For each block:
            Let C = cumulative rearrest probability over 12 quarters.
            Let q = constant quarterly rearrest probability in that block.

            Survival over 12 quarters:
                (1 − q)^12 = 1 − C

            Solving for q:
                q = 1 − (1 − C)^(1/12)

        This ensures that, in expectation, the simulated cohort matches the
        published BJS cumulative rates at 3, 6, and 9 years *before* individual
        risk or policy effects are applied.
        """

        # ─────────────────────────────────────────────
        # 0–3 years after release (first 12 quarters)
        #
        # Target cumulative rearrest at 3 years:
        #     C3 = 0.68
        #
        # Convert to quarterly hazard:
        #     q1 = 1 − (1 − 0.68)^(1/12)
        #        ≈ 0.090584
        # ─────────────────────────────────────────────
        if quarters_since_study < 12:
            return 0.0905841939

        # ─────────────────────────────────────────────
        # 3–6 years after release (next 12 quarters)
        #
        # Target cumulative rearrest at 6 years:
        #     C6 = 0.79
        #
        # Conditional on surviving the first block, the additional rearrest
        # probability in years 3–6 is:
        #     C6 − C3 = 0.79 − 0.68
        #
        # This is converted into a constant quarterly hazard over the second
        # 12-quarter block:
        #     q2 = 1 − ((1 − C6) / (1 − C3))^(1/12)
        #        ≈ 0.034492
        # ─────────────────────────────────────────────
        elif quarters_since_study < 24:
            return 0.0344922228

        # ─────────────────────────────────────────────
        # 6–9 years after release (and beyond)
        #
        # Target cumulative rearrest at 9 years:
        #     C9 = 0.83
        #
        # Conditional on surviving the first two blocks, the additional
        # rearrest probability in years 6–9 is:
        #     C9 − C6 = 0.83 − 0.79
        #
        # Converted to a constant quarterly hazard:
        #     q3 = 1 − ((1 − C9) / (1 − C6))^(1/12)
        #        ≈ 0.017455
        #
        # This declining hazard reflects empirical desistance over time.
        # ─────────────────────────────────────────────
        else:
            return 0.0174549571
    '''
    '''
    def _bjs_baseline_quarterly_prob(self, quarters_since_study: int) -> float:
        """
        Baseline quarterly rearrest probability calibrated to BJS cumulative
        rearrest rates (Alper, Durose & Markman, 2018, NCJ 250975).

        Targets: 3yr=68%  6yr=79%  9yr=83%

        Raw BJS-derived hazards (assuming unselected cohort):
            q1 = 1-(1-0.68)^(1/12) = 0.0906
            q2 = 1-((1-0.79)/(1-0.68))^(1/12) = 0.0345
            q3 = 1-((1-0.83)/(1-0.79))^(1/12) = 0.0175

        Adjusted for warmup survivor selection bias:
            The study cohort is drawn from agents who survived a 144-month
            warmup rather than a fresh prison-release cohort. Survivor
            selection produces a lower-risk cohort than the BJS population,
            requiring downward adjustment of the baseline hazard by factor
            1/1.20 to reproduce BJS aggregate targets:
                q1_adj = 0.0906 / 1.20 = 0.0755
                q2_adj = 0.0345 / 1.20 = 0.0287
                q3_adj = 0.0175 / 1.20 = 0.0145

        Source: Windrum et al. (2007) — ABM calibration methodology supports
        adjusting structural parameters to match empirical targets when
        population composition differs from the reference dataset.
        """
        # No divisor — normalization fix already accounts for survivor selection
        if quarters_since_study < 12:
            return 0.0906
        elif quarters_since_study < 24:
            return 0.0345
        else:
            return 0.0175
    '''

    def _bjs_baseline_quarterly_prob(self, quarters_since_study: int) -> float:
        """
        Baseline quarterly rearrest probability anchored to BJS NCJ 250975.

        Four-tier hazard schedule:
            Quarters 0-3  (Year 1):   base_q1                         (year-1 spike)
            Quarters 4-11 (Years 2-3): base_q1 × decay_1y              (year-1 → 2-3 drop)
            Quarters 12-23 (Years 3-6): base_q1 × decay_1y × decay_3y  (3-6 desistance)
            Quarters 24+   (Years 6-9): base_q1 × decay_1y × decay_3y × decay_6y

        BJS-anchored defaults (Alper, Durose & Markman, 2018, NCJ 250975):
            bjs_1yr_target = 0.439 → q1 = 0.1343   (year 1, quarterly)
            decay_1y       = 0.524 (q_23/q1 = 0.0704/0.1343)
            decay_3y       = 0.381
            decay_6y       = 0.507
        """
        # Year-1 quarterly hazard derived from BJS 1yr cumulative target
        bjs_1yr = 0.439
        cal = getattr(self.model, "calibration", {}) or {}
        base = 1.0 - (1.0 - bjs_1yr) ** (1.0 / 4.0)   # q1 ≈ 0.1343

        if quarters_since_study < 4:
            return base

        decay_1y = float(cal.get("Risk_Effect_Decay_After_1Y", 0.524))

        if quarters_since_study < 12:
            return base * decay_1y

        decay_3y = float(cal.get("Risk_Effect_Decay_After_3Y", 0.50))

        if quarters_since_study < 24:
            return base * decay_1y * decay_3y

        decay_6y = float(cal.get("Risk_Effect_Decay_After_6Y", 0.508))
        return base * decay_1y * decay_3y * decay_6y
    
    def compute_dynamic_reentry_adjustment(self) -> float:
        """
        Counterfactual-only Dynamic Reentry Adjustment (DRA).

        Purpose
        -------
        Allows intervention-induced changes in modifiable post-release
        characteristics to affect quarterly rearrest log-odds without
        recalculating the frozen baseline risk classification.

        Important
        ---------
        - Baseline calibration is unchanged.
        - If dynamic_reentry_adjustment is False, returns 0.
        - If employment_cf == 0 and program_cf == 0, returns 0.
        - reentry_hazard_cf is a separate system-level experiment.
        """

        # ---------------------------------------------------
        # 1. DRA must be explicitly enabled
        # ---------------------------------------------------
        if not getattr(self.model, "dynamic_reentry_adjustment", False):
            self.dynamic_reentry_adjustment_value = 0.0
            self.dra_employment_effect = 0.0
            self.dra_program_effect = 0.0
            return 0.0

        # ---------------------------------------------------
        # 2. Study-period / eligible-agent guard
        # ---------------------------------------------------
        if (
            self.model.current_month <= self.model.warmup_months
            or not getattr(self, "study_eligible_agent", False)
        ):
            self.dynamic_reentry_adjustment_value = 0.0
            self.dra_employment_effect = 0.0
            self.dra_program_effect = 0.0
            return 0.0

        employment_cf = float(
            getattr(self.model, "employment_cf", 0.0)
        )

        program_cf = float(
            getattr(self.model, "program_cf", 0.0)
        )

        # Exact baseline equivalence:
        # no intervention -> no DRA
        if np.isclose(employment_cf, 0.0) and np.isclose(program_cf, 0.0):
            self.dynamic_reentry_adjustment_value = 0.0
            self.dra_employment_effect = 0.0
            self.dra_program_effect = 0.0
            return 0.0

        # ---------------------------------------------------
        # 3. Employment contribution
        # ---------------------------------------------------
        # Current model bounds Percent_Days_Employed at 0-40.
        employment = float(
            np.clip(
                getattr(self, "Percent_Days_Employed", 0.0),
                0.0,
                40.0
            )
        )

        employment_norm = employment / 40.0

        # Existing NIJ-like protective direction/weight
        beta_employment = float(
            self.model.weights.get("Percent_Days_Employed", -0.95)
        )

        employment_effect = (
            beta_employment
            * employment_norm
            * employment_cf
        )

        # ---------------------------------------------------
        # 4. Program contribution
        # ---------------------------------------------------
        # Program participation is generated during supervision.
        # Therefore, its DRA effect is applied only while the
        # agent is actively in Supervision.
        #
        # This prevents the last observed Program_Attendances
        # value from continuing to affect rearrest after the
        # agent leaves supervision.
        program_effect = 0.0
        if self.justice_state == "Supervision":
            program_attendance = float(
                np.clip(
                    getattr(self, "Program_Attendances", 0.0),
                    0.0,
                    4.0
                )
            )

            program_norm = program_attendance / 4.0

            beta_program = float(self.model.weights.get("Program_Attendances",-0.38))

            program_effect = ( beta_program * program_norm * program_cf )

        # ---------------------------------------------------
        # 5. Combined dynamic adjustment
        # ---------------------------------------------------
        dra = employment_effect + program_effect

        # Conservative safeguard.
        dra = float(np.clip(dra, -1.0, 1.0))

        # Diagnostics
        self.dra_employment_effect = float(employment_effect)
        self.dra_program_effect = float(program_effect)
        self.dynamic_reentry_adjustment_value = dra

        return dra

    def evaluate_recidivism(self, phase: str):
        """
        SUMMARY 
        -------------------------------
        This function answers one simple question:
            "Did this person get rearrested this quarter?"

        It does that in a realistic, policy-friendly way:

        1) We only count time when the person is actually in the community
        (Free or on Supervision). Time spent in Trial/Prison does NOT count
        as "time at risk" for rearrest.

        2) We only run the rearrest lottery once every 3 community months
        (i.e., once per quarter), because BJS rearrest benchmarks are
        multi-year outcomes and quarter-by-quarter is a clean time step.

        3) We start from a BJS-based "baseline" quarterly chance of rearrest.
        Baseline means: what the *average* person would face in the real world
        at that point in time since release (early years are riskier; later years
        are calmer due to desistance).

        4) Then we adjust the baseline up/down for:
         - Individual risk (higher risk → higher chance; lower risk → lower chance)
         - Optional bias (differential detection/contact), if enabled
         NOTE: Supervision-phase monitoring is handled exclusively via the
         violation/revocation pipeline in handle_supervision_and_programs.
         This function is only called for Free-phase agents.

        5) Finally, we draw one random number to decide if rearrest happens.
        """

        # Only defined for community phases (rearrest happens in the community, not in prison/trial).
        if phase not in ("Free", "Supervision"):
            return False

        # Safety check: do not evaluate unless the person is currently in the community.
        if getattr(self, "justice_state", None) not in ("Free", "Supervision"):
            return False

        # Warm-up vs study period:
        # - Warm-up: recidivism can be "non-absorbing" (agents can cycle for calibration)
        # - Study: once recidivated, they should stop being at risk (absorbing event)
        in_warmup = self.model.current_month <= self.model.warmup_months

        # If we are in the study period and the agent already recidivated,
        # they should not be evaluated again (prevents double-counting).
        if (not in_warmup) and getattr(self, "recidivated_agent", False):
            return False

        # ─────────────────────────────────────────
        # 1) Quarter-of-exposure gate
        # ─────────────────────────────────────────
        # "community_months_at_risk" should be increased by the caller once per month
        # only when the person is Free or on Supervision.
        community_months = getattr(self, "community_months_at_risk", 0)

        # If exposure wasn't tracked, we can't evaluate rearrest properly.
        if community_months <= 0:
            return False

        # Only draw rearrest once every 3 community months (one quarter of exposure).
        # This prevents us from accidentally running the lottery every month.
        if (community_months % 3) != 0:
            return False

        # Convert community exposure to "quarters at risk" (1, 2, 3, ...).
        quarters_at_risk = community_months // 3

        # ── Step 2: Load parameters ────────────────────────────────────────────────
        cal = getattr(self.model, "calibration", {}) or {}
        risk_contrast_strength = float(cal.get("Risk_Contrast_Strength", 0.0))
        #supervision_monitoring_intensity = float(cal.get("Supervision_Monitoring_Intensity", 1.0))
        supervision_monitoring_intensity = float(self.get_effective_supervision_intensity())
       
        # ── Step 3: Phase-specific hazard adjustment ───────────────────────────────
        base_q = self._bjs_baseline_quarterly_prob(quarters_at_risk)
        # System-level reentry environment counterfactual
        reentry_hazard_cf = float(
            getattr(self.model, "reentry_hazard_cf", 0.0)
        )

        reentry_hazard_cf = float(
            np.clip(reentry_hazard_cf, -0.95, 0.95)
        )

        base_q *= (1.0 - reentry_hazard_cf)

        base_q = max(1e-6, min(1.0 - 1e-6, base_q))
        log_odds = self.logit(base_q)

        if phase == "Free":
            pass  # desistance handled in _bjs_baseline_quarterly_prob()
 
        elif phase == "Supervision":
            # ── New criminal arrest component only ────────────────────────────────
            # Represents direct new criminal arrests during supervision that are
            # NOT mediated by the violation detection pipeline.
            #
            # Empirical basis for 0.40 scalar:
            # Durose, Cooper & Snyder (2014, BJS NCJ 244205) report that among
            # state prisoners released in 2005, approximately 40% of supervision-era
            # rearrests involved new criminal conduct independent of technical
            # violation detection — the remainder were detection-mediated revocations
            # and violations handled by the violation pipeline (Draw 1).
            # Petersilia (2003, Ch. 4) corroborates: roughly 30-40% of parole
            # rearrests reflect new offenses rather than supervision non-compliance.
            # Midpoint of the empirical range (0.35-0.40) rounded to 0.40.
            #
            # This value is treated as a structural constant, not a calibration
            # parameter. It partitions the BJS baseline hazard between the two
            # rearrest channels and is not swept in OAT calibration.
            NEW_CRIME_FRACTION = 0.40   # Durose et al. (2014); Petersilia (2003)

            base_q = base_q * NEW_CRIME_FRACTION
            base_q = max(1e-6, min(1.0 - 1e-6, base_q))
            log_odds = self.logit(base_q)

            # Supervision monitoring decay applies to the new-crime channel
            # because new criminal arrests are also subject to declining
            # supervision intensity over time
            # ── SMI gate — only apply supervision monitoring to supervised agents ──────
            # Unsupervised agents (Supervision_Term == 0 or is_supervised == False)
            # are not subject to monitoring intensity. SMI should have zero effect
            # on unconditionally released individuals — they are governed solely by
            # the BJS baseline hazard and individual risk contrast (γ).
            if getattr(self, "is_supervised", True):
                # Time-decaying supervision monitoring intensity
                # After_3Y and After_6Y multipliers mirror the desistance curve,
                # reflecting reduced supervision contact intensity over time.
                # Source: Petersilia (2003); Taxman (2012)
                if quarters_at_risk >= 24:
                    supervision_monitoring_intensity *= float(
                        cal.get("Supervision_Monitoring_Decay_After_6Y", 0.35))
                elif quarters_at_risk >= 12:
                    supervision_monitoring_intensity *= float(
                        cal.get("Supervision_Monitoring_Decay_After_3Y", 1.0))
                log_odds += math.log(max(supervision_monitoring_intensity, 1e-6))

        # ── Step 4: Risk contrast (γ) — applies to both phases ────────────────────
        #risk_0_1 = self.compute_adjusted_risk_score(phase) or 0.0
        #risk_0_1 = max(0.0, min(1.0, risk_0_1))
        #log_odds += risk_contrast_strength * (risk_0_1 - 0.5)
       
        # ── Step 4: Risk contrast (γ) — PCRA-derived tier contrasts ──────────────
        # Contrasts derived from logit(tier_target_3yr) - logit(BJS_aggregate_3yr)
        # using BJS-scaled PCRA targets (pcra_to_bjs.py, Step 4 constrained values):
        #   Low:         logit(0.462) - logit(0.68) = -0.929
        #   LowModerate: logit(0.720) - logit(0.68) = +0.207
        #   Moderate:    logit(0.845) - logit(0.68) = +1.096
        #   High:        logit(0.910) - logit(0.68) = +1.671
        # γ scales the magnitude of this spread — γ=1.0 reproduces the PCRA
        # log-odds differentials exactly; γ<1.0 compresses, γ>1.0 amplifies.
        _TIER_CONTRAST = {
            "Low":         -1.0852,
            "LowModerate": +0.0116,
            "Moderate":    +0.7630,
            "High":        +1.3807
        }
        _tier     = self.get_pcra_tier()
        _contrast = _TIER_CONTRAST.get(_tier, 0.0)
        log_odds += risk_contrast_strength * _contrast

        # ── Diagnostic — remove after confirming Fix B works ─────────────────────
        #if not hasattr(self.model, "_gamma_printed"):
        #    print(f"  γ={risk_contrast_strength:.4f}  tier={_tier}  "
        #        f"contrast={_contrast:+.2f}  "
        #        f"shift={risk_contrast_strength * _contrast:+.4f}")
        self.model._gamma_printed = True

        # ── Step 5: Group bias — routed via _apply_bias() ────────────────────────
        # Active phases depend on bias_scope:
        #   "supervision_only" → fires during Supervision phase only
        #   "all_channels"     → fires during both Supervision and Free phases
        # Returns 0.0 for fair-baseline runs (bias_factor == 0.0).
        # Source: Nellis (2016). The Color of Justice. The Sentencing Project.
        #         Western (2006). Punishment and Inequality in America.
        # Triggers bias during Supervision phase only not during Free Phase
        if phase == "Supervision":
            log_odds += self._apply_bias("supervision_rearrest")
        #log_odds += self._apply_bias("supervision_rearrest")

        # ── Step 4b: Offense-specific hazard shift (Stage 3) ─────────────────────
        # Absorbs systematic deviations between the NIJ offense weight ordering
        # (PCRA-derived, federal supervised population) and the BJS state-prison
        # cohort rank order. Held at zero by default; calibrated via Stage 3 sweep
        # against BJS NCJ 250975 Table 7.
        offense_shifts = cal.get("offense_hazard_shift", {}) or {}
        log_odds += float(offense_shifts.get(self.offense, 0.0))

        # ── Step 4c: Dynamic Reentry Adjustment ────────────────
        # Counterfactual extension only.
        #
        # Does NOT change:
        #   - frozen_risk_score
        #   - PCRA tier
        #   - calibrated BJS baseline hazard
        #   - Stage 1/2/3 calibration parameters
        #
        # It modifies only the current quarterly log-odds when
        # explicitly enabled for employment/program counterfactuals.
        dra = self.compute_dynamic_reentry_adjustment()
        log_odds += dra


        # ── Final quarterly rearrest probability ───────────────
        p_rearrest = self.safe_sigmoid(log_odds)

        return self.model.np_rng.random() < p_rearrest

    def step(self):
        try:
            # Reset monthly flag
            self.committed_new_offense_this_month = False

            # Guard: skip agents with missing weights
            if self.model.weights is None:
                print(f"⚠️ Agent {self.unique_id} missing weights")
                return

            self.months_in_state += 1

            # Guard: skip agents who have already exited via recidivism
            # community_months_at_risk must NOT increment after exit
            if self.exited_due_to_recidivism:
                return

            # Increment community exposure ONCE per month, AFTER exit guard
            # Only active agents in the community accrue exposure
            if (self.justice_state in ("Free", "Supervision")
                    and not getattr(self, "recidivated_agent", False)):
                self.community_months_at_risk = getattr(self, "community_months_at_risk", 0) + 1

            self.justice_state_changed = False

            # Modular justice state logic
            if self.justice_state == "Trial":
                self.handle_trial()
            elif self.justice_state == "Prison":
                self.handle_prison()
            elif self.justice_state == "Supervision":
                self.handle_supervision()
            elif self.justice_state == "Free":
                self.handle_free()

            # Movement logic
            self.move_agent()

        except Exception as e:
            print(f"Agent {self.unique_id} step error: {e}")

    def evaluate_quarterly_recidivism(self, phase):
        """
        Evaluates whether the agent recidivates during the current quarter.

        DESIGN INTENT
        -------------
        WARM-UP PERIOD:
        • Recidivism is transient
        • Agents may cycle back to Trial
        • No outcome flags are written
        • No absorbing state

        STUDY PERIOD:
        • First recidivism is absorbing
        • Agent exits the system
        • Rearrest outcome flags (3/6/9 yrs) written once
        """

        current_month = self.model.current_month
        warmup_end = self.model.warmup_months
        in_warmup = current_month <= warmup_end
        # Reset monthly event flag (so monthly rate isn't sticky)
        self.committed_new_offense_this_month = False

        # ======================================================
        # STUDY PERIOD GUARD (ABSORBING EVENT)
        # ======================================================
        # During the study period, once an agent has recidivated,
        # they must be removed from the risk process.
        if not in_warmup and getattr(self, "recidivated_agent", False):
            return

        # ======================================================
        # QUARTERLY RECIDIVISM DRAW (COMMON TO BOTH PERIODS)
        # ======================================================
        # PHASE-AWARE RECIDIVISM DRAW
        # Supervision agents draw from evaluate_recidivism just like Free agents.
        # The supervision_monitoring_intensity term in evaluate_recidivism (step 5)
        # adds a log-odds boost for supervised agents, producing elevated rearrest
        # probability during supervision without a separate parallel pipeline.
        # Option B — dual pipeline, phase-exclusive
        if phase == "Supervision":
            # ── Draw 1: Violation/revocation pipeline ─────────────────────────────
            # Detection-mediated rearrests — violations and revocations that
            # result in a formal arrest. Conversion probabilities grounded in:
            #   Durose et al. (2014, NCJ 244205): revocation → arrest = 0.30
            #   Skeem et al. (2014): violation → arrest = 0.12
            recidivated = getattr(self, "rearrest_quarterly", False)
            self.rearrest_quarterly = False

            # ── Draw 2: New criminal arrest channel ───────────────────────────────
            # Direct new offenses that produce arrest records without going through
            # the supervision violation pathway. Approximately 40% of supervision-era
            # rearrests are new criminal arrests not preceded by a detected violation
            # (Durose et al., 2014; Petersilia, 2003).
            # Only fires if Draw 1 did not already produce a rearrest this quarter.
            if not recidivated:
                recidivated = self.evaluate_recidivism(phase="Supervision")

        elif phase == "Free":
            # BJS baseline is sole mechanism after supervision
            recidivated = self.evaluate_recidivism(phase="Free")
        else:
            return

        if not recidivated:
            return
        # ======================================================
        # COMMON EVENT METADATA
        # ======================================================
        self.committed_new_offense_this_month = True
        self.rearrest_month = current_month

        # ======================================================
        # WARM-UP PERIOD (NON-ABSORBING)
        # ======================================================
        if in_warmup:
            # Warm-up rearrests don't permanently mark the agent or map to study years.
            # Send agent back to Trial; reset community clock so next community spell
            # draws from the correct BJS hazard bucket (q1 = 0.0906, years 0-3).
            self.rearrest_year = None
            self.recidivated_agent = False
            self.justice_state = "Trial"
            self.justice_state_changed = True
            self.months_in_state = 0
            self.community_months_at_risk = 0
            return

        # ======================================================
        # STUDY PERIOD (ABSORBING)
        # ======================================================
        self.recidivated_agent = True
        self.exited_due_to_recidivism = True

        # Year of first arrest, in range [1, 9], from community-months-at-risk.
        # Used by validator charts and by the rearrest_{n}_yrs window flags below.
        self.rearrest_year = max(1, min(9, math.ceil(self.community_months_at_risk / 12)))

        # Window flags — rearrest_{1..9}_yrs is True if first arrest fell within N years.
        # Equivalent to (rearrest_year <= N); kept as individual flags for backward
        # compatibility with calculate_flag_rate() and downstream analysis scripts.
        for window in range(1, 10):
            setattr(self, f"rearrest_{window}_yrs", self.rearrest_year <= window)
        

        

    # ─────────────────────────────────────────────
    # ⚖️ Trial Phase
    # ─────────────────────────────────────────────
    def handle_trial(self):
        """
        Handles agent decision-making during the Trial phase.
        Transitions to Free, Supervision, or Prison based on risk-adjusted probabilities,
        informed by recidivism theory and BJS empirical findings.
        """

        try:
            # ─── 1. Wait until trial threshold is met ───
            if self.months_in_state >= self.trial_wait_threshold:

                # ─── 2. Compute adjusted risk score for trial phase ───
                risk = self.compute_adjusted_risk_score("Trial")

                # NEW:
                # Note: trial_bias_shift is 0.0 unless bias_scope == "all_channels".
                # Under "supervision_only", _apply_bias("trial") returns 0.0 unconditionally,
                # keeping trial outcomes group-neutral (fair-baseline behavior).
                # Source: Schlesinger (2005). Justice Quarterly, 22(2) — trial disparity
                #         is the "all_channels" hypothesis, not the default scope.
                cal = getattr(self.model, "calibration", {}) or {}
                trial_bias_shift = self._apply_bias("trial")

                

                # ─── 3. Recidivism Theory Integration ───
                # Based on BJS studies, recidivism risk is highest within 3 years post-release,
                # and varies by offense type, prior history, and supervision exposure.
                # Diversion and release are less likely for high-risk agents; conviction more likely.
                # See: https://bjs.ojp.gov/recidivism-program 

                # ── Trial outcome probabilities ───────────────────────────────────────
                # Anchored to BJS prosecutorial statistics:
                #
                # RELEASE (dismissal / acquittal / nolle prosequi):
                #   BJS "Felony Defendants in Large Urban Counties" (NCJ 243777, 2009)
                #   reports ~24% of felony arrests result in dismissal or acquittal.
                #   Modelled here as base 0.24 deflated by risk (higher-risk agents
                #   are less likely to have charges dropped), with a floor of 0.05
                #   (even very high-risk agents retain a small dismissal probability).
                #   Source: Cohen & Kyckelhahn (2010). BJS NCJ 228944, Table 9.
                #
                # DIVERSION (probation / supervision without incarceration):
                #   BJS reports ~31% of convicted felons receive a probation-only
                #   sentence (non-incarceration). Treating this as the diversion
                #   pathway gives a base of ~0.31, again risk-deflated.
                #   Source: Rosenmerkel, S., Durose, M. & Farole, D. (2009).
                #           BJS NCJ 226846, Table 1.2.
                #
                # The remaining probability mass (~45% at average risk) routes to Prison,
                # consistent with BJS finding ~45% of convicted felons receive
                # an incarceration sentence.
                #
                # Note: trial_bias_shift is 0.0 unless bias_scope includes "all"
                # pathways (Phase 2 experimental mode only).
                #
                # STRUCTURAL SIMPLIFICATION — risk deflation functional form:
                # The linear `base - base * risk` deflation is a modelling choice.
                # No published study directly reports how dismissal/diversion rates
                # vary continuously with a composite NIJ risk score. The linear form
                # is the simplest monotone function that (a) keeps base rates at the
                # BJS-cited values when risk = 0, (b) reduces them proportionally as
                # risk rises, and (c) hits the floor at risk = 1. It is acknowledged
                # as a structural simplification in the dissertation (Chapter 4, §4.3).
                
                #release_prob   = max(0.05, 0.24 - 0.24 * risk + trial_bias_shift)
                #diversion_prob = max(0.05, 0.31 - 0.31 * risk + trial_bias_shift)
                # CORRECT — use risk (already computed) in the probability formulas
                release_prob   = max(0.05, 0.24 * (1 - risk**1.5) + trial_bias_shift)
                diversion_prob = max(0.05, 0.31 * (1 - risk**1.5) + trial_bias_shift)
                
  

                # ── Sentence and supervision terms ────────────────────────────────────
                # Both are computed here for all agents reaching sentencing, but
                # Supervision_Term is only acted upon if new_state == "Supervision",
                # and Prison_Years only if new_state == "Prison".
                # Agents diverted or released carry these attributes as latent values
                # that have no behavioural effect unless the agent re-enters the
                # trial phase in a subsequent cycle.
                self.Supervision_Term = self.derive_supervision_duration()
                self.Prison_Years = self.assign_sentence_length()

                # ─── 4. Draw random outcome ───
                new_state = None
                r = self.model.np_rng.random()
                if r < release_prob:
                    # 🎯 Agent is released (dismissed or acquitted)
                    new_state = "Free"
                    self.is_supervised = False
                    self.Supervision_Term = 0
                    #self.justice_state = "Free"
                    #self.handle_free()
                    # self.transition_log.append((self.model.current_month, "Trial→Free", round(risk, 3)))

                elif r < release_prob + diversion_prob:
                    # ~78% of diversions also go to supervision
                    if self.model.np_rng.random() < 0.78:
                        new_state = "Supervision"
                        self.is_supervised = True
                    else:
                        new_state = "Free"
                        self.is_supervised = False
                        self.Supervision_Term = 0
                else:
                    # 🎯 Agent is convicted and sent to prison
                    new_state = "Prison"
                    # Reset sentence counter so warm-up re-entries don't carry over
                    # accumulated months from prior stints (which would cause
                    # sentence_complete to fire immediately if the new sentence is
                    # shorter than the previously accumulated time_served_months).
                    self.time_served_months = 0

                # ─── 5. Update justice state and reset timer ───
                # Set Age_at_Release once for agents released or diverted from Trial
                if new_state in ("Free", "Supervision"):
                    age_at_entry = getattr(self, "Age_at_Entry", 18)
                    years_in_system = (self.model.current_month - self.agent_entry_month) // 12
                    self.Age_at_Release = age_at_entry + years_in_system

                self.justice_state_changed = (new_state != self.justice_state)
                self.justice_state = new_state
                self.months_in_state = 0
        except Exception as e:
                print(f"Agent {self.unique_id} has error during Trial transition: {e}")
    # ─────────────────────────────────────────────
    # ⛓️ Prison Phase
    # ─────────────────────────────────────────────


    def _update_conviction_history(self):
        offense = self.offense

        # Ensure all required attributes exist
        required_attrs = [
            "Prior_Conviction_Episodes_Felony", "Prior_Arrest_Episodes_Felony",
            "Prior_Conviction_Episodes_Drug", "Prior_Arrest_Episodes_Drug",
            "Prior_Conviction_Episodes_Property", "Prior_Arrest_Episodes_Property",
            "Prior_Conviction_Episodes_Misd", "Prior_Arrest_Episodes_Misd",
            "Prior_Conviction_Episodes_Violent", "Prior_Arrest_Episodes_Violent",
                ]
        for attr in required_attrs:
            if not hasattr(self, attr):
                setattr(self, attr, 0)

        # Update based on offense type
        if offense in ["Violent/Non-Sex"]:
            if self.Prior_Conviction_Episodes_Felony < 5:
                self.Prior_Arrest_Episodes_Felony += 1
                self.Prior_Conviction_Episodes_Felony += 1
        elif offense in ["Violent-SexOffender"]:
            if self.Prior_Conviction_Episodes_Violent < 5:
                self.Prior_Arrest_Episodes_Violent += 1
                self.Prior_Conviction_Episodes_Violent += 1
                 #print("Updated Felony Convictions:", self.Prior_Conviction_Episodes_Felony)
        elif offense == "Drug":
            if self.Prior_Conviction_Episodes_Drug < 5:
                self.Prior_Arrest_Episodes_Drug += 1
                self.Prior_Conviction_Episodes_Drug += 1
                #print("Updated Drug Convictions:", self.Prior_Conviction_Episodes_Drug)
        elif offense == "Property":
            if self.Prior_Conviction_Episodes_Property < 5:
                self.Prior_Arrest_Episodes_Property += 1
                self.Prior_Conviction_Episodes_Property += 1
                #print("Updated Property Convictions:", self.Prior_Conviction_Episodes_Property)

        elif offense == "Other(PublicOrder)":
            if self.Prior_Conviction_Episodes_Misd < 5:
                self.Prior_Arrest_Episodes_Misd += 1
                self.Prior_Conviction_Episodes_Misd += 1
                #print("Updated Misdemeanor Convictions:", self.Prior_Conviction_Episodes_Misd)
                #print("Updated Violent Convictions:", self.Prior_Conviction_Episodes_Violent)

    def _update_correctional_education(self):
        # Reference list of education levels
        education_levels = ["Less than HighSchool Diploma", "High School Diploma", "College"]
        
        current_level = getattr(self, "Education_Level", education_levels[0])
        current_index = education_levels.index(current_level)

        # Upgrade logic only if not already at highest level
        if current_index < len(education_levels) - 1:
            if current_level == "Less than HighSchool Diploma":
                upgrade_prob = 0.26  # GED attainment
            elif current_level == "High School Diploma":
                upgrade_prob = 0.05  # College-level participation
            else:
                upgrade_prob = 0.0

            if self.model.np_rng.random() < upgrade_prob:
                self.Education_Level = education_levels[current_index + 1]



    def handle_prison(self):
        """
        Handles prison-phase progression, release logic, and transition to supervision or freedom.
        Updates employment status, conviction history, and justice state.
        """
        try:
            # ─── 1. Advance Sentence ───
            self.time_served_months += 1

            # ─── 2. Reset Employment ───
            self.Percent_Days_Employed = 0

            # ─── 3a. Reset peer influence boost at Prison entry ───
            # Ensures each Prison spell computes a fresh boost from current cell
            # neighbors. Only reset on the first month (time_served == 1) to avoid
            # wiping the boost mid-sentence while peer influence is still active.
            if self.time_served_months == 1:
                self.peer_influence_boost = 0.0

            # ─── 3. Record Rearrest Month ───
            # Arrest/Rearrest is recorded during Trial stage

            # ─── 4. Check for Release ───
            sentence_complete = self.time_served_months >= self.Prison_Years
            # ── Gang affiliation dynamics during incarceration ──────────────────────
            # Two processes operate monthly:
            #
            # (A) RECRUITMENT (non-affiliated agents):
            #   Pyrooz, D.C. & Sweeten, G. (2015). Gang membership between ages 5 and 17.
            #   Journal of Adolescent Health, 56(4), 414-419.
            #   Annual joining rate ~2-3% among justice-involved individuals.
            #   Monthly probability: 1-(1-0.025)^(1/12) ≈ 0.0021. A conservative
            #   0.003/month is used to reflect elevated incarceration-environment exposure.
            #   Decker, S.H. & Pyrooz, D.C. (2011). Leaving the gang. Justice Quarterly.
            #
            # (B) DESISTANCE (affiliated agents):
            #   Pyrooz & Sweeten (2015) report ~27% annual leaving rate for incarcerated
            #   gang members. Monthly desistance: 1-(1-0.27)^(1/12) ≈ 0.026/month.
            #
            # The previous code ran `if Gang_Affiliated: p(keep) = 0.15`,
            # which caused 85% monthly affiliation loss — wiping gang membership
            # almost entirely within 6 months in prison, contrary to research showing
            # gang ties typically persist or intensify during incarceration.
            if not getattr(self, "Gang_Affiliated", False):
                # (A) Non-affiliated: monthly recruitment
                if self.model.np_rng.random() < 0.003:
                    self.Gang_Affiliated = True
            else:
                # (B) Affiliated: monthly desistance (~27% annual)
                if self.model.np_rng.random() < 0.026:
                    self.Gang_Affiliated = False

            if sentence_complete:
                # Update conviction history and release month
                self._update_conviction_history()
                # Update correctional education attainment
                self._update_correctional_education()
                self.release_month = self.model.current_month
                self.months_in_state = 0
                # Set Age_at_Release once at the moment of release (static snapshot)
                age_at_entry = getattr(self, "Age_at_Entry", 18)
                years_in_system = (self.model.current_month - self.agent_entry_month) // 12
                self.Age_at_Release = age_at_entry + years_in_system
                # Update the risk score based on life events in prison
                self.normalized_score = self.compute_adjusted_risk_score("Prison")

                # ── Post-release supervision assignment ──────────────────────────────────
                # Base rate: 78% of releases go to mandatory supervision.
                # Source: BJS Survey of State Prison Releases (2018, NCJ 252614, Table 6):
                #   83% state; ~70% federal (BJS NCJ 251461). Blended to 0.78 to reflect
                #   the 20–25% unconditional release target for the mixed release cohort.
                #
                # Bias channel ("all_channels" scope only):
                #   _apply_bias("supervision_assign") adds a group-differentiated log-odds
                #   shift. Under "supervision_only" scope this returns 0.0, keeping
                #   supervision assignment group-neutral.
                #   Source: Hartney, C. & Vuong, L. (2009). Created Equal: Racial and
                #           Ethnic Disparities in the US Criminal Justice System.
                #           National Council on Crime and Delinquency.
                SUPERVISION_RATE = 0.78
                #sup_log_odds = self.logit(base_sup_rate)
                sup_log_odds = self.logit(SUPERVISION_RATE) + self._apply_bias("supervision_assign")
                supervision_required = (
                    self.Supervision_Term > 0
                    and self.model.np_rng.random() < self.safe_sigmoid(sup_log_odds)
                    )
                # Store supervision status as agent attribute — used for SMI independence check
                self.is_supervised = supervision_required

                # Clear Prison-phase peer influence boost on release.
                # Lasting peer effects are already captured through Gang_Affiliated
                # dynamics (agents join/leave gangs during incarceration). Carrying
                # the raw additive boost indefinitely into Free/Supervision would
                # permanently inflate post-release risk with no empirical basis for
                # a non-decaying scarring effect of this magnitude.
                self.peer_influence_boost = 0.0

                if supervision_required:
                    # Transition to supervision
                    self.justice_state = "Supervision"
                    self.justice_state_changed = True
                else:
                    # Unconditional release — no supervision monitoring applies
                    self.Supervision_Term = 0   # zero out so SMI gate works correctly
                    self.justice_state = "Free"
                    self.justice_state_changed = True
        except Exception as e:
                print(f"Agent {self.unique_id} has error during Prison transition: {e}")

    # ─────────────────────────────────────────────
    # 🧭 Supervision Phase
    # ─────────────────────────────────────────────

    def get_effective_supervision_intensity(self):
        cal = getattr(self.model, "calibration", {}) or {}

        calibrated_smi = float(
            cal.get("Supervision_Monitoring_Intensity", 1.0)
        )

        tier = self.get_pcra_tier()

        multipliers = getattr(
            self.model,
            "smi_tier_multipliers",
            {
                "Low": 1.0,
                "LowModerate": 1.0,
                "Moderate": 1.0,
                "High": 1.0,
            }
        )

        tier_multiplier = float(
            multipliers.get(tier, 1.0)
        )

        return calibrated_smi * tier_multiplier

    def handle_supervision_and_programs(self, n_programs=4):
        """
        Simulate yearly supervision and program behaviors for an agent.

        Empirical grounding:
        - Prior revocations: ~20% convert to BJS-style rearrest
        (Durose et al., 2014, 2018; Pew Charitable Trusts, 2019)
        - Technical violations detected: ~10–15% convert to BJS-style rearrest
        (Skeem et al., 2014; Wodahl et al., 2011; Austin et al., 2000–2010)

        Includes:
        - Prior revocations (population mean ~10%)
        - Technical violations (population mean ~45%) → converted into BJS-style rearrest events
        - Program attendance per session (population mean ~45%)
        - Unexcused absences tied to attended sessions (~20–40% of attendees)
        - Supervision intensity modulates detection probability (global + group-differentiated)

        Returns
        -------
        revoked : bool
            True if agent is revoked this quarter.
        """

        # Ensure counters exist
        if not hasattr(self, "Program_Attendances"):
            self.Program_Attendances = 0
        if not hasattr(self, "Program_UnexcusedAbsences"):
            self.Program_UnexcusedAbsences = 0

        revoked = False
        rearrest_this_quarter = False

        # ─── 1. Prior Supervision Revocation ───
        # Base population mean ~10% revocation; elevated +5% for MH/SA or gang-affiliated
        strength_revocation = 6
        mean_percent_revocation = 10  # population-level mean
        if getattr(self, "Condition_MH_SA", False) or getattr(self, "Gang_Affiliated", False):
            mean_percent_revocation += 5  # elevated risk

        # Ensure Beta parameters > 0
        a_rev = max((mean_percent_revocation / 100) * strength_revocation, 0.01)
        b_rev = max(((100 - mean_percent_revocation) / 100) * strength_revocation, 0.01)
        p_revocation = self.model.np_rng.beta(a_rev, b_rev)
        self.Prior_Revocations_Supervision = self.model.np_rng.random() < p_revocation

        # Only a fraction of revocations result in BJS-style rearrest
        # empirical: 20% of revocations → rearrest (Durose et al., 2014, 2018; Pew, 2019)
        # ── Conversion: revocation → BJS-style rearrest ───────────────────────
        # UPDATED: 0.20 → 0.45
        # Original source (Durose et al., 2014; Pew, 2019) reflects federal
        # probation population. BJS NCJ 250975 (Alper et al., 2018) reports
        # prison-release populations have substantially higher rearrest rates.
        # Back-calculation from quarterly BJS hazard (q1 ≈ 0.0906) against
        # observed violation/revocation rates gives conversion ≈ 0.38-0.45.
        # 0.45 is the upper end, consistent with the higher-risk prison-release
        # cohort relative to federal probationers.
        if self.Prior_Revocations_Supervision:
            revoked = True
            conversion_prob = 0.30   # Durose et al. (2014, NCJ 244205)
            rearrest_this_quarter = self.model.np_rng.random() < conversion_prob

        # ─── 2. Technical Violations ─────────────────────────────────────────────
        # Base rate 45%: Pew Charitable Trusts (2019). One in Five. Technical violations.
        strength_violation = 6
        mean_percent_violation = 45

        # ── Violation rate elevation +10 pp ───────────────────────────────────────
        # Gang affiliation and unemployment are consistently associated with higher
        # technical violation rates in supervision research:
        #   Skeem, J. et al. (2014). Psychological Services, 11(3): gang-involved
        #   and unemployed supervisees face higher violation detection rates.
        #   Wodahl, E.J. et al. (2011). Journal of Criminal Justice, 39(6): low
        #   employment is among the strongest predictors of technical violations.
        # The +10 pp shift is a calibration-range choice — no study directly
        # reports a percentage-point increment for these risk factors in isolation.
        # It represents roughly a 22% relative increase (45% → 55%) consistent
        # with the elevated odds ratios reported in Skeem et al. (2014).
        # Note: this parameter affects violation occurrence, not detection.
        # Detection bias is modelled separately via the log-odds mechanism below.

        # ── Continuous employment-related violation adjustment ────────────────
        # Replace the former hard <30% employment threshold with a smooth
        # employment-deficit function.
        #
        # Percent_Days_Employed is bounded to the model's 0–40% range:
        #   0% employed  -> +10.0 pp violation risk
        #  10% employed ->  +7.5 pp
        #  20% employed ->  +5.0 pp
        #  30% employed ->  +2.5 pp
        #  40% employed ->  +0.0 pp
        #
        # Gang affiliation retains the original +10 pp elevation. Using max()
        # preserves the original maximum +10 pp adjustment when either risk
        # factor is present, rather than stacking them to +20 pp.

        employment = float(
            np.clip(
                getattr(self, "Percent_Days_Employed", 0.0),
                0.0,
                40.0
            )
        )
        employment_deficit = 1.0 - (employment / 40.0)
        employment_violation_penalty = 10.0 * employment_deficit
        gang_violation_penalty = 10.0 if getattr(self, "Gang_Affiliated", False) else 0.0

        mean_percent_violation += max(
            employment_violation_penalty,
            gang_violation_penalty
        )

        a_viol = max((mean_percent_violation / 100) * strength_violation, 0.01)
        b_viol = max(((100 - mean_percent_violation) / 100) * strength_violation, 0.01)
        
        # A) Occurrence (unbiased)
        p_violation_occur = self.model.np_rng.beta(a_viol, b_viol)
        violation_occurs = self.model.np_rng.random() < p_violation_occur

        # ── Detection probability: global intensity + group-differentiated bias ──
        cal            = getattr(self.model, "calibration", {}) or {}

        # baseline detection rate (tunable); keep modest
        # Urban Institute / PJI research on supervision — officers typically conduct
        # 1–4 check-ins per month, violations discovered per check-in run roughly
        # 20–40% depending on supervision intensity level — midpoint = 0.30
        p_detect_base = float(cal.get("tech_detect_base", 0.30))
        p_detect_base = max(1e-6, min(1 - 1e-6, p_detect_base))

        # ── Global supervision monitoring intensity (time-decaying) ───────────────
        # Supervision_Monitoring_Intensity scales the baseline detection rate for
        # all agents. Decays over time as supervision becomes less intensive in
        # later years, mirroring the desistance curve in the BJS hazard blocks.
        # After_3Y and After_6Y multipliers are shared with evaluate_recidivism
        # so both pathways use the same time-varying intensity curve.
        quarters_at_risk = self.community_months_at_risk // 3
        #global_intensity = float(cal.get("Supervision_Monitoring_Intensity", 1.0))
        global_intensity = float(self.get_effective_supervision_intensity())
        # Default 1.0 → log(1.0) = 0 → no effect when key is absent from calibration dict.
        # Set > 1.0 to increase detection intensity; < 1.0 to decrease it.
        if quarters_at_risk >= 24:
            global_intensity *= float(cal.get("Supervision_Monitoring_Decay_After_6Y", 0.20))
        elif quarters_at_risk >= 12:
            global_intensity *= float(cal.get("Supervision_Monitoring_Decay_After_3Y", 0.35))
        
        # ── Counterfactual supervision burden ─────────────────────
        if getattr(self, "is_supervised", True):
            self.cumulative_supervision_exposure += float(global_intensity)
            self.supervision_exposure_events += 1

        # ── Log-odds assembly ─────────────────────────────────────────────────────
        # Three additive components in log-odds space:
        #   (1) logit(p_detect_base)      — baseline detection rate (Urban Institute)
        #   (2) log(global_intensity)     — global monitoring intensity scalar,
        #                                   time-decaying after 3 and 6 years
        #                                   (Petersilia, 2003; Taxman, 2012)
        #   (3) _apply_bias("detect")     — group-differentiated detection shift.
        #                                   Active under both bias scopes.
        #                                   Returns 0.0 when bias_factor == 0.0
        #                                   (fair-baseline run) or when agent is
        #                                   not supervised.
        #                                   Source: Skeem et al. (2014).
        #                                   Psychological Services, 11(3).
        if getattr(self, "is_supervised", True):
            log_odds_detect = (
                                self.logit(p_detect_base)
                                + math.log(max(global_intensity, 1e-6))
                                + self._apply_bias("detect")
                            )
        else:
            # Unsupervised — no detection pipeline applies at all
            # This branch should never be reached in practice because unsupervised
            # agents have Supervision_Term = 0 and never enter handle_supervision.
            # Guard included for defensive correctness.
            log_odds_detect = self.logit(p_detect_base)

        p_detect = self.safe_sigmoid(log_odds_detect)

        violation_detected = violation_occurs and (self.model.np_rng.random() < p_detect)
        self.Violations_Technical = violation_detected

        # Only a fraction of detected violations convert to BJS-style rearrest
        # empirical: 10% of technical violations → rearrest (Skeem et al., 2014;
        # Wodahl et al., 2011)
        # ── Conversion: detected violation → BJS-style rearrest ──────────────
        # The revocation-to-rearrest conversion probability was set at 0.50, 
        # derived via back-calculation from the BJS quarterly rearrest hazard 
        # (q₁ = 0.0906; Alper et al., 2018) and confirmed within an empirically 
        # defensible range. The original value of 0.20, sourced from Durose et al. (2014) 
        # and Pew Charitable Trusts (2019), reflects federal probation populations 
        # whose 5-year rearrest rates are approximately 32 percentage points lower 
        # than the state prison-release cohort modelled here (Alper et al., 2018). 
        # Research on high-risk parole populations documents that revocations 
        # frequently involve underlying new criminal conduct recorded administratively 
        # as technical violations, blurring the boundary between revocation and rearrest 
        # (Petersilia, 2003; Pew, 2011; Travis, 2005; Grattet et al., 2011). 
        # The value of 0.50 sits at the upper end of the back-calculated range of 0.38–0.50,
        # consistent with this higher-risk population, and was confirmed by reproducing the
        #  BJS 3-year cumulative rearrest rate of 68% (MAE = 0.002) in the calibrated model."

        if violation_detected:
            conversion_prob = 0.12 # Skeem et al. (2014); Wodahl et al. (2011)
            rearrest_this_quarter |= self.model.np_rng.random() < conversion_prob

        # ─── 3. Program Attendance & Unexcused Absences ───
        strength_attendance = 6
        if getattr(self, "Condition_MH_SA", False) or getattr(self, "Condition_Cog_Ed", False) or getattr(self, "Condition_Other", False):
            mean_percent_attend = 40
        else:
            mean_percent_attend = 45
        
        # ============================================================
        # COUNTERFACTUAL PROGRAM INTERVENTION
        # Active only during study period
        # Examples:
        #
        # Baseline 45%
        # +20% intervention -> 45 * 1.20 = 54%
        # +50% intervention -> 45 * 1.50 = 67.5%
        #
        # -20% intervention -> 45 * 0.80 = 36%
        # ============================================================

        if (
                self.model.current_month > self.model.warmup_months
                and getattr(self, "study_eligible_agent", False)
            ):

            program_cf = float(
                getattr(self.model, "program_cf", 0.0)
            )

            mean_percent_attend *= (
                1.0 + program_cf
            )

        mean_percent_attend = float(
            np.clip(
                mean_percent_attend,
                0.1,
                99.9
            )
        )

        a_att = max((mean_percent_attend / 100) * strength_attendance, 0.01)
        b_att = max(((100 - mean_percent_attend) / 100) * strength_attendance, 0.01)
        p_attend = self.model.np_rng.beta(a_att, b_att)

        attended_sessions = 0
        unexcused_absences = 0
        for _ in range(n_programs):
            if self.model.np_rng.random() < p_attend:
                attended_sessions += 1

                # Per-session unexcused absence probability
                # Base 0.30: midpoint of Pew (2019) / Urban Institute (2019) range (20-40%).
                base_absence_prob = 0.3

                # ── Absence elevation +0.10 ───────────────────────────────────────
                # Gang affiliation and unemployment are associated with programme
                # non-compliance in the reentry literature:
                #   Aos et al. (2006). WSIPP: gang-involved participants show lower
                #   attendance completion rates across correctional programmes.
                #   Brewster (2001). Journal of Drug Issues 31(1): unemployment
                #   predicts treatment non-completion among supervisees.
                # +0.10 (33% relative increase on base 0.30) is a calibration-range
                # choice; no study directly quantifies this increment in isolation.
                # Impact is low-stakes: programme attendance is a risk-score input
                # feature with NIJ weight −0.38 (attendances) and +0.11 (absences),
                # so this parameter influences risk scores only indirectly.
                # ── Continuous employment-related absence adjustment ────────────
                employment = float(
                    np.clip(
                        getattr(self, "Percent_Days_Employed", 0.0),
                        0.0,
                        40.0
                    )
                )

                employment_deficit = 1.0 - (employment / 40.0)

                # Maximum employment-related increase = +0.10
                employment_absence_penalty = 0.10 * employment_deficit

                # Preserve original gang-related maximum = +0.10
                gang_absence_penalty = (
                    0.10 if getattr(self, "Gang_Affiliated", False) else 0.0
                )

                # Do not stack the two effects
                base_absence_prob += max(
                    employment_absence_penalty,
                    gang_absence_penalty
                )

                base_absence_prob = min(base_absence_prob, 0.8)

                if self.model.np_rng.random() < base_absence_prob:
                    unexcused_absences += 1

        self.Program_Attendances = attended_sessions
        self.Program_UnexcusedAbsences = unexcused_absences

        # ─── 4. Feed supervision-mediated rearrest to quarterly evaluation ───
        # Detected technical violations and revocations can produce BJS-style
        # rearrest events through their empirically grounded conversion
        # probabilities. These events form the supervision-mediated rearrest
        # pathway, while evaluate_recidivism() separately handles new-crime
        # rearrests using the calibrated BJS hazard
        self.rearrest_quarterly = rearrest_this_quarter

        return revoked

 
    def handle_supervision(self):
        try:
            # ─── 1. Initialize Supervision Start ───
            # months_in_state is incremented in step() before handle_supervision()
            # runs, so the first call always sees months_in_state == 1, not 0.
            if self.months_in_state == 1:
                self.supervision_start_month = self.model.current_month
        
            #months_since_start = self.model.current_month - self.supervision_start_month
            community_months = getattr(self, "community_months_at_risk", 0)
            # ─────────────────────────────────────────
            # 2. Quarterly Checks (run once per quarter)
            # ─────────────────────────────────────────

            # Program/revocation checks use supervision-relative quarter timing
            if community_months > 0 and community_months % 3 == 0:

                # Post-release employment evolves during supervision too
                self.update_post_release_employment()

                # Handle supervision dynamics (programs, revocations, violations)
                revoked = self.handle_supervision_and_programs()
                self.evaluate_quarterly_recidivism("Supervision")

                if revoked and not self.exited_due_to_recidivism:
                    self.justice_state = "Trial"
                    self.months_in_state = 0
                    return


            # ─── 3. Transition to Free if Supervision Ends ───
            if ( self.months_in_state >= self.Supervision_Term) and not self.exited_due_to_recidivism:
                # Align exposure clock to the nearest completed quarter boundary so the
                # first Free-phase draw fires correctly without silently skipping a quarter.
                self.community_months_at_risk = (self.community_months_at_risk // 3) * 3
                self.justice_state = "Free"
                self.justice_state_changed = True
                self.months_in_state = 0
                # Update the risk score based on life  events in supervision
                self.normalized_score = self.compute_adjusted_risk_score("Supervision")
                #self.handle_free()
        except Exception as e:
                print(f"Agent {self.unique_id} has error during Supervision transition: {e}")
                traceback.print_exc()


    # ─────────────────────────────────────────────
    # 🕊️ Free Phase
    # ─────────────────────────────────────────────
    def update_post_release_employment(self):
        """
        Quarterly update for post-release employment status.

        Behavioral logic:
        -----------------
        - National benchmark: ~67% of releasees are employed at least once
        during the 4 years post-release (BJS, 2018).
        - Individual cap: Each agent's total Percent_Days_Employed cannot exceed 40%.
        (i.e., even consistently working agents do not surpass ~40% of days employed.)
        - Employment evolves quarterly, affected by individual-level risk and protective factors.

        Data reference:
        ---------------
        Alper, M., Durose, M. R., & Markman, J. (2018).
        2018 Update on Prisoner Recidivism: A 9-Year Follow-up Period (2005–2014).
        Bureau of Justice Statistics, NCJ 250975.

         Counterfactual parameter
            ------------------------
            model.employment_cf

                0.00  -> calibrated baseline
            +0.10  -> 10% increase in employment opportunity
            +0.20  -> 20% increase
            +0.50  -> 50% increase

            -0.10  -> 10% decrease
            -0.50  -> 50% decrease

            The intervention changes the probability of obtaining employment.
            It does NOT modify the calibrated risk weight for employment.

        """

        # ─────────────────────────────────────────────────────────────
        # INITIALIZATION (First quarter after release)
        # ─────────────────────────────────────────────────────────────
        # Initialize employment attributes if not already set
        if not hasattr(self, "Percent_Days_Employed"):
            self.Percent_Days_Employed = 0.0  # % of total days post-release employed

        if not hasattr(self, "IsEmployed"):
            self.IsEmployed = False # Binary employment status flag

        # Assign whether this agent will EVER gain employment post-release
        # → About 67% of individuals are expected to become employed at least once
        if not hasattr(self, "WillEverBeEmployed"):
            self.WillEverBeEmployed = (
                self.model.np_rng.random() < 0.67
            )

        # ─────────────────────────────────────────────────────────────
        # QUARTERLY UPDATE (For employable agents only)
        # ─────────────────────────────────────────────────────────────
        if getattr(self, "WillEverBeEmployed", False) and self.Percent_Days_Employed < 40.0:
            # ── Employment gain rate — analytically derived ───────────────────────
            # BJS NCJ 250975 (Alper et al., 2018) reports ~67% of releasees gain
            # employment at least once in the 4 years post-release (16 quarters).
            # For 67% of the WillEverBeEmployed cohort to record at least one gain
            # event within 16 quarters, the per-quarter gain probability p satisfies:
            #
            #   1 - (1 - p)^16 = 0.67  →  p = 1 - 0.33^(1/16) ≈ 0.065
            #
            # A value of 0.12 is used rather than the theoretical 0.065 because:
            #   (a) WillEverBeEmployed already screens out 33% of agents upfront;
            #   (b) the gain check stops once Percent_Days_Employed ≥ 40%, so many
            #       agents exhaust the cap before 16 quarters, reducing effective
            #       exposure. A higher per-quarter probability compensates so that
            #       full-cohort employment penetration reaches ~67% by year 4.
            # Sensitivity: ±0.04 around 0.12 shifts population employment
            # penetration by ≈ ±5 pp — within the uncertainty of the BJS estimate.
            # Source: Alper, Durose & Markman (2018). BJS NCJ 250975.
            base_prob = 0.12

            # ── Individual-level employment adjustments ──────────────────────────
            # Adjustments are applied as additive shifts to the base quarterly
            # gain probability. Direction of each effect is empirically supported;
            # specific magnitudes (−0.04, +0.03) were chosen to produce plausible
            # relative employment gaps while keeping all probabilities positive.
            # They are acknowledged as calibration-range choices in the dissertation.

            # ↓ Gang affiliation: Western (2002, Am. Soc. Rev. 67(4)) documents
            #   substantially reduced employment rates for gang-involved releasees.
            #   MH/SA: Draine et al. (2002, Psych. Services 53(10)) report MH/SA
            #   conditions associated with ~30-40% lower employment rates post-release.
            #   A −0.04 quarterly shift (≈ −1 third-quartile employment event per year)
            #   captures this direction conservatively.
            if getattr(self, "Gang_Affiliated", False) or getattr(self, "Condition_MH_SA", False):
                base_prob -= 0.04

            # ↑ Education ≥ HS: Heckman, Stixrud & Urzua (2006, J. Political Economy
            #   114(4)) document education as a robust predictor of post-release
            #   employment. A +0.03 quarterly shift is a conservative representation
            #   of this well-established relationship. The small magnitude reflects
            #   that within the justice-involved population, the education premium
            #   is attenuated by criminal record stigma.
            #   Source: Pager (2003). American Journal of Sociology, 108(5), 937-975.
            if getattr(self, "Education_Level", "") in ["High School Diploma", "College"]:
                base_prob += 0.03

            # ============================================================
            # COUNTERFACTUAL EMPLOYMENT INTERVENTION
            # Active only during study period
            # Example:
            # -0.50 -> 50% fewer employment opportunities
            #  0.00 -> calibrated baseline
            # +0.50 -> 50% more employment opportunities
            # ============================================================

            if (
                    self.model.current_month > self.model.warmup_months
                    and getattr(self, "study_eligible_agent", False)
                ):

                employment_cf = float(
                    getattr(self.model, "employment_cf", 0.0)
                )

                base_prob *= (1.0 + employment_cf)

            base_prob = float(
                np.clip(base_prob, 0.0, 1.0)
            )

            # Stochastic employment gain event for this quarter
            if self.model.np_rng.random() < base_prob:
                # Each successful quarter increases employment days by 5–10%
                gain = self.model.np_rng.uniform(5.0, 10.0)

                # Cap overall employment intensity at 40%
                self.Percent_Days_Employed = min(40.0, self.Percent_Days_Employed + gain)

        # ─────────────────────────────────────────────────────────────
        # STATUS UPDATE (Employment indicator)
        # ─────────────────────────────────────────────────────────────
        # Binary flag indicating whether the agent has been employed at least once
        self.IsEmployed = self.Percent_Days_Employed > 0


    def update_residence_changes(self):
        """
        Quarterly update for agent residence changes.

        Base rate (0.25/yr):
          Visher & Travis (2003). Annual Review of Sociology, 29, 89-113.
          Post-release residential instability is well documented; ~25% annual
          residential change is a conservative midpoint for the released population.
          Roman & Travis (2006). Urban Institute: Where Will I Sleep Tonight?
          confirms high mobility rates in the first two years post-release.

        Elevated rate (0.35/yr) for MH/SA or gang affiliation:
          Direction: MH/SA conditions and gang involvement are consistently
          associated with greater housing instability in reentry research.
            Metraux & Culhane (2004). Social Service Review 78(2): MH/SA
            conditions strongly predict residential instability post-release.
            Roman & Travis (2006): gang-involved releasees face heightened
            housing barriers driving higher mobility.
          Magnitude: +10 pp is a calibration-range choice. No study directly
          reports an annual percentage-point increment for these factors in
          isolation. The 0.35 value (40% relative increase on base) is
          consistent with the elevated odds ratios in Metraux & Culhane (2004)
          and is acknowledged as an approximation in the dissertation.

        Temporal conversion: annual → quarterly using the correct formula
          p_quarter = 1 - (1 - p_annual)^0.25

        Quarterly post-release residential-instability transition.

        residence_cf:
            0.00 = calibrated baseline
            +0.20 = 20% reduction in residential-instability probability
            +0.80 = 80% reduction
            -0.20 = 20% increase in residential-instability probability
            -0.80 = 80% increase

        The counterfactual modifies the probability of a residential transition,
        not the Residence_Changes risk coefficient itself.
        """

        # ------------------------------------------------------------
        # 1. Baseline residential-instability probability
        # ------------------------------------------------------------
        if (
            getattr(self, "Condition_MH_SA", False)
            or getattr(self, "Gang_Affiliated", False)
        ):
            p_annual = 0.35
        else:
            p_annual = 0.25

        # Correct annual -> quarterly probability conversion
        move_prob = 1.0 - (1.0 - p_annual) ** 0.25

        # ------------------------------------------------------------
        # 2. Housing-stability counterfactual
        # ------------------------------------------------------------
        in_study = ( self.model.current_month > self.model.warmup_months and getattr(self, "study_eligible_agent", False) )

        if in_study:
            residence_cf = float( getattr(self.model, "residence_cf", 0.0)  )

            # Positive CF = greater stability = lower instability probability
            # Negative CF = greater instability = higher probability
            move_prob *= (1.0 - residence_cf)

        move_prob = float(np.clip(move_prob, 0.0, 1.0))

        # ------------------------------------------------------------
        # 3. Mechanism tracking
        # ------------------------------------------------------------
        if in_study:
            self.residence_move_opportunities += 1

        # ------------------------------------------------------------
        # 4. Single residential-transition draw
        # ------------------------------------------------------------
        if self.model.np_rng.random() < move_prob:

            if not hasattr(self, "Residence_Changes"):
                self.Residence_Changes = 0

            self.Residence_Changes += 1

            if in_study:
                self.residence_changes_during_study += 1


    def handle_free(self):
        """
        Handles agent behavior during the Free phase.
        Includes initialization, probabilistic life transitions, and quarterly recidivism evaluation.
        To simulate behavioral deterioration during post-release supervision, agents may probabilistically 
        acquire new conditions such as mental health, cognitive, or housing-related challenges. 
        This transition is modeled with a 7% quarterly probability, reflecting empirical findings 
        that behavioral health risks increase substantially
        in the first year post-release (SAMHSA, 2022; Jang et al., 2025; Chamberlain et al., 2019).
        
        """
        try:
            # ─── 1. Initialize Free Phase ───
            # months_in_state is incremented in step() before handle_free() runs,
            # so the first call always sees months_in_state == 1, not 0.
            if self.months_in_state == 1:
                self.free_start_month = self.model.current_month
            
            # State-relative time since entering Free (good for life events)
            self.MonthsCrimeFree = self.model.current_month - self.free_start_month
            
            # every month while in community (warm-up + study), handled at step
            #if self.justice_state in ("Free", "Supervision") and not getattr(self, "recidivated_agent", False):
            #    self.community_months_at_risk = getattr(self, "community_months_at_risk", 0) + 1

            # call monthly; rearrest draw will happen only when exposure hits quarter boundary
            self.evaluate_quarterly_recidivism("Free")   # or "Supervision"

            # ─── 2. Probabilistic Life Transitions ───
            if self.MonthsCrimeFree > 0 and self.MonthsCrimeFree % 3 == 0:
                self.update_post_release_employment()
                        
                # ── Post-release gang recruitment (Free phase) ─────────────────────
                # Pyrooz, D.C. & Sweeten, G. (2015). Gang membership between ages 5 and 17.
                # Journal of Adolescent Health, 56(4), 414-419.
                # Annual post-release gang joining rate ~2-3% for justice-involved adults.
                # Monthly: 1-(1-0.025)^(1/12) ≈ 0.0021. Quarterly: 1-(1-0.025)^(3/12) ≈ 0.0062.
                #
                # Elevated risk for younger releasees (age < 25) and those without a
                # high school diploma, consistent with:
                # Thornberry, T.P. et al. (2003). Gangs and delinquency in developmental
                # perspective. Cambridge University Press. (age and education as risk factors)
                #
                # FIX: Previous value of 6-9% quarterly (~24-34% annually) was 8-12x
                # higher than published estimates. Corrected to 0.62% base quarterly rate
                # (~2.5% annually), with a 50% relative uplift for the elevated-risk group.
                #
                # Only applies to currently non-affiliated agents.
                if not getattr(self, "Gang_Affiliated", False):
                    base_prob = 0.0062   # ~2.5% annual, expressed quarterly
                    if self.Age_at_Release < 25 or self.Education_Level == "Less than HighSchool Diploma":
                        base_prob = 0.0093   # ~3.7% annual for elevated-risk group
                    if self.model.np_rng.random() < base_prob:
                        self.Gang_Affiliated = True
            
            
                # Mental Health Condition Changes
                # ─── Develop New Behavioral Condition ───
                if self.model.np_rng.random() < 0.07:  # 7% chance per quarter
                    possible_conditions = []
                    if not self.Condition_MH_SA:
                        possible_conditions.append("Condition_MH_SA")
                    if not self.Condition_Cog_Ed:
                        possible_conditions.append("Condition_Cog_Ed")
                    if not self.Condition_Other:
                        possible_conditions.append("Condition_Other")
                    if possible_conditions:
                        idx = self.model.np_rng.integers(0, len(possible_conditions))
                        new_condition = possible_conditions[idx]
                        setattr(self, new_condition, True)
            # self.transition_log.append((self.model.current_month, "Free→DevelopedCondition", new_condition))

                # Add dependent if below cap
                # Yearly dependent gain 7% , connverting to quarterly 0.018
                if self.Dependents < 5:
                    if self.model.np_rng.random() < 0.018:
                        self.Dependents += 1

                self.update_residence_changes()

        except Exception as e:
                print(f"Agent {self.unique_id} has error during Free transition: {e}")



    # ─────────────────────────────────────────────
    # 🗺️ Movement Logic
    # ─────────────────────────────────────────────
    def move_agent(self):
        if self.pos:
            try:
                neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
                if neighbors:
                    new_pos = neighbors[self.model.np_rng.integers(0, len(neighbors))]
                    self.model.grid.move_agent(self, new_pos)
                    self.pos = new_pos
            except Exception as e:
                print(f"Agent {self.unique_id} movement error: {e}")
                
    



        
    def get_agent_vars(self):
        """
        Returns a dictionary of agent attributes for export.
        Fully synchronized with generate_synthetic_agent() field names.
        Includes synthetic profile, justice state, risk scores, and recidivism flags.
        """
        return {
            # ───────────── Identification ─────────────
            "ID": self.unique_id,

            # ───────────── Demographics ─────────────
            "Gender": self.Gender,
            "Race": self.Race,
            "Age_at_Entry": self.Age_at_Entry,
            "Age_at_Release": self.Age_at_Release,

            # ───────────── Social Risk Factors ─────────────
            "Gang_Affiliated": self.Gang_Affiliated,
            "Dependents": self.Dependents,
            "Residence_Changes": self.Residence_Changes,
            "Education_Level": self.Education_Level,

            # ───────────── Supervision Profile ─────────────
            "Supervision_Risk_Score": self.Supervision_Risk_Score,
            "Supervision_Level_First": self.Supervision_Level_First,
            "Supervision_Term": self.Supervision_Term,

            # ───────────── Criminal Offense ─────────────
            "offense": self.offense,
            "Prison_Years": self.Prison_Years,
            "Prison_Release_Month": self.release_month,

            # ───────────── Conviction History ─────────────
            "Prior_Conviction_Episodes_Felony": self.Prior_Conviction_Episodes_Felony,
            "Prior_Conviction_Episodes_Misd": self.Prior_Conviction_Episodes_Misd,
            "Prior_Conviction_Episodes_Violent": self.Prior_Conviction_Episodes_Violent,
            "Prior_Conviction_Episodes_Property": self.Prior_Conviction_Episodes_Property,
            "Prior_Conviction_Episodes_Drug": self.Prior_Conviction_Episodes_Drug,

            # ───────────── Revocations ─────────────
            "Prior_Revocations_Supervision": self.Prior_Revocations_Supervision,

            # ───────────── Behavioral Conditions ─────────────
            "Condition_MH_SA": self.Condition_MH_SA,
            "Condition_Cog_Ed": self.Condition_Cog_Ed,
            "Condition_Other": self.Condition_Other,

            # ───────────── Supervision Violations ─────────────
            #"Violations_ElectronicMonitoring": self.Violations_ElectronicMonitoring,
            #"Violations_Instruction": self.Violations_Instruction,
            #"Violations_FailToReport": self.Violations_FailToReport,
            #"Violations_MoveWithoutPermission": self.Violations_MoveWithoutPermission,
            "Violations_Technical":self.Violations_Technical,
            #"Delinquency_Reports": self.Delinquency_Reports,

            # ───────────── Program Participation ─────────────
            "Program_Attendances": self.Program_Attendances,
            "Program_UnexcusedAbsences": self.Program_UnexcusedAbsences,

            # ───────────── Employment Profile ─────────────
            "Percent_Days_Employed": self.Percent_Days_Employed,
            #"Jobs_Per_Year": self.Jobs_Per_Year,
            #"Employment_Exempt": self.Employment_Exempt,

            # ───────────── Justice System State ─────────────
            "justice_state": self.justice_state,
            "months_in_state": self.months_in_state,
            "cohort": self.cohort,
            "entry_month": self.entry_month,
            "study_eligible_agent": self.study_eligible_agent,

            # ───────────── Supervision Counterfactual Metrics ─────────────
            "pcra_tier": self.get_pcra_tier(),

            "effective_supervision_intensity": (
                self.get_effective_supervision_intensity()
                if getattr(self, "is_supervised", False)
                else 0.0
            ),

            "cumulative_supervision_exposure":
                self.cumulative_supervision_exposure,

            "supervision_exposure_events":
                self.supervision_exposure_events,

            "mean_supervision_intensity": (
                self.cumulative_supervision_exposure
                / self.supervision_exposure_events
                if self.supervision_exposure_events > 0
                else 0.0
            ),
            # ───────────── Residence Counterfactual Metrics ─────────────
            "residence_changes_during_study":
                self.residence_changes_during_study,
            "residence_move_opportunities":
                self.residence_move_opportunities,
            "residence_cf":
                float(getattr(self.model, "residence_cf", 0.0)),
            
            "dynamic_reentry_adjustment":
                getattr(self, "dynamic_reentry_adjustment_value", 0.0),

            "dra_employment_effect":
                getattr(self, "dra_employment_effect", 0.0),

            "dra_program_effect":
                getattr(self, "dra_program_effect", 0.0),

            "employment_cf":
                float(getattr(self.model, "employment_cf", 0.0)),

            "program_cf":
                float(getattr(self.model, "program_cf", 0.0)),

            # ───────────── Risk and Recidivism Outcomes ─────────────
            "dynamic_risk_score": self.dynamic_risk_score,
            "recidivated_agent": self.recidivated_agent,
            "rearrest_month": self.rearrest_month,
            "rearrest_1_yrs": self.rearrest_1_yrs,
            "rearrest_2_yrs": self.rearrest_2_yrs,
            "rearrest_3_yrs": self.rearrest_3_yrs,
            "rearrest_4_yrs": self.rearrest_4_yrs,
            "rearrest_5_yrs": self.rearrest_5_yrs,
            "rearrest_6_yrs": self.rearrest_6_yrs,
            "rearrest_7_yrs": self.rearrest_7_yrs,
            "rearrest_8_yrs": self.rearrest_8_yrs,
            "rearrest_9_yrs": self.rearrest_9_yrs
        }