import pandas as pd
import os
import datetime
from collections import defaultdict
import math
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from recidivism_abm.agents import Person
from recidivism_abm.scoring import compute_risk_score, normalize_score
import random
from recidivism_abm.generate_synthetic_agent import generate_synthetic_agent  # Adjust path as needed
from recidivism_abm.config.risk_config  import get_flat_risk_weights,get_peer_influence_config,get_global_calibration_params,get_pcra_tier_targets
from concurrent.futures import ThreadPoolExecutor, as_completed



class RecidivismModel(Model):
    def __init__(self, initial_agents,bias_factor,
                  monthly_intake,
                 warmup_months, study_months,enable_peer_influence,
                 weights=None,peer_config=None,seed=40
                 ,mode = "realistic"
                 ,output_directory='Results_Run1'
                 ,calibration_params=None
                 # Counterfactual intervention parameters
                ,employment_cf=0.0
                ,program_cf=0.0  
                ,residence_cf=0.0
                ,reentry_hazard_cf=0.0 
                ,dynamic_reentry_adjustment=False           
                 ):
        #print("Model initialized")
        super().__init__()

        # 1. Store slider parameters
        self.seed = seed
        self.mode = mode
        self.employment_cf = float(employment_cf)
        self.program_cf = float(program_cf)
        # Housing-stability counterfactual
        self.residence_cf = float(residence_cf)
        self.reentry_hazard_cf = float(reentry_hazard_cf)
        self.dynamic_reentry_adjustment = bool(dynamic_reentry_adjustment)
        self.initial_agents = initial_agents
        self.monthly_intake = monthly_intake
        self.warmup_months = warmup_months
        self.study_months = study_months #max(study_months, warmup_months + 1)
        self.max_months = self.warmup_months + self.study_months
        self.bias_factor = bias_factor
        self.monthly_recidivism_rate = []
        self.phase_score_cache = {}
        self.risk_midpoint = 0
        self.risk_spread = 0
        self._tier_rate_cache = {}   # populated once per step, read by DataCollector
        self.np_rng = np.random.default_rng(self.seed)


        self.bjs_targets = {    1: 0.439, # 43.9% rearrested in 1 year
                                3: 0.684, # 68% rearrested within 3 years
                                6: 0.794, # 77% within 5 years
                                9: 0.834  # 83% within 10 years
                            }
        # ── BJS per-offense cumulative rearrest targets (Stage 3) ──────────────
        # Source: Alper, Durose & Markman (2018). BJS NCJ 250975, Table 7.
        # Used by compute_offense_loss() to calibrate offense_hazard_shift.
        #Table 7 - Cumulative percent of prisoners released in 30 states in 2005
        #who were arrested following release, by most serious
        #commitment offense and type of post-release arrest offense
        self.bjs_offense_targets = {
            "Violent":             {3: 0.622, 6: 0.742, 9: 0.787},   
            "Property":            {3: 0.75, 6: 0.844, 9: 0.878},
            "Drug":                {3: 0.686, 6: 0.798, 9: 0.838},
            "Other(PublicOrder)":  {3: 0.65, 6: 0.769, 9: 0.819},
        }
        # ── PCRA tier-stratified rearrest targets ──────────────────────────────
        self.pcra_tier_targets = get_pcra_tier_targets()
      
        # 🔧 Load peer influence config with checkbox toggle
        self.weights = weights or get_flat_risk_weights()
        self.peer_config = peer_config or get_peer_influence_config()
        self.enable_peer_influence = enable_peer_influence  # use constructor arg, not peer_config["enabled"]
        # Use injected calibration if provided, otherwise fall back to config defaults.
        # CRITICAL: apply_calibration must run BEFORE create_agent() so agents
        # inherit the correct knob values from the start.
        self.calibration = calibration_params if calibration_params is not None \
                        else get_global_calibration_params()
        self.apply_calibration(self.calibration)

        self.peer_influence_weights = self.peer_config
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        # 2. Initialize Mesa components
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width=40, height=10, torus=False)
        self.agent_counter = 0
        self.current_month = 1
        self.running = True
        self.time_series = []

        # 3. Create initial agents
        for _ in range(self.initial_agents):
            self.create_agent()

        # 4. Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Year": lambda m: m.current_year,
                # Existing reporters...
                "Trial": lambda m: sum(1 for a in m.schedule.agents if a.justice_state == "Trial"),
                "Prison": lambda m: sum(1 for a in m.schedule.agents if a.justice_state == "Prison"),
                "Supervision": lambda m: sum(1 for a in m.schedule.agents if a.justice_state == "Supervision"),
                "Free": lambda m: sum(1 for a in m.schedule.agents if a.justice_state == "Free"),
                "CalibrationError_1yr": lambda m: m.calibration_error_by_window(1),
                "CalibrationError_3yr": lambda m: m.calibration_error_by_window(3),
                "CalibrationError_6yr": lambda m: m.calibration_error_by_window(6),
                "CalibrationError_9yr": lambda m: m.calibration_error_by_window(9),
                "CumulativeRecidivismRate": lambda m: m.compute_cumulative_recidivism_rate(),
                "MonthlyRecidivismRate": lambda m: m.compute_monthly_recidivism_rate(),
                #"RecidivismRate_3yr": lambda m: m.percent_study_recidivists_in_window(3),
                #"RecidivismRate_5yr": lambda m: m.percent_study_recidivists_in_window(5),
                #"RecidivismRate_9yr": lambda m: m.percent_study_recidivists_in_window(9),
                "RecidivismRate_1yr": lambda m: m.calculate_flag_rate("rearrest_1_yrs"),
                "RecidivismRate_3yr": lambda m: m.calculate_flag_rate("rearrest_3_yrs"),
                "RecidivismRate_6yr": lambda m: m.calculate_flag_rate("rearrest_6_yrs"),
                "RecidivismRate_9yr": lambda m: m.calculate_flag_rate("rearrest_9_yrs"),
                # 3-year tier rates
                "RecidivismRate_3yr_Low":         lambda m: m.calculate_flag_rate_by_tier("rearrest_3_yrs", "Low"),
                "RecidivismRate_3yr_LowModerate": lambda m: m.calculate_flag_rate_by_tier("rearrest_3_yrs", "LowModerate"),
                "RecidivismRate_3yr_Moderate":    lambda m: m.calculate_flag_rate_by_tier("rearrest_3_yrs", "Moderate"),
                "RecidivismRate_3yr_High":        lambda m: m.calculate_flag_rate_by_tier("rearrest_3_yrs", "High"),

                # 6-year tier rates
                "RecidivismRate_6yr_Low":         lambda m: m.calculate_flag_rate_by_tier("rearrest_6_yrs", "Low"),
                "RecidivismRate_6yr_LowModerate": lambda m: m.calculate_flag_rate_by_tier("rearrest_6_yrs", "LowModerate"),
                "RecidivismRate_6yr_Moderate":    lambda m: m.calculate_flag_rate_by_tier("rearrest_6_yrs", "Moderate"),
                "RecidivismRate_6yr_High":        lambda m: m.calculate_flag_rate_by_tier("rearrest_6_yrs", "High"),

                # 9-year tier rates
                "RecidivismRate_9yr_Low":         lambda m: m.calculate_flag_rate_by_tier("rearrest_9_yrs", "Low"),
                "RecidivismRate_9yr_LowModerate": lambda m: m.calculate_flag_rate_by_tier("rearrest_9_yrs", "LowModerate"),
                "RecidivismRate_9yr_Moderate":    lambda m: m.calculate_flag_rate_by_tier("rearrest_9_yrs", "Moderate"),
                "RecidivismRate_9yr_High":        lambda m: m.calculate_flag_rate_by_tier("rearrest_9_yrs", "High"),
                # ── Tier calibration errors — 3yr ────────────────────────────────────────
                "CalibError_3yr_Low":         lambda m: m.calibration_error_by_tier(3, "Low"),
                "CalibError_3yr_LowModerate": lambda m: m.calibration_error_by_tier(3, "LowModerate"),
                "CalibError_3yr_Moderate":    lambda m: m.calibration_error_by_tier(3, "Moderate"),
                "CalibError_3yr_High":        lambda m: m.calibration_error_by_tier(3, "High"),

                # ── Tier calibration errors — 6yr ────────────────────────────────────────
                "CalibError_6yr_Low":         lambda m: m.calibration_error_by_tier(6, "Low"),
                "CalibError_6yr_LowModerate": lambda m: m.calibration_error_by_tier(6, "LowModerate"),
                "CalibError_6yr_Moderate":    lambda m: m.calibration_error_by_tier(6, "Moderate"),
                "CalibError_6yr_High":        lambda m: m.calibration_error_by_tier(6, "High"),

                # ── Tier calibration errors — 9yr ────────────────────────────────────────
                "CalibError_9yr_Low":         lambda m: m.calibration_error_by_tier(9, "Low"),
                "CalibError_9yr_LowModerate": lambda m: m.calibration_error_by_tier(9, "LowModerate"),
                "CalibError_9yr_Moderate":    lambda m: m.calibration_error_by_tier(9, "Moderate"),
                "CalibError_9yr_High":        lambda m: m.calibration_error_by_tier(9, "High")              

            },
            agent_reporters={
                "JusticeState": lambda a: a.justice_state,
                "Recidivated": lambda a: getattr(a, "recidivated_agent", False),
                "RearrestMonth": lambda a: getattr(a, "rearrest_month", None),
                "rearrest_1_yrs": lambda a: a.rearrest_1_yrs,
                "rearrest_3_yrs": lambda a: a.rearrest_3_yrs,
                "rearrest_6_yrs": lambda a: a.rearrest_6_yrs,
                "rearrest_9_yrs": lambda a: a.rearrest_9_yrs
            }
        )
 
    @property
    def current_year(self):
        return self.current_month // 12
    @property
    def agents(self):
        return self.schedule.agents
        
    def apply_calibration(self, calibration: dict) -> None:
        """
        Apply calibration dict to the model and expose commonly used knobs as attributes.
        This guarantees sensitivity sweeps actually affect dynamics.
        """
        self.calibration = calibration or {}

        self.Risk_Contrast_Strength = float(
            self.calibration.get("Risk_Contrast_Strength", 0.0)
        )
        self.Supervision_Monitoring_Intensity = float(
            self.calibration.get("Supervision_Monitoring_Intensity", 0.35)
        )

        self.Risk_Effect_Decay_After_3Y = float(
            self.calibration.get("Risk_Effect_Decay_After_3Y", 1.0)
        )
        self.Risk_Effect_Decay_After_6Y = float(
            self.calibration.get("Risk_Effect_Decay_After_6Y", 1.0)
        )
        self.Supervision_Monitoring_Decay_After_3Y = float(
            self.calibration.get("Supervision_Monitoring_Decay_After_3Y", 1.0)
        )
        self.Supervision_Monitoring_Decay_After_6Y = float(
            self.calibration.get("Supervision_Monitoring_Decay_After_6Y", 1.0)
        )

        # calibration as the single source of truth for person.py reads.
        if "bias_factor" not in self.calibration or self.calibration["bias_factor"] == 0.0:
            self.calibration["bias_factor"] = float(getattr(self, "bias_factor", 0.0))


        # If risk weights are provided in calibration, use them
        if "risk_weights" in self.calibration and isinstance(self.calibration["risk_weights"], dict):
            self.weights = dict(self.calibration["risk_weights"])
        
        return
    
    def calculate_flag_rate(self, flag_name):
        try:
            years = int(flag_name.split("_")[1].replace("yr", ""))
        except (IndexError, ValueError):
            return None

        # Guard 1: study period hasn't started
        if self.current_month <= self.warmup_months:
            return None

        # Guard 2: wait until the calendar window has elapsed
        # Community-time flags accumulate continuously — readable
        # as soon as warmup + window months have passed
        min_study_months = self.warmup_months + years * 12
        if self.current_month < min_study_months:
            return None

        eligible_agents = [
            a for a in self.schedule.agents
            if getattr(a, "study_eligible_agent", False)
        ]
        if not eligible_agents:
            return 0.0

        flagged = sum(1 for a in eligible_agents if getattr(a, flag_name, False))
        return round(flagged / len(eligible_agents), 3)

    def calculate_flag_rate_by_tier(self, flag_name_or_years, tier: str = None):
        # ── Resolve attribute name ────────────────────────────────────────────
        if isinstance(flag_name_or_years, int):
            attr = f"rearrest_{flag_name_or_years}_yrs"
        else:
            attr = flag_name_or_years

        # ── Guard: neutral values before study period starts ─────────────────
        neutral = {"Low": 0.0, "LowModerate": 0.0, "Moderate": 0.0, "High": 0.0}
        if self.current_month <= self.warmup_months:
            return 0.0 if tier is not None else neutral

        # ── Read from cache if available (DataCollector path) ─────────────────
        cache = getattr(self, "_tier_rate_cache", {})
        if cache:
            if tier is not None:
                return cache.get((attr, tier), 0.0)
            return {
                t: cache.get((attr, t), 0.0)
                for t in ("Low", "LowModerate", "Moderate", "High")
            }

        # ── Full scan fallback (compute_gamma_loss, end-of-run calls) ─────────
        counts = {"Low": 0, "LowModerate": 0, "Moderate": 0, "High": 0}
        totals = {"Low": 0, "LowModerate": 0, "Moderate": 0, "High": 0}

        for agent in self.schedule.agents:
            if not getattr(agent, "study_eligible_agent", False):
                continue
            t = agent.get_pcra_tier()
            totals[t] += 1
            if getattr(agent, attr, False):
                counts[t] += 1

        rates = {
            t: (counts[t] / totals[t]) if totals[t] > 0 else 0.0
            for t in counts
        }

        if tier is not None:
            return rates.get(tier, 0.0)
        return rates

    def calculate_flag_rate_by_offense(self, flag_name_or_years, offense: str = None):
        """
        Per-offense cumulative rearrest rate for a given window.
        Mirrors calculate_flag_rate_by_tier() API.
        """
        if isinstance(flag_name_or_years, int):
            attr = f"rearrest_{flag_name_or_years}_yrs"
        else:
            attr = flag_name_or_years

        offenses = ["Violent", "Drug",
                    "Property", "Other(PublicOrder)"]
        neutral = {o: 0.0 for o in offenses}

        if self.current_month <= self.warmup_months:
            return 0.0 if offense is not None else neutral

        counts = {o: 0 for o in offenses}
        totals = {o: 0 for o in offenses}

        for agent in self.schedule.agents:
            if not getattr(agent, "study_eligible_agent", False):
                continue
            o = getattr(agent, "offense", "Other(PublicOrder)")
            if o not in totals:
                continue
            totals[o] += 1
            if getattr(agent, attr, False):
                counts[o] += 1

        rates = {o: (counts[o] / totals[o]) if totals[o] > 0 else 0.0
                for o in offenses}

        if offense is not None:
            return rates.get(offense, 0.0)
        return rates

    def compute_gamma_loss(self) -> float:
        """
        Computes mean absolute error between simulated tier-stratified rearrest
        rates and PCRA-BJS tier targets across all windows and tiers.

        Used to calibrate Risk_Contrast_Strength (γ):
            γ* = argmin compute_gamma_loss()

        A γ that produces tier rates matching the PCRA-BJS targets at 3, 6, and
        9 years is the identified value — the aggregate BJS constraint alone
        cannot identify γ because tier spread is not observable from averages.

        Returns
        -------
        float  Mean absolute error across all tier × window combinations.
        """
        targets = getattr(self, "pcra_tier_targets", {})
        if not targets:
            return float("nan")

        errors = []
        for window_years in (3, 6, 9):
            simulated = self.calculate_flag_rate_by_tier(window_years)
            for tier, target_rate in targets.items():
                sim_rate = simulated.get(tier, 0.0)
                target   = target_rate[window_years]
                errors.append(abs(sim_rate - target))

        return sum(errors) / len(errors) if errors else float("nan")
    
    def compute_offense_loss(self) -> float:
        """
        Mean absolute error between simulated and BJS per-offense cumulative
        rearrest rates across offense × window cells (5 offenses × 3 windows).

        Used to calibrate offense_hazard_shift in Stage 3. Analogous to
        compute_gamma_loss() but stratified by offense instead of risk tier.
        """
        targets = getattr(self, "bjs_offense_targets", {})
        if not targets:
            return float("nan")

        errors = []
        for window_years in (3, 6, 9):
            simulated = self.calculate_flag_rate_by_offense(window_years)
            for offense, target_rates in targets.items():
                sim_rate = simulated.get(offense, 0.0)
                target   = target_rates[window_years]
                errors.append(abs(sim_rate - target))

        return sum(errors) / len(errors) if errors else float("nan")
    
    def calibration_error_by_tier(self, years: int, tier: str) -> float:
        """
        Absolute error between simulated and PCRA-BJS target
        for a specific window and tier. Used by DataCollector.

        Parameters
        ----------
        years : int   — 3, 6, or 9
        tier  : str   — "Low", "LowModerate", "Moderate", "High"

        Returns
        -------
        float  |simulated_rate − pcra_bjs_target| or None before study starts.
        """
        if self.current_month <= self.warmup_months:
            return None

        targets = getattr(self, "pcra_tier_targets", {})
        if not targets or tier not in targets:
            return None

        # Integer mode → full dict, extract the one tier needed
        simulated   = self.calculate_flag_rate_by_tier(years)
        sim_rate    = simulated.get(tier, 0.0)
        target_rate = targets[tier][years]

        return round(abs(sim_rate - target_rate), 4)

    def _compute_phase_score_cache(self):
        """
        Compute raw risk scores ONCE per month and cache them by justice phase.
        """

        self.phase_score_cache = {
            "Trial": [],
            "Prison": [],
            "Supervision": [],
            "Free": []
        }

        for agent in self.schedule.agents:
            if not hasattr(agent, "justice_state"):
                continue

            # Compute raw score ONCE
            agent.raw_score = compute_risk_score(agent, self.weights) or 0.0

            phase = agent.justice_state
            if phase in self.phase_score_cache:
                self.phase_score_cache[phase].append(agent.raw_score)

 
    # ── Add this method ───────────────────────────────────────────────────────
    def _refresh_tier_rate_cache(self):
        """
        Computes all tier × window rearrest rates in a single agent scan
        and stores them in self._tier_rate_cache.

        Called once per step() before DataCollector.collect(). All 12
        DataCollector tier reporters read from this cache — no repeated scans.

        Cache key format: (attr, tier)
        e.g. ("rearrest_3_yrs", "Low") → 0.214
        """
        # Clear previous month
        self._tier_rate_cache = {}

        if self.current_month <= self.warmup_months:
            return

        attrs = ["rearrest_1_yrs","rearrest_3_yrs", "rearrest_6_yrs", "rearrest_9_yrs"]

        counts = {attr: {"Low": 0, "LowModerate": 0, "Moderate": 0, "High": 0} for attr in attrs}
        totals = {"Low": 0, "LowModerate": 0, "Moderate": 0, "High": 0}

        # ── Single pass over all agents ───────────────────────────────────────
        for agent in self.schedule.agents:
            if not getattr(agent, "study_eligible_agent", False):
                continue
            tier = agent.get_pcra_tier()
            totals[tier] += 1
            for attr in attrs:
                if getattr(agent, attr, False):
                    counts[attr][tier] += 1

        # ── Write to cache ────────────────────────────────────────────────────
        for attr in attrs:
            for tier in ("Low", "LowModerate", "Moderate", "High"):
                total = totals[tier]
                self._tier_rate_cache[(attr, tier)] = (
                    counts[attr][tier] / total if total > 0 else 0.0
                )

    def print_normalized_risk_percentiles(self):
        """
        Print percentiles of normalized (0–1) risk for the warm-up study-eligible cohort.
        Call this right after compute_reference_risk_stats().
        """
        risks = []

        for agent in self.schedule.agents:
            # same cohort used for reference stats
            if getattr(agent, "study_eligible_agent", False) and getattr(agent, "justice_state", None) in ("Free", "Supervision"):
                raw = agent.normalize_risk_absolute()
                risks.append(raw)

        if not risks:
            print("⚠️ No agents available for normalized risk percentile check.")
            return

        p5, p25, p50, p75, p95 = np.percentile(risks, [5, 25, 50, 75, 95])

        print("📊 Normalized Risk Percentiles (Post–Warm-up)")
        print(f"  5th  percentile: {p5:.3f}")
        print(f" 25th percentile: {p25:.3f}")
        print(f" 50th percentile: {p50:.3f}")
        print(f" 75th percentile: {p75:.3f}")
        print(f" 95th percentile: {p95:.3f}")

    # ─────────────────────────────────────────────
    # Risk Reference Statistics (Warm-up Phase)
    # ─────────────────────────────────────────────
    def compute_reference_risk_stats(self):
        """
        Compute reference statistics for risk normalization using the warm-up cohort.

        These values are FIXED after warm-up and used throughout the study period.

        Outputs stored on the model:
        - self.risk_midpoint       : Mean raw risk score (robust to right-skew)
        - self.risk_spread         : IQR-based dispersion (floor=0.5)
        - self.tier_boundary_low   : 25th percentile of normalized scores
        - self.tier_boundary_lowmod: 50th percentile of normalized scores  
        - self.tier_boundary_mod   : 75th percentile of normalized scores
        
        Tier boundaries are derived from the actual reference population
        distribution, guaranteeing exactly 25% per tier regardless of
        score distribution shape. This is required for γ identification
        against PCRA tier targets which assume equal-sized tiers.
        """

        # ── 1. Select reference agents ────────────────────────────────────────
        reference_agents = [
            a for a in self.schedule.agents
            if getattr(a, "study_eligible_agent", False)
            and getattr(a, "justice_state", None) in ("Free", "Supervision")
        ]

        #cal = getattr(self, "calibration", {}) or {}
        #ref_weights = cal.get("baseline_risk_weights", self.weights)

        #raw_scores = []
        #for agent in reference_agents:
            #raw = compute_risk_score(agent, weights=ref_weights) or 0.0
        raw_scores = []
        for agent in reference_agents:
            raw = compute_risk_score(agent, weights=self.weights) or 0.0
            # Clamp extreme outliers — scores outside [-20, 20] are data anomalies
            if -20 <= raw <= 20:
                raw_scores.append(raw)
            agent.raw_score = raw

        # ── 2. Safety fallback ────────────────────────────────────────────────
        if not raw_scores:
            self.risk_midpoint        = 0.0
            self.risk_spread          = 1.0
            self.tier_boundary_low    = 0.25
            self.tier_boundary_lowmod = 0.50
            self.tier_boundary_mod    = 0.75
            return

        raw_scores.sort()

        # ── 3. Midpoint — use mean (robust against right-skew) ────────────────
        # Median caused asymmetric tier compression because the survivor
        # population's right-skewed score distribution placed more mass
        # above the median than below, inflating High tier at expense of Low.
        self.risk_midpoint = float(sum(raw_scores) / len(raw_scores))

        # ── 4. Spread — IQR-based with corrected floor ────────────────────────
        def percentile(sorted_values, pct):
            n = len(sorted_values)
            if n == 0:
                return 0.0
            k = (n - 1) * (pct / 100.0)
            f = int(math.floor(k))
            c = int(math.ceil(k))
            if f == c:
                return sorted_values[f]
            return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

        q25 = percentile(raw_scores, 25)
        q75 = percentile(raw_scores, 75)
        iqr = max(1e-9, q75 - q25)

        # Floor of 0.5 — lower than original 1.0 which was artificially
        # compressing all scores toward 0.5 (IQR≈0.97 → iqr/1.35≈0.72,
        # which the old floor of 1.0 was overriding entirely)
        self.risk_spread = float(max(0.5, iqr / 1.35))

        # ── 5. Population-derived tier boundaries ─────────────────────────────
        # Fixed boundaries (0.25/0.50/0.75) produced severely imbalanced tiers
        # (Low≈7%, High≈27%) because the sigmoid-normalized score distribution
        # does not span [0,1] uniformly — it is bounded below by agent attribute
        # floors accumulated during 144-month warmup cycling.
        #
        # Computing boundaries from actual percentiles of the reference
        # population guarantees exactly 25% per tier, which is required for
        # γ (Risk_Contrast_Strength) identification against PCRA tier targets.
        # PCRA Table 6 (Federal Probation, Sept 2023) assumes roughly equal
        # tier sizes; mismatched tier sizes make γ unidentifiable.
        normalized_scores = []
        for raw in raw_scores:
            z = (raw - self.risk_midpoint) / max(self.risk_spread, 1e-6)
            # Inline safe_sigmoid to avoid agent method dependency here
            if z > 60:
                n_score = 1.0
            elif z < -60:
                n_score = 0.0
            elif z >= 0:
                n_score = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                n_score = ez / (1.0 + ez)
            normalized_scores.append(n_score)

        normalized_scores.sort()
        n = len(normalized_scores)

        self.tier_boundary_low    = float(normalized_scores[n // 4])
        self.tier_boundary_lowmod = float(normalized_scores[n // 2])
        self.tier_boundary_mod    = float(normalized_scores[3 * n // 4])

        # ── 6. Diagnostic print (uncomment to verify) ─────────────────────────
        # print(f"📌 Risk reference stats:")
        # print(f"   midpoint={self.risk_midpoint:.3f}  spread={self.risk_spread:.3f}")
        # print(f"   tier boundaries: "
        #       f"{self.tier_boundary_low:.3f} / "
        #       f"{self.tier_boundary_lowmod:.3f} / "
        #       f"{self.tier_boundary_mod:.3f}")
        

    def percent_study_recidivists_in_window(self, window: int) -> float:
        if self.current_month <= self.warmup_months:
            return None

        # Define the cutoff month for the recidivism window
        cutoff_month = self.warmup_months + window * 12

        # Filter agents eligible for study
        eligible_agents = [
            a for a in self.schedule.agents
            if getattr(a, "study_eligible_agent", False)
        ]
        if not eligible_agents:
            return 0.0

        # Count recidivists who were rearrested within the window
        recidivists = sum(
            1 for a in eligible_agents
            if getattr(a, "recidivated_agent", False)
            and isinstance(getattr(a, "rearrest_month", None), (int, float))
            and self.warmup_months < a.rearrest_month <= cutoff_month
        )

        return round(recidivists / len(eligible_agents), 3)


    def reset_agent_for_study(self,agent):
        # Dynamic risk and outcomes
        #agent.dynamic_risk_score = None
        agent.rearrest_within_3yr = False
        agent.reincarceration_within_3yr = False
        agent.recidivated_agent = False
        agent.rearrest_month = None
        agent.rearrest_year=None
        agent.rearrest_1_yrs = False 
        agent.rearrest_2_yrs=False
        agent.rearrest_3_yrs=False
        agent.rearrest_4_yrs=False
        agent.rearrest_5_yrs = False 
        agent.rearrest_6_yrs=False
        agent.rearrest_7_yrs=False
        agent.rearrest_8_yrs=False
        agent.rearrest_9_yrs=False
        agent.community_months_at_risk = 0
        agent.exited_due_to_recidivism = False

        # Optional: flag reset completion
        agent.reset_done = True      
        # Only reset supervision clock for agents currently in Supervision
        # Agents in Free have no supervision clock to reset
        if agent.justice_state == "Supervision":
            agent.supervision_start_month = self.current_month
        agent.months_in_state = 0

        # FIX: Reset rearrest_quarterly so no stale True value
        # from warmup carries into the first study-period quarter.
        agent.rearrest_quarterly = False
        agent.exited_due_to_recidivism = False  

            # ADD: invalidate cached tier so it recomputes with new boundaries
        if hasattr(agent, "_cached_tier"):
            del agent._cached_tier
        if hasattr(agent, "_cached_tier_month"):
            del agent._cached_tier_month
        
        # At the END of reset_agent_for_study(), after all resets:
        #cal = getattr(self, "calibration", {}) or {}
        #ref_weights = cal.get("baseline_risk_weights", self.weights)
        #agent.frozen_risk_score = compute_risk_score(agent, weights=ref_weights)
        agent.frozen_risk_score = compute_risk_score(agent, weights=self.weights)
        
        return
    
    def _recompute_tier_boundaries_post_reset(self):
        """
        Recompute population-derived tier boundaries from the study cohort
        after reset_agent_for_study() has run for all agents.
        Uses same weights as compute_reference_risk_stats() for consistency.
        """
        study_agents = [
            a for a in self.schedule.agents
            if getattr(a, "study_eligible_agent", False)
        ]
        if not study_agents:
            return

        # Match weights used in compute_reference_risk_stats()
        #cal = getattr(self, "calibration", {}) or {}
        #ref_weights = cal.get("baseline_risk_weights", self.weights)

        normalized_scores = []
        #for agent in study_agents:
        #    raw = compute_risk_score(agent, weights=ref_weights) or 0.0
        for agent in study_agents:
            raw = compute_risk_score(agent, weights=self.weights) or 0.0
            agent.raw_score = raw  # refresh cached raw score
            z = (raw - self.risk_midpoint) / max(self.risk_spread, 1e-6)
            if z > 60:    n_score = 1.0
            elif z < -60: n_score = 0.0
            elif z >= 0:  n_score = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                n_score = ez / (1.0 + ez)
            normalized_scores.append(n_score)

        normalized_scores.sort()
        n = len(normalized_scores)
        #self.tier_boundary_low    = float(normalized_scores[n // 4])
        #self.tier_boundary_lowmod = float(normalized_scores[n // 2])
        #self.tier_boundary_mod    = float(normalized_scores[3 * n // 4])
        self.tier_boundary_low    = float(normalized_scores[int(0.291 * n)])
        self.tier_boundary_lowmod = float(normalized_scores[int((0.291 + 0.356) * n)])
        self.tier_boundary_mod    = float(normalized_scores[int((0.291 + 0.356 + 0.239) * n)])

    
    def _finalize_warmup(self):
        try:

            eligible = [a for a in self.schedule.agents 
                if a.justice_state in ("Free", "Supervision")]

            raw_scores = [compute_risk_score(a, self.weights) for a in eligible]
            '''
            if raw_scores:
                import numpy as np
                print(f"\n📊 WARMUP SURVIVOR DIAGNOSTIC")
                print(f"   N eligible agents: {len(eligible)}")
                print(f"   Raw score median:  {np.median(raw_scores):.3f}")
                print(f"   Raw score mean:    {np.mean(raw_scores):.3f}")
                print(f"   Raw score std:     {np.std(raw_scores):.3f}")
                print(f"   Pct Gang:          {sum(a.Gang_Affiliated for a in eligible)/len(eligible):.1%}")
                print(f"   Pct MH_SA:         {sum(a.Condition_MH_SA for a in eligible)/len(eligible):.1%}")
                print(f"   Mean age:          {sum(a.Age_at_Release for a in eligible)/len(eligible):.1f}")
            '''
            # Mark study-eligible agents
            for agent in self.schedule.agents:
                agent.study_eligible_agent = agent.justice_state in ["Free","Supervision"]
                
            # Compute fixed reference risk stats from warm-up cohort
            self.compute_reference_risk_stats()

            # REMOVE the old tier boundary print that fires here —
            # boundaries will be recomputed from study cohort below
            # print(f"📌 Tier boundaries (population-derived): ...")  ← DELETE THIS

            # Export warm-up cohort
            if getattr(self, "export_csv", True):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(
                    self.output_directory,
                    f"{self.seed}_agents_free_warmupperiod_{timestamp}.csv"
                )
                warmup_agents = [
                    a.get_agent_vars()
                    for a in self.schedule.agents
                    if isinstance(a, Person)
                ]
                pd.DataFrame(warmup_agents).to_csv(filename, index=False)
                #print(f"✅ _finalize_warmup called after month {self.current_month - 1} completed "
                #f"(current_month now={self.current_month}, warmup_months={self.warmup_months})")
                #print(f"✅ Exported {len(warmup_agents)} warm-up agents → {filename}")

            # Purge non-study agents from schedule AND grid
            eligible_agents = [
                a for a in self.schedule.agents
                if getattr(a, "study_eligible_agent", False)
            ]
        
            for agent in self.schedule.agents:
                if not getattr(agent, "study_eligible_agent", False):
                    if agent.pos is not None:
                        self.grid.remove_agent(agent)

            # Create a new scheduler and add only eligible agents
            new_schedule = RandomActivation(self)
            for agent in eligible_agents:
                new_schedule.add(agent)

            # Replace the old schedule
            self.schedule = new_schedule

            # Reset study agents
            for agent in self.schedule.agents:
                self.reset_agent_for_study(agent)

            # ADD: Recompute tier boundaries from study cohort post-reset
            # Must run AFTER reset loop (cache cleared) and AFTER schedule
            # replacement (only study agents remain).
            # Uses same weights as compute_reference_risk_stats() so
            # normalization is consistent.
            self._recompute_tier_boundaries_post_reset()
            #print(f"📌 Tier boundaries (study cohort post-reset): "
            #    f"Low<{self.tier_boundary_low:.3f}  "
            #    f"LowMod<{self.tier_boundary_lowmod:.3f}  "
            #    f"Mod<{self.tier_boundary_mod:.3f}")


        except Exception as e:
            print(f"❌ Warm-up finalization failed: {e}")


    def _export_final_cohort(self):
        # Skip during parallel calibration runs
        if not getattr(self, "export_csv", True):
            return
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.output_directory,
                f"{self.seed}_study_cohort_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
            )
            study_agents = [
                a.get_agent_vars()
                for a in self.schedule.agents
                if isinstance(a, Person) and getattr(a, "study_eligible_agent", False)
            ]
            pd.DataFrame(study_agents).to_csv(filename, index=False)
            #print(f"✅ Exported {len(study_agents)} study agents → {filename}")
        except Exception as e:
            print(f"❌ Final cohort export failed: {e}")

    def _track_justice_states(self):
        state_counts = {
        "Month": self.current_month,
        "trial": 0,
        "prison": 0,
        "supervision": 0,
        "free": 0
        }
        for agent in self.schedule.agents:
            if isinstance(agent, Person) and getattr(agent, "study_eligible_agent", False):
                key = agent.justice_state.lower()
                state_counts[key] = state_counts.get(key, 0) + 1
        self.time_series.append(state_counts)


    def step(self):
        if not self.running:
            return
        try:
            
            # ─── 1. Simulation Completion ───
            if self.current_month >= self.max_months:
                self._export_final_cohort()
                self.running = False
                return

            # ─── 2. Track Justice State Counts ───
            self._track_justice_states()

            # ─── 3. Advance Agents (SHUFFLED ORDER) ───
            self.staggered_intake()

            # Get a shuffled copy of agents using the seeded RNG.
            # This ensures:
            #   - Different activation order every month (no first-mover bias)
            #   - Same seed → same shuffle sequence (reproducible across runs)
            #   - Different seeds → different sequences (genuine variance across replications)
            # NOTE: We do NOT call self.schedule.step() because it would activate
            # ALL agents including non-study agents after warmup. Instead we
            # manually shuffle and filter, giving us full control.
            agents_this_month = list(self.schedule.agents)
            self.np_rng.shuffle(agents_this_month)

            for agent in agents_this_month:
                try:
                    # During study period, skip agents not eligible for study
                    if self.current_month >= self.warmup_months and not getattr(agent, "study_eligible_agent", False):
                        continue
                    # Skip agents who have already exited via recidivism
                    if getattr(agent, "exited_due_to_recidivism", False):
                        continue
                    agent.step()
                except Exception as e:
                    print(f"⚠️ Agent {agent.unique_id} crashed: {e}")
            
            # ─── 4. Compute risk scores ONCE for this month ───
            self._compute_phase_score_cache()
            # ─── 4b. Refresh tier rate cache ONCE for DataCollector ──────────────────
            self._refresh_tier_rate_cache()
            
            # ─── 5. Collect Data ───
            # In step() — only collect full dataset during study period
            # During warmup collect only the lightweight reporters
            if self.current_month <= self.warmup_months:
                # Skip tier reporters during warmup — they always return 0
                pass
            else:
                self.datacollector.collect(self)
            self.current_month += 1
            
            # ─── 6. Warm-up Finalization ───
            if self.current_month == self.warmup_months + 1:
                self._finalize_warmup()
                

        except Exception as e:
            print(f"❌ Step error: {e}")
            self.running = False

    def create_agent(self):
        # Generate synthetic data for this agent
        agent_data = generate_synthetic_agent(self.agent_counter,self.mode)

        # Create agent with synthetic attributes
        agent = Person(self.agent_counter, self, agent_data)
        agent.agent_data = agent_data
        agent.initialize_attributes_from_data()
        # Initialize justice state and cohort
        agent.justice_state = "Trial"
        agent.months_in_state = 0
        agent.entry_month = self.current_month
        agent.cohort = "warmup" if self.current_month < self.warmup_months else "study"

        # Place agent on grid
        x = int(self.np_rng.integers(0, self.grid.width))
        y = int(self.np_rng.integers(0, self.grid.height))
        self.grid.place_agent(agent, (x, y))
        agent.pos = (x, y)

        # Add to schedule
        self.schedule.add(agent)
        self.agent_counter += 1

    def staggered_intake(self):
        if self.current_month < self.warmup_months:
            intake_percentage = self.monthly_intake / 100  # 5% per month
            intake = int(self.initial_agents * intake_percentage)
            for _ in range(intake):
                self.create_agent()
                

    def calibration_error_by_window(self, years: int) -> float:
        if self.current_month <= self.warmup_months:
            return None
        # Same buffer as calculate_flag_rate — must be consistent
        min_study_months = self.warmup_months + years * 12
        if self.current_month < min_study_months:
            return None

        # Use agent-level flag instead of percent_study_recidivists_in_window
        flag_name = f"rearrest_{years}_yrs"
        observed = self.calculate_flag_rate(flag_name)
        target = self.bjs_targets.get(years, 0.0)
        
        return abs(observed - target) if observed is not None else None



    
    def count_recidivists_during_study(self):
        return sum(
            1 for a in self.schedule.agents
            if isinstance(a, Person)
            and a.recidivated_agent
        )
    def recidivism_rate_by_year(self) -> float:
        year = self.current_year
        eligible = [
            a for a in self.schedule.agents
            if getattr(a, "study_eligible_agent", False)
        ]
        recidivists = [
            a for a in eligible
            if getattr(a, "recidivated_agent", False) and getattr(a, "rearrest_month", -1) // 12 == year
        ]
        return len(recidivists) / len(eligible) if eligible else 0.0

    def compute_cumulative_recidivism_rate(self):
        recidivated = 0
        total = 0
        if self.current_month <= self.warmup_months:
        #    return None  # or 0.0, or skip logging during warm-up
            for agent in self.schedule.agents:
                if hasattr(agent, "recidivated_agent"):
                    total += 1
                    if agent.recidivated_agent:
                        recidivated += 1
        rate = recidivated / total if total > 0 else 0.0
        #self.monthly_recidivism_rate.append(rate)
        return rate
    
    def compute_monthly_recidivism_rate(self):
        recidivated = sum(1 for a in self.schedule.agents if a.committed_new_offense_this_month)
        total = len(self.schedule.agents)
        rate = recidivated / total if total > 0 else 0.0
        self.monthly_recidivism_rate.append(rate)
        return rate


    def fairness_metrics(self):
        groups = {}
        for a in self.schedule.agents:
            key = (a.race, a.supervision_level)
            if key not in groups:
                groups[key] = {"count": 0, "rearrested": 0}
            groups[key]["count"] += 1
            groups[key]["rearrested"] += a.rearrest_within_3yr
        return {
            f"{k[0]}_{k[1]}": round(v["rearrested"] / v["count"], 3) if v["count"] > 0 else 0
            for k, v in groups.items()
        }
    def compute_risk_score_bins(self):
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        counts = {label: 0 for label in labels}

        for agent in self.schedule.agents:
            score = agent.risk_score if hasattr(agent, "risk_score") else 0.0
            for i in range(len(bins)-1):
                if bins[i] <= score < bins[i+1]:
                    counts[labels[i]] += 1
                    break
        return counts
    def get_time_series(self):
        return pd.DataFrame(self.time_series)

    def verify_activation_shuffle(self, n_months=3):
        """
        Diagnostic: confirms that np_rng produces different activation orders
        across consecutive shuffles of the current agent list.

        NOTE: Past shuffle states cannot be reconstructed — np_rng is a persistent
        stateful RNG that has evolved through all prior draws. This method verifies
        that the shuffle mechanism is working right now, not what happened in past months.

        Example output (good — different IDs each shuffle):
            Shuffle 1 | First 5: [302, 87, 415, 203, 56]
            Shuffle 2 | First 5: [91, 447, 12, 388, 201]
            Shuffle 3 | First 5: [503, 34, 277, 88, 412]

        Example output (bad — same IDs every shuffle, RNG broken):
            Shuffle 1 | First 5: [0, 1, 2, 3, 4]
            Shuffle 2 | First 5: [0, 1, 2, 3, 4]
        """
        print(f"\n🔀 Activation Order Verification ({n_months} consecutive shuffles, current month={self.current_month})")
        print("─" * 55)

        agents = list(self.schedule.agents)
        for i in range(1, n_months + 1):
            sample = agents.copy()
            self.np_rng.shuffle(sample)
            ids = [a.unique_id for a in sample[:5]]
            print(f"  Shuffle {i} | First 5: {ids}")

        print("─" * 55)
        print("✅ If IDs differ each month, shuffling is working correctly.\n")