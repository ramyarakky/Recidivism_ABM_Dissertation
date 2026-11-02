from mesa import Agent
import random
from scoring import compute_risk_score, normalize_score
import numpy as np

class Person(Agent):
    def __init__(self, unique_id, model, agent_data):
        super().__init__(unique_id, model)
        self.agent_data = {}
        self.unique_id = unique_id
        self.model = model
        self.rand_val= random.random()
        self.exited_due_to_recidivism = False
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
            
        self.trial_wait_threshold = model.random.randint(1, 4)
        self.raw_score = compute_risk_score(self,weights=self.model.weights) or 0.0
        #print('Raw score:',self.raw_score)
        self.months_in_state = 0
        self.time_served_months = 0
        self.sentence_length_months = 0
        self.dynamic_risk_score = 0.0
        self.rearrest_within_3yr = 0
        #self.reincarceration_within_3yr = 0
        self.committed_new_offense_this_month = False
        self.recidivated_agent = False
        self.study_eligible_agent = False
        self.free_at_warmup_end = False
        self.free_start_month = 0
        self.supervision_start_month=0
        self.agent_entry_month = self.model.current_month
        self.rearrest_month = None
        self.release_month = None
        self.recidivism_month = None
        self.bias_exposure = model.bias_factor
        self.justice_state_changed = False
        self.normalized_score =0.0
        self.recidivated_count = 0
        # Optional flags for analysis
        self.rearrest_3_yrs = False
        self.rearrest_6_yrs = False
        self.rearrest_9_yrs = False
        #self.transition_log = []
        
    def initialize_attributes_from_data(self):
        for key, value in self.agent_data.items():
            setattr(self, key, value)

    def display_all_attributes(self):
        print(f"\n🧠 Agent {self.unique_id} Full Attribute Snapshot:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")

    def normalize_score_by_phase(self, phase):
        # Collect raw scores of all agents in the same justice phase
        self.raw_score = compute_risk_score(self,weights=self.model.weights)
        #print("Agent:",self.unique_id,"Phase:",phase," Raw Score:",self.raw_score)
        scores = [
            agent.raw_score
            for agent in self.model.schedule.agents
            if hasattr(agent, "justice_state") and agent.justice_state == phase and hasattr(agent, "raw_score")
        ]
        

        # Sort scores and compute rank
        sorted_scores = sorted(scores)
        total_score = self.raw_score
        try:
            rank = sorted_scores.index(total_score)
        except ValueError:
            rank = 0  # fallback if score not found

        # Normalize to percentile
        normalized_score = rank / len(sorted_scores) if len(sorted_scores) > 0 else 0.0
        #print("Agent:",self.unique_id,"Phase:",phase," Score:",normalized_score)
        return normalized_score

    def compute_adjusted_risk_score(self, phase: str) -> float:
        """
        Computes normalized dynamic risk score for a given justice phase.
        Adds peer influence (Prison) if enabled. Returns a score bounded between 0 and 1.
        """
        # ─── Peer Influence Calculation (Prison Phase Only) ───
        if phase == "Prison" and self.model.enable_peer_influence:
            config = self.model.peer_influence_weights
            boost = 0.0

            for mate in self.model.grid.get_cell_list_contents([self.pos]):
                if mate != self:
                    # Influence from recidivated peers
                    if getattr(mate, "recidivated_agent", False):
                        offense = getattr(mate, "offense", "Other(PublicOrder)")
                        multiplier = config["offense_multiplier"].get(offense, 1.0)
                        
                        try:
                            boost += config["base_influence"] * multiplier
                            #print('Boost',boost)
                        except TypeError as e:
                            print(f"Agent {self.unique_id} failed multiplication: {e}")
                    # Additional influence from gang-affiliated peers
                    if getattr(mate, "gang_affiliated", False):
                        boost += config.get("gang_affiliation_boost", 0.05)  # Optional extra boost

            self.peer_influence_boost = min(boost, config["max_influence"])
            

        # ─── Compute Final Score ───
        score = self.normalize_score_by_phase(phase)
        return max(0.0, min(score, 1.0))

    def assign_sentence_length(self):
        """
        Assigns prison sentence length (in months) based on offense type and normalized risk score.
        Calibrated to reflect BJS/USSC sentencing norms and reduce early exits for high-risk agents.
        """

        # ─── 1. Normalize risk score for prison phase ───
        risk = self.normalize_score_by_phase("Prison")
        #print(risk)
        # ─── 2. Base sentence ranges by offense type (BJS/USSC-inspired) ───
        ranges = {
            "Violent-SexOffender": (48, 120),
            "Violent/Non-Sex":     (36, 96),
            "Drug":                (18, 60),
            "Property":            (6, 48),
            "Other(PublicOrder)":  (6, 60)
        }
        min_months, max_months = ranges.get(self.offense, (6, 120))

        
        try:
            # ─── 3. Apply nonlinear scaling to risk ───
            scaled = risk * 0.25  # dampens inflation for low-risk agents
            #print(scaled)
            sentence = min_months + (max_months - min_months) * scaled
            #print("Sentence b4",sentence)
            # ─── 4. Round to nearest 3-month block ───
            sentence = 3 * round(sentence / 3)
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
        Calculates supervision duration based on offense type, dynamic risk score, and prior convictions.
        Also assigns supervision level, risk code, and supervision risk score (1–10 scale).
        """

        # ─── Supervision Profile by Offense ───
        offense = getattr(self, "offense", "Other(PublicOrder)")
        profile_map = {
            "Violent-SexOffender": ("High", "VSO", 10, 48),
            "Violent/Non-Sex":     ("High", "VNS", 9, 36),
            "Drug":                ("Specialized", "DRG", 8, 48),
            "Property":            ("Standard", "PRP", 5, 24),
            "Other(PublicOrder)":  ("Standard", "PUB", 3, 12)
        }
        level, code, risk_score, bonus = profile_map.get(offense, ("Standard", "UNK", 1, 0))

        # ─── Assign Supervision Attributes ───
        self.Supervision_Level_First = level
        self.Supervision_Risk_Code = code
        self.Supervision_Risk_Score = max(1, min(risk_score, 10))  # Clamp to 1–10

        try:
            # ─── Duration Components ───
            risk = self.compute_adjusted_risk_score("Supervision") or 0.0  # Range: 0.0–1.0
            if risk is None or risk_score is None:
                raise ValueError(f"Agent {self.unique_id} has missing values: risk={risk}, risk_score={risk_score}")

            base = 6 + 30 * risk                                     # Scaled base duration
            severity_boost = 1.2 * risk_score                        # Up to +12 months

            felony = getattr(self, "Prior_Conviction_Episodes_Felony", 0)
            misd = getattr(self, "Prior_Conviction_Episodes_Misd", 0)
            prior_boost = 2 * felony + misd

            # ─── Final Duration ───
            duration = base + severity_boost + bonus + prior_boost
            duration = max(3, min(duration, 48))                     # Clamp to 3–48 months
            duration = 3 * round(duration / 3)                       # Round to nearest 3-month block

            self.supervision_duration = duration
        except TypeError as e:
            print(f"Agent {self.unique_id} failed multiplication during derive_supervision_duration: {e}")

        return duration


    def evaluate_recidivism_by_percentile(self, phase: str, 
                                          #duration: int, 
                                          threshold: float, risk_cutoff: float = 0.5):
        """
        Evaluate recidivism risk using percentile rank and absolute score.
        Applies behavioral noise, cohort comparison, and stochastic rearrest logic.

        Parameters:
        - phase (str): Justice state ("Free" or "Supervision")
        - duration (int): Evaluation window in months
        - threshold (float): Percentile cutoff for high-risk classification
        - risk_cutoff (float): Absolute score threshold for high-risk flag

        Theoretical Basis:
        - RNR, Procedural Justice, Desistance Theory, Empirical Calibration
        """

        if phase not in ["Free","Supervision"]:
            return False

        # ─── 1. Adjust Risk Score with Behavioral Noise ───
        base_score = self.compute_adjusted_risk_score(phase)
        noise = random.uniform(-0.05, 0.05)
        adjusted_score = max(0.0, min(1.0, base_score + noise))
        self.dynamic_risk_score = adjusted_score

        # ─── 2. Compute Percentile Rank ───
        
        cohort_scores = [
            #a.compute_adjusted_risk_score(phase)
            a.dynamic_risk_score
            for a in self.model.schedule.agents
            if a.justice_state == phase and a != self
        ]
        percentile = (
            sum(1 for score in cohort_scores if adjusted_score > score) / len(cohort_scores)
            if cohort_scores else 0.5
        )

        # ─── 3. Flag High-Risk Status ───
        high_risk = adjusted_score >= risk_cutoff
        high_percentile = percentile > threshold

        # ─── 4. Rearrest Probability by Score Tier ───
        rearrest_prob = (
                        0.95 if adjusted_score >= 0.8 else
                        0.85 if adjusted_score >= 0.7 else
                        0.65 if adjusted_score >= 0.6 else
                        0.45 if adjusted_score >= 0.5 else
                        0.25
                    )

        #print("Rearrest Prob:",rearrest_prob)
        rand_val = random.random()

        # ─── 5. Rearrest Decision ───
        if (high_risk or high_percentile) and rand_val < rearrest_prob:
            delay = random.randint(1, 6)
            #self.rearrest_month = self.model.current_month + delay
            self.recidivated_agent = True
            #self.recidivism_month = self.rearrest_month
        else:
            #self.rearrest_month = self.rearrest_month #getattr(self, "rearrest_month", None)
            self.recidivated_agent = self.recidivated_agent  #getattr(self, "recidivated_agent", False)
            delay = None

        # ─── 6. Update Age at Release ───
        entry_age = self.agent_entry_month #getattr(self, "Age_at_Entry", self.model.current_month)
        months_in_system = self.model.current_month - entry_age
        self.Age_at_Release = round(months_in_system//12+self.Age_at_Entry, 1)

        # ─── 7. Log Evaluation ───
        #self.transition_log.append(
        #    (self.model.current_month, f"{phase}→RiskEval", {
        #        "adjusted_score": round(adjusted_score, 3),
        #        "percentile": round(percentile, 3),
        #        "rearrest_prob": round(rearrest_prob, 2),
        #        "rand_val": round(rand_val, 3),
        #        "delay": delay,
        #        "recidivated": self.recidivated_agent
        #    })
        #)

        return self.recidivated_agent

    def step(self):
        try:
            #self.display_all_attributes()
            #print(f"Agent {self.unique_id} data: {self.agent_data}")
            #print(f"Agent {self.unique_id} weights: {self.model.weights}")
            # Reset monthly flag
            self.committed_new_offense_this_month = False
            if self.agent_data is None or self.model.weights is None:
                print(f"⚠️ Agent {self.unique_id} missing data or weights")
                return
            
            self.months_in_state += 1
            if  self.exited_due_to_recidivism :
                #print(f"⏸ Agent {self.unique_id} skipped due to recidivism count of {self.recidivated_count}.")
                return
            #self.transition_log.append((self.model.current_month, self.justice_state))
            self.justice_state_changed = False
            #print("Agent State:",self.justice_state)
            
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

    def evaluate_quarterly_recidivism(self,phase):
        """
        Evaluates whether the agent has recidivated during the current quarter.
        Updates justice state and offense flags based on the model's current month.
        """
        recidivated = self.evaluate_recidivism_by_percentile(
            phase=phase,
            #duration=self.MonthsCrimeFree,
            threshold=0.25
        )

        if recidivated:
            self.recidivated_agent = True
            self.committed_new_offense_this_month = True
            self.rearrest_month = self.model.current_month
            self.rearrest_year = self.rearrest_month // 12
            
            if self.model.current_month <= self.model.warmup_months:
                self.justice_state = "Trial"
                self.justice_state_changed = True
                self.months_in_state = 0
                self.handle_trial()
            elif recidivated and self.model.current_month > self.model.warmup_months:
                self.committed_new_offense_this_month = True
                self.recidivated_agent = True
                self.rearrest_month = self.model.current_month
                self.rearrest_year = self.rearrest_month // 12
                self.exited_due_to_recidivism = True
                # ✅ Update rearrest flags if agent has been rearrested
                if isinstance(self.rearrest_month, (int, float)):
                    for window in [3, 6, 9]:
                        cutoff = self.model.warmup_months + window * 12
                        #print('cutoff',cutoff)
                        attr_name = f"rearrest_{window}_yrs"
                        setattr(self, attr_name, self.model.warmup_months < self.rearrest_month <= cutoff)

        

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
                #print('risk',risk)
                

                # ─── 3. Recidivism Theory Integration ───
                # Based on BJS studies, recidivism risk is highest within 3 years post-release,
                # and varies by offense type, prior history, and supervision exposure.
                # Diversion and release are less likely for high-risk agents; conviction more likely.
                # See: https://bjs.ojp.gov/recidivism-program 

                diversion_prob = max(0.05, 0.2 - 0.2 * risk + self.model.bias_factor)  # Lower diversion for high-risk agents
                release_prob = max(0.02, 0.1 - 0.1 * risk + self.model.bias_factor)  # Higher chance of release for low-risk agents
                #print('diversion_prob',diversion_prob,':','release_prob',release_prob )
                # Sentencing Process

                self.Supervision_Term = self.derive_supervision_duration()
                self.Prison_Years = self.assign_sentence_length()

                # ─── 4. Draw random outcome ───
                new_state = None

                if random.random() < release_prob:
                    # 🎯 Agent is released (dismissed or acquitted)
                    new_state = "Free"
                    #self.justice_state = "Free"
                    self.handle_free()
                    # self.transition_log.append((self.model.current_month, "Trial→Free", round(risk, 3)))

                elif random.random() < release_prob + diversion_prob:
                    # 🎯 Agent is diverted to supervision
                    new_state = "Supervision"
                    self.handle_supervision()
                    # self.transition_log.append((self.model.current_month, "Trial→Diversion", round(risk, 3)))

                else:
                    # 🎯 Agent is convicted and sent to prison
                    new_state = "Prison"
                    self.handle_prison()
                    # self.transition_log.append((self.model.current_month, "Trial→Prison", round(risk, 3)))

                # ─── 5. Update justice state and reset timer ───
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

            if random.random() < upgrade_prob:
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

            # ─── 3. Record Rearrest Month ───
            # Arrest/Rearrest is recorded during Trial stage

            # ─── 4. Check for Release ───
            sentence_complete = self.time_served_months >= self.Prison_Years
            # Example: 70% chance of requiring supervision if Supervision_Term > 0
            if self.Supervision_Term > 0:
                supervision_required = random.random() < 0.7
            else:
                supervision_required = False
            # ─── Gang Affiliation Assignment (Prison Phase Only) ───
            # The study “Gang Affiliation and Prisoner Reentry” analyzes recidivism patterns
            # and confirms that 13–20% of incarcerated individuals are gang-affiliated
            if getattr(self, "gang_affiliated", False):
                self.Gang_Affiliated = self.random.random() < 0.15

            if sentence_complete:
                # Update conviction history and release month
                self._update_conviction_history()
                # Update correctional education attainment
                self._update_correctional_education()
                self.release_month = self.model.current_month
                self.months_in_state = 0
                # Update the risk score based on life  events in prison
                self.normalized_score = self.compute_adjusted_risk_score("Prison")
                
                if supervision_required:
                    # Transition to supervision
                    self.justice_state = "Supervision"
                    self.justice_state_changed = True
                    self.handle_supervision()
                else:
                    # Transition to freedom
                    self.justice_state = "Free"
                    self.justice_state_changed = True
                    self.handle_free()
        except Exception as e:
                print(f"Agent {self.unique_id} has error during Prison transition: {e}")

    # ─────────────────────────────────────────────
    # 🧭 Supervision Phase
    # ─────────────────────────────────────────────

    def handle_supervision_and_programs(self, n_programs=4):
        """
        Simulate yearly supervision and program behaviors for an agent.

        Includes:
        - Prior revocations (population mean ~10%)
        - Technical violations (population mean ~45%)
        - Program attendance per session (population mean ~45%)
        - Unexcused absences tied to attended sessions (~20–40% of attendees)
        """

        # Ensure counters exist
        if not hasattr(self, "Program_Attendances"):
            self.Program_Attendances = 0
        if not hasattr(self, "Program_UnexcusedAbsences"):
            self.Program_UnexcusedAbsences = 0

        revoked = False

        # ─── 1. Prior Supervision Revocation ───
        strength_revocation = 6
        mean_percent_revocation = 10  # population-level mean
        if getattr(self, "Condition_MH_SA", False) or getattr(self, "Gang_Affiliated", False):
            mean_percent_revocation += 5  # elevated risk

        # Ensure Beta parameters > 0
        a_rev = max((mean_percent_revocation / 100) * strength_revocation, 0.01)
        b_rev = max(((100 - mean_percent_revocation) / 100) * strength_revocation, 0.01)
        p_revocation = np.random.beta(a_rev, b_rev)
        self.Prior_Revocations_Supervision = np.random.random() < p_revocation

        if self.Prior_Revocations_Supervision:
            revoked = True

        # ─── 2. Technical Violations ───
        strength_violation = 6
        mean_percent_violation = 45
        if getattr(self, "Gang_Affiliated", False) or getattr(self, "Percent_Days_Employed", 100) < 30:
            mean_percent_violation += 10

        a_viol = max((mean_percent_violation / 100) * strength_violation, 0.01)
        b_viol = max(((100 - mean_percent_violation) / 100) * strength_violation, 0.01)
        p_violation = np.random.beta(a_viol, b_viol)
        self.Violations_Technical = np.random.random() < p_violation

        # ─── 3. Program Attendance & Unexcused Absences ───
        strength_attendance = 6
        if getattr(self, "Condition_MH_SA", False) or getattr(self, "Condition_Cog_Ed", False) or getattr(self, "Condition_Other", False):
            mean_percent_attend = 40
        else:
            mean_percent_attend = 45

        a_att = max((mean_percent_attend / 100) * strength_attendance, 0.01)
        b_att = max(((100 - mean_percent_attend) / 100) * strength_attendance, 0.01)
        p_attend = np.random.beta(a_att, b_att)

        attended_sessions = 0
        unexcused_absences = 0
        for _ in range(n_programs):
            if np.random.random() < p_attend:
                attended_sessions += 1

                # Per-session unexcused absence probability
                base_absence_prob = 0.3  # midpoint of Pew/Urban range
                if getattr(self, "Gang_Affiliated", False) or getattr(self, "Percent_Days_Employed", 100) < 30:
                    base_absence_prob += 0.1  # elevated risk
                base_absence_prob = min(base_absence_prob, 0.8)  # cap at 80%

                if np.random.random() < base_absence_prob:
                    unexcused_absences += 1

        self.Program_Attendances += attended_sessions
        self.Program_UnexcusedAbsences += unexcused_absences

        return revoked



    def handle_supervision_violations_and_revocation(self):
        """
        Applies supervision violations and evaluates revocation risk.
        Includes random assignment of violations and optional escalation to Trial.
        """

        # ─── 1. Prior Supervision Revocation ─── 10% 
        revoked = False
        base_prob = 0.10
        if self.Condition_MH_SA or self.Gang_Affiliated: #or self.Delinquency_Reports:
            base_prob += 0.05  # elevated risk
        self.Prior_Revocations_Supervision = random.random() < base_prob
        if self.Prior_Revocations_Supervision:
            revoked = True

        # ─── 2. Supervision Violations ─── 45%
        self.Violations_Technical = random.random() < 0.45
        
        
        # ─── 3. Delinquency_Reports ─── 30% or 40%
        #base_prob = 0.30
        #if self.Condition_MH_SA or self.Gang_Affiliated or self.Percent_Days_Employed < 30:
        #    base_prob += 0.10  # elevated risk
        #self.Delinquency_Reports = random.random() < base_prob
        

        # ─── 4. Program Attendance 45% ───
        if self.Condition_MH_SA or self.Condition_Cog_Ed or self.Condition_Other:
            self.Program_Attendances += int(random.random() < 0.45)

        # ─── Yearly Unexcused Absence Evaluation ───
        base_absence_prob = 0.15
        if self.Gang_Affiliated or self.Percent_Days_Employed < 30:
            base_absence_prob += 0.10  # elevated risk
        # Convert quarterly probability to yearly probability
        yearly_prob = 1 - (1 - base_absence_prob) ** 4  # 4 quarters in a year
        if random.random() < yearly_prob:
            self.Program_UnexcusedAbsences += 1


        return revoked

        
        # self.transition_log.append((self.model.current_month, "Supervision→Trial", round(self.dynamic_risk_score, 3)))

    def handle_supervision(self):
        try:
            # ─── 1. Initialize Supervision Start ───
            if self.months_in_state == 0:
                self.supervision_start_month = self.model.current_month
        
            months_since_start = self.model.current_month - self.supervision_start_month

            # ─── 2. Annual Revocation Checks ───
            if months_since_start % 12 == 0:
                #self.handle_supervision_violations_and_revocation()
                #if self.handle_supervision_violations_and_revocation():
                if self.handle_supervision_and_programs():
                    self.evaluate_quarterly_recidivism("Supervision")
                    self.justice_state = "Trial"
                    
                    
                        
            # ─── 3. Transition to Free if Supervision Ends ───
            if self.months_in_state >= self.Supervision_Term:
                self.justice_state = "Free"
                self.justice_state_changed = True
                self.months_in_state = 0
                # Update the risk score based on life  events in prison
                self.normalized_score = self.compute_adjusted_risk_score("Prison")
                self.handle_free()
        except Exception as e:
                print(f"Agent {self.unique_id} has error during Supervision transition: {e}")


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
        - Individual cap: Each agent’s total Percent_Days_Employed cannot exceed 40%.
        (i.e., even consistently working agents do not surpass ~40% of days employed.)
        - Employment evolves quarterly, affected by individual-level risk and protective factors.

        Data reference:
        ---------------
        Alper, M., Durose, M. R., & Markman, J. (2018).
        2018 Update on Prisoner Recidivism: A 9-Year Follow-up Period (2005–2014).
        Bureau of Justice Statistics, NCJ 250975.
        """

        # ─────────────────────────────────────────────────────────────
        # INITIALIZATION (First quarter after release)
        # ─────────────────────────────────────────────────────────────
        if not hasattr(self, "Percent_Days_Employed"):
            # Initialize employment attributes if not already set
            self.Percent_Days_Employed = 0.0  # % of total days post-release employed
            self.IsEmployed = False           # Binary employment status flag

            # Assign whether this agent will EVER gain employment post-release
            # → About 67% of individuals are expected to become employed at least once
            self.WillEverBeEmployed = random.random() < 0.67

        # ─────────────────────────────────────────────────────────────
        # QUARTERLY UPDATE (For employable agents only)
        # ─────────────────────────────────────────────────────────────
        if getattr(self, "WillEverBeEmployed", False) and self.Percent_Days_Employed < 40.0:
            # Base quarterly probability of employment gain.
            # 12% chosen so that over ~16 quarters (4 years), a majority reach some employment.
            base_prob = 0.12

            # Adjust for individual risk/advantage factors:

            # ↓ Gang affiliation or MH/SA issues reduce employment likelihood
            if getattr(self, "Gang_Affiliated", False) or getattr(self, "Condition_MH_SA", False):
                base_prob -= 0.04

            # ↑ Higher education slightly boosts job acquisition odds
            if getattr(self, "Education_Level", "") in ["High School Diploma", "College"]:
                base_prob += 0.03

            # Stochastic employment gain event for this quarter
            if random.random() < base_prob:
                # Each successful quarter increases employment days by 5–10%
                gain = random.uniform(5.0, 10.0)

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

        - Base annual probability: 25%
        - Elevated to 35% for agents with MH/SA or gang affiliation
        - Converted to quarterly probability (~7–9% per quarter)
        - Tracks cumulative residence changes
        """

        # Determine annual probability based on risk factors
        if getattr(self, "Condition_MH_SA", False) or getattr(self, "Gang_Affiliated", False):
            p_annual = 0.35
        else:
            p_annual = 0.25

        # Convert annual probability to quarterly probability
        p_quarter = 1 - (1 - p_annual) ** 0.25

        # Determine if agent moves this quarter
        if random.random() < p_quarter:
            if not hasattr(self, "Residence_Changes"):
                self.Residence_Changes = 0
            self.Residence_Changes += 1


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
            if self.months_in_state == 0:
                self.free_start_month = self.model.current_month

            self.MonthsCrimeFree = self.model.current_month - self.free_start_month

            # ─── 2. Probabilistic Life Transitions ───
            if self.MonthsCrimeFree % 3 == 0 :
                self.update_post_release_employment()
                        
                # Gang involvement if not already affiliated,
                # 25% yearly chance becomes a ~6.9% quarterly chance.
                base_prob = 0.06  # ~6% quarterly chance
                if self.Age_at_Release < 25 or self.Education_Level == "Less than HighScool Diploma":
                    base_prob += 0.03  # elevated risk
                if random.random() < base_prob:
                    self.Gang_Affiliated = True
            
            
                # Mental Health Condition Changes
                # ─── Develop New Behavioral Condition ───
                if random.random() < 0.07:  # 7% chance per quarter
                    possible_conditions = []
                    if not self.Condition_MH_SA:
                        possible_conditions.append("Condition_MH_SA")
                    if not self.Condition_Cog_Ed:
                        possible_conditions.append("Condition_Cog_Ed")
                    if not self.Condition_Other:
                        possible_conditions.append("Condition_Other")
                    if possible_conditions:
                        new_condition = random.choice(possible_conditions)
                        setattr(self, new_condition, True)
            # self.transition_log.append((self.model.current_month, "Free→DevelopedCondition", new_condition))

                # Add dependent if below cap
                # Yearly dependent gain 7% , connverting to quarterly 0.018
                if self.Dependents < 5:
                    if random.random() < 0.018:
                        self.Dependents += 1
                        
                self.update_residence_changes()

                # ─── 3. Quarterly Recidivism Evaluation ───
                self.evaluate_quarterly_recidivism("Free")
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
                    new_pos = self.random.choice(neighbors)
                    self.model.grid.move_agent(self, new_pos)
                    self.pos = new_pos
            except Exception as e:
                print(f"Agent {self.unique_id} movement error: {e}")
                
    



        """
        Returns a dictionary of agent attributes for export.
        Fully synchronized with generate_synthetic_agent() field names.
        Includes synthetic profile, justice state, risk scores, and recidivism flags.
        """
    def get_agent_vars(self):
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
            "Supervision_Duration": self.Supervision_Term,

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

            # ───────────── Risk and Recidivism Outcomes ─────────────
            "dynamic_risk_score": self.dynamic_risk_score,
            "recidivated_agent": self.recidivated_agent,
            "rearrest_month": self.rearrest_month,
            "rearrest_3_yrs": self.rearrest_3_yrs,
            "rearrest_6_yrs": self.rearrest_6_yrs,
            "rearrest_9_yrs": self.rearrest_9_yrs
        }

