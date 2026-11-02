import numpy as np

class VectorizedRecidivismModelWithSpatialHash:
    def __init__(self, n_agents=1_000_000, grid_size=(1000,1000), warmup_months=3, seed=42):
        np.random.seed(seed)
        self.n_agents = n_agents
        self.grid_width, self.grid_height = grid_size
        self.warmup_months = warmup_months
        self.current_month = 0

        # ─── Demographics ───
        self.gender = np.random.choice(['Male', 'Female'], n_agents)
        self.race = np.random.choice(['White', 'Black', 'Hispanic', 'Other'], n_agents)
        self.age_at_entry = np.random.randint(18, 60, n_agents)
        self.age_at_release = self.age_at_entry.copy()
        self.gang_affiliated = np.random.rand(n_agents) < 0.1
        self.dependents = np.random.randint(0, 5, n_agents)
        self.residence_changes = np.random.randint(0, 3, n_agents)
        self.education_level = np.random.choice(
            ['Less than HighSchool', 'HighSchool', 'College'], n_agents
        )

        # ─── Justice State: 0=Trial,1=Prison,2=Supervision,3=Free ───
        self.justice_state = np.where(np.arange(n_agents) < warmup_months, 0, 3)
        self.months_in_state = np.zeros(n_agents, dtype=int)

        # ─── Offense Types ───
        self.offense = np.random.choice(
            ["Violent-SexOffender", "Violent/Non-Sex", "Drug", "Property", "Other(PublicOrder)"],
            n_agents
        )

        # ─── Prior Convictions ───
        self.prior_felony = np.random.randint(0, 5, n_agents)
        self.prior_misd = np.random.randint(0, 5, n_agents)

        # ─── Risk / Scores ───
        self.raw_score = np.random.rand(n_agents)
        self.dynamic_risk_score = np.zeros(n_agents)
        self.supervision_term = np.zeros(n_agents)
        self.sentence_length = np.zeros(n_agents)

        # ─── Recidivism ───
        self.recidivated = np.zeros(n_agents, dtype=bool)
        self.rearrest_month = np.full(n_agents, -1)

        # ─── Grid positions ───
        self.pos_x = np.random.randint(0, self.grid_width, n_agents)
        self.pos_y = np.random.randint(0, self.grid_height, n_agents)

        # ─── Peer influence ───
        self.enable_peer_influence = True
        self.peer_influence_base = 0.05
        self.peer_influence_max = 0.3
        self.cohort_ids = np.random.randint(0, 100, n_agents)  # 100 synthetic cohorts

    # ─────────── Normalize Risk ───────────
    def normalize_risk(self, mask):
        scores = self.raw_score[mask]
        ranks = scores.argsort().argsort()
        self.dynamic_risk_score[mask] = ranks / max(1, len(scores)-1)

    # ─────────── Spatial Hash Peer Influence ───────────
    def apply_peer_influence(self):
        if not self.enable_peer_influence:
            return

        # Compute flat cell IDs
        flat_cells = self.pos_y * self.grid_width + self.pos_x
        recid_flat = np.zeros_like(flat_cells, dtype=int)
        recid_flat[self.recidivated] = 1

        # Count recidivated agents per cell
        recid_count_per_cell = np.bincount(flat_cells, weights=recid_flat, minlength=self.grid_width*self.grid_height)

        # Apply boost: dynamic_risk_score += base * #recid in same cell
        boost = self.peer_influence_base * recid_count_per_cell[flat_cells]
        self.dynamic_risk_score += boost
        self.dynamic_risk_score = np.clip(self.dynamic_risk_score, 0, 1)

        # Cohort-based influence
        unique_cohorts, cohort_idx = np.unique(self.cohort_ids, return_inverse=True)
        cohort_recid_count = np.bincount(cohort_idx, weights=self.recidivated)
        cohort_boost = 0.01 * cohort_recid_count[cohort_idx]
        self.dynamic_risk_score = np.clip(self.dynamic_risk_score + cohort_boost, 0, self.peer_influence_max)

    # ─────────── Assign Supervision & Sentence ───────────
    def assign_sentence_and_supervision(self):
        min_max = {
            "Violent-SexOffender": (48, 120),
            "Violent/Non-Sex": (36, 96),
            "Drug": (18, 60),
            "Property": (6, 48),
            "Other(PublicOrder)": (6, 60)
        }
        for offense, (min_m, max_m) in min_max.items():
            mask = self.offense == offense
            risk = self.dynamic_risk_score[mask]
            scaled = risk * 0.25
            sentence = 3 * np.round((min_m + (max_m - min_m) * scaled) / 3)
            self.sentence_length[mask] = np.clip(sentence, min_m, max_m)
            self.supervision_term[mask] = 6 + 30*risk + 2*self.prior_felony[mask] + self.prior_misd[mask]
            self.supervision_term[mask] = 3*np.round(np.clip(self.supervision_term[mask],3,48)/3)

    # ─────────── Evaluate Recidivism ───────────
    def evaluate_recidivism(self, mask):
        adjusted_score = self.dynamic_risk_score[mask] + np.random.uniform(-0.02,0.02, mask.sum())
        adjusted_score = np.clip(adjusted_score,0,1)
        years_since_release = self.current_month / 12
        desistance_factor = np.where(
            years_since_release<=3,
            max(0.35, 1-0.12*years_since_release),
            max(0.35, 0.64 - 0.05*max(0, years_since_release-3))
        )
        combined_risk = np.clip(adjusted_score*0.85 + adjusted_score*0.15,0,1)*desistance_factor
        rearrest_prob = 0.12 + (1-0.12)/(1 + np.exp(-8*(combined_risk-0.5)))
        recid = np.random.rand(mask.sum()) < rearrest_prob
        self.recidivated[mask] = recid
        self.rearrest_month[mask] = np.where(recid, self.current_month, self.rearrest_month[mask])

    # ─────────── Move Agents ───────────
    def move_agents(self):
        dx = np.random.randint(-1,2,self.n_agents)
        dy = np.random.randint(-1,2,self.n_agents)
        self.pos_x = np.clip(self.pos_x + dx, 0, self.grid_width-1)
        self.pos_y = np.clip(self.pos_y + dy, 0, self.grid_height-1)

    # ─────────── Simulation Step ───────────
    def step(self):
        self.months_in_state += 1
        self.move_agents()

        # Trial → Prison
        trial_mask = self.justice_state==0
        trial_done = self.months_in_state[trial_mask]>=3
        self.justice_state[trial_mask] = np.where(trial_done,1,self.justice_state[trial_mask])
        self.months_in_state[trial_mask & trial_done] = 0

        # Prison → Supervision
        prison_mask = self.justice_state==1
        self.assign_sentence_and_supervision()
        prison_done = self.months_in_state[prison_mask]>=self.sentence_length[prison_mask]
        self.justice_state[prison_mask] = np.where(prison_done,2,self.justice_state[prison_mask])
        self.months_in_state[prison_mask & prison_done] = 0

        # Supervision → Free / Recidivism
        supervision_mask = self.justice_state==2
        self.normalize_risk(supervision_mask)
        self.apply_peer_influence()
        self.evaluate_recidivism(supervision_mask)
        recid_supervision = supervision_mask & self.recidivated
        self.justice_state[recid_supervision] = 0
        self.months_in_state[recid_supervision] = 0
        supervision_done = supervision_mask & (~self.recidivated) & (self.months_in_state>=self.supervision_term)
        self.justice_state[supervision_done] = 3
        self.months_in_state[supervision_done] = 0

        # Free → Trial (recidivism)
        free_mask = self.justice_state==3
        self.normalize_risk(free_mask)
        self.apply_peer_influence()
        self.evaluate_recidivism(free_mask)
        recid_free = free_mask & self.recidivated
        self.justice_state[recid_free] = 0
        self.months_in_state[recid_free] = 0

        self.current_month += 1

# ─────────── Run Simulation ───────────
model = VectorizedRecidivismModelWithSpatialHash(n_agents=1_000_000)
for month in range(12):
    model.step()
print("Simulation complete. Recidivated:", np.sum(model.recidivated))
