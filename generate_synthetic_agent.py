import random

def stratified_conditions(mode="realistic") -> dict:
    """
    Assigns behavioral and supervision-related conditions:
    - Condition_MH_SA: Mental Health / Substance Abuse
    - Condition_Cog_Ed: Cognitive or Educational Needs
    - Condition_Other: Other supervision-related conditions (e.g., housing, medical)

    Modes:
    - 'realistic': Uses subgroup prevalence from NIJ, SAMHSA, and peer-reviewed studies.
    - 'equalized': Uses aggregate population averages to remove intake bias while preserving realism.

    References:
    - NIJ Report (2022): ~70% of incarcerated individuals have MH/SA needs.
    - Academic Pediatrics (2017): ~30–40% of justice-involved youth have cognitive/educational needs.
    - NIJ Reentry Report (2020): ~20–30% face other supervision conditions (housing, medical, parenting).
    """

    if mode == "equalized":
        return {
            "Condition_MH_SA":  random.choices([True, False], weights=[0.65, 0.35])[0],
            "Condition_Cog_Ed": random.choices([True, False], weights=[0.35, 0.65])[0],
            "Condition_Other":  random.choices([True, False], weights=[0.25, 0.75])[0],
        }
    else:
        return {
            "Condition_MH_SA":  random.choices([True, False], weights=[0.70, 0.30])[0],
            "Condition_Cog_Ed": random.choices([True, False], weights=[0.40, 0.60])[0],
            "Condition_Other":  random.choices([True, False], weights=[0.30, 0.70])[0],
        }


def stratified_dependents(race: str, gender: str, mode="realistic") -> int:
    """
    Assigns number of dependents (0, 1, or 2) based on race and gender.
    In 'realistic' mode, uses research-backed probabilities.
    In 'equalized' mode, uses population-average weights to eliminate intake bias.
    """

    if mode == "equalized":
        return random.choices([0, 1, 2], weights=[0.35, 0.39, 0.26])[0]

    prob_map = {
        ("Female", "Black"):    0.80,
        ("Female", "White"):    0.70,
        ("Female", "Hispanic"): 0.75,
        ("Male",   "Black"):    0.60,
        ("Male",   "White"):    0.50,
        ("Male",   "Hispanic"): 0.55,
    }

    base_prob = prob_map.get((gender, race), 0.55)
    weights   = [1 - base_prob, base_prob * 0.6, base_prob * 0.4]
    return random.choices([0, 1, 2], weights=weights)[0]


# ── Stratified Offense Assignment (4-category collapse of V1) ───────────────
def stratified_offense(race: str, gender: str, mode="realistic") -> str:
    """
    Demographic-stratified offense assignment.

    Four categories: Violent, Drug, Property, Other(PublicOrder).

    Each per-demographic weight vector is the V1 5-category distribution
    with Violent-SexOffender + Violent/Non-Sex summed into a single
    Violent category. Original V1 source proportions preserved otherwise.

    Order: [Violent, Drug, Property, Other(PublicOrder)]

    Reference: BJS Prisoners in 2022 (NCJ 307149);
               BJS Recidivism of Prisoners Released in 2005 (NCJ 250975).
    """

    OFFENSES = ["Violent", "Drug", "Property", "Other(PublicOrder)"]

    # Per-row values are V1: [V-SO + V/NS, Drug, Property, Other]
    OFFENSE_WEIGHTS = {
        "Male": {
            # V1 Black male:    [0.05, 0.35, 0.20, 0.30, 0.10]
            #                   → Violent=0.40, Drug=0.20, Property=0.30, Other=0.10
            "Black":    [0.40, 0.20, 0.30, 0.10],

            # V1 White male:    [0.05, 0.30, 0.20, 0.30, 0.15]
            #                   → Violent=0.35, Drug=0.20, Property=0.30, Other=0.15
            "White":    [0.35, 0.20, 0.30, 0.15],

            # V1 Hispanic male: [0.05, 0.30, 0.25, 0.25, 0.15]
            #                   → Violent=0.35, Drug=0.25, Property=0.25, Other=0.15
            "Hispanic": [0.35, 0.25, 0.25, 0.15],
        },
        "Female": {
            # V1 Black female:    [0.02, 0.25, 0.30, 0.25, 0.18]
            #                     → Violent=0.27, Drug=0.30, Property=0.25, Other=0.18
            "Black":    [0.27, 0.30, 0.25, 0.18],

            # V1 White female:    [0.02, 0.28, 0.30, 0.25, 0.15]
            #                     → Violent=0.30, Drug=0.30, Property=0.25, Other=0.15
            "White":    [0.30, 0.30, 0.25, 0.15],

            # V1 Hispanic female: [0.02, 0.25, 0.35, 0.20, 0.18]
            #                     → Violent=0.27, Drug=0.35, Property=0.20, Other=0.18
            "Hispanic": [0.27, 0.35, 0.20, 0.18],
        },
    }

    # V1 default (population-average fallback) preserved, collapsed:
    # [0.04, 0.30, 0.25, 0.27, 0.14] → Violent=0.34, Drug=0.25, Property=0.27, Other=0.14
    DEFAULT_WEIGHTS = [0.34, 0.25, 0.27, 0.14]

    # Equalized mode: V1 used [0.05, 0.25, 0.30, 0.25, 0.15]
    # Collapsed: Violent=0.30, Drug=0.30, Property=0.25, Other=0.15
    if mode == "equalized":
        return random.choices(OFFENSES, weights=[0.30, 0.30, 0.25, 0.15])[0]

    weights = OFFENSE_WEIGHTS.get(gender, {}).get(race, DEFAULT_WEIGHTS)
    return random.choices(OFFENSES, weights=weights, k=1)[0]

# ── Stratified Education Assignment ──────────────────────────────────────────
def stratified_education(race: str, mode="realistic") -> str:

    LEVELS = ["Less than HighSchool Diploma", "High School Diploma", "College"]

    if mode == "equalized":
        return random.choices(LEVELS, weights=[0.39, 0.44, 0.17])[0]

    weights = (
        [0.45, 0.45, 0.10] if race == "Black"    else
        [0.30, 0.45, 0.25] if race == "White"    else
        [0.39, 0.44, 0.17]                         # Hispanic (matches aggregate)
    )
    return random.choices(LEVELS, weights=weights)[0]


# ── Agent Initialisation ──────────────────────────────────────────────────────
def generate_synthetic_agent(agent_id: int, mode="realistic") -> dict:

    # ── Demographics ─────────────────────────────────────────────────────────
    gender = (
        random.choice(["Male", "Female"]) if mode == "equalized"
        else random.choices(["Male", "Female"], weights=[0.93, 0.07])[0]
    )

    # Three groups only — White / Black / Hispanic.
    # Weights re-normalised from BJS proportions after removing "Other":
    #   Raw BJS: White 0.30, Black 0.33, Hispanic 0.23  (sum = 0.86)
    #   Normalised: White 0.349, Black 0.384, Hispanic 0.267
    # Source: BJS Prisoners in 2022 (NCJ 307149)
    RACE_GROUPS   = ["White", "Black", "Hispanic"]
    RACE_WEIGHTS  = [0.349,   0.384,   0.267]

    race = (
        random.choices(RACE_GROUPS, k=1)[0]          if mode == "equalized"
        else random.choices(RACE_GROUPS, weights=RACE_WEIGHTS, k=1)[0]
    )

    offense = stratified_offense(race, gender, mode)
    age_at_entry = random.randint(18, 50)

    return {
        "ID":               agent_id,
        "Gender":           gender,
        "Race":             race,
        "Age_at_Entry":     age_at_entry,
        "Age_at_Release":   age_at_entry,  # updated to actual release age in handle_free()
        "Education_Level":  stratified_education(race, mode),
        "offense":          offense,
        "Prison_Years":     0,

        # ── Social Risk Factors ───────────────────────────────────────────
        "Gang_Affiliated":   random.random() < 0.1,
        "Dependents":        stratified_dependents(race, gender, mode),
        "Residence_Changes": 0,

        # ── Behavioral Conditions ─────────────────────────────────────────
        **stratified_conditions(mode),

        # ── Placeholder for Simulation-Derived Traits ─────────────────────
        "Supervision_Risk_Score":            0,
        "Supervision_Level_First":           None,
        "Supervision_Term":                  0,
        "Prior_Conviction_Episodes_Felony":  0,
        "Prior_Conviction_Episodes_Misd":    0,
        "Prior_Conviction_Episodes_Violent": 0,
        "Prior_Conviction_Episodes_Property":0,
        "Prior_Conviction_Episodes_Drug":    0,
        "Prior_Revocations_Supervision":     False,

        # ── Supervision Violations ────────────────────────────────────────
        # Binary flag for technical violations (electronic monitoring failures,
        # instruction non-compliance, reporting lapses, unauthorised movement,
        # delinquency reports). Reflects ~45% of supervision revocations being
        # technical (Pew, 2019).
        "Violations_Technical":      0,
        "Delinquency_Reports":       0,

        # ── Program Participation ─────────────────────────────────────────
        "Program_Attendances":       0,
        "Program_UnexcusedAbsences": 0,

        # ── Employment Profile ────────────────────────────────────────────
        "Percent_Days_Employed": random.uniform(45.0, 65.0),

        # ── Recidivism Outcomes ───────────────────────────────────────────
        "Recidivism_Within_3years":  0,
        "Recidivism_Arrest_Year1":   0,
        "Recidivism_Arrest_Year2":   0,
        "Recidivism_Arrest_Year3":   0,
    }