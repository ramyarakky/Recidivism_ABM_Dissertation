import random

def generate_synthetic_agent(agent_id: int) -> dict:
    """
    Generates synthetic agent data for recidivism simulation.
    Variables reflect criminological theory:
    - Risk-Need-Responsivity (RNR)
    - Social Learning Theory (SLT)
    - Procedural Justice (PJ)
    - Empirical predictors from BJS and meta-analyses
    """

    return {
        # ───────────── Identification ─────────────
        "ID": agent_id,

        # ───────────── Demographics ─────────────
        #"Gender": random.choice(["Male", "Female"]),
        #"Race": random.choice(["White", "Black", "Hispanic", "Other"]),
        "Gender" : random.choices(["Male", "Female"], weights=[0.93, 0.07])[0],
        "Race" : random.choices(["White", "Black", "Hispanic", "Other"], weights=[0.57, 0.38, 0.20, 0.05])[0],

        "Age_at_Entry": random.randint(18,50),  # Older age → lower recidivism (BJS)
        "Age_at_Release": 0,
        # ───────────── Social Risk Factors ─────────────
        "Gang_Affiliated": random.random() < 0.1,  # SLT: gang ties increase peer modeling risk
        "Dependents": random.randint(0, 2),         # RNR: family bonds reduce risk
        "Residence_Changes": random.randint(0, 5),  # Housing instability → higher risk
        "Education_Level": random.choices(
            ["Less than HighScool Diploma", "High School Diploma", "College"],
            weights=[0.39, 0.44, 0.17]
        )[0],  # Higher education → lower recidivism

        # ───────────── Supervision Profile ─────────────
        "Supervision_Risk_Score": 0, #random.randint(1, 10),  # RNR: static risk score
        "Supervision_Level_First": None, #random.choices(
            #["Standard", "High", "Specialized"],
            #weights=[0.6, 0.3, 0.1]
        #)[0],  # PJ: supervision intensity affects compliance
        "Supervision_Term": 0,
        # ───────────── Criminal Offense ─────────────
        "offense": random.choices(
            ["Violent-SexOffender", "Violent/Non-Sex", "Drug", "Property", "Other(PublicOrder)"],
            weights=[0.05, 0.25, 0.3, 0.25, 0.15]
        )[0],  # BJS: offense type predicts rearrest likelihood
        "Prison_Years": 0, #random.randint(1, 10),  # Longer incarceration → mixed effects

        # ───────────── Arrest History ─────────────
        "Prior_Arrest_Episodes_Felony": 0, #random.randint(0, 2),
        "Prior_Arrest_Episodes_Misd": 0, #random.randint(0, 2),
        "Prior_Arrest_Episodes_Violent": 0, #random.randint(0, 2),
        "Prior_Arrest_Episodes_Property": 0, # random.randint(0, 2),
        "Prior_Arrest_Episodes_Drug": 0, # random.randint(0, 2),
        "Prior_Arrest_Episodes_PPViolationCharges": 0, # random.randint(0, 2),
        "Prior_Arrest_Episodes_DVCharges": 0, # random.randint(0, 2),
        "Prior_Arrest_Episodes_GunCharges": 0, #random.randint(0, 2),

        # ───────────── Conviction History ─────────────
        "Prior_Conviction_Episodes_Felony": 0, #random.randint(0, 2),
        "Prior_Conviction_Episodes_Misd": 0, #random.randint(0, 2),
        "Prior_Conviction_Episodes_Violent": 0, #random.randint(0, 2),
        "Prior_Conviction_Episodes_Property": 0, #random.randint(0, 2),
        "Prior_Conviction_Episodes_Drug": 0, #random.randint(0, 2),
        "Prior_Conviction_Episodes_PPViolationCharges": 0, #random.randint(0, 2),
        "Prior_Conviction_Episodes_DVCharges": 0, #random.randint(0, 2),
        "Prior_Conviction_Episodes_GunCharges": 0, #random.randint(0, 2),

        # ───────────── Revocations ─────────────
        "Prior_Revocations_Parole": False, #random.choice([True, False]),
        "Prior_Revocations_Probation":False, #random.choice([True, False]),

        # ───────────── Behavioral Conditions ─────────────
        "Condition_MH_SA": random.choice([True, False]), # Mental health or substance abuse
        "Condition_Cog_Ed": random.choice([True, False]), # Cognitive/educational deficits
        "Condition_Other": random.choice([True, False]),  # Other behavioral flags

        # ───────────── Supervision Violations ─────────────
        "Violations_ElectronicMonitoring": 0, #random.randint(0, 1),
        "Violations_Instruction": 0, #random.randint(0, 1),
        "Violations_FailToReport": 0, #random.randint(0, 1),
        "Violations_MoveWithoutPermission": 0, #random.randint(0, 1),
        "Delinquency_Reports": 0, #random.randint(0, 1),  # PJ: procedural violations

        # ───────────── Program Participation ─────────────
        "Program_Attendances": 0, #random.randint(0, 5),  # RNR: rehab reduces risk
        "Program_UnexcusedAbsences": 0,  #random.randint(0, 3),  # Noncompliance increases risk

        # ───────────── Drug Testing ─────────────
        "Avg_Days_per_DrugTest": random.randint(10, 60),
        "DrugTests_THC_Positive": random.randint(0, 1),
        "DrugTests_Cocaine_Positive": random.randint(0, 1),
        "DrugTests_Meth_Positive": random.randint(0, 1),
        "DrugTests_Other_Positive": random.randint(0, 1),

        # ───────────── Employment Profile ─────────────
        "Percent_Days_Employed": 0,
        "Jobs_Per_Year": random.randint(0, 5),
        "Employment_Exempt": random.random() < 0.1,  # Disability, age, etc.

        # ───────────── Recidivism Outcomes (for benchmarking only) ─────────────
        "Recidivism_Within_3years": 0,
        "Recidivism_Arrest_Year1": 0,
        "Recidivism_Arrest_Year2": 0,
        "Recidivism_Arrest_Year3": 0
    }
