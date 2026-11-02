# stress_test_dashboard_with_GA.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os, sys, datetime
import io
from pandas import ExcelWriter
import itertools
import uuid
import math
import random

# Ensure imports work relative to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_peer_influence_config, get_flat_risk_weights

# -----------------------
# CONFIG
# -----------------------

ALL_KEYS = [
    "Age_at_Release", "Percent_Days_Employed", "Dependents",
    "Education_None", "Education_LessthanHSdiploma", "Education_HighSchoolDiploma",
    "Education_Atleastsomecollege", "Gang_Affiliated", "Residence_Changes",
    "offense_Violent-SexOffender", "offense_Violent/Non-Sex", "offense_Drug",
    "offense_Property", "offense_Other(PublicOrder)", "Supervision_Risk_Score",
    "Supervision_Level_First_Standard", "Supervision_Level_First_High",
    "Supervision_Level_First_Specialized", "Prior_Conviction_Episodes_Violent",
    "Prior_Conviction_Episodes_Property", "Prior_Conviction_Episodes_Drug",
    "Prior_Conviction_Episodes_Misd", "Prior_Conviction_Episodes_Felony",
    "Prior_Revocations_Supervision", "Condition_MH_SA", "Condition_Cog_Ed",
    "Condition_Other", "Violations_Technical", "Program_Attendances",
    "Program_UnexcusedAbsences", "Threshold", "Desistance_Rate"
]

PEER_CONFIG = get_peer_influence_config()
TARGET_YEARS = [3, 6, 9]

# -----------------------
# MODEL RUNNER
# -----------------------

def run_model(weights, seed, agents, warmup=144, study=108, enable_peer=True):
    """
    Run a single model instance with the provided weights and seed.
    Returns recid dict for TARGET_YEARS.
    """
    w = weights.copy()
    m = RecidivismModel(
        initial_agents=agents,
        bias_factor=w.get("Bias_Factor", 0),
        monthly_intake=10,
        warmup_months=warmup,
        study_months=study,
        enable_peer_influence=enable_peer,
        weights=w,
        peer_config=PEER_CONFIG,
        seed=seed,
        mode="realistic"
    )
    while m.running:
        m.step()

    recid = {y: m.calculate_flag_rate(f"rearrest_{y}_yrs") for y in TARGET_YEARS}
    non_recid = {y: 1 - recid[y] for y in TARGET_YEARS}
    return {"recid": recid, "non_recid": non_recid}

# -----------------------
# SCORING / OBJECTIVE
# -----------------------

def score_against_targets(model_recid, target_recid, metric="absolute"):
    """
    model_recid: dict {year: rate}
    target_recid: dict {year: target_rate}
    metric: 'absolute' -> sum absolute differences
            'squared' -> sum squared differences
            'relative' -> sum absolute relative diffs
    Returns scalar score (higher -> worse).
    """
    s = 0.0
    for y in TARGET_YEARS:
        m = model_recid[y]
        t = target_recid.get(y, 0.0)
        if metric == "absolute":
            s += abs(m - t)
        elif metric == "squared":
            s += (m - t) ** 2
        elif metric == "relative":
            denom = max(t, 1e-6)
            s += abs(m - t) / denom
        else:
            s += abs(m - t)
    return s

# -----------------------
# PARAMETER BUILD HELPERS
# -----------------------

def apply_multipliers_to_weights(base_weights, multipliers_map):
    """
    base_weights: dict
    multipliers_map: dict {feature: multiplier}
    Returns a new weights dict where base_weights[feature] *= multiplier.
    If base weight is zero, create a small base scaled by multiplier.
    """
    w = base_weights.copy()
    for f, mul in multipliers_map.items():
        base_val = w.get(f, 0.0)
        if base_val == 0.0:
            # create a small base so multiplier still has effect
            w[f] = (np.sign(mul) if mul != 0 else 0.0) * (abs(mul) * 1.0e-2)
        else:
            w[f] = base_val * mul
    return w

# -----------------------
# EVALUATION / STRESS SWEEP
# -----------------------

def evaluate_parameter_set(weights, seeds, agents, target_recid, enable_peer=True, n_jobs=None):
    """
    Run the model across seeds (in parallel) with a given weights dict and compute:
      - average recid (dict)
      - score against target
    Returns dict with recid, non_recid, score
    """
    if n_jobs is None:
        n_jobs = min(cpu_count(), max(1, len(seeds)))

    run_results = Parallel(n_jobs=n_jobs)(
        delayed(run_model)(weights, s, agents, enable_peer=enable_peer) for s in seeds
    )

    avg_recid = {y: np.mean([r["recid"][y] for r in run_results]) for y in TARGET_YEARS}
    avg_nonrecid = {y: np.mean([r["non_recid"][y] for r in run_results]) for y in TARGET_YEARS}
    score = score_against_targets(avg_recid, target_recid, metric="absolute")
    return {"weights": weights, "recid": avg_recid, "non_recid": avg_nonrecid, "score": score}

# -----------------------
# RANDOM + GRID (kept for completeness)
# -----------------------

def generate_extreme_grid(base_weights, features_to_vary, multipliers):
    combos = []
    for combo in itertools.product(multipliers, repeat=len(features_to_vary)):
        mp = {feat: mul for feat, mul in zip(features_to_vary, combo)}
        combos.append(apply_multipliers_to_weights(base_weights, mp))
    return combos

def generate_random_perturbations(base_weights, n_samples=200, scale_range=(-10.0, 10.0), features_subset=None, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    feats = list(features_subset) if features_subset else ALL_KEYS
    samples = []
    for _ in range(n_samples):
        mp = {}
        for f in feats:
            mul = rng.uniform(scale_range[0], scale_range[1])
            mp[f] = mul
        samples.append(apply_multipliers_to_weights(base_weights, mp))
    return samples

# -----------------------
# GENETIC ALGORITHM (Adversarial Search)
# -----------------------

def initialize_population(base_weights, features, pop_size, multiplier_bounds, rng):
    """
    Initialize population as list of multiplier maps (dicts: feature->multiplier).
    multiplier_bounds: (min, max)
    """
    pop = []
    for _ in range(pop_size):
        mp = {f: rng.uniform(multiplier_bounds[0], multiplier_bounds[1]) for f in features}
        pop.append(mp)
    return pop

def crossover(parent_a, parent_b, rng):
    """Uniform crossover between two multiplier maps -> child"""
    child = {}
    for k in parent_a.keys():
        child[k] = parent_a[k] if rng.rand() < 0.5 else parent_b[k]
    return child

def mutate(indiv, mutation_rate, mutation_scale, rng):
    """
    Mutate an individual's multipliers:
      - with probability mutation_rate per gene, add gaussian noise scaled by mutation_scale
    """
    child = indiv.copy()
    for k in child.keys():
        if rng.rand() < mutation_rate:
            # gaussian perturbation
            noise = rng.normal(loc=0.0, scale=mutation_scale)
            child[k] = child[k] + noise
    return child

# -----------------------
# GENETIC ALGORITHM (Adversarial Search)
# -----------------------

def tournament_selection(pop_with_scores, k=3, rng=None):
    """Select one individual using k-tournament selection (higher score = better)"""
    if rng is None:
        rng = np.random.RandomState()
    
    # Fix: select indices instead of full objects to avoid 1D error
    indices = rng.choice(len(pop_with_scores), size=min(k, len(pop_with_scores)), replace=False)
    contenders = [pop_with_scores[i] for i in indices]
    
    best = max(contenders, key=lambda x: x[1])
    return best[0]

def genetic_adversary_search(base_weights,
                             features,
                             agents,
                             seeds,
                             target_recid,
                             pop_size=80,
                             generations=25,
                             multiplier_bounds=(-10.0, 10.0),
                             crossover_rate=0.8,
                             mutation_rate=0.15,
                             mutation_scale=1.0,
                             elite_frac=0.05,
                             rng_seed=0,
                             n_jobs=None,
                             enable_peer=True):
    """
    Run a simple GA to maximize score_against_targets (i.e., find parameter sets that break the model).
    Returns sorted DataFrame of evaluated candidates (descending score).
    """
    rng = np.random.RandomState(rng_seed)
    if n_jobs is None:
        n_jobs = max(1, min(cpu_count(), len(seeds)))

    # initialize population as multiplier maps
    population = initialize_population(base_weights, features, pop_size, multiplier_bounds, rng)

    evaluated = []  # store tuples (mult_map, score, recid, nonrecid, weights_full)

    # helper to evaluate a multiplier map -> returns dict with score etc.
    def eval_multiplier_map(mult_map):
        weights_full = apply_multipliers_to_weights(base_weights, mult_map)
        res = evaluate_parameter_set(weights_full, seeds, agents, target_recid, enable_peer=enable_peer, n_jobs=n_jobs)
        return {"mult_map": mult_map, "score": res["score"], "recid": res["recid"], "non_recid": res["non_recid"], "weights": res["weights"]}

    # evaluate initial population
    with st.spinner("Evaluating initial GA population..."):
        pop_results = [eval_multiplier_map(ind) for ind in population]
    evaluated.extend(pop_results)

    # GA loop
    for gen in range(generations):
        # sort by score descending (higher = more broken)
        pop_results = sorted(pop_results, key=lambda x: x["score"], reverse=True)
        elites_count = max(1, int(math.ceil(elite_frac * pop_size)))
        elites = [p["mult_map"] for p in pop_results[:elites_count]]

        # build mating pool (we'll use tournament selection)
        pop_with_scores = [(p["mult_map"], p["score"]) for p in pop_results]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            # parent selection
            parent_a = tournament_selection(pop_with_scores, k=3, rng=rng)
            parent_b = tournament_selection(pop_with_scores, k=3, rng=rng)
            # crossover
            if rng.rand() < crossover_rate:
                child = crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()
            # mutation
            child = mutate(child, mutation_rate, mutation_scale, rng)
            # clip child multipliers to multiplier_bounds
            for k in child:
                child[k] = float(np.clip(child[k], multiplier_bounds[0], multiplier_bounds[1]))
            new_pop.append(child)

        # evaluate new population
        with st.spinner(f"GA generation {gen+1}/{generations}: evaluating {len(new_pop)} individuals..."):
            pop_results = [eval_multiplier_map(ind) for ind in new_pop]
        evaluated.extend(pop_results)

    # convert evaluated to DataFrame rows
    rows = []
    for r in evaluated:
        row = {
            "id": str(uuid.uuid4())[:8],
            "score": r["score"]
        }
        for y in TARGET_YEARS:
            row[f"recid_{y}yr"] = r["recid"][y]
            row[f"nonrecid_{y}yr"] = r["non_recid"][y]
        # include a few weight snippets if present
        for k in ["Age_at_Release","Percent_Days_Employed","Gang_Affiliated","Residence_Changes","Condition_MH_SA","Violations_Technical","Program_Attendances"]:
            row[f"w_{k}"] = r["weights"].get(k, np.nan)
        row["weights_full"] = r["weights"]
        row["mult_map"] = r["mult_map"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


# -----------------------
# STREAMLIT DASHBOARD (Stress Test + GA)
# -----------------------

def stress_test_dashboard():
    st.title("🛠️ Stress Test: Break the Recidivism Model (with GA)")

    st.markdown("This tool searches for parameter sets (extreme + random + GA) that produce recidivism rates far from your target rates. Use the Genetic Algorithm (GA) to search adversarially.")

    # basic run controls
    agents = st.number_input("Agents", min_value=100, max_value=5000, value=1000, step=100)
    seed_start = st.number_input("Seed Start", min_value=1, max_value=9999, value=10)
    seed_end = st.number_input("Seed End", min_value=seed_start+1, max_value=10000, value=15)
    seeds = list(range(seed_start, seed_end))

    # targets
    st.subheader("Target recidivism rates (for scoring)")
    t3 = st.number_input("3-year target", min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.2f")
    t6 = st.number_input("6-year target", min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.2f")
    t9 = st.number_input("9-year target", min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.2f")
    target_recid = {3: t3, 6: t6, 9: t9}

    base_weights = get_flat_risk_weights()

    # mode selection: Grid/Random (existing) or GA adversary
    mode = st.radio("Search mode", options=["Grid+Random", "Genetic Adversary (GA)"], index=1)

    if mode == "Grid+Random":
        st.subheader("Grid + Random configuration")
        grid_feat_input = st.text_input("Grid features (comma separated)", value="Age_at_Release,Percent_Days_Employed,Gang_Affiliated,Residence_Changes,Condition_MH_SA,Violations_Technical,Program_Attendances")
        grid_features = [s.strip() for s in grid_feat_input.split(",") if s.strip()]
        grid_multipliers_text = st.text_input("Grid multipliers (comma separated)", value="-10,-5,-2,-1,-0.5,0,0.5,1,2,5,10")
        grid_multipliers = [float(x.strip()) for x in grid_multipliers_text.split(",") if x.strip()]
        random_samples = st.number_input("Random samples", min_value=0, max_value=5000, value=300)
        random_scale_min = st.number_input("Random scale min", value=-10.0, format="%.2f")
        random_scale_max = st.number_input("Random scale max", value=10.0, format="%.2f")
        rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0)
        run_button = st.button("Run Grid+Random Stress Test")

        if run_button:
            if not seeds:
                st.error("No seeds selected. Set Seed Start < Seed End.")
                return

            start_time = datetime.datetime.now()
            status = st.empty()
            status.text("Building candidate parameter sets...")

            # candidates
            candidates = []
            candidates += generate_extreme_grid(base_weights, grid_features, grid_multipliers)
            candidates += generate_random_perturbations(base_weights, n_samples=int(random_samples), scale_range=(float(random_scale_min), float(random_scale_max)), features_subset=None, rng=np.random.RandomState(int(rng_seed)))

            # dedupe
            unique = {}
            for w in candidates:
                items = tuple(sorted((k, float(v)) for k, v in w.items()))
                unique[items] = w
            candidate_list = list(unique.values())

            # evaluate sequentially (each evaluation runs seeds in parallel)
            rows = []
            for i, w in enumerate(candidate_list, start=1):
                status.text(f"Evaluating candidate {i}/{len(candidate_list)}")
                res = evaluate_parameter_set(w, seeds, agents, target_recid, enable_peer=True)
                row = {"id": str(uuid.uuid4())[:8], "score": res["score"]}
                for y in TARGET_YEARS:
                    row[f"recid_{y}yr"] = res["recid"][y]
                    row[f"nonrecid_{y}yr"] = res["non_recid"][y]
                for k in ["Age_at_Release","Percent_Days_Employed","Gang_Affiliated","Residence_Changes","Condition_MH_SA","Violations_Technical","Program_Attendances"]:
                    row[f"w_{k}"] = res["weights"].get(k, np.nan)
                row["weights_full"] = res["weights"]
                rows.append(row)

            df_results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            status.text(f"Grid+Random stress test complete in {elapsed:.1f}s — {len(df_results)} candidates evaluated.")
            st.subheader("Top results")
            st.dataframe(df_results.head(20))

            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", csv_bytes, file_name="stress_test_grid_random_results.csv")

    else:
        # GA mode
        st.subheader("Genetic Adversary (GA) configuration")
        features_input = st.text_input("GA feature set (comma separated)", value="Age_at_Release,Percent_Days_Employed,Gang_Affiliated,Residence_Changes,Condition_MH_SA,Violations_Technical,Program_Attendances")
        ga_features = [s.strip() for s in features_input.split(",") if s.strip()]

        pop_size = st.number_input("Population size", min_value=4, max_value=1000, value=80)
        generations = st.number_input("Generations", min_value=1, max_value=500, value=20)
        multiplier_min = st.number_input("Multiplier min", value=-10.0, format="%.2f")
        multiplier_max = st.number_input("Multiplier max", value=10.0, format="%.2f")
        crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.8)
        mutation_rate = st.slider("Mutation rate (per gene)", 0.0, 1.0, 0.15)
        mutation_scale = st.number_input("Mutation scale (stddev)", value=1.0, format="%.2f")
        elite_frac = st.slider("Elite fraction", 0.0, 0.5, 0.05)
        rng_seed = st.number_input("GA random seed", min_value=0, max_value=999999, value=0)
        ga_n_jobs = st.number_input("Per-evaluation parallel jobs (per candidate)", min_value=1, max_value=cpu_count(), value=min(cpu_count(), max(1, len(seeds))))

        run_ga = st.button("Run Genetic Adversary Search")

        if run_ga:
            if not seeds:
                st.error("No seeds selected. Set Seed Start < Seed End.")
                return
            if multiplier_min >= multiplier_max:
                st.error("Multiplier min must be < multiplier max.")
                return

            start_time = datetime.datetime.now()
            st.info("Starting GA adversarial search. This may take a while depending on pop/gens and model runtime.")
            df_results = genetic_adversary_search(
                base_weights=base_weights,
                features=ga_features,
                agents=agents,
                seeds=seeds,
                target_recid=target_recid,
                pop_size=int(pop_size),
                generations=int(generations),
                multiplier_bounds=(float(multiplier_min), float(multiplier_max)),
                crossover_rate=float(crossover_rate),
                mutation_rate=float(mutation_rate),
                mutation_scale=float(mutation_scale),
                elite_frac=float(elite_frac),
                rng_seed=int(rng_seed),
                n_jobs=int(ga_n_jobs),
                enable_peer=True
            )

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            st.success(f"GA complete — evaluated {len(df_results)} individual evaluations in {elapsed:.1f}s")
            st.subheader("Top GA adversarial findings")
            st.dataframe(df_results.head(20))

            worst = df_results.iloc[0]
            st.subheader("Worst candidate detail (top 1)")
            st.write(f"Score: {worst['score']:.4f}")
            for y in TARGET_YEARS:
                st.write(f"  {y}-year: {worst[f'recid_{y}yr']:.4f}  (target: {target_recid[y]:.4f})")
            st.json({"weights_snippet": {k: worst.get(f"w_{k}", None) for k in ["Age_at_Release","Percent_Days_Employed","Gang_Affiliated","Residence_Changes","Condition_MH_SA","Violations_Technical","Program_Attendances"]}, "mult_map": worst["mult_map"]})

            # exports
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("Download GA Results CSV", csv_bytes, file_name="stress_test_ga_results.csv")
            output = io.BytesIO()
            with ExcelWriter(output, engine="xlsxwriter") as writer:
                df_results.to_excel(writer, sheet_name="GAResults", index=False)
                worksheet = writer.sheets["GAResults"]
                worksheet.set_column(0, 0, 14)
                worksheet.set_column(1, 6, 14)
            output.seek(0)
            st.download_button("Download GA Excel Workbook", output, file_name="stress_test_ga_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    stress_test_dashboard()
