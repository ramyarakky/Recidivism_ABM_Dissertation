"""
calibration_bjs.py

DE-based calibration engine to tune your RecidivismModel to BJS 3/6/9-year benchmarks.
Assumes run_model(weights_dict, seed, agents, warmup, study) exists and returns
a dict with keys 3,6,9 for recidivism means per run (matching your earlier code).
"""

import os
import json
import csv
import time
import math
import logging
import numpy as np
from functools import partial
from multiprocessing import cpu_count
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed

# -------------------------
# Logging + output folder
# -------------------------
os.makedirs("Calibration_Multiseed", exist_ok=True)
logging.basicConfig(
    filename="Calibration_Multiseed/full_calibration.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log_txt_path = "Calibration_Multiseed/log.txt"
log_csv_template = "Calibration_Multiseed/{stage}_history.csv"

# Keep a human-readable tab-delimited running log
log_file = open(log_txt_path, "a")
if os.stat(log_txt_path).st_size == 0:
    # write header if empty
    log_file.write("Iteration\t" +
                   "Params\tRecidivism_3yr\tRecidivism_6yr\tRecidivism_9yr\tLoss\tTime\n")
    log_file.flush()


# -------------------------
# BJS Targets (fixed)
# -------------------------
BJS_TARGETS = {3: 0.68, 6: 0.79, 9: 0.83}

# Weights for loss (you can change to emphasize 3-year more)
LOSS_WEIGHTS = {3: 0.5, 6: 0.3, 9: 0.2}


# -------------------------
# DEFAULT BASE WEIGHTS
# Insert your provided weights as baseline defaults
# (only keys shown here that matter for calibration — others will be kept in model)
# -------------------------
DEFAULT_WEIGHTS = {
    'Age_at_Release': 0.73,
    'Percent_Days_Employed': -0.95,
    'Dependents': 0.1605,
    'Education_None': -0.048999992331901866,
    'Education_LessthanHSdiploma': 0.154,
    'Education_HighSchoolDiploma': 0.00016,
    'Education_Atleastsomecollege': -0.08,
    'offense_Violent-SexOffender': -0.205,
    'offense_Violent/Non-Sex': 0.099,
    'offense_Drug': -0.047,
    'offense_Property': 0.137,
    'offense_Other(PublicOrder)': 0.089,
    'Prior_Conviction_Episodes_Violent': 0.086,
    'Prior_Conviction_Episodes_Property': 0.027,
    'Prior_Conviction_Episodes_Drug': 0.026,
    'Prior_Conviction_Episodes_Misd': 0.39,
    'Prior_Conviction_Episodes_Felony': 0.242,
    # Global/others
    'Desistance_Rate': 0.00667,
    "Violations_Technical": 0.8,
    "Program_UnexcusedAbsences": 0.4,
    "Program_Attendances": -0.4,
    "Prior_Revocations_Supervision": 0.4,
    "Residence_Changes": 0.2,
    "Condition_MH_SA": 0.4,
    "Condition_Cog_Ed": -0.0,
    "Condition_Other": 0.1,
    "Supervision_Level_First_Standard": -0.1,
    "Supervision_Level_First_High": 0.1,
    "Supervision_Level_First_Specialized": 0.1,
    "Supervision_Risk_Score": 0.0,
    "Gang_Affiliated": 0.8,
    "Threshold": 0.10,
    "Bias_Factor": 0.03,
    # mapping parameters (not necessarily in your original base; added so evaluate uses them)
    "Floor_Prob": 0.05,
    "Logistic_Center": 0.60,
    "Logistic_Scale": 10.0
}

# -------------------------
# Tunable parameter list + bounds
# Ordered list -> differential_evolution expects vector
# -------------------------
TUNABLE_PARAMS = [
    ("Program_Attendances", -0.4, (-1.0, 0.1)),
    ("Program_UnexcusedAbsences", 0.4, (0.0, 1.0)),
    ("Violations_Technical", 0.8, (0.0, 2.0)),
    ("Prior_Revocations_Supervision", 0.4, (0.0, 1.0)),
    ("Gang_Affiliated", 0.8, (0.0, 2.0)),
    ("Supervision_Risk_Score", 0.0, (-0.5, 0.5)),
    ("Threshold", 0.10, (0.05, 0.30)),
    ("Bias_Factor", 0.03, (0.0, 0.10)),
    ("Desistance_Rate", 0.00667, (0.002, 0.05)),
    ("Floor_Prob", 0.05, (0.01, 0.20)),
    ("Logistic_Center", 0.60, (0.45, 0.70)),
    ("Logistic_Scale", 10.0, (6.0, 20.0))
]

# helper: generate initial vector and bounds for DE
initial_vector = [p[1] for p in TUNABLE_PARAMS]
de_bounds = [p[2] for p in TUNABLE_PARAMS]
param_names = [p[0] for p in TUNABLE_PARAMS]


# -------------------------
# Wrapper to apply parameter vector to a weights dict
# -------------------------
def params_to_weights_dict(x_vector, base_weights=None):
    base = dict(DEFAULT_WEIGHTS) if base_weights is None else dict(base_weights)
    for name, val in zip(param_names, x_vector):
        base[name] = float(val)
    return base


# -------------------------
# Loss function used by DE
# - runs evaluate() across multiple seeds (replicates)
# - computes weighted squared error to BJS targets
# -------------------------
def calibration_loss(x_vector,
                     seeds=(100,101,102),
                     agents=400,
                     warmup=144,
                     study=108,
                     verbose=False):
    """
    x_vector: array-like parameter vector in same order as param_names.
    Returns scalar loss (lower is better).
    """
    start_time = time.time()
    weights_dict = params_to_weights_dict(x_vector)

    # NOTE: run_model must exist in your environment (import or same module)
    # run_model(weights_dict, seed, agents, warmup, study) -> returns dict with keys 3,6,9
    # We run multiple seeds and average the resulting recidivism rates.
    try:
        from __main__ import run_model  # if script run inline
    except Exception:
        # fallback: import if run_model is in another module (adjust if needed)
        try:
            from model_runner import run_model  # example alternative
        except Exception as exc:
            raise RuntimeError("run_model not found in __main__ or model_runner. "
                               "Place run_model in same script or adjust import.") from exc

    # parallel run across seeds
    n_jobs = min(cpu_count(), len(seeds))
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_model)(weights_dict, int(s), agents, warmup, study) for s in seeds
    )

    # compute mean recidivism across seeds
    mean_rates = {}
    for k in [3, 6, 9]:
        vals = [res[k] for res in results]
        mean_rates[k] = float(np.mean(vals))

    # weighted squared error loss
    loss = sum(LOSS_WEIGHTS[k] * (mean_rates[k] - BJS_TARGETS[k]) ** 2 for k in BJS_TARGETS)

    elapsed = time.time() - start_time

    # Logging - CSV and human log
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # prepare readable params
    param_pairs = ",".join(f"{name}={val:.5f}" for name, val in zip(param_names, x_vector))
    log_line = (f"{timestamp}\tLoss:{loss:.6f}\t3yr:{mean_rates[3]:.4f}\t6yr:{mean_rates[6]:.4f}\t"
                f"9yr:{mean_rates[9]:.4f}\tTime:{elapsed:.1f}s\tParams:{param_pairs}\n")
    log_file.write(log_line)
    log_file.flush()
    logging.info(log_line.strip())

    if verbose:
        print(log_line.strip())

    return float(loss)


# -------------------------
# Run differential evolution
# -------------------------
def run_de_calibration(popsize=15, maxiter=40, seeds=(100,101,102), agents=300, warmup=144, study=108):
    """
    Executes DE. Adjust popsize/maxiter to trade runtime vs search thoroughness.
    """
    print("Starting DE calibration to BJS targets (this may take a while)...")
    callback_iter = {"count": 0}

    def de_callback(xk, convergence=None):
        callback_iter["count"] += 1
        it = callback_iter["count"]
        # Evaluate metrics for printing
        try:
            loss_val = calibration_loss(xk, seeds=seeds, agents=agents, warmup=warmup, study=study, verbose=False)
        except Exception as e:
            loss_val = float("nan")
        print(f"[DE] Iter {it} | Loss {loss_val:.6f}")

    result = differential_evolution(
        func=partial(calibration_loss, seeds=seeds, agents=agents, warmup=warmup, study=study),
        bounds=de_bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        updating='deferred',
        workers=-1,
        callback=de_callback
    )

    # final best parameters
    best_x = result.x
    best_weights = params_to_weights_dict(best_x)

    # Save best weights & the DE result
    with open("Calibration_Multiseed/bjs_calibrated_weights.json", "w") as f:
        json.dump(best_weights, f, indent=2)

    with open("Calibration_Multiseed/bjs_de_result.json", "w") as f:
        json.dump({
            "x": best_x.tolist(),
            "fun": float(result.fun),
            "nit": int(result.nit),
            "message": result.message
        }, f, indent=2)

    print("DE complete. Best loss:", result.fun)
    print("Saved calibrated weights to Calibration_Multiseed/bjs_calibrated_weights.json")
    return best_weights, result


# -------------------------
# If run as script, execute calibration
# -------------------------
if __name__ == "__main__":
    # quick-run settings - tune these to control runtime
    POPSIZE = 12
    MAXITER = 30
    SEEDS = list(range(200, 203))  # 3 replicates by default
    AGENTS = 400
    WARMUP = 144
    STUDY = 108

    try:
        best_weights, de_result = run_de_calibration(popsize=POPSIZE,
                                                     maxiter=MAXITER,
                                                     seeds=SEEDS,
                                                     agents=AGENTS,
                                                     warmup=WARMUP,
                                                     study=STUDY)
    finally:
        log_file.close()
