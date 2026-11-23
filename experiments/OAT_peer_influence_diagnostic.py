#!/usr/bin/env python3
"""Measure actual peer exposure during Prison-phase risk-score evaluations.

Nonzero exposure is counted when Person.compute_adjusted_risk_score("Prison")
is called and the computed peer_influence_boost > 0. This is NOT the fraction
of all released prisoners or the fraction whose rearrest was caused by peers.

Run from recidivism_abm/experiments:
 python OAT_peer_influence_diagnostic.py --workers 4 --initial-agents 3000
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from recidivism_abm.agents.person import Person
from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    #get_uncalibrated_params,
    get_global_calibration_params,
)
SEEDS = [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087]
SCENARIOS = [
    ("peer_off", False, 0.06),
    ("zero_weight", True, 0.00),
    ("baseline_006", True, 0.06),
]
FIELDS = [
    "scenario", "seed", "max_peer_effect", "enabled",
    "prison_evaluations", "nonzero_evaluations", "pct_nonzero_evaluations",
    "distinct_prison_agents", "distinct_exposed_agents",
    "pct_exposed_prison_agents", "mean_boost_all_evaluations",
    "mean_boost_nonzero_evaluations", "max_observed_boost",
    "mean_exposure_share_nonzero", "rate_3yr", "rate_6yr", "rate_9yr",
]

def run_one(task):
    scenario, enabled, weight, seed, initial_agents, intake, warmup, study, outdir = task
    from recidivism_abm.model.recidivism_model import RecidivismModel
    from recidivism_abm.agents.person import Person
    from recidivism_abm.config.risk_config import (
        get_global_calibration_params, get_peer_influence_config,
    )

    counts = defaultdict(int)
    total_boost = 0.0
    nonzero_boost = 0.0
    max_boost = 0.0
    all_agents = set()
    exposed_agents = set()

    original = Person.compute_adjusted_risk_score

    def observed_adjusted_score(self, phase):
        nonlocal total_boost, nonzero_boost, max_boost
        result = original(self, phase)
        if phase == "Prison":
            counts["evaluations"] += 1
            all_agents.add(self.unique_id)
            boost = float(getattr(self, "peer_influence_boost", 0.0))
            if not math.isfinite(boost) or boost < 0:
                raise RuntimeError(f"Invalid peer boost {boost!r}")
            total_boost += boost
            max_boost = max(max_boost, boost)
            if boost > 1e-12:
                counts["nonzero"] += 1
                nonzero_boost += boost
                exposed_agents.add(self.unique_id)
        return result

    # Each worker runs one simulation at a time. Restore after completion.
    Person.compute_adjusted_risk_score = observed_adjusted_score
    try:
        peer_config = dict(get_peer_influence_config())
        peer_config["max_peer_effect"] = weight
        run_dir = Path(outdir) / "runs" / scenario / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = RecidivismModel(
            initial_agents=initial_agents,
            monthly_intake=intake,
            warmup_months=warmup,
            study_months=study,
            bias_factor=0.0,
            enable_peer_influence=enabled,
            peer_config=peer_config,
            calibration_params=get_global_calibration_params(),
            seed=seed,
            output_directory=str(run_dir),
        )
        if not math.isclose(
            float(model.peer_influence_weights["max_peer_effect"]), weight, abs_tol=1e-12
        ):
            raise RuntimeError("Model did not accept peer weight override")
        if hasattr(model, "export_csv"):
            model.export_csv = False
        model.max_months = warmup + study
        while model.running:
            model.step()
        n = counts["evaluations"]
        k = counts["nonzero"]
        if n == 0:
            raise RuntimeError(
                "No Prison-phase adjusted-risk evaluations observed. "
                "Check that the instrumented method is used in the current model."
            )
        rates = {}
        for yr in (3, 6, 9):
            value = float(model.calculate_flag_rate(f"rearrest_{yr}_yrs"))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError(f"Invalid {yr}-year rate: {value}")
            rates[f"rate_{yr}yr"] = value
        return {
            "scenario": scenario, "seed": seed, "max_peer_effect": weight,
            "enabled": enabled, "prison_evaluations": n,
            "nonzero_evaluations": k, "pct_nonzero_evaluations": 100*k/n,
            "distinct_prison_agents": len(all_agents),
            "distinct_exposed_agents": len(exposed_agents),
            "pct_exposed_prison_agents": 100*len(exposed_agents)/len(all_agents),
            "mean_boost_all_evaluations": total_boost/n,
            "mean_boost_nonzero_evaluations": nonzero_boost/k if k else 0.0,
            "max_observed_boost": max_boost,
            "mean_exposure_share_nonzero": (
                nonzero_boost/k/weight if k and weight > 0 else 0.0
            ),
            **rates,
        }
    finally:
        Person.compute_adjusted_risk_score = original

def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--initial-agents", type=int, default=3000)
    parser.add_argument("--monthly-intake", type=int, default=10)
    parser.add_argument("--warmup-months", type=int, default=144)
    parser.add_argument("--study-months", type=int, default=108)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--out-dir", default="OAT_Peer_Exposure_Diagnostic_Output")
    args = parser.parse_args()
    if args.workers < 1 or args.initial_agents < 1 or not args.seeds:
        parser.error("Workers, initial agents, and seeds must be positive")
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (scenario, enabled, weight, seed, args.initial_agents,
         args.monthly_intake, args.warmup_months, args.study_months, str(outdir))
        for scenario, enabled, weight in SCENARIOS for seed in args.seeds
    ]
    print(f"Running {len(tasks)} simulations ({len(SCENARIOS)} scenarios x "
          f"{len(args.seeds)} seeds) with {args.workers} workers", flush=True)
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{len(results)+len(errors)}/{len(tasks)}] "
                      f"{result['scenario']} seed={result['seed']}: "
                      f"exposed agents={result['pct_exposed_prison_agents']:.2f}%, "
                      f"mean nonzero boost={result['mean_boost_nonzero_evaluations']:.6f}",
                      flush=True)
            except Exception as exc:
                errors.append({"scenario": task[0], "seed": task[3], "error": str(exc)})
                print(f"ERROR {task[0]} seed={task[3]}: {exc}", flush=True)
    save_csv(outdir / "peer_exposure_per_seed.csv",
             sorted(results, key=lambda r: (r["scenario"], r["seed"])), FIELDS)
    if errors:
        import json
        (outdir / "errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(f"{len(errors)} simulations failed; see errors.json")
    summaries = []
    metric_names = [
        "pct_nonzero_evaluations", "pct_exposed_prison_agents",
        "mean_boost_all_evaluations", "mean_boost_nonzero_evaluations",
        "max_observed_boost", "mean_exposure_share_nonzero",
        "rate_3yr", "rate_6yr", "rate_9yr",
    ]
    for scenario, enabled, weight in SCENARIOS:
        subset = [r for r in results if r["scenario"] == scenario]
        summary = {"scenario": scenario, "enabled": enabled,
                   "max_peer_effect": weight, "n_seeds": len(subset)}
        for metric in metric_names:
            vals = [r[metric] for r in subset]
            summary[f"mean_{metric}"] = sum(vals)/len(vals)
            summary[f"sd_{metric}"] = (
                math.sqrt(sum((x-summary[f"mean_{metric}"])**2 for x in vals)/(len(vals)-1))
                if len(vals)>1 else 0.0
            )
        summaries.append(summary)
    save_csv(outdir / "peer_exposure_summary.csv", summaries, list(summaries[0]))
    print("\nSummary (mean across seeds):")
    for r in summaries:
        print(f"  {r['scenario']:<15} "
              f"agents exposed={r['mean_pct_exposed_prison_agents']:.2f}%  "
              f"nonzero evaluations={r['mean_pct_nonzero_evaluations']:.2f}%  "
              f"mean boost (nonzero)={r['mean_mean_boost_nonzero_evaluations']:.6f}")
    print(f"\nSaved CSVs to {outdir}")

if __name__ == "__main__":
    main()
