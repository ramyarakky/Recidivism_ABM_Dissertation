#!/usr/bin/env python3
"""Peer influence OAT sensitivity and ablation for the Recidivism ABM.

Uses the model's peer_config constructor argument; does not edit risk_config.py.
Each scenario uses identical seeds and the locked three-stage calibration.
The zero-weight/peer-enabled case is a diagnostic for the peer-off ablation.

Example:
    python peer_influence_sensitivity.py --workers 4
    python peer_influence_sensitivity.py --workers 8 --initial-agents 3000
"""
import argparse
import csv
import json
import math
import os
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    #get_uncalibrated_params,
    get_global_calibration_params,
)

SEEDS = [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087]
SCENARIOS = [
    ("peer_off", False, 0.06),
    ("zero", True, 0.00),
    ("low_003", True, 0.03),
    ("lower_004", True, 0.04),
    ("baseline_006", True, 0.06),
    ("upper_008", True, 0.08),
    ("high_009", True, 0.09),
    ("stress_012", True, 0.12),
]
BJS = {3: 0.684, 6: 0.794, 9: 0.834}


def run_one(task):
    scenario, enabled, weight, seed, initial_agents, intake, warmup, study, outdir = task
    from recidivism_abm.model.recidivism_model import RecidivismModel
    from recidivism_abm.config.risk_config import (
        get_global_calibration_params, get_peer_influence_config,
    )
    peer_config = dict(get_peer_influence_config())
    peer_config["max_peer_effect"] = float(weight)
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
    # Check that the model actually accepted the override.
    observed = model.peer_influence_weights["max_peer_effect"]
    if not math.isclose(observed, weight, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"Peer weight override failed: requested {weight}, got {observed}")
    if hasattr(model, "export_csv"):
        model.export_csv = False
    model.max_months = warmup + study
    while model.running:
        model.step()
    rates = {}
    for yr in (3, 6, 9):
        rate = model.calculate_flag_rate(f"rearrest_{yr}_yrs")
        if rate is None or not math.isfinite(float(rate)) or not 0 <= float(rate) <= 1:
            raise RuntimeError(f"Invalid {yr}-year rate for {scenario} seed {seed}: {rate}")
        rates[f"rate_{yr}yr"] = float(rate)
    return {"scenario": scenario, "enabled": enabled, "max_peer_effect": weight,
            "seed": seed, **rates}


def mean(vals):
    return sum(vals) / len(vals)


def sd(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_charts(summary, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV files still saved")
        return
    ordered = [r for r in summary if r["enabled"]]
    x = [r["max_peer_effect"] for r in ordered]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for yr, marker in ((3, "o"), (6, "s"), (9, "^")):
        y = [100*r[f"mean_{yr}yr"] for r in ordered]
        ax.plot(x, y, marker=marker, linewidth=1.8, label=f"{yr}-year rearrest")
    ax.axvline(0.06, color="gray", linestyle="--", label="Baseline weight = 0.06")
    ax.set_xlabel("Maximum peer-influence weight")
    ax.set_ylabel("Cumulative rearrest rate (%)")
    ax.set_title("Peer-influence weight sensitivity — calibrated ABM")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir)/"peer_weight_response.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for yr, marker in ((3, "o"), (6, "s"), (9, "^")):
        y = [r[f"paired_delta_{yr}yr_pp"] for r in ordered]
        ax.plot(x, y, marker=marker, linewidth=1.8, label=f"{yr}-year difference")
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0.06, color="gray", linestyle="--")
    ax.set_xlabel("Maximum peer-influence weight")
    ax.set_ylabel("Change vs peer OFF (percentage points)")
    ax.set_title("Paired peer-influence effect by weight")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir)/"peer_effect_vs_off.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--initial-agents", type=int, default=3000)
    p.add_argument("--monthly-intake", type=int, default=10)
    p.add_argument("--warmup-months", type=int, default=144)
    p.add_argument("--study-months", type=int, default=108)
    p.add_argument("--out-dir", default="OAT_Peer_Influence_Sensitivity_Output")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()
    if args.workers < 1 or args.initial_agents < 1 or args.monthly_intake < 0 or not args.seeds:
        p.error("workers, initial-agents and seeds must be positive; monthly-intake must be nonnegative")
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = [(name, enabled, weight, seed, args.initial_agents,
              args.monthly_intake, args.warmup_months, args.study_months, str(outdir))
             for name, enabled, weight in SCENARIOS for seed in args.seeds]
    print(f"Running {len(tasks)} simulations ({len(SCENARIOS)} scenarios x {len(args.seeds)} seeds), "
          f"{args.workers} workers", flush=True)
    start = time.monotonic()
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                print(f"[{len(results)+len(errors)}/{len(tasks)}] "
                      f"{result['scenario']} seed={result['seed']} "
                      f"3y={result['rate_3yr']:.2%}", flush=True)
            except Exception as exc:
                errors.append({"scenario": task[0], "seed": task[3], "error": str(exc)})
                print(f"ERROR {task[0]} seed={task[3]}: {exc}", flush=True)
    fields = ["scenario", "enabled", "max_peer_effect", "seed",
              "rate_3yr", "rate_6yr", "rate_9yr"]
    write_csv(outdir/"peer_influence_per_seed.csv",
              sorted(results, key=lambda r: (r["scenario"], r["seed"])), fields)
    if errors:
        (outdir/"errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(f"{len(errors)} simulations failed; no summary produced. See errors.json")

    by_scenario = {name: {r["seed"]: r for r in results if r["scenario"] == name}
                   for name, _, _ in SCENARIOS}
    reference = by_scenario["peer_off"]
    summary = []
    paired = []
    for name, enabled, weight in SCENARIOS:
        subset = by_scenario[name]
        row = {"scenario": name, "enabled": enabled, "max_peer_effect": weight,
               "n_seeds": len(subset)}
        for yr in (3, 6, 9):
            rates = [subset[s][f"rate_{yr}yr"] for s in args.seeds]
            diffs = [100*(subset[s][f"rate_{yr}yr"] - reference[s][f"rate_{yr}yr"])
                     for s in args.seeds]
            row[f"mean_{yr}yr"] = mean(rates)
            row[f"sd_{yr}yr"] = sd(rates)
            row[f"paired_delta_{yr}yr_pp"] = mean(diffs)
            row[f"paired_sd_{yr}yr_pp"] = sd(diffs)
            row[f"paired_se_{yr}yr_pp"] = sd(diffs)/math.sqrt(len(diffs))
        summary.append(row)
        for s in args.seeds:
            record = {"scenario": name, "seed": s, "max_peer_effect": weight}
            for yr in (3, 6, 9):
                record[f"delta_{yr}yr_pp"] = 100*(subset[s][f"rate_{yr}yr"] - reference[s][f"rate_{yr}yr"])
            paired.append(record)
    write_csv(outdir/"peer_influence_summary.csv", summary, list(summary[0].keys()))
    write_csv(outdir/"peer_influence_paired_differences.csv", paired, list(paired[0].keys()))
    make_charts(summary, outdir)
    print("\nResults (mean paired difference vs peer OFF, percentage points):")
    for r in summary:
        print(f"  {r['scenario']:<16} weight={r['max_peer_effect']:.2f} "
              f"3y={r['paired_delta_3yr_pp']:+.3f} "
              f"6y={r['paired_delta_6yr_pp']:+.3f} "
              f"9y={r['paired_delta_9yr_pp']:+.3f}")
    print(f"\nSaved to {outdir} in {(time.monotonic()-start)/60:.1f} min")
    print("Note: paired SE is provided; compute t-based CIs for dissertation reporting.")
    print("Check that peer_off and zero produce comparable outcomes; otherwise investigate the toggle path.")


if __name__ == "__main__":
    main()
