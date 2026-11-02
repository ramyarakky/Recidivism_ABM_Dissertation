"""
phase2_bias_all_channels.py  —  Phase 2: Intersectional Bias (all_channels scope)
==================================================================================
Recidivism ABM — PhD Dissertation Tool

HYPOTHESIS TESTED
-----------------
bias_scope = "all_channels"  →  structural embedding hypothesis
  Disparity arises across ALL pipeline stages:
    trial + sentence + supervision_assign + detect + supervision_rearrest
  Predicted BJS signal: racial and gender gaps PERSIST through year 9
  (vs supervision_only where gaps decay as agents exit supervision)

COMPARISON WITH supervision_only
---------------------------------
  This script is the structural-embedding comparison arm.
  Run AFTER phase2_bias_supervision_only.py completes Stages 1–3.
  Compare calibrated bias values and gap trajectories between the two
  scope conditions to test which hypothesis better fits BJS data.

  Key expected differences:
  - Smaller absolute bias values needed (more pipeline stages active)
  - Female minimum expected at a less extreme log-odds value
  - 9yr gaps should be larger / closer to BJS under all_channels

CHANGES VS supervision_only
----------------------------
  1. BIAS_SCOPE = "all_channels"
  2. output_directory = "phase2_bias_all_channels_output"
  3. GENDER_SWEEP_COARSE extended to -5.0 (was -3.0)
     Rationale: all_channels activates trial-stage effects which can
     produce large Female-protective bias values at small log-odds.
     Previous supervision_only minimum was interior at ~-1.4;
     all_channels minimum may be shallower — extend for safety.
  4. All docstrings updated to reflect scope and hypothesis.

BIAS ARCHITECTURE (v6 — aligned with _apply_bias() refactor)
--------------------------------------------------------------
All bias routing handled exclusively by _apply_bias(phase) in person.py.
The bias_scope enum is the SOLE routing mechanism:

  "supervision_only"  →  detect + supervision_rearrest only
  "all_channels"      →  also activates trial + sentence + supervision_assign

REMOVED (stale) KEYS:
  bias_on_trial        → replaced by bias_scope routing in _apply_bias()
  bias_on_tech_detect  → replaced by _apply_bias("detect")
  These keys are no longer read by person.py.

PERFORMANCE OPTIMIZATIONS (identical to supervision_only v6)
-------------------------------------------------------------
  COARSE: 500 agents × 5 seeds × 5 reps = 25 runs/point
  FINE:   1000 agents × 10 seeds × 10 reps = 100 runs/point
  Expected runtime: ~3.5 hr on 124-core HPC
"""

import os
import sys
import json
import math
import warnings
from typing import Dict, Optional, Set
import multiprocessing
warnings.filterwarnings("ignore")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_global_calibration_params,
    get_peer_influence_config,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "initial_agents":        1000,   # fine pass + validation
    "monthly_intake":        10,
    "warmup_months":         144,
    "study_months":          108,
    "enable_peer_influence": True,
    "output_directory":      "phase2_bias_all_channels_output",
    "N_REPS":                10,
    "SEEDS":                 [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "bjs_targets":           {3: 0.684, 6: 0.794, 9: 0.834},
}

# ── Coarse-pass reduced settings ─────────────────────────────────────────────
AGENTS_COARSE = 500
SEEDS_COARSE  = CONFIG["SEEDS"][:5]

# ── BJS subgroup targets ──────────────────────────────────────────────────────
# Source: Alper, M., Durose, M.R. & Markman, J. (2018).
#         2018 Update on Prisoner Recidivism: A 9-Year Follow-up Period.
#         BJS NCJ 250975. Table 8 (race), Table 9 (gender).
BJS_SUBGROUP = {
    3: {"Male": 0.68, "Female": 0.58,
        "White": 0.66, "Black": 0.72, "Hispanic": 0.68},
    6: {"Male": 0.79, "Female": 0.70,
        "White": 0.77, "Black": 0.82, "Hispanic": 0.79},
    9: {"Male": 0.84, "Female": 0.77,
        "White": 0.81, "Black": 0.87, "Hispanic": 0.84},
}

# ── Baseline offsets ──────────────────────────────────────────────────────────
# Populated at runtime by measure_baseline_offsets(). Do not set manually.
# FEMALE_OFFSET_OVERRIDE = 0.0: treat Female as direct BJS target (no offset).
# Under all_channels scope, additional trial/sentence pathways are active even
# at bias_factor=0, so the zero-bias baseline Female rate may differ from the
# supervision_only baseline. The live measurement captures this correctly.
# Setting 0.0 treats BJS 58% as the direct calibration target regardless.
FEMALE_OFFSET_OVERRIDE = 0.0  # type: Optional[float]
BASELINE_OFFSETS: dict  = {}
ADJUSTED_TARGETS: dict  = {}
_BASELINE_MEASURED: bool = False

# ── Sweep ranges ──────────────────────────────────────────────────────────────
RACE_REFERENCE   = "White"
GENDER_REFERENCE = "Male"
BIAS_FACTOR      = 1.0

# ── BIAS_SCOPE: "all_channels" (structural embedding hypothesis) ──────────────
# This is the ONLY difference from supervision_only that affects model output.
# All payloads use this value. CLI --scope override is available but defaults
# to all_channels — changing it here would defeat the purpose of this script.
BIAS_SCOPE = "all_channels"

RACE_SWEEP = {
    "Black":    np.round(np.arange(0.0,  1.51, 0.1),  2).tolist(),
    "Hispanic": np.round(np.arange(-0.6, 0.41, 0.05), 2).tolist(),
}
BLACK_SWEEP_FINE_HALF_WIDTH = 0.20
BLACK_SWEEP_FINE_STEP       = 0.02

# Extended to -5.0 for all_channels.
# Rationale: under all_channels, the Female-protective effect is distributed
# across more pipeline stages, so the MAE minimum may require a larger
# negative log-odds than under supervision_only (-1.4 typical). -5.0 gives
# ample headroom; if minimum is near -5.0, extend to -7.0 and rerun.
GENDER_SWEEP_COARSE          = np.round(np.arange(-5.0, 0.51, 0.1), 2).tolist()
GENDER_SWEEP_FINE_HALF_WIDTH = 0.30
GENDER_SWEEP_FINE_STEP       = 0.02
GENDER_SWEEP                 = {"Female": GENDER_SWEEP_COARSE}   # legacy compat

N_REPS_COARSE        = 5
N_REPS_FINE          = 10
FINE_REPS_MULTIPLIER = N_REPS_FINE // CONFIG["N_REPS"]   # legacy compat

SENSITIVITY_LEVELS = np.round(np.arange(0.6, 1.41, 0.1), 2).tolist()

# ── HPC config ────────────────────────────────────────────────────────────────
_HPC_CORES         = 128
_MEM_PER_CPU_GB    = 4
_OS_OVERHEAD_CORES = 4
_MAX_WORKERS       = _HPC_CORES - _OS_OVERHEAD_CORES   # = 124

_WORKER_BATCH_SIZE        = 20
_WORKER_BATCH_SIZE_COARSE = 25

COLORS = {
    "Male": "#2E75B6", "Female": "#ED7D31",
    "White": "#70AD47", "Black": "#C00000", "Hispanic": "#7030A0",
    "Other": "#808080", "bjs": "#1A3D5C", "sim": "#2E75B6", "grid": "#E8E8E8",
}
WINDOWS = [3, 6, 9]

# ── Dissertation-quality figure defaults ─────────────────────────────────────
DISS_DPI        = 300
DISS_TITLE      = 16
DISS_SUBTITLE   = 13
DISS_AXIS       = 12
DISS_TICK       = 10
DISS_LEGEND     = 10
DISS_NOTE       = 9

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": DISS_TICK,
    "axes.titlesize": DISS_SUBTITLE,
    "axes.labelsize": DISS_AXIS,
    "xtick.labelsize": DISS_TICK,
    "ytick.labelsize": DISS_TICK,
    "legend.fontsize": DISS_LEGEND,
    "figure.titlesize": DISS_TITLE,
    "figure.dpi": 120,
    "savefig.dpi": DISS_DPI,
})


# =============================================================================
# PAYLOAD BUILDER — single source of truth for bias keys
# =============================================================================

def _build_payload(bias_factor: float, group_bias_race: dict,
                   group_bias_gender: dict, bias_scope: str,
                   task_id: float = 0.0, agents: int = None) -> dict:
    """
    Centralised payload constructor.

    bias_scope="all_channels" activates all pipeline stages in _apply_bias():
      trial + sentence + supervision_assign + detect + supervision_rearrest

    REMOVED vs v5: bias_on_trial, bias_on_tech_detect (stale keys).
    bias_scope is the sole routing key — see _apply_bias() in person.py.
    """
    return {
        "__task_id__":           task_id,
        "__warmup__":            CONFIG["warmup_months"],
        "__study__":             CONFIG["study_months"],
        "__agents__":            agents if agents is not None else CONFIG["initial_agents"],
        "__intake__":            CONFIG["monthly_intake"],
        "__bias_factor__":       float(bias_factor),
        "__group_bias_race__":   dict(group_bias_race),
        "__group_bias_gender__": dict(group_bias_gender),
        "bias_scope":            bias_scope,
        "tech_detect_base":      0.30,
    }


# =============================================================================
# WORKER
# =============================================================================

def _worker(task: tuple) -> tuple:
    """
    Execute a batch of seeds for one payload configuration.

    all_channels scope: _apply_bias() returns non-zero for ALL phases —
    trial, sentence, supervision_assign, detect, supervision_rearrest.
    This implements the structural embedding hypothesis (Nellis, 2016).

    Performance changes vs v5: single tmpdir per batch, cal built once per
    batch, export_csv=False immediately after construction, BLAS threads=1.
    """
    import os as _os
    for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_env] = "1"

    import tempfile, shutil

    payload, seeds = task
    payload = dict(payload)

    task_id = float(payload.pop("__task_id__", 0.0))
    warmup  = int(payload.pop("__warmup__",   CONFIG["warmup_months"]))
    study   = int(payload.pop("__study__",    CONFIG["study_months"]))
    agents  = int(payload.pop("__agents__",   CONFIG["initial_agents"]))
    intake  = int(payload.pop("__intake__",   CONFIG["monthly_intake"]))

    cal = get_global_calibration_params()
    cal["bias_factor"]       = float(payload.pop("__bias_factor__", BIAS_FACTOR))
    cal["group_bias_race"]   = payload.pop("__group_bias_race__",
                                           {"White": 0.0, "Black": 0.0,
                                            "Hispanic": 0.0, "Other": 0.0})
    cal["group_bias_gender"] = payload.pop("__group_bias_gender__",
                                           {"Male": 0.0, "Female": 0.0})
    # "all_channels" — sole bias routing key, replaces stale bias_on_trial etc.
    cal["bias_scope"]       = payload.pop("bias_scope", BIAS_SCOPE)
    cal["tech_detect_base"] = float(payload.pop("tech_detect_base", 0.30))

    peer_config = get_peer_influence_config()
    weights     = get_flat_risk_weights()
    results     = []

    _batch_tmp = tempfile.mkdtemp(prefix=f"abm_batch_{int(task_id * 1000)}_")
    try:
        for seed in seeds:
            try:
                model = RecidivismModel(
                    initial_agents        = agents,
                    bias_factor           = cal["bias_factor"],
                    monthly_intake        = intake,
                    warmup_months         = warmup,
                    study_months          = study,
                    enable_peer_influence = CONFIG["enable_peer_influence"],
                    weights               = weights,
                    peer_config           = peer_config,
                    calibration_params    = cal,
                    seed                  = seed,
                )
                model.export_csv = False
                while model.running:
                    model.step()
                results.append(_extract_subgroup_rates(model))
            except Exception as e:
                print(f"  Warning: Worker error (tid={task_id:.4f}, seed={seed}): {e}",
                      flush=True)
    finally:
        shutil.rmtree(_batch_tmp, ignore_errors=True)

    return (task_id, results)


# =============================================================================
# SUBGROUP EXTRACTION
# =============================================================================

def _extract_subgroup_rates(model) -> dict:
    from collections import defaultdict

    eligible = [a for a in model.schedule.agents
                if getattr(a, "study_eligible_agent", False)]
    if not eligible:
        return {}

    groups = defaultdict(list)
    for a in eligible:
        race   = getattr(a, "Race",   None)
        gender = getattr(a, "Gender", None)
        groups["__all__"].append(a)
        if race:            groups[race].append(a)
        if gender:          groups[gender].append(a)
        if race and gender: groups[f"{race}_{gender}"].append(a)

    windows_flags = {3: "rearrest_3_yrs", 6: "rearrest_6_yrs", 9: "rearrest_9_yrs"}
    rates = {}
    for win, flag in windows_flags.items():
        rates[win] = {}
        for key, members in groups.items():
            n = len(members)
            if n == 0:
                continue
            flagged = sum(1 for a in members if getattr(a, flag, False))
            out_key = "Overall" if key == "__all__" else key
            rates[win][out_key] = flagged / n
    return rates


# =============================================================================
# BATCH RUNNER
# =============================================================================

def _run_batch(overrides_list: list, n_workers: int, desc: str = "Running",
               n_reps_override: int = None, batch_size: int = None,
               seeds_override: list = None) -> dict:
    seeds  = seeds_override if seeds_override is not None else CONFIG["SEEDS"]
    n_reps = n_reps_override if n_reps_override is not None else CONFIG["N_REPS"]
    _bs    = batch_size if batch_size is not None else _WORKER_BATCH_SIZE

    tasks = []
    for task_id, payload in overrides_list:
        d = dict(payload)
        d["__task_id__"] = float(task_id)
        all_seeds = [seed + rep * 10_000
                     for rep in range(n_reps) for seed in seeds]
        for i in range(0, len(all_seeds), _bs):
            tasks.append((d, all_seeds[i : i + _bs]))

    raw = {float(tid): [] for tid, _ in overrides_list}

    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc=f"  {desc}", ncols=82) as bar:
            for fut in as_completed(futures):
                try:
                    tid, run_results = fut.result()
                    if tid in raw:
                        raw[tid].extend(run_results)
                    else:
                        nearest = min(raw.keys(), key=lambda k: abs(k - tid))
                        raw[nearest].extend(run_results)
                except Exception as e:
                    print(f"\n  Warning: Future error: {e}", flush=True)
                finally:
                    bar.update(1)

    return raw


# =============================================================================
# AGGREGATION + METRICS
# =============================================================================

def _aggregate_subgroup_rates(runs: list) -> dict:
    if not runs:
        return {}
    all_groups = set()
    for r in runs:
        for win in r:
            all_groups.update(r[win].keys())
    result = {}
    for win in WINDOWS:
        result[win] = {}
        for grp in all_groups:
            vals = [r[win][grp] for r in runs
                    if win in r and r[win].get(grp) is not None]
            if not vals:
                result[win][grp] = {"mean": None, "ci95": None, "n": 0}
                continue
            arr  = np.array(vals, dtype=np.float64)
            n    = len(arr)
            mean = float(arr.mean())
            std  = float(arr.std(ddof=min(1, n - 1)))
            sem  = std / math.sqrt(n) if n > 1 else std
            tc   = float(scipy_stats.t.ppf(0.975, df=max(1, n - 1)))
            result[win][grp] = {"mean": mean, "ci95": tc * sem, "n": n}
    return result


def _compute_disparity_metrics(agg: dict, ref_race: str = "White",
                                ref_gender: str = "Male") -> dict:
    metrics = {}
    for win in WINDOWS:
        metrics[win] = {}
        ref_race_mean   = agg[win].get(ref_race,   {}).get("mean") or 0.0
        ref_gender_mean = agg[win].get(ref_gender, {}).get("mean") or 0.0
        baseline        = agg[win].get("Overall",  {}).get("mean") or 0.0
        for grp, vals in agg[win].items():
            mean = vals.get("mean")
            if mean is None:
                continue
            ref = ref_gender_mean if grp in ("Male", "Female") else (
                  baseline if "_" in grp else ref_race_mean)
            metrics[win][grp] = {
                "mean":    mean,
                "ci95":    vals.get("ci95", 0.0),
                "gap":     round(mean - ref, 4),
                "ratio":   round(mean / ref, 4) if ref > 0 else None,
                "abs_dev": round(mean - baseline, 4),
            }
    return metrics


# =============================================================================
# BASELINE OFFSET MEASUREMENT
# =============================================================================

def measure_baseline_offsets(n_workers: int, outdir: str) -> None:
    """
    Zero-bias baseline measurement under all_channels scope.

    NOTE: The all_channels zero-bias baseline may differ from supervision_only
    because additional pipeline pathways are active even at bias_factor=0.
    The live measurement captures the correct all_channels baseline rates.

    FEMALE_OFFSET_OVERRIDE = 0.0: calibrate directly to BJS 58% target.
    """
    global BASELINE_OFFSETS, ADJUSTED_TARGETS, _BASELINE_MEASURED

    cp = os.path.join(outdir, "baseline_offsets.json")
    if os.path.exists(cp):
        print("  > Baseline offsets — loading checkpoint")
        with open(cp) as f:
            saved = json.load(f)
        BASELINE_OFFSETS = {int(k): v for k, v in saved["offsets"].items()}
        ADJUSTED_TARGETS = {int(k): v for k, v in saved["adjusted"].items()}
        _BASELINE_MEASURED = True
        for win in WINDOWS:
            o = BASELINE_OFFSETS[win]
            print(f"     {win}yr  Black:{o['Black']:+.4f}  Hispanic:{o['Hispanic']:+.4f}  "
                  f"Female:{o['Female']:+.4f}  White:{o['White']:+.4f}  Male:{o['Male']:+.4f}")
        return

    print(f"\n  > Measuring baseline offsets (all_channels scope, "
          f"{len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps = "
          f"{len(SEEDS_COARSE) * N_REPS_COARSE} runs, {AGENTS_COARSE} agents)...")

    payload = _build_payload(
        bias_factor=0.0,
        group_bias_race={"White": 0.0, "Black": 0.0, "Hispanic": 0.0, "Other": 0.0},
        group_bias_gender={"Male": 0.0, "Female": 0.0},
        # Use all_channels even for baseline — this captures any structural
        # rate differences that arise when all pipeline stages are active,
        # ensuring the offset is computed under the correct conditions.
        bias_scope=BIAS_SCOPE,
        task_id=0.0,
        agents=AGENTS_COARSE,
    )

    raw = _run_batch(
        [(0.0, payload)], n_workers,
        desc="Baseline measurement",
        n_reps_override=N_REPS_COARSE,
        batch_size=_WORKER_BATCH_SIZE_COARSE,
        seeds_override=SEEDS_COARSE,
    )
    agg = _aggregate_subgroup_rates(raw.get(0.0, []))

    if not agg or not all(win in agg for win in WINDOWS):
        raise RuntimeError(
            "Baseline measurement produced no results — all workers failed.\n"
            "Check worker errors above.")

    groups = ["Male", "Female", "White", "Black", "Hispanic"]
    offsets = {}
    adjusted = {}
    for win in WINDOWS:
        offsets[win] = {}
        adjusted[win] = {}
        for grp in groups:
            sim = agg[win].get(grp, {}).get("mean")
            bjs = BJS_SUBGROUP[win].get(grp)
            if sim is not None and bjs is not None:
                raw_offset = round(sim - bjs, 4)
                offsets[win][grp]  = (FEMALE_OFFSET_OVERRIDE
                                      if grp == "Female" and FEMALE_OFFSET_OVERRIDE is not None
                                      else raw_offset)
                adjusted[win][grp] = round(bjs - offsets[win][grp], 4)
            else:
                offsets[win][grp]  = 0.0
                adjusted[win][grp] = bjs or 0.0

    BASELINE_OFFSETS = offsets
    ADJUSTED_TARGETS = adjusted
    _BASELINE_MEASURED = True

    with open(cp, "w") as f:
        json.dump({
            "offsets":         {str(k): v for k, v in offsets.items()},
            "adjusted":        {str(k): v for k, v in adjusted.items()},
            "measured_at":     pd.Timestamp.now().isoformat(),
            "seeds_used":      SEEDS_COARSE,
            "agents_used":     AGENTS_COARSE,
            "female_override": FEMALE_OFFSET_OVERRIDE,
            "bias_scope":      BIAS_SCOPE,
        }, f, indent=2)

    print(f"  OK  Baseline offsets measured -> {cp}")
    for win in WINDOWS:
        o = BASELINE_OFFSETS[win]
        print(f"     {win}yr  Black:{o['Black']:+.4f}  Hispanic:{o['Hispanic']:+.4f}  "
              f"Female:{o['Female']:+.4f}  White:{o['White']:+.4f}  Male:{o['Male']:+.4f}")


# =============================================================================
# STAGE 1 — RACE CALIBRATION
# =============================================================================

def _sweep_race_group(race_group: str, sweep_vals: list, n_workers: int,
                      desc: str, n_reps_override: int = None,
                      is_coarse: bool = False) -> list:
    """
    Sweep race bias under all_channels scope.
    All stages active — gap expected to persist through year 9.
    Source: Nellis (2016); Western (2006).
    """
    agents = AGENTS_COARSE if is_coarse else CONFIG["initial_agents"]
    seeds  = SEEDS_COARSE  if is_coarse else CONFIG["SEEDS"]
    bs     = _WORKER_BATCH_SIZE_COARSE if is_coarse else _WORKER_BATCH_SIZE

    overrides_list = []
    for i, val in enumerate(sweep_vals):
        race_bias = {"White": 0.0, "Black": 0.0, "Hispanic": 0.0, "Other": 0.0}
        race_bias[race_group] = float(val)
        payload = _build_payload(
            bias_factor=BIAS_FACTOR,
            group_bias_race=race_bias,
            group_bias_gender={"Male": 0.0, "Female": 0.0},
            bias_scope=BIAS_SCOPE,
            task_id=float(i),
            agents=agents,
        )
        overrides_list.append((float(i), payload))

    raw = _run_batch(overrides_list, n_workers, desc=desc,
                     n_reps_override=n_reps_override, batch_size=bs,
                     seeds_override=seeds)

    results = []
    for i, val in enumerate(sweep_vals):
        agg = _aggregate_subgroup_rates(raw.get(float(i), []))
        if not agg:
            continue
        maes = [abs(agg[win].get(race_group, {}).get("mean", 0) - ADJUSTED_TARGETS[win].get(race_group, 0))
                for win in WINDOWS
                if agg[win].get(race_group, {}).get("mean") is not None
                and ADJUSTED_TARGETS[win].get(race_group) is not None]
        results.append({"value": val, "mae": float(np.mean(maes)) if maes else 999.0, "agg": agg})

    results.sort(key=lambda x: x["mae"])
    return results


def run_stage1_race(n_workers: int, outdir: str) -> dict:
    print("\n" + "=" * 70)
    print("  STAGE 1 — RACE CALIBRATION  (two-pass for Black, all_channels)")
    print("=" * 70)

    best_params = {}

    for grp in ["Hispanic", "Black"]:
        cp = os.path.join(outdir, f"stage1_{grp}.json")
        if os.path.exists(cp):
            print(f"  > {grp} — loading checkpoint")
            with open(cp) as f:
                best_params[grp] = json.load(f)
            continue

        if grp == "Hispanic":
            sweep_vals = RACE_SWEEP["Hispanic"]
            print(f"\n  > Sweeping {grp}  ({len(sweep_vals)} pts, coarse only)")
            results = _sweep_race_group(grp, sweep_vals, n_workers,
                                        f"Stage1 {grp}",
                                        n_reps_override=N_REPS_COARSE,
                                        is_coarse=True)
            best = results[0]
            best_params[grp] = {
                "value":       best["value"], "mae": best["mae"],
                "agg":         {str(k): v for k, v in best["agg"].items()},
                "all_results": [{"value": r["value"], "mae": r["mae"]} for r in results],
                "adjusted_target": {str(w): ADJUSTED_TARGETS[w][grp] for w in WINDOWS},
                "baseline_offset": {str(w): BASELINE_OFFSETS[w][grp] for w in WINDOWS},
            }
            print(f"  OK  Best {grp} = {best['value']:.2f}  (MAE={best['mae']:.4f})")

        else:  # Black — two pass
            coarse_vals = RACE_SWEEP["Black"]
            print(f"\n  Pass 1 (coarse): {len(coarse_vals)} pts  "
                  f"[{AGENTS_COARSE} agents, {len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")
            coarse_results = _sweep_race_group(grp, coarse_vals, n_workers,
                                               "Stage1 Black coarse",
                                               n_reps_override=N_REPS_COARSE,
                                               is_coarse=True)
            coarse_best = coarse_results[0]
            print(f"  OK  Coarse best = {coarse_best['value']:.2f}  "
                  f"(MAE={coarse_best['mae']:.4f})")

            hw = BLACK_SWEEP_FINE_HALF_WIDTH
            fine_vals = np.round(np.arange(
                round(coarse_best["value"] - hw, 3),
                round(coarse_best["value"] + hw + BLACK_SWEEP_FINE_STEP, 3),
                BLACK_SWEEP_FINE_STEP), 3).tolist()
            print(f"\n  Pass 2 (fine):   {len(fine_vals)} pts  "
                  f"[{CONFIG['initial_agents']} agents, "
                  f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")
            fine_results = _sweep_race_group(grp, fine_vals, n_workers,
                                             "Stage1 Black fine",
                                             n_reps_override=N_REPS_FINE,
                                             is_coarse=False)
            fine_best = fine_results[0]
            print(f"  OK  Fine best = {fine_best['value']:.3f}  "
                  f"(MAE={fine_best['mae']:.4f})")

            best_params[grp] = {
                "value":          fine_best["value"], "mae": fine_best["mae"],
                "agg":            {str(k): v for k, v in fine_best["agg"].items()},
                "coarse_results": [{"value": r["value"], "mae": r["mae"]} for r in coarse_results],
                "fine_results":   [{"value": r["value"], "mae": r["mae"]} for r in fine_results],
                "all_results":    [{"value": r["value"], "mae": r["mae"]} for r in coarse_results],
                "adjusted_target":{str(w): ADJUSTED_TARGETS[w][grp] for w in WINDOWS},
                "baseline_offset":{str(w): BASELINE_OFFSETS[w][grp] for w in WINDOWS},
                "coarse_best":    coarse_best["value"],
                "coarse_best_mae":coarse_best["mae"],
            }

        with open(cp, "w") as f:
            json.dump(best_params[grp], f, indent=2)

    return best_params


# =============================================================================
# STAGE 2 — GENDER CALIBRATION
# =============================================================================

def _sweep_female(race_bias: dict, sweep_vals: list, n_workers: int,
                  desc: str, n_reps_override: int = None,
                  is_coarse: bool = False) -> list:
    """
    Sweep Female bias under all_channels scope.

    bias_scope=all_channels — matches Stage 1 scope. Calibrated race bias
    values are only valid under the scope they were identified with.

    Under all_channels, Female-protective bias operates across trial,
    sentencing, and supervision stages simultaneously. This reflects the
    chivalry hypothesis in its broadest form (Daly, 1994; Steffensmeier
    et al., 1993; Doerner & Demuth, 2010).
    """
    agents = AGENTS_COARSE if is_coarse else CONFIG["initial_agents"]
    seeds  = SEEDS_COARSE  if is_coarse else CONFIG["SEEDS"]
    bs     = _WORKER_BATCH_SIZE_COARSE if is_coarse else _WORKER_BATCH_SIZE

    overrides_list = []
    for i, val in enumerate(sweep_vals):
        payload = _build_payload(
            bias_factor=BIAS_FACTOR,
            group_bias_race=race_bias,
            group_bias_gender={"Male": 0.0, "Female": float(val)},
            bias_scope=BIAS_SCOPE,
            task_id=float(i),
            agents=agents,
        )
        overrides_list.append((float(i), payload))

    raw = _run_batch(overrides_list, n_workers, desc=desc,
                     n_reps_override=n_reps_override, batch_size=bs,
                     seeds_override=seeds)

    results = []
    for i, val in enumerate(sweep_vals):
        agg = _aggregate_subgroup_rates(raw.get(float(i), []))
        if not agg:
            continue
        maes = [abs(agg[win].get("Female", {}).get("mean", 0) - ADJUSTED_TARGETS[win].get("Female", 0))
                for win in WINDOWS
                if agg[win].get("Female", {}).get("mean") is not None
                and ADJUSTED_TARGETS[win].get("Female") is not None]
        results.append({"value": val, "mae": float(np.mean(maes)) if maes else 999.0, "agg": agg})

    results.sort(key=lambda x: x["mae"])
    return results


def run_stage2_gender(race_params: dict, n_workers: int, outdir: str) -> dict:
    print("\n" + "=" * 70)
    print("  STAGE 2 — GENDER CALIBRATION  (two-pass, all_channels scope)")
    print("=" * 70)

    cp = os.path.join(outdir, "stage2_gender.json")
    if os.path.exists(cp):
        print("  > Female — loading checkpoint")
        with open(cp) as f:
            return json.load(f)

    race_bias = {
        "White": 0.0,
        "Black":    float(race_params.get("Black",    {}).get("value", 0.0)),
        "Hispanic": float(race_params.get("Hispanic", {}).get("value", 0.0)),
        "Other": 0.0,
    }
    print(f"  Race bias fixed at: {race_bias}")

    coarse_vals = GENDER_SWEEP_COARSE
    print(f"\n  Pass 1 (coarse): {len(coarse_vals)} pts  "
          f"[{AGENTS_COARSE} agents, {len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")
    print(f"  Range: {coarse_vals[0]:.1f} → {coarse_vals[-1]:.1f}  "
          f"(extended to -5.0 for all_channels scope)")
    coarse_results = _sweep_female(race_bias, coarse_vals, n_workers,
                                   "Stage2 Female coarse",
                                   n_reps_override=N_REPS_COARSE,
                                   is_coarse=True)
    coarse_best = coarse_results[0]
    print(f"  OK  Coarse best = {coarse_best['value']:.2f}  "
          f"(MAE={coarse_best['mae']:.4f})")

    # Boundary check — if minimum is at or near -5.0, the true minimum
    # may be outside the sweep range. Flag this for the user.
    if coarse_best["value"] <= coarse_vals[0] + 0.20:
        print(f"  ⚠  WARNING: Coarse minimum near left boundary ({coarse_vals[0]:.1f}).")
        print(f"     True minimum may be below {coarse_vals[0]:.1f}.")
        print(f"     Re-run with: GENDER_SWEEP_COARSE extended to -7.0")
        print(f"     Command: python phase2_bias_all_channels.py --stage 2 --rerun")

    hw = GENDER_SWEEP_FINE_HALF_WIDTH
    fine_vals = np.round(np.arange(
        round(coarse_best["value"] - hw, 3),
        round(coarse_best["value"] + hw + GENDER_SWEEP_FINE_STEP, 3),
        GENDER_SWEEP_FINE_STEP), 3).tolist()
    print(f"\n  Pass 2 (fine):   {len(fine_vals)} pts  "
          f"[{CONFIG['initial_agents']} agents, "
          f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")
    fine_results = _sweep_female(race_bias, fine_vals, n_workers,
                                 "Stage2 Female fine",
                                 n_reps_override=N_REPS_FINE,
                                 is_coarse=False)
    fine_best = fine_results[0]
    print(f"  OK  Fine best = {fine_best['value']:.3f}  "
          f"(MAE={fine_best['mae']:.4f})")

    best_params = {"Female": {
        "value":          fine_best["value"], "mae": fine_best["mae"],
        "agg":            {str(k): v for k, v in fine_best["agg"].items()},
        "coarse_results": [{"value": r["value"], "mae": r["mae"]} for r in coarse_results],
        "fine_results":   [{"value": r["value"], "mae": r["mae"]} for r in fine_results],
        "all_results":    [{"value": r["value"], "mae": r["mae"]} for r in coarse_results],
        "adjusted_target":{str(w): ADJUSTED_TARGETS[w]["Female"] for w in WINDOWS},
        "baseline_offset":{str(w): BASELINE_OFFSETS[w]["Female"] for w in WINDOWS},
        "coarse_best":    coarse_best["value"],
        "coarse_best_mae":coarse_best["mae"],
    }}
    with open(cp, "w") as f:
        json.dump(best_params, f, indent=2)
    return best_params


# =============================================================================
# STAGE 3 — INTERSECTIONAL VALIDATION + SENSITIVITY
# =============================================================================

def run_stage3_intersectional(race_params: dict, gender_params: dict,
                               n_workers: int, outdir: str) -> dict:
    print("\n" + "=" * 70)
    print("  STAGE 3 — INTERSECTIONAL VALIDATION + SENSITIVITY (all_channels)")
    print("=" * 70)

    race_bias = {
        "White": 0.0,
        "Black":    float(race_params.get("Black",    {}).get("value", 0.0)),
        "Hispanic": float(race_params.get("Hispanic", {}).get("value", 0.0)),
        "Other": 0.0,
    }
    gender_bias = {
        "Male": 0.0,
        "Female": float(gender_params.get("Female", {}).get("value", 0.0)),
    }
    print(f"  Calibrated race bias:   {race_bias}")
    print(f"  Calibrated gender bias: {gender_bias}")

    cp_val = os.path.join(outdir, "stage3_validation.json")
    if os.path.exists(cp_val):
        print("  > Validation — loading checkpoint")
        with open(cp_val) as f:
            validation = json.load(f)
    else:
        print("\n  > Running intersectional validation (full agents + seeds)...")
        payload = _build_payload(
            bias_factor=BIAS_FACTOR,
            group_bias_race=race_bias,
            group_bias_gender=gender_bias,
            # Must match the scope used in Stages 1 and 2.
            # all_channels: all pipeline stages active for validation.
            bias_scope=BIAS_SCOPE,
            task_id=0.0,
            agents=CONFIG["initial_agents"],
        )
        raw  = _run_batch([(0.0, payload)], n_workers, "Stage3 Validation")
        agg  = _aggregate_subgroup_rates(raw.get(0.0, []))
        metrics = _compute_disparity_metrics(agg)
        validation = {
            "race_bias": race_bias, "gender_bias": gender_bias,
            "bias_factor": BIAS_FACTOR, "bias_scope": BIAS_SCOPE,
            "agg":     {str(k): v for k, v in agg.items()},
            "metrics": {str(k): v for k, v in metrics.items()},
        }
        with open(cp_val, "w") as f:
            json.dump(validation, f, indent=2)
        print("  OK  Validation complete")

    cp_sens = os.path.join(outdir, "stage3_sensitivity.json")
    if os.path.exists(cp_sens):
        print("  > Sensitivity — loading checkpoint")
        with open(cp_sens) as f:
            sensitivity = json.load(f)
    else:
        print(f"\n  > Sweeping bias_factor  "
              f"({len(SENSITIVITY_LEVELS)} levels)  "
              f"[{AGENTS_COARSE} agents, coarse settings]")
        overrides_list = []
        for i, bf in enumerate(SENSITIVITY_LEVELS):
            payload = _build_payload(
                bias_factor=float(bf),
                group_bias_race=race_bias,
                group_bias_gender=gender_bias,
                bias_scope=BIAS_SCOPE,
                task_id=float(i),
                agents=AGENTS_COARSE,
            )
            overrides_list.append((float(i), payload))

        raw = _run_batch(overrides_list, n_workers, "Stage3 Sensitivity",
                         n_reps_override=N_REPS_COARSE,
                         batch_size=_WORKER_BATCH_SIZE_COARSE,
                         seeds_override=SEEDS_COARSE)
        sens_results = []
        for i, bf in enumerate(SENSITIVITY_LEVELS):
            agg = _aggregate_subgroup_rates(raw.get(float(i), []))
            if not agg:
                continue
            metrics = _compute_disparity_metrics(agg)
            sens_results.append({
                "bias_factor": bf,
                "agg":     {str(k): v for k, v in agg.items()},
                "metrics": {str(k): v for k, v in metrics.items()},
            })
        sensitivity = {"levels": sens_results}
        with open(cp_sens, "w") as f:
            json.dump(sensitivity, f, indent=2)
        print("  OK  Sensitivity sweep complete")

    return {"validation": validation, "sensitivity": sensitivity}


# =============================================================================
# CHARTS (identical structure to supervision_only, labels updated for scope)
# =============================================================================

def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Shared dissertation-friendly axis styling."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=DISS_AXIS, labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=DISS_AXIS, labelpad=7)
    if title:
        ax.set_title(
            title, fontsize=DISS_SUBTITLE, fontweight="bold",
            pad=10, color="#1A3D5C"
        )
    ax.grid(True, color=COLORS["grid"], linewidth=0.7,
            linestyle="--", alpha=0.85, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=DISS_TICK)


def plot_stage1_calibration(race_params: dict, outdir: str) -> str:
    SMOOTH_SIGMA = 2.0
    black_data    = race_params.get("Black", {})
    coarse_rows_b = black_data.get("coarse_results", black_data.get("all_results", []))
    fine_rows_b   = black_data.get("fine_results", [])
    best_b        = black_data.get("value")
    coarse_best_b = black_data.get("coarse_best", best_b)

    ncols_b = 2 if fine_rows_b else 1
    fig_b, axes_b = plt.subplots(1, ncols_b, figsize=(7 * ncols_b, 5))
    fig_b.patch.set_facecolor("#FAFAFA")
    if ncols_b == 1:
        axes_b = [axes_b]

    ax = axes_b[0]
    if coarse_rows_b:
        vals = [r["value"] for r in coarse_rows_b]
        maes = [r["mae"]   for r in coarse_rows_b]
        ax.plot(vals, maes, color=COLORS["Black"], linewidth=0.9,
                alpha=0.35, linestyle="--", zorder=2, label="MAE (raw)")
        ax.plot(vals, gaussian_filter1d(maes, sigma=SMOOTH_SIGMA),
                color=COLORS["Black"], linewidth=2.2, zorder=3,
                label=f"MAE (smoothed σ={SMOOTH_SIGMA})")
        ax.axvline(coarse_best_b, color="#333333", linewidth=1.5,
                   linestyle="--", zorder=4, label=f"Coarse best={coarse_best_b:.2f}")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=8)
    _style_ax(ax, "Black Bias (log-odds)", "MAE vs Adjusted BJS Target",
              f"Stage 1 — Black  Coarse  0.0→1.5\nbias_scope={BIAS_SCOPE}")

    if fine_rows_b:
        ax2 = axes_b[1]
        fvals = [r["value"] for r in fine_rows_b]
        fmaes = [r["mae"]   for r in fine_rows_b]
        ax2.plot(fvals, fmaes, color=COLORS["Black"], linewidth=0.9,
                 alpha=0.35, linestyle="--", zorder=2, label="MAE (raw)")
        ax2.plot(fvals, gaussian_filter1d(fmaes, sigma=1.5),
                 color=COLORS["Black"], linewidth=2.2, zorder=3,
                 label="MAE (smoothed σ=1.5)")
        ax2.axvline(best_b, color="#333333", linewidth=1.5, linestyle="--",
                    label=f"Fine best={best_b:.3f} (MAE={black_data['mae']:.4f})")
        ax2.axvline(coarse_best_b, color="#888888", linewidth=1.0, linestyle=":",
                    label=f"Coarse best={coarse_best_b:.2f}")
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=8)
        _style_ax(ax2, "Black Bias (log-odds)", "",
                  f"Pass 2: Fine  ±{BLACK_SWEEP_FINE_HALF_WIDTH:.2f} around coarse best")

    fig_b.suptitle(
        f"Stage 1 — Black Race Bias Calibration (two-pass)\nbias_scope={BIAS_SCOPE}",
        fontsize=12, fontweight="bold", color="#1A3D5C", y=1.04)
    plt.tight_layout()
    path_b = os.path.join(outdir, "phase2_stage1_black_calibration.png")
    plt.savefig(path_b, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig_b.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 Black -> {path_b}")

    hisp_data = race_params.get("Hispanic", {})
    rows_h    = hisp_data.get("all_results", [])
    best_h    = hisp_data.get("value")
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    fig_h.patch.set_facecolor("#FAFAFA")
    if rows_h:
        vals = [r["value"] for r in rows_h]
        maes = [r["mae"]   for r in rows_h]
        ax_h.plot(vals, maes, color=COLORS["Hispanic"], linewidth=0.9,
                  alpha=0.35, linestyle="--", zorder=2, label="MAE (raw)")
        ax_h.plot(vals, gaussian_filter1d(maes, sigma=SMOOTH_SIGMA),
                  color=COLORS["Hispanic"], linewidth=2.2, zorder=3,
                  label=f"MAE (smoothed σ={SMOOTH_SIGMA})")
        ax_h.axvline(best_h, color="#333333", linewidth=1.5, linestyle="--", zorder=4,
                     label=f"Best={best_h:.2f} (MAE={hisp_data['mae']:.4f})")
        ax_h.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax_h.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_h.legend(fontsize=9)
    _style_ax(ax_h, "Hispanic Bias (log-odds)", "MAE vs Adjusted BJS Target",
              f"Stage 1 — Hispanic Race Bias\nbias_scope={BIAS_SCOPE}")
    plt.tight_layout()
    path_h = os.path.join(outdir, "phase2_stage1_hispanic_calibration.png")
    plt.savefig(path_h, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig_h.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 Hispanic -> {path_h}")

    fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 5))
    fig_c.patch.set_facecolor("#FAFAFA")
    for ax_c, (grp, clr, rows, bv) in zip(
        axes_c,
        [("Black", COLORS["Black"], coarse_rows_b, coarse_best_b),
         ("Hispanic", COLORS["Hispanic"], rows_h, best_h)]):
        if not rows:
            continue
        vals = [r["value"] for r in rows]
        maes = [r["mae"]   for r in rows]
        ax_c.plot(vals, maes, color=clr, linewidth=0.9, alpha=0.35, linestyle="--", zorder=2)
        ax_c.plot(vals, gaussian_filter1d(maes, sigma=SMOOTH_SIGMA),
                  color=clr, linewidth=2.0, zorder=3,
                  label=f"Best={bv:.2f} (MAE={race_params[grp]['mae']:.4f})")
        ax_c.axvline(bv, color="#333333", linewidth=1.5, linestyle="--", zorder=4)
        ax_c.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax_c.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_c.legend(fontsize=9)
        _style_ax(ax_c, f"{grp} Bias (log-odds)", "MAE vs Adjusted BJS Target",
                  f"Stage 1 — {grp}")
    fig_c.suptitle(
        f"Stage 1: Race Bias Calibration — MAE Loss Curves\n"
        f"bias_scope={BIAS_SCOPE}  |  Dashed = raw  |  Solid = smoothed",
        fontsize=12, fontweight="bold", color="#1A3D5C", y=1.04)
    plt.tight_layout()
    path_combined = os.path.join(outdir, "phase2_stage1_race_calibration.png")
    plt.savefig(path_combined, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig_c.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 combined -> {path_combined}")
    return path_combined


def plot_stage2_calibration(gender_params: dict, outdir: str) -> str:
    data        = gender_params.get("Female", {})
    coarse_rows = data.get("coarse_results", data.get("all_results", []))
    fine_rows   = data.get("fine_results", [])
    best_v      = data.get("value")
    coarse_best = data.get("coarse_best", best_v)

    ncols = 2 if fine_rows else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.patch.set_facecolor("#FAFAFA")
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    if coarse_rows:
        vals = [r["value"] for r in coarse_rows]
        maes = [r["mae"]   for r in coarse_rows]
        ax.plot(vals, maes, color=COLORS["Female"], linewidth=2.0, zorder=3,
                label="MAE (coarse)")
        ax.axvline(coarse_best, color="#333333", linewidth=1.5, linestyle="--",
                   label=f"Coarse best={coarse_best:.2f}")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9)
    _style_ax(ax, "Female Bias (log-odds)", "MAE vs Adjusted BJS Target",
              f"Stage 2 — Female  Coarse  −5.0→+0.5  (step 0.1)\n"
              f"Extended range for {BIAS_SCOPE} scope")

    if fine_rows:
        ax2 = axes[1]
        fvals = [r["value"] for r in fine_rows]
        fmaes = [r["mae"]   for r in fine_rows]
        ax2.plot(fvals, fmaes, color="#C55A11", linewidth=2.0, zorder=3,
                 label="MAE (fine)")
        ax2.axvline(best_v, color="#333333", linewidth=1.5, linestyle="--",
                    label=f"Fine best={best_v:.3f} (MAE={data['mae']:.4f})")
        ax2.axvline(coarse_best, color="#888888", linewidth=1.0, linestyle=":",
                    label=f"Coarse best={coarse_best:.2f}")
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=9)
        _style_ax(ax2, "Female Bias (log-odds)", "",
                  f"Pass 2: Fine  ±{GENDER_SWEEP_FINE_HALF_WIDTH:.2f} around coarse best")

    fig.suptitle(
        f"Stage 2 — Female Gender Bias Calibration (two-pass)\n"
        f"bias_scope={BIAS_SCOPE}  |  consistent with Stage 1",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_stage2_gender_calibration.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 2 -> {path}")
    return path


def plot_stage3_intersectional(validation: dict, outdir: str) -> str:
    agg     = {int(k): v for k, v in validation["agg"].items()}
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.91, bottom=0.06)
    race_groups   = ["White", "Black", "Hispanic"]
    gender_groups = ["Male", "Female"]

    for col, win in enumerate(WINDOWS):
        ax  = fig.add_subplot(gs[0, col])
        x   = np.arange(len(race_groups))
        bjs_vals = [BJS_SUBGROUP[win].get(g, 0) for g in race_groups]
        sim_vals = [agg[win].get(g, {}).get("mean", 0) or 0 for g in race_groups]
        sim_cis  = [agg[win].get(g, {}).get("ci95",  0) or 0 for g in race_groups]
        ax.bar(x - 0.2, bjs_vals, width=0.35, color="#AAAAAA",
               alpha=0.7, label="BJS Target", zorder=3)
        ax.bar(x + 0.2, sim_vals, width=0.35,
               color=[COLORS[g] for g in race_groups], alpha=0.85,
               label="Simulation", zorder=3)
        ax.errorbar(x + 0.2, sim_vals, yerr=sim_cis,
                    fmt="none", color="#333333", elinewidth=1.5, capsize=5, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(race_groups, fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(0.4, 1.0)
        _style_ax(ax, "", "Rearrest Rate", f"{win}-Year Rate by Race")
        if col == 0:
            ax.legend(fontsize=8)

    ax_g = fig.add_subplot(gs[1, 0])
    x    = np.arange(len(gender_groups))
    bjs_g = [BJS_SUBGROUP[3].get(g, 0) for g in gender_groups]
    sim_g = [agg[3].get(g, {}).get("mean", 0) or 0 for g in gender_groups]
    ci_g  = [agg[3].get(g, {}).get("ci95",  0) or 0 for g in gender_groups]
    ax_g.bar(x - 0.2, bjs_g, width=0.35, color="#AAAAAA",
             alpha=0.7, label="BJS Target", zorder=3)
    ax_g.bar(x + 0.2, sim_g, width=0.35,
             color=[COLORS[g] for g in gender_groups], alpha=0.85,
             label="Simulation", zorder=3)
    ax_g.errorbar(x + 0.2, sim_g, yerr=ci_g, fmt="none",
                  color="#333333", elinewidth=1.5, capsize=5, zorder=4)
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(gender_groups, fontsize=9)
    ax_g.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_g.set_ylim(0.4, 1.0)
    ax_g.legend(fontsize=8)
    _style_ax(ax_g, "", "Rearrest Rate", "3-Year Rate by Gender")

    ax_h = fig.add_subplot(gs[1, 1:])
    intersect_groups = [f"{r}_{g}" for r in race_groups for g in gender_groups]
    matrix = np.zeros((len(WINDOWS), len(intersect_groups)))
    for i, win in enumerate(WINDOWS):
        for j, grp in enumerate(intersect_groups):
            matrix[i, j] = agg[win].get(grp, {}).get("mean", 0) or 0
    im = ax_h.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.4, vmax=0.95)
    ax_h.set_xticks(range(len(intersect_groups)))
    ax_h.set_xticklabels([g.replace("_", "\n") for g in intersect_groups], fontsize=8)
    ax_h.set_yticks(range(len(WINDOWS)))
    ax_h.set_yticklabels([f"{w}-Year" for w in WINDOWS], fontsize=9)
    for i in range(len(WINDOWS)):
        for j in range(len(intersect_groups)):
            ax_h.text(j, i, f"{matrix[i, j]:.2%}", ha="center", va="center",
                      fontsize=8, fontweight="bold",
                      color="white" if matrix[i, j] > 0.75 else "#222222")
    plt.colorbar(im, ax=ax_h, shrink=0.8).ax.tick_params(labelsize=8)
    _style_ax(ax_h, "", "", "Intersectional Rearrest Rates (Race × Gender)")

    fig.suptitle(
        f"Stage 3 — Intersectional Validation  (scope={BIAS_SCOPE})\n"
        f"Black={validation['race_bias']['Black']:.2f}  "
        f"Hispanic={validation['race_bias']['Hispanic']:.2f}  "
        f"Female={validation['gender_bias']['Female']:.2f}  "
        f"bias_factor={validation['bias_factor']:.1f}",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=0.98)
    path = os.path.join(outdir, "phase2_stage3_intersectional.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 3 intersectional -> {path}")
    return path


def plot_stage3_sensitivity(sensitivity: dict, outdir: str) -> str:
    levels = sensitivity.get("levels", [])
    if not levels:
        return ""
    bfs = [r["bias_factor"] for r in levels]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#FAFAFA")
    for row, (grp, ref, title, clr) in enumerate([
        ("Black",  "White", "Race Gap: Black − White",   COLORS["Black"]),
        ("Female", "Male",  "Gender Gap: Female − Male", COLORS["Female"]),
    ]):
        for col, win in enumerate(WINDOWS):
            ax   = axes[row][col]
            gaps = [({int(k): v for k, v in r["metrics"].items()}
                     .get(win, {}).get(grp, {}).get("gap", 0) or 0)
                    for r in levels]
            ax.plot(bfs, gaps, color=clr, linewidth=2.0, label="Gap (pp)", zorder=3)
            ax.axvline(BIAS_FACTOR, color="#888888", linewidth=1.0,
                       linestyle="--", label="Baseline (1.0)")
            ax.axhline(0, color="#333333", linewidth=0.7)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.legend(fontsize=8)
            _style_ax(ax, "bias_factor", "Gap (pp)", f"{title}\n{win}-Year")
    fig.suptitle(
        f"Stage 3 — Bias Factor Sensitivity  (scope={BIAS_SCOPE})\n"
        "±40% perturbation of bias_factor",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_stage3_sensitivity.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 3 sensitivity -> {path}")
    return path


def plot_disparity_summary(validation: dict, outdir: str) -> str:
    metrics = {int(k): v for k, v in validation["metrics"].items()}
    groups  = ["Black", "Hispanic", "Female"]
    fig, axes = plt.subplots(len(WINDOWS), 3, figsize=(16, 11))
    fig.patch.set_facecolor("#FAFAFA")
    titles = ["Absolute Gap\n(group − reference, pp)",
              "Rate Ratio\n(group / reference)",
              "Absolute Deviation\nfrom Aggregate Baseline (pp)"]

    for row, win in enumerate(WINDOWS):
        for col, (metric, title) in enumerate(zip(["gap", "ratio", "abs_dev"], titles)):
            ax   = axes[row][col]
            vals = [metrics[win].get(g, {}).get(metric, 0) or 0 for g in groups]
            cis  = [metrics[win].get(g, {}).get("ci95",   0) or 0 for g in groups]
            xpos = np.arange(len(groups))
            ax.bar(xpos, vals, color=[COLORS[g] for g in groups],
                   alpha=0.85, edgecolor="white", width=0.55, zorder=3)
            if metric != "ratio":
                ax.errorbar(xpos, vals, yerr=cis, fmt="none",
                            color="#333333", elinewidth=1.5, capsize=5, zorder=4)
                ax.axhline(0, color="#333333", linewidth=0.8)
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            else:
                ax.axhline(1.0, color="#333333", linewidth=0.8,
                           linestyle="--", label="Reference (1.0)")
                ax.legend(fontsize=7)
            for xi, v in enumerate(vals):
                ax.text(xi, v + (0.001 if v >= 0 else -0.005),
                        f"{v:.3f}", ha="center", fontsize=8, color="#333333")
            ax.set_xticks(xpos)
            ax.set_xticklabels(groups, fontsize=9)
            _style_ax(ax, "", title if col == 0 else "",
                      f"{win}-Year | {title.split(chr(10))[0]}" if row == 0 else "")

    fig.suptitle(
        "Phase 2 Disparity Summary — All Three Metrics\n"
        "Gap = group − reference  |  Ratio = group / reference  |  "
        f"Abs Dev = group − baseline\nbias_scope={BIAS_SCOPE}",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_disparity_summary.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Disparity summary -> {path}")
    return path



# =============================================================================
# DISSERTATION-READY VALIDATION FIGURES + TABLES
# =============================================================================

def _diss_agg(validation: dict) -> dict:
    return {int(k): v for k, v in validation["agg"].items()}


def _diss_ref(group: str) -> str:
    return "Male" if group == "Female" else "White"


def _diss_bjs_gap(win: int, group: str) -> float:
    ref = _diss_ref(group)
    return BJS_SUBGROUP[win][group] - BJS_SUBGROUP[win][ref]


def plot_diss_overall_fit(validation: dict, outdir: str) -> str:
    """D1 — Overall BJS anchor validation after all_channels calibration."""
    agg = _diss_agg(validation)
    bjs = np.array([CONFIG["bjs_targets"][w] * 100 for w in WINDOWS])
    sim = np.array([agg[w]["Overall"]["mean"] * 100 for w in WINDOWS])
    delta = sim - bjs
    mae = float(np.mean(np.abs(delta)))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.patch.set_facecolor("#FAFAFA")

    ax.plot(WINDOWS, bjs, marker="D", markersize=8, linewidth=2.7,
            color=COLORS["bjs"], label="BJS empirical target", zorder=4)
    ax.plot(WINDOWS, sim, marker="o", markersize=8, linewidth=2.7,
            color=COLORS["sim"], label="Calibrated ABM", zorder=5)

    for w, y, d in zip(WINDOWS, sim, delta):
        ax.annotate(f"{y:.1f}%\\nΔ {d:+.1f} pp", (w, y),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=DISS_TICK, fontweight="bold")

    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([f"{w}-Year" for w in WINDOWS])
    _style_ax(ax, "Follow-up Window", "Cumulative Rearrest Rate (%)",
              "Aggregate Empirical Validation")
    ax.legend(frameon=False)
    ax.text(0.02, 0.04, f"Aggregate MAE = {mae:.2f} pp",
            transform=ax.transAxes, fontsize=DISS_TICK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB"))

    fig.suptitle("All-Channels Bias Calibration: Aggregate Fit to BJS",
                 fontsize=DISS_TITLE, fontweight="bold",
                 color="#1A3D5C", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(outdir, "D1_all_channels_overall_bjs_validation.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Dissertation chart D1 -> {path}")
    return path


def plot_diss_subgroup_fit(validation: dict, outdir: str) -> str:
    """D2 — Absolute race/gender subgroup rates versus BJS."""
    agg = _diss_agg(validation)
    groups = ["White", "Black", "Hispanic", "Male", "Female"]
    x = np.arange(len(groups))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2), sharey=True)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, win in zip(axes, WINDOWS):
        bjs = np.array([BJS_SUBGROUP[win][g] * 100 for g in groups])
        sim = np.array([agg[win].get(g, {}).get("mean", np.nan) * 100
                        for g in groups])
        ci = np.array([agg[win].get(g, {}).get("ci95", 0.0) * 100
                       for g in groups])

        ax.bar(x - width/2, bjs, width, color="#B7B7B7",
               alpha=0.85, label="BJS target", zorder=3)
        ax.bar(x + width/2, sim, width,
               color=[COLORS[g] for g in groups],
               alpha=0.88, label="Calibrated ABM", zorder=3)
        ax.errorbar(x + width/2, sim, yerr=ci, fmt="none",
                    ecolor="#333333", elinewidth=1.2,
                    capsize=4, zorder=4)

        for i, (target, value) in enumerate(zip(bjs, sim)):
            ax.text(i + width/2, value + 0.8, f"{value-target:+.1f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.set_ylim(45, 95)
        _style_ax(ax, "",
                  "Rearrest Rate (%)" if win == WINDOWS[0] else "",
                  f"{win}-Year Follow-up")

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("All-Channels Calibration: Absolute Subgroup Fit to BJS",
                 fontsize=DISS_TITLE, fontweight="bold",
                 color="#1A3D5C", y=1.01)
    fig.text(0.5, 0.015,
             "Labels above ABM bars show ABM − BJS error in percentage points; "
             "error bars show 95% simulation confidence intervals.",
             ha="center", fontsize=DISS_NOTE)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = os.path.join(outdir, "D2_all_channels_subgroup_bjs_validation.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Dissertation chart D2 -> {path}")
    return path


def plot_diss_gap_fit(validation: dict, outdir: str) -> str:
    """D3 — ABM demographic disparities versus direct BJS disparities."""
    agg = _diss_agg(validation)
    comparisons = [
        ("Black", "White", "Black − White"),
        ("Hispanic", "White", "Hispanic − White"),
        ("Female", "Male", "Female − Male"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (group, ref, title) in zip(axes, comparisons):
        bjs = np.array([_diss_bjs_gap(w, group) * 100 for w in WINDOWS])
        sim = np.array([(agg[w][group]["mean"] - agg[w][ref]["mean"]) * 100
                        for w in WINDOWS])
        mae = float(np.mean(np.abs(sim - bjs)))

        ax.plot(WINDOWS, bjs, marker="D", markersize=7,
                linewidth=2.5, color=COLORS["bjs"],
                label="BJS disparity")
        ax.plot(WINDOWS, sim, marker="o", markersize=7,
                linewidth=2.5, color=COLORS[group],
                label="ABM disparity")
        ax.axhline(0, color="#666666", linewidth=0.9)

        for w, s, t in zip(WINDOWS, sim, bjs):
            ax.annotate(f"|err|={abs(s-t):.1f}", (w, s),
                        xytext=(0, 10 if s >= 0 else -17),
                        textcoords="offset points",
                        ha="center", fontsize=8.5)

        ax.set_xticks(WINDOWS)
        ax.set_xticklabels([f"{w}yr" for w in WINDOWS])
        _style_ax(ax, "Follow-up Window",
                  "Group − Reference Gap (pp)",
                  f"{title}\\nGap MAE = {mae:.2f} pp")

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("All-Channels Calibration: Demographic Disparity Validation",
                 fontsize=DISS_TITLE, fontweight="bold",
                 color="#1A3D5C", y=1.02)
    fig.text(0.5, 0.015,
             "Gap MAE is the mean absolute difference between simulated and "
             "BJS group-reference gaps across 3-, 6-, and 9-year windows.",
             ha="center", fontsize=DISS_NOTE)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = os.path.join(outdir, "D3_all_channels_demographic_gap_validation.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Dissertation chart D3 -> {path}")
    return path


def plot_diss_parameter_identification(race_params: dict,
                                       gender_params: dict,
                                       outdir: str) -> str:
    """D4 — Explain how Black/Hispanic/Female beta values were selected."""
    series = [
        ("Black", race_params.get("Black", {}), COLORS["Black"]),
        ("Hispanic", race_params.get("Hispanic", {}), COLORS["Hispanic"]),
        ("Female", gender_params.get("Female", {}), COLORS["Female"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (group, data, color) in zip(axes, series):
        coarse = data.get("coarse_results", data.get("all_results", []))
        fine = data.get("fine_results", [])
        best = data.get("value")
        mae = data.get("mae")

        if coarse:
            vals = np.array([r["value"] for r in coarse], dtype=float)
            errs = np.array([r["mae"] * 100 for r in coarse], dtype=float)
            order = np.argsort(vals)
            ax.plot(vals[order], errs[order], linestyle="--",
                    linewidth=1.5, alpha=0.55, color=color,
                    label="Coarse search")

        if fine:
            vals = np.array([r["value"] for r in fine], dtype=float)
            errs = np.array([r["mae"] * 100 for r in fine], dtype=float)
            order = np.argsort(vals)
            ax.plot(vals[order], errs[order], marker="o",
                    markersize=4, linewidth=2.2, color=color,
                    label="Fine validation")

        if best is not None:
            ax.axvline(best, linestyle="--", linewidth=1.5,
                       color="#333333")
            mae_txt = "—" if mae is None else f"{mae*100:.2f} pp"
            ax.text(0.04, 0.95,
                    f"Selected β = {best:+.3f}\\n"
                    f"OR = {math.exp(best):.3f}×\\n"
                    f"Calibration MAE = {mae_txt}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.35",
                              fc="white", ec="#BBBBBB"))

        ax.axvline(0, linestyle=":", linewidth=0.9, color="#777777")
        _style_ax(ax, "Bias Parameter β (log-odds)",
                  "Calibration MAE (pp)" if group == "Black" else "",
                  group)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Identification of All-Channels Demographic Bias Parameters",
                 fontsize=DISS_TITLE, fontweight="bold",
                 color="#1A3D5C", y=1.02)
    fig.text(0.5, 0.015,
             "Each coefficient is selected at the minimum subgroup-rate "
             "calibration error under the all_channels specification.",
             ha="center", fontsize=DISS_NOTE)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    path = os.path.join(outdir, "D4_all_channels_parameter_identification.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Dissertation chart D4 -> {path}")
    return path


def build_diss_tables(validation: dict,
                      race_params: dict,
                      gender_params: dict,
                      outdir: str) -> dict:
    """Export figure-ready validation data as CSV."""
    agg = _diss_agg(validation)

    overall_rows = []
    for w in WINDOWS:
        bjs = CONFIG["bjs_targets"][w]
        sim = agg[w]["Overall"]["mean"]
        overall_rows.append({
            "Window": f"{w}-Year",
            "BJS Rate (%)": bjs * 100,
            "ABM Rate (%)": sim * 100,
            "ABM-BJS (pp)": (sim-bjs) * 100,
            "Absolute Error (pp)": abs(sim-bjs) * 100,
        })
    overall_df = pd.DataFrame(overall_rows)

    subgroup_rows = []
    for w in WINDOWS:
        for group in ["White", "Black", "Hispanic", "Male", "Female"]:
            bjs = BJS_SUBGROUP[w][group]
            sim = agg[w][group]["mean"]
            ci = agg[w][group].get("ci95", 0.0)
            subgroup_rows.append({
                "Window": f"{w}-Year",
                "Group": group,
                "BJS Rate (%)": bjs * 100,
                "ABM Rate (%)": sim * 100,
                "95% CI Half-Width (pp)": ci * 100,
                "ABM-BJS (pp)": (sim-bjs) * 100,
                "Absolute Error (pp)": abs(sim-bjs) * 100,
            })
    subgroup_df = pd.DataFrame(subgroup_rows)

    gap_rows = []
    for w in WINDOWS:
        for group, ref in [
            ("Black", "White"),
            ("Hispanic", "White"),
            ("Female", "Male"),
        ]:
            bjs_gap = _diss_bjs_gap(w, group)
            sim_gap = agg[w][group]["mean"] - agg[w][ref]["mean"]
            gap_rows.append({
                "Window": f"{w}-Year",
                "Comparison": f"{group} - {ref}",
                "BJS Gap (pp)": bjs_gap * 100,
                "ABM Gap (pp)": sim_gap * 100,
                "Gap Error (pp)": (sim_gap-bjs_gap) * 100,
                "Absolute Gap Error (pp)": abs(sim_gap-bjs_gap) * 100,
            })
    gap_df = pd.DataFrame(gap_rows)

    parameter_rows = []
    for group, data in [
        ("Black", race_params.get("Black", {})),
        ("Hispanic", race_params.get("Hispanic", {})),
        ("Female", gender_params.get("Female", {})),
    ]:
        beta = data.get("value")
        if beta is None:
            continue
        ref = _diss_ref(group)
        rate_err = []
        gap_err = []
        for w in WINDOWS:
            rate_err.append(abs(agg[w][group]["mean"] -
                                BJS_SUBGROUP[w][group]))
            gap_err.append(abs(
                (agg[w][group]["mean"] - agg[w][ref]["mean"]) -
                _diss_bjs_gap(w, group)
            ))
        parameter_rows.append({
            "Group": group,
            "Bias Beta": beta,
            "Odds Ratio": math.exp(beta),
            "Calibration MAE (pp)": data.get("mae", np.nan) * 100,
            "Final Subgroup Rate MAE (pp)": np.mean(rate_err) * 100,
            "Final Gap MAE (pp)": np.mean(gap_err) * 100,
        })
    parameter_df = pd.DataFrame(parameter_rows)

    intersectional_rows = []
    for w in WINDOWS:
        for race in ["White", "Black", "Hispanic"]:
            for gender in ["Male", "Female"]:
                key = f"{race}_{gender}"
                val = agg[w].get(key, {}).get("mean")
                ci = agg[w].get(key, {}).get("ci95")
                if val is not None:
                    intersectional_rows.append({
                        "Window": f"{w}-Year",
                        "Race": race,
                        "Gender": gender,
                        "ABM Rearrest Rate (%)": val * 100,
                        "95% CI Half-Width (pp)":
                            ci * 100 if ci is not None else np.nan,
                    })
    intersectional_df = pd.DataFrame(intersectional_rows)

    paths = {
        "overall": os.path.join(outdir, "D_table_all_channels_overall_fit.csv"),
        "subgroup": os.path.join(outdir, "D_table_all_channels_subgroup_fit.csv"),
        "gap": os.path.join(outdir, "D_table_all_channels_gap_fit.csv"),
        "parameters": os.path.join(outdir, "D_table_all_channels_parameter_summary.csv"),
        "intersectional": os.path.join(outdir, "D_table_all_channels_intersectional_rates.csv"),
    }

    overall_df.to_csv(paths["overall"], index=False)
    subgroup_df.to_csv(paths["subgroup"], index=False)
    gap_df.to_csv(paths["gap"], index=False)
    parameter_df.to_csv(paths["parameters"], index=False)
    intersectional_df.to_csv(paths["intersectional"], index=False)

    return {
        "overall": overall_df,
        "subgroup": subgroup_df,
        "gap": gap_df,
        "parameters": parameter_df,
        "intersectional": intersectional_df,
        "paths": paths,
    }


def plot_diss_parameter_table(validation: dict,
                              race_params: dict,
                              gender_params: dict,
                              outdir: str) -> str:
    """D5 — Final parameter/validation summary table."""
    tables = build_diss_tables(validation, race_params, gender_params, outdir)
    df = tables["parameters"]

    display = []
    for _, r in df.iterrows():
        display.append([
            r["Group"],
            f"{r['Bias Beta']:+.3f}",
            f"{r['Odds Ratio']:.3f}×",
            f"{r['Calibration MAE (pp)']:.2f}",
            f"{r['Final Subgroup Rate MAE (pp)']:.2f}",
            f"{r['Final Gap MAE (pp)']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.axis("off")

    table = ax.table(
        cellText=display,
        colLabels=[
            "Group", "Bias β", "Odds Ratio",
            "Calibration\\nMAE (pp)",
            "Final Subgroup\\nRate MAE (pp)",
            "Final Gap\\nMAE (pp)",
        ],
        cellLoc="center", colLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.75)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C7C7C7")
        if row == 0:
            cell.set_facecolor("#E7EDF4")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")

    ax.set_title(
        "All-Channels Bias Calibration: Parameter and Validation Summary",
        fontsize=DISS_TITLE, fontweight="bold",
        color="#1A3D5C", pad=16
    )
    fig.text(
        0.5, 0.035,
        "Calibration MAE is the parameter-selection objective; final subgroup-rate "
        "and gap MAEs are Stage 3 validation diagnostics.",
        ha="center", fontsize=DISS_NOTE
    )
    plt.tight_layout(rect=[0.01, 0.07, 0.99, 0.94])

    path = os.path.join(outdir, "D5_all_channels_parameter_summary_table.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Dissertation chart D5 -> {path}")
    return path


def plot_diss_intersectional_table(validation: dict, outdir: str) -> str:
    """D6 — Compact race × gender rates at the three BJS anchor windows."""
    agg = _diss_agg(validation)
    rows = [
        ("White Male", "White_Male"),
        ("White Female", "White_Female"),
        ("Black Male", "Black_Male"),
        ("Black Female", "Black_Female"),
        ("Hispanic Male", "Hispanic_Male"),
        ("Hispanic Female", "Hispanic_Female"),
    ]

    data = []
    for label, key in rows:
        vals = []
        for w in WINDOWS:
            v = agg[w].get(key, {}).get("mean")
            vals.append("—" if v is None else f"{v*100:.2f}%")
        data.append([label] + vals)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=["Intersectional Group", "3-Year", "6-Year", "9-Year"],
        cellLoc="center", colLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C7C7C7")
        if row == 0:
            cell.set_facecolor("#E7EDF4")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")

    ax.set_title(
        "Intersectional Rearrest Rates Under the All-Channels Specification",
        fontsize=DISS_TITLE, fontweight="bold",
        color="#1A3D5C", pad=16
    )
    fig.text(
        0.5, 0.035,
        "Race and gender coefficients operate simultaneously across trial, "
        "sentencing, supervision assignment, detection, and supervision rearrest.",
        ha="center", fontsize=DISS_NOTE
    )
    plt.tight_layout(rect=[0.02, 0.07, 0.98, 0.94])

    path = os.path.join(outdir, "D6_all_channels_intersectional_table.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Dissertation chart D6 -> {path}")
    return path


def generate_dissertation_outputs(validation: dict,
                                   race_params: dict,
                                   gender_params: dict,
                                   outdir: str) -> None:
    """Generate dissertation figures/tables without additional ABM simulation."""
    print("\\n  Generating dissertation-ready all_channels outputs...")
    plot_diss_overall_fit(validation, outdir)
    plot_diss_subgroup_fit(validation, outdir)
    plot_diss_gap_fit(validation, outdir)
    plot_diss_parameter_identification(race_params, gender_params, outdir)
    plot_diss_parameter_table(validation, race_params, gender_params, outdir)
    plot_diss_intersectional_table(validation, outdir)

    print("    CSV exports:")
    print("      D_table_all_channels_overall_fit.csv")
    print("      D_table_all_channels_subgroup_fit.csv")
    print("      D_table_all_channels_gap_fit.csv")
    print("      D_table_all_channels_parameter_summary.csv")
    print("      D_table_all_channels_intersectional_rates.csv")


# =============================================================================
# WORKER DETECTION
# =============================================================================

def detect_workers() -> tuple:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        n = max(1, min(int(slurm_cpus) - _OS_OVERHEAD_CORES, _MAX_WORKERS))
        return n, f"SLURM_CPUS_PER_TASK={slurm_cpus} -> {n} workers"
    if _PSUTIL_AVAILABLE:
        physical = psutil.cpu_count(logical=False)
        logical  = psutil.cpu_count(logical=True)
        if physical and physical > 0:
            n = max(1, min(physical - 1, _MAX_WORKERS))
            return n, f"psutil: {physical} physical -> {n} workers"
        n = max(1, min((logical or 2) - 1, _MAX_WORKERS))
        return n, f"psutil fallback: {n} workers"
    logical = multiprocessing.cpu_count()
    n = max(1, min(logical - 1, _MAX_WORKERS))
    return n, f"multiprocessing: {logical} logical -> {n} workers"


# =============================================================================
# MAIN
# =============================================================================

def main(n_workers: int, stages: list = None, force_rerun: bool = False,
         bias_scope: str = None):
    global BIAS_SCOPE
    import os as _os
    for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_env] = "1"

    if stages is None:
        stages = [1, 2, 3]
    if bias_scope is not None:
        BIAS_SCOPE = bias_scope

    outdir = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("  RECIDIVISM ABM — PHASE 2 BIAS SIMULATION (all_channels)")
    print("=" * 70)
    print(f"  Workers        : {n_workers} / {_MAX_WORKERS} max")
    print(f"  Agents (fine)  : {CONFIG['initial_agents']}  (coarse: {AGENTS_COARSE})")
    print(f"  Seeds  (fine)  : {len(CONFIG['SEEDS'])}  (coarse: {len(SEEDS_COARSE)})")
    print(f"  Reps (c/f)     : {N_REPS_COARSE} / {N_REPS_FINE}  "
          f"→ {len(SEEDS_COARSE)*N_REPS_COARSE} / "
          f"{len(CONFIG['SEEDS'])*N_REPS_FINE} runs/pt")
    print(f"  Stages         : {stages}")
    print(f"  bias_scope     : {BIAS_SCOPE}  (structural embedding hypothesis)")
    print(f"  REMOVED KEYS   : bias_on_trial, bias_on_tech_detect")
    print(f"  Female sweep   : {GENDER_SWEEP_COARSE[0]:.1f} → {GENDER_SWEEP_COARSE[-1]:.1f}"
          f"  (extended vs supervision_only)")
    print(f"  Output dir     : {outdir}")
    print("=" * 70)

    if force_rerun:
        import glob
        STAGE_CP = {
            1: ["stage1_Black.json", "stage1_Hispanic.json"],
            2: ["stage2_gender.json"],
            3: ["stage3_validation.json", "stage3_sensitivity.json"],
        }
        target_fnames = {f for s in stages for f in STAGE_CP.get(s, [])}
        for cp in glob.glob(os.path.join(outdir, "stage*.json")):
            if os.path.basename(cp) in target_fnames:
                os.remove(cp)
                print(f"    deleted: {os.path.basename(cp)}")
        if 1 in stages:
            bp = os.path.join(outdir, "baseline_offsets.json")
            if os.path.exists(bp):
                os.remove(bp)
                print(f"    deleted: baseline_offsets.json (will remeasure)")

    measure_baseline_offsets(n_workers, outdir)

    STAGE_CP = {
        1: ["stage1_Black.json", "stage1_Hispanic.json"],
        2: ["stage2_gender.json"],
        3: ["stage3_validation.json", "stage3_sensitivity.json"],
    }
    print("\n  Checkpoint status:")
    for s, fnames in STAGE_CP.items():
        for fname in fnames:
            cp = os.path.join(outdir, fname)
            if os.path.exists(cp):
                try:
                    with open(cp) as f:
                        d = json.load(f)
                    val = d.get("value") or d.get("Female", {}).get("value", "?")
                    mae = d.get("mae")   or d.get("Female", {}).get("mae",   "?")
                    print(f"    Stage {s}  {fname:<35} FOUND  bias={val}  MAE={mae}")
                except Exception:
                    print(f"    Stage {s}  {fname:<35} FOUND (unreadable)")
            else:
                print(f"    Stage {s}  {fname:<35} missing — will run")
    print()

    race_params = {}
    gender_params = {}
    stage3_results = {}

    if 1 in stages:
        race_params = run_stage1_race(n_workers, outdir)
        print("\n  Generating Stage 1 charts...")
        plot_stage1_calibration(race_params, outdir)

    if 2 in stages:
        if not race_params:
            for grp in ["Black", "Hispanic"]:
                cp = os.path.join(outdir, f"stage1_{grp}.json")
                if os.path.exists(cp):
                    with open(cp) as f:
                        race_params[grp] = json.load(f)
        gender_params = run_stage2_gender(race_params, n_workers, outdir)
        print("\n  Generating Stage 2 charts...")
        plot_stage2_calibration(gender_params, outdir)

    if 3 in stages:
        if not race_params:
            for grp in ["Black", "Hispanic"]:
                cp = os.path.join(outdir, f"stage1_{grp}.json")
                if os.path.exists(cp):
                    with open(cp) as f:
                        race_params[grp] = json.load(f)
        if not gender_params:
            cp = os.path.join(outdir, "stage2_gender.json")
            if os.path.exists(cp):
                with open(cp) as f:
                    gender_params = json.load(f)
        stage3_results = run_stage3_intersectional(
            race_params, gender_params, n_workers, outdir)
        print("\n  Generating Stage 3 charts...")
        plot_stage3_intersectional(stage3_results["validation"], outdir)
        plot_stage3_sensitivity(stage3_results["sensitivity"], outdir)
        plot_disparity_summary(stage3_results["validation"], outdir)

        generate_dissertation_outputs(
            stage3_results["validation"],
            race_params,
            gender_params,
            outdir,
        )

    summary = {
        "bias_factor": BIAS_FACTOR, "bias_scope": BIAS_SCOPE,
        "reference_race": RACE_REFERENCE, "reference_gender": GENDER_REFERENCE,
        "calibrated_race_bias":   {k: v.get("value") for k, v in race_params.items()},
        "calibrated_gender_bias": {k: v.get("value") for k, v in gender_params.items()},
        "bjs_targets":    BJS_SUBGROUP,
        "baseline_offsets": BASELINE_OFFSETS,
        "adjusted_targets": ADJUSTED_TARGETS,
        "performance_settings": {
            "agents_coarse": AGENTS_COARSE, "seeds_coarse": SEEDS_COARSE,
            "n_reps_coarse": N_REPS_COARSE, "n_reps_fine": N_REPS_FINE,
        },
    }
    json_path = os.path.join(outdir, "phase2_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  OK  Summary -> {json_path}")

    print("\n" + "=" * 70)
    print("  FINAL CALIBRATION SUMMARY")
    print("=" * 70)
    rows = [(grp, race_params[grp].get("value"), race_params[grp].get("mae"))
            for grp in ["Black", "Hispanic"] if grp in race_params]
    if "Female" in gender_params:
        rows.append(("Female", gender_params["Female"].get("value"),
                     gender_params["Female"].get("mae")))
    print(
        f"  {'Group':<12} {'Bias (log-odds)':>16} {'Odds Ratio':>12} "
        f"{'Cal MAE':>9} {'Final Rate':>11} {'Final Gap':>10}"
    )
    print(f"  {'-'*76}")

    validation_agg = {}
    if stage3_results and stage3_results.get("validation"):
        validation_agg = {
            int(k): v for k, v in stage3_results["validation"]["agg"].items()
        }

    for grp, b, m in rows:
        if b is None:
            continue

        rate_mae = None
        gap_mae = None

        if validation_agg:
            ref = "Male" if grp == "Female" else "White"
            rate_errors = []
            gap_errors = []

            for win in WINDOWS:
                sim = validation_agg[win].get(grp, {}).get("mean")
                if sim is not None:
                    rate_errors.append(abs(sim - BJS_SUBGROUP[win][grp]))

                g = validation_agg[win].get(grp, {}).get("mean")
                r = validation_agg[win].get(ref, {}).get("mean")
                if g is not None and r is not None:
                    bjs_gap = BJS_SUBGROUP[win][grp] - BJS_SUBGROUP[win][ref]
                    gap_errors.append(abs((g-r) - bjs_gap))

            if rate_errors:
                rate_mae = float(np.mean(rate_errors))
            if gap_errors:
                gap_mae = float(np.mean(gap_errors))

        rate_txt = f"{rate_mae:.4f}" if rate_mae is not None else "-"
        gap_txt = f"{gap_mae:.4f}" if gap_mae is not None else "-"

        print(
            f"  {grp:<12} {b:>16.3f} {math.exp(b):>11.3f}x "
            f"{m:>9.4f} {rate_txt:>11} {gap_txt:>10}"
        )

    print(f"  {'White':<12} {'Reference (0.00)':>16} {'1.000x':>12} "
          f"{'-':>9} {'-':>11} {'-':>10}")
    print(f"  {'Male':<12} {'Reference (0.00)':>16} {'1.000x':>12} "
          f"{'-':>9} {'-':>11} {'-':>10}")

    if validation_agg:
        aggregate_mae = np.mean([
            abs(validation_agg[w]["Overall"]["mean"] - CONFIG["bjs_targets"][w])
            for w in WINDOWS
        ])
        print(f"\n  Overall aggregate BJS MAE: {aggregate_mae*100:.2f} pp")

    print("=" * 76)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import os as _os
    for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ.setdefault(_env, "1")

    multiprocessing.freeze_support()
    args = sys.argv[1:]

    if "--replot" in args:
        outdir = CONFIG["output_directory"]
        race_params = {}
        gender_params = {}
        for grp in ["Black", "Hispanic"]:
            cp = os.path.join(outdir, f"stage1_{grp}.json")
            if os.path.exists(cp):
                with open(cp) as f:
                    race_params[grp] = json.load(f)
        cp = os.path.join(outdir, "stage2_gender.json")
        if os.path.exists(cp):
            with open(cp) as f:
                gender_params = json.load(f)
        cp3 = os.path.join(outdir, "stage3_validation.json")
        if os.path.exists(cp3):
            with open(cp3) as f:
                validation = json.load(f)
            sensitivity = {}
            cp3s = os.path.join(outdir, "stage3_sensitivity.json")
            if os.path.exists(cp3s):
                with open(cp3s) as f:
                    sensitivity = json.load(f)
            plot_stage1_calibration(race_params, outdir)
            plot_stage2_calibration(gender_params, outdir)
            plot_stage3_intersectional(validation, outdir)
            plot_stage3_sensitivity(sensitivity, outdir)
            plot_disparity_summary(validation, outdir)

            generate_dissertation_outputs(
                validation,
                race_params,
                gender_params,
                outdir,
            )
        print("OK  Charts regenerated.")
        sys.exit(0)

    force_rerun = "--rerun" in args

    cli_scope = None
    if "--scope" in args:
        idx = args.index("--scope")
        try:
            scope_arg = args[idx + 1]
            if scope_arg in ("supervision_only", "all_channels"):
                BIAS_SCOPE = scope_arg
                cli_scope  = scope_arg
                print(f"\n  Scope override: BIAS_SCOPE = {BIAS_SCOPE}")
            else:
                print(f"  Unknown scope '{scope_arg}'. "
                      f"Use: supervision_only | all_channels")
                sys.exit(1)
        except IndexError:
            print("Usage: python phase2_bias_all_channels.py "
                  "--scope supervision_only|all_channels")
            sys.exit(1)

    stages = None
    if "--stage" in args:
        idx = args.index("--stage")
        try:
            stages = [int(args[idx + 1])]
        except (IndexError, ValueError):
            print("Usage: python phase2_bias_all_channels.py --stage {1|2|3}")
            sys.exit(1)

    if "--cores" in args:
        idx = args.index("--cores")
        try:
            n_workers = int(args[idx + 1])
            print(f"\n  Manual override: {n_workers} workers")
        except (IndexError, ValueError):
            print("Usage: python phase2_bias_all_channels.py [--cores N] "
                  "[--stage N] [--scope all_channels|supervision_only] "
                  "[--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  Core auto-detection: {reason}")

    main(n_workers=n_workers, stages=stages, force_rerun=force_rerun,
         bias_scope=cli_scope)