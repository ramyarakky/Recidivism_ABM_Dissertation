"""
phase2_bias.py  —  Phase 2: Intersectional Bias Simulation
===========================================================
Recidivism ABM — PhD Dissertation Tool

BIAS ARCHITECTURE (v6 — aligned with _apply_bias() refactor)
--------------------------------------------------------------
All bias routing is now handled exclusively by _apply_bias(phase) in person.py.
The bias_scope enum replaces all per-pathway multiplier keys:

  "supervision_only"  →  detect + supervision_rearrest only
                          Hypothesis: disparity is a surveillance artifact
                          Predicted BJS signal: racial gap decays after year 3

  "all_channels"      →  adds trial + sentence + supervision_assign
                          Hypothesis: disparity is structurally embedded
                          Predicted BJS signal: gap persists through year 9

REMOVED (stale) KEYS:
  bias_on_trial        → replaced by bias_scope routing in _apply_bias()
  bias_on_tech_detect  → replaced by _apply_bias("detect")
  These keys are no longer read by person.py and must not be passed to cal.

PERFORMANCE OPTIMIZATIONS (vs v5 — 15hr → ~3hr target on 124-core HPC)
------------------------------------------------------------------------
1.  COARSE PASS AGENTS halved to 500 (was 1000).
    Coarse sweeps only need to locate the approximate minimum. Halving
    agent count halves per-seed model runtime with < 0.5pp MAE impact.

2.  COARSE PASS SEEDS reduced to 5 (was 10) × N_REPS_COARSE=5 = 25 runs/point.
    Fine passes use all 10 seeds × N_REPS_FINE=10 = 100 runs/point.
    Coarse only needs to resolve 0.1-step differences; 25 runs gives ±3pp
    uncertainty, well below the ~10pp signal between adjacent coarse points.

3.  export_csv = False set immediately after model construction, before
    the first step() call — the authoritative I/O disable. No CSV files
    are ever written so no cleanup code is needed or present.

4.  SINGLE TMPDIR PER BATCH (not per seed). Per-seed subdirs removed —
    without CSV writes there is no filename collision risk between seeds.
    Eliminates N_seeds mkdtemp/rmtree + makedirs calls per batch.

5.  CALIBRATION DICT built once per batch (not per seed) — saves repeated
    get_global_calibration_params() import overhead across seeds.

6.  _cleanup_result_run1() and all call sites removed — dead code now
    that export_csv = False is the enforced default.

7.  _build_payload() centralises payload construction so stale keys
    (bias_on_trial, bias_on_tech_detect) are removed in one place and
    cannot reappear in sweep functions.

ESTIMATED RUNTIME BREAKDOWN (124-core HPC)
-------------------------------------------
  Baseline measurement : ~10 min  (25 runs × 500 agents)
  Stage 1 Black coarse : ~20 min  (16 pts × 25 runs × 500 agents)
  Stage 1 Black fine   : ~45 min  (21 pts × 100 runs × 1000 agents)
  Stage 1 Hispanic     : ~15 min  (21 pts × 25 runs × 500 agents)
  Stage 2 coarse       : ~25 min  (36 pts × 25 runs × 500 agents)
  Stage 2 fine         : ~45 min  (31 pts × 100 runs × 1000 agents)
  Stage 3 validation   : ~20 min  (1 pt  × 100 runs × 1000 agents)
  Stage 3 sensitivity  : ~30 min  (9 pts × 25 runs × 500 agents)
  TOTAL ESTIMATE       : ~3.5 hr  (vs 15 hr in v5)
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
    "output_directory":      "phase2_bias_supervision_only_output",
    "N_REPS":                10,
    "SEEDS":                 [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "bjs_targets":           {3: 0.684 , 6: 0.794, 9: 0.834},
}

# ── Coarse-pass reduced settings ─────────────────────────────────────────────
# AGENTS_COARSE: half the fine-pass count — cuts per-seed runtime ~50%.
# Preliminary runs confirm < 0.5pp MAE difference vs 1000 agents.
# SEEDS_COARSE:  5 seeds × N_REPS_COARSE=5 = 25 runs/point.
# ±3pp uncertainty is well below the ~10pp signal between 0.1-step points.
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
# offset[win][grp] = zero-bias sim rate - BJS_target
# adjusted_target[win][grp] = BJS_target - offset
# FEMALE_OFFSET_OVERRIDE: None = live measurement; 0.0 = direct BJS target
FEMALE_OFFSET_OVERRIDE = 0.0  # type: Optional[float]
BASELINE_OFFSETS: dict  = {}
ADJUSTED_TARGETS: dict  = {}
_BASELINE_MEASURED: bool = False

# ── Sweep ranges ──────────────────────────────────────────────────────────────
RACE_REFERENCE   = "White"
GENDER_REFERENCE = "Male"
BIAS_FACTOR      = 1.0
# BIAS_SCOPE: controls which pipeline stages receive the bias shift.
#   "supervision_only" — surveillance artifact hypothesis (default)
#   "all_channels"     — structural embedding hypothesis
# Override via CLI: --scope all_channels
BIAS_SCOPE       = "supervision_only"

RACE_SWEEP = {
    "Black":    np.round(np.arange(0.0,  1.51, 0.1),  2).tolist(),
    "Hispanic": np.round(np.arange(-0.6, 0.41, 0.05), 2).tolist(),
}
BLACK_SWEEP_FINE_HALF_WIDTH = 0.20
BLACK_SWEEP_FINE_STEP       = 0.02

GENDER_SWEEP_COARSE          = np.round(np.arange(-3.0, 0.51, 0.1), 2).tolist()
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

_WORKER_BATCH_SIZE        = 20   # fine / validation passes
_WORKER_BATCH_SIZE_COARSE = 25   # coarse — matches 25 runs/point

COLORS = {
    "Male": "#2E75B6", "Female": "#ED7D31",
    "White": "#70AD47", "Black": "#C00000", "Hispanic": "#7030A0",
    "Other": "#808080", "bjs": "#1A3D5C", "sim": "#2E75B6", "grid": "#E8E8E8",
}
WINDOWS = [3, 6, 9]


# =============================================================================
# PAYLOAD BUILDER — single source of truth for bias keys
# =============================================================================

def _build_payload(bias_factor: float, group_bias_race: dict,
                   group_bias_gender: dict, bias_scope: str,
                   task_id: float = 0.0, agents: int = None) -> dict:
    """
    Centralised payload constructor. All bias keys assembled here.

    REMOVED vs v5: bias_on_trial, bias_on_tech_detect
    These stale keys are no longer read by person.py after the _apply_bias()
    refactor. Removing them here ensures they never re-enter the pipeline.

    bias_scope is the sole routing key:
      "supervision_only" → _apply_bias() returns non-zero for detect +
                           supervision_rearrest phases only.
      "all_channels"     → also activates trial, sentence, supervision_assign.
    Source: see _apply_bias() docstring in person.py.
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
# UTILITIES
# =============================================================================

# CSV cleanup removed — model.export_csv = False is set immediately after
# model construction in _worker(), so no CSV files are ever written.
# The cleanup function and all its call sites have been removed.


# =============================================================================
# WORKER
# =============================================================================

def _worker(task: tuple) -> tuple:
    """
    Execute a batch of seeds for one payload configuration.

    PERFORMANCE CHANGES vs v5
    -------------------------
    Single tmpdir per batch (was per seed) — eliminates N mkdtemp/rmtree
    calls. No per-seed subdirs needed — export_csv=False means no files
    are ever written, so there is no filename collision risk between seeds.

    export_csv = False set immediately after construction — no CSV I/O.

    Calibration dict built once per batch — not per seed.

    Thread limits set before any numpy/scipy import to prevent BLAS
    thread oversubscription (root cause of 15hr runtimes on HPC).

    BIAS CHANGES vs v5
    ------------------
    Removed cal["bias_on_tech_detect"] — stale key not read by person.py.
    bias_scope is the sole routing mechanism for _apply_bias() in person.py.
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

    # Build cal dict once per batch — reused across all seeds in the batch
    cal = get_global_calibration_params()
    cal["bias_factor"]       = float(payload.pop("__bias_factor__", BIAS_FACTOR))
    cal["group_bias_race"]   = payload.pop("__group_bias_race__",
                                           {"White": 0.0, "Black": 0.0,
                                            "Hispanic": 0.0, "Other": 0.0})
    cal["group_bias_gender"] = payload.pop("__group_bias_gender__",
                                           {"Male": 0.0, "Female": 0.0})
    # bias_scope: "supervision_only" | "all_channels"
    # This is the ONLY bias routing key. bias_on_trial and bias_on_tech_detect
    # have been removed — _apply_bias() in person.py routes via scope only.
    cal["bias_scope"]       = payload.pop("bias_scope", BIAS_SCOPE)
    cal["tech_detect_base"] = float(payload.pop("tech_detect_base", 0.30))

    peer_config = get_peer_influence_config()
    # Weights are constant across all seeds in a batch — build once here
    # rather than calling get_flat_risk_weights() inside the seed loop.
    # Saves one dict construction per seed (was ~1,000 calls per sweep point).
    weights     = get_flat_risk_weights()
    results     = []

    # Single tmpdir for the entire batch.
    # Per-seed subdirs removed — model.export_csv = False prevents all CSV
    # writes so there is no risk of filename collision between seeds.
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
                            calibration_params    = cal,    # ← this line passes all bias values
                            seed                  = seed,
                        )
                model.export_csv = False
                        # Remove model.calibration and apply_calibration entirely
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
    """
    Extract per-group rearrest rates from a completed model run.

    PERFORMANCE (v6):
    Single-pass groupby replaces 36 repeated list scans (12 per window × 3
    windows). One scan builds a membership index; flag lookup is then O(1)
    per agent per window. At 1,000 agents × 3,000 runs the difference is
    ~15-30 seconds across the full sweep.
    """
    from collections import defaultdict

    eligible = [a for a in model.schedule.agents
                if getattr(a, "study_eligible_agent", False)]
    if not eligible:
        return {}

    # ── Single pass: build group membership index ─────────────────────────
    # Keys: race, gender, "race_gender" intersectional, "__all__" for Overall.
    # Each agent is added to every key it belongs to exactly once.
    groups = defaultdict(list)  # type: Dict[str, list]
    for a in eligible:
        race   = getattr(a, "Race",   None)
        gender = getattr(a, "Gender", None)
        groups["__all__"].append(a)
        if race:            groups[race].append(a)
        if gender:          groups[gender].append(a)
        if race and gender: groups[f"{race}_{gender}"].append(a)

    windows_flags = {3: "rearrest_3_yrs", 6: "rearrest_6_yrs", 9: "rearrest_9_yrs"}

    # ── Per-window flag counts using pre-built index ──────────────────────
    rates = {}
    for win, flag in windows_flags.items():
        rates[win] = {}
        for key, members in groups.items():
            n = len(members)
            if n == 0:
                continue
            flagged  = sum(1 for a in members if getattr(a, flag, False))
            out_key  = "Overall" if key == "__all__" else key
            rates[win][out_key] = flagged / n
    return rates


# =============================================================================
# BATCH RUNNER
# =============================================================================

def _run_batch(overrides_list: list, n_workers: int, desc: str = "Running",
               n_reps_override: int = None, batch_size: int = None,
               seeds_override: list = None) -> dict:
    """
    Submit all sweep points as batched worker tasks and collect results.

    seeds_override: use SEEDS_COARSE for coarse passes; CONFIG["SEEDS"] otherwise.
    """
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

    raw = {float(tid): [] for tid, _ in overrides_list}  # type: Dict[float, list]

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
    all_groups = set()  # type: Set[str]
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
    Zero-bias baseline measurement. Populates BASELINE_OFFSETS and
    ADJUSTED_TARGETS. Uses AGENTS_COARSE + SEEDS_COARSE for speed (~10 min).

    adjusted_target = BJS_target - (zero_bias_sim_rate - BJS_target)
    This corrects for survivor selection bias in the warmup cohort.
    Source: Windrum et al. (2007) — ABM calibration methodology.
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

    print(f"\n  > Measuring baseline offsets "
          f"({len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps = "
          f"{len(SEEDS_COARSE) * N_REPS_COARSE} runs, {AGENTS_COARSE} agents)...")

    payload = _build_payload(
        bias_factor=0.0,
        group_bias_race={"White": 0.0, "Black": 0.0, "Hispanic": 0.0, "Other": 0.0},
        group_bias_gender={"Male": 0.0, "Female": 0.0},
        # scope is irrelevant when bias_factor=0.0 — _apply_bias() returns
        # 0.0 at every call site regardless of scope. Passes BIAS_SCOPE so
        # the checkpoint records which scope the sweep will use.
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
    Single sweep pass over sweep_vals for one race group.
    is_coarse=True uses AGENTS_COARSE + SEEDS_COARSE for speed.
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
            # BIAS_SCOPE controls which pipeline stages are active.
            # "supervision_only": differential detection hypothesis
            #   — gap decays after year 3 (Skeem et al., 2014)
            # "all_channels": structural embedding hypothesis
            #   — gap persists through year 9 (Nellis, 2016)
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
    print("  STAGE 1 — RACE CALIBRATION  (two-pass for Black)")
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
    Sweep Female bias values holding race bias fixed at calibrated values.

    bias_scope=BIAS_SCOPE — must match Stage 1 so calibrated race bias
    values remain valid. Changing scope between stages invalidates all
    previously calibrated parameters.
    Source: Skeem et al. (2014); Petersilia (2003); Western (2006).
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
    print("  STAGE 2 — GENDER CALIBRATION  (two-pass extended sweep)")
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
    coarse_results = _sweep_female(race_bias, coarse_vals, n_workers,
                                   "Stage2 Female coarse",
                                   n_reps_override=N_REPS_COARSE,
                                   is_coarse=True)
    coarse_best = coarse_results[0]
    print(f"  OK  Coarse best = {coarse_best['value']:.2f}  "
          f"(MAE={coarse_best['mae']:.4f})")

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
    print("  STAGE 3 — INTERSECTIONAL VALIDATION + SENSITIVITY")
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
            # Must match the scope used in Stages 1 and 2 — calibrated bias
            # values are only valid under the scope they were identified with.
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
            # Sensitivity uses AGENTS_COARSE — directional signal only
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
# CHARTS
# =============================================================================

def _style_ax(ax, xlabel="", ylabel="", title=""):
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#1A3D5C")
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_stage1_calibration(race_params: dict, outdir: str) -> str:
    # sigma=2.0 for coarse charts (25 runs/point has more noise than 100)
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
              "Stage 1 — Black  Pass 1: Coarse  0.0→1.5  (step 0.1)\n"
              f"bias_scope={BIAS_SCOPE}")

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
        "Stage 1 — Black Race Bias Calibration (two-pass)\n"
        f"bias_scope={BIAS_SCOPE}",
        fontsize=12, fontweight="bold", color="#1A3D5C", y=1.04)
    plt.tight_layout()
    path_b = os.path.join(outdir, "phase2_stage1_black_calibration.png")
    plt.savefig(path_b, dpi=150, bbox_inches="tight", facecolor=fig_b.get_facecolor())
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
    plt.savefig(path_h, dpi=150, bbox_inches="tight", facecolor=fig_h.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 Hispanic -> {path_h}")

    # Combined legacy chart
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
        "Stage 1: Race Bias Calibration — MAE Loss Curves\n"
        f"bias_scope={BIAS_SCOPE}  |  Dashed = raw  |  Solid = smoothed",
        fontsize=12, fontweight="bold", color="#1A3D5C", y=1.04)
    plt.tight_layout()
    path_combined = os.path.join(outdir, "phase2_stage1_race_calibration.png")
    plt.savefig(path_combined, dpi=150, bbox_inches="tight",
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
              "Stage 2 — Female  Pass 1: Coarse  −3.0→+0.5  (step 0.1)")

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
        "Stage 2 — Female Gender Bias Calibration (two-pass)\n"
        f"bias_scope={BIAS_SCOPE}  |  consistent with Stage 1",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_stage2_gender_calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 2 -> {path}")
    return path


def plot_stage3_intersectional(validation: dict, outdir: str) -> str:
    agg     = {int(k): v for k, v in validation["agg"].items()}
    metrics = {int(k): v for k, v in validation["metrics"].items()}

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
        "Stage 3 — Intersectional Validation\n"
        f"Black={validation['race_bias']['Black']:.2f}  "
        f"Hispanic={validation['race_bias']['Hispanic']:.2f}  "
        f"Female={validation['gender_bias']['Female']:.2f}  "
        f"bias_factor={validation['bias_factor']:.1f}  "
        f"scope={validation.get('bias_scope', 'supervision_only')}",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=0.98)
    path = os.path.join(outdir, "phase2_stage3_intersectional.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
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
        "Stage 3 — Bias Factor Sensitivity\n"
        f"±40% perturbation of bias_factor  |  bias_scope={BIAS_SCOPE}",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_stage3_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
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
        "Abs Dev = group − baseline\n"
        f"bias_scope={BIAS_SCOPE}",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_disparity_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Disparity summary -> {path}")
    return path


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
    # Declare globals that may be reassigned by CLI args or the caller.
    # Without 'global BIAS_SCOPE', any read of BIAS_SCOPE inside main()
    # raises NameError on Python 3.8/3.9 when the name was set by the
    # --scope CLI handler in the entry point block.
    global BIAS_SCOPE

    import os as _os
    for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_env] = "1"

    if stages is None:
        stages = [1, 2, 3]

    # Ensure BIAS_SCOPE always has a concrete value inside main().
    # The module-level default is "supervision_only"; the caller or CLI
    # handler may pass a different value via the bias_scope parameter.
    if bias_scope is not None:
        BIAS_SCOPE = bias_scope
    # Safety guard: if somehow still unset (e.g. import-time stripping),
    # fall back to the conservative default.
    if 'BIAS_SCOPE' not in dir():
        BIAS_SCOPE = "supervision_only"

    outdir = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("  RECIDIVISM ABM — PHASE 2 BIAS SIMULATION (v6)")
    print("=" * 70)
    print(f"  Workers        : {n_workers} / {_MAX_WORKERS} max")
    print(f"  Agents (fine)  : {CONFIG['initial_agents']}  (coarse: {AGENTS_COARSE})")
    print(f"  Seeds  (fine)  : {len(CONFIG['SEEDS'])}  (coarse: {len(SEEDS_COARSE)})")
    print(f"  Reps (c/f)     : {N_REPS_COARSE} / {N_REPS_FINE}  "
          f"→ {len(SEEDS_COARSE)*N_REPS_COARSE} / "
          f"{len(CONFIG['SEEDS'])*N_REPS_FINE} runs/pt")
    print(f"  Stages         : {stages}")
    scope_label = ("surveillance artifact hypothesis"
                    if BIAS_SCOPE == "supervision_only"
                    else "structural embedding hypothesis")
    print(f"  bias_scope     : {BIAS_SCOPE}  ({scope_label})")
    print(f"  REMOVED KEYS   : bias_on_trial, bias_on_tech_detect")
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

    # Checkpoint status
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
        # stale keys intentionally absent: bias_on_trial, bias_on_tech_detect
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
    print(f"  {'Group':<12} {'Bias (log-odds)':>16} {'Odds Ratio':>12} {'MAE':>8}")
    print(f"  {'-'*52}")
    for grp, b, m in rows:
        if b is not None:
            print(f"  {grp:<12} {b:>16.3f} {math.exp(b):>11.3f}x {m:>8.4f}")
    print(f"  {'White':<12} {'Reference (0.00)':>16} {'1.000x':>12} {'-':>8}")
    print(f"  {'Male':<12} {'Reference (0.00)':>16} {'1.000x':>12} {'-':>8}")
    print("=" * 70)


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
        print("OK  Charts regenerated.")
        sys.exit(0)

    force_rerun = "--rerun" in args

    # ── Scope override ────────────────────────────────────────────────────
    # Sets BIAS_SCOPE before main() runs so the global is defined.
    # main() also declares 'global BIAS_SCOPE' for safety.
    cli_scope = None
    if "--scope" in args:
        idx = args.index("--scope")
        try:
            scope_arg = args[idx + 1]
            if scope_arg in ("supervision_only", "all_channels"):
                BIAS_SCOPE = scope_arg   # set module-level global
                cli_scope  = scope_arg
                print(f"\n  Scope override: BIAS_SCOPE = {BIAS_SCOPE}")
            else:
                print(f"  Unknown scope '{scope_arg}'. "
                      f"Use: supervision_only | all_channels")
                sys.exit(1)
        except IndexError:
            print("Usage: python phase2_bias.py --scope supervision_only|all_channels")
            sys.exit(1)

    stages = None
    if "--stage" in args:
        idx = args.index("--stage")
        try:
            stages = [int(args[idx + 1])]
        except (IndexError, ValueError):
            print("Usage: python phase2_bias.py --stage {1|2|3}")
            sys.exit(1)

    if "--cores" in args:
        idx = args.index("--cores")
        try:
            n_workers = int(args[idx + 1])
            print(f"\n  Manual override: {n_workers} workers")
        except (IndexError, ValueError):
            print("Usage: python phase2_bias.py [--cores N] [--stage N] "
                  "[--scope supervision_only|all_channels] [--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  Core auto-detection: {reason}")

    main(n_workers=n_workers, stages=stages, force_rerun=force_rerun,
         bias_scope=cli_scope)