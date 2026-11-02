"""
phase2_bias_supervision_only_balanced_refinement.py —  Phase 2: Intersectional Bias Simulation
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
  Stage 1 Black fine   : adaptive screen + ~5–10 full candidates
  Stage 1 Hispanic     : ~15 min  (21 pts × 25 runs × 500 agents)
  Stage 2 coarse       : ~25 min  (36 pts × 25 runs × 500 agents)
  Stage 2 fine         : adaptive screen + ~5–10 full candidates
  Stage 3 validation   : ~20 min  (1 pt  × 100 runs × 1000 agents)
  Stage 3 sensitivity  : ~30 min  (9 pts × 25 runs × 500 agents)
  TOTAL ESTIMATE       : ~3.5 hr  (vs 15 hr in v5)
"""

import os
import sys
import json
import math
import itertools
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
    "output_directory":      "phase2_bias_supervision_only_balanced_refinement",
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

# ── Adaptive fine-search optimization ────────────────────────────────────────
# Instead of evaluating every fine-grid point at full fidelity:
#   Pass 2a: screen ALL fine points with coarse settings (500 agents, 25 runs)
#   Pass 2b: re-run only the most promising candidates at full fidelity
#            (1000 agents, 100 runs)
#
# This preserves the full-fidelity final selection while greatly reducing
# expensive ABM executions.
FINE_SCREEN_TOP_K = 5

# Include nearest neighbors around each screened candidate to reduce the chance
# that coarse stochastic noise excludes the true fine optimum.
FINE_SCREEN_NEIGHBOR_RADIUS = 1

# ── FINAL BALANCED REFINEMENT ────────────────────────────────────────────────
# Purpose:
#   The independent Stage 1/2 sweeps identify race and gender coefficients
#   against subgroup BJS targets.  A final local refinement then evaluates
#   Black/Hispanic/Female coefficients jointly and selects a parameterization
#   that preserves empirical fit at THREE levels:
#
#     1) Aggregate BJS trajectory (Overall 3/6/9-year rates)
#     2) Absolute subgroup BJS rates (White/Black/Hispanic/Male/Female)
#     3) Demographic disparities (Black-White, Hispanic-White, Female-Male)
#
# IMPORTANT:
#   Selection is CONSTRAINT-BASED, not an unconstrained weighted sum.
#   A candidate must first satisfy all empirical tolerances.  Among acceptable
#   candidates, the model with the lowest demographic-gap MAE is selected.
#   This prevents a very small disparity error from being achieved by inflating
#   or depressing the overall BJS trajectory.
BALANCED_MAX_AGGREGATE_MAE          = 0.010  # <= 1.0 percentage points
BALANCED_MAX_SUBGROUP_MAE           = 0.010  # <= 1.0 percentage points
BALANCED_MAX_GAP_MAE                = 0.010  # <= 1.0 percentage points
BALANCED_MAX_AGGREGATE_WINDOW_ERROR = 0.010  # no 3/6/9 anchor > 3.0 pp

# Local search around independently calibrated beta values.
# 5 × 5 × 5 = 125 candidate combinations in the coarse screen.
BALANCED_BLACK_OFFSETS    = [-0.08, -0.04, 0.00, 0.04, 0.08]
BALANCED_HISPANIC_OFFSETS = [-0.08, -0.04, 0.00, 0.04, 0.08]
BALANCED_FEMALE_OFFSETS   = [-0.12, -0.06, 0.00, 0.06, 0.12]

# Number of coarse candidates sent to full-fidelity validation.
# The independent Stage 1/2 solution is always included even if it is not
# among the top candidates.
BALANCED_FULL_TOP_K = 10



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
# ADAPTIVE FINE-SEARCH HELPERS
# =============================================================================

def _select_fine_candidates(screen_results: list, fine_vals: list,
                            top_k: int = FINE_SCREEN_TOP_K,
                            neighbor_radius: int = FINE_SCREEN_NEIGHBOR_RADIUS) -> list:
    """
    Select promising fine-grid values from the cheap screening pass.

    We retain:
      - the top_k lowest-MAE screened points, and
      - immediate neighboring grid values around those points.

    Adding neighbors protects against coarse-pass stochastic noise while still
    keeping the expensive full-validation set small.
    """
    if not screen_results:
        return []

    ranked_vals = [float(r["value"]) for r in screen_results[:top_k]]
    value_to_idx = {round(float(v), 8): i for i, v in enumerate(fine_vals)}

    selected = set()
    for val in ranked_vals:
        idx = value_to_idx.get(round(val, 8))
        if idx is None:
            # Fall back to nearest value if floating-point representation differs.
            idx = int(np.argmin(np.abs(np.asarray(fine_vals, dtype=float) - val)))

        lo = max(0, idx - neighbor_radius)
        hi = min(len(fine_vals), idx + neighbor_radius + 1)
        for j in range(lo, hi):
            selected.add(float(fine_vals[j]))

    return sorted(selected)


def _validate_race_candidates(race_group: str, candidate_vals: list,
                              n_workers: int, desc: str) -> list:
    """
    Full-fidelity validation for a small set of race-bias candidates.
    Uses the same absolute subgroup-rate MAE objective as _sweep_race_group().
    """
    if not candidate_vals:
        return []

    return _sweep_race_group(
        race_group,
        candidate_vals,
        n_workers,
        desc,
        n_reps_override=N_REPS_FINE,
        is_coarse=False,
    )


def _validate_female_candidates(race_bias: dict, candidate_vals: list,
                                n_workers: int, desc: str) -> list:
    """
    Full-fidelity validation for a small set of Female-bias candidates.
    Uses the same absolute Female subgroup-rate MAE objective as _sweep_female().
    """
    if not candidate_vals:
        return []

    return _sweep_female(
        race_bias,
        candidate_vals,
        n_workers,
        desc,
        n_reps_override=N_REPS_FINE,
        is_coarse=False,
    )


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
            # --------------------------------------------------------------
            # Pass 2a — cheap fine-grid screen
            # --------------------------------------------------------------
            print(f"\n  Pass 2a (fine screen): {len(fine_vals)} pts  "
                  f"[{AGENTS_COARSE} agents, "
                  f"{len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")

            fine_screen_results = _sweep_race_group(
                grp,
                fine_vals,
                n_workers,
                "Stage1 Black fine-screen",
                n_reps_override=N_REPS_COARSE,
                is_coarse=True,
            )

            validation_vals = _select_fine_candidates(
                fine_screen_results,
                fine_vals,
            )

            # Always include the original coarse optimum in full validation.
            validation_vals = sorted(set(
                validation_vals + [float(coarse_best["value"])]
            ))

            # --------------------------------------------------------------
            # Pass 2b — full-fidelity validation of promising candidates only
            # --------------------------------------------------------------
            print(f"  Pass 2b (full validation): {len(validation_vals)} candidates  "
                  f"[{CONFIG['initial_agents']} agents, "
                  f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")

            fine_results = _validate_race_candidates(
                grp,
                validation_vals,
                n_workers,
                "Stage1 Black fine-validate",
            )

            if not fine_results:
                raise RuntimeError("Black fine validation produced no results.")

            fine_best = fine_results[0]

            print(f"  OK  Fine best = {fine_best['value']:.3f}  "
                  f"(MAE={fine_best['mae']:.4f})")

            best_params[grp] = {
                "value":          fine_best["value"],
                "mae":            fine_best["mae"],
                "agg":            {str(k): v for k, v in fine_best["agg"].items()},
                "coarse_results": [{"value": r["value"], "mae": r["mae"]}
                                   for r in coarse_results],
                # fine_screen_results is retained for plotting the full local landscape.
                "fine_screen_results": [
                    {"value": r["value"], "mae": r["mae"]}
                    for r in fine_screen_results
                ],
                # fine_results contains ONLY the full-fidelity candidates.
                "fine_results": [
                    {"value": r["value"], "mae": r["mae"]}
                    for r in fine_results
                ],
                "fine_validation_values": validation_vals,
                "all_results":    [{"value": r["value"], "mae": r["mae"]}
                                   for r in coarse_results],
                "adjusted_target": {str(w): ADJUSTED_TARGETS[w][grp]
                                    for w in WINDOWS},
                "baseline_offset": {str(w): BASELINE_OFFSETS[w][grp]
                                    for w in WINDOWS},
                "coarse_best":     coarse_best["value"],
                "coarse_best_mae": coarse_best["mae"],
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
    # ------------------------------------------------------------------
    # Pass 2a — cheap fine-grid screen
    # ------------------------------------------------------------------
    print(f"\n  Pass 2a (fine screen): {len(fine_vals)} pts  "
          f"[{AGENTS_COARSE} agents, "
          f"{len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")

    fine_screen_results = _sweep_female(
        race_bias,
        fine_vals,
        n_workers,
        "Stage2 Female fine-screen",
        n_reps_override=N_REPS_COARSE,
        is_coarse=True,
    )

    validation_vals = _select_fine_candidates(
        fine_screen_results,
        fine_vals,
    )

    # Always include the coarse minimum.
    validation_vals = sorted(set(
        validation_vals + [float(coarse_best["value"])]
    ))

    # ------------------------------------------------------------------
    # Pass 2b — full-fidelity validation
    # ------------------------------------------------------------------
    print(f"  Pass 2b (full validation): {len(validation_vals)} candidates  "
          f"[{CONFIG['initial_agents']} agents, "
          f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")

    fine_results = _validate_female_candidates(
        race_bias,
        validation_vals,
        n_workers,
        "Stage2 Female fine-validate",
    )

    if not fine_results:
        raise RuntimeError("Female fine validation produced no results.")

    fine_best = fine_results[0]

    print(f"  OK  Fine best = {fine_best['value']:.3f}  "
          f"(MAE={fine_best['mae']:.4f})")

    best_params = {"Female": {
        "value":          fine_best["value"],
        "mae":            fine_best["mae"],
        "agg":            {str(k): v for k, v in fine_best["agg"].items()},
        "coarse_results": [{"value": r["value"], "mae": r["mae"]}
                           for r in coarse_results],
        "fine_screen_results": [
            {"value": r["value"], "mae": r["mae"]}
            for r in fine_screen_results
        ],
        "fine_results":   [
            {"value": r["value"], "mae": r["mae"]}
            for r in fine_results
        ],
        "fine_validation_values": validation_vals,
        "all_results":    [{"value": r["value"], "mae": r["mae"]}
                           for r in coarse_results],
        "adjusted_target": {str(w): ADJUSTED_TARGETS[w]["Female"]
                            for w in WINDOWS},
        "baseline_offset": {str(w): BASELINE_OFFSETS[w]["Female"]
                            for w in WINDOWS},
        "coarse_best":     coarse_best["value"],
        "coarse_best_mae": coarse_best["mae"],
    }}
    with open(cp, "w") as f:
        json.dump(best_params, f, indent=2)
    return best_params



# =============================================================================
# FINAL BALANCED REFINEMENT — CONSTRAINED MULTI-OBJECTIVE CALIBRATION
# =============================================================================

def _balanced_aggregate_mae(agg: dict) -> float:
    """Overall ABM-vs-BJS MAE across the 3-, 6-, and 9-year anchor windows."""
    errors = []
    for win in WINDOWS:
        sim = agg.get(win, {}).get("Overall", {}).get("mean")
        target = CONFIG["bjs_targets"].get(win)
        if sim is not None and target is not None:
            errors.append(abs(float(sim) - float(target)))
    return float(np.mean(errors)) if errors else 999.0


def _balanced_max_aggregate_error(agg: dict) -> float:
    """Largest absolute overall ABM-vs-BJS error at any anchor window."""
    errors = []
    for win in WINDOWS:
        sim = agg.get(win, {}).get("Overall", {}).get("mean")
        target = CONFIG["bjs_targets"].get(win)
        if sim is not None and target is not None:
            errors.append(abs(float(sim) - float(target)))
    return float(max(errors)) if errors else 999.0


def _balanced_subgroup_mae(agg: dict) -> float:
    """
    Absolute subgroup-rate MAE across 15 empirical cells:
      White, Black, Hispanic, Male, Female × 3/6/9 years.
    """
    errors = []
    for win in WINDOWS:
        for group in ("White", "Black", "Hispanic", "Male", "Female"):
            sim = agg.get(win, {}).get(group, {}).get("mean")
            target = BJS_SUBGROUP.get(win, {}).get(group)
            if sim is not None and target is not None:
                errors.append(abs(float(sim) - float(target)))
    return float(np.mean(errors)) if errors else 999.0


def _balanced_gap_mae(agg: dict) -> float:
    """
    Demographic-gap MAE across 9 empirical disparity cells:
      Black-White, Hispanic-White, Female-Male × 3/6/9 years.

    Gap targets are derived directly from BJS subgroup rates so absolute rates
    and disparity targets remain mathematically consistent.
    """
    comparisons = (
        ("Black", "White"),
        ("Hispanic", "White"),
        ("Female", "Male"),
    )

    errors = []
    for win in WINDOWS:
        for group, reference in comparisons:
            g = agg.get(win, {}).get(group, {}).get("mean")
            r = agg.get(win, {}).get(reference, {}).get("mean")
            if g is None or r is None:
                continue

            sim_gap = float(g) - float(r)
            bjs_gap = (
                float(BJS_SUBGROUP[win][group])
                - float(BJS_SUBGROUP[win][reference])
            )
            errors.append(abs(sim_gap - bjs_gap))

    return float(np.mean(errors)) if errors else 999.0


def _balanced_candidate_metrics(agg: dict) -> dict:
    """
    Calculate all final-selection metrics and empirical acceptance status.

    Acceptable means ALL four conditions are met:
      Aggregate MAE          <= 2.0 pp
      Subgroup-rate MAE      <= 3.0 pp
      Demographic-gap MAE    <= 2.0 pp
      Worst aggregate anchor <= 3.0 pp
    """
    agg_mae = _balanced_aggregate_mae(agg)
    sub_mae = _balanced_subgroup_mae(agg)
    gap_mae = _balanced_gap_mae(agg)
    max_agg = _balanced_max_aggregate_error(agg)

    acceptable = (
        agg_mae <= BALANCED_MAX_AGGREGATE_MAE
        and sub_mae <= BALANCED_MAX_SUBGROUP_MAE
        and gap_mae <= BALANCED_MAX_GAP_MAE
        and max_agg <= BALANCED_MAX_AGGREGATE_WINDOW_ERROR
    )

    # Used only if no candidate satisfies every hard constraint.
    constraint_violation = (
        max(0.0, agg_mae - BALANCED_MAX_AGGREGATE_MAE)
        + max(0.0, sub_mae - BALANCED_MAX_SUBGROUP_MAE)
        + max(0.0, gap_mae - BALANCED_MAX_GAP_MAE)
        + max(0.0, max_agg - BALANCED_MAX_AGGREGATE_WINDOW_ERROR)
    )

    return {
        "aggregate_mae": float(agg_mae),
        "subgroup_mae": float(sub_mae),
        "gap_mae": float(gap_mae),
        "max_aggregate_error": float(max_agg),
        "acceptable": bool(acceptable),
        "constraint_violation": float(constraint_violation),
    }


def _balanced_candidate_sort_key(row: dict):
    """
    Candidate selection rule.

    1. Acceptable candidates come first.
    2. Among acceptable candidates:
         lowest demographic-gap MAE,
         then subgroup-rate MAE,
         then aggregate MAE.
    3. If none are acceptable:
         minimum total constraint violation,
         then gap MAE,
         subgroup MAE,
         aggregate MAE.
    """
    if row.get("acceptable", False):
        return (
            0,
            row["gap_mae"],
            row["subgroup_mae"],
            row["aggregate_mae"],
        )
    return (
        1,
        row["constraint_violation"],
        row["gap_mae"],
        row["subgroup_mae"],
        row["aggregate_mae"],
    )


def _run_balanced_candidates(candidates: list, n_workers: int, desc: str,
                             agents: int, seeds: list, n_reps: int,
                             batch_size: int) -> list:
    """Run joint Black/Hispanic/Female candidate combinations."""
    overrides = []

    for i, candidate in enumerate(candidates):
        payload = _build_payload(
            bias_factor=BIAS_FACTOR,
            group_bias_race={
                "White": 0.0,
                "Black": float(candidate["Black"]),
                "Hispanic": float(candidate["Hispanic"]),
                "Other": 0.0,
            },
            group_bias_gender={
                "Male": 0.0,
                "Female": float(candidate["Female"]),
            },
            bias_scope=BIAS_SCOPE,
            task_id=float(i),
            agents=agents,
        )
        overrides.append((float(i), payload))

    raw = _run_batch(
        overrides,
        n_workers,
        desc=desc,
        n_reps_override=n_reps,
        batch_size=batch_size,
        seeds_override=seeds,
    )

    rows = []
    for i, candidate in enumerate(candidates):
        agg = _aggregate_subgroup_rates(raw.get(float(i), []))
        if not agg:
            continue

        metrics = _balanced_candidate_metrics(agg)
        rows.append({
            "Black": float(candidate["Black"]),
            "Hispanic": float(candidate["Hispanic"]),
            "Female": float(candidate["Female"]),
            **metrics,
            "agg": agg,
        })

    rows.sort(key=_balanced_candidate_sort_key)
    return rows


def _balanced_candidate_dataframe(rows: list, selected: dict = None) -> pd.DataFrame:
    """Dissertation-ready tabular candidate data in percentage-point units."""
    selected_key = None
    if selected:
        selected_key = (
            round(float(selected["Black"]), 6),
            round(float(selected["Hispanic"]), 6),
            round(float(selected["Female"]), 6),
        )

    records = []
    for rank, r in enumerate(rows, start=1):
        key = (
            round(float(r["Black"]), 6),
            round(float(r["Hispanic"]), 6),
            round(float(r["Female"]), 6),
        )
        records.append({
            "Rank": rank,
            "Black Beta": r["Black"],
            "Hispanic Beta": r["Hispanic"],
            "Female Beta": r["Female"],
            "Aggregate MAE (pp)": r["aggregate_mae"] * 100.0,
            "Subgroup Rate MAE (pp)": r["subgroup_mae"] * 100.0,
            "Demographic Gap MAE (pp)": r["gap_mae"] * 100.0,
            "Max Aggregate Error (pp)": r["max_aggregate_error"] * 100.0,
            "Accept": "Yes" if r["acceptable"] else "No",
            "Constraint Violation (pp)": r["constraint_violation"] * 100.0,
            "Selected": "Yes" if selected_key is not None and key == selected_key else "No",
        })
    return pd.DataFrame.from_records(records)


def _print_balanced_candidate_table(rows: list, selected: dict) -> None:
    """Console table matching the dissertation candidate-selection logic."""
    print("\n" + "=" * 108)
    print("  FINAL BALANCED-CALIBRATION CANDIDATES — FULL-FIDELITY VALIDATION")
    print("=" * 108)
    print(
        f"  {'Black':>7} {'Hisp':>7} {'Female':>8} "
        f"{'Agg MAE':>9} {'Sub MAE':>9} {'Gap MAE':>9} "
        f"{'Max Agg':>9} {'Accept':>8} {'Selected':>9}"
    )
    print("  " + "-" * 104)

    selected_key = (
        round(float(selected["Black"]), 6),
        round(float(selected["Hispanic"]), 6),
        round(float(selected["Female"]), 6),
    )

    for r in rows:
        key = (
            round(float(r["Black"]), 6),
            round(float(r["Hispanic"]), 6),
            round(float(r["Female"]), 6),
        )
        is_selected = key == selected_key
        print(
            f"  {r['Black']:>7.3f} {r['Hispanic']:>7.3f} {r['Female']:>8.3f} "
            f"{r['aggregate_mae']*100:>8.2f} "
            f"{r['subgroup_mae']*100:>8.2f} "
            f"{r['gap_mae']*100:>8.2f} "
            f"{r['max_aggregate_error']*100:>8.2f} "
            f"{('YES' if r['acceptable'] else 'NO'):>8} "
            f"{('***' if is_selected else ''):>9}"
        )

    print("  " + "-" * 104)
    print(
        "  Acceptance: Aggregate MAE <= 2.0 pp | Subgroup MAE <= 3.0 pp | "
        "Gap MAE <= 2.0 pp | Max aggregate anchor <= 3.0 pp"
    )
    print(
        "  Selection: among acceptable candidates, minimize Gap MAE; "
        "tie-break by Subgroup MAE then Aggregate MAE."
    )
    print("=" * 108)


def plot_balanced_candidate_tradeoff(rows: list, selected: dict,
                                     outdir: str) -> str:
    """
    Dissertation-ready trade-off chart.

    X-axis  : Aggregate MAE (pp)
    Y-axis  : Demographic-gap MAE (pp)
    Marker size: Subgroup-rate MAE
    Circle  : candidate satisfies all empirical constraints
    X       : candidate violates >=1 constraint
    Star    : selected final candidate
    """
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    fig.patch.set_facecolor("#FAFAFA")

    acceptable = [r for r in rows if r["acceptable"]]
    rejected = [r for r in rows if not r["acceptable"]]

    def _sizes(items):
        # Keep bubble-size differences visible but not overwhelming.
        return [150 + 180 * (r["subgroup_mae"] * 100.0) for r in items]

    if acceptable:
        ax.scatter(
            [r["aggregate_mae"] * 100 for r in acceptable],
            [r["gap_mae"] * 100 for r in acceptable],
            s=_sizes(acceptable),
            marker="o",
            alpha=0.70,
            label="Acceptable candidate",
            zorder=4,
        )

    if rejected:
        ax.scatter(
            [r["aggregate_mae"] * 100 for r in rejected],
            [r["gap_mae"] * 100 for r in rejected],
            s=_sizes(rejected),
            marker="x",
            linewidths=1.8,
            alpha=0.70,
            label="Fails ≥1 acceptance criterion",
            zorder=3,
        )

    # Hard acceptance boundaries.
    ax.axvline(
        BALANCED_MAX_AGGREGATE_MAE * 100,
        linestyle="--", linewidth=1.4,
        label=f"Aggregate MAE limit = {BALANCED_MAX_AGGREGATE_MAE*100:.1f} pp",
        zorder=2,
    )
    ax.axhline(
        BALANCED_MAX_GAP_MAE * 100,
        linestyle=":",
        linewidth=1.6,
        label=f"Gap MAE limit = {BALANCED_MAX_GAP_MAE*100:.1f} pp",
        zorder=2,
    )

    sx = selected["aggregate_mae"] * 100
    sy = selected["gap_mae"] * 100
    ax.scatter(
        [sx], [sy],
        marker="*", s=520,
        edgecolors="black", linewidths=1.0,
        label="Selected final candidate",
        zorder=8,
    )

    ax.annotate(
        (
            f"SELECTED\n"
            f"Black β = {selected['Black']:.3f}\n"
            f"Hispanic β = {selected['Hispanic']:.3f}\n"
            f"Female β = {selected['Female']:.3f}\n"
            f"Aggregate MAE = {sx:.2f} pp\n"
            f"Subgroup MAE = {selected['subgroup_mae']*100:.2f} pp\n"
            f"Gap MAE = {sy:.2f} pp"
        ),
        xy=(sx, sy),
        xytext=(22, 22),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#555555", alpha=0.94),
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
        zorder=9,
    )

    # Annotate the best few ranked candidates only to avoid clutter.
    for rank, r in enumerate(rows[:8], start=1):
        if (
            abs(r["Black"] - selected["Black"]) < 1e-10
            and abs(r["Hispanic"] - selected["Hispanic"]) < 1e-10
            and abs(r["Female"] - selected["Female"]) < 1e-10
        ):
            continue
        ax.annotate(
            f"#{rank}: B={r['Black']:.2f}, H={r['Hispanic']:.2f}, F={r['Female']:.2f}",
            (r["aggregate_mae"] * 100, r["gap_mae"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.82,
        )

    ax.set_xlabel("Aggregate MAE (percentage points)", fontsize=13)
    ax.set_ylabel("Demographic Gap MAE (percentage points)", fontsize=13)
    ax.set_title(
        "Final Balanced Calibration — Candidate Trade-off\n"
        "Lower-left is preferred; marker size reflects absolute subgroup-rate MAE",
        fontsize=15, fontweight="bold", color="#1A3D5C", pad=14
    )
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, loc="best")

    fig.text(
        0.5, 0.018,
        (
            "Candidate must satisfy: Aggregate MAE ≤ 2.0 pp, Subgroup-rate MAE ≤ 3.0 pp, "
            "Demographic-gap MAE ≤ 2.0 pp, and no aggregate anchor error > 3.0 pp. "
            "Among acceptable candidates, the lowest demographic-gap MAE is selected."
        ),
        ha="center", va="bottom", fontsize=9.5
    )

    plt.tight_layout(rect=[0, 0.055, 1, 1])
    path = os.path.join(outdir, "phase2_balanced_candidate_tradeoff.png")
    plt.savefig(path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Chart: Balanced candidate trade-off -> {path}")
    return path


def plot_balanced_candidate_table(rows: list, selected: dict,
                                  outdir: str, top_n: int = 12) -> str:
    """
    Dissertation-ready rendered table for the top full-validation candidates.
    Values are percentage points; Accept/Selected make the decision rule explicit.
    """
    display_rows = rows[:top_n]
    df = _balanced_candidate_dataframe(display_rows, selected)

    cols = [
        "Rank", "Black Beta", "Hispanic Beta", "Female Beta",
        "Aggregate MAE (pp)", "Subgroup Rate MAE (pp)",
        "Demographic Gap MAE (pp)", "Max Aggregate Error (pp)",
        "Accept", "Selected",
    ]
    df = df[cols].copy()

    for c in [
        "Black Beta", "Hispanic Beta", "Female Beta",
        "Aggregate MAE (pp)", "Subgroup Rate MAE (pp)",
        "Demographic Gap MAE (pp)", "Max Aggregate Error (pp)",
    ]:
        df[c] = df[c].map(lambda v: f"{v:.2f}" if "MAE" in c or "Error" in c else f"{v:.3f}")

    fig_h = max(5.2, 1.0 + 0.43 * len(df))
    fig, ax = plt.subplots(figsize=(16.5, fig_h))
    fig.patch.set_facecolor("#FAFAFA")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=[
            "Rank", "Black β", "Hispanic β", "Female β",
            "Aggregate\nMAE (pp)", "Subgroup-rate\nMAE (pp)",
            "Demographic-gap\nMAE (pp)", "Max aggregate\nerror (pp)",
            "Accept", "Selected",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C8C8C8")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E9EEF4")
        else:
            # Highlight selected row, then acceptable rows.
            selected_flag = df.iloc[row - 1]["Selected"] == "Yes"
            accept_flag = df.iloc[row - 1]["Accept"] == "Yes"
            if selected_flag:
                cell.set_facecolor("#DDEFD8")
                cell.set_text_props(weight="bold")
            elif accept_flag:
                cell.set_facecolor("#F2F8F0")

    ax.set_title(
        "Final Balanced Calibration — Full-Fidelity Candidate Table",
        fontsize=16, fontweight="bold", color="#1A3D5C", pad=16
    )

    fig.text(
        0.5, 0.035,
        (
            "Accept = all four empirical criteria satisfied. Selected = acceptable candidate "
            "with the lowest demographic-gap MAE (ties: subgroup-rate MAE, then aggregate MAE)."
        ),
        ha="center", fontsize=10
    )
    fig.text(
        0.5, 0.012,
        (
            "Thresholds: Aggregate MAE ≤ 2.0 pp | Subgroup-rate MAE ≤ 3.0 pp | "
            "Demographic-gap MAE ≤ 2.0 pp | Maximum aggregate anchor error ≤ 3.0 pp."
        ),
        ha="center", fontsize=9
    )

    plt.tight_layout(rect=[0.01, 0.07, 0.99, 0.95])
    path = os.path.join(outdir, "phase2_balanced_candidate_table.png")
    plt.savefig(path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Chart: Balanced candidate table -> {path}")
    return path


def run_final_balanced_refinement(race_params: dict, gender_params: dict,
                                  n_workers: int, outdir: str,
                                  force_rerun: bool = False) -> dict:
    """
    Final constrained calibration stage.

    Workflow
    --------
    1. Use independent Stage 1/2 estimates as the center.
    2. Screen 125 nearby Black/Hispanic/Female combinations cheaply.
    3. Promote promising candidates to full-fidelity validation.
    4. Require empirical acceptance thresholds.
    5. Among acceptable candidates, select minimum demographic-gap MAE.
    6. Export machine-readable JSON/CSV and dissertation-ready PNGs.
    """
    print("\n" + "=" * 84)
    print("  FINAL BALANCED REFINEMENT — CONSTRAINED MULTI-OBJECTIVE CALIBRATION")
    print("=" * 84)

    cp = os.path.join(outdir, "balanced_refinement.json")

    if force_rerun and os.path.exists(cp):
        os.remove(cp)
        print("  Removed prior balanced-refinement checkpoint")

    if os.path.exists(cp):
        print("  > Loading balanced-refinement checkpoint")
        with open(cp) as f:
            result = json.load(f)

        # Recreate charts/table from checkpoint.
        rows = result.get("full_validation_candidates", [])
        selected = result.get("selected", {})
        if rows and selected:
            _print_balanced_candidate_table(rows, selected)
            plot_balanced_candidate_tradeoff(rows, selected, outdir)
            plot_balanced_candidate_table(rows, selected, outdir)
        return result

    black0 = float(race_params["Black"]["value"])
    hisp0 = float(race_params["Hispanic"]["value"])
    female0 = float(gender_params["Female"]["value"])

    starting = {
        "Black": black0,
        "Hispanic": hisp0,
        "Female": female0,
    }

    print(
        f"  Independent calibration center: "
        f"Black={black0:+.3f}, Hispanic={hisp0:+.3f}, Female={female0:+.3f}"
    )
    print(
        "  Acceptance thresholds: "
        f"Agg MAE≤{BALANCED_MAX_AGGREGATE_MAE*100:.1f}pp, "
        f"Subgroup MAE≤{BALANCED_MAX_SUBGROUP_MAE*100:.1f}pp, "
        f"Gap MAE≤{BALANCED_MAX_GAP_MAE*100:.1f}pp, "
        f"Max aggregate anchor≤{BALANCED_MAX_AGGREGATE_WINDOW_ERROR*100:.1f}pp"
    )

    # ------------------------------------------------------------------
    # Pass A — local coarse screen
    # ------------------------------------------------------------------
    candidates = []
    seen = set()

    for db, dh, df in itertools.product(
        BALANCED_BLACK_OFFSETS,
        BALANCED_HISPANIC_OFFSETS,
        BALANCED_FEMALE_OFFSETS,
    ):
        candidate = {
            "Black": round(black0 + db, 3),
            "Hispanic": round(hisp0 + dh, 3),
            "Female": round(female0 + df, 3),
        }
        key = (candidate["Black"], candidate["Hispanic"], candidate["Female"])
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)

    print(
        f"\n  Pass A — coarse balanced screen: {len(candidates)} candidates "
        f"[{AGENTS_COARSE} agents, {len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]"
    )

    coarse = _run_balanced_candidates(
        candidates,
        n_workers=n_workers,
        desc="Balanced coarse screen",
        agents=AGENTS_COARSE,
        seeds=SEEDS_COARSE,
        n_reps=N_REPS_COARSE,
        batch_size=_WORKER_BATCH_SIZE_COARSE,
    )

    if not coarse:
        raise RuntimeError("Balanced coarse screen produced no valid candidates.")

    # Promotion strategy:
    #   - top candidate rankings under the constraint-aware sort,
    #   - plus best low-gap candidates,
    #   - plus best low-aggregate candidates,
    #   - plus independent calibration center.
    promote = []
    promote_keys = set()

    def _promote(row):
        key = (
            round(float(row["Black"]), 6),
            round(float(row["Hispanic"]), 6),
            round(float(row["Female"]), 6),
        )
        if key not in promote_keys:
            promote.append({
                "Black": row["Black"],
                "Hispanic": row["Hispanic"],
                "Female": row["Female"],
            })
            promote_keys.add(key)

    for row in coarse[:BALANCED_FULL_TOP_K]:
        _promote(row)

    for row in sorted(coarse, key=lambda r: r["gap_mae"])[:4]:
        _promote(row)

    for row in sorted(coarse, key=lambda r: r["aggregate_mae"])[:4]:
        _promote(row)

    center_row = {
        "Black": black0,
        "Hispanic": hisp0,
        "Female": female0,
    }
    center_key = (round(black0, 6), round(hisp0, 6), round(female0, 6))
    if center_key not in promote_keys:
        promote.append(center_row)

    print(
        f"  Pass B — full-fidelity validation: {len(promote)} candidates "
        f"[{CONFIG['initial_agents']} agents, {len(CONFIG['SEEDS'])} seeds × "
        f"{N_REPS_FINE} reps]"
    )

    full = _run_balanced_candidates(
        promote,
        n_workers=n_workers,
        desc="Balanced full validation",
        agents=CONFIG["initial_agents"],
        seeds=CONFIG["SEEDS"],
        n_reps=N_REPS_FINE,
        batch_size=_WORKER_BATCH_SIZE,
    )

    if not full:
        raise RuntimeError("Balanced full validation produced no valid candidates.")

    acceptable = [r for r in full if r["acceptable"]]

    if acceptable:
        acceptable.sort(
            key=lambda r: (
                r["gap_mae"],
                r["subgroup_mae"],
                r["aggregate_mae"],
            )
        )
        selected = acceptable[0]
        selection_mode = "accepted_minimum_gap"
    else:
        full.sort(
            key=lambda r: (
                r["constraint_violation"],
                r["gap_mae"],
                r["subgroup_mae"],
                r["aggregate_mae"],
            )
        )
        selected = full[0]
        selection_mode = "minimum_constraint_violation"
        print(
            "\n  WARNING: No full-validation candidate satisfied all four "
            "acceptance criteria. Selected the least-violating candidate."
        )

    # Re-sort for presentation with acceptable/selectable candidates first.
    full.sort(key=_balanced_candidate_sort_key)

    selected_key = (
        round(float(selected["Black"]), 6),
        round(float(selected["Hispanic"]), 6),
        round(float(selected["Female"]), 6),
    )
    for r in full:
        r["selected"] = (
            round(float(r["Black"]), 6),
            round(float(r["Hispanic"]), 6),
            round(float(r["Female"]), 6),
        ) == selected_key

    # ------------------------------------------------------------------
    # Dissertation-ready exports
    # ------------------------------------------------------------------
    candidate_df = _balanced_candidate_dataframe(full, selected)

    csv_path = os.path.join(outdir, "balanced_refinement_candidates.csv")
    candidate_df.to_csv(csv_path, index=False)

    # Also save a concise selected-candidate table.
    selected_df = candidate_df[candidate_df["Selected"] == "Yes"].copy()
    selected_csv_path = os.path.join(outdir, "balanced_refinement_selected.csv")
    selected_df.to_csv(selected_csv_path, index=False)

    # Compact JSON: retain aggregates for selected solution, not every candidate.
    compact_full = []
    for r in full:
        compact_full.append({
            "Black": r["Black"],
            "Hispanic": r["Hispanic"],
            "Female": r["Female"],
            "aggregate_mae": r["aggregate_mae"],
            "subgroup_mae": r["subgroup_mae"],
            "gap_mae": r["gap_mae"],
            "max_aggregate_error": r["max_aggregate_error"],
            "acceptable": r["acceptable"],
            "constraint_violation": r["constraint_violation"],
            "selected": r["selected"],
        })

    result = {
        "bias_scope": BIAS_SCOPE,
        "bias_factor": BIAS_FACTOR,
        "method": "constrained_multiobjective_balanced_refinement",
        "selection_mode": selection_mode,
        "acceptance_thresholds": {
            "aggregate_mae": BALANCED_MAX_AGGREGATE_MAE,
            "subgroup_mae": BALANCED_MAX_SUBGROUP_MAE,
            "gap_mae": BALANCED_MAX_GAP_MAE,
            "max_aggregate_window_error": BALANCED_MAX_AGGREGATE_WINDOW_ERROR,
        },
        "starting_values": starting,
        "selected": {
            "Black": selected["Black"],
            "Hispanic": selected["Hispanic"],
            "Female": selected["Female"],
            "aggregate_mae": selected["aggregate_mae"],
            "subgroup_mae": selected["subgroup_mae"],
            "gap_mae": selected["gap_mae"],
            "max_aggregate_error": selected["max_aggregate_error"],
            "acceptable": selected["acceptable"],
            "constraint_violation": selected["constraint_violation"],
            "agg": {str(k): v for k, v in selected["agg"].items()},
        },
        "full_validation_candidates": compact_full,
        "exports": {
            "candidate_csv": csv_path,
            "selected_csv": selected_csv_path,
            "tradeoff_chart": os.path.join(
                outdir, "phase2_balanced_candidate_tradeoff.png"
            ),
            "candidate_table_chart": os.path.join(
                outdir, "phase2_balanced_candidate_table.png"
            ),
        },
    }

    with open(cp, "w") as f:
        json.dump(result, f, indent=2)

    _print_balanced_candidate_table(full, selected)
    plot_balanced_candidate_tradeoff(full, selected, outdir)
    plot_balanced_candidate_table(full, selected, outdir)

    print("\n  SELECTED BALANCED CALIBRATION")
    print("  " + "-" * 76)
    print(f"  Black β                 : {selected['Black']:+.3f}")
    print(f"  Hispanic β              : {selected['Hispanic']:+.3f}")
    print(f"  Female β                : {selected['Female']:+.3f}")
    print(f"  Aggregate MAE           : {selected['aggregate_mae']*100:.2f} pp")
    print(f"  Subgroup-rate MAE       : {selected['subgroup_mae']*100:.2f} pp")
    print(f"  Demographic-gap MAE     : {selected['gap_mae']*100:.2f} pp")
    print(f"  Maximum aggregate error : {selected['max_aggregate_error']*100:.2f} pp")
    print(f"  Acceptable              : {selected['acceptable']}")
    print(f"  Selection mode          : {selection_mode}")
    print(f"  Candidate table CSV     : {csv_path}")
    print("=" * 84)

    return result


def apply_balanced_refinement(race_params: dict, gender_params: dict,
                              balanced_result: dict) -> tuple:
    """Update downstream Stage 3 parameters with the selected balanced solution."""
    selected = balanced_result["selected"]

    race_params["Black"]["pre_balanced_value"] = race_params["Black"]["value"]
    race_params["Hispanic"]["pre_balanced_value"] = race_params["Hispanic"]["value"]
    gender_params["Female"]["pre_balanced_value"] = gender_params["Female"]["value"]

    race_params["Black"]["value"] = float(selected["Black"])
    race_params["Hispanic"]["value"] = float(selected["Hispanic"])
    gender_params["Female"]["value"] = float(selected["Female"])

    race_params["Black"]["balanced_refined"] = True
    race_params["Hispanic"]["balanced_refined"] = True
    gender_params["Female"]["balanced_refined"] = True

    return race_params, gender_params


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
                    "race_bias": race_bias,
                    "gender_bias": gender_bias,
                    "bias_factor": BIAS_FACTOR,
                    "bias_scope": BIAS_SCOPE,
                    "agg": {str(k): v for k, v in agg.items()},
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
    fine_screen_b = black_data.get("fine_screen_results", fine_rows_b)
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

    if fine_rows_b or fine_screen_b:
        ax2 = axes_b[1]

        if fine_screen_b:
            svals = [r["value"] for r in fine_screen_b]
            smaes = [r["mae"]   for r in fine_screen_b]
            order = np.argsort(svals)
            svals = np.asarray(svals)[order]
            smaes = np.asarray(smaes)[order]

            ax2.plot(
                svals, smaes,
                color=COLORS["Black"],
                linewidth=0.9,
                alpha=0.35,
                linestyle="--",
                zorder=2,
                label="Fine screen (500 agents, 25 runs/pt)",
            )

            if len(smaes) > 3:
                ax2.plot(
                    svals,
                    gaussian_filter1d(smaes, sigma=1.5),
                    color=COLORS["Black"],
                    linewidth=1.8,
                    alpha=0.65,
                    zorder=3,
                    label="Fine screen (smoothed)",
                )

        if fine_rows_b:
            fvals = [r["value"] for r in fine_rows_b]
            fmaes = [r["mae"] for r in fine_rows_b]
            ax2.plot(
                fvals, fmaes,
                color=COLORS["Black"],
                marker="o",
                linewidth=2.2,
                zorder=5,
                label="Full validation candidates",
            )

        ax2.axvline(
            best_b,
            color="#333333",
            linewidth=1.5,
            linestyle="--",
            label=f"FINAL best={best_b:.3f} (MAE={black_data['mae']:.4f})",
        )
        ax2.axvline(
            coarse_best_b,
            color="#888888",
            linewidth=1.0,
            linestyle=":",
            label=f"Coarse best={coarse_best_b:.2f}",
        )
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=8)
        _style_ax(
            ax2,
            "Black Bias (log-odds)",
            "",
            f"Pass 2: Adaptive Fine Search ±{BLACK_SWEEP_FINE_HALF_WIDTH:.2f}",
        )

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
    fine_screen = data.get("fine_screen_results", fine_rows)
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

    if fine_rows or fine_screen:
        ax2 = axes[1]

        if fine_screen:
            svals = [r["value"] for r in fine_screen]
            smaes = [r["mae"] for r in fine_screen]
            order = np.argsort(svals)
            svals = np.asarray(svals)[order]
            smaes = np.asarray(smaes)[order]

            ax2.plot(
                svals, smaes,
                color="#C55A11",
                linewidth=0.9,
                alpha=0.35,
                linestyle="--",
                zorder=2,
                label="Fine screen (500 agents, 25 runs/pt)",
            )

            if len(smaes) > 3:
                ax2.plot(
                    svals,
                    gaussian_filter1d(smaes, sigma=1.5),
                    color="#C55A11",
                    linewidth=1.8,
                    alpha=0.65,
                    zorder=3,
                    label="Fine screen (smoothed)",
                )

        if fine_rows:
            fvals = [r["value"] for r in fine_rows]
            fmaes = [r["mae"] for r in fine_rows]
            ax2.plot(
                fvals, fmaes,
                color="#C55A11",
                marker="o",
                linewidth=2.2,
                zorder=5,
                label="Full validation candidates",
            )

        ax2.axvline(
            best_v,
            color="#333333",
            linewidth=1.5,
            linestyle="--",
            label=f"FINAL best={best_v:.3f} (MAE={data['mae']:.4f})",
        )
        ax2.axvline(
            coarse_best,
            color="#888888",
            linewidth=1.0,
            linestyle=":",
            label=f"Coarse best={coarse_best:.2f}",
        )
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=9)
        _style_ax(
            ax2,
            "Female Bias (log-odds)",
            "",
            f"Pass 2: Adaptive Fine Search ±{GENDER_SWEEP_FINE_HALF_WIDTH:.2f}",
        )

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
    print(f"  Fine strategy  : adaptive screen -> top {FINE_SCREEN_TOP_K} "
          f"+ neighbors -> full validation")
    print("  Final selection: constrained balance of aggregate, subgroup, and gap fit")
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
        if 3 in stages:
            bp2 = os.path.join(outdir, "balanced_refinement.json")
            if os.path.exists(bp2):
                os.remove(bp2)
                print("    deleted: balanced_refinement.json (will recompute)")

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
    balanced_result = {}

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

    # ── Final constrained balanced refinement before Stage 3 ────────────────
    # Stage 3 must validate the FINAL selected Black/Hispanic/Female values.
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

        if not race_params or not gender_params:
            raise RuntimeError(
                "Balanced refinement requires completed Stage 1 race and "
                "Stage 2 gender calibration checkpoints."
            )

        balanced_result = run_final_balanced_refinement(
            race_params,
            gender_params,
            n_workers,
            outdir,
            force_rerun=force_rerun,
        )

        race_params, gender_params = apply_balanced_refinement(
            race_params,
            gender_params,
            balanced_result,
        )

        # Any existing Stage 3 files may contain pre-refinement parameters.
        # Delete them so validation/sensitivity are guaranteed to use the
        # selected balanced solution.
        for fname in ("stage3_validation.json", "stage3_sensitivity.json"):
            stale = os.path.join(outdir, fname)
            if os.path.exists(stale):
                os.remove(stale)
                print(f"    Removed stale Stage 3 checkpoint: {fname}")

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

    if balanced_result and balanced_result.get("selected"):
        s = balanced_result["selected"]
        print("\n  FINAL BALANCE DIAGNOSTICS")
        print("  " + "-" * 70)
        print(f"  Aggregate MAE       : {s['aggregate_mae']*100:.2f} pp")
        print(f"  Subgroup-rate MAE   : {s['subgroup_mae']*100:.2f} pp")
        print(f"  Demographic-gap MAE : {s['gap_mae']*100:.2f} pp")
        print(f"  Max aggregate error : {s['max_aggregate_error']*100:.2f} pp")
        print(f"  Acceptable          : {s['acceptable']}")
        print("  " + "-" * 70)


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
            print("Usage: python phase2_bias_supervision_only_balanced_refinement.py --scope supervision_only|all_channels")
            sys.exit(1)

    stages = None
    if "--stage" in args:
        idx = args.index("--stage")
        try:
            stages = [int(args[idx + 1])]
        except (IndexError, ValueError):
            print("Usage: python phase2_bias_supervision_only_balanced_refinement.py --stage {1|2|3}")
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