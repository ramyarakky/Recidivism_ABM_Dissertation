"""
phase2_bias_supervision_only_sequential_optimized.py  —  Phase 2: Intersectional Bias Simulation
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
  Stage 1 Black fine   : adaptive screen + small full-validation subset
  Stage 1 Hispanic     : ~15 min  (21 pts × 25 runs × 500 agents)
  Stage 2 coarse       : ~25 min  (36 pts × 25 runs × 500 agents)
  Stage 2 fine         : adaptive screen + small full-validation subset
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
    "output_directory":      "phase2_bias_supervision_only_sequential_optimized",
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
BASELINE_AGG: dict      = {}   # raw zero-bias subgroup rates for before/after charts
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
# Preserve the sequential calibration objective, but avoid running every fine
# grid point at full fidelity.
#
# Pass 2a: all fine-grid points are screened cheaply
#          (500 agents × 5 seeds × 5 reps = 25 runs/point)
# Pass 2b: only the best screened points + immediate neighbors are rerun
#          at full fidelity (1000 agents × 10 seeds × 10 reps = 100 runs/point)
FINE_SCREEN_TOP_K = 5
FINE_SCREEN_NEIGHBOR_RADIUS = 1

# ── Dissertation figure settings ─────────────────────────────────────────────
DISS_DPI            = 300
DISS_TITLE_SIZE     = 15
DISS_AXIS_SIZE      = 12
DISS_TICK_SIZE      = 10
DISS_LEGEND_SIZE    = 10

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
    global BASELINE_OFFSETS, ADJUSTED_TARGETS, BASELINE_AGG, _BASELINE_MEASURED

    cp = os.path.join(outdir, "baseline_offsets.json")
    if os.path.exists(cp):
        print("  > Baseline offsets — loading checkpoint")
        with open(cp) as f:
            saved = json.load(f)

        # New checkpoint format stores the raw zero-bias subgroup rates used
        # by the dissertation before/after calibration figures. Older
        # checkpoints did not preserve raw Female baseline values because the
        # Female target override can replace the stored offset. Re-measure once
        # rather than silently reconstructing an inaccurate baseline.
        if "baseline_agg" not in saved:
            print("  NOTE: legacy baseline checkpoint lacks raw baseline rates.")
            print("        Re-measuring zero-bias baseline once for accurate charts.")
            os.remove(cp)
        else:
            BASELINE_OFFSETS = {int(k): v for k, v in saved["offsets"].items()}
            ADJUSTED_TARGETS = {int(k): v for k, v in saved["adjusted"].items()}
            BASELINE_AGG = {int(k): v for k, v in saved["baseline_agg"].items()}
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
    raw_offsets = {}
    adjusted = {}

    for win in WINDOWS:
        offsets[win] = {}
        raw_offsets[win] = {}
        adjusted[win] = {}

        for grp in groups:
            sim = agg[win].get(grp, {}).get("mean")
            bjs = BJS_SUBGROUP[win].get(grp)

            if sim is not None and bjs is not None:
                raw_offset = round(float(sim) - float(bjs), 4)
                raw_offsets[win][grp] = raw_offset

                # Preserve existing dissertation calibration behavior:
                # Female can use the direct BJS target when override=0.0.
                offsets[win][grp] = (
                    FEMALE_OFFSET_OVERRIDE
                    if grp == "Female" and FEMALE_OFFSET_OVERRIDE is not None
                    else raw_offset
                )
                adjusted[win][grp] = round(
                    float(bjs) - float(offsets[win][grp]), 4
                )
            else:
                raw_offsets[win][grp] = 0.0
                offsets[win][grp] = 0.0
                adjusted[win][grp] = bjs or 0.0

    BASELINE_OFFSETS = offsets
    ADJUSTED_TARGETS = adjusted
    BASELINE_AGG = agg
    _BASELINE_MEASURED = True

    with open(cp, "w") as f:
        json.dump({
            "offsets":         {str(k): v for k, v in offsets.items()},
            "raw_offsets":     {str(k): v for k, v in raw_offsets.items()},
            "adjusted":        {str(k): v for k, v in adjusted.items()},
            "baseline_agg":    {str(k): v for k, v in agg.items()},
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
    Select a small, robust full-validation set from the cheap fine screen.

    The best top_k screened values are retained together with immediate
    neighboring grid points. This protects against stochastic rank changes
    while avoiding full-fidelity evaluation of the entire fine grid.
    """
    if not screen_results:
        return []

    value_to_idx = {
        round(float(v), 8): i for i, v in enumerate(fine_vals)
    }
    selected = set()

    for row in screen_results[:top_k]:
        val = float(row["value"])
        idx = value_to_idx.get(round(val, 8))

        if idx is None:
            idx = int(
                np.argmin(
                    np.abs(np.asarray(fine_vals, dtype=float) - val)
                )
            )

        lo = max(0, idx - neighbor_radius)
        hi = min(len(fine_vals), idx + neighbor_radius + 1)

        for j in range(lo, hi):
            selected.add(float(fine_vals[j]))

    return sorted(selected)


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
            # Pass 2a — cheap full-grid screen
            print(f"\n  Pass 2a (fine screen): {len(fine_vals)} pts  "
                  f"[{AGENTS_COARSE} agents, "
                  f"{len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")

            fine_screen_results = _sweep_race_group(
                grp, fine_vals, n_workers,
                "Stage1 Black fine-screen",
                n_reps_override=N_REPS_COARSE,
                is_coarse=True,
            )

            validation_vals = _select_fine_candidates(
                fine_screen_results, fine_vals
            )
            validation_vals = sorted(
                set(validation_vals + [float(coarse_best["value"])])
            )

            # Pass 2b — full-fidelity confirmation of only promising points
            print(f"  Pass 2b (full validation): {len(validation_vals)} candidates  "
                  f"[{CONFIG['initial_agents']} agents, "
                  f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")

            fine_results = _sweep_race_group(
                grp, validation_vals, n_workers,
                "Stage1 Black fine-validate",
                n_reps_override=N_REPS_FINE,
                is_coarse=False,
            )

            if not fine_results:
                raise RuntimeError("Black fine validation produced no results.")

            fine_best = fine_results[0]

            print(f"  OK  Fine best = {fine_best['value']:.3f}  "
                  f"(MAE={fine_best['mae']:.4f})")

            best_params[grp] = {
                "value": fine_best["value"],
                "mae": fine_best["mae"],
                "agg": {str(k): v for k, v in fine_best["agg"].items()},
                "coarse_results": [
                    {"value": r["value"], "mae": r["mae"]}
                    for r in coarse_results
                ],
                "fine_screen_results": [
                    {"value": r["value"], "mae": r["mae"]}
                    for r in fine_screen_results
                ],
                "fine_results": [
                    {"value": r["value"], "mae": r["mae"]}
                    for r in fine_results
                ],
                "fine_validation_values": validation_vals,
                "all_results": [
                    {"value": r["value"], "mae": r["mae"]}
                    for r in coarse_results
                ],
                "adjusted_target": {
                    str(w): ADJUSTED_TARGETS[w][grp] for w in WINDOWS
                },
                "baseline_offset": {
                    str(w): BASELINE_OFFSETS[w][grp] for w in WINDOWS
                },
                "coarse_best": coarse_best["value"],
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
    # Pass 2a — cheap fine-grid screen
    print(f"\n  Pass 2a (fine screen): {len(fine_vals)} pts  "
          f"[{AGENTS_COARSE} agents, "
          f"{len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")

    fine_screen_results = _sweep_female(
        race_bias, fine_vals, n_workers,
        "Stage2 Female fine-screen",
        n_reps_override=N_REPS_COARSE,
        is_coarse=True,
    )

    validation_vals = _select_fine_candidates(
        fine_screen_results, fine_vals
    )
    validation_vals = sorted(
        set(validation_vals + [float(coarse_best["value"])])
    )

    # Pass 2b — full-fidelity confirmation
    print(f"  Pass 2b (full validation): {len(validation_vals)} candidates  "
          f"[{CONFIG['initial_agents']} agents, "
          f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")

    fine_results = _sweep_female(
        race_bias, validation_vals, n_workers,
        "Stage2 Female fine-validate",
        n_reps_override=N_REPS_FINE,
        is_coarse=False,
    )

    if not fine_results:
        raise RuntimeError("Female fine validation produced no results.")

    fine_best = fine_results[0]

    print(f"  OK  Fine best = {fine_best['value']:.3f}  "
          f"(MAE={fine_best['mae']:.4f})")

    best_params = {"Female": {
        "value": fine_best["value"],
        "mae": fine_best["mae"],
        "agg": {str(k): v for k, v in fine_best["agg"].items()},
        "coarse_results": [
            {"value": r["value"], "mae": r["mae"]}
            for r in coarse_results
        ],
        "fine_screen_results": [
            {"value": r["value"], "mae": r["mae"]}
            for r in fine_screen_results
        ],
        "fine_results": [
            {"value": r["value"], "mae": r["mae"]}
            for r in fine_results
        ],
        "fine_validation_values": validation_vals,
        "all_results": [
            {"value": r["value"], "mae": r["mae"]}
            for r in coarse_results
        ],
        "adjusted_target": {
            str(w): ADJUSTED_TARGETS[w]["Female"] for w in WINDOWS
        },
        "baseline_offset": {
            str(w): BASELINE_OFFSETS[w]["Female"] for w in WINDOWS
        },
        "coarse_best": coarse_best["value"],
        "coarse_best_mae": coarse_best["mae"],
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
            svals = np.array([r["value"] for r in fine_screen_b], dtype=float)
            smaes = np.array([r["mae"] for r in fine_screen_b], dtype=float)
            order = np.argsort(svals)
            svals, smaes = svals[order], smaes[order]

            ax2.plot(
                svals, smaes,
                color=COLORS["Black"], linewidth=1.1,
                alpha=0.45, linestyle="--", zorder=2,
                label="Fine screen (coarse fidelity)"
            )
            if len(smaes) > 3:
                ax2.plot(
                    svals, gaussian_filter1d(smaes, sigma=1.3),
                    color=COLORS["Black"], linewidth=2.0,
                    alpha=0.70, zorder=3,
                    label="Fine screen (smoothed)"
                )

        if fine_rows_b:
            fvals = [r["value"] for r in fine_rows_b]
            fmaes = [r["mae"] for r in fine_rows_b]
            ax2.plot(
                fvals, fmaes,
                color=COLORS["Black"], marker="o",
                linewidth=2.2, zorder=5,
                label="Full-validation candidates"
            )

        ax2.axvline(
            best_b, color="#333333", linewidth=1.6, linestyle="--",
            label=f"Final best={best_b:.3f} (MAE={black_data['mae']:.4f})"
        )
        ax2.axvline(
            coarse_best_b, color="#888888", linewidth=1.0, linestyle=":",
            label=f"Coarse best={coarse_best_b:.2f}"
        )
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=8)
        _style_ax(
            ax2, "Black Bias (log-odds)", "",
            f"Pass 2: Adaptive Fine Search ±{BLACK_SWEEP_FINE_HALF_WIDTH:.2f}"
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
            svals = np.array([r["value"] for r in fine_screen], dtype=float)
            smaes = np.array([r["mae"] for r in fine_screen], dtype=float)
            order = np.argsort(svals)
            svals, smaes = svals[order], smaes[order]

            ax2.plot(
                svals, smaes,
                color="#C55A11", linewidth=1.1,
                alpha=0.45, linestyle="--", zorder=2,
                label="Fine screen (coarse fidelity)"
            )
            if len(smaes) > 3:
                ax2.plot(
                    svals, gaussian_filter1d(smaes, sigma=1.3),
                    color="#C55A11", linewidth=2.0,
                    alpha=0.70, zorder=3,
                    label="Fine screen (smoothed)"
                )

        if fine_rows:
            fvals = [r["value"] for r in fine_rows]
            fmaes = [r["mae"] for r in fine_rows]
            ax2.plot(
                fvals, fmaes,
                color="#C55A11", marker="o",
                linewidth=2.2, zorder=5,
                label="Full-validation candidates"
            )

        ax2.axvline(
            best_v, color="#333333", linewidth=1.6, linestyle="--",
            label=f"Final best={best_v:.3f} (MAE={data['mae']:.4f})"
        )
        ax2.axvline(
            coarse_best, color="#888888", linewidth=1.0, linestyle=":",
            label=f"Coarse best={coarse_best:.2f}"
        )
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=9)
        _style_ax(
            ax2, "Female Bias (log-odds)", "",
            f"Pass 2: Adaptive Fine Search ±{GENDER_SWEEP_FINE_HALF_WIDTH:.2f}"
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
# DISSERTATION-READY VALIDATION TABLES + CHARTS
# =============================================================================

def _bjs_gap(win: int, group: str) -> float:
    """Return the BJS group-reference disparity for the requested window."""
    if group == "Black":
        return BJS_SUBGROUP[win]["Black"] - BJS_SUBGROUP[win]["White"]
    if group == "Hispanic":
        return BJS_SUBGROUP[win]["Hispanic"] - BJS_SUBGROUP[win]["White"]
    if group == "Female":
        return BJS_SUBGROUP[win]["Female"] - BJS_SUBGROUP[win]["Male"]
    raise ValueError(f"Unsupported disparity group: {group}")


def build_dissertation_validation_tables(validation: dict,
                                         race_params: dict,
                                         gender_params: dict,
                                         outdir: str) -> dict:
    """
    Export transparent dissertation tables for:
      1. overall BJS-vs-ABM anchors,
      2. subgroup BJS-vs-ABM rates,
      3. demographic BJS-vs-ABM gaps,
      4. calibrated beta / OR / error summary.

    Calibration remains sequential. These tables are validation diagnostics;
    they do NOT change the selected coefficients.
    """
    agg = {int(k): v for k, v in validation["agg"].items()}

    # ---------------------- overall ----------------------
    overall_rows = []
    for win in WINDOWS:
        sim = agg[win].get("Overall", {}).get("mean")
        bjs = CONFIG["bjs_targets"][win]
        overall_rows.append({
            "Window": f"{win}-Year",
            "BJS Rate (%)": bjs * 100,
            "ABM Rate (%)": (sim * 100) if sim is not None else np.nan,
            "ABM - BJS (pp)": ((sim - bjs) * 100) if sim is not None else np.nan,
            "Absolute Error (pp)": (abs(sim - bjs) * 100) if sim is not None else np.nan,
        })
    overall_df = pd.DataFrame(overall_rows)

    # ---------------------- subgroup ---------------------
    subgroup_rows = []
    groups = ["White", "Black", "Hispanic", "Male", "Female"]
    for win in WINDOWS:
        for group in groups:
            sim = agg[win].get(group, {}).get("mean")
            ci = agg[win].get(group, {}).get("ci95")
            bjs = BJS_SUBGROUP[win][group]
            subgroup_rows.append({
                "Window": f"{win}-Year",
                "Group": group,
                "BJS Rate (%)": bjs * 100,
                "ABM Rate (%)": (sim * 100) if sim is not None else np.nan,
                "95% CI Half-Width (pp)": (ci * 100) if ci is not None else np.nan,
                "ABM - BJS (pp)": ((sim - bjs) * 100) if sim is not None else np.nan,
                "Absolute Error (pp)": (abs(sim - bjs) * 100) if sim is not None else np.nan,
            })
    subgroup_df = pd.DataFrame(subgroup_rows)

    # ---------------------- disparities ------------------
    gap_rows = []
    refs = {
        "Black": "White",
        "Hispanic": "White",
        "Female": "Male",
    }
    for win in WINDOWS:
        for group, ref in refs.items():
            g = agg[win].get(group, {}).get("mean")
            r = agg[win].get(ref, {}).get("mean")
            if g is None or r is None:
                continue
            sim_gap = g - r
            bjs_gap = _bjs_gap(win, group)
            gap_rows.append({
                "Window": f"{win}-Year",
                "Comparison": f"{group} - {ref}",
                "BJS Gap (pp)": bjs_gap * 100,
                "ABM Gap (pp)": sim_gap * 100,
                "Gap Error (pp)": (sim_gap - bjs_gap) * 100,
                "Absolute Gap Error (pp)": abs(sim_gap - bjs_gap) * 100,
            })
    gap_df = pd.DataFrame(gap_rows)

    # ---------------------- parameter summary ------------
    param_rows = []
    for group, container in (
        ("Black", race_params.get("Black", {})),
        ("Hispanic", race_params.get("Hispanic", {})),
        ("Female", gender_params.get("Female", {})),
    ):
        beta = container.get("value")
        if beta is None:
            continue

        # direct empirical subgroup-rate MAE from FINAL Stage 3 validation
        group_abs_errors = []
        gap_abs_errors = []
        for win in WINDOWS:
            sim = agg[win].get(group, {}).get("mean")
            if sim is not None:
                group_abs_errors.append(
                    abs(sim - BJS_SUBGROUP[win][group])
                )

            ref = "Male" if group == "Female" else "White"
            g = agg[win].get(group, {}).get("mean")
            r = agg[win].get(ref, {}).get("mean")
            if g is not None and r is not None:
                gap_abs_errors.append(
                    abs((g - r) - _bjs_gap(win, group))
                )

        param_rows.append({
            "Group": group,
            "Bias Beta": beta,
            "Odds Ratio exp(Beta)": math.exp(beta),
            "Sequential Calibration MAE (pp)": container.get("mae", np.nan) * 100,
            "Final BJS Subgroup Rate MAE (pp)": (
                np.mean(group_abs_errors) * 100
                if group_abs_errors else np.nan
            ),
            "Final BJS Gap MAE (pp)": (
                np.mean(gap_abs_errors) * 100
                if gap_abs_errors else np.nan
            ),
        })

    param_rows.extend([
        {
            "Group": "White",
            "Bias Beta": 0.0,
            "Odds Ratio exp(Beta)": 1.0,
            "Sequential Calibration MAE (pp)": np.nan,
            "Final BJS Subgroup Rate MAE (pp)": np.nan,
            "Final BJS Gap MAE (pp)": np.nan,
        },
        {
            "Group": "Male",
            "Bias Beta": 0.0,
            "Odds Ratio exp(Beta)": 1.0,
            "Sequential Calibration MAE (pp)": np.nan,
            "Final BJS Subgroup Rate MAE (pp)": np.nan,
            "Final BJS Gap MAE (pp)": np.nan,
        },
    ])
    params_df = pd.DataFrame(param_rows)

    paths = {
        "overall_csv": os.path.join(outdir, "dissertation_overall_fit.csv"),
        "subgroup_csv": os.path.join(outdir, "dissertation_subgroup_fit.csv"),
        "gap_csv": os.path.join(outdir, "dissertation_gap_fit.csv"),
        "parameter_csv": os.path.join(outdir, "dissertation_calibration_summary.csv"),
    }

    overall_df.to_csv(paths["overall_csv"], index=False)
    subgroup_df.to_csv(paths["subgroup_csv"], index=False)
    gap_df.to_csv(paths["gap_csv"], index=False)
    params_df.to_csv(paths["parameter_csv"], index=False)

    return {
        "overall": overall_df,
        "subgroup": subgroup_df,
        "gap": gap_df,
        "parameters": params_df,
        "paths": paths,
    }


def plot_calibration_effect(validation: dict, outdir: str) -> str:
    """
    Show whether demographic calibration actually bridges BJS subgroup rates.

    Each panel compares:
      Zero-bias ABM -> Calibrated ABM -> BJS
    at 3/6/9 years for Black, Hispanic, and Female.
    """
    if not BASELINE_AGG:
        print("    Calibration effect chart skipped — baseline aggregate unavailable.")
        return None

    agg = {int(k): v for k, v in validation["agg"].items()}
    groups = ["Black", "Hispanic", "Female"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, group in zip(axes, groups):
        baseline = [
            BASELINE_AGG[w].get(group, {}).get("mean", np.nan) * 100
            for w in WINDOWS
        ]
        calibrated = [
            agg[w].get(group, {}).get("mean", np.nan) * 100
            for w in WINDOWS
        ]
        bjs = [
            BJS_SUBGROUP[w][group] * 100
            for w in WINDOWS
        ]

        ax.plot(
            WINDOWS, bjs, marker="D", linewidth=2.6,
            label="BJS target", zorder=5
        )
        ax.plot(
            WINDOWS, baseline, marker="o", linewidth=1.8,
            linestyle="--", label="Zero-bias ABM", zorder=3
        )
        ax.plot(
            WINDOWS, calibrated, marker="o", linewidth=2.6,
            label="Calibrated ABM", zorder=4
        )

        for w, y, target in zip(WINDOWS, calibrated, bjs):
            if np.isfinite(y):
                ax.annotate(
                    f"Δ {y-target:+.1f} pp",
                    (w, y),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=9,
                )

        ax.set_xticks(WINDOWS)
        ax.set_xticklabels([f"{w}yr" for w in WINDOWS])
        ax.set_title(group, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=DISS_TICK_SIZE)

    axes[0].set_ylabel(
        "Cumulative Rearrest Rate (%)", fontsize=DISS_AXIS_SIZE
    )
    axes[0].legend(fontsize=DISS_LEGEND_SIZE)

    fig.suptitle(
        "Demographic Bias Calibration Effect: Zero-Bias vs Calibrated ABM vs BJS",
        fontsize=DISS_TITLE_SIZE,
        fontweight="bold",
        color="#1A3D5C",
        y=1.01,
    )

    fig.text(
        0.5, 0.015,
        "Calibration objective remains sequential subgroup-rate fit; this figure is an empirical validation of the resulting shift.",
        ha="center", fontsize=9.5
    )

    plt.tight_layout(rect=[0, 0.045, 1, 0.97])
    path = os.path.join(outdir, "phase2_calibration_effect_before_after.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Chart: Calibration effect -> {path}")
    return path


def plot_overall_empirical_fit(validation: dict, outdir: str) -> str:
    """Overall 3/6/9-year BJS anchor validation."""
    agg = {int(k): v for k, v in validation["agg"].items()}

    bjs = [CONFIG["bjs_targets"][w] * 100 for w in WINDOWS]
    sim = [agg[w]["Overall"]["mean"] * 100 for w in WINDOWS]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    fig.patch.set_facecolor("#FAFAFA")

    ax.plot(WINDOWS, bjs, marker="D", linewidth=2.7, label="BJS target")
    ax.plot(WINDOWS, sim, marker="o", linewidth=2.7, label="Calibrated ABM")

    for w, y, t in zip(WINDOWS, sim, bjs):
        ax.annotate(
            f"{y:.1f}%\nΔ {y-t:+.1f} pp",
            (w, y),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9.5
        )

    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([f"{w}-Year" for w in WINDOWS])
    ax.set_ylabel("Cumulative Rearrest Rate (%)", fontsize=DISS_AXIS_SIZE)
    ax.set_xlabel("Follow-up Window", fontsize=DISS_AXIS_SIZE)
    ax.set_title(
        "Aggregate BJS Validation After Demographic Calibration",
        fontsize=DISS_TITLE_SIZE, fontweight="bold", color="#1A3D5C"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=DISS_TICK_SIZE)
    ax.legend(fontsize=DISS_LEGEND_SIZE)

    mae = np.mean([
        abs(agg[w]["Overall"]["mean"] - CONFIG["bjs_targets"][w])
        for w in WINDOWS
    ]) * 100

    ax.text(
        0.02, 0.04,
        f"Aggregate anchor MAE = {mae:.2f} pp",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB")
    )

    plt.tight_layout()
    path = os.path.join(outdir, "phase2_overall_empirical_fit.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Chart: Overall empirical fit -> {path}")
    return path


def plot_subgroup_empirical_fit(validation: dict, outdir: str) -> str:
    """BJS vs calibrated ABM subgroup rates at every anchor window."""
    agg = {int(k): v for k, v in validation["agg"].items()}
    groups = ["White", "Black", "Hispanic", "Male", "Female"]
    x = np.arange(len(groups))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, win in zip(axes, WINDOWS):
        bjs = [BJS_SUBGROUP[win][g] * 100 for g in groups]
        sim = [agg[win][g]["mean"] * 100 for g in groups]

        ax.bar(x - width / 2, bjs, width, label="BJS target", alpha=0.70)
        ax.bar(x + width / 2, sim, width, label="Calibrated ABM", alpha=0.90)

        for i, (target, value) in enumerate(zip(bjs, sim)):
            ax.text(
                i + width / 2, value + 0.8,
                f"{value-target:+.1f}",
                ha="center", va="bottom",
                fontsize=8.5
            )

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.set_title(f"{win}-Year", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("Cumulative Rearrest Rate (%)", fontsize=DISS_AXIS_SIZE)
    axes[0].legend(fontsize=DISS_LEGEND_SIZE)

    fig.suptitle(
        "Absolute Subgroup Fit to BJS After Sequential Bias Calibration",
        fontsize=DISS_TITLE_SIZE,
        fontweight="bold",
        color="#1A3D5C",
        y=1.01
    )
    fig.text(
        0.5, 0.01,
        "Labels above ABM bars show simulation − BJS in percentage points.",
        ha="center", fontsize=9.5
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = os.path.join(outdir, "phase2_subgroup_empirical_fit.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Chart: Subgroup empirical fit -> {path}")
    return path


def plot_demographic_gap_fit(validation: dict, outdir: str) -> str:
    """Direct BJS-vs-ABM disparity validation."""
    agg = {int(k): v for k, v in validation["agg"].items()}
    comparisons = [
        ("Black", "White", "Black − White"),
        ("Hispanic", "White", "Hispanic − White"),
        ("Female", "Male", "Female − Male"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), sharey=True)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (group, ref, title) in zip(axes, comparisons):
        bjs = [_bjs_gap(w, group) * 100 for w in WINDOWS]
        sim = [
            (
                agg[w][group]["mean"]
                - agg[w][ref]["mean"]
            ) * 100
            for w in WINDOWS
        ]

        ax.plot(WINDOWS, bjs, marker="D", linewidth=2.5, label="BJS gap")
        ax.plot(WINDOWS, sim, marker="o", linewidth=2.5, label="ABM gap")
        ax.axhline(0, linewidth=0.8, color="#666666")

        gap_mae = np.mean(
            [abs(a - b) for a, b in zip(sim, bjs)]
        )

        for w, value, target in zip(WINDOWS, sim, bjs):
            ax.annotate(
                f"|err| {abs(value-target):.1f}",
                (w, value),
                textcoords="offset points",
                xytext=(0, 9 if value >= 0 else -15),
                ha="center", fontsize=8.5
            )

        ax.set_xticks(WINDOWS)
        ax.set_xticklabels([f"{w}yr" for w in WINDOWS])
        ax.set_title(
            f"{title}\nGap MAE = {gap_mae:.2f} pp",
            fontsize=12.5, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("Group − Reference Gap (percentage points)",
                       fontsize=DISS_AXIS_SIZE)
    axes[0].legend(fontsize=DISS_LEGEND_SIZE)

    fig.suptitle(
        "Demographic Disparity Validation Against BJS",
        fontsize=DISS_TITLE_SIZE,
        fontweight="bold",
        color="#1A3D5C",
        y=1.02
    )
    fig.text(
        0.5, 0.012,
        "Gap MAE is a validation diagnostic; sequential parameter selection is based on subgroup-rate MAE.",
        ha="center", fontsize=9.5
    )

    plt.tight_layout(rect=[0, 0.045, 1, 0.97])
    path = os.path.join(outdir, "phase2_demographic_gap_fit.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Chart: Demographic gap fit -> {path}")
    return path


def plot_calibration_summary_table(tables: dict, outdir: str) -> str:
    """Rendered dissertation-ready summary of final calibrated parameters."""
    df = tables["parameters"].copy()

    display = df.copy()
    display["Bias Beta"] = display["Bias Beta"].map(lambda x: f"{x:+.3f}")
    display["Odds Ratio exp(Beta)"] = display["Odds Ratio exp(Beta)"].map(
        lambda x: f"{x:.3f}×"
    )

    for col in [
        "Sequential Calibration MAE (pp)",
        "Final BJS Subgroup Rate MAE (pp)",
        "Final BJS Gap MAE (pp)",
    ]:
        display[col] = display[col].map(
            lambda x: "—" if pd.isna(x) else f"{x:.2f}"
        )

    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    fig.patch.set_facecolor("#FAFAFA")
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=[
            "Group", "Bias β", "Odds Ratio",
            "Sequential\nCalibration MAE (pp)",
            "Final Subgroup\nRate MAE (pp)",
            "Final Gap\nMAE (pp)",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C8C8C8")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E9EEF4")

    ax.set_title(
        "Final Sequential Bias Calibration — Parameter and Validation Summary",
        fontsize=DISS_TITLE_SIZE,
        fontweight="bold",
        color="#1A3D5C",
        pad=14
    )

    fig.text(
        0.5, 0.03,
        "Sequential calibration MAE is the optimization objective; final subgroup-rate and gap MAEs are independent Stage 3 validation diagnostics.",
        ha="center", fontsize=9.5
    )

    plt.tight_layout(rect=[0.01, 0.065, 0.99, 0.94])
    path = os.path.join(outdir, "phase2_calibration_summary_table.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Chart: Calibration summary table -> {path}")
    return path


def generate_dissertation_outputs(validation: dict,
                                   race_params: dict,
                                   gender_params: dict,
                                   outdir: str) -> dict:
    """Build all dissertation-ready validation tables, CSVs, and figures."""
    print("\n  Generating dissertation-ready empirical validation outputs...")

    tables = build_dissertation_validation_tables(
        validation, race_params, gender_params, outdir
    )

    plot_calibration_effect(validation, outdir)
    plot_overall_empirical_fit(validation, outdir)
    plot_subgroup_empirical_fit(validation, outdir)
    plot_demographic_gap_fit(validation, outdir)
    plot_calibration_summary_table(tables, outdir)

    print("    CSV: dissertation_overall_fit.csv")
    print("    CSV: dissertation_subgroup_fit.csv")
    print("    CSV: dissertation_gap_fit.csv")
    print("    CSV: dissertation_calibration_summary.csv")

    return tables


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
    print(f"  Fine strategy  : adaptive screen -> top {FINE_SCREEN_TOP_K} + neighbors -> full validation")
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

        # Additional dissertation-facing empirical validation outputs.
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
        "zero_bias_baseline_agg": BASELINE_AGG,
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
    print(
        f"  {'Group':<12} {'Bias (log-odds)':>16} {'Odds Ratio':>12} "
        f"{'Cal MAE':>9} {'Final Rate':>11} {'Final Gap':>10}"
    )
    print(f"  {'-'*76}")

    validation_agg = {}
    if stage3_results and stage3_results.get("validation"):
        validation_agg = {
            int(k): v
            for k, v in stage3_results["validation"]["agg"].items()
        }

    for grp, b, m in rows:
        if b is None:
            continue

        rate_mae = None
        gap_mae = None

        if validation_agg:
            rate_errors = []
            gap_errors = []
            ref = "Male" if grp == "Female" else "White"

            for win in WINDOWS:
                sim = validation_agg[win].get(grp, {}).get("mean")
                if sim is not None:
                    rate_errors.append(
                        abs(sim - BJS_SUBGROUP[win][grp])
                    )

                g = validation_agg[win].get(grp, {}).get("mean")
                r = validation_agg[win].get(ref, {}).get("mean")
                if g is not None and r is not None:
                    gap_errors.append(
                        abs((g - r) - _bjs_gap(win, grp))
                    )

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

    print(
        f"  {'White':<12} {'Reference (0.00)':>16} "
        f"{'1.000x':>12} {'-':>9} {'-':>11} {'-':>10}"
    )
    print(
        f"  {'Male':<12} {'Reference (0.00)':>16} "
        f"{'1.000x':>12} {'-':>9} {'-':>11} {'-':>10}"
    )

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
        # Rebuild dissertation figures from existing checkpoints when possible.
        cp_val = os.path.join(outdir, "stage3_validation.json")
        if os.path.exists(cp_val):
            with open(cp_val) as f:
                validation = json.load(f)

            race_params = {}
            for grp in ("Black", "Hispanic"):
                cp = os.path.join(outdir, f"stage1_{grp}.json")
                if os.path.exists(cp):
                    with open(cp) as f:
                        race_params[grp] = json.load(f)

            gender_params = {}
            cp_g = os.path.join(outdir, "stage2_gender.json")
            if os.path.exists(cp_g):
                with open(cp_g) as f:
                    gender_params = json.load(f)

            if race_params and gender_params:
                generate_dissertation_outputs(
                    validation, race_params, gender_params, outdir
                )

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
            print("Usage: python phase2_bias_supervision_only_sequential_optimized.py --scope supervision_only|all_channels")
            sys.exit(1)

    stages = None
    if "--stage" in args:
        idx = args.index("--stage")
        try:
            stages = [int(args[idx + 1])]
        except (IndexError, ValueError):
            print("Usage: python phase2_bias_supervision_only_sequential_optimized.py --stage {1|2|3}")
            sys.exit(1)

    if "--cores" in args:
        idx = args.index("--cores")
        try:
            n_workers = int(args[idx + 1])
            print(f"\n  Manual override: {n_workers} workers")
        except (IndexError, ValueError):
            print("Usage: python phase2_bias_supervision_only_sequential_optimized.py [--cores N] [--stage N] "
                  "[--scope supervision_only|all_channels] [--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  Core auto-detection: {reason}")

    main(n_workers=n_workers, stages=stages, force_rerun=force_rerun,
         bias_scope=cli_scope)