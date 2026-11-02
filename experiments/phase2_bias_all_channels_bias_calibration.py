"""
phase2_bias_all_channels_gap_fast.py  —  Phase 2: Structural Bias Calibration
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
    "output_directory":      "phase2_bias_all_channels_bias_output",
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

# ── Phase 2 disparity targets ─────────────────────────────────────────────────
# Phase 1 calibrates the overall BJS recidivism trajectory.
# Phase 2 calibrates demographic DISPARITIES (group - reference), not mirrored
# subgroup rates. Race groups use White as the reference; Female uses Male.
#
# BJS target gaps:
#   Black - White     : +0.06, +0.05, +0.06 at 3/6/9 years
#   Hispanic - White  : +0.02, +0.02, +0.03
#   Female - Male     : -0.10, -0.09, -0.07
#
# Zero-bias results are retained only as an experimental control/diagnostic.
ZERO_BIAS_RATES: dict = {}
ZERO_BIAS_GAPS: dict = {}
_BASELINE_MEASURED: bool = False

# ── Sweep ranges ──────────────────────────────────────────────────────────────
RACE_REFERENCE   = "White"
GENDER_REFERENCE = "Male"
BIAS_FACTOR      = 1.0
# BIAS_SCOPE: controls which pipeline stages receive the bias shift.
#   "supervision_only" — surveillance artifact hypothesis (default)
#   "all_channels"     — structural embedding hypothesis
# Override via CLI: --scope all_channels
BIAS_SCOPE = "all_channels"

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

# Adaptive fine search: screen every fine-grid value cheaply, then run the
# expensive full-fidelity simulation only for the best few candidates.
# This preserves the final 100-run selection while avoiding 100 runs at every
# fine-grid point.
FINE_SCREEN_TOP_K     = 5

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

# ── Dissertation figure typography ─────────────────────────────────────────────
# Applies consistently to diagnostic and committee-facing charts.
# DejaVu Serif ships with Matplotlib and is reliable on HPC systems.
DISS_FONT_FAMILY  = "DejaVu Serif"
DISS_FONT_SIZE    = 13
DISS_TICK_SIZE    = 12
DISS_LABEL_SIZE   = 14
DISS_TITLE_SIZE   = 15
DISS_SUPTITLE_SIZE = 17
DISS_LEGEND_SIZE  = 11
DISS_DPI          = 300

plt.rcParams.update({
    "font.family": DISS_FONT_FAMILY,
    "font.size": DISS_FONT_SIZE,
    "axes.labelsize": DISS_LABEL_SIZE,
    "axes.titlesize": DISS_TITLE_SIZE,
    "axes.titleweight": "bold",
    "xtick.labelsize": DISS_TICK_SIZE,
    "ytick.labelsize": DISS_TICK_SIZE,
    "legend.fontsize": DISS_LEGEND_SIZE,
    "figure.titlesize": DISS_SUPTITLE_SIZE,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": DISS_DPI,
})
WINDOWS = [3, 6, 9]

BJS_GAP_TARGETS = {
    win: {
        "Black": BJS_SUBGROUP[win]["Black"] - BJS_SUBGROUP[win]["White"],
        "Hispanic": BJS_SUBGROUP[win]["Hispanic"] - BJS_SUBGROUP[win]["White"],
        "Female": BJS_SUBGROUP[win]["Female"] - BJS_SUBGROUP[win]["Male"],
    }
    for win in WINDOWS
}


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

    # No temporary directory is needed: export_csv is disabled for every run.
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
            while model.running:
                model.step()
            results.append(_extract_subgroup_rates(model))
        except Exception as e:
            print(f"  Warning: Worker error (tid={task_id:.4f}, seed={seed}): {e}",
                  flush=True)

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

    # Dynamic seed chunking. The old 20/25-seed batches severely underused a
    # 124-core node (e.g., one validation point produced only ~5 worker tasks).
    # Keep at least ~2 tasks per worker whenever enough simulations exist.
    total_runs = max(1, len(overrides_list) * len(seeds) * n_reps)
    requested_bs = batch_size if batch_size is not None else _WORKER_BATCH_SIZE
    adaptive_bs = max(1, math.ceil(total_runs / max(1, n_workers * 2)))
    _bs = max(1, min(requested_bs, adaptive_bs))

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


def _reference_group(group: str) -> str:
    """Return the BJS reference group used for Phase 2 disparity calibration."""
    return "Male" if group == "Female" else "White"


def _compute_gap_mae(agg: dict, group: str) -> float:
    """
    Primary Phase 2 calibration objective.

    Compares the simulated subgroup gap (group - reference) with the observed
    BJS subgroup gap at 3, 6, and 9 years.
    """
    reference = _reference_group(group)
    errors = []

    for win in WINDOWS:
        group_mean = agg.get(win, {}).get(group, {}).get("mean")
        ref_mean = agg.get(win, {}).get(reference, {}).get("mean")
        target_gap = BJS_GAP_TARGETS.get(win, {}).get(group)

        if group_mean is None or ref_mean is None or target_gap is None:
            continue

        simulated_gap = group_mean - ref_mean
        errors.append(abs(simulated_gap - target_gap))

    return float(np.mean(errors)) if errors else 999.0


def _compute_rate_mae(agg: dict, group: str) -> float:
    """
    Diagnostic only: absolute subgroup-rate MAE versus BJS.

    This metric is reported but is NOT used to select the calibrated bias.
    """
    errors = []

    for win in WINDOWS:
        sim_rate = agg.get(win, {}).get(group, {}).get("mean")
        target_rate = BJS_SUBGROUP.get(win, {}).get(group)

        if sim_rate is None or target_rate is None:
            continue

        errors.append(abs(sim_rate - target_rate))

    return float(np.mean(errors)) if errors else 999.0


def _select_fine_candidates(screen_results: list, top_k: int = FINE_SCREEN_TOP_K) -> list:
    """Return the best unique fine-grid values from a cheap screening pass."""
    if not screen_results:
        return []
    selected = []
    seen = set()
    for row in sorted(screen_results, key=lambda x: x["gap_mae"]):
        value = float(row["value"])
        if value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) >= top_k:
            break
    return sorted(selected)


def _extract_gap_summary(agg: dict) -> dict:
    """Return simulated group-reference gaps for Black, Hispanic, and Female."""
    gaps = {}
    for win in WINDOWS:
        gaps[win] = {}
        for group in ("Black", "Hispanic", "Female"):
            reference = _reference_group(group)
            group_mean = agg.get(win, {}).get(group, {}).get("mean")
            ref_mean = agg.get(win, {}).get(reference, {}).get("mean")
            gaps[win][group] = (
                None if group_mean is None or ref_mean is None
                else float(group_mean - ref_mean)
            )
    return gaps


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
# ZERO-BIAS CONTROL MEASUREMENT
# =============================================================================

def measure_zero_bias_baseline(n_workers: int, outdir: str) -> None:
    """
    Measure the zero-explicit-bias control condition.

    The zero-bias run is retained as a diagnostic showing how much subgroup
    disparity emerges from the synthetic population and lifecycle before an
    explicit Phase 2 bias term is introduced.

    IMPORTANT:
      - It does NOT create mirrored "adjusted targets".
      - Phase 2 calibration targets are the observed BJS group-reference gaps.
    """
    global ZERO_BIAS_RATES, ZERO_BIAS_GAPS, _BASELINE_MEASURED

    cp = os.path.join(outdir, "zero_bias_baseline.json")
    if os.path.exists(cp):
        print("  > Zero-bias baseline — loading checkpoint")
        with open(cp) as f:
            saved = json.load(f)
        ZERO_BIAS_RATES = {int(k): v for k, v in saved["rates"].items()}
        ZERO_BIAS_GAPS = {int(k): v for k, v in saved["gaps"].items()}
        _BASELINE_MEASURED = True
        for win in WINDOWS:
            g = ZERO_BIAS_GAPS[win]
            print(
                f"     {win}yr  "
                f"Black-White:{g['Black']:+.4f}  "
                f"Hispanic-White:{g['Hispanic']:+.4f}  "
                f"Female-Male:{g['Female']:+.4f}"
            )
        return

    print(
        f"\n  > Measuring zero-bias baseline "
        f"({len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps = "
        f"{len(SEEDS_COARSE) * N_REPS_COARSE} runs, {AGENTS_COARSE} agents)..."
    )

    payload = _build_payload(
        bias_factor=0.0,
        group_bias_race={"White": 0.0, "Black": 0.0, "Hispanic": 0.0, "Other": 0.0},
        group_bias_gender={"Male": 0.0, "Female": 0.0},
        # Scope is irrelevant at bias_factor=0.0 because _apply_bias() returns 0.
        bias_scope=BIAS_SCOPE,
        task_id=0.0,
        agents=AGENTS_COARSE,
    )

    raw = _run_batch(
        [(0.0, payload)], n_workers,
        desc="Zero-bias baseline",
        n_reps_override=N_REPS_COARSE,
        batch_size=_WORKER_BATCH_SIZE_COARSE,
        seeds_override=SEEDS_COARSE,
    )
    agg = _aggregate_subgroup_rates(raw.get(0.0, []))

    if not agg or not all(win in agg for win in WINDOWS):
        raise RuntimeError(
            "Zero-bias baseline produced no results — all workers failed.\n"
            "Check worker errors above."
        )

    rates = {
        win: {
            grp: agg[win].get(grp, {}).get("mean")
            for grp in ("Male", "Female", "White", "Black", "Hispanic", "Overall")
        }
        for win in WINDOWS
    }
    gaps = _extract_gap_summary(agg)

    ZERO_BIAS_RATES = rates
    ZERO_BIAS_GAPS = gaps
    _BASELINE_MEASURED = True

    with open(cp, "w") as f:
        json.dump({
            "rates": {str(k): v for k, v in rates.items()},
            "gaps": {str(k): v for k, v in gaps.items()},
            "bjs_gap_targets": {str(k): v for k, v in BJS_GAP_TARGETS.items()},
            "measured_at": pd.Timestamp.now().isoformat(),
            "seeds_used": SEEDS_COARSE,
            "agents_used": AGENTS_COARSE,
            "bias_scope": BIAS_SCOPE,
        }, f, indent=2)

    print(f"  OK  Zero-bias baseline measured -> {cp}")
    for win in WINDOWS:
        g = ZERO_BIAS_GAPS[win]
        print(
            f"     {win}yr  "
            f"Black-White:{g['Black']:+.4f}  "
            f"Hispanic-White:{g['Hispanic']:+.4f}  "
            f"Female-Male:{g['Female']:+.4f}"
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
        gap_mae = _compute_gap_mae(agg, race_group)
        rate_mae = _compute_rate_mae(agg, race_group)
        results.append({
            "value": val,
            # Keep "mae" as an alias for backward-compatible plotting/checkpoint code.
            "mae": gap_mae,
            "gap_mae": gap_mae,
            "rate_mae": rate_mae,
            "agg": agg,
        })

    # Bias selection is based ONLY on disparity fit.
    results.sort(key=lambda x: x["gap_mae"])
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
                "value":       best["value"], "mae": best["gap_mae"], "gap_mae": best["gap_mae"],
                "agg":         {str(k): v for k, v in best["agg"].items()},
                "all_results": [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in results],
                "gap_target": {str(w): BJS_GAP_TARGETS[w][grp] for w in WINDOWS},
                "rate_target": {str(w): BJS_SUBGROUP[w][grp] for w in WINDOWS},
                "zero_bias_gap": {str(w): ZERO_BIAS_GAPS.get(w, {}).get(grp) for w in WINDOWS},
                "rate_mae": best.get("rate_mae"),
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
            print(f"\n  Pass 2a (fine screen): {len(fine_vals)} pts  "
                  f"[{AGENTS_COARSE} agents, "
                  f"{len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")
            fine_screen_results = _sweep_race_group(
                grp, fine_vals, n_workers,
                "Stage1 Black fine-screen",
                n_reps_override=N_REPS_COARSE,
                is_coarse=True,
            )
            candidate_vals = _select_fine_candidates(fine_screen_results)
            print(f"  Pass 2b (full validation): {len(candidate_vals)} candidates  "
                  f"[{CONFIG['initial_agents']} agents, "
                  f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")
            fine_results = _sweep_race_group(
                grp, candidate_vals, n_workers,
                "Stage1 Black fine-validate",
                n_reps_override=N_REPS_FINE,
                is_coarse=False,
            )
            fine_best = fine_results[0]
            print(f"  OK  Fine best = {fine_best['value']:.3f}  "
                  f"(Gap MAE={fine_best['mae']:.4f})")

            best_params[grp] = {
                "value":          fine_best["value"], "mae": fine_best["gap_mae"], "gap_mae": fine_best["gap_mae"],
                "agg":            {str(k): v for k, v in fine_best["agg"].items()},
                "coarse_results": [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in coarse_results],
                "fine_screen_results": [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in fine_screen_results],
                "fine_results":   [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in fine_results],
                "all_results":    [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in coarse_results],
                "gap_target":      {str(w): BJS_GAP_TARGETS[w][grp] for w in WINDOWS},
                "rate_target":     {str(w): BJS_SUBGROUP[w][grp] for w in WINDOWS},
                "zero_bias_gap":   {str(w): ZERO_BIAS_GAPS.get(w, {}).get(grp) for w in WINDOWS},
                "rate_mae":        fine_best.get("rate_mae"),
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
        gap_mae = _compute_gap_mae(agg, "Female")
        rate_mae = _compute_rate_mae(agg, "Female")
        results.append({
            "value": val,
            # Keep "mae" as an alias for backward-compatible plotting/checkpoint code.
            "mae": gap_mae,
            "gap_mae": gap_mae,
            "rate_mae": rate_mae,
            "agg": agg,
        })

    # Bias selection is based ONLY on Female-Male disparity fit.
    results.sort(key=lambda x: x["gap_mae"])
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
    print(f"\n  Pass 2a (fine screen): {len(fine_vals)} pts  "
          f"[{AGENTS_COARSE} agents, "
          f"{len(SEEDS_COARSE)} seeds × {N_REPS_COARSE} reps]")
    fine_screen_results = _sweep_female(
        race_bias, fine_vals, n_workers,
        "Stage2 Female fine-screen",
        n_reps_override=N_REPS_COARSE,
        is_coarse=True,
    )
    candidate_vals = _select_fine_candidates(fine_screen_results)
    print(f"  Pass 2b (full validation): {len(candidate_vals)} candidates  "
          f"[{CONFIG['initial_agents']} agents, "
          f"{len(CONFIG['SEEDS'])} seeds × {N_REPS_FINE} reps]")
    fine_results = _sweep_female(
        race_bias, candidate_vals, n_workers,
        "Stage2 Female fine-validate",
        n_reps_override=N_REPS_FINE,
        is_coarse=False,
    )
    fine_best = fine_results[0]
    print(f"  OK  Fine best = {fine_best['value']:.3f}  "
          f"(Gap MAE={fine_best['mae']:.4f})")

    best_params = {"Female": {
        "value":          fine_best["value"], "mae": fine_best["gap_mae"], "gap_mae": fine_best["gap_mae"],
        "agg":            {str(k): v for k, v in fine_best["agg"].items()},
        "coarse_results": [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in coarse_results],
        "fine_screen_results": [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in fine_screen_results],
        "fine_results":   [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in fine_results],
        "all_results":    [{"value": r["value"], "mae": r["mae"], "rate_mae": r.get("rate_mae")} for r in coarse_results],
        "gap_target":      {str(w): BJS_GAP_TARGETS[w]["Female"] for w in WINDOWS},
        "rate_target":     {str(w): BJS_SUBGROUP[w]["Female"] for w in WINDOWS},
        "zero_bias_gap":   {str(w): ZERO_BIAS_GAPS.get(w, {}).get("Female") for w in WINDOWS},
        "rate_mae":        fine_best.get("rate_mae"),
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
    """Apply consistent dissertation-ready typography to an axis."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=DISS_LABEL_SIZE, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=DISS_LABEL_SIZE, labelpad=8)
    if title:
        ax.set_title(title, fontsize=DISS_TITLE_SIZE, fontweight="bold",
                     pad=10, color="#1A3D5C")
    ax.grid(True, color=COLORS["grid"], linewidth=0.65, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=DISS_TICK_SIZE)


def plot_stage1_calibration(race_params: dict, outdir: str) -> str:
    """Plot Stage 1 calibration curves in ascending bias-value order.

    Sweep functions sort rows by MAE so results[0] is the optimum. Plotting those
    rows directly creates zig-zag curves, so every series is sorted by ``value``.
    """
    SMOOTH_SIGMA = 2.0

    black_data = race_params.get("Black", {})
    coarse_rows_b = sorted(
        black_data.get("coarse_results", black_data.get("all_results", [])),
        key=lambda r: r["value"],
    )
    fine_screen_b = sorted(
        black_data.get("fine_screen_results", []), key=lambda r: r["value"]
    )
    fine_rows_b = sorted(
        black_data.get("fine_results", []), key=lambda r: r["value"]
    )
    best_b = black_data.get("value")
    coarse_best_b = black_data.get("coarse_best", best_b)

    ncols_b = 2 if (fine_screen_b or fine_rows_b) else 1
    fig_b, axes_b = plt.subplots(1, ncols_b, figsize=(7 * ncols_b, 5))
    fig_b.patch.set_facecolor("#FAFAFA")
    if ncols_b == 1:
        axes_b = [axes_b]

    ax = axes_b[0]
    if coarse_rows_b:
        vals = [r["value"] for r in coarse_rows_b]
        maes = [r["mae"] for r in coarse_rows_b]
        ax.plot(vals, maes, color=COLORS["Black"], linewidth=0.9,
                alpha=0.35, linestyle="--", zorder=2, label="Gap MAE (raw)")
        ax.plot(vals, gaussian_filter1d(maes, sigma=SMOOTH_SIGMA),
                color=COLORS["Black"], linewidth=2.2, zorder=3,
                label=f"Gap MAE (smoothed σ={SMOOTH_SIGMA})")
        ax.axvline(coarse_best_b, color="#333333", linewidth=1.5,
                   linestyle="--", zorder=4, label=f"Coarse best={coarse_best_b:.2f}")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=DISS_TICK_SIZE)
    _style_ax(ax, "Black Bias (log-odds)", "Gap MAE vs BJS Disparity Target",
              "Stage 1 — Black Pass 1: Coarse 0.0→1.5 (step 0.1)\n"
              f"bias_scope={BIAS_SCOPE}")

    if ncols_b == 2:
        ax2 = axes_b[1]
        if fine_screen_b:
            svals = [r["value"] for r in fine_screen_b]
            smaes = [r["mae"] for r in fine_screen_b]
            ax2.plot(svals, smaes, color=COLORS["Black"], linewidth=1.0,
                     alpha=0.35, linestyle="--", zorder=2,
                     label="Fine screen (500 agents, 25 runs/pt)")
            ax2.plot(svals, gaussian_filter1d(smaes, sigma=1.2),
                     color=COLORS["Black"], linewidth=1.7, alpha=0.65,
                     zorder=3, label="Fine screen (smoothed)")
        if fine_rows_b:
            fvals = [r["value"] for r in fine_rows_b]
            fmaes = [r["mae"] for r in fine_rows_b]
            ax2.plot(fvals, fmaes, color=COLORS["Black"], linewidth=2.0,
                     marker="o", markersize=5, zorder=5,
                     label="Full validation candidates")
        if best_b is not None:
            ax2.axvline(best_b, color="#333333", linewidth=1.7,
                        linestyle="--", zorder=6,
                        label=f"FINAL best={best_b:.3f} (Gap MAE={black_data['mae']:.4f})")
        if coarse_best_b is not None:
            ax2.axvline(coarse_best_b, color="#888888", linewidth=1.0,
                        linestyle=":", zorder=4,
                        label=f"Coarse best={coarse_best_b:.2f}")
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=DISS_TICK_SIZE)
        _style_ax(ax2, "Black Bias (log-odds)", "Gap MAE",
                  f"Pass 2: Adaptive fine search ±{BLACK_SWEEP_FINE_HALF_WIDTH:.2f} around coarse best")

    fig_b.suptitle(
        "Stage 1 — Black Race Bias Calibration (adaptive fine search)\n"
        f"bias_scope={BIAS_SCOPE}",
        fontsize=DISS_TITLE_SIZE, fontweight="bold", color="#1A3D5C", y=1.04)
    plt.tight_layout()
    path_b = os.path.join(outdir, "phase2_stage1_black_calibration.png")
    plt.savefig(path_b, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig_b.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 Black -> {path_b}")

    hisp_data = race_params.get("Hispanic", {})
    rows_h = sorted(hisp_data.get("all_results", []), key=lambda r: r["value"])
    best_h = hisp_data.get("value")
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    fig_h.patch.set_facecolor("#FAFAFA")
    if rows_h:
        vals = [r["value"] for r in rows_h]
        maes = [r["mae"] for r in rows_h]
        ax_h.plot(vals, maes, color=COLORS["Hispanic"], linewidth=0.9,
                  alpha=0.35, linestyle="--", zorder=2, label="Gap MAE (raw)")
        ax_h.plot(vals, gaussian_filter1d(maes, sigma=SMOOTH_SIGMA),
                  color=COLORS["Hispanic"], linewidth=2.2, zorder=3,
                  label=f"Gap MAE (smoothed σ={SMOOTH_SIGMA})")
        if best_h is not None:
            ax_h.axvline(best_h, color="#333333", linewidth=1.5,
                         linestyle="--", zorder=4,
                         label=f"Best={best_h:.2f} (Gap MAE={hisp_data['mae']:.4f})")
        ax_h.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax_h.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_h.legend(fontsize=DISS_TICK_SIZE)
    _style_ax(ax_h, "Hispanic Bias (log-odds)", "Gap MAE vs BJS Disparity Target",
              f"Stage 1 — Hispanic Race Bias\nbias_scope={BIAS_SCOPE}")
    plt.tight_layout()
    path_h = os.path.join(outdir, "phase2_stage1_hispanic_calibration.png")
    plt.savefig(path_h, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig_h.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 Hispanic -> {path_h}")

    fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 5))
    fig_c.patch.set_facecolor("#FAFAFA")

    ax_c = axes_c[0]
    if coarse_rows_b:
        vals = [r["value"] for r in coarse_rows_b]
        maes = [r["mae"] for r in coarse_rows_b]
        ax_c.plot(vals, maes, color=COLORS["Black"], linewidth=0.9,
                  alpha=0.35, linestyle="--", zorder=2, label="Coarse Gap MAE (raw)")
        ax_c.plot(vals, gaussian_filter1d(maes, sigma=SMOOTH_SIGMA),
                  color=COLORS["Black"], linewidth=2.0, zorder=3,
                  label="Coarse Gap MAE (smoothed)")
        if coarse_best_b is not None:
            ax_c.axvline(coarse_best_b, color="#888888", linewidth=1.0,
                         linestyle=":", zorder=4,
                         label=f"Coarse best={coarse_best_b:.2f}")
        if best_b is not None:
            ax_c.axvline(best_b, color="#333333", linewidth=1.6,
                         linestyle="--", zorder=5,
                         label=f"FINAL best={best_b:.3f} (Gap MAE={black_data['mae']:.4f})")
        ax_c.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle=":")
        ax_c.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_c.legend(fontsize=DISS_TICK_SIZE)
    _style_ax(ax_c, "Black Bias (log-odds)", "Gap MAE vs BJS Disparity Target",
              "Stage 1 — Black")

    ax_c = axes_c[1]
    if rows_h:
        vals = [r["value"] for r in rows_h]
        maes = [r["mae"] for r in rows_h]
        ax_c.plot(vals, maes, color=COLORS["Hispanic"], linewidth=0.9,
                  alpha=0.35, linestyle="--", zorder=2, label="Gap MAE (raw)")
        ax_c.plot(vals, gaussian_filter1d(maes, sigma=SMOOTH_SIGMA),
                  color=COLORS["Hispanic"], linewidth=2.0, zorder=3,
                  label="Gap MAE (smoothed)")
        if best_h is not None:
            ax_c.axvline(best_h, color="#333333", linewidth=1.6,
                         linestyle="--", zorder=5,
                         label=f"Best={best_h:.2f} (Gap MAE={hisp_data['mae']:.4f})")
        ax_c.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax_c.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_c.legend(fontsize=DISS_TICK_SIZE)
    _style_ax(ax_c, "Hispanic Bias (log-odds)", "Gap MAE vs BJS Disparity Target",
              "Stage 1 — Hispanic")

    fig_c.suptitle(
        "Stage 1: Race Bias Calibration — Gap-MAE Loss Curves\n"
        f"bias_scope={BIAS_SCOPE}",
        fontsize=DISS_TITLE_SIZE, fontweight="bold", color="#1A3D5C", y=1.04)
    plt.tight_layout()
    path_combined = os.path.join(outdir, "phase2_stage1_race_calibration.png")
    plt.savefig(path_combined, dpi=DISS_DPI, bbox_inches="tight",
                facecolor=fig_c.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 combined -> {path_combined}")
    return path_combined


def plot_stage2_calibration(gender_params: dict, outdir: str) -> str:
    """Plot Female calibration in ascending bias-value order."""
    data = gender_params.get("Female", {})
    coarse_rows = sorted(
        data.get("coarse_results", data.get("all_results", [])),
        key=lambda r: r["value"],
    )
    fine_screen = sorted(data.get("fine_screen_results", []), key=lambda r: r["value"])
    fine_rows = sorted(data.get("fine_results", []), key=lambda r: r["value"])
    best_v = data.get("value")
    coarse_best = data.get("coarse_best", best_v)

    ncols = 2 if (fine_screen or fine_rows) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.patch.set_facecolor("#FAFAFA")
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    if coarse_rows:
        vals = [r["value"] for r in coarse_rows]
        maes = [r["mae"] for r in coarse_rows]
        ax.plot(vals, maes, color=COLORS["Female"], linewidth=1.0,
                alpha=0.4, linestyle="--", zorder=2, label="Gap MAE (raw)")
        ax.plot(vals, gaussian_filter1d(maes, sigma=1.5),
                color=COLORS["Female"], linewidth=2.0, zorder=3,
                label="Gap MAE (smoothed)")
        if coarse_best is not None:
            ax.axvline(coarse_best, color="#333333", linewidth=1.5,
                       linestyle="--", label=f"Coarse best={coarse_best:.2f}")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=DISS_TICK_SIZE)
    _style_ax(ax, "Female Bias (log-odds)", "Gap MAE vs BJS Disparity Target",
              "Stage 2 — Female Pass 1: Coarse −3.0→+0.5 (step 0.1)")

    if ncols == 2:
        ax2 = axes[1]
        if fine_screen:
            svals = [r["value"] for r in fine_screen]
            smaes = [r["mae"] for r in fine_screen]
            ax2.plot(svals, smaes, color=COLORS["Female"], linewidth=1.0,
                     alpha=0.35, linestyle="--", zorder=2,
                     label="Fine screen (500 agents, 25 runs/pt)")
            ax2.plot(svals, gaussian_filter1d(smaes, sigma=1.2),
                     color=COLORS["Female"], linewidth=1.7, alpha=0.65,
                     zorder=3, label="Fine screen (smoothed)")
        if fine_rows:
            fvals = [r["value"] for r in fine_rows]
            fmaes = [r["mae"] for r in fine_rows]
            ax2.plot(fvals, fmaes, color=COLORS["Female"], linewidth=2.0,
                     marker="o", markersize=5, zorder=5,
                     label="Full validation candidates")
        if best_v is not None:
            ax2.axvline(best_v, color="#333333", linewidth=1.7,
                        linestyle="--", zorder=6,
                        label=f"FINAL best={best_v:.3f} (Gap MAE={data['mae']:.4f})")
        if coarse_best is not None:
            ax2.axvline(coarse_best, color="#888888", linewidth=1.0,
                        linestyle=":", zorder=4,
                        label=f"Coarse best={coarse_best:.2f}")
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=DISS_TICK_SIZE)
        _style_ax(ax2, "Female Bias (log-odds)", "Gap MAE",
                  f"Pass 2: Adaptive fine search ±{GENDER_SWEEP_FINE_HALF_WIDTH:.2f} around coarse best")

    fig.suptitle(
        "Stage 2 — Female Gender Bias Calibration (adaptive fine search)\n"
        f"bias_scope={BIAS_SCOPE}  |  consistent with Stage 1",
        fontsize=DISS_SUPTITLE_SIZE, fontweight="bold", color="#1A3D5C", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_stage2_gender_calibration.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
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
        ax.set_xticklabels(race_groups, fontsize=DISS_TICK_SIZE)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(0.4, 1.0)
        _style_ax(ax, "", "Rearrest Rate", f"{win}-Year Rate by Race")
        if col == 0:
            ax.legend(fontsize=DISS_TICK_SIZE)

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
    ax_g.set_xticklabels(gender_groups, fontsize=DISS_TICK_SIZE)
    ax_g.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_g.set_ylim(0.4, 1.0)
    ax_g.legend(fontsize=DISS_TICK_SIZE)
    _style_ax(ax_g, "", "Rearrest Rate", "3-Year Rate by Gender")

    ax_h = fig.add_subplot(gs[1, 1:])
    intersect_groups = [f"{r}_{g}" for r in race_groups for g in gender_groups]
    matrix = np.zeros((len(WINDOWS), len(intersect_groups)))
    for i, win in enumerate(WINDOWS):
        for j, grp in enumerate(intersect_groups):
            matrix[i, j] = agg[win].get(grp, {}).get("mean", 0) or 0
    im = ax_h.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.4, vmax=0.95)
    ax_h.set_xticks(range(len(intersect_groups)))
    ax_h.set_xticklabels([g.replace("_", "\n") for g in intersect_groups], fontsize=DISS_TICK_SIZE)
    ax_h.set_yticks(range(len(WINDOWS)))
    ax_h.set_yticklabels([f"{w}-Year" for w in WINDOWS], fontsize=DISS_TICK_SIZE)
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
        fontsize=DISS_SUPTITLE_SIZE, fontweight="bold", color="#1A3D5C", y=0.98)
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
            ax.legend(fontsize=DISS_TICK_SIZE)
            _style_ax(ax, "bias_factor", "Gap (pp)", f"{title}\n{win}-Year")
    fig.suptitle(
        "Stage 3 — Bias Factor Sensitivity\n"
        f"±40% perturbation of bias_factor  |  bias_scope={BIAS_SCOPE}",
        fontsize=DISS_SUPTITLE_SIZE, fontweight="bold", color="#1A3D5C", y=1.01)
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
            ax.set_xticklabels(groups, fontsize=DISS_TICK_SIZE)
            _style_ax(ax, "", title if col == 0 else "",
                      f"{win}-Year | {title.split(chr(10))[0]}" if row == 0 else "")

    fig.suptitle(
        "Phase 2 Disparity Summary — All Three Metrics\n"
        "Gap = group − reference  |  Ratio = group / reference  |  "
        "Abs Dev = group − baseline\n"
        f"bias_scope={BIAS_SCOPE}",
        fontsize=DISS_SUPTITLE_SIZE, fontweight="bold", color="#1A3D5C", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_disparity_summary.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
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
# DISSERTATION / COMMITTEE-FACING CHARTS
# =============================================================================

def _committee_loss_rows(param_data: dict) -> list:
    """Prefer high-fidelity validated points; fall back to coarse/screen rows."""
    rows = param_data.get("fine_results", [])
    if not rows:
        rows = param_data.get("fine_screen_results", [])
    if not rows:
        rows = param_data.get("all_results", param_data.get("coarse_results", []))
    return sorted(rows, key=lambda r: r["value"])


def _plot_committee_loss(ax, param_data: dict, group: str, reference: str):
    rows = _committee_loss_rows(param_data)
    if not rows:
        return

    x = [float(r["value"]) for r in rows]
    y_pp = [100.0 * float(r.get("gap_mae", r.get("mae", 0.0))) for r in rows]

    ax.plot(x, y_pp, linewidth=2.3, marker="o", markersize=5)

    best = float(param_data["value"])
    best_mae_pp = 100.0 * float(param_data.get("gap_mae", param_data.get("mae", 0.0)))

    ax.axvline(best, linestyle="--", linewidth=1.6)
    ax.scatter([best], [best_mae_pp], s=55, zorder=5)

    ax.annotate(
        f"Calibrated β = {best:.2f}\nDisparity MAE = {best_mae_pp:.2f} pp",
        xy=(best, best_mae_pp),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=DISS_LEGEND_SIZE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )

    _style_ax(
        ax,
        "Bias parameter (β)",
        f"Mean absolute error in {group}–{reference} disparity (percentage points)",
        f"{group}–{reference} disparity calibration",
    )


def plot_committee_race_calibration(race_params: dict, outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    _plot_committee_loss(axes[0], race_params["Black"], "Black", "White")
    _plot_committee_loss(axes[1], race_params["Hispanic"], "Hispanic", "White")

    fig.suptitle("Calibration of Race-Based Rearrest Disparities",
                 fontsize=DISS_SUPTITLE_SIZE, fontweight="bold")

    fig.text(
        0.5, 0.01,
        "Disparity MAE is the mean absolute difference between simulated and BJS "
        "group-reference rearrest-rate gaps across the 3-, 6-, and 9-year follow-up periods. "
        "Lower values indicate closer agreement.",
        ha="center", va="bottom", fontsize=DISS_LEGEND_SIZE
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.92])
    path = os.path.join(outdir, "phase2_committee_race_calibration.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Committee chart: race calibration -> {path}")
    return path


def plot_committee_gender_calibration(gender_params: dict, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("white")

    _plot_committee_loss(ax, gender_params["Female"], "Female", "Male")

    fig.suptitle("Calibration of Female–Male Rearrest Disparity",
                 fontsize=DISS_SUPTITLE_SIZE, fontweight="bold")

    fig.text(
        0.5, 0.01,
        "Disparity MAE is the mean absolute difference between simulated and BJS "
        "Female–Male rearrest-rate gaps across the 3-, 6-, and 9-year follow-up periods. "
        "Lower values indicate closer agreement.",
        ha="center", va="bottom", fontsize=DISS_LEGEND_SIZE
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    path = os.path.join(outdir, "phase2_committee_gender_calibration.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Committee chart: gender calibration -> {path}")
    return path


def plot_committee_disparity_fit(validation: dict, outdir: str) -> str:
    agg = {int(k): v for k, v in validation["agg"].items()}

    comparisons = [
        ("Black–White", "Black", "White"),
        ("Hispanic–White", "Hispanic", "White"),
        ("Female–Male", "Female", "Male"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, (label, group, ref) in zip(axes, comparisons):
        bjs = [100.0 * BJS_GAP_TARGETS[w][group] for w in WINDOWS]
        sim = [
            100.0 * (
                (agg[w].get(group, {}).get("mean") or 0.0)
                - (agg[w].get(ref, {}).get("mean") or 0.0)
            )
            for w in WINDOWS
        ]

        x = np.arange(len(WINDOWS))
        width = 0.34
        ax.bar(x - width/2, bjs, width, label="BJS observed")
        ax.bar(x + width/2, sim, width, label="Simulated")
        ax.axhline(0, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}-year" for w in WINDOWS])
        _style_ax(ax, "Follow-up period", "Rearrest-rate gap (percentage points)",
                  label)

    axes[0].legend(fontsize=DISS_LEGEND_SIZE)
    fig.suptitle("Observed and Simulated Rearrest Disparities After Calibration",
                 fontsize=DISS_SUPTITLE_SIZE, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    path = os.path.join(outdir, "phase2_committee_disparity_fit.png")
    plt.savefig(path, dpi=DISS_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Committee chart: disparity fit -> {path}")
    return path

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
        BIAS_SCOPE = "all_channels"

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
            bp = os.path.join(outdir, "zero_bias_baseline.json")
            if os.path.exists(bp):
                os.remove(bp)
                print(f"    deleted: zero_bias_baseline.json (will remeasure)")

    measure_zero_bias_baseline(n_workers, outdir)

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
        "bjs_targets": BJS_SUBGROUP,
        "bjs_gap_targets": BJS_GAP_TARGETS,
        "zero_bias_rates": ZERO_BIAS_RATES,
        "zero_bias_gaps": ZERO_BIAS_GAPS,
        "calibration_objective": "group_gap_mae",
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
    print("  FINAL ALL-CHANNELS CALIBRATION SUMMARY")
    print("=" * 70)
    rows = [
        (
            grp,
            race_params[grp].get("value"),
            race_params[grp].get("gap_mae", race_params[grp].get("mae")),
            race_params[grp].get("rate_mae"),
        )
        for grp in ["Black", "Hispanic"] if grp in race_params
    ]
    if "Female" in gender_params:
        rows.append((
            "Female",
            gender_params["Female"].get("value"),
            gender_params["Female"].get("gap_mae", gender_params["Female"].get("mae")),
            gender_params["Female"].get("rate_mae"),
        ))
    print(f"  {'Group':<12} {'Latent bias β':>16} {'exp(β)':>12} {'Gap MAE':>10} {'Rate MAE':>10}")
    print(f"  {'-'*74}")
    for grp, b, gap_m, rate_m in rows:
        if b is not None:
            gap_txt = f"{gap_m:.4f}" if gap_m is not None else "-"
            rate_txt = f"{rate_m:.4f}" if rate_m is not None else "-"
            print(
                f"  {grp:<12} {b:>16.3f} {math.exp(b):>11.3f}x "
                f"{gap_txt:>10} {rate_txt:>10}"
            )
    print(f"  {'White':<12} {'Reference (0.00)':>16} {'1.000x':>12} {'-':>10} {'-':>10}")
    print(f"  {'Male':<12} {'Reference (0.00)':>16} {'1.000x':>12} {'-':>10} {'-':>10}")
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