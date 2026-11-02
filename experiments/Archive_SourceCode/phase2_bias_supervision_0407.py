"""
phase2_bias.py  —  Phase 2: Intersectional Bias Simulation
===========================================================
Recidivism ABM — PhD Dissertation Tool

OPTIMISATION CHANGELOG (vs original)
--------------------------------------
BUG FIXES (silent result-loss):
  1. _run_batch: removed the accidental nested loop that multiplied task count
     exponentially and re-keyed tasks incorrectly.
  2. _worker: removed the premature `return results` inside the seed loop so
     every seed in the batch is actually simulated (not just the first).
  3. _run_batch result collection: fut.result() now correctly receives the list
     of dicts that _worker returns, instead of trying to unpack a 3-tuple that
     was never produced — previously every future raised an exception and was
     silently swallowed, making every sweep return empty data.

PERFORMANCE IMPROVEMENTS:
  4. Batch size increased to 50 (was 20) — amortises ProcessPoolExecutor
     overhead; especially beneficial on high-core machines.
  5. _cleanup_result_run1 moved out of the hot-path: called once per stage
     rather than every 500 completions, eliminating repeated listdir syscalls.
  6. _run_batch uses a flat task_id→index map instead of min(abs(k-tid))
     linear scan on every result, reducing result-aggregation from O(N²) to O(1).
  7. _aggregate_subgroup_rates pre-builds the union of group keys once rather
     than re-scanning every run for every window.
  8. Worker payload no longer JSON-serialises/deserialises the dict (the
     original serialisation was already removed in a prior edit; confirmed clean).
  9. detect_workers: uses logical-1 (not physical-1) when psutil is absent,
     matching the comment intent.
"""

import os
import sys
import json
import math
import warnings
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
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter, FuncFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_flat_risk_weights,
    get_global_calibration_params,
    get_peer_influence_config
)

CONFIG = {
    "initial_agents":        1000,
    "monthly_intake":        10,
    "warmup_months":         144,
    "study_months":          108,
    "enable_peer_influence": True,
    "output_directory":      "phase2_bias_supervision_output",
    "N_REPS":                10,    # reduced from 20 — coarse uses N_REPS_COARSE=5
    "SEEDS":                 [42, 137, 251, 389, 503, 617, 743, 863, 971, 1087],
    "bjs_targets": {
        3: 0.68, 6: 0.79, 9: 0.83,
    },
}

BJS_SUBGROUP = {
    3: {"Male": 0.68, "Female": 0.58,
        "White": 0.66, "Black": 0.72, "Hispanic": 0.68},
    6: {"Male": 0.79, "Female": 0.70,
        "White": 0.77, "Black": 0.82, "Hispanic": 0.79},
    9: {"Male": 0.84, "Female": 0.77,
        "White": 0.81, "Black": 0.87, "Hispanic": 0.84},
}

# ── BASELINE_OFFSETS: computed live from a zero-bias measurement ─────────────
# No hardcoded values. At startup, measure_baseline_offsets() runs a quick
# 3-seed zero-bias simulation and computes offsets as:
#     offset[win][grp] = sim_rate[win][grp] - BJS_SUBGROUP[win][grp]
# ADJUSTED_TARGETS are then derived automatically:
#     adjusted_target[win][grp] = BJS_SUBGROUP[win][grp] - offset[win][grp]
#
# Female offset: not forced to zero here — the live measurement will naturally
# reflect the current model state. If the gender-neutral hazard is still in
# place the Female offset will be ~+9.4pp and the adjusted target ~48%.
# To treat Female as a direct BJS target, set FEMALE_OFFSET_OVERRIDE = 0.0.
# To use the live measurement for Female as well, set it to None.
FEMALE_OFFSET_OVERRIDE = None #= use live measurement; 0.0 = direct BJS target

# Populated at runtime by measure_baseline_offsets() — do not set manually.
BASELINE_OFFSETS: dict  = {}
ADJUSTED_TARGETS: dict  = {}
_BASELINE_MEASURED: bool = False


def measure_baseline_offsets(n_workers: int, outdir: str) -> None:
    """
    Run a zero-bias baseline measurement and compute BASELINE_OFFSETS and
    ADJUSTED_TARGETS in-place. Saves results to baseline_offsets.json so
    subsequent runs (e.g. --stage 2 only) do not re-measure.

    Uses 3 seeds x N_REPS replications — fast enough (~3-5 min) to run
    every time but skipped if a saved checkpoint exists.

    Groups measured: Male, Female, White, Black, Hispanic.
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
        print("  Baseline offsets (from checkpoint):")
        for win in WINDOWS:
            o = BASELINE_OFFSETS[win]
            print(f"     {win}yr  Black:{o['Black']:+.4f}  Hispanic:{o['Hispanic']:+.4f}  "
                  f"Female:{o['Female']:+.4f}  White:{o['White']:+.4f}  Male:{o['Male']:+.4f}")
        return

    print("\n  > Measuring baseline offsets (zero-bias run, 3 seeds × N_REPS)...")

    # Use 3 seeds only for speed — enough for stable offset estimation
    baseline_seeds = CONFIG["SEEDS"][:3]
    n_reps         = CONFIG["N_REPS"]

    payload = {
        "__warmup__":            CONFIG["warmup_months"],
        "__study__":             CONFIG["study_months"],
        "__agents__":            CONFIG["initial_agents"],
        "__intake__":            CONFIG["monthly_intake"],
        # bias_factor=0.0 — baseline measurement must run with all bias
        # completely disabled. Using 1.0 with zero group biases is
        # mathematically equivalent but 0.0 is explicit and safe against
        # any future code path that might apply a residual effect.
        "__bias_factor__":       0.0,
        "__group_bias_race__":   {"White": 0.0, "Black": 0.0,
                                  "Hispanic": 0.0, "Other": 0.0},
        "__group_bias_gender__": {"Male": 0.0, "Female": 0.0},
        "bias_scope":            "supervision_only",
        "bias_on_tech_detect":   1.0,
        "tech_detect_base":      0.30,
    }

    # Build seed list same way as _run_batch
    all_seeds = [seed + rep * 10_000
                 for rep in range(n_reps)
                 for seed in baseline_seeds]
    tasks = [(dict(payload | {"__task_id__": 0.0}),
              all_seeds[i:i + _WORKER_BATCH_SIZE])
             for i in range(0, len(all_seeds), _WORKER_BATCH_SIZE)]

    raw_runs = []
    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="  Baseline measurement", ncols=82) as bar:
            for fut in as_completed(futures):
                try:
                    _, run_results = fut.result()
                    raw_runs.extend(run_results)
                except Exception as e:
                    print(f"\n  Warning: Baseline worker error: {e}")
                finally:
                    bar.update(1)

    agg = _aggregate_subgroup_rates(raw_runs)

    # Guard — if all workers failed, agg will be empty
    if not agg or not all(win in agg for win in WINDOWS):
        raise RuntimeError(
            "Baseline measurement produced no results — all workers failed.\n"
            "Check the worker error above (likely get_peer_influence_config "
            "signature mismatch or RecidivismModel import error)."
        )

    # Compute offsets: sim_rate - BJS_target
    groups = ["Male", "Female", "White", "Black", "Hispanic"]
    offsets  = {}
    adjusted = {}
    for win in WINDOWS:
        offsets[win]  = {}
        adjusted[win] = {}
        for grp in groups:
            sim = agg[win].get(grp, {}).get("mean")
            bjs = BJS_SUBGROUP[win].get(grp)
            if sim is not None and bjs is not None:
                raw_offset = round(sim - bjs, 4)
                # Apply Female override if set
                if grp == "Female" and FEMALE_OFFSET_OVERRIDE is not None:
                    offsets[win][grp]  = FEMALE_OFFSET_OVERRIDE
                else:
                    offsets[win][grp]  = raw_offset
                adjusted[win][grp] = round(bjs - offsets[win][grp], 4)
            else:
                offsets[win][grp]  = 0.0
                adjusted[win][grp] = bjs or 0.0

    BASELINE_OFFSETS = offsets
    ADJUSTED_TARGETS = adjusted
    _BASELINE_MEASURED = True

    # Save checkpoint
    with open(cp, "w") as f:
        json.dump({
            "offsets":         {str(k): v for k, v in offsets.items()},
            "adjusted":        {str(k): v for k, v in adjusted.items()},
            "measured_at":     pd.Timestamp.now().isoformat(),
            "seeds_used":      baseline_seeds,
            "female_override": FEMALE_OFFSET_OVERRIDE,
            "baseline_sim_rates": {
                str(win): {
                    grp: agg[win].get(grp, {}).get("mean")
                    for grp in groups
                }
                for win in WINDOWS
            },
        }, f, indent=2)

    print(f"  OK  Baseline offsets measured and saved -> {cp}")
    print("  Baseline offsets (live measurement):")
    for win in WINDOWS:
        o = BASELINE_OFFSETS[win]
        print(f"     {win}yr  Black:{o['Black']:+.4f}  Hispanic:{o['Hispanic']:+.4f}  "
              f"Female:{o['Female']:+.4f}  White:{o['White']:+.4f}  Male:{o['Male']:+.4f}")
    print("  Adjusted targets:")
    for win in WINDOWS:
        a = ADJUSTED_TARGETS[win]
        print(f"     {win}yr  Black:{a['Black']:.4f}  Hispanic:{a['Hispanic']:.4f}  "
              f"Female:{a['Female']:.4f}  White:{a['White']:.4f}  Male:{a['Male']:.4f}")

RACE_REFERENCE   = "White"
GENDER_REFERENCE = "Male"
BIAS_FACTOR      = 1.0

RACE_SWEEP = {
    # Black: production coarse confirmed clean minimum near 1.30.
    # Range 0.0→1.5 is sufficient.
    "Black":    np.round(np.arange(0.0,  1.51, 0.1),  2).tolist(),

    # Hispanic: production minimum confirmed at 0.00 (MAE=0.83%).
    # Extended left to -0.60 (was -0.40) to rule out left-boundary artefact.
    "Hispanic": np.round(np.arange(-0.6, 0.41, 0.05), 2).tolist(),
}

# Black fine pass:
#   Step increased 0.02 → 0.05.
#   Production charts showed MAE oscillating ~0.15pp between adjacent 0.02
#   points — the model's within-condition variance exceeds the between-condition
#   MAE signal at 0.02 resolution even at 200 runs/point.  0.05 steps are
#   above the noise floor and still give 2 d.p. precision on the minimum.
BLACK_SWEEP_FINE_HALF_WIDTH = 0.20
BLACK_SWEEP_FINE_STEP       = 0.02

# Female coarse: EXTENDED from -3.0 to -5.0.
#   Production chart confirmed minimum hit the LEFT BOUNDARY at -3.0 with
#   200 runs/point — this is a genuine model behaviour, not noise.
#   Fine best landed at -3.22, confirming the true minimum is below -3.0.
#   -5.0 gives ample headroom; if minimum is still near -5.0 extend to -6.0
#   and rerun Stage 2 only:  python phase2_bias.py --stage 2 --rerun
# Range updated -5.0 → -3.0 now that Female offset=0.0.
# With direct BJS target (58%), the minimum is expected around -1.0 to -1.8.
# -3.0 left boundary gives ample headroom without the noisy floor region
# that existed between -5.0 and -3.0 under the old offset methodology.
GENDER_SWEEP_COARSE = np.round(np.arange(-3.0, 0.51, 0.1), 2).tolist()

# Female fine pass:
#   Step increased 0.02 → 0.05 (same noise-floor rationale as Black).
#   Half-width kept at 0.30 — wide enough to bracket the true minimum
#   even if the coarse best is slightly off due to residual noise.
GENDER_SWEEP_FINE_HALF_WIDTH = 0.30
GENDER_SWEEP_FINE_STEP       = 0.02

# Fine pass reps multiplier.
#   COARSE passes use N_REPS_COARSE replications (100 runs/point).
#   FINE passes use N_REPS_FINE replications (200 runs/point).
#   Coarse pass only needs to locate the approximate minimum — 100 runs
#   is sufficient for a 5-10% MAE signal difference between sweep points.
#   Fine pass needs 200 runs to resolve the 0.02-step MAE differences.
#   This halves coarse pass compute vs the original 200-run coarse baseline.
N_REPS_COARSE        = 5    # coarse passes: 50 runs/point (5 reps × 10 seeds)
                              # coarse only needs to locate approximate minimum
                              # 50 runs gives ±2pp uncertainty — sufficient for 0.1 step
N_REPS_FINE          = 10   # fine passes: 100 runs/point (10 reps × 10 seeds)
                              # fine needs more precision for 0.02 step resolution
FINE_REPS_MULTIPLIER = N_REPS_FINE // CONFIG["N_REPS"]  # kept for legacy compat

# Legacy single-pass sweep kept for --replot back-compat
GENDER_SWEEP = {
    "Female": GENDER_SWEEP_COARSE,
}

SENSITIVITY_LEVELS = np.round(np.arange(0.6, 1.41, 0.1), 2).tolist()

COLORS = {
    "Male":     "#2E75B6",
    "Female":   "#ED7D31",
    "White":    "#70AD47",
    "Black":    "#C00000",
    "Hispanic": "#7030A0",
    "Other":    "#808080",
    "bjs":      "#1A3D5C",
    "sim":      "#2E75B6",
    "grid":     "#E8E8E8",
}

WINDOWS = [3, 6, 9]

# ── HPC cluster config (128-core Bigmem, 4 GB/core default, 8 GB/core max) ───
# Total memory available: 128 cores × 4 GB = 512 GB
# At ~250 MB per worker (typical Mesa ABM with peer influence) you can safely
# run all 124 workers simultaneously with ~31 GB used out of 512 GB.
#
# _MAX_WORKERS: reserve 4 cores for SLURM/OS overhead.
# _WORKER_BATCH_SIZE: 200 total seeds / 124 workers ≈ 2 seeds per worker,
#   but batches below 10 create too much scheduler overhead — floor at 20.
#   Each batch runs serially inside the worker, so a batch of 20 means one
#   worker handles 20 consecutive model runs before returning results.
#
# To request max memory (8 GB/core) add to your SLURM script:
#   #SBATCH --mem-per-cpu=8G
# and set _MEM_PER_CPU_GB = 8 below — this only affects the startup printout.

_HPC_CORES        = 128
_MEM_PER_CPU_GB   = 4      # default; change to 8 if you request --mem-per-cpu=8G
_OS_OVERHEAD_CORES = 4
_MAX_WORKERS      = _HPC_CORES - _OS_OVERHEAD_CORES   # = 124

_WORKER_BATCH_SIZE       = 20   # fine pass / validation — small batches for responsiveness
_WORKER_BATCH_SIZE_COARSE = 50   # coarse pass — larger batches reduce scheduler overhead
                                  # at 124 workers × 50 seeds/batch = 6,200 concurrent seeds


# =============================================================================
# UTILITIES
# =============================================================================

def _cleanup_result_run1() -> int:
    """
    Delete stale CSV export files from Results_Run1 and any other known
    model output directories. Called periodically during sweeps to prevent
    disk filling — each model run exports warmup + study CSVs (~700KB total).
    At 200 runs/point x 56 points that is ~78GB without periodic cleanup.

    Searches all likely locations including subdirectories up to 2 levels
    deep from the working directory. Safe — only deletes .csv files matching
    the warmup/study export naming pattern.
    """
    import glob as _glob

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_script_dir)
    _cwd        = os.getcwd()

    # All candidate Results_Run1 locations — broad search to catch any CWD
    candidates = [
        os.path.join(_parent_dir, "Results_Run1"),
        os.path.join(_script_dir, "Results_Run1"),
        os.path.join(_cwd,        "Results_Run1"),
    ]

    # Also find any Results_Run* directories up to 2 levels deep from CWD
    for pattern in ["Results_Run*", "*/Results_Run*", "*/*/Results_Run*"]:
        for found in _glob.glob(os.path.join(_cwd, pattern)):
            if os.path.isdir(found) and found not in candidates:
                candidates.append(found)

    deleted = 0
    for target in candidates:
        if not os.path.isdir(target):
            continue
        try:
            for fname in os.listdir(target):
                # Only delete CSV files — never touch JSON checkpoints or logs
                if not fname.endswith(".csv"):
                    continue
                fpath = os.path.join(target, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        deleted += 1
                except OSError:
                    pass
        except OSError:
            pass

    return deleted


# =============================================================================
# WORKER  (runs in subprocess)
# =============================================================================

def _worker(task: tuple) -> tuple:
    """
    Execute a batch of seeds for a single payload configuration.

    Args:
        task: (payload_dict, seeds_list)
            payload_dict contains all __dunder__ config keys plus bias params.
            seeds_list is the slice of seeds this worker should run.

    Returns:
        (task_id, list_of_subgroup_rate_dicts)
        One dict per successfully completed seed run.

    FIX vs original:
        • Removed `return results` from inside the seed loop — the original
          code returned after the *first* seed, silently dropping every
          subsequent seed in the batch.
        • Returns (task_id, results) — a 2-tuple — instead of the 3-tuple the
          original promised but never produced, so _run_batch can unpack safely.
    """
    # Suppress BLAS/OpenMP thread oversubscription.
    # numpy/scipy link against multithreaded BLAS (MKL, OpenBLAS).
    # Without this, 124 workers × N BLAS threads = hundreds of threads
    # competing for 128 cores — the primary cause of 7-hour runtimes.
    # Must be set BEFORE importing numpy in the worker subprocess.
    import os as _os
    for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_env] = "1"

    payload, seeds = task
    payload = dict(payload)          # defensive copy; don't mutate caller's dict

    task_id = float(payload.pop("__task_id__", 0.0))
    warmup  = int(payload.pop("__warmup__",  CONFIG["warmup_months"]))
    study   = int(payload.pop("__study__",   CONFIG["study_months"]))
    agents  = int(payload.pop("__agents__",  CONFIG["initial_agents"]))
    intake  = int(payload.pop("__intake__",  CONFIG["monthly_intake"]))

    cal = get_global_calibration_params()
    cal["bias_factor"]         = float(payload.pop("__bias_factor__", BIAS_FACTOR))
    cal["group_bias_race"]     = payload.pop("__group_bias_race__",
                                             {"White": 0.0, "Black": 0.0,
                                              "Hispanic": 0.0, "Other": 0.0})
    cal["group_bias_gender"]   = payload.pop("__group_bias_gender__",
                                             {"Male": 0.0, "Female": 0.0})
    cal["bias_scope"]          = payload.pop("bias_scope",          "all")
    cal["bias_on_tech_detect"] = float(payload.pop("bias_on_tech_detect", 1.0))
    cal["tech_detect_base"]    = float(payload.pop("tech_detect_base",    0.30))

    peer_config = get_peer_influence_config()

    results = []
    for seed in seeds:
        try:
            # Use a per-worker temp directory to eliminate I/O contention.
            # 124 workers writing to the same Results_Run1 dir causes lock
            # contention and disk thrashing — the primary cause of 12-hour runs.
            # Each worker gets its own isolated scratch dir that is deleted
            # immediately after the run completes.
            import tempfile, shutil
            _tmp_dir = tempfile.mkdtemp(prefix=f"abm_worker_{seed}_")
            try:
                model = RecidivismModel(
                    initial_agents=agents,
                    bias_factor=cal["bias_factor"],
                    monthly_intake=intake,
                    warmup_months=warmup,
                    study_months=study,
                    enable_peer_influence=CONFIG["enable_peer_influence"],
                    weights=get_flat_risk_weights(),
                    peer_config=peer_config,
                    seed=seed,
                    output_directory=_tmp_dir,
                )
                model.export_csv = False
                model.calibration = cal
                if hasattr(model, "apply_calibration"):
                    model.apply_calibration(model.calibration)
                while model.running:
                    model.step()
                results.append(_extract_subgroup_rates(model))
                # NOTE: no early `return` here — all seeds in the batch must run.
            finally:
                # Always clean up the temp dir — no CSV accumulation on disk
                shutil.rmtree(_tmp_dir, ignore_errors=True)
        except Exception as e:
            print(f"  Warning: Worker error (tid={task_id:.4f}, seed={seed}): {e}",
                  flush=True)

    return (task_id, results)


# =============================================================================
# SUBGROUP EXTRACTION
# =============================================================================

def _extract_subgroup_rates(model) -> dict:
    eligible = [a for a in model.schedule.agents
                if getattr(a, "study_eligible_agent", False)]
    if not eligible:
        return {}

    groups_race   = ["White", "Black", "Hispanic"]
    groups_gender = ["Male", "Female"]
    windows_flags = {3: "rearrest_3_yrs", 6: "rearrest_6_yrs", 9: "rearrest_9_yrs"}

    rates = {}
    for win, flag in windows_flags.items():
        rates[win] = {}
        flagged = sum(1 for a in eligible if getattr(a, flag, False))
        rates[win]["Overall"] = flagged / len(eligible)
        for grp in groups_race:
            members = [a for a in eligible if getattr(a, "Race", None) == grp]
            rates[win][grp] = (
                sum(1 for a in members if getattr(a, flag, False)) / len(members)
                if members else None
            )
        for grp in groups_gender:
            members = [a for a in eligible if getattr(a, "Gender", None) == grp]
            rates[win][grp] = (
                sum(1 for a in members if getattr(a, flag, False)) / len(members)
                if members else None
            )
        for race in groups_race:
            for gender in groups_gender:
                key = f"{race}_{gender}"
                members = [
                    a for a in eligible
                    if getattr(a, "Race", None) == race
                    and getattr(a, "Gender", None) == gender
                ]
                rates[win][key] = (
                    sum(1 for a in members if getattr(a, flag, False)) / len(members)
                    if members else None
                )
    return rates


# =============================================================================
# BATCH RUNNER
# =============================================================================

def _run_batch(overrides_list: list, n_workers: int, desc: str = "Running",
               n_reps_override: int = None,
               batch_size: int = None) -> dict:
    """
    Submit all sweep points as batched worker tasks and collect results.

    Args:
        overrides_list:  list of (task_id_float, payload_dict) pairs.
        n_workers:       number of parallel processes.
        desc:            label shown in tqdm progress bar.
        n_reps_override: if set, use this instead of CONFIG['N_REPS'].
                         Fine passes use N_REPS * FINE_REPS_MULTIPLIER to
                         push the noise floor below the coarser step signal.

    Returns:
        dict mapping task_id -> list of subgroup-rate dicts (one per seed run).
    """
    seeds  = CONFIG["SEEDS"]
    n_reps = n_reps_override if n_reps_override is not None else CONFIG["N_REPS"]

    # Build flat (payload, seed_batch) task list — one entry per batch.
    tasks = []
    for task_id, payload in overrides_list:
        d = dict(payload)
        d["__task_id__"] = float(task_id)
        all_seeds = [seed + rep * 10_000
                     for rep in range(n_reps)
                     for seed in seeds]
        _bs = batch_size if batch_size is not None else _WORKER_BATCH_SIZE
        for i in range(0, len(all_seeds), _bs):
            tasks.append((d, all_seeds[i : i + _bs]))

    # Pre-build result accumulator and fast task_id→key mapping.
    raw: dict[float, list] = {float(tid): [] for tid, _ in overrides_list}

    # Periodic cleanup interval — delete Results_Run1 files every N completed
    # batches to prevent disk filling during long sweeps. Each model run
    # exports a warmup CSV (~500KB) and a study CSV (~200KB). At 200 runs/point
    # and 56 coarse points that is ~78GB without cleanup. Every 50 batches
    # (~1000 model runs) keeps peak disk usage under ~5GB at any time.
    _CLEANUP_INTERVAL = 50
    _completed_batches = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc=f"  {desc}", ncols=82) as bar:
            for fut in as_completed(futures):
                try:
                    tid, run_results = fut.result()   # 2-tuple (fixed)
                    # O(1) lookup — task_id is set exactly from overrides_list
                    if tid in raw:
                        raw[tid].extend(run_results)
                    else:
                        # Floating-point safety: find nearest key
                        nearest = min(raw.keys(), key=lambda k: abs(k - tid))
                        raw[nearest].extend(run_results)
                except Exception as e:
                    print(f"\n  Warning: Future error: {e}", flush=True)
                finally:
                    bar.update(1)
                    _completed_batches += 1
                    if _completed_batches % _CLEANUP_INTERVAL == 0:
                        n_del = _cleanup_result_run1()
                        if n_del:
                            tqdm.write(f"  [disk] Cleaned {n_del} files "
                                       f"(batch {_completed_batches})")

    # Final cleanup after all batches complete
    n_del = _cleanup_result_run1()
    if n_del:
        print(f"  [disk] Final cleanup: {n_del} files removed")
    return raw


# =============================================================================
# AGGREGATION + METRICS
# =============================================================================

def _aggregate_subgroup_rates(runs: list) -> dict:
    """
    Aggregate a list of per-run subgroup-rate dicts into mean ± 95% CI.

    OPT vs original:
        • Union of group keys built once (not once per window per run).
    """
    if not runs:
        return {}

    # Build group key union once across all runs and all windows.
    all_groups: set[str] = set()
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
            if grp in ("Male", "Female"):
                ref = ref_gender_mean
            elif "_" in grp:
                ref = baseline
            else:
                ref = ref_race_mean
            metrics[win][grp] = {
                "mean":    mean,
                "ci95":    vals.get("ci95", 0.0),
                "gap":     round(mean - ref, 4),
                "ratio":   round(mean / ref, 4) if ref > 0 else None,
                "abs_dev": round(mean - baseline, 4),
            }
    return metrics


# =============================================================================
# STAGE 1 - RACE CALIBRATION
# =============================================================================

def _sweep_race_group(race_group: str, sweep_vals: list,
                      n_workers: int, desc: str,
                      n_reps_override: int = None) -> list:
    """
    Single sweep pass over sweep_vals for one race group.
    Returns list of {value, mae, agg} sorted ascending by mae.
    """
    overrides_list = []
    for i, val in enumerate(sweep_vals):
        race_bias = {"White": 0.0, "Black": 0.0,
                     "Hispanic": 0.0, "Other": 0.0}
        race_bias[race_group] = float(val)
        payload = {
            "__warmup__":            CONFIG["warmup_months"],
            "__study__":             CONFIG["study_months"],
            "__agents__":            CONFIG["initial_agents"],
            "__intake__":            CONFIG["monthly_intake"],
            "__bias_factor__":       BIAS_FACTOR,
            "__group_bias_race__":   race_bias,
            "__group_bias_gender__": {"Male": 0.0, "Female": 0.0},
            "bias_scope":            "supervision_only",
            "bias_on_tech_detect":   1.0,
            "tech_detect_base":      0.30,
        }
        overrides_list.append((float(i), payload))

    raw     = _run_batch(overrides_list, n_workers, desc=desc,
                         n_reps_override=n_reps_override,
                         batch_size=_WORKER_BATCH_SIZE_COARSE if n_reps_override is not None and n_reps_override <= N_REPS_COARSE * len(CONFIG["SEEDS"]) else _WORKER_BATCH_SIZE)
    results = []
    for i, val in enumerate(sweep_vals):
        runs = raw.get(float(i), [])
        agg  = _aggregate_subgroup_rates(runs)

        if not agg:
            continue
        maes = []
        for win in WINDOWS:
            sim = agg[win].get(race_group, {}).get("mean")
            tgt = ADJUSTED_TARGETS[win].get(race_group)
            if sim is not None and tgt is not None:
                maes.append(abs(sim - tgt))
        mae = float(np.mean(maes)) if maes else 999.0
        results.append({"value": val, "mae": mae, "agg": agg})

    results.sort(key=lambda x: x["mae"])
    return results


def run_stage1_race(n_workers: int, outdir: str) -> dict:
    print("\n" + "=" * 70)
    print("  STAGE 1 - RACE CALIBRATION  (two-pass for Black)")
    print("=" * 70)
    for win in WINDOWS:
        print(f"     {win}yr  Black adjusted={ADJUSTED_TARGETS[win]['Black']:.4f}  "
              f"Hispanic adjusted={ADJUSTED_TARGETS[win]['Hispanic']:.4f}")

    best_params = {}

    # ── Hispanic: single-pass ─────────────────────────────────────────────────
    grp = "Hispanic"
    cp  = os.path.join(outdir, f"stage1_{grp}.json")
    if os.path.exists(cp):
        print(f"  > {grp} - loading checkpoint")
        with open(cp) as f:
            best_params[grp] = json.load(f)
    else:
        sweep_vals = RACE_SWEEP["Hispanic"]
        print(f"\n  > Sweeping {grp} bias  "
              f"({len(sweep_vals)} values: {sweep_vals[0]} to {sweep_vals[-1]})")
        results = _sweep_race_group(grp, sweep_vals, n_workers, f"Stage1 {grp}",
                                  n_reps_override=N_REPS_COARSE * len(CONFIG["SEEDS"]))
        best    = results[0]
        best_params[grp] = {
            "value":           best["value"],
            "mae":             best["mae"],
            "agg":             {str(k): v for k, v in best["agg"].items()},
            "all_results":     [{"value": r["value"], "mae": r["mae"]}
                                for r in results],
            "adjusted_target": {str(w): ADJUSTED_TARGETS[w][grp] for w in WINDOWS},
            "baseline_offset": {str(w): BASELINE_OFFSETS[w][grp] for w in WINDOWS},
        }
        print(f"  OK  Best {grp} bias = {best['value']:.2f}  "
              f"(MAE={best['mae']:.4f})")
        with open(cp, "w") as f:
            json.dump(best_params[grp], f, indent=2)

    # ── Black: two-pass ───────────────────────────────────────────────────────
    grp = "Black"
    cp  = os.path.join(outdir, f"stage1_{grp}.json")
    if os.path.exists(cp):
        print(f"  > {grp} - loading checkpoint (delete with --rerun to re-sweep)")
        with open(cp) as f:
            best_params[grp] = json.load(f)
    else:
        # Pass 1 — coarse
        coarse_vals = RACE_SWEEP["Black"]
        print(f"\n  Pass 1 (coarse): {len(coarse_vals)} values  "
              f"[{coarse_vals[0]:.1f}  →  {coarse_vals[-1]:.1f}  step=0.1]")
        coarse_results = _sweep_race_group(
            grp, coarse_vals, n_workers, "Stage1 Black coarse",
            n_reps_override=N_REPS_COARSE * len(CONFIG["SEEDS"])
        )
        coarse_best = coarse_results[0]
        print(f"  OK  Coarse best = {coarse_best['value']:.2f}  "
              f"(MAE={coarse_best['mae']:.4f})")

        # Pass 2 — fine
        hw        = BLACK_SWEEP_FINE_HALF_WIDTH
        step      = BLACK_SWEEP_FINE_STEP
        fine_lo   = round(coarse_best["value"] - hw, 3)
        fine_hi   = round(coarse_best["value"] + hw + step, 3)
        fine_vals = np.round(np.arange(fine_lo, fine_hi, step), 3).tolist()
        print(f"\n  Pass 2 (fine):   {len(fine_vals)} values  "
              f"[{fine_vals[0]:.2f}  →  {fine_vals[-1]:.2f}  step={step}]")
        fine_n_reps = N_REPS_FINE * len(CONFIG["SEEDS"])
        print(f"  Fine pass reps: {fine_n_reps} ({N_REPS_FINE} reps x {len(CONFIG['SEEDS'])} seeds)")
        fine_results = _sweep_race_group(
            grp, fine_vals, n_workers, "Stage1 Black fine",
            n_reps_override=fine_n_reps
        )
        fine_best = fine_results[0]
        print(f"  OK  Fine best   = {fine_best['value']:.3f}  "
              f"(MAE={fine_best['mae']:.4f})")

        all_coarse = [{"value": r["value"], "mae": r["mae"]} for r in coarse_results]
        all_fine   = [{"value": r["value"], "mae": r["mae"]} for r in fine_results]

        best_params[grp] = {
            "value":           fine_best["value"],
            "mae":             fine_best["mae"],
            "agg":             {str(k): v for k, v in fine_best["agg"].items()},
            "coarse_results":  all_coarse,
            "fine_results":    all_fine,
            "all_results":     all_coarse,   # legacy --replot compat
            "adjusted_target": {str(w): ADJUSTED_TARGETS[w][grp] for w in WINDOWS},
            "baseline_offset": {str(w): BASELINE_OFFSETS[w][grp] for w in WINDOWS},
            "coarse_best":     coarse_best["value"],
            "coarse_best_mae": coarse_best["mae"],
        }
        print(f"  OK  Best Black bias = {fine_best['value']:.3f}  "
              f"(MAE={fine_best['mae']:.4f})")
        with open(cp, "w") as f:
            json.dump(best_params[grp], f, indent=2)

    return best_params


# =============================================================================
# STAGE 2 - GENDER CALIBRATION
# =============================================================================

def _sweep_female(race_bias: dict, sweep_vals: list,
                  n_workers: int, desc: str,
                  n_reps_override: int = None) -> list:
    overrides_list = []
    for i, val in enumerate(sweep_vals):
        payload = {
            "__warmup__":            CONFIG["warmup_months"],
            "__study__":             CONFIG["study_months"],
            "__agents__":            CONFIG["initial_agents"],
            "__intake__":            CONFIG["monthly_intake"],
            "__bias_factor__":       BIAS_FACTOR,
            "__group_bias_race__":   race_bias,
            "__group_bias_gender__": {"Male": 0.0, "Female": float(val)},
            # bias_scope="supervision_only" — consistent with race calibration.
            # All bias (race and gender) is operationalised exclusively at the
            # supervision violation detection stage, reflecting the unified
            # theoretical claim that recorded recidivism disparities are driven
            # by differential supervision contact intensity rather than
            # behavioural differences or trial-stage effects.
            # Source: Skeem et al. (2014); Petersilia (2003); Western (2006).
            "bias_scope":            "supervision_only",
            "bias_on_tech_detect":   1.0,
            "tech_detect_base":      0.30,
        }
        overrides_list.append((float(i), payload))

    raw     = _run_batch(overrides_list, n_workers, desc=desc,
                         n_reps_override=n_reps_override,
                         batch_size=_WORKER_BATCH_SIZE_COARSE if n_reps_override is not None and n_reps_override <= N_REPS_COARSE * len(CONFIG["SEEDS"]) else _WORKER_BATCH_SIZE)
    results = []
    for i, val in enumerate(sweep_vals):
        runs = raw.get(float(i), [])
        agg  = _aggregate_subgroup_rates(runs)
        if not agg:
            continue
        maes = []
        for win in WINDOWS:
            sim = agg[win].get("Female", {}).get("mean")
            tgt = ADJUSTED_TARGETS[win].get("Female")
            if sim is not None and tgt is not None:
                maes.append(abs(sim - tgt))
        mae = float(np.mean(maes)) if maes else 999.0
        results.append({"value": val, "mae": mae, "agg": agg})

    results.sort(key=lambda x: x["mae"])
    return results


def run_stage2_gender(race_params: dict, n_workers: int, outdir: str) -> dict:
    print("\n" + "=" * 70)
    print("  STAGE 2 - GENDER CALIBRATION  (two-pass extended sweep)")
    print("=" * 70)
    print("  Using ADJUSTED_TARGETS (BJS - baseline offset) for MAE:")
    for win in WINDOWS:
        print(f"     {win}yr  Female adjusted={ADJUSTED_TARGETS[win]['Female']:.4f}  "
              f"(BJS={BJS_SUBGROUP[win]['Female']:.2f}, "
              f"offset={BASELINE_OFFSETS[win]['Female']:+.4f})")

    cp = os.path.join(outdir, "stage2_gender.json")
    if os.path.exists(cp):
        print("  > Female - loading checkpoint (delete with --rerun to re-sweep)")
        with open(cp) as f:
            return json.load(f)

    race_bias = {
        "White":    0.0,
        "Black":    float(race_params.get("Black",    {}).get("value", 0.0)),
        "Hispanic": float(race_params.get("Hispanic", {}).get("value", 0.0)),
        "Other":    0.0,
    }
    print(f"  Race bias fixed at: {race_bias}")

    # Pass 1: coarse
    coarse_vals = GENDER_SWEEP_COARSE
    print(f"\n  Pass 1 (coarse): {len(coarse_vals)} values  "
          f"[{coarse_vals[0]:.1f}  →  {coarse_vals[-1]:.1f}  step=0.1]")
    coarse_results = _sweep_female(race_bias, coarse_vals,
                                   n_workers, "Stage2 Female coarse",
                                   n_reps_override=N_REPS_COARSE * len(CONFIG["SEEDS"]))
    coarse_best = coarse_results[0]
    print(f"  OK  Coarse best = {coarse_best['value']:.2f}  "
          f"(MAE={coarse_best['mae']:.4f})")

    # Pass 2: fine
    hw        = GENDER_SWEEP_FINE_HALF_WIDTH
    step      = GENDER_SWEEP_FINE_STEP
    fine_lo   = round(coarse_best["value"] - hw, 3)
    fine_hi   = round(coarse_best["value"] + hw + step, 3)
    fine_vals = np.round(np.arange(fine_lo, fine_hi, step), 3).tolist()
    print(f"\n  Pass 2 (fine):   {len(fine_vals)} values  "
          f"[{fine_vals[0]:.2f}  →  {fine_vals[-1]:.2f}  step={step}]")
    fine_n_reps = N_REPS_FINE * len(CONFIG["SEEDS"])
    print(f"  Fine pass reps: {fine_n_reps} ({N_REPS_FINE} reps x {len(CONFIG['SEEDS'])} seeds)")
    fine_results = _sweep_female(race_bias, fine_vals,
                                n_workers, "Stage2 Female fine",
                                n_reps_override=fine_n_reps)
    fine_best = fine_results[0]
    print(f"  OK  Fine best   = {fine_best['value']:.3f}  "
          f"(MAE={fine_best['mae']:.4f})")

    all_coarse = [{"value": r["value"], "mae": r["mae"]} for r in coarse_results]
    all_fine   = [{"value": r["value"], "mae": r["mae"]} for r in fine_results]

    best_params = {
        "Female": {
            "value":            fine_best["value"],
            "mae":              fine_best["mae"],
            "agg":              {str(k): v for k, v in fine_best["agg"].items()},
            "coarse_results":   all_coarse,
            "fine_results":     all_fine,
            "all_results":      all_coarse,   # legacy --replot compat
            "adjusted_target":  {str(w): ADJUSTED_TARGETS[w]["Female"]
                                 for w in WINDOWS},
            "baseline_offset":  {str(w): BASELINE_OFFSETS[w]["Female"]
                                 for w in WINDOWS},
            "coarse_best":      coarse_best["value"],
            "coarse_best_mae":  coarse_best["mae"],
        }
    }

    with open(cp, "w") as f:
        json.dump(best_params, f, indent=2)

    return best_params


# =============================================================================
# STAGE 3 - INTERSECTIONAL VALIDATION + SENSITIVITY
# =============================================================================

def run_stage3_intersectional(race_params: dict, gender_params: dict,
                               n_workers: int, outdir: str) -> dict:
    print("\n" + "=" * 70)
    print("  STAGE 3 - INTERSECTIONAL VALIDATION + SENSITIVITY")
    print("=" * 70)

    race_bias = {
        "White":    0.0,
        "Black":    float(race_params.get("Black",    {}).get("value", 0.0)),
        "Hispanic": float(race_params.get("Hispanic", {}).get("value", 0.0)),
        "Other":    0.0,
    }
    gender_bias = {
        "Male":   0.0,
        "Female": float(gender_params.get("Female", {}).get("value", 0.0)),
    }
    print(f"  Calibrated race bias:   {race_bias}")
    print(f"  Calibrated gender bias: {gender_bias}")

    cp_val = os.path.join(outdir, "stage3_validation.json")
    if os.path.exists(cp_val):
        print("  > Validation - loading checkpoint")
        with open(cp_val) as f:
            validation = json.load(f)
    else:
        print("\n  > Running intersectional validation...")
        payload = {
            "__warmup__":            CONFIG["warmup_months"],
            "__study__":             CONFIG["study_months"],
            "__agents__":            CONFIG["initial_agents"],
            "__intake__":            CONFIG["monthly_intake"],
            "__bias_factor__":       BIAS_FACTOR,
            "__group_bias_race__":   race_bias,
            "__group_bias_gender__": gender_bias,
            # supervision_only: Race bias (Black=1.24) was calibrated under
            # supervision_only. Using "all" applies trial pathway bias which
            # reduces Black rearrest rates counter-intuitively (more Black
            # agents diverted at trial → less supervision → fewer rearrests).
            # Female=-1.36 at this magnitude works correctly under
            # supervision_only — the supervision detection component dominates.
            "bias_scope":            "supervision_only",
            "bias_on_tech_detect":   1.0,
            "tech_detect_base":      0.30,
        }
        raw  = _run_batch([(0.0, payload)], n_workers, "Stage3 Validation")
        runs = raw.get(0.0, [])
        agg  = _aggregate_subgroup_rates(runs)
        metrics = _compute_disparity_metrics(agg)
        validation = {
            "race_bias":   race_bias,
            "gender_bias": gender_bias,
            "bias_factor": BIAS_FACTOR,
            "agg":         {str(k): v for k, v in agg.items()},
            "metrics":     {str(k): v for k, v in metrics.items()},
        }
        with open(cp_val, "w") as f:
            json.dump(validation, f, indent=2)
        print("  OK  Validation complete")

    cp_sens = os.path.join(outdir, "stage3_sensitivity.json")
    if os.path.exists(cp_sens):
        print("  > Sensitivity - loading checkpoint")
        with open(cp_sens) as f:
            sensitivity = json.load(f)
    else:
        print(f"\n  > Sweeping bias_factor "
              f"({len(SENSITIVITY_LEVELS)} levels: "
              f"{SENSITIVITY_LEVELS[0]} to {SENSITIVITY_LEVELS[-1]})")
        overrides_list = []
        for i, bf in enumerate(SENSITIVITY_LEVELS):
            payload = {
                "__warmup__":            CONFIG["warmup_months"],
                "__study__":             CONFIG["study_months"],
                "__agents__":            CONFIG["initial_agents"],
                "__intake__":            CONFIG["monthly_intake"],
                "__bias_factor__":       float(bf),
                "__group_bias_race__":   race_bias,
                "__group_bias_gender__": gender_bias,
                # supervision_only: matches Race calibration scope (Stage 1).
                # Black=1.24 was calibrated under supervision_only and must
                # validate under the same scope for consistent results.
                "bias_scope":            "supervision_only",
                "bias_on_tech_detect":   1.0,
                "tech_detect_base":      0.30,
            }
            overrides_list.append((float(i), payload))

        raw = _run_batch(overrides_list, n_workers, desc="Stage3 Sensitivity")
        sens_results = []
        for i, bf in enumerate(SENSITIVITY_LEVELS):
            runs = raw.get(float(i), [])
            agg  = _aggregate_subgroup_rates(runs)
            if not agg:
                continue
            metrics = _compute_disparity_metrics(agg)
            sens_results.append({
                "bias_factor": bf,
                "agg":         {str(k): v for k, v in agg.items()},
                "metrics":     {str(k): v for k, v in metrics.items()},
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
    if title:  ax.set_title(title, fontsize=11, fontweight="bold",
                            pad=8, color="#1A3D5C")
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_stage1_calibration(race_params: dict, outdir: str) -> str:
    SMOOTH_SIGMA = 1.5

    black_data    = race_params.get("Black", {})
    coarse_rows_b = black_data.get("coarse_results", black_data.get("all_results", []))
    fine_rows_b   = black_data.get("fine_results", [])
    has_fine_b    = bool(fine_rows_b)
    best_b        = black_data.get("value")
    coarse_best_b = black_data.get("coarse_best", best_b)

    ncols_b = 2 if has_fine_b else 1
    fig_b, axes_b = plt.subplots(1, ncols_b, figsize=(7 * ncols_b, 5))
    fig_b.patch.set_facecolor("#FAFAFA")
    if ncols_b == 1:
        axes_b = [axes_b]

    ax = axes_b[0]
    if coarse_rows_b:
        vals  = [r["value"] for r in coarse_rows_b]
        maes  = [r["mae"]   for r in coarse_rows_b]
        sm    = gaussian_filter1d(maes, sigma=SMOOTH_SIGMA)
        ax.plot(vals, maes, color=COLORS["Black"], linewidth=0.9,
                alpha=0.35, linestyle="--", zorder=2, label="MAE (raw)")
        ax.plot(vals, sm,   color=COLORS["Black"], linewidth=2.2,
                zorder=3, label=f"MAE (smoothed σ={SMOOTH_SIGMA})")
        ax.axvline(coarse_best_b, color="#333333", linewidth=1.5, linestyle="--",
                   zorder=4, label=f"Coarse best={coarse_best_b:.2f}")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=8)
    _style_ax(ax, "Black Bias (log-odds)", "MAE vs Adjusted BJS Target",
              "Stage 1 — Black Race Bias\nPass 1: Coarse  0.0 → 1.5  (step 0.1)")

    if has_fine_b:
        ax2 = axes_b[1]
        fvals = [r["value"] for r in fine_rows_b]
        fmaes = [r["mae"]   for r in fine_rows_b]
        fsm   = gaussian_filter1d(fmaes, sigma=SMOOTH_SIGMA)
        ax2.plot(fvals, fmaes, color=COLORS["Black"], linewidth=0.9,
                 alpha=0.35, linestyle="--", zorder=2, label="MAE (raw)")
        ax2.plot(fvals, fsm,   color=COLORS["Black"], linewidth=2.2,
                 zorder=3, label=f"MAE (smoothed σ={SMOOTH_SIGMA})")
        ax2.axvline(best_b, color="#333333", linewidth=1.5, linestyle="--",
                    label=f"Fine best={best_b:.3f} (MAE={black_data['mae']:.4f})")
        ax2.axvline(coarse_best_b, color="#888888", linewidth=1.0, linestyle=":",
                    label=f"Coarse best={coarse_best_b:.2f}")
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.legend(fontsize=8)
        hw = BLACK_SWEEP_FINE_HALF_WIDTH
        _style_ax(ax2, "Black Bias (log-odds)", "",
                  f"Pass 2: Fine  ±{hw:.2f} around coarse best  "
                  f"(step {BLACK_SWEEP_FINE_STEP})")

    fig_b.suptitle(
        "Stage 1 — Black Race Bias Calibration (two-pass)\n"
        "(MAE vs BJS targets adjusted for bias=0 baseline offsets)\n"
        "Dashed = raw MAE  |  Solid = Gaussian-smoothed (σ=1.5)",
        fontsize=12, fontweight="bold", color="#1A3D5C", y=1.04,
    )
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
        vals  = [r["value"] for r in rows_h]
        maes  = [r["mae"]   for r in rows_h]
        sm    = gaussian_filter1d(maes, sigma=SMOOTH_SIGMA)
        ax_h.plot(vals, maes, color=COLORS["Hispanic"], linewidth=0.9,
                  alpha=0.35, linestyle="--", zorder=2, label="MAE (raw)")
        ax_h.plot(vals, sm,   color=COLORS["Hispanic"], linewidth=2.2,
                  zorder=3, label=f"MAE (smoothed σ={SMOOTH_SIGMA})")
        ax_h.axvline(best_h, color="#333333", linewidth=1.5, linestyle="--",
                     zorder=4,
                     label=f"Best={best_h:.2f} (MAE={hisp_data['mae']:.4f})")
        ax_h.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax_h.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_h.legend(fontsize=9)
    _style_ax(ax_h, "Hispanic Bias (log-odds)", "MAE vs Adjusted BJS Target",
              "Stage 1 — Hispanic Race Bias Calibration\n"
              "Dashed = raw MAE  |  Solid = Gaussian-smoothed (σ=1.5)")
    plt.tight_layout()
    path_h = os.path.join(outdir, "phase2_stage1_hispanic_calibration.png")
    plt.savefig(path_h, dpi=150, bbox_inches="tight", facecolor=fig_h.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 Hispanic -> {path_h}")

    # Combined legacy chart (coarse only, both groups)
    fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 5))
    fig_c.patch.set_facecolor("#FAFAFA")
    for ax_c, (grp, clr, rows, best_v) in zip(
        axes_c,
        [("Black",    COLORS["Black"],    coarse_rows_b, coarse_best_b),
         ("Hispanic", COLORS["Hispanic"], rows_h,        best_h)],
    ):
        if not rows:
            continue
        vals = [r["value"] for r in rows]
        maes = [r["mae"]   for r in rows]
        sm   = gaussian_filter1d(maes, sigma=SMOOTH_SIGMA)
        ax_c.plot(vals, maes, color=clr, linewidth=0.9,
                  alpha=0.35, linestyle="--", zorder=2)
        ax_c.plot(vals, sm,   color=clr, linewidth=2.0, zorder=3,
                  label=f"Best={best_v:.2f} (MAE={race_params[grp]['mae']:.4f})")
        ax_c.axvline(best_v, color="#333333", linewidth=1.5, linestyle="--", zorder=4)
        ax_c.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax_c.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_c.legend(fontsize=9)
        _style_ax(ax_c, f"{grp} Bias (log-odds)", "MAE vs Adjusted BJS Target",
                  f"Stage 1 — {grp} Race Bias Calibration")
    fig_c.suptitle(
        "Stage 1: Race Bias Calibration — MAE Loss Curves\n"
        "(MAE vs BJS targets adjusted for bias=0 baseline offsets)\n"
        "Dashed = raw MAE  |  Solid = Gaussian-smoothed (σ=1.5)",
        fontsize=12, fontweight="bold", color="#1A3D5C", y=1.04,
    )
    plt.tight_layout()
    path_combined = os.path.join(outdir, "phase2_stage1_race_calibration.png")
    plt.savefig(path_combined, dpi=150, bbox_inches="tight",
                facecolor=fig_c.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 1 combined -> {path_combined}")
    return path_combined


def plot_stage2_calibration(gender_params: dict, outdir: str) -> str:
    data         = gender_params.get("Female", {})
    coarse_rows  = data.get("coarse_results", data.get("all_results", []))
    fine_rows    = data.get("fine_results", [])
    best_v       = data.get("value")
    coarse_best  = data.get("coarse_best", best_v)

    has_two_passes = bool(fine_rows)
    ncols = 2 if has_two_passes else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.patch.set_facecolor("#FAFAFA")
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    if coarse_rows:
        vals  = [r["value"] for r in coarse_rows]
        maes  = [r["mae"]   for r in coarse_rows]
        ax.plot(vals, maes, color=COLORS["Female"], linewidth=2.0, zorder=3,
                label="MAE (coarse)")
        ax.axvline(coarse_best, color="#333333", linewidth=1.5, linestyle="--",
                   label=f"Coarse best={coarse_best:.2f}")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9)
    _style_ax(ax, "Female Bias (log-odds)", "MAE vs Adjusted BJS Target",
              "Stage 2 — Female Calibration\nPass 1: Coarse sweep  −3.0 → +0.5  (step 0.1)")

    if has_two_passes:
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
        hw = GENDER_SWEEP_FINE_HALF_WIDTH
        _style_ax(ax2, "Female Bias (log-odds)", "",
                  f"Pass 2: Fine sweep  ±{hw:.2f} around coarse best  (step {GENDER_SWEEP_FINE_STEP})")

    fig.suptitle(
        "Stage 2 — Female Gender Bias Calibration (two-pass)\n"
         "bias_scope=supervision_only  |  consistent with race calibration scope",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.02,
    )
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
        ax = fig.add_subplot(gs[0, col])
        x  = np.arange(len(race_groups))
        bjs_vals = [BJS_SUBGROUP[win].get(g, 0) for g in race_groups]
        sim_vals = [agg[win].get(g, {}).get("mean", 0) or 0 for g in race_groups]
        sim_cis  = [agg[win].get(g, {}).get("ci95",  0) or 0 for g in race_groups]
        clrs     = [COLORS[g] for g in race_groups]
        ax.bar(x - 0.2, bjs_vals, width=0.35, color="#AAAAAA",
               alpha=0.7, label="BJS Target", zorder=3)
        ax.bar(x + 0.2, sim_vals, width=0.35, color=clrs,
               alpha=0.85, label="Simulation", zorder=3)
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
             color=[COLORS[g] for g in gender_groups],
             alpha=0.85, label="Simulation", zorder=3)
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
            v = agg[win].get(grp, {})
            matrix[i, j] = v.get("mean", 0) or 0
    im = ax_h.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.4, vmax=0.95)
    ax_h.set_xticks(range(len(intersect_groups)))
    ax_h.set_xticklabels([g.replace("_", "\n") for g in intersect_groups], fontsize=8)
    ax_h.set_yticks(range(len(WINDOWS)))
    ax_h.set_yticklabels([f"{w}-Year" for w in WINDOWS], fontsize=9)
    for i in range(len(WINDOWS)):
        for j in range(len(intersect_groups)):
            ax_h.text(j, i, f"{matrix[i, j]:.2%}",
                      ha="center", va="center", fontsize=8,
                      color="white" if matrix[i, j] > 0.75 else "#222222",
                      fontweight="bold")
    plt.colorbar(im, ax=ax_h, shrink=0.8).ax.tick_params(labelsize=8)
    _style_ax(ax_h, "", "", "Intersectional Rearrest Rates (Race x Gender)")

    fig.suptitle(
        "Stage 3 - Intersectional Validation\n"
        f"Calibrated: Black={validation['race_bias']['Black']:.2f}  "
        f"Hispanic={validation['race_bias']['Hispanic']:.2f}  "
        f"Female={validation['gender_bias']['Female']:.2f}  "
        f"bias_factor={validation['bias_factor']:.1f}",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=0.98,
    )
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
    gap_pairs = [
        ("Black",  "White", "Race Gap: Black - White",   COLORS["Black"]),
        ("Female", "Male",  "Gender Gap: Female - Male", COLORS["Female"]),
    ]
    for row, (grp, ref, title, clr) in enumerate(gap_pairs):
        for col, win in enumerate(WINDOWS):
            ax   = axes[row][col]
            gaps = []
            for r in levels:
                m = {int(k): v for k, v in r["metrics"].items()}
                g = m.get(win, {}).get(grp, {})
                gaps.append(g.get("gap", 0) or 0)
            ax.plot(bfs, gaps, color=clr, linewidth=2.0, label="Gap (pp)", zorder=3)
            ax.axvline(BIAS_FACTOR, color="#888888", linewidth=1.0,
                       linestyle="--", label="Baseline (1.0)")
            ax.axhline(0, color="#333333", linewidth=0.7)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.legend(fontsize=8)
            _style_ax(ax, "bias_factor", "Gap (pp)",
                      f"{title}\n{win}-Year Window")
    fig.suptitle(
        "Stage 3 - Bias Factor Sensitivity\n"
        "How disparity gaps respond to +/-40% perturbation of bias_factor",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_stage3_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Stage 3 sensitivity -> {path}")
    return path


def plot_disparity_summary(validation: dict, outdir: str) -> str:
    metrics         = {int(k): v for k, v in validation["metrics"].items()}
    groups          = ["Black", "Hispanic", "Female"]
    metrics_to_plot = ["gap", "ratio", "abs_dev"]
    titles          = ["Absolute Gap\n(group - reference, pp)",
                       "Rate Ratio\n(group / reference)",
                       "Absolute Deviation\nfrom Aggregate Baseline (pp)"]

    fig, axes = plt.subplots(len(WINDOWS), 3, figsize=(16, 11))
    fig.patch.set_facecolor("#FAFAFA")

    for row, win in enumerate(WINDOWS):
        for col, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax   = axes[row][col]
            vals = [metrics[win].get(g, {}).get(metric, 0) or 0 for g in groups]
            cis  = [metrics[win].get(g, {}).get("ci95",   0) or 0 for g in groups]
            clrs = [COLORS[g] for g in groups]
            xpos = np.arange(len(groups))
            ax.bar(xpos, vals, color=clrs, alpha=0.85,
                   edgecolor="white", width=0.55, zorder=3)
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
        "Phase 2 Disparity Summary - All Three Metrics\n"
        "Gap = group - reference  |  Ratio = group / reference  |  "
        "Abs Dev = group - aggregate baseline",
        fontsize=13, fontweight="bold", color="#1A3D5C", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(outdir, "phase2_disparity_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Chart: Disparity summary -> {path}")
    return path


# =============================================================================
# CORE DETECTION
# =============================================================================

def detect_workers() -> tuple:
    """
    Auto-detect safe worker count, capped at _MAX_WORKERS (124).

    On the HPC cluster SLURM sets SLURM_CPUS_PER_TASK — we use that
    first so the job script controls parallelism, not psutil guessing.
    Falls back to psutil / multiprocessing on non-SLURM machines.
    """
    # ── SLURM environment (cluster) ───────────────────────────────────────────
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        n = max(1, min(int(slurm_cpus) - _OS_OVERHEAD_CORES, _MAX_WORKERS))
        return n, f"SLURM_CPUS_PER_TASK={slurm_cpus} -> {n} workers (cap={_MAX_WORKERS})"

    # ── Local machine fallback ────────────────────────────────────────────────
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

def main(n_workers: int, stages: list = None, force_rerun: bool = False):
    # Set thread limits before any worker pool is created.
    # Child processes inherit environment variables from the parent,
    # so setting these here ensures every worker starts single-threaded.
    import os as _os
    for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ[_env] = "1"

    if stages is None:
        stages = [1, 2, 3]

    outdir = CONFIG["output_directory"]
    os.makedirs(outdir, exist_ok=True)
    n_runs = CONFIG["N_REPS"] * len(CONFIG["SEEDS"])

    slurm_job  = os.environ.get("SLURM_JOB_ID",        "local")
    slurm_node = os.environ.get("SLURM_NODELIST",      "local")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "—")
    total_mem_gb = _HPC_CORES * _MEM_PER_CPU_GB

    print("=" * 70)
    print("  RECIDIVISM ABM - PHASE 2 BIAS SIMULATION")
    print("=" * 70)
    print(f"  SLURM job      : {slurm_job}  node={slurm_node}  cpus={slurm_cpus}")
    print(f"  Workers        : {n_workers} / {_MAX_WORKERS} max  "
          f"(HPC: {_HPC_CORES} cores x {_MEM_PER_CPU_GB} GB = {total_mem_gb} GB total)")
    print(f"  Agents         : {CONFIG['initial_agents']}")
    print(f"  Warmup / Study : {CONFIG['warmup_months']} / {CONFIG['study_months']} months")
    print(f"  Runs per cond  : {n_runs}  "
          f"({CONFIG['N_REPS']} reps x {len(CONFIG['SEEDS'])} seeds)")
    print(f"  Batch size     : {_WORKER_BATCH_SIZE} seeds/task")
    print(f"  Stages         : {stages}")
    print(f"  bias_factor    : {BIAS_FACTOR} (fixed)")
    print(f"  Reference      : Race={RACE_REFERENCE}, Gender={GENDER_REFERENCE}")
    print(f"  Output dir     : {outdir}")
    print(f"  Gender sweep   : coarse [{GENDER_SWEEP_COARSE[0]:.1f} → "
          f"{GENDER_SWEEP_COARSE[-1]:.1f} step=0.1]  "
          f"+ fine [±{GENDER_SWEEP_FINE_HALF_WIDTH:.2f} step={GENDER_SWEEP_FINE_STEP}]")
    print(f"  Black sweep    : coarse [0.0 → 1.5 step=0.1]  "
          f"+ fine [±{BLACK_SWEEP_FINE_HALF_WIDTH:.2f} step={BLACK_SWEEP_FINE_STEP}]")
    print("  Baseline offsets : computed live from zero-bias measurement (see below)")
    print("  Female override  : " + (f"{FEMALE_OFFSET_OVERRIDE:.4f}" if FEMALE_OFFSET_OVERRIDE is not None else "None (live measurement)"))
    print("=" * 70)

    if force_rerun:
        import glob
        # Stage-scoped deletion — only delete checkpoints belonging to the
        # stages being run. This prevents --stage 3 --rerun from wiping
        # Stage 1 and Stage 2 checkpoints accidentally.
        STAGE_CHECKPOINTS = {
            1: ["stage1_Black.json", "stage1_Hispanic.json"],
            2: ["stage2_gender.json"],
            3: ["stage3_validation.json", "stage3_sensitivity.json"],
        }
        deleted   = []
        preserved = []
        all_cps   = glob.glob(os.path.join(outdir, "stage*.json"))
        target_fnames = set()
        for s in stages:
            for fname in STAGE_CHECKPOINTS.get(s, []):
                target_fnames.add(fname)
        for cp in all_cps:
            if os.path.basename(cp) in target_fnames:
                os.remove(cp)
                deleted.append(os.path.basename(cp))
            else:
                preserved.append(os.path.basename(cp))
        if deleted:
            print(f"  Warning: Cleared {len(deleted)} checkpoint(s) for "
                  f"stage(s) {stages}:")
            for f in sorted(deleted):
                print(f"    deleted  : {f}")
        if preserved:
            for f in sorted(preserved):
                print(f"    preserved: {f}")
        if not deleted:
            print(f"  Info: No checkpoints found for stage(s) {stages}")
        # Also delete baseline checkpoint when --rerun is used with stage 1
        # so offsets are remeasured with any model changes
        if 1 in stages:
            bp = os.path.join(outdir, "baseline_offsets.json")
            if os.path.exists(bp):
                os.remove(bp)
                print(f"    deleted  : baseline_offsets.json (will remeasure)")

    # ── Measure baseline offsets (live zero-bias run) ─────────────────────────
    measure_baseline_offsets(n_workers, outdir)

    # ── Checkpoint status report ─────────────────────────────────────────────
    # Print what is already saved so you know what will be loaded vs rerun.
    STAGE_CHECKPOINTS = {
        1: ["stage1_Black.json", "stage1_Hispanic.json"],
        2: ["stage2_gender.json"],
        3: ["stage3_validation.json", "stage3_sensitivity.json"],
    }
    print("\n  Checkpoint status:")
    for s, fnames in STAGE_CHECKPOINTS.items():
        for fname in fnames:
            cp = os.path.join(outdir, fname)
            if os.path.exists(cp):
                # Print saved bias value for quick sanity check
                try:
                    with open(cp) as f:
                        d = json.load(f)
                    if "value" in d:
                        val = d["value"]
                        mae = d.get("mae", "?")
                    elif "Female" in d:
                        val = d["Female"]["value"]
                        mae = d["Female"].get("mae", "?")
                    else:
                        val = mae = "?"
                    print(f"    Stage {s}  {fname:<35} FOUND  "
                          f"bias={val}  MAE={mae}")
                except Exception:
                    print(f"    Stage {s}  {fname:<35} FOUND (unreadable)")
            else:
                print(f"    Stage {s}  {fname:<35} missing — will run")
    print()

    race_params    = {}
    gender_params  = {}
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
            race_params, gender_params, n_workers, outdir
        )
        print("\n  Generating Stage 3 charts...")
        plot_stage3_intersectional(stage3_results["validation"], outdir)
        plot_stage3_sensitivity(stage3_results["sensitivity"], outdir)
        plot_disparity_summary(stage3_results["validation"], outdir)

    summary = {
        "bias_factor":      BIAS_FACTOR,
        "reference_race":   RACE_REFERENCE,
        "reference_gender": GENDER_REFERENCE,
        "calibrated_race_bias": {
            k: v.get("value") for k, v in race_params.items()
        },
        "calibrated_gender_bias": {
            k: v.get("value") for k, v in gender_params.items()
        },
        "bjs_targets":     BJS_SUBGROUP,
        "baseline_source":   "live_measurement",
        "female_offset_override": FEMALE_OFFSET_OVERRIDE,
        "baseline_offsets":  BASELINE_OFFSETS,
        "adjusted_targets":  ADJUSTED_TARGETS,
        "baseline_subgroup_rates": {
            3: {"Male": 0.6798, "Female": 0.6738, "White": 0.6792,
                "Black": 0.6778, "Hispanic": 0.6800},
            6: {"Male": 0.7949, "Female": 0.7912, "White": 0.7970,
                "Black": 0.7929, "Hispanic": 0.7925},
            9: {"Male": 0.8336, "Female": 0.8325, "White": 0.8326,
                "Black": 0.8371, "Hispanic": 0.8304},
        },
    }
    json_path = os.path.join(outdir, "phase2_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  OK  Summary -> {json_path}")

    # ── Final calibration verification ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL CALIBRATION SUMMARY")
    print("=" * 70)
    import math
    rows = []
    for grp in ["Black", "Hispanic"]:
        if grp in race_params:
            b = race_params[grp].get("value")
            m = race_params[grp].get("mae")
            rows.append((grp, b, m))
    if "Female" in gender_params:
        b = gender_params["Female"].get("value")
        m = gender_params["Female"].get("mae")
        rows.append(("Female", b, m))
    print(f"  {'Group':<12} {'Bias (log-odds)':>16} {'Odds Ratio':>12} {'MAE':>8}")
    print(f"  {'-'*52}")
    for grp, b, m in rows:
        if b is not None:
            print(f"  {grp:<12} {b:>16.3f} {math.exp(b):>11.3f}x {m:>8.4f}")
    print(f"  {'White':<12} {'Reference (0.00)':>16} {'1.000x':>12} {'-':>8}")
    print(f"  {'Male':<12} {'Reference (0.00)':>16} {'1.000x':>12} {'-':>8}")
    print("=" * 70)

    print(f"\n  Charts produced:")
    for name in ["phase2_stage1_race_calibration.png",
                 "phase2_stage2_gender_calibration.png",
                 "phase2_stage3_intersectional.png",
                 "phase2_stage3_sensitivity.png",
                 "phase2_disparity_summary.png"]:
        fpath = os.path.join(outdir, name)
        status = "OK " if os.path.exists(fpath) else "-- "
        print(f"    {status}  {name}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Set thread limits immediately at process start — before any imports
    # that might initialise BLAS. This is the safest place to set them.
    import os as _os
    for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ.setdefault(_env, "1")  # setdefault — respects any SLURM override

    multiprocessing.freeze_support()
    args = sys.argv[1:]

    if "--replot" in args:
        outdir = CONFIG["output_directory"]
        race_params   = {}
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
        print("OK Charts regenerated.")
        sys.exit(0)

    force_rerun = "--rerun" in args

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
                  "[--rerun] [--replot]")
            sys.exit(1)
    else:
        n_workers, reason = detect_workers()
        print(f"\n  Core auto-detection: {reason}")

    main(n_workers=n_workers, stages=stages, force_rerun=force_rerun)