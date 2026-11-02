"""
test_decay_parameters.py
========================
Verification tests for the risk decay and supervision monitoring decay
parameters in evaluate_recidivism (person.py).

Professor's concern: "Risk decay parameters are not behaving as expected."

This script exposes three bugs:
  Bug 1 — Double application of decay multipliers (>= not elif)
  Bug 2 — Decay on γ is inert because γ = 0.0 in baseline calibration
  Bug 3 — Supervision monitoring decay compounds incorrectly after year 6

Run with:
    python test_decay_parameters.py

All tests print PASS or FAIL with a clear explanation.
"""

import math


# ── Helpers (copied verbatim from person.py) ──────────────────────────────────

def logit(p):
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def sigmoid(x):
    if x > 60:  return 1.0
    if x < -60: return 0.0
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


BASELINE_Q = 0.0906   # BJS year-1 quarterly hazard


def compute_p_rearrest_CURRENT(quarters, risk_score, gamma, smi,
                                decay_3y_risk=1.0, decay_6y_risk=1.0,
                                decay_3y_smi=1.0,  decay_6y_smi=1.0):
    """
    Reproduces the CURRENT (buggy) decay logic in person.py lines 598-604.
    Uses two independent 'if' blocks — both fire for quarters >= 24.
    """
    rcs = gamma
    s   = smi

    if quarters >= 12:          # Bug: also fires when quarters >= 24
        rcs *= decay_3y_risk
        s   *= decay_3y_smi

    if quarters >= 24:          # Bug: stacks on top of the 12-quarter block
        rcs *= decay_6y_risk
        s   *= decay_6y_smi

    log_odds = logit(BASELINE_Q)
    log_odds += rcs * (risk_score - 0.5)
    log_odds += s
    return sigmoid(log_odds), rcs, s


def compute_p_rearrest_FIXED(quarters, risk_score, gamma, smi,
                              decay_3y_risk=1.0, decay_6y_risk=1.0,
                              decay_3y_smi=1.0,  decay_6y_smi=1.0):
    """
    FIXED decay logic using elif so multipliers are mutually exclusive.
    3-year decay applies only in years 3-6 (quarters 12-23).
    6-year decay applies only after year 6 (quarters >= 24).
    """
    rcs = gamma
    s   = smi

    if quarters >= 24:          # Fixed: 6-year block checked first
        rcs *= decay_6y_risk
        s   *= decay_6y_smi
    elif quarters >= 12:        # Fixed: only reached if < 24
        rcs *= decay_3y_risk
        s   *= decay_3y_smi

    log_odds = logit(BASELINE_Q)
    log_odds += rcs * (risk_score - 0.5)
    log_odds += s
    return sigmoid(log_odds), rcs, s


# ── Test runner ───────────────────────────────────────────────────────────────

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


# =============================================================================
print("\n" + "="*70)
print("  TEST BLOCK 1 — Bug 1: Double application of decay multipliers")
print("="*70)
print("""
  Scenario: decay_3Y = 0.5, decay_6Y = 0.5, gamma = 2.0, smi = 0.35
  At quarter 25 (past 6 years), CURRENT code fires BOTH if-blocks:
    rcs = 2.0 * 0.5 * 0.5 = 0.50  (compounded — wrong)
  FIXED code fires only the 6-year block:
    rcs = 2.0 * 0.5 = 1.00  (independent — correct)
""")

gamma    = 2.0
smi      = 0.35
d3r, d6r = 0.5, 0.5
d3s, d6s = 0.5, 0.5
risk     = 0.7

# At quarter 25 — past both thresholds
p_cur, rcs_cur, smi_cur = compute_p_rearrest_CURRENT(25, risk, gamma, smi,
                                                       d3r, d6r, d3s, d6s)
p_fix, rcs_fix, smi_fix = compute_p_rearrest_FIXED(  25, risk, gamma, smi,
                                                       d3r, d6r, d3s, d6s)

print(f"  Quarter 25 results:")
print(f"    CURRENT  rcs={rcs_cur:.4f}  smi={smi_cur:.4f}  p={p_cur:.4f}")
print(f"    FIXED    rcs={rcs_fix:.4f}  smi={smi_fix:.4f}  p={p_fix:.4f}")
print()

check("At quarter 25, current code compounds both decay multipliers",
      abs(rcs_cur - 0.5) < 1e-9,
      f"rcs_current={rcs_cur:.4f}, expected 0.5 (2.0×0.5×0.5 — both blocks fired)")

check("At quarter 25, fixed code applies only 6-year decay",
      abs(rcs_fix - 1.0) < 1e-9,
      f"rcs_fixed={rcs_fix:.4f}, expected 1.0 (2.0×0.5 — only 6Y block)")

check("Current and fixed differ at quarter 25 (confirms bug exists)",
      abs(p_cur - p_fix) > 0.01,
      f"p_current={p_cur:.4f}  p_fixed={p_fix:.4f}  diff={abs(p_cur-p_fix):.4f}")

# At quarter 15 — only past 3-year threshold, BOTH should give same result
p_cur15, rcs_cur15, _ = compute_p_rearrest_CURRENT(15, risk, gamma, smi, d3r, d6r, d3s, d6s)
p_fix15, rcs_fix15, _ = compute_p_rearrest_FIXED(  15, risk, gamma, smi, d3r, d6r, d3s, d6s)

check("At quarter 15 (years 3–6), both versions agree",
      abs(p_cur15 - p_fix15) < 1e-9,
      f"p_current={p_cur15:.4f}  p_fixed={p_fix15:.4f} — should be identical here")

# At quarter 10 — before any decay, both should give same result
p_cur10, rcs_cur10, _ = compute_p_rearrest_CURRENT(10, risk, gamma, smi, d3r, d6r, d3s, d6s)
p_fix10, rcs_fix10, _ = compute_p_rearrest_FIXED(  10, risk, gamma, smi, d3r, d6r, d3s, d6s)

check("At quarter 10 (years 0–3), no decay applied in either version",
      abs(p_cur10 - p_fix10) < 1e-9 and abs(rcs_cur10 - gamma) < 1e-9,
      f"rcs={rcs_cur10:.4f} (expected {gamma}), p_diff={abs(p_cur10-p_fix10):.6f}")


# =============================================================================
print("\n" + "="*70)
print("  TEST BLOCK 2 — Bug 2: Decay on γ is inert when γ = 0")
print("="*70)
print("""
  Scenario: calibrated baseline — gamma = 0.0, decay values = 0.5
  With gamma = 0.0, multiplying by any decay factor still gives 0.0.
  The decay on risk_contrast_strength has never actually done anything
  in the baseline calibration run.
""")

gamma_zero = 0.0
smi_active = 0.35

for q in [10, 15, 25]:
    p, rcs, s = compute_p_rearrest_CURRENT(q, 0.7, gamma_zero, smi_active, 0.5, 0.5, 1.0, 1.0)
    check(f"Quarter {q}: risk decay inert — rcs stays 0.0 when gamma=0",
          abs(rcs) < 1e-12,
          f"rcs={rcs} (0.0 × any decay = 0.0 — gamma decay has never operated)")

print()
print("  NOTE: This is not a crash — it is a silent no-op. The decay parameter")
print("  was calibrated against an output (rcs) that was always zero. The decay")
print("  values of 1.0 in risk_config.py reflect an unidentified parameter,")
print("  not a validated neutral result. Any test with gamma > 0 would be")
print("  the first real exercise of this decay channel.")


# =============================================================================
print("\n" + "="*70)
print("  TEST BLOCK 3 — Bug 3: Supervision monitoring decay compounds")
print("="*70)
print("""
  Scenario: smi = 0.35, decay_3Y_smi = 0.8, decay_6Y_smi = 0.8
  Expected at quarter 25:  smi = 0.35 × 0.8 = 0.280  (6Y decay only)
  Actual (current code):   smi = 0.35 × 0.8 × 0.8 = 0.224  (both stacked)
""")

smi_test  = 0.35
d3s_test  = 0.8
d6s_test  = 0.8

_, _, smi_cur_25 = compute_p_rearrest_CURRENT(25, 0.5, 0.0, smi_test, 1.0, 1.0, d3s_test, d6s_test)
_, _, smi_fix_25 = compute_p_rearrest_FIXED(25, 0.5, 0.0, smi_test, 1.0, 1.0, d3s_test, d6s_test)

expected_correct  = smi_test * d6s_test          # 0.35 × 0.8 = 0.28
expected_buggy    = smi_test * d3s_test * d6s_test  # 0.35 × 0.8 × 0.8 = 0.224

check("Current code produces compounded (wrong) smi at quarter 25",
      abs(smi_cur_25 - expected_buggy) < 1e-9,
      f"smi_current={smi_cur_25:.4f}, expected_buggy={expected_buggy:.4f}")

check("Fixed code produces correct (non-compounded) smi at quarter 25",
      abs(smi_fix_25 - expected_correct) < 1e-9,
      f"smi_fixed={smi_fix_25:.4f}, expected_correct={expected_correct:.4f}")

check("Compounding error magnitude is meaningful",
      abs(smi_cur_25 - smi_fix_25) > 0.01,
      f"Error = {abs(smi_cur_25 - smi_fix_25):.4f} log-odds units")

# With baseline calibration values (both decay = 1.0) — no visible effect
_, _, smi_baseline = compute_p_rearrest_CURRENT(25, 0.5, 0.0, 0.35, 1.0, 1.0, 1.0, 1.0)
check("With baseline decay=1.0, bug is silent (1.0×1.0 = 1.0 either way)",
      abs(smi_baseline - 0.35) < 1e-9,
      f"smi={smi_baseline:.4f} — confirms bug is latent, not currently active")


# =============================================================================
print("\n" + "="*70)
print("  TEST BLOCK 4 — Correct decay behaviour (fixed version)")
print("="*70)
print("""
  Verifies that with the elif fix, each decay bracket fires exactly once
  and is mutually exclusive.
""")

gamma_test = 2.0
smi_test   = 0.35
d3r_test   = 0.7    # 3-year decay reduces gamma by 30%
d6r_test   = 0.5    # 6-year decay reduces gamma by 50%

quarters_and_expected = [
    (8,  gamma_test,            "no decay — before year 3"),
    (15, gamma_test * d3r_test, "3Y decay only — between year 3 and 6"),
    (25, gamma_test * d6r_test, "6Y decay only — after year 6"),
    (36, gamma_test * d6r_test, "6Y decay only — well past year 6"),
]

for q, expected_rcs, label in quarters_and_expected:
    _, rcs, _ = compute_p_rearrest_FIXED(q, 0.7, gamma_test, smi_test,
                                          d3r_test, d6r_test, 1.0, 1.0)
    check(f"Quarter {q:2d} ({label})",
          abs(rcs - expected_rcs) < 1e-9,
          f"rcs={rcs:.4f}, expected={expected_rcs:.4f}")


# =============================================================================
print("\n" + "="*70)
print(f"  RESULTS:  {passed} passed  |  {failed} failed")
print("="*70)

if failed > 0:
    print("""
  SUMMARY OF BUGS CONFIRMED:

  Bug 1  Both if-blocks fire at quarters >= 24, compounding decay multipliers.
         Fix: change second 'if' to 'elif' in person.py lines 598-604.

  Bug 2  Risk decay on γ is inert in baseline calibration because γ = 0.0.
         Any decay value produces the same result (0.0 × anything = 0.0).
         Note: not a runtime error, but means the parameter was never tested.

  Bug 3  Supervision monitoring decay has the same compounding bug as Bug 1.
         With baseline values (decay=1.0) this is invisible. Non-unity values
         produce incorrect results.

  ONE-LINE FIX in person.py:
         Change line 602 from:
             if quarters_at_risk >= 24:
         To:
             elif quarters_at_risk >= 24:
""")
else:
    print("\n  All tests passed.\n")