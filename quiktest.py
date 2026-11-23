from recidivism_abm.config.risk_config import (
    get_global_calibration_params,
    get_bias_sweep_params,
)

# No-bias baseline
baseline = get_global_calibration_params()
assert baseline["bias_factor"] == 0.0
assert baseline["group_bias_race"]["Black"] == 0.0
assert baseline["group_bias_gender"]["Female"] == 0.0

# Calibrated bias
biased = get_global_calibration_params({
    "bias_factor": 1.0,
    "bias_scope": "supervision_only",
})
assert biased["bias_factor"] == 1.0

# Fractional bias
partial = get_global_calibration_params({
    "bias_factor": 0.5,
})
assert partial["bias_factor"] == 0.5

# Explicit sweep coefficients must not be overwritten
sweep = get_bias_sweep_params(
    bias_factor=0.5,
    scope="supervision_only",
    group_bias_race={"White": 0.0, "Black": 0.2},
    group_bias_gender={"Male": 0.0, "Female": -0.3},
)
assert sweep["bias_factor"] == 0.5
assert sweep["group_bias_race"]["Black"] == 0.2
assert sweep["group_bias_gender"]["Female"] == -0.3

print("PASS: Risk configuration tests")