import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pandas as pd
import plotly.graph_objects as go

# Ensure project-root imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_uncalibrated_params, get_global_calibration_params,get_flat_risk_weights

model = RecidivismModel(
    initial_agents=1000,
    bias_factor=0.0,
    monthly_intake=10,
    warmup_months=144,
    study_months=108,
    enable_peer_influence=True,
    weights=get_flat_risk_weights(),
    seed=42,
    mode="realistic",
    output_directory="smoke_test",
    calibration_params=get_uncalibrated_params(),
)
model.export_csv = False

for _ in range(252):   # 144 warmup + 108 study
    if not model.running: break
    model.step()

print(f"Year 1: {model.calculate_flag_rate('rearrest_1_yrs'):.3f}")
print(f"Year 3: {model.calculate_flag_rate('rearrest_3_yrs'):.3f}")
print(f"Year 6: {model.calculate_flag_rate('rearrest_6_yrs'):.3f}")
print(f"Year 9: {model.calculate_flag_rate('rearrest_9_yrs'):.3f}")