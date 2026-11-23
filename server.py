"""Dissertation demonstration dashboard for the legacy Mesa ModularServer API.

Designed for Mesa 1.1.x and the existing RecidivismModel implementation.
The dashboard shows ONE seed/configuration at a time; it does not calculate
multiseed confidence intervals or transition-event counts.
"""

import os
import sys
from html import escape

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

# This file lives inside the recidivism_abm package. Running `python run.py`
# from that directory must still make its parent importable.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.dirname(PACKAGE_DIR)
if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import (
    get_global_calibration_params,
    get_uncalibrated_params,
)

BJS_TARGETS = {3: 0.684, 6: 0.794, 9: 0.834}
STATE_COLORS = {
    "Trial": "#E5A12B",
    "Prison": "#D95A52",
    "Supervision": "#4D83B9",
    "Free": "#34966D",
}


def _card(title, content, subtitle=""):
    subtitle_html = (
        f'<div style="color:#64748b;font-size:12px;margin-top:4px">{subtitle}</div>'
        if subtitle else ""
    )
    return f"""
    <section style="box-sizing:border-box;width:100%;background:white;
        border:1px solid #e2e8f0;border-radius:12px;padding:20px 22px;
        margin:0 0 16px;box-shadow:0 2px 8px rgba(20,35,55,.04);">
        <div style="font-size:18px;font-weight:700;color:#20334b;">{title}</div>
        {subtitle_html}
        {content}
    </section>
    """


class DashboardRecidivismModel(RecidivismModel):
    """Select calibration before base model constructs any agents."""

    def __init__(self, calibration_mode="Calibrated", **kwargs):
        calibration_params = (
            get_global_calibration_params()
            if calibration_mode == "Calibrated"
            else get_uncalibrated_params()
        )
        self.calibration_mode = calibration_mode
        super().__init__(calibration_params=calibration_params, **kwargs)


class DashboardHeader(TextElement):
    def render(self, model):
        current = int(model.current_month)
        total_months = int(model.warmup_months + model.study_months)
        complete = current >= total_months or not model.running
        phase = "Completed" if complete else (
            "Warm-up" if current < model.warmup_months else "Study follow-up"
        )
        pct = min(100, max(0, 100 * current / max(1, total_months)))
        phase_color = "#25845e" if complete else "#315e91"
        content = f"""
        <div style="display:flex;flex-wrap:wrap;gap:12px;margin:16px 0;">
            {self._metric('Configuration', escape(str(model.calibration_mode)))}
            {self._metric('Seed', str(model.seed))}
            {self._metric('Simulation month', f'{current} / {total_months}')}
            {self._metric('Phase', phase, phase_color)}
        </div>
        <div style="background:#e9eff5;border-radius:20px;height:10px;overflow:hidden;">
            <div style="background:#3b82a6;width:{pct:.1f}%;height:100%;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;color:#64748b;
            font-size:12px;margin-top:7px;">
            <span>Warm-up: {model.warmup_months} months</span>
            <span>{pct:.0f}% complete</span>
            <span>Follow-up: {model.study_months} months</span>
        </div>
        """
        return _card(
            "Recidivism ABM | Simulation overview",
            content,
            "Single-seed demonstration • Calibration comparison against BJS benchmarks",
        )

    @staticmethod
    def _metric(label, value, color="#20334b"):
        return f"""
        <div style="flex:1;min-width:125px;background:#f5f8fb;border-radius:8px;
            padding:12px 14px;">
            <div style="font-size:11px;color:#64748b;">{label}</div>
            <div style="font-size:17px;font-weight:700;color:{color};margin-top:4px;">
                {value}
            </div>
        </div>
        """


class JusticeStateWorkflow(TextElement):
    """Current state populations; arrows are conceptual, not measured flows."""

    def render(self, model):
        counts = {name: 0 for name in STATE_COLORS}
        for agent in model.schedule.agents:
            state = getattr(agent, "justice_state", None)
            if state in counts:
                counts[state] += 1
        total = len(model.schedule.agents)
        stages = []
        for name, color in STATE_COLORS.items():
            count = counts[name]
            pct = count / total * 100 if total else 0
            stages.append(f"""
                <div style="flex:1 1 145px;min-width:145px;background:#f8fafc;
                    border:1px solid #e2e8f0;border-top:5px solid {color};
                    border-radius:10px;text-align:center;padding:14px 10px;">
                    <div style="font-weight:700;color:#334155;font-size:14px;">{name}</div>
                    <div style="font-weight:800;color:{color};font-size:27px;
                        margin:8px 0 3px;">{count:,}</div>
                    <div style="color:#64748b;font-size:12px;">{pct:.1f}% of agents</div>
                </div>
            """)
        arrow = '<div style="font-size:25px;color:#94a3b8;align-self:center;">&#8594;</div>'
        flow = arrow.join(stages)
        content = f"""
            <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;
                margin-top:18px;">{flow}</div>
            <div style="margin-top:16px;background:#eff5f9;border-radius:8px;
                padding:11px 14px;color:#40566d;font-size:13px;">
                <b>Current population:</b> {total:,} agents
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Study rearrests recorded:</b>
                {model.count_recidivists_during_study():,}
            </div>
            <div style="font-size:11px;color:#64748b;margin-top:10px;">
                Arrows illustrate the main justice-system pathway, not measured
                monthly transition counts. Rearrest is an outcome, not a fifth state.
            </div>
        """
        return _card(
            "Justice-system pathway",
            content,
            "Live population counts by agent state",
        )


class ABMvsBJSComparison(TextElement):
    """Grouped horizontal bars and errors for the completed single-seed run."""

    def render(self, model):
        total_months = model.warmup_months + model.study_months
        complete = model.current_month >= total_months or not model.running
        rows = []
        errors = []
        for years, target in BJS_TARGETS.items():
            # Do not interpret partial follow-up rates as final validation.
            rate = model.calculate_flag_rate(f"rearrest_{years}_yrs") if complete else None
            if rate is not None:
                rate = float(rate)
                errors.append(abs(rate - target))
                label = f"{rate * 100:.1f}%"
                delta = (rate - target) * 100
                diff_label = f"{delta:+.1f} pp"
                abm_width = max(0, min(100, rate * 100))
                abm_bar = f'<div style="width:{abm_width:.1f}%;height:100%;background:#2f6fb2;border-radius:4px;"></div>'
            else:
                label, diff_label, abm_bar = "Pending", "—", ""
            rows.append(f"""
                <div style="border-bottom:1px solid #edf1f5;padding:13px 0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;
                        margin-bottom:9px;gap:8px;">
                        <b style="font-size:14px;color:#26384e;">{years}-year rearrest</b>
                        <span style="font-size:12px;color:#64748b;">ABM − BJS: {diff_label}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <span style="width:38px;font-size:12px;color:#475569;">ABM</span>
                        <div style="flex:1;background:#edf2f7;height:16px;border-radius:4px;overflow:hidden;">{abm_bar}</div>
                        <b style="width:68px;text-align:right;font-size:13px;color:#2f6fb2;">{label}</b>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="width:38px;font-size:12px;color:#475569;">BJS</span>
                        <div style="flex:1;background:#edf2f7;height:16px;border-radius:4px;overflow:hidden;">
                            <div style="width:{target*100:.1f}%;height:100%;background:#e07b39;border-radius:4px;"></div>
                        </div>
                        <b style="width:68px;text-align:right;font-size:13px;color:#e07b39;">{target*100:.1f}%</b>
                    </div>
                </div>
            """)
        if len(errors) == 3:
            mae = sum(errors) / len(errors) * 100
            status = f"<b>Single-seed MAE:</b> {mae:.2f} percentage points"
        else:
            status = "<b>Validation status:</b> Awaiting completion of 108-month study follow-up"
        content = f"""
            <div style="margin-top:12px;">{''.join(rows)}</div>
            <div style="background:#eef5f9;color:#26445e;padding:12px 14px;
                border-radius:8px;margin-top:16px;font-size:14px;">{status}</div>
            <div style="font-size:11px;color:#64748b;margin-top:10px;">
                This panel summarizes one simulation seed. Confidence intervals and
                robustness conclusions require independent replicated runs.
            </div>
        """
        return _card(
            "Empirical validation | ABM vs BJS",
            content,
            "Cumulative first rearrest at 3, 6 and 9 years",
        )


# Existing DataCollector already reports Trial, Prison, Supervision and Free.
chart_justice_states = ChartModule(
    [
        {"Label": "Trial", "Color": STATE_COLORS["Trial"]},
        {"Label": "Prison", "Color": STATE_COLORS["Prison"]},
        {"Label": "Supervision", "Color": STATE_COLORS["Supervision"]},
        {"Label": "Free", "Color": STATE_COLORS["Free"]},
    ],
    data_collector_name="datacollector",
    canvas_height=290,
    canvas_width=790,
)


# Lock observation windows to the dissertation's calibration design.
model_params = {
    "initial_agents": UserSettableParameter(
        "slider", "Initial agents", 1500, 500, 5000, 500
    ),
    "warmup_months": 144,
    "study_months": 108,
    "monthly_intake": 10,
    "bias_factor": UserSettableParameter(
        "choice", "Demographic bias", value=0, choices=[0, 1]
    ),
    "enable_peer_influence": UserSettableParameter(
        "checkbox", "Enable peer influence", True
    ),
    "calibration_mode": UserSettableParameter(
        "choice", "Calibration", value="Calibrated",
        choices=["Calibrated", "Uncalibrated"]
    ),
    "seed": UserSettableParameter("number", "Simulation seed", value=1000),
}

server = ModularServer(
    DashboardRecidivismModel,
    [
        DashboardHeader(),
        JusticeStateWorkflow(),
        ABMvsBJSComparison(),
        chart_justice_states,
    ],
    "Recidivism ABM | Dissertation demonstration",
    model_params,
)
