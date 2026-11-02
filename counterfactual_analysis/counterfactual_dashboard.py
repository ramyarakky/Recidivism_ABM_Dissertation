#!/usr/bin/env python3
"""
counterfactual_dashboard.py
===============================================================
Interactive dissertation dashboard for Recidivism ABM
counterfactual experiments with literature-informed effect-size sensitivity.

Reads model outputs from:
    counterfactual_output/model_results/

Does NOT rerun the ABM.

Expected files:
    employment_summary.csv
    employment_paired_effects.csv
    employment_raw.csv
    program_summary.csv
    program_paired_effects.csv
    program_raw.csv

Run:
    python counterfactual_dashboard.py

Optional:
    python counterfactual_dashboard.py --port 8050
    python counterfactual_dashboard.py --host 0.0.0.0

Then open:
    http://127.0.0.1:8050
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dash import Dash, Input, Output, dcc, html, dash_table
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit(
        "\nDash/Plotly is required for the interactive dashboard.\n"
        "Install once with:\n\n"
        "    pip install dash plotly\n"
    ) from exc


# ================================================================
# PATHS
# ================================================================


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "counterfactual_output"
RESULTS_DIR = OUTPUT_ROOT / "model_results"

WINDOWS = (3, 6, 9)

BJS_TARGETS = {
    3: 0.684,
    6: 0.794,
    9: 0.834,
}


# ================================================================
# STYLE
# ================================================================

PAGE_BG = "#F4F5F1"
CARD_BG = "#FFFFFF"
TEXT = "#1F2926"
MUTED = "#68726D"
BORDER = "#DDE2DC"

EMPLOYMENT_COLOR = "#2B6E68"
PROGRAM_COLOR = "#A6672F"
BASELINE_COLOR = "#8D9690"

CHART_COLORS = {
    3: "#2F6FB2",
    6: "#27AE60",
    9: "#8E44AD",
}


def card_style():
    return {
        "backgroundColor": CARD_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "18px",
        "boxShadow": "0 1px 2px rgba(0,0,0,0.03)",
    }


# ================================================================
# DATA LOADERS
# ================================================================

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_experiment(experiment: str):
    summary = _read_csv(
        RESULTS_DIR / f"{experiment}_summary.csv"
    )
    paired = _read_csv(
        RESULTS_DIR / f"{experiment}_paired_effects.csv"
    )
    raw = _read_csv(
        RESULTS_DIR / f"{experiment}_raw.csv"
    )

    return summary, paired, raw


def available_experiments():
    options = []

    for exp in ("employment", "program"):
        summary = RESULTS_DIR / f"{exp}_summary.csv"
        paired = RESULTS_DIR / f"{exp}_paired_effects.csv"

        if summary.exists() and paired.exists():
            options.append(exp)

    return options


def available_strengths(experiment: str):
    """Return literature-informed sensitivity scenarios found in results."""
    summary, _, _ = load_experiment(experiment)
    if summary.empty or "strength_label" not in summary.columns:
        return [("legacy", 0.0, 1.0)]

    cols = ["strength_label", "intervention_strength"]
    strengths = summary[cols].drop_duplicates().copy()
    strengths["assumed_or"] = np.exp(strengths["intervention_strength"].astype(float))

    preferred = (
        ["null", "small", "moderate", "strong"]
        if experiment == "employment"
        else ["null", "central", "moderate", "strong"]
    )
    order = {name: i for i, name in enumerate(preferred)}
    strengths["_order"] = strengths["strength_label"].map(order).fillna(99)
    strengths = strengths.sort_values(["_order", "intervention_strength"])

    return [
        (str(r["strength_label"]), float(r["intervention_strength"]), float(r["assumed_or"]))
        for _, r in strengths.iterrows()
    ]


def filter_strength(df: pd.DataFrame, strength_label: str) -> pd.DataFrame:
    if df.empty or "strength_label" not in df.columns:
        return df.copy()
    return df[df["strength_label"].astype(str) == str(strength_label)].copy()


def default_strength(experiment: str):
    labels = [x[0] for x in available_strengths(experiment)]
    preferred = "moderate" if experiment == "employment" else "central"
    if preferred in labels:
        return preferred
    return labels[0] if labels else None


# ================================================================
# FIGURES
# ================================================================

def make_rate_figure(
    summary: pd.DataFrame,
    experiment: str,
):
    fig = go.Figure()

    if summary.empty:
        fig.add_annotation(
            text="No model results available",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    x = summary["cf_percent"]

    for years in WINDOWS:
        mean_col = f"rearrest_{years}yr_mean"
        lo_col = f"rearrest_{years}yr_ci95_low"
        hi_col = f"rearrest_{years}yr_ci95_high"

        y = summary[mean_col]
        lo = summary[lo_col]
        hi = summary[hi_col]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"{years}-Year",
                line=dict(
                    color=CHART_COLORS[years],
                    width=3,
                ),
                marker=dict(size=8),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(hi - y),
                    arrayminus=(y - lo),
                    thickness=1.5,
                    width=4,
                ),
                hovertemplate=(
                    "CF change: %{x:.0f}%<br>"
                    "Rearrest: %{y:.2%}<extra></extra>"
                ),
            )
        )

        if years in BJS_TARGETS:
            fig.add_hline(
                y=BJS_TARGETS[years],
                line_dash="dot",
                line_width=1,
                line_color=CHART_COLORS[years],
                opacity=0.35,
            )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1.5,
        line_color=BASELINE_COLOR,
    )

    fig.update_layout(
        title=(
            f"{experiment.capitalize()} Counterfactual: "
            "Rearrest Rate by Follow-Up Horizon"
        ),
        xaxis_title=(
            f"{experiment.capitalize()} counterfactual change (%)"
        ),
        yaxis_title="Cumulative rearrest rate",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Follow-up",
        margin=dict(l=60, r=30, t=70, b=60),
    )

    return fig


def make_delta_figure(
    paired: pd.DataFrame,
    experiment: str,
):
    fig = go.Figure()

    if paired.empty:
        fig.add_annotation(
            text="No paired-effect results available",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    x = paired["cf_percent"]

    for years in WINDOWS:
        mean_col = f"delta_{years}yr_pp_mean"
        lo_col = f"delta_{years}yr_pp_ci95_low"
        hi_col = f"delta_{years}yr_pp_ci95_high"

        y = paired[mean_col]
        lo = paired[lo_col]
        hi = paired[hi_col]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"{years}-Year",
                line=dict(
                    color=CHART_COLORS[years],
                    width=3,
                ),
                marker=dict(size=8),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(hi - y),
                    arrayminus=(y - lo),
                    thickness=1.5,
                    width=4,
                ),
                hovertemplate=(
                    "CF change: %{x:.0f}%<br>"
                    "Δ: %{y:+.2f} pp<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1.5,
        line_color=BASELINE_COLOR,
    )
    fig.add_vline(
        x=0,
        line_dash="dot",
        line_width=1,
        line_color=BASELINE_COLOR,
    )

    fig.update_layout(
        title=(
            f"{experiment.capitalize()} Counterfactual: "
            "Change from Calibrated Baseline"
        ),
        xaxis_title=(
            f"{experiment.capitalize()} counterfactual change (%)"
        ),
        yaxis_title="Change from baseline (percentage points)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Follow-up",
        margin=dict(l=60, r=30, t=70, b=60),
    )

    return fig


def make_mechanism_figure(
    summary: pd.DataFrame,
    experiment: str,
):
    """
    Mechanism validation chart.

    Confirms that the intervention parameter is actually changing the
    intended ABM state variable before interpreting rearrest outcomes.
    """
    fig = go.Figure()

    if summary.empty:
        fig.add_annotation(
            text="No mechanism results available",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    if experiment == "employment":
        metric = "mean_percent_days_employed_mean"
        ylabel = "Mean percent days employed"
        title = "Mechanism Check: Employment Responds to the Intervention"
        subtitle = (
            "Higher employment counterfactual levels should increase "
            "observed post-release employment."
        )
        color = EMPLOYMENT_COLOR
    else:
        metric = "mean_program_attendances_mean"
        ylabel = "Mean program attendances"
        title = "Mechanism Check: Program Participation Responds to the Intervention"
        subtitle = (
            "Higher program counterfactual levels should increase "
            "observed program participation."
        )
        color = PROGRAM_COLOR

    if metric not in summary.columns:
        fig.add_annotation(
            text=f"{metric} not found in summary output",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    plot_df = summary.sort_values("cf_percent").copy()

    fig.add_trace(
        go.Scatter(
            x=plot_df["cf_percent"],
            y=plot_df[metric],
            mode="lines+markers",
            name="Observed mechanism",
            line=dict(color=color, width=3),
            marker=dict(size=9),
            hovertemplate=(
                "Counterfactual: %{x:+.0f}%<br>"
                "Observed mean: %{y:.3f}<extra></extra>"
            ),
        )
    )

    # Highlight the calibrated baseline if present.
    baseline = plot_df[
        np.isclose(plot_df["cf_level"].astype(float), 0.0)
    ]

    if not baseline.empty:
        baseline_y = float(baseline.iloc[0][metric])

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[baseline_y],
                mode="markers",
                name="Calibrated baseline",
                marker=dict(
                    size=14,
                    symbol="diamond",
                    color=BASELINE_COLOR,
                ),
                hovertemplate=(
                    "Baseline (0%)<br>"
                    f"{ylabel}: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1.2,
        line_color=BASELINE_COLOR,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>"
        ),
        xaxis_title=(
            f"{experiment.capitalize()} counterfactual change (%)"
        ),
        yaxis_title=ylabel,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=65, r=30, t=90, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig



def make_pathway_figures(summary: pd.DataFrame, experiment: str):
    """
    Pathway diagnostic.

    Employment:
        CF -> % Days Employed -> Technical Violations
           -> Program Absences -> 3-Year Rearrest

    Program:
        CF -> Program Attendances -> Program Absences
           -> Technical Violations -> 3-Year Rearrest

    These are diagnostic co-movement plots, not proof of a causal chain.
    """
    if summary.empty:
        return [go.Figure() for _ in range(4)]

    plot_df = summary.sort_values("cf_percent").copy()

    if experiment == "employment":
        metrics = [
            (
                "mean_percent_days_employed_mean",
                "1. Employment Response",
                "Mean Percent Days Employed",
                False,
            ),
            (
                "mean_technical_violations_mean",
                "2. Technical Violations",
                "Mean Technical Violations",
                False,
            ),
            (
                "mean_program_unexcused_absences_mean",
                "3. Program Absences",
                "Mean Unexcused Absences",
                False,
            ),
            (
                "rearrest_3yr_mean",
                "4. Rearrest Outcome",
                "3-Year Rearrest Rate",
                True,
            ),
        ]
        x_title = "Employment Counterfactual Change (%)"
    else:
        metrics = [
            (
                "mean_program_attendances_mean",
                "1. Program Participation Response",
                "Mean Program Attendances",
                False,
            ),
            (
                "mean_program_unexcused_absences_mean",
                "2. Program Absences",
                "Mean Unexcused Absences",
                False,
            ),
            (
                "mean_technical_violations_mean",
                "3. Technical Violations",
                "Mean Technical Violations",
                False,
            ),
            (
                "rearrest_3yr_mean",
                "4. Rearrest Outcome",
                "3-Year Rearrest Rate",
                True,
            ),
        ]
        x_title = "Program Counterfactual Change (%)"

    figures = []

    for column, title, ylabel, is_rate in metrics:
        fig = go.Figure()

        if column not in plot_df.columns:
            fig.add_annotation(
                text=f"{column} not found in summary output",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(
                title=title,
                template="plotly_white",
                height=340,
            )
            figures.append(fig)
            continue

        fig.add_trace(
            go.Scatter(
                x=plot_df["cf_percent"],
                y=plot_df[column],
                mode="lines+markers",
                marker=dict(size=8),
                line=dict(width=3),
                hovertemplate=(
                    "Counterfactual: %{x:+.0f}%<br>"
                    "Value: %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_width=1.2,
            line_color=BASELINE_COLOR,
        )

        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=ylabel,
            template="plotly_white",
            showlegend=False,
            hovermode="x unified",
            margin=dict(l=55, r=20, t=60, b=55),
            height=340,
        )

        if is_rate:
            fig.update_yaxes(tickformat=".1%")

        figures.append(fig)

    return figures


def make_seed_distribution(
    raw: pd.DataFrame,
    experiment: str,
    selected_level: float,
):
    fig = go.Figure()

    if raw.empty:
        fig.add_annotation(
            text="No raw replication data available",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    subset = raw[
        np.isclose(
            raw["cf_level"].astype(float),
            float(selected_level),
        )
    ]

    if subset.empty:
        fig.add_annotation(
            text="No replications for selected level",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    for years in WINDOWS:
        col = f"rearrest_{years}yr"

        fig.add_trace(
            go.Box(
                y=subset[col],
                name=f"{years}-Year",
                boxpoints="all",
                jitter=0.28,
                pointpos=0,
                marker=dict(
                    size=4,
                    color=CHART_COLORS[years],
                    opacity=0.45,
                ),
                line=dict(
                    color=CHART_COLORS[years],
                ),
                hovertemplate=(
                    f"{years}-Year<br>"
                    "Rearrest: %{y:.2%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            "Distribution Across 200 Replications "
            f"at {selected_level * 100:+.0f}%"
        ),
        yaxis_title="Rearrest rate",
        yaxis_tickformat=".0%",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=30, t=70, b=50),
    )

    return fig


def make_sensitivity_figure(
    paired: pd.DataFrame,
    experiment: str,
    years: int = 3,
):
    """Compare all literature-informed beta assumptions on one chart."""
    fig = go.Figure()

    if paired.empty or "strength_label" not in paired.columns:
        fig.add_annotation(
            text="Sensitivity results are not available in the current output files.",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig

    col = f"delta_{years}yr_pp_mean"
    for label, beta, assumed_or in available_strengths(experiment):
        subset = filter_strength(paired, label).sort_values("cf_percent")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["cf_percent"],
                y=subset[col],
                mode="lines+markers",
                name=f"{label.capitalize()} (β={beta:+.3f}, OR={assumed_or:.2f})",
                hovertemplate=(
                    "CF: %{x:+.0f}%<br>"
                    f"{years}-Year Δ: %{{y:+.2f}} pp<br>"
                    f"β={beta:+.3f}, OR={assumed_or:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_width=1.3, line_color=BASELINE_COLOR)
    fig.add_vline(x=0, line_dash="dot", line_width=1, line_color=BASELINE_COLOR)
    fig.update_layout(
        title=(
            f"{experiment.capitalize()} Literature-Informed Sensitivity Analysis: "
            f"{years}-Year Rearrest"
        ),
        xaxis_title=f"{experiment.capitalize()} counterfactual change (%)",
        yaxis_title="Change from calibrated baseline (percentage points)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Effect-size assumption",
        margin=dict(l=60, r=30, t=75, b=60),
    )
    return fig


# ================================================================
# SUMMARY CARDS
# ================================================================

def baseline_value(summary, years):
    if summary.empty:
        return None

    row = summary[
        np.isclose(
            summary["cf_level"].astype(float),
            0.0,
        )
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0][f"rearrest_{years}yr_mean"]
    )


def selected_effect(paired, level, years):
    if paired.empty:
        return None, None, None

    row = paired[
        np.isclose(
            paired["cf_level"].astype(float),
            float(level),
        )
    ]

    if row.empty:
        return None, None, None

    r = row.iloc[0]

    return (
        float(r[f"delta_{years}yr_pp_mean"]),
        float(r[f"delta_{years}yr_pp_ci95_low"]),
        float(r[f"delta_{years}yr_pp_ci95_high"]),
    )


# ================================================================
# APP
# ================================================================

experiments = available_experiments()

if not experiments:
    raise SystemExit(
        "\nNo counterfactual output files were found.\n"
        f"Expected results under:\n    {RESULTS_DIR}\n\n"
        "Run counterfactual_analysis.py first."
    )

default_experiment = experiments[0]

app = Dash(__name__)
app.title = "Recidivism ABM Counterfactual Dashboard"


app.layout = html.Div(
    style={
        "backgroundColor": PAGE_BG,
        "minHeight": "100vh",
        "fontFamily": "Arial, sans-serif",
        "color": TEXT,
        "padding": "24px 30px",
    },
    children=[
        html.Div(
            [
                html.H1(
                    "Recidivism ABM Counterfactual Analysis",
                    style={
                        "marginBottom": "4px",
                        "fontFamily": "Georgia, serif",
                        "fontSize": "29px",
                    },
                ),
                html.Div(
                    (
                        "Interactive view of employment and program "
                        "counterfactual experiments. Model outputs remain "
                        "separate from static dissertation charts."
                    ),
                    style={
                        "color": MUTED,
                        "fontSize": "14px",
                        "marginBottom": "22px",
                    },
                ),
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "260px 1fr",
                "gap": "22px",
            },
            children=[
                html.Div(
                    style=card_style(),
                    children=[
                        html.H3(
                            "Scenario Controls",
                            style={
                                "marginTop": 0,
                                "fontFamily": "Georgia, serif",
                            },
                        ),
                        html.Label(
                            "Counterfactual family",
                            style={
                                "fontWeight": 600,
                                "fontSize": "13px",
                            },
                        ),
                        dcc.Dropdown(
                            id="experiment",
                            options=[
                                {
                                    "label": e.capitalize(),
                                    "value": e,
                                }
                                for e in experiments
                            ],
                            value=default_experiment,
                            clearable=False,
                            style={"marginTop": "6px"},
                        ),

                        html.Div(
                            style={"height": "18px"}
                        ),

                        html.Label(
                            "Effect-size assumption",
                            style={
                                "fontWeight": 600,
                                "fontSize": "13px",
                            },
                        ),
                        dcc.Dropdown(
                            id="strength",
                            clearable=False,
                            style={"marginTop": "6px"},
                        ),
                        html.Div(
                            id="strength-note",
                            style={
                                "fontSize": "11px",
                                "color": MUTED,
                                "lineHeight": 1.45,
                                "marginTop": "7px",
                            },
                        ),

                        html.Div(
                            style={"height": "18px"}
                        ),

                        html.Label(
                            "Counterfactual level",
                            style={
                                "fontWeight": 600,
                                "fontSize": "13px",
                            },
                        ),
                        dcc.Dropdown(
                            id="cf-level",
                            clearable=False,
                            style={"marginTop": "6px"},
                        ),

                        html.Hr(
                            style={
                                "border": 0,
                                "borderTop": f"1px solid {BORDER}",
                                "margin": "22px 0",
                            }
                        ),

                        html.Div(
                            id="run-info",
                            style={
                                "fontSize": "12.5px",
                                "color": MUTED,
                                "lineHeight": 1.7,
                            },
                        ),
                    ],
                ),

                html.Div(
                    children=[
                        html.Div(
                            id="metric-cards",
                            style={
                                "display": "grid",
                                "gridTemplateColumns":
                                    "repeat(4, minmax(160px, 1fr))",
                                "gap": "14px",
                                "marginBottom": "20px",
                            },
                        ),

                        html.Div(
                            style=card_style(),
                            children=[
                                dcc.Graph(
                                    id="rate-chart",
                                    config={
                                        "displaylogo": False,
                                        "toImageButtonOptions": {
                                            "format": "png",
                                            "filename":
                                                "counterfactual_rearrest_rates",
                                            "scale": 2,
                                        },
                                    },
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "20px",
                                "marginTop": "20px",
                            },
                            children=[
                                html.Div(
                                    style=card_style(),
                                    children=[
                                        dcc.Graph(
                                            id="delta-chart",
                                            config={
                                                "displaylogo": False
                                            },
                                        )
                                    ],
                                ),
                                html.Div(
                                    style=card_style(),
                                    children=[
                                        html.Div(
                                            "Mechanism validation",
                                            style={
                                                "fontSize": "12px",
                                                "fontWeight": 600,
                                                "color": MUTED,
                                                "marginBottom": "4px",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.4px",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="mechanism-chart",
                                            config={
                                                "displaylogo": False,
                                                "toImageButtonOptions": {
                                                    "format": "png",
                                                    "filename": "counterfactual_mechanism_check",
                                                    "scale": 2,
                                                },
                                            },
                                        )
                                    ],
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                **card_style(),
                                "marginTop": "20px",
                            },
                            children=[
                                dcc.Graph(
                                    id="distribution-chart",
                                    config={
                                        "displaylogo": False
                                    },
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                **card_style(),
                                "marginTop": "20px",
                            },
                            children=[
                                html.H2(
                                    "Literature-Informed Sensitivity Comparison",
                                    style={
                                        "fontFamily": "Georgia, serif",
                                        "marginTop": 0,
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.P(
                                    (
                                        "Compares rearrest changes across all effect-size "
                                        "assumptions while keeping the BJS-calibrated baseline fixed."
                                    ),
                                    style={
                                        "color": MUTED,
                                        "fontSize": "13px",
                                        "marginTop": 0,
                                    },
                                ),
                                dcc.Graph(
                                    id="sensitivity-chart",
                                    config={"displaylogo": False},
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                **card_style(),
                                "marginTop": "20px",
                            },
                            children=[
                                html.H2(
                                    "Pathway Diagnostic",
                                    style={
                                        "fontFamily": "Georgia, serif",
                                        "marginTop": 0,
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.P(
                                    (
                                        "Tracks where the counterfactual intervention "
                                        "translates—or fails to translate—into downstream "
                                        "model mechanisms and 3-year rearrest."
                                    ),
                                    style={
                                        "color": MUTED,
                                        "fontSize": "13px",
                                        "marginTop": 0,
                                        "marginBottom": "16px",
                                    },
                                ),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns":
                                            "repeat(2, minmax(350px, 1fr))",
                                        "gap": "18px",
                                    },
                                    children=[
                                        dcc.Graph(
                                            id="pathway-employment",
                                            config={"displaylogo": False},
                                        ),
                                        dcc.Graph(
                                            id="pathway-violations",
                                            config={"displaylogo": False},
                                        ),
                                        dcc.Graph(
                                            id="pathway-absences",
                                            config={"displaylogo": False},
                                        ),
                                        dcc.Graph(
                                            id="pathway-rearrest",
                                            config={"displaylogo": False},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    (
                                        "Interpretation note: these panels are pathway "
                                        "diagnostics showing co-movement within the ABM "
                                        "experiment; they do not by themselves establish "
                                        "a causal ordering among the intermediate variables."
                                    ),
                                    style={
                                        "fontSize": "11px",
                                        "color": MUTED,
                                        "marginTop": "8px",
                                    },
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                **card_style(),
                                "marginTop": "20px",
                            },
                            children=[
                                html.H3(
                                    "Counterfactual Summary Table",
                                    style={
                                        "marginTop": 0,
                                        "fontFamily": "Georgia, serif",
                                    },
                                ),
                                dash_table.DataTable(
                                    id="summary-table",
                                    page_size=20,
                                    style_table={
                                        "overflowX": "auto"
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#EEF1EC",
                                        "border": f"1px solid {BORDER}",
                                    },
                                    style_cell={
                                        "padding": "8px",
                                        "fontSize": "12px",
                                        "textAlign": "center",
                                        "border": f"1px solid {BORDER}",
                                    },
                                ),
                            ],
                        ),
                    ]
                ),
            ],
        ),
    ],
)


# ================================================================
# CALLBACKS
# ================================================================

@app.callback(
    Output("strength", "options"),
    Output("strength", "value"),
    Output("strength-note", "children"),
    Input("experiment", "value"),
)
def update_strengths(experiment):
    strengths = available_strengths(experiment)
    options = []
    for label, beta, assumed_or in strengths:
        if label == "legacy":
            text = "Legacy output (no sensitivity dimension)"
        else:
            text = f"{label.capitalize()} — β={beta:+.3f}, OR={assumed_or:.2f}"
        options.append({"label": text, "value": label})

    default = default_strength(experiment)
    note = (
        "β values are literature-informed sensitivity assumptions, not BJS "
        "calibration parameters. Employment uses a heterogeneous evidence range; "
        "program 'central' uses the pooled rearrest OR≈0.89."
    )
    return options, default, note


@app.callback(
    Output("cf-level", "options"),
    Output("cf-level", "value"),
    Input("experiment", "value"),
)
def update_levels(experiment):
    summary, _, _ = load_experiment(experiment)

    if summary.empty:
        return [], None

    levels = sorted(
        summary["cf_level"].astype(float).unique()
    )

    options = [
        {
            "label": (
                "Baseline (0%)"
                if np.isclose(v, 0.0)
                else f"{v * 100:+.0f}%"
            ),
            "value": float(v),
        }
        for v in levels
    ]

    default = 0.0 if any(
        np.isclose(v, 0.0) for v in levels
    ) else float(levels[0])

    return options, default


@app.callback(
    Output("rate-chart", "figure"),
    Output("delta-chart", "figure"),
    Output("mechanism-chart", "figure"),
    Output("distribution-chart", "figure"),
    Output("sensitivity-chart", "figure"),
    Output("pathway-employment", "figure"),
    Output("pathway-violations", "figure"),
    Output("pathway-absences", "figure"),
    Output("pathway-rearrest", "figure"),
    Output("metric-cards", "children"),
    Output("run-info", "children"),
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Input("experiment", "value"),
    Input("cf-level", "value"),
    Input("strength", "value"),
)
def update_dashboard(experiment, level, strength):
    summary_all, paired_all, raw_all = load_experiment(experiment)

    if level is None:
        level = 0.0
    if strength is None:
        strength = default_strength(experiment)

    summary = filter_strength(summary_all, strength)
    paired = filter_strength(paired_all, strength)
    raw = filter_strength(raw_all, strength)

    rate_fig = make_rate_figure(
        summary,
        experiment,
    )
    delta_fig = make_delta_figure(
        paired,
        experiment,
    )
    mechanism_fig = make_mechanism_figure(
        summary,
        experiment,
    )
    dist_fig = make_seed_distribution(
        raw,
        experiment,
        level,
    )

    sensitivity_fig = make_sensitivity_figure(
        paired_all,
        experiment,
        years=3,
    )

    pathway_figs = make_pathway_figures(
        summary,
        experiment,
    )

    cards = []

    baseline_3 = baseline_value(
        summary,
        3,
    )

    delta3, delta3_lo, delta3_hi = selected_effect(
        paired,
        level,
        3,
    )

    selected_row = summary[
        np.isclose(
            summary["cf_level"].astype(float),
            float(level),
        )
    ]

    selected_3 = None
    n_replications = None
    mechanism_value = None

    if not selected_row.empty:
        r = selected_row.iloc[0]
        selected_3 = float(r["rearrest_3yr_mean"])
        n_replications = int(r["n_replications"])

        metric_col = (
            "mean_percent_days_employed_mean"
            if experiment == "employment"
            else "mean_program_attendances_mean"
        )

        if metric_col in r.index:
            mechanism_value = float(r[metric_col])

    card_specs = [
        (
            "3-Year Baseline",
            (
                f"{baseline_3:.1%}"
                if baseline_3 is not None
                else "—"
            ),
            "Calibrated 0% counterfactual",
        ),
        (
            "3-Year Selected Scenario",
            (
                f"{selected_3:.1%}"
                if selected_3 is not None
                else "—"
            ),
            f"{float(level) * 100:+.0f}% intervention",
        ),
        (
            "3-Year Change",
            (
                f"{delta3:+.2f} pp"
                if delta3 is not None
                else "—"
            ),
            (
                f"95% CI [{delta3_lo:+.2f}, {delta3_hi:+.2f}]"
                if delta3_lo is not None
                else "Paired baseline difference"
            ),
        ),
        (
            (
                "Avg. Days Employed"
                if experiment == "employment"
                else "Avg. Program Attendances"
            ),
            (
                f"{mechanism_value:.2f}"
                if mechanism_value is not None
                else "—"
            ),
            "Mechanism check",
        ),
    ]

    for title, value, note in card_specs:
        cards.append(
            html.Div(
                style=card_style(),
                children=[
                    html.Div(
                        title,
                        style={
                            "fontSize": "12px",
                            "color": MUTED,
                            "marginBottom": "6px",
                        },
                    ),
                    html.Div(
                        value,
                        style={
                            "fontFamily": "Georgia, serif",
                            "fontSize": "26px",
                            "fontWeight": 600,
                        },
                    ),
                    html.Div(
                        note,
                        style={
                            "fontSize": "11px",
                            "color": MUTED,
                            "marginTop": "5px",
                        },
                    ),
                ],
            )
        )

    run_info = html.Div(
        [
            html.Div(
                [
                    html.B("Source: "),
                    str(RESULTS_DIR),
                ]
            ),
            html.Div(
                [
                    html.B("Selected family: "),
                    experiment.capitalize(),
                ]
            ),
            html.Div(
                [
                    html.B("Effect assumption: "),
                    str(strength).capitalize(),
                ]
            ),
            html.Div(
                [
                    html.B("β / OR: "),
                    (
                        f"{float(summary.iloc[0]['intervention_strength']):+.3f} / "
                        f"{float(summary.iloc[0]['assumed_or']):.2f}"
                        if (not summary.empty and
                            'intervention_strength' in summary.columns and
                            'assumed_or' in summary.columns)
                        else "—"
                    ),
                ]
            ),
            html.Div(
                [
                    html.B("Replications/level: "),
                    str(n_replications or "—"),
                ]
            ),
            html.Div(
                [
                    html.B("Design: "),
                    "10 runs × 20 seeds",
                ]
            ),
            html.Div(
                [
                    html.B("Warm-up/study: "),
                    "144 / 108 months",
                ]
            ),
            html.Div(
                [
                    html.B("Matched seeds: "),
                    "Yes",
                ]
            ),
        ]
    )

    # Dissertation-friendly table.
    table = summary.copy()

    table_display = pd.DataFrame(
        {
            "Effect": (
                table["strength_label"].astype(str).str.capitalize()
                if "strength_label" in table.columns
                else "Legacy"
            ),
            "β": (
                table["intervention_strength"].map(lambda x: f"{x:+.3f}")
                if "intervention_strength" in table.columns
                else "—"
            ),
            "OR": (
                table["assumed_or"].map(lambda x: f"{x:.2f}")
                if "assumed_or" in table.columns
                else "—"
            ),
            "CF Change": table["cf_percent"].map(
                lambda x: f"{x:+.0f}%"
            ),
            "N": table["n_replications"],
            "3-Year": table["rearrest_3yr_mean"].map(
                lambda x: f"{x:.2%}"
            ),
            "3-Year 95% CI": table.apply(
                lambda r:
                    f"[{r['rearrest_3yr_ci95_low']:.2%}, "
                    f"{r['rearrest_3yr_ci95_high']:.2%}]",
                axis=1,
            ),
            "6-Year": table["rearrest_6yr_mean"].map(
                lambda x: f"{x:.2%}"
            ),
            "9-Year": table["rearrest_9yr_mean"].map(
                lambda x: f"{x:.2%}"
            ),
        }
    )

    paired_map = paired.set_index("cf_level") \
        if not paired.empty else pd.DataFrame()

    if not paired.empty:
        table_display["Δ 3-Year (pp)"] = [
            (
                f"{paired_map.loc[level_, 'delta_3yr_pp_mean']:+.2f}"
                if level_ in paired_map.index
                else "—"
            )
            for level_ in table["cf_level"]
        ]

    columns = [
        {
            "name": col,
            "id": col,
        }
        for col in table_display.columns
    ]

    return (
        rate_fig,
        delta_fig,
        mechanism_fig,
        dist_fig,
        sensitivity_fig,
        pathway_figs[0],
        pathway_figs[1],
        pathway_figs[2],
        pathway_figs[3],
        cards,
        run_info,
        table_display.to_dict("records"),
        columns,
    )


# ================================================================
# MAIN
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print()
    print("=" * 72)
    print("RECIDIVISM ABM — INTERACTIVE COUNTERFACTUAL DASHBOARD")
    print("=" * 72)
    print(f"Model results -> {RESULTS_DIR}")
    print(f"Experiments   -> {', '.join(available_experiments())}")
    print(f"Dashboard     -> http://{args.host}:{args.port}")
    print("=" * 72)
    print()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
