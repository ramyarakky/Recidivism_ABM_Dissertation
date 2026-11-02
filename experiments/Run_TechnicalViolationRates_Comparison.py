import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from recidivism_abm.model.recidivism_model import RecidivismModel

# ─────────────────────────────────────────────
# Simulation Parameters
# ─────────────────────────────────────────────
MASTER_SEED = 424242
base_params = {
    "initial_agents":     1500,
    "warmup_months":       144,
    "study_months":        108,
    "monthly_intake":       10,
    "bias_factor":           0,
    "enable_peer_influence": True,
}

RACE_GROUPS   = ["White", "Black", "Hispanic"]
GENDER_GROUPS = ["Male", "Female"]

# ─────────────────────────────────────────────
# Seed generation
# ─────────────────────────────────────────────
def generate_seeds(master_seed: int, n_runs: int, n_seeds: int) -> np.ndarray:
    rng   = np.random.default_rng(master_seed)
    seeds = rng.choice(
        np.iinfo(np.int32).max,
        size=n_runs * n_seeds,
        replace=False,
    ).astype(np.int64)
    return seeds.reshape(n_runs, n_seeds)


def export_seed_schedule(seeds_matrix: np.ndarray, master_seed: int,
                         filename: str = None) -> pd.DataFrame:
    if filename is None:
        filename = f"simulation_seeds_master_{master_seed}.csv"
    n_runs, n_seeds = seeds_matrix.shape
    df_seeds = pd.DataFrame({
        "run_id":      np.repeat(np.arange(1, n_runs + 1), n_seeds),
        "seed":        seeds_matrix.reshape(-1).astype(np.int64),
        "master_seed": master_seed,
    })
    df_seeds.to_csv(filename, index=False)
    print(f"\n  Seed schedule exported to: {filename}")
    return df_seeds


# ─────────────────────────────────────────────
# Technical violation rates
# ─────────────────────────────────────────────
def compute_tech_violation_rates(model) -> dict:
    agents   = list(getattr(model.schedule, "agents", []))
    eligible = [a for a in agents if getattr(a, "study_eligible_agent", True)]
    if not eligible:
        eligible = agents

    def rate(group_agents):
        if not group_agents:
            return None
        flagged = sum(
            1 for a in group_agents
            if bool(getattr(a, "Violations_Technical", False))
        )
        return flagged / len(group_agents)

    by_gender = {g: rate([a for a in eligible
                           if getattr(a, "Gender", None) == g])
                 for g in GENDER_GROUPS}

    by_race = {r: rate([a for a in eligible
                         if getattr(a, "Race", None) == r])
               for r in RACE_GROUPS}

    # Gender x Race intersection — key as "gender_race" string for easy CSV storage
    by_gender_race = {
        f"{g}_{r}": rate([a for a in eligible
                          if getattr(a, "Gender", None) == g
                          and getattr(a, "Race",   None) == r])
        for g in GENDER_GROUPS
        for r in RACE_GROUPS
    }

    return {
        "tech_by_gender":      by_gender,
        "tech_by_race":        by_race,
        "tech_by_gender_race": by_gender_race,
        "n_eligible":          len(eligible),
    }


# ─────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────
def run_model_for_seed(job):
    run_id, seed = job
    start_time   = time.perf_counter()

    params         = base_params.copy()
    params["seed"] = int(seed)

    model            = RecidivismModel(**params)
    model.max_months = model.warmup_months + model.study_months

    while model.running:
        model.step()

    tech_rates          = compute_tech_violation_rates(model)
    model.current_month = model.warmup_months + 108
    elapsed             = time.perf_counter() - start_time

    # Flatten gender x race keys into individual columns
    gr_cols = {f"tech_{k.lower()}": v
               for k, v in tech_rates["tech_by_gender_race"].items()}

    return {
        "run_id":      int(run_id),
        "seed":        int(seed),
        "rate_3yr":    model.calculate_flag_rate("rearrest_3_yrs"),
        "rate_6yr":    model.calculate_flag_rate("rearrest_6_yrs"),
        "rate_9yr":    model.calculate_flag_rate("rearrest_9_yrs"),
        "runtime_sec": elapsed,
        # Gender marginals
        "tech_male":     tech_rates["tech_by_gender"]["Male"],
        "tech_female":   tech_rates["tech_by_gender"]["Female"],
        # Race marginals
        "tech_white":    tech_rates["tech_by_race"]["White"],
        "tech_black":    tech_rates["tech_by_race"]["Black"],
        "tech_hispanic": tech_rates["tech_by_race"]["Hispanic"],
        # Gender x Race intersection columns
        # keys: tech_male_white, tech_male_black, tech_male_hispanic,
        #       tech_female_white, tech_female_black, tech_female_hispanic
        **gr_cols,
    }


# ─────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────
def run_simulation(master_seed: int, n_runs: int, n_seeds: int,
                   max_workers=None, verbose=True) -> pd.DataFrame:
    seeds_matrix = generate_seeds(master_seed, n_runs, n_seeds)
    export_seed_schedule(seeds_matrix, master_seed)

    jobs = [
        (run_id, int(seed))
        for run_id in range(1, n_runs + 1)
        for seed in seeds_matrix[run_id - 1]
    ]
    total_jobs = len(jobs)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    chunksize = max(1, total_jobs // (max_workers * 4))

    if verbose:
        print(f"\n  Running {total_jobs} simulations | "
              f"workers={max_workers} | chunksize={chunksize}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(run_model_for_seed, jobs,
                                    chunksize=chunksize)
        results = list(tqdm(
            results_iter, total=total_jobs,
            desc="Simulations",
            miniters=max(1, total_jobs // 100),
            smoothing=0.1,
        ))

    df = pd.DataFrame(results)
    df["master_seed"] = master_seed

    if verbose:
        runtime_summary = (
            df.groupby("run_id")["runtime_sec"]
              .mean()
              .rename("avg_runtime_sec")
              .reset_index()
        )
        print("\n  Avg runtime per run:")
        print(runtime_summary)

    return df


# ─────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────
def summarize_tech_violation_rates(df: pd.DataFrame) -> dict:
    return {
        "Gender": {
            "Male":   df["tech_male"].mean(),
            "Female": df["tech_female"].mean(),
        },
        "Race": {
            "White":    df["tech_white"].mean(),
            "Black":    df["tech_black"].mean(),
            "Hispanic": df["tech_hispanic"].mean(),
        },
        "GenderRace": {
            (g, r): df[f"tech_{g.lower()}_{r.lower()}"].mean()
            for g in GENDER_GROUPS
            for r in RACE_GROUPS
        },
    }


# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
def plot_tech_violation_by_gender(df: pd.DataFrame):
    rates = summarize_tech_violation_rates(df)["Gender"]
    fig   = go.Figure()
    fig.add_bar(
        x=list(rates.keys()),
        y=list(rates.values()),
        text=[f"{v:.1%}" for v in rates.values()],
        textposition="outside",
        marker_color=["#4C78A8", "#F58518"],
        name="Technical Violation Rate",
    )
    fig.update_layout(
        title="Detected Technical Violation Rate by Gender",
        yaxis_title="Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white",
    )
    try:
        fig.write_image("Technical_Violation_Rates_ByGender.png")
        print("\n  Chart saved: Technical_Violation_Rates_ByGender.png")
    except Exception:
        print("  Install kaleido to enable PNG export")
    fig.show()


def plot_tech_violation_by_race(df: pd.DataFrame):
    rates  = summarize_tech_violation_rates(df)["Race"]
    COLORS = {"White": "#9E9E9E", "Black": "#E45756", "Hispanic": "#72B7B2"}
    fig    = go.Figure()
    fig.add_bar(
        x=list(rates.keys()),
        y=list(rates.values()),
        text=[f"{v:.1%}" for v in rates.values()],
        textposition="outside",
        marker_color=[COLORS[r] for r in rates],
        name="Technical Violation Rate",
    )
    fig.update_layout(
        title="Detected Technical Violation Rate by Race",
        yaxis_title="Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white",
    )
    try:
        fig.write_image("Technical_Violation_Rates_ByRace.png")
        print("\n  Chart saved: Technical_Violation_Rates_ByRace.png")
    except Exception:
        print("  Install kaleido to enable PNG export")
    fig.show()


def plot_tech_violation_gender_race_heatmap(df: pd.DataFrame, n_runs: int = None):
    """
    Gender x Race intersection heatmap of technical violation rates.
    Rows = Gender (Female top, Male bottom), Cols = Race (White, Black, Hispanic).
    Matches the reference chart style with annotated percentages.
    """
    summary = summarize_tech_violation_rates(df)["GenderRace"]

    # Gender order: Male top, Female bottom (reversed so autorange works correctly)
    gender_order = list(reversed(GENDER_GROUPS))   # ["Female", "Male"]

    z_matrix = [
        [summary[(g, r)] for r in RACE_GROUPS]
        for g in gender_order
    ]
    all_vals = [summary[(g, r)] for g in GENDER_GROUPS for r in RACE_GROUPS]

    # Tight colorscale: span around actual data range so all cells get
    # a distinct colour even when values are very close together
    v_min = min(all_vals)
    v_max = max(all_vals)
    v_pad = max((v_max - v_min) * 0.5, 0.005)   # at least 0.5pp padding
    z_min = max(0.0, v_min - v_pad)
    z_max = min(1.0, v_max + v_pad)

    # Adaptive text colour: dark on light cells, white on dark cells
    midpoint = (z_min + z_max) / 2
    font_colors = [
        [
            "white" if summary[(g, r)] >= midpoint else "#1a3d5c"
            for r in RACE_GROUPS
        ]
        for g in gender_order
    ]

    n_label = f"n={len(df)} runs" if n_runs is None else f"n={n_runs} runs"
    bias    = int(base_params.get("bias_factor", 0))

    fig = go.Figure()

    # Base heatmap layer
    fig.add_trace(go.Heatmap(
        z          = z_matrix,
        x          = RACE_GROUPS,
        y          = gender_order,
        zmin       = z_min,
        zmax       = z_max,
        colorscale = "Blues",
        showscale  = True,
        colorbar   = dict(
            title      = dict(text="Rate", side="right"),
            tickformat = ".1%",
            tickvals   = [z_min, (z_min + z_max) / 2, z_max],
        ),
        hovertemplate=(
            "Gender: %{y}<br>"
            "Race: %{x}<br>"
            "Tech Violation Rate: %{z:.1%}<extra></extra>"
        ),
    ))

    # Per-cell text overlay — one Scatter trace per cell so each gets its
    # own font colour (Heatmap textfont only accepts a single global colour)
    for ri, g in enumerate(gender_order):
        for ci, r in enumerate(RACE_GROUPS):
            val  = summary[(g, r)]
            fcol = font_colors[ri][ci]
            fig.add_trace(go.Scatter(
                x=[r], y=[g],
                mode="text",
                text=[f"{val:.1%}"],
                textfont=dict(size=16, color=fcol, family="Arial Black"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(
            text=(
                "Technical Violation Rate — Gender × Race Intersection"
                f"<br><sup>bias={bias}, {n_label}</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        xaxis=dict(title="", side="bottom", tickfont=dict(size=14)),
        yaxis=dict(title="", tickfont=dict(size=14)),
        template="plotly_white",
        width=780,
        height=440,
        margin=dict(l=100, r=130, t=110, b=70),
    )

    try:
        fig.write_image("Technical_Violation_Rates_GenderRace.png")
        print("\n  Chart saved: Technical_Violation_Rates_GenderRace.png")
    except Exception:
        print("  Install kaleido to enable PNG export")
    fig.show()


def plot_clustered_bars_with_ci(df: pd.DataFrame):
    labels      = ["3-Year", "6-Year", "9-Year"]
    model_means = np.array([df["rate_3yr"].mean(),
                             df["rate_6yr"].mean(),
                             df["rate_9yr"].mean()])
    bjs         = np.array([0.68, 0.79, 0.83])
    n           = len(df)
    model_std   = np.array([df["rate_3yr"].std(ddof=1),
                             df["rate_6yr"].std(ddof=1),
                             df["rate_9yr"].std(ddof=1)])
    ci95        = 1.96 * (model_std / np.sqrt(n))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=model_means,
        name="Model mean (±95% CI)",
        marker_color="#4C78A8",
        error_y=dict(type="data", array=ci95, thickness=2),
        text=[f"{v:.0%}" for v in model_means],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=bjs,
        name="BJS observed",
        marker_color="#9E9E9E",
        text=[f"{v:.0%}" for v in bjs],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title="Recidivism ABM Model Mean vs BJS Observed Recidivism (±95% CI)",
        yaxis_title="Recidivism rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="center", x=0.5),
        bargap=0.35,
    )
    try:
        fig.write_image("recidivism_vs_bjs_targets.png")
        print("\n  Chart saved: recidivism_vs_bjs_targets.png")
    except Exception:
        print("  Install kaleido to enable PNG export")
    fig.show()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    n_seeds     = 20
    n_runs      = 10
    max_workers = os.cpu_count() or 1

    df_results = run_simulation(
        master_seed=MASTER_SEED,
        n_runs=n_runs,
        n_seeds=n_seeds,
        max_workers=max_workers,
        verbose=True,
    )

    plot_tech_violation_by_gender(df_results)
    plot_tech_violation_by_race(df_results)
    plot_tech_violation_gender_race_heatmap(df_results, n_runs=n_runs * n_seeds)

    print("\n  Total simulations:", len(df_results))
    print(df_results.head())