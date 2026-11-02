import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os, sys, datetime

# Ensure imports work relative to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from recidivism_abm.model.recidivism_model import RecidivismModel
from recidivism_abm.config.risk_config import get_peer_influence_config, get_flat_risk_weights

# -----------------------
# CONFIG
# -----------------------

ALL_KEYS = [
    "Age_at_Release", "Percent_Days_Employed", "Dependents",
    "Education_None", "Education_LessthanHSdiploma", "Education_HighSchoolDiploma",
    "Education_Atleastsomecollege", "Gang_Affiliated", "Residence_Changes",
    "offense_Violent-SexOffender", "offense_Violent/Non-Sex", "offense_Drug",
    "offense_Property", "offense_Other(PublicOrder)", "Supervision_Risk_Score",
    "Supervision_Level_First_Standard", "Supervision_Level_First_High",
    "Supervision_Level_First_Specialized", "Prior_Conviction_Episodes_Violent",
    "Prior_Conviction_Episodes_Property", "Prior_Conviction_Episodes_Drug",
    "Prior_Conviction_Episodes_Misd", "Prior_Conviction_Episodes_Felony",
    "Prior_Revocations_Supervision", "Condition_MH_SA", "Condition_Cog_Ed",
    "Condition_Other", "Violations_Technical", "Program_Attendances",
    "Program_UnexcusedAbsences", "Threshold", "Desistance_Rate"
]

PEER_CONFIG = get_peer_influence_config()
TARGET_YEARS = [3, 6, 9]
BASELINE_RATES = {3: 0.69, 6: 0.79, 9: 0.83}  # optional visual baseline anchor

# Ensure Charts folder exists
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "Charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# -----------------------
# MODEL RUNNER
# -----------------------

def run_model(weights, seed, agents, warmup=144, study=108):
    m = RecidivismModel(
        initial_agents=agents,
        bias_factor=weights.get("Bias_Factor", 0),
        monthly_intake=10,
        warmup_months=warmup,
        study_months=study,
        enable_peer_influence=True,
        weights=weights,
        peer_config=PEER_CONFIG,
        seed=seed,
        mode="realistic"
    )
    while m.running:
        m.step()

    recid = {y: m.calculate_flag_rate(f"rearrest_{y}_yrs") for y in TARGET_YEARS}
    return {"recid": recid}

# -----------------------
# SENSITIVITY EVALUATION
# -----------------------

def evaluate_change(feature, base_weights, pct_change, seeds, agents):
    # Construct weight variants
    w_plus = base_weights.copy()
    w_minus = base_weights.copy()
    

    st.write("Available weight keys:", list(base_weights.keys()))
    # Guard for missing feature in weights
    if feature not in w_plus:
        raise KeyError(f"Feature '{feature}' not found in base weights.")

    w_plus[feature] *= (1 + pct_change)
    w_minus[feature] *= (1 - pct_change)

    n_jobs = min(cpu_count(), len(seeds))

    # Run scenarios
    results_base = Parallel(n_jobs=n_jobs)(
        delayed(run_model)(base_weights, s, agents) for s in seeds
    )
    results_plus = Parallel(n_jobs=n_jobs)(
        delayed(run_model)(w_plus, s, agents) for s in seeds
    )
    results_minus = Parallel(n_jobs=n_jobs)(
        delayed(run_model)(w_minus, s, agents) for s in seeds
    )

    # Summarize across seeds
    def summarize(results):
        return {y: float(np.mean([r["recid"][y] for r in results])) for y in TARGET_YEARS}

    return (
        summarize(results_base),
        summarize(results_plus),
        summarize(results_minus),
        float(base_weights[feature]),
        float(w_plus[feature]),
        float(w_minus[feature]),
    )

# -----------------------
# STREAMLIT DASHBOARD
# -----------------------

def streamlit_dashboard():
    st.title("📊 Sensitivity Analysis Dashboard")
    st.caption("Compare overall recidivism rates at 3, 6, and 9 years against baseline for selected features, with annotated weight values.")

    # Controls
    pct = st.slider("Percent Change (±)", 5, 50, 10, help="Adjust feature weight by ±%") / 100
    agents = st.number_input("Agents", min_value=100, max_value=20000, value=1000, step=100)
    seed_start = st.number_input("Seed Start", min_value=1, max_value=9999, value=10)
    seed_end = st.number_input("Seed End (exclusive)", min_value=seed_start + 1, max_value=10000, value=15)
    warmup = st.number_input("Warmup months", min_value=0, max_value=240, value=144)
    study = st.number_input("Study months", min_value=12, max_value=240, value=108)

    seeds = list(range(seed_start, seed_end))
    base_weights = get_flat_risk_weights()

    # Feature selection
    selected_features = st.multiselect(
        "Select up to 5 features for sensitivity analysis",
        ALL_KEYS,
        default=["Age_at_Release", "Percent_Days_Employed", "Gang_Affiliated", "Program_Attendances", "Violations_Technical","Condition_MH_SA"]
    )

    # Validation
    if len(selected_features) > 6:
        st.warning("Please select no more than 6 features.")
    run_btn = st.button("Run Sensitivity Analysis")

    if run_btn:
        if seed_start >= seed_end:
            st.error("Seed End must be greater than Seed Start.")
            return
        if len(selected_features) == 0:
            st.error("Please select at least one feature.")
            return

        rows = []
        progress = st.progress(0, text="Starting sensitivity runs...")
        status = st.empty()
        log = st.empty()
        log_lines = []
        t0 = datetime.datetime.now()
        timestamp_tag = t0.strftime("%Y%m%d_%H%M%S")

        # Evaluate each feature
        for i, f in enumerate(selected_features, start=1):
            status.text(f"Evaluating {f} ({i}/{len(selected_features)})")
            try:
                base_recid, plus_recid, minus_recid, base_w, plus_w, minus_w = evaluate_change(
                    f, base_weights, pct, seeds, agents
                )
            except Exception as e:
                st.error(f"Error evaluating feature '{f}': {e}")
                continue

            for y in TARGET_YEARS:
                rows.append({
                    "Feature": f,
                    "Year": y,
                    "Baseline": base_recid[y],
                    "Recid_Plus": plus_recid[y],
                    "Recid_Minus": minus_recid[y],
                    "Base_Weight": base_w,
                    "Weight_Plus": plus_w,
                    "Weight_Minus": minus_w
                })

            progress.progress(i / len(selected_features), text=f"Completed {i}/{len(selected_features)}")
            log_lines.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Done: {f} | BaseW={base_w:.4f}, W+={plus_w:.4f}, W-={minus_w:.4f}")
            log.text_area("Run log", "\n".join(log_lines), height=180)

        t1 = datetime.datetime.now()
        status.text(f"✅ Completed in {(t1 - t0).total_seconds():.1f}s")

        # Build DataFrame and coerce dtypes to avoid matplotlib indexing issues
        df = pd.DataFrame(rows).sort_values(["Feature", "Year"]).copy()
        df["Year"] = df["Year"].astype(int)
        for col in ["Baseline", "Recid_Plus", "Recid_Minus", "Base_Weight", "Weight_Plus", "Weight_Minus"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        st.subheader("Results table")
        st.dataframe(df, use_container_width=True)

        # -----------------------
        # Individual charts
        # -----------------------
        st.subheader("📈 Overall recidivism rates vs baseline (individual features)")

        for f in selected_features:
            feature_df = df[df["Feature"] == f].sort_values("Year").copy()
            if feature_df.empty:
                continue

            base_w = float(feature_df["Base_Weight"].iloc[0])
            plus_w = float(feature_df["Weight_Plus"].iloc[0])
            minus_w = float(feature_df["Weight_Minus"].iloc[0])

            x = feature_df["Year"].to_numpy(dtype=float)
            y_base = feature_df["Baseline"].to_numpy(dtype=float)
            y_plus = feature_df["Recid_Plus"].to_numpy(dtype=float)
            y_minus = feature_df["Recid_Minus"].to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y_base, marker="o", color="gray", label="Baseline")
            ax.plot(x, y_plus, marker="o", color="green", label=f"{f} (+{int(pct*100)}%)")
            ax.plot(x, y_minus, marker="o", color="red", label=f"{f} (-{int(pct*100)}%)")

            # Safe scalar annotations
            for xi, yi_plus, yi_minus in zip(x, y_plus, y_minus):
                ax.annotate(f"W+={plus_w:.3f}\nR={float(yi_plus):.3f}",
                            (float(xi), float(yi_plus)),
                            textcoords="offset points", xytext=(0, 10), ha="center", color="green", fontsize=9)
                ax.annotate(f"W-={minus_w:.3f}\nR={float(yi_minus):.3f}",
                            (float(xi), float(yi_minus)),
                            textcoords="offset points", xytext=(0, -20), ha="center", color="red", fontsize=9)

            ax.set_xticks(TARGET_YEARS)
            ax.set_xlabel("Years")
            ax.set_ylabel("Recidivism rate (proportion)")
            ax.set_title(f"{f} | BaseW={base_w:.3f}, W+={plus_w:.3f}, W-={minus_w:.3f}")
            ax.grid(alpha=0.2)
            ax.legend()
            st.pyplot(fig)

            # Save chart
            chart_path = os.path.join(CHARTS_DIR, f"{f}_recidivism_{timestamp_tag}.png")
            fig.savefig(chart_path, bbox_inches="tight")
            plt.close(fig)

        # -----------------------
        # Combined overlay chart
        # -----------------------
        st.subheader("📊 Combined recidivism rates vs baseline (all selected features)")
        fig, ax = plt.subplots(figsize=(12, 7))

        x_ref = np.array(TARGET_YEARS, dtype=float)
        y_ref = np.array([BASELINE_RATES[y] for y in TARGET_YEARS], dtype=float)
        ax.plot(x_ref, y_ref, marker="o", color="black", linewidth=2, label="Baseline (reference)")

        for f in selected_features:
            feature_df = df[df["Feature"] == f].sort_values("Year").copy()
            if feature_df.empty:
                continue

            x = feature_df["Year"].to_numpy(dtype=float)
            y_plus = feature_df["Recid_Plus"].to_numpy(dtype=float)
            y_minus = feature_df["Recid_Minus"].to_numpy(dtype=float)
            plus_w = float(feature_df["Weight_Plus"].iloc[0])
            minus_w = float(feature_df["Weight_Minus"].iloc[0])

            ax.plot(x, y_plus, marker="o", linestyle="-",
                    label=f"{f} (+{int(pct*100)}%, W={plus_w:.3f})")
            ax.plot(x, y_minus, marker="x", linestyle="--",
                    label=f"{f} (-{int(pct*100)}%, W={minus_w:.3f})")

        ax.set_xticks(TARGET_YEARS)
        ax.set_xlabel("Years")
        ax.set_ylabel("Recidivism rate (proportion)")
        ax.set_title("Combined sensitivity overlay")
        ax.grid(alpha=0.2)
        ax.legend(ncol=2)
        st.pyplot(fig)

        combined_chart_path = os.path.join(CHARTS_DIR, f"combined_recidivism_{timestamp_tag}.png")
        fig.savefig(combined_chart_path, bbox_inches="tight")
        plt.close(fig)

        # Download CSV
        st.download_button(
            "Download results CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"recidivism_results_{timestamp_tag}.csv"
        )

if __name__ == "__main__":
    streamlit_dashboard()
