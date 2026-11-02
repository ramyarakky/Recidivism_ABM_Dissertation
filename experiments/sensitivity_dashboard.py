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
    non_recid = {y: 1 - recid[y] for y in TARGET_YEARS}
    return {"recid": recid, "non_recid": non_recid}

# -----------------------
# SENSITIVITY EVALUATION
# -----------------------

def evaluate_change(feature, base_weights, pct_change, seeds, agents):
    w_plus = base_weights.copy()
    w_minus = base_weights.copy()

    w_plus[feature] *= (1 + pct_change)
    w_minus[feature] *= (1 - pct_change)

    n_jobs = min(cpu_count(), len(seeds))

    results_plus = Parallel(n_jobs=n_jobs)(
        delayed(run_model)(w_plus, s, agents) for s in seeds
    )
    results_minus = Parallel(n_jobs=n_jobs)(
        delayed(run_model)(w_minus, s, agents) for s in seeds
    )

    def summarize(results, key):
        return {y: np.mean([r[key][y] for r in results]) for y in TARGET_YEARS}

    return summarize(results_plus, "recid"), summarize(results_minus, "recid"), \
           summarize(results_plus, "non_recid"), summarize(results_minus, "non_recid")

# -----------------------
# STREAMLIT DASHBOARD
# -----------------------

def streamlit_dashboard():
    st.title("📊 Sensitivity Analysis Dashboard")

    pct = st.slider("Percent Change (±)", 5, 50, 10) / 100
    agents = st.number_input("Agents", min_value=100, max_value=5000, value=1000)
    seed_start = st.number_input("Seed Start", min_value=1, max_value=999, value=10)
    seed_end = st.number_input("Seed End", min_value=seed_start+1, max_value=1000, value=15)
    seeds = list(range(seed_start, seed_end))

    base_weights = get_flat_risk_weights()

    if st.button("Run Sensitivity Analysis"):
        if not seeds:
            st.error("No seeds selected. Please set Seed Start < Seed End.")
            return

        rows = []
        progress_bar = st.progress(0, text="Running sensitivity analysis...")
        status_text = st.empty()
        log_box = st.empty()   # live log panel

        total_features = len(ALL_KEYS)
        log_messages = []
        start_time = datetime.datetime.now()

        for i, f in enumerate(ALL_KEYS, start=1):
            status_text.text(f"Evaluating feature {i}/{total_features}: {f}")
            plus_recid, minus_recid, plus_non, minus_non = evaluate_change(f, base_weights, pct, seeds, agents)

            for y in TARGET_YEARS:
                rows.append({
                    "Feature": f,
                    "Year": y,
                    "Recid_Plus": plus_recid[y],
                    "Recid_Minus": minus_recid[y],
                    "NonRecid_Plus": plus_non[y],
                    "NonRecid_Minus": minus_non[y],
                    "Impact(%)": abs(plus_recid[y] - minus_recid[y]) * 100
                })

            # Update progress bar
            progress_bar.progress(i / total_features, text=f"Completed {i}/{total_features} features")

            # Add timestamped log entry
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_messages.append(f"[{timestamp}] ✔ Finished {f}")
            log_box.text_area("Run Log", "\n".join(log_messages), height=250)

        end_time = datetime.datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        status_text.text(f"✅ Sensitivity analysis complete! Total runtime: {elapsed:.1f} seconds")

        df = pd.DataFrame(rows)
        if df.empty:
            st.warning("No results generated. Check seeds and agent count.")
            return

        st.dataframe(df)

        # Plot top 5 features per year (grouped bars)
        for y in TARGET_YEARS:
            st.subheader(f"{y}-Year Rearrest Sensitivity (Top 5 Features)")
            subset = df[df["Year"] == y].nlargest(5, "Impact(%)")
            if subset.empty:
                st.warning(f"No results for {y}-year analysis.")
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(subset["Feature"]))
            ax.bar(x - 0.3, subset["Recid_Plus"], width=0.3, label="Recidivism +")
            ax.bar(x, subset["Recid_Minus"], width=0.3, label="Recidivism -")
            ax.bar(x + 0.3, subset["NonRecid_Plus"], width=0.3, alpha=0.6, label="Non-Recidivism +")
            ax.bar(x + 0.6, subset["NonRecid_Minus"], width=0.3, alpha=0.6, label="Non-Recidivism -")

            ax.set_xticks(x)
            ax.set_xticklabels(subset["Feature"], rotation=90)
            ax.set_ylabel("Rate")
            ax.set_title(f"Sensitivity Analysis ({y}-Year)")
            ax.legend()
            st.pyplot(fig)

        # Line plots across years for top features
        st.subheader("📈 Sensitivity Evolution Across Years")
        top_features = df.groupby("Feature")["Impact(%)"].mean().nlargest(5).index

        fig, ax = plt.subplots(figsize=(12, 6))
        for f in top_features:
            feature_data = df[df["Feature"] == f].sort_values("Year")
            ax.plot(feature_data["Year"], feature_data["Recid_Plus"], marker="o", label=f"{f} (Recidivism +)")
            ax.plot(feature_data["Year"], feature_data["Recid_Minus"], marker="x", linestyle="--", label=f"{f} (Recidivism -)")
        ax.set_xlabel("Years")
        ax.set_ylabel("Recidivism Rate")
        ax.set_title("Sensitivity Evolution Across 3, 6, 9 Years")
        ax.legend()
        st.pyplot(fig)

        # Download CSV
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="sensitivity_results.csv"
        )

if __name__ == "__main__":
    streamlit_dashboard()
