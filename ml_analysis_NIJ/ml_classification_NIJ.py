#!/usr/bin/env python3
"""
Dissertation-Ready ML Classification Benchmark
================================================

Purpose
-------
Evaluate multiple machine-learning classifiers on the NIJ Recidivism Challenge
dataset using a rigorous train/test + cross-validation design.

Design
------
1. Load and clean the NIJ dataset.
2. Prediction horizons:
      - 1-Year cumulative rearrest
      - 2-Year cumulative rearrest
      - 3-Year cumulative rearrest
3. Remove identifiers, competition split indicator, and future recidivism
   outcome columns to prevent target leakage.
4. Create an explicit stratified 80/20 train/holdout-test split.
5. Run 5-fold Stratified Cross-Validation ONLY on the training set.
6. Fit each model on the entire training set.
7. Generate actual predictions on the untouched holdout test set.
8. Compare models using:
      - Accuracy
      - Balanced Accuracy
      - Precision
      - Recall / Sensitivity
      - Specificity
      - F1
      - ROC-AUC
      - PR-AUC / Average Precision
      - Brier Score
9. Evaluate subgroup fairness for Race and Gender using:
      - Disparate Impact Ratio
      - Demographic Parity Difference
      - TPR Difference
      - FPR Difference
      - subgroup ROC-AUC
      - subgroup Brier Score
10. Save dissertation-ready tables and high-resolution figures.

Models
------
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- Naive Bayes
- Support Vector Machine
- XGBoost (when installed)

Usage
-----
python ml_classification_dissertation.py

or

python ml_classification_dissertation.py \
    --input "/path/to/NIJ_s_Recidivism_Challenge_Full_Dataset.csv" \
    --output "ml_classification_output"

Replot existing results without retraining:
python ml_classification_dissertation.py \
    --output "ml_classification_output" \
    --replot

Optional XGBoost installation:
python -m pip install xgboost
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*default behavior of `mode` typically preserves the axis.*",
    category=FutureWarning,
)
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_DATA = "NIJ_s_Recidivism_Challenge_Full_Dataset.csv"
HORIZONS = {
    "1_year": {
        "label": "1-Year",
        "description": "Rearrest within the first year",
    },
    "2_year": {
        "label": "2-Year",
        "description": "Cumulative rearrest within the first two years",
    },
    "3_year": {
        "label": "3-Year",
        "description": "Cumulative rearrest within the first three years",
    },
}
CURRENT_HORIZON_LABEL = "3-Year"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
BOOTSTRAP_REPS = 1000

FAIRNESS_DIMENSIONS = {
    "Race": "WHITE",
    "Gender": "M",
}

FAIRNESS_DISPLAY_LABELS = {
    "BLACK": "Black",
    "WHITE": "White",
    "F": "Female",
    "M": "Male",
}

# These columns are excluded because they identify records, define the
# original challenge split, or contain future recidivism outcomes.
LEAKAGE_OR_NONPREDICTOR_COLUMNS = [
    "ID",
    "Training_Sample",
    "Recidivism_Within_3years",
    "Recidivism_Arrest_Year1",
    "Recidivism_Arrest_Year2",
    "Recidivism_Arrest_Year3",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dissertation-ready classification benchmark for NIJ recidivism data."
    )
    parser.add_argument(
        "--data", "--input",
        dest="data",
        default=DEFAULT_DATA,
        help="Path to NIJ CSV file. --input is accepted as an alias for --data.",
    )
    parser.add_argument(
        "--output",
        default="ml_classification_output",
        help="Output directory.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Holdout test proportion (default: 0.20).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=N_SPLITS,
        help="Number of stratified CV folds on training data (default: 5).",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=BOOTSTRAP_REPS,
        help="Bootstrap repetitions for 95%% CI on test ROC-AUC/F1 (default: 1000).",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help=(
            "Recreate dissertation tables and figures from previously saved "
            "CSV prediction/result files without rerunning model training."
        ),
    )
    return parser.parse_args()


def make_output_dirs(root: Path):
    dirs = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "predictions": root / "predictions",
        "reports": root / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_one_hot_encoder():
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def get_probability_scores(fitted_pipeline, X):
    """
    Return a score suitable for ROC/PR analysis.
    Prefer predict_proba; fall back to decision_function.
    """
    if hasattr(fitted_pipeline, "predict_proba"):
        return fitted_pipeline.predict_proba(X)[:, 1]

    if hasattr(fitted_pipeline, "decision_function"):
        scores = fitted_pipeline.decision_function(X)
        # Map arbitrary decision scores to [0,1] only for presentation/Brier fallback.
        return 1.0 / (1.0 + np.exp(-scores))

    return fitted_pipeline.predict(X).astype(float)


def bootstrap_metric_ci(
    y_true,
    y_pred,
    y_prob,
    metric_name,
    n_bootstrap=1000,
    seed=42,
):
    """Nonparametric bootstrap 95% confidence interval."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]

        # ROC-AUC is undefined if a bootstrap sample contains only one class.
        if len(np.unique(yt)) < 2 and metric_name == "roc_auc":
            continue

        if metric_name == "roc_auc":
            value = roc_auc_score(yt, y_prob[idx])
        elif metric_name == "f1":
            value = f1_score(yt, y_pred[idx], zero_division=0)
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric_name}")

        values.append(value)

    if not values:
        return np.nan, np.nan

    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def save_dataframe_table_image(
    df,
    path,
    title,
    decimals=3,
    max_rows=20,
):
    """
    Save a clean publication-style table as PNG.
    Intended for compact dissertation/result tables.
    """
    table_df = df.head(max_rows).copy()

    for col in table_df.columns:
        if pd.api.types.is_numeric_dtype(table_df[col]):
            table_df[col] = table_df[col].map(
                lambda x: f"{x:.{decimals}f}" if pd.notna(x) else ""
            )

    n_rows, n_cols = table_df.shape
    fig_width = max(10, n_cols * 1.55)
    fig_height = max(3.0, 0.48 * n_rows + 1.7)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)

    # Bold header and alternating row emphasis without hardcoding a palette.
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _tight_axis_limits(values, lower_bound=None, upper_bound=None, pad_fraction=0.10, min_pad=0.01, include=None):
    """Return compact plot limits with small padding while preserving meaningful reference values."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if include is not None:
        extra = np.atleast_1d(include).astype(float)
        extra = extra[np.isfinite(extra)]
        if extra.size:
            arr = np.concatenate([arr, extra]) if arr.size else extra
    if arr.size == 0:
        return lower_bound, upper_bound

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    pad = max(span * pad_fraction, min_pad)
    if span == 0:
        pad = max(abs(lo) * pad_fraction, min_pad)

    lo -= pad
    hi += pad
    if lower_bound is not None:
        lo = max(float(lower_bound), lo)
    if upper_bound is not None:
        hi = min(float(upper_bound), hi)
    if hi <= lo:
        hi = lo + max(min_pad * 2, 0.02)
    return lo, hi


def _best_model_map_text(best_models: pd.DataFrame, horizon_order=None):
    """Create a dissertation-friendly horizon -> selected model label."""
    if best_models is None or best_models.empty:
        return "Selected model unavailable"
    d = best_models.copy()
    if horizon_order is not None:
        order = {h: i for i, h in enumerate(horizon_order)}
        d["__order"] = d["Horizon"].map(order).fillna(len(order))
        d = d.sort_values("__order")
    return "; ".join(
        f"{row['Horizon']}: {row['Model']}" for _, row in d.iterrows()
    )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def _to_binary_series(series: pd.Series, name: str) -> pd.Series:
    """Robustly convert an NIJ outcome column to binary 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="coerce")
        # NIJ outcome fields are expected to be 0/1 or NaN.
        return out.astype("Int64")

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "nan": np.nan,
            "none": np.nan,
            "": np.nan,
        })
    )
    if normalized.isna().all() and series.notna().any():
        raise ValueError(f"Outcome column '{name}' could not be mapped to binary 0/1.")
    return normalized.astype("Int64")


def derive_horizon_target(df: pd.DataFrame, horizon_key: str):
    """
    Derive cumulative NIJ rearrest outcomes.

    NIJ provides year-specific rearrest indicators for Years 1-3 plus
    Recidivism_Within_3years. We therefore analyze three empirically supported
    follow-up horizons:

      1-Year: Year1
      2-Year: Year1 OR Year2
      3-Year: Recidivism_Within_3years

    For 2-Year, a record is positive if rearrest occurred in either Year 1 or
    Year 2. For 3-Year, the challenge's cumulative 3-year indicator is used.
    """
    required = [
        "Recidivism_Arrest_Year1",
        "Recidivism_Arrest_Year2",
        "Recidivism_Arrest_Year3",
        "Recidivism_Within_3years",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "NIJ dataset is missing required rearrest outcome columns: "
            + ", ".join(missing)
        )

    y1 = _to_binary_series(df["Recidivism_Arrest_Year1"], "Recidivism_Arrest_Year1")
    y2 = _to_binary_series(df["Recidivism_Arrest_Year2"], "Recidivism_Arrest_Year2")
    y3_cumulative = _to_binary_series(
        df["Recidivism_Within_3years"], "Recidivism_Within_3years"
    )

    if horizon_key == "1_year":
        y = y1
        source = "Recidivism_Arrest_Year1"
    elif horizon_key == "2_year":
        # Preserve missingness only when both component fields are missing.
        work = pd.concat([y1, y2], axis=1)
        y = work.max(axis=1, skipna=True).astype("Int64")
        y[work.isna().all(axis=1)] = pd.NA
        source = "Recidivism_Arrest_Year1 OR Recidivism_Arrest_Year2"
    elif horizon_key == "3_year":
        y = y3_cumulative
        source = "Recidivism_Within_3years"
    else:
        raise ValueError(f"Unknown horizon_key: {horizon_key}")

    return y, source


def load_and_prepare_data(data_path: Path, horizon_key: str):
    df = pd.read_csv(data_path)

    y_raw, target_source = derive_horizon_target(df, horizon_key)
    valid = y_raw.notna()

    df_h = df.loc[valid].copy()
    y = y_raw.loc[valid].astype(int)

    exclude = [
        c for c in LEAKAGE_OR_NONPREDICTOR_COLUMNS if c in df_h.columns
    ]

    X = df_h.drop(columns=exclude).copy()

    # Drop completely empty predictor columns if any.
    all_null = X.columns[X.isna().all()].tolist()
    if all_null:
        X = X.drop(columns=all_null)

    return df_h, X, y, exclude, all_null, target_source

def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    categorical_features = [
        c for c in X.columns if c not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", safe_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb_status = "included"
    except ImportError:
        xgb_status = "not installed; skipped"

    return models, xgb_status



# ---------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------

def _safe_rate(num, den):
    return float(num / den) if den else np.nan


def subgroup_fairness_metrics(
    protected_values,
    y_true,
    y_pred,
    y_prob,
    reference_group,
    dimension,
    model_name,
):
    """
    Calculate subgroup fairness metrics relative to a specified reference group.

    Positive class in this analysis is predicted rearrest, an adverse outcome.
    Accordingly, Disparate Impact Ratio (DIR) is interpreted as the relative
    predicted-positive/rearrest classification rate:
        DIR = subgroup predicted-positive rate / reference predicted-positive rate.

    Also reports:
      - Demographic parity difference
      - TPR difference (equal opportunity difference)
      - FPR difference
      - subgroup ROC-AUC
      - subgroup Brier score
    """
    normalized_groups = (
        pd.Series(protected_values)
        .astype(str)
        .str.strip()
        .str.upper()
    )

    reference_group = str(reference_group).strip().upper()

    work = pd.DataFrame({
        "Group": normalized_groups.values,
        "Observed": np.asarray(y_true, dtype=int),
        "Predicted": np.asarray(y_pred, dtype=int),
        "Probability": np.asarray(y_prob, dtype=float),
    })

    rows = []
    for group, g in work.groupby("Group", sort=True):
        y = g["Observed"].to_numpy(dtype=int)
        pred = g["Predicted"].to_numpy(dtype=int)
        prob = g["Probability"].to_numpy(dtype=float)

        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        predicted_positive_rate = float(np.mean(pred))
        observed_positive_rate = float(np.mean(y))
        tpr = _safe_rate(tp, tp + fn)
        fpr = _safe_rate(fp, fp + tn)
        auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
        brier = brier_score_loss(y, prob)

        rows.append({
            "Model": model_name,
            "Dimension": dimension,
            "Group": group,
            "Reference_Group": reference_group,
            "N": int(len(g)),
            "Observed_Rearrest_Rate": observed_positive_rate,
            "Predicted_Positive_Rate": predicted_positive_rate,
            "TPR": tpr,
            "FPR": fpr,
            "ROC_AUC": auc,
            "Brier": brier,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    out = pd.DataFrame(rows)

    ref = out[out["Group"] == str(reference_group)]
    if ref.empty:
        raise ValueError(
            f"Reference group '{reference_group}' not found for fairness dimension '{dimension}'. "
            f"Available groups: {out['Group'].tolist()}"
        )

    ref_rate = float(ref["Predicted_Positive_Rate"].iloc[0])
    ref_tpr = float(ref["TPR"].iloc[0])
    ref_fpr = float(ref["FPR"].iloc[0])

    out["Reference_Predicted_Positive_Rate"] = ref_rate
    out["Disparate_Impact_Ratio"] = (
        out["Predicted_Positive_Rate"] / ref_rate
        if ref_rate != 0 else np.nan
    )
    out["Demographic_Parity_Difference"] = (
        out["Predicted_Positive_Rate"] - ref_rate
    )
    out["TPR_Difference"] = out["TPR"] - ref_tpr
    out["FPR_Difference"] = out["FPR"] - ref_fpr

    out["Group_Display"] = (
        out["Group"]
        .map(FAIRNESS_DISPLAY_LABELS)
        .fillna(out["Group"])
    )
    out["Reference_Group_Display"] = (
        out["Reference_Group"]
        .map(FAIRNESS_DISPLAY_LABELS)
        .fillna(out["Reference_Group"])
    )

    return out


def bootstrap_fairness_ci(
    protected_values,
    y_true,
    y_pred,
    y_prob,
    reference_group,
    dimension,
    model_name,
    n_boot=500,
    seed=42,
):
    """
    Bootstrap 95% CIs for the comparative fairness metrics.
    Resampling is done at the individual-row level within the holdout set.
    """
    protected = np.asarray(
        pd.Series(protected_values)
        .astype(str)
        .str.strip()
        .str.upper()
    )
    reference_group = str(reference_group).strip().upper()
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    base = subgroup_fairness_metrics(
        protected, y_true, y_pred, y_prob,
        reference_group, dimension, model_name
    )

    metric_names = [
        "Disparate_Impact_Ratio",
        "Demographic_Parity_Difference",
        "TPR_Difference",
        "FPR_Difference",
        "ROC_AUC",
        "Brier",
    ]

    rng = np.random.default_rng(seed)
    values = {
        (group, metric): []
        for group in base["Group"].astype(str)
        for metric in metric_names
    }

    n = len(y_true)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        try:
            boot = subgroup_fairness_metrics(
                protected[idx],
                y_true[idx],
                y_pred[idx],
                y_prob[idx],
                reference_group,
                dimension,
                model_name,
            )
        except ValueError:
            continue

        for _, row in boot.iterrows():
            group = str(row["Group"])
            for metric in metric_names:
                val = row[metric]
                if pd.notna(val) and np.isfinite(val):
                    values[(group, metric)].append(float(val))

    for metric in metric_names:
        lows = []
        highs = []
        for group in base["Group"].astype(str):
            arr = np.asarray(values[(group, metric)], dtype=float)
            if len(arr):
                lo, hi = np.percentile(arr, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            lows.append(lo)
            highs.append(hi)
        base[f"{metric}_CI95_Low"] = lows
        base[f"{metric}_CI95_High"] = highs

    return base


def plot_fairness_disparate_impact(fair_df, dimension, reference_group, path):
    d = fair_df[fair_df["Dimension"] == dimension].copy()
    if d.empty:
        return

    # Average across models for a high-level dissertation comparison.
    agg = (
        d.groupby("Group", as_index=False)
        .agg(
            DIR_Mean=("Disparate_Impact_Ratio", "mean"),
            DIR_SD=("Disparate_Impact_Ratio", "std"),
        )
        .sort_values("DIR_Mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9.5, max(5.8, 0.55 * len(agg) + 2.5)))
    ax.barh(
        agg["Group"].map(FAIRNESS_DISPLAY_LABELS).fillna(agg["Group"]),
        agg["DIR_Mean"],
        xerr=agg["DIR_SD"].fillna(0),
        capsize=3,
    )
    ax.axvline(1.0, linestyle="--", linewidth=1.3, label="Parity (DIR = 1.0)")
    ax.set_xlabel("Disparate Impact Ratio")
    ax.set_title(
        f"NIJ {CURRENT_HORIZON_LABEL} Recidivism: Disparate Impact Ratio by {dimension}",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)

    fig.text(
        0.5, 0.01,
        f"Bars show the mean across classifiers; error bars show between-model SD. "
        f"Reference group: {reference_group}. Positive classification = predicted rearrest.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_fairness_rate_differences(fair_df, dimension, reference_group, path):
    d = fair_df[fair_df["Dimension"] == dimension].copy()
    if d.empty:
        return

    agg = (
        d.groupby("Group", as_index=False)
        .agg(
            DPD=("Demographic_Parity_Difference", "mean"),
            TPR_Diff=("TPR_Difference", "mean"),
            FPR_Diff=("FPR_Difference", "mean"),
        )
    )

    groups = agg["Group"].map(FAIRNESS_DISPLAY_LABELS).fillna(agg["Group"]).tolist()
    x = np.arange(len(groups))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.bar(x - width, agg["DPD"], width, label="Demographic parity difference")
    ax.bar(x, agg["TPR_Diff"], width, label="TPR difference")
    ax.bar(x + width, agg["FPR_Diff"], width, label="FPR difference")
    ax.axhline(0.0, linestyle="--", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Difference from reference group")
    ax.set_title(
        f"NIJ {CURRENT_HORIZON_LABEL} Recidivism: Fairness Rate Differences by {dimension}",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=1)

    fig.text(
        0.5, 0.01,
        f"Values are averaged across classifiers. Zero indicates parity with {reference_group}. "
        f"Positive FPR differences indicate more false-positive rearrest classifications than the reference group.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_subgroup_auc_brier(fair_df, dimension, path):
    d = fair_df[fair_df["Dimension"] == dimension].copy()
    if d.empty:
        return

    agg = (
        d.groupby("Group", as_index=False)
        .agg(
            ROC_AUC=("ROC_AUC", "mean"),
            Brier=("Brier", "mean"),
        )
        .sort_values("ROC_AUC", ascending=False)
    )

    x = np.arange(len(agg))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    ax.bar(x - width/2, agg["ROC_AUC"], width, label="Subgroup ROC-AUC")
    ax.bar(x + width/2, agg["Brier"], width, label="Subgroup Brier")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["Group"].map(FAIRNESS_DISPLAY_LABELS).fillna(agg["Group"]), rotation=20, ha="right")
    score_values = pd.concat([agg["ROC_AUC"], agg["Brier"]], ignore_index=True)
    y_lo, y_hi = _tight_axis_limits(score_values, lower_bound=0.0, upper_bound=1.0, pad_fraction=0.08, min_pad=0.02)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Score")
    ax.set_title(
        f"NIJ {CURRENT_HORIZON_LABEL} Recidivism: Subgroup Discrimination and Calibration by {dimension}",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.text(
        0.5, 0.01,
        "ROC-AUC: higher is better. Brier score: lower is better. Bars show means across classifiers.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_fairness_heatmap(fair_df, dimension, path):
    """
    Heatmap of fairness metrics averaged across models for each subgroup.
    """
    d = fair_df[fair_df["Dimension"] == dimension].copy()
    if d.empty:
        return

    metrics = [
        "Disparate_Impact_Ratio",
        "Demographic_Parity_Difference",
        "TPR_Difference",
        "FPR_Difference",
        "ROC_AUC",
        "Brier",
    ]
    labels = [
        "DIR",
        "DP difference",
        "TPR difference",
        "FPR difference",
        "ROC-AUC",
        "Brier ↓",
    ]

    agg = d.groupby("Group", as_index=False)[metrics].mean()
    matrix = agg[metrics].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 0.55 * len(agg) + 2.4)))
    image = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels(agg["Group"].map(FAIRNESS_DISPLAY_LABELS).fillna(agg["Group"]))
    ax.set_title(
        f"NIJ {CURRENT_HORIZON_LABEL} Recidivism: Fairness Summary by {dimension}",
        fontweight="bold",
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=8.5,
                    color="white" if abs(val) > 0.75 else "black",
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Metric value")

    fig.text(
        0.5, 0.01,
        "Values are mean subgroup metrics across classifiers. DIR parity = 1.0; rate-difference parity = 0.0; "
        "higher ROC-AUC and lower Brier are preferable.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_fairness_dissertation_table(fair_df, tables_dir):
    cols = [
        "Model", "Dimension", "Group", "Group_Display",
        "Reference_Group", "Reference_Group_Display", "N",
        "Observed_Rearrest_Rate", "Predicted_Positive_Rate",
        "Disparate_Impact_Ratio", "Demographic_Parity_Difference",
        "TPR", "TPR_Difference", "FPR", "FPR_Difference",
        "ROC_AUC", "Brier",
    ]
    available = [c for c in cols if c in fair_df.columns]
    out = fair_df[available].copy()
    out.to_csv(tables_dir / "fairness_suite_all_models.csv", index=False)

    for dimension in FAIRNESS_DIMENSIONS:
        d = out[out["Dimension"] == dimension].copy()
        d.to_csv(
            tables_dir / f"fairness_suite_{dimension.lower()}.csv",
            index=False,
        )

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

def plot_class_distribution(y_train, y_test, path):
    rows = []
    for split_name, y_split in [("Training", y_train), ("Holdout Test", y_test)]:
        counts = pd.Series(y_split).value_counts().sort_index()
        total = len(y_split)
        for cls in [0, 1]:
            rows.append(
                {
                    "Split": split_name,
                    "Class": "No Rearrest" if cls == 0 else "Rearrest",
                    "Count": int(counts.get(cls, 0)),
                    "Percent": 100 * counts.get(cls, 0) / total,
                }
            )

    plot_df = pd.DataFrame(rows)

    x = np.arange(2)
    width = 0.35

    train_pct = (
        plot_df[plot_df["Split"] == "Training"]
        .set_index("Class")
        .reindex(["No Rearrest", "Rearrest"])["Percent"]
        .values
    )
    test_pct = (
        plot_df[plot_df["Split"] == "Holdout Test"]
        .set_index("Class")
        .reindex(["No Rearrest", "Rearrest"])["Percent"]
        .values
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width / 2, train_pct, width, label="Training")
    ax.bar(x + width / 2, test_pct, width, label="Holdout Test")

    ax.set_xticks(x)
    ax.set_xticklabels(["No Rearrest", "Rearrest"])
    ax.set_ylabel("Percent of observations")
    ax.set_title("Class Distribution After Stratified 80/20 Split", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_df


def plot_cv_vs_test(results_df, metric, path):
    cv_col = f"CV_{metric}_Mean"
    test_col = f"Test_{metric}"

    plot_df = results_df.sort_values(test_col, ascending=True)
    y = np.arange(len(plot_df))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.barh(y - height / 2, plot_df[cv_col], height, label="5-fold CV (training)")
    ax.barh(y + height / 2, plot_df[test_col], height, label="Holdout test")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Model"])
    axis_values = pd.concat([plot_df[cv_col], plot_df[test_col]], ignore_index=True)
    floor = 0.50 if metric.upper() == "ROC_AUC" else 0.0
    x_lo, x_hi = _tight_axis_limits(axis_values, lower_bound=floor, upper_bound=1.0, pad_fraction=0.10, min_pad=0.015)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(
        f"Cross-Validation vs Holdout Test {metric.replace('_', ' ')}",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(results_df, path):
    """
    Dissertation-style grouped performance chart modeled after the primary
    dissertation ML analysis figure. Higher values are better.
    """
    metrics = [
        ("Test_ROC_AUC", "ROC-AUC"),
        ("Test_PR_AUC", "PR-AUC"),
        ("Test_F1", "F1"),
        ("Test_Accuracy", "Accuracy"),
    ]

    d = results_df.copy()
    # Preserve a consistent model order while keeping all available models.
    d = d.sort_values("Test_ROC_AUC", ascending=False).reset_index(drop=True)

    x = np.arange(len(d))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    for i, (metric, label) in enumerate(metrics):
        if metric not in d.columns:
            continue
        ax.bar(
            x + (i - (len(metrics) - 1) / 2) * width,
            d[metric].astype(float),
            width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(d["Model"], rotation=28, ha="right")
    plotted_values = []
    for metric, _ in metrics:
        if metric in d.columns:
            plotted_values.extend(pd.to_numeric(d[metric], errors="coerce").dropna().tolist())
    y_lo, y_hi = _tight_axis_limits(plotted_values, lower_bound=0.0, upper_bound=1.0, pad_fraction=0.10, min_pad=0.02)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Predictive Performance: NIJ {CURRENT_HORIZON_LABEL} Recidivism",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.01,
        "Higher values indicate better performance. All values shown are from the untouched 20% holdout test set.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_cv_metrics_heatmap(results_df, path):
    """
    Dissertation-ready heatmap of mean 5-fold training-set CV metrics.
    Brier is displayed with the other metrics but lower values are better.
    """
    metric_columns = [
        "CV_Accuracy_Mean",
        "CV_Precision_Mean",
        "CV_Recall_Mean",
        "CV_F1_Mean",
        "CV_ROC_AUC_Mean",
        "CV_PR_AUC_Mean",
        "CV_Brier_Mean",
    ]
    labels = [
        "Accuracy",
        "Precision",
        "Sensitivity",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "Brier ↓",
    ]

    available = [(c, l) for c, l in zip(metric_columns, labels) if c in results_df.columns]
    if not available:
        return

    d = results_df.copy()
    d = d.sort_values("Test_ROC_AUC", ascending=False).reset_index(drop=True)

    cols = [c for c, _ in available]
    display_labels = [l for _, l in available]
    matrix = d[cols].astype(float).to_numpy()

    fig_height = max(6.5, 0.55 * len(d) + 2.5)
    fig, ax = plt.subplots(figsize=(12.8, fig_height))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["Model"].tolist())
    ax.set_title(
        f"NIJ {CURRENT_HORIZON_LABEL} Recidivism: 5-Fold Cross-Validation Performance",
        fontweight="bold",
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color="white" if val < 0.30 or val > 0.78 else "black",
                    fontsize=8.5,
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean CV score")

    fig.text(
        0.5,
        0.01,
        "Stratified 5-fold cross-validation uses the 80% training set only. Higher is better except Brier (lower is better).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_cv_roc_auc_stability(results_df, path):
    """Plot mean training-set CV ROC-AUC with SD across the five folds."""
    if not {"CV_ROC_AUC_Mean", "CV_ROC_AUC_SD"}.issubset(results_df.columns):
        return

    d = (
        results_df.copy()
        .sort_values("CV_ROC_AUC_Mean", ascending=False)
        .reset_index(drop=True)
    )

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10.8, max(6.5, 0.50 * len(d) + 2.0)))
    ax.errorbar(
        d["CV_ROC_AUC_Mean"].astype(float),
        y,
        xerr=d["CV_ROC_AUC_SD"].fillna(0).astype(float),
        fmt="o",
        capsize=4,
        linewidth=1.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(d["Model"])
    ax.set_xlabel("Cross-validation ROC-AUC (mean ± SD)")
    ax.set_title(
        f"NIJ {CURRENT_HORIZON_LABEL} Recidivism: Model Stability Across CV Folds",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    fig.text(
        0.5,
        0.01,
        "Error bars show standard deviation across the five stratified training-set validation folds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_brier_ranked(results_df, path):
    """Rank models by holdout Brier score; lower is better."""
    if "Test_Brier" not in results_df.columns:
        return

    d = results_df.sort_values("Test_Brier", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    ax.barh(d["Model"], d["Test_Brier"].astype(float))
    ax.set_xlabel("Brier Score")
    ax.set_title(
        f"Probability Calibration Error: NIJ {CURRENT_HORIZON_LABEL} Recidivism",
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8.5)

    fig.text(
        0.5,
        0.01,
        "Lower Brier scores indicate more accurate probabilistic predictions on the untouched holdout test set.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)

def plot_roc_curves(curve_store, path):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    for model_name, values in curve_store.items():
        fpr, tpr = values["roc"]
        auc_value = values["auc"]
        ax.plot(fpr, tpr, linewidth=1.7, label=f"{model_name} ({auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves: NIJ {CURRENT_HORIZON_LABEL} Recidivism Prediction", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Curves are based exclusively on predictions for the untouched 20% holdout test set.",
        ha="center", fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(curve_store, prevalence, path):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    for model_name, values in curve_store.items():
        precision, recall = values["pr"]
        ap = values["ap"]
        ax.plot(recall, precision, linewidth=1.7, label=f"{model_name} ({ap:.3f})")

    ax.axhline(prevalence, linestyle="--", linewidth=1, label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision–Recall Curves: NIJ {CURRENT_HORIZON_LABEL} Recidivism Prediction",
        fontweight="bold",
    )
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Dashed baseline represents rearrest prevalence in the untouched holdout test set.",
        ha="center", fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, model_name, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))

    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Rearrest", "Rearrest"])
    ax.set_yticklabels(["No Rearrest", "Rearrest"])
    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("Observed Outcome")
    ax.set_title(
        f"{model_name}: NIJ {CURRENT_HORIZON_LABEL} Recidivism Confusion Matrix",
        fontweight="bold",
    )

    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            # Use high-contrast black/white text only.
            text_color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

    plt.tight_layout()
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(calibration_store, path):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    for model_name, (mean_pred, frac_pos) in calibration_store.items():
        ax.plot(mean_pred, frac_pos, marker="o", markersize=3, linewidth=1.4, label=model_name)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rearrest rate")
    ax.set_title(
        f"Calibration Curves: Predicted vs. Observed NIJ {CURRENT_HORIZON_LABEL} Recidivism",
        fontweight="bold",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Calibration is estimated from quantile bins on the untouched holdout test set; the dashed diagonal indicates perfect calibration.",
        ha="center", fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def get_transformed_feature_names(fitted_preprocessor):
    try:
        return fitted_preprocessor.get_feature_names_out()
    except Exception:
        return None


def extract_feature_importance(fitted_pipeline, model_name):
    """
    Extract feature importance/absolute coefficient where supported.
    Returns None for models without a straightforward importance measure.
    """
    model = fitted_pipeline.named_steps["classifier"]
    prep = fitted_pipeline.named_steps["preprocessor"]
    names = get_transformed_feature_names(prep)

    if names is None:
        return None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        importance_type = "Feature importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).ravel()
        importance_type = "Absolute coefficient"
    else:
        return None

    if len(values) != len(names):
        return None

    out = pd.DataFrame(
        {
            "Feature": names,
            "Importance": values,
            "Importance_Type": importance_type,
            "Model": model_name,
        }
    )

    # Improve labels.
    out["Feature"] = (
        out["Feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )

    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def plot_top_features(feature_df, model_name, path, top_n=15):
    top = feature_df.head(top_n).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_xlabel(top["Importance_Type"].iloc[0])
    ax.set_title(
        f"Top Predictors: NIJ {CURRENT_HORIZON_LABEL} Recidivism — {model_name}",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Larger values indicate greater model-specific predictive contribution.",
        ha="center", fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Replot mode
# ---------------------------------------------------------------------

def model_name_from_prediction_filename(path: Path):
    """Recover display name by matching saved model names to filename."""
    stem = path.stem.replace("_holdout_predictions", "")
    known = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient Boosting",
        "adaboost": "AdaBoost",
        "k_nearest_neighbors": "K-Nearest Neighbors",
        "naive_bayes": "Naive Bayes",
        "support_vector_machine": "Support Vector Machine",
        "xgboost": "XGBoost",
    }
    return known.get(stem, stem.replace("_", " ").title())


def replot_from_saved_results(dirs):
    """
    Recreate figures/tables from saved analysis outputs.

    No model is fitted and no cross-validation/bootstrap analysis is rerun.
    ROC, PR, calibration and confusion-matrix figures are reconstructed from
    the saved holdout prediction files. Feature-importance plots are rebuilt
    from saved feature-importance CSV files.
    """
    print("\n" + "=" * 88)
    print(" REPLOT MODE — USING SAVED RESULTS ONLY")
    print("=" * 88)

    results_path = dirs["tables"] / "model_performance_full.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Cannot replot because this file is missing:\n  {results_path}\n"
            "Run the full analysis once before using --replot."
        )

    results_df = pd.read_csv(results_path)

    required = {
        "Model",
        "CV_ROC_AUC_Mean",
        "CV_F1_Mean",
        "Test_ROC_AUC",
        "Test_F1",
        "Test_Precision",
        "Test_Recall",
        "Test_Specificity",
    }
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(
            "Saved model-performance file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    results_df = results_df.sort_values(
        ["Test_ROC_AUC", "Test_F1"], ascending=False
    ).reset_index(drop=True)

    # Ensure rank reflects current sorting.
    if "Rank_ROC_AUC" in results_df.columns:
        results_df["Rank_ROC_AUC"] = np.arange(1, len(results_df) + 1)
    else:
        results_df.insert(0, "Rank_ROC_AUC", np.arange(1, len(results_df) + 1))

    results_df.to_csv(results_path, index=False)

    # Dissertation table
    compact_cols = [
        "Rank_ROC_AUC",
        "Model",
        "CV_ROC_AUC_Mean",
        "CV_ROC_AUC_SD",
        "Test_ROC_AUC",
        "Test_ROC_AUC_CI_Low",
        "Test_ROC_AUC_CI_High",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_Specificity",
        "Test_F1",
        "Test_PR_AUC",
        "Test_Brier",
    ]
    compact_cols = [c for c in compact_cols if c in results_df.columns]
    compact = results_df[compact_cols].copy()

    rename_map = {
        "Rank_ROC_AUC": "Rank",
        "CV_ROC_AUC_Mean": "CV ROC-AUC",
        "CV_ROC_AUC_SD": "CV SD",
        "Test_ROC_AUC": "Test ROC-AUC",
        "Test_ROC_AUC_CI_Low": "AUC CI Low",
        "Test_ROC_AUC_CI_High": "AUC CI High",
        "Test_Accuracy": "Accuracy",
        "Test_Precision": "Precision",
        "Test_Recall": "Recall",
        "Test_Specificity": "Specificity",
        "Test_F1": "F1",
        "Test_PR_AUC": "PR-AUC",
        "Test_Brier": "Brier",
    }
    compact = compact.rename(columns=rename_map)
    compact.to_csv(
        dirs["tables"] / "model_performance_dissertation_table.csv",
        index=False,
    )
    save_dataframe_table_image(
        compact,
        dirs["figures"] / "02_model_performance_dissertation_table.png",
        "Classification Performance on NIJ Recidivism Data",
        decimals=3,
        max_rows=20,
    )

    # CV vs test and metric-comparison figures
    plot_cv_vs_test(
        results_df,
        "ROC_AUC",
        dirs["figures"] / "03_cv_vs_test_roc_auc.png",
    )
    plot_cv_vs_test(
        results_df,
        "F1",
        dirs["figures"] / "04_cv_vs_test_f1.png",
    )
    plot_metric_comparison(
        results_df,
        dirs["figures"] / "ML1_nij_replot_test_performance.png",
    )
    plot_cv_metrics_heatmap(
        results_df,
        dirs["figures"] / "ML22_nij_replot_cv_all_metrics.png",
    )
    plot_cv_roc_auc_stability(
        results_df,
        dirs["figures"] / "ML23_nij_replot_cv_roc_auc_stability.png",
    )
    plot_brier_ranked(
        results_df,
        dirs["figures"] / "ML5_nij_replot_brier.png",
    )

    # Rebuild holdout-derived figures.
    prediction_files = sorted(
        p for p in dirs["predictions"].glob("*_holdout_predictions.csv")
        if p.name != "all_models_holdout_predictions.csv"
    )
    if not prediction_files:
        raise FileNotFoundError(
            f"No individual holdout prediction CSV files found in:\n"
            f"  {dirs['predictions']}\n"
            "ROC/PR/calibration/confusion-matrix figures require these saved files."
        )

    curve_store = {}
    calibration_store = {}
    prevalence_values = []

    for prediction_path in prediction_files:
        pred = pd.read_csv(prediction_path)

        required_pred = {"Actual", "Predicted", "Predicted_Probability"}
        missing_pred = required_pred.difference(pred.columns)
        if missing_pred:
            print(
                f"WARNING: Skipping {prediction_path.name}; missing "
                f"{sorted(missing_pred)}"
            )
            continue

        model_name = model_name_from_prediction_filename(prediction_path)
        y_true = pred["Actual"].astype(int).to_numpy()
        y_pred = pred["Predicted"].astype(int).to_numpy()
        y_prob = pred["Predicted_Probability"].astype(float).to_numpy()

        prevalence_values.append(float(np.mean(y_true)))

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)

        curve_store[model_name] = {
            "roc": (fpr, tpr),
            "auc": auc_value,
            "pr": (pr_precision, pr_recall),
            "ap": ap_value,
        }

        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=10,
            strategy="quantile",
        )
        calibration_store[model_name] = (mean_pred, frac_pos)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        safe_name = prediction_path.stem.replace("_holdout_predictions", "")
        plot_confusion_matrix(
            cm,
            model_name,
            dirs["figures"] / f"ML6_nij_replot_confusion_matrix_{safe_name}.png",
        )

    if curve_store:
        plot_roc_curves(
            curve_store,
            dirs["figures"] / "ML2_nij_replot_roc_curves.png",
        )
        prevalence = (
            float(np.mean(prevalence_values))
            if prevalence_values else 0.5
        )
        plot_pr_curves(
            curve_store,
            prevalence=prevalence,
            path=dirs["figures"] / "ML3_nij_replot_precision_recall.png",
        )

    if calibration_store:
        plot_calibration_curves(
            calibration_store,
            dirs["figures"] / "ML4_nij_replot_calibration.png",
        )

    # Rebuild fairness-suite plots from saved CSV output.
    fairness_path = dirs["tables"] / "fairness_suite_all_models.csv"
    if fairness_path.exists():
        fairness_all = pd.read_csv(fairness_path)
        for dimension, reference_group in FAIRNESS_DIMENSIONS.items():
            d = fairness_all[fairness_all["Dimension"] == dimension]
            if d.empty:
                continue

            plot_fairness_disparate_impact(
                fairness_all,
                dimension,
                reference_group,
                dirs["figures"] / f"ML8_nij_disparate_impact_{dimension.lower()}.png",
            )
            plot_fairness_rate_differences(
                fairness_all,
                dimension,
                reference_group,
                dirs["figures"] / f"ML9_nij_fairness_rate_differences_{dimension.lower()}.png",
            )
            plot_subgroup_auc_brier(
                fairness_all,
                dimension,
                dirs["figures"] / f"ML10_nij_subgroup_auc_brier_{dimension.lower()}.png",
            )
            plot_fairness_heatmap(
                fairness_all,
                dimension,
                dirs["figures"] / f"ML11_nij_fairness_heatmap_{dimension.lower()}.png",
            )
    else:
        print("Fairness suite CSV not found; skipping fairness replot.")

    # Rebuild feature-importance plots from saved CSVs.
    feature_files = sorted(
        dirs["tables"].glob("*_feature_importance.csv")
    )
    feature_count = 0
    for feature_path in feature_files:
        feature_df = pd.read_csv(feature_path)
        if feature_df.empty or not {"Feature", "Importance"}.issubset(feature_df.columns):
            continue

        if "Model" in feature_df.columns and feature_df["Model"].notna().any():
            model_name = str(feature_df["Model"].dropna().iloc[0])
        else:
            model_name = feature_path.stem.replace(
                "_feature_importance", ""
            ).replace("_", " ").title()

        if "Importance_Type" not in feature_df.columns:
            feature_df["Importance_Type"] = "Importance"

        safe_name = feature_path.stem.replace("_feature_importance", "")
        plot_top_features(
            feature_df.sort_values("Importance", ascending=False),
            model_name,
            dirs["figures"] / f"ML7_nij_replot_features_{safe_name}.png",
            top_n=15,
        )
        feature_count += 1

    print("\nReplot complete.")
    print(f"Figures recreated in: {dirs['figures'].resolve()}")
    print(f"Dissertation table recreated in: {dirs['tables'].resolve()}")
    print(f"Prediction files used: {len(curve_store)}")
    print(f"Feature-importance figures recreated: {feature_count}")
    print("\nNo classifiers were retrained.")


# ---------------------------------------------------------------------
# Cross-horizon dissertation charts
# ---------------------------------------------------------------------

def plot_cross_horizon_model_performance(combined_results, figures_dir):
    """Compare holdout ROC-AUC and F1 for every model across NIJ horizons."""
    if combined_results.empty:
        return

    horizon_order = ["1-Year", "2-Year", "3-Year"]

    for metric, ylabel, filename in [
        ("Test_ROC_AUC", "Holdout ROC-AUC", "ML30_cross_horizon_roc_auc.png"),
        ("Test_F1", "Holdout F1", "ML31_cross_horizon_f1.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11.5, 7.0))
        for model in combined_results["Model"].drop_duplicates():
            g = (
                combined_results[combined_results["Model"] == model]
                .set_index("Horizon")
                .reindex(horizon_order)
            )
            if g[metric].notna().any():
                y_values = g[metric].astype(float).to_numpy()
                x_values = np.arange(len(horizon_order))
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2,
                    label=model,
                )

        ax.set_xticks(np.arange(len(horizon_order)))
        ax.set_xticklabels(horizon_order)

        ax.set_xlabel("Rearrest prediction horizon")
        ax.set_ylabel(ylabel)
        metric_values = pd.to_numeric(combined_results[metric], errors="coerce")
        floor = 0.50 if metric == "Test_ROC_AUC" else 0.0
        y_lo, y_hi = _tight_axis_limits(metric_values, lower_bound=floor, upper_bound=1.0, pad_fraction=0.10, min_pad=0.015)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"NIJ Model Performance Across 1-, 2-, and 3-Year Rearrest Horizons: {ylabel}",
            fontweight="bold",
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.text(
            0.5, 0.01,
            "Each point is performance on the horizon-specific untouched 20% holdout test set.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.045, 1, 1])
        fig.savefig(figures_dir / filename, dpi=350, bbox_inches="tight")
        plt.close(fig)


def plot_cross_horizon_best_model_fairness(combined_fairness, combined_results, figures_dir):
    """
    Compare fairness of each horizon's best-ROC-AUC classifier.
    Separate figures are produced for Race and Gender.
    """
    if combined_fairness.empty or combined_results.empty:
        return

    best_models = (
        combined_results.sort_values(
            ["Horizon", "Test_ROC_AUC"], ascending=[True, False]
        )
        .groupby("Horizon", as_index=False)
        .first()[["Horizon", "Model"]]
    )

    d = combined_fairness.merge(
        best_models,
        on=["Horizon", "Model"],
        how="inner",
    )

    horizon_order = ["1-Year", "2-Year", "3-Year"]
    best_model_text = _best_model_map_text(best_models, horizon_order)

    for dimension, reference_group in FAIRNESS_DIMENSIONS.items():
        sub = d[d["Dimension"] == dimension].copy()
        if sub.empty:
            continue

        groups = [
            g for g in sub["Group"].drop_duplicates().tolist()
            if str(g) != str(reference_group)
        ]

        # DIR across horizons
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        for group in groups:
            g = (
                sub[sub["Group"] == group]
                .set_index("Horizon")
                .reindex(horizon_order)
            )
            ax.plot(
                np.arange(len(horizon_order)),
                g["Disparate_Impact_Ratio"].astype(float).to_numpy(),
                marker="o",
                linewidth=2,
                label=str(group),
            )
        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Parity = 1.0")
        ax.set_xticks(np.arange(len(horizon_order)))
        ax.set_xticklabels(horizon_order)
        ax.set_xlabel("Rearrest prediction horizon")
        ax.set_ylabel("Disparate Impact Ratio")
        dir_values = pd.to_numeric(sub["Disparate_Impact_Ratio"], errors="coerce")
        y_lo, y_hi = _tight_axis_limits(dir_values, lower_bound=0.0, pad_fraction=0.12, min_pad=0.03, include=1.0)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"Disparate Impact Across Horizons by {dimension}\nSelected models — {best_model_text}",
            fontweight="bold",
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.text(
            0.5, 0.01,
            f"Selected classifier by horizon: {best_model_text}. Reference group: {reference_group}. "
            f"Positive classification denotes predicted rearrest.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.045, 1, 1])
        fig.savefig(
            figures_dir / f"ML32_cross_horizon_DIR_{dimension.lower()}.png",
            dpi=350, bbox_inches="tight",
        )
        plt.close(fig)

        # FPR difference across horizons -- especially important for recidivism.
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        for group in groups:
            g = (
                sub[sub["Group"] == group]
                .set_index("Horizon")
                .reindex(horizon_order)
            )
            ax.plot(
                np.arange(len(horizon_order)),
                g["FPR_Difference"].astype(float).to_numpy(),
                marker="o",
                linewidth=2,
                label=str(group),
            )
        ax.axhline(0.0, linestyle="--", linewidth=1.2, label="Parity = 0.0")
        ax.set_xticks(np.arange(len(horizon_order)))
        ax.set_xticklabels(horizon_order)
        ax.set_xlabel("Rearrest prediction horizon")
        ax.set_ylabel("FPR difference from reference")
        fpr_values = pd.to_numeric(sub["FPR_Difference"], errors="coerce")
        y_lo, y_hi = _tight_axis_limits(fpr_values, lower_bound=-1.0, upper_bound=1.0, pad_fraction=0.12, min_pad=0.02, include=0.0)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"False-Positive Rate Disparity Across Horizons by {dimension}\nSelected models — {best_model_text}",
            fontweight="bold",
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.text(
            0.5, 0.01,
            f"Selected classifier by horizon: {best_model_text}. Positive values indicate more false-positive "
            f"rearrest classifications than {reference_group}.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.045, 1, 1])
        fig.savefig(
            figures_dir / f"ML33_cross_horizon_FPR_difference_{dimension.lower()}.png",
            dpi=350, bbox_inches="tight",
        )
        plt.close(fig)


def build_cross_horizon_outputs(output_root: Path):
    """Combine horizon-specific performance/fairness tables and build dissertation charts."""
    cross_tables = output_root / "cross_horizon" / "tables"
    cross_figures = output_root / "cross_horizon" / "figures"
    cross_tables.mkdir(parents=True, exist_ok=True)
    cross_figures.mkdir(parents=True, exist_ok=True)

    result_frames = []
    fairness_frames = []

    for horizon_key, info in HORIZONS.items():
        hroot = output_root / horizon_key
        perf_path = hroot / "tables" / "model_performance_full.csv"
        fair_path = hroot / "tables" / "fairness_suite_all_models.csv"

        if perf_path.exists():
            d = pd.read_csv(perf_path)
            d.insert(0, "Horizon", info["label"])
            result_frames.append(d)

        if fair_path.exists():
            f = pd.read_csv(fair_path)
            f.insert(0, "Horizon", info["label"])
            fairness_frames.append(f)

    combined_results = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames else pd.DataFrame()
    )
    combined_fairness = (
        pd.concat(fairness_frames, ignore_index=True)
        if fairness_frames else pd.DataFrame()
    )

    if not combined_results.empty:
        combined_results.to_csv(
            cross_tables / "DISSERTATION_model_performance_all_horizons.csv",
            index=False,
        )

        best = (
            combined_results.sort_values(
                ["Horizon", "Test_ROC_AUC", "Test_F1"],
                ascending=[True, False, False],
            )
            .groupby("Horizon", as_index=False)
            .first()
        )
        best.to_csv(
            cross_tables / "DISSERTATION_best_model_by_horizon.csv",
            index=False,
        )

        plot_cross_horizon_model_performance(combined_results, cross_figures)

    if not combined_fairness.empty:
        combined_fairness.to_csv(
            cross_tables / "DISSERTATION_fairness_all_horizons.csv",
            index=False,
        )

    if not combined_results.empty and not combined_fairness.empty:
        plot_cross_horizon_best_model_fairness(
            combined_fairness,
            combined_results,
            cross_figures,
        )


def safe_filename(name):
    """
    Convert a model/display name into a filesystem-safe lowercase name.

    Examples:
        Logistic Regression -> logistic_regression
        K-Nearest Neighbors  -> k_nearest_neighbors
        Support Vector Machine -> support_vector_machine
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
# ---------------------------------------------------------------------
# Single-horizon analysis
# ---------------------------------------------------------------------

def run_horizon_analysis(args, data_path: Path, output_root: Path, horizon_key: str):
    global CURRENT_HORIZON_LABEL

    info = HORIZONS[horizon_key]
    CURRENT_HORIZON_LABEL = info["label"]
    horizon_slug = horizon_key
    dirs = make_output_dirs(output_root / horizon_slug)

    print("\n" + "=" * 88)
    print(f" NIJ {CURRENT_HORIZON_LABEL.upper()} RECIDIVISM CLASSIFICATION")
    print("=" * 88)

    df, X, y, excluded_columns, all_null, target_source = load_and_prepare_data(
        data_path, horizon_key
    )
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    models, xgb_status = build_models()

    print(f"\nDataset rows:           {len(df):,}")
    print(f"Original columns:       {df.shape[1]}")
    print(f"Predictor columns:      {X.shape[1]}")
    print(f"Numeric predictors:     {len(numeric_features)}")
    print(f"Categorical predictors: {len(categorical_features)}")
    print(f"Horizon:                {CURRENT_HORIZON_LABEL}")
    print(f"Outcome source:         {target_source}")
    print(f"XGBoost:                {xgb_status}")

    # Horizon-specific stratified holdout split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Holdout records:  {len(X_test):,}")
    print(f"Training prevalence: {y_train.mean():.3f}")
    print(f"Test prevalence:     {y_test.mean():.3f}")

    class_distribution = plot_class_distribution(
        y_train,
        y_test,
        dirs["figures"] / f"01_{horizon_slug}_class_distribution.png",
    )
    class_distribution.to_csv(
        dirs["tables"] / "class_distribution.csv",
        index=False,
    )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "neg_brier": "neg_brier_score",
    }

    result_rows = []
    prediction_frame = pd.DataFrame(index=X_test.index)
    prediction_frame["Actual"] = y_test.astype(int)
    prediction_frame["Race"] = X_test["Race"].astype(str).values if "Race" in X_test else ""
    prediction_frame["Gender"] = X_test["Gender"].astype(str).values if "Gender" in X_test else ""

    curve_store = {}
    calibration_store = {}
    fairness_frames = []

    for i, (model_name, classifier) in enumerate(models.items(), start=1):
        print(f"\n[{i}/{len(models)}] {CURRENT_HORIZON_LABEL} — {model_name}")
        start_time = time.time()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", clone(classifier)),
            ]
        )

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test).astype(int)
        y_prob = get_probability_scores(pipeline, X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_specificity = specificity_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_prob)
        test_ap = average_precision_score(y_test, y_prob)
        test_brier = brier_score_loss(y_test, y_prob)

        auc_ci_low, auc_ci_high = bootstrap_metric_ci(
            y_test, y_pred, y_prob, "roc_auc",
            n_bootstrap=args.bootstrap_reps, seed=RANDOM_STATE,
        )
        f1_ci_low, f1_ci_high = bootstrap_metric_ci(
            y_test, y_pred, y_prob, "f1",
            n_bootstrap=args.bootstrap_reps, seed=RANDOM_STATE,
        )

        elapsed = time.time() - start_time

        row = {
            "Model": model_name,
            "CV_Accuracy_Mean": np.mean(cv_scores["test_accuracy"]),
            "CV_Accuracy_SD": np.std(cv_scores["test_accuracy"], ddof=1),
            "CV_Balanced_Accuracy_Mean": np.mean(cv_scores["test_balanced_accuracy"]),
            "CV_Balanced_Accuracy_SD": np.std(cv_scores["test_balanced_accuracy"], ddof=1),
            "CV_Precision_Mean": np.mean(cv_scores["test_precision"]),
            "CV_Precision_SD": np.std(cv_scores["test_precision"], ddof=1),
            "CV_Recall_Mean": np.mean(cv_scores["test_recall"]),
            "CV_Recall_SD": np.std(cv_scores["test_recall"], ddof=1),
            "CV_F1_Mean": np.mean(cv_scores["test_f1"]),
            "CV_F1_SD": np.std(cv_scores["test_f1"], ddof=1),
            "CV_ROC_AUC_Mean": np.mean(cv_scores["test_roc_auc"]),
            "CV_ROC_AUC_SD": np.std(cv_scores["test_roc_auc"], ddof=1),
            "CV_PR_AUC_Mean": np.mean(cv_scores["test_average_precision"]),
            "CV_PR_AUC_SD": np.std(cv_scores["test_average_precision"], ddof=1),
            "CV_Brier_Mean": -np.mean(cv_scores["test_neg_brier"]),
            "CV_Brier_SD": np.std(-cv_scores["test_neg_brier"], ddof=1),
            "Test_Accuracy": test_accuracy,
            "Test_Balanced_Accuracy": test_balanced_accuracy,
            "Test_Precision": test_precision,
            "Test_Recall": test_recall,
            "Test_Specificity": test_specificity,
            "Test_F1": test_f1,
            "Test_ROC_AUC": test_auc,
            "Test_ROC_AUC_CI_Low": auc_ci_low,
            "Test_ROC_AUC_CI_High": auc_ci_high,
            "Test_F1_CI_Low": f1_ci_low,
            "Test_F1_CI_High": f1_ci_high,
            "Test_PR_AUC": test_ap,
            "Test_Brier": test_brier,
            "Runtime_Seconds": elapsed,
        }
        result_rows.append(row)

        safe_name = safe_filename(model_name)

        pred_out = pd.DataFrame({
            "Actual": y_test.astype(int).values,
            "Predicted": y_pred,
            "Predicted_Probability": y_prob,
            "Race": X_test["Race"].astype(str).values if "Race" in X_test else "",
            "Gender": X_test["Gender"].astype(str).values if "Gender" in X_test else "",
        }, index=X_test.index)
        pred_out.to_csv(
            dirs["predictions"] / f"{safe_name}_holdout_predictions.csv",
            index_label="Original_Row_Index",
        )

        prediction_frame[f"Predicted_{safe_name}"] = y_pred
        prediction_frame[f"Probability_{safe_name}"] = y_prob

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
        curve_store[model_name] = {
            "roc": (fpr, tpr),
            "auc": test_auc,
            "pr": (pr_precision, pr_recall),
            "ap": test_ap,
        }

        frac_pos, mean_pred = calibration_curve(
            y_test, y_prob, n_bins=10, strategy="quantile"
        )
        calibration_store[model_name] = (mean_pred, frac_pos)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        pd.DataFrame(
            cm,
            index=["Actual_No_Rearrest", "Actual_Rearrest"],
            columns=["Predicted_No_Rearrest", "Predicted_Rearrest"],
        ).to_csv(dirs["tables"] / f"{safe_name}_confusion_matrix.csv")

        plot_confusion_matrix(
            cm,
            model_name,
            dirs["figures"] /
            f"ML6_nij_{horizon_slug}_confusion_matrix_{safe_name}.png",
        )

        report_df = pd.DataFrame(
            classification_report(
                y_test, y_pred,
                target_names=["No Rearrest", "Rearrest"],
                output_dict=True,
                zero_division=0,
            )
        ).T
        report_df.to_csv(
            dirs["reports"] / f"{safe_name}_classification_report.csv"
        )

        for dimension, reference_group in FAIRNESS_DIMENSIONS.items():
            if dimension not in X_test.columns:
                continue
            try:
                fair = bootstrap_fairness_ci(
                    protected_values=X_test[dimension],
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    reference_group=reference_group,
                    dimension=dimension,
                    model_name=model_name,
                    n_boot=min(args.bootstrap_reps, 500),
                    seed=RANDOM_STATE,
                )
                fairness_frames.append(fair)
                fair.to_csv(
                    dirs["tables"] /
                    f"{safe_name}_fairness_{dimension.lower()}.csv",
                    index=False,
                )
            except ValueError as exc:
                print(f"    WARNING: {exc}")

        importance_df = extract_feature_importance(pipeline, model_name)
        if importance_df is not None:
            importance_df.to_csv(
                dirs["tables"] / f"{safe_name}_feature_importance.csv",
                index=False,
            )
            plot_top_features(
                importance_df,
                model_name,
                dirs["figures"] /
                f"ML7_nij_{horizon_slug}_features_{safe_name}.png",
                top_n=15,
            )

        print(
            f"    CV ROC-AUC {row['CV_ROC_AUC_Mean']:.3f} ± "
            f"{row['CV_ROC_AUC_SD']:.3f} | "
            f"Holdout ROC-AUC {test_auc:.3f} | F1 {test_f1:.3f}"
        )

    results_df = (
        pd.DataFrame(result_rows)
        .sort_values(["Test_ROC_AUC", "Test_F1"], ascending=False)
        .reset_index(drop=True)
    )
    results_df.insert(0, "Rank_ROC_AUC", np.arange(1, len(results_df) + 1))
    results_df.to_csv(dirs["tables"] / "model_performance_full.csv", index=False)

    compact_cols = [
        "Rank_ROC_AUC", "Model", "CV_ROC_AUC_Mean", "CV_ROC_AUC_SD",
        "Test_ROC_AUC", "Test_ROC_AUC_CI_Low", "Test_ROC_AUC_CI_High",
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_Specificity",
        "Test_F1", "Test_PR_AUC", "Test_Brier",
    ]
    compact = results_df[compact_cols].copy()
    compact.columns = [
        "Rank", "Model", "CV ROC-AUC", "CV SD", "Test ROC-AUC",
        "AUC CI Low", "AUC CI High", "Accuracy", "Precision", "Recall",
        "Specificity", "F1", "PR-AUC", "Brier",
    ]
    compact.to_csv(
        dirs["tables"] / "model_performance_dissertation_table.csv",
        index=False,
    )
    save_dataframe_table_image(
        compact,
        dirs["figures"] / "02_model_performance_dissertation_table.png",
        f"Classification Performance on NIJ {CURRENT_HORIZON_LABEL} Rearrest",
        decimals=3,
        max_rows=20,
    )

    cv_table = results_df[
        [
            "Model", "CV_Accuracy_Mean", "CV_Accuracy_SD",
            "CV_Precision_Mean", "CV_Precision_SD",
            "CV_Recall_Mean", "CV_Recall_SD",
            "CV_F1_Mean", "CV_F1_SD",
            "CV_ROC_AUC_Mean", "CV_ROC_AUC_SD",
        ]
    ].copy()
    cv_table.to_csv(dirs["tables"] / "cross_validation_summary.csv", index=False)

    prediction_frame.to_csv(
        dirs["predictions"] / "all_models_holdout_predictions.csv",
        index_label="Original_Row_Index",
    )

    if fairness_frames:
        fairness_all = pd.concat(fairness_frames, ignore_index=True)
        save_fairness_dissertation_table(fairness_all, dirs["tables"])

        for dimension, reference_group in FAIRNESS_DIMENSIONS.items():
            if fairness_all[fairness_all["Dimension"] == dimension].empty:
                continue
            plot_fairness_disparate_impact(
                fairness_all, dimension, reference_group,
                dirs["figures"] /
                f"ML8_nij_{horizon_slug}_disparate_impact_{dimension.lower()}.png",
            )
            plot_fairness_rate_differences(
                fairness_all, dimension, reference_group,
                dirs["figures"] /
                f"ML9_nij_{horizon_slug}_fairness_rate_differences_{dimension.lower()}.png",
            )
            plot_subgroup_auc_brier(
                fairness_all, dimension,
                dirs["figures"] /
                f"ML10_nij_{horizon_slug}_subgroup_auc_brier_{dimension.lower()}.png",
            )
            plot_fairness_heatmap(
                fairness_all, dimension,
                dirs["figures"] /
                f"ML11_nij_{horizon_slug}_fairness_heatmap_{dimension.lower()}.png",
            )

    plot_cv_vs_test(
        results_df, "ROC_AUC",
        dirs["figures"] / f"03_{horizon_slug}_cv_vs_test_roc_auc.png",
    )
    plot_cv_vs_test(
        results_df, "F1",
        dirs["figures"] / f"04_{horizon_slug}_cv_vs_test_f1.png",
    )
    plot_metric_comparison(
        results_df,
        dirs["figures"] / f"ML1_nij_{horizon_slug}_test_performance.png",
    )
    plot_cv_metrics_heatmap(
        results_df,
        dirs["figures"] / f"ML22_nij_{horizon_slug}_cv_all_metrics.png",
    )
    plot_cv_roc_auc_stability(
        results_df,
        dirs["figures"] / f"ML23_nij_{horizon_slug}_cv_roc_auc_stability.png",
    )
    plot_brier_ranked(
        results_df,
        dirs["figures"] / f"ML5_nij_{horizon_slug}_brier.png",
    )
    plot_roc_curves(
        curve_store,
        dirs["figures"] / f"ML2_nij_{horizon_slug}_roc_curves.png",
    )
    plot_pr_curves(
        curve_store,
        prevalence=float(y_test.mean()),
        path=dirs["figures"] / f"ML3_nij_{horizon_slug}_precision_recall.png",
    )
    plot_calibration_curves(
        calibration_store,
        dirs["figures"] / f"ML4_nij_{horizon_slug}_calibration.png",
    )

    metadata = {
        "data_file": str(data_path),
        "horizon_key": horizon_key,
        "horizon_label": CURRENT_HORIZON_LABEL,
        "outcome_source": target_source,
        "n_rows": int(len(df)),
        "n_original_columns": int(df.shape[1]),
        "n_predictors": int(X.shape[1]),
        "target_prevalence": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": float(args.test_size),
        "cv_folds": int(args.cv_folds),
        "random_state": RANDOM_STATE,
        "bootstrap_reps": int(args.bootstrap_reps),
        "excluded_columns": excluded_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": list(models.keys()),
        "xgboost_status": xgb_status,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
    }
    with open(
        dirs["reports"] / "analysis_metadata.json",
        "w", encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    best_auc = results_df.iloc[0]
    print(
        f"\n{CURRENT_HORIZON_LABEL} best holdout ROC-AUC: "
        f"{best_auc['Model']} = {best_auc['Test_ROC_AUC']:.3f}"
    )

    return results_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global CURRENT_HORIZON_LABEL

    args = parse_args()
    data_path = Path(args.data)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.replot:
        print("\nREPLOT MODE — REGENERATING ALL THREE NIJ HORIZONS")
        for horizon_key, info in HORIZONS.items():
            CURRENT_HORIZON_LABEL = info["label"]
            dirs = make_output_dirs(output_root / horizon_key)
            results_path = dirs["tables"] / "model_performance_full.csv"
            if results_path.exists():
                print(f"\nReplotting {info['label']}...")
                replot_from_saved_results(dirs)
            else:
                print(
                    f"\nSkipping {info['label']}: no saved results at {results_path}"
                )
        build_cross_horizon_outputs(output_root)
        print("\nCross-horizon figures regenerated.")
        return

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("\n" + "=" * 88)
    print(" NIJ MULTI-HORIZON DISSERTATION ML CLASSIFICATION BENCHMARK")
    print("=" * 88)
    print(f"Data:   {data_path}")
    print(f"Output: {output_root.resolve()}")
    print(
        "\nHorizons: 1-Year, cumulative 2-Year, cumulative 3-Year rearrest. "
        "All recidivism outcome fields are excluded from predictors."
    )

    for horizon_key in HORIZONS:
        run_horizon_analysis(args, data_path, output_root, horizon_key)

    build_cross_horizon_outputs(output_root)

    print("\n" + "=" * 88)
    print(" MULTI-HORIZON ANALYSIS COMPLETE")
    print("=" * 88)
    print(f"1-Year outputs: {output_root / '1_year'}")
    print(f"2-Year outputs: {output_root / '2_year'}")
    print(f"3-Year outputs: {output_root / '3_year'}")
    print(f"Cross-horizon outputs: {output_root / 'cross_horizon'}")
    print("\nKey cross-horizon dissertation charts:")
    print("  ML30_cross_horizon_roc_auc.png")
    print("  ML31_cross_horizon_f1.png")
    print("  ML32_cross_horizon_DIR_race.png / gender.png")
    print("  ML33_cross_horizon_FPR_difference_race.png / gender.png")


if __name__ == "__main__":
    main()
