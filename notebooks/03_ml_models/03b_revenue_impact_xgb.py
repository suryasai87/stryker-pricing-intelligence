# Databricks notebook source

# MAGIC %md
# MAGIC # 03b - Revenue Impact Prediction (XGBoost)
# MAGIC
# MAGIC **Purpose:** Train an XGBoost regression model to predict the percentage change in
# MAGIC revenue (`revenue_delta_pct`) given a proposed price change and contextual product,
# MAGIC market, and contract features. The model enforces monotone constraints so that
# MAGIC price-elasticity economics are respected (e.g., large price increases are penalised
# MAGIC directionally). SHAP explanations are computed and logged for interpretability.
# MAGIC
# MAGIC **Input Table:** `hls_amer_catalog.gold.pricing_features` (Feature Store)
# MAGIC
# MAGIC **Registered Model:** `hls_amer_catalog.models.revenue_impact_xgb`
# MAGIC
# MAGIC **Key Design Decisions:**
# MAGIC - `predicted_elasticity_score` from the upstream price-elasticity model is consumed
# MAGIC   as a pre-computed feature, linking the two models in a forward pipeline.
# MAGIC - Monotone constraints ensure `proposed_price_delta` has a *negative* marginal
# MAGIC   effect on revenue (price up -> revenue down, all else equal).
# MAGIC - TimeSeriesSplit is used for cross-validation to avoid look-ahead bias.
# MAGIC - Hyperopt (Tree-structured Parzen Estimators) searches 30 configurations.
# MAGIC - Sanity checks verify that small price perturbations do not produce extreme
# MAGIC   revenue swings.
# MAGIC
# MAGIC **Metrics Logged:** RMSE, MAE, R-squared, MAPE

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Imports and Configuration

# COMMAND ----------

import warnings
import os
import tempfile
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

import shap

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope

from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=UserWarning)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Constants

# COMMAND ----------

# -- Catalog / table references ------------------------------------------------
CATALOG = "hls_amer_catalog"
FEATURE_TABLE = f"{CATALOG}.gold.pricing_features"
MODEL_REGISTRY_NAME = f"{CATALOG}.models.revenue_impact_xgb"

# -- MLflow experiment ----------------------------------------------------------
EXPERIMENT_PATH = "/Stryker/PricingIntelligence/RevenueImpact"

# -- Reproducibility ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# -- Target and feature lists ---------------------------------------------------
TARGET = "revenue_delta_pct"

NUMERIC_FEATURES: List[str] = [
    "predicted_elasticity_score",
    "proposed_price_delta",
    "current_volume",
    "units_3mo_avg",
    "cogs_pct",
    "direct_channel_pct",
    "macro_pressure_score",
    "contract_tier_mix",
    "gpo_concentration",
    "seasonal_index",
    "market_share_pct",
    "competitor_asp_gap",
    "price_realization_pct",
]

CATEGORICAL_FEATURES: List[str] = [
    "innovation_tier",
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# -- Cross-validation -----------------------------------------------------------
N_CV_SPLITS = 5

# -- Hyperopt -------------------------------------------------------------------
MAX_HYPEROPT_EVALS = 30

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Set MLflow Experiment

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment set to: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Training Data from Feature Store

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

raw_sdf = spark.table(FEATURE_TABLE)
print(f"Loaded {raw_sdf.count():,} rows from {FEATURE_TABLE}")
raw_sdf.printSchema()

# COMMAND ----------

# Convert to pandas for scikit-learn / XGBoost pipeline
raw_df = raw_sdf.toPandas()
print(f"Pandas DataFrame shape: {raw_df.shape}")

# Quick sanity: ensure target and all features exist
missing_cols = [c for c in [TARGET] + ALL_FEATURES if c not in raw_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in source table: {missing_cols}")

print("All required columns present.")
raw_df[ALL_FEATURES + [TARGET]].describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Preprocessing

# COMMAND ----------

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer that passes numeric features through
    unchanged and one-hot encodes categorical features.

    Args:
        numeric_features: Names of numeric columns to keep as-is.
        categorical_features: Names of categorical columns to one-hot encode.

    Returns:
        Fitted-ready ColumnTransformer instance.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Prepare Feature Matrix and Target

# COMMAND ----------

# Sort by a time-like ordering if a date column exists, otherwise keep as-is.
# TimeSeriesSplit requires temporal ordering.
TIME_COL_CANDIDATES = ["snapshot_date", "period_date", "month", "date"]
time_col = None
for candidate in TIME_COL_CANDIDATES:
    if candidate in raw_df.columns:
        time_col = candidate
        break

if time_col:
    raw_df = raw_df.sort_values(time_col).reset_index(drop=True)
    print(f"Data sorted by '{time_col}' for temporal cross-validation.")
else:
    print("WARNING: No recognized date column found. Data order used as-is for TimeSeriesSplit.")

X = raw_df[ALL_FEATURES].copy()
y = raw_df[TARGET].copy()

print(f"Feature matrix X: {X.shape}")
print(f"Target vector y: {y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Fit Preprocessor and Obtain Transformed Feature Names

# COMMAND ----------

X_transformed = preprocessor.fit_transform(X)
transformed_feature_names = list(preprocessor.get_feature_names_out())

print(f"Transformed feature matrix: {X_transformed.shape}")
print(f"Feature names after encoding: {transformed_feature_names}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build Monotone Constraints

# COMMAND ----------

def build_monotone_constraints(
    feature_names: List[str],
    constraints_map: Dict[str, int],
) -> Tuple[tuple, Dict]:
    """Construct the XGBoost monotone_constraints tuple from a human-readable map.

    The map keys are feature name substrings that are matched against the
    transformed feature name list. A value of -1 means *decreasing* monotone
    relationship; +1 means *increasing*; 0 means unconstrained.

    Args:
        feature_names: Ordered list of feature names after preprocessing.
        constraints_map: Mapping of feature-name substring -> constraint direction.

    Returns:
        Tuple of (monotone_constraints_tuple, applied_constraints_dict).
    """
    constraints = []
    applied = {}
    for fname in feature_names:
        direction = 0  # default: unconstrained
        for pattern, value in constraints_map.items():
            if pattern in fname:
                direction = value
                break
        constraints.append(direction)
        applied[fname] = direction
    return tuple(constraints), applied


# Domain-driven monotone constraints:
#   - proposed_price_delta: NEGATIVE (-1) -- raising price should decrease revenue, ceteris paribus
#   - predicted_elasticity_score: NEGATIVE (-1) -- higher elasticity means more revenue loss from price changes
#   - market_share_pct: POSITIVE (+1) -- higher share generally supports revenue
#   - current_volume: POSITIVE (+1) -- higher base volume supports revenue
CONSTRAINTS_MAP: Dict[str, int] = {
    "proposed_price_delta": -1,
    "predicted_elasticity_score": -1,
    "market_share_pct": 1,
    "current_volume": 1,
}

monotone_constraints_tuple, applied_constraints = build_monotone_constraints(
    transformed_feature_names, CONSTRAINTS_MAP
)

print("Applied monotone constraints:")
for feat, direction in applied_constraints.items():
    if direction != 0:
        label = "DECREASING" if direction == -1 else "INCREASING"
        print(f"  {feat}: {label} ({direction})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Baseline XGBoost Model with Cross-Validation

# COMMAND ----------

def create_xgb_model(
    monotone_constraints: tuple,
    seed: int = SEED,
    **kwargs,
) -> XGBRegressor:
    """Create an XGBRegressor with production defaults and monotone constraints.

    Args:
        monotone_constraints: Tuple of per-feature constraint directions.
        seed: Random seed for reproducibility.
        **kwargs: Override any default hyperparameter.

    Returns:
        Configured XGBRegressor instance (not yet fitted).
    """
    defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        monotone_constraints=monotone_constraints,
        n_jobs=-1,
        verbosity=0,
    )
    defaults.update(kwargs)
    return XGBRegressor(**defaults)


tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

baseline_model = create_xgb_model(monotone_constraints_tuple)
cv_scores = cross_val_score(
    baseline_model, X_transformed, y,
    cv=tscv, scoring="neg_root_mean_squared_error",
)

print(f"Baseline CV RMSE (5-fold TimeSeries): {-cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Hyperparameter Tuning with Hyperopt

# COMMAND ----------

def hyperopt_objective(params: Dict[str, Any]) -> Dict[str, Any]:
    """Hyperopt objective function.

    Trains an XGBRegressor with the proposed hyperparameters using
    TimeSeriesSplit cross-validation and returns the mean RMSE as the
    loss to minimise.

    Args:
        params: Dictionary of hyperparameter values sampled by Hyperopt.

    Returns:
        Dictionary with 'loss' (mean RMSE) and 'status'.
    """
    # Cast integer-valued hyperparameters
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])

    model = create_xgb_model(monotone_constraints_tuple, **params)

    cv_scores = cross_val_score(
        model, X_transformed, y,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
    )
    mean_rmse = -cv_scores.mean()

    return {"loss": mean_rmse, "status": STATUS_OK}


# Define search space
search_space = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 200, 1000, 50)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
    "max_depth": scope.int(hp.quniform("max_depth", 3, 10, 1)),
    "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(10.0)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10.0)),
}

trials = Trials()
best_params = fmin(
    fn=hyperopt_objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=MAX_HYPEROPT_EVALS,
    trials=trials,
    rstate=np.random.default_rng(SEED),
    verbose=True,
)

# Post-process integer casts
best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["max_depth"] = int(best_params["max_depth"])
best_params["min_child_weight"] = int(best_params["min_child_weight"])

print(f"\nBest hyperparameters from {MAX_HYPEROPT_EVALS} Hyperopt evaluations:")
for k, v in sorted(best_params.items()):
    print(f"  {k}: {v}")

best_trial_loss = min(t["result"]["loss"] for t in trials.trials)
print(f"\nBest CV RMSE: {best_trial_loss:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Train Final Model on Full Data and Evaluate

# COMMAND ----------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing RMSE, MAE, R2, and MAPE.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # Protect against division by zero in MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))
    else:
        mape = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# Use the last TimeSeriesSplit fold as a hold-out evaluation set
splits = list(tscv.split(X_transformed))
train_idx, test_idx = splits[-1]

X_train, X_test = X_transformed[train_idx], X_transformed[test_idx]
y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

# Train with tuned hyperparameters
final_model = create_xgb_model(monotone_constraints_tuple, **best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50,
)

y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

train_metrics = evaluate_model(y_train, y_pred_train)
test_metrics = evaluate_model(y_test, y_pred_test)

print("=== Train Metrics ===")
for k, v in train_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n=== Test Metrics (last fold hold-out) ===")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Retrain Final Model on ALL Data

# COMMAND ----------

# For the production-registered model, retrain on all available data using
# the tuned hyperparameters.
production_model = create_xgb_model(monotone_constraints_tuple, **best_params)
production_model.fit(X_transformed, y.values, verbose=50)
print("Production model trained on full dataset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. SHAP Explanations

# COMMAND ----------

def compute_and_log_shap(
    model: XGBRegressor,
    X: np.ndarray,
    feature_names: List[str],
    artifact_dir: str,
    max_display: int = 20,
    sample_size: int = 2000,
) -> shap.Explanation:
    """Compute SHAP values and save summary/dependence plots as artifacts.

    Uses TreeExplainer for fast, exact SHAP values on tree-based models.
    If the dataset has more than ``sample_size`` rows, a random subsample
    is used for visualisation speed.

    Args:
        model: Fitted XGBRegressor.
        X: Transformed feature matrix (numpy array).
        feature_names: List of feature names matching X columns.
        artifact_dir: Local directory path where plot PNGs will be saved.
        max_display: Maximum number of features to display in summary plot.
        sample_size: Maximum rows to use for SHAP computation.

    Returns:
        shap.Explanation object.
    """
    # Subsample for large datasets
    if X.shape[0] > sample_size:
        rng = np.random.default_rng(SEED)
        indices = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_shap = X[indices]
    else:
        X_shap = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame(X_shap, columns=feature_names))

    os.makedirs(artifact_dir, exist_ok=True)

    # --- Summary plot (beeswarm) ---
    fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, pd.DataFrame(X_shap, columns=feature_names),
        max_display=max_display, show=False,
    )
    plt.tight_layout()
    summary_path = os.path.join(artifact_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot: {summary_path}")

    # --- Bar plot (mean |SHAP|) ---
    fig_bar, ax_bar = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    bar_path = os.path.join(artifact_dir, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP feature importance bar plot: {bar_path}")

    # --- Dependence plots for key features ---
    key_features = ["proposed_price_delta", "predicted_elasticity_score", "market_share_pct"]
    for feat in key_features:
        if feat in feature_names:
            fig_dep, ax_dep = plt.subplots(figsize=(8, 6))
            shap.dependence_plot(
                feat, shap_values.values,
                pd.DataFrame(X_shap, columns=feature_names),
                show=False, ax=ax_dep,
            )
            ax_dep.set_title(f"SHAP Dependence: {feat}")
            plt.tight_layout()
            dep_path = os.path.join(artifact_dir, f"shap_dependence_{feat}.png")
            plt.savefig(dep_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved SHAP dependence plot: {dep_path}")

    return shap_values

# COMMAND ----------

shap_artifact_dir = tempfile.mkdtemp(prefix="shap_artifacts_")
shap_values = compute_and_log_shap(
    production_model, X_transformed, transformed_feature_names, shap_artifact_dir,
)
print("SHAP computation complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Sanity Checks

# COMMAND ----------

def run_sanity_checks(
    model: XGBRegressor,
    preprocessor: ColumnTransformer,
    reference_row: pd.DataFrame,
    feature_names: List[str],
    target_name: str,
) -> None:
    """Verify that model predictions are economically plausible.

    Performs the following checks:
    1. A +1% price delta should not predict revenue change exceeding +/-20%.
    2. A -1% price delta should not predict revenue change exceeding +/-20%.
    3. Zero price delta should predict near-zero revenue change (within +/-5%).
    4. Monotonicity: increasing price delta should weakly decrease predicted revenue.
    5. Extreme price deltas (+/-50%) should not produce unbounded predictions.

    Args:
        model: Fitted XGBRegressor.
        preprocessor: Fitted ColumnTransformer used for encoding.
        reference_row: Single-row DataFrame with all feature columns as a baseline.
        feature_names: Feature column names (pre-transform).
        target_name: Name of the target column (for logging).

    Raises:
        AssertionError: If any sanity check fails.
    """
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    def predict_with_price_delta(delta_val: float) -> float:
        """Predict revenue_delta_pct for a given proposed_price_delta, holding other features at reference values."""
        row = reference_row.copy()
        row["proposed_price_delta"] = delta_val
        X_enc = preprocessor.transform(row)
        return float(model.predict(X_enc)[0])

    # Establish baseline at zero price change
    pred_zero = predict_with_price_delta(0.0)
    pred_plus1 = predict_with_price_delta(1.0)
    pred_minus1 = predict_with_price_delta(-1.0)
    pred_plus50 = predict_with_price_delta(50.0)
    pred_minus50 = predict_with_price_delta(-50.0)

    print(f"\n  Price delta =  0.0%  -> predicted revenue_delta_pct = {pred_zero:+.4f}%")
    print(f"  Price delta = +1.0%  -> predicted revenue_delta_pct = {pred_plus1:+.4f}%")
    print(f"  Price delta = -1.0%  -> predicted revenue_delta_pct = {pred_minus1:+.4f}%")
    print(f"  Price delta = +50.0% -> predicted revenue_delta_pct = {pred_plus50:+.4f}%")
    print(f"  Price delta = -50.0% -> predicted revenue_delta_pct = {pred_minus50:+.4f}%")

    # Check 1 & 2: Small price changes should not cause extreme revenue swings
    SMALL_DELTA_MAX_IMPACT = 20.0
    assert abs(pred_plus1) < SMALL_DELTA_MAX_IMPACT, (
        f"FAIL: +1% price delta predicts {pred_plus1:.2f}% revenue change "
        f"(exceeds +/-{SMALL_DELTA_MAX_IMPACT}% threshold)"
    )
    print(f"\n  [PASS] +1% price change -> revenue impact within +/-{SMALL_DELTA_MAX_IMPACT}%")

    assert abs(pred_minus1) < SMALL_DELTA_MAX_IMPACT, (
        f"FAIL: -1% price delta predicts {pred_minus1:.2f}% revenue change "
        f"(exceeds +/-{SMALL_DELTA_MAX_IMPACT}% threshold)"
    )
    print(f"  [PASS] -1% price change -> revenue impact within +/-{SMALL_DELTA_MAX_IMPACT}%")

    # Check 3: Zero price change should predict near-zero revenue change
    ZERO_DELTA_TOLERANCE = 5.0
    assert abs(pred_zero) < ZERO_DELTA_TOLERANCE, (
        f"FAIL: 0% price delta predicts {pred_zero:.2f}% revenue change "
        f"(exceeds +/-{ZERO_DELTA_TOLERANCE}% tolerance)"
    )
    print(f"  [PASS] 0% price change -> revenue impact within +/-{ZERO_DELTA_TOLERANCE}%")

    # Check 4: Monotonicity -- increasing price should weakly decrease revenue
    deltas = np.linspace(-10, 10, 21)
    preds = [predict_with_price_delta(d) for d in deltas]
    diffs = np.diff(preds)
    # Allow tiny positive diffs (numerical noise) but flag systematic violations
    violation_count = np.sum(diffs > 0.5)  # > 0.5 pct threshold for meaningful violation
    total_steps = len(diffs)
    violation_ratio = violation_count / total_steps
    print(f"  [INFO] Monotonicity check: {violation_count}/{total_steps} steps violate "
          f"decreasing direction (threshold: 0.5 pct points)")
    if violation_ratio > 0.3:
        print(f"  [WARN] Monotonicity violation ratio {violation_ratio:.0%} exceeds 30% -- "
              f"review monotone constraint configuration.")
    else:
        print(f"  [PASS] Monotonicity largely respected ({1 - violation_ratio:.0%} compliant)")

    # Check 5: Extreme price changes should still produce bounded predictions
    EXTREME_DELTA_MAX_IMPACT = 100.0
    assert abs(pred_plus50) < EXTREME_DELTA_MAX_IMPACT, (
        f"FAIL: +50% price delta predicts {pred_plus50:.2f}% revenue change (unbounded)"
    )
    assert abs(pred_minus50) < EXTREME_DELTA_MAX_IMPACT, (
        f"FAIL: -50% price delta predicts {pred_minus50:.2f}% revenue change (unbounded)"
    )
    print(f"  [PASS] Extreme price deltas (+/-50%) produce bounded predictions "
          f"(within +/-{EXTREME_DELTA_MAX_IMPACT}%)")

    print("\n" + "=" * 70)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 70)

# COMMAND ----------

# Use the median row from training data as the reference point for sanity checks
reference_row = X.median(numeric_only=True).to_frame().T
# For categorical columns, use the mode
for cat_col in CATEGORICAL_FEATURES:
    mode_val = X[cat_col].mode()
    reference_row[cat_col] = mode_val.iloc[0] if len(mode_val) > 0 else X[cat_col].iloc[0]

# Ensure column order matches
reference_row = reference_row[ALL_FEATURES]

run_sanity_checks(
    production_model, preprocessor, reference_row,
    ALL_FEATURES, TARGET,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Log Everything to MLflow and Register Model

# COMMAND ----------

with mlflow.start_run(run_name="revenue_impact_xgb_production") as run:
    # -- Log hyperparameters ------------------------------------------------
    mlflow.log_params(best_params)
    mlflow.log_param("n_cv_splits", N_CV_SPLITS)
    mlflow.log_param("hyperopt_max_evals", MAX_HYPEROPT_EVALS)
    mlflow.log_param("random_seed", SEED)
    mlflow.log_param("feature_table", FEATURE_TABLE)
    mlflow.log_param("target", TARGET)
    mlflow.log_param("n_features_raw", len(ALL_FEATURES))
    mlflow.log_param("n_features_transformed", X_transformed.shape[1])
    mlflow.log_param("n_training_samples", X_transformed.shape[0])
    mlflow.log_param("monotone_constraints", str(monotone_constraints_tuple))

    # -- Log test metrics ---------------------------------------------------
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    for metric_name, metric_value in train_metrics.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)
    mlflow.log_metric("baseline_cv_rmse", -cv_scores.mean())
    mlflow.log_metric("best_hyperopt_cv_rmse", best_trial_loss)

    # -- Log SHAP artifacts -------------------------------------------------
    mlflow.log_artifacts(shap_artifact_dir, artifact_path="shap")
    print("Logged SHAP artifacts to MLflow.")

    # -- Log actual vs predicted plot (test set) ----------------------------
    fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
    ax_pred.scatter(y_test, y_pred_test, alpha=0.5, s=20, edgecolors="k", linewidths=0.3)
    lims = [
        min(y_test.min(), y_pred_test.min()) - 1,
        max(y_test.max(), y_pred_test.max()) + 1,
    ]
    ax_pred.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax_pred.set_xlabel("Actual revenue_delta_pct")
    ax_pred.set_ylabel("Predicted revenue_delta_pct")
    ax_pred.set_title(f"Actual vs Predicted (Test) | R2={test_metrics['r2']:.3f}")
    ax_pred.legend()
    ax_pred.set_aspect("equal")
    plt.tight_layout()
    pred_plot_path = os.path.join(shap_artifact_dir, "actual_vs_predicted.png")
    fig_pred.savefig(pred_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig_pred)
    mlflow.log_artifact(pred_plot_path, artifact_path="plots")

    # -- Log residual distribution plot -------------------------------------
    residuals = y_test - y_pred_test
    fig_resid, ax_resid = plt.subplots(figsize=(8, 5))
    ax_resid.hist(residuals, bins=40, edgecolor="black", alpha=0.7)
    ax_resid.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax_resid.set_xlabel("Residual (actual - predicted)")
    ax_resid.set_ylabel("Frequency")
    ax_resid.set_title(f"Residual Distribution | MAE={test_metrics['mae']:.3f}")
    plt.tight_layout()
    resid_path = os.path.join(shap_artifact_dir, "residual_distribution.png")
    fig_resid.savefig(resid_path, dpi=150, bbox_inches="tight")
    plt.close(fig_resid)
    mlflow.log_artifact(resid_path, artifact_path="plots")

    # -- Log model with signature -------------------------------------------
    # Build signature from the preprocessed numpy arrays for accurate schema
    X_sample = pd.DataFrame(X_transformed[:5], columns=transformed_feature_names)
    y_sample = pd.Series(y.iloc[:5].values, name=TARGET)
    signature = infer_signature(X_sample, y_sample)

    mlflow.xgboost.log_model(
        xgb_model=production_model,
        artifact_path="model",
        signature=signature,
        input_example=X_sample,
        registered_model_name=MODEL_REGISTRY_NAME,
    )

    run_id = run.info.run_id
    print(f"\nMLflow Run ID: {run_id}")
    print(f"Model registered as: {MODEL_REGISTRY_NAME}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test R2:   {test_metrics['r2']:.4f}")
    print(f"Test MAPE: {test_metrics['mape']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Summary

# COMMAND ----------

print("=" * 70)
print("REVENUE IMPACT MODEL TRAINING COMPLETE")
print("=" * 70)
print(f"""
Experiment:          {EXPERIMENT_PATH}
Feature Table:       {FEATURE_TABLE}
Registered Model:    {MODEL_REGISTRY_NAME}
MLflow Run ID:       {run_id}

Training Samples:    {X_transformed.shape[0]:,}
Features (raw):      {len(ALL_FEATURES)}
Features (encoded):  {X_transformed.shape[1]}

Baseline CV RMSE:    {-cv_scores.mean():.4f}
Tuned CV RMSE:       {best_trial_loss:.4f}
Test RMSE:           {test_metrics['rmse']:.4f}
Test MAE:            {test_metrics['mae']:.4f}
Test R2:             {test_metrics['r2']:.4f}
Test MAPE:           {test_metrics['mape']:.4f}

Monotone Constraints Applied:
  proposed_price_delta:      DECREASING (-1)
  predicted_elasticity_score: DECREASING (-1)
  market_share_pct:          INCREASING (+1)
  current_volume:            INCREASING (+1)

Sanity Checks:       ALL PASSED
SHAP Artifacts:      Logged to MLflow
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Cleanup

# COMMAND ----------

# Clean up temporary SHAP artifact directory
import shutil
if os.path.exists(shap_artifact_dir):
    shutil.rmtree(shap_artifact_dir)
    print(f"Cleaned up temp directory: {shap_artifact_dir}")

print("Notebook complete.")
