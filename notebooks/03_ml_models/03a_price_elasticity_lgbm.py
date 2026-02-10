# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Price Elasticity Model -- LightGBM
# MAGIC
# MAGIC **Purpose**: Train a LightGBM regressor that predicts the percentage change in
# MAGIC units sold (`volume_delta_pct`) as a function of price changes and a rich set
# MAGIC of control features.  The trained model is registered in Unity Catalog and can
# MAGIC be served via Databricks Model Serving for real-time pricing decisions.
# MAGIC
# MAGIC **Pipeline stages**:
# MAGIC 1. Load gold-layer feature table from Unity Catalog Feature Store.
# MAGIC 2. Pre-process (one-hot encoding, missing-value imputation, time-series split).
# MAGIC 3. Bayesian hyper-parameter search with Hyperopt.
# MAGIC 4. Train final model with best parameters on full training window.
# MAGIC 5. Evaluate on held-out test fold (RMSE, MAE, R-squared, MAPE).
# MAGIC 6. Compute SHAP explanations and log artefacts to MLflow.
# MAGIC 7. Generate per-category elasticity curves.
# MAGIC 8. Register model in Unity Catalog model registry.
# MAGIC 9. Run sanity checks (elasticity direction).
# MAGIC
# MAGIC **Owner**: Pricing Intelligence Team
# MAGIC **Catalog**: `hls_amer_catalog`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 -- Environment & Imports

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

from lightgbm import LGBMRegressor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 -- Configuration

# COMMAND ----------

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
EXPERIMENT_PATH: str = "/Stryker/PricingIntelligence/PriceElasticity"
FEATURE_TABLE: str = "hls_amer_catalog.gold.pricing_features"
MODEL_NAME_UC: str = "hls_amer_catalog.models.price_elasticity_lgbm"

TARGET_COL: str = "volume_delta_pct"
PRIMARY_FEATURE: str = "price_delta_pct"

CATEGORICAL_FEATURES: list[str] = [
    "product_category",
]

NUMERIC_FEATURES: list[str] = [
    "price_delta_pct",
    "avg_pocket_price",
    "avg_list_price",
    "discount_depth_avg",
    "price_realization_avg",
    "seasonal_index_avg",
    "competitor_asp_gap",
    "contract_mix_score",
    "macro_pressure_score",
    "innovation_tier",
    "market_share_pct",
    "patent_years_remaining",
    "gpo_concentration",
]

ALL_FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

N_CV_FOLDS: int = 5
HYPEROPT_MAX_EVALS: int = 50

np.random.seed(RANDOM_SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 -- MLflow Experiment

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.autolog(disable=True)  # we control logging explicitly

print(f"MLflow experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 -- Load Data from Feature Store

# COMMAND ----------

def load_feature_table(table_name: str) -> pd.DataFrame:
    """Read a Unity Catalog feature table into a Pandas DataFrame.

    Reads the specified table via Spark and converts to Pandas for local
    processing.

    Parameters
    ----------
    table_name : str
        Fully-qualified Unity Catalog table name
        (e.g. ``hls_amer_catalog.gold.pricing_features``).

    Returns
    -------
    pd.DataFrame
        The feature table as a Pandas DataFrame.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    feature_df = spark.read.table(table_name)
    return feature_df.toPandas()


raw_df = load_feature_table(FEATURE_TABLE)
print(f"Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns from {FEATURE_TABLE}")
display(raw_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 -- Data Pre-processing

# COMMAND ----------

def build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for pre-processing.

    Categorical columns are one-hot encoded (unknown categories are
    ignored at inference time).  Numeric columns have missing values
    filled with the column median.

    Parameters
    ----------
    categorical_features : list[str]
        Column names to one-hot encode.
    numeric_features : list[str]
        Column names to impute with median strategy.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Fitted-ready transformer that can be embedded in a Pipeline.
    """
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_features),
            ("num", num_pipeline, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    categorical_features: list[str],
    numeric_features: list[str],
) -> list[str]:
    """Extract human-readable feature names after transformation.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A *fitted* ColumnTransformer.
    categorical_features : list[str]
        Original categorical column names.
    numeric_features : list[str]
        Original numeric column names.

    Returns
    -------
    list[str]
        Ordered list of feature names matching the transformed array columns.
    """
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    return cat_names + numeric_features

# COMMAND ----------

# Validate required columns exist
missing_cols = [c for c in ALL_FEATURES + [TARGET_COL] if c not in raw_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in feature table: {missing_cols}")

# Drop rows where target is null
df = raw_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
print(f"Rows after dropping null targets: {len(df):,}")

X = df[ALL_FEATURES].copy()
y = df[TARGET_COL].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 -- Time-Series Cross-Validation Split

# COMMAND ----------

def create_time_series_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test index pairs using TimeSeriesSplit.

    This preserves temporal ordering so that the model never trains on
    future data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector (unused but kept for API symmetry).
    n_splits : int, optional
        Number of folds, by default 5.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of (train_indices, test_indices) tuples.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X, y))


splits = create_time_series_splits(X, y, n_splits=N_CV_FOLDS)
print(f"Created {len(splits)} time-series folds")
for i, (train_idx, test_idx) in enumerate(splits):
    print(f"  Fold {i}: train={len(train_idx):,}  test={len(test_idx):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 -- Hyperopt Hyper-parameter Tuning

# COMMAND ----------

def build_hyperopt_search_space() -> dict:
    """Define the Hyperopt search space for LightGBM hyper-parameters.

    The search space covers tree structure, regularisation, and
    stochastic training knobs that have the most impact on LightGBM
    performance for tabular regression tasks.

    Returns
    -------
    dict
        A dictionary of Hyperopt distributions keyed by parameter name.
    """
    space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 50)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 12, 1)),
        "num_leaves": scope.int(hp.quniform("num_leaves", 15, 127, 1)),
        "min_child_samples": scope.int(hp.quniform("min_child_samples", 5, 100, 5)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(10.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10.0)),
    }
    return space


def hyperopt_objective(params: dict) -> dict:
    """Evaluate a single Hyperopt trial using cross-validated RMSE.

    Fits a LightGBM model for each time-series fold, computes RMSE on
    the test portion, and returns the mean RMSE as the loss to minimise.

    Parameters
    ----------
    params : dict
        Hyper-parameter configuration sampled by Hyperopt.

    Returns
    -------
    dict
        Dictionary with keys ``loss`` (mean CV RMSE) and ``status``.
    """
    int_params = ["n_estimators", "max_depth", "num_leaves", "min_child_samples"]
    for p in int_params:
        params[p] = int(params[p])

    fold_rmses: list[float] = []

    for train_idx, test_idx in splits:
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        preprocessor = build_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
        X_train_t = preprocessor.fit_transform(X_train_fold)
        X_test_t = preprocessor.transform(X_test_fold)

        model = LGBMRegressor(
            **params,
            random_state=RANDOM_SEED,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(
            X_train_t,
            y_train_fold,
            eval_set=[(X_test_t, y_test_fold)],
            callbacks=[],
        )

        preds = model.predict(X_test_t)
        rmse = np.sqrt(mean_squared_error(y_test_fold, preds))
        fold_rmses.append(rmse)

    mean_rmse = float(np.mean(fold_rmses))
    return {"loss": mean_rmse, "status": STATUS_OK}

# COMMAND ----------

print(f"Starting Hyperopt search with {HYPEROPT_MAX_EVALS} evaluations ...")

search_space = build_hyperopt_search_space()
trials = Trials()

best_raw = fmin(
    fn=hyperopt_objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=HYPEROPT_MAX_EVALS,
    trials=trials,
    rstate=np.random.default_rng(RANDOM_SEED),
    verbose=True,
)

# Convert raw Hyperopt output to usable parameter dict
best_params = {
    "n_estimators": int(best_raw["n_estimators"]),
    "learning_rate": float(best_raw["learning_rate"]),
    "max_depth": int(best_raw["max_depth"]),
    "num_leaves": int(best_raw["num_leaves"]),
    "min_child_samples": int(best_raw["min_child_samples"]),
    "subsample": float(best_raw["subsample"]),
    "colsample_bytree": float(best_raw["colsample_bytree"]),
    "reg_alpha": float(best_raw["reg_alpha"]),
    "reg_lambda": float(best_raw["reg_lambda"]),
}

print("\nBest hyper-parameters from Hyperopt:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

best_cv_rmse = min(t["result"]["loss"] for t in trials.trials)
print(f"\nBest CV RMSE: {best_cv_rmse:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 -- Train Final Model

# COMMAND ----------

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
    params: dict,
    splits: list[tuple[np.ndarray, np.ndarray]],
    random_state: int = 42,
) -> tuple[LGBMRegressor, ColumnTransformer, list[str], dict]:
    """Train the final LightGBM model on the last fold's training data.

    Uses the last TimeSeriesSplit fold so that the model is trained on
    the maximum amount of historical data while retaining a temporally
    valid hold-out test set for reporting.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature DataFrame.
    y : pd.Series
        Full target Series.
    categorical_features : list[str]
        Categorical column names.
    numeric_features : list[str]
        Numeric column names.
    params : dict
        LightGBM hyper-parameters (from Hyperopt or manual).
    splits : list[tuple[np.ndarray, np.ndarray]]
        Time-series fold indices.
    random_state : int, optional
        Reproducibility seed, by default 42.

    Returns
    -------
    tuple
        (fitted_model, fitted_preprocessor, feature_names, fold_metrics)
    """
    all_fold_metrics: list[dict] = []

    # Evaluate across all folds for reporting
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]

        pre = build_preprocessor(categorical_features, numeric_features)
        X_tr_t = pre.fit_transform(X_tr)
        X_te_t = pre.transform(X_te)

        mdl = LGBMRegressor(
            **params,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        mdl.fit(X_tr_t, y_tr)
        preds = mdl.predict(X_te_t)

        fold_metrics = {
            "fold": fold_idx,
            "rmse": float(np.sqrt(mean_squared_error(y_te, preds))),
            "mae": float(mean_absolute_error(y_te, preds)),
            "r2": float(r2_score(y_te, preds)),
            "mape": float(mean_absolute_percentage_error(y_te, preds)),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        all_fold_metrics.append(fold_metrics)

    # Final model trained on last fold's training data
    last_train_idx, last_test_idx = splits[-1]
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    X_train_final = preprocessor.fit_transform(X.iloc[last_train_idx])
    y_train_final = y.iloc[last_train_idx]

    feature_names = get_feature_names_from_preprocessor(
        preprocessor, categorical_features, numeric_features
    )

    final_model = LGBMRegressor(
        **params,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
    )
    final_model.fit(X_train_final, y_train_final)

    return final_model, preprocessor, feature_names, all_fold_metrics

# COMMAND ----------

# Use best hyperopt params, but also define fallback defaults matching the spec
final_params = {
    "n_estimators": best_params.get("n_estimators", 500),
    "learning_rate": best_params.get("learning_rate", 0.05),
    "max_depth": best_params.get("max_depth", 8),
    "num_leaves": best_params.get("num_leaves", 31),
    "min_child_samples": best_params.get("min_child_samples", 20),
    "subsample": best_params.get("subsample", 0.8),
    "colsample_bytree": best_params.get("colsample_bytree", 0.8),
    "reg_alpha": best_params.get("reg_alpha", 0.1),
    "reg_lambda": best_params.get("reg_lambda", 0.1),
}

model, preprocessor, feature_names, fold_metrics = train_final_model(
    X, y,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    final_params,
    splits,
    random_state=RANDOM_SEED,
)

metrics_df = pd.DataFrame(fold_metrics)
print("\nPer-fold evaluation metrics:")
display(metrics_df)

mean_metrics = metrics_df[["rmse", "mae", "r2", "mape"]].mean().to_dict()
print("\nMean metrics across folds:")
for k, v in mean_metrics.items():
    print(f"  {k}: {v:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 -- Test Set Evaluation (Final Fold Hold-Out)

# COMMAND ----------

last_train_idx, last_test_idx = splits[-1]
X_test_final = preprocessor.transform(X.iloc[last_test_idx])
y_test_final = y.iloc[last_test_idx]

test_preds = model.predict(X_test_final)

test_metrics = {
    "test_rmse": float(np.sqrt(mean_squared_error(y_test_final, test_preds))),
    "test_mae": float(mean_absolute_error(y_test_final, test_preds)),
    "test_r2": float(r2_score(y_test_final, test_preds)),
    "test_mape": float(mean_absolute_percentage_error(y_test_final, test_preds)),
    "test_n": len(last_test_idx),
}

print("Hold-out test set metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 -- SHAP Explanations

# COMMAND ----------

def compute_and_log_shap(
    model: LGBMRegressor,
    X_transformed: np.ndarray,
    feature_names: list[str],
    artifact_dir: str = "/tmp/shap_artifacts",
    max_display: int = 20,
    top_n_dependence: int = 3,
) -> shap.Explanation:
    """Compute SHAP values and save explanation plots as MLflow artefacts.

    Generates three types of plots:
    - **Beeswarm (summary)**: shows global feature importance with value
      direction.
    - **Bar chart**: mean |SHAP| per feature.
    - **Dependence plots**: for the top-N most important features, showing
      marginal effect on prediction.

    All plots are saved as PNG files and logged to the active MLflow run.

    Parameters
    ----------
    model : LGBMRegressor
        A fitted LightGBM model.
    X_transformed : np.ndarray
        The pre-processed feature matrix (same shape used for prediction).
    feature_names : list[str]
        Human-readable names matching columns of ``X_transformed``.
    artifact_dir : str, optional
        Local directory for temporary plot files, by default "/tmp/shap_artifacts".
    max_display : int, optional
        Maximum features to show in summary plots, by default 20.
    top_n_dependence : int, optional
        Number of top features for dependence plots, by default 3.

    Returns
    -------
    shap.Explanation
        The computed SHAP explanation object.
    """
    import os
    os.makedirs(artifact_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names

    # --- Beeswarm summary plot ---
    fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        features=X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    summary_path = os.path.join(artifact_dir, "shap_beeswarm_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(summary_path, artifact_path="shap")

    # --- Bar chart ---
    fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        features=X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    bar_path = os.path.join(artifact_dir, "shap_feature_importance_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(bar_path, artifact_path="shap")

    # --- Dependence plots for top-N features ---
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n_dependence]

    for rank, idx in enumerate(top_indices):
        feat_name = feature_names[idx]
        fig_dep, ax_dep = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            idx,
            shap_values.values,
            X_transformed,
            feature_names=feature_names,
            show=False,
            ax=ax_dep,
        )
        ax_dep.set_title(f"SHAP Dependence: {feat_name} (rank {rank + 1})")
        plt.tight_layout()
        dep_path = os.path.join(artifact_dir, f"shap_dependence_{rank + 1}_{feat_name}.png")
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(dep_path, artifact_path="shap")

    print(f"Logged {2 + top_n_dependence} SHAP artefacts to MLflow.")
    return shap_values

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 -- Elasticity Curves by Category

# COMMAND ----------

def generate_elasticity_curves(
    df: pd.DataFrame,
    model: LGBMRegressor,
    preprocessor: ColumnTransformer,
    categorical_features: list[str],
    numeric_features: list[str],
    target_col: str,
    primary_feature: str,
    artifact_dir: str = "/tmp/elasticity_artifacts",
) -> pd.DataFrame:
    """Generate and plot elasticity curves for each product category.

    For each unique category in the data, this function computes the
    predicted ``volume_delta_pct`` across the observed range of
    ``price_delta_pct`` values.  A scatter-with-trend plot is saved per
    category and logged to MLflow.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with raw features and target.
    model : LGBMRegressor
        Fitted LightGBM model.
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    categorical_features : list[str]
        Categorical column names.
    numeric_features : list[str]
        Numeric column names.
    target_col : str
        Name of the target column.
    primary_feature : str
        Name of the price-change feature (x-axis).
    artifact_dir : str, optional
        Local directory for temporary plot files.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns: category, price_delta_pct,
        predicted_volume_delta_pct, actual_volume_delta_pct.
    """
    import os
    os.makedirs(artifact_dir, exist_ok=True)

    all_features = categorical_features + numeric_features
    categories = df["product_category"].dropna().unique()
    curve_records: list[dict] = []

    # -- Combined plot --
    fig_all, ax_all = plt.subplots(figsize=(14, 8))

    for cat in sorted(categories):
        cat_df = df[df["product_category"] == cat].copy()
        if len(cat_df) < 10:
            continue

        X_cat = cat_df[all_features]
        X_cat_t = preprocessor.transform(X_cat)
        preds = model.predict(X_cat_t)

        for i, (_, row) in enumerate(cat_df.iterrows()):
            curve_records.append({
                "category": cat,
                "price_delta_pct": row[primary_feature],
                "predicted_volume_delta_pct": float(preds[i]),
                "actual_volume_delta_pct": row[target_col],
            })

        # Sort for line plot
        sort_idx = np.argsort(cat_df[primary_feature].values)
        x_sorted = cat_df[primary_feature].values[sort_idx]
        y_sorted = preds[sort_idx]

        ax_all.plot(x_sorted, y_sorted, label=cat, linewidth=1.5, alpha=0.8)

        # -- Per-category plot --
        fig_cat, ax_cat = plt.subplots(figsize=(10, 6))
        ax_cat.scatter(
            cat_df[primary_feature].values,
            cat_df[target_col].values,
            alpha=0.3, s=15, label="Actual", color="grey",
        )
        ax_cat.plot(x_sorted, y_sorted, color="blue", linewidth=2, label="Predicted")
        ax_cat.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax_cat.axvline(0, color="black", linestyle="--", linewidth=0.5)
        ax_cat.set_xlabel("Price Change (%)")
        ax_cat.set_ylabel("Volume Change (%)")
        ax_cat.set_title(f"Elasticity Curve: {cat}")
        ax_cat.legend()
        plt.tight_layout()

        safe_name = cat.replace("/", "_").replace(" ", "_").lower()
        cat_path = os.path.join(artifact_dir, f"elasticity_curve_{safe_name}.png")
        plt.savefig(cat_path, dpi=150, bbox_inches="tight")
        plt.close(fig_cat)
        mlflow.log_artifact(cat_path, artifact_path="elasticity_curves")

    # Finalize combined plot
    ax_all.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax_all.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax_all.set_xlabel("Price Change (%)")
    ax_all.set_ylabel("Predicted Volume Change (%)")
    ax_all.set_title("Elasticity Curves by Category")
    ax_all.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    combined_path = os.path.join(artifact_dir, "elasticity_curves_all_categories.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig_all)
    mlflow.log_artifact(combined_path, artifact_path="elasticity_curves")

    curves_df = pd.DataFrame(curve_records)
    print(f"Generated elasticity curves for {len(categories)} categories.")
    return curves_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11 -- MLflow Run: Log Everything

# COMMAND ----------

with mlflow.start_run(run_name="price_elasticity_lgbm_final") as run:

    # --- Log parameters ---
    mlflow.log_params(final_params)
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("n_cv_folds", N_CV_FOLDS)
    mlflow.log_param("hyperopt_max_evals", HYPEROPT_MAX_EVALS)
    mlflow.log_param("feature_table", FEATURE_TABLE)
    mlflow.log_param("target", TARGET_COL)
    mlflow.log_param("n_features", len(feature_names))
    mlflow.log_param("n_training_rows", len(splits[-1][0]))

    # --- Log fold metrics ---
    for fm in fold_metrics:
        fold_i = fm["fold"]
        mlflow.log_metric(f"fold_{fold_i}_rmse", fm["rmse"])
        mlflow.log_metric(f"fold_{fold_i}_mae", fm["mae"])
        mlflow.log_metric(f"fold_{fold_i}_r2", fm["r2"])
        mlflow.log_metric(f"fold_{fold_i}_mape", fm["mape"])

    # --- Log mean CV metrics ---
    mlflow.log_metric("cv_mean_rmse", mean_metrics["rmse"])
    mlflow.log_metric("cv_mean_mae", mean_metrics["mae"])
    mlflow.log_metric("cv_mean_r2", mean_metrics["r2"])
    mlflow.log_metric("cv_mean_mape", mean_metrics["mape"])

    # --- Log test metrics ---
    for k, v in test_metrics.items():
        mlflow.log_metric(k, v)

    # --- Log best Hyperopt CV RMSE ---
    mlflow.log_metric("hyperopt_best_cv_rmse", best_cv_rmse)

    # --- SHAP ---
    print("Computing SHAP values ...")
    shap_values = compute_and_log_shap(
        model=model,
        X_transformed=X_test_final,
        feature_names=feature_names,
        artifact_dir="/tmp/shap_artifacts",
        max_display=20,
        top_n_dependence=3,
    )

    # --- Elasticity curves ---
    print("Generating elasticity curves ...")
    curves_df = generate_elasticity_curves(
        df=df.iloc[last_test_idx],
        model=model,
        preprocessor=preprocessor,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=NUMERIC_FEATURES,
        target_col=TARGET_COL,
        primary_feature=PRIMARY_FEATURE,
    )

    # Save curves data as CSV artefact
    curves_csv_path = "/tmp/elasticity_artifacts/elasticity_curves_data.csv"
    curves_df.to_csv(curves_csv_path, index=False)
    mlflow.log_artifact(curves_csv_path, artifact_path="elasticity_curves")

    # --- Model signature ---
    X_sample = X_test_final[:5] if isinstance(X_test_final, np.ndarray) else X_test_final.head(5)
    y_sample = model.predict(X_sample)
    signature = infer_signature(X_sample, y_sample)

    # --- Log model ---
    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_sample[:1],
    )

    run_id = run.info.run_id
    print(f"\nMLflow run completed: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12 -- Register Model in Unity Catalog

# COMMAND ----------

def register_model_to_unity_catalog(
    run_id: str,
    model_artifact_path: str,
    model_name: str,
) -> None:
    """Register the logged MLflow model in Unity Catalog.

    Sets the Unity Catalog model registry URI, registers the model, and
    adds a description and alias for the latest version.

    Parameters
    ----------
    run_id : str
        The MLflow run ID where the model artifact was logged.
    model_artifact_path : str
        The artifact path within the run (e.g. ``"model"``).
    model_name : str
        Fully-qualified Unity Catalog model name
        (e.g. ``"hls_amer_catalog.models.price_elasticity_lgbm"``).
    """
    mlflow.set_registry_uri("databricks-uc")

    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    from mlflow import MlflowClient
    client = MlflowClient(registry_uri="databricks-uc")

    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=(
            "LightGBM price-elasticity model. Predicts volume_delta_pct from "
            "price changes and control features. Trained with Hyperopt tuning "
            f"and {N_CV_FOLDS}-fold time-series CV."
        ),
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version.version,
    )

    print(f"Model registered: {model_name} v{model_version.version}")
    print(f"Alias 'champion' set to v{model_version.version}")


register_model_to_unity_catalog(
    run_id=run_id,
    model_artifact_path="model",
    model_name=MODEL_NAME_UC,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13 -- Sanity Checks

# COMMAND ----------

def run_sanity_checks(
    model: LGBMRegressor,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    primary_feature: str,
    target_col: str,
) -> dict:
    """Run economic sanity checks on the trained elasticity model.

    The key check is that the model predicts a *negative* relationship
    between price increases and volume -- i.e., when prices go up,
    units sold should (on average) go down.  This is the fundamental
    law of demand.

    Additional checks:
    - Correlation between predicted and actual should be positive.
    - Median elasticity should be negative for price increases.

    Parameters
    ----------
    model : LGBMRegressor
        Fitted model.
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    df : pd.DataFrame
        DataFrame with raw features and target.
    categorical_features : list[str]
        Categorical column names.
    numeric_features : list[str]
        Numeric column names.
    primary_feature : str
        Name of the price-change feature.
    target_col : str
        Name of the target column.

    Returns
    -------
    dict
        Dictionary of sanity-check results with boolean pass/fail flags.
    """
    all_features = categorical_features + numeric_features
    X_all = df[all_features]
    X_all_t = preprocessor.transform(X_all)
    preds = model.predict(X_all_t)

    results: dict = {}

    # Check 1: Negative elasticity for price increases
    price_up_mask = df[primary_feature].values > 0
    if price_up_mask.sum() > 0:
        avg_volume_pred_when_price_up = float(np.mean(preds[price_up_mask]))
        results["avg_pred_volume_when_price_up"] = avg_volume_pred_when_price_up
        results["elasticity_negative_for_price_increase"] = avg_volume_pred_when_price_up < 0
    else:
        results["elasticity_negative_for_price_increase"] = None
        results["avg_pred_volume_when_price_up"] = None

    # Check 2: Positive elasticity for price decreases
    price_down_mask = df[primary_feature].values < 0
    if price_down_mask.sum() > 0:
        avg_volume_pred_when_price_down = float(np.mean(preds[price_down_mask]))
        results["avg_pred_volume_when_price_down"] = avg_volume_pred_when_price_down
        results["elasticity_positive_for_price_decrease"] = avg_volume_pred_when_price_down > 0
    else:
        results["elasticity_positive_for_price_decrease"] = None
        results["avg_pred_volume_when_price_down"] = None

    # Check 3: Prediction-actual correlation
    actual = df[target_col].values
    correlation = float(np.corrcoef(preds, actual)[0, 1])
    results["pred_actual_correlation"] = correlation
    results["correlation_positive"] = correlation > 0

    # Check 4: Overall elasticity sign (slope of price_delta -> volume_delta)
    if price_up_mask.sum() > 0 and price_down_mask.sum() > 0:
        coef = np.polyfit(df[primary_feature].values, preds, deg=1)
        results["linear_slope_price_to_volume"] = float(coef[0])
        results["slope_negative"] = coef[0] < 0
    else:
        results["linear_slope_price_to_volume"] = None
        results["slope_negative"] = None

    return results

# COMMAND ----------

sanity_results = run_sanity_checks(
    model=model,
    preprocessor=preprocessor,
    df=df,
    categorical_features=CATEGORICAL_FEATURES,
    numeric_features=NUMERIC_FEATURES,
    primary_feature=PRIMARY_FEATURE,
    target_col=TARGET_COL,
)

print("Sanity Check Results:")
print("-" * 60)
for key, value in sanity_results.items():
    status = ""
    if isinstance(value, bool):
        status = " PASS" if value else " WARN -- review model"
    print(f"  {key}: {value}{status}")

# Critical assertion: elasticity should be negative for price increases
if sanity_results.get("elasticity_negative_for_price_increase") is False:
    print(
        "\nWARNING: Model does NOT show negative elasticity for price increases. "
        "This violates basic demand economics. Investigate feature engineering, "
        "data quality, or model specification."
    )
elif sanity_results.get("elasticity_negative_for_price_increase") is True:
    print("\nSanity check PASSED: price increases predict volume decreases (law of demand).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14 -- Summary

# COMMAND ----------

print("=" * 70)
print("  PRICE ELASTICITY MODEL -- TRAINING COMPLETE")
print("=" * 70)
print(f"  Feature table     : {FEATURE_TABLE}")
print(f"  Target            : {TARGET_COL}")
print(f"  Primary feature   : {PRIMARY_FEATURE}")
print(f"  Model             : LightGBM ({len(feature_names)} transformed features)")
print(f"  Hyperopt evals    : {HYPEROPT_MAX_EVALS}")
print(f"  CV folds          : {N_CV_FOLDS} (TimeSeriesSplit)")
print(f"  Best CV RMSE      : {best_cv_rmse:.6f}")
print(f"  Test RMSE         : {test_metrics['test_rmse']:.6f}")
print(f"  Test MAE          : {test_metrics['test_mae']:.6f}")
print(f"  Test R-squared    : {test_metrics['test_r2']:.6f}")
print(f"  Test MAPE         : {test_metrics['test_mape']:.6f}")
print(f"  MLflow run ID     : {run_id}")
print(f"  Registered model  : {MODEL_NAME_UC}")
print(f"  Elasticity check  : {'PASS' if sanity_results.get('elasticity_negative_for_price_increase') else 'WARN'}")
print("=" * 70)
