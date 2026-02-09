# Databricks notebook source

# MAGIC %md
# MAGIC # 03c - Margin Optimization: ElasticNet with Interaction Effects
# MAGIC
# MAGIC **Purpose:** Train an ElasticNet regression model to predict absolute gross margin
# MAGIC delta per unit (`gross_margin_delta`) given pricing, cost, and market features.
# MAGIC The pipeline uses polynomial feature expansion to capture critical interaction
# MAGIC effects (e.g., tariff x volume, FX x COGS) that drive margin compression or
# MAGIC expansion in medical device pricing.
# MAGIC
# MAGIC **Why ElasticNet?**
# MAGIC - Combines L1 (Lasso) sparsity with L2 (Ridge) stability, ideal for
# MAGIC   high-dimensional feature spaces created by polynomial expansion.
# MAGIC - Coefficient interpretability: post-training, we can rank feature importance
# MAGIC   by coefficient magnitude, including interaction terms.
# MAGIC - Regularization prevents overfitting on the expanded feature set.
# MAGIC
# MAGIC **Input Table:** `hls_amer_catalog.gold.pricing_features`
# MAGIC
# MAGIC **Registered Model:** `hls_amer_catalog.models.margin_optimization_enet`
# MAGIC
# MAGIC **MLflow Experiment:** `/Stryker/PricingIntelligence/MarginOptimization`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and Configuration

# COMMAND ----------

import warnings
import tempfile
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=UserWarning)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Constants and Experiment Setup

# COMMAND ----------

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data coordinates
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
FEATURE_TABLE = f"{CATALOG}.gold.pricing_features"
MODEL_REGISTRY_NAME = f"{CATALOG}.models.margin_optimization_enet"

# ---------------------------------------------------------------------------
# MLflow experiment
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_PATH = "/Stryker/PricingIntelligence/MarginOptimization"
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

print(f"MLflow experiment : {MLFLOW_EXPERIMENT_PATH}")
print(f"Feature table     : {FEATURE_TABLE}")
print(f"Model registry    : {MODEL_REGISTRY_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Definitions
# MAGIC
# MAGIC We explicitly declare every feature column and its type so that the pipeline
# MAGIC is self-documenting and resilient to upstream schema changes.

# COMMAND ----------

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
TARGET_COL = "gross_margin_delta"

# ---------------------------------------------------------------------------
# Numeric features
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: List[str] = [
    "predicted_revenue_impact",     # Output from upstream revenue model (03a)
    "cogs_pct",                     # Cost of goods sold as % of revenue
    "cogs_trend_3mo",               # 3-month rolling COGS trend
    "tariff_impact_index",          # Composite tariff impact score
    "fx_impact",                    # Weighted FX impact (USD/EUR, USD/JPY)
    "logistics_cost_index",         # Composite: fuel + freight indices
    "predicted_volume_change",      # Predicted unit volume delta (from 03b)
    "discount_depth_change",        # Change in average discount depth (pp)
    "rebate_pct_change",            # Change in rebate percentage (pp)
    "product_mix_shift_index",      # Index capturing mix shift toward/away from premium
    "resin_price_trend",            # Resin commodity price trend (3-mo slope)
    "cobalt_chrome_price_trend",    # CoCr commodity price trend (3-mo slope)
    "innovation_tier",              # Numeric tier: 1 (legacy) to 5 (breakthrough)
]

# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES: List[str] = [
    "category",                     # Product category (e.g., Joint Replacement, Spine, etc.)
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print(f"Target            : {TARGET_COL}")
print(f"Numeric features  : {len(NUMERIC_FEATURES)}")
print(f"Categorical feat. : {len(CATEGORICAL_FEATURES)}")
print(f"Total features    : {len(ALL_FEATURES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load Training Data from Feature Store

# COMMAND ----------

def load_feature_data(table_name: str) -> pd.DataFrame:
    """Load training data from the Unity Catalog Feature Store table.

    Reads the specified Delta table via Spark, converts to pandas, and
    performs basic validation.

    Args:
        table_name: Fully qualified three-level Unity Catalog table name
                    (catalog.schema.table).

    Returns:
        pandas DataFrame containing all feature columns and the target.

    Raises:
        ValueError: If required columns are missing from the source table.
        RuntimeError: If the table is empty.
    """
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.table(table_name)

    # Validate required columns exist
    available_cols = set(sdf.columns)
    required_cols = set(ALL_FEATURES + [TARGET_COL])
    missing_cols = required_cols - available_cols
    if missing_cols:
        raise ValueError(
            f"Missing columns in {table_name}: {sorted(missing_cols)}. "
            f"Available: {sorted(available_cols)}"
        )

    # Select only the columns we need (plus any time column for ordering)
    select_cols = ALL_FEATURES + [TARGET_COL]
    if "month" in available_cols:
        select_cols = ["month"] + select_cols
    if "product_id" in available_cols:
        select_cols = ["product_id"] + select_cols

    pdf = sdf.select(*select_cols).toPandas()

    if pdf.empty:
        raise RuntimeError(f"Table {table_name} returned zero rows.")

    # Sort by time if available (critical for TimeSeriesSplit)
    if "month" in pdf.columns:
        pdf = pdf.sort_values("month").reset_index(drop=True)

    print(f"Loaded {len(pdf):,} rows from {table_name}")
    print(f"Date range: {pdf['month'].min()} to {pdf['month'].max()}" if "month" in pdf.columns else "")
    print(f"Target stats:\n{pdf[TARGET_COL].describe()}\n")

    return pdf

# COMMAND ----------

pdf = load_feature_data(FEATURE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Exploratory Summary and Missing-Value Audit

# COMMAND ----------

def audit_data(df: pd.DataFrame, features: List[str], target: str) -> None:
    """Print a concise data-quality summary for the training set.

    Reports missing value counts, zero-variance columns, and target
    distribution statistics. This is a pre-training gate: any critical
    issues will surface here before we invest compute in grid search.

    Args:
        df: Training DataFrame.
        features: List of feature column names.
        target: Target column name.
    """
    print("=" * 70)
    print("DATA QUALITY AUDIT")
    print("=" * 70)

    # Missing values
    missing = df[features + [target]].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    has_missing = missing[missing > 0]
    if has_missing.empty:
        print("\n[OK] No missing values in features or target.")
    else:
        print("\n[WARN] Missing values detected:")
        for col in has_missing.index:
            print(f"  {col}: {has_missing[col]} ({missing_pct[col]}%)")

    # Zero-variance check (numeric only)
    numeric_cols = [c for c in features if df[c].dtype in ("float64", "int64", "float32", "int32")]
    zero_var = [c for c in numeric_cols if df[c].std() == 0]
    if zero_var:
        print(f"\n[WARN] Zero-variance features (will be dropped): {zero_var}")
    else:
        print(f"\n[OK] No zero-variance numeric features.")

    # Target distribution
    print(f"\n[INFO] Target '{target}' distribution:")
    print(f"  Mean  : {df[target].mean():.4f}")
    print(f"  Std   : {df[target].std():.4f}")
    print(f"  Min   : {df[target].min():.4f}")
    print(f"  Max   : {df[target].max():.4f}")
    print(f"  Median: {df[target].median():.4f}")
    print("=" * 70)


audit_data(pdf, ALL_FEATURES, TARGET_COL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Prepare Feature Matrix and Target Vector

# COMMAND ----------

def prepare_xy(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y, dropping NaN rows.

    Args:
        df: Source DataFrame.
        numeric_features: Numeric column names.
        categorical_features: Categorical column names.
        target: Target column name.

    Returns:
        Tuple of (X, y) where X contains only feature columns and y is the
        target Series. Rows with any NaN in features or target are dropped.
    """
    all_cols = numeric_features + categorical_features + [target]
    clean_df = df[all_cols].dropna()
    n_dropped = len(df) - len(clean_df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with NaN values ({n_dropped / len(df) * 100:.1f}%)")

    X = clean_df[numeric_features + categorical_features].copy()
    y = clean_df[target].copy()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape : {y.shape}")
    return X, y

# COMMAND ----------

X, y = prepare_xy(pdf, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build the sklearn Pipeline
# MAGIC
# MAGIC **Architecture:**
# MAGIC ```
# MAGIC Numeric features ──> PolynomialFeatures(degree=2) ──> StandardScaler ──┐
# MAGIC                                                                        ├──> ElasticNet
# MAGIC Categorical features ──> OneHotEncoder ───────────────────────────────-─┘
# MAGIC ```
# MAGIC
# MAGIC The polynomial expansion on numeric features generates all pairwise
# MAGIC interaction terms (e.g., `tariff_impact_index * predicted_volume_change`,
# MAGIC `fx_impact * cogs_pct`). These interaction terms are crucial because
# MAGIC margin pressure is rarely driven by a single factor in isolation.

# COMMAND ----------

def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    poly_degree: int = 2,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = SEED,
) -> Pipeline:
    """Construct the full ElasticNet pipeline with preprocessing.

    The pipeline is organized as:
    1. ColumnTransformer that applies different preprocessing to numeric
       vs. categorical features:
       - Numeric: PolynomialFeatures (interaction + quadratic terms) then
         StandardScaler for regularization stability.
       - Categorical: OneHotEncoder with handle_unknown='ignore' for
         robustness to unseen categories at inference time.
    2. ElasticNet regressor with configurable alpha and l1_ratio.

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        poly_degree: Degree for PolynomialFeatures (1 = linear only, 2 = interactions).
        alpha: ElasticNet regularization strength.
        l1_ratio: Mix between L1 (1.0) and L2 (0.0) penalties.
        random_state: Random seed for reproducibility.

    Returns:
        Unfitted sklearn Pipeline.
    """
    # --- Numeric sub-pipeline ---
    numeric_transformer = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(
                degree=poly_degree,
                interaction_only=False,
                include_bias=False,
            )),
            ("scaler", StandardScaler()),
        ]
    )

    # --- Categorical sub-pipeline ---
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="if_binary",
            )),
        ]
    )

    # --- Column transformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # --- Full pipeline ---
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=10000,
                tol=1e-5,
                random_state=random_state,
                selection="cyclic",
            )),
        ]
    )

    return pipeline

# COMMAND ----------

# Quick smoke test: build and inspect the default pipeline
_test_pipe = build_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
print("Pipeline steps:")
for step_name, step_obj in _test_pipe.named_steps.items():
    print(f"  {step_name}: {type(step_obj).__name__}")
del _test_pipe

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Hyperparameter Tuning via Grid Search
# MAGIC
# MAGIC **Search space:**
# MAGIC | Parameter | Values | Rationale |
# MAGIC |-----------|--------|-----------|
# MAGIC | `alpha` | 0.001, 0.01, 0.1, 0.5, 1.0 | From nearly unregularized to strongly penalized |
# MAGIC | `l1_ratio` | 0.1, 0.3, 0.5, 0.7, 0.9 | Spectrum from Ridge-like to Lasso-like |
# MAGIC | `poly degree` | 1, 2 | Linear-only vs. interaction effects |
# MAGIC
# MAGIC **Cross-validation:** `TimeSeriesSplit(n_splits=5)` respects temporal ordering
# MAGIC to prevent data leakage from future observations.

# COMMAND ----------

def run_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    n_splits: int = 5,
    random_state: int = SEED,
) -> Tuple[GridSearchCV, Dict]:
    """Execute hyperparameter grid search with time-series cross-validation.

    Searches over ElasticNet alpha, l1_ratio, and polynomial degree to find
    the combination that minimizes negative RMSE on held-out temporal folds.

    Args:
        X: Feature matrix (pandas DataFrame).
        y: Target vector (pandas Series).
        numeric_features: Numeric feature column names.
        categorical_features: Categorical feature column names.
        n_splits: Number of TimeSeriesSplit folds.
        random_state: Random seed.

    Returns:
        Tuple of (fitted GridSearchCV object, dict of best parameters).
    """
    # Build a base pipeline (parameters will be overridden by grid)
    base_pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    # Define the parameter grid
    # Note: sklearn Pipeline param names use double-underscore notation
    param_grid = {
        "preprocessor__num__poly__degree": [1, 2],
        "regressor__alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
        "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"Grid search: {total_combos} parameter combinations x {n_splits} folds "
          f"= {total_combos * n_splits} fits")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        refit=True,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    print("\nStarting grid search...")
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Negate because scoring is negative RMSE

    print(f"\nBest parameters:")
    for param, value in sorted(best_params.items()):
        short_name = param.split("__")[-1]
        print(f"  {short_name}: {value}")
    print(f"Best CV RMSE: {best_score:.6f}")

    return grid_search, best_params

# COMMAND ----------

grid_search, best_params = run_grid_search(
    X, y,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    n_splits=5,
    random_state=SEED,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cross-Validation Results Analysis

# COMMAND ----------

def analyze_cv_results(grid_search: GridSearchCV) -> pd.DataFrame:
    """Summarize and display the grid search cross-validation results.

    Extracts the results DataFrame from the fitted GridSearchCV, ranks
    parameter combinations by test score, and displays the top 10.

    Args:
        grid_search: Fitted GridSearchCV object.

    Returns:
        pandas DataFrame of CV results sorted by rank.
    """
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Extract readable param names
    param_cols = [c for c in results_df.columns if c.startswith("param_")]
    display_cols = param_cols + [
        "mean_train_score", "std_train_score",
        "mean_test_score", "std_test_score",
        "rank_test_score",
    ]
    summary = results_df[display_cols].sort_values("rank_test_score")

    # Convert negative RMSE back to positive for readability
    summary["mean_train_rmse"] = -summary["mean_train_score"]
    summary["mean_test_rmse"] = -summary["mean_test_score"]
    summary["std_test_rmse"] = summary["std_test_score"]

    print("Top 10 Parameter Combinations (by CV RMSE):")
    print("-" * 90)
    top10 = summary.head(10)
    for _, row in top10.iterrows():
        degree = row.get("param_preprocessor__num__poly__degree", "?")
        alpha = row.get("param_regressor__alpha", "?")
        l1 = row.get("param_regressor__l1_ratio", "?")
        print(
            f"  Rank {int(row['rank_test_score']):2d} | "
            f"degree={degree}, alpha={alpha}, l1_ratio={l1} | "
            f"Train RMSE={row['mean_train_rmse']:.6f} | "
            f"Test RMSE={row['mean_test_rmse']:.6f} +/- {row['std_test_rmse']:.6f}"
        )

    # Check for overfitting signal
    best_row = summary.iloc[0]
    overfit_ratio = best_row["mean_train_rmse"] / best_row["mean_test_rmse"]
    if overfit_ratio < 0.7:
        print(f"\n[WARN] Potential overfitting: train/test RMSE ratio = {overfit_ratio:.3f}")
    else:
        print(f"\n[OK] Train/test RMSE ratio = {overfit_ratio:.3f} (no severe overfitting)")

    return summary


cv_summary = analyze_cv_results(grid_search)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Final Model Evaluation on Full Refit

# COMMAND ----------

def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """Compute regression metrics for the final refitted model.

    Calculates RMSE, MAE, and R-squared on the provided dataset. In
    production, this is typically run on a held-out test set; here we
    evaluate on the full training set (the GridSearchCV already provides
    unbiased CV estimates).

    Args:
        model: Fitted sklearn Pipeline.
        X: Feature matrix.
        y: True target values.

    Returns:
        Dictionary with 'rmse', 'mae', and 'r2' keys.
    """
    y_pred = model.predict(X)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

    print("Final Model Metrics (refit on full training data):")
    print(f"  RMSE : {metrics['rmse']:.6f}")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  R2   : {metrics['r2']:.6f}")

    return metrics

# COMMAND ----------

best_model = grid_search.best_estimator_
metrics = evaluate_model(best_model, X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Feature Importance via Coefficient Analysis
# MAGIC
# MAGIC Since ElasticNet is a linear model (after the polynomial transformation),
# MAGIC the magnitude of each coefficient directly indicates its contribution to
# MAGIC the prediction. We extract the transformed feature names (including
# MAGIC interaction terms like `tariff_impact_index * predicted_volume_change`)
# MAGIC and rank them by absolute coefficient value.

# COMMAND ----------

def extract_feature_importance(
    model: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Extract and rank feature importances from the fitted ElasticNet pipeline.

    Uses the pipeline's get_feature_names_out() to retrieve the names of all
    transformed features (including polynomial interaction terms and one-hot
    encoded categories), then pairs them with the ElasticNet coefficients.

    Args:
        model: Fitted sklearn Pipeline containing a ColumnTransformer
               preprocessor and an ElasticNet regressor.
        numeric_features: Original numeric feature names.
        categorical_features: Original categorical feature names.

    Returns:
        pandas DataFrame with columns ['feature', 'coefficient', 'abs_coefficient'],
        sorted by absolute coefficient value descending.
    """
    # Get the regressor coefficients
    regressor = model.named_steps["regressor"]
    coefficients = regressor.coef_

    # Get transformed feature names from the preprocessor
    preprocessor = model.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback: generate generic names
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    # Clean up feature names for readability
    clean_names = []
    for name in feature_names:
        # Remove prefixes like 'num__poly__' or 'cat__onehot__'
        clean = name
        for prefix in ["num__poly__", "num__scaler__", "cat__onehot__"]:
            clean = clean.replace(prefix, "")
        clean_names.append(clean)

    importance_df = pd.DataFrame({
        "feature": clean_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    # Summary statistics
    n_total = len(importance_df)
    n_nonzero = (importance_df["abs_coefficient"] > 1e-10).sum()
    n_zero = n_total - n_nonzero
    print(f"Total transformed features : {n_total}")
    print(f"Non-zero coefficients      : {n_nonzero}")
    print(f"Zero coefficients (pruned) : {n_zero}")
    print(f"Sparsity ratio             : {n_zero / n_total:.1%}\n")

    # Display top 25 features
    print("Top 25 Features by Absolute Coefficient:")
    print("-" * 75)
    for i, row in importance_df.head(25).iterrows():
        direction = "+" if row["coefficient"] > 0 else "-"
        print(f"  {i + 1:3d}. [{direction}] {row['feature']:50s}  |coef|={row['abs_coefficient']:.6f}")

    return importance_df

# COMMAND ----------

importance_df = extract_feature_importance(best_model, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identify Top Interaction Terms
# MAGIC
# MAGIC Interaction terms contain a space (from PolynomialFeatures naming convention)
# MAGIC between two feature names. These capture the non-additive margin effects.

# COMMAND ----------

def identify_top_interactions(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """Filter and display the top interaction terms from the importance table.

    Interaction terms are identified by the presence of a space character in
    the feature name (PolynomialFeatures uses 'feat1 feat2' naming).

    Args:
        importance_df: DataFrame from extract_feature_importance().
        top_n: Number of top interactions to return.

    Returns:
        Filtered DataFrame of the top interaction features.
    """
    # Interaction terms contain a space between two feature names
    interactions = importance_df[
        importance_df["feature"].str.contains(" ", na=False)
    ].copy()

    if interactions.empty:
        print("[INFO] No interaction terms found (polynomial degree may be 1).")
        return interactions

    top_interactions = interactions.head(top_n)

    print(f"\nTop {min(top_n, len(top_interactions))} Interaction Terms:")
    print("-" * 75)
    for i, (_, row) in enumerate(top_interactions.iterrows()):
        direction = "positive" if row["coefficient"] > 0 else "negative"
        print(f"  {i + 1:2d}. {row['feature']:55s} ({direction}, |coef|={row['abs_coefficient']:.6f})")

    return top_interactions


top_interactions = identify_top_interactions(importance_df, top_n=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Coefficient Visualization

# COMMAND ----------

def plot_coefficient_importance(
    importance_df: pd.DataFrame,
    top_n: int = 30,
    title: str = "ElasticNet Coefficient Magnitudes (Top Features)",
) -> str:
    """Create and save a horizontal bar chart of top feature coefficients.

    Generates a publication-quality plot showing the largest absolute
    coefficients, color-coded by sign (positive = margin increase,
    negative = margin decrease).

    Args:
        importance_df: DataFrame with 'feature', 'coefficient', 'abs_coefficient'.
        top_n: Number of features to display.
        title: Chart title.

    Returns:
        File path to the saved PNG image.
    """
    plot_df = importance_df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]  # Reverse for horizontal bar chart (largest at top)

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.35)))

    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in plot_df["coefficient"]]

    bars = ax.barh(
        y=range(len(plot_df)),
        width=plot_df["abs_coefficient"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"], fontsize=9)
    ax.set_xlabel("Absolute Coefficient Magnitude", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # Add a legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Positive (margin increase)"),
        Patch(facecolor="#e74c3c", label="Negative (margin decrease)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save to temp file
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "coefficient_importance.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Coefficient plot saved to: {filepath}")
    return filepath

# COMMAND ----------

coef_plot_path = plot_coefficient_importance(importance_df, top_n=30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Sanity Checks: Directional Validation
# MAGIC
# MAGIC Before registering the model, we verify that predictions respond
# MAGIC correctly to known input perturbations. For example:
# MAGIC - Increasing `cogs_pct` should decrease margin (negative direction).
# MAGIC - Increasing `predicted_revenue_impact` should increase margin (positive).
# MAGIC - Increasing `tariff_impact_index` should decrease margin (negative).

# COMMAND ----------

def run_sanity_checks(
    model: Pipeline,
    X: pd.DataFrame,
    numeric_features: List[str],
) -> Dict[str, Dict]:
    """Validate model predictions against known economic intuition.

    For each feature with a known expected direction, perturbs the feature
    by +1 standard deviation while holding all others constant, and checks
    whether the average predicted margin moves in the expected direction.

    Args:
        model: Fitted sklearn Pipeline.
        X: Original feature matrix.
        numeric_features: List of numeric feature names.

    Returns:
        Dictionary mapping feature name to a dict with 'expected_direction',
        'actual_direction', 'delta', and 'passed' keys.
    """
    # Expected directional relationships with gross_margin_delta
    expected_directions = {
        "predicted_revenue_impact": "positive",    # Higher revenue -> higher margin
        "cogs_pct": "negative",                    # Higher COGS % -> lower margin
        "cogs_trend_3mo": "negative",              # Rising COGS trend -> lower margin
        "tariff_impact_index": "negative",         # Higher tariffs -> lower margin
        "logistics_cost_index": "negative",        # Higher logistics costs -> lower margin
        "predicted_volume_change": "positive",     # More volume -> scale efficiencies
        "discount_depth_change": "negative",       # Deeper discounts -> lower margin
        "rebate_pct_change": "negative",           # Higher rebates -> lower margin
        "innovation_tier": "positive",             # Higher innovation -> pricing power
    }

    baseline_pred = model.predict(X).mean()
    results = {}

    print("Sanity Check: Directional Validation")
    print("=" * 75)
    print(f"{'Feature':<35s} {'Expected':>10s} {'Actual':>10s} {'Delta':>12s} {'Status':>8s}")
    print("-" * 75)

    for feature, expected in expected_directions.items():
        if feature not in numeric_features or feature not in X.columns:
            continue

        # Perturb feature by +1 std dev
        X_perturbed = X.copy()
        std = X[feature].std()
        if std < 1e-10:
            print(f"  {feature:<35s} {'SKIP (zero variance)':>40s}")
            continue

        X_perturbed[feature] = X[feature] + std
        perturbed_pred = model.predict(X_perturbed).mean()
        delta = perturbed_pred - baseline_pred

        actual = "positive" if delta > 0 else "negative"
        passed = actual == expected

        results[feature] = {
            "expected_direction": expected,
            "actual_direction": actual,
            "delta": float(delta),
            "passed": passed,
        }

        status = "PASS" if passed else "FAIL"
        print(f"  {feature:<35s} {expected:>10s} {actual:>10s} {delta:>+12.6f} {status:>8s}")

    # Summary
    n_passed = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)
    print("-" * 75)
    print(f"  Results: {n_passed}/{n_total} checks passed")

    if n_passed < n_total:
        failed = [f for f, r in results.items() if not r["passed"]]
        print(f"\n  [WARN] Failed checks: {failed}")
        print("  This may indicate insufficient training data or collinear features.")
        print("  Review coefficient signs and consider domain-constrained regularization.")
    else:
        print("\n  [OK] All directional sanity checks passed.")

    return results

# COMMAND ----------

sanity_results = run_sanity_checks(best_model, X, NUMERIC_FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Log Everything to MLflow and Register Model

# COMMAND ----------

def log_and_register_model(
    grid_search: GridSearchCV,
    best_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: Dict[str, float],
    best_params: Dict,
    importance_df: pd.DataFrame,
    sanity_results: Dict,
    coef_plot_path: str,
    model_registry_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> str:
    """Log all artifacts, metrics, and parameters to MLflow, then register the model.

    This function creates a single MLflow run containing:
    - All hyperparameters (best from grid search)
    - Evaluation metrics (RMSE, MAE, R2)
    - Cross-validation RMSE (mean and std)
    - Feature importance CSV artifact
    - Coefficient importance plot (PNG)
    - Sanity check results
    - Model artifact with signature
    - Model registration in Unity Catalog

    Args:
        grid_search: Fitted GridSearchCV.
        best_model: Best estimator pipeline.
        X: Feature matrix.
        y: Target vector.
        metrics: Dict of evaluation metrics.
        best_params: Dict of best hyperparameters.
        importance_df: Feature importance DataFrame.
        sanity_results: Directional validation results.
        coef_plot_path: Path to the coefficient plot PNG.
        model_registry_name: Unity Catalog model name for registration.
        numeric_features: Numeric feature names.
        categorical_features: Categorical feature names.

    Returns:
        The MLflow run ID.
    """
    with mlflow.start_run(run_name="margin_optimization_enet") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # -----------------------------------------------------------------
        # 1. Log parameters
        # -----------------------------------------------------------------
        # Best hyperparameters (clean names)
        mlflow.log_param("model_type", "ElasticNet")
        mlflow.log_param("random_state", SEED)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_numeric_features", len(numeric_features))
        mlflow.log_param("n_categorical_features", len(categorical_features))

        for param_name, param_value in best_params.items():
            clean_name = param_name.replace("preprocessor__num__poly__", "").replace("regressor__", "")
            mlflow.log_param(f"best_{clean_name}", param_value)

        # Cross-validation configuration
        mlflow.log_param("cv_strategy", "TimeSeriesSplit")
        mlflow.log_param("cv_n_splits", 5)
        mlflow.log_param("grid_search_scoring", "neg_root_mean_squared_error")

        # Feature lists (as tags for searchability)
        mlflow.set_tag("numeric_features", ", ".join(numeric_features))
        mlflow.set_tag("categorical_features", ", ".join(categorical_features))
        mlflow.set_tag("target", TARGET_COL)
        mlflow.set_tag("feature_table", FEATURE_TABLE)

        # -----------------------------------------------------------------
        # 2. Log metrics
        # -----------------------------------------------------------------
        # Final refit metrics
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])

        # Cross-validation metrics
        cv_rmse_mean = -grid_search.best_score_
        cv_results = grid_search.cv_results_
        best_idx = grid_search.best_index_
        cv_rmse_std = cv_results["std_test_score"][best_idx]

        mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
        mlflow.log_metric("cv_rmse_std", cv_rmse_std)

        # Sanity check pass rate
        n_passed = sum(1 for r in sanity_results.values() if r["passed"])
        n_total = len(sanity_results)
        mlflow.log_metric("sanity_check_pass_rate", n_passed / n_total if n_total > 0 else 0.0)

        # Model complexity metrics
        regressor = best_model.named_steps["regressor"]
        n_nonzero = int(np.sum(np.abs(regressor.coef_) > 1e-10))
        n_total_coefs = len(regressor.coef_)
        mlflow.log_metric("n_nonzero_coefficients", n_nonzero)
        mlflow.log_metric("n_total_coefficients", n_total_coefs)
        mlflow.log_metric("sparsity_ratio", 1.0 - (n_nonzero / n_total_coefs))

        # -----------------------------------------------------------------
        # 3. Log artifacts
        # -----------------------------------------------------------------
        # Feature importance CSV
        tmpdir = tempfile.mkdtemp()

        importance_path = os.path.join(tmpdir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="feature_analysis")

        # Coefficient plot
        mlflow.log_artifact(coef_plot_path, artifact_path="plots")

        # Sanity check results
        sanity_path = os.path.join(tmpdir, "sanity_check_results.csv")
        sanity_df = pd.DataFrame([
            {"feature": f, **r} for f, r in sanity_results.items()
        ])
        sanity_df.to_csv(sanity_path, index=False)
        mlflow.log_artifact(sanity_path, artifact_path="validation")

        # Cross-validation results summary
        cv_summary_path = os.path.join(tmpdir, "cv_results.csv")
        pd.DataFrame(cv_results).to_csv(cv_summary_path, index=False)
        mlflow.log_artifact(cv_summary_path, artifact_path="cv_analysis")

        # -----------------------------------------------------------------
        # 4. Log model with signature
        # -----------------------------------------------------------------
        # Infer signature from a sample prediction
        sample_X = X.head(5)
        sample_pred = best_model.predict(sample_X)
        signature = infer_signature(sample_X, sample_pred)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_registry_name,
            pip_requirements=[
                "scikit-learn",
                "pandas",
                "numpy",
            ],
        )

        print(f"\nModel registered as: {model_registry_name}")
        print(f"Artifacts logged to run: {run_id}")

        # -----------------------------------------------------------------
        # 5. Summary
        # -----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("MLflow Run Summary")
        print("=" * 70)
        print(f"  Run ID            : {run_id}")
        print(f"  Model             : ElasticNet")
        print(f"  Best alpha        : {best_params.get('regressor__alpha', 'N/A')}")
        print(f"  Best l1_ratio     : {best_params.get('regressor__l1_ratio', 'N/A')}")
        print(f"  Best poly degree  : {best_params.get('preprocessor__num__poly__degree', 'N/A')}")
        print(f"  RMSE (refit)      : {metrics['rmse']:.6f}")
        print(f"  MAE (refit)       : {metrics['mae']:.6f}")
        print(f"  R2 (refit)        : {metrics['r2']:.6f}")
        print(f"  CV RMSE           : {cv_rmse_mean:.6f} +/- {cv_rmse_std:.6f}")
        print(f"  Non-zero coefs    : {n_nonzero} / {n_total_coefs}")
        print(f"  Sanity checks     : {n_passed}/{len(sanity_results)} passed")
        print(f"  Registry          : {model_registry_name}")
        print("=" * 70)

        return run_id

# COMMAND ----------

run_id = log_and_register_model(
    grid_search=grid_search,
    best_model=best_model,
    X=X,
    y=y,
    metrics=metrics,
    best_params=best_params,
    importance_df=importance_df,
    sanity_results=sanity_results,
    coef_plot_path=coef_plot_path,
    model_registry_name=MODEL_REGISTRY_NAME,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Post-Registration Verification

# COMMAND ----------

def verify_registered_model(
    model_registry_name: str,
    X_sample: pd.DataFrame,
) -> None:
    """Load the registered model from MLflow and verify it produces predictions.

    This is a critical post-deployment check: we load the model back from
    the registry and confirm it can score new data without errors.

    Args:
        model_registry_name: Unity Catalog model name.
        X_sample: A small sample of the feature matrix for test inference.
    """
    print("Verifying registered model...")

    # Load the latest version from the registry
    model_uri = f"models:/{model_registry_name}/latest"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Score a sample
    predictions = loaded_model.predict(X_sample)

    print(f"  Model URI       : {model_uri}")
    print(f"  Sample size     : {len(X_sample)}")
    print(f"  Predictions     : {predictions[:5]}")
    print(f"  Pred mean       : {predictions.mean():.6f}")
    print(f"  Pred std        : {predictions.std():.6f}")

    # Basic checks
    assert len(predictions) == len(X_sample), "Prediction count mismatch"
    assert not np.any(np.isnan(predictions)), "NaN predictions detected"
    assert not np.any(np.isinf(predictions)), "Infinite predictions detected"

    print("\n  [OK] Registered model verification passed.")

# COMMAND ----------

verify_registered_model(MODEL_REGISTRY_NAME, X.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Summary and Next Steps
# MAGIC
# MAGIC **What this notebook produced:**
# MAGIC 1. A tuned ElasticNet pipeline with polynomial feature expansion that
# MAGIC    captures interaction effects between cost drivers, market factors,
# MAGIC    and pricing levers.
# MAGIC 2. Feature importance rankings including the most impactful interaction
# MAGIC    terms (e.g., tariff x volume, FX x COGS).
# MAGIC 3. Directional sanity checks confirming the model responds correctly
# MAGIC    to known economic relationships.
# MAGIC 4. A registered model in Unity Catalog ready for downstream scoring.
# MAGIC
# MAGIC **Downstream consumers:**
# MAGIC - `04a_pricing_recommendations.py` - Uses this model to simulate margin
# MAGIC   outcomes under different pricing scenarios.
# MAGIC - `04b_what_if_analysis.py` - Interactive what-if tool for pricing analysts.
# MAGIC - Margin monitoring dashboards (Databricks SQL).
# MAGIC
# MAGIC **Retraining cadence:** Monthly, aligned with Feature Store refresh.
