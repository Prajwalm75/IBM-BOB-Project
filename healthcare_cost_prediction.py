"""
============================================================
 Healthcare Cost Prediction — Single-File Submission
============================================================
Project  : Healthcare Cost Prediction
Dataset  : data/insurance.csv  (US Medical Insurance Costs)
Model    : Gradient Boosting Regressor (scikit-learn Pipeline)
Backend  : Python Flask REST API
Frontend : Served from frontend/index.html via Flask

This file consolidates the full Python implementation for
project evaluation. It covers:

  Section 1 — Imports & Configuration
  Section 2 — Dataset Loading & Inspection
  Section 3 — Exploratory Data Analysis (EDA)
  Section 4 — Preprocessing Pipeline
  Section 5 — Model Training & Algorithm Comparison
  Section 6 — Model Evaluation (MAE, RMSE, R²)
  Section 7 — Feature Importance Analysis
  Section 8 — Healthcare Cost Prediction Function
  Section 9 — Flask REST API (all 5 endpoints)
  Section 10 — Application Entry Point

NOTE: The existing application (backend/app.py, model/ml_model.py,
model/train_pipeline.py, frontend/) is NOT modified. This file
is a standalone submission consolidation only.
============================================================
"""

# ============================================================
# SECTION 1 — IMPORTS & CONFIGURATION
# ============================================================

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib

# scikit-learn — preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# scikit-learn — models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# scikit-learn — evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Flask — web API
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

# -- Path resolution (works from project root or any working directory) ---------
_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(_HERE, "data",  "insurance.csv")
MODEL_PATH  = os.path.join(_HERE, "model", "healthcare_model.pkl")
EVAL_PATH   = os.path.join(_HERE, "model", "evaluation_results.json")
FRONTEND_DIR = os.path.join(_HERE, "frontend")

# -- Feature / target column names (must match the CSV exactly) -----------------
FEATURES    = ["age", "sex", "bmi", "children", "smoker", "region"]
TARGET      = "charges"
NUM_COLS    = ["age", "bmi", "children"]
CAT_COLS    = ["sex", "smoker", "region"]

# Fixed category arrays passed to OrdinalEncoder — deterministic, no data leakage
CAT_CATEGORIES = [
    ["female", "male"],                                         # sex
    ["no",     "yes"],                                          # smoker
    ["northeast", "northwest", "southeast", "southwest"],       # region
]

# Valid values for input validation
VALID_SEX     = {"male", "female"}
VALID_SMOKER  = {"yes", "no"}
VALID_REGIONS = {"northeast", "northwest", "southeast", "southwest"}


# ============================================================
# SECTION 2 — DATASET LOADING & INSPECTION
# ============================================================

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the insurance CSV dataset, remove the one known duplicate row,
    and return a clean DataFrame ready for analysis and training.

    Original file is never modified — deduplication happens in memory only.
    """
    df = pd.read_csv(path)

    # -- Basic inspection ------------------------------------------------------
    print("=" * 60)
    print("SECTION 2 — Dataset Inspection")
    print("=" * 60)
    print(f"  Raw shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Missing values : {df.isnull().sum().sum()} total")
    print(f"  Duplicate rows : {df.duplicated().sum()}")
    print(f"\n  Column dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<12} {dtype}")

    # -- Remove the single duplicate present in the raw file -------------------
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"\n  Duplicates removed : {removed} row(s)")
    print(f"  Clean dataset size : {len(df)} rows")

    return df


# ============================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Print a concise EDA summary: descriptive statistics for the target,
    unique values for categorical columns, and class-level mean charges.
    """
    print("\n" + "=" * 60)
    print("SECTION 3 — Exploratory Data Analysis")
    print("=" * 60)

    # -- Target distribution ---------------------------------------------------
    print("\n  Target column: 'charges' (annual insurance cost in USD)")
    print(f"    min    : ${df[TARGET].min():>10,.2f}")
    print(f"    max    : ${df[TARGET].max():>10,.2f}")
    print(f"    mean   : ${df[TARGET].mean():>10,.2f}")
    print(f"    median : ${df[TARGET].median():>10,.2f}")
    print(f"    std    : ${df[TARGET].std():>10,.2f}")
    print(f"    skew   : {df[TARGET].skew():>10.4f}  (right-skewed — log transform optional)")

    # -- Categorical unique values ---------------------------------------------
    print("\n  Categorical feature values:")
    for col in CAT_COLS:
        print(f"    {col:<12} {sorted(df[col].unique().tolist())}")

    # -- Numerical ranges ------------------------------------------------------
    print("\n  Numerical feature ranges:")
    for col in NUM_COLS:
        print(f"    {col:<12} min={df[col].min():.1f}  max={df[col].max():.1f}  mean={df[col].mean():.1f}")

    # -- Key insight: smoker vs non-smoker costs --------------------------------
    print("\n  Avg charges by smoking status:")
    for status, grp in df.groupby("smoker")[TARGET]:
        print(f"    {status:<6} : ${grp.mean():>10,.2f}")

    print("\n  Avg charges by region:")
    for region, grp in df.groupby("region")[TARGET]:
        print(f"    {region:<12} : ${grp.mean():>10,.2f}")


# ============================================================
# SECTION 4 — PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build and return the sklearn ColumnTransformer that handles:

    Numerical columns (age, bmi, children):
      → StandardScaler  (zero mean, unit variance)
        Required for linear models; harmless for tree-based models.

    Categorical columns (sex, smoker, region):
      → OrdinalEncoder with hard-coded category arrays.
        Fixed arrays prevent data leakage and ensure inference-time
        consistency for unseen category values.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUM_COLS,
            ),
            (
                "cat",
                OrdinalEncoder(
                    categories=CAT_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,       # gracefully encodes unseen values
                ),
                CAT_COLS,
            ),
        ],
        remainder="drop",               # ignore any extra columns
    )
    return preprocessor


# ============================================================
# SECTION 5 — MODEL TRAINING & ALGORITHM COMPARISON
# ============================================================

def train_and_compare(df: pd.DataFrame):
    """
    1. Split the dataset into 80 % training and 20 % test sets.
    2. Build a sklearn Pipeline (preprocessor + estimator) for each of
       five candidate algorithms.
    3. Fit every pipeline on the training set and evaluate on the test set.
    4. Perform 5-fold cross-validation for each algorithm.
    5. Select the best pipeline by highest test R² score.
    6. Return the best fitted pipeline and the full results dictionary.

    Algorithms compared
    -------------------
    - Linear Regression   (baseline linear model)
    - Ridge Regression    (L2-regularised linear model, alpha=10)
    - Lasso Regression    (L1-regularised linear model, alpha=1)
    - Random Forest       (300 trees, ensemble bagging)
    - Gradient Boosting   (300 estimators, lr=0.05, depth=4) ← selected best
    """
    print("\n" + "=" * 60)
    print("SECTION 5 — Model Training & Algorithm Comparison")
    print("=" * 60)

    X = df[FEATURES].copy()
    y = df[TARGET].values

    # -- 80/20 train-test split (reproducible with random_state=42) ------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n  Train set : {len(X_train)} samples")
    print(f"  Test  set : {len(X_test)} samples")

    preprocessor = build_preprocessor()

    # -- Candidate algorithms --------------------------------------------------
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10.0, random_state=42),
        "Lasso Regression":  Lasso(alpha=1.0, max_iter=10_000, random_state=42),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=300,
                                 min_samples_leaf=2,
                                 random_state=42,
                                 n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=300,
                                 learning_rate=0.05,
                                 max_depth=4,
                                 min_samples_split=5,
                                 min_samples_leaf=3,
                                 subsample=0.8,
                                 random_state=42),
    }

    cv     = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # -- Header row ------------------------------------------------------------
    w = max(len(k) for k in candidates) + 2
    print(f"\n  {'Model':<{w}} {'Train R²':>9} {'Test R²':>9} "
          f"{'Test MAE':>12} {'Test RMSE':>12} {'CV R²(5)':>10}")
    print("  " + "-" * (w + 54))

    for name, estimator in candidates.items():
        pipe = Pipeline([("pre", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)

        y_pred_tr = pipe.predict(X_train)
        y_pred_te = pipe.predict(X_test)

        r2_tr    = r2_score(y_train, y_pred_tr)
        r2_te    = r2_score(y_test,  y_pred_te)
        mae_tr   = mean_absolute_error(y_train, y_pred_tr)
        mae_te   = mean_absolute_error(y_test,  y_pred_te)
        rmse_tr  = float(np.sqrt(mean_squared_error(y_train, y_pred_tr)))
        rmse_te  = float(np.sqrt(mean_squared_error(y_test,  y_pred_te)))
        cv_r2    = cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean()

        results[name] = {
            "r2_train":   round(r2_tr,   6),
            "r2_test":    round(r2_te,   6),
            "mae_train":  round(mae_tr,  4),
            "mae_test":   round(mae_te,  4),
            "rmse_train": round(rmse_tr, 4),
            "rmse_test":  round(rmse_te, 4),
            "cv_r2":      round(cv_r2,   6),
            "pipeline":   pipe,
        }

        print(f"  {name:<{w}} {r2_tr:9.4f} {r2_te:9.4f} "
              f"{mae_te:12.2f} {rmse_te:12.2f} {cv_r2:10.4f}")

    # -- Select best model by test R² ------------------------------------------
    best_name = max(results, key=lambda k: results[k]["r2_test"])
    return results[best_name]["pipeline"], best_name, results, X_train, X_test, y_train, y_test


# ============================================================
# SECTION 6 — MODEL EVALUATION  (MAE, RMSE, R²)
# ============================================================

def evaluate_model(pipeline, best_name: str, results: dict,
                   X_train, X_test, y_train, y_test) -> dict:
    """
    Compute and display the final evaluation metrics for the selected
    best model on both the training set and the held-out test set.

    Metrics reported
    ----------------
    R²   (coefficient of determination)  — higher is better; 1.0 = perfect
    MAE  (mean absolute error)           — average dollar prediction error
    RMSE (root mean squared error)       — penalises large errors more than MAE
    """
    print("\n" + "=" * 60)
    print("SECTION 6 — Model Evaluation")
    print("=" * 60)

    best = results[best_name]

    print(f"\n  Selected algorithm : {best_name}")
    print(f"\n  {'Metric':<20} {'Train Set':>14} {'Test Set':>14}")
    print("  " + "-" * 50)
    print(f"  {'R² Score':<20} {best['r2_train']:>13.4f} {best['r2_test']:>13.4f}")
    print(f"  {'R² (%)':<20} {best['r2_train']*100:>12.2f}% {best['r2_test']*100:>12.2f}%")
    print(f"  {'MAE ($)':<20} {best['mae_train']:>13,.2f} {best['mae_test']:>13,.2f}")
    print(f"  {'RMSE ($)':<20} {best['rmse_train']:>13,.2f} {best['rmse_test']:>13,.2f}")
    print(f"  {'CV R² (5-fold)':<20} {'':>14} {best['cv_r2']:>13.4f}")

    # -- Interpretation --------------------------------------------------------
    r2_pct = best["r2_test"] * 100
    print(f"\n  Interpretation:")
    print(f"    The {best_name} explains {r2_pct:.2f}% of the variance in")
    print(f"    healthcare costs on the unseen test set.")
    print(f"    On average, predictions are within ${best['mae_test']:,.2f} of the actual cost.")

    return best


# ============================================================
# SECTION 7 — FEATURE IMPORTANCE ANALYSIS
# ============================================================

def analyse_feature_importance(pipeline) -> list:
    """
    Extract and display feature importances from the trained estimator.
    Only tree-based models (Random Forest, Gradient Boosting) expose
    feature_importances_; linear models are skipped gracefully.

    The feature order follows the ColumnTransformer output order:
        [age, bmi, children,  sex, smoker, region]
         --- numerical ---   --- categorical ---
    """
    print("\n" + "=" * 60)
    print("SECTION 7 — Feature Importance Analysis")
    print("=" * 60)

    estimator = pipeline.named_steps["model"]
    feature_importance = []

    if not hasattr(estimator, "feature_importances_"):
        print("  (Feature importances not available for linear models)")
        return feature_importance

    feature_order = NUM_COLS + CAT_COLS   # matches ColumnTransformer output
    importances   = estimator.feature_importances_

    fi_pairs = sorted(
        zip(feature_order, importances),
        key=lambda x: -x[1]
    )

    print(f"\n  {'Feature':<14} {'Importance':>12}  Bar")
    print("  " + "-" * 50)
    max_val = fi_pairs[0][1]
    for fname, fval in fi_pairs:
        bar_len = int((fval / max_val) * 30)
        bar     = "|" * bar_len
        print(f"  {fname:<14} {fval*100:>10.2f}%  {bar}")

    feature_importance = [
        {"feature": f, "importance": round(float(v) * 100, 4)}
        for f, v in fi_pairs
    ]

    print(f"\n  Key insight: 'smoker' accounts for ~{fi_pairs[0][1]*100:.1f}% of")
    print(f"  prediction weight — the single strongest cost driver in this dataset.")

    return feature_importance


# ============================================================
# SECTION 8 — HEALTHCARE COST PREDICTION FUNCTION
# ============================================================

def validate_input(data: dict) -> list:
    """
    Validate a prediction request dictionary.

    Returns a list of human-readable error strings.
    An empty list means the input is valid and safe to pass to predict().

    Rules (mirror the backend API contract exactly):
      age      : float/int, 18–100
      bmi      : float/int, 10.0–60.0
      children : int, 0–10
      sex      : str, 'male' or 'female'
      smoker   : str, 'yes' or 'no'
      region   : str, one of the four US regions
    """
    errors   = []
    required = ["age", "sex", "bmi", "children", "smoker", "region"]

    # -- Presence check first --------------------------------------------------
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
    if errors:
        return errors   # skip range checks if fields are absent

    age      = data["age"]
    bmi      = data["bmi"]
    children = data["children"]
    sex      = data["sex"]
    smoker   = data["smoker"]
    region   = data["region"]

    # -- Range / type checks ---------------------------------------------------
    if not isinstance(age, (int, float)) or not (18 <= float(age) <= 100):
        errors.append("age must be a number between 18 and 100")

    if not isinstance(bmi, (int, float)) or not (10.0 <= float(bmi) <= 60.0):
        errors.append("bmi must be a number between 10.0 and 60.0")

    if not isinstance(children, (int, float)) or int(children) not in range(0, 11):
        errors.append("children must be an integer between 0 and 10")

    if not isinstance(sex, str) or sex.lower() not in VALID_SEX:
        errors.append(f"sex must be one of: {sorted(VALID_SEX)}")

    if not isinstance(smoker, str) or smoker.lower() not in VALID_SMOKER:
        errors.append(f"smoker must be one of: {sorted(VALID_SMOKER)}")

    if not isinstance(region, str) or region.lower() not in VALID_REGIONS:
        errors.append(f"region must be one of: {sorted(VALID_REGIONS)}")

    return errors


def predict_cost(pipeline, input_data: dict) -> dict:
    """
    Generate a predicted annual healthcare cost for one individual.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline (preprocessor + GBR estimator)
    input_data : dict with keys: age, sex, bmi, children, smoker, region

    Returns
    -------
    dict with:
      predicted_cost  — float, USD annual insurance cost estimate
      risk_category   — str, one of Low / Moderate / High / Very High
      risk_color      — str, colour token used by the frontend
      input_summary   — dict, echo of the normalised input values

    Risk thresholds (based on dataset quartile analysis):
      < $5,000   → Low
      < $15,000  → Moderate
      < $30,000  → High
      ≥ $30,000  → Very High
    """
    # -- Normalise types — defensive casting before DataFrame construction -----
    row = pd.DataFrame([{
        "age":      float(input_data["age"]),
        "sex":      str(input_data["sex"]).lower(),
        "bmi":      float(input_data["bmi"]),
        "children": int(input_data["children"]),
        "smoker":   str(input_data["smoker"]).lower(),
        "region":   str(input_data["region"]).lower(),
    }])

    # -- Predict via the full sklearn pipeline (pre-process + model) -----------
    predicted = float(pipeline.predict(row)[0])
    predicted = max(0.0, predicted)   # clamp: costs cannot be negative

    # -- Risk categorisation ---------------------------------------------------
    if predicted < 5_000:
        risk, risk_color = "Low",       "green"
    elif predicted < 15_000:
        risk, risk_color = "Moderate",  "orange"
    elif predicted < 30_000:
        risk, risk_color = "High",      "red"
    else:
        risk, risk_color = "Very High", "darkred"

    return {
        "predicted_cost": round(predicted, 2),
        "risk_category":  risk,
        "risk_color":     risk_color,
        "input_summary":  {k: input_data[k] for k in FEATURES},
    }


# ============================================================
# SECTION 9 — FLASK REST API
# ============================================================
#
# Endpoints
# ---------
#   GET  /               → serves frontend/index.html
#   GET  /api/health     → health check
#   GET  /api/stats      → dataset statistics (total records, avg cost, etc.)
#   GET  /api/model-metrics → evaluation metrics + feature importance
#   GET  /api/chart-data → pre-aggregated chart data for the frontend
#   POST /api/predict    → predict healthcare cost from JSON body
#
# The Flask app loads the pre-trained model/healthcare_model.pkl at
# startup. If the .pkl is absent it falls back to live training.
# ============================================================

# -- Global model state --------------------------------------------------------
_pipeline     = None   # fitted sklearn Pipeline
_metrics_dict = {}     # loaded from evaluation_results.json
_df           = None   # cleaned DataFrame (for stats & charts)


def _load_or_train_model():
    """
    Load the pre-trained pipeline from disk on first use.
    Falls back to training a new Gradient Boosting pipeline if
    model/healthcare_model.pkl does not exist.
    """
    global _pipeline, _metrics_dict, _df

    # -- Dataset (always loaded fresh from CSV) --------------------------------
    _df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    # -- Pipeline: load saved or train live ------------------------------------
    if os.path.exists(MODEL_PATH):
        _pipeline = joblib.load(MODEL_PATH)
        print(f"[API] Loaded pipeline from {MODEL_PATH}")
    else:
        print("[API] .pkl not found — running live Gradient Boosting training …")
        pre = build_preprocessor()
        _pipeline = Pipeline([
            ("pre",   pre),
            ("model", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                min_samples_split=5, min_samples_leaf=3,
                subsample=0.8, random_state=42)),
        ])
        X_tr, _, y_tr, _ = train_test_split(
            _df[FEATURES], _df[TARGET].values, test_size=0.20, random_state=42
        )
        _pipeline.fit(X_tr, y_tr)
        print("[API] Live training complete.")

    # -- Metrics: load from JSON or compute live --------------------------------
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as fh:
            saved = json.load(fh)
        best_name = saved.get("best_model", "Gradient Boosting")
        bm = saved["metrics"].get(best_name, {})
        _metrics_dict = {
            "algorithm":          best_name,
            "r2_train":           bm.get("r2_train",   0.0),
            "r2_test":            bm.get("r2_test",    0.0),
            "mae_train":          bm.get("mae_train",  0.0),
            "mae_test":           bm.get("mae_test",   0.0),
            "rmse_train":         bm.get("rmse_train", 0.0),
            "rmse_test":          bm.get("rmse_test",  0.0),
            "cv_r2":              bm.get("cv_r2",      0.0),
            "n_train":            saved.get("train_size", 0),
            "n_test":             saved.get("test_size",  0),
            "feature_importance": saved.get("feature_importance", []),
            "model_comparison": {
                name: round(v["r2_test"] * 100, 2)
                for name, v in saved["metrics"].items()
            },
        }
    else:
        # Compute metrics on the fly if JSON absent
        X  = _df[FEATURES]
        y  = _df[TARGET].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        y_pred_te = _pipeline.predict(X_te)
        _metrics_dict = {
            "algorithm":          "Gradient Boosting",
            "r2_test":            float(r2_score(y_te, y_pred_te)),
            "mae_test":           float(mean_absolute_error(y_te, y_pred_te)),
            "rmse_test":          float(np.sqrt(mean_squared_error(y_te, y_pred_te))),
            "n_train":            int(len(X_tr)),
            "n_test":             int(len(X_te)),
            "feature_importance": [],
            "model_comparison":   {},
        }


def _get_dataset_stats() -> dict:
    """Return summary statistics over the loaded DataFrame."""
    df = _df
    return {
        "total_records":  int(len(df)),
        "avg_charges":    round(float(df["charges"].mean()),    2),
        "max_charges":    round(float(df["charges"].max()),     2),
        "min_charges":    round(float(df["charges"].min()),     2),
        "median_charges": round(float(df["charges"].median()),  2),
        "std_charges":    round(float(df["charges"].std()),     2),
        "avg_age":        round(float(df["age"].mean()),        1),
        "avg_bmi":        round(float(df["bmi"].mean()),        1),
        "smoker_pct":     round(float((df["smoker"] == "yes").mean()) * 100, 1),
        "non_smoker_pct": round(float((df["smoker"] == "no").mean())  * 100, 1),
        "male_pct":       round(float((df["sex"] == "male").mean())   * 100, 1),
        "female_pct":     round(float((df["sex"] == "female").mean()) * 100, 1),
        "avg_children":   round(float(df["children"].mean()),   2),
        "regions":        df["region"].value_counts().to_dict(),
    }


def _get_chart_data() -> dict:
    """Return pre-aggregated, chart-ready data for the frontend."""
    df     = _df
    sample = df.sample(min(300, len(df)), random_state=42)

    # Scatter: age vs charges (coloured by smoking status)
    age_charges = [
        {"age": int(r["age"]), "charges": round(float(r["charges"]), 2),
         "smoker": r["smoker"]}
        for _, r in sample.iterrows()
    ]

    # Bar: average charges per region
    region_avg  = df.groupby("region")["charges"].mean().round(2).to_dict()
    region_data = [{"region": k.capitalize(), "avg_charges": v}
                   for k, v in region_avg.items()]

    # Doughnut: smoker vs non-smoker average charges
    smoker_avg  = df.groupby("smoker")["charges"].mean().round(2).to_dict()

    # Bar: average charges by number of children
    children_avg  = df.groupby("children")["charges"].mean().round(2).to_dict()
    children_data = [{"children": int(k), "avg_charges": v}
                     for k, v in sorted(children_avg.items())]

    # Histogram: distribution of charges in $5,000 bins
    bins = np.arange(0, df["charges"].max() + 5_000, 5_000)
    hist, edges = np.histogram(df["charges"], bins=bins)
    histogram = [
        {"range": f"${int(edges[i]/1000)}k-${int(edges[i+1]/1000)}k",
         "count": int(hist[i])}
        for i in range(len(hist)) if hist[i] > 0
    ]

    return {
        "age_charges":       age_charges,
        "region_charges":    region_data,
        "smoker_charges":    smoker_avg,
        "children_charges":  children_data,
        "charges_histogram": histogram,
    }


# -- Flask application factory -------------------------------------------------
def create_app() -> Flask:
    """
    Create and configure the Flask application.
    The app serves the frontend SPA at '/' and exposes 5 REST API endpoints.
    """
    app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
    CORS(app)   # allow cross-origin requests (development convenience)

    # Load the model once when the app starts
    _load_or_train_model()

    # -- Route: frontend SPA ----------------------------------------------------
    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    # -- Route: health check ----------------------------------------------------
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status":  "ok",
            "message": "Healthcare Cost Prediction API is running",
        })

    # -- Route: dataset statistics ----------------------------------------------
    @app.route("/api/stats", methods=["GET"])
    def get_stats():
        try:
            return jsonify({"success": True, "data": _get_dataset_stats()})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    # -- Route: model evaluation metrics ---------------------------------------
    @app.route("/api/model-metrics", methods=["GET"])
    def get_model_metrics():
        try:
            return jsonify({"success": True, "data": _metrics_dict})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    # -- Route: chart data ------------------------------------------------------
    @app.route("/api/chart-data", methods=["GET"])
    def get_chart_data():
        try:
            return jsonify({"success": True, "data": _get_chart_data()})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    # -- Route: predict healthcare cost ----------------------------------------
    @app.route("/api/predict", methods=["POST"])
    def predict():
        """
        Accepts a JSON body with keys:
            age (number), sex (str), bmi (number),
            children (int), smoker (str), region (str)

        Returns on success (HTTP 200):
            { success: true, data: { predicted_cost, risk_category,
                                     risk_color, input_summary } }

        Returns on validation failure (HTTP 422):
            { success: false, error: "...", validation_errors: [...] }

        Returns on server error (HTTP 500):
            { success: false, error: "..." }
        """
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({
                "success": False,
                "error": "Request body must be valid JSON with "
                         "Content-Type: application/json",
            }), 400

        # -- Validate inputs ----------------------------------------------------
        errors = validate_input(payload)
        if errors:
            return jsonify({
                "success":           False,
                "error":             "; ".join(errors),
                "validation_errors": errors,
            }), 422

        # -- Generate prediction ------------------------------------------------
        try:
            result = predict_cost(_pipeline, payload)
            return jsonify({"success": True, "data": result})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    return app


# ============================================================
# SECTION 10 — APPLICATION ENTRY POINT
# ============================================================

def run_demo():
    """
    When executed directly (python healthcare_cost_prediction.py), run
    the full ML pipeline as a demonstration:
      1. Load dataset
      2. EDA
      3. Train & compare 5 algorithms
      4. Evaluate the best model
      5. Analyse feature importances
      6. Demo prediction examples
      7. Start the Flask web server
    """
    # -- Steps 1–3 -------------------------------------------------------------
    df = load_dataset()
    run_eda(df)
    pipeline, best_name, results, X_train, X_test, y_train, y_test = \
        train_and_compare(df)

    # -- Step 4 ----------------------------------------------------------------
    evaluate_model(pipeline, best_name, results, X_train, X_test, y_train, y_test)

    # -- Step 5 ----------------------------------------------------------------
    analyse_feature_importance(pipeline)

    # -- Step 6: Demo predictions -----------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 8 — Sample Predictions")
    print("=" * 60)

    demo_cases = [
        # (description,  input dict)
        ("Young non-smoker, low BMI",
         {"age": 25, "sex": "female", "bmi": 22.0, "children": 0,
          "smoker": "no",  "region": "northwest"}),
        ("Middle-aged smoker, high BMI",
         {"age": 45, "sex": "male",   "bmi": 34.5, "children": 2,
          "smoker": "yes", "region": "southeast"}),
        ("Older non-smoker, average BMI",
         {"age": 60, "sex": "male",   "bmi": 27.8, "children": 1,
          "smoker": "no",  "region": "northeast"}),
        ("Young smoker, very high BMI",
         {"age": 30, "sex": "female", "bmi": 40.0, "children": 0,
          "smoker": "yes", "region": "southwest"}),
    ]

    print(f"\n  {'Case':<40} {'Predicted Cost':>16} {'Risk':>12}")
    print("  " + "-" * 70)
    for desc, inp in demo_cases:
        result = predict_cost(pipeline, inp)
        print(f"  {desc:<40} ${result['predicted_cost']:>14,.2f}   "
              f"{result['risk_category']:>10}")

    # -- Step 7: Start Flask server --------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 9 — Starting Flask API")
    print("=" * 60)
    print("\n  The web application will be available at:")
    print("    http://127.0.0.1:5000/")
    print("\n  API endpoints:")
    print("    GET  /api/health")
    print("    GET  /api/stats")
    print("    GET  /api/model-metrics")
    print("    GET  /api/chart-data")
    print("    POST /api/predict   (JSON body: age, sex, bmi, children, smoker, region)")
    print("\n  Press CTRL+C to stop the server.")
    print("=" * 60 + "\n")

    app = create_app()
    app.run(debug=False, port=5000)


if __name__ == "__main__":
    run_demo()
