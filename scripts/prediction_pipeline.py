"""
scripts/prediction_pipeline.py
--------------------------------
Step 2 of the wildfire detection pipeline.

Architecture
------------
Three models run in sequence on the latest clean NASA FIRMS data:

  1. DBSCAN        — spatial hotspot clustering (unsupervised)
  2. XGBoost       — risk classification + FRP regression
  3. Random Forest — risk classification + FRP regression

XGBoost and Random Forest predictions are combined into a final
ensemble output using a conservative rule: the higher risk level
wins, and FRP is averaged between both models. This produces a
more robust and reliable prediction than either model alone.

Outputs
-------
  data/latest_predictions.csv
  data/latest_predictions_world_1day.csv  (backup copy)
"""

import os
import sys
import glob
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import (
    DATA_DIR,
    DBSCAN_MODEL_PATH,
    DBSCAN_PARAMS_PATH,
    XGBOOST_CLASSIFIER_PATH,
    XGBOOST_REGRESSOR_PATH,
    RF_CLASSIFIER_PATH,
    RF_REGRESSOR_PATH,
    LATEST_CLEAN_FIRMS_PATH,
    LATEST_PREDICTIONS_PATH,
    LATEST_PREDICTIONS_WORLD_PATH,
    PREDICTION_FEATURES,
    RISK_LABELS,
)

# Risk level order for comparison
_RISK_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Extreme": 3}


# ============================================================
# DATA LOADING
# ============================================================

def _load_latest_clean_firms():
    """Return the latest clean FIRMS CSV path."""
    if os.path.exists(LATEST_CLEAN_FIRMS_PATH):
        return LATEST_CLEAN_FIRMS_PATH
    # Fallback: most recent timestamped file
    candidates = glob.glob(os.path.join(DATA_DIR, "firms_clean_world_1day_*.csv"))
    if candidates:
        return max(candidates, key=os.path.getctime)
    raise FileNotFoundError(
        "No clean FIRMS data found. Run pipeline_nasa_firms.py first."
    )


def _prepare_features(df):
    missing = [c for c in PREDICTION_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in clean data: {missing}")
    X = df[PREDICTION_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


# ============================================================
# MODEL LOADERS
# ============================================================

def _load_xgboost():
    from xgboost import XGBClassifier, XGBRegressor

    if not os.path.exists(XGBOOST_CLASSIFIER_PATH):
        raise FileNotFoundError(f"XGBoost classifier not found: {XGBOOST_CLASSIFIER_PATH}")
    if not os.path.exists(XGBOOST_REGRESSOR_PATH):
        raise FileNotFoundError(f"XGBoost regressor not found: {XGBOOST_REGRESSOR_PATH}")

    clf = XGBClassifier()
    clf.load_model(XGBOOST_CLASSIFIER_PATH)

    reg = XGBRegressor()
    reg.load_model(XGBOOST_REGRESSOR_PATH)

    print("  XGBoost classifier : loaded")
    print("  XGBoost regressor  : loaded")
    return clf, reg


def _load_random_forest():
    clf, reg = None, None

    if os.path.exists(RF_CLASSIFIER_PATH):
        clf = joblib.load(RF_CLASSIFIER_PATH)
        print("  RF classifier      : loaded")
    else:
        warnings.warn(
            f"Random Forest classifier not found at {RF_CLASSIFIER_PATH}. "
            "XGBoost-only predictions will be used.",
            UserWarning, stacklevel=2,
        )

    if os.path.exists(RF_REGRESSOR_PATH):
        reg = joblib.load(RF_REGRESSOR_PATH)
        print("  RF regressor       : loaded")
    else:
        warnings.warn(
            f"Random Forest regressor not found at {RF_REGRESSOR_PATH}. "
            "XGBoost-only FRP will be used.",
            UserWarning, stacklevel=2,
        )

    return clf, reg


def _load_dbscan_params():
    defaults = {"eps": 0.0005, "min_samples": 5}

    if os.path.exists(DBSCAN_PARAMS_PATH):
        with open(DBSCAN_PARAMS_PATH) as f:
            params = json.load(f)
        eps         = params.get("eps",         defaults["eps"])
        min_samples = params.get("min_samples", defaults["min_samples"])
        print(f"  DBSCAN params      : eps={eps}, min_samples={min_samples} (from JSON)")
        return eps, min_samples

    if os.path.exists(DBSCAN_MODEL_PATH):
        obj = joblib.load(DBSCAN_MODEL_PATH)
        if isinstance(obj, dict):
            p = obj.get("params", {})
            eps         = p.get("eps",         defaults["eps"])
            min_samples = p.get("min_samples", defaults["min_samples"])
        else:
            eps         = getattr(obj, "eps",         defaults["eps"])
            min_samples = getattr(obj, "min_samples", defaults["min_samples"])
        print(f"  DBSCAN params      : eps={eps}, min_samples={min_samples} (from model)")
        return eps, min_samples

    print(f"  DBSCAN params      : using defaults eps={defaults['eps']}, min_samples={defaults['min_samples']}")
    return defaults["eps"], defaults["min_samples"]


# ============================================================
# PREDICTION STEPS
# ============================================================

def run_xgboost(df, X):
    print("\n[XGBoost] Running predictions...")
    clf, reg = _load_xgboost()

    df["xgb_predicted_frp"]     = np.expm1(reg.predict(X)).clip(min=0)
    risk_ids                     = clf.predict(X)
    df["xgb_predicted_risk"]    = pd.Series(risk_ids).map(RISK_LABELS).values
    df["xgb_risk_confidence"]   = clf.predict_proba(X).max(axis=1).round(4)

    print(f"  XGBoost risk distribution:")
    print(df["xgb_predicted_risk"].value_counts().to_string())
    return df


def run_random_forest(df, X):
    print("\n[Random Forest] Running predictions...")
    clf, reg = _load_random_forest()

    if reg is not None:
        df["rf_predicted_frp"] = np.expm1(reg.predict(X)).clip(min=0)
    else:
        df["rf_predicted_frp"] = np.nan

    if clf is not None:
        risk_ids                  = clf.predict(X)
        df["rf_predicted_risk"]   = pd.Series(risk_ids).map(RISK_LABELS).values
        df["rf_risk_confidence"]  = clf.predict_proba(X).max(axis=1).round(4)
        print(f"  RF risk distribution:")
        print(df["rf_predicted_risk"].value_counts().to_string())
    else:
        df["rf_predicted_risk"]  = np.nan
        df["rf_risk_confidence"] = np.nan

    return df


def compute_ensemble(df):
    """
    Ensemble strategy:
    - predicted_risk : conservative rule — higher risk level wins between XGBoost and RF
    - predicted_frp  : average of both models (or XGBoost alone if RF unavailable)
    - risk_confidence: average of both model confidences
    """
    def _higher_risk(xgb_val, rf_val):
        if pd.isna(rf_val) or rf_val not in _RISK_ORDER:
            return xgb_val
        if pd.isna(xgb_val) or xgb_val not in _RISK_ORDER:
            return rf_val
        return xgb_val if _RISK_ORDER[xgb_val] >= _RISK_ORDER[rf_val] else rf_val

    df["predicted_risk"] = [
        _higher_risk(x, r)
        for x, r in zip(df["xgb_predicted_risk"], df["rf_predicted_risk"])
    ]

    rf_available = df["rf_predicted_frp"].notna().all()
    if rf_available:
        df["predicted_frp"] = (df["xgb_predicted_frp"] + df["rf_predicted_frp"]) / 2
    else:
        df["predicted_frp"] = df["xgb_predicted_frp"]

    rf_conf_available = df["rf_risk_confidence"].notna().all()
    if rf_conf_available:
        df["risk_confidence"] = (df["xgb_risk_confidence"] + df["rf_risk_confidence"]) / 2
    else:
        df["risk_confidence"] = df["xgb_risk_confidence"]

    return df


def run_dbscan(df):
    print("\n[DBSCAN] Running spatial hotspot clustering...")
    eps, min_samples = _load_dbscan_params()

    coords     = df[["latitude", "longitude"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    coords_rad = np.radians(coords.values)

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1,
    )
    df["cluster_id"] = dbscan.fit_predict(coords_rad)

    n_clusters  = df[df["cluster_id"] != -1]["cluster_id"].nunique()
    n_noise     = (df["cluster_id"] == -1).sum()
    noise_ratio = n_noise / len(df) * 100

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise:,}  ({noise_ratio:.1f}%)")
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("WORLD 1-DAY PREDICTION PIPELINE")
    print("Models: XGBoost + Random Forest (ensemble) + DBSCAN")
    print("=" * 70)

    data_path = _load_latest_clean_firms()
    print(f"\nInput  : {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Shape  : {df.shape}")

    X = _prepare_features(df)
    print(f"Features: {X.shape[1]} columns, {X.shape[0]:,} rows")

    df = run_xgboost(df, X)
    df = run_random_forest(df, X)
    df = compute_ensemble(df)
    df = run_dbscan(df)

    df["prediction_time"]   = pd.Timestamp.now()
    df["prediction_scope"]  = "World"
    df["prediction_period"] = "1 day"
    df["source_clean_file"] = os.path.basename(data_path)

    df.to_csv(LATEST_PREDICTIONS_PATH,       index=False)
    df.to_csv(LATEST_PREDICTIONS_WORLD_PATH, index=False)

    print("\n" + "=" * 70)
    print("Prediction pipeline completed.")
    print(f"Output : {LATEST_PREDICTIONS_PATH}")
    print("=" * 70)

    print("\nEnsemble risk distribution:")
    print(df["predicted_risk"].value_counts().to_string())

    preview_cols = [c for c in [
        "latitude", "longitude", "frp",
        "xgb_predicted_risk", "rf_predicted_risk", "predicted_risk",
        "predicted_frp", "risk_confidence", "cluster_id",
    ] if c in df.columns]
    print(f"\nSample output ({len(df):,} total rows):")
    print(df[preview_cols].head(5).to_string())


if __name__ == "__main__":
    main()