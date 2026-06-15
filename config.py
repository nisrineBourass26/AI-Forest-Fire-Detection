"""
config.py
---------
Central configuration for the AI-Based Forest Fire Detection system.
All scripts import paths and constants from here.
"""

import os

# ============================================================
# ROOT
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODEL_DIR  = os.path.join(ROOT_DIR, "models", "saved_models")
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
WEB_DIR    = os.path.join(ROOT_DIR, "web")

# ============================================================
# SAVED MODEL PATHS
# ============================================================

# DBSCAN
DBSCAN_MODEL_PATH  = os.path.join(MODEL_DIR, "dbscan_model.joblib")
DBSCAN_PARAMS_PATH = os.path.join(MODEL_DIR, "dbscan_params.json")

# XGBoost
XGBOOST_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "xgboost_classifier.json")
XGBOOST_REGRESSOR_PATH  = os.path.join(MODEL_DIR, "xgboost_regressor.json")

# Random Forest
RF_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "rf_classifier.joblib")
RF_REGRESSOR_PATH  = os.path.join(MODEL_DIR, "rf_regressor.joblib")

# Forecasting — Prophet + LSTM ensemble
PROPHET_MODEL_PATH       = os.path.join(MODEL_DIR, "ForecastingGlobal_prophet.joblib")
LSTM_MODEL_PATH          = os.path.join(MODEL_DIR, "ForecastingGlobal_lstm.keras")
FORECASTING_SCALER_PATH  = os.path.join(MODEL_DIR, "ForecastingGlobal_scaler.joblib")
FORECASTING_LSTM_META_PATH = os.path.join(MODEL_DIR, "ForecastingGlobal_metadata.joblib")
FORECASTING_REPORT_PATH  = os.path.join(MODEL_DIR, "rfg_forecasting_meta.json")

# ============================================================
# DATA FILE PATHS
# ============================================================

LATEST_RAW_FIRMS_PATH         = os.path.join(DATA_DIR, "latest_raw_firms.csv")
LATEST_CLEAN_FIRMS_PATH       = os.path.join(DATA_DIR, "latest_clean_firms.csv")
LATEST_PREDICTIONS_PATH       = os.path.join(DATA_DIR, "latest_predictions.csv")
LATEST_PREDICTIONS_WORLD_PATH = os.path.join(DATA_DIR, "latest_predictions_world_1day.csv")
LATEST_PREDICTIONS_COUNTRIES  = os.path.join(DATA_DIR, "latest_predictions_with_countries.csv")
LATEST_FORECAST_PATH          = os.path.join(DATA_DIR, "latest_forecast.csv")
FORECAST_CONTEXT_PATH         = os.path.join(DATA_DIR, "forecast_context.csv")
PROPHET_DATA_GLOBAL_PATH      = os.path.join(DATA_DIR, "prophetData_global.csv")

# ============================================================
# FEATURE SETS
# ============================================================

# Features used by XGBoost and Random Forest
PREDICTION_FEATURES = [
    "latitude",
    "longitude",
    "brightness",
    "bright_t31",
    "brightness_anomaly",
    "bright_t31_anomaly",
    "confidence_num",
    "is_day",
    "satellite_code",
    "scan",
    "track",
    "day_of_year",
    "month",
    "week_of_year",
    "year",
    "season",
]

# Risk level labels (classifier output → human-readable)
RISK_LABELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Extreme",
}

# ============================================================
# PIPELINE SETTINGS
# ============================================================

FORECAST_HORIZON_DAYS = 7
CONTEXT_DAYS_TO_KEEP  = 90

# PostgreSQL connection (reads from environment variables first,
# falls back to local development defaults)
DB_USER     = os.environ.get("DB_USER",     "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres123")
DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_PORT     = os.environ.get("DB_PORT",     "5433")
DB_NAME     = os.environ.get("DB_NAME",     "forest_fire_pfe")

PREDICTIONS_TABLE = "fire_predictions"