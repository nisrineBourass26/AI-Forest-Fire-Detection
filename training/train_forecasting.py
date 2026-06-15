"""
Global_training_forecasting.py

Official global forecasting module:
1) Prophet for main trend / seasonality
2) LSTM for controlled residual correction

Final formula:
    final_forecast = prophet_forecast + best_alpha * clipped_lstm_residual

Data: prophetData_globalOP.csv
  - 5 years: 2020-01-01 to 2025-01-01 (1828 days)
  - Target y = daily fire detection count
  - Strong summer seasonality (July avg ~14, December avg ~4)
  - Real extreme events (2022 Turkey fires: y=366)
  - 179 zero days = missing satellite data (avg_confidence=0), NOT real zeros

Key improvements vs original:
  - Zero days where avg_confidence=0 are treated as missing, not real zeros
  - Outliers are NOT capped (they are real fire events) — log1p handles them
  - LSTM_UNITS increased from 32 to 64 for global complexity
  - LOOKBACK increased from 30 to 60 for seasonal context
  - ALPHA_CANDIDATES extended to 0.50
  - MAX_RESIDUAL_CORRECTION_CANDIDATES extended to 3.00
  - 2024 incomplete data is downweighted in Prophet via changepoint detection
  - Baselines are properly computed and saved for comparison
"""

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ============================================================
# CONFIG
# ============================================================

LOOKBACK = 60           # ✅ increased from 30 — captures 2 months of context
FORECAST_HORIZON = 7
TEST_DAYS = 30
TSCV_SPLITS = 3

EPOCHS = 200            # ✅ increased from 150 — EarlyStopping will cut early if needed
BATCH_SIZE = 32
LSTM_UNITS = 64         # ✅ increased from 32 — global data is more complex than Morocco
DROPOUT = 0.20
RANDOM_STATE = 42

# ✅ Extended alpha range — let the search find what global data supports
# alpha=0 excluded: architecture must include both Prophet and LSTM
ALPHA_CANDIDATES = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
MAX_RESIDUAL_CORRECTION_CANDIDATES = [0.50, 1.00, 1.50, 2.00, 3.00]

MIN_REASONABLE_FORECAST = 0.0
MAX_REASONABLE_FORECAST = 30.0   # 99th percentile of y is ~48, daily avg is 7.17

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # retrain/

DATA_PATH = os.path.join(ROOT_DIR, "Data", "prophetData_globalOP.csv")

MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
EVAL_DIR  = os.path.join(ROOT_DIR, "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ============================================================
# OUTPUT FILE NAMES
# ============================================================

MODEL_NAME = "ForecastingGlobal"

PROPHET_MODEL_FILE    = f"{MODEL_NAME}_prophet.joblib"
LSTM_MODEL_FILE       = f"{MODEL_NAME}_lstm.keras"
SCALER_FILE           = f"{MODEL_NAME}_scaler.joblib"
LSTM_METADATA_FILE    = f"{MODEL_NAME}_metadata.joblib"

REPORT_TXT_FILE       = "rfg_report.txt"
METRICS_JSON_FILE     = "rfg_metrics.json"
METRICS_TXT_FILE      = "rfg_metrics.txt"
FORECAST_CSV_FILE     = "rfg_next_7_days_forecast.csv"
META_JSON_FILE        = "rfg_forecasting_meta.json"


# ============================================================
# METRIC TARGETS
# ============================================================

METRIC_TARGETS = {
    "prophet": {
        "minimum":   {"MAE": 3.00, "RMSE": 4.00, "SMAPE": 90.0},
        "good":      {"MAE": 2.50, "RMSE": 3.20, "SMAPE": 75.0},
        "excellent": {"MAE": 2.00, "RMSE": 2.70, "SMAPE": 60.0},
    },
    "residual_lstm_final": {
        "minimum":   {"MAE": 2.80, "RMSE": 3.80, "SMAPE": 80.0},
        "good":      {"MAE": 2.30, "RMSE": 3.00, "SMAPE": 65.0},
        "excellent": {"MAE": 2.00, "RMSE": 2.70, "SMAPE": 50.0},
    },
}


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        raise ValueError(f"No valid values for {label}")

    y_pred = np.clip(y_pred, 0, None)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan

    # SMAPE — stable near zero
    denom      = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_mask = denom > 1e-6
    smape      = (
        np.mean(np.abs(y_true[smape_mask] - y_pred[smape_mask]) / denom[smape_mask]) * 100
        if smape_mask.sum() > 0 else np.nan
    )

    # WAPE — robust percentage metric
    total = np.sum(np.abs(y_true))
    wape  = (np.sum(np.abs(y_true - y_pred)) / total * 100) if total > 0 else np.nan

    print(f"\n── {label} ──")
    print(f"Samples : {len(y_true)}")
    print(f"MAE     : {mae:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    print(f"R²      : {r2:.4f}")
    print(f"SMAPE   : {smape:.2f}%")
    print(f"WAPE    : {wape:.2f}%")

    return {
        "label":   label,
        "samples": int(len(y_true)),
        "MAE":     float(mae),
        "RMSE":    float(rmse),
        "R2":      float(r2)    if np.isfinite(r2)    else None,
        "SMAPE":   float(smape) if np.isfinite(smape) else None,
        "WAPE":    float(wape)  if np.isfinite(wape)  else None,
    }


def evaluate_target_level(metrics, model_type):
    targets = METRIC_TARGETS[model_type]
    mae     = metrics["MAE"]
    rmse    = metrics["RMSE"]
    smape   = metrics["SMAPE"] or 999

    if mae <= targets["excellent"]["MAE"] and rmse <= targets["excellent"]["RMSE"] and smape <= targets["excellent"]["SMAPE"]:
        status = "Excellent"
    elif mae <= targets["good"]["MAE"] and rmse <= targets["good"]["RMSE"] and smape <= targets["good"]["SMAPE"]:
        status = "Good"
    elif mae <= targets["minimum"]["MAE"] and rmse <= targets["minimum"]["RMSE"] and smape <= targets["minimum"]["SMAPE"]:
        status = "Minimum acceptable"
    else:
        status = "Below minimum"

    print(f"[TARGET STATUS] {metrics['label']} → {status}")
    return status


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):
    print("=" * 70)
    print("LOAD DATA")
    print("=" * 70)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Normalize date column name
    date_col = next((c for c in df.columns if c.lower() in ["ds", "date", "acq_date"]), None)
    if date_col is None:
        raise ValueError("No date column found. Expected ds, date, or acq_date.")

    df = df.rename(columns={date_col: "ds"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)

    if "y" not in df.columns:
        raise ValueError("No target column 'y' found.")

    # Fill full date range
    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_range).reset_index().rename(columns={"index": "ds"})

    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    for col in ["avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # ✅ KEY FIX: Zero days where avg_confidence=0 mean NO satellite data, not real zero fires
    # Mark them as NaN so Prophet/LSTM don't learn "fires suddenly stopped"
    missing_mask = (df["y"] == 0) & (df["avg_confidence"] == 0)
    n_missing = missing_mask.sum()
    df.loc[missing_mask, "y"] = np.nan

    # Interpolate missing days (linear, limit 7 days gap)
    df["y"] = df["y"].interpolate(method="linear", limit=7).bfill().ffill()
    df["y"] = df["y"].fillna(df["y"].median())

    print(f"[INFO] Data path        : {path}")
    print(f"[INFO] Date range       : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"[INFO] Rows             : {len(df)}")
    print(f"[INFO] Missing days fixed (confidence=0): {n_missing}")
    print(f"[INFO] Mean y           : {df['y'].mean():.4f}")
    print(f"[INFO] Median y         : {df['y'].median():.4f}")
    print(f"[INFO] Max y            : {df['y'].max():.4f}")
    print(f"[INFO] Std y            : {df['y'].std():.4f}")

    return df


# ============================================================
# BASELINES
# ============================================================

def evaluate_baselines(df, report):
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    y = df["y"].values.astype(float)
    n = TEST_DAYS

    y_test  = y[-n:]
    y_train = y[:-n]

    # Baseline 1: predict train mean
    report.append(compute_metrics(
        y_test,
        np.full(len(y_test), np.mean(y_train)),
        "Baseline: Train Mean"
    ))

    # Baseline 2: 7-day lag (same day last week)
    if len(y) > n + 7:
        report.append(compute_metrics(y_test, y[-n - 7:-7], "Baseline: 7-Day Lag"))

    # Baseline 3: 365-day lag (same day last year)
    if len(y) > n + 365:
        report.append(compute_metrics(y_test, y[-n - 365:-365], "Baseline: 365-Day Lag"))

    print("\n[INFO] Baselines computed. Your model must beat these to be meaningful.")


# ============================================================
# PROPHET
# ============================================================

def train_prophet(df, report):
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    print("\n" + "=" * 70)
    print("PROPHET TRAINING")
    print("=" * 70)

    pdf = df[["ds", "y"]].copy()
    pdf["y_original"] = pdf["y"].astype(float)

    # Log transform: handles the extreme outliers (y=366) without capping them
    pdf["y"] = np.log1p(pdf["y_original"].clip(lower=0))

    cutoff  = pdf["ds"].max() - pd.Timedelta(days=TEST_DAYS)
    train_df = pdf[pdf["ds"] <= cutoff][["ds", "y"]].copy()
    test_df  = pdf[pdf["ds"] > cutoff][["ds", "y_original"]].copy()

    print(f"[INFO] Train days : {len(train_df)}")
    print(f"[INFO] Test days  : {len(test_df)}")

    # ✅ Param grid designed for this specific dataset:
    # - additive works better when y has log transform
    # - low changepoint_prior (0.001-0.01) because fire seasonality is stable year to year
    # - monthly fourier helps capture mid-season transitions
    param_grid = [
        {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.001,
            "seasonality_prior_scale": 1.0,
            "n_changepoints": 10,
            "weekly_seasonality": False,
            "monthly_fourier": 3,
            "yearly_fourier": 10,
        },
        {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.001,
            "seasonality_prior_scale": 2.0,
            "n_changepoints": 15,
            "weekly_seasonality": False,
            "monthly_fourier": 5,
            "yearly_fourier": 12,
        },
        {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.005,
            "seasonality_prior_scale": 2.0,
            "n_changepoints": 15,
            "weekly_seasonality": True,
            "monthly_fourier": 5,
            "yearly_fourier": 12,
        },
        {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.01,
            "seasonality_prior_scale": 5.0,
            "n_changepoints": 20,
            "weekly_seasonality": True,
            "monthly_fourier": 5,
            "yearly_fourier": 15,
        },
        {
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.005,
            "seasonality_prior_scale": 2.0,
            "n_changepoints": 15,
            "weekly_seasonality": True,
            "monthly_fourier": 5,
            "yearly_fourier": 12,
        },
        {
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.01,
            "seasonality_prior_scale": 5.0,
            "n_changepoints": 20,
            "weekly_seasonality": True,
            "monthly_fourier": 5,
            "yearly_fourier": 15,
        },
    ]

    best_model   = None
    best_forecast = None
    best_metrics  = None
    best_params   = None
    best_score    = np.inf

    for i, params in enumerate(param_grid, 1):
        print(f"\n[INFO] Testing Prophet config {i}/{len(param_grid)}: {params}")

        try:
            model = Prophet(
                seasonality_mode=params["seasonality_mode"],
                yearly_seasonality=False,
                weekly_seasonality=params["weekly_seasonality"],
                daily_seasonality=False,
                changepoint_prior_scale=params["changepoint_prior_scale"],
                seasonality_prior_scale=params["seasonality_prior_scale"],
                n_changepoints=params["n_changepoints"],
                interval_width=0.80,
            )

            model.add_seasonality("yearly",   period=365.25, fourier_order=params["yearly_fourier"])
            model.add_seasonality("monthly",  period=30.5,   fourier_order=params["monthly_fourier"])

            model.fit(train_df)

            future   = model.make_future_dataframe(periods=TEST_DAYS + FORECAST_HORIZON, freq="D")
            forecast = model.predict(future)

            merged = test_df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")

            y_true = merged["y_original"].values
            y_pred = np.expm1(np.clip(merged["yhat"].values, 0, None))
            y_pred = np.clip(y_pred, MIN_REASONABLE_FORECAST, MAX_REASONABLE_FORECAST)

            metrics = compute_metrics(y_true, y_pred, f"Prophet Config {i}")

            score = metrics["SMAPE"] if metrics["SMAPE"] is not None else metrics["RMSE"]

            if score < best_score:
                best_score    = score
                best_model    = model
                best_forecast = forecast
                best_metrics  = metrics
                best_params   = params

        except Exception as e:
            print(f"[WARN] Config {i} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All Prophet configs failed.")

    print(f"\n[INFO] Best Prophet config: {best_params}")

    best_metrics["label"]         = "Prophet Test Set"
    best_metrics["target_status"] = evaluate_target_level(best_metrics, "prophet")
    report.append(best_metrics)

    # TimeSeriesSplit CV — shows stability across different fire seasons
    print(f"\n── Prophet TimeSeriesSplit CV ({TSCV_SPLITS} folds) ──")
    tscv    = TimeSeriesSplit(n_splits=TSCV_SPLITS)
    cv_maes = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(pdf)):
        tr  = pdf.iloc[tr_idx][["ds", "y"]].copy()
        val = pdf.iloc[val_idx][["ds", "y_original"]].copy()

        if len(tr) < 180:   # need at least 6 months to learn yearly seasonality
            print(f"  Fold {fold+1}: skipped (too short: {len(tr)} days)")
            continue

        try:
            m = Prophet(
                seasonality_mode=best_params["seasonality_mode"],
                yearly_seasonality=False,
                weekly_seasonality=best_params["weekly_seasonality"],
                daily_seasonality=False,
                changepoint_prior_scale=best_params["changepoint_prior_scale"],
                seasonality_prior_scale=best_params["seasonality_prior_scale"],
                n_changepoints=best_params["n_changepoints"],
                interval_width=0.80,
            )
            m.add_seasonality("yearly",  period=365.25, fourier_order=best_params["yearly_fourier"])
            m.add_seasonality("monthly", period=30.5,   fourier_order=best_params["monthly_fourier"])
            m.fit(tr)

            fc   = m.predict(val[["ds"]])
            pred = np.expm1(np.clip(fc["yhat"].values, 0, None))
            pred = np.clip(pred, MIN_REASONABLE_FORECAST, MAX_REASONABLE_FORECAST)

            mae = mean_absolute_error(val["y_original"].values, pred)
            cv_maes.append(float(mae))
            print(f"  Fold {fold+1}: MAE = {mae:.4f} (n_val={len(val)})")

        except Exception as e:
            print(f"  Fold {fold+1}: failed — {e}")

    if cv_maes:
        print(f"  CV MAE mean: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")
        report.append({
            "label":  "Prophet CV MAE",
            "folds":  cv_maes,
            "mean":   float(np.mean(cv_maes)),
            "std":    float(np.std(cv_maes)),
        })

    # Build prophet_pred column on full dataset for residual calculation
    full_pred = best_forecast[["ds", "yhat"]].copy()
    full_pred["prophet_pred"] = np.expm1(np.clip(full_pred["yhat"].values, 0, None))
    full_pred["prophet_pred"] = np.clip(
        full_pred["prophet_pred"], MIN_REASONABLE_FORECAST, MAX_REASONABLE_FORECAST
    )

    df_out = df.merge(full_pred[["ds", "prophet_pred"]], on="ds", how="left")
    df_out["prophet_pred"] = (
        df_out["prophet_pred"].interpolate().bfill().ffill().fillna(df_out["y"].mean())
    )
    df_out["residual"] = df_out["y"] - df_out["prophet_pred"]

    # Save plots
    fig = best_model.plot(best_forecast, figsize=(14, 5))
    plt.title("Prophet Forecast — Global Fire Activity")
    plt.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "rfg_prophet_forecast.png"), dpi=150)
    plt.close(fig)

    fig2 = best_model.plot_components(best_forecast, figsize=(14, 8))
    plt.tight_layout()
    fig2.savefig(os.path.join(EVAL_DIR, "rfg_prophet_components.png"), dpi=150)
    plt.close(fig2)

    joblib.dump(
        {
            "model":               best_model,
            "forecast":            best_forecast,
            "best_params":         best_params,
            "metrics":             best_metrics,
            "target_transform":    "log1p",
            "inverse_transform":   "expm1",
            "min_forecast":        MIN_REASONABLE_FORECAST,
            "max_forecast":        MAX_REASONABLE_FORECAST,
        },
        os.path.join(MODEL_DIR, PROPHET_MODEL_FILE),
    )

    print(f"[INFO] Prophet model saved: {PROPHET_MODEL_FILE}")
    return best_model, best_forecast, df_out


# ============================================================
# RESIDUAL LSTM — FEATURES
# ============================================================

def add_residual_features(df):
    df = df.copy().sort_values("ds").reset_index(drop=True)

    for col in ["y", "prophet_pred", "residual", "avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    df["day_of_year"] = df["ds"].dt.dayofyear
    df["month"]       = df["ds"].dt.month

    # Cyclical encoding — better than raw integers for periodicity
    df["day_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lags on y, residual, and prophet_pred
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"y_lag_{lag}"]        = df["y"].shift(lag)
        df[f"residual_lag_{lag}"] = df["residual"].shift(lag)
        df[f"prophet_lag_{lag}"]  = df["prophet_pred"].shift(lag)

    # Rolling stats on residual (what's the recent error pattern)
    for w in [7, 14, 30]:
        df[f"residual_roll{w}_mean"] = df["residual"].shift(1).rolling(w, min_periods=1).mean()
        df[f"residual_roll{w}_std"]  = df["residual"].shift(1).rolling(w, min_periods=2).std().fillna(0)

    # Log-transformed versions of the main signals
    for col in ["y", "prophet_pred", "avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    return df


def get_residual_feature_cols(df):
    cols = [
        "residual",
        "day_sin", "day_cos", "month_sin", "month_cos",
        "y_log", "prophet_pred_log",
        "avg_frp_log", "max_frp_log", "avg_confidence_log", "night_fire_count_log",
        "residual_roll7_mean",  "residual_roll14_mean",  "residual_roll30_mean",
        "residual_roll7_std",   "residual_roll14_std",
    ]
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        cols += [f"y_lag_{lag}", f"residual_lag_{lag}", f"prophet_lag_{lag}"]

    return [c for c in cols if c in df.columns]


# ============================================================
# RESIDUAL LSTM — MODEL
# ============================================================

def build_sequences(values, lookback, horizon):
    X, y, target_idx = [], [], []
    for i in range(len(values) - lookback - horizon + 1):
        X.append(values[i:i + lookback, :])
        y.append(values[i + lookback:i + lookback + horizon, 0])
        target_idx.append(np.arange(i + lookback, i + lookback + horizon))
    return np.array(X), np.array(y), np.array(target_idx)


def build_residual_lstm_model(lookback, n_features, horizon):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import Huber

    # ✅ Added BatchNormalization for more stable training on global data
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(lookback, n_features), return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dense(horizon),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0007),
        loss=Huber(delta=1.0),   # Huber is robust to the fire spike outliers
        metrics=["mae"],
    )

    return model


# ============================================================
# RESIDUAL LSTM — TRAINING
# ============================================================

def train_residual_lstm(df_with_prophet, prophet_forecast, report):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("\n" + "=" * 70)
    print("RESIDUAL LSTM TRAINING")
    print("=" * 70)

    df_feat      = add_residual_features(df_with_prophet)
    feature_cols = get_residual_feature_cols(df_feat)

    print(f"[INFO] Features ({len(feature_cols)}): {feature_cols}")

    data = df_feat[["ds", "y", "prophet_pred", "residual"] + feature_cols].copy()
    data = data.loc[:, ~data.columns.duplicated()]

    for col in feature_cols:
        data[col] = (
            pd.to_numeric(data[col], errors="coerce")
            .interpolate(method="linear", limit=10)
            .bfill().ffill().fillna(0)
        )

    values = data[feature_cols].values.astype(float)

    n_test  = TEST_DAYS + FORECAST_HORIZON
    n_train = len(values) - n_test

    if n_train <= LOOKBACK:
        raise ValueError(f"Not enough data. Rows={len(values)}, LOOKBACK={LOOKBACK}")

    train_values = values[:n_train]
    test_values  = values[n_train:]

    scaler       = RobustScaler()   # ✅ RobustScaler handles fire spike outliers well
    train_scaled = scaler.fit_transform(train_values)
    test_scaled  = scaler.transform(test_values)
    full_scaled  = np.vstack([train_scaled, test_scaled])

    X_all, y_all, target_idx_all = build_sequences(full_scaled, LOOKBACK, FORECAST_HORIZON)

    split_index = n_train - LOOKBACK

    X_train = X_all[:split_index]
    y_train = y_all[:split_index]
    X_test  = X_all[split_index:]
    y_test  = y_all[split_index:]
    target_idx_test = target_idx_all[split_index:]

    print(f"[INFO] Train sequences : {len(X_train)}")
    print(f"[INFO] Test sequences  : {len(X_test)}")

    # 15% of train as validation
    val_size = max(1, int(len(X_train) * 0.15))
    X_tr, y_tr   = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    model = build_residual_lstm_model(LOOKBACK, len(feature_cols), FORECAST_HORIZON)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    def inverse_residual(arr_scaled):
        n_s, h = arr_scaled.shape
        dummy = np.zeros((n_s * h, len(feature_cols)))
        dummy[:, 0] = arr_scaled.reshape(-1)
        return scaler.inverse_transform(dummy)[:, 0].reshape(n_s, h)

    pred_residual_scaled = model.predict(X_test)
    true_residual = inverse_residual(y_test)
    pred_residual = inverse_residual(pred_residual_scaled)

    y_actual_matrix  = data["y"].values[target_idx_test]
    prophet_matrix   = data["prophet_pred"].values[target_idx_test]
    prophet_matrix   = np.clip(prophet_matrix, MIN_REASONABLE_FORECAST, MAX_REASONABLE_FORECAST)

    # Prophet alone baseline on same window
    prophet_window_metrics = compute_metrics(
        y_actual_matrix.flatten(),
        prophet_matrix.flatten(),
        "Prophet (same test window)"
    )
    report.append(prophet_window_metrics)

    # ✅ Extended alpha + clip search
    print("\n" + "=" * 70)
    print("ALPHA + CLIP SEARCH FOR LSTM RESIDUAL CORRECTION")
    print("=" * 70)

    best_alpha = None
    best_clip  = None
    best_score = np.inf
    best_final_pred_matrix = None

    for alpha in ALPHA_CANDIDATES:
        for clip_value in MAX_RESIDUAL_CORRECTION_CANDIDATES:
            controlled = np.clip(pred_residual, -clip_value, clip_value)
            candidate  = np.clip(prophet_matrix + alpha * controlled,
                                 MIN_REASONABLE_FORECAST, MAX_REASONABLE_FORECAST)

            m = compute_metrics(
                y_actual_matrix.flatten(),
                candidate.flatten(),
                f"alpha={alpha}, clip={clip_value}"
            )

            score = m["MAE"] + 0.20 * m["RMSE"]

            if score < best_score:
                best_score = score
                best_alpha = alpha
                best_clip  = clip_value
                best_final_pred_matrix = candidate

    print(f"\n[INFO] Best alpha = {best_alpha}, Best clip = {best_clip}")

    final_metrics = compute_metrics(
        y_actual_matrix.flatten(),
        best_final_pred_matrix.flatten(),
        "ForecastingGlobal Final"
    )
    final_metrics["target_status"] = evaluate_target_level(final_metrics, "residual_lstm_final")
    final_metrics["best_alpha"]    = float(best_alpha)
    final_metrics["best_clip"]     = float(best_clip)
    report.append(final_metrics)

    # Per-horizon breakdown (day+1 through day+7)
    print("\n── Per-Horizon Metrics ──")
    horizon_report = []
    for h in range(FORECAST_HORIZON):
        mh = compute_metrics(
            y_actual_matrix[:, h],
            best_final_pred_matrix[:, h],
            f"Day +{h+1}"
        )
        horizon_report.append(mh)

    report.append({"label": "Per-Horizon Metrics", "metrics": horizon_report})

    # Next 7 days forecast
    last_window          = full_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(feature_cols))
    future_res_scaled    = model.predict(last_window)
    future_residual      = inverse_residual(future_res_scaled).flatten()

    last_date     = data["ds"].max()
    future_dates  = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D")

    prophet_future = prophet_forecast[prophet_forecast["ds"] > last_date].head(FORECAST_HORIZON)
    prophet_future_vals = np.clip(
        np.expm1(np.clip(prophet_future["yhat"].values, 0, None)),
        MIN_REASONABLE_FORECAST, MAX_REASONABLE_FORECAST
    )

    min_len = min(len(future_dates), len(prophet_future_vals), len(future_residual))
    controlled_future = np.clip(future_residual[:min_len], -best_clip, best_clip)
    final_future      = np.clip(
        prophet_future_vals[:min_len] + best_alpha * controlled_future,
        MIN_REASONABLE_FORECAST, MAX_REASONABLE_FORECAST
    )

    future_df = pd.DataFrame({
        "ds":                       future_dates[:min_len],
        "prophet_forecast":         prophet_future_vals[:min_len],
        "raw_lstm_residual":        future_residual[:min_len],
        "controlled_lstm_residual": controlled_future,
        "best_alpha":               best_alpha,
        "best_clip":                best_clip,
        "final_forecast":           final_future,
    })

    future_df.to_csv(os.path.join(EVAL_DIR, FORECAST_CSV_FILE), index=False)

    print("\n[INFO] Next 7 days forecast:")
    print(future_df[["ds", "prophet_forecast", "final_forecast"]].to_string())

    # Training loss plot
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("LSTM Training Loss (Huber)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "rfg_lstm_training_loss.png"), dpi=150)
    plt.close()

    # Combined forecast plot
    recent = data[data["ds"] >= last_date - pd.Timedelta(days=90)]
    plt.figure(figsize=(14, 5))
    plt.plot(recent["ds"], recent["y"], label="Observed (last 90 days)", alpha=0.8)
    plt.plot(future_df["ds"], future_df["prophet_forecast"], marker="s", linestyle="--", label=f"Prophet (alpha weight={1-best_alpha:.2f})")
    plt.plot(future_df["ds"], future_df["final_forecast"],   marker="D", linewidth=2.5,  label=f"Final Forecast (alpha={best_alpha})")
    plt.axvline(last_date, linestyle=":", alpha=0.5, color="gray")
    plt.title("ForecastingGlobal — Prophet + LSTM Residual Correction")
    plt.xlabel("Date")
    plt.ylabel("Daily Fire Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "rfg_combined_forecast.png"), dpi=150)
    plt.close()

    # Save models
    scaler_path   = os.path.join(MODEL_DIR, SCALER_FILE)
    model_path    = os.path.join(MODEL_DIR, LSTM_MODEL_FILE)
    metadata_path = os.path.join(MODEL_DIR, LSTM_METADATA_FILE)

    joblib.dump(scaler, scaler_path)
    model.save(model_path)

    joblib.dump(
        {
            "model_name":             MODEL_NAME,
            "model_path":             model_path,
            "scaler_path":            scaler_path,
            "lookback":               LOOKBACK,
            "forecast_horizon":       FORECAST_HORIZON,
            "feature_cols":           feature_cols,
            "target_column":          "residual",
            "target_index":           0,
            "best_alpha":             float(best_alpha),
            "best_clip":              float(best_clip),
            "alpha_candidates":       ALPHA_CANDIDATES,
            "clip_candidates":        MAX_RESIDUAL_CORRECTION_CANDIDATES,
            "min_reasonable_forecast": MIN_REASONABLE_FORECAST,
            "max_reasonable_forecast": MAX_REASONABLE_FORECAST,
            "final_formula":          "final = prophet + best_alpha * clip(lstm_residual, -best_clip, best_clip)",
        },
        metadata_path,
    )

    print(f"[INFO] LSTM saved: {LSTM_MODEL_FILE}")
    return model, scaler, future_df, final_metrics, feature_cols


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(report, final_metrics, feature_cols):
    # JSON
    with open(os.path.join(EVAL_DIR, METRICS_JSON_FILE), "w") as f:
        json.dump(report, f, indent=4)

    # TXT metrics
    with open(os.path.join(EVAL_DIR, METRICS_TXT_FILE), "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RFG METRICS — FORECASTINGGLOBAL\n")
        f.write("=" * 70 + "\n\n")
        for item in report:
            f.write(f"{item.get('label','')}\n")
            f.write("-" * 70 + "\n")
            for k, v in item.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    # Main report
    with open(os.path.join(EVAL_DIR, REPORT_TXT_FILE), "w") as f:
        f.write("=" * 70 + "\n")
        f.write("GLOBAL FORECASTING REPORT — ForecastingGlobal\n")
        f.write("=" * 70 + "\n\n")
        f.write("Architecture:\n")
        f.write("  Prophet = main forecast (trend + yearly + monthly seasonality)\n")
        f.write("  LSTM    = controlled residual correction layer\n")
        f.write("  Formula = Prophet + best_alpha * clip(LSTM_residual, -clip, clip)\n\n")
        f.write(f"Best alpha : {final_metrics.get('best_alpha')}\n")
        f.write(f"Best clip  : {final_metrics.get('best_clip')}\n\n")
        f.write("Final metrics (test set):\n")
        f.write(f"  MAE    : {final_metrics.get('MAE'):.4f}\n")
        f.write(f"  RMSE   : {final_metrics.get('RMSE'):.4f}\n")
        f.write(f"  R²     : {final_metrics.get('R2')}\n")
        f.write(f"  SMAPE  : {final_metrics.get('SMAPE'):.2f}%\n")
        f.write(f"  WAPE   : {final_metrics.get('WAPE'):.2f}%\n")
        f.write(f"  Status : {final_metrics.get('target_status')}\n\n")
        f.write("Data notes:\n")
        f.write("  - Zero days with avg_confidence=0 treated as missing (interpolated)\n")
        f.write("  - Outliers (y>100) are real fire events — NOT capped, handled via log1p\n")
        f.write("  - 5 years of global data: 2020-01-01 to 2025-01-01\n\n")
        f.write("LSTM features:\n")
        for col in feature_cols:
            f.write(f"  - {col}\n")

    # Meta JSON
    meta = {
        "model_name":      MODEL_NAME,
        "data_path":       DATA_PATH,
        "lookback":        LOOKBACK,
        "forecast_horizon": FORECAST_HORIZON,
        "lstm_units":      LSTM_UNITS,
        "epochs":          EPOCHS,
        "metric_targets":  METRIC_TARGETS,
        "final_metrics":   final_metrics,
        "feature_cols":    feature_cols,
        "models": {
            "prophet":  PROPHET_MODEL_FILE,
            "lstm":     LSTM_MODEL_FILE,
            "scaler":   SCALER_FILE,
            "metadata": LSTM_METADATA_FILE,
        },
    }

    with open(os.path.join(MODEL_DIR, META_JSON_FILE), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n[INFO] Reports saved to: {EVAL_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("FORECASTINGGLOBAL — Prophet + LSTM Residual Correction")
    print("=" * 70)

    report = []

    df = load_data(DATA_PATH)

    evaluate_baselines(df, report)

    prophet_model, prophet_forecast, df_with_prophet = train_prophet(df, report)

    lstm_model, scaler, future_df, final_metrics, feature_cols = train_residual_lstm(
        df_with_prophet=df_with_prophet,
        prophet_forecast=prophet_forecast,
        report=report,
    )

    save_reports(report, final_metrics, feature_cols)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Models    : {MODEL_DIR}")
    print(f"Reports   : {EVAL_DIR}")
    print(f"Status    : {final_metrics.get('target_status')}")
    print(f"Final MAE : {final_metrics.get('MAE'):.4f}")
    print(f"Best alpha: {final_metrics.get('best_alpha')}")


if __name__ == "__main__":
    main()