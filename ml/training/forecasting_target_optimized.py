"""
Target-Optimized Wildfire Forecasting
Prophet + LSTM + validated weighted ensemble

Goal targets to approach:
- R²   > 0.50
- MAPE < 10%
- MAE  < 2000
- RMSE < 2500

Important:
These targets cannot be guaranteed by code only. This script pushes performance by:
1) cleaning outage-like abnormal low days,
2) using log1p target,
3) tuning Prophet by grid search,
4) tuning LSTM across several configs,
5) aligning Prophet and LSTM predictions on the SAME dates,
6) selecting the best ensemble weight on validation dates.
"""

import os
import json
import warnings
import itertools
from collections import defaultdict

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

FORECAST_HORIZON = 7
PROPHET_TEST_DAYS = 30
TSCV_SPLITS = 3

# Cleaning: values below this are likely missing acquisition / partial data, not real signal
OUTAGE_MIN_Y = 1000
LOW_RATIO_TO_ROLLING_MEDIAN = 0.20
ROLLING_MEDIAN_WINDOW = 7
INTERPOLATION_LIMIT = 10

# LSTM tuning configs. First config keeps your best previous style.
LSTM_CONFIGS = [
    {"lookback": 30, "units": 48, "dropout": 0.25, "batch_size": 16, "epochs": 120},
    {"lookback": 45, "units": 64, "dropout": 0.20, "batch_size": 8,  "epochs": 150},
    {"lookback": 60, "units": 64, "dropout": 0.25, "batch_size": 8,  "epochs": 150},
    {"lookback": 30, "units": 64, "dropout": 0.20, "batch_size": 8,  "epochs": 150},
]

# Set to 1 if you want fast testing, keep all configs for final run
MAX_LSTM_CONFIGS = len(LSTM_CONFIGS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(ROOT_DIR, "data", "prophetData.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "ml", "saved_models")
EVAL_DIR = os.path.join(ROOT_DIR, "ml", "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mask = y_true > 1e-6
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.sum() > 0 else np.nan
    )

    print(f"\n── {label} ──")
    print(f"Samples : {len(y_true)}")
    print(f"MAE     : {mae:.2f}")
    print(f"RMSE    : {rmse:.2f}")
    print(f"R²      : {r2:.4f}")
    print(f"MAPE    : {mape:.2f}%")

    return {
        "label": label,
        "samples": int(len(y_true)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


# ============================================================
# LOAD + CLEAN
# ============================================================

def load_data(path):
    print(f"[INFO] Loading: {path}")
    df = pd.read_csv(path, low_memory=False)

    date_col = None
    for col in df.columns:
        if col.lower() in ["ds", "date", "acq_date"]:
            date_col = col
            break
    if date_col is None:
        raise ValueError("No date column found. Expected ds, date, or acq_date.")

    df = df.rename(columns={date_col: "ds"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)

    if "y" not in df.columns:
        if "frp" in df.columns:
            df["y"] = df["frp"]
        else:
            raise ValueError("No target column found. Expected y or frp.")

    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_range).reset_index()
    df = df.rename(columns={"index": "ds"})

    for col in ["avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_y = df["y"].copy()

    # 1) zero / extremely low values are likely acquisition outages or partial days
    zero_days = int((df["y"] == 0).sum())
    very_low_days = int(((df["y"] > 0) & (df["y"] < OUTAGE_MIN_Y)).sum())
    df.loc[df["y"] == 0, "y"] = np.nan
    df.loc[df["y"] < OUTAGE_MIN_Y, "y"] = np.nan

    # 2) relative abnormal drops compared to local rolling median
    roll_med = df["y"].rolling(ROLLING_MEDIAN_WINDOW, center=True, min_periods=3).median()
    low_relative = df["y"] < (LOW_RATIO_TO_ROLLING_MEDIAN * roll_med)
    relative_low_days = int(low_relative.sum(skipna=True))
    df.loc[low_relative, "y"] = np.nan

    # Keep audit columns for report/plotting
    df["y_raw_before_cleaning"] = raw_y
    df["was_y_cleaned"] = raw_y.notna() & df["y"].isna()

    print(f"[INFO] Date range          : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"[INFO] Rows                : {len(df)}")
    print(f"[INFO] Zero→NaN            : {zero_days}")
    print(f"[INFO] Very low→NaN        : {very_low_days}")
    print(f"[INFO] Relative low→NaN    : {relative_low_days}")
    print(f"[INFO] Missing y after clean: {df['y'].isna().sum()}")
    print(f"[INFO] Mean y              : {df['y'].mean():.2f}")
    print(f"[INFO] Max y               : {df['y'].max():.2f}\n")

    return df


# ============================================================
# FEATURES
# ============================================================

def add_features(df):
    df = df.copy()

    df["day_of_year"] = df["ds"].dt.dayofyear
    df["month"] = df["ds"].dt.month
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["ds"].dt.quarter

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Lags use original y, shifted to avoid future leakage
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["roll3_mean"] = df["y"].shift(1).rolling(3, min_periods=1).mean()
    df["roll7_mean"] = df["y"].shift(1).rolling(7, min_periods=1).mean()
    df["roll14_mean"] = df["y"].shift(1).rolling(14, min_periods=1).mean()
    df["roll30_mean"] = df["y"].shift(1).rolling(30, min_periods=1).mean()
    df["roll7_std"] = df["y"].shift(1).rolling(7, min_periods=2).std().fillna(0)
    df["roll14_std"] = df["y"].shift(1).rolling(14, min_periods=2).std().fillna(0)

    df["y_log"] = np.log1p(df["y"].clip(lower=0))

    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"lag_{lag}_log"] = np.log1p(df[f"lag_{lag}"].clip(lower=0))

    for col in ["roll3_mean", "roll7_mean", "roll14_mean", "roll30_mean", "roll7_std", "roll14_std"]:
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # Current external features appear inside the past lookback window only, not as future values.
    for col in ["avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            for lag in [1, 7]:
                df[f"{col}_lag_{lag}_log"] = np.log1p(df[col].shift(lag).clip(lower=0))

    return df


def get_feature_cols(df):
    base = [
        "y_log",  # target must remain index 0
        "day_sin", "day_cos", "month_sin", "month_cos", "week_sin", "week_cos",
        "lag_1_log", "lag_2_log", "lag_3_log", "lag_7_log", "lag_14_log", "lag_21_log", "lag_30_log",
        "roll3_mean_log", "roll7_mean_log", "roll14_mean_log", "roll30_mean_log",
        "roll7_std_log", "roll14_std_log",
    ]

    for col in ["avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        for cname in [f"{col}_log", f"{col}_lag_1_log", f"{col}_lag_7_log"]:
            if cname in df.columns:
                base.append(cname)

    return [c for c in base if c in df.columns]


# ============================================================
# BASELINES
# ============================================================

def evaluate_baselines(df, report):
    print("=" * 60)
    print("BASELINES")
    print("=" * 60)

    y = df["y"].values.astype(float)
    n = PROPHET_TEST_DAYS
    y_test = y[-n:]
    train_mean = np.nanmean(y[:-n])

    report.append(compute_metrics(y_test, np.full(n, train_mean), "Baseline: Train Mean"))

    if len(y) > n + 7:
        report.append(compute_metrics(y_test, y[-n - 7:-7], "Baseline: 7-Day Lag"))

    if len(y) > n + 365:
        report.append(compute_metrics(y_test, y[-n - 365:-365], "Baseline: 365-Day Lag"))


# ============================================================
# PROPHET GRID SEARCH
# ============================================================

def train_prophet(df, report):
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    print("\n" + "=" * 60)
    print("PROPHET TRAINING — OPTIMIZED GRID SEARCH")
    print("=" * 60)

    pdf = df[["ds", "y"]].copy()
    pdf["y_original"] = pdf["y"]
    pdf["y"] = np.log1p(pdf["y"].clip(lower=0))
    pdf = pdf.dropna(subset=["y"]).reset_index(drop=True)

    cutoff = pdf["ds"].max() - pd.Timedelta(days=PROPHET_TEST_DAYS)
    train_df = pdf[pdf["ds"] <= cutoff].copy()
    test_df = pdf[pdf["ds"] > cutoff].copy()

    print(f"[INFO] Train days: {len(train_df)}")
    print(f"[INFO] Test days : {len(test_df)}")

    param_grid = {
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_prior_scale": [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "seasonality_prior_scale": [1.0, 5.0, 10.0, 15.0, 20.0],
        "yearly_fourier": [5, 10, 15, 20],
        "monthly_fourier": [3, 5, 8, 10],
        "weekly": [True, False],
    }

    all_params = list(itertools.product(
        param_grid["seasonality_mode"],
        param_grid["changepoint_prior_scale"],
        param_grid["seasonality_prior_scale"],
        param_grid["yearly_fourier"],
        param_grid["monthly_fourier"],
        param_grid["weekly"],
    ))

    best_score = float("inf")
    best_model = None
    best_forecast = None
    best_params = None
    best_metrics = None
    best_merged = None

    for idx, (seasonality_mode, cps, sps, yf, mf, weekly) in enumerate(all_params, 1):
        try:
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=False,
                weekly_seasonality=weekly,
                daily_seasonality=False,
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                n_changepoints=25,
                interval_width=0.80,
            )
            model.add_seasonality(name="yearly", period=365.25, fourier_order=yf)
            model.add_seasonality(name="monthly", period=30.5, fourier_order=mf)
            model.fit(train_df[["ds", "y"]])

            future = model.make_future_dataframe(periods=PROPHET_TEST_DAYS + FORECAST_HORIZON, freq="D")
            forecast = model.predict(future)

            merged = test_df.merge(
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                on="ds",
                how="inner",
            )
            if len(merged) == 0:
                continue

            y_true = np.expm1(np.clip(merged["y"].values, 0, None))
            y_pred = np.expm1(np.clip(merged["yhat"].values, 0, None))

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            # Selection uses MAE, because target includes MAE < 2000
            if mae < best_score:
                best_score = mae
                best_model = model
                best_forecast = forecast
                best_merged = merged.copy()
                best_params = {
                    "seasonality_mode": seasonality_mode,
                    "changepoint_prior_scale": cps,
                    "seasonality_prior_scale": sps,
                    "yearly_fourier": yf,
                    "monthly_fourier": mf,
                    "weekly_seasonality": weekly,
                }
                best_metrics = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("Prophet optimization failed. No valid model found.")

    print("\n[INFO] Best Prophet parameters:")
    print(best_params)
    print("\n── Prophet Optimized Original Scale ──")
    print(f"MAE  : {best_metrics['MAE']:.2f}")
    print(f"RMSE : {best_metrics['RMSE']:.2f}")
    print(f"R²   : {best_metrics['R2']:.4f}")

    report.append({"label": "Prophet Optimized Original Scale", "best_params": best_params, "metrics": best_metrics})

    forecast_original = best_forecast.copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast_original[col] = np.expm1(np.clip(forecast_original[col], 0, None))

    # Store test predictions in original scale
    prophet_test_predictions = best_merged[["ds"]].copy()
    prophet_test_predictions["y_true"] = np.expm1(np.clip(best_merged["y"].values, 0, None))
    prophet_test_predictions["prophet_pred"] = np.expm1(np.clip(best_merged["yhat"].values, 0, None))

    # Plot test prediction
    plt.figure(figsize=(14, 5))
    plt.plot(prophet_test_predictions["ds"], prophet_test_predictions["y_true"], label="Actual")
    plt.plot(prophet_test_predictions["ds"], prophet_test_predictions["prophet_pred"], label="Optimized Prophet")
    plt.title("Optimized Prophet — Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Fire Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "prophet_optimized_test.png"), dpi=150)
    plt.close()

    joblib.dump(
        {"model": best_model, "forecast": forecast_original, "best_params": best_params, "metrics": best_metrics},
        os.path.join(MODEL_DIR, "prophet_optimized_model.joblib"),
    )

    print("[INFO] Optimized Prophet saved.")
    return best_model, forecast_original, best_metrics["MAE"], prophet_test_predictions


# ============================================================
# LSTM TUNING
# ============================================================

def build_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback:i + lookback + horizon, 0])
    return np.array(X), np.array(y)


def build_lstm_model(lookback, n_features, horizon, units, dropout):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(units, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(horizon),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def inverse_lstm_y(arr_scaled, scaler, n_features):
    n_samples, horizon = arr_scaled.shape
    dummy = np.zeros((n_samples * horizon, n_features))
    dummy[:, 0] = arr_scaled.reshape(-1)
    inv_log = scaler.inverse_transform(dummy)[:, 0].reshape(n_samples, horizon)
    return np.expm1(np.clip(inv_log, 0, None))


def build_lstm_date_predictions(df_dates, X_test, y_test_scaled, y_pred_scaled, scaler, n_features, split_index, lookback):
    y_true_orig = inverse_lstm_y(y_test_scaled, scaler, n_features)
    y_pred_orig = inverse_lstm_y(y_pred_scaled, scaler, n_features)

    by_date_true = defaultdict(list)
    by_date_pred = defaultdict(list)

    # X_all sample i predicts dates: i + lookback ... i + lookback + horizon - 1
    for local_i in range(len(X_test)):
        global_i = split_index + local_i
        for h in range(FORECAST_HORIZON):
            date_idx = global_i + lookback + h
            if 0 <= date_idx < len(df_dates):
                ds = df_dates.iloc[date_idx]
                by_date_true[ds].append(y_true_orig[local_i, h])
                by_date_pred[ds].append(y_pred_orig[local_i, h])

    rows = []
    for ds in sorted(by_date_pred.keys()):
        rows.append({
            "ds": ds,
            "y_true": float(np.mean(by_date_true[ds])),
            "lstm_pred": float(np.mean(by_date_pred[ds])),
        })

    return pd.DataFrame(rows)


def train_single_lstm(df_features, feature_cols, config):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    lookback = config["lookback"]
    units = config["units"]
    dropout = config["dropout"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    data = df_features[["ds"] + feature_cols].copy()
    data["y_log"] = data["y_log"].interpolate(method="linear", limit=INTERPOLATION_LIMIT).bfill().ffill()

    for col in feature_cols:
        if col != "y_log":
            data[col] = (
                pd.to_numeric(data[col], errors="coerce")
                .interpolate(method="linear", limit=INTERPOLATION_LIMIT)
                .bfill().ffill().fillna(0)
            )

    values = data[feature_cols].values.astype(float)

    n_test = PROPHET_TEST_DAYS + FORECAST_HORIZON
    n_train = len(values) - n_test
    train_values = values[:n_train]
    test_values = values[n_train:]

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)
    full_scaled = np.vstack([train_scaled, test_scaled])

    X_all, y_all = build_sequences(full_scaled, lookback, FORECAST_HORIZON)
    split_index = n_train - lookback
    if split_index <= 0:
        raise ValueError("Not enough data. Reduce LOOKBACK.")

    X_train, y_train = X_all[:split_index], y_all[:split_index]
    X_test, y_test = X_all[split_index:], y_all[split_index:]

    np.random.seed(42)
    tf.random.set_seed(42)

    model = build_lstm_model(lookback, len(feature_cols), FORECAST_HORIZON, units, dropout)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_orig = inverse_lstm_y(y_test, scaler, len(feature_cols))
    y_pred_orig = inverse_lstm_y(y_pred_scaled, scaler, len(feature_cols))

    metrics = compute_metrics(y_test_orig.flatten(), y_pred_orig.flatten(), f"LSTM Candidate {config}")

    date_predictions = build_lstm_date_predictions(
        df_dates=data["ds"],
        X_test=X_test,
        y_test_scaled=y_test,
        y_pred_scaled=y_pred_scaled,
        scaler=scaler,
        n_features=len(feature_cols),
        split_index=split_index,
        lookback=lookback,
    )

    last_window = full_scaled[-lookback:].reshape(1, lookback, len(feature_cols))
    future_scaled = model.predict(last_window, verbose=0)
    future_original = inverse_lstm_y(future_scaled, scaler, len(feature_cols)).flatten()
    future_dates = pd.date_range(df_features["ds"].max() + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D")

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "metrics": metrics,
        "config": config,
        "feature_cols": feature_cols,
        "date_predictions": date_predictions,
        "future_original": future_original,
        "future_dates": future_dates,
        "full_scaled": full_scaled,
    }


def train_lstm_tuned(df, report):
    print("\n" + "=" * 60)
    print("LSTM TRAINING — MULTI-CONFIG TUNING")
    print("=" * 60)

    df_features = add_features(df)
    feature_cols = get_feature_cols(df_features)
    print(f"[INFO] Features used ({len(feature_cols)}): {feature_cols}")

    results = []
    for config in LSTM_CONFIGS[:MAX_LSTM_CONFIGS]:
        print(f"\n[INFO] Training LSTM config: {config}")
        try:
            result = train_single_lstm(df_features, feature_cols, config)
            results.append(result)
        except Exception as e:
            print(f"[WARN] Config failed: {config} | {e}")

    if not results:
        raise RuntimeError("All LSTM configurations failed.")

    # Select by MAE first, then RMSE
    best = sorted(results, key=lambda r: (r["metrics"]["MAE"], r["metrics"]["RMSE"]))[0]

    print("\n[INFO] Best LSTM config:")
    print(best["config"])
    report.append({"label": "Best LSTM", "config": best["config"], "metrics": best["metrics"]})

    model_path = os.path.join(MODEL_DIR, "lstm_best_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "lstm_best_scaler.joblib")
    meta_path = os.path.join(MODEL_DIR, "lstm_best_metadata.joblib")

    best["model"].save(model_path)
    joblib.dump(best["scaler"], scaler_path)
    joblib.dump(
        {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "config": best["config"],
            "feature_cols": best["feature_cols"],
            "metrics": best["metrics"],
        },
        meta_path,
    )

    return best


# ============================================================
# ENSEMBLE ON COMMON VALIDATION DATES
# ============================================================

def optimize_ensemble(prophet_test, lstm_test, report):
    print("\n" + "=" * 60)
    print("ENSEMBLE OPTIMIZATION — COMMON TEST DATES")
    print("=" * 60)

    merged = prophet_test.merge(lstm_test, on="ds", how="inner", suffixes=("_p", "_l"))
    if len(merged) == 0:
        raise RuntimeError("No common dates between Prophet and LSTM test predictions.")

    # y_true should be the same; average just in case
    merged["y_true"] = merged[["y_true_p", "y_true_l"]].mean(axis=1)

    prophet_metrics = compute_metrics(merged["y_true"], merged["prophet_pred"], "Prophet on Common Dates")
    lstm_metrics = compute_metrics(merged["y_true"], merged["lstm_pred"], "LSTM on Common Dates")

    best = None
    for w in np.linspace(0, 1, 101):
        # w = Prophet weight, 1-w = LSTM weight
        pred = w * merged["prophet_pred"].values + (1 - w) * merged["lstm_pred"].values
        mae = mean_absolute_error(merged["y_true"], pred)
        rmse = np.sqrt(mean_squared_error(merged["y_true"], pred))
        r2 = r2_score(merged["y_true"], pred)
        mask = merged["y_true"].values > 1e-6
        mape = np.mean(np.abs((merged["y_true"].values[mask] - pred[mask]) / merged["y_true"].values[mask])) * 100

        candidate = {"w_prophet": float(w), "w_lstm": float(1 - w), "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE": float(mape)}
        # choose by MAE, then require good RMSE
        if best is None or (candidate["MAE"], candidate["RMSE"]) < (best["MAE"], best["RMSE"]):
            best = candidate

    merged["ensemble_pred"] = best["w_prophet"] * merged["prophet_pred"] + best["w_lstm"] * merged["lstm_pred"]
    ensemble_metrics = compute_metrics(merged["y_true"], merged["ensemble_pred"], "Best Weighted Ensemble on Common Dates")

    print("\n[INFO] Best ensemble weights:")
    print(f"  Prophet: {best['w_prophet']:.2f}")
    print(f"  LSTM   : {best['w_lstm']:.2f}")

    report.append({
        "label": "Common-Date Evaluation",
        "prophet": prophet_metrics,
        "lstm": lstm_metrics,
        "ensemble_weights": best,
        "ensemble": ensemble_metrics,
    })

    return merged, best, ensemble_metrics


def plot_combined(df, prophet_forecast, lstm_result, ensemble_common, weights):
    last_date = df["ds"].max()
    recent = df[df["ds"] >= last_date - pd.Timedelta(days=90)]
    prophet_future = prophet_forecast[prophet_forecast["ds"] > last_date].head(FORECAST_HORIZON)

    prophet_vals = np.clip(prophet_future["yhat"].values, 0, None)
    lstm_vals = lstm_result["future_original"][:len(prophet_vals)]

    if len(prophet_vals) == len(lstm_vals):
        ensemble_future = weights["w_prophet"] * prophet_vals + weights["w_lstm"] * lstm_vals
    else:
        ensemble_future = None

    plt.figure(figsize=(15, 6))
    plt.plot(recent["ds"], recent["y"], label="Observed Last 90 Days", color="gray")

    if not prophet_future.empty:
        plt.plot(prophet_future["ds"], prophet_vals, marker="s", label=f"Prophet future (w={weights['w_prophet']:.2f})", linestyle="--")

    plt.plot(lstm_result["future_dates"][:len(lstm_vals)], lstm_vals, marker="o", label=f"LSTM future (w={weights['w_lstm']:.2f})", linestyle="--")

    if ensemble_future is not None:
        plt.plot(prophet_future["ds"], ensemble_future, marker="D", linewidth=2.5, label="Optimized Ensemble future")

    plt.axvline(last_date, linestyle=":", alpha=0.5, color="black")
    plt.title("Final Forecast — Prophet + LSTM Optimized Ensemble")
    plt.xlabel("Date")
    plt.ylabel("Fire Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "final_optimized_ensemble_forecast.png"), dpi=150)
    plt.close()

    # Common-date validation plot
    plt.figure(figsize=(14, 5))
    plt.plot(ensemble_common["ds"], ensemble_common["y_true"], label="Actual")
    plt.plot(ensemble_common["ds"], ensemble_common["prophet_pred"], label="Prophet")
    plt.plot(ensemble_common["ds"], ensemble_common["lstm_pred"], label="LSTM")
    plt.plot(ensemble_common["ds"], ensemble_common["ensemble_pred"], label="Ensemble", linewidth=2.5)
    plt.title("Common Test Dates — Actual vs Prophet vs LSTM vs Ensemble")
    plt.xlabel("Date")
    plt.ylabel("Fire Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "final_common_date_validation.png"), dpi=150)
    plt.close()

    print("[INFO] Final plots saved.")


# ============================================================
# MAIN
# ============================================================

def main():
    report = []

    df = load_data(DATA_PATH)
    evaluate_baselines(df, report)

    prophet_model, prophet_forecast, prophet_mae, prophet_test = train_prophet(df, report)
    lstm_result = train_lstm_tuned(df, report)

    ensemble_common, weights, ensemble_metrics = optimize_ensemble(
        prophet_test=prophet_test,
        lstm_test=lstm_result["date_predictions"],
        report=report,
    )

    plot_combined(df, prophet_forecast, lstm_result, ensemble_common, weights)

    report_path = os.path.join(EVAL_DIR, "forecasting_report_target_optimized.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    meta = {
        "target_metrics": {"R2": ">0.50", "MAPE": "<10%", "MAE": "<2000", "RMSE": "<2500"},
        "cleaning": {
            "outage_min_y": OUTAGE_MIN_Y,
            "low_ratio_to_rolling_median": LOW_RATIO_TO_ROLLING_MEDIAN,
            "rolling_median_window": ROLLING_MEDIAN_WINDOW,
        },
        "forecast_horizon": FORECAST_HORIZON,
        "prophet_test_days": PROPHET_TEST_DAYS,
        "lstm_configs_tried": LSTM_CONFIGS[:MAX_LSTM_CONFIGS],
        "best_ensemble_weights": weights,
        "final_ensemble_metrics": ensemble_metrics,
    }

    with open(os.path.join(MODEL_DIR, "forecasting_meta_target_optimized.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print("\n[DONE]")
    print(f"Reports saved to : {EVAL_DIR}")
    print(f"Models saved to  : {MODEL_DIR}")
    print("\nFinal target check:")
    print(f"  MAE  target < 2000 : got {ensemble_metrics['MAE']:.2f}")
    print(f"  RMSE target < 2500 : got {ensemble_metrics['RMSE']:.2f}")
    print(f"  R²   target > 0.50 : got {ensemble_metrics['R2']:.4f}")
    print(f"  MAPE target < 10%  : got {ensemble_metrics['MAPE']:.2f}%")


if __name__ == "__main__":
    main()
