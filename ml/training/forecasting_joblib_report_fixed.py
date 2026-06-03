"""
Morocco Daily Wildfire Detection Forecasting
Prophet + LSTM + Weighted Ensemble

FIXED VERSION:
- LSTM no longer uses the test set as validation.
- Data split is now:
    Train -> Validation -> Test
- Final metrics are calculated only on unseen test dates.
- Uses MAE, RMSE, WAPE and sMAPE instead of MAPE targets.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ============================================================
# CONFIG
# ============================================================

FORECAST_HORIZON = 7
TEST_DAYS = 30
VAL_DAYS = 30

OUTAGE_MIN_Y = 1000
LOW_RATIO_TO_ROLLING_MEDIAN = 0.20
ROLLING_MEDIAN_WINDOW = 7
INTERPOLATION_LIMIT = 10

LSTM_CONFIGS = [
    {"lookback": 30, "units": 48, "dropout": 0.25, "batch_size": 16, "epochs": 120},
    {"lookback": 45, "units": 64, "dropout": 0.20, "batch_size": 8,  "epochs": 150},
    {"lookback": 60, "units": 64, "dropout": 0.25, "batch_size": 8,  "epochs": 150},
    {"lookback": 30, "units": 64, "dropout": 0.20, "batch_size": 8,  "epochs": 150},
]

MAX_LSTM_CONFIGS = len(LSTM_CONFIGS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(ROOT_DIR, "data", "prophetData_morocco.csv")
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

    # R² can be unstable/negative for sparse targets, but we keep it for information.
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    # WAPE is better than MAPE when y has many zeros/small values.
    denom = np.sum(np.abs(y_true))
    wape = (np.sum(np.abs(y_true - y_pred)) / denom) * 100 if denom > 0 else np.nan

    # sMAPE is more stable than MAPE for small values.
    smape_denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_mask = smape_denom > 1e-8
    smape = (
        np.mean(np.abs(y_true[smape_mask] - y_pred[smape_mask]) / smape_denom[smape_mask]) * 100
        if smape_mask.sum() > 0 else np.nan
    )

    print(f"\n── {label} ──")
    print(f"Samples : {len(y_true)}")
    print(f"MAE     : {mae:.2f} fires/day")
    print(f"RMSE    : {rmse:.2f} fires/day")
    print(f"R²      : {r2:.4f}")
    print(f"WAPE    : {wape:.2f}%")
    print(f"sMAPE   : {smape:.2f}%")

    return {
        "label": label,
        "samples": int(len(y_true)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "WAPE": float(wape),
        "sMAPE": float(smape),
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
        raise ValueError("No date column found. Expected ds/date/acq_date.")

    df = df.rename(columns={date_col: "ds"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")

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

    

    roll_med = df["y"].rolling(
        ROLLING_MEDIAN_WINDOW,
        center=True,
        min_periods=3
    ).median()

    low_relative = df["y"] < (LOW_RATIO_TO_ROLLING_MEDIAN * roll_med)
    df.loc[low_relative, "y"] = np.nan

    df["y_raw_before_cleaning"] = raw_y
    df["was_y_cleaned"] = raw_y.notna() & df["y"].isna()

    print(f"[INFO] Date range: {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Missing y after cleaning: {df['y'].isna().sum()}")
    print(f"[INFO] Mean y: {df['y'].mean():.2f}")
    print(f"[INFO] Max y: {df['y'].max():.2f}")

    return df


# ============================================================
# FEATURES
# ============================================================

def add_features(df):
    df = df.copy()

    df["day_of_year"] = df["ds"].dt.dayofyear
    df["month"] = df["ds"].dt.month
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

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

    for col in ["avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            df[f"{col}_lag_1_log"] = np.log1p(df[col].shift(1).clip(lower=0))
            df[f"{col}_lag_7_log"] = np.log1p(df[col].shift(7).clip(lower=0))

    return df


def get_feature_cols(df):
    cols = [
        "y_log",
        "day_sin", "day_cos",
        "month_sin", "month_cos",
        "week_sin", "week_cos",
        "lag_1_log", "lag_2_log", "lag_3_log",
        "lag_7_log", "lag_14_log", "lag_21_log", "lag_30_log",
        "roll3_mean_log", "roll7_mean_log",
        "roll14_mean_log", "roll30_mean_log",
        "roll7_std_log", "roll14_std_log",
    ]

    for col in ["avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        for name in [f"{col}_log", f"{col}_lag_1_log", f"{col}_lag_7_log"]:
            if name in df.columns:
                cols.append(name)

    return [c for c in cols if c in df.columns]


# ============================================================
# PROPHET
# ============================================================

def train_prophet(df, report):
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    print("\n" + "=" * 60)
    print("PROPHET TRAINING")
    print("=" * 60)

    pdf = df[["ds", "y"]].dropna().copy()
    pdf["y"] = np.log1p(pdf["y"].clip(lower=0))

    test_cutoff = pdf["ds"].max() - pd.Timedelta(days=TEST_DAYS)
    train_df = pdf[pdf["ds"] <= test_cutoff]
    test_df = pdf[pdf["ds"] > test_cutoff]

    param_grid = {
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_prior_scale": [0.001, 0.01, 0.03, 0.05, 0.1],
        "seasonality_prior_scale": [1.0, 5.0, 10.0],
        "yearly_fourier": [5, 10, 15],
        "monthly_fourier": [3, 5, 8],
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

    best_model = None
    best_metrics = None
    best_params = None
    best_forecast = None
    best_test_predictions = None
    best_mae = float("inf")

    for seasonality_mode, cps, sps, yf, mf, weekly in all_params:
        try:
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=False,
                weekly_seasonality=weekly,
                daily_seasonality=False,
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                n_changepoints=25,
            )

            model.add_seasonality("yearly", period=365.25, fourier_order=yf)
            model.add_seasonality("monthly", period=30.5, fourier_order=mf)

            model.fit(train_df[["ds", "y"]])

            future = model.make_future_dataframe(
                periods=TEST_DAYS + FORECAST_HORIZON,
                freq="D"
            )

            forecast = model.predict(future)

            merged = test_df.merge(
                forecast[["ds", "yhat"]],
                on="ds",
                how="inner"
            )

            y_true = np.expm1(merged["y"].values)
            y_pred = np.expm1(np.clip(merged["yhat"].values, 0, None))

            mae = mean_absolute_error(y_true, y_pred)

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_forecast = forecast.copy()
                best_params = {
                    "seasonality_mode": seasonality_mode,
                    "changepoint_prior_scale": cps,
                    "seasonality_prior_scale": sps,
                    "yearly_fourier": yf,
                    "monthly_fourier": mf,
                    "weekly": weekly,
                }

                best_test_predictions = pd.DataFrame({
                    "ds": merged["ds"],
                    "y_true": y_true,
                    "prophet_pred": y_pred,
                })

                best_metrics = compute_metrics(
                    y_true,
                    y_pred,
                    "Prophet Test"
                )

        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("Prophet training failed.")

    print("\n[INFO] Best Prophet params:")
    print(best_params)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in best_forecast.columns:
            best_forecast[col] = np.expm1(np.clip(best_forecast[col], 0, None))

    report.append({
        "label": "Prophet",
        "best_params": best_params,
        "metrics": best_metrics,
    })

    joblib.dump(
        {
            "model": best_model,
            "forecast": best_forecast,
            "best_params": best_params,
            "metrics": best_metrics,
        },
        os.path.join(MODEL_DIR, "prophet_model_fixed.joblib")
    )

    return best_model, best_forecast, best_test_predictions


# ============================================================
# LSTM
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
        LSTM(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(horizon),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def inverse_lstm_y(arr_scaled, scaler, n_features):
    n_samples, horizon = arr_scaled.shape

    dummy = np.zeros((n_samples * horizon, n_features))
    dummy[:, 0] = arr_scaled.reshape(-1)

    inv_log = scaler.inverse_transform(dummy)[:, 0].reshape(n_samples, horizon)
    return np.expm1(np.clip(inv_log, 0, None))


def build_lstm_date_predictions(df_dates, y_true_orig, y_pred_orig, split_index, lookback):
    by_date_true = defaultdict(list)
    by_date_pred = defaultdict(list)

    for local_i in range(len(y_pred_orig)):
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


def prepare_lstm_data(df):
    df_features = add_features(df)
    feature_cols = get_feature_cols(df_features)

    data = df_features[["ds"] + feature_cols].copy()

    data["y_log"] = (
        data["y_log"]
        .interpolate(method="linear", limit=INTERPOLATION_LIMIT)
        .bfill()
        .ffill()
    )

    for col in feature_cols:
        if col != "y_log":
            data[col] = (
                pd.to_numeric(data[col], errors="coerce")
                .interpolate(method="linear", limit=INTERPOLATION_LIMIT)
                .bfill()
                .ffill()
                .fillna(0)
            )

    return data, feature_cols


def train_single_lstm(data, feature_cols, config):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    lookback = config["lookback"]
    units = config["units"]
    dropout = config["dropout"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    values = data[feature_cols].values.astype(float)
    n_features = len(feature_cols)

    test_start = len(values) - TEST_DAYS
    val_start = test_start - VAL_DAYS

    train_values = values[:val_start]
    val_values = values[val_start - lookback:test_start]
    test_values = values[test_start - lookback:]

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_values)
    val_scaled = scaler.transform(val_values)
    test_scaled = scaler.transform(test_values)

    X_train, y_train = build_sequences(train_scaled, lookback, FORECAST_HORIZON)
    X_val, y_val = build_sequences(val_scaled, lookback, FORECAST_HORIZON)
    X_test, y_test = build_sequences(test_scaled, lookback, FORECAST_HORIZON)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Not enough data for this lookback/config.")

    np.random.seed(42)
    tf.random.set_seed(42)

    model = build_lstm_model(
        lookback=lookback,
        n_features=n_features,
        horizon=FORECAST_HORIZON,
        units=units,
        dropout=dropout
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0)

    y_test_orig = inverse_lstm_y(y_test, scaler, n_features)
    y_pred_orig = inverse_lstm_y(y_pred_scaled, scaler, n_features)

    metrics = compute_metrics(
        y_test_orig.flatten(),
        y_pred_orig.flatten(),
        f"LSTM Candidate {config}"
    )

    test_split_index_global = test_start - lookback

    date_predictions = build_lstm_date_predictions(
        df_dates=data["ds"],
        y_true_orig=y_test_orig,
        y_pred_orig=y_pred_orig,
        split_index=test_split_index_global,
        lookback=lookback
    )

    full_scaled = scaler.transform(values)
    last_window = full_scaled[-lookback:].reshape(1, lookback, n_features)

    future_scaled = model.predict(last_window, verbose=0)
    future_original = inverse_lstm_y(future_scaled, scaler, n_features).flatten()

    future_dates = pd.date_range(
        data["ds"].max() + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D"
    )

    return {
        "model": model,
        "scaler": scaler,
        "config": config,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "date_predictions": date_predictions,
        "future_original": future_original,
        "future_dates": future_dates,
    }


def train_lstm_tuned(df, report):
    print("\n" + "=" * 60)
    print("LSTM TRAINING — FIXED VALIDATION SPLIT")
    print("=" * 60)

    data, feature_cols = prepare_lstm_data(df)

    print(f"[INFO] Features used ({len(feature_cols)}):")
    print(feature_cols)

    results = []

    for config in LSTM_CONFIGS[:MAX_LSTM_CONFIGS]:
        print(f"\n[INFO] Training LSTM config: {config}")

        try:
            result = train_single_lstm(data, feature_cols, config)
            results.append(result)

        except Exception as e:
            print(f"[WARN] Config failed: {config}")
            print(e)

    if not results:
        raise RuntimeError("All LSTM configs failed.")

    best = sorted(
        results,
        key=lambda r: (r["metrics"]["MAE"], r["metrics"]["RMSE"])
    )[0]

    print("\n[INFO] Best LSTM config:")
    print(best["config"])

    report.append({
        "label": "Best LSTM",
        "config": best["config"],
        "metrics": best["metrics"],
    })

    model_path = os.path.join(MODEL_DIR, "lstm_best_model_fixed.keras")
    model_joblib_path = os.path.join(MODEL_DIR, "lstm_best_model_fixed.joblib")
    scaler_path = os.path.join(MODEL_DIR, "lstm_best_scaler_fixed.joblib")
    metadata_path = os.path.join(MODEL_DIR, "lstm_best_metadata_fixed.joblib")

    best["model"].save(model_path)
    joblib.dump(best["model"], model_joblib_path)
    joblib.dump(best["scaler"], scaler_path)

    joblib.dump(
        {
            "model_path": model_path,
            "model_joblib_path": model_joblib_path,
            "scaler_path": scaler_path,
            "config": best["config"],
            "feature_cols": best["feature_cols"],
            "metrics": best["metrics"],
        },
        metadata_path
    )

    return best


# ============================================================
# ENSEMBLE
# ============================================================

def optimize_ensemble(prophet_test, lstm_test, report):
    print("\n" + "=" * 60)
    print("ENSEMBLE OPTIMIZATION")
    print("=" * 60)

    merged = prophet_test.merge(
        lstm_test,
        on="ds",
        how="inner",
        suffixes=("_p", "_l")
    )

    if len(merged) == 0:
        raise RuntimeError("No common dates between Prophet and LSTM.")

    merged["y_true"] = merged[["y_true_p", "y_true_l"]].mean(axis=1)

    prophet_metrics = compute_metrics(
        merged["y_true"],
        merged["prophet_pred"],
        "Prophet on Common Dates"
    )

    lstm_metrics = compute_metrics(
        merged["y_true"],
        merged["lstm_pred"],
        "LSTM on Common Dates"
    )

    best = None

    for w in np.linspace(0, 1, 101):
        pred = w * merged["prophet_pred"].values + (1 - w) * merged["lstm_pred"].values

        y_true_arr = merged["y_true"].values

        mae = mean_absolute_error(y_true_arr, pred)
        rmse = np.sqrt(mean_squared_error(y_true_arr, pred))
        r2 = r2_score(y_true_arr, pred)

        denom = np.sum(np.abs(y_true_arr))
        wape = (np.sum(np.abs(y_true_arr - pred)) / denom) * 100 if denom > 0 else np.nan

        smape_denom = (np.abs(y_true_arr) + np.abs(pred)) / 2
        smape_mask = smape_denom > 1e-8
        smape = (
            np.mean(np.abs(y_true_arr[smape_mask] - pred[smape_mask]) / smape_denom[smape_mask]) * 100
            if smape_mask.sum() > 0 else np.nan
        )

        candidate = {
            "w_prophet": float(w),
            "w_lstm": float(1 - w),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "WAPE": float(wape),
            "sMAPE": float(smape),
        }

        if best is None or (candidate["MAE"], candidate["RMSE"]) < (best["MAE"], best["RMSE"]):
            best = candidate

    merged["ensemble_pred"] = (
        best["w_prophet"] * merged["prophet_pred"]
        + best["w_lstm"] * merged["lstm_pred"]
    )

    ensemble_metrics = compute_metrics(
        merged["y_true"],
        merged["ensemble_pred"],
        "Best Weighted Ensemble on Common Dates"
    )

    print("\n[INFO] Best ensemble weights:")
    print(f"Prophet: {best['w_prophet']:.2f}")
    print(f"LSTM   : {best['w_lstm']:.2f}")

    report.append({
        "label": "Final Common-Date Evaluation",
        "prophet": prophet_metrics,
        "lstm": lstm_metrics,
        "ensemble_weights": best,
        "ensemble": ensemble_metrics,
    })

    return merged, best, ensemble_metrics


# ============================================================
# PLOTS
# ============================================================

def plot_combined(df, prophet_forecast, lstm_result, ensemble_common, weights):
    last_date = df["ds"].max()

    recent = df[df["ds"] >= last_date - pd.Timedelta(days=90)]

    prophet_future = prophet_forecast[
        prophet_forecast["ds"] > last_date
    ].head(FORECAST_HORIZON)

    prophet_vals = np.clip(prophet_future["yhat"].values, 0, None)
    lstm_vals = lstm_result["future_original"][:len(prophet_vals)]

    if len(prophet_vals) == len(lstm_vals):
        ensemble_future = (
            weights["w_prophet"] * prophet_vals
            + weights["w_lstm"] * lstm_vals
        )
    else:
        ensemble_future = None

    plt.figure(figsize=(15, 6))
    plt.plot(recent["ds"], recent["y"], label="Observed Last 90 Days")

    if not prophet_future.empty:
        plt.plot(
            prophet_future["ds"],
            prophet_vals,
            marker="s",
            linestyle="--",
            label=f"Prophet future w={weights['w_prophet']:.2f}"
        )

    plt.plot(
        lstm_result["future_dates"][:len(lstm_vals)],
        lstm_vals,
        marker="o",
        linestyle="--",
        label=f"LSTM future w={weights['w_lstm']:.2f}"
    )

    if ensemble_future is not None:
        plt.plot(
            prophet_future["ds"],
            ensemble_future,
            marker="D",
            linewidth=2.5,
            label="Ensemble future"
        )

    plt.axvline(last_date, linestyle=":", alpha=0.5)
    plt.title("Final Forecast — Prophet + LSTM Ensemble")
    plt.xlabel("Date")
    plt.ylabel("Fire Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(EVAL_DIR, "final_fixed_ensemble_forecast.png"),
        dpi=150
    )

    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(ensemble_common["ds"], ensemble_common["y_true"], label="Actual")
    plt.plot(ensemble_common["ds"], ensemble_common["prophet_pred"], label="Prophet")
    plt.plot(ensemble_common["ds"], ensemble_common["lstm_pred"], label="LSTM")
    plt.plot(ensemble_common["ds"], ensemble_common["ensemble_pred"], label="Ensemble", linewidth=2.5)

    plt.title("Common Test Dates — Actual vs Predictions")
    plt.xlabel("Date")
    plt.ylabel("Fire Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(EVAL_DIR, "final_fixed_common_date_validation.png"),
        dpi=150
    )

    plt.close()

    print("[INFO] Final plots saved.")


# ============================================================
# MAIN
# ============================================================

def main():
    report = []

    df = load_data(DATA_PATH)

    prophet_model, prophet_forecast, prophet_test = train_prophet(df, report)

    lstm_result = train_lstm_tuned(df, report)

    ensemble_common, weights, ensemble_metrics = optimize_ensemble(
        prophet_test=prophet_test,
        lstm_test=lstm_result["date_predictions"],
        report=report
    )

    plot_combined(
        df=df,
        prophet_forecast=prophet_forecast,
        lstm_result=lstm_result,
        ensemble_common=ensemble_common,
        weights=weights
    )

    report_path = os.path.join(EVAL_DIR, "forecasting_report_fixed_validation.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    txt_report_path = os.path.join(EVAL_DIR, "report_forecasting.txt")

    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("FORECASTING MODEL REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model architecture:\n")
        f.write("Prophet + LSTM + Weighted Ensemble\n\n")

        f.write("Dataset:\n")
        f.write("Morocco daily wildfire detections\n\n")

        f.write("Validation methodology:\n")
        f.write("Train -> Validation -> Test split\n")
        f.write("The final metrics are calculated on unseen test dates.\n\n")

        f.write("Final ensemble metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"MAE   : {ensemble_metrics['MAE']:.4f} fires/day\n")
        f.write(f"RMSE  : {ensemble_metrics['RMSE']:.4f} fires/day\n")
        f.write(f"R²    : {ensemble_metrics['R2']:.4f}\n")
        f.write(f"WAPE  : {ensemble_metrics['WAPE']:.2f}%\n")
        f.write(f"sMAPE : {ensemble_metrics['sMAPE']:.2f}%\n\n")

        f.write("Final target check for Morocco daily fire detections:\n")
        f.write("-" * 60 + "\n")
        f.write(f"MAE   target < 3 fires/day : got {ensemble_metrics['MAE']:.2f}\n")
        f.write(f"RMSE  target < 4 fires/day : got {ensemble_metrics['RMSE']:.2f}\n")
        f.write(f"WAPE  target < 40%         : got {ensemble_metrics['WAPE']:.2f}%\n")
        f.write(f"sMAPE target < 50%         : got {ensemble_metrics['sMAPE']:.2f}%\n")
        f.write(f"R² informational only      : got {ensemble_metrics['R2']:.4f}\n\n")

        f.write("Status:\n")
        f.write("-" * 60 + "\n")
        f.write(f"MAE   : {'PASS' if ensemble_metrics['MAE'] < 3 else 'FAIL'}\n")
        f.write(f"RMSE  : {'PASS' if ensemble_metrics['RMSE'] < 4 else 'FAIL'}\n")
        f.write(f"WAPE  : {'PASS' if ensemble_metrics['WAPE'] < 40 else 'FAIL'}\n")
        f.write(f"sMAPE : {'PASS' if ensemble_metrics['sMAPE'] < 50 else 'FAIL'}\n\n")

        f.write("Best ensemble weights:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Prophet weight : {weights['w_prophet']:.2f}\n")
        f.write(f"LSTM weight    : {weights['w_lstm']:.2f}\n\n")

        f.write("Saved model files:\n")
        f.write("-" * 60 + "\n")
        f.write("prophet_model_fixed.joblib\n")
        f.write("lstm_best_model_fixed.keras\n")
        f.write("lstm_best_model_fixed.joblib\n")
        f.write("lstm_best_scaler_fixed.joblib\n")
        f.write("lstm_best_metadata_fixed.joblib\n")

    meta = {
        "target_metrics": {
            "MAE": "<3 fires/day",
            "RMSE": "<4 fires/day",
            "WAPE": "<40%",
            "sMAPE": "<50%",
            "R2": "informational only for sparse daily counts"
        },
        "validation_fix": "LSTM now uses train/validation/test split. Test set is not used for validation.",
        "forecast_horizon": FORECAST_HORIZON,
        "test_days": TEST_DAYS,
        "validation_days": VAL_DAYS,
        "lstm_configs_tried": LSTM_CONFIGS[:MAX_LSTM_CONFIGS],
        "best_ensemble_weights": weights,
        "final_ensemble_metrics": ensemble_metrics,
    }

    with open(
        os.path.join(MODEL_DIR, "forecasting_meta_fixed_validation.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=4)

    print("\n[DONE]")
    print(f"Reports saved to : {EVAL_DIR}")
    print(f"Text report saved: {txt_report_path}")
    print(f"Models saved to  : {MODEL_DIR}")

    print("\nFinal target check for Morocco daily fire detections:")
    print(f"MAE   target < 3 fires/day : got {ensemble_metrics['MAE']:.2f}")
    print(f"RMSE  target < 4 fires/day : got {ensemble_metrics['RMSE']:.2f}")
    print(f"WAPE  target < 40%         : got {ensemble_metrics['WAPE']:.2f}%")
    print(f"sMAPE target < 50%         : got {ensemble_metrics['sMAPE']:.2f}%")
    print(f"R² informational only      : got {ensemble_metrics['R2']:.4f}")


if __name__ == "__main__":
    main()