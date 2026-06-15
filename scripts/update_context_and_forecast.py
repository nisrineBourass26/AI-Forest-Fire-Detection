"""
scripts/update_context_and_forecast.py
----------------------------------------
Step 3 of the wildfire detection pipeline.

Forecasting Architecture
------------------------
Two complementary models produce the final 7-day global fire
activity forecast:

  Prophet  — captures long-term trend and yearly seasonality.
             Provides the baseline structural forecast.

  LSTM     — residual correction model. Trained on the difference
             between Prophet's predictions and actual values.
             Captures short-term patterns Prophet misses.

Final forecast formula (from training metadata):
  final = prophet_pred + alpha * clip(lstm_residual, -clip_val, clip_val)

  alpha = 0.03  (conservative blend — LSTM correction is small)
  clip  = 3.0   (prevents LSTM from over-correcting)

This ensemble is more accurate than Prophet alone, especially
for short-term fluctuations in fire activity.

Outputs
-------
  data/latest_forecast.csv
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import (
    FORECAST_CONTEXT_PATH,
    LATEST_CLEAN_FIRMS_PATH,
    LATEST_FORECAST_PATH,
    PROPHET_MODEL_PATH,
    LSTM_MODEL_PATH,
    FORECASTING_SCALER_PATH,
    FORECASTING_LSTM_META_PATH,
    FORECAST_HORIZON_DAYS,
    CONTEXT_DAYS_TO_KEEP,
)


# ============================================================
# HELPERS
# ============================================================

def _check_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _unwrap_prophet(obj):
    """Support both a bare Prophet object and {"model": Prophet} dict."""
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _load_lstm_model(path):
    """
    Load the LSTM .keras file, handling Keras version mismatches.

    The model was saved with Keras 3.x which includes 'quantization_config'
    in the Dense layer config. Older Keras versions don't recognise this
    parameter and raise a TypeError on load.

    Fix: load with compile=False and custom_objects that silently accept
    unknown keyword arguments in Dense layers.
    """
    # Strategy 1: try normal load first (works if versions match)
    try:
        from tensorflow.keras.models import load_model
        model = load_model(path, compile=False, safe_mode=False)
        print("  LSTM model loaded (standard load).")
        return model
    except Exception as e1:
        pass

    # Strategy 2: patch Dense to accept unknown kwargs, then load
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        class _PatchedDense(layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        model = tf.keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
            custom_objects={"Dense": _PatchedDense},
        )
        print("  LSTM model loaded (patched Dense — Keras version mismatch handled).")
        return model
    except Exception as e2:
        pass

    # Strategy 3: load weights only by rebuilding architecture from metadata
    try:
        import tensorflow as tf

        # Architecture is known from the error log:
        # Input(60, 37) -> LSTM(64) -> BatchNorm -> Dropout(0.2) -> Dense(32, relu) -> Dense(7)
        n_features = 37  # from build_config in error
        lookback   = 60

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(lookback, n_features)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(7,  activation="linear"),
        ])

        # Load only the weights (works across versions)
        model.load_weights(path)
        print("  LSTM model loaded (weights-only — architecture rebuilt from config).")
        return model
    except Exception as e3:
        raise RuntimeError(
            f"Could not load LSTM model from {path}.\n"
            f"  Attempt 1 (standard):       {e1}\n"
            f"  Attempt 2 (patched Dense):  {e2}\n"
            f"  Attempt 3 (weights-only):   {e3}\n"
            "Try upgrading Keras: pip install --upgrade keras tensorflow"
        )


# ============================================================
# STEP 1 — Load forecast context
# ============================================================

def load_forecast_context():
    _check_file(FORECAST_CONTEXT_PATH, "forecast_context.csv")

    context = pd.read_csv(FORECAST_CONTEXT_PATH, low_memory=False)

    required = ["ds", "y", "avg_frp", "max_frp", "avg_confidence", "night_fire_count"]
    missing  = [c for c in required if c not in context.columns]
    if missing:
        raise ValueError(f"forecast_context.csv missing columns: {missing}")

    context["ds"] = pd.to_datetime(context["ds"], errors="coerce")
    context = context.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)

    for col in ["y", "avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        context[col] = pd.to_numeric(context[col], errors="coerce").fillna(0)

    if context.empty:
        raise ValueError("forecast_context.csv is empty after cleaning.")

    print(f"  Context loaded: {len(context)} rows, "
          f"{context['ds'].min().date()} → {context['ds'].max().date()}")
    return context


# ============================================================
# STEP 2 — Optional live update from latest_clean_firms.csv
# ============================================================

def _aggregate_latest_firms():
    if not os.path.exists(LATEST_CLEAN_FIRMS_PATH):
        print("  latest_clean_firms.csv not found — skipping live update.")
        return pd.DataFrame()

    df = pd.read_csv(LATEST_CLEAN_FIRMS_PATH, low_memory=False)

    if "datetime" in df.columns:
        df["ds"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
    elif "acq_date" in df.columns:
        df["ds"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.normalize()
    else:
        raise ValueError("latest_clean_firms.csv needs 'datetime' or 'acq_date'.")

    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)

    if "confidence_num" in df.columns:
        df["conf"] = pd.to_numeric(df["confidence_num"], errors="coerce").fillna(0)
    elif "confidence" in df.columns:
        _map = {"l": 30, "low": 30, "n": 60, "nominal": 60, "medium": 60, "h": 90, "high": 90}
        raw  = df["confidence"].astype(str).str.lower().str.strip()
        df["conf"] = raw.map(_map).fillna(pd.to_numeric(raw, errors="coerce")).fillna(0)
    else:
        df["conf"] = 0

    if "is_night" in df.columns:
        df["night"] = pd.to_numeric(df["is_night"], errors="coerce").fillna(0)
    elif "daynight" in df.columns:
        df["night"] = df["daynight"].astype(str).str.upper().eq("N").astype(int)
    else:
        df["night"] = 0

    df = df.dropna(subset=["ds"])
    if df.empty:
        return pd.DataFrame()

    daily = (
        df.groupby("ds")
        .agg(y=("frp", "size"), avg_frp=("frp", "mean"),
             max_frp=("frp", "max"), avg_confidence=("conf", "mean"),
             night_fire_count=("night", "sum"))
        .reset_index()
    )
    daily["y"]                = daily["y"].astype(int)
    daily["night_fire_count"] = daily["night_fire_count"].astype(int)
    return daily


def update_context(context):
    new_daily = _aggregate_latest_firms()
    if new_daily.empty:
        return context

    context   = context.copy()
    new_daily["ds"] = pd.to_datetime(new_daily["ds"])
    context["ds"]   = pd.to_datetime(context["ds"])

    replace_dates = set(new_daily["ds"].dt.date)
    context = context[~context["ds"].dt.date.isin(replace_dates)]

    updated = (
        pd.concat([context, new_daily], ignore_index=True)
        .groupby("ds")
        .agg(y=("y", "sum"), avg_frp=("avg_frp", "mean"),
             max_frp=("max_frp", "max"), avg_confidence=("avg_confidence", "mean"),
             night_fire_count=("night_fire_count", "sum"))
        .reset_index()
        .sort_values("ds")
    )

    full_range = pd.date_range(updated["ds"].min(), updated["ds"].max(), freq="D")
    updated = updated.set_index("ds").reindex(full_range).reset_index().rename(columns={"index": "ds"})
    updated["y"]                = updated["y"].fillna(0).astype(int)
    updated["avg_frp"]          = updated["avg_frp"].fillna(0)
    updated["max_frp"]          = updated["max_frp"].fillna(0)
    updated["avg_confidence"]   = updated["avg_confidence"].fillna(0)
    updated["night_fire_count"] = updated["night_fire_count"].fillna(0).astype(int)

    updated = updated.tail(CONTEXT_DAYS_TO_KEEP).copy()
    updated.to_csv(FORECAST_CONTEXT_PATH, index=False)

    print(f"  Context updated: {len(updated)} rows, "
          f"{updated['ds'].min().date()} → {updated['ds'].max().date()}")
    return updated


# ============================================================
# STEP 3 — Prophet baseline forecast
# ============================================================

def run_prophet_forecast(context):
    _check_file(PROPHET_MODEL_PATH, "Prophet model")
    prophet_obj = joblib.load(PROPHET_MODEL_PATH)
    model       = _unwrap_prophet(prophet_obj)

    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object from {PROPHET_MODEL_PATH} has no predict() method.")

    last_date    = pd.to_datetime(context["ds"]).max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON_DAYS, freq="D",
    )
    future   = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).clip(lower=0)

    print(f"  Prophet forecast: {len(result)} days, "
          f"mean={result['yhat'].mean():.3f}, max={result['yhat'].max():.3f}")
    return result


# ============================================================
# STEP 4 — LSTM residual correction
# ============================================================

def _build_lstm_features(context, prophet_forecast, meta):
    """
    Reconstruct the feature matrix the LSTM expects, using the
    rolling context + Prophet forecast values.
    Returns a numpy array of shape (lookback, n_features).
    """
    feature_cols = meta["feature_cols"]
    lookback     = meta["lookback"]

    ctx = context.copy().tail(lookback + 60).reset_index(drop=True)
    ctx["ds"] = pd.to_datetime(ctx["ds"])

    # Log features
    for col_base in ["y", "avg_frp", "max_frp", "avg_confidence", "night_fire_count"]:
        if col_base in ctx.columns:
            ctx[f"{col_base}_log"] = np.log1p(ctx[col_base].clip(lower=0))

    ctx["prophet_pred_log"] = np.log1p(ctx["y"].clip(lower=0))

    # Residual
    ctx["residual"] = ctx["y_log"] - ctx["prophet_pred_log"]

    # Cyclic features
    ctx["day_of_year"] = ctx["ds"].dt.dayofyear
    ctx["day_sin"]     = np.sin(2 * np.pi * ctx["day_of_year"] / 365)
    ctx["day_cos"]     = np.cos(2 * np.pi * ctx["day_of_year"] / 365)
    ctx["month"]       = ctx["ds"].dt.month
    ctx["month_sin"]   = np.sin(2 * np.pi * ctx["month"] / 12)
    ctx["month_cos"]   = np.cos(2 * np.pi * ctx["month"] / 12)

    # Rolling stats
    ctx["residual_roll7_mean"]  = ctx["residual"].rolling(7,  min_periods=1).mean()
    ctx["residual_roll14_mean"] = ctx["residual"].rolling(14, min_periods=1).mean()
    ctx["residual_roll30_mean"] = ctx["residual"].rolling(30, min_periods=1).mean()
    ctx["residual_roll7_std"]   = ctx["residual"].rolling(7,  min_periods=1).std().fillna(0)
    ctx["residual_roll14_std"]  = ctx["residual"].rolling(14, min_periods=1).std().fillna(0)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        ctx[f"y_lag_{lag}"]        = ctx["y_log"].shift(lag)
        ctx[f"residual_lag_{lag}"] = ctx["residual"].shift(lag)
        ctx[f"prophet_lag_{lag}"]  = ctx["prophet_pred_log"].shift(lag)

    ctx = ctx.fillna(0)

    window = ctx.tail(lookback).copy()

    # Add any missing feature columns as zeros
    for f in feature_cols:
        if f not in window.columns:
            window[f] = 0.0

    return window[feature_cols].values


def run_lstm_correction(context, prophet_result, meta):
    """
    Load the LSTM model and compute residual corrections for all 7 days.
    Falls back to zero correction (Prophet-only) if loading fails.
    """
    _check_file(LSTM_MODEL_PATH,         "LSTM model")
    _check_file(FORECASTING_SCALER_PATH, "Forecasting scaler")

    try:
        lstm_model = _load_lstm_model(LSTM_MODEL_PATH)
    except Exception as e:
        print(f"\n  WARNING: LSTM model could not be loaded — using Prophet-only forecast.")
        print(f"  Reason: {e}")
        print("  Tip: pip install --upgrade keras tensorflow")
        return pd.Series(0.0, index=prophet_result["ds"])

    scaler       = joblib.load(FORECASTING_SCALER_PATH)
    lookback     = meta["lookback"]
    feature_cols = meta["feature_cols"]

    X_window = _build_lstm_features(context, prophet_result, meta)

    if len(X_window) < lookback:
        print(f"  WARNING: context too short ({len(X_window)} < {lookback}). "
              "Using Prophet-only forecast.")
        return pd.Series(0.0, index=prophet_result["ds"])

    X_scaled = scaler.transform(X_window)
    X_input  = X_scaled.reshape(1, lookback, len(feature_cols))

    raw_correction = lstm_model.predict(X_input, verbose=0)
    corrections    = raw_correction.flatten()

    if len(corrections) < FORECAST_HORIZON_DAYS:
        corrections = np.repeat(corrections[0], FORECAST_HORIZON_DAYS)

    corrections = corrections[:FORECAST_HORIZON_DAYS]

    print(f"  LSTM corrections: mean={corrections.mean():.4f}, std={corrections.std():.4f}")
    return pd.Series(corrections, index=prophet_result["ds"])


# ============================================================
# STEP 5 — Blend Prophet + LSTM → final forecast
# ============================================================

def blend_forecasts(prophet_result, lstm_corrections, meta):
    """
    Apply the ensemble formula from training:
      final = prophet + alpha * clip(lstm_residual, -clip_val, clip_val)
    """
    alpha    = meta.get("best_alpha", 0.03)
    clip_val = meta.get("best_clip",  3.0)

    result = prophet_result.copy()

    corrections_clipped        = lstm_corrections.clip(lower=-clip_val, upper=clip_val)
    result["lstm_correction"]  = corrections_clipped.values
    result["final_forecast"]   = (result["yhat"] + alpha * result["lstm_correction"]).clip(lower=0)
    result["prophet_forecast"] = result["yhat"]
    result["prophet_lower"]    = result["yhat_lower"]
    result["prophet_upper"]    = result["yhat_upper"]
    result["forecast_lower"]   = (result["prophet_lower"] + alpha * result["lstm_correction"]).clip(lower=0)
    result["forecast_upper"]   = (result["prophet_upper"] + alpha * result["lstm_correction"]).clip(lower=0)

    result["forecast_value"]        = result["final_forecast"].round(4)
    result["forecast_model"]        = "Prophet + LSTM ensemble"
    result["forecast_scope"]        = "Global"
    result["forecast_horizon_days"] = FORECAST_HORIZON_DAYS
    result["forecast_label"]        = "Global Fire Activity Forecast Index"
    result["ensemble_alpha"]        = alpha
    result["ensemble_clip"]         = clip_val
    result["generated_at"]          = pd.Timestamp.now()

    for col in ["forecast_value", "prophet_forecast", "forecast_lower", "forecast_upper"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").round(4)

    print(f"\n  Final ensemble forecast (alpha={alpha}, clip={clip_val}):")
    print(f"  Range: {result['forecast_value'].min():.3f} — {result['forecast_value'].max():.3f}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GLOBAL FIRE ACTIVITY FORECASTING — Prophet + LSTM Ensemble")
    print("=" * 70)

    _check_file(FORECASTING_LSTM_META_PATH, "LSTM metadata")
    with open(FORECASTING_LSTM_META_PATH, "rb") as f:
        meta = joblib.load(f)

    print(f"\n  LSTM metadata: lookback={meta['lookback']}, "
          f"horizon={meta['forecast_horizon']}, "
          f"alpha={meta['best_alpha']}, clip={meta['best_clip']}")

    print("\n[1] Loading forecast context...")
    context = load_forecast_context()

    print("\n[2] Updating context from latest NASA data...")
    context = update_context(context)

    print("\n[3] Running Prophet baseline forecast...")
    prophet_result = run_prophet_forecast(context)

    print("\n[4] Running LSTM residual correction...")
    lstm_corrections = run_lstm_correction(context, prophet_result, meta)

    print("\n[5] Computing ensemble forecast...")
    final = blend_forecasts(prophet_result, lstm_corrections, meta)

    final.to_csv(LATEST_FORECAST_PATH, index=False)
    print(f"\nForecast saved: {LATEST_FORECAST_PATH}")
    print("\nForecast preview:")
    print(final[["ds", "prophet_forecast", "lstm_correction",
                  "forecast_value", "forecast_lower", "forecast_upper"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()