"""
training/train_xgboost.py
=========================
Wildfire Risk Prediction — XGBoost Classifier + Regressor

DATA NOTES (fire_archive_SV-C2_762343.csv):
  - 441,297 VIIRS SNPP detections, 7 days (2025-01-05 to 2025-01-11)
  - Only type=0 kept (vegetation fires). Types 1/2/3 are volcanoes/industrial/offshore.
  - Severe class imbalance: 81% Low, 17% Moderate, 1.4% High, 0.1% Extreme
  - Single satellite (SNPP) — satellite/instrument columns are zero-variance, excluded
  - version column is always 2 — excluded

LEAKAGE DECISIONS:
  - frp_per_km2 (frp / scan*track) is EXCLUDED — 0.94 correlation with FRP = pure leakage
  - brightness and bright_t31 are INCLUDED — thermally sensitive but NOT derived from FRP
  - frp_log is the regression TARGET, not a feature

SPLIT STRATEGY:
  - Temporal split by date (not random) — preserves time structure
  - Last 2 days = test, first 5 days = train
  - This is the correct approach for satellite time-series data

EVALUATION:
  - Classifier: F1 macro + per-class report + confusion matrix + 5-fold stratified CV
  - Regressor: metrics on BOTH log-FRP and raw FRP scale + 5-fold CV
  - Feature importance plots saved for both models
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.neighbors import BallTree
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier, XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")


# ── PATHS ──────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "Data", "fire_archive_SV-C2_762343.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
EVAL_DIR  = os.path.join(ROOT_DIR, "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)


# ── 1. LOAD + FILTER ───────────────────────────────────────────────────────────

def load_data(path):
    print(f"[INFO] Loading {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Raw shape: {df.shape}")

    # Keep only real vegetation fires
    # type=0: vegetation  |  type=1: volcano  |  type=2: industrial  |  type=3: offshore
    before = len(df)
    df = df[df["type"] == 0].copy()
    print(f"[INFO] After type=0 filter: {len(df):,} rows (removed {before - len(df):,})")

    # Remove physically impossible FRP=0 detections
    df = df[df["frp"] > 0].copy()
    print(f"[INFO] After FRP>0 filter: {len(df):,} rows")

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"])

    print(f"[INFO] Date range: {df['acq_date'].min().date()} to {df['acq_date'].max().date()}")
    return df


# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    # Temporal
    df["day_of_year"]  = df["acq_date"].dt.dayofyear
    df["month"]        = df["acq_date"].dt.month
    df["week_of_year"] = df["acq_date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["acq_date"].dt.year
    df["season"]       = df["month"].map({
        12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3
    }).fillna(0).astype(int)

    # Hour from HHMM integer
    df["hour"]     = (df["acq_time"] // 100).clip(0, 23).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Day/night
    df["is_day"] = df["daynight"].astype(str).str.upper().eq("D").astype(int)

    # Confidence
    conf_map = {"l": 30, "n": 60, "h": 90}
    df["confidence_num"] = (
        df["confidence"].astype(str).str.lower().str.strip()
        .map(conf_map).fillna(60).astype(float)
    )

    # Brightness anomaly (safe — not derived from FRP)
    df["brightness_anomaly"] = df["brightness"] - df["brightness"].mean()
    df["bright_t31_anomaly"] = df["bright_t31"] - df["bright_t31"].mean()

    # Pixel geometry ratio (scan/track measures pixel shape at swath edge — sensor artifact)
    df["scan"]  = pd.to_numeric(df["scan"],  errors="coerce").fillna(df["scan"].median())
    df["track"] = pd.to_numeric(df["track"], errors="coerce").fillna(df["track"].median())
    df["track_scan_ratio"] = (df["track"] / (df["scan"] + 1e-9)).clip(0, 5)

    # FRP target (NOT a feature — only used as y)
    df["frp_log"] = np.log1p(df["frp"].clip(lower=0))

    # Spatial fire density (~50km neighborhood count)
    # This is NOT leakage: it uses spatial coordinates, not FRP values.
    # A fire in a dense cluster is fundamentally different from an isolated detection.
    print("[INFO] Computing fire_density_50km ...")
    try:
        coords_rad = np.radians(df[["latitude", "longitude"]].values)
        tree   = BallTree(coords_rad, metric="haversine")
        counts = tree.query_radius(coords_rad, r=0.008, count_only=True)  # 0.008 rad ~ 50 km
        df["fire_density_50km"] = (counts - 1).clip(0)
    except Exception as e:
        print(f"[WARN] fire_density_50km failed ({e}), setting to 0")
        df["fire_density_50km"] = 0

    print("[INFO] Feature engineering done.")
    return df


# ── 3. RISK LABEL ──────────────────────────────────────────────────────────────

def build_risk_label(df):
    """
    4-class label from FRP thresholds.

    Brightness correlates with FRP (~0.35) but is a raw sensor band,
    not derived from FRP — so using it as a feature is acceptable.
    frp_per_km2 (scan*track) has 0.94 correlation = leakage — excluded.
    """
    df["risk_label"] = pd.cut(
        df["frp"],
        bins=[-np.inf, 10, 50, 150, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    print("\n[INFO] Risk label distribution:")
    for k, name in enumerate(["Low", "Moderate", "High", "Extreme"]):
        n = (df["risk_label"] == k).sum()
        print(f"  {name:10s}: {n:7,d} ({n/len(df)*100:.1f}%)")
    return df


# ── 4. FEATURE LIST ────────────────────────────────────────────────────────────

FEATURES = [
    "latitude", "longitude",
    "brightness", "bright_t31",
    "brightness_anomaly", "bright_t31_anomaly",
    "scan", "track", "track_scan_ratio",
    "confidence_num", "is_day",
    "hour_sin", "hour_cos",
    "day_of_year", "month", "week_of_year", "year", "season",
    "fire_density_50km",
]


def select_features(df):
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features (will be skipped): {missing}")
    print(f"\n[INFO] Features used ({len(available)}): {available}")
    return available


# ── 5. TEMPORAL SPLIT ──────────────────────────────────────────────────────────

def temporal_split(df):
    """
    Split by date — correct for satellite time-series data.
    Random split would leak future observations into training.
    Last 2 of 7 days = test.
    """
    dates      = sorted(df["acq_date"].unique())
    split_date = dates[-2]
    train = df[df["acq_date"] < split_date].copy()
    test  = df[df["acq_date"] >= split_date].copy()
    print(f"\n[INFO] Train: {train['acq_date'].min().date()} to {train['acq_date'].max().date()} ({len(train):,} rows)")
    print(f"[INFO] Test : {test['acq_date'].min().date()} to {test['acq_date'].max().date()}  ({len(test):,} rows)")
    return train, test


# ── 6. OPTUNA TUNING ───────────────────────────────────────────────────────────

def tune_classifier(X_tr, y_tr, n_trials=40):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "eval_metric": "mlogloss", "tree_method": "hist",
            "random_state": 42, "n_jobs": -1,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(XGBClassifier(**params), X_tr, y_tr,
                               cv=skf, scoring="f1_macro", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[INFO] Best CV F1 macro: {study.best_value:.4f}")
    return study.best_params


def tune_regressor(X_tr, y_tr, n_trials=40):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method": "hist", "random_state": 42, "n_jobs": -1,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(XGBRegressor(**params), X_tr, y_tr,
                               cv=kf, scoring="r2", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[INFO] Best CV R²: {study.best_value:.4f}")
    return study.best_params


# ── 7. EVALUATION ──────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_tr, y_tr, X_te, y_te, features, lines):
    y_pred = model.predict(X_te)
    names  = ["Low", "Moderate", "High", "Extreme"]

    rep = classification_report(y_te, y_pred, target_names=names, zero_division=0)
    print("\n── XGBoost Classifier — Test Set ──\n" + rep)
    lines.append("── XGBoost Classifier — Test Set ──\n" + rep)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="f1_macro", n_jobs=-1)
    cv_text = (
        f"\nXGBoost Classifier — 5-Fold CV F1 (macro):\n"
        f"  Per fold : {[f'{s:.4f}' for s in cv]}\n"
        f"  Mean ± Std: {cv.mean():.4f} ± {cv.std():.4f}\n"
    )
    print(cv_text)
    lines.append(cv_text)

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Low","Mod","High","Ext"]).plot(ax=ax, colorbar=False)
    ax.set_title("XGBoost — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "xgb_confusion_matrix.png"), dpi=150)
    plt.close()

    _plot_importance(model.feature_importances_, features,
                     "xgb_importance_classifier.png", "XGBoost Classifier — Feature Importance")


def evaluate_regressor(model, X_tr, y_tr, X_te, y_te, features, lines):
    y_pred = model.predict(X_te)

    # Log-scale
    mae_l  = mean_absolute_error(y_te, y_pred)
    rmse_l = np.sqrt(mean_squared_error(y_te, y_pred))
    r2_l   = r2_score(y_te, y_pred)

    # Raw MW scale
    y_raw  = np.expm1(y_te)
    p_raw  = np.expm1(y_pred)
    mae_r  = mean_absolute_error(y_raw, p_raw)
    rmse_r = np.sqrt(mean_squared_error(y_raw, p_raw))
    r2_r   = r2_score(y_raw, p_raw)

    text = (
        "\n── XGBoost Regressor — Test Set ──\n"
        "\n[log(FRP) scale — training scale]\n"
        f"  MAE  : {mae_l:.4f}\n  RMSE : {rmse_l:.4f}\n  R²   : {r2_l:.4f}\n"
        "\n[Raw FRP (MW) — real-world interpretation]\n"
        f"  MAE  : {mae_r:.2f} MW\n  RMSE : {rmse_r:.2f} MW\n  R²   : {r2_r:.4f}\n"
        "\nNote: High R² on log scale is expected — log1p compresses the range.\n"
        "Raw-scale MAE is the meaningful metric: average error in megawatts.\n"
    )
    print(text)
    lines.append(text)

    # 5-fold CV
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=-1)
    cv_text = (
        f"\nXGBoost Regressor — 5-Fold CV R² (log-FRP):\n"
        f"  Per fold : {[f'{s:.4f}' for s in cv]}\n"
        f"  Mean ± Std: {cv.mean():.4f} ± {cv.std():.4f}\n"
    )
    print(cv_text)
    lines.append(cv_text)

    # Scatter
    idx = np.random.choice(len(y_te), min(5000, len(y_te)), replace=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te[idx], y_pred[idx], alpha=0.3, s=5, color="#BA7517")
    lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Actual log(FRP)")
    ax.set_ylabel("Predicted log(FRP)")
    ax.set_title("XGBoost Regressor — Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "xgb_regression_scatter.png"), dpi=150)
    plt.close()

    _plot_importance(model.feature_importances_, features,
                     "xgb_importance_regressor.png", "XGBoost Regressor — Feature Importance")


def _plot_importance(importances, features, fname, title):
    n   = min(len(importances), len(features))
    imp = np.array(importances[:n])
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([features[i] for i in idx], imp[idx], color="#185FA5")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, fname), dpi=150)
    plt.close()


# ── 8. MAIN ────────────────────────────────────────────────────────────────────

def main():
    lines = []

    df = load_data(DATA_PATH)
    df = engineer_features(df)
    df = build_risk_label(df)

    features     = select_features(df)
    train, test  = temporal_split(df)

    needed = features + ["risk_label", "frp_log"]
    train  = train[needed].dropna()
    test   = test[needed].dropna()

    X_tr, X_te   = train[features].values, test[features].values
    yc_tr, yc_te = train["risk_label"].values.astype(int), test["risk_label"].values.astype(int)
    yr_tr, yr_te = train["frp_log"].values.astype(float),  test["frp_log"].values.astype(float)

    # ── CLASSIFIER ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1/4 — Tuning XGBoost Classifier")
    print("="*60)

    clf_p = tune_classifier(X_tr, yc_tr)
    clf_p.update({"eval_metric":"mlogloss","tree_method":"hist","random_state":42,"n_jobs":-1})

    # Sample weights for class imbalance
    sw  = compute_sample_weight("balanced", yc_tr)
    clf = XGBClassifier(**clf_p)
    clf.fit(X_tr, yc_tr, sample_weight=sw, eval_set=[(X_te, yc_te)], verbose=False)
    clf.save_model(os.path.join(MODEL_DIR, "xgboost_classifier.json"))

    print("\nSTEP 2/4 — Evaluating Classifier")
    evaluate_classifier(clf, X_tr, yc_tr, X_te, yc_te, features, lines)

    # ── REGRESSOR ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3/4 — Tuning XGBoost Regressor")
    print("="*60)

    reg_p = tune_regressor(X_tr, yr_tr)
    reg_p.update({"tree_method":"hist","random_state":42,"n_jobs":-1})

    reg = XGBRegressor(**reg_p)
    reg.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
    reg.save_model(os.path.join(MODEL_DIR, "xgboost_regressor.json"))

    print("\nSTEP 4/4 — Evaluating Regressor")
    evaluate_regressor(reg, X_tr, yr_tr, X_te, yr_te, features, lines)

    # ── SAVE ──────────────────────────────────────────────────
    with open(os.path.join(EVAL_DIR, "xgboost_report.txt"), "w") as f:
        f.write("\n\n".join(lines))

    with open(os.path.join(MODEL_DIR, "xgboost_best_params.json"), "w") as f:
        json.dump({"classifier": clf_p, "regressor": reg_p}, f, indent=2)

    print(f"\n[DONE] Models  → {MODEL_DIR}")
    print(f"[DONE] Reports → {EVAL_DIR}")


if __name__ == "__main__":
    main()