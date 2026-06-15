"""
training/train_random_forest.py
================================
Wildfire Risk Prediction — Random Forest Classifier + Regressor

Same data decisions and leakage rules as train_xgboost.py.
See that file for full notes.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.neighbors import BallTree
from sklearn.utils.class_weight import compute_sample_weight

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

    before = len(df)
    df = df[df["type"] == 0].copy()
    print(f"[INFO] After type=0 filter: {len(df):,} (removed {before - len(df):,} non-vegetation)")

    df = df[df["frp"] > 0].copy()
    print(f"[INFO] After FRP>0 filter: {len(df):,}")

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"])
    print(f"[INFO] Date range: {df['acq_date'].min().date()} to {df['acq_date'].max().date()}")
    return df


# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    df["day_of_year"]  = df["acq_date"].dt.dayofyear
    df["month"]        = df["acq_date"].dt.month
    df["week_of_year"] = df["acq_date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["acq_date"].dt.year
    df["season"]       = df["month"].map({
        12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3
    }).fillna(0).astype(int)

    df["hour"]     = (df["acq_time"] // 100).clip(0, 23).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["is_day"] = df["daynight"].astype(str).str.upper().eq("D").astype(int)

    conf_map = {"l": 30, "n": 60, "h": 90}
    df["confidence_num"] = (
        df["confidence"].astype(str).str.lower().str.strip()
        .map(conf_map).fillna(60).astype(float)
    )

    df["brightness_anomaly"] = df["brightness"] - df["brightness"].mean()
    df["bright_t31_anomaly"] = df["bright_t31"] - df["bright_t31"].mean()

    df["scan"]  = pd.to_numeric(df["scan"],  errors="coerce").fillna(df["scan"].median())
    df["track"] = pd.to_numeric(df["track"], errors="coerce").fillna(df["track"].median())
    df["track_scan_ratio"] = (df["track"] / (df["scan"] + 1e-9)).clip(0, 5)

    df["frp_log"] = np.log1p(df["frp"].clip(lower=0))

    # Spatial fire density
    print("[INFO] Computing fire_density_50km ...")
    try:
        coords_rad = np.radians(df[["latitude", "longitude"]].values)
        tree   = BallTree(coords_rad, metric="haversine")
        counts = tree.query_radius(coords_rad, r=0.008, count_only=True)
        df["fire_density_50km"] = (counts - 1).clip(0)
    except Exception as e:
        print(f"[WARN] fire_density_50km failed ({e})")
        df["fire_density_50km"] = 0

    print("[INFO] Feature engineering done.")
    return df


# ── 3. RISK LABEL ──────────────────────────────────────────────────────────────

def build_risk_label(df):
    df["risk_label"] = pd.cut(
        df["frp"], bins=[-np.inf, 10, 50, 150, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)

    print("\n[INFO] Risk label distribution:")
    for k, name in enumerate(["Low", "Moderate", "High", "Extreme"]):
        n = (df["risk_label"] == k).sum()
        print(f"  {name:10s}: {n:7,d} ({n/len(df)*100:.1f}%)")
    return df


# ── 4. FEATURES ────────────────────────────────────────────────────────────────

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
        print(f"[WARN] Missing: {missing}")
    print(f"\n[INFO] Features ({len(available)}): {available}")
    return available


# ── 5. TEMPORAL SPLIT ──────────────────────────────────────────────────────────

def temporal_split(df):
    dates      = sorted(df["acq_date"].unique())
    split_date = dates[-2]
    train = df[df["acq_date"] < split_date].copy()
    test  = df[df["acq_date"] >= split_date].copy()
    print(f"\n[INFO] Train: {train['acq_date'].min().date()} to {train['acq_date'].max().date()} ({len(train):,})")
    print(f"[INFO] Test : {test['acq_date'].min().date()} to {test['acq_date'].max().date()}  ({len(test):,})")
    return train, test


# ── 6. OPTUNA TUNING ───────────────────────────────────────────────────────────

def tune_classifier(X_tr, y_tr, n_trials=30):
    def objective(trial):
        p = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(RandomForestClassifier(**p), X_tr, y_tr,
                               cv=skf, scoring="f1_macro", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[INFO] Best CV F1 macro: {study.best_value:.4f}")
    return study.best_params


def tune_regressor(X_tr, y_tr, n_trials=30):
    def objective(trial):
        p = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42, "n_jobs": -1,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(RandomForestRegressor(**p), X_tr, y_tr,
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
    print("\n── RF Classifier — Test Set ──\n" + rep)
    lines.append("── RF Classifier — Test Set ──\n" + rep)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="f1_macro", n_jobs=-1)
    cv_text = (
        f"\nRF Classifier — 5-Fold CV F1 (macro):\n"
        f"  Per fold  : {[f'{s:.4f}' for s in cv]}\n"
        f"  Mean ± Std: {cv.mean():.4f} ± {cv.std():.4f}\n"
    )
    print(cv_text)
    lines.append(cv_text)

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Low","Mod","High","Ext"]).plot(ax=ax, colorbar=False)
    ax.set_title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "rf_confusion_matrix.png"), dpi=150)
    plt.close()

    _plot_importance(model.feature_importances_, features,
                     "rf_importance_classifier.png", "RF Classifier — Feature Importance")


def evaluate_regressor(model, X_tr, y_tr, X_te, y_te, features, lines):
    y_pred = model.predict(X_te)

    mae_l  = mean_absolute_error(y_te, y_pred)
    rmse_l = np.sqrt(mean_squared_error(y_te, y_pred))
    r2_l   = r2_score(y_te, y_pred)

    y_raw  = np.expm1(y_te)
    p_raw  = np.expm1(y_pred)
    mae_r  = mean_absolute_error(y_raw, p_raw)
    rmse_r = np.sqrt(mean_squared_error(y_raw, p_raw))
    r2_r   = r2_score(y_raw, p_raw)

    text = (
        "\n── RF Regressor — Test Set ──\n"
        "\n[log(FRP) scale]\n"
        f"  MAE  : {mae_l:.4f}\n  RMSE : {rmse_l:.4f}\n  R²   : {r2_l:.4f}\n"
        "\n[Raw FRP (MW)]\n"
        f"  MAE  : {mae_r:.2f} MW\n  RMSE : {rmse_r:.2f} MW\n  R²   : {r2_r:.4f}\n"
        "\nNote: Raw-scale MAE is the meaningful metric for real-world interpretation.\n"
    )
    print(text)
    lines.append(text)

    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=-1)
    cv_text = (
        f"\nRF Regressor — 5-Fold CV R² (log-FRP):\n"
        f"  Per fold  : {[f'{s:.4f}' for s in cv]}\n"
        f"  Mean ± Std: {cv.mean():.4f} ± {cv.std():.4f}\n"
    )
    print(cv_text)
    lines.append(cv_text)

    idx = np.random.choice(len(y_te), min(5000, len(y_te)), replace=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te[idx], y_pred[idx], alpha=0.3, s=5, color="#0F6E56")
    lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Actual log(FRP)")
    ax.set_ylabel("Predicted log(FRP)")
    ax.set_title("RF Regressor — Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "rf_regression_scatter.png"), dpi=150)
    plt.close()

    _plot_importance(model.feature_importances_, features,
                     "rf_importance_regressor.png", "RF Regressor — Feature Importance")


def _plot_importance(importances, features, fname, title):
    n   = min(len(importances), len(features))
    imp = np.array(importances[:n])
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([features[i] for i in idx], imp[idx], color="#0F6E56")
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
    print("STEP 1/4 — Tuning RF Classifier")
    print("="*60)

    clf_p = tune_classifier(X_tr, yc_tr)
    clf_p.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1})

    clf = RandomForestClassifier(**clf_p)
    clf.fit(X_tr, yc_tr)

    with open(os.path.join(MODEL_DIR, "rf_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print("\nSTEP 2/4 — Evaluating Classifier")
    evaluate_classifier(clf, X_tr, yc_tr, X_te, yc_te, features, lines)

    # ── REGRESSOR ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3/4 — Tuning RF Regressor")
    print("="*60)

    reg_p = tune_regressor(X_tr, yr_tr)
    reg_p.update({"random_state": 42, "n_jobs": -1})

    reg = RandomForestRegressor(**reg_p)
    reg.fit(X_tr, yr_tr)

    with open(os.path.join(MODEL_DIR, "rf_regressor.pkl"), "wb") as f:
        pickle.dump(reg, f)

    print("\nSTEP 4/4 — Evaluating Regressor")
    evaluate_regressor(reg, X_tr, yr_tr, X_te, yr_te, features, lines)

    # ── SAVE ──────────────────────────────────────────────────
    with open(os.path.join(EVAL_DIR, "rf_report.txt"), "w") as f:
        f.write("\n\n".join(lines))

    with open(os.path.join(MODEL_DIR, "rf_best_params.json"), "w") as f:
        json.dump({"classifier": clf_p, "regressor": reg_p}, f, indent=2)

    print(f"\n[DONE] Models  → {MODEL_DIR}")
    print(f"[DONE] Reports → {EVAL_DIR}")


if __name__ == "__main__":
    main()