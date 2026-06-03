"""
train_random_forest.py
Wildfire Risk Prediction — Random Forest
Classification + Regression

Important fixes:
1. Removed FRP leakage from features.
2. Fixed feature importance IndexError.
3. Used KFold for regression instead of StratifiedKFold.
"""

import os
import json
import warnings
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(ROOT_DIR, "data", "fires.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "ml", "saved_models")
EVAL_DIR = os.path.join(ROOT_DIR, "ml", "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ── 1. Load data ───────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from {path}")

    df = pd.read_csv(path, low_memory=False)

    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}\n")

    return df


# ── 2. Feature engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_col = next(
        (c for c in df.columns if "acq_date" in c.lower() or "date" in c.lower()),
        None
    )

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        df["day_of_year"] = df[date_col].dt.dayofyear
        df["month"] = df[date_col].dt.month
        df["week_of_year"] = df[date_col].dt.isocalendar().week.astype("Int64")
        df["year"] = df[date_col].dt.year

        df["season"] = df["month"].map({
            12: 0, 1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2,
            9: 3, 10: 3, 11: 3
        }).fillna(0).astype(int)

    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
        df["frp_log"] = np.log1p(df["frp"].clip(lower=0))

    if "scan" in df.columns:
        df["scan"] = pd.to_numeric(df["scan"], errors="coerce")

    if "track" in df.columns:
        df["track"] = pd.to_numeric(df["track"], errors="coerce")

    bright_cols = [c for c in df.columns if "bright" in c.lower()]

    for bc in bright_cols:
        df[bc] = pd.to_numeric(df[bc], errors="coerce")
        df[f"{bc}_anomaly"] = df[bc] - df[bc].mean()

    if "daynight" in df.columns:
        df["is_day"] = (
            df["daynight"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("D")
            .astype(int)
        )

    if "satellite" in df.columns:
        df["satellite_code"] = LabelEncoder().fit_transform(
            df["satellite"].astype(str)
        )

    if "confidence" in df.columns:
        conf = df["confidence"].astype(str).str.strip().str.lower()

        text_map = {
            "low": 30,
            "nominal": 60,
            "medium": 60,
            "high": 90
        }

        df["confidence_num"] = conf.map(text_map)

        mask = df["confidence_num"].isna()

        df.loc[mask, "confidence_num"] = pd.to_numeric(
            conf[mask],
            errors="coerce"
        )

        df["confidence_num"] = df["confidence_num"].fillna(
            df["confidence_num"].median()
        )

    return df


# ── 3. Build risk label ────────────────────────────────────────────────────────

def build_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    if "frp" not in df.columns:
        raise ValueError("Column 'frp' not found.")

    bins = [-np.inf, 10, 50, 150, np.inf]
    labels = [0, 1, 2, 3]

    df["risk_label"] = pd.cut(
        df["frp"],
        bins=bins,
        labels=labels
    ).astype(int)

    print("[INFO] Risk label distribution:")
    print(df["risk_label"].value_counts().sort_index().to_string(), "\n")

    return df


# ── 4. Feature selection ───────────────────────────────────────────────────────
# IMPORTANT:
# We removed frp_log and frp_per_km2 from features to avoid data leakage.

CANDIDATE_FEATURES = [
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


def select_features(df: pd.DataFrame) -> list:
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]

    print(f"[INFO] Features selected ({len(available)}): {available}\n")

    return available


# ── 5. Optuna tuning ───────────────────────────────────────────────────────────

def tune_rf_classifier(X_train, y_train, n_trials=20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None]
            ),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = RandomForestClassifier(**params)

        skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[INFO] Best classifier CV F1 macro: {study.best_value:.4f}")
    print(f"[INFO] Best params: {study.best_params}\n")

    return study.best_params


def tune_rf_regressor(X_train, y_train, n_trials=20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None]
            ),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = RandomForestRegressor(**params)

        kf = KFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[INFO] Best regressor CV RMSE: {-study.best_value:.4f}")
    print(f"[INFO] Best params: {study.best_params}\n")

    return study.best_params


# ── 6. Evaluation ──────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, features, report_lines):
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Moderate", "High", "Extreme"]
    )

    print("── Classification Report ──")
    print(report)

    report_lines.append("── Random Forest Classifier ──\n" + report)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=["Low", "Mod", "High", "Ext"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)

    ax.set_title("Random Forest — Confusion Matrix")

    plt.tight_layout()
    plt.savefig(
        os.path.join(EVAL_DIR, "rf_confusion_matrix.png"),
        dpi=150
    )
    plt.close()

    _plot_importance(
        model.feature_importances_,
        features,
        "rf_feature_importance_classifier.png",
        "Random Forest Classifier — Feature Importance"
    )


def evaluate_regressor(model, X_test, y_test, features, report_lines):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    lines = (
        "── Random Forest Regressor (log-FRP) ──\n"
        f"MAE  : {mae:.4f}\n"
        f"RMSE : {rmse:.4f}\n"
        f"R²   : {r2:.4f}\n"
    )

    print(lines)

    report_lines.append(lines)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        y_test[:2000],
        y_pred[:2000],
        alpha=0.3,
        s=8,
        color="#0F6E56"
    )

    lims = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max())
    ]

    ax.plot(lims, lims, "r--", linewidth=1)

    ax.set_xlabel("Actual log-FRP")
    ax.set_ylabel("Predicted log-FRP")
    ax.set_title("Random Forest Regressor — Actual vs Predicted")

    plt.tight_layout()
    plt.savefig(
        os.path.join(EVAL_DIR, "rf_regression_scatter.png"),
        dpi=150
    )
    plt.close()

    _plot_importance(
        model.feature_importances_,
        features,
        "rf_feature_importance_regressor.png",
        "Random Forest Regressor — Feature Importance"
    )


def _plot_importance(importances, features, filename, title):
    """
    Safe feature importance plot.
    Fixes:
    IndexError: list index out of range
    """

    n = min(len(importances), len(features))

    importances = np.array(importances[:n])
    feature_names = features[:n]

    idx = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="#0F6E56"
    )

    ax.set_xlabel("Importance score")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, filename), dpi=150)
    plt.close()


# ── 7. Main ────────────────────────────────────────────────────────────────────

def main():
    report_lines = []

    df = load_data(DATA_PATH)

    df = engineer_features(df)
    df = build_risk_label(df)

    features = select_features(df)

    target_clf = "risk_label"
    target_reg = "frp_log" if "frp_log" in df.columns else "frp"

    needed_columns = features + [target_clf, target_reg]

    df_clean = df[needed_columns].dropna()

    print(f"[INFO] Rows after dropping NaN: {len(df_clean)}\n")

    if len(df_clean) == 0:
        raise ValueError("No rows left after dropping NaN values.")

    X = df_clean[features].values
    y_clf = df_clean[target_clf].values.astype(int)
    y_reg = df_clean[target_reg].values.astype(float)

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X,
        y_clf,
        y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_clf
    )

    print(f"[INFO] Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}\n")

    # ── Classifier ─────────────────────────────────────────────────────────────

    print("=" * 60)
    print("STEP 1/4 — Tuning Random Forest Classifier")
    print("=" * 60)

    best_clf_params = tune_rf_classifier(
        X_tr,
        yc_tr,
        n_trials=20
    )

    best_clf_params.update({
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    })

    clf = RandomForestClassifier(**best_clf_params)

    clf.fit(X_tr, yc_tr)

    with open(os.path.join(MODEL_DIR, "rf_classifier.joblib"), "wb") as f:
        pickle.dump(clf, f)

    print("[INFO] RF Classifier saved.\n")

    print("STEP 2/4 — Evaluating Classifier")

    evaluate_classifier(
        clf,
        X_te,
        yc_te,
        features,
        report_lines
    )

    # ── Regressor ──────────────────────────────────────────────────────────────

    print("=" * 60)
    print("STEP 3/4 — Tuning Random Forest Regressor")
    print("=" * 60)

    best_reg_params = tune_rf_regressor(
        X_tr,
        yr_tr,
        n_trials=20
    )

    best_reg_params.update({
        "random_state": 42,
        "n_jobs": -1
    })

    reg = RandomForestRegressor(**best_reg_params)

    reg.fit(X_tr, yr_tr)

    with open(os.path.join(MODEL_DIR, "rf_regressor.joblib"), "wb") as f:
        pickle.dump(reg, f)

    print("[INFO] RF Regressor saved.\n")

    print("STEP 4/4 — Evaluating Regressor")

    evaluate_regressor(
        reg,
        X_te,
        yr_te,
        features,
        report_lines
    )

    # ── Save report and params ─────────────────────────────────────────────────

    report_path = os.path.join(EVAL_DIR, "rf_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))

    with open(
        os.path.join(MODEL_DIR, "rf_best_params.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "classifier": best_clf_params,
                "regressor": best_reg_params
            },
            f,
            indent=2
        )

    print(f"\n[DONE] Reports saved to {EVAL_DIR}")
    print(f"[DONE] Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()