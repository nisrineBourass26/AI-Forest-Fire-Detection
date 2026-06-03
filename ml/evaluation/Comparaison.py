# ==============================
# Random Forest vs XGBoost Evaluation
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


# 1. Load your dataset
df = pd.read_csv("../../data/fires.csv")   # change file name


# 2. Separate features and target
# Change "target" to the real name of your label column
X = df.drop(columns=["target"])
y = df["target"]


# 3. Train/test split
# Use stratify because fire datasets are usually imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Define models
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)


models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# 5. Train and evaluate
results = []

for name, model in models.items():
    print(f"\n==============================")
    print(f"Training {name}")
    print(f"==============================")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # For ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    })

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


# 6. Results table
results_df = pd.DataFrame(results)
print("\n\nFinal Comparison:")
print(results_df)


# 7. Save results
results_df.to_csv("model_comparison_results.csv", index=False)


# 8. Save the best model based on F1-score
best_model_name = results_df.sort_values(by="F1-score", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

joblib.dump(best_model, "best_fire_risk_model.pkl")

print(f"\nBest model saved: {best_model_name}")