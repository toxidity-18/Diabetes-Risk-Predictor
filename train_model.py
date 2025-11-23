import pandas as pd
import numpy as np
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# 1. Load dataset
df = pd.read_csv("diabetes.csv")

# 2. Handle zeros in some columns (treat as missing, replace with median)
cols_with_zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero_as_missing:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# 3. Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Define models
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# 7. Train models
log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)  # Random Forest does NOT need scaled input

# 8. Evaluate models
def evaluate_model(name, model, X_te, y_te, use_proba=True):
    """
    Returns metrics plus raw pieces needed for confusion matrix & curves.
    """
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)

    if use_proba and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_proba)
    else:
        y_proba = None
        auc = np.nan

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"AUC      : {auc:.3f}")

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "y_true": y_te,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

results = []

# Logistic Regression (uses scaled inputs)
results.append(
    evaluate_model(
        "Logistic Regression",
        log_reg,
        X_test_scaled,
        y_test,
        use_proba=True
    )
)

# Random Forest (uses unscaled inputs)
results.append(
    evaluate_model(
        "Random Forest",
        rf,
        X_test,
        y_test,
        use_proba=True
    )
)

# 9. Choose best model by AUC (fall back to accuracy if AUC is NaN)
def score_for_selection(r):
    # np.nan_to_num will replace NaN AUC with 0.0
    return (np.nan_to_num(r["auc"]), r["accuracy"])

best = max(results, key=score_for_selection)
print(f"\n>>> Best model selected: {best['name']}")

# 10. Save best model + scaler info
model_package = {
    "model_name": best["name"],
    "scaler": scaler,
    "feature_names": list(X.columns),
}

if best["name"] == "Logistic Regression":
    model_package["model"] = log_reg
    model_package["uses_scaled_input"] = True
else:
    model_package["model"] = rf
    model_package["uses_scaled_input"] = False

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("\nSaved best model to diabetes_model.pkl")

# 11. Save feature importances for Random Forest (for interpretability)
if best["name"] == "Random Forest":
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fi_df.to_csv("feature_importances.csv", index=False)
    print("Saved feature importances to feature_importances.csv")
else:
    print("Best model is Logistic Regression – no tree-based importances saved.")

# 12. Build detailed metrics for the best model
y_true_best = best["y_true"]
y_pred_best = best["y_pred"]
y_proba_best = best["y_proba"]

# Confusion matrix
cm = confusion_matrix(y_true_best, y_pred_best).tolist()

# ROC curve and PR curve (only if probabilities available)
roc_data = None
pr_data = None
if y_proba_best is not None:
    fpr, tpr, _ = roc_curve(y_true_best, y_proba_best)
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

    precision_arr, recall_arr, _ = precision_recall_curve(y_true_best, y_proba_best)
    pr_data = {
        "precision": precision_arr.tolist(),
        "recall": recall_arr.tolist()
    }

# 13. Save metrics for the best model to JSON
metrics = {
    "best_model": best["name"],
    "accuracy": best["accuracy"],
    "recall": best["recall"],
    "precision": best["precision"],
    "f1": best["f1"],
    "auc": float(best["auc"]) if not np.isnan(best["auc"]) else None,
    "confusion_matrix": cm,
    "roc_curve": roc_data,
    "pr_curve": pr_data,
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model performance metrics (including CM & curves) to model_metrics.json")
