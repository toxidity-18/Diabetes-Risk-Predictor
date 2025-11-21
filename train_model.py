# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. Load dataset
df = pd.read_csv("diabetes.csv")

# 2. Basic cleaning (optional but common for Pima dataset)
# Some columns contain zeros which are physiologically impossible
# (e.g. blood pressure 0). We can treat zeros as missing and impute.
cols_with_zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero_as_missing:
    df[col].replace(0, np.nan, inplace=True)
    df[col].fillna(df[col].median(), inplace=True)

# 3. Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scaling (Logistic Regression works better with scaled features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save model + scaler together
model_artifact = {
    "model": model,
    "scaler": scaler,
    "feature_names": list(X.columns)
}

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model_artifact, f)

print("Model saved to diabetes_model.pkl")
