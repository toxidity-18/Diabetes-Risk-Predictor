
## 1. Proper virtual environment & dependencies

* Created a **virtual environment** for the project:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
* Installed the required libraries inside the venv:

  ```bash
  pip install streamlit pandas scikit-learn numpy
  ```
* This fixed the `ModuleNotFoundError: No module named 'pandas'` and keeps your project clean.

---

## 2. Training script upgrade (`train_model.py`)

We turned `train_model.py` from a simple single-model trainer into a **model comparison + metrics exporter**:

### a) Data cleaning

* Replaced **0 values** in medically impossible columns with `NaN` then filled them with **median**:

  * `Glucose`
  * `BloodPressure`
  * `SkinThickness`
  * `Insulin`
  * `BMI`

### b) Multiple models and comparison

* Train **two models**:

  * `LogisticRegression` (on **scaled** features)
  * `RandomForestClassifier` (on **unscaled** features)
* Evaluate both using:

  * Accuracy
  * Precision
  * Recall
  * AUC

### c) Best model selection logic

* Automatically choose the **best model by AUC**, with accuracy as backup.
* Save a **model package** that includes:

  * `model` (best model object)
  * `scaler` (for Logistic Regression; still saved even if RF wins)
  * `feature_names`
  * `uses_scaled_input` (True/False)
  * `model_name` (string label for the app)

### d) Extra artifacts for interpretability & reporting

* If **Random Forest** is best:

  * Save `feature_importances.csv` with feature names and importance scores.
* Save `model_metrics.json`:

  ```json
  {
    "best_model": "...",
    "accuracy": ...,
    "recall": ...,
    "precision": ...,
    "f1": ...
  }
  ```

  so the app can display proper performance statistics.

---

## 3. Streamlit app upgrades (`app.py`)

We did a pretty big glow-up here.

### a) Fixed scaling & model compatibility

* Load the new **model package**:

  * `model`
  * `scaler`
  * `feature_names`
  * `uses_scaled_input`
  * `model_name`
* At prediction time:

  * If `uses_scaled_input` → use `scaler.transform(input_data)`
  * If not → use raw input (and wrap in a `DataFrame` with `feature_names` to avoid the `UserWarning` about feature names).

### b) Added a proper **Reset Inputs** button

* Introduced `st.session_state["reset_key"]`.
* All sidebar widgets use keys that include `reset_key`, e.g. `key=f"bmi_{key_suffix}"`.
* A **Reset Inputs** button:

  * Increments `reset_key`
  * Calls `st.rerun()` (we replaced deprecated `st.experimental_rerun()`).
* This means all inputs snap back to default values without hacks.

### c) Removed infinite rerun bug

* You had:

  ```python
  if "reset_key" not in st.session_state:
      st.session_state["reset_key"] = 0
      st.rerun()
  ```

  That causes a rerun loop.
* We changed it to:

  ```python
  if "reset_key" not in st.session_state:
      st.session_state["reset_key"] = 0
  ```

  No rerun there — only when pressing the reset button.

### d) Better risk prediction output

* Use `predict_proba` when available to get a probability of diabetes (class 1).
* Map probability → **risk levels**:

  * `< 0.3` → Low
  * `0.3 – 0.6` → Medium
  * `> 0.6` → High
* Show:

  * Prediction message
  * Probability (formatted to 2 decimal places)
  * Risk level
  * Risk-specific educational guidance

### e) Local anonymous usage logging

* Every prediction is logged to `usage_log.csv` with:

  * timestamp
  * input values
  * prediction (0/1)
  * probability
  * risk_level
* If the file exists → append; otherwise → create.
* This enables simple analytics and doesn’t store names/IDs.

---

## 4. UI structure and new tabs

The app now has **three tabs**:

### 🔮 Tab 1: Risk Prediction

* Sidebar inputs for:

  * Pregnancies
  * Glucose
  * BloodPressure
  * SkinThickness
  * Insulin
  * BMI
  * DiabetesPedigreeFunction
  * Age
* Shows:

  * A summary of *your input*.
  * Warnings if values are outside typical ranges.
  * Predict / Reset buttons.
  * Prediction + probability + risk level.
  * General health tips.
  * Strong disclaimer that this is **not medical advice**.

### ℹ️ Tab 2: About & SDG 3

* Project overview.
* Clear link to **SDG 3: Good Health & Well-being**.
* Explains:

  * Model type (`model_name`).
  * Features used.
  * Offline nature of the tool.
* Disclaimer about non-clinical use.

### 📈 Tab 3: Usage Stats

* Reads `usage_log.csv` (if present).
* Shows:

  * Total predictions made.
  * Average predicted probability.
  * Bar chart of **Risk Level Distribution** (Low / Medium / High).
  * Table of the **last 10 predictions**.
  * **Download button** to export the usage log as CSV.
* If `feature_importances.csv` exists:

  * Bar chart of feature importance for the Random Forest.
* If `model_metrics.json` exists:

  * Shows:

    * Best model
    * Accuracy
    * Recall
    * Precision
    * F1-score

---

## 5. Warnings / errors we already fixed

* `ModuleNotFoundError: No module named 'pandas'` → fixed via venv + install.
* `AttributeError: module 'streamlit' has no attribute 'experimental_rerun'`
  → replaced with `st.rerun()`.
* `UserWarning: X does not have valid feature names`
  → handled by wrapping numpy array into `pd.DataFrame(..., columns=feature_names)` when using RandomForest.

.
