import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import json
from datetime import datetime

import matplotlib.pyplot as plt  # for confusion matrix & curves
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor - SDG 3",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Session state for reset
# -----------------------------
if "reset_key" not in st.session_state:
    st.session_state["reset_key"] = 0

# -----------------------------
# Load trained model
# -----------------------------
with open("diabetes_model.pkl", "rb") as f:
    model_artifact = pickle.load(f)

model = model_artifact["model"]
scaler = model_artifact["scaler"]
feature_names = model_artifact["feature_names"]

# new fields for upgraded training script (with defaults for backward-compat)
uses_scaled_input = model_artifact.get("uses_scaled_input", True)
model_name = model_artifact.get("model_name", "Logistic Regression")

# -----------------------------
# App title
# -----------------------------
st.title("🩺 Diabetes Risk Predictor")
st.write("""
This app uses a machine learning model to estimate the **risk of type 2 diabetes**
based on basic health parameters.

It supports **SDG 3: Good Health & Well-being** by promoting awareness of
non-communicable diseases and encouraging early check-ups.
""")

# Create tabs: Prediction, About, Usage Stats
tab_predict, tab_about, tab_stats = st.tabs(
    ["🔮 Risk Prediction", "ℹ️ About & SDG 3", "📈 Usage Stats"]
)

# ======================================================
# TAB 1: Prediction
# ======================================================
with tab_predict:
    st.header("Enter Your Health Information")

    st.sidebar.header("Input Health Parameters")

    def user_input_features():
        """
        Use reset_key in widget keys so that when it changes,
        Streamlit recreates the inputs with default values.
        """
        key_suffix = st.session_state["reset_key"]

        Pregnancies = st.sidebar.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1, key=f"preg_{key_suffix}"
        )
        Glucose = st.sidebar.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=300, value=120, key=f"glu_{key_suffix}"
        )
        BloodPressure = st.sidebar.number_input(
            "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, key=f"bp_{key_suffix}"
        )
        SkinThickness = st.sidebar.number_input(
            "Skin Thickness (mm)", min_value=0, max_value=100, value=20, key=f"skin_{key_suffix}"
        )
        Insulin = st.sidebar.number_input(
            "Insulin (mu U/ml)", min_value=0, max_value=900, value=80, key=f"ins_{key_suffix}"
        )
        BMI = st.sidebar.number_input(
            "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, key=f"bmi_{key_suffix}"
        )
        DiabetesPedigreeFunction = st.sidebar.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            key=f"dpf_{key_suffix}"
        )
        Age = st.sidebar.number_input(
            "Age (years)", min_value=1, max_value=120, value=30, key=f"age_{key_suffix}"
        )

        data = np.array([
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ], dtype=float).reshape(1, -1)

        return data

    input_data = user_input_features()

    # Show user input
    st.subheader("Your Input")
    st.write({
        "Pregnancies": int(input_data[0, 0]),
        "Glucose": float(input_data[0, 1]),
        "BloodPressure": float(input_data[0, 2]),
        "SkinThickness": float(input_data[0, 3]),
        "Insulin": float(input_data[0, 4]),
        "BMI": float(input_data[0, 5]),
        "DiabetesPedigreeFunction": float(input_data[0, 6]),
        "Age": int(input_data[0, 7]),
    })

    # Basic input validation / sanity checks
    warnings = []
    if input_data[0, 1] < 50 or input_data[0, 1] > 250:
        warnings.append("Glucose value is outside a typical range (50–250 mg/dL).")
    if input_data[0, 2] < 40 or input_data[0, 2] > 180:
        warnings.append("Blood Pressure value is outside a typical range (40–180 mm Hg).")
    if input_data[0, 5] < 10 or input_data[0, 5] > 60:
        warnings.append("BMI value is outside a typical range (10–60 kg/m²).")
    if input_data[0, 7] < 10 or input_data[0, 7] > 100:
        warnings.append("Age is outside a typical adult range (10–100 years).")

    if warnings:
        st.warning("Please double-check your inputs:\n\n- " + "\n- ".join(warnings))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        predict_clicked = st.button("Predict Risk")
    with col2:
        reset_clicked = st.button("Reset Inputs")

    # Reset button logic
    if reset_clicked:
        st.session_state["reset_key"] += 1
        st.rerun()

    # Prediction button logic
    if predict_clicked:
        # Decide whether to scale input based on the saved model package
        if uses_scaled_input:
            model_input = scaler.transform(input_data)
        else:
            model_input = input_data

        # Optional: wrap in DataFrame for models trained with feature names (Random Forest)
        if not uses_scaled_input:
            model_input = pd.DataFrame(model_input, columns=feature_names)

        prediction = model.predict(model_input)[0]

        # probability of diabetes (class 1) if available
        if hasattr(model, "predict_proba"):
            prediction_proba = float(model.predict_proba(model_input)[0][1])
        else:
            # fallback: use 0/1 prediction as a rough probability
            prediction_proba = float(prediction)

        # Determine risk level from probability
        if prediction_proba < 0.3:
            risk_level = "Low"
            msg = "Currently low estimated risk. Keep maintaining a healthy lifestyle."
        elif prediction_proba < 0.6:
            risk_level = "Medium"
            msg = "Moderate estimated risk. Consider lifestyle improvements and regular check-ups."
        else:
            risk_level = "High"
            msg = "High estimated risk. Please consider consulting a healthcare professional."

        st.subheader("Prediction Result")
        st.caption(f"Model used: **{model_name}**")

        if prediction == 1:
            st.error(
                f"⚠️ The model predicts **diabetes risk**.\n\n"
                f"Estimated probability of diabetes: **{prediction_proba:.2f}**"
            )
        else:
            st.success(
                f"✅ The model predicts **low diabetes risk**.\n\n"
                f"Estimated probability of diabetes: **{prediction_proba:.2f}**"
            )

        st.write(f"**Risk level:** {risk_level}")
        st.info(msg)

        # Risk-specific advice
        st.markdown("### 🩺 Educational Health Guidance (Non-medical)")

        if risk_level == "High":
            st.warning("""
- Please consider visiting a health facility for proper diabetes screening.  
- Discuss your results with a qualified healthcare professional.  
- Focus on reducing sugary foods and increasing physical activity.  
            """)
        elif risk_level == "Medium":
            st.info("""
- You may benefit from lifestyle changes such as more exercise and a healthier diet.  
- Try to monitor your blood pressure, weight, and blood sugar where possible.  
- Consider regular check-ups, especially if you have a family history of diabetes.  
            """)
        else:  # Low
            st.success("""
- Keep up your healthy habits!  
- Maintain a balanced diet and regular physical activity.  
- Continue going for periodic health check-ups.  
            """)

        st.markdown("#### 💡 General Health Tips")
        st.write("""
- Eat more vegetables, fruits, and whole grains.  
- Aim for at least **150 minutes of moderate exercise per week** (e.g., brisk walking).  
- Limit sugary drinks and highly processed foods.  
- Avoid smoking and reduce alcohol intake.  
- If you have a family history of diabetes or other risk factors, get regular health screenings.
        """)
        st.caption("These are general tips, not personalized medical advice.")

        # Log usage (anonymous, local)
        log_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "Pregnancies": input_data[0, 0],
            "Glucose": input_data[0, 1],
            "BloodPressure": input_data[0, 2],
            "SkinThickness": input_data[0, 3],
            "Insulin": input_data[0, 4],
            "BMI": input_data[0, 5],
            "DiabetesPedigreeFunction": input_data[0, 6],
            "Age": input_data[0, 7],
            "prediction": int(prediction),
            "probability": float(prediction_proba),
            "risk_level": risk_level
        }

        log_file = "usage_log.csv"
        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)
            df_log = pd.concat([df_log, pd.DataFrame([log_row])], ignore_index=True)
        else:
            df_log = pd.DataFrame([log_row])

        df_log.to_csv(log_file, index=False)
        st.caption("This prediction was saved locally (anonymous) to 'usage_log.csv'.")

    st.markdown("---")
    st.caption("""
⚠️ This tool is a **demonstration** and **not a medical diagnosis**.
Always consult a qualified healthcare professional for medical advice.
""")

# ======================================================
# TAB 2: About & SDG 3
# ======================================================
with tab_about:
    st.header("About this Project & SDG 3")

    st.markdown("""
### 🎯 Project Overview
This project implements a simple **Diabetes Risk Predictor** using a
machine learning model trained on the public **Pima Indians Diabetes Dataset**.

The app estimates the probability that a person might have diabetes based on:
- Age  
- BMI  
- Blood Pressure  
- Glucose level  
- Insulin and skin thickness  
- Number of pregnancies  
- Family history (diabetes pedigree function)  

The model and application run **entirely offline** once installed.
""")

    st.markdown("""
### 🌍 Connection to SDG 3: Good Health & Well-being

This project supports **Sustainable Development Goal 3** by:

- Raising **awareness** of diabetes risk, a major non-communicable disease.  
- Encouraging **early screening** and check-ups.  
- Providing a simple, low-cost decision support tool that can work in
  **low-resource or low-connectivity environments**.

It is not meant to replace doctors but to show how **data and AI** can assist
public health and preventive care.
""")

    st.markdown(f"""
### 🧠 Model Information

- **Selected algorithm:** {model_name}  
- **Input features:** 8 numerical health parameters  
- **Output:** Probability of diabetes (0 to 1) and a binary prediction (Yes/No)  

The model was evaluated on a held-out test set. In practice, any AI model used
for real healthcare decisions would need much more data, validation, and expert review.
""")

    st.markdown("""
### ⚠️ Important Disclaimer

This application is **for educational and demonstration purposes only**.
It is **not approved for clinical use** and must not be used as a substitute
for professional medical advice, diagnosis, or treatment.
""")


# ======================================================
# TAB 3: Usage Stats
# ======================================================
with tab_stats:
    st.header("📈 Usage Statistics & Model Performance")

    # -------------------------
    # A. Local usage analytics
    # -------------------------
    log_file = "usage_log.csv"

    st.subheader("App Usage (Local Only)")
    if not os.path.exists(log_file):
        st.info(
            "No usage data found yet. Make some predictions in the "
            "**🔮 Risk Prediction** tab and they will appear here."
        )
    else:
        df_log = pd.read_csv(log_file)

        # Top-level numbers
        total_preds = len(df_log)
        avg_prob = df_log["probability"].mean()

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.metric("Total Predictions", total_preds)
        with col_a2:
            st.metric("Avg. Predicted Diabetes Probability", f"{avg_prob:.2f}")

        # Risk level distribution
        st.markdown("#### Risk Level Distribution")
        risk_counts = (
            df_log["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
        )
        st.bar_chart(risk_counts)

        # Recent logs
        st.markdown("#### Recent Predictions (Last 10)")
        st.dataframe(df_log.tail(10))
        st.caption("Data is stored locally in 'usage_log.csv' and does not contain names or IDs.")

        # Optional: allow export of CSV
        csv_bytes = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Usage Log as CSV",
            data=csv_bytes,
            file_name="usage_log.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # -------------------------
    # B. Model performance section
    # -------------------------
    st.subheader("Model Performance (Test Set)")
    metrics_file = "model_metrics.json"

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        # Top metrics in columns
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Best Model", metrics["best_model"])
        with col_m2:
            st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        with col_m3:
            st.metric("Recall", f"{metrics['recall']:.2f}")
        with col_m4:
            st.metric("Precision", f"{metrics['precision']:.2f}")
        with col_m5:
            st.metric("F1-score", f"{metrics['f1']:.2f}")

        # ---- Confusion Matrix ----
        cm = metrics.get("confusion_matrix", None)
        if cm is not None:
            st.markdown("### Confusion Matrix")

            cm_array = np.array(cm)

            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_array,
                display_labels=["No Diabetes (0)", "Diabetes (1)"],
            )
            disp.plot(ax=ax_cm, values_format="d", colorbar=False)
            ax_cm.set_xlabel("Predicted label")
            ax_cm.set_ylabel("True label")
            st.pyplot(fig_cm)

        # Put ROC & PR curves side-by-side if both exist
        roc_curve_data = metrics.get("roc_curve", None)
        pr_curve_data = metrics.get("pr_curve", None)

        if roc_curve_data or pr_curve_data:
            st.markdown("### Evaluation Curves")
            col_c1, col_c2 = st.columns(2)

            # ---- ROC Curve ----
            if roc_curve_data is not None:
                with col_c1:
                    st.markdown("#### ROC Curve")
                    fpr = roc_curve_data.get("fpr", [])
                    tpr = roc_curve_data.get("tpr", [])

                    if len(fpr) > 0 and len(tpr) > 0:
                        fig_roc, ax_roc = plt.subplots()
                        ax_roc.plot(fpr, tpr, label="ROC curve")
                        ax_roc.plot([0, 1], [0, 1], linestyle="--")
                        ax_roc.set_xlabel("False Positive Rate")
                        ax_roc.set_ylabel("True Positive Rate")
                        ax_roc.set_title("Receiver Operating Characteristic")
                        st.pyplot(fig_roc)
                    else:
                        st.info("ROC data is empty.")

            # ---- Precision–Recall Curve ----
            if pr_curve_data is not None:
                with col_c2:
                    st.markdown("#### Precision–Recall Curve")
                    precision_vals = pr_curve_data.get("precision", [])
                    recall_vals = pr_curve_data.get("recall", [])

                    if len(precision_vals) > 0 and len(recall_vals) > 0:
                        fig_pr, ax_pr = plt.subplots()
                        ax_pr.plot(recall_vals, precision_vals)
                        ax_pr.set_xlabel("Recall")
                        ax_pr.set_ylabel("Precision")
                        ax_pr.set_title("Precision–Recall Curve")
                        st.pyplot(fig_pr)
                    else:
                        st.info("Precision–Recall data is empty.")
    else:
        st.info(
            "Model performance summary not found. Run `train_model.py` "
            "to generate model_metrics.json (and plots data)."
        )

    st.markdown("---")

    # -------------------------
    # C. Feature importance block
    # -------------------------
    st.subheader("Feature Importance (if available)")

    if os.path.exists("feature_importances.csv"):
        fi_df = pd.read_csv("feature_importances.csv")
        st.write("The chart below shows which features contributed most in the Random Forest model:")
        st.bar_chart(fi_df.set_index("feature")["importance"])
    else:
        st.info(
            "Feature importance is only available when the Random Forest model "
            "is selected as the best model during training."
        )
