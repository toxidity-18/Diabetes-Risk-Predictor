import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor - SDG 3",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
with open("diabetes_model.pkl", "rb") as f:
    model_artifact = pickle.load(f)

model = model_artifact["model"]
scaler = model_artifact["scaler"]
feature_names = model_artifact["feature_names"]

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
        Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
        BloodPressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        SkinThickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        Insulin = st.sidebar.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
        BMI = st.sidebar.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        DiabetesPedigreeFunction = st.sidebar.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01
        )
        Age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=30)

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

    # Prediction button
    if st.button("Predict Risk"):
        # scale input using the same scaler
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]  # probability of diabetes (class 1)

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

        # General health tips (non-medical)
        st.markdown("### 💡 General Health Tips (Non-medical)")
        st.write("""
- Try to maintain a balanced diet with vegetables, fruits, and whole grains.  
- Aim for regular physical activity (e.g., walking, light exercise).  
- Limit sugary drinks and highly processed foods.  
- Avoid smoking and excessive alcohol.  
- Go for regular health check-ups if possible.
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
Logistic Regression model trained on the public **Pima Indians Diabetes Dataset**.

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

    st.markdown("""
### 🧠 Model Information

- **Algorithm:** Logistic Regression  
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
    st.header("📈 Usage Statistics (Local)")

    log_file = "usage_log.csv"

    if not os.path.exists(log_file):
        st.info(
            "No usage data found yet. Make some predictions in the "
            "**🔮 Risk Prediction** tab and they will appear here."
        )
    else:
        df_log = pd.read_csv(log_file)

        st.write(f"Total predictions made: **{len(df_log)}**")

        # Average probability
        avg_prob = df_log["probability"].mean()
        st.write(f"Average predicted diabetes probability: **{avg_prob:.2f}**")

        # Count by risk level
        st.subheader("Risk Level Distribution")
        risk_counts = df_log["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)

        st.bar_chart(risk_counts)

        # Show recent logs
        st.subheader("Recent Predictions (Last 10)")
        st.dataframe(df_log.tail(10))
        st.caption("Data is stored locally in 'usage_log.csv' and does not contain names or IDs.")
