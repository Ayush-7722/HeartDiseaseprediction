import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model Files
# -----------------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result {
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 35px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter your health information to get a prediction</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Input Section
# -----------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )


with col2:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )


oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)


# -----------------------------
# Prediction Button
# -----------------------------
st.markdown("---")

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    # Create input dictionary
    raw_input = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Scale
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Result
    if prediction == 1:
        st.error(
            "⚠️ Higher Risk Detected\n\n"
            "The model predicts a possibility of heart disease."
        )
    else:
        st.success(
            "✅ Lower Risk Detected\n\n"
            "The model does not predict heart disease."
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">Heart Disease Prediction • Machine Learning Project</div>',
    unsafe_allow_html=True
)