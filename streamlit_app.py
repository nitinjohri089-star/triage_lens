"""Streamlit interface for the emergency triage prediction API."""

import os

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Emergency Triage",
    page_icon="🏥",
    layout="centered",
)

st.title("Emergency Triage Classifier")
st.caption("Enter patient measurements to request a model prediction.")

api_url = st.sidebar.text_input(
    "Prediction API URL",
    value=os.getenv("TRIAGE_API_URL", "http://127.0.0.1:8000"),
).rstrip("/")

with st.sidebar:
    if st.button("Check API connection"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            st.success("Prediction API is ready.")
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API: {exc}")

st.info(
    "This project is for demonstration and educational use only. "
    "Its predictions are not medical advice and must not guide patient care."
)

with st.form("patient_features"):
    st.subheader("Patient measurements")
    model_name = st.selectbox(
        "Regression model",
        options=["Logistic Regression", "Linear Regression"],
    )
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=35)
        sex = st.selectbox("Sex", options=["Female", "Male"])
        temperature_f = st.number_input(
            "Temperature (°F)",
            min_value=70.0,
            max_value=115.0,
            value=98.6,
            step=0.1,
        )
        pulse = st.number_input(
            "Pulse (bpm)", min_value=0, max_value=300, value=80
        )
        respiratory_rate = st.number_input(
            "Respiratory rate (breaths/min)",
            min_value=0,
            max_value=100,
            value=16,
        )

    with col2:
        systolic_bp = st.number_input(
            "Systolic blood pressure (mmHg)",
            min_value=0,
            max_value=350,
            value=120,
        )
        diastolic_bp = st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=0,
            max_value=250,
            value=80,
        )
        oxygen_saturation = st.number_input(
            "Oxygen saturation (%)",
            min_value=0.0,
            max_value=100.0,
            value=98.0,
            step=0.1,
        )
        pain_scale = st.number_input(
            "Pain scale (0-10)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=1.0,
        )
        ambulance_arrival = st.selectbox(
            "Arrived by ambulance", options=["No", "Yes"]
        )

    submitted = st.form_submit_button("Predict triage category", type="primary")

if submitted:
    patient = {
        "age": age,
        "sex": sex,
        "temperature_f": temperature_f,
        "pulse": pulse,
        "respiratory_rate": respiratory_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "oxygen_saturation": oxygen_saturation,
        "pain_scale": pain_scale,
        "ambulance_arrival": ambulance_arrival,
        "model": model_name,
    }

    try:
        response = requests.post(
            f"{api_url}/predict",
            json=patient,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
    else:
        st.subheader("Model result")
        st.metric("Predicted category", result["prediction"])
        if result["model"] == "Logistic Regression":
            probabilities = pd.DataFrame(
                [
                    {"Triage category": label, "Probability": probability}
                    for label, probability in result["probabilities"].items()
                ]
            ).sort_values("Probability", ascending=False)
            st.bar_chart(
                probabilities.set_index("Triage category"),
                y="Probability",
            )
            st.dataframe(
                probabilities.style.format({"Probability": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.metric("Linear regression raw score", f"{result['score']:.3f}")
            st.caption(
                "This score is not a probability. Scores at least 0.5 map to "
                "Urgent; lower scores map to Non-Urgent."
            )
