
import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Saved Model
# ----------------------------
model = joblib.load("maintenance_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="SMART Maintenance AI",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------
# Title
# ----------------------------
st.title("🤖 SMART Maintenance AI")
st.write("### Predict Whether a Machine Needs Maintenance")

st.markdown("---")

# ----------------------------
# User Inputs
# ----------------------------

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=300.0
)

process_temp = st.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=310.0
)

rpm = st.number_input(
    "Rotational Speed (RPM)",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=120
)

st.markdown("---")

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Machine Health"):

    machine_type = encoder.transform([machine_type])[0]

    input_df = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("## Prediction Result")

    if prediction == 0:
        st.success("✅ Machine is Healthy")
    else:
        st.error("⚠️ Maintenance Required")

    st.write(f"Confidence : **{max(probability)*100:.2f}%**")

    st.markdown("---")

    st.subheader("Input Summary")

    st.write(input_df)
