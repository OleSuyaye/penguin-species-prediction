import streamlit as st
import pandas as pd
import joblib


# Load the saved model
model = joblib.load("models/penguin_logistic_regression.pkl")


# App title
st.title("Penguin Species Prediction")

st.write(
    "Enter the penguin measurements below to predict whether it is Adelie or Gentoo."
)


# User inputs
island = st.selectbox(
    "Island",
    ["Biscoe", "Dream", "Torgersen"]
)

bill_length = st.number_input(
    "Bill Length (mm)",
    min_value=0.0
)

bill_depth = st.number_input(
    "Bill Depth (mm)",
    min_value=0.0
)

flipper_length = st.number_input(
    "Flipper Length (mm)",
    min_value=0.0
)

body_mass = st.number_input(
    "Body Mass (g)",
    min_value=0.0
)


# Prediction button
if st.button("Predict Species"):

    input_data = pd.DataFrame({
        "island": [island],
        "bill_length_mm": [bill_length],
        "bill_depth_mm": [bill_depth],
        "flipper_length_mm": [flipper_length],
        "body_mass_g": [body_mass]
    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction")

    st.write(f"Predicted Species: **{prediction}**")

    st.write(
        f"Adelie Probability: {probabilities[0]:.2%}"
    )

    st.write(
        f"Gentoo Probability: {probabilities[1]:.2%}"
    )