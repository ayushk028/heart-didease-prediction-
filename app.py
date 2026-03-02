import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("heart_model.pkl", "rb"))

st.title("❤️ Heart Disease Prediction System")

st.write("Enter Patient Details:")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak, slope,
                            ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Risk Probability: {probability*100:.2f}%")

    if probability > 0.7:
        st.error("High Risk of Heart Disease")
    elif probability > 0.4:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")