import streamlit as st 
import numpy as np
import pickle 

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# App Title
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# User Input (ALL with unique keys)
pregnancies = st.number_input("Pregnancies", min_value=0, key="preg")
glucose = st.number_input("Glucose Level", min_value=50, key="glucose")
blood_pressure = st.number_input("Blood Pressure", min_value=40, key="bp")
skin_thickness = st.number_input("Skin Thickness", min_value=0, key="skin")
insulin = st.number_input("Insulin Level", min_value=0, key="insulin")
bmi = st.number_input("BMI", min_value=10.0, key="bmi")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, key="dpf")
age = st.number_input("Age", min_value=1, key="age")

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error("The patient is likely diabetic.")
    else:
        st.success("The patient is not diabetic.")

    st.write(f"Probability of Diabetes: {prob:.2f}")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app predicts diabetes using ML model trained on clinical data.")