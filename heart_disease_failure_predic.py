import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained Naive Bayes model
naive_bayes_model = load('naive_bayes_model.joblib')

# Imputer to handle NaN values
imputer = SimpleImputer(strategy='mean')

def preprocess_input(data):
    # Convert categorical inputs to numerical values
    sex = 1 if data['Sex'] == "Male" else 0
    cp_mapping = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_mapping[data['ChestPainType']]
    fbs = 1 if data['FastingBS'] == "True" else 0
    restecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_mapping[data['RestingECG']]
    exang = 1 if data['ExerciseAngina'] == "Yes" else 0
    slope_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = slope_mapping[data['ST_Slope']]
    
    # Create DataFrame for input
    processed_data = pd.DataFrame([[
        data['Age'], sex, cp, data['RestingBP'], data['Cholesterol'], fbs, restecg,
        data['MaxHR'], exang, data['Oldpeak'], slope
    ]], columns=[
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ])
    
    # Impute missing values
    processed_data = imputer.fit_transform(processed_data)
    
    return processed_data

def predict_and_display(data):
    # Preprocess the input data
    processed_data = preprocess_input(data)

    # Predict using the loaded model
    try:
        predictions = naive_bayes_model.predict(processed_data)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        return
    
    # Display predictions
    st.header("Prediction Results")
    st.write(f"The predicted outcome is: **{'Heart Disease' if predictions[0] == 1 else 'No Heart Disease'}**")
    
    # Plot histogram of predictions
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    prediction_counts.plot(kind='bar', ax=ax)
    ax.set_title("Number of Heart Disease Predictions")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def main():
    st.title("Heart Disease Prediction App - Naive Bayes")

    st.header("Manual Input")

    # Manually input the features
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["True", "False"], index=1)
    restecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], index=0)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"], index=1)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"], index=0)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"], index=0)

    # Gather input data into a dictionary
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': cp,
        'RestingBP': trestbps,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'RestingECG': restecg,
        'MaxHR': thalach,
        'ExerciseAngina': exang,
        'Oldpeak': oldpeak,
        'ST_Slope': slope
    }
    
    # When the user clicks the Predict button
    if st.button("Predict"):
        predict_and_display(input_data)

if __name__ == '__main__':
    main()
