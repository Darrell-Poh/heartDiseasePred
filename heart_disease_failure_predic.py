import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
naive_bayes_model = load('naive_bayes_model.joblib')

# Get the feature names expected by the model
expected_features = naive_bayes_model.feature_names_in_

def preprocess_input(data):
    # Convert categorical inputs to numerical values
    sex = 1 if data['sex'] == "Male" else 0
    cp_mapping = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_mapping[data['cp']]
    fbs = 1 if data['fbs'] == "True" else 0
    restecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_mapping[data['restecg']]
    exang = 1 if data['exang'] == "Yes" else 0
    slope_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = slope_mapping[data['slope']]
    thal_mapping = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }
    thal = thal_mapping[data['thal']]
    
    # Return processed features as a DataFrame
    processed_data = pd.DataFrame([[
        data['age'], sex, cp, data['trestbps'], data['chol'], fbs, restecg,
        data['thalach'], exang, data['oldpeak'], slope, data['ca'], thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])
    
    # Handle missing values using SimpleImputer (mean strategy)
    imputer = SimpleImputer(strategy='mean')
    processed_data = pd.DataFrame(imputer.fit_transform(processed_data), columns=processed_data.columns)

    return processed_data

def predict_and_display(data):
    # Preprocess the input data
    processed_data = preprocess_input(data)
    
    # Ensure the data contains the correct features and order
    processed_data = processed_data.reindex(columns=expected_features)

    # Predict using the loaded model
    predictions = naive_bayes_model.predict(processed_data)
    
    # Display predictions
    st.header("Prediction Results")
    st.write(f"The predicted outcome is: **{'Heart Disease' if predictions[0] == 1 else 'No Heart Disease'}**")
    
    # Plot histogram of predictions (if desired)
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
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["True", "False"])
    restecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

    # Gather input data into a dictionary
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # When the user clicks the Predict button
    if st.button("Predict"):
        predict_and_display(input_data)

if __name__ == '__main__':
    main()
