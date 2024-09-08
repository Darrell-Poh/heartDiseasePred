import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained Naive Bayes model
naive_bayes_model = load('naive_bayes_model.joblib')

# Load label encoders
label_encoders = {
    'Sex': LabelEncoder(),
    'ChestPainType': LabelEncoder(),
    'RestingECG': LabelEncoder(),
    'ExerciseAngina': LabelEncoder(),
    'ST_Slope': LabelEncoder()
}

# Define categorical mappings for prediction
label_encoders['Sex'].classes_ = np.array(['Female', 'Male'])
label_encoders['ChestPainType'].classes_ = np.array(['Asymptomatic', 'Atypical Angina', 'Non-anginal Pain', 'Typical Angina'])
label_encoders['RestingECG'].classes_ = np.array(['Left Ventricular Hypertrophy', 'Normal', 'ST-T Wave Abnormality'])
label_encoders['ExerciseAngina'].classes_ = np.array(['No', 'Yes'])
label_encoders['ST_Slope'].classes_ = np.array(['Downsloping', 'Flat', 'Upsloping'])

def preprocess_input(data):
    # Convert categorical inputs to numerical values
    data['Sex'] = label_encoders['Sex'].transform([data['Sex']])[0]
    data['ChestPainType'] = label_encoders['ChestPainType'].transform([data['ChestPainType']])[0]
    data['RestingECG'] = label_encoders['RestingECG'].transform([data['RestingECG']])[0]
    data['ExerciseAngina'] = label_encoders['ExerciseAngina'].transform([data['ExerciseAngina']])[0]
    data['ST_Slope'] = label_encoders['ST_Slope'].transform([data['ST_Slope']])[0]
    
    # Create DataFrame for input
    processed_data = pd.DataFrame([[
        data['Age'], data['Sex'], data['ChestPainType'], data['RestingBP'], data['Cholesterol'], data['FastingBS'], 
        data['RestingECG'], data['MaxHR'], data['ExerciseAngina'], data['Oldpeak'], data['ST_Slope']
    ]], columns=[
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ])
    
    # Ensure no missing values
    processed_data.fillna(method='ffill', inplace=True)
    
    return processed_data

def predict_and_display(data):
    # Preprocess input data
    processed_data = preprocess_input(data)
    
    # Predict using the loaded model
    predictions = naive_bayes_model.predict(processed_data)
    
    # Display predictions
    result = "Heart Disease" if predictions[0] == 1 else "No Heart Disease"
    st.write(f"The predicted outcome is: {result}")

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
    
    predict_and_display(input_data)

if __name__ == '__main__':
    main()
