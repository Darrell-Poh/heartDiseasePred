import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

# Load the trained model and label encoders
naive_bayes_model = load('naive_bayes_model.joblib')
label_encoders = load('label_encoders.joblib')

# Define the feature names
model_expected_features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Encode categorical features using label encoders
    for column, le in label_encoders.items():
        if column in df.columns:
            df[column] = le.transform(df[column])
    
    # Ensure all features are included and ordered correctly
    processed_data = df[model_expected_features]
    
    # Handle any missing values if necessary
    processed_data = processed_data.fillna(0)  # Fill missing values with a default value, e.g., 0
    
    return processed_data

def predict_and_display(input_data):
    processed_data = preprocess_input(input_data)
    
    # Predict using the loaded model
    predictions = naive_bayes_model.predict(processed_data)
    
    # Display predictions
    result = "Heart Disease" if predictions[0] == 1 else "No Heart Disease"
    st.write(f"Prediction: {result}")

def main():
    st.title('Heart Disease Prediction')
    
    # Create input fields for user
    age = st.number_input('Age', min_value=1, max_value=120, value=30)
    sex = st.selectbox('Sex', ['M', 'F'])
    chest_pain_type = st.selectbox('ChestPainType', ['ATA', 'NAP', 'TA', 'ASY'])
    resting_bp = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    fasting_bs = st.selectbox('Fasting Blood Sugar', [0, 1])
    resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input('Maximum Heart Rate', min_value=0, max_value=300, value=150)
    exercise_angina = st.selectbox('Exercise Induced Angina', ['Y', 'N'])
    oldpeak = st.number_input('Oldpeak', min_value=-10.0, max_value=10.0, value=0.0)
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])
    
    # Collect input data into a dictionary
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    
    # Button to make prediction
    if st.button('Predict'):
        predict_and_display(input_data)

if __name__ == '__main__':
    main()
