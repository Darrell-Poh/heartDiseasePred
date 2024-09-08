import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
naive_bayes_model = load('naive_bayes_model.joblib')

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
    
    return processed_data

def predict_and_display(data):
    # Preprocess the input data
    processed_data = preprocess_input(data)
    
    # Predict using the loaded model
    predictions = naive_bayes_model.predict(processed_data)
    
    # Display predictions
    results_df = pd.DataFrame(processed_data, columns=processed_data.columns)
    results_df['Prediction'] = predictions
    
    # Show predictions in a table
    st.header("Prediction Results")
    st.table(results_df)
    
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

    st.header("File Upload")

    # Upload a CSV file
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if not data.empty:
            # Predict and display results
            predict_and_display(data)

if __name__ == '__main__':
    main()
