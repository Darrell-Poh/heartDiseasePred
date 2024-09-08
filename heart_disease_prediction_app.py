import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "naive_bayes_model.joblib"

naive_bayes_model = load(file_path)

# Main function
def main():
    st.title("Heart Disease Prediction App - Naive Bayes")

    # Manually input the features
    st.header("Manual Input")

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

    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp = cp_mapping[cp]
    fbs = 1 if fbs == "True" else 0
    restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    restecg = restecg_mapping[restecg]
    exang = 1 if exang == "Yes" else 0
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_mapping[slope]
    thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal = thal_mapping[thal]

    # Prepare data for prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame(input_data, columns=columns)

    # Predict and display results
    predict_and_display(input_df)

# Function to predict and display results
def predict_and_display(data):
    # Predict using the loaded model
    predictions = naive_bayes_model.predict(data)

    # Display predictions
    results_df = data.copy()
    results_df['Prediction'] = predictions

    # Show predictions in a table
    with st.expander("Show/Hide Prediction Table"):
        st.table(results_df)

    # Plot histogram of predictions
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()  # This line should be aligned properly
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    prediction_counts.plot(kind='bar', ax=ax)
    ax.set_title("Number of Heart Disease Predictions")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    st.pyplot(fig)

if __name__ == '__main__':
    main()
