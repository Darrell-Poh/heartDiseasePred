import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder


naive_bayes_model = load('naive_bayes_model.joblib')


label_encoders = {
    'Sex': LabelEncoder(),
    'ChestPainType': LabelEncoder(),
    'RestingECG': LabelEncoder(),
    'ExerciseAngina': LabelEncoder(),
    'ST_Slope': LabelEncoder()
}

def preprocess_input(data):
   
    data['Sex'] = label_encoders['Sex'].fit_transform(data['Sex'])
    data['ChestPainType'] = label_encoders['ChestPainType'].fit_transform(data['ChestPainType'])
    data['RestingECG'] = label_encoders['RestingECG'].fit_transform(data['RestingECG'])
    data['ExerciseAngina'] = label_encoders['ExerciseAngina'].fit_transform(data['ExerciseAngina'])
    data['ST_Slope'] = label_encoders['ST_Slope'].fit_transform(data['ST_Slope'])
    return data

def predict_and_display(input_data):
    processed_data = preprocess_input(input_data)
    
  
    processed_data = processed_data.drop(['HeartDisease'], axis=1, errors='ignore')
    
  
    predictions = naive_bayes_model.predict(processed_data)
    
    # Display predictions
    result = "Heart Disease" if predictions[0] == 1 else "No Heart Disease"
    st.write(f"Prediction: {result}")

def main():
    st.title("Heart Disease Prediction App - Naive Bayes")

    st.header("Manual Input")

   
    age = st.number_input("Age", min_value=30, max_value=100, value=50)
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

   
    sex = 1 if sex == "Male" else 0
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp = cp_mapping[cp]
    fbs = 1 if fbs == "True" else 0
    restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    restecg = restecg_mapping[restecg]
    exang = 1 if exang == "Yes" else 0
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_mapping[slope]

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [cp],
        'RestingBP': [trestbps],
        'Cholesterol': [chol],
        'FastingBS': [fbs],
        'RestingECG': [restecg],
        'MaxHR': [thalach],
        'ExerciseAngina': [exang],
        'Oldpeak': [oldpeak],
        'ST_Slope': [slope],
        'HeartDisease': [0]  
    })

  
    if st.button('Predict'):
        predict_and_display(input_data)

if __name__ == '__main__':
    main()
