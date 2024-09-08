{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0dcd3877-21b9-4936-bf2b-ec3191074907",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from joblib import load\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "21afb001-144f-4d7b-9cdb-5d9a07b2ed45",
   "metadata": {},
   "outputs": [],
   "source": [
    "naive_bayes_model = load('naive_bayes_model.joblib')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b221864b-32c9-4ad2-92a7-80cc0272be98",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    st.title(\"Heart Disease Prediction App - Naive Bayes\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "561c6728-7854-496b-ab30-ee0aef06613e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-08 17:56:55.105 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:56:55.217 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-09-08 17:56:55.218 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=1, _parent=DeltaGenerator())"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    " st.sidebar.title(\"Options\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "77bb27d3-f6d0-4a0e-bfca-c7995d89e83e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-08 17:57:13.799 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:13.800 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:13.804 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:13.807 Session state does not function when running a script without `streamlit run`\n",
      "2024-09-08 17:57:13.809 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:13.810 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "option = st.sidebar.selectbox(\"Choose how to input data\", [\"Enter data manually\", \"Upload file\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0feb5d35-a894-47b0-ae15-9d3a6be45e96",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-08 17:57:34.004 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.006 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.007 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.008 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.011 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.012 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.013 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.015 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.017 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.020 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.022 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.023 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.025 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.027 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.028 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.030 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.032 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.033 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.034 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.036 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.037 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.038 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.041 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.042 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.042 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.044 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.046 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.048 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.051 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.052 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.053 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.055 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.055 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.058 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.059 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.061 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.063 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.064 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.065 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.067 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.068 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.070 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.072 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.073 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.074 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.076 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.077 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.077 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.078 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.081 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.084 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.086 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.088 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.089 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.090 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.093 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.094 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.094 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.097 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.098 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.099 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.101 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.102 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.103 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.104 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 17:57:34.106 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if option == \"Enter data manually\":\n",
    "        st.header(\"Manual Input\")\n",
    "\n",
    "        # Manually input the features\n",
    "        age = st.number_input(\"Age\", min_value=0, max_value=120, value=50)\n",
    "        sex = st.selectbox(\"Sex\", options=[\"Male\", \"Female\"])\n",
    "        cp = st.selectbox(\"Chest Pain Type\", options=[\"Typical Angina\", \"Atypical Angina\", \"Non-anginal Pain\", \"Asymptomatic\"])\n",
    "        trestbps = st.number_input(\"Resting Blood Pressure\", min_value=80, max_value=200, value=120)\n",
    "        chol = st.number_input(\"Serum Cholesterol (mg/dl)\", min_value=100, max_value=400, value=200)\n",
    "        fbs = st.selectbox(\"Fasting Blood Sugar > 120 mg/dl\", options=[\"True\", \"False\"])\n",
    "        restecg = st.selectbox(\"Resting Electrocardiographic Results\", options=[\"Normal\", \"ST-T Wave Abnormality\", \"Left Ventricular Hypertrophy\"])\n",
    "        thalach = st.number_input(\"Maximum Heart Rate Achieved\", min_value=60, max_value=220, value=150)\n",
    "        exang = st.selectbox(\"Exercise Induced Angina\", options=[\"Yes\", \"No\"])\n",
    "        oldpeak = st.number_input(\"ST Depression Induced by Exercise\", min_value=0.0, max_value=6.0, step=0.1, value=1.0)\n",
    "        slope = st.selectbox(\"Slope of the Peak Exercise ST Segment\", options=[\"Upsloping\", \"Flat\", \"Downsloping\"])\n",
    "        ca = st.number_input(\"Number of Major Vessels Colored by Fluoroscopy\", min_value=0, max_value=3, value=0)\n",
    "        thal = st.selectbox(\"Thalassemia\", options=[\"Normal\", \"Fixed Defect\", \"Reversible Defect\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "7b13ee3e-b248-4813-b421-6d27e7b70be4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert categorical inputs to numerical values\n",
    "sex = 1 if sex == \"Male\" else 0\n",
    "cp_mapping = {\n",
    "    \"Typical Angina\": 0,\n",
    "    \"Atypical Angina\": 1,\n",
    "    \"Non-anginal Pain\": 2,\n",
    "    \"Asymptomatic\": 3\n",
    "}\n",
    "cp = cp_mapping[cp]\n",
    "fbs = 1 if fbs == \"True\" else 0\n",
    "restecg_mapping = {\n",
    "    \"Normal\": 0,\n",
    "    \"ST-T Wave Abnormality\": 1,\n",
    "    \"Left Ventricular Hypertrophy\": 2\n",
    "}\n",
    "restecg = restecg_mapping[restecg]\n",
    "exang = 1 if exang == \"Yes\" else 0\n",
    "slope_mapping = {\n",
    "    \"Upsloping\": 0,\n",
    "    \"Flat\": 1,\n",
    "    \"Downsloping\": 2\n",
    "}\n",
    "slope = slope_mapping[slope]\n",
    "thal_mapping = {\n",
    "    \"Normal\": 1,\n",
    "    \"Fixed Defect\": 2,\n",
    "    \"Reversible Defect\": 3\n",
    "}\n",
    "thal = thal_mapping[thal]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ea6da725-d5ce-45b6-b943-bd902e49dc3f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-08 18:11:06.304 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 18:11:06.306 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if option == \"Enter data manually\":\n",
    "    st.header(\"Manual Input\")\n",
    "    # [Code for manual input]\n",
    "    \n",
    "elif option == \"Upload file\":\n",
    "    st.header(\"File Upload\")\n",
    "    \n",
    "    # Upload a CSV file\n",
    "    uploaded_file = st.file_uploader(\"Choose a file\", type=['csv'])\n",
    "    if uploaded_file is not None:\n",
    "        data = pd.read_csv(uploaded_file)\n",
    "        if not data.empty:\n",
    "            predict_and_display(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "5ec4070f-f7f0-4bc1-953c-89616cad9d87",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_and_display(data):\n",
    "    # Predict using the loaded model\n",
    "    predictions = naive_bayes_model.predict(data)\n",
    "    \n",
    "    # Display predictions\n",
    "    results_df = data.copy()\n",
    "    results_df['Prediction'] = predictions\n",
    "\n",
    "    # Show predictions in a table\n",
    "    with st.expander(\"Show/Hide Prediction Table\"):\n",
    "        st.table(results_df)\n",
    "    \n",
    "    # Plot histogram of predictions\n",
    "    st.write(\"Histogram of Predictions:\")\n",
    "    fig, ax = plt.subplots()  # This line should be aligned with other lines inside the function\n",
    "    prediction_counts = pd.Series(predictions).value_counts().sort_index()\n",
    "    prediction_counts.plot(kind='bar', ax=ax)\n",
    "    ax.set_title(\"Number of Heart Disease Predictions\")\n",
    "    ax.set_xlabel(\"Prediction\")\n",
    "    ax.set_ylabel(\"Count\")\n",
    "    st.pyplot(fig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "c34e14e9-e607-4549-9cdc-95986d9f3132",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-08 18:13:58.652 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-08 18:13:58.654 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "878705b0-7d2e-4d21-a7a7-4f99cc7c2b61",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
