import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("salary_model.pkl")  # Do NOT unpack this; just the model

st.title("🧑‍💼 Employee Salary Predictor")

# User inputs
age = st.number_input("Age", 18, 100, step=1)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                       'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                                       'Assoc-acdm', 'Assoc-voc', 'Doctorate', '5th-6th', 'Prof-school',
                                       '12th', '1st-4th', '10th', 'Preschool', '7th-8th'])
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                         'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                         'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                         'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                         'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                             'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
gender = st.radio("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", 0)
capital_loss = st.number_input("Capital Loss", 0)
hours_per_week = st.slider("Hours Per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                                                 'India', 'England', 'Cuba', 'Jamaica', 'South', 'China',
                                                 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan'])

# Predict
if st.button("Predict"):
    input_dict = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': 100000,  # placeholder
        'education': education,
        'educational-num': 10,  # placeholder
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    input_df = pd.DataFrame([input_dict])

    # Same preprocessing as in training
    # 👉 Only needed if you used LabelEncoder or OneHotEncoder in model_training.py

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("💰 Likely earns *more than 50K*")
    else:
        st.warning("💼 Likely earns *50K or less*")