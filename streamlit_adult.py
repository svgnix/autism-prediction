


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and column names
with open('ANN_adult_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open("column_names_adult.pkl", "rb") as f:
    columns = pickle.load(f)

# Assume scaler used in training is saved and reused here
sc = StandardScaler()

# Title
st.title("ASD Screening Prediction")

st.header("Please fill in the following details:")

# A1 to A10 Scores - Yes (1) / No (0) using radio buttons
# Full-text questions for A1 to A10
questions = {
    "A1_Score": "I often notice small sounds when others do not.",
    "A2_Score": "I usually concentrate more on the whole picture, rather than the small details.",
    "A3_Score": "I find it easy to do more than one thing at once.",
    "A4_Score": "If there is an interruption, I can switch back to what I was doing very quickly.",
    "A5_Score": "I find it easy to ‘read between the lines’ when someone is talking to me.",
    "A6_Score": "I know how to tell if someone listening to me is getting bored.",
    "A7_Score": "When I’m reading a story, I find it difficult to work out the characters’ intentions.",
    "A8_Score": "I like to collect information about categories of things (e.g. types of car, bird, train, plant, etc).",
    "A9_Score": "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "A10_Score": "I find it difficult to work out people’s intentions."
}

a_scores = {}
st.subheader("Screening Questions (A1 to A10)")
for key, question in questions.items():
    a_scores[key] = 1 if st.radio(question, ["No", "Yes"], index=0) == "Yes" else 0


# Age input
age = st.number_input("Age", min_value=1, max_value=100, value=25)

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])
gender = 'm' if gender == "Male" else 'f'

# Ethnicity
ethnicities = ['White-European', 'Latino', 'Others', 'Black', 'Asian', 'Middle Eastern ', 'Pasifika', 
               'South Asian', 'Hispanic', 'Turkish', 'others']
ethnicity = st.selectbox("Ethnicity", ethnicities)

# Relation
relations = ['Self', 'Parent', 'Health care professional', 'Relative', 'Others']
relation = st.selectbox("Relation", relations)

# Family PDD history
family_pdd = st.radio("Family history of PDD?", ["No", "Yes"], horizontal=True).lower()

# Jaundice
jaundice = st.selectbox("Jaundice (experienced in childhood)?", ["No", "Yes"]).lower()

# Country of Residence
countries = ['United States', 'Brazil', 'Spain', 'Egypt', 'New Zealand', 'Bahamas', 'Burundi', 'Austria', 'Argentina',
             'Jordan', 'Ireland', 'United Arab Emirates', 'Afghanistan', 'Lebanon', 'United Kingdom', 'South Africa',
             'Italy', 'Pakistan', 'Bangladesh', 'Chile', 'France', 'China', 'Australia', 'Canada', 'Saudi Arabia',
             'Netherlands', 'Romania', 'Sweden', 'Tonga', 'Oman', 'India', 'Philippines', 'Sri Lanka', 'Sierra Leone',
             'Ethiopia', 'Viet Nam', 'Iran', 'Costa Rica', 'Germany', 'Mexico', 'Russia', 'Armenia', 'Iceland',
             'Nicaragua', 'Hong Kong', 'Japan', 'Ukraine', 'Kazakhstan', 'AmericanSamoa', 'Uruguay', 'Serbia',
             'Portugal', 'Malaysia', 'Ecuador', 'Niger', 'Belgium', 'Bolivia', 'Aruba', 'Finland', 'Turkey', 'Nepal',
             'Indonesia', 'Angola', 'Azerbaijan', 'Iraq', 'Czech Republic', 'Cyprus']
country_of_res = st.selectbox("Country of Residence", countries)

# Submit button
if st.button("Predict ASD"):
    # Create initial DataFrame
    input_data = pd.DataFrame([{
        **a_scores,
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'relation': relation,
        'family_pdd': family_pdd,
        'jaundice': jaundice,
        'country_of_res': country_of_res
    }])

    # One-hot encode categorical columns
    input_data = pd.get_dummies(input_data, columns=["country_of_res", "ethnicity", "relation"])

    # Add missing dummy columns with 0
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training
    input_data = input_data[columns]

    # Encode gender, jaundice, family_pdd using label encoding
    le = LabelEncoder()
    input_data['gender'] = le.fit(['f', 'm']).transform(input_data['gender'])
    input_data['jaundice'] = le.fit(['no', 'yes']).transform(input_data['jaundice'])
    input_data['family_pdd'] = le.fit(['no', 'yes']).transform(input_data['family_pdd'])

    # Scale input
    scaled_input = sc.fit_transform(input_data)  # Use transform if you have trained scaler

    # Predict
    prediction = model.predict(scaled_input.astype(np.float32))
    result = "Yes" if prediction[0] > 0.5 else "No"

    st.subheader(f"ASD Prediction: {result}")
    # st.write(f"Model raw output: {prediction[0]}")
