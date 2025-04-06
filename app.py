import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="ASD Screening", layout="centered")
st.title("ASD Screening Prediction")

# Section: Choose the type of prediction
screening_type = st.radio("Select Screening Type", ["Toddler", "Adult"])

# ---------------------------- TODDLER SECTION ----------------------------
if screening_type == "Toddler":
    @st.cache_resource
    def load_toddler_model_and_columns():
        with open('LR_Toddler_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('output_column_names.pkl', 'rb') as f:
            required_columns = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, required_columns, scaler

    model, required_columns, scaler = load_toddler_model_and_columns()

    st.subheader("Fill in Toddler Information")

    st.markdown("### Questionnaire (A1 - A10)")
    q = {}
    for i in range(1, 11):
        question = f"A{i}"
        q[question] = 1 if st.radio(f"{question}:", ["Yes", "No"]) == "Yes" else 0

    age_mons = st.number_input("Age in Months", min_value=1, step=1)
    sex = st.radio("Sex", ["Male", "Female"])
    sex = 'm' if sex == "Male" else 'f'

    ethnicity = st.selectbox("Ethnicity", [
        'middle eastern', 'White European', 'Hispanic', 'black', 'asian', 
        'south asian', 'Native Indian', 'Others', 'Latino', 'mixed', 'Pacifica'
    ])

    jaundice = st.radio("Jaundice", ["yes", "no"])
    family_mem = st.radio("Family member with ASD", ["yes", "no"])

    if st.button("Predict ASD (Toddler)"):
        input_dict = {**q, 'Age_Mons': age_mons, 'Sex': sex, 'Ethnicity': ethnicity,
                      'Jaundice': jaundice, 'Family_mem_with_ASD': family_mem}
        input_data = pd.DataFrame([input_dict])

        # One-hot encode ethnicity
        input_data = pd.get_dummies(input_data, columns=["Ethnicity"])
        for col in required_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[required_columns]

        le = LabelEncoder()
        input_data['Sex'] = le.fit(['f', 'm']).transform(input_data['Sex'])
        input_data['Jaundice'] = le.fit(['no', 'yes']).transform(input_data['Jaundice'])
        input_data['Family_mem_with_ASD'] = le.fit(['no', 'yes']).transform(input_data['Family_mem_with_ASD'])

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input.astype(np.float32))

        st.subheader("ASD Prediction:")
        st.write("Yes" if prediction[0] > 0.5 else "No")

# ---------------------------- ADULT SECTION ----------------------------
elif screening_type == "Adult":
    @st.cache_resource
    def load_adult_model_and_columns():
        with open('ANN_adult_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open("column_names_adult.pkl", "rb") as f:
            columns = pickle.load(f)
        return model, columns

    model, columns = load_adult_model_and_columns()
    with open('scaler_adult.pkl', 'rb') as f:
        scaler = pickle.load(f)

    st.subheader("Fill in Adult Information")

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
    for key, question in questions.items():
        a_scores[key] = 1 if st.radio(question, ["No", "Yes"], index=0) == "Yes" else 0

    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 'm' if gender == "Male" else 'f'

    ethnicity = st.selectbox("Ethnicity", [
        'White-European', 'Latino', 'Others', 'Black', 'Asian', 'Middle Eastern ', 'Pasifika', 
        'South Asian', 'Hispanic', 'Turkish', 'others'
    ])

    relation = st.selectbox("Relation", ['Self', 'Parent', 'Health care professional', 'Relative', 'Others'])
    family_pdd = st.radio("Family history of PDD?", ["No", "Yes"]).lower()
    jaundice = st.selectbox("Jaundice (experienced in childhood)?", ["No", "Yes"]).lower()

    country_of_res = st.selectbox("Country of Residence", [
        'United States', 'Brazil', 'Spain', 'Egypt', 'New Zealand', 'Bahamas', 'Burundi', 'Austria', 'Argentina',
        'Jordan', 'Ireland', 'United Arab Emirates', 'Afghanistan', 'Lebanon', 'United Kingdom', 'South Africa',
        'Italy', 'Pakistan', 'Bangladesh', 'Chile', 'France', 'China', 'Australia', 'Canada', 'Saudi Arabia',
        'Netherlands', 'Romania', 'Sweden', 'Tonga', 'Oman', 'India', 'Philippines', 'Sri Lanka', 'Sierra Leone',
        'Ethiopia', 'Viet Nam', 'Iran', 'Costa Rica', 'Germany', 'Mexico', 'Russia', 'Armenia', 'Iceland',
        'Nicaragua', 'Hong Kong', 'Japan', 'Ukraine', 'Kazakhstan', 'AmericanSamoa', 'Uruguay', 'Serbia',
        'Portugal', 'Malaysia', 'Ecuador', 'Niger', 'Belgium', 'Bolivia', 'Aruba', 'Finland', 'Turkey', 'Nepal',
        'Indonesia', 'Angola', 'Azerbaijan', 'Iraq', 'Czech Republic', 'Cyprus'
    ])

    if st.button("Predict ASD (Adult)"):
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

        input_data = pd.get_dummies(input_data, columns=["country_of_res", "ethnicity", "relation"])
        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[columns]

        le = LabelEncoder()
        input_data['gender'] = le.fit(['f', 'm']).transform(input_data['gender'])
        input_data['jaundice'] = le.fit(['no', 'yes']).transform(input_data['jaundice'])
        input_data['family_pdd'] = le.fit(['no', 'yes']).transform(input_data['family_pdd'])

        scaled_input = scaler.fit_transform(input_data)
        prediction = model.predict(scaled_input.astype(np.float32))

        st.subheader("ASD Prediction:")
        st.write("Yes" if prediction[0] > 0.5 else "No")
