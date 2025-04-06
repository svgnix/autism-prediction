


import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model and required columns
@st.cache_resource
def load_model_and_columns():
    with open('LR_Toddler_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('output_column_names.pkl', 'rb') as f:
        required_columns = pickle.load(f)
    return model, required_columns

model, required_columns = load_model_and_columns()

st.title("ASD Prediction Application")
st.write("Fill in the information below to predict if your child may have ASD.")

st.header("Questionnaire (A1 - A10)")
# Each question returns "Yes" or "No". We then store 1 for Yes, and 0 for No.
q1 = st.radio("A1: Does your child look at you when you call his/her name?", ("Yes", "No"))
q2 = st.radio("A2: How easy is it for you to get eye contact with your child?", ("Yes", "No"))
q3 = st.radio("A3: Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)", ("Yes", "No"))
q4 = st.radio("A4: Does your child point to share interest with you? (e.g. pointing at an interesting sight)", ("Yes", "No"))
q5 = st.radio("A5: Does your child pretend? (e.g. care for dolls, talk on a toy phone)", ("Yes", "No"))
q6 = st.radio("A6: Does your child follow where you’re looking?", ("Yes", "No"))
q7 = st.radio("A7: If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging)", ("Yes", "No"))
q8 = st.radio("A8: Would you describe your child’s first words as meaningful? (Yes/No selection)", ("Yes", "No"))
q9 = st.radio("A9: Does your child use simple gestures? (e.g. wave goodbye)", ("Yes", "No"))
q10 = st.radio("A10: Does your child stare at nothing with no apparent purpose?", ("Yes", "No"))

# Convert the responses to 1 or 0
answers = {
    'A1': 1 if q1 == "Yes" else 0,
    'A2': 1 if q2 == "Yes" else 0,
    'A3': 1 if q3 == "Yes" else 0,
    'A4': 1 if q4 == "Yes" else 0,
    'A5': 1 if q5 == "Yes" else 0,
    'A6': 1 if q6 == "Yes" else 0,
    'A7': 1 if q7 == "Yes" else 0,
    'A8': 1 if q8 == "Yes" else 0,
    'A9': 1 if q9 == "Yes" else 0,
    'A10': 1 if q10 == "Yes" else 0
}

st.header("Additional Information")
age_mons = st.number_input("Age in Months", min_value=1, step=1)
# qchat_score = st.number_input("Qchat-10 Score", min_value=0, step=1)

# Sex: radio selection that stores 'm' for Male and 'f' for Female.
sex = st.radio("Sex", ("Male", "Female"))
sex = 'm' if sex == "Male" else 'f'

# Ethnicity: dropdown with provided options.
ethnicity_options = [
    'middle eastern', 'White European', 'Hispanic', 'black', 'asian', 
    'south asian', 'Native Indian', 'Others', 'Latino', 'mixed', 'Pacifica'
]
ethnicity = st.selectbox("Ethnicity", ethnicity_options)

# Jaundice: yes or no selection.
jaundice = st.radio("Jaundice", ("yes", "no"))

# Family member with ASD: yes or no selection.
family_mem = st.radio("Family member with ASD", ("yes", "no"))

# Who completed the test: dropdown with provided options.
who_completed_options = [
    'family member', 'Health Care Professional', 'Health care professional', 'Self', 'Others'
]
# who_completed = st.selectbox("Who completed the test", who_completed_options)

if st.button("Predict ASD"):
    # Build the input DataFrame
    input_dict = {
        'A1': [answers['A1']],
        'A2': [answers['A2']],
        'A3': [answers['A3']],
        'A4': [answers['A4']],
        'A5': [answers['A5']],
        'A6': [answers['A6']],
        'A7': [answers['A7']],
        'A8': [answers['A8']],
        'A9': [answers['A9']],
        'A10': [answers['A10']],
        'Age_Mons': [age_mons],
        # 'Qchat-10-Score': [qchat_score],
        'Sex': [sex],
        'Ethnicity': [ethnicity],
        'Jaundice': [jaundice],
        'Family_mem_with_ASD': [family_mem],
        # 'Who completed the test': [who_completed]
    }
    input_data = pd.DataFrame(input_dict)
    
    # Create dummy variables for categorical features
    input_data = pd.get_dummies(input_data, columns=["Ethnicity"])
    
    # Ensure all required columns (from training) are present in the input data
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing dummy columns with zeros
    # Reorder columns to match the training data
    input_data = input_data[required_columns]
    
    # Apply LabelEncoder transformations for binary variables.
    # Note: The encoders are set to the specific order used during training.
    le = LabelEncoder()
    input_data['Sex'] = le.fit(['f', 'm']).transform(input_data['Sex'])
    input_data['Jaundice'] = le.fit(['no', 'yes']).transform(input_data['Jaundice'])
    input_data['Family_mem_with_ASD'] = le.fit(['no', 'yes']).transform(input_data['Family_mem_with_ASD'])

    @st.cache_resource
    def load_scaler():
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    
    scaler = load_scaler()

    
# Predict using the pre-trained model (convert input to float32 for Keras compatibility)
    import numpy as np
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input.astype(np.float32))



    
    # Output the result
    st.subheader("ASD Prediction:")
    if prediction[0] > 0.5:
        st.write("Yes")
    else:
        st.write("No")

    # st.write(prediction)

    # st.subheader("Complete Input Data:")
    # st.dataframe(input_data)



