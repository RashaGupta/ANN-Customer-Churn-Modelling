import streamlit as st
import joblib
import numpy as np
from keras.models import load_model

# Load the model (we used Keras to train it)
model = load_model('customer_churn_model.h5')  

st.title('Customer Churn Prediction')

# Input fields
credit_score = st.sidebar.number_input('Credit Score', min_value=100, max_value=1000, step=1)
geography = st.sidebar.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.number_input('Age', min_value=18, max_value=100, step=1)
tenure = st.sidebar.number_input('Tenure (in years)', min_value=0, max_value=10, step=1)
balance = st.sidebar.number_input('Account Balance', min_value=0.0, step=0.01)
num_products = st.sidebar.number_input('Number of Products', min_value=1, max_value=4, step=1)
has_cr_card = st.sidebar.selectbox('Has Credit Card?', ['Yes', 'No'])
is_active_member = st.sidebar.selectbox('Is Active Member?', ['Yes', 'No'])
estimated_salary = st.sidebar.number_input('Estimated Salary (in Euros)', min_value=0.0, step=0.01)

# Predict button
if st.button('Predict'):
    # Convert categorical data
    geography_encoded = [1, 0, 0]  # Default encoding for 'France'
    if geography == 'Spain':
        geography_encoded = [0, 1, 0]
    elif geography == 'Germany':
        geography_encoded = [0, 0, 1]

    # Encode gender, has_cr_card, and is_active_member
    gender_encoded = 1 if gender == 'Male' else 0
    has_cr_card_encoded = 1 if has_cr_card == 'Yes' else 0
    is_active_member_encoded = 1 if is_active_member == 'Yes' else 0

    features = np.array([[*geography_encoded, credit_score, gender_encoded, age, tenure, balance, num_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary]])
    prediction = model.predict(features)
    prediction = np.argmax(prediction, axis=1)
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")