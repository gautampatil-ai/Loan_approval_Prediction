import os
import pickle
import streamlit as st
import pandas as pd

st.title("Loan Approval Prediction")

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Create input UI form
if model:
    st.subheader("Enter Loan Details")
    
    # Add input fields matching your model's expected features
    # Example fields:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

    if st.button("Predict"):
        # Construct DataFrame
        input_data = pd.DataFrame([{
            'ApplicantIncome': applicant_income,
            'LoanAmount': loan_amount,
            'Credit_History': credit_history,
            # Add remaining features your model needs...
        }])
        
        prediction = model.predict(input_data)[0]
        st.write(f"### Prediction Result: **{prediction}**")
