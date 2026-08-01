import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Load model cleanly from file
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("`model.pkl` file not found in directory!")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏦 Loan Risk Scoring Dashboard")
st.write("Predict loan approval status using machine learning.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        loan_id = st.number_input("Loan ID", value=1001)
        no_of_dependents = st.number_input("Dependents", value=2, min_value=0)
        education = st.selectbox("Education", options=[(1, "Graduate"), (0, "Not Graduate")], format_func=lambda x: x[1])[0]
        self_employed = st.selectbox("Employment Status", options=[(0, "Salaried"), (1, "Self Employed")], format_func=lambda x: x[1])[0]
        income_annum = st.number_input("Annual Income ($)", value=7500000)
        loan_amount = st.number_input("Loan Amount ($)", value=15000000)
    
    with col2:
        loan_term = st.number_input("Loan Term (Years)", value=12)
        cibil_score = st.number_input("CIBIL Score", value=750, min_value=300, max_value=900)
        residential_assets = st.number_input("Residential Asset Value ($)", value=4000000)
        commercial_assets = st.number_input("Commercial Asset Value ($)", value=2500000)
        luxury_assets = st.number_input("Luxury Asset Value ($)", value=8000000)
        bank_assets = st.number_input("Bank Asset Value ($)", value=5000000)

    submitted = st.form_submit_button("Predict Approval Status")

if submitted:
    features = np.array([[
        loan_id, no_of_dependents, education, self_employed,
        income_annum, loan_amount, loan_term, cibil_score,
        residential_assets, commercial_assets, luxury_assets, bank_assets
    ]])
    
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    st.divider()
    if pred == 1:
        st.success(f"✅ **LOAN APPROVED** (Confidence: {proba[1]*100:.1f}%)")
    else:
        st.error(f"❌ **LOAN REJECTED** (Risk Probability: {proba[0]*100:.1f}%)")
