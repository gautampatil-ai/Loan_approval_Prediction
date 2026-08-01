import os
import pickle
import numpy as np
import streamlit as st

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")

if model is None:
    st.error("Model file `model.pkl` not found!")
    st.stop()

# Inputs
col1, col2 = st.columns(2)

with col1:
    loan_id = st.number_input("Loan ID", value=1001)
    dependents = st.number_input("Dependents", value=2, min_value=0, max_value=10)
    education = st.selectbox("Education", options=[0, 1], format_func=lambda x: "Graduate" if x == 0 else "Not Graduate")
    self_employed = st.selectbox("Employment", options=[0, 1], format_func=lambda x: "Salaried" if x == 0 else "Self Employed")
    cibil_score = st.slider("CIBIL Score", 300, 900, 750)
    income_annum = st.number_input("Annual Income ($)", value=6500000)

with col2:
    loan_amount = st.number_input("Loan Amount ($)", value=15000000)
    loan_term = st.number_input("Loan Term (Years)", value=12)
    res_assets = st.number_input("Residential Assets ($)", value=4000000)
    comm_assets = st.number_input("Commercial Assets ($)", value=2500000)
    lux_assets = st.number_input("Luxury Assets ($)", value=8000000)
    bank_assets = st.number_input("Bank Assets ($)", value=5000000)

if st.button("Predict Eligibility", type="primary"):
 import pandas as pd

features = pd.DataFrame([{
    "loan_id": loan_id,
    "no_of_dependents": dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": res_assets,
    "commercial_assets_value": comm_assets,
    "luxury_assets_value": lux_assets,
    "bank_asset_value": bank_assets
}])
    
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {max(probs)*100:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected. (Confidence: {max(probs)*100:.2f}%)")
