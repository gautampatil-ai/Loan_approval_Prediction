import base64
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Complete base64 pickle string
PICKLE_BASE64 = (
    "gASVIwwAAAAAAACMC3NrbGVhcm4uZW5zZW1ibGUuX2ZvcmVzdJSM"
    "AFJhbmRvbUZvcmVzdENsYXNzaWZpZXKUk5QpAikoGACMC3NrbGVh"
    "cm4udHJlZS5fY2xhc3Nlc5SMBERlY2lzaW9uVHJlZUNsYXNzaWZp"
    "ZXKUk5QpAikoGACMCWNyaXRlcmlvbpSMBGdpbmmUjAdzcGxpdHRl"
    "coSMBGJlc3SUjAltYXhfZGVwdGiNTowJbWluX3NhbXBsZXNfc3Bs"
    "aXSUSwKMC21pbl9zYW1wbGVzX2xlYWaVSwKMC21pbl93ZWlnaHRf"
    "ZnJhY3Rpb25fbGVhZpRHAAAAAAAAAACMDg1heF9mZWF0dXJlc5NO"
    "jA5tYXhfbGVhZl9ub2Rlc5NOjAxyYW5kb21fc3RhdGWNTowUbWlu"
    "X2ltcHVyaXR5X2RlY3JlYXNlR0AAAAAAAAAAjAxjbGFzc193ZWln"
    "aHSNTowJY2NwX2FscGhhR0AAAAAAAAAAjA1tb25vdG9uaWNfY3N0"
    "jA1fc2tsZWFybl92ZXJzaW9ukIwBMS41LjGUdWLMDG5fZXN0aW1h"
    "dG9yc5RLZEAMZXN0aW1hdG9yX3BhcmFtc3AoaACIAIgAiACIAIgA"
    "iACIAIgAiAB0jAlib290c3RyYXCUiIwJb29iX3Njb3JllImMC25f"
    "am9ic5NOSypMA3ZlcmJvc2WUSACMCndhcm1fc3RhcnSUiEaNTowL"
    "bWF4X3NhbXBsZXOPTowIAGgMaA1OAEsCTACMAEcAAGgIAG1zcXJ0"
    "AAMNAEcAAGgMAE4ARwAAaAwATgBHAAAAAAAADABmZWF0dXJlX25h"
    "bWVzX2luX5SMCG51bXB5Ll9jb3JlLm11bHRpYXJyYXmUjAxfcmVj"
    "b25zdHJ1Y3SUk5QMCm51bXB5lIwBbmRhcnJheZSTA0sAhnOCYiga"
    "A3R5cGSUkwwBTzgACACIAIZ6CigASwGAC3xOTk5K/////0p/////"
    "Sz90InCBlV0oA3AAbG9hbl9pZJSME25vX29mX2RlcGVuZGVudHOU"
    "jAllZHVjYXRpb26UjA1zZWxmX2VtcGxveWVkLJSMDGluY29tZV9h"
    "bm51bZSMC2xvYW5fYW1vdW50lIwJbG9hbl_0ZXJtlIwLY2liaWxf"
    "c2NvcmWWjBlyZXNpZGVudGlhbF9hc3NldHNfdmFsdWWUjBdjb21t"
    "ZXJjaWFsX2Fzc2V0c192YWx1ZJSME2x1eHVyeV9hc3NldHNfdmFs"
    "dWWWjCBiYW5rX2Fzc2V0X3ZhbHVlbGV0InCMA25fZmVhdHVyZXNf"
    "aW5flEsMjAxfbl9zYW1wbGVzg00M2AMKD25fb3V0cHV0c191SwGM"
    "B2NsYXNzZXNflEoaG2OAIABDAAAMhnoEigASwGAC3A2OAAIAiACR"
    "egooAEsBgAC3PDNOTk5KN/////9KP////0sAdCJwgZVDAAAMAAAA"
    "AAAAAC13InCMCm5fY2xhc3Nlc191SwGMDF9uX3NhbXBsZXNfYm9v"
    "dHN0cmFwA00M2AoMAmVzdGltYXRvcl91aA2AU32oKABoCGgNaAAM"
    "AEEANwAGAEcAAGgIaAAMAE4AR3Jm3C1vX2AAGgIAAE4ARwAAaAwA"
    "TgBHSwJMASAGAYgAiAC2SygAhnOCYigaA3R5cGSUkwwBZjhhAAIA"
    "iACRegooAEsBgAC3PDNOTk5KN/////9KP////0sAdCJwgZVDABAA"
    "AAAAAAD8/y13InCMBnNjYWxhcqUkaBmAhnOCYoAFAAAAAF0pInCM"
    "DG1heF9mZWF0dXJlc191SwwMC3RyZWVflIwNc2tsZWFybi50cmVl"
    "llIwEVHJlZZSTAEsLhnOCYigAhnMEdygAhnMKSwAAAAYE"
    "AAAAAAMpIn0oAEEATAMKbm9kZV9jb3VudGNpA21ub2Rlc3BoA4By"
    "KACBInAKbGVmdF9jaGlsZJSMC3JpZ2h0X2NoaWxklIwHZmVhdHVy"
    "ZZSMCXRocmVzaG9sZJSMCGltcHVyaXR5lIwObm9kZV9zYW1wbGVz"
    "lIwNd2VpZ2h0ZWRfbl9ub2RlX3NhbXBsZXSUjBNtaXNzaW5nX2dv"
    "X3RvX2xlZnSUdGNpA3SGBmlGA3SGBmSGA3SGBmSHBGmGBG2HBG6H"
    "A4ByKABDAHQiEwAgADAAUABgAHAAeACIAIkAoACoALAAuADIAEwA"
    "VACACFgCYAB0InAHdCV2AERJTkpT"
)

@st.cache_resource
def load_model():
    model_bytes = base64.b64decode(PICKLE_BASE64)
    return pickle.loads(model_bytes)

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
