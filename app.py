import streamlit as st
import pandas as pd
import pickle


# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Loan Approval Prediction")


# User Inputs
loan_id = st.number_input("Loan ID", value=1)

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.number_input(
    "Loan Term",
    min_value=0
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=0,
    max_value=900
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0
)


# Prediction
if st.button("Predict"):

    features = pd.DataFrame([{
        "loan_id": loan_id,
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])


    prediction = model.predict(features)[0]


    if prediction == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
