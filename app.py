import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Self-Contained In-Memory Model Trainer & Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    # 1. Generate clean training data matching dataset schema
    np.random.seed(42)
    n_samples = 1200

    data = {
        "no_of_dependents": np.random.randint(0, 6, n_samples),
        "education": np.random.choice([" Graduate", " Not Graduate"], n_samples),
        "self_employed": np.random.choice([" No", " Yes"], n_samples),
        "income_annum": np.random.randint(200000, 9900000, n_samples),
        "loan_amount": np.random.randint(300000, 39500000, n_samples),
        "loan_term": np.random.randint(2, 21, n_samples),
        "cibil_score": np.random.randint(300, 901, n_samples),
        "residential_assets_value": np.random.randint(0, 29100000, n_samples),
        "commercial_assets_value": np.random.randint(0, 19400000, n_samples),
        "luxury_assets_value": np.random.randint(200000, 39200000, n_samples),
        "bank_asset_value": np.random.randint(0, 14700000, n_samples),
    }

    df = pd.DataFrame(data)

    # Business rule logic for synthetic dataset target creation
    loan_status_prob = (df["cibil_score"] > 600) & (df["loan_amount"] < df["income_annum"] * 5)
    df["loan_status"] = np.where(loan_status_prob, " Approved", " Rejected")

    # Fit Encoders
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    # Train XGBoost Model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X, y)

    return model, encoders, X.columns.tolist()


# -----------------------------------------------------------------------------
# 3. NOW CALL THE FUNCTION (After it has been defined)
# -----------------------------------------------------------------------------
model, encoders, feature_names = load_model_assets()

if not feature_names:
    feature_names = [
        'no_of_dependents', 'education', 'self_employed', 'income_annum',
        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]


# -----------------------------------------------------------------------------
# 4. User Interface Inputs
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🏦 Smart Loan Eligibility Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Risk Assessment Engine using XGBoost</div>', unsafe_allow_html=True)

st.subheader("📋 Borrower Information")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 👤 Demographics & Employment")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
    
    if isinstance(encoders, dict) and 'education' in encoders and hasattr(encoders['education'], 'classes_'):
        education_opts = list(encoders['education'].classes_)
    else:
        education_opts = [" Graduate", " Not Graduate"]

    if isinstance(encoders, dict) and 'self_employed' in encoders and hasattr(encoders['self_employed'], 'classes_'):
        self_employed_opts = list(encoders['self_employed'].classes_)
    else:
        self_employed_opts = [" No", " Yes"]

    education = st.selectbox("Education Level", options=education_opts)
    self_employed = st.selectbox("Self Employed?", options=self_employed_opts)
    income_annum = st.number_input("Annual Income ($)", min_value=0, value=500000, step=10000)

with col2:
    st.markdown("#### 💰 Financials & Collateral")
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=1500000, step=10000)
    loan_term = st.slider("Loan Term (Years)", min_value=1, max_value=30, value=10)
    cibil_score = st.slider("CIBIL / Credit Score", min_value=300, max_value=900, value=750)
    
    asset_col_a, asset_col_b = st.columns(2)
    with asset_col_a:
        residential_assets = st.number_input("Residential Assets ($)", min_value=0, value=200000, step=5000)
        commercial_assets = st.number_input("Commercial Assets ($)", min_value=0, value=100000, step=5000)
    with asset_col_b:
        luxury_assets = st.number_input("Luxury Assets ($)", min_value=0, value=50000, step=5000)
        bank_asset_value = st.number_input("Bank Asset Value ($)", min_value=0, value=100000, step=5000)


# -----------------------------------------------------------------------------
# 5. Prediction Execution
# -----------------------------------------------------------------------------
input_dict = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets,
    'commercial_assets_value': commercial_assets,
    'luxury_assets_value': luxury_assets,
    'bank_asset_value': bank_asset_value
}

df_input = pd.DataFrame([input_dict])

for col in ['education', 'self_employed']:
    if isinstance(encoders, dict) and col in encoders and hasattr(encoders[col], 'transform'):
        try:
            df_input[col] = encoders[col].transform(df_input[col])
        except Exception:
            df_input[col] = 0
    else:
        df_input[col] = df_input[col].map({' Graduate': 0, ' Not Graduate': 1, ' No': 0, ' Yes': 1}).fillna(0)

df_input = df_input[feature_names]
st.divider()

if st.button("🚀 Evaluate Loan Application", use_container_width=True, type="primary"):
    prediction = model.predict(df_input)[0]
    probabilities = model.predict_proba(df_input)[0]

    st.subheader("📊 Evaluation Results")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1 or str(prediction).strip() in ['1', 'Approved', ' Approved']:
            st.success("### Result: APPROVED ✅")
            st.balloons()
        else:
            st.error("### Result: REJECTED ❌")

    with res_col2:
        approval_prob = probabilities[1] * 100 if len(probabilities) > 1 else 100
        st.write(f"**Approval Probability:** `{approval_prob:.2f}%`")
        st.progress(int(approval_prob))
