import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional Data Science aesthetic
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
    .card {
        background-color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Load Trained Model Assets
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    try:
        with open("model_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], data["encoders"], data["feature_names"]
    except FileNotFoundError:
        st.error("Error: `model_data.pkl` file not found. Please train and save the model first.")
        st.stop()

model, encoders, feature_names = load_model_assets()


# -----------------------------------------------------------------------------
# 3. Header & Sidebar Section
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🏦 Smart Loan Eligibility Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Risk Assessment Engine using XGBoost</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=80)
    st.title("Model Insights")
    st.info("""
    **Algorithm:** XGBoost Classifier  
    **Trees:** 300 | **Max Depth:** 6  
    **Learning Rate:** 0.05  
    """)
    st.divider()
    st.write("💡 *Provide borrower financial details in the main panel to compute approval likelihood.*")


# -----------------------------------------------------------------------------
# 4. User Input Form
# -----------------------------------------------------------------------------
st.subheader("📋 Borrower Information")

col1, col2 = st.columns(2, gap="large")

# Mapping dict for user inputs to map back using LabelEncoders
input_data = {}

with col1:
    st.markdown("#### 👤 Demographics & Employment")
    
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
    
    # Categorical Inputs (handling label encoding automatically)
    education_opts = list(encoders['education'].classes_) if 'education' in encoders else [" Graduate", " Not Graduate"]
    education = st.selectbox("Education Level", options=education_opts)
    
    self_employed_opts = list(encoders['self_employed'].classes_) if 'self_employed' in encoders else [" No", " Yes"]
    self_employed = st.selectbox("Self Employed?", options=self_employed_opts)
    
    income_annum = st.number_input("Annual Income ($)", min_value=0, value=500000, step=10000)


with col2:
    st.markdown("#### 💰 Financials & Collateral")
    
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=1500000, step=10000)
    loan_term = st.slider("Loan Term (Years)", min_value=1, max_value=30, value=10)
    cibil_score = st.slider("CIBIL / Credit Score", min_value=300, max_value=900, value=750)
    
    st.markdown("##### Assets Valuation")
    asset_col_a, asset_col_b = st.columns(2)
    with asset_col_a:
        residential_assets = st.number_input("Residential Assets ($)", min_value=0, value=200000, step=5000)
        commercial_assets = st.number_input("Commercial Assets ($)", min_value=0, value=100000, step=5000)
    with asset_col_b:
        luxury_assets = st.number_input("Luxury Assets ($)", min_value=0, value=50000, step=5000)
        bank_asset_value = st.number_input("Bank Asset Value ($)", min_value=0, value=100000, step=5000)


# -----------------------------------------------------------------------------
# 5. Data Processing & Prediction Logic
# -----------------------------------------------------------------------------
# Assemble input dataframe matching feature names
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

# Encode categorical values using stored LabelEncoders
for col, encoder in encoders.items():
    if col in df_input.columns:
        df_input[col] = encoder.transform(df_input[col])

# Ensure correct feature ordering
df_input = df_input[feature_names]

st.divider()

# Prediction Action
if st.button("🚀 Evaluate Loan Application", use_container_width=True, type="primary"):
    
    prediction = model.predict(df_input)[0]
    probabilities = model.predict_proba(df_input)[0]

    # Convert numeric output to class label if encoder exists
    if 'loan_status' in encoders:
        status_label = encoders['loan_status'].inverse_transform([prediction])[0].strip()
    else:
        status_label = "Approved" if prediction == 1 else "Rejected"

    # Display Results
    st.subheader("📊 Evaluation Results")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if "Approved" in status_label or prediction == 1:
            st.success(f"### Result: APPROVED ✅")
            st.balloons()
        else:
            st.error(f"### Result: REJECTED ❌")

    with res_col2:
        approval_prob = probabilities[1] * 100 if len(probabilities) > 1 else 100
        st.write(f"**Approval Confidence Rate:** `{approval_prob:.2f}%`")
        st.progress(int(approval_prob))

    # Detailed Summary Card
    st.markdown("---")
    st.markdown("#### 🔍 Application Summary Overview")
    metric_c1, metric_c2, metric_c3 = st.columns(3)
    
    total_assets = residential_assets + commercial_assets + luxury_assets + bank_asset_value
    debt_to_income = round(loan_amount / max(income_annum, 1), 2)

    metric_c1.metric("Credit Score Rating", f"{cibil_score}", delta="Good" if cibil_score >= 700 else "Low")
    metric_c2.metric("Total Asset Backing", f"${total_assets:,}")
    metric_c3.metric("Loan-to-Income Ratio", f"{debt_to_income}x")
