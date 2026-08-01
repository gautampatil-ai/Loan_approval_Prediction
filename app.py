import os
import pickle
import pandas as pd
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Enterprise Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for professional UI
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #64748B; margin-bottom: 2rem; }
    .card { background-color: #F8FAFC; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# 1. Cached Model Loading Function
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found at `{model_path}`. Please run `train.py` first.")
        st.stop()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

pipeline = load_pipeline()

# Header Section
st.markdown('<div class="main-header">🏦 Enterprise Loan Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluate applicant default probability and approval confidence in real-time.</div>', unsafe_allow_html=True)

# 2. Sidebar Input Form
st.sidebar.header("📋 Applicant Profile")

with st.sidebar.form(key="applicant_form"):
    st.subheader("Personal & Demographic")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
    education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Employment Status", options=["No", "Yes"], format_func=lambda x: "Self Employed" if x == "Yes" else "Salaried")
    
    st.subheader("Financial Metrics")
    income_annum = st.number_input("Annual Income ($)", min_value=0, value=7500000, step=50000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000000, step=50000)
    loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=12, step=1)
    cibil_score = st.slider("CIBIL Credit Score", min_value=300, max_value=900, value=750, step=1)

    st.subheader("Valued Assets ($)")
    residential_assets_value = st.number_input("Residential Asset Value", min_value=0, value=4000000, step=25000)
    commercial_assets_value = st.number_input("Commercial Asset Value", min_value=0, value=2500000, step=25000)
    luxury_assets_value = st.number_input("Luxury Asset Value", min_value=0, value=8000000, step=25000)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=5000000, step=25000)

    submit_button = st.form_submit_button(label="Evaluate Loan Eligibility")

# 3. Prediction & Output Display
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Entered Profile Summary")
    summary_df = pd.DataFrame({
        "Metric": ["Dependents", "Education", "Self Employed", "Annual Income", "Loan Requested", "Loan Term", "CIBIL Score"],
        "Value": [
            no_of_dependents, education, "Yes" if self_employed == "Yes" else "No",
            f"${income_annum:,}", f"${loan_amount:,}", f"{loan_term} Years", cibil_score
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Model Evaluation Result")

    if submit_button:
        try:
            # Construct DataFrame matching training dataset structure
            input_data = pd.DataFrame([{
                'loan_id': 1001,
                'no_of_dependents': int(no_of_dependents),
                'education': str(education),      # Pass string directly: "Graduate" or "Not Graduate"
                'self_employed': str(self_employed), # Pass string directly: "Yes" or "No"
                'income_annum': float(income_annum),
                'loan_amount': float(loan_amount),
                'loan_term': float(loan_term),
                'cibil_score': float(cibil_score),
                'residential_assets_value': float(residential_assets_value),
                'commercial_assets_value': float(commercial_assets_value),
                'luxury_assets_value': float(luxury_assets_value),
                'bank_asset_value': float(bank_asset_value)
            }])

            # Execute pipeline prediction
            prediction = pipeline.predict(input_data)[0]
            
            # Predict Probabilities
            has_proba = hasattr(pipeline, "predict_proba")
            if has_proba:
                probabilities = pipeline.predict_proba(input_data)[0]
                prob_approved = probabilities[1]
                prob_rejected = probabilities[0]

            # Render UI based on Prediction
            if prediction == 1:
                st.success("🎉 **LOAN APPLICATION APPROVED**")
                if has_proba:
                    st.metric(label="Approval Confidence", value=f"{prob_approved * 100:.1f}%")
                    st.progress(float(prob_approved))
                st.info("Applicant meets credit risk standards and asset valuation thresholds.")
            else:
                st.error("❌ **LOAN APPLICATION REJECTED**")
                if has_proba:
                    st.metric(label="Rejection Risk Probability", value=f"{prob_rejected * 100:.1f}%")
                    st.progress(float(prob_rejected))
                st.warning("Applicant does not meet minimum risk criteria or asset ratio bounds.")

        except Exception as e:
            st.error(f"An error occurred during evaluation: {str(e)}")
            st.exception(e)
    else:
        st.info("👈 Adjust applicant parameters in the sidebar and click **Evaluate Loan Eligibility**.")
