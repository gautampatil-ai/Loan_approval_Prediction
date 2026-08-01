import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

print("⏳ Generating synthetic training data based on your dataset schema...")

# 1. Create realistic dummy data matching your schema
np.random.seed(42)
n_samples = 1000

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

# Target heuristic: higher CIBIL & lower loan-to-income ratio -> approved
loan_status_prob = (df["cibil_score"] > 600) & (
    df["loan_amount"] < df["income_annum"] * 5
)
df["loan_status"] = np.where(loan_status_prob, " Approved", " Rejected")

# 2. Fit Encoders
encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. Features & Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 4. Train XGBoost Model
print("🧠 Training XGBoost Model...")
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42
)
model.fit(X, y)

# 5. Save everything to model_data.pkl
model_data = {
    "model": model,
    "encoders": encoders,
    "feature_names": X.columns.tolist(),
}

with open("model_data.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Success! 'model_data.pkl' generated successfully in your folder.")
