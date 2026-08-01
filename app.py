import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def load_model():
    """Loads the pre-trained RandomForestClassifier model."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# Standard web interface rendering without HTML page files
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Eligibility AI Predictor</title>
</head>
<body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px;">

    <h2>Loan Approval Prediction Engine</h2>

    <form action="/predict-form" method="POST" style="max-width: 600px; background: #1e293b; padding: 20px; border-radius: 8px;">
        <h3>Applicant Info</h3>
        
        <label>Loan ID:</label><br>
        <input type="number" name="loan_id" value="1001" required><br><br>

        <label>Number of Dependents:</label><br>
        <input type="number" name="no_of_dependents" value="2" min="0" max="10" required><br><br>

        <label>Education Status:</label><br>
        <select name="education">
            <option value="0">Graduate</option>
            <option value="1">Not Graduate</option>
        </select><br><br>

        <label>Employment Status:</label><br>
        <select name="self_employed">
            <option value="0">Salaried / Employed</option>
            <option value="1">Self Employed</option>
        </select><br><br>

        <label>CIBIL Score (300 - 900):</label><br>
        <input type="number" name="cibil_score" value="750" min="300" max="900" required><br><br>

        <h3>Financial Info</h3>

        <label>Annual Income ($):</label><br>
        <input type="number" name="income_annum" value="6500000" required><br><br>

        <label>Requested Loan Amount ($):</label><br>
        <input type="number" name="loan_amount" value="15000000" required><br><br>

        <label>Loan Term (Years):</label><br>
        <input type="number" name="loan_term" value="12" required><br><br>

        <h3>Declared Asset Values</h3>

        <label>Residential Assets ($):</label><br>
        <input type="number" name="residential_assets_value" value="4000000" required><br><br>

        <label>Commercial Assets ($):</label><br>
        <input type="number" name="commercial_assets_value" value="2500000" required><br><br>

        <label>Luxury Assets ($):</label><br>
        <input type="number" name="luxury_assets_value" value="8000000" required><br><br>

        <label>Bank Assets ($):</label><br>
        <input type="number" name="bank_asset_value" value="5000000" required><br><br>

        <button type="submit" style="background: #4f46e5; color: white; padding: 10px 20px; border: none; cursor: pointer;">Submit Request</button>
    </form>

    {% if result %}
    <div style="margin-top: 20px; padding: 20px; background: #334155; border-radius: 8px; max-width: 600px;">
        <h3>Prediction Result</h3>
        <p><strong>Status:</strong> {{ result.status }}</p>
        <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
        <p><strong>Approval Probability:</strong> {{ result.probability_approved }}%</p>
        <p><strong>Rejection Probability:</strong> {{ result.probability_rejected }}%</p>
    </div>
    {% endif %}

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_TEMPLATE, result=None)

@app.route('/predict-form', methods=['POST'])
def predict_form():
    result = run_prediction(request.form)
    return render_template_string(INDEX_TEMPLATE, result=result)

@app.route('/predict', methods=['POST'])
def predict_json():
    data = request.get_json(force=True) if request.is_json else request.form
    result = run_prediction(data)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

def run_prediction(data):
    try:
        if model is None:
            return {'error': 'Model file not found. Place model.pkl in root directory.'}

        features = [
            float(data.get('loan_id', 1)),
            float(data.get('no_of_dependents', 0)),
            float(data.get('education', 0)),
            float(data.get('self_employed', 0)),
            float(data.get('income_annum', 0)),
            float(data.get('loan_amount', 0)),
            float(data.get('loan_term', 0)),
            float(data.get('cibil_score', 0)),
            float(data.get('residential_assets_value', 0)),
            float(data.get('commercial_assets_value', 0)),
            float(data.get('luxury_assets_value', 0)),
            float(data.get('bank_asset_value', 0))
        ]

        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        confidence = round(float(np.max(probabilities)) * 100, 2)
        status = "Approved" if prediction == 1 else "Rejected"

        return {
            'status': status,
            'prediction': int(prediction),
            'confidence': confidence,
            'probability_approved': round(float(probabilities[1]) * 100, 2) if len(probabilities) > 1 else 100.0,
            'probability_rejected': round(float(probabilities[0]) * 100, 2) if len(probabilities) > 1 else 0.0
        }

    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
