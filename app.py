import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# ------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint expecting a JSON payload."""
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    try:
        # Parse JSON request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400

        # Convert input payload to DataFrame (handles single object or list of objects)
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format. Provide a JSON object or array."}), 400

        # Generate predictions
        predictions = model.predict(input_df).tolist()

        # Get probabilities if the model supports it
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df).tolist()

        response = {
            "predictions": predictions
        }
        if probabilities:
            response["probabilities"] = probabilities

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Run dev server
    app.run(host='0.0.0.0', port=5000, debug=True)
