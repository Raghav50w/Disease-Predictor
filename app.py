"""
Multi-Disease Predictor Flask Backend.

Dynamically loads all trained models from the models/ folder and
exposes endpoints for disease prediction.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, static_folder="static")
CORS(app)

# Structure: diseases = {
#   "diabetes": { 
#       "scaler": ..., 
#       "features": [...],
#       "models": { "random_forest": ..., "naive_bayes": ..., ... }
#   },
#   ...
# }
diseases = {}

def load_all_models():
    """Scan models/ folder and load all disease models dynamically."""
    global diseases
    diseases = {}

    scaler_files = glob.glob(os.path.join(MODELS_DIR, "*_scaler.pkl"))
    
    for scaler_path in sorted(scaler_files):
        disease_name = os.path.basename(scaler_path).replace("_scaler.pkl", "")
        features_path = os.path.join(MODELS_DIR, f"{disease_name}_features.pkl")

        if not os.path.exists(features_path):
            continue

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(features_path, "rb") as f:
            features = pickle.load(f)

        diseases[disease_name] = {
            "scaler": scaler,
            "features": features,
            "models": {}
        }
        
        # Load all available models for this disease
        model_files = glob.glob(os.path.join(MODELS_DIR, f"{disease_name}_*.pkl"))
        for model_path in model_files:
            if "scaler" in model_path or "features" in model_path:
                continue
            
            # Extract model type (e.g., "random_forest")
            basename = os.path.basename(model_path)
            model_type = basename.replace(f"{disease_name}_", "").replace(".pkl", "")
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                diseases[disease_name]["models"][model_type] = model

        print(f"Loaded: {disease_name} (Models: {list(diseases[disease_name]['models'].keys())})")

# Load models on startup
load_all_models()


# -----------------------------------------------
# Routes
# -----------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/diseases", methods=["GET"])
def get_diseases():
    """Returns available diseases, their input fields, and available models."""
    result = {}
    for name, data in diseases.items():
        display_name = name.replace("_", " ").title()
        models_available = list(data["models"].keys())
        
        result[name] = {
            "display_name": display_name,
            "features": data["features"],
            "models": models_available
        }
    return jsonify(result)

@app.route("/predict", methods=["POST"])
def predict():
    """Predict risk for a single disease using a selected model."""
    data = request.get_json()
    disease = data.get("disease")
    model_type = data.get("model_type", "random_forest") # Default to random_forest

    if not disease or disease not in diseases:
        return jsonify({"error": f"Unknown disease. Available: {list(diseases.keys())}"}), 400

    disease_data = diseases[disease]
    
    if model_type not in disease_data["models"]:
        return jsonify({"error": f"Model '{model_type}' not available for {disease}"}), 400

    features = disease_data["features"]
    model = disease_data["models"][model_type]
    scaler = disease_data["scaler"]

    try:
        values = [float(data.get(f, 0)) for f in features]
        X = np.array([values])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400

    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        probability = round(float(proba[1]) * 100, 1)

    return jsonify({
        "prediction": prediction,
        "label": "positive" if prediction == 1 else "negative",
        "disease": disease,
        "model_used": model_type,
        "probability": probability
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
