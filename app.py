from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os

MODEL_NB_PATH = "naive_bayes_model.pkl"
MODEL_MLP_PATH = "mlp_model.pkl"
SCALER_PATH = "scaler.pkl"

app = Flask(__name__, static_folder="static")
CORS(app)

if os.path.exists(MODEL_NB_PATH):
    loaded_nb = pickle.load(open(MODEL_NB_PATH, "rb"))
if os.path.exists(MODEL_MLP_PATH):
    loaded_mlp = pickle.load(open(MODEL_MLP_PATH, "rb"))
if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, "rb"))

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not (os.path.exists(MODEL_NB_PATH) and os.path.exists(MODEL_MLP_PATH) and os.path.exists(SCALER_PATH)):
        return jsonify({"error": "Models not found. Train models first."}), 500

    data = request.get_json()
    try:
        age = float(data["age"])
        glucose = float(data["glucose"])
        insulin = float(data["insulin"])
        bmi = float(data["bmi"])
    except:
        return jsonify({"error": "Invalid or missing input values"}), 400

    model_type = data.get("model_type", "naive_bayes")
    features = np.array([[age, glucose, insulin, bmi]])

    if model_type == "mlp":
        pred = loaded_mlp.predict(scaler.transform(features))
    elif model_type == "naive_bayes":
        pred = loaded_nb.predict(features)
    else:
        return jsonify({"error": "Invalid model_type"}), 400

    result = int(pred[0])
    label = "diabetic" if result == 1 else "non-diabetic"

    return jsonify({"prediction": result, "label": label})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)


