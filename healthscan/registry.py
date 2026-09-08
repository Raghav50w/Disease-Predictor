"""Loads the trained models from disk and runs predictions."""

import json
import pickle
from pathlib import Path

from healthscan.config import (
    DEFAULT_MODEL,
    DISEASE_META,
    METRICS_FILENAME,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
)


class ModelRegistry:
    """Holds every trained model in memory so predictions don't reload files."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.diseases = {}
        self.metrics = {}

    def load(self):
        """Read the artifacts for every disease listed in DISEASE_META."""
        for disease in DISEASE_META:
            features_path = self.models_dir / f"{disease}_features.json"
            scaler_path = self.models_dir / f"{disease}_scaler.json"

            features = json.loads(features_path.read_text(encoding="utf-8"))
            scaler = json.loads(scaler_path.read_text(encoding="utf-8"))

            models = {}
            for model_type in MODEL_DISPLAY_NAMES:
                model_path = self.models_dir / f"{disease}_{model_type}.pkl"
                with open(model_path, "rb") as f:
                    models[model_type] = pickle.load(f)

            self.diseases[disease] = {
                "features": features,
                "scaler": scaler,
                "models": models,
            }

        metrics_path = self.models_dir / METRICS_FILENAME
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    @property
    def is_ready(self):
        return len(self.diseases) > 0

    def default_model(self, disease):
        """The model the UI pre-selects, and the one used if the request omits it."""
        models = self.diseases[disease]["models"]
        if DEFAULT_MODEL in models:
            return DEFAULT_MODEL
        return sorted(models)[0]

    def catalog(self):
        """The /api/diseases payload: everything the frontend needs to draw a form."""
        catalog = {}

        for disease, artifacts in self.diseases.items():
            meta = DISEASE_META[disease]

            features = []
            for name in artifacts["features"]:
                feature = {"name": name}
                feature.update(meta["features"][name])
                features.append(feature)

            models = []
            for model_type in sorted(artifacts["models"]):
                models.append(
                    {
                        "name": model_type,
                        "display_name": MODEL_DISPLAY_NAMES[model_type],
                    }
                )

            catalog[disease] = {
                "display_name": meta["display_name"],
                "description": meta["description"],
                "features": features,
                "default_model": self.default_model(disease),
                "models": models,
            }

        return {"diseases": catalog}

    def predict(self, disease, model_type, values):
        """Run one prediction. `values` must already be validated, in feature order."""
        artifacts = self.diseases[disease]
        scaler = artifacts["scaler"]
        model = artifacts["models"][model_type]

        # Apply the same scaling the model was trained with.
        scaled = []
        for i, value in enumerate(values):
            scaled.append((value - scaler["mean"][i]) / scaler["scale"][i])

        prediction = int(model.predict([scaled])[0])

        # Perceptron has no predict_proba, so it reports no probability.
        probability = None
        if hasattr(model, "predict_proba"):
            positive = model.predict_proba([scaled])[0][1]
            probability = round(float(positive) * 100, 1)

        display_name = DISEASE_META[disease]["display_name"]
        if prediction == 1:
            label = "positive"
            level = "High"
        else:
            label = "negative"
            level = "Low"

        return {
            "disease": disease,
            "model_used": model_type,
            "prediction": prediction,
            "label": label,
            "summary": f"{level} risk of {display_name}",
            "probability": probability,
        }
