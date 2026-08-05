"""Artifact discovery, catalog construction, and inference."""

from __future__ import annotations

import numpy as np
import pytest

from healthscan.api.registry import InferenceError, ModelRegistry, RegistryError
from healthscan.config import DEFAULT_MODEL


def test_loads_all_trained_models(registry):
    assert registry.is_ready
    assert set(registry.diseases) == {"diabetes"}
    assert set(registry.diseases["diabetes"].models) == {
        "random_forest",
        "naive_bayes",
        "mlp",
        "perceptron",
    }


def test_scaler_and_features_are_not_loaded_as_models(registry):
    """`*_scaler.pkl` and `*_features.pkl` match the model glob too."""
    assert "scaler" not in registry.diseases["diabetes"].models
    assert "features" not in registry.diseases["diabetes"].models


def test_feature_order_is_preserved(registry):
    assert registry.diseases["diabetes"].features == ["Glucose", "Insulin", "BMI", "Age"]


def test_missing_models_directory_is_not_fatal(tmp_path):
    registry = ModelRegistry(tmp_path / "does-not-exist")
    registry.load()

    assert registry.is_ready is False
    assert registry.catalog()["diseases"] == {}


def test_incomplete_artifact_set_is_skipped(models_dir):
    """A scaler with no matching features file must not half-load."""
    (models_dir / "diabetes_features.pkl").unlink()

    registry = ModelRegistry(models_dir)
    registry.load()

    assert registry.diseases == {}


def test_unknown_disease_raises(registry):
    with pytest.raises(RegistryError, match="Unknown disease"):
        registry.get("malaria")


def test_unknown_model_raises(registry):
    with pytest.raises(RegistryError, match="not available"):
        registry.get_model("diabetes", "xgboost")


def test_broken_model_raises_inference_error(registry):
    """A model that blows up must surface as InferenceError, not a raw traceback."""

    class Broken:
        def predict(self, X):
            raise ValueError("shape mismatch")

    registry.diseases["diabetes"].models["broken"] = Broken()

    with pytest.raises(InferenceError):
        registry.predict("diabetes", "broken", [180.0, 100.0, 35.0, 50.0])


def test_predict_returns_a_binary_label(registry):
    result = registry.predict("diabetes", "random_forest", [180.0, 100.0, 35.0, 50.0])

    assert result["prediction"] in (0, 1)
    assert result["label"] in ("positive", "negative")
    assert result["disease"] == "diabetes"
    assert result["summary"].endswith("risk of Diabetes")


def test_probability_is_a_percentage(registry):
    result = registry.predict("diabetes", "random_forest", [180.0, 100.0, 35.0, 50.0])
    assert 0.0 <= result["probability"] <= 100.0


def test_perceptron_has_no_probability(registry):
    """Perceptron exposes no predict_proba; the API must return null, not crash."""
    result = registry.predict("diabetes", "perceptron", [180.0, 100.0, 35.0, 50.0])
    assert result["probability"] is None


def test_probability_uses_classes_not_column_one(registry):
    """Positive-class probability is indexed via classes_, not hardcoded [1]."""

    class ReversedClasses:
        classes_ = np.array([1, 0])

        def predict(self, X):
            return np.array([1])

        def predict_proba(self, X):
            # Positive class sits in column 0 here.
            return np.array([[0.9, 0.1]])

    registry.diseases["diabetes"].models["reversed"] = ReversedClasses()
    result = registry.predict("diabetes", "reversed", [180.0, 100.0, 35.0, 50.0])

    assert result["probability"] == 90.0


def test_catalog_exposes_feature_metadata(registry):
    catalog = registry.catalog()
    diabetes = catalog["diseases"]["diabetes"]

    assert diabetes["display_name"] == "Diabetes"

    glucose = next(f for f in diabetes["features"] if f["name"] == "Glucose")
    assert glucose["label"] == "Plasma Glucose"
    assert glucose["unit"] == "mg/dL"
    assert glucose["min"] == 40
    assert glucose["max"] == 400


def test_catalog_default_model_matches_the_api_default(registry):
    """The UI pre-selects this; it must be what /api/predict uses when omitted."""
    assert registry.catalog()["diseases"]["diabetes"]["default_model"] == DEFAULT_MODEL


def test_default_model_falls_back_when_not_trained(registry):
    del registry.diseases["diabetes"].models[DEFAULT_MODEL]

    fallback = registry.default_model("diabetes")

    assert fallback in registry.diseases["diabetes"].models


def test_catalog_attaches_holdout_metrics(registry):
    models = registry.catalog()["diseases"]["diabetes"]["models"]
    forest = next(m for m in models if m["name"] == "random_forest")

    assert forest["display_name"] == "Random Forest"
    assert 0.0 <= forest["metrics"]["accuracy"] <= 1.0


def test_catalog_handles_absent_metrics(models_dir):
    (models_dir / "metrics.json").unlink()

    registry = ModelRegistry(models_dir)
    registry.load()

    models = registry.catalog()["diseases"]["diabetes"]["models"]
    assert all(model["metrics"] is None for model in models)
