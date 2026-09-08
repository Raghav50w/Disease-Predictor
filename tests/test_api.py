"""The HTTP endpoints, and the input rules they enforce."""

import pytest

from healthscan.config import DISEASE_META
from healthscan.schemas import validate_features


def test_predict_happy_path(client, valid_payload):
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["model_used"] == "random_forest"
    assert body["summary"] in ("Low risk of Diabetes", "High risk of Diabetes")


@pytest.mark.parametrize("model_type", ["random_forest", "naive_bayes", "mlp", "perceptron"])
def test_every_model_serves_predictions(client, valid_payload, model_type):
    valid_payload["model_type"] = model_type
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["model_used"] == model_type


def test_diseases_catalog_shape(client):
    """The payload the whole frontend is built from."""
    response = client.get("/api/diseases")

    assert response.status_code == 200
    diseases = response.json()["diseases"]

    # Diabetes is listed first, so it is the one the dropdown opens on.
    assert list(diseases)[0] == "diabetes"

    diabetes = diseases["diabetes"]
    assert [f["name"] for f in diabetes["features"]] == ["Glucose", "Insulin", "BMI", "Age"]
    assert {m["name"] for m in diabetes["models"]} == {
        "random_forest",
        "naive_bayes",
        "mlp",
        "perceptron",
    }


def test_missing_field_returns_400(client, valid_payload):
    del valid_payload["Glucose"]
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 400
    assert "Glucose" in response.json()["fields"]


def test_out_of_range_returns_400(client, valid_payload):
    valid_payload["Glucose"] = 99999
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 400
    assert "at most" in response.json()["fields"]["Glucose"]


def test_unknown_disease_returns_404(client, valid_payload):
    valid_payload["disease"] = "malaria"
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 404
    assert "Unknown disease" in response.json()["error"]


def test_index_serves_the_frontend(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"HealthScan" in response.content


def test_returns_values_in_model_feature_order(diabetes_features):
    """Feature order is the one bug that fails silently.

    Sent in the wrong order the model still answers, it is just answering a
    different question, so nothing errors and the result is quietly wrong.
    """
    payload = {"BMI": 28.4, "Age": 45, "Glucose": 120, "Insulin": 85}

    values, errors = validate_features("diabetes", diabetes_features, payload)

    assert errors == {}
    assert values == [120.0, 85.0, 28.4, 45.0]


@pytest.mark.parametrize("disease", sorted(DISEASE_META))
def test_configured_defaults_pass_their_own_validation(disease):
    """The form pre-fills every field with its `default`.

    A default outside its own min/max would break the form the moment someone
    clicked submit without touching anything.
    """
    features = DISEASE_META[disease]["features"]
    payload = {}
    for name, meta in features.items():
        payload[name] = meta["default"]

    values, errors = validate_features(disease, list(features), payload)

    assert errors == {}
    assert len(values) == len(features)
