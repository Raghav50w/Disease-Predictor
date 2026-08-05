"""HTTP-level behaviour, including every error path."""

from __future__ import annotations

import pytest


def test_healthz_reports_ready(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.json["ready"] is True
    assert response.json["diseases"] == ["diabetes"]


def test_diseases_catalog_shape(client):
    response = client.get("/api/diseases")

    assert response.status_code == 200
    diabetes = response.json["diseases"]["diabetes"]
    assert [f["name"] for f in diabetes["features"]] == ["Glucose", "Insulin", "BMI", "Age"]
    assert {m["name"] for m in diabetes["models"]} == {
        "random_forest",
        "naive_bayes",
        "mlp",
        "perceptron",
    }


def test_metrics_endpoint(client):
    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert "diabetes" in response.json["diseases"]


def test_predict_happy_path(client, valid_payload):
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json
    assert body["prediction"] in (0, 1)
    assert body["model_used"] == "random_forest"
    assert body["summary"] in ("Low risk of Diabetes", "High risk of Diabetes")


def test_predict_defaults_to_random_forest(client, valid_payload):
    valid_payload.pop("model_type")
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json["model_used"] == "random_forest"


@pytest.mark.parametrize("model_type", ["random_forest", "naive_bayes", "mlp", "perceptron"])
def test_every_model_serves_predictions(client, valid_payload, model_type):
    valid_payload["model_type"] = model_type
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json["model_used"] == model_type


def test_missing_field_returns_400_with_field_detail(client, valid_payload):
    del valid_payload["Glucose"]
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 400
    assert "Glucose" in response.json["fields"]


def test_out_of_range_returns_400(client, valid_payload):
    valid_payload["Glucose"] = 99999
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 400
    assert "Glucose" in response.json["fields"]


def test_unknown_disease_returns_404(client, valid_payload):
    valid_payload["disease"] = "malaria"
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 404
    assert "Unknown disease" in response.json["error"]


def test_unknown_model_returns_404(client, valid_payload):
    valid_payload["model_type"] = "xgboost"
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 404
    assert "not available" in response.json["error"]


def test_missing_disease_returns_400(client, valid_payload):
    del valid_payload["disease"]
    response = client.post("/api/predict", json=valid_payload)

    assert response.status_code == 400


def test_non_json_body_returns_400_not_500(client):
    response = client.post("/api/predict", data="not json", content_type="application/json")

    assert response.status_code == 400
    assert response.json["error"]


def test_form_encoded_body_returns_400(client):
    response = client.post("/api/predict", data={"disease": "diabetes"})

    assert response.status_code == 400


def test_unknown_route_returns_json_not_html(client):
    response = client.get("/api/nope")

    assert response.status_code == 404
    assert response.content_type.startswith("application/json")


def test_wrong_method_returns_json(client):
    response = client.get("/api/predict")

    assert response.status_code == 405
    assert response.content_type.startswith("application/json")


def test_index_serves_the_frontend(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"HealthScan" in response.data
