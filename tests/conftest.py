"""Shared test fixtures.

Tests run against the real datasets and the real trained models, so run
`python -m healthscan.ml.preprocess` and `python -m healthscan.ml.train`
before `pytest`.
"""

import pytest
from fastapi.testclient import TestClient

from healthscan.config import DISEASE_META
from healthscan.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """A request body that satisfies every diabetes validation rule."""
    return {
        "disease": "diabetes",
        "model_type": "random_forest",
        "Glucose": 120,
        "Insulin": 85,
        "BMI": 28.4,
        "Age": 45,
    }


@pytest.fixture
def diabetes_features():
    return list(DISEASE_META["diabetes"]["features"])
