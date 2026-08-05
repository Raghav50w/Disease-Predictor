"""Shared fixtures.

Tests never depend on the real ``models/`` directory - each one gets a
freshly trained set of tiny artifacts in a temporary directory, so the suite
passes on a clean clone before anything has been trained.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from healthscan.api import create_app
from healthscan.api.registry import ModelRegistry
from healthscan.config import DISEASE_META, METRICS_FILENAME, RANDOM_STATE, TEST_SIZE
from healthscan.ml.train import train_disease


@pytest.fixture(scope="session")
def diabetes_csv(tmp_path_factory):
    """A small, learnable synthetic dataset with the real diabetes columns."""
    tmp_path = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(RANDOM_STATE)
    n = 160

    glucose = rng.uniform(70, 190, n)
    insulin = rng.uniform(20, 300, n)
    bmi = rng.uniform(18, 45, n)
    age = rng.uniform(21, 70, n)

    # A deterministic signal so trained models are better than chance.
    outcome = ((glucose > 130) & (bmi > 30)).astype(int)

    df = pd.DataFrame(
        {
            "Glucose": glucose,
            "Insulin": insulin,
            "BMI": bmi,
            "Age": age,
            "Outcome": outcome,
        }
    )

    path = tmp_path / "diabetes.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def _trained_models(tmp_path_factory, diabetes_csv):
    """Train once per session - fitting an MLP for every test is far too slow."""
    out = tmp_path_factory.mktemp("trained") / "models"
    result = train_disease(diabetes_csv, "diabetes", models_dir=out)

    metrics = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "sklearn_version": "test",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "diseases": {"diabetes": result},
    }
    (out / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


@pytest.fixture
def models_dir(tmp_path, _trained_models):
    """A private copy of the trained artifacts.

    Copied rather than shared because some tests delete artifacts to exercise
    the incomplete-artifact paths.
    """
    out = tmp_path / "models"
    shutil.copytree(_trained_models, out)
    return out


@pytest.fixture
def registry(models_dir):
    reg = ModelRegistry(models_dir)
    reg.load()
    return reg


@pytest.fixture
def client(models_dir):
    app = create_app(models_dir=models_dir)
    app.config.update(TESTING=True)
    return app.test_client()


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
