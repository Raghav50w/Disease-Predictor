"""End-to-end training behaviour and the honesty of the reported metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

from healthscan.config import RANDOM_STATE
from healthscan.ml.evaluate import evaluate, holdout_scores, make_pipeline
from healthscan.ml.train import (
    get_models,
    impute_invalid_zeros,
    resolve_feature_columns,
    train_disease,
)


def test_writes_every_expected_artifact(models_dir):
    expected = {
        "diabetes_scaler.pkl",
        "diabetes_features.pkl",
        "diabetes_random_forest.pkl",
        "diabetes_naive_bayes.pkl",
        "diabetes_mlp.pkl",
        "diabetes_perceptron.pkl",
    }
    assert expected <= {p.name for p in models_dir.glob("*.pkl")}


def test_result_records_split_sizes(diabetes_csv, tmp_path):
    result = train_disease(diabetes_csv, "diabetes", models_dir=tmp_path / "models")

    assert result["n_samples"] == result["n_train"] + result["n_test"]
    assert result["n_features"] == 4
    assert result["n_test"] == pytest.approx(result["n_samples"] * 0.2, abs=1)


def test_metrics_include_cv_and_holdout(diabetes_csv, tmp_path):
    result = train_disease(diabetes_csv, "diabetes", models_dir=tmp_path / "models")
    forest = result["models"]["random_forest"]

    assert set(forest) == {"cross_validation", "holdout"}
    assert set(forest["holdout"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert set(forest["cross_validation"][metric]) == {"mean", "std"}


def test_scores_are_valid_probabilities(diabetes_csv, tmp_path):
    result = train_disease(diabetes_csv, "diabetes", models_dir=tmp_path / "models")

    for model in result["models"].values():
        for name, value in model["holdout"].items():
            if value is not None:
                assert 0.0 <= value <= 1.0, name


def test_perceptron_gets_roc_auc_via_decision_function(diabetes_csv, tmp_path):
    """Perceptron has no predict_proba, so ROC-AUC must not come back null."""
    result = train_disease(diabetes_csv, "diabetes", models_dir=tmp_path / "models")

    assert result["models"]["perceptron"]["holdout"]["roc_auc"] is not None


def test_training_is_deterministic(diabetes_csv, tmp_path):
    first = train_disease(diabetes_csv, "diabetes", models_dir=tmp_path / "a")
    second = train_disease(diabetes_csv, "diabetes", models_dir=tmp_path / "b")

    assert first["models"]["random_forest"] == second["models"]["random_forest"]


def test_holdout_scores_are_not_training_scores():
    """Guard against the leakage bug: an overfit model must not score 1.0 on unseen noise."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 2, size=200)

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    forest = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
    fitted = make_pipeline(forest).fit(X_train, y_train)

    # Perfect on data it memorised, near chance on data it has never seen.
    assert holdout_scores(fitted, X_train, y_train)["accuracy"] > 0.95
    assert holdout_scores(fitted, X_test, y_test)["accuracy"] < 0.75


def test_evaluate_returns_both_sections():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)

    result = evaluate(Perceptron(random_state=RANDOM_STATE), X[:90], y[:90], X[90:], y[90:])

    assert result["cross_validation"]["accuracy"]["mean"] > 0.7
    assert result["holdout"]["accuracy"] > 0.7


# ------------------------------------------------------------------ imputation


def test_imputes_only_flagged_columns():
    """`impute_zero` is per-feature; a real zero elsewhere must survive."""
    df = pd.DataFrame(
        {
            "Glucose": [0, 100, 140],  # impute_zero: True
            "Insulin": [50, 80, 120],
            "BMI": [25.0, 30.0, 35.0],
            "Age": [0, 40, 50],  # impute_zero: False
        }
    )

    result = impute_invalid_zeros(df, "diabetes", list(df.columns))

    assert result.loc[0, "Glucose"] == 120  # median of the positive values
    assert result.loc[0, "Age"] == 0  # left alone


def test_imputation_does_not_mutate_the_caller_frame():
    df = pd.DataFrame({"Glucose": [0, 100, 140], "Insulin": [1, 2, 3], "BMI": [1.0, 2.0, 3.0]})
    original = df.copy()

    impute_invalid_zeros(df, "diabetes", list(df.columns))

    pd.testing.assert_frame_equal(df, original)


# ------------------------------------------------------- column resolution


def test_feature_order_comes_from_config_not_csv_order():
    df = pd.DataFrame({"Age": [1], "BMI": [1], "Outcome": [0], "Insulin": [1], "Glucose": [1]})

    features, target = resolve_feature_columns(df, "diabetes")

    assert features == ["Glucose", "Insulin", "BMI", "Age"]
    assert target == "Outcome"


def test_missing_column_is_a_clear_error():
    df = pd.DataFrame({"Glucose": [1], "Outcome": [0]})

    with pytest.raises(ValueError, match="missing columns"):
        resolve_feature_columns(df, "diabetes")


def test_unconfigured_disease_is_an_error_not_a_guess():
    """Guessing feature order from the CSV would train a model nothing can label."""
    df = pd.DataFrame({"a": [1], "b": [2], "target": [0]})

    with pytest.raises(ValueError, match="no DISEASE_META entry"):
        resolve_feature_columns(df, "not_configured")


def test_all_configured_models_are_unfitted():
    for name, model in get_models().items():
        assert not hasattr(model, "classes_"), name
