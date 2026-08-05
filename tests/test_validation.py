"""Validation rules for prediction requests."""

from __future__ import annotations

import pytest

from healthscan.api.validation import ValidationError, parse_features
from healthscan.config import DISEASE_META


def parse(payload, features):
    return parse_features("diabetes", features, payload)


def test_returns_values_in_model_feature_order(diabetes_features):
    payload = {"BMI": 28.4, "Age": 45, "Glucose": 120, "Insulin": 85}
    assert parse(payload, diabetes_features) == [120.0, 85.0, 28.4, 45.0]


def test_accepts_numeric_strings(diabetes_features):
    payload = {"Glucose": "120", "Insulin": "85", "BMI": "28.4", "Age": "45"}
    assert parse(payload, diabetes_features) == [120.0, 85.0, 28.4, 45.0]


def test_missing_field_is_rejected_not_defaulted_to_zero(diabetes_features):
    """A silent 0 here would produce a confident answer from a fabricated value."""
    payload = {"Insulin": 85, "BMI": 28.4, "Age": 45}

    with pytest.raises(ValidationError) as exc:
        parse(payload, diabetes_features)

    assert "Glucose" in exc.value.errors
    assert "required" in exc.value.errors["Glucose"]


def test_empty_string_is_treated_as_missing(diabetes_features):
    payload = {"Glucose": "  ", "Insulin": 85, "BMI": 28.4, "Age": 45}

    with pytest.raises(ValidationError) as exc:
        parse(payload, diabetes_features)

    assert "Glucose" in exc.value.errors


def test_non_numeric_value_is_rejected(diabetes_features):
    payload = {"Glucose": "high", "Insulin": 85, "BMI": 28.4, "Age": 45}

    with pytest.raises(ValidationError) as exc:
        parse(payload, diabetes_features)

    assert "must be a number" in exc.value.errors["Glucose"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("Glucose", 99999, "at most"),
        ("Glucose", 5, "at least"),
        ("Age", -1, "at least"),
        ("BMI", 500, "at most"),
    ],
)
def test_out_of_range_values_are_rejected(diabetes_features, field, value, expected):
    payload = {"Glucose": 120, "Insulin": 85, "BMI": 28.4, "Age": 45, field: value}

    with pytest.raises(ValidationError) as exc:
        parse(payload, diabetes_features)

    assert expected in exc.value.errors[field]


def test_non_finite_values_are_rejected(diabetes_features):
    payload = {"Glucose": float("nan"), "Insulin": 85, "BMI": 28.4, "Age": 45}

    with pytest.raises(ValidationError) as exc:
        parse(payload, diabetes_features)

    assert "finite" in exc.value.errors["Glucose"]


def test_all_errors_are_reported_together(diabetes_features):
    """A user fixing a form should see every problem at once."""
    payload = {"Glucose": "abc", "BMI": 999}

    with pytest.raises(ValidationError) as exc:
        parse(payload, diabetes_features)

    assert set(exc.value.errors) == {"Glucose", "Insulin", "BMI", "Age"}


def test_non_dict_payload_is_rejected(diabetes_features):
    with pytest.raises(ValidationError):
        parse(["not", "a", "dict"], diabetes_features)


def test_unknown_feature_is_an_error_not_an_unvalidated_passthrough():
    """No DISEASE_META entry means no bounds, so this must fail loudly."""
    with pytest.raises(KeyError):
        parse_features("diabetes", ["mystery_column"], {"mystery_column": 12345})


@pytest.mark.parametrize("disease", sorted(DISEASE_META))
def test_configured_defaults_pass_their_own_validation(disease):
    """The frontend pre-fills every field with its `default`.

    A default outside its own min/max would make the form fail the moment a
    visitor clicked submit without touching anything.
    """
    features = DISEASE_META[disease]["features"]
    payload = {name: meta["default"] for name, meta in features.items()}

    values = parse_features(disease, list(features), payload)

    assert values == [float(meta["default"]) for meta in features.values()]


@pytest.mark.parametrize("disease", sorted(DISEASE_META))
def test_enumerated_defaults_are_a_listed_option(disease):
    """A default that is not among a select's options renders as a blank box."""
    for name, meta in DISEASE_META[disease]["features"].items():
        if "options" in meta:
            allowed = {choice["value"] for choice in meta["options"]}
            assert meta["default"] in allowed, name
