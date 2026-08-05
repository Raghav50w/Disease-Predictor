"""Request validation for ``/api/predict``.

A missing field is an error rather than a silent zero, and every value is
checked against the physiological bounds declared in ``config.DISEASE_META``.
"""

from __future__ import annotations

import math

from healthscan.config import feature_meta


class ValidationError(Exception):
    """Raised when a prediction request is malformed.

    ``errors`` maps a field name (or ``"_"`` for request-level problems) to a
    human-readable message, so the frontend can highlight the offending input.
    """

    def __init__(self, message: str, errors: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or {}


def parse_features(disease: str, features: list[str], payload: dict) -> list[float]:
    """Validate ``payload`` and return feature values in model order.

    Raises ``ValidationError`` listing every problem at once, rather than
    failing on the first one, so a user fixing a form sees all of it.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    errors: dict[str, str] = {}
    values: list[float] = []

    for feature in features:
        meta = feature_meta(disease, feature)
        label = meta["label"]
        raw = payload.get(feature)

        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors[feature] = f"{label} is required."
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[feature] = f"{label} must be a number."
            continue

        if not math.isfinite(value):
            errors[feature] = f"{label} must be a finite number."
            continue

        low, high = meta["min"], meta["max"]
        if value < low:
            errors[feature] = f"{label} must be at least {_fmt(low)}{_unit(meta)}."
        elif value > high:
            errors[feature] = f"{label} must be at most {_fmt(high)}{_unit(meta)}."

        values.append(value)

    if errors:
        raise ValidationError("One or more inputs are invalid.", errors)

    return values


def _unit(meta: dict) -> str:
    unit = meta["unit"]
    return f" {unit}" if unit else ""


def _fmt(number: float) -> str:
    return f"{number:g}"
