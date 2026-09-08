"""Request shape and input checks for /api/predict."""

import math

from pydantic import BaseModel, ConfigDict

from healthscan.config import DISEASE_META


class PredictRequest(BaseModel):
    """The JSON body of a prediction request.

    Feature values (Glucose, BMI, ...) differ per disease, so they arrive as
    extra fields and are checked by `validate_features` against the bounds in
    DISEASE_META.
    """

    model_config = ConfigDict(extra="allow")

    disease: str
    model_type: str | None = None

    def feature_values(self):
        return self.model_extra or {}


def validate_features(disease, features, values):
    """Check every feature against its DISEASE_META bounds.

    Returns (numbers, errors). `numbers` is in the same order as `features`.
    Every problem is collected so a user fixing the form sees all of them at
    once, rather than one per submit.
    """
    numbers = []
    errors = {}

    for name in features:
        meta = DISEASE_META[disease]["features"][name]
        label = meta["label"]
        raw = values.get(name)

        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors[name] = f"{label} is required."
            continue

        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue

        if not math.isfinite(number):
            errors[name] = f"{label} must be a finite number."
            continue

        unit = ""
        if meta["unit"]:
            unit = " " + meta["unit"]

        if number < meta["min"]:
            errors[name] = f"{label} must be at least {meta['min']:g}{unit}."
        elif number > meta["max"]:
            errors[name] = f"{label} must be at most {meta['max']:g}{unit}."

        numbers.append(number)

    return numbers, errors
