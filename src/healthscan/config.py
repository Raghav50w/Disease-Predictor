"""Single source of truth for paths, training constants, and feature metadata.

``DISEASE_META`` is the important part. Every consumer reads feature
information from here rather than redefining it:

* ``ml.preprocess`` uses the ``features`` keys to decide which raw columns to
  keep, and ``target_column`` for the label.
* ``ml.train`` uses the ``features`` order as the model's feature order, and
  ``impute_zero`` to decide which zeros are physiologically impossible and
  should be median-imputed.
* ``api.validation`` uses ``min``/``max`` as request validation bounds.
* the frontend renders labels, units, help text, and input bounds straight
  from the ``/api/diseases`` payload.

A disease must have an entry here to be trained or served - there is no
"guess it from the CSV" fallback, so the config and the artifacts cannot
disagree about feature order or bounds.

Each feature's ``default`` is a typical value for that population, roughly the
dataset mean. The frontend pre-fills every field with it and also uses it as
the placeholder, so the form is submittable on load and a visitor can get a
result without hunting for plausible clinical values. One key rather than a
separate ``placeholder``/``default`` pair, so the two cannot drift apart.

Adding a fourth disease means adding an entry here plus a CSV in
``datasets/``; no HTML or JavaScript changes are required.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = Path(os.environ.get("HEALTHSCAN_DATASETS_DIR", BASE_DIR / "datasets"))
RAW_DIR = DATASETS_DIR / "raw"
MODELS_DIR = Path(os.environ.get("HEALTHSCAN_MODELS_DIR", BASE_DIR / "models"))
STATIC_DIR = Path(os.environ.get("HEALTHSCAN_STATIC_DIR", BASE_DIR / "static"))

METRICS_FILENAME = "metrics.json"

# Training constants.
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model selected when a request omits ``model_type``, and pre-selected by the
# frontend. Kept here so the API default and the UI default cannot disagree.
DEFAULT_MODEL = "random_forest"

# Human-readable names for the model keys produced by ``ml.train``.
MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "naive_bayes": "Naive Bayes",
    "mlp": "Neural Network (MLP)",
    "perceptron": "Perceptron",
}

# Column layout of the raw UCI source files, shared by preprocessing and the
# feature-importance analysis so the lists cannot drift apart.
CLEVELAND_COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "st_depression",
    "st_slope",
    "num_vessels",
    "thalassemia",
    "target",
]

WDBC_FEATURE_NAMES = [
    f"{base}_{suffix}"
    for suffix in ("mean", "se", "worst")
    for base in (
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    )
]
WDBC_COLUMNS = ["id", "diagnosis", *WDBC_FEATURE_NAMES]


DISEASE_META = {
    "diabetes": {
        "display_name": "Diabetes",
        "description": "Pima Indians Diabetes Database (768 records, female patients aged 21+).",
        "target_column": "Outcome",
        "features": {
            "Glucose": {
                "label": "Plasma Glucose",
                "unit": "mg/dL",
                "min": 40,
                "max": 400,
                "step": 1,
                "default": 120,
                "help": "2-hour plasma glucose from an oral glucose tolerance test.",
                "impute_zero": True,
            },
            # min is 14, not 0: training median-imputes every zero insulin
            # (374 of 768 rows), so the models have never seen a zero and a
            # request carrying one would get a meaningless answer.
            "Insulin": {
                "label": "Serum Insulin",
                "unit": "mu U/mL",
                "min": 14,
                "max": 900,
                "step": 1,
                "default": 80,
                "help": "2-hour serum insulin. Leave near 80 if unknown.",
                "impute_zero": True,
            },
            "BMI": {
                "label": "Body Mass Index",
                "unit": "kg/m²",
                "min": 10,
                "max": 70,
                "step": 0.1,
                "default": 32.0,
                "help": "Weight in kg divided by height in metres squared.",
                "impute_zero": True,
            },
            "Age": {
                "label": "Age",
                "unit": "years",
                "min": 18,
                "max": 120,
                "step": 1,
                "default": 35,
                "help": "Patient age in years.",
                "impute_zero": False,
            },
        },
    },
    "heart_disease": {
        "display_name": "Heart Disease",
        "description": "Cleveland Heart Disease dataset (303 records), reduced to the six most predictive features.",
        "target_column": "target",
        "features": {
            "chest_pain_type": {
                "label": "Chest Pain Type",
                "unit": "",
                "min": 1,
                "max": 4,
                "step": 1,
                "default": 4,
                "help": "Category of chest pain reported by the patient.",
                "impute_zero": False,
                "options": [
                    {"value": 1, "label": "Typical angina"},
                    {"value": 2, "label": "Atypical angina"},
                    {"value": 3, "label": "Non-anginal pain"},
                    {"value": 4, "label": "Asymptomatic"},
                ],
            },
            "thalassemia": {
                "label": "Thalassemia",
                "unit": "",
                "min": 3,
                "max": 7,
                "step": 1,
                "default": 3,
                "help": "Result of the thallium stress test.",
                "impute_zero": False,
                "options": [
                    {"value": 3, "label": "Normal"},
                    {"value": 6, "label": "Fixed defect"},
                    {"value": 7, "label": "Reversible defect"},
                ],
            },
            "num_vessels": {
                "label": "Major Vessels Coloured",
                "unit": "count",
                "min": 0,
                "max": 3,
                "step": 1,
                "default": 0,
                "help": "Number of major vessels (0-3) coloured by fluoroscopy.",
                "impute_zero": False,
                "options": [
                    {"value": 0, "label": "0 vessels"},
                    {"value": 1, "label": "1 vessel"},
                    {"value": 2, "label": "2 vessels"},
                    {"value": 3, "label": "3 vessels"},
                ],
            },
            "max_heart_rate": {
                "label": "Maximum Heart Rate",
                "unit": "bpm",
                "min": 60,
                "max": 220,
                "step": 1,
                "default": 150,
                "help": "Highest heart rate achieved during exercise testing.",
                "impute_zero": False,
            },
            "st_depression": {
                "label": "ST Depression",
                "unit": "mm",
                "min": 0,
                "max": 10,
                "step": 0.1,
                "default": 1.0,
                "help": "ST depression induced by exercise relative to rest.",
                "impute_zero": False,
            },
            "age": {
                "label": "Age",
                "unit": "years",
                "min": 18,
                "max": 120,
                "step": 1,
                "default": 54,
                "help": "Patient age in years.",
                "impute_zero": False,
            },
        },
    },
    "breast_cancer": {
        "display_name": "Breast Cancer",
        "description": "Wisconsin Diagnostic Breast Cancer dataset (569 records), reduced to the six most predictive features.",
        "target_column": "target",
        "features": {
            "perimeter_worst": {
                "label": "Perimeter (worst)",
                "unit": "mm",
                "min": 40,
                "max": 300,
                "step": 0.01,
                "default": 107.26,
                "help": "Largest cell-nucleus perimeter measured in the sample.",
                "impute_zero": False,
            },
            "area_worst": {
                "label": "Area (worst)",
                "unit": "mm²",
                "min": 150,
                "max": 5000,
                "step": 0.1,
                "default": 880.58,
                "help": "Largest cell-nucleus area measured in the sample.",
                "impute_zero": False,
            },
            "concave_points_worst": {
                "label": "Concave Points (worst)",
                "unit": "",
                "min": 0,
                "max": 0.5,
                "step": 0.0001,
                "default": 0.1146,
                "help": "Largest count of concave contour portions, normalised.",
                "impute_zero": False,
            },
            "concave_points_mean": {
                "label": "Concave Points (mean)",
                "unit": "",
                "min": 0,
                "max": 0.5,
                "step": 0.0001,
                "default": 0.0489,
                "help": "Mean count of concave contour portions, normalised.",
                "impute_zero": False,
            },
            "radius_worst": {
                "label": "Radius (worst)",
                "unit": "mm",
                "min": 5,
                "max": 50,
                "step": 0.01,
                "default": 16.27,
                "help": "Largest mean distance from centre to perimeter points.",
                "impute_zero": False,
            },
            "area_mean": {
                "label": "Area (mean)",
                "unit": "mm²",
                "min": 100,
                "max": 3000,
                "step": 0.1,
                "default": 654.89,
                "help": "Mean cell-nucleus area across the sample.",
                "impute_zero": False,
            },
        },
    },
}


def disease_meta(disease: str) -> dict | None:
    """Return the metadata block for ``disease``, or ``None`` if unknown."""
    return DISEASE_META.get(disease)


def feature_meta(disease: str, feature: str) -> dict:
    """Return metadata for a single feature.

    Raises ``KeyError`` for an unconfigured feature. That is deliberate: the
    alternative was a permissive fallback with no ``min``/``max``, which would
    have quietly sent an unvalidated number into a model.
    """
    return DISEASE_META[disease]["features"][feature]
