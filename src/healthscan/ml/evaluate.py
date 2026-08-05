"""Model evaluation helpers.

Everything here operates on *unscaled* features and wraps the estimator in a
``Pipeline`` with its own ``StandardScaler``. That matters: scaling before
splitting - or before cross-validation - leaks test-fold statistics into
training and inflates every score. The pipeline refits the scaler inside each
fold, so the numbers written to ``metrics.json`` are honest.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthscan.config import CV_FOLDS, RANDOM_STATE

SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")


def make_pipeline(estimator) -> Pipeline:
    """Wrap ``estimator`` in a scaler pipeline so scaling is fold-local."""
    return Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])


def _decision_scores(fitted: Pipeline, X) -> np.ndarray | None:
    """Continuous scores for ROC-AUC.

    ``Perceptron`` has no ``predict_proba``, so fall back to
    ``decision_function``. Probabilities are indexed via ``classes_`` rather
    than assuming the positive class sits at column 1.
    """
    model = fitted[-1]
    if hasattr(model, "predict_proba"):
        proba = fitted.predict_proba(X)
        positive_index = list(model.classes_).index(1)
        return proba[:, positive_index]
    if hasattr(model, "decision_function"):
        return fitted.decision_function(X)
    return None


def holdout_scores(fitted: Pipeline, X_test, y_test) -> dict[str, float | None]:
    """Score a fitted pipeline against the held-out test set."""
    y_pred = fitted.predict(X_test)
    scores: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    y_score = _decision_scores(fitted, X_test)
    scores["roc_auc"] = None if y_score is None else float(roc_auc_score(y_test, y_score))
    return scores


def cross_validation_scores(estimator, X_train, y_train) -> dict[str, dict[str, float]]:
    """Stratified k-fold CV on the training split only.

    Returns ``{metric: {"mean": ..., "std": ...}}``.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        make_pipeline(estimator),
        X_train,
        y_train,
        cv=cv,
        scoring=list(SCORING),
        error_score="raise",
    )

    summary = {}
    for metric in SCORING:
        fold_scores = results[f"test_{metric}"]
        summary[metric] = {
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
        }
    return summary


def evaluate(estimator, X_train, y_train, X_test, y_test) -> dict:
    """Full evaluation: k-fold CV on train, then a single held-out test score."""
    fitted = make_pipeline(estimator).fit(X_train, y_train)
    return {
        "cross_validation": cross_validation_scores(estimator, X_train, y_train),
        "holdout": holdout_scores(fitted, X_test, y_test),
    }
