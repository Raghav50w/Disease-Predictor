"""Scores a model: 5-fold cross-validation on the training split, then a
single score against the held-out test split.

Everything here takes *unscaled* features and wraps the model in a Pipeline
with its own StandardScaler. That matters: scaling before cross-validation
would let each fold's scaler see the rows it is about to be scored on, which
inflates the numbers. The Pipeline refits the scaler inside every fold.
"""

import numpy as np
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


def make_pipeline(estimator):
    """Glue a scaler and a model together so they fit and predict as one unit."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def holdout_scores(fitted, X_test, y_test):
    """Score a fitted pipeline against data it has never seen."""
    y_pred = fitted.predict(X_test)

    scores = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # ROC-AUC needs a confidence score, not just a 0/1 answer. Perceptron has
    # no predict_proba, so it reports no ROC-AUC.
    model = fitted[-1]
    if hasattr(model, "predict_proba"):
        positive = fitted.predict_proba(X_test)[:, 1]
        scores["roc_auc"] = float(roc_auc_score(y_test, positive))
    else:
        scores["roc_auc"] = None

    return scores


def cross_validation_scores(estimator, X_train, y_train):
    """5-fold stratified cross-validation. Returns {metric: {mean, std}}."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        make_pipeline(estimator),
        X_train,
        y_train,
        cv=folds,
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


def evaluate(estimator, X_train, y_train, X_test, y_test):
    """Cross-validate on the training split, then score once on the test split."""
    fitted = make_pipeline(estimator).fit(X_train, y_train)
    return {
        "cross_validation": cross_validation_scores(estimator, X_train, y_train),
        "holdout": holdout_scores(fitted, X_test, y_test),
    }
