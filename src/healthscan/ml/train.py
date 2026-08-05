"""Train and evaluate every model for every disease dataset.

    python -m healthscan.ml.train

For each CSV in ``datasets/`` this script:

1. splits the data (stratified 80/20),
2. cross-validates each estimator on the training split,
3. scores it once against the held-out test split,
4. refits the estimator on the full dataset and pickles that artifact,
5. writes every score to ``models/metrics.json``.

Step 4 is deliberate: the reported numbers come from data the model never
saw, while the shipped artifact is trained on everything available.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from healthscan.config import (
    DATASETS_DIR,
    DISEASE_META,
    METRICS_FILENAME,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    feature_meta,
)
from healthscan.ml.evaluate import evaluate

log = logging.getLogger(__name__)


def get_models() -> dict:
    """Fresh, unfitted estimator instances keyed by artifact name."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "perceptron": Perceptron(max_iter=1000, random_state=RANDOM_STATE),
    }


def impute_invalid_zeros(df: pd.DataFrame, disease: str, feature_cols: list[str]) -> pd.DataFrame:
    """Replace physiologically impossible zeros with the column median.

    Which columns qualify comes from the ``impute_zero`` flag in
    ``DISEASE_META`` - an explicit per-feature decision, rather than substring
    matching on the column name, which would silently mangle any future column
    that merely contained "bmi" or "insulin".
    """
    df = df.copy()
    for col in feature_cols:
        if not feature_meta(disease, col)["impute_zero"]:
            continue
        positive = df[col] > 0
        if positive.any():
            median = df.loc[positive, col].median()
            n_replaced = int((df[col] == 0).sum())
            if n_replaced:
                df.loc[df[col] == 0, col] = median
                log.info("  imputed %d zero(s) in %s with median=%.2f", n_replaced, col, median)
    return df


def resolve_feature_columns(df: pd.DataFrame, disease: str) -> tuple[list[str], str]:
    """Determine feature columns and target column for a dataset.

    Feature order comes from ``DISEASE_META``, never from the CSV, so the API,
    the UI, and the trained model cannot disagree about it. A dataset with no
    metadata entry is an error rather than a guess - guessing "every column
    except the last" would train a model whose inputs nothing else can label.
    """
    meta = DISEASE_META.get(disease)
    if meta is None:
        raise ValueError(f"{disease}: no DISEASE_META entry; add one in healthscan.config")

    feature_cols = list(meta["features"])
    target_col = meta["target_column"]
    missing = [c for c in [*feature_cols, target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{disease}: dataset is missing columns {missing}")
    return feature_cols, target_col


def train_disease(csv_path: Path, disease: str, models_dir: Path = MODELS_DIR) -> dict:
    """Train, evaluate, and persist every model for one dataset."""
    log.info("=" * 62)
    log.info("  %s", disease)
    log.info("=" * 62)

    df = pd.read_csv(csv_path)
    feature_cols, target_col = resolve_feature_columns(df, disease)
    df = impute_invalid_zeros(df, disease, feature_cols)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    _dump(models_dir / f"{disease}_features.pkl", feature_cols)

    # The shipped scaler is fit on the full dataset, matching the shipped
    # models. Evaluation never touches it - see healthscan.ml.evaluate.
    scaler = StandardScaler().fit(X)
    _dump(models_dir / f"{disease}_scaler.pkl", scaler)
    X_scaled = scaler.transform(X)

    results = {}
    for name, estimator in get_models().items():
        log.info("  training %s...", name)
        results[name] = evaluate(estimator, X_train, y_train, X_test, y_test)

        estimator.fit(X_scaled, y)
        _dump(models_dir / f"{disease}_{name}.pkl", estimator)

        holdout = results[name]["holdout"]
        cv = results[name]["cross_validation"]["accuracy"]
        log.info(
            "    cv accuracy %.3f (+/- %.3f) | test accuracy %.3f | test roc_auc %s",
            cv["mean"],
            cv["std"],
            holdout["accuracy"],
            "n/a" if holdout["roc_auc"] is None else f"{holdout['roc_auc']:.3f}",
        )

    return {
        "display_name": DISEASE_META[disease]["display_name"],
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "models": results,
    }


def _dump(path: Path, obj) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _clear_artifacts(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*models_dir.glob("*.pkl"), models_dir / METRICS_FILENAME]:
        if stale.exists():
            stale.unlink()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not DATASETS_DIR.is_dir():
        log.error("datasets directory not found: %s", DATASETS_DIR)
        log.error("run `python -m healthscan.ml.preprocess` first")
        return 1

    csv_files = sorted(DATASETS_DIR.glob("*.csv"))
    if not csv_files:
        log.error("no .csv files found in %s", DATASETS_DIR)
        return 1

    _clear_artifacts(MODELS_DIR)
    log.info("found %d dataset(s): %s", len(csv_files), [p.name for p in csv_files])

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "diseases": {},
    }

    for csv_path in csv_files:
        disease = csv_path.stem
        metrics["diseases"][disease] = train_disease(csv_path, disease)

    metrics_path = MODELS_DIR / METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log.info("=" * 62)
    log.info("  artifacts written to %s", MODELS_DIR)
    log.info("  metrics written to %s", metrics_path)
    log.info("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
