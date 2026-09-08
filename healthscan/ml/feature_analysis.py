"""Feature-importance analysis behind the six-feature selection.

Diagnostic only - it trains a Random Forest on the *full* raw datasets and
prints ranked ``feature_importances_``, which is how the shortlist in
``config.DISEASE_META`` was chosen. Nothing here writes artifacts.

    python -m healthscan.ml.feature_analysis
"""

import logging
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from healthscan.config import (
    CLEVELAND_COLUMNS,
    DISEASE_META,
    RANDOM_STATE,
    RAW_DIR,
    WDBC_COLUMNS,
)

log = logging.getLogger(__name__)


def _report(disease: str, X: pd.DataFrame, y: pd.Series, top_n: int | None = None) -> None:
    """Fit a forest and print features ranked by importance."""
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = set(DISEASE_META[disease]["features"])

    log.info("")
    log.info("=" * 56)
    log.info(" %s - feature importance (%d raw features)", disease, len(X.columns))
    log.info("=" * 56)
    for rank, (feature, importance) in enumerate(importances.head(top_n).items(), 1):
        marker = "  <-- SELECTED" if feature in selected else ""
        log.info("  %2d. %-24s %.4f%s", rank, feature, importance, marker)


def analyze_heart_disease() -> None:
    raw_path = RAW_DIR / "processed.cleveland.data"
    if not raw_path.exists():
        log.warning("raw heart disease data not found at %s", raw_path)
        return

    df = pd.read_csv(raw_path, header=None, names=CLEVELAND_COLUMNS, na_values="?")
    df = df.fillna(df.median())
    df["target"] = (df["target"] > 0).astype(int)

    _report("heart_disease", df.drop(columns=["target"]), df["target"])


def analyze_breast_cancer() -> None:
    raw_path = RAW_DIR / "wdbc.data"
    if not raw_path.exists():
        log.warning("raw breast cancer data not found at %s", raw_path)
        return

    df = pd.read_csv(raw_path, header=None, names=WDBC_COLUMNS).drop(columns=["id"])
    y = df.pop("diagnosis").map({"M": 1, "B": 0})

    _report("breast_cancer", df, y, top_n=15)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    analyze_heart_disease()
    analyze_breast_cancer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
