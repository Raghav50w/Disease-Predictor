"""Convert the raw UCI source files into the clean CSVs used for training.

Reads from ``datasets/raw/`` and writes one CSV per disease into
``datasets/``, with the target as the final column. Idempotent - safe to
re-run at any time.

    python -m healthscan.ml.preprocess
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from healthscan.config import (
    CLEVELAND_COLUMNS,
    DATASETS_DIR,
    DISEASE_META,
    RAW_DIR,
    WDBC_COLUMNS,
)

log = logging.getLogger(__name__)


def _keep_features(disease: str) -> list[str]:
    """Feature columns to retain for ``disease``, in DISEASE_META order."""
    return list(DISEASE_META[disease]["features"])


def _fill_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs with the column median, logging each column touched."""
    for col in df.columns:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log.info("  filled %s missing values with median=%.2f", col, median)
    return df


def build_heart_disease(raw_dir: Path = RAW_DIR, out_dir: Path = DATASETS_DIR) -> Path:
    """processed.cleveland.data -> heart_disease.csv."""
    raw_path = raw_dir / "processed.cleveland.data"
    df = pd.read_csv(raw_path, header=None, names=CLEVELAND_COLUMNS, na_values="?")
    df = _fill_median(df)

    # The raw target is a 0-4 severity score; collapse it to presence/absence.
    df["target"] = (df["target"] > 0).astype(int)

    keep = [*_keep_features("heart_disease"), "target"]
    df = df[keep]

    out_path = out_dir / "heart_disease.csv"
    df.to_csv(out_path, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
    return out_path


def build_breast_cancer(raw_dir: Path = RAW_DIR, out_dir: Path = DATASETS_DIR) -> Path:
    """wdbc.data -> breast_cancer.csv."""
    raw_path = raw_dir / "wdbc.data"
    df = pd.read_csv(raw_path, header=None, names=WDBC_COLUMNS)

    df = df.drop(columns=["id"])
    target = df.pop("diagnosis").map({"M": 1, "B": 0})

    df = df[_keep_features("breast_cancer")]
    df["target"] = target

    out_path = out_dir / "breast_cancer.csv"
    df.to_csv(out_path, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
    return out_path


def build_diabetes(raw_dir: Path = RAW_DIR, out_dir: Path = DATASETS_DIR) -> Path:
    """diabetes.csv is already tabular; normalise column order and copy it."""
    raw_path = raw_dir / "diabetes.csv"
    df = pd.read_csv(raw_path)

    keep = [*_keep_features("diabetes"), DISEASE_META["diabetes"]["target_column"]]
    df = df[keep]

    out_path = out_dir / "diabetes.csv"
    df.to_csv(out_path, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_path.name, len(df), len(df.columns))
    return out_path


BUILDERS = {
    "heart_disease": build_heart_disease,
    "breast_cancer": build_breast_cancer,
    "diabetes": build_diabetes,
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not RAW_DIR.is_dir():
        log.error("raw data directory not found: %s", RAW_DIR)
        return 1

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for name, builder in BUILDERS.items():
        log.info("processing %s...", name)
        builder()

    log.info("all datasets prepared in %s", DATASETS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
