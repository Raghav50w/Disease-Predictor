"""Dataset preparation from the committed raw UCI files.

These run against the real ``datasets/raw/`` files, which are committed, so
they check the actual row counts and target encodings rather than a fixture's.
"""

from __future__ import annotations

import pandas as pd
import pytest

from healthscan.config import DISEASE_META, RAW_DIR
from healthscan.ml.preprocess import (
    BUILDERS,
    build_breast_cancer,
    build_diabetes,
    build_heart_disease,
)

pytestmark = pytest.mark.skipif(
    not RAW_DIR.is_dir(), reason="raw datasets are not present in this checkout"
)


@pytest.fixture(params=sorted(BUILDERS))
def built(request, tmp_path):
    """Build one dataset into a temp directory and return (name, dataframe)."""
    name = request.param
    out_path = BUILDERS[name](out_dir=tmp_path)
    return name, pd.read_csv(out_path)


def test_columns_match_config_with_target_last(built):
    name, df = built
    meta = DISEASE_META[name]

    assert list(df.columns) == [*meta["features"], meta["target_column"]]


def test_target_is_binary(built):
    _, df = built
    assert set(df.iloc[:, -1].unique()) <= {0, 1}


def test_no_missing_values_remain(built):
    _, df = built
    assert not df.isna().any().any()


def test_expected_row_counts(built):
    name, df = built
    assert len(df) == {"diabetes": 768, "heart_disease": 303, "breast_cancer": 569}[name]


def test_heart_target_is_binarised_from_severity(tmp_path):
    """The raw Cleveland target is a 0-4 severity score, not a flag."""
    df = pd.read_csv(build_heart_disease(out_dir=tmp_path))

    # 164 healthy / 139 with any degree of disease.
    assert df["target"].sum() == 139


def test_breast_cancer_diagnosis_is_mapped(tmp_path):
    """M -> 1, B -> 0, and the patient id column is dropped."""
    df = pd.read_csv(build_breast_cancer(out_dir=tmp_path))

    assert "id" not in df.columns
    assert df["target"].sum() == 212  # malignant cases in WDBC


def test_builders_are_idempotent(tmp_path):
    first = pd.read_csv(build_diabetes(out_dir=tmp_path))
    second = pd.read_csv(build_diabetes(out_dir=tmp_path))

    pd.testing.assert_frame_equal(first, second)
