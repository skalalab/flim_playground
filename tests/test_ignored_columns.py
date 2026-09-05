"""Ignore excludes columns from analysis regardless of dtype.
Numeric labels such as plate_number must stay out of feature selection and
coercion warnings when the user marks them Ignore.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io


def _features(groups):
    return [col for cols in groups.values() for col in cols]


def _frame():
    return pd.DataFrame({"row": ["a", "b"], "feat": [1.0, 2.0],
                         "plate_number": [3, 4], "notes": ["x", "y"]})


def test_a_numeric_column_can_be_ignored():
    """Numerical columns marked Ignore are excluded from features and the analysis frame."""
    out, groups, _w, error = dataset_io.get_features(
        _frame(), [], use_data_extraction=False, unique_row_id_col="row",
        ignored_cols=["plate_number"])

    assert error == ""
    assert "plate_number" not in _features(groups)
    assert "plate_number" not in out.columns
    assert _features(groups) == ["feat"]


def test_a_text_column_is_still_dropped_when_ignored():
    out, _groups, _w, _error = dataset_io.get_features(
        _frame(), [], use_data_extraction=False, unique_row_id_col="row",
        ignored_cols=["notes"])
    assert "notes" not in out.columns


def test_ignoring_every_measurement_is_an_error_not_an_empty_plot():
    _out, _groups, _w, error = dataset_io.get_features(
        _frame(), [], use_data_extraction=False, unique_row_id_col="row",
        ignored_cols=["feat", "plate_number"])
    assert "No feature found" in error


def test_a_deliberately_ignored_column_is_not_reported_as_a_surprise():
    """Intentionally ignored columns produce no pruning warning."""
    _out, _groups, warning, _error = dataset_io.get_features(
        _frame(), [], use_data_extraction=False, unique_row_id_col="row",
        ignored_cols=["notes", "plate_number"])

    assert "not analysed" not in warning


def test_an_unexpected_column_is_still_reported():
    """Pruning warnings still name unconfigured columns that were not marked Ignore."""
    _out, _groups, warning, _error = dataset_io.get_features(
        _frame(), [], use_data_extraction=False, unique_row_id_col="row")

    assert "notes" in warning
    assert "not analysed" in warning


def test_no_ignored_columns_behaves_exactly_as_before():
    """The parameter defaults to empty, so every existing caller is unaffected."""
    without = dataset_io.get_features(
        _frame(), [], use_data_extraction=False, unique_row_id_col="row")
    empty = dataset_io.get_features(
        _frame(), [], use_data_extraction=False, unique_row_id_col="row",
        ignored_cols=[])

    assert list(without[0].columns) == list(empty[0].columns)
    assert without[1] == empty[1]
    assert without[2] == empty[2]


def test_an_ignored_column_is_not_coerced_to_numeric():
    """Ignored columns skip numeric coercion without a conversion warning.
    Checking only their absence from the output frame would miss unnecessary coercion
    before pruning, so assert the warning text too.
    """
    vals = [str(i) for i in range(200)]
    vals[0] = "n/a"  # 0.5% non-numeric: inside the 1% rule, so it would convert
    df = pd.DataFrame({"row": [f"r{i}" for i in range(200)],
                       "feat": [float(i) for i in range(200)],
                       "code": vals})

    out, groups, warning, _error = dataset_io.get_features(
        df, [], use_data_extraction=False, unique_row_id_col="row",
        ignored_cols=["code"])

    assert "code" not in _features(groups)
    assert "code" not in out.columns
    assert "converted to NaN" not in warning
