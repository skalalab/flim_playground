"""Categorical columns match by exact name; the frame keeps the file's spelling."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io

# ------------------------------------------------------------ exact matching

def test_an_exactly_named_categorical_is_still_handled():
    """An exact categorical match is stringified and filled with "N/A"."""
    df = pd.DataFrame({"id": ["a", "b"], "treatment": ["DMSO", None], "f": [1.0, 2.0]})
    fixed, _warning, error = dataset_io.check_and_fix_df(df, ["treatment"], "id", "")
    assert error == ""
    assert list(fixed["treatment"]) == ["DMSO", "N/A"]


def test_a_differently_spelled_categorical_is_not_matched():
    """`Treatments` does not read as the configured `treatment`."""
    df = pd.DataFrame({"id": ["a", "b"], "Treatments": ["DMSO", None], "f": [1.0, 2.0]})
    fixed, _warning, error = dataset_io.check_and_fix_df(df, ["treatment"], "id", "")
    assert error == ""
    # Untouched: not renamed, not stringified, not "N/A"-filled.
    assert "Treatments" in fixed.columns
    assert "treatment" not in fixed.columns
    assert fixed["Treatments"].isna().any()


def test_matching_is_case_sensitive():
    """Categorical matching is case-sensitive."""
    df = pd.DataFrame({"id": ["a", "b"], "Treatment": ["DMSO", "PBS"], "f": [1.0, 2.0]})
    fixed, _warning, _error = dataset_io.check_and_fix_df(df, ["treatment"], "id", "")
    assert "Treatment" in fixed.columns
    assert "treatment" not in fixed.columns


def test_a_plural_column_is_not_folded_onto_its_singular():
    """Distinct singular and plural headers survive without a collision warning."""
    df = pd.DataFrame({"id": ["a", "b"], "ns": ["x", "y"], "n": ["p", "q"]})
    fixed, warning, _error = dataset_io.check_and_fix_df(df, ["n"], "id", "")
    assert {"n", "ns"} <= set(fixed.columns)
    assert "also reads as the categorical column" not in warning


# Header preservation

def test_the_frame_keeps_the_files_own_spelling():
    """A differently spelled configured name does not rename the file's header."""
    df = pd.DataFrame({"id": ["a", "b"], "Cell Line": ["A549", "MCF7"], "f": [1.0, 2.0]})
    fixed, _warning, _error = dataset_io.check_and_fix_df(df, ["cell_line"], "id", "")
    assert "Cell Line" in fixed.columns
    assert "cell_line" not in fixed.columns


def test_two_headers_that_used_to_collide_both_survive():
    """Headers differing in punctuation remain separate without collision warnings."""
    df = pd.DataFrame({"id": ["a", "b"], "IL-18": ["hi", "lo"], "IL_18": ["p", "q"]})
    fixed, warning, _error = dataset_io.check_and_fix_df(df, ["IL-18", "IL_18"], "id", "")
    assert {"IL-18", "IL_18"} <= set(fixed.columns)
    assert "also reads as the categorical column" not in warning


def test_the_fov_column_is_matched_exactly_too():
    """check_and_fix_df prepends the FOV name to the categorical list."""
    df = pd.DataFrame({"id": ["a", "b"], "well": [1, None], "f": [1.0, 2.0]})
    fixed, _warning, error = dataset_io.check_and_fix_df(df, [], "id", "well")
    assert error == ""
    assert list(fixed["well"]) == ["1", "N/A"]


# Export excludes fuzzy matching

def test_match_col_name_is_gone():
    assert not hasattr(dataset_io, "match_col_name")


def test_match_col_name_is_not_inlined_into_exported_scripts():
    """Export source contains no reference to the removed fuzzy-matching helper."""
    from src import export_script

    src = Path(export_script.__file__).read_text()
    assert "match_col_name" not in src


# Actionable extraction warnings

def test_dropped_columns_point_at_the_home_page_for_extraction_data():
    """Unmatched extraction categoricals produce advice to correct their Configuration names."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "Treatments": ["x", "y"],
                       "feat": [1.0, 2.0]})
    _out, _groups, warning, error = dataset_io.get_features(
        df, ["treatment"], use_data_extraction=True, unique_row_id_col="cell_id")
    assert error == ""
    assert "Treatments" in warning
    assert "Home page" in warning


def test_the_user_table_branch_does_not_point_at_the_home_page():
    """A user table's names come off its own headers, so that advice would misdirect."""
    df = pd.DataFrame({"row": ["a", "b"], "notes": ["x", "y"], "feat": [1.0, 2.0]})
    _out, _groups, warning, error = dataset_io.get_features(
        df, [], use_data_extraction=False, unique_row_id_col="row")
    assert error == ""
    assert "notes" in warning
    assert "Home page" not in warning


# Spreadsheet header normalization

def test_numeric_spreadsheet_headers_are_still_stringified():
    """Spreadsheet headers remain strings for categorical membership and downstream lookups."""
    df = pd.DataFrame({"id": ["a", "b"], 2024: ["x", "y"], "f": [1.0, 2.0]})
    df.columns = [str(c) for c in df.columns]
    fixed, _warning, error = dataset_io.check_and_fix_df(df, ["2024"], "id", "")
    assert error == ""
    assert list(fixed["2024"]) == ["x", "y"]
