"""Column roles are inferred from dtype, emptiness, and identifier uniqueness."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.column_roles import (
    ROLE_CATEGORICAL,
    ROLE_IGNORE,
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
    detect_column_roles,
)


def test_numeric_columns_are_measurements():
    """The repeated integer is ineligible for Row ID and remains numerical."""
    df = pd.DataFrame({"g": ["a", "a", "b"], "feat": [1.5, 2.5, 3.5], "n": [3, 4, 4]})
    roles = detect_column_roles(df)
    assert roles["feat"] == ROLE_NUMERICAL
    assert roles["n"] == ROLE_NUMERICAL


def test_a_repeating_text_column_is_categorical():
    df = pd.DataFrame({"treatment": ["ctrl", "drug", "ctrl", "drug"],
                       "feat": [1.0, 2.0, 3.0, 4.0]})
    assert detect_column_roles(df)["treatment"] == ROLE_CATEGORICAL


def test_an_all_unique_text_column_is_the_row_id():
    df = pd.DataFrame({"cell_id": ["a", "b", "c"], "t": ["x", "x", "y"],
                       "feat": [1.0, 2.0, 3.0]})
    roles = detect_column_roles(df)
    assert roles["cell_id"] == ROLE_ROW_ID
    assert roles["t"] == ROLE_CATEGORICAL


def test_an_identifier_must_remain_unique_as_text():
    df = pd.DataFrame({"cell_id": [1, "1", "c"],
                       "uuid": ["a", "b", "c"], "feat": [1.5, 2.5, 3.5]})

    roles = detect_column_roles(df)

    assert roles["cell_id"] == ROLE_CATEGORICAL
    assert roles["uuid"] == ROLE_ROW_ID


def test_a_fractional_float_is_never_the_row_id():
    """Distinct fractional measurements are ineligible for Row ID."""
    df = pd.DataFrame({"feat": [1.5, 2.5, 3.5], "t": ["x", "x", "y"]})
    roles = detect_column_roles(df)
    assert roles["feat"] == ROLE_NUMERICAL
    assert ROLE_ROW_ID not in roles.values()


def test_a_non_finite_value_disqualifies_a_numeric_column():
    """inf has no fractional part to test; it is not a whole number either."""
    df = pd.DataFrame({"feat": [1.0, 2.0, float("inf")], "t": ["x", "x", "y"]})
    roles = detect_column_roles(df)
    assert roles["feat"] == ROLE_NUMERICAL
    assert ROLE_ROW_ID not in roles.values()


def test_only_one_column_is_given_the_row_id_role():
    """Two all-unique text columns: the leftmost wins, the other is categorical."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "uuid": ["p", "q"], "feat": [1.0, 2.0]})
    roles = detect_column_roles(df)
    assert roles["cell_id"] == ROLE_ROW_ID
    assert roles["uuid"] == ROLE_CATEGORICAL


def test_an_identifier_is_found_wherever_it_sits_in_the_file():
    """Identifier detection checks every column, including the last one."""
    df = pd.DataFrame({"feat": [1.5, 2.5, 3.5], "g": ["a", "a", "b"],
                       "wine_id": [1, 2, 3]})
    roles = detect_column_roles(df)
    assert roles["wine_id"] == ROLE_ROW_ID
    assert roles["feat"] == ROLE_NUMERICAL


def test_an_identifier_qualifies_in_either_dtype():
    """Whole-number identifiers qualify as integers or floats, including spreadsheet
    values.
    """
    for values, dtype in (([1, 2, 3], "int64"), ([1.0, 2.0, 3.0], "float64")):
        df = pd.DataFrame({"g": ["a", "a", "b"], "cell_id": values})
        assert df["cell_id"].dtype == dtype     # Whole-valued spreadsheet columns can use either dtype.
        assert detect_column_roles(df)["cell_id"] == ROLE_ROW_ID, values


def test_an_identifier_with_gaps_still_qualifies():
    """Unique whole-number identifiers need not be consecutive."""
    df = pd.DataFrame({"cell_id": [1, 2, 4, 5], "feat": [1.5, 2.5, 3.5, 4.5]})
    assert detect_column_roles(df)["cell_id"] == ROLE_ROW_ID


def test_a_boolean_column_is_never_the_identifier():
    """Boolean values remain categorical even when both rows are unique and whole-valued."""
    df = pd.DataFrame({"flag": [True, False], "feat": [1.5, 2.5]})
    assert ROLE_ROW_ID not in detect_column_roles(df).values()


def test_a_boolean_column_is_a_category_not_a_measurement():
    """Boolean columns remain categorical despite pandas treating their dtype as numeric.
    """
    df = pd.DataFrame({"cell_id": [1, 2, 3], "is_responder": [True, False, True],
                       "tau": [0.4, 0.5, 0.6]})
    roles = detect_column_roles(df)
    assert roles == {"cell_id": ROLE_ROW_ID, "is_responder": ROLE_CATEGORICAL,
                     "tau": ROLE_NUMERICAL}


def test_an_integer_flag_column_is_still_a_measurement():
    """An integer 0/1 column may be a count, so its values do not make it categorical.
    """
    df = pd.DataFrame({"flag": [0, 1, 1, 0], "feat": [1.5, 2.5, 3.5, 4.5]})
    assert detect_column_roles(df)["flag"] == ROLE_NUMERICAL


def test_the_leftmost_qualifying_column_takes_the_role():
    """The leftmost identifier candidate wins; other numeric candidates remain
    measurements.
    """
    df = pd.DataFrame({"cell_id": [1, 2, 3], "npix": [7, 8, 9], "feat": [1.5, 2.5, 3.5]})
    roles = detect_column_roles(df)
    assert roles["cell_id"] == ROLE_ROW_ID
    assert roles["npix"] == ROLE_NUMERICAL


def test_a_whole_numbered_measurement_is_claimed_when_nothing_else_qualifies():
    """A unique integer measurement can become the Row ID; the review table can correct it.
    """
    df = pd.DataFrame({"g": ["a", "a"], "feat": [1.5, 2.5], "npix": [3, 4]})
    assert detect_column_roles(df)["npix"] == ROLE_ROW_ID


def test_a_repeating_integer_column_is_a_measurement_not_an_id():
    df = pd.DataFrame({"plate": [1, 1, 2], "feat": [1.5, 2.5, 3.5]})
    roles = detect_column_roles(df)
    assert roles["plate"] == ROLE_NUMERICAL
    assert ROLE_ROW_ID not in roles.values()


def test_the_fov_role_is_never_guessed():
    """Nothing distinguishes a FOV column from any other categorical."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "image_name": ["f1", "f1"],
                       "well": ["w1", "w1"], "feat": [1.0, 2.0]})
    assert "fov" not in detect_column_roles(df).values()


def test_nothing_is_ignored_by_guess_while_it_still_has_values():
    """Nonempty text columns are categorical; only empty columns are inferred as Ignore."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "notes": ["p", "q"], "feat": [1.0, 2.0]})
    assert "ignore" not in detect_column_roles(df).values()


def test_a_column_with_no_values_at_all_is_ignored():
    """An all-NaN column is ignored even when pandas assigns it a numerical dtype."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "notes": [float("nan")] * 2,
                       "feat": [1.0, 2.0]})
    assert df["notes"].dtype == "float64"   # pandas assigns an all-NaN column a numeric dtype.
    assert detect_column_roles(df)["notes"] == ROLE_IGNORE


def test_an_empty_column_of_any_dtype_is_ignored():
    """A spreadsheet can hand back an object column of Nones for the same shape."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "notes": [None, None],
                       "feat": [1.0, 2.0]})
    assert detect_column_roles(df)["notes"] == ROLE_IGNORE


def test_an_empty_column_does_not_become_the_row_id():
    """Emptiness is checked before candidacy, so an all-NaN column never qualifies."""
    df = pd.DataFrame({"blank": [None, None], "cell_id": ["a", "b"],
                       "feat": [1.0, 2.0]})
    roles = detect_column_roles(df)
    assert roles["blank"] == ROLE_IGNORE
    assert roles["cell_id"] == ROLE_ROW_ID


def test_an_empty_frame_has_no_roles():
    assert detect_column_roles(pd.DataFrame()) == {}


def test_every_column_gets_exactly_one_role():
    df = pd.DataFrame({"cell_id": ["a", "b"], "t": ["x", "x"], "feat": [1.0, 2.0]})
    roles = detect_column_roles(df)
    assert set(roles) == set(df.columns)


# ------------------------------------------- the app's wrapper, with coercion

def test_detect_roles_applies_the_same_1_percent_coercion_the_analysis_does():
    """Role detection applies the analysis coercion rule to mostly numerical text columns.
    """
    vals = [str(i / 2) for i in range(200)]
    vals[0] = "n/a"
    df = pd.DataFrame({"cell_id": [str(i) for i in range(200)], "feat": vals})

    assert detect_column_roles(df)["feat"] == ROLE_CATEGORICAL   # raw: still object
    assert dataset_io.detect_roles(df)["feat"] == ROLE_NUMERICAL  # coerced first


def test_detect_roles_does_not_mutate_the_caller_frame():
    vals = [str(i / 2) for i in range(200)]
    df = pd.DataFrame({"cell_id": [str(i) for i in range(200)], "feat": vals})

    dataset_io.detect_roles(df)
    assert df["feat"].dtype == object
