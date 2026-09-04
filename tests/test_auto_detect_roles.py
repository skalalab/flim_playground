"""Guessing a role for every column of a file the profile has never seen.

Deliberately threshold-free. Every rule is either a dtype question or a uniqueness
question, so there is no magic number to tune and no silent drift when data changes
shape. The guesses it gets wrong are wrong *visibly* -- a free-text column guessed as
categorical shows its 1204 distinct values in the review table's preview -- and cost
one dropdown to fix. A cardinality threshold would trade that for failures nobody sees.
"""
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
    """`n` repeats on purpose -- an all-distinct whole-number column is an id candidate,
    which is a different rule, pinned below."""
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


def test_a_fractional_float_is_never_the_row_id():
    """Nearly every measurement column is all-unique; that must not make it an id.

    A fractional part is what rules a column out, not the float dtype -- 1.1, 1.2, 1.3
    is a measurement however distinct it is. This is the rule that keeps the 25
    all-distinct float columns of a real extraction table out of the running.
    """
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
    """Position carries no information, so the rule reads none from it.

    Real files put the identifier last as readily as first -- `wine_id` is column 13
    of 13, `flower_id` column 6 of 6 -- and a rule that only looked at column 0 missed
    both while claiming to be a uniqueness rule.
    """
    df = pd.DataFrame({"feat": [1.5, 2.5, 3.5], "g": ["a", "a", "b"],
                       "wine_id": [1, 2, 3]})
    roles = detect_column_roles(df)
    assert roles["wine_id"] == ROLE_ROW_ID
    assert roles["feat"] == ROLE_NUMERICAL


def test_an_identifier_qualifies_in_either_dtype():
    """1, 2, 3 is an identifier whether it is stored as int64 or float64.

    The pair that keeps the rule off the dtype: the int form is the ordinary `cell_id`,
    and the float form -- 1.0, 2.0, 3.0 -- is what a spreadsheet, or
    coerce_majority_numeric_cols beside a column that needed a NaN, hands back. Refusing
    every float would miss exactly the column the coercion had just created.
    """
    for values, dtype in (([1, 2, 3], "int64"), ([1.0, 2.0, 3.0], "float64")):
        df = pd.DataFrame({"g": ["a", "a", "b"], "cell_id": values})
        assert df["cell_id"].dtype == dtype     # the dtypes that made this necessary
        assert detect_column_roles(df)["cell_id"] == ROLE_ROW_ID, values


def test_an_identifier_with_gaps_still_qualifies():
    """Ids surviving a filter -- 1, 2, 4, 5 -- are as much an identifier as 1, 2, 3, 4.

    Nothing requires the numbering to be a consecutive run: the test is whether the
    column is a bijection with the rows, not whether it counts them.
    """
    df = pd.DataFrame({"cell_id": [1, 2, 4, 5], "feat": [1.5, 2.5, 3.5, 4.5]})
    assert detect_column_roles(df)["cell_id"] == ROLE_ROW_ID


def test_a_boolean_column_is_never_the_identifier():
    """Two rows make a bool column all-distinct and its values are whole, so both
    questions pass -- and it is still a two-level category, not an identifier."""
    df = pd.DataFrame({"flag": [True, False], "feat": [1.5, 2.5]})
    assert ROLE_ROW_ID not in detect_column_roles(df).values()


def test_the_leftmost_qualifying_column_takes_the_role():
    """Two whole-number bijections: the leftmost wins, the other stays a measurement.

    This is the tie-break that carries what the old first-position rule used to enforce
    as a filter -- an identifier beside an all-distinct integer measurement wins by
    sitting where identifiers usually sit, rather than by the measurement being refused.
    Demoting the loser to Categorical would take it out of the analysis.
    """
    df = pd.DataFrame({"cell_id": [1, 2, 3], "npix": [7, 8, 9], "feat": [1.5, 2.5, 3.5]})
    roles = detect_column_roles(df)
    assert roles["cell_id"] == ROLE_ROW_ID
    assert roles["npix"] == ROLE_NUMERICAL


def test_a_whole_numbered_measurement_is_claimed_when_nothing_else_qualifies():
    """The cost of reading no signal but the values, pinned rather than left to be found.

    With no identifier column in the file, an all-distinct whole-numbered measurement
    takes the role -- `npix` here, a lifetime rounded to whole picoseconds in a real one.
    One measurement out of the pickers until a dropdown puts it back. Accepted against
    the alternative, which missed `wine_id` and `flower_id` outright, and visible: the
    review table shows this guess beside the column's own preview before any save.
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
    """Ignore stays a decision, not an assumption -- see the empty column below."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "notes": ["p", "q"], "feat": [1.0, 2.0]})
    assert "ignore" not in detect_column_roles(df).values()


def test_a_column_with_no_values_at_all_is_ignored():
    """The one exception, and the reason it is not a real exception.

    A header with nothing under it comes off a CSV as float64 NaN, so the dtype rule
    reads it as a measurement. check_and_fix_df removes it before get_features sees
    it either way, so guessing Ignore pre-empts no decision -- it just stops the
    review table offering an empty column as something to plot.
    """
    df = pd.DataFrame({"cell_id": ["a", "b"], "notes": [float("nan")] * 2,
                       "feat": [1.0, 2.0]})
    assert df["notes"].dtype == "float64"   # the dtype that made this necessary
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
    """A feature column with a stray string is a measurement, not a category.

    dataset_io.detect_roles runs coerce_majority_numeric_cols first, so the guess
    matches what get_features will actually make of the column.
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
