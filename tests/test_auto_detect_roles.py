"""Role inference uses explicit name hints and preserves assigned roles."""
import pandas as pd
import pytest

from src import dataset_io
from src.column_roles import (
    ROLE_CATEGORICAL, ROLE_IGNORE, ROLE_NUMERICAL, ROLE_ROW_ID,
    detect_column_roles,
)


@pytest.mark.parametrize("values, expected", [
    ([100, 200, 300], ROLE_NUMERICAL),
    ([1.0, 2.0, 3.0], ROLE_NUMERICAL),
    ([1.5, 2.5, float("inf")], ROLE_NUMERICAL),
    (["a", "b", "c"], ROLE_CATEGORICAL),
    ([True, False, True], ROLE_CATEGORICAL),
    ([None, None, None], ROLE_IGNORE),
    (pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), ROLE_CATEGORICAL),
    (pd.to_timedelta([1, 2, 3], unit="D"), ROLE_CATEGORICAL),
])
def test_values_never_guess_an_id_without_a_name_hint(values, expected):
    df = pd.DataFrame({"area": values})
    assert detect_column_roles(df) == {"area": expected}
    assert dataset_io.detect_roles(df) == {"area": expected}


@pytest.mark.parametrize("values", [
    [1, 2, 3], [1.2, 2.2, 3.2], ["a", "b", "c"],
    [1, "1", "c"], [None, None, None], ["a", None, "c"],
    ["a", "a", "c"], [True, False, True],
])
def test_leftmost_exact_id_hint_is_assigned_before_values_are_checked(values):
    df = pd.DataFrame({"area": [100, 200, 300], "cell_id": values,
                       "uuid": ["x", "y", "z"]})
    for detect in (detect_column_roles, dataset_io.detect_roles):
        roles = detect(df, id_hints=["uuid", "cell_id"])
        assert roles == {"area": ROLE_NUMERICAL, "cell_id": ROLE_ROW_ID,
                         "uuid": ROLE_CATEGORICAL}


def test_id_names_match_exactly_and_guess_can_be_disabled():
    df = pd.DataFrame({"Cell_ID": [1, 2], "cell_id": [3, 4]})
    assert detect_column_roles(df, id_hints=["CELL_ID"]) == {
        "Cell_ID": ROLE_NUMERICAL, "cell_id": ROLE_NUMERICAL}
    assert ROLE_ROW_ID not in detect_column_roles(
        df, id_hints=["cell_id"], guess_row_id=False).values()


@pytest.mark.parametrize("role", [ROLE_ROW_ID, ROLE_NUMERICAL, ROLE_CATEGORICAL, ROLE_IGNORE])
def test_explicit_roles_beat_hints_including_a_cleared_id(role):
    df = pd.DataFrame({"cell_id": [None, None], "uuid": [1, 2], "code": [3, 4]})
    assigned = {"cell_id": role, "code": ROLE_NUMERICAL, "absent": ROLE_ROW_ID}
    roles = dataset_io.detect_roles(
        df, id_hints=["cell_id", "uuid"], categorical_hints=["code"],
        assigned_roles=assigned)
    assert roles["cell_id"] == role
    assert roles["code"] == ROLE_NUMERICAL
    assert roles["uuid"] == (ROLE_NUMERICAL if role == ROLE_ROW_ID else ROLE_ROW_ID)
    assert "absent" not in roles


@pytest.mark.parametrize("values, expected", [
    ([1, 2, 3], ROLE_CATEGORICAL), (["1", "2", "3"], ROLE_CATEGORICAL),
    ([None, None, None], ROLE_IGNORE),
])
def test_categorical_hint_beats_numeric_values_but_not_emptiness(values, expected):
    df = pd.DataFrame({"code": values})
    assert dataset_io.detect_roles(df, categorical_hints=["code"]) == {"code": expected}


def test_an_empty_frame_has_no_roles():
    assert detect_column_roles(pd.DataFrame()) == {}


@pytest.mark.parametrize("valid_count, bad_count, null_count, expected", [
    (99, 1, 0, ROLE_NUMERICAL), (98, 2, 0, ROLE_CATEGORICAL),
    (98, 1, 100, ROLE_CATEGORICAL), (99, 1, 100, ROLE_NUMERICAL),
    (100, 0, 5, ROLE_NUMERICAL),
])
def test_numeric_string_coercion_uses_non_null_one_percent_boundary(
        valid_count, bad_count, null_count, expected):
    values = [str(i / 2) for i in range(valid_count)] + ["stray"] * bad_count + [None] * null_count
    df = pd.DataFrame({"feature": values})
    original = df.copy(deep=True)
    assert dataset_io.detect_roles(df)["feature"] == expected
    roles, _groups, numeric = dataset_io.build_working_copy(df)
    assert roles["feature"] == expected
    assert ("feature" in numeric) == (expected == ROLE_NUMERICAL)
    pd.testing.assert_frame_equal(df, original)


def test_coercion_keeps_raw_id_labels_and_interpretation_preserves_rows(monkeypatch):
    # Numeric inference must never rewrite zero-padded raw identifier labels.
    df = pd.DataFrame({"cell_id": ["001", "002", "003"], "feature": [1, 2, 3]})
    original = df.copy(deep=True)
    roles, groups, _numeric = dataset_io.build_working_copy(df, id_hints=["cell_id"])
    monkeypatch.setattr(dataset_io, "_render_warning", lambda *_: None)
    monkeypatch.setattr(dataset_io.st, "write", lambda *_: None)
    interpreted, _groups, complete, row_id = dataset_io.interpret_table(
        df, [], "cell_id", "", feature_groups=groups, use_data_extraction=False)
    assert roles["cell_id"] == ROLE_ROW_ID
    assert complete and row_id == "cell_id"
    assert interpreted["cell_id"].tolist() == ["001", "002", "003"]
    assert len(interpreted) == len(df)
    pd.testing.assert_frame_equal(df, original)


def test_rescued_numeric_strings_keep_the_normal_interpretation_warning(monkeypatch):
    values = [str(i) for i in range(99)] + ["stray"]
    df = pd.DataFrame({"feature": values})
    captured = []
    monkeypatch.setattr(dataset_io, "_render_warning", captured.append)
    monkeypatch.setattr(dataset_io.st, "write", lambda *_: None)
    interpreted, _groups, complete, _id = dataset_io.interpret_table(
        df, [], "", "", feature_groups={}, use_data_extraction=False)
    assert complete and interpreted["feature"].isna().sum() == 1
    assert "1 non-numeric value" in captured[0]
    assert df["feature"].tolist() == values


def test_doubtful_columns_flag_codes_and_row_labels_but_not_measurements():
    """Flags point at guesses worth checking: few whole-number levels or near-unique text."""
    n = 2000
    df = pd.DataFrame({
        "plate": [1, 2, 3] * 666 + [1, 2],
        # 60 levels are under 10% of rows but over the cap, like pixel areas.
        "area": [100 + i % 60 for i in range(n)],
        "lifetime": [0.5 + i % 3 for i in range(n)],
        "notes": [f"note {i}" for i in range(n)],
        # Varied but repeated text could not be a Row ID.
        "label": [f"s{i % 1500}" for i in range(n)],
        "treatment": ["DMSO", "drug"] * (n // 2),
        # Comma decimals and percentages are numbers the reader keeps as text.
        "tau": [f"{i},5" for i in range(n)],
        "pct": [f"{i / 10:.1f}%" for i in range(n)],
        # A measurement with placeholder text, and doses beside "control", a category.
        "lod": [str(0.5 + i) if i % 4 else "below LOD" for i in range(n)],
        "dose": [["0", "10", "100", "control"][i % 4] for i in range(n)],
    })
    doubts = dataset_io.doubtful_columns(df, dataset_io.numeric_column_names(df))
    assert {col: (roles, fix) for col, (roles, fix, _msg) in doubts.items()} == {
        "plate": ((ROLE_NUMERICAL,), ROLE_CATEGORICAL),
        "notes": ((ROLE_CATEGORICAL,), ROLE_IGNORE),
        "label": ((ROLE_CATEGORICAL,), ROLE_IGNORE),
        "tau": ((ROLE_CATEGORICAL, ROLE_NUMERICAL), ROLE_IGNORE),
        "pct": ((ROLE_CATEGORICAL, ROLE_NUMERICAL), ROLE_IGNORE),
        "lod": ((ROLE_CATEGORICAL, ROLE_NUMERICAL), ROLE_IGNORE)}
    assert doubts["plate"][2].startswith(
        "`plate` (Numerical): 3 whole-number values in 2,000 rows.")
    assert "500 rows say `below LOD`" in doubts["lod"][2]
    assert "decimal commas (`0,5`)" in doubts["tau"][2]
    assert "extra characters (`0.0%`)" in doubts["pct"][2]
    assert "make it the Row ID" in doubts["notes"][2]
    assert "Row ID" not in doubts["label"][2]
