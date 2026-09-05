"""Pure review-table rules: previews, working-copy construction, roles, and validation.
"""
import pandas as pd
import pytest

from src.column_roles import (
    ROLE_CATEGORICAL,
    ROLE_IGNORE,
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
    UNGROUPED_LABEL,
    column_preview,
    detect_column_groups,
    enforce_role_invariants,
    row_id_notice,
    validate_roles,
)
from src.dataset_io import build_working_copy, detect_roles
from src.widgets.analysis_config_widgets import (
    apply_column_groups,
    column_groups,
)

# ------------------------------------------------------------------ column_preview

def test_preview_of_an_empty_column_says_it_will_be_dropped():
    assert column_preview(pd.Series([None, None], dtype=object)) == "empty — will be dropped"


def test_preview_of_a_measurement_is_its_range():
    assert column_preview(pd.Series([291.4, 500.0, 812.7])) == "291.4 – 812.7"


def test_preview_of_a_constant_measurement_is_the_single_value():
    assert column_preview(pd.Series([7.0, 7.0])) == "7"


def test_preview_of_a_category_is_a_value_and_its_level_count():
    assert column_preview(pd.Series(["DMSO", "PD-L1", "DMSO"])) == "DMSO (2 levels)"


def test_preview_of_a_single_level_category_is_singular():
    assert column_preview(pd.Series(["batch A", "batch A"])) == "batch A (1 level)"


def test_preview_counts_levels_over_the_whole_column_not_the_shown_value():
    free_text = pd.Series([f"note {i}" for i in range(1204)])
    assert column_preview(free_text) == "note 0 (1204 levels)"


def test_preview_of_a_boolean_column_is_a_level_count_not_a_range():
    """Boolean previews show values and level counts, matching their categorical role.
    """
    assert column_preview(pd.Series([True, False, True])) == "True (2 levels)"


def test_preview_of_a_rescued_column_is_a_range_not_a_level_count():
    """Numerical previews use the coerced values even when the input column contains text.
    """
    values = pd.Series([f"{0.4 + i * 0.001:.3f}" for i in range(199)] + ["n/a"])
    assert column_preview(values) == "0.400 (200 levels)"          # what the dtype says
    assert column_preview(values, numeric=True) == "0.4 – 0.598"   # what the analysis reads


def test_a_preview_asked_for_a_range_it_cannot_give_falls_back():
    """Boolean and nonnumeric columns fall back to a categorical preview."""
    assert column_preview(pd.Series([True, False]), numeric=True) == "True (2 levels)"
    assert column_preview(pd.Series(["a", "b"]), numeric=True) == "a (2 levels)"


# ------------------------------------------------------- reserved ungrouped label

def test_a_guessed_group_named_uncategorized_merges_into_the_ungrouped_slot():
    """Merge the reserved group label into the ungrouped slot to avoid duplicate options.
    """
    df = pd.DataFrame({"cell_id": [1, 2, 3],
                       "Uncategorized_a": [1.0, 2.0, 3.0],
                       "Uncategorized_b": [4.0, 5.0, 6.0]})
    # Raw grouping finds the prefix; the working copy normalizes the reserved label.
    assert detect_column_groups(["Uncategorized_a", "Uncategorized_b"]) == {
        "Uncategorized_a": UNGROUPED_LABEL, "Uncategorized_b": UNGROUPED_LABEL}

    _roles, groups, _numeric = build_working_copy(df)
    assert groups == {}


def test_a_stored_group_named_uncategorized_merges_too():
    """Normalize the reserved group label when reading an older profile."""
    df = pd.DataFrame({"cell_id": [1, 2, 3], "Area": [1.0, 2.0, 3.0]})
    _roles, groups, _numeric = build_working_copy(
        df, profile_roles={"Area": ROLE_NUMERICAL},
        profile_groups={"Area": UNGROUPED_LABEL})
    assert groups == {}


# ----------------------------------------------------------- enforce_role_invariants

def test_the_column_just_assigned_keeps_the_row_id_and_the_other_is_demoted():
    roles = {"cell_id": ROLE_ROW_ID, "row_index": ROLE_ROW_ID}
    roles, _groups, notices = enforce_role_invariants(
        roles, {}, numeric_cols={"cell_id", "row_index"},
        previous_roles={"cell_id": ROLE_ROW_ID, "row_index": ROLE_NUMERICAL})
    assert roles == {"cell_id": ROLE_NUMERICAL, "row_index": ROLE_ROW_ID}
    assert any("cell_id" in notice for notice in notices)


def test_a_field_of_view_column_is_an_ordinary_categorical():
    """Field-of-view names have the same categorical role and invariants as other text
    columns.
    """
    frame = pd.DataFrame({"image_name": ["fov1", "fov1", "fov2"],
                          "treatment": ["a", "b", "a"],
                          "npix": [10, 20, 30]})
    roles = detect_roles(frame, guess_row_id=False)
    assert roles["image_name"] == ROLE_CATEGORICAL
    assert roles["image_name"] == roles["treatment"]

    fixed, _groups, notices = enforce_role_invariants(
        roles, {}, numeric_cols={"npix"}, previous_roles=roles)
    assert fixed == roles
    assert notices == []


def test_a_demoted_measurement_goes_back_to_numerical_not_categorical():
    """A numeric column displaced as Row ID returns to the Numerical role."""
    roles = {"npix": ROLE_ROW_ID, "cell_id": ROLE_ROW_ID}
    roles, _groups, _notices = enforce_role_invariants(
        roles, {}, numeric_cols={"npix", "cell_id"},
        previous_roles={"npix": ROLE_ROW_ID, "cell_id": ROLE_NUMERICAL})
    assert roles == {"npix": ROLE_NUMERICAL, "cell_id": ROLE_ROW_ID}


def test_with_no_previous_state_the_first_holder_keeps_the_role():
    roles = {"a": ROLE_ROW_ID, "b": ROLE_ROW_ID}
    roles, _groups, _notices = enforce_role_invariants(roles, {}, numeric_cols=set())
    assert roles == {"a": ROLE_ROW_ID, "b": ROLE_CATEGORICAL}


def test_a_group_survives_only_on_a_numerical_row():
    roles = {"Area": ROLE_CATEGORICAL}
    _roles, groups, _notices = enforce_role_invariants(roles, {"Area": "morphology"})
    assert groups == {}


def test_ignoring_a_grouped_column_removes_it_from_its_group():
    roles = {"Area": ROLE_IGNORE, "Perimeter": ROLE_NUMERICAL}
    _roles, groups, _notices = enforce_role_invariants(
        roles, {"Area": "morphology", "Perimeter": "morphology"})
    assert groups == {"Perimeter": "morphology"}


def test_a_group_on_a_numerical_row_is_left_alone():
    roles = {"Area": ROLE_NUMERICAL}
    _roles, groups, _notices = enforce_role_invariants(roles, {"Area": "morphology"})
    assert groups == {"Area": "morphology"}


# -------------------------------------------------------------------- validate_roles

def test_a_table_with_no_numerical_column_cannot_be_used():
    assert "Numerical" in validate_roles({"a": ROLE_CATEGORICAL, "b": ROLE_ROW_ID})


def test_one_numerical_column_is_enough():
    assert validate_roles({"a": ROLE_CATEGORICAL, "b": ROLE_NUMERICAL}) == ""


# --------------------------------------------------------------------- row_id_notice

def test_a_table_with_no_row_id_says_what_will_identify_its_rows():
    """The gate explains generated row numbers before the loader adds them."""
    notice = row_id_notice({"treatment": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL})
    assert "row number" in notice.lower(), notice


def test_an_ignored_column_does_not_count_as_an_identifier():
    """An ignored former identifier is excluded from the identifier notice."""
    assert row_id_notice({"cell_id": ROLE_IGNORE, "Area": ROLE_NUMERICAL})


def test_a_table_with_a_row_id_is_told_nothing():
    assert row_id_notice({"cell_id": ROLE_ROW_ID, "Area": ROLE_NUMERICAL}) == ""


# --------------------------------------------------------------- build_working_copy

@pytest.fixture
def frame():
    return pd.DataFrame({
        "cell_id": [1, 2, 3],
        "treatment": ["DMSO", "PD-L1", "DMSO"],
        "nadh_t1_mean": [480.0, 470.0, 490.0],
        "nadh_t2_mean": [2900.0, 2880.0, 2910.0],
        "nadh_t3_mean": [6100.0, 6050.0, 6150.0],
    })


def test_with_no_profile_every_column_is_detected(frame):
    roles, _groups, _numeric = build_working_copy(frame)
    assert roles == {
        "cell_id": ROLE_ROW_ID,
        "treatment": ROLE_CATEGORICAL,
        "nadh_t1_mean": ROLE_NUMERICAL,
        "nadh_t2_mean": ROLE_NUMERICAL,
        "nadh_t3_mean": ROLE_NUMERICAL,
    }


def test_a_matched_column_keeps_the_role_the_profile_stored(frame):
    """The profile calls treatment a measurement; nothing may re-detect it as a category."""
    roles, _groups, _numeric = build_working_copy(frame, profile_roles={"treatment": ROLE_NUMERICAL})
    assert roles["treatment"] == ROLE_NUMERICAL


def test_a_profile_that_calls_the_identifier_a_measurement_is_obeyed():
    """A saved Numerical role outranks an identifier guess."""
    frame = pd.DataFrame({"alcohol": [9.4, 9.8, 10.1],
                          "quality": [5, 5, 6],
                          "wine_id": [1, 2, 3]})
    stored = {"alcohol": ROLE_NUMERICAL, "quality": ROLE_NUMERICAL,
              "wine_id": ROLE_NUMERICAL}

    assert detect_roles(frame)["wine_id"] == ROLE_ROW_ID       # what the guess says
    roles, _groups, _numeric = build_working_copy(frame, profile_roles=stored)
    assert roles == stored                                     # what the profile says


def test_a_new_column_is_still_guessed_beside_a_profile_that_named_no_identifier():
    """A profile with no Row ID does not prevent a new column from becoming one."""
    frame = pd.DataFrame({"alcohol": [9.4, 9.8, 10.1], "wine_id": [1, 2, 3]})
    roles, _groups, _numeric = build_working_copy(frame, profile_roles={"alcohol": ROLE_NUMERICAL})
    assert roles["wine_id"] == ROLE_ROW_ID


def test_a_matched_column_keeps_the_group_the_profile_stored(frame):
    profile_roles = {"nadh_t1_mean": ROLE_NUMERICAL, "nadh_t2_mean": ROLE_NUMERICAL}
    profile_groups = {"nadh_t1_mean": "lifetime", "nadh_t2_mean": "lifetime"}
    _roles, groups, _numeric = build_working_copy(
        frame, profile_roles=profile_roles, profile_groups=profile_groups)
    assert groups["nadh_t1_mean"] == "lifetime"
    assert groups["nadh_t2_mean"] == "lifetime"


def test_a_new_column_joins_an_existing_group_by_prefix(frame):
    """Rule 2: the group exists because the user made it, so nothing is invented."""
    profile_roles = {"nadh_t1_mean": ROLE_NUMERICAL}
    profile_groups = {"nadh_t1_mean": "nadh"}
    _roles, groups, _numeric = build_working_copy(
        frame, profile_roles=profile_roles, profile_groups=profile_groups)
    assert groups["nadh_t3_mean"] == "nadh"


def test_a_new_column_follows_its_siblings_into_a_renamed_group(frame):
    """New siblings follow the saved group name after a user renames it."""
    profile_roles = {"nadh_t1_mean": ROLE_NUMERICAL, "nadh_t2_mean": ROLE_NUMERICAL}
    profile_groups = {"nadh_t1_mean": "NADH lifetime", "nadh_t2_mean": "NADH lifetime"}
    _roles, groups, _numeric = build_working_copy(
        frame, profile_roles=profile_roles, profile_groups=profile_groups)
    assert groups["nadh_t3_mean"] == "NADH lifetime"
    assert set(groups.values()) == {"NADH lifetime"}


def test_a_sibling_the_profile_knows_attracts_even_when_this_file_lacks_it():
    """The profile is the memory: the renamed group holds no column of *this* file."""
    # Fractional values keep this single-column fixture in the Numerical role.
    df = pd.DataFrame({"nadh_t3_mean": [6100.4, 6050.7]})
    profile_roles = {"nadh_t1_mean": ROLE_NUMERICAL}
    profile_groups = {"nadh_t1_mean": "NADH lifetime"}
    _roles, groups, _numeric = build_working_copy(
        df, profile_roles=profile_roles, profile_groups=profile_groups)
    assert groups == {"nadh_t3_mean": "NADH lifetime"}


def test_grouping_never_reconsiders_a_column_the_profile_knows(frame):
    """A deliberately ungrouped saved column stays ungrouped when new siblings arrive.
    """
    profile_roles = {"nadh_t1_mean": ROLE_NUMERICAL}
    _roles, groups, _numeric = build_working_copy(
        frame, profile_roles=profile_roles, profile_groups={})
    assert "nadh_t1_mean" not in groups
    assert groups["nadh_t2_mean"] == "nadh"
    assert groups["nadh_t3_mean"] == "nadh"


def test_a_column_the_file_lacks_gets_no_row(frame):
    roles, _groups, _numeric = build_working_copy(frame, profile_roles={"fad_t1_mean": ROLE_NUMERICAL})
    assert "fad_t1_mean" not in roles


def test_an_ignored_column_stays_ignored_on_re_upload(frame):
    roles, _groups, _numeric = build_working_copy(frame, profile_roles={"treatment": ROLE_IGNORE})
    assert roles["treatment"] == ROLE_IGNORE


# ------------------------------------------------------------------- group storage

def test_column_groups_inverts_the_stored_mapping():
    cfg = {"feature_groups": {"nadh": ["a", "b"], "fad": ["c"]}}
    assert column_groups(cfg) == {"a": "nadh", "b": "nadh", "c": "fad"}


def test_apply_column_groups_writes_the_storage_format_back():
    cfg = {}
    apply_column_groups(cfg, {"a": "nadh", "b": "nadh", "c": "fad"})
    assert cfg["feature_groups"] == {"nadh": ["a", "b"], "fad": ["c"]}


def test_a_group_the_user_created_but_left_empty_survives_a_save():
    cfg = {}
    apply_column_groups(cfg, {"a": "nadh"}, group_names=["nadh", "morphology"])
    assert cfg["feature_groups"] == {"nadh": ["a"], "morphology": []}


# --------------------------------------------------------------- numeric_column_names

def test_a_column_of_measurements_with_a_stray_na_still_counts_as_numeric():
    """The same 1% rule get_features will apply, so the review table's demotions agree
    with the analysis' own reading of the frame."""
    from src.dataset_io import numeric_column_names

    values = [1.0] * 300 + ["n/a"]
    assert "mostly" in numeric_column_names(pd.DataFrame({"mostly": values}))


def test_a_column_that_is_mostly_text_is_not_numeric():
    from src.dataset_io import numeric_column_names

    values = [1.0] * 200 + ["below LOD"] * 100
    assert numeric_column_names(pd.DataFrame({"half": values})) == set()


def test_build_working_copy_hands_back_the_same_numeric_set_as_the_accessor():
    """The working copy and numeric-column accessor agree after coercion."""
    from src.dataset_io import numeric_column_names

    frame = pd.DataFrame({
        "cell_id": ["c1", "c2", "c3"],
        "treatment": ["DMSO", "PD-L1", "DMSO"],
        "mostly": [1.0, 2.0, "n/a"],          # inside the 1% rule
        "text": ["below LOD", "b", "c"],
        "Area": [100.0, 120.0, 140.0],
    })

    _roles, _groups, numeric = build_working_copy(frame)

    assert numeric == numeric_column_names(frame)
    assert "Area" in numeric and "text" not in numeric


# ---------------------------------------------------- the gate's blocking message

def test_blocking_a_comma_decimal_table_says_why_it_has_no_measurements():
    """A gate rejection for comma decimals includes the decimal-format hint."""
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"cell_id": [1, 2], "t1": ["480,5", "471,2"]})
    roles, _groups, _numeric = build_working_copy(df)
    reason = review_blocking_reason(df, roles)
    assert "Numerical" in reason
    assert "decimal point as a comma" in reason


def test_a_usable_table_has_no_blocking_reason():
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"cell_id": [1, 2], "t1": [480.5, 471.2]})
    roles, _groups, _numeric = build_working_copy(df)
    assert review_blocking_reason(df, roles) == ""


def test_a_table_with_no_numbers_at_all_is_blocked_without_a_decimal_hint():
    """all_text.csv: nothing here is a near-miss measurement, so the hint would mislead."""
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"cell_id": ["a", "b"], "treatment": ["DMSO", "PD-L1"]})
    roles, _groups, _numeric = build_working_copy(df)
    reason = review_blocking_reason(df, roles)
    assert "Numerical" in reason
    assert "comma" not in reason


def test_a_new_unique_column_does_not_take_the_row_id_the_profile_already_names(frame):
    """A newly guessed identifier cannot displace the saved identifier present in the file.
    """
    wider = frame.copy()
    wider.insert(0, "uuid", ["a1", "b2", "c3"])
    roles, _groups, _numeric = build_working_copy(wider, profile_roles={"cell_id": ROLE_ROW_ID})
    assert roles["cell_id"] == ROLE_ROW_ID
    assert roles["uuid"] != ROLE_ROW_ID


def test_the_guess_comes_back_when_the_profiles_identifier_is_not_in_this_file(frame):
    """A missing saved identifier does not suppress a valid identifier guess in this file.
    """
    renamed = frame.rename(columns={"cell_id": "roi_id"})
    roles, _groups, _numeric = build_working_copy(
        renamed, profile_roles={"cell_id": ROLE_ROW_ID, "treatment": ROLE_CATEGORICAL})
    assert [col for col, role in roles.items() if role == ROLE_ROW_ID] == ["roi_id"]


# --------------------------- the gate answers the loader's questions, not just its own

def test_an_empty_row_id_column_is_blocked_at_the_gate_not_two_screens_later():
    """The gate and loader both reject an empty identifier, even when its headers match a
    profile.
    """
    from src.dataset_io import check_and_fix_df, review_blocking_reason

    df = pd.DataFrame({"cell_id": [None, None], "treatment": ["a", "b"], "Area": [1.0, 2.0]})
    roles = {"cell_id": ROLE_ROW_ID, "treatment": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL}

    assert check_and_fix_df(df, ["treatment"], "cell_id", "")[2] != ""
    reason = review_blocking_reason(df, roles)
    assert "cell_id" in reason and "Row ID" in reason


def test_a_row_id_that_repeats_is_blocked_where_the_dropdown_is():
    """The gate and loader both reject repeated identifiers without deleting duplicate
    rows.
    """
    from src.dataset_io import check_and_fix_df, review_blocking_reason

    df = pd.DataFrame({"cell_id": [1, 2, 1, 2], "image_name": ["A", "A", "B", "B"],
                       "Area": [1.0, 2.0, 3.0, 4.0]})
    roles = {"cell_id": ROLE_ROW_ID, "image_name": ROLE_CATEGORICAL,
             "Area": ROLE_NUMERICAL}

    frame, _warning, error = check_and_fix_df(df, ["image_name"], "cell_id", "")
    assert frame is None and "cell_id" in error

    reason = review_blocking_reason(df, roles)
    assert "cell_id" in reason and "Row ID" in reason
    assert "2 of 4" in reason, reason


def test_a_row_id_blank_in_some_rows_is_blocked():
    """Missing identifiers block review before pandas could treat them as duplicate labels."""
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"cell_id": ["a", None, None], "Area": [1.0, 2.0, 3.0]})
    roles = {"cell_id": ROLE_ROW_ID, "Area": ROLE_NUMERICAL}
    reason = review_blocking_reason(df, roles)
    assert "cell_id" in reason and "2 of 3" in reason, reason


def test_a_one_to_one_row_id_is_left_alone():
    """The check has to be a bijection test, not a suspicion: unique and complete passes."""
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"cell_id": ["A01_1", "A01_2", "A02_1"], "Area": [1.0, 2.0, 3.0]})
    roles = {"cell_id": ROLE_ROW_ID, "Area": ROLE_NUMERICAL}
    assert review_blocking_reason(df, roles) == ""


def test_a_table_with_no_identifier_at_all_is_not_blocked():
    """Row numbers are invented for it -- a blank identifier is a choice, not a fault."""
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"treatment": ["a", "b"], "Area": [1.0, 2.0]})
    roles = {"treatment": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL}
    assert review_blocking_reason(df, roles) == ""


def test_columns_marked_numerical_that_hold_no_numbers_are_blocked():
    """A Numerical label cannot make a text-only column plottable."""
    from src.dataset_io import get_features, review_blocking_reason

    df = pd.DataFrame({"cell_id": ["c1", "c2"], "notes": ["ok", "fine"]})
    roles = {"cell_id": ROLE_ROW_ID, "notes": ROLE_NUMERICAL}

    assert get_features(df.copy(), [], use_data_extraction=False,
                        unique_row_id_col="cell_id", ignored_cols=[], feature_groups={})[3] != ""
    assert review_blocking_reason(df, roles) != ""


def test_one_usable_measurement_is_enough_to_pass():
    """Not every Numerical column has to hold numbers -- get_features needs one feature."""
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"cell_id": [1, 2], "notes": ["ok", "fine"], "Area": [1.0, 2.0]})
    roles = {"cell_id": ROLE_ROW_ID, "notes": ROLE_NUMERICAL, "Area": ROLE_NUMERICAL}
    assert review_blocking_reason(df, roles) == ""


def test_an_empty_column_marked_numerical_does_not_count_as_a_measurement():
    """pandas types an all-blank CSV column float64, so a dtype test alone calls it a
    number -- but check_and_fix_df drops it before get_features ever looks at it."""
    from src.dataset_io import review_blocking_reason

    df = pd.DataFrame({"cell_id": [1, 2], "blank": [None, None]})
    roles = {"cell_id": ROLE_ROW_ID, "blank": ROLE_NUMERICAL}
    assert review_blocking_reason(df, roles) != ""
