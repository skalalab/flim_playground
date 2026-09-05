"""Saving, renaming, deleting, and applying analysis profiles."""
import pytest

from src.column_roles import ROLE_CATEGORICAL, ROLE_IGNORE, ROLE_NUMERICAL, ROLE_ROW_ID


@pytest.fixture
def acw(tmp_path, monkeypatch):
    from src.widgets import analysis_config_widgets as module

    monkeypatch.setattr(module, "_ANALYSIS_CONFIG_PATH", tmp_path / "analysis_config.toml")
    module.st.session_state.pop("current_profile", None)
    return module


ROLES = {"cell_id": ROLE_ROW_ID, "treatment": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL}


def test_saving_a_working_copy_creates_a_profile_that_knows_exactly_those_columns(acw):
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    assert acw.profile_known_columns(acw._get_profile_config("pdl1")) == set(ROLES)


def test_saving_a_working_copy_makes_that_profile_current(acw):
    """Saving a profile makes it current."""
    acw.save_working_copy("pdl1", ROLES, {})
    assert acw._get_current_profile() == "pdl1"


def test_saving_over_a_profile_forgets_the_columns_this_file_did_not_have(acw):
    """Save means 'this profile now describes this file', or the next upload re-asks."""
    acw.save_working_copy("pdl1", dict(ROLES, fad_t1_mean=ROLE_NUMERICAL), {})
    acw.save_working_copy("pdl1", ROLES, {})
    assert "fad_t1_mean" not in acw.profile_known_columns(acw._get_profile_config("pdl1"))


def test_the_group_a_column_was_saved_with_comes_back(acw):
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    assert acw.column_groups(acw._get_profile_config("pdl1")) == {"Area": "morphology"}


def test_list_profiles_reports_what_has_been_saved(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    acw.save_working_copy("iris", ROLES, {})
    assert set(acw.list_profiles()) >= {"pdl1", "iris"}


def test_a_profile_missing_the_newer_keys_is_topped_up_on_read(acw):
    """Reading a legacy profile supplies missing defaults without rewriting the saved file.
    """
    import toml

    path = acw._ANALYSIS_CONFIG_PATH
    path.write_text(toml.dumps({
        "current_profile": "legacy",
        "profiles": {"legacy": {"all_numerical_features": ["Area"]}},
    }))
    before = path.read_text()

    profile_cfg = acw._get_profile_config("legacy")

    assert profile_cfg["unique_row_id_col"] == ""
    assert profile_cfg["categorical_cols"] == []
    assert path.read_text() == before, "the top-up wrote the config"


def test_the_topped_up_categorical_list_is_not_shared_between_profiles(acw):
    """Each migrated profile receives its own categorical list."""
    import toml

    acw._ANALYSIS_CONFIG_PATH.write_text(toml.dumps({
        "current_profile": "a",
        "profiles": {"a": {"all_numerical_features": []},
                     "b": {"all_numerical_features": []}},
    }))
    cfg = acw._migrate_old_config_to_profiles(
        toml.loads(acw._ANALYSIS_CONFIG_PATH.read_text()))

    cfg["profiles"]["a"]["categorical_cols"].append("treatment")

    assert cfg["profiles"]["b"]["categorical_cols"] == []


def test_renaming_a_profile_keeps_its_columns(acw):
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    assert acw.rename_profile("pdl1", "PD-L1 rep 3") == ""
    assert "pdl1" not in acw.list_profiles()
    assert acw.column_groups(acw._get_profile_config("PD-L1 rep 3")) == {"Area": "morphology"}


def test_renaming_the_current_profile_follows_it(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    acw.rename_profile("pdl1", "renamed")
    assert acw._get_current_profile() == "renamed"


def test_renaming_onto_an_existing_name_is_refused(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    acw.save_working_copy("iris", ROLES, {})
    assert acw.rename_profile("pdl1", "iris") != ""
    assert "pdl1" in acw.list_profiles()


def test_renaming_to_a_blank_name_is_refused(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    assert acw.rename_profile("pdl1", "   ") != ""


def test_deleting_a_profile_removes_it(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    acw.save_working_copy("iris", ROLES, {})
    acw.delete_profile("iris")
    assert "iris" not in acw.list_profiles()


def test_deleting_the_current_profile_moves_current_to_a_survivor(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    acw.save_working_copy("iris", ROLES, {})
    acw.delete_profile("iris")
    assert acw._get_current_profile() == "pdl1"


def test_deleting_the_last_profile_is_allowed(acw):
    """Deleting the final profile leaves an empty analysis config."""
    acw.save_working_copy("pdl1", ROLES, {})
    for name in list(acw.list_profiles()):
        acw.delete_profile(name)
    assert acw.all_profile_columns() == {} or all(
        not cols for cols in acw.all_profile_columns().values())


def test_a_saved_profile_matches_the_file_it_was_saved_from(acw):
    """The whole point of Save: the next upload of this file skips the gate."""
    from src.profile_matching import exact_match

    acw.save_working_copy("pdl1", ROLES, {})
    assert exact_match(set(ROLES), acw.all_profile_columns()) == "pdl1"


def test_the_profile_cap_is_enforced_by_the_saver(acw):
    for i in range(acw.MAX_PROFILES):
        acw.save_working_copy(f"p{i}", ROLES, {})
    assert acw.save_working_copy("one too many", ROLES, {}) != ""
    assert "one too many" not in acw.list_profiles()


def test_the_at_the_cap_message_names_a_section_that_is_on_screen(acw):
    """The cap error names the always-visible management section and explains overwrite.
    """
    for i in range(acw.MAX_PROFILES):
        acw.save_working_copy(f"p{i}", ROLES, {})
    message = acw.save_working_copy("one too many", ROLES, {})

    assert acw.MANAGE_LABEL in message, message
    assert "above" not in message, message
    assert "existing name" in message, message   # the escape hatch that always works


def test_saving_over_an_existing_profile_is_never_capped(acw):
    for i in range(acw.MAX_PROFILES):
        acw.save_working_copy(f"p{i}", ROLES, {})
    assert acw.save_working_copy("p0", ROLES, {}) == ""


# ------------------------------- what the working copy hands to interpret_table

def test_profile_roles_and_groups_reads_a_named_profile_not_the_current_one(acw):
    """The file picks the profile, so the matched one need not be the active one."""
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    acw.save_working_copy("iris", {"petal": ROLE_NUMERICAL}, {})   # now current
    roles, groups, names = acw.profile_roles_and_groups("pdl1")
    assert roles["cell_id"] == ROLE_ROW_ID
    assert groups == {"Area": "morphology"}
    assert names == ["morphology"]


def test_an_empty_group_survives_the_round_trip_it_was_saved_for(acw):
    """Empty groups survive through stored group names, which the column mapping cannot
    hold.
    """
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"},
                          group_names=["morphology", "lifetime"])
    _roles, groups, names = acw.profile_roles_and_groups("pdl1")
    assert groups == {"Area": "morphology"}
    assert names == ["morphology", "lifetime"]


def test_no_profile_may_be_named_the_choosers_own_auto_detect_row(acw):
    """Two rows under one widget key raises, and the gate cannot render at all."""
    assert acw.save_working_copy(acw.AUTO_DETECT, ROLES, {})
    assert acw.list_profiles() == []
    acw.save_working_copy("pdl1", ROLES, {})
    assert acw.rename_profile("pdl1", acw.AUTO_DETECT)
    assert acw.list_profiles() == ["pdl1"]


def test_a_name_the_config_file_cannot_store_is_refused(acw):
    """Reject profile names that the TOML serializer cannot read back unchanged."""
    for name in ("run\\2026", 'PD-L1 "high"', "plate\t2"):
        assert acw.save_working_copy(name, ROLES, {}), name
    assert acw.list_profiles() == []

    acw.save_working_copy("pdl1", ROLES, {})
    assert acw.rename_profile("pdl1", "run\\2026")
    assert acw.list_profiles() == ["pdl1"]


def test_the_refusal_names_the_character_that_caused_it(acw):
    """Name the invalid character, including invisible tabs."""
    assert "a backslash" in acw.save_working_copy("run\\2026", ROLES, {})
    assert "a tab" in acw.save_working_copy("plate\t2", ROLES, {})
    both = acw.save_working_copy('a\\b"c', ROLES, {})
    assert "a backslash or a double quote" in both


def test_the_names_people_actually_type_still_save(acw):
    """Supported punctuation and Unicode names survive a save-and-read round trip."""
    for name in ("run 2.0", "PD-L1 (high)", "anti-PD1", "Wenxuan\'s plate", "ünïcode"):
        assert acw.save_working_copy(name, ROLES, {}) == "", name
        # Read the profile using the exact name that saved it.
        assert acw.profile_known_columns(acw._get_profile_config(name)) == set(ROLES), name


def test_working_copy_arguments_carry_the_identifier():
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments(ROLES, {})
    assert args["unique_row_id_col"] == "cell_id"


def test_a_field_of_view_column_is_a_categorical_like_any_other():
    """The user-table handoff includes FOV names as categoricals and no fov_name_col argument."""
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments(dict(ROLES, image_name=ROLE_CATEGORICAL), {})
    assert "image_name" in args["categorical_cols"]
    assert "treatment" in args["categorical_cols"]
    assert "fov_name_col" not in args


def test_the_clustering_columns_the_plots_invent_stay_categorical():
    """GMM_group and friends are added to the frame later; get_features must not
    offer them as measurements."""
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments(ROLES, {})
    assert {"GMM_group", "2D_GMM_group", "k_means_cluster"} <= set(args["categorical_cols"])


@pytest.mark.parametrize("role, is_categorical", [
    (ROLE_NUMERICAL, False),
    (ROLE_IGNORE, False),
    (ROLE_CATEGORICAL, True),
])
def test_a_file_column_named_after_a_clustering_one_keeps_the_role_it_was_given(
        role, is_categorical):
    """A file column keeps its reviewed role when its name matches a generated cluster
    column.
    """
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments(dict(ROLES, k_means_cluster=role), {})
    assert ("k_means_cluster" in args["categorical_cols"]) is is_categorical
    # Names absent from the file remain available for generated clustering labels.
    assert {"GMM_group", "2D_GMM_group"} <= set(args["categorical_cols"])


def test_a_clustering_name_the_file_owns_reaches_the_plots_as_a_measurement():
    """A numerical file column with a clustering name reaches the plots with numeric
    values.
    """
    import pandas as pd

    from src.dataset_io import interpret_table
    from src.widgets.analysis_config_widgets import working_copy_arguments

    df = pd.DataFrame({"cell_id": [1, 2, 3, 4],
                       "treatment": ["ctrl", "ctrl", "drug", "drug"],
                       "k_means_cluster": [1.5, 2.5, 3.5, 4.5]})
    roles = {"cell_id": ROLE_ROW_ID, "treatment": ROLE_CATEGORICAL,
             "k_means_cluster": ROLE_NUMERICAL}
    args = working_copy_arguments(roles, {})

    out, groups, upload_complete, _ = interpret_table(
        df, args["categorical_cols"], args["unique_row_id_col"], None,
        ignored_cols=args["ignored_cols"], feature_groups=args["feature_groups"],
        use_data_extraction=False)

    assert upload_complete
    assert "k_means_cluster" in {col for cols in groups.values() for col in cols}
    assert list(out["k_means_cluster"]) == [1.5, 2.5, 3.5, 4.5]


def test_an_ignored_column_reaches_get_features_by_name():
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments(dict(ROLES, notes=ROLE_IGNORE), {})
    assert args["ignored_cols"] == ["notes"]


def test_the_groups_travel_by_argument_rather_than_being_read_from_the_active_profile():
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments(ROLES, {"Area": "morphology"})
    assert args["feature_groups"] == {"morphology": ["Area"]}


def test_a_table_with_no_row_id_hands_over_a_blank_name():
    """Blank means 'this table has none', which resolve_row_id_col turns into numbers."""
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments({"Area": ROLE_NUMERICAL}, {})
    assert args["unique_row_id_col"] == ""
