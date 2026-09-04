"""Creating, renaming and deleting analysis profiles without the old config panel.

Under the data-driven design a profile is born from a file (`Save as`) rather than
from a name typed into an empty form, and "start over" is delete. These are the four
operations the chooser's rows are a thin wrapper around -- the ✏️ and 🗑️ beside each
profile, on the one screen that lists them.
"""
import pytest

from src.column_roles import ROLE_CATEGORICAL, ROLE_NUMERICAL, ROLE_ROW_ID


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
    """_save_profile_config already sets current_profile, which is what 'creates' means."""
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
    """The one thing `dataset_config_widget` did, folded into the migration.

    Only a profile written before those keys existed can lack them -- every Save goes
    through `apply_column_roles`, which writes both. The top-up is a read-time shape
    guarantee, so it must not write: the file on disk keeps whatever it held until a
    Save replaces the profile whole.
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
    """The top-up seeds a list per profile, never one shared default.

    Shared, a caller that mutated the list it read would hand the next profile
    someone else's categoricals -- and the accessors hand this list straight out.
    """
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
    """'Start over' is delete, and a file no longer needs a profile to exist to load."""
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
    """It used to send the user to a 🗑️ "in the list above" -- the chooser's rows.

    That list holds only profiles sharing a column with the file, so for a file sharing
    none it is empty, and that is exactly the file most likely to be the one hitting the
    cap: a brand new shape with nowhere to go. The message names the manage section
    instead, which renders on every opening of the gate, below the button this error
    appears beside. `MANAGE_LABEL` is shared with the expander that draws it, so a rename
    of the section cannot leave the sentence pointing at nothing.
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
    """The mapping cannot express it, so the names have to come off the stored keys.

    Read back off `column_groups`' values instead and the group is saved and then lost
    on the next upload of the same file -- which is apply_column_groups' `group_names`
    argument doing nothing at all.
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


def test_working_copy_arguments_carry_the_identifier():
    from src.widgets.analysis_config_widgets import working_copy_arguments

    args = working_copy_arguments(ROLES, {})
    assert args["unique_row_id_col"] == "cell_id"


def test_a_field_of_view_column_is_a_categorical_like_any_other():
    """No FOV role, no fov_name_col in the hand-off: image_name is just a category.

    The hand-off must not carry the key at all -- interpret_table takes the FOV name
    positionally, and a "" smuggled through here would read as a designated column that
    happens to be blank rather than as "this branch has none".
    """
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


def test_an_ignored_column_reaches_get_features_by_name():
    from src.column_roles import ROLE_IGNORE
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
