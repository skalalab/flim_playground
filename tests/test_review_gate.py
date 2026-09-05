"""Review-gate decisions in bare mode: widgets return defaults and no buttons are pressed.
Page interaction is covered in test_review_page.py.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.column_roles import (
    NO_GROUP,
    ROLE_CATEGORICAL,
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
)
from src.widgets import review_table_widget as gate


class _Upload:
    """Only what the gate touches: a name. The frame is passed separately."""

    def __init__(self, name):
        self.name = name


@pytest.fixture
def acw(tmp_path, monkeypatch):
    from src.widgets import analysis_config_widgets as module

    monkeypatch.setattr(module, "_ANALYSIS_CONFIG_PATH", tmp_path / "analysis_config.toml")
    module.st.session_state.clear()
    return module


def _frame():
    return pd.DataFrame({
        "cell_id": [1, 2, 3],
        "treatment": ["DMSO", "PD-L1", "DMSO"],
        "Area": [100.0, 120.0, 140.0],
    })


ROLES = {"cell_id": ROLE_ROW_ID, "treatment": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL}


def _wide_frame():
    """Two measurements share a prefix so a saved profile can override their inferred
    group.
    """
    return pd.DataFrame({
        "cell_id": [1, 2, 3],
        "treatment": ["DMSO", "PD-L1", "DMSO"],
        "nadh_t1": [0.4, 0.5, 0.6],
        "nadh_t2": [2.1, 2.2, 2.3],
    })


_WIDE_ROLES = {"cell_id": ROLE_ROW_ID, "treatment": ROLE_CATEGORICAL,
               "nadh_t1": ROLE_NUMERICAL, "nadh_t2": ROLE_NUMERICAL}


def test_a_file_no_profile_describes_opens_the_gate(acw):
    assert gate.review_gate(_Upload("pdl1_rep1.csv"), _frame()) is None


def test_an_exact_match_skips_the_gate_entirely(acw):
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    decision = gate.review_gate(_Upload("pdl1_rep2.csv"), _frame())
    assert decision is not None
    assert decision["profile"] == "pdl1"


def test_an_auto_applied_profile_brings_its_groups(acw):
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    decision = gate.review_gate(_Upload("pdl1_rep2.csv"), _frame())
    assert decision["groups"] == {"Area": "morphology"}


def test_a_file_with_one_extra_column_is_not_an_exact_match(acw):
    """Containment is deliberately not enough: auto-applying would drop a measurement."""
    acw.save_working_copy("pdl1", ROLES, {})
    wider = _frame().assign(Perimeter=[10.0, 11.0, 12.0])
    assert gate.review_gate(_Upload("pdl1_rep3.csv"), wider) is None


def test_an_empty_profile_never_claims_a_file(acw):
    acw._get_profile_config("blank")
    acw._save_profile_config("blank", {})
    assert gate.review_gate(_Upload("anything.csv"), _frame()) is None


def test_two_profiles_with_the_same_columns_send_the_user_to_the_chooser(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    acw.save_working_copy("pdl1-again", ROLES, {})
    assert gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()) is None


def test_the_decision_survives_a_rerun_of_the_same_file(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    first = gate.review_gate(_Upload("pdl1_rep2.csv"), _frame())
    assert gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()) == first


def test_a_different_file_reopens_the_gate(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    assert gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()) is not None
    other = pd.DataFrame({"sepal": [1.0, 2.0], "species": ["a", "b"]})
    assert gate.review_gate(_Upload("iris.csv"), other) is None


def test_a_confirmed_decision_records_the_configured_row_id(acw):
    """Exports need the configured identifier name, including a blank name for generated
    IDs.
    """
    acw.save_working_copy("pdl1", ROLES, {})
    assert gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()) is not None
    assert gate.configured_row_id() == "cell_id"


def test_a_new_upload_cannot_inherit_the_last_files_configured_row_id(acw):
    """The configured identifier belongs to one upload and must be cleared for the next.
    """
    acw.save_working_copy("pdl1", ROLES, {})
    gate.review_gate(_Upload("pdl1_rep2.csv"), _frame())
    assert gate.configured_row_id() == "cell_id"

    other = pd.DataFrame({"sepal": [1.0, 2.0], "species": ["a", "b"]})
    assert gate.review_gate(_Upload("iris.csv"), other) is None      # the gate opens
    assert gate.configured_row_id() == ""


def test_a_table_with_no_row_id_records_a_blank_not_the_invented_name(acw):
    """Blank is the answer the script re-invents "Row number" from."""
    frame = pd.DataFrame({"treatment": ["DMSO", "PD-L1"], "Area": [1.0, 2.0]})
    roles = {"treatment": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL}
    acw.save_working_copy("no_id", roles, {})
    assert gate.review_gate(_Upload("no_id.csv"), frame) is not None
    assert gate.configured_row_id() == ""


def test_the_same_columns_under_a_different_filename_reopen_the_gate(acw):
    """The fingerprint is name plus columns: a second replicate is a different file, and
    only an exact profile match may skip its gate -- which it then does."""
    acw.save_working_copy("pdl1", ROLES, {})
    gate.review_gate(_Upload("rep2.csv"), _frame())
    decision = gate.review_gate(_Upload("rep3.csv"), _frame())
    assert decision["profile"] == "pdl1"


def test_saving_inside_the_gate_does_not_slam_it_shut(acw):
    """Save makes the profile match the file exactly. Auto-apply is an entry decision
    taken once per file, or that Save would close the table mid-edit."""
    gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())      # opens the gate
    acw.save_working_copy("pdl1", ROLES, {})                  # the Save button's work
    assert gate.review_gate(_Upload("pdl1_rep1.csv"), _frame()) is None


def test_a_confirmed_decision_reports_auto_detect_as_no_profile(acw):
    gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())
    gate.st.session_state._review_confirmed = True
    gate.st.session_state._review_source = gate.AUTO_DETECT
    gate.st.session_state._review_roles = ROLES
    assert gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())["profile"] is None


def test_a_profile_just_saved_is_the_one_the_summary_names(acw):
    """Save as binds auto-detected roles to the profile named in the analysis summary."""
    gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())
    gate.st.session_state._review_roles = ROLES
    gate.st.session_state._review_source = gate.AUTO_DETECT
    gate.st.session_state._review_saved_as = "step6-check"
    gate.st.session_state._review_confirmed = True
    assert gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())["profile"] == "step6-check"


# ------------------------------------------------------ the applied-profile summary


def _summary(monkeypatch, decision):
    """Capture visible summary text after removing CSS and markup."""
    shown = []
    monkeypatch.setattr(gate.st, "markdown", lambda msg, **k: shown.append(msg))
    gate.applied_summary(decision)
    without_css = re.sub(r"<style>.*?</style>", "", " ".join(shown), flags=re.DOTALL)
    return re.sub(r"<[^>]+>", "", without_css).replace("&nbsp;", " ")


def test_the_summary_names_the_profile_and_tallies_the_roles(acw, monkeypatch):
    """The summary identifies the applied profile and counts the roles offered to analysis.
    """
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    shown = _summary(monkeypatch, gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()))

    assert "pdl1" in shown
    assert "1 Categorical" in shown and "1 Numerical" in shown
    assert "Row ID" not in shown, shown


def test_the_summary_leaves_out_the_roles_no_column_holds(acw, monkeypatch):
    """A tally of five roles, three of them zero, buries the two that matter."""
    acw.save_working_copy("pdl1", ROLES, {})
    shown = _summary(monkeypatch, gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()))
    assert "0 " not in shown and "FOV" not in shown and "Ignore" not in shown


def test_the_summary_calls_an_unsaved_working_copy_auto_detected(acw, monkeypatch):
    """"Auto-detected" is a real answer, not a missing one: it is what tells the user
    nothing on disk describes this file yet."""
    gate.review_gate(_Upload("iris.csv"), _frame())
    gate.st.session_state._review_source = gate.AUTO_DETECT
    gate.st.session_state._review_roles = ROLES
    gate.st.session_state._review_confirmed = True
    shown = _summary(monkeypatch, gate.review_gate(_Upload("iris.csv"), _frame()))
    assert "Auto-detected" in shown


def test_the_summary_reports_the_name_a_save_just_gave(acw, monkeypatch):
    gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())
    gate.st.session_state._review_roles = ROLES
    gate.st.session_state._review_source = gate.AUTO_DETECT
    gate.st.session_state._review_saved_as = "step6-check"
    gate.st.session_state._review_confirmed = True
    shown = _summary(monkeypatch, gate.review_gate(_Upload("pdl1_rep1.csv"), _frame()))
    assert "step6-check" in shown and "Auto-detected" not in shown


def test_the_profile_in_use_is_the_matched_one_not_the_last_saved(acw):
    """An exact match uses the matched profile while current_profile follows the last
    write.
    """
    acw.save_working_copy("pdl1", ROLES, {})
    acw.save_working_copy("iris", {"Sepal length": ROLE_NUMERICAL}, {})
    assert gate.st.session_state.current_profile == "iris"

    decision = gate.review_gate(_Upload("pdl1_rep2.csv"), _frame())

    assert decision["profile"] == "pdl1"
    assert gate._applied_profile() == "pdl1"
    assert gate.st.session_state.current_profile == "iris"


# ------------------------------------------------------- the way out of the gate


def test_an_applied_profile_is_written_back_to_itself():
    """An applied profile is the sole save target for its working copy."""
    assert gate.exit_actions("pdl1", reopened=False) == [("save", "pdl1")]


def test_auto_detect_can_only_leave_by_naming_a_new_profile():
    assert gate.exit_actions(None, reopened=False) == [("save_as_new", None)]


def test_reopening_adds_a_way_out_that_changes_nothing():
    """Reopened review offers Cancel without writing the profile."""
    assert gate.exit_actions("pdl1", reopened=True) == [("save", "pdl1"), ("cancel", None)]


def test_no_state_can_leave_the_gate_without_saving():
    """Initial review requires saving a profile before analysis."""
    for applied in ("pdl1", None):
        for reopened in (True, False):
            kinds = [kind for kind, _ in gate.exit_actions(applied, reopened=reopened)]
            assert kinds.count("save") + kinds.count("save_as_new") == 1
            assert "use" not in kinds


# ------------------------------------------------------------ reopening with the pencil


def test_reopening_an_exact_match_asks_no_question(acw):
    """Reopening an exact match suppresses the chooser and retains its save target."""
    acw.save_working_copy("pdl1", ROLES, {})
    gate.review_gate(_Upload("rep2.csv"), _frame())            # auto-applied, no gate
    gate.reopen_gate()

    assert gate.review_gate(_Upload("rep2.csv"), _frame()) is None
    assert gate.st.session_state._review_confirmed is False
    assert gate.st.session_state._review_reopened is True
    assert gate.st.session_state._review_chooser is False


def test_a_file_no_profile_fits_is_always_asked_which_profile(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    wider = _frame().assign(Perimeter=[10.0, 11.0, 12.0])
    assert gate.review_gate(_Upload("rep3.csv"), wider) is None
    assert gate.st.session_state._review_chooser is True
    assert gate.st.session_state.get("_review_reopened") is not True


def test_a_new_file_forgets_that_the_last_one_was_reopened(acw):
    """Otherwise the second file's gate offers a Cancel that has nothing to cancel to."""
    acw.save_working_copy("pdl1", ROLES, {})
    gate.review_gate(_Upload("rep2.csv"), _frame())
    gate.reopen_gate()
    gate.review_gate(_Upload("rep2.csv"), _frame())

    other = pd.DataFrame({"sepal": [1.0, 2.0], "species": ["a", "b"]})
    gate.review_gate(_Upload("iris.csv"), other)
    assert gate.st.session_state.get("_review_reopened") is not True


def test_the_unpicked_chooser_names_no_button_that_is_not_on_screen(acw, monkeypatch):
    """`_render_gate` returns before the table and the button row while nothing is
    picked, so a caption telling the user to press one of them points at empty space."""
    shown = []
    monkeypatch.setattr(gate.st, "caption", lambda msg, **k: shown.append(msg))
    gate._chooser(_frame(), acw.all_profile_columns())
    text = " ".join(shown)
    assert text, "the chooser says nothing at all while unpicked"
    assert "Use this" not in text and "Save" not in text


@pytest.mark.parametrize("ids", [[None, None, None], [1, "1", "a"]])
def test_a_profile_whose_roles_no_longer_work_opens_the_table_instead_of_applying(acw, ids):
    """Matching headers cannot auto-apply a profile whose identifier is blank or non-
    unique.
    """
    acw.save_working_copy("pdl1", ROLES, {})
    invalid = _frame().assign(cell_id=ids)

    assert gate.review_gate(_Upload("rep2.csv"), invalid) is None        # opened, not applied
    assert gate.st.session_state._review_roles["cell_id"] == ROLE_ROW_ID  # on pdl1's roles
    assert gate.st.session_state._review_source == "pdl1"                 # ... and bound to it
    assert gate.st.session_state._review_chooser is False                 # nothing to choose


def test_a_profile_that_still_works_applies_without_a_word(acw):
    """The guard above must not cost the ordinary case its silence."""
    acw.save_working_copy("pdl1", ROLES, {})
    assert gate.review_gate(_Upload("rep2.csv"), _frame())["profile"] == "pdl1"


# ------------------------------------- deleting the profile the working copy is using

def _delete(name):
    """Press Delete on that row's confirm. Bare mode, so the widgets are stepped over."""
    gate._delete_and_refresh(name)


def test_deleting_the_profile_in_force_orphans_the_working_copy(acw, monkeypatch):
    """Deleting a profile applied through Save as clears its working-copy binding."""
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    gate.review_gate(_Upload("flowers.csv"), _frame())          # opens, nothing picked
    gate._load_working_copy(_frame(), gate.AUTO_DETECT)
    acw.save_working_copy("foo", ROLES, {})
    gate.st.session_state._review_saved_as = "foo"
    gate.st.session_state._review_reopened = True
    gate.st.session_state._review_chooser = False
    # Save as keeps Auto-detect as the source and binds the applied profile separately.
    assert gate.st.session_state._review_source == gate.AUTO_DETECT
    assert gate._applied_profile() == "foo"

    _delete("foo")

    assert acw.list_profiles() == []
    assert "_review_saved_as" not in gate.st.session_state
    assert "_review_roles" not in gate.st.session_state
    assert gate._applied_profile() is None
    # Clear both the write target and the previous decision used by Cancel.
    assert gate.exit_actions(gate._applied_profile(),
                             gate.st.session_state.get("_review_reopened", False)) == [
        ("save_as_new", None)]
    assert gate.st.session_state._review_chooser is True


def test_deleting_a_different_profile_leaves_the_working_copy_alone(acw, monkeypatch):
    """The other half of the same predicate: pruning an unrelated profile mid-review must
    not throw away the roles the user is part-way through setting."""
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    acw.save_working_copy("foo", ROLES, {})
    acw.save_working_copy("bystander", {"petal": ROLE_NUMERICAL}, {})
    gate.review_gate(_Upload("flowers.csv"), _frame())
    gate._load_working_copy(_frame(), "foo")

    _delete("bystander")

    assert acw.list_profiles() == ["foo"]
    assert gate._applied_profile() == "foo"
    assert gate.st.session_state._review_roles           # the edit in flight survives


# --------------------------------------------------------------------- feature groups

def test_a_group_the_file_cannot_fill_comes_back_with_the_working_copy(acw):
    """Reload empty groups from their saved names as well as the column-to-group mapping.
    """
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"},
                          group_names=["morphology", "lifetime"])
    gate._load_working_copy(_frame(), "pdl1")
    assert gate.st.session_state._review_group_names == ["morphology", "lifetime"]
    assert gate.st.session_state._review_groups == {"Area": "morphology"}


def test_a_new_column_joins_an_empty_group_that_shares_its_name(acw):
    """New columns can join an existing empty group by matching its name."""
    acw.save_working_copy("pdl1", ROLES, {}, group_names=["nadh"])
    wider = _frame().assign(nadh_t1_mean=[0.4, 0.5, 0.6])
    gate._load_working_copy(wider, "pdl1")
    assert gate.st.session_state._review_groups == {"nadh_t1_mean": "nadh"}


def test_a_group_cannot_be_renamed_to_the_ungrouped_marker(acw, monkeypatch):
    """Renaming to NO_GROUP is rejected so it cannot duplicate the ungrouped option."""
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    acw.save_working_copy("pdl1", ROLES, {"Area": "morphology"})
    gate._load_working_copy(_frame(), "pdl1")

    gate._rename_group("morphology", NO_GROUP)

    assert gate.st.session_state._review_group_names == ["morphology"]
    assert gate.st.session_state._review_groups == {"Area": "morphology"}


def test_a_group_takes_every_column_the_bar_hands_it(acw, monkeypatch):
    """Bulk assignment moves every selected measurement to the destination group."""
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    acw.save_working_copy("pdl1", _WIDE_ROLES, {})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._apply_group(["nadh_t1", "nadh_t2"], "morphology")

    assert gate.st.session_state._review_groups == {
        "nadh_t1": "morphology", "nadh_t2": "morphology"}
    # Retain the destination for another assignment; clear only the selection ticks.
    gen = gate.st.session_state[gate._GEN]
    assert gate.st.session_state[f"review_group_target_{gen}"] == "morphology"


def test_adding_a_group_makes_it_and_points_the_destination_at_it(acw, monkeypatch):
    """Adding a group registers its name and selects it as the bulk-assignment destination.
    """
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    acw.save_working_copy("pdl1", _WIDE_ROLES, {})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._add_group("  lifetime  ")

    assert gate.st.session_state._review_group_names == ["lifetime"]
    assert gate.st.session_state._review_groups == {}, "Add fills nothing"
    gen = gate.st.session_state[gate._GEN]
    assert gate.st.session_state[f"review_group_target_{gen}"] == "lifetime"


def test_a_blank_name_makes_no_group(acw, monkeypatch):
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    warned = []
    monkeypatch.setattr(gate.st, "warning", lambda msg, *a, **k: warned.append(msg))
    acw.save_working_copy("pdl1", _WIDE_ROLES, {})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._add_group("   ")

    assert warned
    assert gate.st.session_state._review_group_names == []


def test_the_bar_can_take_columns_out_of_their_group(acw, monkeypatch):
    """NO_GROUP as the destination is the bulk un-assign -- delete's verb, without
    destroying the group the other members still sit in."""
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    acw.save_working_copy("pdl1", _WIDE_ROLES,
                          {"nadh_t1": "lifetime", "nadh_t2": "lifetime"})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._apply_group(["nadh_t1"], NO_GROUP)

    assert gate.st.session_state._review_groups == {"nadh_t2": "lifetime"}
    assert "lifetime" in gate.st.session_state._review_group_names


def test_a_name_that_is_already_a_group_is_refused(acw, monkeypatch):
    """Duplicate group names would create indistinguishable dropdown options."""
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    warned = []
    monkeypatch.setattr(gate.st, "warning", lambda msg, *a, **k: warned.append(msg))
    acw.save_working_copy("pdl1", _WIDE_ROLES, {"nadh_t2": "lifetime"})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._add_group("lifetime")

    assert warned
    assert gate.st.session_state._review_group_names == ["lifetime"]


def test_renaming_a_group_onto_another_merges_them(acw, monkeypatch):
    """Renaming onto an existing group merges members without duplicating dropdown options.
    """
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    acw.save_working_copy("pdl1", _WIDE_ROLES,
                          {"nadh_t1": "lifetime", "nadh_t2": "morphology"})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._rename_group("lifetime", "morphology")

    assert gate.st.session_state._review_groups == {
        "nadh_t1": "morphology", "nadh_t2": "morphology"}
    assert gate.st.session_state._review_group_names == ["morphology"]
    # The selection follows the group, so Delete and Rename stay live on it.
    gen = gate.st.session_state[gate._GEN]
    assert gate.st.session_state[f"review_group_target_{gen}"] == "morphology"


def test_a_group_cannot_be_called_uncategorized_either(acw, monkeypatch):
    """The displayed ungrouped label is reserved to keep dropdown options distinguishable.
    """
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    warned = []
    monkeypatch.setattr(gate.st, "warning", lambda msg, *a, **k: warned.append(msg))
    acw.save_working_copy("pdl1", _WIDE_ROLES, {})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._add_group(gate.UNGROUPED_LABEL)

    assert warned
    assert gate.st.session_state._review_group_names == []


def test_deleting_a_group_drops_its_members_to_uncategorized(acw, monkeypatch):
    """Deleting a group unassigns its members."""
    monkeypatch.setattr(gate.st, "rerun", lambda *a, **k: None)
    acw.save_working_copy("pdl1", _WIDE_ROLES,
                          {"nadh_t1": "lifetime", "nadh_t2": "lifetime"})
    gate._load_working_copy(_wide_frame(), "pdl1")

    gate._delete_group("lifetime")

    assert gate.st.session_state._review_groups == {}
    assert gate.st.session_state._review_group_names == []


def test_only_a_measurement_can_be_picked(acw):
    """Ignore stale selection ticks when a row has just lost its Numerical role."""
    acw.save_working_copy("pdl1", _WIDE_ROLES, {})
    frame = _wide_frame()
    gate._load_working_copy(frame, "pdl1")
    for col in frame.columns:
        gate.st.session_state[gate._pick_key(col)] = True

    assert gate._picked_columns(frame) == ["nadh_t1", "nadh_t2"]


# ------------------------------------------------- one read of the config per rerun

@pytest.fixture
def count_config_reads(monkeypatch):
    """Count `analysis_config.toml` parses, the way the gate actually reaches them."""
    from src import config as config_module

    calls = []
    real = config_module.toml.load
    monkeypatch.setattr(config_module.toml, "load",
                        lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])
    return calls


def test_the_gate_reads_the_config_once_per_rerun(acw, count_config_reads):
    """Share one config read within a rerun; later reruns must see intervening writes.
    """
    # A near-match, so the gate opens rather than auto-applying: the confirmed path
    # returns above every config read and would score zero without proving anything.
    near = dict(ROLES)
    near.pop("Area")
    acw.save_working_copy("pdl1", near, {})
    assert gate.review_gate(_Upload("rep1.csv"), _frame()) is None
    gate._load_working_copy(_frame(), gate.AUTO_DETECT)      # a pick, so all of it draws

    count_config_reads.clear()
    assert gate.review_gate(_Upload("rep1.csv"), _frame()) is None

    assert len(count_config_reads) == 1, count_config_reads


def test_a_profile_saved_inside_a_run_is_visible_to_the_next_read(acw, count_config_reads):
    """A profile written during a rerun is available to the next config read."""
    acw.save_working_copy("pdl1", ROLES, {})
    assert gate.review_gate(_Upload("rep1.csv"), _frame()) is not None

    acw.save_working_copy("second", {"petal": ROLE_NUMERICAL}, {})

    assert acw.list_profiles() == ["pdl1", "second"]
    assert set(acw.all_profile_columns()) == {"pdl1", "second"}
