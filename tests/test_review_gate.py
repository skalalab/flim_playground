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
def acw(isolated_config_paths):
    from src.widgets import analysis_config_widgets as module

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


def test_no_profile_uses_inference_immediately(acw):
    assert gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())["profile"] is None
    assert not acw._ANALYSIS_CONFIG_PATH.exists()


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
    assert gate.review_gate(_Upload("pdl1_rep3.csv"), wider)["profile"] is None


def test_an_empty_profile_never_claims_a_file(acw):
    acw._get_profile_config("blank")
    acw._save_profile_config("blank", {})
    assert gate.review_gate(_Upload("anything.csv"), _frame())["profile"] is None


def test_multiple_exact_profiles_use_inference(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    acw.save_working_copy("pdl1-again", ROLES, {})
    assert gate.review_gate(_Upload("pdl1_rep2.csv"), _frame())["profile"] is None


def test_the_decision_survives_a_rerun_of_the_same_file(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    first = gate.review_gate(_Upload("pdl1_rep2.csv"), _frame())
    assert gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()) == first


def test_a_different_file_reopens_the_gate(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    assert gate.review_gate(_Upload("pdl1_rep2.csv"), _frame()) is not None
    other = pd.DataFrame({"note": ["a", "b"]})
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

    other = pd.DataFrame({"note": ["a", "b"]})
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
    gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())
    gate.reopen_gate()
    acw.save_working_copy("pdl1", ROLES, {})                  # the Save button's work
    assert gate.review_gate(_Upload("pdl1_rep1.csv"), _frame()) is None


def test_a_confirmed_decision_reports_auto_detect_as_no_profile(acw):
    assert gate.review_gate(_Upload("pdl1_rep1.csv"), _frame())["profile"] is None


def test_a_profile_just_saved_is_the_one_the_summary_names(acw):
    at = _click(_app(), "review_reopen")
    gen = at.session_state._review_file_gen
    next(w for w in at.text_input if w.key == f"review_save_as_name_{gen}").set_value("step6-check").run()
    at = _click(at, f"review_save_as_new_{gen}")
    assert at.session_state._review_saved_as == "step6-check"
    assert at.session_state._review_source == "step6-check"


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
    shown = _summary(monkeypatch, gate.review_gate(_Upload("iris.csv"), _frame()))
    assert "Auto-detected" in shown


def test_the_summary_reports_the_name_a_save_just_gave(acw):
    at = _click(_app(), "review_reopen")
    gen = at.session_state._review_file_gen
    next(w for w in at.text_input if w.key == f"review_save_as_name_{gen}").set_value("step6-check").run()
    at = _click(at, f"review_save_as_new_{gen}")
    assert any("<b>step6-check</b>" in m.value for m in at.markdown)


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
    assert gate.exit_actions("pdl1", reopened=False) == [("use", None), ("save", "pdl1")]


def test_auto_detect_can_use_without_saving():
    assert gate.exit_actions(None, reopened=False) == [("use", None), ("save_as_new", None)]


def test_reopening_adds_a_way_out_that_changes_nothing():
    """Reopened review offers Cancel without writing the profile."""
    assert gate.exit_actions("pdl1", reopened=True) == [("use", None), ("save", "pdl1"), ("cancel", None)]


def test_every_editor_offers_use_without_saving():
    """Every valid draft can resume analysis without persistence."""
    for applied in ("pdl1", None):
        for reopened in (True, False):
            kinds = [kind for kind, _ in gate.exit_actions(applied, reopened=reopened)]
            assert kinds.count("save") + kinds.count("save_as_new") == 1
            assert "use" in kinds


# ------------------------------------------------------------ reopening with the pencil


def test_reopening_an_exact_match_offers_optional_profiles(acw):
    """Reopening retains the draft, offers optional profiles, and enables Cancel."""
    acw.save_working_copy("pdl1", ROLES, {})
    at = _click(_app(_frame().to_dict("list")), "review_reopen")

    assert at.session_state._review_confirmed is False
    assert {"review_pick_pdl1", "review_use", "review_cancel"} <= {b.key for b in at.button}


def test_partial_profile_is_optional_on_reopen(acw):
    acw.save_working_copy("pdl1", ROLES, {})
    wider = _frame().assign(Perimeter=[10.0, 11.0, 12.0])
    at = _click(_app(wider.to_dict("list")), "review_reopen")

    assert at.session_state._review_source == gate.AUTO_DETECT
    assert {"review_pick_pdl1", "review_use", "review_cancel"} <= {b.key for b in at.button}


def test_a_new_file_forgets_that_the_last_one_was_reopened(acw):
    """Otherwise the second file's gate offers a Cancel that has nothing to cancel to."""
    acw.save_working_copy("pdl1", ROLES, {})
    at = _click(_app(_frame().to_dict("list")), "review_reopen")

    at.session_state.data = {"note": ["a", "b"]}
    at.run()

    assert not at.exception
    assert at.session_state._review_source == gate.AUTO_DETECT
    assert not any(b.key == "review_cancel" for b in at.button)


def test_unrelated_profiles_do_not_offer_an_optional_section(acw, monkeypatch):
    """There is no optional profile section when no profile shares a column."""
    shown = []
    monkeypatch.setattr(gate.st, "caption", lambda msg, **k: shown.append(msg))
    gate._chooser(_frame(), acw.all_profile_columns())
    text = " ".join(shown)
    assert text == ""
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


def test_a_profile_that_still_works_applies_without_a_word(acw):
    """The guard above must not cost the ordinary case its silence."""
    acw.save_working_copy("pdl1", ROLES, {})
    assert gate.review_gate(_Upload("rep2.csv"), _frame())["profile"] == "pdl1"


# ------------------------------------- deleting the profile the working copy is using

def _delete(name):
    """Press Delete on that row's confirm. Bare mode, so the widgets are stepped over."""
    gate._delete_and_refresh(name, _frame())


def test_deleting_the_profile_in_force_rebuilds_inference(acw, monkeypatch):
    monkeypatch.setattr(gate.st, "rerun", lambda: None)
    acw.save_working_copy("foo", ROLES, {})
    gate.review_gate(_Upload("flowers.csv"), _frame())
    gate.reopen_gate()
    _delete("foo")
    assert acw.list_profiles() == []
    assert gate.st.session_state._review_roles == ROLES
    assert gate._applied_profile() is None
    assert "_review_snapshot" not in gate.st.session_state
    assert gate.st.session_state._review_confirmed is False


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
    assert gate.review_gate(_Upload("rep1.csv"), _frame()) is not None
    gate.reopen_gate()
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


def _gate_app():
    import pandas as pd
    import streamlit as st
    from src.widgets import review_table_widget as gate
    st.button("Replace upload", key="review_replace", on_click=gate.reset_upload_review)
    frame = pd.DataFrame(st.session_state.get("data", {
        "cell_id": [1, 2, 3], "treatment": ["a", "b", "a"],
        "Area": [1.5, 2.5, 3.5], "Length": [4.5, 5.5, 6.5]}))
    upload = type("Upload", (), {"name": "same.csv"})()
    review_slot = st.empty()
    summary_slot = st.empty()
    review_slot.empty()
    summary_slot.empty()
    with review_slot.container():
        decision = gate.review_gate(upload, frame)
    if decision is None:
        summary_slot.empty()
        st.stop()
    review_slot.empty()
    with summary_slot.container():
        gate.applied_summary(decision)


def _app(data=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_gate_app)
    if data is not None:
        at.session_state.data = data
    at.run()
    assert not at.exception
    return at


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    assert not at.exception, [e.value for e in at.exception]
    if at.error:
        return at
    # AppTest 1.54 merges st.rerun deltas into its prior tree. Render current
    # state after the real handler, retaining live inputs and excluding removed
    # widgets and button triggers. No session decisions are changed here.
    from streamlit.proto.WidgetStates_pb2 import WidgetStates
    from streamlit.testing.v1.element_tree import get_widget_state
    live = WidgetStates()
    for node in at._tree:
        if node.type == "button":
            continue
        try:
            widget = get_widget_state(node)
        except KeyError:
            continue
        if widget is not None:
            live.widgets.append(widget)
    return at._run(widget_state=live)


def _edit(at, column, role):
    gen = at.session_state._review_editor_gen
    return next(w for w in at.selectbox if w.key == f"review_role_{gen}_{column}").select(role).run()


@pytest.mark.parametrize("source", [False, True])
def test_temporary_use_then_cancel_restores_complete_confirmed_copy(acw, source):
    if source:
        acw.save_working_copy("saved", {**ROLES, "Length": ROLE_NUMERICAL},
                              {"Area": "shape"}, group_names=["shape", "empty"])
    before = acw._ANALYSIS_CONFIG_PATH.read_bytes() if source else None
    at = _click(_app(), "review_reopen")
    at = _edit(at, "cell_id", "Ignore")
    at = _edit(at, "Length", "Ignore")
    at = _click(at, "review_use")
    assert at.session_state._review_confirmed
    assert at.session_state._review_configured_row_id == ""
    assert any(("saved · edited for this upload" if source else "This upload only")
               in m.value for m in at.markdown)
    expected = {k: at.session_state[k] for k in (
        "_review_roles", "_review_groups", "_review_group_names", "_review_known_cols",
        "_review_baseline", "_review_source")}
    at = _click(at, "review_reopen")
    at = _edit(at, "Area", "Ignore")
    assert next(b for b in at.button if b.key == "review_use").disabled
    at = _click(at, "review_cancel")
    assert at.session_state._review_confirmed
    assert {k: at.session_state[k] for k in expected} == expected
    assert at.session_state._review_previous_roles == expected["_review_roles"]
    assert at.session_state._review_configured_row_id == ""
    assert (acw._ANALYSIS_CONFIG_PATH.read_bytes() if source else None) == before


def test_optional_reapply_current_saved_profile_discards_draft(acw):
    acw.save_working_copy("Auto-detect", {**ROLES, "Length": ROLE_NUMERICAL}, {})
    at = _click(_app(), "review_reopen")
    assert any("Apply a saved profile (optional)" in m.value for m in at.markdown)
    at = _edit(at, "Length", "Ignore")
    at = _click(at, "review_pick_Auto-detect")
    assert at.session_state._review_roles["Length"] == ROLE_NUMERICAL
    assert not at.session_state._review_confirmed
    at = _click(at, "review_use")
    assert any("<b>Auto-detect</b>" in m.value for m in at.markdown)


def test_uploader_callback_resets_even_same_name_and_headers(acw):
    at = _app()
    at = _click(at, "review_reopen")
    at = _edit(at, "Length", "Ignore")
    at = _click(at, "review_use")
    # Call the exported callback in the script's own Session State context.
    from streamlit.testing.v1 import AppTest
    reset = AppTest.from_string('''import streamlit as st
from src.widgets.review_table_widget import reset_upload_review
st.button("Replace", on_click=reset_upload_review)
''')
    for key, value in at.session_state.filtered_state.items():
        if key.startswith("_review_"):
            reset.session_state[key] = value
    reset.run().button[0].click().run()
    assert "_review_roles" not in reset.session_state
    assert "_review_snapshot" not in reset.session_state
    assert "_review_configured_row_id" not in reset.session_state


@pytest.mark.parametrize("group", [NO_GROUP, gate.UNGROUPED_LABEL])
def test_legacy_ungrouped_exact_source_is_clean_after_noop_use(acw, group):
    acw._save_profile_config("legacy", {
        "unique_row_id_col": "cell_id", "categorical_cols": ["treatment"],
        "all_numerical_features": ["Area", "Length"], "feature_groups": {group: ["Area"]}})
    at = _click(_app(), "review_reopen")
    at = _click(at, "review_use")
    assert any("<b>legacy</b>" in m.value for m in at.markdown)
    assert not any("edited for this upload" in m.value for m in at.markdown)


@pytest.mark.parametrize("failed_write", [False, True])
def test_save_clears_temporary_status_or_failure_keeps_use_available(acw, monkeypatch, failed_write):
    at = _click(_app(), "review_reopen")
    at = _edit(at, "Length", "Categorical")
    gen = at.session_state._review_file_gen
    at = next(w for w in at.text_input if w.key == f"review_save_as_name_{gen}").set_value("new").run()
    if failed_write:
        original = Path.open
        def read_only(path, mode="r", *args, **kwargs):
            if path == acw._ANALYSIS_CONFIG_PATH and "w" in mode:
                raise PermissionError("read-only configuration")
            return original(path, mode, *args, **kwargs)
        monkeypatch.setattr(Path, "open", read_only)
    at = _click(at, f"review_save_as_new_{gen}")
    assert not at.exception
    if failed_write:
        assert not at.session_state._review_confirmed
        assert any("read-only configuration" in e.value for e in at.error)
        assert not acw._ANALYSIS_CONFIG_PATH.exists()
        at = _click(at, "review_use")
        assert at.session_state._review_confirmed
        assert any("This upload only" in m.value for m in at.markdown)
    else:
        assert at.session_state._review_confirmed
        assert acw.profile_roles_and_groups("new")[0]["Length"] == ROLE_CATEGORICAL
        assert any("<b>new</b>" in m.value for m in at.markdown)
        assert at.session_state._review_known_cols == set(at.session_state._review_roles)


@pytest.mark.parametrize("delete_source", ["draft", "snapshot", "last"])
def test_delete_cannot_restore_deleted_source_and_keeps_editor_populated(acw, delete_source):
    roles = {**ROLES, "Length": ROLE_NUMERICAL}
    acw.save_working_copy("A", roles, {})
    if delete_source != "last":
        acw.save_working_copy("B", {"Length": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL}, {})
    at = _click(_app(), "review_reopen")
    if delete_source != "last":
        at = _click(at, "review_pick_B")
    deleted = "B" if delete_source == "draft" else "A"
    at = _click(at, f"review_arm_delete_{deleted}")
    at = _click(at, f"review_delete_{deleted}")
    assert not at.exception
    assert not at.session_state._review_confirmed
    assert "_review_snapshot" not in at.session_state
    assert not any(b.key == "review_cancel" for b in at.button)
    assert at.session_state._review_roles
    if delete_source == "snapshot":
        assert at.session_state._review_source == "B"
        assert at.session_state._review_roles["Length"] == ROLE_CATEGORICAL
    else:
        assert at.session_state._review_source == gate.AUTO_DETECT


def test_rename_updates_both_draft_and_confirmed_snapshot_source(acw):
    acw.save_working_copy("A", {**ROLES, "Length": ROLE_NUMERICAL}, {})
    at = _click(_app(), "review_reopen")
    gen = at.session_state._review_file_gen
    at = next(w for w in at.text_input if w.key == f"review_rename_{gen}_A").set_value("renamed").run()
    at = _click(at, "review_rename_submit_A")
    assert at.session_state._review_source == "renamed"
    assert at.session_state._review_snapshot["_review_source"] == "renamed"
    at = _click(at, "review_cancel")
    assert any("<b>renamed</b>" in m.value for m in at.markdown)


@pytest.mark.parametrize("save", [False, True])
def test_only_save_teaches_categorical_hint_to_a_new_schema(acw, save):
    at = _click(_app(), "review_reopen")
    at = _edit(at, "Length", "Categorical")
    if save:
        gen = at.session_state._review_file_gen
        at = next(w for w in at.text_input if w.key == f"review_save_as_name_{gen}").set_value("learned").run()
        at = _click(at, f"review_save_as_new_{gen}")
    else:
        at = _click(at, "review_use")
    # A new schema must build a new working copy with read-only combined hints.
    next_data = {"id": ["a", "b", "c"], "Length": [1.5, 2.5, 3.5], "Other": [2.5, 3.5, 4.5]}
    at.session_state.data = next_data
    at.run()
    assert at.session_state._review_roles["Length"] == (ROLE_CATEGORICAL if save else ROLE_NUMERICAL)
    if save:
        acw.delete_profile("learned")
        at.run()
        assert at.session_state._review_roles["Length"] == ROLE_CATEGORICAL
        at.session_state.data = {**next_data, "extra": [7.5, 8.5, 9.5]}
        at.run()
        assert at.session_state._review_roles["Length"] == ROLE_NUMERICAL


@pytest.mark.parametrize("saved", [False, True])
def test_group_metadata_snapshot_is_independent_and_cancel_restores_temporary_groups(acw, saved):
    if saved:
        acw.save_working_copy("saved", {**ROLES, "Length": ROLE_NUMERICAL}, {}, group_names=["empty"])
    at = _click(_app(), "review_reopen")
    gen = at.session_state._review_editor_gen
    next(w for w in at.text_input if w.key == f"review_group_name_{gen}").set_value("temporary").run()
    at = _click(at, "review_add_group")
    gen = at.session_state._review_editor_gen
    next(w for w in at.selectbox if w.key == _group_key_for_test(gen, "Area")).select("temporary").run()
    at = _click(at, "review_use")
    expected_names = list(at.session_state._review_group_names)
    at = _click(at, "review_reopen")
    gen = at.session_state._review_editor_gen
    next(w for w in at.text_input if w.key == f"review_group_name_{gen}").set_value("discard").run()
    at = _click(at, "review_add_group")
    assert at.session_state._review_snapshot["_review_group_names"] == expected_names
    at = _click(at, "review_cancel")
    assert at.session_state._review_group_names == expected_names
    assert at.session_state._review_groups == {"Area": "temporary"}


def _group_key_for_test(gen, column):
    return gate._group_key(gen, column, True)


def test_use_remains_available_at_profile_cap_and_failed_save_adds_no_slot(acw):
    for index in range(acw.MAX_PROFILES):
        acw.save_working_copy(f"p{index}", ROLES, {})
    at = _click(_app(), "review_reopen")
    gen = at.session_state._review_file_gen
    next(w for w in at.text_input if w.key == f"review_save_as_name_{gen}").set_value("excess").run()
    at = _click(at, f"review_save_as_new_{gen}")
    assert not at.session_state._review_confirmed
    assert any("20 profiles" in e.value for e in at.error)
    at = _click(at, "review_use")
    assert at.session_state._review_confirmed
    assert len(acw.list_profiles()) == acw.MAX_PROFILES


def test_complete_headers_include_ignored_columns_for_exact_matching(acw):
    from src.column_roles import ROLE_IGNORE
    acw.save_working_copy("complete", {**ROLES, "noise": ROLE_IGNORE}, {})
    assert gate.review_gate(_Upload("same.csv"), _frame())["profile"] is None
    gate.reset_upload_review()
    decision = gate.review_gate(_Upload("same.csv"), _frame().assign(noise=["x", "y", "z"]))
    assert decision["profile"] == "complete"
    assert decision["roles"]["noise"] == ROLE_IGNORE


def test_same_upload_reruns_keep_decisions_but_replacement_callback_revalidates_raw_values(acw):
    acw.save_working_copy("saved", {**ROLES, "Length": ROLE_NUMERICAL}, {})
    before = acw._ANALYSIS_CONFIG_PATH.read_bytes()
    at = _app()
    at.session_state.data = {"cell_id": [1, 1, 3], "treatment": ["a", "b", "a"],
                             "Area": [1.5, 2.5, 3.5], "Length": [4.5, 5.5, 6.5]}
    at.run()
    assert at.session_state._review_confirmed
    at = _click(at, "review_replace")
    assert not at.session_state._review_confirmed
    assert at.session_state._review_roles["cell_id"] == ROLE_ROW_ID
    assert any("cell_id" in e.value for e in at.error)
    assert next(b for b in at.button if b.key == "review_use").disabled
    assert not any(b.key == "review_cancel" for b in at.button)
    assert acw._ANALYSIS_CONFIG_PATH.read_bytes() == before


@pytest.mark.parametrize("absent", [False, True])
def test_saved_profile_schema_changes_count_as_upload_edits(acw, absent):
    roles = dict(ROLES)
    if absent:
        roles["absent"] = ROLE_NUMERICAL
    acw.save_working_copy("partial", roles, {})
    at = _click(_app(), "review_reopen")
    at = _click(at, "review_pick_partial")
    at = _click(at, "review_use")
    assert any("partial · edited for this upload" in m.value for m in at.markdown)


def test_no_usable_measurement_opens_populated_editor_and_blocks_both_acceptance_actions(acw):
    at = _app({"note": ["a", "b", "c"]})
    assert at.session_state._review_roles == {"note": ROLE_CATEGORICAL}
    assert at.selectbox
    assert next(b for b in at.button if b.key == "review_use").disabled
    assert next(b for b in at.button if str(b.key).startswith("review_save_as_new_")).disabled
    assert not any(b.key == "review_cancel" for b in at.button)
    assert at.error
    assert not acw._ANALYSIS_CONFIG_PATH.exists()


def test_noop_use_keeps_auto_detected_status_and_does_not_write(acw):
    at = _click(_app(), "review_reopen")
    at = _click(at, "review_use")
    assert at.session_state._review_confirmed
    assert any("<b>Auto-detected</b>" in m.value for m in at.markdown)
    assert not acw._ANALYSIS_CONFIG_PATH.exists()


def test_exact_name_save_as_overwrite_confirmation_works_at_cap(acw):
    for index in range(acw.MAX_PROFILES):
        acw.save_working_copy(f"p{index}", ROLES, {})
    at = _click(_app(), "review_reopen")
    gen = at.session_state._review_file_gen
    next(w for w in at.text_input if w.key == f"review_save_as_name_{gen}").set_value("p0").run()
    before = acw._ANALYSIS_CONFIG_PATH.read_bytes()
    at = _click(at, f"review_save_as_new_{gen}")
    assert acw._ANALYSIS_CONFIG_PATH.read_bytes() == before
    assert not at.session_state._review_confirmed
    assert at.session_state._review_overwrite_armed == "p0"
    at = _click(at, f"review_save_as_confirm_{gen}")
    assert at.session_state._review_confirmed
    assert at.session_state._review_source == "p0"
    assert len(acw.list_profiles()) == acw.MAX_PROFILES
    assert set(acw.profile_roles_and_groups("p0")[0]) == {"cell_id", "treatment", "Area", "Length"}
