"""The Data Analysis page wired to the gate, driven end to end under AppTest.

The gap these close: the page has no file_uploader accessor, so the other suites call
`review_gate` in bare mode, where every widget returns its default and nothing is ever
picked or clicked. Everything past that point -- the working copy reaching the page, a
role changed in the table, the roles the page then hands to the plots, a rejection after
the gate closed -- is only reachable here.

The uploader and the reader are replaced, so these exercise the page's own wiring rather
than file parsing: `read_table` has its own suite.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.widgets import analysis_config_widgets as acw
from src.widgets.review_table_widget import AUTO_DETECT

_PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")


class _Upload:
    def __init__(self, name="table.csv"):
        self.name = name


def _frame():
    return pd.DataFrame({
        "cell_id": [1, 2, 3, 4],
        "image_name": ["A01", "A01", "A02", "A02"],
        "treatment": ["DMSO", "PD-L1", "DMSO", "PD-L1"],
        "Area": [100.0, 120.0, 140.0, 160.0],
    })


@pytest.fixture
def page(tmp_path, monkeypatch):
    """A Data Analysis page whose uploader always holds `frame`, on a private config."""
    monkeypatch.setattr(acw, "_ANALYSIS_CONFIG_PATH", tmp_path / "analysis_config.toml")

    state = {"frame": _frame(), "name": "table.csv", "warning": ""}

    def fake_uploader(*_args, **_kwargs):
        return _Upload(state["name"])

    def fake_read_table(_upload):
        return state["frame"].copy(), {}, ",", state["warning"], ""

    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    monkeypatch.setattr(dataset_io, "read_table", fake_read_table)
    return state


def _run(profiles=None, current="p", path=None):
    from streamlit.testing.v1 import AppTest

    if profiles is not None:
        path.write_text(toml.dumps({"current_profile": current, "profiles": profiles}))
    at = AppTest.from_file(_PAGE)
    at.run(timeout=90)
    # Turn off "Use Dataset from Data Extraction" -- the branch the gate lives on.
    at.checkbox[0].uncheck().run(timeout=90)
    return at


def _pick(at, value):
    """Click `value`'s row in the chooser.

    A row is a button labelled "pdl1  —  3 shared · ...", so it is matched by prefix
    rather than by the whole label. Buttons rather than one radio group because the rows
    carry their own ✏️ and 🗑️, and Streamlit cannot split a radio across rows.
    """
    for widget in at.button:
        if str(widget.label).startswith(value):
            widget.click().run(timeout=90)
            return at
    raise AssertionError(f"no chooser row starting {value!r}: "
                         f"{[str(b.label) for b in at.button]} {[e.value for e in at.error]}")


def _by_key(at, kind, key):
    matches = [w for w in getattr(at, kind) if w.key == key]
    assert matches, f"no {kind} keyed {key!r}: {[w.key for w in getattr(at, kind)]}"
    return matches[0]


# --------------------------------------------- no FOV column on the user-table branch

def test_a_user_table_never_has_a_designated_fov_column(page, tmp_path):
    """There is no FOV role, so nothing on this branch can name one.

    image_name is present in the fixture and reads as an ordinary categorical. If a
    designated column ever came back, the four point plots would grow a hover line no
    role stands behind.
    """
    at = _run({}, path=tmp_path / "analysis_config.toml")
    at = _pick(at, AUTO_DETECT)
    at.session_state._review_confirmed = True
    at.run(timeout=90)
    assert at.session_state.effective_fov_name_col is None
    assert at.session_state._review_roles["image_name"] == "categorical"


def test_a_legacy_profiles_fov_name_never_leaks_into_a_file_that_did_not_match_it(page, tmp_path):
    """`other` is merely the profile on disk; this file was never matched against it.

    It still carries fov_name_col, being written before the role was dropped. That key
    is read as one more categorical and never as a designated column, on this profile
    or any other.
    """
    at = _run({"other": {"fov_name_col": "image_name", "categorical_cols": ["image_name"],
                         "all_numerical_features": ["something_else"]}},
              current="other", path=tmp_path / "analysis_config.toml")
    at = _pick(at, AUTO_DETECT)
    at.session_state._review_confirmed = True
    at.run(timeout=90)
    assert at.session_state.effective_fov_name_col is None


# ------------------------------------------------------------------- the chooser

def test_a_second_file_does_not_inherit_the_first_files_chooser_pick(page, tmp_path):
    """Nothing is preselected, so a mismatched upload cannot damage a saved profile."""
    config = tmp_path / "analysis_config.toml"
    at = _run({"pdl1": {"unique_row_id_col": "cell_id", "categorical_cols": ["treatment"],
                        "all_numerical_features": ["Area"]}}, path=config)
    at = _pick(at, "pdl1")
    assert at.session_state._review_source == "pdl1"

    page["frame"] = pd.DataFrame({"sepal": [1.0, 2.0], "species": ["a", "b"]})
    page["name"] = "iris.csv"
    at.run(timeout=90)
    assert "_review_source" not in at.session_state


# --------------------------------------- a profile whose roles stopped fitting the file

def test_a_matching_profile_that_can_no_longer_load_the_file_opens_the_table(page, tmp_path):
    """The one question is asked in one place, and that place is the table.

    Same headers, blank identifier: still an exact match, because a profile remembers
    which column is the identifier and never whether it holds anything. This used to
    auto-apply, fail in check_and_fix_df, and leave an error on screen with no table
    behind it. Now the gate asks its own question before stepping aside, so the file
    lands in the table that can fix it -- with the reason beside the disabled button and
    no chooser, since only one profile can ever know these columns.
    """
    config = tmp_path / "analysis_config.toml"
    page["frame"] = _frame().assign(cell_id=[None] * 4)      # blank in every row
    at = _run({"pdl1": {"unique_row_id_col": "cell_id",
                        "categorical_cols": ["image_name", "treatment"],
                        "all_numerical_features": ["Area"]}}, path=config)

    assert at.session_state._review_confirmed is False        # opened, not applied
    assert at.session_state._review_source == "pdl1"          # ... on pdl1's own roles
    assert at.session_state._review_chooser is False
    assert not any(str(b.label).startswith(AUTO_DETECT) for b in at.button)

    assert any("cell_id" in err.value for err in at.error), [err.value for err in at.error]
    save = [b for b in at.button if "Save to pdl1" in b.label]
    assert save and save[0].disabled, [(b.label, b.disabled) for b in at.button]


def test_the_reopened_table_can_only_write_back_to_the_profile_it_came_from(page, tmp_path):
    """The rule that keeps two profiles from ever holding the same column set.

    A working copy that came from `pdl1` has one save target, so the table it is
    edited in cannot mint a second profile over `pdl1`'s own columns. Cancel is the
    way out that writes nothing, and there is no way out that plots without saving.
    """
    config = tmp_path / "analysis_config.toml"
    at = _run({"pdl1": {"unique_row_id_col": "cell_id",
                        "categorical_cols": ["image_name", "treatment"],
                        "all_numerical_features": ["Area"]}}, path=config)
    next(b for b in at.button if b.key == "review_reopen").click().run(timeout=90)

    labels = [b.label for b in at.button]
    assert any("Save to pdl1" in label for label in labels), labels
    assert any(label == "Cancel" for label in labels), labels
    assert not any("Save as" in label for label in labels), labels
    assert not any("Use this" in label for label in labels), labels


# ------------------------------------------------------------------ the review table

def test_a_role_changed_in_the_table_reaches_the_working_copy(page, tmp_path):
    """That this is checkable at all is the point of the rows.

    Under `st.data_editor` the editing itself was an AppTest blind spot: every rule was
    proved on the pure functions and the wiring between them taken on faith.
    """
    at = _pick(_run({}, path=tmp_path / "analysis_config.toml"), AUTO_DETECT)
    gen = at.session_state["_review_editor_gen"]
    assert at.session_state._review_roles["Area"] == "numerical"

    _by_key(at, "selectbox", f"review_role_{gen}_Area").set_value("Ignore").run(timeout=90)
    assert at.session_state._review_roles["Area"] == "ignore"


def _numbering(at):
    return [str(note.value) for note in at.info if "row number" in str(note.value).lower()]


def test_a_table_with_no_row_id_is_told_its_rows_will_be_numbered(page, tmp_path):
    """The gate is the last screen that can say it.

    Saving with no Row ID is allowed -- the role is optional, and the exit advice offers
    it -- but the column that replaces it is invented by `resolve_row_id_col` after the
    gate has closed, so nothing downstream ever announces where those numbers came from.
    """
    at = _pick(_run({}, path=tmp_path / "analysis_config.toml"), AUTO_DETECT)
    gen = at.session_state["_review_editor_gen"]
    assert at.session_state._review_roles["cell_id"] == "row_id"
    assert not _numbering(at), "announced numbering while a column still holds Row ID"

    _by_key(at, "selectbox", f"review_role_{gen}_cell_id").set_value("Ignore").run(timeout=90)
    assert _numbering(at), [str(note.value) for note in at.info]


def test_a_second_row_id_is_taken_back_and_the_notice_survives_the_rekey(page, tmp_path):
    """Two identifiers cannot both stand, and the repair has to reach the widget.

    The column just clicked keeps the role; the other is demoted to what it can hold --
    Numerical here, since demoting a measurement to Categorical would take it out of the
    analysis. The rows are re-keyed to carry that correction back, so the notice has to
    outlive the rerun in session state or nothing on screen explains the change.
    """
    at = _pick(_run({}, path=tmp_path / "analysis_config.toml"), AUTO_DETECT)
    gen = at.session_state["_review_editor_gen"]
    assert at.session_state._review_roles["cell_id"] == "row_id"

    _by_key(at, "selectbox", f"review_role_{gen}_image_name").set_value("Row ID").run(timeout=90)

    roles = at.session_state._review_roles
    assert roles["image_name"] == "row_id"
    assert roles["cell_id"] == "numerical"
    assert any("cell_id" in str(note.value) for note in at.info), [str(n.value) for n in at.info]


def test_a_group_can_only_be_given_to_a_measurement(page, tmp_path):
    """The one rule a per-cell dropdown cannot express, still enforced after the edit."""
    at = _pick(_run({"pdl1": {"unique_row_id_col": "cell_id",
                              "categorical_cols": ["treatment"],
                              "all_numerical_features": ["Area"],
                              "feature_groups": {"shape": ["Area"]}}},
                    current="pdl1", path=tmp_path / "analysis_config.toml"), "pdl1")
    gen = at.session_state["_review_editor_gen"]
    assert at.session_state._review_groups.get("Area") == "shape"

    _by_key(at, "selectbox", f"review_role_{gen}_Area").set_value("Categorical").run(timeout=90)
    assert "Area" not in at.session_state._review_groups


# ------------------------------ picking lives in the chooser, maintenance below the table

_PDL1 = {"unique_row_id_col": "cell_id", "categorical_cols": ["treatment"],
         "all_numerical_features": ["Area"]}
# The frame's four columns exactly, so this one auto-applies and renders no gate.
_EXACT = {"unique_row_id_col": "cell_id", "categorical_cols": ["image_name", "treatment"],
          "all_numerical_features": ["Area"]}
_IRIS = {"unique_row_id_col": "", "categorical_cols": ["species"],
         "all_numerical_features": ["sepal"]}
# Shares `treatment` and `Area` with the frame, so it earns a chooser row where _IRIS,
# which shares nothing, does not.
_PARTIAL = {"unique_row_id_col": "", "categorical_cols": ["treatment", "species"],
            "all_numerical_features": ["Area"]}


def test_the_profiles_panel_exists_only_inside_the_gate(page, tmp_path):
    """It came back, but as gate furniture rather than a fixture above the plot.

    The panel was retired because it sat over the plot in every session to offer two
    controls wanted a handful of times in a profile's life. That objection was about
    where it lived, not whether it should exist -- so it lives in the gate now, which
    is open only while a file's roles are being decided.
    """
    at = _run({"pdl1": _PDL1}, current="pdl1", path=tmp_path / "analysis_config.toml")
    assert not [e for e in at.expander if "Manage saved profiles" in e.label], \
        [e.label for e in at.expander]                       # unpicked: chooser only

    _pick(at, "pdl1")
    assert [e for e in at.expander if "Manage saved profiles" in e.label], \
        [e.label for e in at.expander]


def test_the_chooser_rows_only_pick(page, tmp_path):
    """Rename and delete left these rows when the zero-shared cutoff arrived.

    A chooser row that carried a \U0001f5d1\ufe0f could only ever offer it for a profile the
    chooser had decided to list, which is the wrong list for a maintenance control.
    """
    at = _pick(_run({"pdl1": _PDL1, "partial": _PARTIAL}, current="pdl1",
                    path=tmp_path / "analysis_config.toml"), "pdl1")
    picks = {b.key for b in at.button if str(b.key).startswith("review_pick_")}
    assert picks == {"review_pick_pdl1", "review_pick_partial",
                     f"review_pick_{AUTO_DETECT}"}, picks


def test_manage_lists_every_profile_including_one_sharing_no_column(page, tmp_path):
    """The whole reason it is a separate list. `iris` shares nothing with this file, so
    the chooser drops it -- and before the split that dropped its \u270f\ufe0f and \U0001f5d1\ufe0f too,
    leaving a profile that could not be reached from anywhere."""
    at = _pick(_run({"pdl1": _PDL1, "iris": _IRIS}, current="pdl1",
                    path=tmp_path / "analysis_config.toml"), "pdl1")

    labels = [str(b.label) for b in at.button]
    assert not any(label.startswith("iris") for label in labels), labels   # not a candidate

    keys = {b.key for b in at.button}
    assert {"review_arm_delete_pdl1", "review_arm_delete_iris"} <= keys, keys
    assert not any(AUTO_DETECT in str(key) and
                   str(key).startswith(("review_arm_delete_", "review_rename_"))
                   for key in keys), keys


def test_manage_is_reachable_on_the_reopen_of_an_exact_match(page, tmp_path):
    """The case that had no way in at all.

    An exact match renders no gate, and the \u270f\ufe0f reopens one whose chooser is suppressed
    -- `chooser_is_needed` is False, because the applied profile already describes the
    file. So the one screen that listed profiles was the one screen this user never saw.
    """
    at = _run({"pdl1": _EXACT, "iris": _IRIS}, current="pdl1",
              path=tmp_path / "analysis_config.toml")
    _by_key(at, "button", "review_reopen").click().run(timeout=90)

    assert at.session_state._review_chooser is False
    keys = {b.key for b in at.button}
    assert {"review_arm_delete_pdl1", "review_arm_delete_iris"} <= keys, keys


def test_a_new_profile_is_named_in_the_row_rather_than_behind_a_popover(page, tmp_path):
    """Naming a new profile is one gesture, not two.

    Every new file shape has to make this write -- nothing reaches a plot unsaved -- so
    the box that names it belongs beside the button that uses it, on screen and typed
    into directly. The row says nothing else: what the name is *for* is the one thing
    the button already says.
    """
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({}, path=config), AUTO_DETECT)
    gen = at.session_state["_review_file_gen"]

    _by_key(at, "text_input", f"review_save_as_name_{gen}").set_value("pdl1").run(timeout=90)
    next(b for b in at.button if "Save as" in str(b.label)).click().run(timeout=90)

    assert "pdl1" in acw.list_profiles(), acw.list_profiles()
    assert at.session_state._review_confirmed is True
    assert not any("recognised next time" in str(c.value) for c in at.caption), \
        [str(c.value) for c in at.caption]


def test_deleting_the_picked_profile_sends_the_gate_back_to_choosing(page, tmp_path):
    """The working copy came from a profile that no longer exists, so there is nothing
    for the table to write back to. Leave the pick standing and the save button offers
    to write to a profile that is gone."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"pdl1": _PDL1, "iris": _IRIS}, current="pdl1", path=config), "pdl1")
    assert at.session_state._review_source == "pdl1"

    _by_key(at, "button", "review_arm_delete_pdl1").click().run(timeout=90)
    _by_key(at, "button", "review_delete_pdl1").click().run(timeout=90)

    assert acw.list_profiles() == ["iris"]
    assert "_review_source" not in at.session_state
    assert not any(str(b.label).startswith("pdl1") for b in at.button)


def test_deleting_one_profile_leaves_no_row_armed(page, tmp_path):
    """The bug this confirm replaced a popover to fix.

    A popover keeps its open state in the browser and takes no key, so its identity was
    its slot: delete the first of two and the survivor slid up into that slot and
    inherited an *open* confirm -- one click from destroying a profile the user had never
    named. The armed row is a profile name in session state now, and the delete clears it.
    """
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"pdl1": _PDL1, "partial": _PARTIAL, "iris": _IRIS},
                    current="pdl1", path=config), "pdl1")

    _by_key(at, "button", "review_arm_delete_iris").click().run(timeout=90)
    assert at.session_state._review_delete_armed == "iris"
    # Exactly one confirm on screen, and it names the row that was clicked.
    confirms = {b.key for b in at.button if str(b.key).startswith("review_delete_")}
    assert confirms == {"review_delete_iris", "review_delete_cancel_iris"}, confirms

    # Deleting a profile the working copy did not come from, so the gate stays open and
    # the surviving rows re-render -- which is where the inherited confirm used to show up.
    _by_key(at, "button", "review_delete_iris").click().run(timeout=90)

    assert "_review_delete_armed" not in at.session_state
    keys = {b.key for b in at.button}
    assert not [k for k in keys if str(k).startswith("review_delete_")], keys
    assert {"review_arm_delete_pdl1", "review_arm_delete_partial"} <= keys, keys


def test_deleting_the_applied_profile_on_the_reopen_path_reopens_the_chooser(page, tmp_path):
    """The case the chooser-visible one does not reach.

    `_review_chooser` is decided once per opening, and on the ✏️ path it is False -- the
    applied profile describes this file exactly, so there is nothing to choose. Delete that
    profile and there is: the working copy has nowhere to be written back to, and the screen
    has to ask again. Left at False the gate rendered a table with no roles behind it.
    """
    at = _run({"pdl1": _EXACT, "iris": _IRIS}, current="pdl1",
              path=tmp_path / "analysis_config.toml")
    _by_key(at, "button", "review_reopen").click().run(timeout=90)
    assert at.session_state._review_chooser is False

    _by_key(at, "button", "review_arm_delete_pdl1").click().run(timeout=90)
    _by_key(at, "button", "review_delete_pdl1").click().run(timeout=90)

    assert acw.list_profiles() == ["iris"]
    assert at.session_state._review_chooser is True
    assert "_review_source" not in at.session_state
    # The previous decision was the profile just deleted, so there is nothing to fall back
    # to. Left standing, Cancel on an auto-detected copy confirmed it straight to the plots
    # without a save -- the one thing the gate exists to prevent.
    assert "_review_reopened" not in at.session_state

    # One clean pass before reading the widgets: the click that deleted rendered the old
    # screen up to the delete button and then `st.rerun()`, so `at.button` after it holds
    # both passes -- a stale "Cancel" and the fresh chooser in one list.
    at.run(timeout=90)
    labels = [str(b.label) for b in at.button]
    assert not [e.value for e in at.exception], [e.value for e in at.exception]
    assert AUTO_DETECT in labels, labels
    assert "Cancel" not in labels, labels
    assert not any(label.startswith("pdl1") for label in labels), labels


def test_keep_disarms_the_row_without_deleting(page, tmp_path):
    """A confirm that cannot be backed out of is a delete button with extra steps."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"pdl1": _PDL1, "iris": _IRIS}, current="pdl1", path=config), "pdl1")

    _by_key(at, "button", "review_arm_delete_pdl1").click().run(timeout=90)
    _by_key(at, "button", "review_delete_cancel_pdl1").click().run(timeout=90)

    assert "_review_delete_armed" not in at.session_state
    assert acw.list_profiles() == ["pdl1", "iris"]


def test_renaming_a_profile_moves_the_working_copys_save_target(page, tmp_path):
    """The working copy is bound to its profile by name. Rename the profile without
    following it and the one write the gate offers points at nothing."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"pdl1": _PDL1}, current="pdl1", path=config), "pdl1")

    gen = at.session_state["_review_file_gen"]
    _by_key(at, "text_input", f"review_rename_{gen}_pdl1").set_value("pdl2").run(timeout=90)
    _by_key(at, "button", "review_rename_submit_pdl1").click().run(timeout=90)

    assert acw.list_profiles() == ["pdl2"]
    assert at.session_state._review_source == "pdl2"
    assert any("Save to pdl2" in str(b.label) for b in at.button), [b.label for b in at.button]


# ------------------------------------------------- the harness's own bare-mode reset

def _reset_body():
    """The autouse fixture's generator, so a test can drive its teardown directly."""
    import conftest

    return conftest._forget_bare_mode_containers._get_wrapped_function()


def _run_teardown():
    """Drive the fixture past its `yield`, which is where the reset lives."""
    gen = _reset_body()()
    next(gen)
    with pytest.raises(StopIteration):
        next(gen)


def test_the_bare_mode_form_mark_is_cleared_off_the_singleton_itself():
    """The reset every AppTest in this process depends on -- see tests/conftest.py.

    A bare-mode `st.form` has no ScriptRunContext to build a child block against, so it
    writes its id onto the *main* DeltaGenerator, and the mark outlives the `with`. The
    next `st.button` anywhere in the process then reads it. Cleared on a copy instead,
    the singleton would stay marked and a later suite would die inside a form it never
    rendered.
    """
    from streamlit.delta_generator_singletons import get_default_dg_stack_value

    with st.form("bare_mode_form", border=False):
        pass
    main_dg = get_default_dg_stack_value()[0]
    assert main_dg._form_data is not None, "st.form no longer marks the main generator"

    _run_teardown()

    assert main_dg._form_data is None


def test_the_reset_degrades_to_a_no_op_when_streamlits_internals_move(monkeypatch):
    """The import lives in the fixture body so a rename cannot fail *collection*.

    `streamlit.delta_generator_singletons` is private with no API guarantee. Imported at
    module scope, a renamed symbol takes down every test in the suite with a traceback
    pointing nowhere near the cause.
    """
    monkeypatch.setitem(sys.modules, "streamlit.delta_generator_singletons", None)

    _run_teardown()     # raises if the ImportError is not swallowed


def test_what_the_reader_says_about_the_file_is_shown_while_the_gate_is_open(page, tmp_path):
    """The scope warning must not wait behind a Save the user cannot reach.

    read_table returns it; interpret_table used to be the only thing that rendered it,
    and interpret_table runs only *after* the gate is saved. So a workbook whose table
    sits on sheet 2 opened the gate on the cover sheet's one junk column, blocked Save
    with "no column is marked Numerical" -- unfixable, there is no measurement to mark --
    and withheld the one line that explains it. The two states this pins are the ones
    that made it invisible: the gate open, and the gate blocked.
    """
    page["warning"] = ("Warning: 'book.xlsx' has 2 sheets and only the first one, "
                       "'ReadMe', was read ('Data' skipped).")
    at = _run(profiles={}, current="", path=tmp_path / "analysis_config.toml")

    # Two stages, because the warning has to survive both: the chooser, which is what a
    # first upload lands on, and the table behind it.
    assert [b for b in at.button if str(b.label).startswith(AUTO_DETECT)], \
        f"chooser not open; wrong state under test: {[str(b.label) for b in at.button]}"
    assert [m for m in at.markdown if "ReadMe" in str(m.value)], \
        f"reader warning not rendered on the chooser: {[str(m.value) for m in at.markdown][:6]}"
    at = _pick(at, AUTO_DETECT)
    shown = [str(m.value) for m in at.markdown if "ReadMe" in str(m.value)]
    assert shown, f"reader warning not rendered while the table is open: {[str(m.value) for m in at.markdown][:6]}"

    # ... and still shown when the gate is blocked, which is when it is needed most:
    # nothing the user can do in the table clears the block, so this line is the only
    # thing pointing at the file.
    page["frame"] = pd.DataFrame({"cell_id": [1, 2], "note": ["a", "b"]})
    at = _pick(_run(profiles={}, current="", path=tmp_path / "analysis_config.toml"), AUTO_DETECT)
    save = [b for b in at.button if "Save" in str(b.label)]   # labelled "💾 Save as"
    assert save and all(b.disabled for b in save), \
        f"expected Save blocked with no numerical column: {[(b.label, b.disabled) for b in at.button]}"
    assert [m for m in at.markdown if "ReadMe" in str(m.value)], "warning lost once the gate blocked"
