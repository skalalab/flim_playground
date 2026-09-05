"""End-to-end review-gate interaction through AppTest. The uploader and reader are stubbed;
file parsing is covered separately.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.column_roles import NO_GROUP, code_span
from src.widgets import analysis_config_widgets as acw
from src.widgets.filter_widgets import selection_key
from src.widgets.review_table_widget import (
    AUTO_DETECT,
    GROUP_HELP,
    GROUP_SECTION,
    _group_key,
)

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
    """Click a chooser row by its profile-name prefix, ignoring the displayed match counts.
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
    """User-table field-of-view columns are ordinary categoricals with no designated hover
    role.
    """
    at = _run({}, path=tmp_path / "analysis_config.toml")
    at = _pick(at, AUTO_DETECT)
    at.session_state._review_confirmed = True
    at.run(timeout=90)
    assert at.session_state.effective_fov_name_col is None
    assert at.session_state._review_roles["image_name"] == "categorical"


def test_a_legacy_profiles_fov_name_never_leaks_into_a_file_that_did_not_match_it(page, tmp_path):
    """An unmatched legacy profile cannot designate a field-of-view column for this file.
    """
    at = _run({"other": {"fov_name_col": "image_name", "categorical_cols": ["image_name"],
                         "all_numerical_features": ["something_else"]}},
              current="other", path=tmp_path / "analysis_config.toml")
    at = _pick(at, AUTO_DETECT)
    at.session_state._review_confirmed = True
    at.run(timeout=90)
    assert at.session_state.effective_fov_name_col is None


@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "custom-groups"])
@pytest.mark.parametrize("harmonic,suffix", [(1, "1st"), (2, "2nd")])
def test_user_tables_offer_and_render_phasor_from_coordinate_names(
        page, tmp_path, grouped, harmonic, suffix):
    g_col = f"Lifetime fit free_NADH_ch1: G({suffix})"
    s_col = f"Lifetime fit free_NADH_ch1: S({suffix})"
    page["frame"] = _frame().assign(**{
        g_col: [0.2, 0.3, 0.4, 0.5],
        s_col: [0.15, 0.25, 0.35, 0.45],
    })
    profile = {"unique_row_id_col": "cell_id",
               "categorical_cols": ["image_name", "treatment"],
               "all_numerical_features": ["Area", g_col, s_col],
               "feature_groups": {"Real": [g_col], "Imaginary": [s_col]} if grouped else {}}
    at = _run({"p": profile}, path=tmp_path / "analysis_config.toml")
    at = at.radio[0].set_value("### **Bivariate**").run(timeout=90)

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state.phasor_available is True
    assert "Phasor Plot" in at.radio[1].options

    at = at.radio[1].set_value("Phasor Plot").run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert _by_key(at, "selectbox", "analysis_control_phasor_harmonic_NADH_ch1").value == harmonic
    assert at.get("plotly_chart")
    assert at.get("download_button")


@pytest.mark.parametrize("unavailable_role", ["missing", "ignore", "categorical"])
def test_user_tables_hide_phasor_without_a_complete_numerical_pair(
        page, tmp_path, unavailable_role):
    g_col = "Lifetime fit free_ch1: G(1st)"
    s_col = "Lifetime fit free_ch1: S(1st)"
    page["frame"] = _frame().assign(**{g_col: [0.2, 0.3, 0.4, 0.5]})
    profile = {"unique_row_id_col": "cell_id",
               "categorical_cols": ["image_name", "treatment"],
               "all_numerical_features": ["Area", g_col]}
    if unavailable_role != "missing":
        page["frame"][s_col] = [0.15, 0.25, 0.35, 0.45]
        if unavailable_role == "ignore":
            profile["ignored_cols"] = [s_col]
        else:
            profile["categorical_cols"].append(s_col)
    at = _run({"p": profile}, path=tmp_path / "analysis_config.toml")
    at = at.radio[0].set_value("### **Bivariate**").run(timeout=90)

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state._review_confirmed is True
    assert at.session_state.phasor_available is False
    assert "Phasor Plot" not in at.radio[1].options


def test_the_read_line_spans_the_filename_it_names(page, tmp_path):
    """A Markdown caption displays the uploaded filename literally."""
    page["name"] = "*draft*.csv"
    at = _run({}, path=tmp_path / "analysis_config.toml")

    captions = [str(caption.value) for caption in at.caption]
    read_line = [text for text in captions if text.startswith("Read ")]
    assert read_line, captions
    assert code_span("*draft*.csv") in read_line[0], read_line[0]


# ------------------------------------------ a reopening does not reset the page below


def _plotting(tmp_path, name="p", measurements=None):
    """Open an auto-applied profile, select a feature, and narrow the treatment filter.
    """
    at = _run({name: {"unique_row_id_col": "cell_id",
                      "categorical_cols": ["image_name", "treatment"],
                      "all_numerical_features": measurements or ["Area"], "ignored_cols": []}},
              current=name, path=tmp_path / "analysis_config.toml")
    menu = [w for w in at.selectbox if "_menu_" in str(w.key)][0]
    at = menu.select("Area").run(timeout=90)
    filter_widget = _by_key(at, "multiselect", selection_key("treatment"))
    return filter_widget.set_value(["DMSO"]).run(timeout=90)


def _configuration(at):
    """What the user set, as the page holds it."""
    state = at.session_state.filtered_state
    return {key: state.get(key) for key in state if "_menu_" in key} | {
        "filter": state.get(selection_key("treatment")),
        "color": state.get("vis_encoding_color_by"),
    }


def _pencil(at):
    return _by_key(at, "button", "review_reopen").click().run(timeout=90)


def test_reopening_the_table_does_not_reset_the_plot_configuration(page, tmp_path):
    """Hidden analysis widgets keep their settings while review owns the screen."""
    at = _plotting(tmp_path)
    before = _configuration(at)
    assert before["filter"] == ["DMSO"], before
    assert at.get("plotly_chart")
    assert at.get("download_button")

    at = _pencil(at)

    assert at.session_state._review_confirmed is False
    assert _configuration(at) == before
    _assert_review_only(at)


def _assert_review_only(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.get("plotly_chart")
    assert not at.get("download_button")
    assert not [w for w in at.selectbox if "_menu_" in str(w.key)]
    assert not [w for w in at.multiselect if w.key == "vis_encoding_color_by"]
    assert not [b for b in at.button if b.key == "review_reopen"]
    assert at.session_state.vis_df is None
    assert at.session_state.analysis_columns is None


def test_profile_names_do_not_make_review_buttons_persistent(page, tmp_path):
    at = _pencil(_plotting(tmp_path, name="p_multiselect"))
    for _ in range(2):
        at.run(timeout=90)
        _assert_review_only(at)
        assert not at.error, [e.value for e in at.error]

    at = next(b for b in at.button if str(b.label) == "Cancel").click().run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert at.get("plotly_chart")


def test_unsaved_role_changes_keep_the_plot_and_exports_hidden(page, tmp_path):
    """An export must never combine saved plot settings with unsaved Ignore roles."""
    at = _pencil(_plotting(tmp_path))
    gen = at.session_state["_review_editor_gen"]

    at = _by_key(at, "selectbox", f"review_role_{gen}_Area").select("Ignore").run(timeout=90)

    assert at.session_state._review_roles["Area"] == "ignore"
    _assert_review_only(at)

    at = [b for b in at.button if str(b.label) == "Cancel"][0].click().run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert at.get("plotly_chart")
    assert at.get("download_button")
    assert at.session_state._review_roles["Area"] == "numerical"


@pytest.mark.parametrize("exit_action", ["Cancel", "Save"])
@pytest.mark.parametrize("hidden_runs", [0, 2], ids=["immediate", "after-reruns"])
def test_review_restores_plot_options(
        page, tmp_path, monkeypatch, exit_action, hidden_runs):
    from streamlit.elements.lib import policies

    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    at = _plotting(tmp_path)
    separate = next(w for w in at.selectbox if w.label == "Separate by")
    at = separate.select("image_name").run(timeout=90)
    comparison_label = "Statistical Comparison between Two Groups"
    comparison = next(w for w in at.radio if w.label == comparison_label)
    at = comparison.set_value("Welch's t-test").run(timeout=90)
    at = next(w for w in at.checkbox if w.label == "Log Y").check().run(timeout=90)
    at = _by_key(at, "number_input", "plot_point_size").set_value(9).run(timeout=90)
    before = _configuration(at)

    at = _pencil(at)
    _assert_review_only(at)
    for _ in range(hidden_runs):
        at.run(timeout=90)
        _assert_review_only(at)
    button = next(b for b in at.button if (
        str(b.label) == "Cancel" if exit_action == "Cancel"
        else str(b.label).startswith("💾 Save to")))
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    at = button.click().run(timeout=90)

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state._review_confirmed is True
    assert _configuration(at) == before
    assert next(w for w in at.selectbox if w.label == "Separate by").value == "image_name"
    assert next(w for w in at.radio if w.label == comparison_label).value == "Welch's t-test"
    assert next(w for w in at.checkbox if w.label == "Log Y").value is True
    assert _by_key(at, "number_input", "plot_point_size").value == 9
    assert not at.warning, [w.value for w in at.warning]
    assert at.get("plotly_chart")
    assert at.get("download_button")
    assert _by_key(at, "button", "review_reopen")


@pytest.mark.parametrize("analysis_type,method", [
    ("Bivariate", "Phasor Plot"),
    ("Multivariate", "Classification"),
])
def test_plot_style_survives_switching_analysis_modules(page, tmp_path, analysis_type, method):
    g_col = "Lifetime fit free_ch1: G(1st)"
    s_col = "Lifetime fit free_ch1: S(1st)"
    page["frame"] = _frame().assign(**{
        g_col: [0.2, 0.3, 0.4, 0.5], s_col: [0.15, 0.25, 0.35, 0.45],
    })
    at = _plotting(tmp_path, measurements=["Area", g_col, s_col])
    expected = {"plot_point_size": 11, "plot_axis_label_size": 20,
                "plot_legend_size": 16, "plot_colormap": "Set2", "plot_show_group_counts": True}
    for key in ("plot_point_size", "plot_axis_label_size", "plot_legend_size"):
        at = _by_key(at, "number_input", key).set_value(expected[key]).run(timeout=90)
    at = _by_key(at, "selectbox", "plot_colormap").select("Set2").run(timeout=90)
    at = _by_key(at, "checkbox", "plot_show_group_counts").check().run(timeout=90)
    original = json.loads(at.get("plotly_chart")[0].proto.spec)

    # The destination initially needs feature selections, so its styling widgets
    # are absent. Settings must survive cleanup on this and subsequent runs.
    at = at.radio[0].set_value(f"### **{analysis_type}**").run(timeout=90)
    assert not at.get("plotly_chart")
    assert not any(w.key == "plot_point_size" for w in at.number_input)
    for _ in range(2):
        assert {key: at.session_state.filtered_state.get(key) for key in expected} == expected
        at.run(timeout=90)
        assert not at.exception, [e.value for e in at.exception]

    at = at.radio[1].set_value(method).run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert {key: at.session_state.filtered_state.get(key) for key in expected} == expected
    if method == "Phasor Plot":
        phasor = json.loads(at.get("plotly_chart")[0].proto.spec)
        assert phasor["layout"]["legend"]["font"]["size"] == 16
        assert _by_key(at, "number_input", "plot_point_size").value == 11

    at = at.radio[0].set_value("### **Univariate**").run(timeout=90)
    menu = next(w for w in at.selectbox if "_menu_" in str(w.key))
    at = menu.select("Area").run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    for kind, key in (("number_input", "plot_point_size"),
                      ("number_input", "plot_axis_label_size"),
                      ("number_input", "plot_legend_size"),
                      ("selectbox", "plot_colormap"),
                      ("checkbox", "plot_show_group_counts")):
        widget = _by_key(at, kind, key)
        assert widget.value == expected[key]
        assert widget.proto.set_value, key
    restored = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert restored["layout"]["xaxis"]["title"]["font"]["size"] == 20
    assert restored["layout"]["legend"]["font"]["size"] == 16
    for field in ("name", "marker"):
        assert [trace.get(field) for trace in restored["data"]] == [
            trace.get(field) for trace in original["data"]]


def test_saving_new_roles_discards_an_invalid_feature_selection(page, tmp_path):
    page["frame"] = _frame().assign(Perimeter=[1.5, 2.5, 3.5, 4.5])
    at = _pencil(_plotting(tmp_path, measurements=["Area", "Perimeter"]))
    gen = at.session_state["_review_editor_gen"]
    at = _by_key(at, "selectbox", f"review_role_{gen}_Area").select("Ignore").run(timeout=90)
    _assert_review_only(at)
    at = next(b for b in at.button if str(b.label).startswith("💾 Save to")).click().run(timeout=90)

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state._review_confirmed is True
    assert "Area" not in at.session_state.analysis_columns
    assert "Perimeter" in at.session_state.analysis_columns
    assert next(w for w in at.selectbox if "_menu_" in str(w.key)).value == "Select"


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
    """Matching headers with an invalid identifier open review with an explanation and Save
    disabled.
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
    """A reopened working copy saves to its source profile or cancels without writing.
    """
    config = tmp_path / "analysis_config.toml"
    at = _run({"pdl1": {"unique_row_id_col": "cell_id",
                        "categorical_cols": ["image_name", "treatment"],
                        "all_numerical_features": ["Area"]}}, path=config)
    next(b for b in at.button if b.key == "review_reopen").click().run(timeout=90)

    labels = [b.label for b in at.button]
    assert any("Save to pdl1" in label for label in labels), labels
    assert any(label == "Cancel" for label in labels), labels
    assert not any("Save profile as" in label for label in labels), labels
    assert not any("Use this" in label for label in labels), labels


# ------------------------------------------------------------------ the review table

def test_a_role_changed_in_the_table_reaches_the_working_copy(page, tmp_path):
    """A role selection updates the working copy through the rendered row widget."""
    at = _pick(_run({}, path=tmp_path / "analysis_config.toml"), AUTO_DETECT)
    gen = at.session_state["_review_editor_gen"]
    assert at.session_state._review_roles["Area"] == "numerical"

    _by_key(at, "selectbox", f"review_role_{gen}_Area").set_value("Ignore").run(timeout=90)
    assert at.session_state._review_roles["Area"] == "ignore"


def _numbering(at):
    return [str(note.value) for note in at.info if "row number" in str(note.value).lower()]


def test_a_table_with_no_row_id_is_told_its_rows_will_be_numbered(page, tmp_path):
    """Review explains generated row numbers before the loader creates them."""
    at = _pick(_run({}, path=tmp_path / "analysis_config.toml"), AUTO_DETECT)
    gen = at.session_state["_review_editor_gen"]
    assert at.session_state._review_roles["cell_id"] == "row_id"
    assert not _numbering(at), "announced numbering while a column still holds Row ID"

    _by_key(at, "selectbox", f"review_role_{gen}_cell_id").set_value("Ignore").run(timeout=90)
    assert _numbering(at), [str(note.value) for note in at.info]


def test_a_second_row_id_is_taken_back_and_the_notice_survives_the_rekey(page, tmp_path):
    """The newly selected identifier wins, and the demotion notice survives re-keyed row
    widgets.
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


# ------------------------------------------------------- bulk group assignment

def _shape_frame():
    """Two measurements with no shared prefix start ungrouped for assignment tests."""
    return pd.DataFrame({
        "cell_id": [1, 2, 3],
        "treatment": ["DMSO", "PD-L1", "DMSO"],
        "area": [10.0, 11.0, 12.0],
        "perimeter": [4.0, 4.4, 4.8],
    })


def _fresh_gate(page, tmp_path, frame=None):
    """An open table over `frame`, on a config holding no profiles at all."""
    page["frame"] = frame if frame is not None else _shape_frame()
    return _pick(_run({}, current="", path=tmp_path / "analysis_config.toml"), AUTO_DETECT)


def _tick(at, col):
    return _by_key(at, "checkbox", f"review_tick_{at.session_state['_review_file_gen']}_{col}")


def test_create_then_assign_puts_two_ticked_rows_in_one_group(page, tmp_path):
    """Adding a group selects it as the destination; Apply updates both selected row
    dropdowns.
    """
    at = _fresh_gate(page, tmp_path)
    gen = at.session_state["_review_editor_gen"]
    _by_key(at, "text_input", f"review_group_name_{gen}").set_value("shape").run(timeout=90)
    _by_key(at, "button", "review_add_group").click().run(timeout=90)

    assert at.session_state._review_group_names == ["shape"]
    assert at.session_state._review_groups == {}, "Add fills nothing on its own"
    gen = at.session_state["_review_editor_gen"]
    assert _by_key(at, "selectbox", f"review_group_target_{gen}").value == "shape"

    _tick(at, "area").check().run(timeout=90)
    _tick(at, "perimeter").check().run(timeout=90)
    _by_key(at, "button", "review_apply_group").click().run(timeout=90)

    assert at.session_state._review_groups == {"area": "shape", "perimeter": "shape"}
    gen = at.session_state["_review_editor_gen"]
    assert _by_key(at, "selectbox", _group_key(gen, "area", True)).value == "shape"
    # Clear completed selections so the next assignment includes only newly ticked rows.
    assert _tick(at, "area").value is False


def test_only_a_measurement_row_offers_a_tick(page, tmp_path):
    """A group is a measurement's to hold, so a tick anywhere else could not be acted on."""
    at = _fresh_gate(page, tmp_path)
    file_gen = at.session_state["_review_file_gen"]
    keys = {widget.key for widget in at.checkbox}

    assert f"review_tick_{file_gen}_area" in keys
    assert f"review_tick_{file_gen}_perimeter" in keys
    assert f"review_tick_{file_gen}_treatment" not in keys
    assert f"review_tick_{file_gen}_cell_id" not in keys


def test_a_tick_survives_a_correction_elsewhere_in_the_table(page, tmp_path):
    """Selection ticks survive when a role correction re-keys the editor dropdowns."""
    at = _fresh_gate(page, tmp_path)
    _tick(at, "area").check().run(timeout=90)
    before = at.session_state["_review_editor_gen"]

    _by_key(at, "selectbox", f"review_role_{before}_treatment").set_value("Row ID").run(timeout=90)

    assert at.session_state["_review_editor_gen"] > before, "no correction fired"
    assert _tick(at, "area").value is True


def test_a_measurements_ungrouped_slot_is_named_after_where_it_goes(page, tmp_path):
    """Ungrouped labels depend on the role while both dropdowns retain the same NO_GROUP
    value.
    """
    at = _fresh_gate(page, tmp_path)
    gen = at.session_state["_review_editor_gen"]
    measurement = _by_key(at, "selectbox", _group_key(gen, "area", True))
    categorical = _by_key(at, "selectbox", _group_key(gen, "treatment", False))

    assert measurement.options == ["Uncategorized"]
    # A group does not apply here, so there is nothing to be un-grouped from.
    assert categorical.options == [NO_GROUP]
    assert measurement.value == NO_GROUP == categorical.value


def test_the_group_cell_is_rekeyed_when_the_role_flips(page, tmp_path):
    """Streamlit fixes option labels at a widget key, so changing roles must re-key the
    group cell. AppTest reports fresh labels even without a new key; assert the key
    itself.
    """
    at = _fresh_gate(page, tmp_path)
    gen = at.session_state["_review_editor_gen"]
    role = _by_key(at, "selectbox", f"review_role_{gen}_area")
    assert at.session_state._review_roles["area"] == "numerical"
    assert _group_key(gen, "area", True) != _group_key(gen, "area", False)

    role.set_value("Categorical").run(timeout=90)

    gen = at.session_state["_review_editor_gen"]
    assert at.session_state._review_roles["area"] == "categorical"
    demoted = _by_key(at, "selectbox", _group_key(gen, "area", False))
    assert demoted.options == [NO_GROUP], "a group cannot apply, so it must not read Uncategorized"
    assert demoted.proto.disabled

    _by_key(at, "selectbox", f"review_role_{gen}_area").set_value("Numerical").run(timeout=90)

    gen = at.session_state["_review_editor_gen"]
    promoted = _by_key(at, "selectbox", _group_key(gen, "area", True))
    assert promoted.options == ["Uncategorized"], "it has somewhere to fall to again"
    assert not promoted.proto.disabled


def test_the_group_section_explains_itself_on_hover_not_on_a_line(page, tmp_path):
    """Group help appears in the heading tooltip without adding a caption."""
    at = _fresh_gate(page, tmp_path, _shape_frame().assign(nadh_t1=[1.0, 2.0, 3.0],
                                                           nadh_t2=[4.0, 5.0, 6.0]))
    assert at.session_state._review_group_names, "expected auto-grouping to make one"
    heading = next((m for m in at.markdown if GROUP_SECTION in str(m.value)), None)
    assert heading is not None, [str(m.value) for m in at.markdown]
    assert heading.proto.help == GROUP_HELP
    assert "feature pickers" in GROUP_HELP
    assert not any("organise the pickers" in str(c.value) for c in at.caption), \
        [str(c.value) for c in at.caption]


def test_with_no_groups_the_section_stays_and_only_add_is_live(page, tmp_path):
    """An empty group section stays visible with Add enabled and group-dependent controls
    disabled.
    """
    at = _fresh_gate(page, tmp_path)   # area/perimeter: no shared prefix, so no groups

    assert at.session_state._review_group_names == []
    assert any("Feature group management" in str(m.value) for m in at.markdown), \
        [str(m.value) for m in at.markdown]
    assert _by_key(at, "button", "review_add_group").disabled is False
    assert _by_key(at, "button", "review_group_rename").disabled is True
    assert _by_key(at, "button", "review_group_delete").disabled is True


def test_select_all_ticks_every_measurement_and_then_clears(page, tmp_path):
    """One Apply button handles bulk assignment."""
    at = _fresh_gate(page, tmp_path)

    _by_key(at, "button", "review_select_all").click().run(timeout=90)
    assert _tick(at, "area").value is True
    assert _tick(at, "perimeter").value is True
    assert _by_key(at, "button", "review_select_all").label == "Clear"

    _by_key(at, "button", "review_select_all").click().run(timeout=90)
    assert _tick(at, "area").value is False
    assert _by_key(at, "button", "review_select_all").label == "Select all"


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
    """Profile management is visible during review and hidden beside the plot."""
    at = _run({"pdl1": _PDL1}, current="pdl1", path=tmp_path / "analysis_config.toml")
    assert not [e for e in at.expander if "Manage saved profiles" in e.label], \
        [e.label for e in at.expander]                       # unpicked: chooser only

    _pick(at, "pdl1")
    assert [e for e in at.expander if "Manage saved profiles" in e.label], \
        [e.label for e in at.expander]


def test_the_chooser_rows_only_pick(page, tmp_path):
    """Chooser rows select profiles; maintenance controls belong in the complete profile
    list.
    """
    at = _pick(_run({"pdl1": _PDL1, "partial": _PARTIAL}, current="pdl1",
                    path=tmp_path / "analysis_config.toml"), "pdl1")
    picks = {b.key for b in at.button if str(b.key).startswith("review_pick_")}
    assert picks == {"review_pick_pdl1", "review_pick_partial",
                     f"review_pick_{AUTO_DETECT}"}, picks


def test_manage_lists_every_profile_including_one_sharing_no_column(page, tmp_path):
    """Management includes profiles excluded from the chooser because they share no
    columns.
    """
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
    """Reopening an exact match exposes profile management even though its chooser is
    suppressed.
    """
    at = _run({"pdl1": _EXACT, "iris": _IRIS}, current="pdl1",
              path=tmp_path / "analysis_config.toml")
    _by_key(at, "button", "review_reopen").click().run(timeout=90)

    assert at.session_state._review_chooser is False
    keys = {b.key for b in at.button}
    assert {"review_arm_delete_pdl1", "review_arm_delete_iris"} <= keys, keys


def test_a_new_profile_is_named_in_the_row_rather_than_behind_a_popover(page, tmp_path):
    """The new-profile name field is visible beside its Save button."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({}, path=config), AUTO_DETECT)
    gen = at.session_state["_review_file_gen"]

    _by_key(at, "text_input", f"review_save_as_name_{gen}").set_value("pdl1").run(timeout=90)
    next(b for b in at.button if "Save profile as" in str(b.label)).click().run(timeout=90)

    assert "pdl1" in acw.list_profiles(), acw.list_profiles()
    assert at.session_state._review_confirmed is True
    assert not any("recognised next time" in str(c.value) for c in at.caption), \
        [str(c.value) for c in at.caption]


def _save_as(at, name):
    """Type `name` into the Save-as box and press whichever button the row is showing."""
    gen = at.session_state["_review_file_gen"]
    _by_key(at, "text_input", f"review_save_as_name_{gen}").set_value(name).run(timeout=90)
    pressed = [b for b in at.button if str(b.key).startswith(f"review_save_as_")
               and str(b.key).endswith(str(gen))]
    assert len(pressed) == 1, [b.key for b in at.button]
    return pressed[0].click().run(timeout=90)


def test_saving_over_an_existing_profile_asks_first(page, tmp_path):
    """The first press arms overwrite; confirmation replaces the existing profile."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"iris": _IRIS}, current="", path=config), AUTO_DETECT)
    gen = at.session_state["_review_file_gen"]

    at = _save_as(at, "iris")
    assert acw.profile_known_columns(acw._get_profile_config("iris")) == {"species", "sepal"}
    assert "_review_saved_as" not in at.session_state, "wrote on the first press"
    assert at.session_state["_review_overwrite_armed"] == "iris"
    assert _by_key(at, "button", f"review_save_as_confirm_{gen}").label.startswith("⚠️")
    assert any("already exists" in str(c.value) for c in at.caption), \
        [str(c.value) for c in at.caption]

    at = _by_key(at, "button", f"review_save_as_confirm_{gen}").click().run(timeout=90)
    assert acw.profile_known_columns(acw._get_profile_config("iris")) == set(_frame().columns)
    assert at.session_state._review_confirmed is True


def test_retyping_the_name_is_the_cancel(page, tmp_path):
    """Changing the name cancels the overwrite confirmation for the previous name."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"iris": _IRIS}, current="", path=config), AUTO_DETECT)
    gen = at.session_state["_review_file_gen"]

    at = _save_as(at, "iris")
    assert at.session_state["_review_overwrite_armed"] == "iris"

    at = _save_as(at, "iris-2")   # a free name: the row goes back to offering a plain save
    assert _by_key(at, "button", f"review_save_as_new_{gen}"), "still armed after a retype"
    assert acw.profile_known_columns(acw._get_profile_config("iris")) == {"species", "sepal"}
    assert "iris-2" in acw.list_profiles(), acw.list_profiles()


def test_deleting_the_armed_profile_disarms_the_row(page, tmp_path):
    """Deleting the overwrite target restores the row to creating a new profile."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"iris": _IRIS}, current="", path=config), AUTO_DETECT)
    gen = at.session_state["_review_file_gen"]

    at = _save_as(at, "iris")
    assert at.session_state["_review_overwrite_armed"] == "iris"

    at = _by_key(at, "button", "review_arm_delete_iris").click().run(timeout=90)
    at = _by_key(at, "button", "review_delete_iris").click().run(timeout=90)
    assert _by_key(at, "button", f"review_save_as_new_{gen}"), \
        f"row still armed for a deleted profile: {[b.key for b in at.button]}"


def test_deleting_the_picked_profile_sends_the_gate_back_to_choosing(page, tmp_path):
    """Deleting the working copy's source returns the gate to profile selection."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"pdl1": _PDL1, "iris": _IRIS}, current="pdl1", path=config), "pdl1")
    assert at.session_state._review_source == "pdl1"

    _by_key(at, "button", "review_arm_delete_pdl1").click().run(timeout=90)
    _by_key(at, "button", "review_delete_pdl1").click().run(timeout=90)

    assert acw.list_profiles() == ["iris"]
    assert "_review_source" not in at.session_state
    assert not any(str(b.label).startswith("pdl1") for b in at.button)


def test_deleting_one_profile_leaves_no_row_armed(page, tmp_path):
    """A surviving profile must not inherit the deleted row's confirmation state."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"pdl1": _PDL1, "partial": _PARTIAL, "iris": _IRIS},
                    current="pdl1", path=config), "pdl1")

    _by_key(at, "button", "review_arm_delete_iris").click().run(timeout=90)
    assert at.session_state._review_delete_armed == "iris"
    # Exactly one confirm on screen, and it names the row that was clicked.
    confirms = {b.key for b in at.button if str(b.key).startswith("review_delete_")}
    assert confirms == {"review_delete_iris", "review_delete_cancel_iris"}, confirms

    # Deleting an unrelated profile keeps review open while the surviving rows render.
    _by_key(at, "button", "review_delete_iris").click().run(timeout=90)

    assert "_review_delete_armed" not in at.session_state
    keys = {b.key for b in at.button}
    assert not [k for k in keys if str(k).startswith("review_delete_")], keys
    assert {"review_arm_delete_pdl1", "review_arm_delete_partial"} <= keys, keys


def test_deleting_the_applied_profile_on_the_reopen_path_reopens_the_chooser(page, tmp_path):
    """Deleting the applied profile reopens the suppressed chooser and removes its working
    copy.
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
    # Cancel cannot restore the deleted profile's decision.
    assert "_review_reopened" not in at.session_state

    # Inspect the delete-triggered run directly. An extra AppTest run can replay stale
    # editor widgets from its mixed tree after Streamlit has discarded their state.
    assert not [e.value for e in at.exception], [e.value for e in at.exception]
    labels = [str(b.label) for b in at.button]
    assert AUTO_DETECT in labels, labels
    assert not any(label.startswith("pdl1") for label in labels), labels


def test_keep_disarms_the_row_without_deleting(page, tmp_path):
    """Armed deletion can be cancelled without deleting the profile."""
    config = tmp_path / "analysis_config.toml"
    at = _pick(_run({"pdl1": _PDL1, "iris": _IRIS}, current="pdl1", path=config), "pdl1")

    _by_key(at, "button", "review_arm_delete_pdl1").click().run(timeout=90)
    _by_key(at, "button", "review_delete_cancel_pdl1").click().run(timeout=90)

    assert "_review_delete_armed" not in at.session_state
    assert acw.list_profiles() == ["pdl1", "iris"]


def test_renaming_a_profile_moves_the_working_copys_save_target(page, tmp_path):
    """Renaming a profile updates the working-copy binding and save target."""
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
    """Bare-mode forms mark Streamlit's main singleton; the fixture must clear that same
    object for later AppTests.
    """
    from streamlit.delta_generator_singletons import get_default_dg_stack_value

    with st.form("bare_mode_form", border=False):
        pass
    main_dg = get_default_dg_stack_value()[0]
    assert main_dg._form_data is not None, "st.form no longer marks the main generator"

    _run_teardown()

    assert main_dg._form_data is None


def test_the_reset_degrades_to_a_no_op_when_streamlits_internals_move(monkeypatch):
    """The fixture tolerates renamed Streamlit internals without failing test collection.
    """
    monkeypatch.setitem(sys.modules, "streamlit.delta_generator_singletons", None)

    _run_teardown()     # raises if the ImportError is not swallowed


def test_what_the_reader_says_about_the_file_is_shown_while_the_gate_is_open(page, tmp_path):
    """Reader warnings remain visible while review is open, including when validation
    blocks Save.
    """
    page["warning"] = ("Warning: 'book.xlsx' has 2 sheets and only the first one, "
                       "'ReadMe', was read ('Data' skipped).")
    at = _run(profiles={}, current="", path=tmp_path / "analysis_config.toml")

    # Warnings remain visible in both the initial chooser and the review table.
    assert [b for b in at.button if str(b.label).startswith(AUTO_DETECT)], \
        f"chooser not open; wrong state under test: {[str(b.label) for b in at.button]}"
    assert [m for m in at.markdown if "ReadMe" in str(m.value)], \
        f"reader warning not rendered on the chooser: {[str(m.value) for m in at.markdown][:6]}"
    at = _pick(at, AUTO_DETECT)
    shown = [str(m.value) for m in at.markdown if "ReadMe" in str(m.value)]
    assert shown, f"reader warning not rendered while the table is open: {[str(m.value) for m in at.markdown][:6]}"

    # File warnings remain visible when invalid content blocks Save.
    page["frame"] = pd.DataFrame({"cell_id": [1, 2], "note": ["a", "b"]})
    at = _pick(_run(profiles={}, current="", path=tmp_path / "analysis_config.toml"), AUTO_DETECT)
    save = [b for b in at.button if "Save" in str(b.label)]   # labelled "💾 Save profile as"
    assert save and all(b.disabled for b in save), \
        f"expected Save blocked with no numerical column: {[(b.label, b.disabled) for b in at.button]}"
    assert [m for m in at.markdown if "ReadMe" in str(m.value)], "warning lost once the gate blocked"
