"""FD owns its category view and two-mode decoration independently of FC and PP."""

import pytest
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.testing.v1 import AppTest

from src.widgets import visualization_widgets as vw


SEPARATE_KEY = "vis_encoding_fd_separate_by"
MODE_KEY = "vis_encoding_fd_point_mode"
CATEGORY_KEY = "vis_encoding_fd_category"


def app():
    import pandas as pd
    import streamlit as st
    from src.widgets.analysis_widget_state import (
        analysis_control_keys, preserve_analysis_controls,
    )
    from src.widgets.visualization_widgets import visual_encoding_channels_widget

    def open_review():
        if st.session_state.review:
            st.session_state.saved_controls = analysis_control_keys(st.session_state)

    method = st.selectbox("Method", ["FD", "FC", "PP", "DR", "Histogram"], key="method")
    st.checkbox("Review", key="review", on_change=open_review)
    preserve_analysis_controls(st.session_state, st.session_state.get("saved_controls", ()))
    if st.session_state.review:
        st.stop()
    data = pd.DataFrame({
        "day": ["Day 2", "Day 10"] * 4,
        "treatment": ["A", "A", "B", "B"] * 2,
        "dish": ["d1", "d2", "d3", "d4"] * 2,
        "patient": ["p1", "p1", "p2", "p2"] * 2,
    })
    if st.session_state.get("single"):
        data = data.iloc[:1]
    if st.session_state.get("one_replicate"):
        data = data[data["dish"] == "d1"]
    if st.session_state.get("removed"):
        data = data.drop(columns="day")
    categories = st.session_state.get("categoricals", list(data.columns))
    st.session_state.result = visual_encoding_channels_widget(
        data, categories, point_based=method != "Histogram",
        separate_by_available=method != "Histogram", subcolor_available=method == "FC",
        collapse_available=method in {"FD", "FC"},
        separate_by_mode={"FD": "distribution", "PP": "subplots", "DR": "facets"}.get(
            method, "sections"),
    )
    st.session_state.pop("saved_controls", None)


def run(at):
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [e.value for e in at.warning]
    return at


def new(state=None):
    at = AppTest.from_function(app)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    return run(at)


def test_fd_renders_four_aligned_slots_and_only_opacity_shape_modes():
    at = new()
    columns = at.get("column")
    assert len(columns) == 4
    assert [column.weight for column in columns] == pytest.approx([1 / 4.4] * 3 + [1.4 / 4.4])
    assert all(column.proto.vertical_alignment == BlockProto.Column.BOTTOM for column in columns)
    assert columns[0].selectbox(key=SEPARATE_KEY).label == "Separate by"
    assert columns[1].multiselect(key=vw.COLOR_BY_KEY).label == "Color by"
    assert columns[2].selectbox(key=vw.COLLAPSE_BY_KEY).label == "Collapse by"
    assert [option.content for option in columns[3].button_group(key=MODE_KEY).options] == [
        "Opacity", "Shape"]
    assert columns[3].selectbox(key=vw.PICKER_COL_KEY).label == "Shape by"
    assert {box.label for box in at.selectbox} == {"Method", "Separate by", "Collapse by", "Shape by"}
    assert at.session_state.result == (["day"], None, None, None, None, None)


def test_mode_switches_keep_the_shared_column_and_return_only_the_active_channel():
    at = new({vw.PICKER_COL_KEY: "patient", vw.OPACITY_BY_KEY: "dish"})
    assert at.session_state.result[1:3] == (None, "patient")
    assert len(at.button_group) == 1
    for mode, expected in [("opacity", ("patient", None)), ("shape", (None, "patient"))]:
        at.button_group(key=MODE_KEY).set_value(mode)
        run(at)
        assert at.session_state.result[1:3] == expected
        assert at.session_state.result[4] is None
        assert at.selectbox(key=vw.PICKER_COL_KEY).value == "patient"
        assert at.selectbox(key=vw.PICKER_COL_KEY).label == f"{mode.title()} by"
        assert at.multiselect(key=vw.COLOR_BY_KEY).label == "Color by"
    at.button_group(key=MODE_KEY).set_value(None)
    run(at)
    assert at.button_group(key=MODE_KEY).value == "shape"
    assert at.selectbox(key=vw.PICKER_COL_KEY).value == "patient"


def test_legacy_opacity_migrates_once_and_clearing_the_picker_stays_cleared():
    at = new({vw.OPACITY_BY_KEY: "dish"})
    assert at.session_state.result[1:3] == ("dish", None)
    assert at.button_group(key=MODE_KEY).value == "opacity"
    at.selectbox(key=vw.PICKER_COL_KEY).set_value(None)
    run(at)
    at.button_group(key=MODE_KEY).set_value("shape")
    run(at)
    at.button_group(key=MODE_KEY).set_value("opacity")
    run(at)
    assert at.session_state.result[1:3] == (None, None)
    for method in ["PP", "DR"]:
        at.selectbox(key="method").set_value(method)
        run(at)
        assert at.selectbox(key=vw.OPACITY_BY_KEY).value == "dish"
        assert at.selectbox(key=vw.PICKER_COL_KEY).value is None
    at.selectbox(key="method").set_value("FD")
    run(at)
    assert at.button_group(key=MODE_KEY).value == "opacity"
    assert at.session_state.result[1:3] == (None, None)


@pytest.mark.parametrize("first,second", [("FC", "FD"), ("FD", "FC")])
def test_first_entry_to_another_merged_method_keeps_the_shared_picker_cleared(first, second):
    at = new({"method": first, vw.OPACITY_BY_KEY: "dish"})
    assert at.session_state.result[1:3] == ("dish", None)
    at.selectbox(key=vw.PICKER_COL_KEY).set_value(None)
    run(at)
    for method in [second, first]:
        at.selectbox(key="method").set_value(method)
        run(at)
        assert at.selectbox(key=vw.PICKER_COL_KEY).value is None
        assert at.session_state.result[1:3] == (None, None)
        assert at.session_state.result[4] is None
        assert at.session_state[vw.OPACITY_BY_KEY] == "dish"


@pytest.mark.parametrize("first,second", [("FC", "FD"), ("FD", "FC")])
def test_first_entry_after_histogram_does_not_revive_cleared_shared_opacity(first, second):
    at = new({"method": first, vw.OPACITY_BY_KEY: "dish"})
    assert at.session_state.result[1:3] == ("dish", None)
    at.selectbox(key=vw.PICKER_COL_KEY).set_value(None)
    run(at)
    at.selectbox(key="method").set_value("Histogram")
    run(at)
    assert vw.PICKER_COL_KEY not in at.session_state
    assert at.session_state[vw.OPACITY_BY_KEY] == "dish"
    at.selectbox(key="method").set_value(second)
    run(at)
    assert at.selectbox(key=vw.PICKER_COL_KEY).value is None
    assert at.session_state.result[1:3] == (None, None)
    assert at.session_state.result[4] is None


@pytest.mark.parametrize("method", ["FC", "FD"])
def test_initialization_preserves_an_explicitly_empty_shared_picker(method):
    at = new({"method": method, vw.PICKER_COL_KEY: None, vw.OPACITY_BY_KEY: "dish"})
    assert at.selectbox(key=vw.PICKER_COL_KEY).value is None
    assert at.session_state.result[1:3] == (None, None)
    assert at.session_state[vw.OPACITY_BY_KEY] == "dish"


def test_fc_subcolor_and_fd_modes_preserve_independent_intent():
    at = new({"method": "FC", vw.AS_COLOUR_KEY: True, vw.PICKER_COL_KEY: "patient",
              vw.OPACITY_BY_KEY: "dish"})
    assert at.button_group(key=vw.POINT_MODE_KEY).value == "subcolor"
    at.selectbox(key="method").set_value("FD")
    run(at)
    assert at.multiselect(key=vw.COLOR_BY_KEY).label == "Color by"
    assert len(at.button_group) == 1
    assert at.button_group(key=MODE_KEY).value == "shape"
    assert at.session_state.result[1:3] == (None, "patient")
    assert at.session_state.result[4] is None
    at.button_group(key=MODE_KEY).set_value("opacity")
    run(at)
    at.selectbox(key="method").set_value("FC")
    run(at)
    assert [option.content for option in at.button_group(key=vw.POINT_MODE_KEY).options] == [
        "Opacity", "Subcolor", "Shape"]
    assert at.button_group(key=vw.POINT_MODE_KEY).value == "subcolor"
    assert at.session_state.result[4] == "patient"
    at.selectbox(key="method").set_value("FD")
    run(at)
    assert at.button_group(key=MODE_KEY).value == "opacity"
    assert at.session_state.result[1:3] == ("patient", None)


def test_review_preserves_the_active_decoration_and_shared_column():
    at = new({MODE_KEY: "opacity", vw.PICKER_COL_KEY: "patient",
              vw.OPACITY_BY_KEY: "dish"})
    at.checkbox(key="review").check()
    run(at)
    run(at)
    at.checkbox(key="review").uncheck()
    run(at)
    assert at.button_group(key=MODE_KEY).value == "opacity"
    assert at.selectbox(key=vw.PICKER_COL_KEY).value == "patient"
    assert at.session_state.result[1:3] == ("patient", None)
    assert at.session_state[vw.OPACITY_BY_KEY] == "dish"


@pytest.mark.parametrize("mode", ["shape", "opacity"])
def test_separator_then_color_narrow_collapse_without_narrowing_decorations(mode):
    at = new({SEPARATE_KEY: "day", vw.COLOR_BY_KEY: ["day", "treatment"],
              vw.COLLAPSE_BY_KEY: "dish", MODE_KEY: mode, vw.PICKER_COL_KEY: "dish"})
    assert at.session_state.result[0] == ["treatment"]
    assert at.session_state.result[3] == "day"
    assert at.session_state.result[5] == "dish"
    assert "day" not in at.multiselect(key=vw.COLOR_BY_KEY).options
    assert set(at.selectbox(key=vw.COLLAPSE_BY_KEY).options) == {"dish", "patient"}
    assert {"day", "treatment", "dish"} <= set(at.selectbox(key=vw.PICKER_COL_KEY).options)
    at.multiselect(key=vw.COLOR_BY_KEY).set_value(["treatment", "dish"])
    run(at)
    assert at.session_state.result[0] == ["treatment", "dish"]
    assert at.session_state.result[3] == "day"
    assert at.session_state.result[5] is None
    assert at.selectbox(key=vw.PICKER_COL_KEY).value == "dish"


@pytest.mark.parametrize("collapse", ["day", "treatment"])
def test_stale_collapse_selection_never_resets_separator_or_color(collapse):
    at = new({SEPARATE_KEY: "day", vw.COLOR_BY_KEY: ["treatment"],
              vw.COLLAPSE_BY_KEY: collapse})
    assert at.session_state.result[0] == ["treatment"]
    assert at.session_state.result[3] == "day"
    assert at.session_state.result[5] is None


def test_one_remaining_replicate_keeps_selected_collapse_column():
    at = new({SEPARATE_KEY: "day", vw.COLOR_BY_KEY: ["treatment"],
              vw.COLLAPSE_BY_KEY: "dish"})
    at.session_state.one_replicate = True
    run(at)
    assert at.selectbox(key=vw.COLLAPSE_BY_KEY).value == "dish"
    assert at.session_state.result[3] == "day"
    assert at.session_state.result[5] == "dish"


def test_fd_help_describes_category_color_models_and_complete_pair_aggregation():
    at = new({SEPARATE_KEY: "day"})
    help_text = at.multiselect(key=vw.COLOR_BY_KEY).proto.help
    assert "each category and color group" in help_text
    assert "Collapse by" in help_text
    collapse_help = at.selectbox(key=vw.COLLAPSE_BY_KEY).proto.help
    assert "each category and color group" in collapse_help
    assert "MEAN X and Y" in collapse_help
    assert "both measurements" in collapse_help
    assert "Log transforms apply after averaging" in collapse_help
    assert "Subcolor" not in collapse_help


def test_separator_and_mode_survive_review_single_level_filtering_and_method_switches():
    at = new({SEPARATE_KEY: "day", MODE_KEY: "opacity", vw.PICKER_COL_KEY: "patient",
              "vis_encoding_phasor_separate_by": "treatment",
              "vis_encoding_dr_separate_by": ["patient", "dish"],
              "analysis_control_separate_by": "dish"})
    at.session_state.single = True
    run(at)
    assert at.selectbox(key=SEPARATE_KEY).value == "day"
    at.session_state.single = False
    at.checkbox(key="review").check()
    run(at)
    run(at)
    at.checkbox(key="review").uncheck()
    run(at)
    for method, expected in [("FC", "dish"), ("PP", "treatment"),
                             ("DR", ["patient", "dish"]), ("Histogram", None), ("FD", "day")]:
        at.selectbox(key="method").set_value(method)
        run(at)
        assert at.session_state.result[3] == expected
    assert at.button_group(key=MODE_KEY).value == "opacity"


@pytest.mark.parametrize("state", [{"removed": True}, {"categoricals": ["treatment", "dish"]}])
def test_removed_or_retyped_separator_is_pruned(state):
    at = new({SEPARATE_KEY: "day"})
    for key, value in state.items():
        at.session_state[key] = value
    run(at)
    assert at.session_state.result[3] is None
    assert at.session_state[SEPARATE_KEY] is None


def category_app():
    import streamlit as st
    from src.widgets.analysis_widget_state import (
        analysis_control_keys, preserve_analysis_controls,
    )
    from src.widgets import visualization_widgets as vw

    def open_review():
        if st.session_state.review:
            st.session_state.saved_controls = analysis_control_keys(st.session_state)

    preserve_analysis_controls(st.session_state, vw.SEPARATION_KEYS)
    method = st.selectbox("Method", ["FD", "PP", "Hidden"], key="method")
    st.checkbox("Review", key="review", on_change=open_review)
    preserve_analysis_controls(st.session_state, st.session_state.get("saved_controls", ()))
    if st.session_state.review or method == "Hidden":
        st.stop()
    options = st.session_state.get("categories", ["Day 2", "Day 10", "N/A"])
    separator = st.session_state.get("separator", "day")
    widget = (vw.distribution_category_widget if method == "FD" else vw.phasor_category_widget)
    st.session_state.result = widget(options, separator)
    st.session_state.pop("saved_controls", None)


def category_new(state=None):
    at = AppTest.from_function(category_app)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    return run(at)


def test_category_buttons_cannot_deselect_and_fall_back_to_first_available_category():
    at = category_new()
    assert at.session_state.result == "Day 2"
    at.button_group(key=CATEGORY_KEY).set_value("Day 10")
    run(at)
    at.button_group(key=CATEGORY_KEY).set_value(None)
    run(at)
    assert at.session_state.result == "Day 10"
    at.session_state.categories = ["Day 10", "N/A"]
    run(at)
    assert at.session_state.result == "Day 10"
    at.session_state.categories = ["N/A"]
    run(at)
    assert at.session_state.result == "N/A"
    at.session_state.categories = ["Day 2", "N/A"]
    at.session_state.separator = "batch"
    run(at)
    assert at.session_state.result == "Day 2"


def test_category_dropdown_switches_to_buttons_with_the_same_selected_value():
    at = category_new({"categories": [f"Day {i}" for i in range(1, 8)]})
    at.selectbox(key=CATEGORY_KEY).set_value("Day 2")
    run(at)
    at.session_state.categories = [f"Day {i}" for i in range(1, 7)]
    run(at)
    assert at.button_group(key=CATEGORY_KEY).value == "Day 2"
    at.session_state.categories = [f"Day {i}" for i in range(1, 8)]
    run(at)
    assert at.selectbox(key=CATEGORY_KEY).value == "Day 2"


def test_fd_category_is_independent_of_pp_and_survives_review_and_hidden_controls():
    at = category_new()
    at.button_group(key=CATEGORY_KEY).set_value("Day 10")
    run(at)
    at.checkbox(key="review").check()
    run(at)
    at.checkbox(key="review").uncheck()
    run(at)
    assert at.button_group(key=CATEGORY_KEY).value == "Day 10"
    at.selectbox(key="method").set_value("PP")
    run(at)
    assert at.button_group(key=vw.PHASOR_CATEGORY_KEY).value == "Day 2"
    at.button_group(key=vw.PHASOR_CATEGORY_KEY).set_value("N/A")
    run(at)
    at.selectbox(key="method").set_value("Hidden")
    run(at)
    at.selectbox(key="method").set_value("FD")
    run(at)
    assert at.button_group(key=CATEGORY_KEY).value == "Day 10"
    at.selectbox(key="method").set_value("PP")
    run(at)
    assert at.button_group(key=vw.PHASOR_CATEGORY_KEY).value == "N/A"


def test_empty_category_data_has_no_selector():
    at = category_new({"categories": []})
    assert at.session_state.result is None
    assert not at.button_group
    assert len(at.selectbox) == 1
