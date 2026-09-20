"""FD owns its category view and two-mode decoration independently of FC and PP."""

import pytest
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.testing.v1 import AppTest

from src.widgets import visualization_widgets as vw

SEPARATE_KEY = "vis_encoding_fd_separate_by"
MODE_KEY = "vis_encoding_fd_point_mode"


def app():
    import pandas as pd
    import streamlit as st

    from src.widgets.analysis_widget_state import (
        analysis_control_keys,
        preserve_analysis_controls,
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


def marginal_app():
    import streamlit as st

    from src.vis.bivar import distribution_controls

    st.session_state.options = distribution_controls("x", "y")


def test_marginal_plot_type_offers_none_first_and_defaults_to_it():
    at = run(AppTest.from_function(marginal_app))
    widget = at.selectbox(key="marginal_plot_type_selector_x_y")
    assert widget.label == "Marginal Plot Type"
    assert list(widget.options) == ["None", "gaussian fit", "boxplot", "violin"]
    assert widget.value == "None"
    assert at.session_state.options["marginal_plot_type"] == "None"


def test_a_stored_marginal_choice_survives_and_an_unknown_one_is_pruned():
    at = AppTest.from_function(marginal_app)
    at.session_state["marginal_plot_type_selector_x_y"] = "violin"
    run(at)
    assert at.session_state.options["marginal_plot_type"] == "violin"
    # A second AppTest, rather than a second run() on the same one: Streamlit's
    # own widget-state bookkeeping raises ValueError when a *rendered* selectbox's
    # session value is overwritten with something outside its current options,
    # before this app's own pruning ever gets a chance to run. Priming session
    # state before the first run exercises the same pruning code path (the
    # selectbox has not rendered yet, so nothing pre-validates the stored value).
    at = AppTest.from_function(marginal_app)
    at.session_state["marginal_plot_type_selector_x_y"] = "heatmap"
    run(at)
    assert at.session_state.options["marginal_plot_type"] == "None"


def test_two_facet_wrappers_target_different_containers_and_globals():
    import inspect
    import re

    from src.widgets import plot_layout

    source = inspect.getsource(plot_layout.dimension_reduction_chart)
    script = re.search(r"<script>(.*?)</script>", source, re.DOTALL).group(1)
    assert "__CONTAINER_KEY__" in script
    assert "__META_KEY__" in script
    assert "__CLEANUP_GLOBAL__" in script

    def wrapper_app():
        import plotly.graph_objects as go
        import streamlit as st

        from src.widgets.plot_layout import dimension_reduction_chart

        spec = go.Figure(layout={"meta": {"distribution_facet_layout": {"plot_height": .5}}})
        dimension_reduction_chart(spec, key="fd", container_key="distribution_facet_plot",
                                  meta_key="distribution_facet_layout")
        st.session_state.rendered = True

    at = run(AppTest.from_function(wrapper_app))
    html = at.get("html")[0].proto.body
    assert ".st-key-distribution_facet_plot" in html
    assert "distribution_facet_layout" in html
    assert "_flim_distribution_facet_plot_cleanup" in html
    assert "dimension_reduction_plot" not in html
