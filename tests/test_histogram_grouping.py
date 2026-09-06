"""Histogram grouping is independent of point encodings and other methods."""

import pytest
from streamlit.testing.v1 import AppTest

from src.widgets import visualization_widgets as vw


SEPARATE_KEY = "vis_encoding_histogram_separate_by"
# Simulate obsolete settings from sessions that offered Histogram aggregation.
COLLAPSE_KEY = "vis_encoding_histogram_collapse_by"


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

    method = st.selectbox(
        "Method", ["Histogram", "FD", "FC", "PP", "DR", "Grouping only"], key="method")
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
    if st.session_state.get("missing_categories") == "all":
        data["patient"] = [None, float("nan"), pd.NA, None] * 2
    elif st.session_state.get("missing_categories"):
        data["patient"] = ["p1", None] * 4
    if st.session_state.get("single_category"):
        data = data[data["day"] == "Day 2"]
    if st.session_state.get("one_replicate"):
        data = data[data["dish"] == "d1"]
    if st.session_state.get("single_row"):
        data = data.iloc[:1]
    data = data.drop(columns=st.session_state.get("removed", []))
    categories = st.session_state.get("categoricals", list(data.columns))
    st.session_state.result = visual_encoding_channels_widget(
        data, categories,
        point_based=st.session_state.get("point_based", method not in {"Histogram", "Grouping only"}),
        separate_by_available=True, subcolor_available=method == "FC",
        collapse_available=method in {"Histogram", "FD", "FC", "Grouping only"},
        separate_by_mode={"Histogram": "histogram", "FD": "distribution",
                          "PP": "subplots", "DR": "facets"}.get(method, "sections"),
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


@pytest.mark.parametrize("point_based", [False, True])
def test_histogram_has_two_grouping_controls_and_no_aggregation_or_point_decorations(point_based):
    at = new({"point_based": point_based, vw.PICKER_COL_KEY: "dish",
              vw.OPACITY_BY_KEY: "patient", vw.POINT_MODE_KEY: "subcolor"})
    columns = at.get("column")
    assert len(columns) == 2
    assert [column.weight for column in columns] == pytest.approx([0.5, 0.5])
    assert columns[0].selectbox(key=SEPARATE_KEY).label == "Separate by"
    assert columns[1].multiselect(key=vw.COLOR_BY_KEY).label == "Color by"
    assert {box.label for box in at.selectbox} == {"Method", "Separate by"}
    assert not at.button_group
    assert at.session_state.result == (["day"], None, None, None, None, None)


def test_grouping_flags_are_independent_of_point_based():
    at = new({"method": "Grouping only"})
    assert {box.label for box in at.selectbox} == {"Method", "Separate by", "Collapse by"}
    assert not at.button_group


def test_histogram_defaults_do_not_inherit_comparison_or_distribution_grouping():
    at = new({vw.COLLAPSE_BY_KEY: "dish", vw.FD_SEPARATE_BY_KEY: "day",
              "analysis_control_separate_by": "patient"})
    assert at.selectbox(key=SEPARATE_KEY).value is None
    assert at.session_state.result[3:] == (None, None, None)


def test_separation_excludes_its_column_from_color_without_aggregating_units():
    at = new({SEPARATE_KEY: "day", vw.COLOR_BY_KEY: ["day", "treatment"]})
    assert at.session_state.result[0] == ["treatment"]
    assert "day" not in at.multiselect(key=vw.COLOR_BY_KEY).options
    assert at.session_state.result == (["treatment"], None, None, "day", None, None)
    assert "dish" in at.selectbox(key=SEPARATE_KEY).options
    assert "dish" in at.multiselect(key=vw.COLOR_BY_KEY).options

    at.multiselect(key=vw.COLOR_BY_KEY).set_value(["treatment", "dish"])
    run(at)
    assert at.session_state.result == (["treatment", "dish"], None, None, "day", None, None)


@pytest.mark.parametrize("collapse", ["day", "treatment", "dish"])
def test_stale_collapse_is_ignored_without_resetting_separation_or_color(collapse):
    at = new({SEPARATE_KEY: "day", vw.COLOR_BY_KEY: ["treatment"], COLLAPSE_KEY: collapse})
    assert at.session_state.result == (["treatment"], None, None, "day", None, None)
    assert "Collapse by" not in {box.label for box in at.selectbox}


def test_histogram_grouping_survives_other_methods_without_sharing_their_keys():
    at = new({SEPARATE_KEY: "day", COLLAPSE_KEY: "dish",
              vw.COLOR_BY_KEY: ["treatment"], vw.FD_SEPARATE_BY_KEY: "patient",
              vw.COLLAPSE_BY_KEY: "day", "analysis_control_separate_by": "patient"})
    for method in ["FD", "FC", "PP", "DR"]:
        at.selectbox(key="method").set_value(method)
        run(at)
        run(at)  # Expose cleanup of controls hidden by a method change.
        assert at.session_state[SEPARATE_KEY] == "day"
        if method == "FD":
            assert at.session_state.result[3] == "patient"
            assert at.session_state.result[5] == "day"
    at.selectbox(key="method").set_value("Histogram")
    run(at)
    assert at.session_state.result == (["treatment"], None, None, "day", None, None)


def test_histogram_grouping_survives_review():
    at = new({SEPARATE_KEY: "day", COLLAPSE_KEY: "dish", vw.COLOR_BY_KEY: ["treatment"]})
    at.checkbox(key="review").check()
    run(at)
    run(at)
    assert SEPARATE_KEY in at.session_state.saved_controls
    at.checkbox(key="review").uncheck()
    run(at)
    assert at.session_state.result == (["treatment"], None, None, "day", None, None)


@pytest.mark.parametrize("filter_key", ["single_category", "one_replicate", "single_row"])
def test_filtering_to_one_level_keeps_histogram_separation_without_aggregation(filter_key):
    at = new({SEPARATE_KEY: "day", COLLAPSE_KEY: "dish", vw.COLOR_BY_KEY: ["treatment"]})
    at.session_state[filter_key] = True
    run(at)
    assert at.selectbox(key=SEPARATE_KEY).value == "day"
    assert at.session_state.result[3:] == ("day", None, None)


@pytest.mark.parametrize("state", [
    {"removed": ["day", "dish"]}, {"categoricals": ["treatment", "patient"]},
])
def test_removed_or_retyped_grouping_columns_are_cleared(state):
    at = new({SEPARATE_KEY: "day", COLLAPSE_KEY: "dish", vw.COLOR_BY_KEY: ["treatment"]})
    for key, value in state.items():
        at.session_state[key] = value
    run(at)
    assert at.session_state.result[3:] == (None, None, None)
    assert at.session_state[SEPARATE_KEY] is None


def test_missing_values_count_as_a_category_for_color():
    at = new({"missing_categories": True, vw.COLOR_BY_KEY: ["treatment"],
              COLLAPSE_KEY: "patient"})
    assert "patient" in at.multiselect(key=vw.COLOR_BY_KEY).options
    assert at.session_state.result[5] is None
    at.multiselect(key=vw.COLOR_BY_KEY).set_value(["patient"])
    run(at)
    assert at.session_state.result[0] == ["patient"]
    assert at.session_state.result[5] is None


def test_all_missing_values_count_as_one_category():
    at = new({"missing_categories": "all", vw.COLOR_BY_KEY: ["treatment"]})
    assert "patient" not in at.multiselect(key=vw.COLOR_BY_KEY).options


def test_histogram_help_describes_individual_unit_distributions():
    at = new()
    help_text = at.selectbox(key=SEPARATE_KEY).proto.help
    assert "each category" in help_text
    assert "individual" in help_text
    assert "Collapse" not in help_text
    assert "Collapse" not in at.multiselect(key=vw.COLOR_BY_KEY).proto.help


def test_single_level_color_after_filter_preserves_the_selected_grouping():
    def correlated_app():
        import pandas as pd
        import streamlit as st
        from src.widgets.visualization_widgets import visual_encoding_channels_widget
        df = pd.DataFrame({"treatment": ["A", "A", "B", "B"],
                           "dish": ["d1", "d2", "d1", "d2"],
                           "day": ["Day 2", "Day 2", "Day 10", "Day 10"]})
        if st.checkbox("One day", key="one_day"):
            df = df[df.day == "Day 2"]
        st.session_state.result = visual_encoding_channels_widget(
            df, ["treatment", "dish", "day"], point_based=False,
            separate_by_available=True, collapse_available=True, separate_by_mode="histogram")
    at = AppTest.from_function(correlated_app)
    for key, value in {SEPARATE_KEY: "day", COLLAPSE_KEY: "dish", vw.COLOR_BY_KEY: ["treatment"]}.items():
        at.session_state[key] = value
    run(at)
    at.checkbox(key="one_day").check()
    run(at)
    assert at.session_state.result == (["treatment"], None, None, "day", None, None)
