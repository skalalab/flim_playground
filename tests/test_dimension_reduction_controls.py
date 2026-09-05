"""Dimension Reduction facets keep layout separate from point encodings."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from src import dataset_io, export_script
from src.vis import multivar
from src.widgets import analysis_config_widgets as acw
from src.widgets import visualization_widgets as vw


SEPARATE_KEY = "vis_encoding_dr_separate_by"
OBSOLETE_BACKGROUND_KEY = "vis_encoding_dr_show_facet_background"
OBSOLETE_COLUMNS_KEY = "vis_encoding_dr_facet_columns"
PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")


def _widget_app():
    import pandas as pd
    import streamlit as st
    from src.widgets.analysis_widget_state import (
        analysis_control_keys, preserve_analysis_controls,
    )
    from src.widgets.visualization_widgets import visual_encoding_channels_widget

    def open_review():
        if st.session_state.test_review:
            st.session_state.test_saved_controls = analysis_control_keys(st.session_state)

    method = st.selectbox("Method", ["Dimension Reduction", "Feature Comparison", "Histogram"],
                          key="test_method")
    st.checkbox("Review", key="test_review", on_change=open_review)
    if st.session_state.get("test_saved_controls"):
        preserve_analysis_controls(st.session_state, st.session_state.test_saved_controls)
    if st.session_state.test_review:
        st.stop()

    data = pd.DataFrame({"treatment": ["A", "B"] * 4,
                         "patient": ["p1", "p2", "p3", "p4"] * 2,
                         "day": ["d1", "d1", "d2", "d2"] * 2})
    if st.session_state.get("test_one_level"):
        data = data.iloc[[0]]
    if st.session_state.get("test_removed_column"):
        data = data.drop(columns=[st.session_state.test_removed_column])
    categorical_cols = st.session_state.get("test_categories", list(data.columns))
    result = visual_encoding_channels_widget(
        data, categorical_cols, point_based=method != "Histogram",
        subcolor_available=method == "Feature Comparison",
        separate_by_available=method != "Histogram",
        separate_by_mode="facets" if method == "Dimension Reduction" else "sections",
    )
    st.session_state.test_result = result
    st.session_state.pop("test_saved_controls", None)


def _run(at):
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    return at


def _new_widget(state=None):
    at = AppTest.from_function(_widget_app)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    return _run(at)


def test_facets_start_empty_and_limit_ordered_selection_to_two_columns():
    at = _new_widget()
    picker = at.multiselect(key=SEPARATE_KEY)
    assert picker.label == "Separate by"
    assert picker.proto.max_selections == 2
    assert at.session_state.test_result[3] == []
    picker.set_value(["patient", "treatment"])
    _run(at)
    assert at.session_state.test_result[3] == ["patient", "treatment"]
    at.multiselect(key=SEPARATE_KEY).set_value(["treatment", "patient"])
    _run(at)
    assert at.session_state.test_result[3] == ["treatment", "patient"]


def test_separation_can_share_columns_with_every_point_encoding():
    at = _new_widget({SEPARATE_KEY: ["patient", "treatment"],
                      vw.COLOR_BY_KEY: ["patient"], vw.PICKER_COL_KEY: "patient",
                      vw.OPACITY_BY_KEY: "treatment"})
    assert at.session_state.test_result == (
        ["patient"], "treatment", "patient", ["patient", "treatment"], None, None)
    for picker in [at.multiselect(key=vw.COLOR_BY_KEY),
                   at.selectbox(key=vw.PICKER_COL_KEY), at.selectbox(key=vw.OPACITY_BY_KEY)]:
        assert {"patient", "treatment"} <= set(picker.options)


def test_facet_selection_survives_filtering_every_category_to_one_level():
    at = _new_widget({SEPARATE_KEY: ["patient", "treatment"]})
    at.session_state.test_one_level = True
    _run(at)
    assert at.session_state.test_result[3] == ["patient", "treatment"]
    assert at.multiselect(key=SEPARATE_KEY).value == ["patient", "treatment"]


@pytest.mark.parametrize("state", [
    {"test_removed_column": "patient"},
    {"test_categories": ["treatment", "day"]},
])
def test_removed_or_no_longer_categorical_columns_are_pruned(state):
    at = _new_widget({SEPARATE_KEY: ["patient", "treatment"]})
    for key, value in state.items():
        at.session_state[key] = value
    _run(at)
    assert at.session_state.test_result[3] == ["treatment"]
    assert "patient" not in at.multiselect(key=SEPARATE_KEY).options


def test_removed_last_category_clears_separation():
    at = _new_widget({SEPARATE_KEY: ["patient"]})
    at.session_state.test_categories = []
    _run(at)
    assert at.session_state.test_result[3] == []
    assert at.session_state[SEPARATE_KEY] == []


@pytest.mark.parametrize("stored,expected", [
    ("patient", ["patient"]), ("missing", []), (None, []), (7, []),
])
def test_scalar_facet_state_is_normalized_before_the_multiselect(stored, expected):
    at = _new_widget({SEPARATE_KEY: stored})
    assert at.session_state.test_result[3] == expected
    assert at.multiselect(key=SEPARATE_KEY).value == expected


@pytest.mark.parametrize("separate_by", [[], ["patient"], ["patient", "treatment"]])
def test_obsolete_settings_do_not_restore_layout_or_background_controls(separate_by):
    at = _new_widget({SEPARATE_KEY: separate_by, OBSOLETE_COLUMNS_KEY: 4,
                      OBSOLETE_BACKGROUND_KEY: False})
    assert at.session_state.test_result[3] == separate_by
    assert not at.number_input
    assert "Show other groups in gray" not in [widget.label for widget in at.checkbox]
    assert not {OBSOLETE_COLUMNS_KEY, OBSOLETE_BACKGROUND_KEY}.intersection(vw.DR_FACET_KEYS)
    assert ("One selected feature creates one column of maps. With two, the first "
            "sets rows and the second sets columns.") in at.multiselect(key=SEPARATE_KEY).proto.help


def test_facet_selection_survives_method_switches_without_changing_fc_sections():
    at = _new_widget({SEPARATE_KEY: ["patient"],
                      "analysis_control_separate_by": "treatment"})
    at.selectbox(key="test_method").set_value("Feature Comparison")
    _run(at)
    assert at.session_state.test_result[3] == "treatment"
    assert "treatment" not in at.multiselect(key=vw.COLOR_BY_KEY).options
    assert "treatment" not in at.selectbox(key=vw.PICKER_COL_KEY).options
    at.selectbox(key="test_method").set_value("Histogram")
    _run(at)
    at.selectbox(key="test_method").set_value("Dimension Reduction")
    _run(at)
    assert at.session_state.test_result[3] == ["patient"]


def test_facet_selection_survives_review_and_remount_without_warnings():
    at = _new_widget({SEPARATE_KEY: ["patient"]})
    at.checkbox(key="test_review").check()
    _run(at)
    _run(at)
    at.checkbox(key="test_review").uncheck()
    _run(at)
    assert at.session_state.test_result[3] == ["patient"]
    assert at.multiselect(key=SEPARATE_KEY).proto.set_value
    assert not at.number_input


@pytest.fixture
def page(monkeypatch):
    frame = pd.DataFrame({"cell_id": [f"c{i}" for i in range(8)],
                          "treatment": ["A", "B"] * 4,
                          "patient": ["p1", "p2", "p3", "p4"] * 2,
                          "feature_x": [1., 2., 4., 3., 5., 7., 6., 8.],
                          "feature_y": [8., 4., 5., 7., 2., 6., 1., 3.]})
    monkeypatch.setattr(acw, "get_categorical_cols_analysis",
                        lambda *a, **k: ["treatment", "patient"])
    monkeypatch.setattr(acw, "get_fov_name_col_analysis", lambda *a, **k: None)
    monkeypatch.setattr(acw, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        frame.copy(), {"Uncategorized Features": ["feature_x", "feature_y"]},
        True, ",", "cell_id"))
    seen = {}

    def capture_plot(df, **kwargs):
        seen["plot"] = kwargs
        return go.Figure(go.Scatter(x=[1., 2.], y=[2., 1.]))

    def capture_export(state):
        seen["state"] = state
        return "# captured analysis"

    monkeypatch.setattr(multivar, "dimension_reduction_plot", capture_plot)
    monkeypatch.setattr(export_script, "generate_script", capture_export)
    return seen


@pytest.mark.parametrize("separate_by", [["patient"], ["patient", "treatment"]])
def test_page_passes_ordered_facets_and_ignores_obsolete_settings(page, separate_by):
    at = AppTest.from_file(PAGE).run(timeout=90)
    at.radio[0].set_value("### **Multivariate**")
    for key, value in {SEPARATE_KEY: separate_by,
                       OBSOLETE_BACKGROUND_KEY: False, OBSOLETE_COLUMNS_KEY: 4,
                       "analysis_control_dr_method": "PCA",
                       "ms_Uncategorized Features": ["feature_x", "feature_y"]}.items():
        at.session_state[key] = value
    at.run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert "plot" in page, [e.value for e in at.error]
    assert page["plot"]["separate_by"] == separate_by
    assert "show_facet_background" not in page["plot"]
    assert "facet_columns" not in page["plot"]
    assert page["state"]["separate_by"] == separate_by
    assert "show_facet_background" not in page["state"]["method_params"]
    assert "facet_columns" not in page["state"]["method_params"]
    assert "Facet columns" not in [widget.label for widget in at.number_input]
    assert "Show other groups in gray" not in [widget.label for widget in at.checkbox]


def test_facets_offer_group_counts_without_a_color_group(page):
    at = AppTest.from_file(PAGE).run(timeout=90)
    at.radio[0].set_value("### **Multivariate**")
    for key, value in {SEPARATE_KEY: ["patient"], vw.COLOR_BY_KEY: [],
                       "analysis_control_dr_method": "PCA",
                       "ms_Uncategorized Features": ["feature_x", "feature_y"]}.items():
        at.session_state[key] = value
    at.run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert "plot_show_group_counts" in [widget.key for widget in at.checkbox]
    at.checkbox(key="plot_show_group_counts").check().run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert page["state"]["color_by"] == []
    assert page["state"]["separate_by"] == ["patient"]
    assert page["state"]["show_group_counts"] is True
