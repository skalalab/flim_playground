"""The existing overlay slot exposes SuperPlot only with a replicate unit."""
from streamlit.testing.v1 import AppTest

from src.widgets import visualization_widgets as vw
from src.widgets.analysis_widget_state import analysis_control_keys


def _overlay_app():
    import streamlit as st
    from src.widgets.visualization_widgets import comparison_overlay_widget
    collapse = st.selectbox("Collapse", [None, "dish"], key="collapse")
    st.session_state["chosen"] = comparison_overlay_widget("value", ["treatment"], None, collapse)


def test_overlay_reuses_the_boxplot_slot_and_migrates_legacy_selection():
    at = AppTest.from_function(_overlay_app)
    at.session_state["add_boxplot_value_treatment_"] = True
    at.run()
    assert not at.exception
    assert at.selectbox[1].label == "Overlay"
    assert at.selectbox[1].options == ["None", "Boxplot"]
    assert at.selectbox[1].value == "Boxplot"
    assert not at.checkbox


def test_superplot_requires_collapse_and_clearing_collapse_resets_it():
    at = AppTest.from_function(_overlay_app).run()
    assert not at.exception
    at.selectbox[0].set_value("dish").run()
    assert at.selectbox[1].options == ["None", "Boxplot", "SuperPlot"]
    at.selectbox[1].set_value("SuperPlot").run()
    assert at.session_state["chosen"] == "SuperPlot"
    at.selectbox[0].set_value(None).run()
    assert not at.exception
    assert at.session_state["chosen"] == "None"
    at.selectbox[0].set_value("dish").run()
    assert at.session_state["chosen"] == "None"


def test_overlay_settings_survive_column_review():
    assert "comparison_overlay_value_treatment_" in analysis_control_keys({
        "comparison_overlay_value_treatment_": "SuperPlot"})


def _encoding_app():
    import pandas as pd
    import streamlit as st
    from src.widgets.visualization_widgets import visual_encoding_channels_widget
    rows = pd.DataFrame({"treatment": ["control", "drug"] * 2,
                         "dish": ["D1", "D1", "D2", "D2"]})
    subset = st.selectbox("Filter", ["All", "One dish", "One row", "One treatment"])
    if subset == "One dish":
        rows = rows[rows.dish == "D1"]
    elif subset == "One row":
        rows = rows.iloc[:1]
    elif subset == "One treatment":
        rows = rows[rows.treatment == "control"]
    st.session_state["encodings"] = visual_encoding_channels_widget(
        rows, list(rows.columns), separate_by_available=True,
        subcolor_available=True, collapse_available=True)


def test_filtering_to_one_replicate_never_reverts_to_cell_statistics():
    at = AppTest.from_function(_encoding_app)
    at.session_state[vw.COLLAPSE_BY_KEY] = "dish"
    at.session_state[vw.COLOR_BY_KEY] = ["treatment"]
    at.run()
    assert not at.exception
    for subset in ("One dish", "One row"):
        at.selectbox[0].set_value(subset).run()
        assert not at.exception
        assert at.session_state["encodings"][-1] == "dish"


def test_one_treatment_does_not_promote_dish_to_color_and_remove_collapse():
    at = AppTest.from_function(_encoding_app)
    at.session_state[vw.COLLAPSE_BY_KEY] = "dish"
    at.session_state[vw.COLOR_BY_KEY] = ["treatment"]
    at.run()
    at.selectbox[0].set_value("One treatment").run()
    assert not at.exception
    assert at.session_state["encodings"][0] == ["treatment"]
    assert at.session_state["encodings"][-1] == "dish"
