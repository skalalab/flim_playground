"""Analysis controls remount with their saved values and no Streamlit warnings."""

import pytest
from streamlit.testing.v1 import AppTest


_APP = '''
import numpy as np
import streamlit as st
from streamlit.elements.lib import policies
from src.widgets.analysis_widget_state import analysis_control_keys, preserve_analysis_controls
from src.widgets.classification_widgets import classifier_hyperparams_widget
from src.widgets.visualization_widgets import (
    comparison_pair_widget, gmm_hyperParams_widget, histogram_bin_width_widget,
    tsne_hyperParams_widget, umap_hyperParams_widget,
)

# Streamlit normally displays this warning only once per process. Each test must
# detect it independently, including when a control is remounted after review.
policies._shown_default_value_warning = False

def open_review():
    if st.session_state.review_visible:
        st.session_state.saved_controls = analysis_control_keys(st.session_state)

st.checkbox("Review", key="review_visible", on_change=open_review)
if st.session_state.get("saved_controls"):
    preserve_analysis_controls(st.session_state, st.session_state.saved_controls)
if st.session_state.review_visible:
    st.stop()

{controls}
st.session_state.pop("saved_controls", None)
'''


@pytest.mark.parametrize("controls", [
    "umap_hyperParams_widget()",
    "tsne_hyperParams_widget()",
    "gmm_hyperParams_widget()",
    "comparison_pair_widget([('A', 'B'), ('A', 'C')])",
    "histogram_bin_width_widget(np.linspace(1., 101., 50), key='hist_bin_width_Area')",
    *[f"classifier_hyperparams_widget({name!r})" for name in (
        "Random Forest", "Gradient Boosting", "SVM", "Logistic Regression")],
])
def test_controls_resume_without_default_warnings(controls):
    from src.widgets.analysis_widget_state import analysis_control_keys

    at = AppTest.from_string(_APP.format(controls=controls)).run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    if at.number_input:
        widget = at.number_input[0]
        widget.set_value(widget.value + widget.proto.step).run(timeout=30)
    keys = analysis_control_keys(at.session_state.filtered_state)
    expected = {key: at.session_state[key] for key in keys}

    at.checkbox(key="review_visible").check().run(timeout=30)
    at.run(timeout=30)
    assert not at.number_input
    assert not at.slider
    at.checkbox(key="review_visible").uncheck().run(timeout=30)

    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    assert {key: at.session_state[key] for key in keys} == expected
    # AppTest can read retained server state even when the browser receives only
    # the constructor default. The remount must explicitly send the saved value.
    for widget in [*at.number_input, *at.slider, *at.multiselect]:
        assert widget.proto.set_value, widget.key


def test_histogram_width_is_valid_after_the_data_range_narrows():
    controls = """
data_max = st.session_state.get('data_max', 101.)
st.session_state.edges = histogram_bin_width_widget(
    np.linspace(1., data_max, 50), key='hist_bin_width_Area')
"""
    at = AppTest.from_string(_APP.format(controls=controls)).run(timeout=30)
    at.number_input[0].set_value(20.).run(timeout=30)
    at.session_state.data_max = 7.
    at.run(timeout=30)

    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    assert 0 < at.number_input[0].value <= 2.
    assert len(at.session_state.edges) > 1
