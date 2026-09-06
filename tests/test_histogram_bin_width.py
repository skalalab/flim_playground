"""histogram_bin_width_widget returns valid bin edges for constant features,
including numpy's single-bin auto result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets.visualization_widgets import histogram_bin_width_widget


def test_constant_feature_returns_valid_bin_edges():
    edges = histogram_bin_width_widget(pd.Series([5.0, 5.0, 5.0]), key="const_feature")
    edges = np.asarray(edges)
    assert edges.ndim == 1
    assert len(edges) >= 2  # at least one bin
    assert edges[0] <= 5.0 <= edges[-1]


def _width_app():
    import streamlit as st
    from src.widgets.visualization_widgets import histogram_bin_width_widget

    st.session_state.edges = histogram_bin_width_widget(
        st.session_state.sample_values, key="hist_bin_width_test")


@pytest.mark.parametrize("values", [
    np.log10(np.linspace(1000., 1001., 50) + 1e-6),
    np.linspace(1e-8, 2e-8, 50),
])
def test_small_positive_widths_display_their_actual_magnitude(values):
    at = AppTest.from_function(_width_app)
    at.session_state.sample_values = values
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    widget = at.number_input[0]
    displayed = widget.proto.format % widget.value
    assert float(displayed) == pytest.approx(widget.value, rel=1e-8, abs=0)
    assert float(displayed) > 0


def _plot_app():
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.vis.univar import feature_histogram_plot

    logged = st.checkbox("Log X", key="logged")
    values = np.linspace(1000., 1001., 50)
    if logged:
        values = np.log10(values + 1e-6)
    feature = st.session_state.get("selected_feature", "Signal")
    feature_histogram_plot(pd.DataFrame({feature: values}), feature, log_x=logged)


def _run(at):
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    return at


def test_raw_and_log_views_keep_independent_custom_bin_widths():
    at = _run(AppTest.from_function(_plot_app))
    raw_key = at.number_input[0].key
    at.number_input[0].set_value(.037)
    _run(at)
    at.checkbox(key="logged").check()
    _run(at)
    log_key = at.number_input[0].key
    assert log_key != raw_key
    assert 0 < at.number_input[0].value < .001
    at.number_input[0].set_value(.00001234567)
    _run(at)

    for logged, width in [(False, .037), (True, .00001234567), (False, .037)]:
        at.checkbox(key="logged").set_value(logged)
        _run(at)
        _run(at)  # Expose Streamlit cleanup of the inactive scale's widget.
        assert at.number_input[0].value == pytest.approx(width)
        assert at.session_state[raw_key] == pytest.approx(.037)
        assert at.session_state[log_key] == pytest.approx(.00001234567)


@pytest.mark.parametrize("logged,width", [(False, .037), (True, .00001234567)])
def test_existing_shared_width_migrates_to_its_active_scale(logged, width):
    at = AppTest.from_function(_plot_app)
    at.session_state.logged = logged
    at.session_state["hist_bin_width_Signal"] = width
    _run(at)
    assert at.number_input[0].key != "hist_bin_width_Signal"
    assert at.number_input[0].value == pytest.approx(width)


def test_scale_named_features_do_not_inherit_another_features_width():
    at = _run(AppTest.from_function(_plot_app))
    default_width = at.number_input[0].value
    at.number_input[0].set_value(.037)
    _run(at)
    at.session_state.selected_feature = "raw_Signal"
    _run(at)
    assert at.number_input[0].value == pytest.approx(default_width)


@pytest.mark.parametrize("can_log", [True, False])
def test_page_export_captures_the_active_scale_bin_width(monkeypatch, can_log):
    from src import dataset_io, export_script
    from src.widgets import analysis_config_widgets as acw

    values = np.linspace(1000., 1001., 50) if can_log else np.linspace(-1., 1., 50)
    frame = pd.DataFrame({"cell_id": [f"id{i}" for i in range(50)], "Signal": values})
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: [])
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        frame, {"Uncategorized Features": ["Signal"]}, True, ",", "cell_id"))
    captured = []
    monkeypatch.setattr(export_script, "generate_script", lambda state: (
        captured.append(state) or "# captured"))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py"))
    at.session_state["_menu_Uncategorized Features"] = "Signal"
    _run(at)
    at.radio[1].set_value("Feature Histogram")
    at.session_state["analysis_control_hist_bin_width_raw_Signal"] = .037
    at.session_state["analysis_control_hist_bin_width_log10_Signal"] = .00001234567
    at.session_state["hist_bin_width_Signal"] = .08

    for logged in (True, False):
        expected = .00001234567 if logged and can_log else .037
        at.session_state["log_x_hist_Signal"] = logged
        _run(at)
        widget = next(n for n in at.number_input if n.label == "Bin Width")
        assert widget.value == pytest.approx(expected)
        params = captured[-1]["method_params"]
        assert params["log_x"] is (logged and can_log)
        assert params["bin_width"] == pytest.approx(widget.value)


@pytest.mark.parametrize("destination", ["gmm", "Feature Comparison"])
@pytest.mark.parametrize("logged", [False, True])
def test_page_remembers_bin_widths_while_their_controls_are_hidden(monkeypatch, destination, logged):
    from src import dataset_io, export_script
    from src.widgets import analysis_config_widgets as acw

    frame = pd.DataFrame({"cell_id": [f"id{i}" for i in range(50)],
                          "Signal": np.linspace(1000., 1001., 50)})
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: [])
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        frame, {"Uncategorized Features": ["Signal"]}, True, ",", "cell_id"))
    monkeypatch.setattr(export_script, "generate_script", lambda state: "# captured")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py"))
    at.session_state["_menu_Uncategorized Features"] = "Signal"
    _run(at)
    at.radio[1].set_value("Feature Histogram")
    at.session_state["log_x_hist_Signal"] = logged
    _run(at)
    widget = next(n for n in at.number_input if n.label == "Bin Width")
    key = widget.key
    expected = .00001234567 if logged else .037
    widget.set_value(expected)
    _run(at)

    if destination == "gmm":
        at.checkbox(key="analysis_control_apply_gmm").check()
    else:
        at.radio[1].set_value(destination)
    _run(at)
    _run(at)  # Give Streamlit time to clean up the hidden Bin Width widget.
    assert key in at.session_state
    assert at.session_state[key] == pytest.approx(expected)

    if destination == "gmm":
        at.checkbox(key="analysis_control_apply_gmm").uncheck()
    else:
        at.radio[1].set_value("Feature Histogram")
        at.session_state["log_x_hist_Signal"] = logged
    _run(at)
    widget = next(n for n in at.number_input if n.label == "Bin Width")
    assert widget.value == pytest.approx(expected)
