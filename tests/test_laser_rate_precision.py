"""Laser-rate fields use MHz; saved settings and calculation inputs use GHz."""

import inspect

import pytest
import toml
from streamlit.testing.v1 import AppTest

from test_config_flim_settings import _PAGE, _open, _profile, _save_button


INPUT_TYPES = ["Decay (3/4D)", "Decay (3/4D) pixel-prefitted", "Decay (2D)"]
RATES_MHZ = [39.01, 80.0]


def widget_app(tmp_path, function, *args, **state):
    path = tmp_path / "laser_rate_app.py"
    path.write_text(inspect.getsource(function) + f"\n{function.__name__}(*{args!r})\n")
    app = AppTest.from_file(str(path))
    for key, value in state.items():
        app.session_state[key] = value
    app.run(timeout=30)
    assert not app.exception
    return app


def assert_rate(widget, expected):
    assert "MHz" in widget.label
    assert widget.value == pytest.approx(expected, rel=0, abs=5e-10)
    # AppTest bypasses browser formatting when setting a number. Check the text
    # format sent to the browser too: 39.01 MHz must appear as exactly "39.01".
    displayed = widget.proto.format % widget.value
    assert displayed == f"{expected:.2f}"


@pytest.mark.parametrize("input_type", INPUT_TYPES)
def test_configuration_displays_and_saves_precise_laser_rate(tmp_path, monkeypatch, input_type):
    app, path = _open(tmp_path, monkeypatch, {
        "default": _profile(input_type=input_type, extractors=["Lifetime fit free"]),
    })
    key = f"laser_rate_{input_type}_default_mhz"
    assert_rate(app.number_input(key=key), 80.0)
    for rate in RATES_MHZ:
        app.number_input(key=key).set_value(rate).run()
        assert_rate(app.number_input(key=key), rate)
        _save_button(app).click().run()
        assert not app.exception
        assert toml.load(path)["profiles"]["default"][input_type]["laser_rate"] == rate / 1000
        app = AppTest.from_file(_PAGE).run(timeout=30)
        assert_rate(app.number_input(key=key), rate)


def extraction_app(input_type):
    import streamlit as st
    from src.widgets.metadata_widgets import lifetime_data_config_widget

    _, _, rate = lifetime_data_config_widget({"ch1": ["Lifetime fit free"]}, input_type)
    st.session_state["selected_rate"] = rate


@pytest.mark.parametrize("input_type", INPUT_TYPES)
def test_extraction_displays_and_returns_precise_laser_rate(tmp_path, input_type):
    legacy_key = "2D_decay_laser_rate" if input_type == "Decay (2D)" else "laser_rate"
    app = widget_app(tmp_path, extraction_app, input_type, **{legacy_key: 0.03901})
    key = f"{legacy_key}_mhz"
    assert_rate(app.number_input(key=key), 39.01)
    for rate in RATES_MHZ:
        app.number_input(key=key).set_value(rate).run()
        assert not app.exception
        assert_rate(app.number_input(key=key), rate)
        assert app.session_state["selected_rate"] == rate / 1000


def phasor_app():
    import streamlit as st
    from src.widgets.analysis_widget_state import analysis_control_keys, preserve_analysis_controls
    from src.widgets.visualization_widgets import phasor_params_widget

    preserve_analysis_controls(st.session_state, analysis_control_keys(st.session_state))
    if st.checkbox("Review", key="review"):
        st.stop()
    _, _, rate = phasor_params_widget({
        "Lifetime fit free_NADH": [
            "Lifetime fit free_NADH: G(1st)", "Lifetime fit free_NADH: S(1st)",
        ],
    })
    st.session_state["selected_rate"] = rate


def test_phasor_controls_display_and_return_precise_laser_rate(tmp_path):
    app = widget_app(tmp_path, phasor_app, analysis_control_phasor_frequency=0.03901)
    key = "analysis_control_phasor_frequency_mhz"
    assert_rate(app.number_input(key=key), 39.01)
    for rate in RATES_MHZ:
        app.number_input(key=key).set_value(rate).run()
        assert not app.exception
        assert_rate(app.number_input(key=key), rate)
        assert app.session_state["selected_rate"] == rate / 1000
        app.checkbox(key="review").check().run()
        app.checkbox(key="review").uncheck().run()
        assert not app.exception
        assert not app.warning
        assert_rate(app.number_input(key=key), rate)
        assert app.session_state["selected_rate"] == rate / 1000


@pytest.mark.parametrize("input_type", INPUT_TYPES)
def test_configuration_converts_legacy_session_rate_once(tmp_path, monkeypatch, input_type):
    _open(tmp_path, monkeypatch, {
        "default": _profile(input_type=input_type, extractors=["Lifetime fit free"]),
    })
    app = AppTest.from_file(_PAGE)
    app.session_state[f"laser_rate_{input_type}_default"] = 0.03901
    app.run(timeout=30)
    assert_rate(app.number_input(key=f"laser_rate_{input_type}_default_mhz"), 39.01)
    app.run()
    assert_rate(app.number_input(key=f"laser_rate_{input_type}_default_mhz"), 39.01)
    app.number_input(key=f"laser_rate_{input_type}_default_mhz").set_value(60.0).run()
    feature_key = f"{input_type}_ch1_feature_extractors_default"
    app.multiselect(key=feature_key).set_value(["Lifetime fit"]).run()
    app.run()  # Let the hidden MHz widget be cleaned up.
    app.multiselect(key=feature_key).set_value(["Lifetime fit free"]).run()
    assert_rate(app.number_input(key=f"laser_rate_{input_type}_default_mhz"), 60.0)
    assert not app.warning
