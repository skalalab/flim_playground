"""Phasor owns one scalar separator independently of FC and DR."""
from streamlit.testing.v1 import AppTest

KEY = "vis_encoding_phasor_separate_by"
CATEGORY_KEY = "vis_encoding_phasor_category"


def app():
    import pandas as pd
    import streamlit as st
    from src.widgets.visualization_widgets import visual_encoding_channels_widget
    from src.widgets.analysis_widget_state import analysis_control_keys, preserve_analysis_controls
    method = st.selectbox("Method", ["Phasor", "FC", "DR", "Histogram"], key="method")
    preserve_analysis_controls(st.session_state, analysis_control_keys(st.session_state))
    if st.checkbox("Review", key="review"):
        st.stop()
    data = pd.DataFrame({"day": ["D1", "D2"] * 3,
                         "treatment": ["A", "B", "A"] * 2, "dish": ["x", "y"] * 3})
    if st.session_state.get("single"):
        data = data.iloc[:1]
    if st.session_state.get("removed"):
        data = data.drop(columns="day")
    categories = list(data.columns) if not st.session_state.get("noncategorical") else ["treatment", "dish"]
    result = visual_encoding_channels_widget(data, categories,
        point_based=method != "Histogram", separate_by_available=method != "Histogram",
        subcolor_available=method == "FC",
        separate_by_mode={"Phasor": "subplots", "DR": "facets"}.get(method, "sections"))
    st.session_state.result = result


def run(at):
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [e.value for e in at.warning]
    return at


def new(**state):
    at = AppTest.from_function(app)
    for key, value in state.items():
        at.session_state[key] = value
    return run(at)


def test_scalar_picker_excludes_separator_from_color_only():
    at = new(**{KEY: "day", "vis_encoding_color_by": ["day", "treatment"],
                "vis_encoding_picker_col": "day", "vis_encoding_opacity_by": "day"})
    assert at.selectbox(key=KEY).value == "day"
    assert "day" not in at.multiselect(key="vis_encoding_color_by").options
    assert at.session_state.result[:4] == (["treatment"], "day", "day", "day")


def test_separator_survives_review_filter_and_other_methods():
    at = new(**{KEY: "day", "vis_encoding_dr_separate_by": ["dish", "treatment"],
                "analysis_control_separate_by": "dish"})
    at.session_state.single = True
    run(at)
    assert at.selectbox(key=KEY).value == "day"
    at.session_state.single = False
    at.checkbox(key="review").check()
    run(at)
    at.checkbox(key="review").uncheck()
    run(at)
    for method, expected in [("FC", "dish"), ("DR", ["dish", "treatment"]), ("Histogram", None), ("Phasor", "day")]:
        at.selectbox(key="method").set_value(method)
        run(at)
        assert at.session_state.result[3] == expected


def test_removed_or_retyped_column_is_pruned():
    for flag in ["removed", "noncategorical"]:
        at = new(**{KEY: "day"})
        at.session_state[flag] = True
        run(at)
        assert at.session_state.result[3] is None


def test_no_separator_is_default():
    assert new().session_state.result[3] is None


def test_analysis_page_passes_separator_to_plot_and_export(monkeypatch):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from src import dataset_io, export_script
    from src.vis import bivar
    from src.widgets import analysis_config_widgets as acw
    g, s = "Lifetime fit free_ch1: G(1st)", "Lifetime fit free_ch1: S(1st)"
    df = pd.DataFrame({"id": list("abcdefgh"), "day": ["D1", "D2"] * 4,
                       "treatment": ["A", "A", "B", "B"] * 2,
                       g: np.linspace(.2,.7,8), s: np.linspace(.1,.4,8)})
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: ["day", "treatment"])
    monkeypatch.setattr(acw, "get_fov_name_col_analysis", lambda *a, **k: None)
    monkeypatch.setattr(acw, "get_unique_row_id_col", lambda *a, **k: "id")
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k:
        (df.copy(), {"Uncategorized Features": [g,s]}, True, ",", "id"))
    seen = {}
    def capture_plot(df, **kwargs):
        seen["plot"] = kwargs
        return go.Figure(go.Scatter(x=[.2,.3], y=[.1,.2]),
                         layout=dict(meta={"phasor_categories": ["D1", "D2"],
                                           "phasor_separate_by": "day"})), df
    def capture_export(state):
        seen["export"] = state
        return "# test"
    monkeypatch.setattr(bivar, "phasor_plot", capture_plot)
    monkeypatch.setattr(export_script, "generate_script", capture_export)
    page = str(Path(__file__).resolve().parents[1] / "pages/data_analysis.py")
    at = AppTest.from_file(page).run(timeout=90)
    at.radio[0].set_value("### **Bivariate**").run(timeout=90)
    at.radio[1].set_value("Phasor Plot")
    at.session_state[KEY] = "day"
    at.session_state["vis_encoding_color_by"] = ["treatment"]
    at.run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert seen["plot"].get("separate_by") == "day"
    assert seen["export"]["separate_by"] == "day"
    assert seen["export"]["method_params"]["phasor_category"] == "D1"
    assert "phasor_show_other_categories" not in seen["export"]["method_params"]
    assert "day" not in at.multiselect(key="vis_encoding_color_by").options
    elements = list(at)
    keys = [getattr(el, "key", None) for el in elements]
    assert keys.index(KEY) < keys.index(CATEGORY_KEY) < keys.index("k_means_phasor_ch1")
    chart_index = next(i for i, element in enumerate(elements) if element.type == "plotly_chart")
    assert keys.index("k_means_phasor_ch1") < chart_index
    at.button_group(key=CATEGORY_KEY).set_value("D2")
    at.run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert seen["export"]["method_params"]["phasor_category"] == "D2"
    at.checkbox(key="k_means_phasor_ch1").check()
    at.run(timeout=90)
    assert not at.exception, [e.value for e in at.exception]
    assert seen["plot"]["k_means"] is True
    assert seen["export"]["method_params"]["k_means"] is True


def category_app():
    import streamlit as st
    from src.widgets.visualization_widgets import phasor_category_widget
    options = st.session_state.get("categories", ["Day 2", "Day 10", "N/A"])
    separator = st.session_state.get("separator", "day")
    st.session_state.result = phasor_category_widget(options, separator)


def test_category_buttons_remain_selected_and_follow_available_data():
    at = run(AppTest.from_function(category_app))
    assert at.session_state.result == "Day 2"
    assert not at.checkbox
    at.button_group(key=CATEGORY_KEY).set_value("Day 10")
    run(at)
    assert at.session_state.result == "Day 10"
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


def test_many_categories_use_dropdown_and_preserve_selected_value():
    at = AppTest.from_function(category_app)
    at.session_state.categories = [f"Day {i}" for i in range(10)]
    run(at)
    at.selectbox(key=CATEGORY_KEY).set_value("Day 2")
    run(at)
    assert at.session_state.result == "Day 2"
    at.session_state.categories = ["Day 1", "Day 2"]
    run(at)
    assert at.button_group(key=CATEGORY_KEY).value == "Day 2"
    assert at.session_state.result == "Day 2"


def test_chart_wrapper_keeps_native_plot_and_geometry():
    import json
    def chart_app():
        import plotly.graph_objects as go
        from src.widgets.plot_layout import phasor_chart
        fig = go.Figure(go.Scatter(x=[.3,.4], y=[.2,.3]),
                        layout=dict(meta={"phasor_subplot_layout": {"plot_height": 1.2}}))
        phasor_chart(fig, key="test_phasor")
    at = AppTest.from_function(chart_app).run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) == 1
    assert at.get("html")
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["meta"]["phasor_subplot_layout"]["plot_height"] == 1.2
