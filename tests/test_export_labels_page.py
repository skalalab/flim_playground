"""The real analysis page sends the same editable names to both download paths."""
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest


def _with_component_tables(fig, dimensions=2):
    meta = {**(fig.layout.meta or {}), "distribution_statistics": "Statistics only",
        "gmm_component_tables": [{"category": None, "group": "all_data", "features": ["x", "y"][:dimensions],
            "rows": [{"source_label": f"all_data_group{i}", "component": i,
                      "x_mean_sd": "1.00 ± 0.20", "y_mean_sd": "3.00 ± 0.30", "weight": .5}
                     for i in (1, 2)]}]}
    if dimensions == 1:
        meta["gmm_component_tables"][0]["h_index"] = .4
        meta.update(histogram_gmm=True, histogram_feature="x", histogram_separator=None,
            histogram_summaries=[{"category": None, "groups": [{
                "label": "all_data", "color_group": "all_data", "notices": [],
                "components": [(1, "1.00 ± 0.20", "0.50"), (2, "3.00 ± 0.30", "0.50")],
                "h_index": .4, "thresholds": [2.]}]}])
    fig.update_layout(meta=meta)
    return fig


@pytest.mark.parametrize("method,legacy", [("Feature Histogram", "GMM_group"),
    ("2D Feature Distribution", "2D_GMM_group")])
def test_page_captures_export_names_and_disables_both_downloads(monkeypatch, method, legacy):
    from src import dataset_io, export_script
    from src.vis import bivar, univar
    from src.widgets import analysis_config_widgets as acw, selection_widgets as sw, visualization_widgets as vw
    g, s = "Lifetime fit free_ch1: G(1st)", "Lifetime fit free_ch1: S(1st)"
    source = pd.DataFrame({"id": ["a", "b", "c"], "x": [1., 2., 3.], "y": [2., 1., 4.],
                           g: [.2, .3, .4], s: [.1, .2, .1], legacy: ["previous"] * 3})
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: [legacy])
    monkeypatch.setattr(acw, "get_fov_name_col_analysis", lambda *a, **k: None)
    monkeypatch.setattr(acw, "get_unique_row_id_col", lambda *a, **k: "id")
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        source.copy(), {"Uncategorized Features": ["x", "y", g, s]}, True, ",", "id"))
    monkeypatch.setattr(sw, "single_feature_select_widget", lambda *a, **k: "x")
    monkeypatch.setattr(sw, "twod_single_feature_select_widget", lambda *a, **k: ("x", "y"))
    monkeypatch.setattr(vw, "phasor_params_widget", lambda *a, **k: ("ch1", 1, .08))
    figure = lambda: go.Figure(go.Scatter(x=[1, 2, 3], y=[2, 1, 4]),
                               layout={"meta": {"histogram_gmm": False}})
    monkeypatch.setattr(univar, "feature_comparison_plot", lambda *a, **k: figure())
    monkeypatch.setattr(univar, "feature_histogram_plot", lambda *a, **k: figure())
    outputs, captured = [], []

    def model(data, *args, **kwargs):
        result = data.copy()
        result[kwargs.get("label_column") or legacy] = ["all_data_group1", "all_data_group2", None]
        outputs.append(result)
        return figure(), result

    def model_2d(data, *args, **kwargs):
        fig, result = model(data, *args, **kwargs)
        return _with_component_tables(fig), "<table>Original static GMM table</table>", result

    def model_histogram(data, *args, **kwargs):
        vw.gmm_hyperParams_widget()
        fig, result = model(data, *args, **kwargs)
        return _with_component_tables(fig, dimensions=1), result

    monkeypatch.setattr(univar, "feature_gmm_plot", model_histogram)
    monkeypatch.setattr(bivar, "feature_2d_distribution_plot", model_2d)
    monkeypatch.setattr(bivar, "phasor_plot", model)
    monkeypatch.setattr(export_script, "generate_script", lambda state: captured.append(state) or "# analysis")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages/data_analysis.py"))
    at.run(timeout=60)
    if method != "Feature Histogram":
        at.radio[0].set_value("### **Bivariate**").run(timeout=60)
    at.session_state.analysis_control_apply_gmm = True
    at.session_state["fit_gmm_2d_x_y"] = True
    at.radio[1].set_value(method).run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    naming = [w for w in at.text_input if w.label == "Exported column name"]
    assert naming and naming[0].value == legacy + "_2"
    naming[0].set_value(" Cell state α ").run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    assert captured[-1]["derived_export"]["column_name"] == "Cell state α"
    assert outputs[-1][legacy].eq("previous").all()
    if method == "Feature Histogram":
        at.slider(key="fit_gmm_max_components").set_value(4).run(timeout=60)
        at.checkbox(key="log_x_hist_x").check().run(timeout=60)
        naming = next(w for w in at.text_input if w.label == "Exported column name")
        naming.set_value("Cell state α").run(timeout=60)
        at.radio[1].set_value("Feature Comparison").run(timeout=60)
        at.radio[1].set_value("Feature Histogram").run(timeout=60)
        assert not at.exception, [e.value for e in at.exception]
        assert at.slider(key="fit_gmm_max_components").value == 4
        assert at.checkbox(key="log_x_hist_x").value
        assert next(w for w in at.text_input if w.label == "Exported column name").value == "Cell state α"
        assert not at.warning
    downloads = list(at.get("download_button"))
    assert len(downloads) == 2 and not any(w.proto.disabled for w in downloads)
    naming = next(w for w in at.text_input if w.label == "Exported column name")
    naming.set_value("x").run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    assert at.error
    assert all(w.proto.disabled for w in at.get("download_button"))


@pytest.fixture
def grouping_page(monkeypatch):
    """Keep model output stable while the real page owns grouping and export state."""
    from src import dataset_io, export_script
    from src.vis import bivar, univar
    from src.widgets import analysis_config_widgets as acw, selection_widgets as sw

    source = pd.DataFrame({
        "id": ["a", "b", "c", "d"], "x": [-1., 2., 3., 4.], "y": [2., 1., 4., 3.],
        "day": ["D1", "D1", "D2", "D2"], "treatment": ["ctrl", "drug", "ctrl", "drug"],
    })
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: ["day", "treatment"])
    monkeypatch.setattr(acw, "get_fov_name_col_analysis", lambda *a, **k: None)
    monkeypatch.setattr(acw, "get_unique_row_id_col", lambda *a, **k: "id")
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        source.copy(), {"Uncategorized Features": ["x", "y"]}, True, ",", "id"))
    monkeypatch.setattr(sw, "single_feature_select_widget", lambda *a, **k: "x")
    monkeypatch.setattr(sw, "twod_single_feature_select_widget", lambda *a, **k: ("x", "y"))

    def figure():
        return go.Figure(go.Scatter(x=[1, 2], y=[2, 1]),
                         layout={"meta": {"histogram_gmm": False}})

    def model(data, *args, **kwargs):
        result = data.copy()
        result[kwargs["label_column"]] = [f"all_data_group{i % 2 + 1}" for i in range(len(result))]
        return figure(), result

    def model_2d(data, *args, **kwargs):
        fig, result = model(data, *args, **kwargs)
        return _with_component_tables(fig), "<table>Original static GMM table</table>", result

    def model_histogram(data, *args, **kwargs):
        fig, result = model(data, *args, **kwargs)
        return _with_component_tables(fig, dimensions=1), result

    monkeypatch.setattr(univar, "feature_comparison_plot", lambda *a, **k: figure())
    monkeypatch.setattr(univar, "feature_histogram_plot", lambda *a, **k: figure())
    monkeypatch.setattr(univar, "feature_gmm_plot", model_histogram)
    monkeypatch.setattr(bivar, "feature_2d_distribution_plot", model_2d)
    captured = []
    monkeypatch.setattr(export_script, "generate_script", lambda state: captured.append(state) or "# analysis")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages/data_analysis.py"))
    at.run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    return at, captured


def _export_column_input(at):
    return next(w for w in at.text_input if w.label == "Exported column name")


def _edit_component_names(at, rows):
    states = at._tree.get_widget_states()
    states.widgets.add(id=at.dataframe[0].proto.id, string_value=json.dumps({
        "edited_rows": rows, "added_rows": [], "deleted_rows": []}))
    at._run(states, timeout=60)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("dimensions", [1, 2], ids=["1d", "2d"])
def test_gmm_table_renames_use_shared_export_mapping_and_replace_static_tables(grouping_page, dimensions):
    at, captured = grouping_page
    if dimensions == 1:
        at.session_state.analysis_control_apply_gmm = True
        at.radio[1].set_value("Feature Histogram").run(timeout=60)
    else:
        at.session_state["fit_gmm_2d_x_y"] = True
        at.radio[0].set_value("### **Bivariate**").run(timeout=60)
    assert len(at.dataframe) == 1
    assert not any(w.value == "**Export labels**" for w in at.markdown)
    assert not any("Original static GMM table" in w.value for w in at.markdown)
    assert not any("flim-gmm-table" in w.value for w in at.markdown)
    assert not any(w.label.startswith("New name for") for w in at.text_input)
    if dimensions == 1:
        assert not any(w.label.startswith("GMM details") for w in at.expander)
        assert not any("GMM details" in w.value or "All observations" in w.value for w in at.markdown)
        assert "H-index" not in at.dataframe[0].value.columns
        assert any(w.value == "<p><strong>all_data (H-index: 0.400)</strong></p>" for w in at.markdown)
        assert not any(w.value.startswith("H-index for") for w in at.markdown)
        assert any("Threshold for" in w.value for w in at.markdown)
    _edit_component_names(at, {0: {"Name": "Low"}})
    _edit_component_names(at, {0: {"Name": "Low"}, 1: {"Name": "High"}})
    assert captured[-1]["derived_export"]["value_names"] == {
        "all_data_group1": "Low", "all_data_group2": "High"}
    assert not any(w.proto.disabled for w in at.get("download_button"))
    _edit_component_names(at, {0: {"Name": " "}, 1: {"Name": "High"}})
    assert all(w.proto.disabled for w in at.get("download_button"))


def test_histogram_single_component_keeps_read_only_details_without_rename_controls(grouping_page, monkeypatch):
    from src.vis import univar

    def single_component(data, *args, **kwargs):
        fig = _with_component_tables(go.Figure(), dimensions=1)
        fig.layout.meta["gmm_component_tables"] = []
        group = fig.layout.meta["histogram_summaries"][0]["groups"][0]
        group.update(components=group["components"][:1], h_index=0., thresholds=[])
        result = data.copy()
        result[kwargs["label_column"]] = None
        return fig, result

    monkeypatch.setattr(univar, "feature_gmm_plot", single_component)
    at, captured = grouping_page
    at.session_state.analysis_control_apply_gmm = True
    at.radio[1].set_value("Feature Histogram").run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.dataframe and not at.text_input
    assert not any(w.label.startswith("GMM details") for w in at.expander)
    assert any("flim-gmm-table" in w.value for w in at.markdown)
    assert any("H-index" in w.value for w in at.markdown)
    assert any("No group labels" in w.value for w in at.info)
    assert captured[-1]["derived_export"] is None
    assert len(at.get("download_button")) == 1
    assert not at.get("download_button")[0].proto.disabled


def test_histogram_export_names_survive_classification_hiding_color_by(grouping_page):
    at, captured = grouping_page
    at.session_state.analysis_control_apply_gmm = True
    at.radio[1].set_value("Feature Histogram").run(timeout=60)
    at.multiselect(key="vis_encoding_color_by").set_value(["treatment"]).run(timeout=60)
    _export_column_input(at).set_value("Cell state").run(timeout=60)
    assert captured[-1]["derived_export"]["column_name"] == "Cell state"

    at.radio[0].set_value("### **Multivariate**").run(timeout=60)
    at.radio[1].set_value("Classification").run(timeout=60)
    at.radio[0].set_value("### **Univariate**").run(timeout=60)
    at.radio[1].set_value("Feature Histogram").run(timeout=60)

    assert not at.exception, [e.value for e in at.exception]
    assert (at.multiselect(key="vis_encoding_color_by").value,
            _export_column_input(at).value,
            captured[-1]["derived_export"]["column_name"]) == (["treatment"], "Cell state", "Cell state")


def test_2d_export_names_survive_histogram_hiding_collapse_by(grouping_page):
    at, captured = grouping_page
    at.session_state["fit_gmm_2d_x_y"] = True
    at.radio[0].set_value("### **Bivariate**").run(timeout=60)
    at.radio[1].set_value("2D Feature Distribution").run(timeout=60)
    at.multiselect(key="vis_encoding_color_by").set_value(["treatment"]).run(timeout=60)
    at.selectbox(key="vis_encoding_collapse_by").set_value("day").run(timeout=60)
    _export_column_input(at).set_value("Collapsed state").run(timeout=60)
    assert captured[-1]["derived_export"]["column_name"] == "Collapsed state"

    at.radio[0].set_value("### **Univariate**").run(timeout=60)
    at.radio[1].set_value("Feature Histogram").run(timeout=60)
    at.radio[0].set_value("### **Bivariate**").run(timeout=60)
    at.radio[1].set_value("2D Feature Distribution").run(timeout=60)

    assert not at.exception, [e.value for e in at.exception]
    assert (at.selectbox(key="vis_encoding_collapse_by").value,
            _export_column_input(at).value,
            captured[-1]["derived_export"]["column_name"]) == ("day", "Collapsed state", "Collapsed state")


def test_2d_mean_help_uses_effective_scale_when_negative_values_prevent_log(grouping_page):
    at, _ = grouping_page
    at.session_state["fit_gmm_2d_x_y"] = True
    at.session_state["log_x_2d_x_y"] = True
    at.radio[0].set_value("### **Bivariate**").run(timeout=60)
    at.radio[1].set_value("2D Feature Distribution").run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    assert at.checkbox(key="log_x_2d_x_y").value
    help_text = json.loads(at.dataframe[0].proto.columns)["X (mean ± SD)"]["help"]
    assert help_text == "x"
    assert "log₁₀" not in help_text
