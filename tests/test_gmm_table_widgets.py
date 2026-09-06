"""Editable GMM table names stay attached to their original fitted components."""
import json

import pytest
from streamlit.testing.v1 import AppTest


def app(dimensions):
    import pandas as pd
    import streamlit as st
    from src.export_labels import apply_export_labels
    from src.widgets.export_labels_widgets import export_labels_widget

    category = st.selectbox("Category", ["Day 1", "Day 2"], key="category")
    st.checkbox("Hide results", key="hide")
    if st.session_state.hide:
        st.stop()
    data = pd.DataFrame({"x": [1., 2., 3., 4.], "y": [2., 3., 4., 5.], "labels": [
        "Day 1::ctrl_group1", "Day 1::ctrl_group2", "Day 2::ctrl_group1", "Day 2::ctrl_group2"]})
    features = ["x", "y"][:dimensions]
    context = {"features": features, "fit": {"components": st.session_state.get("components", 2), "log_x": True}}
    tables = [{"category": category, "group": "ctrl", "features": features,
        **({"h_index": .49} if dimensions == 1 else {}), "rows": [
        {"source_label": f"{category}::ctrl_group{i}", "component": i,
         "x_mean_sd": "2.00 ± 0.30", "weight": .5,
         **({"y_mean_sd": "3.00 ± 0.40"} if dimensions == 2 else {})}
        for i in (1, 2)]}]
    settings, valid = export_labels_widget(
        data, "labels", method="Feature Histogram" if dimensions == 1 else "2D Feature Distribution",
        context=context, component_tables=tables)
    st.session_state.settings, st.session_state.valid = settings, valid
    st.session_state.csv = apply_export_labels(data, "labels", settings).to_csv(index=False) if valid else None
    st.download_button("Download", st.session_state.csv or "", disabled=not valid)


@pytest.fixture(params=[1, 2], ids=["1d", "2d"])
def editor_app(request):
    return request.param, AppTest.from_function(app, args=(request.param,))


def run(at):
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    return at


def edit(at, rows):
    """Deliver the same cumulative edit payload that the native table sends."""
    table = at.dataframe[0]
    states = at._tree.get_widget_states()
    states.widgets.add(id=table.proto.id, string_value=json.dumps({
        "edited_rows": rows, "added_rows": [], "deleted_rows": []}))
    at._run(states, timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_existing_component_statistics_have_one_editable_name_column(editor_app):
    dimensions, at = editor_app
    run(at)
    assert len(at.dataframe) == 1
    table = at.dataframe[0]
    stats = ["Mean ± SD"] if dimensions == 1 else ["X (mean ± SD)", "Y (mean ± SD)"]
    assert table.value.columns.tolist() == ["#", "Name", *stats, "Weight"]
    assert table.value["Name"].tolist() == ["Day 1::ctrl_group1", "Day 1::ctrl_group2"]
    config = json.loads(table.proto.columns)
    assert config["Name"]["label"] == "✎ Name"
    assert not config["Name"].get("disabled", False)
    assert all(config[col]["disabled"] for col in ["#", *stats, "Weight"])
    if dimensions == 1:
        assert any(item.value == "<p><strong>ctrl (H-index: 0.490)</strong></p>" for item in at.markdown)
    assert config[stats[0]]["help"] == "x (log₁₀)"
    assert config["_index"]["hidden"]
    assert [w.label for w in at.text_input] == ["Exported column name"]
    assert not at.caption


def test_successive_name_edits_keep_the_editor_identity_and_update_csv(editor_app):
    _, at = editor_app
    run(at)
    identity = at.dataframe[0].proto.id
    edit(at, {0: {"Name": " Low <α> "}})
    assert at.dataframe[0].proto.id == identity
    assert at.session_state.settings["value_names"]["Day 1::ctrl_group1"] == "Low <α>"
    edit(at, {0: {"Name": " Low <α> "}, 1: {"Name": "High"}})
    assert at.dataframe[0].proto.id == identity
    assert at.session_state.settings["value_names"]["Day 1::ctrl_group2"] == "High"
    assert "Low <α>" in at.session_state.csv and "High" in at.session_state.csv
    assert at.session_state.valid


def test_category_switching_and_remounting_keep_every_components_name(editor_app):
    _, at = editor_app
    run(at)
    edit(at, {0: {"Name": "Low"}})
    at.selectbox(key="category").set_value("Day 2")
    run(at)
    edit(at, {1: {"Name": "Activated"}})
    at.selectbox(key="category").set_value("Day 1")
    run(at)
    assert at.dataframe[0].value["Name"].tolist() == ["Low", "Day 1::ctrl_group2"]
    assert at.session_state.settings["value_names"]["Day 2::ctrl_group2"] == "Activated"
    at.checkbox(key="hide").check()
    run(at)
    at.checkbox(key="hide").uncheck()
    run(at)
    assert at.dataframe[0].value["Name"].iloc[0] == "Low"
    edit(at, {1: {"Name": "New high"}})
    assert at.session_state.settings["value_names"]["Day 1::ctrl_group1"] == "Low"
    assert at.session_state.settings["value_names"]["Day 1::ctrl_group2"] == "New high"


def test_invalid_hidden_names_still_disable_export_and_shared_names_can_combine(editor_app):
    _, at = editor_app
    run(at)
    edit(at, {0: {"Name": None}})
    assert not at.session_state.valid
    assert at.get("download_button")[0].proto.disabled
    at.selectbox(key="category").set_value("Day 2")
    run(at)
    assert not at.session_state.valid
    assert "Day 1::ctrl_group1" in at.error[0].value
    at.selectbox(key="category").set_value("Day 1")
    run(at)
    edit(at, {0: {"Name": "Shared"}, 1: {"Name": "Shared"}})
    assert at.session_state.valid
    assert any("combine" in item.value for item in at.info)


def test_fit_changes_reset_tables_and_names(editor_app):
    _, at = editor_app
    run(at)
    edit(at, {0: {"Name": "Low"}})
    old_id = at.dataframe[0].proto.id
    at.session_state.components = 3
    run(at)
    assert at.dataframe[0].proto.id != old_id
    assert at.dataframe[0].value["Name"].iloc[0] == "Day 1::ctrl_group1"
    assert at.session_state.settings["value_names"]["Day 1::ctrl_group1"] == "Day 1::ctrl_group1"
