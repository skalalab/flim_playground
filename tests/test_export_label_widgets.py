"""Naming edits survive remounts and invalidate safely when model inputs change."""
import pytest
from streamlit.testing.v1 import AppTest


def app():
    import pandas as pd
    import streamlit as st
    from src.widgets.export_labels_widgets import export_labels_widget

    method = st.selectbox("Method", ["Feature Histogram", "2D Feature Distribution"], key="method")
    if st.checkbox("Review", key="review"):
        st.stop()
    labels = ["ctrl_group1", "drug_group2", None]
    if st.session_state.get("empty"):
        labels = [None] * 3
    data = pd.DataFrame({"x": [1., 3., 5.], "labels": labels})
    if st.session_state.get("changed_data"):
        data.loc[0, "x"] = 2.
    settings, valid = export_labels_widget(
        data, "labels", method=method,
        context={"dataset": st.session_state.get("dataset", "file1"),
                 "max_components": st.session_state.get("components", 2)})
    st.session_state.settings = settings
    st.session_state.valid = valid
    st.download_button("Download", "csv", disabled=not valid)


def run(at):
    at.run(timeout=30)
    assert not at.exception, [item.value for item in at.exception]
    return at


def test_prefilled_names_and_column_edits_survive_method_switch_and_review():
    at = run(AppTest.from_function(app))
    assert at.text_input[0].value == "labels"
    at.text_input[0].set_value("Cell state")
    run(at)
    assert at.session_state.settings["column_name"] == "Cell state"
    at.selectbox(key="method").set_value("2D Feature Distribution")
    run(at)
    assert at.text_input[0].value == "labels"
    at.selectbox(key="method").set_value("Feature Histogram")
    run(at)
    assert at.text_input[0].value == "Cell state"
    at.checkbox(key="review").check()
    run(at)
    at.checkbox(key="review").uncheck()
    run(at)
    assert at.text_input[0].value == "Cell state"
    assert not at.warning


def test_name_collisions_and_blank_names_disable_download():
    at = run(AppTest.from_function(app))
    for invalid in ["x", "  "]:
        at.text_input[0].set_value(invalid)
        run(at)
        assert not at.session_state.valid
        assert at.error
        assert at.get("download_button")[0].proto.disabled
    at.text_input[0].set_value(" State α ")
    run(at)
    assert at.session_state.settings["column_name"] == "State α"
    assert at.session_state.valid


def test_value_edits_are_cumulative_and_survive_remount():
    at = run(AppTest.from_function(app))
    assert [w.value for w in at.text_input] == ["labels", "ctrl_group1", "drug_group2"]
    assert not at.dataframe
    assert not at.caption
    at.text_input[1].set_value("Low")
    run(at)
    assert at.session_state.settings["value_names"]["ctrl_group1"] == "Low"
    at.text_input[2].set_value("Low")
    run(at)
    assert set(at.session_state.settings["value_names"].values()) == {"Low"}
    assert any("combine" in item.value for item in at.info)
    at.checkbox(key="review").check()
    run(at)
    at.checkbox(key="review").uncheck()
    run(at)
    assert set(at.session_state.settings["value_names"].values()) == {"Low"}
    assert [w.value for w in at.text_input[1:]] == ["Low", "Low"]
    at.text_input[1].set_value(" ")
    run(at)
    assert not at.session_state.valid
    assert at.get("download_button")[0].proto.disabled


def test_fit_changes_reset_names_but_display_changes_do_not():
    for key, value in [("components", 3), ("dataset", "file2"), ("changed_data", True)]:
        at = run(AppTest.from_function(app))
        at.text_input[0].set_value("Custom")
        run(at)
        at.session_state.display_color = "blue"
        run(at)
        assert at.text_input[0].value == "Custom"
        at.session_state[key] = value
        run(at)
        assert at.text_input[0].value == "labels"


def test_no_assignments_explains_absence_without_invalidating_script_export():
    at = AppTest.from_function(app)
    at.session_state.empty = True
    run(at)
    assert not at.text_input
    assert at.session_state.settings is None
    assert at.session_state.valid
    assert any("No group labels" in item.value for item in at.info)
    assert any("more than one component" in item.value for item in at.info)


def means_app():
    import pandas as pd
    import streamlit as st
    from src.widgets.export_labels_widgets import export_labels_widget

    # These are already-analyzed values, including any log transform.
    data = pd.DataFrame({"x <raw>": [1., 3., 9., 100.], "y": [4., 8., 20., 100.],
                         "labels": ["ctrl_group1", "ctrl_group1", "ctrl_group2", None]})
    export_labels_widget(data, "labels", method=st.session_state.method,
                         context={"features": st.session_state.features,
                                  "fit": {"log_x": True}})


@pytest.mark.parametrize("method,features", [
    ("Feature Histogram", ["x <raw>"]),
    ("2D Feature Distribution", ["x <raw>", "y"]),
])
def test_group_help_shows_assigned_means_on_the_analyzed_scale(method, features):
    at = AppTest.from_function(means_app)
    at.session_state.method = method
    at.session_state.features = features
    run(at)
    help_text = at.text_input[1].proto.help
    assert "Mean of assigned rows" in help_text
    assert "`x <raw>`" in help_text and "2" in help_text
    assert "log₁₀" in help_text
    if len(features) == 2:
        assert "`y`" in help_text and "6" in help_text
    assert "100" not in help_text and "count" not in help_text.lower()
    assert not at.dataframe and not at.caption
    at.text_input[1].set_value("Low")
    run(at)
    assert at.text_input[1].proto.help == help_text
