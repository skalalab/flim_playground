"""Classify by defaults to the first eligible category other than the configured FOV."""

import pytest
import toml
from streamlit.testing.v1 import AppTest


@pytest.fixture(params=["image_name", "field_of_view"])
def fov_column(request, isolated_config_paths):
    name = request.param
    isolated_config_paths[0].write_text(toml.dumps({
        "current_profile": "selected",
        "profiles": {
            "other": {"fov_name_col": "treatment"},
            "selected": {"fov_name_col": name},
        },
    }))
    return name


def _app(columns):
    import pandas as pd
    import streamlit as st
    from src.widgets.classification_widgets import classifier_options_widget

    frame = pd.DataFrame({column: ["a", "b", "a", "b"] for column in columns})
    frame["feature"] = [1., 2., 3., 4.]
    message, *_ = classifier_options_widget(
        frame, columns, fov_name_col=None, selected_features=["feature"],
        classifier="Random Forest", splits=0.7,
    )
    if message:
        st.error(message)


@pytest.mark.parametrize("position", ["first", "last", "only"])
def test_default_skips_configured_fov_and_uses_first_category(fov_column, position):
    columns = {
        "first": [fov_column, "treatment", "dish"],
        "last": ["treatment", "dish", fov_column],
        "only": [fov_column],
    }[position]
    at = AppTest.from_function(_app, args=(columns,)).run(timeout=30)

    assert not at.exception
    selector = at.multiselect(key="classify_by_multiselect")
    assert selector.value == ([] if position == "only" else ["treatment"])
    assert fov_column in selector.options
    if position == "only":
        assert "Please select at least one category" in at.error[0].value
    else:
        assert not at.error


def test_manual_fov_and_empty_choices_are_retained(fov_column):
    at = AppTest.from_function(_app, args=([fov_column, "treatment"],)).run(timeout=30)
    for selection in ([fov_column], []):
        at.multiselect(key="classify_by_multiselect").set_value(selection).run()
        at.run()

        assert not at.exception
        assert at.multiselect(key="classify_by_multiselect").value == selection


@pytest.mark.parametrize("fov_column", ["field_of_view", ""], indirect=True)
def test_image_name_can_be_default_when_it_is_not_the_configured_fov(fov_column):
    at = AppTest.from_function(_app, args=(["image_name", "treatment"],)).run(timeout=30)

    assert not at.exception
    assert at.multiselect(key="classify_by_multiselect").value == ["image_name"]
