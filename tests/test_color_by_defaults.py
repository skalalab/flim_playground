"""Automatic color grouping skips the configured FOV, preserving user choices."""

import pytest
import toml
from streamlit.testing.v1 import AppTest

from src.widgets.visualization_widgets import COLOR_BY_KEY


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


def _app(mode, columns):
    import pandas as pd
    import streamlit as st
    from src.widgets.visualization_widgets import (
        POINT_MODE_KEY, visual_encoding_channels_widget,
    )

    if mode == "group_by":
        st.session_state[POINT_MODE_KEY] = "subcolor"
        mode = "sections"
    frame = pd.DataFrame({column: ["a", "b", "a", "b"] for column in columns})
    st.session_state.color_result = visual_encoding_channels_widget(
        frame, columns,
        point_based=mode != "histogram",
        separate_by_available=True,
        subcolor_available=mode == "sections",
        collapse_available=mode in {"sections", "distribution"},
        separate_by_mode=mode,
    )[0]


@pytest.mark.parametrize("mode", [
    "sections", "group_by", "histogram", "distribution", "subplots", "facets",
])
@pytest.mark.parametrize("only_fov", [False, True])
@pytest.mark.parametrize("stored", [None, ["removed_column"]], ids=["initial", "fallback"])
def test_color_default_skips_configured_fov(mode, fov_column, only_fov, stored):
    columns = [fov_column] if only_fov else [fov_column, "treatment", "dish"]
    expected = [] if only_fov else ["treatment"]
    at = AppTest.from_function(_app, args=(mode, columns))
    if stored is not None:
        at.session_state[COLOR_BY_KEY] = stored
    at.run()

    assert not at.exception
    assert at.multiselect(key=COLOR_BY_KEY).value == expected
    assert at.session_state.color_result == expected
    assert fov_column in at.multiselect(key=COLOR_BY_KEY).options


@pytest.mark.parametrize("mode", [
    "sections", "group_by", "histogram", "distribution", "subplots", "facets",
])
def test_manual_fov_and_empty_color_choices_are_retained(mode, fov_column):
    at = AppTest.from_function(_app, args=(mode, [fov_column, "treatment"])).run()
    for selection in ([fov_column], []):
        at.multiselect(key=COLOR_BY_KEY).set_value(selection).run()
        at.run()

        assert not at.exception
        assert at.multiselect(key=COLOR_BY_KEY).value == selection
        assert at.session_state.color_result == selection


@pytest.mark.parametrize("fov_column", ["field_of_view", ""], indirect=True)
def test_image_name_can_be_default_when_it_is_not_the_configured_fov(fov_column):
    at = AppTest.from_function(_app, args=("sections", ["image_name", "treatment"])).run()

    assert not at.exception
    assert at.multiselect(key=COLOR_BY_KEY).value == ["image_name"]
    assert at.session_state.color_result == ["image_name"]
