"""State decisions for Feature Comparison's single point-encoding picker."""

import pytest

from src.widgets import encoding_state
from src.widgets import visualization_widgets as vw


@pytest.mark.parametrize(
    "picker,as_colour,opacity,expected",
    [
        (None, False, None, ("shape", None)),
        ("patient", False, "dish", ("shape", "patient")),
        ("patient", True, "dish", ("subcolor", "patient")),
        (None, True, "dish", ("subcolor", None)),
        (None, False, "dish", ("opacity", "dish")),
    ],
)
def test_initial_mode_preserves_legacy_picker_intent(picker, as_colour, opacity, expected):
    assert encoding_state.initial_point_encoding(picker, as_colour, opacity) == expected


@pytest.mark.parametrize(
    "selected,last_mode,expected",
    [
        ("shape", "opacity", "shape"),
        ("subcolor", "shape", "subcolor"),
        ("opacity", "shape", "opacity"),
        (None, "shape", "shape"),
        (None, "subcolor", "subcolor"),
        (None, "opacity", "opacity"),
        ("unknown", "opacity", "opacity"),
        (None, None, "shape"),
    ],
)
def test_clearing_a_segment_retains_the_last_valid_mode(selected, last_mode, expected):
    assert encoding_state.resolve_point_mode(selected, last_mode) == expected


@pytest.mark.parametrize(
    "mode,has_groups,expected",
    [
        ("shape", True, (None, "patient", None)),
        ("subcolor", True, (None, None, "patient")),
        ("opacity", True, ("patient", None, None)),
        ("shape", False, (None, "patient", None)),
        ("subcolor", False, (None, None, None)),
        ("opacity", False, ("patient", None, None)),
    ],
)
def test_only_the_active_role_receives_the_shared_column(mode, has_groups, expected):
    assert encoding_state.point_encoding_channels(mode, "patient", has_groups) == expected


@pytest.mark.parametrize("mode", ["shape", "subcolor", "opacity"])
def test_an_empty_picker_disables_every_role(mode):
    assert encoding_state.point_encoding_channels(mode, None, True) == (None, None, None)


def _widget_app():
    import pandas as pd
    import streamlit as st
    from src.widgets.visualization_widgets import visual_encoding_channels_widget

    method = st.selectbox("Method", ["Feature Comparison", "UMAP", "Histogram"],
                          key="test_method")
    data = pd.DataFrame({
        "group": ["a", "b"] * 4,
        "patient": ["p1", "p2", "p3", "p4"] * 2,
        "dish": ["d1", "d1", "d2", "d2"] * 2,
    })
    st.session_state["test_result"] = visual_encoding_channels_widget(
        data, list(data.columns), point_based=method != "Histogram",
        subcolor_available=method == "Feature Comparison",
        separate_by_available=True, collapse_available=method == "Feature Comparison",
    )


def _run_widget(at):
    at.run()
    assert not at.exception
    assert not at.warning
    return at


def _new_widget(state=None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_widget_app)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    return _run_widget(at)


def test_fc_renders_one_native_mode_selector_and_one_decoration_picker():
    at = _new_widget()
    assert len(at.get("button_group")) == 1
    assert at.get("button_group")[0].label == "Point encoding"
    assert at.session_state["vis_encoding_point_mode"] == "shape"
    assert at.selectbox(key=vw.PICKER_COL_KEY).label == "Shape by"
    assert {box.label for box in at.selectbox} == {
        "Method", "Separate by", "Collapse by", "Shape by",
    }
    assert at.session_state["test_result"] == (["group"], None, None, None, None, None)


def test_native_mode_changes_reuse_the_column_and_relabel_color_immediately():
    at = _new_widget({vw.PICKER_COL_KEY: "patient"})
    assert len(at.get("button_group")) == 1
    for mode, expected in [
        ("subcolor", (["group"], None, None, None, "patient", None)),
        ("opacity", (["group"], "patient", None, None, None, None)),
        ("shape", (["group"], None, "patient", None, None, None)),
    ]:
        at.get("button_group")[0].set_value([mode])
        _run_widget(at)
        assert at.session_state["test_result"] == expected
        assert at.selectbox(key=vw.PICKER_COL_KEY).value == "patient"
        assert at.selectbox(key=vw.PICKER_COL_KEY).label == f"{mode.title()} by"
        assert at.multiselect[0].label == ("Group by" if mode == "subcolor" else "Color by")


def test_clicking_the_active_segment_retains_its_mode_and_column():
    at = _new_widget({vw.AS_COLOUR_KEY: True, vw.PICKER_COL_KEY: "patient"})
    assert len(at.get("button_group")) == 1
    at.get("button_group")[0].set_value([])
    _run_widget(at)
    assert at.session_state["vis_encoding_point_mode"] == "subcolor"
    assert at.session_state["test_result"] == (["group"], None, None, None, "patient", None)


def test_subcolor_without_groups_keeps_the_field_but_disables_its_mapping():
    at = _new_widget({vw.AS_COLOUR_KEY: True, vw.PICKER_COL_KEY: "patient"})
    assert len(at.get("button_group")) == 1
    at.multiselect[0].set_value([])
    _run_widget(at)
    assert at.multiselect[0].label == "Group by"
    assert at.selectbox(key=vw.PICKER_COL_KEY).disabled
    assert at.selectbox(key=vw.PICKER_COL_KEY).value == "patient"
    assert at.session_state["test_result"] == ([], None, None, None, None, None)
    at.multiselect[0].set_value(["group"])
    _run_widget(at)
    assert at.session_state["test_result"] == (["group"], None, None, None, "patient", None)


def test_legacy_opacity_is_migrated_once_and_preserved_for_other_methods():
    at = _new_widget({vw.OPACITY_BY_KEY: "dish"})
    assert len(at.get("button_group")) == 1
    assert at.session_state["test_result"] == (["group"], "dish", None, None, None, None)
    at.selectbox(key=vw.PICKER_COL_KEY).set_value(None)
    _run_widget(at)
    at.selectbox(key="test_method").set_value("UMAP")
    _run_widget(at)
    assert len(at.get("button_group")) == 0
    assert at.selectbox(key=vw.OPACITY_BY_KEY).value == "dish"
    assert at.selectbox(key=vw.PICKER_COL_KEY).value is None
    at.selectbox(key="test_method").set_value("Feature Comparison")
    _run_widget(at)
    assert at.session_state["vis_encoding_point_mode"] == "opacity"
    assert at.session_state["test_result"] == (["group"], None, None, None, None, None)


def test_mode_survives_non_point_methods_while_picker_follows_widget_cleanup():
    at = _new_widget({vw.OPACITY_BY_KEY: "dish"})
    assert len(at.get("button_group")) == 1
    at.selectbox(key="test_method").set_value("Histogram")
    _run_widget(at)
    assert vw.PICKER_COL_KEY not in at.session_state
    at.selectbox(key="test_method").set_value("Feature Comparison")
    _run_widget(at)
    assert at.session_state["vis_encoding_point_mode"] == "opacity"
    assert at.selectbox(key=vw.PICKER_COL_KEY).value is None
