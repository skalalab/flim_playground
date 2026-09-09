"""Comparison sections contain only observed section/color pairs, even with shared labels."""

import contextlib

import numpy as np
import pandas as pd
import pytest

from src.collapse import collapse_rows
from src.vis import univar


@pytest.fixture
def controls(monkeypatch):
    settings = {"overlay": "None"}
    monkeypatch.setattr(univar.st, "session_state", {"plot_show_group_counts": False})
    monkeypatch.setattr(univar.st, "container", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(univar.st, "checkbox", lambda label, value=False, **kwargs: value)
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")
    monkeypatch.setattr(univar, "comparison_overlay_widget", lambda *args: settings["overlay"])
    return settings


def _frame(pairs):
    rows = []
    for pair_index, (batch, treatment) in enumerate(pairs):
        for dish in ["D1", "D2", "D3"]:
            for value in [1., 2., 4.]:
                rows.append(dict(id=f"cell{len(rows)}", batch=batch, treatment=treatment,
                                 dish=dish, value=value + pair_index * 10 + int(dish[-1])))
    return pd.DataFrame(rows)


def _render(source, controls, **kwargs):
    primary, row_id, extra = source, "id", {}
    if controls["overlay"] == "SuperPlot":
        primary, row_id, _ = collapse_rows(source, "dish", ["treatment", "batch"], "id")
        extra = dict(collapse_by="dish", source_df=source, source_row_id_col="id")
    fig = univar.feature_comparison_plot(
        primary, row_id, None, "value", ["treatment"], separate_by="batch", **extra, **kwargs)
    return fig, primary, row_id


def _assert_group_positions(fig, primary, row_id, positions, overlay):
    assert list(fig.layout.xaxis.ticktext) == [color for section, color in positions]
    np.testing.assert_allclose(fig.layout.xaxis.tickvals, list(positions.values()))
    # Collapsed hover labels can repeat across groups, so retain every point.
    expected = sorted((str(row[row_id]), row["value"], positions[(row["batch"], row["treatment"])])
                      for _, row in primary.iterrows())
    actual = []
    for trace in fig.data:
        if trace.text is None:
            continue
        if overlay == "SuperPlot" and (trace.meta or {}).get("superplot_role") != "replicate":
            continue
        actual.extend((str(identity), y, x) for identity, x, y in zip(trace.text, trace.x, trace.y))
    actual.sort()
    assert len(actual) == len(expected)
    for (identity, y, x), (expected_id, value, center) in zip(actual, expected):
        assert identity == expected_id
        assert y == pytest.approx(value)
        assert abs(x - center) <= .35
    if overlay == "Boxplot":
        boxes = [trace for trace in fig.data if trace.type == "box"]
        assert sorted(float(trace.x[0]) for trace in boxes) == sorted(positions.values())


@pytest.mark.parametrize("overlay", ["None", "Boxplot", "SuperPlot"])
@pytest.mark.parametrize("reordered", [False, True])
def test_overlapping_section_and_color_labels_create_only_real_slots(controls, overlay, reordered):
    controls["overlay"] = overlay
    source = _frame([("A", "B"), ("B", "A"), ("B", "C")])
    custom_order = {"separate_groups": ["B", "A"], "compare_groups": ["C", "B", "A"]} if reordered else None
    expected = ({("B", "C"): 0., ("B", "A"): 1., ("A", "B"): 2.5} if reordered else
                {("A", "B"): 0., ("B", "A"): 1.5, ("B", "C"): 2.5})
    fig, primary, row_id = _render(source, controls, custom_order=custom_order)
    _assert_group_positions(fig, primary, row_id, expected, overlay)
    headings = [(annotation.text, annotation.x) for annotation in fig.layout.annotations]
    assert headings == ([("<b>B</b>", .5), ("<b>A</b>", 2.5)] if reordered else
                        [("<b>A</b>", 0.), ("<b>B</b>", 2.)])
    dividers = [shape.x0 for shape in fig.layout.shapes if shape.type == "line" and shape.x0 == shape.x1]
    assert dividers == ([1.75] if reordered else [.75])


@pytest.mark.parametrize("channel", ["shape_by", "opacity_by"])
@pytest.mark.parametrize("label_role", ["section", "color"])
def test_decoration_labels_cannot_create_additional_section_color_slots(controls, channel, label_role):
    source = _frame([("Day 1", "Control"), ("Day 2", "Drug")])
    labels = {"Day 1": "Day 2", "Day 2": "Day 1"} if label_role == "section" else {"Day 1": "Drug", "Day 2": "Control"}
    source["decoration"] = source["batch"].map(labels)
    fig, primary, row_id = _render(source, controls, **{channel: "decoration"})
    _assert_group_positions(fig, primary, row_id,
                            {("Day 1", "Control"): 0., ("Day 2", "Drug"): 1.5}, "None")


def _filtered_app():
    import pandas as pd
    import streamlit as st
    from src.widgets.filter_widgets import filters_widget
    from src.widgets.visualization_widgets import visual_encoding_channels_widget
    from src.vis.univar import feature_comparison_plot

    source = pd.DataFrame({"id": ["a1", "a2", "b1", "b2"], "batch": ["A", "A", "B", "B"],
                           "treatment": ["B", "B", "A", "A"], "value": [1., 2., 3., 4.]})
    filtered = filters_widget(source, ["batch", "treatment"])
    color, opacity, shape, separate, subcolor, collapse = visual_encoding_channels_widget(
        filtered, ["batch", "treatment"], separate_by_available=True,
        subcolor_available=True, collapse_available=True)
    fig = feature_comparison_plot(filtered, "id", None, "value", color, opacity_by=opacity,
                                 shape_by=shape, separate_by=separate, subcolor_by=subcolor)
    st.session_state["filtered_pairs"] = sorted(set(zip(filtered.batch, filtered.treatment)))
    st.session_state["group_slots"] = list(fig.layout.xaxis.ticktext)


def test_real_filters_and_encoding_controls_preserve_only_observed_group_slots():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_filtered_app)
    app.session_state["analysis_control_separate_by"] = "batch"
    app.session_state["vis_encoding_color_by"] = ["treatment"]
    app.session_state["batch_multiselect"] = ["A", "B"]
    app.session_state["treatment_multiselect"] = ["A", "B"]
    app.run(timeout=45)
    assert not app.exception, app.exception
    assert app.session_state["filtered_pairs"] == [("A", "B"), ("B", "A")]
    assert app.session_state["group_slots"] == ["B", "A"]
