"""Lifetime reference sizing must not depend on a user-visible trace name."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

from src.vis import bivar, helpers

NAME = "Lifetime Markers"
G = "Lifetime fit free_ch1: G(1st)"
S = "Lifetime fit free_ch1: S(1st)"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    state = {"plot_show_group_counts": False}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(helpers, "get_context_theme_color", lambda: "black")
    monkeypatch.setattr(bivar, "get_context_theme_color", lambda: "black")
    return state


@pytest.mark.parametrize("trace_type", [go.Scatter, go.Scattergl])
def test_same_name_data_and_legend_follow_independent_size_controls(trace_type):
    fig = go.Figure([
        trace_type(x=[1., 2.], y=[3., 4.], text=["cell1", "cell2"],
                   name=NAME, legendgroup=NAME, mode="markers", showlegend=True,
                   marker=dict(size=5, color="red", symbol="square", opacity=.6)),
        trace_type(x=[5.], y=[6.], text=["cell3"],
                   name=NAME, legendgroup=NAME, mode="markers", showlegend=False,
                   marker=dict(size=5, color="red", symbol="square", opacity=.6)),
    ])
    # A second styling pass also exercises resizing an existing legend-only trace.
    for point_size, legend_size in [(12, 20), (14, 22)]:
        helpers.apply_plot_styling(fig, point_size, 16, legend_size)
        data = [trace for trace in fig.data if trace.text is not None]
        assert len(data) == 2
        assert {trace.marker.size for trace in data} == {point_size}
        assert all(trace.name == NAME and not trace.showlegend for trace in data)
        assert [(x, y) for trace in data for x, y in zip(trace.x, trace.y)] == [
            (1., 3.), (2., 4.), (5., 6.)]
        legend, = [trace for trace in fig.data if trace.showlegend]
        assert legend.name == NAME and list(legend.x) == [None]
        assert legend.marker.size == legend_size
        assert (legend.marker.color, legend.marker.symbol, legend.marker.opacity) == ("red", "square", .6)


def test_actual_reference_keeps_its_size_even_if_its_display_name_changes():
    fig = go.Figure()
    bivar._create_phasor_background(fig, "black")
    reference, = [trace for trace in fig.data if trace.mode == "markers"]
    coordinates = np.array([reference.x, reference.y])
    reference.name = "Reference lifetimes"
    copied = go.Figure(fig)
    helpers.apply_plot_styling(copied, 12, 16, 20)
    reference, = [trace for trace in copied.data if trace.mode == "markers"]
    assert reference.marker.size == 7
    assert reference.visible is not False and not reference.showlegend
    np.testing.assert_array_equal([reference.x, reference.y], coordinates)


def _frame():
    rng = np.random.default_rng(42)
    rows = []
    for day in ["Day 10", "Day 2"]:
        for treatment in [NAME, "Other treatment"]:
            for i in range(12):
                rows.append({"id": f"cell{len(rows)}", "day": day, "treatment": treatment,
                             "shape": ["round", "square"][i % 2],
                             "opacity": ["low", "high"][i % 2],
                             G: .3 + rng.normal(0, .015), S: .2 + rng.normal(0, .015)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("webgl", [False, True], ids=["svg", "webgl"])
@pytest.mark.parametrize("separated", [False, True], ids=["combined", "separated"])
@pytest.mark.parametrize("show_counts", [False, True], ids=["counts-hidden", "counts-shown"])
def test_phasor_reference_and_same_name_data_keep_distinct_styles(
    settings, monkeypatch, webgl, separated, show_counts
):
    settings["plot_show_group_counts"] = show_counts
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", 0 if webgl else 100000)
    source = _frame()
    base, result = bivar.phasor_plot(
        source, "id", None, "ch1", color_by=["treatment"], shape_by="shape", opacity_by="opacity",
        separate_by="day" if separated else None)
    pd.testing.assert_frame_equal(result, source)
    categories = ["Day 2", "Day 10"] if separated else [None]
    for category in categories:
        if category is not None:
            bivar.select_phasor_category(base, category)
        styled = helpers.apply_plot_styling(go.Figure(base), 12, 16, 20)
        reference, = [trace for trace in styled.data
                      if trace.name == NAME and trace.text is None and not trace.showlegend]
        assert reference.marker.size == 7
        assert reference.visible is not False
        points = [trace for trace in styled.data if trace.text is not None and trace.visible is not False]
        assert points and all(trace.marker.size == 12 for trace in points)
        assert {trace.type for trace in points} == {"scattergl" if webgl else "scatter"}
        expected = source if category is None else source[source.day == category]
        actual_points = sorted((str(identity), x, y) for trace in points
                               for identity, x, y in zip(trace.text, trace.x, trace.y))
        assert actual_points == sorted(zip(expected.id, expected[G], expected[S]))
        collision_legend, = [trace for trace in styled.data if trace.showlegend and trace.legendgroup == NAME]
        assert collision_legend.marker.size == 20
        assert list(collision_legend.x) == [None]
        assert ("<br>" in collision_legend.name) == show_counts
        if show_counts:
            assert str(sum(expected.treatment == NAME)) in collision_legend.name
