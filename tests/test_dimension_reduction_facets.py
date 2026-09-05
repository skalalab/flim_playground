"""One embedding, independent encodings, and linked overview/facet panels."""
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.vis import helpers, multivar


@pytest.fixture
def frame():
    return pd.DataFrame({
        "id": [f"cell{i}" for i in range(8)],
        "fov": [f"FOV{i % 2}" for i in range(8)],
        "x": [0., 1., 3., 2., 6., 8., 9., np.nan],
        "y": [3., 2., 5., 1., 4., 8., 2., 0.],
        "type": ["T10", "T2", "T2", "T10", "T2", "T2", None, "removed"],
        "state": ["rest", "active", "rest", "rest", "active", "rest", "active", "removed"],
    })


def plot(frame, **kwargs):
    return multivar.dimension_reduction_plot(
        frame, "id", "fov", ["x", "y"], colored_by=["type", "state"],
        shape_by="type", opacity_by="state", method="PCA", **kwargs)


def foreground(fig, axis=None):
    return [t for t in fig.data if t.text is not None
            and (axis is None or (t.xaxis or "x") == axis)]


def points(fig, axis="x"):
    return {str(i): (x, y, t.marker.color, s, a)
            for t in foreground(fig, axis)
            for i, x, y, s, a in zip(t.text, t.x, t.y, t.marker.symbol, t.marker.opacity)}


@pytest.mark.parametrize("separate", [[], ["type"], ["type", "state"], ["state", "type"]])
def test_facets_reuse_global_coordinates_and_encodings(frame, separate):
    baseline = points(plot(frame))
    fig = plot(frame, separate_by=separate)
    assert points(fig) == baseline
    assert set(baseline) == set(frame.id[:-1])
    repeated = {}
    for axis in {t.xaxis for t in foreground(fig)} - {"x", None}:
        for identifier, value in points(fig, axis).items():
            assert value == baseline[identifier]
            assert identifier not in repeated
            repeated[identifier] = value
    assert repeated == (baseline if separate else {})


def test_matrix_order_empty_intersections_labels_and_linked_ranges(frame):
    fig = plot(frame, separate_by=["type", "state"])
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 7
    # Rows T2, T10, N/A and columns active, rest (natural ordering).
    assert set(points(fig, "x2")) == {"cell1", "cell4"}
    assert set(points(fig, "x3")) == {"cell2", "cell5"}
    assert not points(fig, "x4")
    backgrounds = [t for t in fig.data if t.name == "Other groups"]
    empty_background = next(t for t in backgrounds if t.xaxis == "x4")
    assert len(empty_background.x) == 7
    for t in backgrounds:
        assert t.hoverinfo == "skip" and t.showlegend is False
        assert t.legendgroup is None
    for i in range(2, 8):
        xaxis, yaxis = fig.layout[f"xaxis{i}"], fig.layout[f"yaxis{i}"]
        assert xaxis.matches == "x" and yaxis.matches == "y"
        assert xaxis.range == fig.layout.xaxis.range
        assert yaxis.range == fig.layout.yaxis.range
        assert xaxis.showticklabels is False and yaxis.showticklabels is False
        assert not xaxis.title.text and not yaxis.title.text
        assert (xaxis.domain[1] - xaxis.domain[0]) / (yaxis.domain[1] - yaxis.domain[0]) == pytest.approx(
            (fig.layout.xaxis.domain[1] - fig.layout.xaxis.domain[0]) /
            (fig.layout.yaxis.domain[1] - fig.layout.yaxis.domain[0]))
    labels = {a.text for a in fig.layout.annotations}
    assert {"active", "rest", "T2", "T10", "N/A"} <= labels
    assert "removed" not in labels


def test_counts_are_unique_and_legend_groups_span_panels(frame):
    st.session_state.plot_show_group_counts = True
    try:
        fig = plot(frame, separate_by=["type", "state"])
    finally:
        st.session_state.plot_show_group_counts = False
    color_legend = [t for t in foreground(fig) if t.showlegend]
    assert sum(int(t.name.split("n=")[1].split("<")[0]) for t in color_legend) == 7
    assert len({t.legendgroup for t in color_legend}) == len(color_legend)
    assert all(t.showlegend is False for t in foreground(fig) if t.xaxis != "x")
    assert fig.layout.legend.groupclick == "togglegroup"
    assert not any(a.text.startswith("n=") for a in fig.layout.annotations)


@pytest.mark.parametrize("threshold, renderer", [(49, "scattergl"), (50, "scatter")])
def test_renderer_uses_total_drawn_points_and_context_is_behind(frame, monkeypatch, threshold, renderer):
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", threshold)
    fig = plot(frame, separate_by=["type", "state"])
    drawn = [t for t in fig.data if any(x is not None for x in t.x)]
    assert {t.type for t in drawn} == {renderer}
    assert sum(len(t.x) for t in drawn) == 49
    assert any(t.name == "Other groups" for t in drawn)
    for axis in {t.xaxis for t in drawn}:
        traces = [t for t in drawn if t.xaxis == axis]
        if any(t.name == "Other groups" for t in traces):
            assert traces[0].name == "Other groups"


def test_hover_has_identifiers_and_fov_on_every_foreground_panel(frame):
    fig = plot(frame, separate_by=["type"], row_id_label="Cell <ID>")
    for trace in foreground(fig):
        assert "Cell &lt;ID&gt;" in trace.hovertemplate
        assert "fov" in trace.hovertemplate and "%{customdata}" in trace.hovertemplate
        for identifier, fov in zip(trace.text, trace.customdata):
            assert frame.set_index("id").loc[identifier, "fov"] == fov


def test_display_changes_hit_embedding_cache(frame, monkeypatch):
    multivar.dimension_reduction.clear()
    calls = []
    original = multivar.StandardScaler.fit_transform
    def counted(self, X, *args, **kwargs):
        calls.append(X.copy())
        return original(self, X, *args, **kwargs)
    monkeypatch.setattr(multivar.StandardScaler, "fit_transform", counted)
    plot(frame)
    plot(frame, separate_by=["type"])
    plot(frame, separate_by=["state", "type"])
    assert len(calls) == 1
    assert len(calls[0]) == 7


def test_one_remaining_level_keeps_facet(frame):
    fig = plot(frame[frame.type == "T2"], separate_by=["type"])
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 2


def test_one_selected_feature_always_has_one_column_of_facets(frame):
    fig = plot(frame, separate_by=["type"])
    axes = [fig.layout[f"xaxis{i}"] for i in range(2, 5)]
    assert len({axis.domain for axis in axes}) == 1
    rows = [fig.layout[f"yaxis{i}"].domain for i in range(2, 5)]
    assert rows[0][0] == pytest.approx(rows[1][1])
    assert rows[1][0] == pytest.approx(rows[2][1])
    assert rows[0][1] > rows[1][1] > rows[2][1]


def test_two_features_define_grid_axes_from_their_category_values(frame):
    fig = plot(frame, separate_by=["state", "type"])
    # Reversing the selection makes type's three levels the horizontal axis.
    assert len({fig.layout[f"xaxis{i}"].domain for i in range(2, 8)}) == 3
    assert len({fig.layout[f"yaxis{i}"].domain for i in range(2, 8)}) == 2
    assert set(points(fig, "x2")) == {"cell1", "cell4"}
    assert not points(fig, "x3")
    assert set(points(fig, "x4")) == {"cell6"}


@pytest.mark.parametrize("separate", [[], ["type"], ["type", "state"]])
def test_dimension_reduction_has_black_axis_lines_without_gridlines(frame, separate):
    fig = plot(frame, separate_by=separate)
    # Streamlit's theme adds 8 px of axis padding unless explicitly overridden.
    # With touching facets, that shifts lines into the neighboring point area.
    assert fig.layout.margin.pad == 0
    for name in fig.layout:
        if name.startswith(("xaxis", "yaxis")):
            assert fig.layout[name].showgrid is False
            assert fig.layout[name].zeroline is False
            assert fig.layout[name].showline is True
            assert fig.layout[name].linecolor == "black"
            assert fig.layout[name].linewidth == 1
            assert fig.layout[name].mirror is False


def test_empty_matrix_cells_stay_mounted_with_gray_context(frame):
    fig = plot(frame, separate_by=["type", "state"])
    # Plotly ignores an axis definition unless a trace references its subplot.
    # Gray context retains every empty matrix cell.
    for index in range(2, 8):
        assert any(t.xaxis == f"x{index}" and t.yaxis == f"y{index}" for t in fig.data)
    assert not points(fig, "x4")
    empty_traces = [t for t in fig.data if t.xaxis == "x4"]
    assert len(empty_traces) == 1
    assert empty_traces[0].name == "Other groups"
    assert len(empty_traces[0].x) == 7


@pytest.mark.parametrize("separate", [["type", "state", "fov"], ["type", "type"], "type"])
def test_invalid_separation_is_rejected(frame, separate):
    with pytest.raises(ValueError, match="[Ss]eparat"):
        plot(frame, separate_by=separate)
