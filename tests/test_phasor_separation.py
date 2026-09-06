"""Phasor category views share coordinates, encodings, and row identities."""
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.vis import bivar, helpers

G = "Lifetime fit free_ch1: G(1st)"
S = "Lifetime fit free_ch1: S(1st)"


def frame():
    rng = np.random.default_rng(42)
    rows = []
    for day in ["Day 10", "Day 2", None]:
        for treatment in ["ctrl", "drug"]:
            if day is None and treatment == "drug":
                continue
            for i in range(12):
                rows.append({"id": f"cell{len(rows)}", "day": day,
                             "treatment": treatment, "shape": f"s{i % 2}",
                             "opacity": f"o{i % 3}",
                             G: .3 + .2 * (i % 2) + rng.normal(0, .015),
                             S: .2 + .1 * (i % 2) + rng.normal(0, .015)})
    rows.append(dict(rows[0], id="incomplete", day="Day 99", **{G: np.nan}))
    return pd.DataFrame(rows)


@pytest.fixture
def settings(monkeypatch):
    st.session_state.clear()
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: False)
    yield
    st.session_state.clear()


def plot(data, **kwargs):
    return bivar.phasor_plot(data, "id", None, "ch1", color_by=["treatment"],
                             shape_by="shape", opacity_by="opacity", **kwargs)


def points(fig, axis="x"):
    return [t for t in fig.data if t.text is not None and t.visible is not False
            and (t.xaxis or "x") == axis]


def context(fig):
    return [t for t in fig.data if isinstance(t.meta, dict)
            and t.meta.get("phasor_role") == "context" and t.visible is not False]


def test_category_views_partition_retained_rows_with_global_encodings(settings):
    data = frame()
    original = data.copy(deep=True)
    fig, result = plot(data, separate_by="day")
    combined, _ = plot(data)
    assert len(result) == 60
    pd.testing.assert_frame_equal(data, original)
    styles = {}
    for trace in points(combined, "x"):
        for i, identity in enumerate(trace.text):
            styles[identity] = (trace.marker.color, trace.marker.symbol[i], trace.marker.opacity[i])
    assert fig.layout.meta["phasor_categories"] == ["Day 2", "Day 10", "N/A"]
    assert fig.layout.meta["phasor_category"] == "Day 2"
    for level in ["Day 2", "Day 10", "N/A"]:
        bivar.select_phasor_category(fig, level)
        traces = points(fig)
        expected = result[result.day.fillna("N/A") == level]
        assert sorted(i for t in traces for i in t.text) == sorted(expected.id)
        for t in traces:
            for i, identity in enumerate(t.text):
                assert (t.marker.color, t.marker.symbol[i], t.marker.opacity[i]) == styles[identity]
        assert not any(a.name == "phasor_category_label" for a in fig.layout.annotations)
    assert "xaxis2" not in fig.layout and "yaxis2" not in fig.layout
    assert list(fig.layout.xaxis.domain) == list(fig.layout.yaxis.domain) == [0, 1]
    assert list(fig.layout.xaxis.range) == [-.05, 1.05]
    assert list(fig.layout.yaxis.range) == [-.05, .55]
    assert fig.layout.xaxis.scaleanchor == "y"
    assert len([t for t in fig.data if t.name == "Lifetime Markers"]) == 1
    assert len([a for a in fig.layout.annotations if a.text == "0.5 ns"]) == 1


def test_background_always_shows_only_other_categories(settings):
    fig, result = plot(frame(), separate_by="day")
    for level in ["Day 2", "Day 10", "N/A"]:
        bivar.select_phasor_category(fig, level)
        expected = sorted(map(tuple, result.loc[result.day.fillna("N/A") != level, [G,S]].to_numpy()))
        assert sorted((x,y) for t in context(fig) for x,y in zip(t.x,t.y)) == expected
        for trace in context(fig):
            assert trace.marker.color == "#b8b8b8" and trace.marker.opacity == .18
            assert trace.marker.symbol == "circle" and not trace.showlegend
            assert trace.hoverinfo == "skip"
        assert sum(len(t.text) for t in points(fig)) == (12 if level == "N/A" else 24)
    # Filtering out the selected level falls back to the first natural value.
    bivar.select_phasor_category(fig, "Day 99")
    assert fig.layout.meta["phasor_category"] == "Day 2"


def test_small_and_duplicate_groups_keep_points(settings):
    data = frame().iloc[:3].copy()
    data[[G, S]] = [.3, .2]
    fig, result = plot(data, separate_by="day")
    assert sum(len(t.text) for t in points(fig, "x")) == 3
    pd.testing.assert_frame_equal(result, data)
    assert not [t for t in fig.data if t.name == "Centroids"]


@pytest.mark.parametrize("separate", [["day"], "absent", "treatment"])
def test_invalid_separation_is_rejected(settings, separate):
    with pytest.raises(ValueError, match="Separate by"):
        plot(frame(), separate_by=separate)


def test_counts_and_legend_follow_active_category(settings):
    import plotly.graph_objects as go
    st.session_state.plot_show_group_counts = True
    fig, _ = plot(frame(), separate_by="day")
    styled = helpers.apply_plot_styling(go.Figure(fig), 5, 16, 14)
    names = [t.name for t in styled.data if t.showlegend]
    assert sum("ctrl" in name for name in names) == 1
    assert sum("drug" in name for name in names) == 1
    assert any("ctrl" in name and "12" in name for name in names)
    assert any("drug" in name and "12" in name for name in names)
    bivar.select_phasor_category(fig, "N/A")
    styled = helpers.apply_plot_styling(fig, 5, 16, 14)
    names = [t.name for t in styled.data if t.showlegend]
    assert any("ctrl" in name and "12" in name for name in names)
    assert not any("drug" in name for name in names)


def test_category_points_keep_the_webgl_renderer(settings, monkeypatch):
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", 0)
    fig, _ = plot(frame(), separate_by="day")
    traces = points(fig) + context(fig)
    assert traces and all(t.type == "scattergl" for t in traces)


@pytest.mark.parametrize("color_by", [None, [], "treatment"])
def test_color_inputs_and_repeated_identifiers_keep_positional_membership(settings, color_by):
    data = frame()
    data.index = [0] * len(data)
    data.id = "repeated ID"
    fig, result = bivar.phasor_plot(data, "id", None, "ch1", color_by=color_by,
                                   separate_by="day")
    assert len(result) == 60
    for level in ["Day 2", "Day 10", "N/A"]:
        bivar.select_phasor_category(fig, level)
        actual = sorted((x, y) for t in points(fig) for x, y in zip(t.x, t.y))
        expected = sorted(map(tuple, result.loc[result.day.fillna("N/A") == level, [G,S]].to_numpy()))
        assert actual == expected


def test_styling_keeps_context_faint_and_legend_on_right(settings):
    fig, _ = plot(frame(), separate_by="day")
    styled = helpers.apply_plot_styling(fig, 9, 24, 12)
    assert styled.layout.xaxis.title.font.size == 24
    assert styled.layout.yaxis.title.font.size == 24
    assert all(t.marker.size == 7 for t in styled.data if t.name == "Lifetime Markers")
    assert all(t.marker.size == 9 for t in points(styled))
    assert all(t.marker.size == 7 and t.marker.opacity == .18 for t in context(styled))
    assert styled.layout.legend.orientation == "v"
    assert styled.layout.legend.x == 1.02 and styled.layout.legend.y == 1


def test_phasor_plot_does_not_render_analysis_controls(settings, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Analysis widgets must not render while building a prepared figure")
    monkeypatch.setattr(st, "checkbox", unexpected)
    fig, result = plot(frame(), separate_by="day")
    assert "k_means_cluster" not in result
    assert not any(t.name == "Centroids" for t in fig.data)


def test_empty_coordinates_raise_clear_error(settings):
    data = frame()
    data[G] = np.nan
    with pytest.raises(ValueError, match="No complete G/S"):
        plot(data, separate_by="day")


@pytest.mark.parametrize("separate_by", [None, "day"])
@pytest.mark.parametrize("color_by", [[], ["treatment"]])
def test_phasor_hover_preserves_row_identity_fov_and_previous_annotations(settings, separate_by, color_by):
    data = frame()
    data.index = [0] * len(data)
    data.id = "repeated ID"
    data["fov"] = [f"image{i}" for i in range(len(data))]
    data["treatment"] = data.treatment.replace("ctrl", "ctrl <A>")
    data["k_means_cluster"] = "previous labels"
    fig, result = bivar.phasor_plot(data, "id", "fov", "ch1", color_by=color_by,
                                   separate_by=separate_by, row_id_label="Cell ID")
    assert result.k_means_cluster.eq("previous labels").all()
    assert "k_means_cluster_2" not in result
    pd.testing.assert_frame_equal(result, data.dropna(subset=[G, S]))
    rows = {(row[G], row[S]): row for _, row in result.iterrows()}
    categories = fig.layout.meta["phasor_categories"] if separate_by else [None]
    for category in categories:
        bivar.select_phasor_category(fig, category)
        for trace in points(fig):
            assert "<b>Cell ID:</b> %{text}" in trace.hovertemplate
            assert "<b>fov:</b> %{customdata}" in trace.hovertemplate
            assert "K-Means" not in trace.hovertemplate
            for x, y, identity, fov in zip(trace.x, trace.y, trace.text, trace.customdata):
                row = rows[(x, y)]
                assert (identity, fov) == (row.id, row.fov)


@pytest.mark.parametrize("separate_by", [None, "day"])
def test_category_named_centroids_keeps_point_styling(settings, separate_by):
    data = frame()
    data.treatment = data.treatment.replace("ctrl", "Centroids")
    fig, _ = plot(data, separate_by=separate_by)
    helpers.apply_plot_styling(fig, 30, 24, 20)
    assert all(t.marker.size == 30 for t in points(fig))
