"""Phasor category views share coordinates/encodings and fit local populations."""
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


def test_kmeans_fits_each_panel_color_group_and_qualifies_assignments(settings, monkeypatch):
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: True)
    calls = []
    real_fit = bivar.phasor_kmeans
    def capture(coords, n_clusters):
        calls.append(coords.copy())
        return real_fit(coords, n_clusters)
    monkeypatch.setattr(bivar, "phasor_kmeans", capture)
    fig, result = plot(frame(), separate_by="day")
    assert len(calls) == 5
    for (day, group), rows in result.groupby([result.day.fillna("N/A"), "treatment"], sort=False):
        expected = rows[[G, S]].to_numpy()
        assert any(np.array_equal(expected, call) for call in calls)
        labels, _ = real_fit(expected, 2)
        assert rows.k_means_cluster.tolist() == [f"day={day} | {group}_group{i + 1}" for i in labels]
    assert len([t for t in fig.data if t.name == "Centroids"]) == 5
    assert len([t for t in fig.data if t.name == "Centroids" and t.visible]) == 2
    fig.update_xaxes(range=[.2,.7])
    bivar.select_phasor_category(fig, "N/A")
    assert len(calls) == 5  # View changes do not recompute any models.
    assert len([t for t in fig.data if t.name == "Centroids" and t.visible]) == 1
    assert list(fig.layout.xaxis.range) == [.2,.7]


def test_small_and_duplicate_groups_keep_points_without_clustering(settings, monkeypatch):
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: True)
    notices = []
    monkeypatch.setattr(st, "warning", notices.append)
    data = frame().iloc[:3].copy()
    data[[G, S]] = [.3, .2]
    fig, result = plot(data, separate_by="day")
    assert sum(len(t.text) for t in points(fig, "x")) == 3
    assert result.k_means_cluster.isna().all()
    assert len(notices) == 1 and "Day 10" in notices[0] and "ctrl" in notices[0]
    assert not [t for t in fig.data if t.name == "Centroids"]


def test_collinear_hulls_are_finite_and_closed_by_the_renderer():
    polygon = bivar._cluster_hull_polygon(np.array([[.1,.1], [.2,.2], [.3,.3]]))
    assert polygon.shape[1] == 2
    assert np.isfinite(polygon).all()


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


def test_overlays_keep_the_webgl_renderer(settings, monkeypatch):
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: True)
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", 0)
    fig, _ = plot(frame(), separate_by="day")
    overlays = [t for t in fig.data if t.name == "Centroids" or "boundary" in (t.name or "")]
    assert overlays and all(t.type == "scattergl" for t in overlays)


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


def test_explicit_clustering_params_allow_page_to_render_controls_separately(settings, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Clustering widgets must not render while building a prepared figure")
    monkeypatch.setattr(st, "checkbox", unexpected)
    fig, result = plot(frame(), separate_by="day", k_means=True, k_means_clusters=2)
    assert result.k_means_cluster.notna().all()
    assert any(t.name == "Centroids" and t.visible for t in fig.data)


def test_empty_coordinates_raise_clear_error(settings):
    data = frame()
    data[G] = np.nan
    with pytest.raises(ValueError, match="No complete G/S"):
        plot(data, separate_by="day")
