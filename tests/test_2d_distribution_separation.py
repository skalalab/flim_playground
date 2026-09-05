"""Category views must display and analyze the same complete populations."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

from src.vis import bivar, helpers


def frame():
    rows = []
    for day, offset in [("Day 10", 10), ("Day 2", 0), (None, 20)]:
        for treatment in ["ctrl", "drug"]:
            if day is None and treatment == "drug":
                continue
            for i in range(12):
                rows.append(dict(id=f"cell{len(rows)}", day=day, treatment=treatment,
                                 dish=f"D{i % 3}", shape=f"s{i % 2}", opacity=f"o{i % 3}",
                                 x=1. + i + offset,
                                 y=2. + (i if day == "Day 2" else -i) + offset
                                 + (i % 3) * .2))
    rows.append(dict(rows[0], id="incomplete", day="Day 99", x=np.nan))
    return pd.DataFrame(rows)


def plot(data, **kwargs):
    return bivar.feature_2d_distribution_plot(
        data, "id", None, "x", "y", color_by=["treatment"],
        shape_by="shape", opacity_by="opacity", separate_by="day",
        analysis_options=dict(log_x=False, log_y=False, marginal_plot_type="gaussian fit",
                              fit_regression=True, fit_gmm=True,
                              max_components=2, min_weight_threshold=.1), **kwargs)


def visible(fig, role):
    return [t for t in fig.data if isinstance(t.meta, dict)
            and t.meta.get("distribution_role") == role and t.visible is not False]


@pytest.fixture(autouse=True)
def settings():
    st.session_state.clear()
    st.session_state.plot_show_group_counts = True
    yield
    st.session_state.clear()


def test_category_switch_changes_points_marginals_and_summaries_without_refitting(monkeypatch):
    data = frame()
    original = data.copy(deep=True)
    calls = []
    real = bivar.pearsonr

    def capture(x, y):
        calls.append(np.column_stack([x, y]))
        return real(x, y)

    monkeypatch.setattr(bivar, "pearsonr", capture)
    fig, text, result = plot(data)
    pd.testing.assert_frame_equal(data, original)
    assert len(calls) == 5
    assert fig.layout.meta["distribution_categories"] == ["Day 2", "Day 10", "N/A"]
    x_range, y_range = tuple(fig.layout.xaxis.range), tuple(fig.layout.yaxis.range)
    for category in ["Day 2", "Day 10", "N/A"]:
        bivar.select_distribution_category(fig, category)
        expected = result[result.day.fillna("N/A") == category]
        assert sorted(i for t in visible(fig, "points") for i in t.text) == sorted(expected.id)
        assert sum(len(t.x) for t in visible(fig, "context")) == len(result) - len(expected)
        assert len(visible(fig, "marginal")) == 2 * expected.treatment.nunique()
        assert len(visible(fig, "regression")) == expected.treatment.nunique()
        summary = fig.layout.meta["distribution_summary"]
        assert summary.count("Pearson r =") == expected.treatment.nunique()
        assert len(calls) == 5
        assert tuple(fig.layout.xaxis.range) == x_range
        assert tuple(fig.layout.yaxis.range) == y_range
    assert "**ctrl:**" in text
    assert len(result) == 60
    assert fig.layout.xaxis.scaleanchor is None
    assert fig.layout.yaxis3.matches == "y"


def test_models_fit_each_category_color_group_and_labels_are_qualified(monkeypatch):
    calls = []
    real = bivar._find_best_gmm

    def capture(values, **kwargs):
        calls.append(np.asarray(values).copy())
        return real(values, **kwargs)

    monkeypatch.setattr(bivar, "_find_best_gmm", capture)
    fig, _, result = plot(frame())
    assert len(calls) == 5
    for (day, treatment), group in result.groupby([result.day.fillna("N/A"), "treatment"]):
        assert any(np.array_equal(group[["x", "y"]].to_numpy(), call) for call in calls)
        labels = group["2D_GMM_group"].dropna()
        assert all(label.startswith(f"day={day} | {treatment}_group") for label in labels)
    bivar.select_distribution_category(fig, "Day 10")
    assert len(calls) == 5


def test_categories_preserve_global_encodings_local_counts_and_gray_context():
    fig, _, _ = plot(frame())
    reference = {}
    for category in fig.layout.meta["distribution_categories"]:
        bivar.select_distribution_category(fig, category)
        for trace in visible(fig, "points"):
            key = trace.legendgroup
            reference.setdefault(key, trace.marker.color)
            assert trace.marker.color == reference[key]
        assert {symbol for trace in visible(fig, "points") for symbol in trace.marker.symbol} == {"circle", "square"}
        styled = helpers.apply_plot_styling(go.Figure(fig), 9, 18, 14)
        for trace in visible(styled, "context"):
            assert trace.marker.color == "#b8b8b8"
            assert trace.marker.opacity == .18
            assert trace.marker.size == 7
            assert trace.hoverinfo == "skip" and not trace.showlegend
        legends = [t.name for t in styled.data if t.showlegend]
        assert any("ctrl" in name and "12" in name for name in legends)
        assert sum("drug" in name for name in legends) == (0 if category == "N/A" else 1)


def test_duplicate_indices_and_ids_do_not_merge_category_memberships():
    data = frame()
    data.index = [0] * len(data)
    data.id = "same"
    fig, _, result = plot(data)
    for category in fig.layout.meta["distribution_categories"]:
        bivar.select_distribution_category(fig, category)
        actual = sorted((x, y) for t in visible(fig, "points") for x, y in zip(t.x, t.y))
        expected = sorted(map(tuple, result.loc[result.day.fillna("N/A") == category, ["x", "y"]].to_numpy()))
        assert actual == expected


def test_constant_group_keeps_points_and_available_marginal_with_explanation():
    data = frame().iloc[:3].copy()
    data.y = 1.
    fig, _, _ = plot(data)
    assert sum(len(t.x) for t in visible(fig, "points")) == 3
    assert not visible(fig, "regression")
    assert len(visible(fig, "marginal")) == 1
    assert "constant" in fig.layout.meta["distribution_summary"].lower()


def test_webgl_overlays_use_foreground_renderer(monkeypatch):
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", 0)
    fig, _, _ = plot(frame())
    assert all(t.type == "scattergl" for t in visible(fig, "points") + visible(fig, "context")
               + visible(fig, "regression") + visible(fig, "fit"))


@pytest.mark.parametrize("separator", [["day"], "absent", "treatment"])
def test_invalid_separator_rejected(separator):
    with pytest.raises(ValueError, match="Separate by"):
        bivar.feature_2d_distribution_plot(frame(), "id", None, "x", "y",
                                           color_by=["treatment"], separate_by=separator,
                                           analysis_options={})
