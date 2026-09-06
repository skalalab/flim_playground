"""Histogram panel populations, numerical preparation, and line styling."""
import inspect

import numpy as np
import pandas as pd
import pytest

from src.vis import univar
from src.vis.helpers import apply_plot_styling
from src.vis.histogram import prepare_histogram as prepare


def source():
    return pd.DataFrame({
        "id": [f"row{i}" for i in range(13)],
        "day": ["Day 10"] * 6 + ["Day 2"] * 5 + [None] * 2,
        "treatment": ["ctrl"] * 6 + ["drug"] * 5 + ["ctrl"] * 2,
        "value": [1., 2., 3., 4., 5., np.nan, 10., 10., 10., 10., 10., 20., 22.],
    }, index=[0] * 13)


def groups(prepared):
    return [group for panel in prepared["panels"] for group in panel["groups"]]


def test_shared_bins_include_each_local_observation_once_in_natural_panels():
    original = source()
    before = original.copy(deep=True)
    data = prepare(original, "value", ["treatment"], separate_by="day", bin_width=2)
    assert [p["category"] for p in data["panels"]] == ["Day 2", "Day 10", "N/A"]
    assert data["color_groups"] == ["ctrl", "drug"]
    assert data["color_counts"] == {"ctrl": 7, "drug": 5}
    assert [g["count"] for g in groups(data)] == [5, 5, 2]
    assert len(data["df"]) == 12
    for group in groups(data):
        expected = np.histogram(group["values"], bins=data["bin_edges"])[0]
        np.testing.assert_array_equal(group["counts"], expected)
        assert sum(group["counts"]) == group["count"]
    assert data["y_range"] == pytest.approx([0, 5.5])
    pd.testing.assert_frame_equal(original, before)


def test_undefined_skewness_and_no_color_by_keep_sparse_observations():
    data = prepare(source(), "value", [], separate_by="day")
    assert data["color_groups"] == ["all_data"]
    assert data["color_counts"] == {"all_data": 12}
    local = groups(data)
    assert np.isnan(local[0]["skewness"])  # constant
    assert local[1]["skewness"] == pytest.approx(0)
    assert np.isnan(local[2]["skewness"])  # two samples
    assert sum(g["count"] for g in local) == 12


def test_count_figure_stacks_local_curves_and_styles_every_axis(monkeypatch):
    assert "separate_by" in inspect.signature(univar.feature_histogram_plot).parameters
    monkeypatch.setattr(univar.st, "session_state", {"plot_show_group_counts": True})
    fig = univar.feature_histogram_plot(source(), "value", ["treatment"], separate_by="day")
    assert len(fig.data) == 3
    assert [trace.xaxis for trace in fig.data] == ["x", "x2", "x3"]
    assert [sum(trace.y) for trace in fig.data] == [5, 5, 2]
    assert fig.layout.height >= 900
    assert fig.data[1].line.color == fig.data[2].line.color
    assert len([trace for trace in fig.data if trace.showlegend]) == 3
    assert [trace.name for trace in fig.data] == [
        "drug (n=5)<br>skew=undefined", "ctrl (n=5)<br>skew=0.000",
        "ctrl (n=2)<br>skew=undefined"]
    assert "n=2" in fig.data[2].hovertemplate and "N/A" in fig.data[2].hovertemplate
    fig = apply_plot_styling(fig, 5, 30, 24)
    assert len(fig.data) == 3  # line legends must not turn into marker ghosts
    for i in range(1, 4):
        x = fig.layout["xaxis" + (str(i) if i > 1 else "")]
        y = fig.layout["yaxis" + (str(i) if i > 1 else "")]
        assert x.range == fig.layout.xaxis.range
        assert y.range == fig.layout.yaxis.range
        assert y.range[0] == 0
        assert x.title.font.size == y.title.font.size == 30
        assert x.tickfont.size == y.tickfont.size == 28
        legend_id = "legend" + (str(i) if i > 1 else "")
        assert fig.data[i - 1].legend == legend_id
        legend = fig.layout[legend_id]
        assert legend.x == 1 and legend.y == pytest.approx(y.domain[1])
        assert (legend.xanchor, legend.yanchor) == ("right", "top")
        assert legend.font.size == 24
    assert all(trace.line.color for trace in fig.data)


def bimodal_source():
    rng = np.random.default_rng(923)
    rows = []
    for day, shift in [("Day 10", 100), ("Day 2", 0), (None, 200)]:
        for value in np.r_[rng.normal(2 + shift, .08, 12), rng.normal(8 + shift, .08, 12)]:
            rows.append({"day": day, "treatment": "ctrl", "value": value})
    return pd.DataFrame(rows, index=["duplicate"] * len(rows))


@pytest.mark.parametrize("gmm", [False, True])
@pytest.mark.parametrize("show_counts", [False, True])
def test_each_panel_legend_has_optional_local_counts_and_skewness_only_in_count_mode(
    monkeypatch, gmm, show_counts
):
    monkeypatch.setattr(univar.st, "session_state", {"plot_show_group_counts": show_counts})
    data = prepare(bimodal_source(), "value", ["treatment"], "day", apply_gmm=gmm,
                   max_components=2)
    fig = apply_plot_styling(univar._histogram_figure(data, "tab10", False), 5, 24, 18)
    for row, panel in enumerate(data["panels"], 1):
        legend_id = "legend" + (str(row) if row > 1 else "")
        traces = [trace for trace in fig.data if trace.legend == legend_id]
        assert len(traces) == (3 if gmm else 1)
        assert all(trace.showlegend for trace in traces)
        expected = "ctrl GMM" if gmm else "ctrl"
        if show_counts:
            expected += " (n=24)"
        if not gmm:
            expected += f"<br>skew={panel['groups'][0]['skewness']:.3f}"
        assert traces[0].name == expected
        if gmm:
            assert all("skew" not in trace.name for trace in traces)
            assert [trace.line.dash for trace in traces[1:]] == ["dash", "dot"]
    assert len({trace.legendgroup for trace in fig.data}) == 3


@pytest.mark.parametrize("separate_by", [None, "day"])
@pytest.mark.parametrize("legend_size", [14, 32])
def test_gmm_legends_reserve_space_outside_aligned_plot_axes(monkeypatch, separate_by, legend_size):
    monkeypatch.setattr(univar.st, "session_state", {"plot_show_group_counts": True})
    data = prepare(bimodal_source(), "value", ["treatment"], separate_by, apply_gmm=True,
                   max_components=2)
    fig = apply_plot_styling(univar._histogram_figure(data, "tab10", False), 5, 32, legend_size)
    for row in range(1, len(data["panels"]) + 1):
        suffix = str(row) if row > 1 else ""
        legend = fig.layout["legend" + suffix]
        xaxis, yaxis = fig.layout["xaxis" + suffix], fig.layout["yaxis" + suffix]
        assert xaxis.domain == fig.layout.xaxis.domain
        assert xaxis.range == fig.layout.xaxis.range
        assert legend.x > xaxis.domain[1]
        assert legend.xanchor == "left"
        assert legend.y == pytest.approx(yaxis.domain[1])
        assert legend.yanchor == "top"
    assert fig.layout.margin.autoexpand is True


@pytest.mark.parametrize("theme", ["black", "white"])
def test_category_titles_use_only_the_value_and_match_the_axes(monkeypatch, theme):
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: theme)
    monkeypatch.setattr("src.vis.helpers.get_context_theme_color", lambda: theme)
    fig = univar.feature_histogram_plot(source(), "value", ["treatment"], separate_by="day")
    fig = apply_plot_styling(fig, 5, 36, 20)
    assert [title.text for title in fig.layout.annotations] == ["Day 2", "Day 10", "N/A"]
    assert all(title.font.color == theme for title in fig.layout.annotations)
    assert fig.layout.xaxis3.title.font.color == theme
    assert fig.layout.yaxis.title.font.color == theme


def test_gmm_fits_each_panel_locally_and_assigns_by_position(monkeypatch):
    from src.vis import histogram
    fitted = []
    original = histogram._find_best_gmm

    def recording(values, **kwargs):
        fitted.append(values.copy())
        return original(values, **kwargs)

    monkeypatch.setattr(histogram, "_find_best_gmm", recording)
    frame = bimodal_source()
    data = prepare(frame, "value", ["treatment"], separate_by="day", apply_gmm=True,
                   max_components=2, intersection_threshold=True)
    assert len(fitted) == 3
    for group, values in zip(groups(data), fitted):
        np.testing.assert_array_equal(group["values"], values)
        assert np.ptp(values) < 7
        means = [component["mean"] for component in group["components"]]
        assert means == sorted(means) and len(means) == 2
        assert means[0] < group["thresholds"][0] < means[1]
        assert group["h_index"] > 0
        assigned = data["df"].iloc[group["positions"]]["GMM_group"].tolist()
        assert assigned[:12] == [group["label"] + "_group1"] * 12
        assert assigned[12:] == [group["label"] + "_group2"] * 12
        assert data["x_range"][0] <= min(group["x"])
        assert data["x_range"][1] >= max(group["x"])
        assert data["y_range"][1] > max(group["pdf"])
    assert "GMM_group" not in frame


def test_gmm_fallback_and_sparse_failures_stay_local(monkeypatch):
    from src.vis import histogram
    original = histogram._find_best_gmm

    def selective_fit(values, **kwargs):
        if values.max() > 200:
            raise ValueError("singular fit")
        return original(values, **kwargs)

    def no_intersection(*args):
        raise ValueError("not bracketed")

    monkeypatch.setattr(histogram, "_find_best_gmm", selective_fit)
    monkeypatch.setattr(histogram, "find_intersection", no_intersection)
    frame = pd.concat([bimodal_source(), source()], ignore_index=True)
    data = prepare(frame, "value", ["treatment"], separate_by="day", apply_gmm=True,
                   max_components=2, intersection_threshold=True)
    by_key = {(g["category"], g["color_group"]): g for g in groups(data)}
    fallback = by_key[("Day 2", "ctrl")]
    assert fallback["thresholds"] is None
    assert any("hard assignment" in notice for notice in fallback["notices"])
    assert data["df"].iloc[fallback["positions"]]["GMM_group"].notna().all()
    sparse = by_key[("Day 2", "drug")]
    assert sparse["count"] == 5 and sparse["gmm"] is None
    assert any("distinct observations" in notice for notice in sparse["notices"])
    failed = by_key[("N/A", "ctrl")]
    assert failed["count"] == 26 and failed["gmm"] is None
    assert any("singular fit" in notice for notice in failed["notices"])
    assert data["df"].iloc[failed["positions"]]["GMM_group"].isna().all()
    assert len(data["df"]) == 84


def test_single_component_curve_stays_visible_and_unassigned(monkeypatch):
    assert "separate_by" in inspect.signature(univar.feature_gmm_plot).parameters
    monkeypatch.setattr(univar, "gmm_hyperParams_widget", lambda: (1, .1))
    monkeypatch.setattr(univar.st, "checkbox", lambda *a, **k: False)
    fig, out = univar.feature_gmm_plot(source(), "value", [], separate_by="day")
    assert len(out) == 12 and out["GMM_group"].isna().all()
    curves = [trace for trace in fig.data if len(trace.x) > 1]
    assert len(curves) == 2
    assert all("GMM" in trace.name for trace in curves)
    assert not any("Component" in trace.name for trace in fig.data)


@pytest.mark.parametrize("separator", [["day"], "absent", "treatment"])
def test_invalid_histogram_separators_are_rejected(separator):
    with pytest.raises(ValueError, match="Separate by"):
        prepare(source(), "value", ["treatment"], separate_by=separator)


@pytest.mark.parametrize("values", [[2.], [2., 2., 2.]])
def test_a_single_bin_has_a_visible_count_marker_after_styling(values):
    data = pd.DataFrame({"value": values})
    fig = univar.feature_histogram_plot(data, "value")
    fig = apply_plot_styling(fig, 5, 24, 18)
    trace = fig.data[0]
    assert list(trace.y) == [len(values)]
    assert trace.mode == "lines+markers"
    assert trace.marker.color == trace.line.color
    assert trace.showlegend
    assert len(fig.data) == 1


@pytest.mark.parametrize("gmm", [False, True])
def test_stacked_panels_align_and_show_only_the_bottom_x_axis(gmm):
    prepared = prepare(source(), "value", ["treatment"], separate_by="day", apply_gmm=gmm)
    fig = apply_plot_styling(univar._histogram_figure(prepared, "tab10", False), 5, 36, 20)
    axes = [fig.layout.xaxis, fig.layout.xaxis2, fig.layout.xaxis3]
    assert all(axis.domain == axes[-1].domain for axis in axes)
    assert all(axis.range == axes[-1].range for axis in axes)
    assert all(axis.matches == "x3" for axis in axes[:-1])
    assert [axis.visible for axis in axes] == [False, False, True]
    assert not fig.layout.xaxis.title.text
    assert not fig.layout.xaxis2.title.text
    assert fig.layout.xaxis3.title.text == "value"


def test_large_fonts_leave_room_between_category_headers_and_previous_axis_ticks():
    fig = univar.feature_histogram_plot(source(), "value", ["treatment"], separate_by="day")
    fig = apply_plot_styling(fig, 5, 36, 20)
    plot_height = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    axes = [fig.layout.yaxis, fig.layout.yaxis2, fig.layout.yaxis3]
    for upper, lower in zip(axes, axes[1:]):
        gap_pixels = (upper.domain[0] - lower.domain[1]) * plot_height
        assert gap_pixels >= 2 * 36 + 16  # title, tick label, and padding
    for axis, title in zip(axes, fig.layout.annotations):
        assert title.y == pytest.approx(axis.domain[1])


def test_styling_an_empty_separated_population_keeps_a_valid_empty_figure():
    data = pd.DataFrame({"value": [np.nan], "day": ["Day 2"]})
    fig = univar.feature_histogram_plot(data, "value", separate_by="day")
    fig = apply_plot_styling(fig, 5, 36, 20)
    assert not fig.data
    assert fig.layout.yaxis.range == (0, 1)


def test_gmm_does_not_allocate_count_bins_or_use_a_saved_count_width(monkeypatch):
    from src.vis import histogram

    def no_count_binning(*a, **k):
        pytest.fail("Density fits must not allocate unused count bins")

    monkeypatch.setattr(histogram, "histogram_bin_edges", no_count_binning)
    data = pd.DataFrame({"value": np.r_[np.linspace(0, 1, 200), 1e7]})
    prepared = histogram.prepare_histogram(data, "value", apply_gmm=True, max_components=1,
                                           bin_width=.1)
    assert prepared["x_range"] == [0, 1e7]
    assert len(prepared["bin_edges"]) == 2
    assert groups(prepared)[0]["gmm"] is not None
    assert groups(prepared)[0]["count"] == 201


def test_separated_threshold_values_stay_in_details_without_overlapping_chart_labels(monkeypatch):
    monkeypatch.setattr(univar, "gmm_hyperParams_widget", lambda: (2, .1))
    monkeypatch.setattr(univar.st, "checkbox", lambda *a, **k: True)
    fig, _ = univar.feature_gmm_plot(bimodal_source(), "value", ["treatment"], separate_by="day")
    assert len(fig.layout.shapes) == 3
    assert [annotation.text for annotation in fig.layout.annotations] == [
        "Day 2", "Day 10", "N/A"]
    assert fig.layout.yaxis.title.text == "Density"
    assert all(panel["groups"][0]["thresholds"] for panel in fig.layout.meta["histogram_summaries"])


def test_nonfinite_gmm_group_reports_local_failure_and_retains_other_categories():
    data = pd.DataFrame({"day": ["good", "good", "bad", "bad"], "value": [1., 2., np.inf, 3.]})
    prepared = prepare(data, "value", separate_by="day", apply_gmm=True, max_components=1)
    local = {group["category"]: group for group in groups(prepared)}
    assert local["good"]["gmm"] is not None
    assert local["bad"]["gmm"] is None
    assert any("GMM fitting failed" in notice for notice in local["bad"]["notices"])
    assert len(prepared["df"]) == 4
    assert np.isfinite(prepared["x_range"]).all()
