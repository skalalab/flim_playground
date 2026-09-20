"""Regression checks against the app for the older standalone export paths."""

import contextlib
import re
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.collapse import collapse_rows
from src.dataset_io import check_and_fix_df
from src.export_script import generate_script
from src.vis import bivar, univar
from src.vis.helpers import apply_plot_styling


G = "Lifetime fit free_Ch1: G(1st)"
S = "Lifetime fit free_Ch1: S(1st)"


@pytest.fixture(autouse=True)
def _headless_widgets(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(st, "session_state", {"plot_show_group_counts": True})
    monkeypatch.setattr(st, "container", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(st, "columns", lambda spec, **kw: [
        contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))])
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw: value)
    monkeypatch.setattr(st, "number_input", lambda label, value=0, **kw: value)
    monkeypatch.setattr(univar, "comparison_overlay_widget", lambda *a: "None")


def _state(method, **overrides):
    state = dict(
        method=method, csv_filename="data.csv", unique_row_id_col="id", fov_name_col=None,
        categorical_cols=["treatment", "shape", "opacity", "dish"],
        color_by=["treatment"], shape_by=None, opacity_by=None, separate_by=None,
        subcolor_by=None, show_group_counts=True, point_size=12, axis_label_size=12,
        legend_size=10, method_params={},
    )
    state.update(overrides)
    return state


def _source():
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(8)],
        "treatment": ["A", "A", "B", "B", "B", "C", "C", "C"],
        "shape": ["lost", "lost", "round", "square", "round", "square", "round", "square"],
        "opacity": ["0", "0", "1", "2", None, "1", "2", None],
        "dish": ["D1", "D2", "D1", "D2", "D1", "D1", "D2", "D1"],
        "x": np.arange(8, dtype=float) + 1,
        "y": [1., 3., 2., 5., 4., 8., 7., 10.],
        G: np.linspace(.2, .8, 8), S: np.linspace(.1, .4, 8),
    })


def _run(tmp_path, monkeypatch, source, state, save=False):
    path = tmp_path / state["csv_filename"]
    source.to_csv(path, index=False)
    # The app and script must read the same bytes, including float rounding.
    app_df = pd.read_csv(path, index_col=False, low_memory=False)
    app_df, _, error = check_and_fix_df(
        app_df, state["categorical_cols"], state["unique_row_id_col"], None)
    assert not error
    script = generate_script(state)
    if save:
        script = script.replace("SAVE_DERIVED_DATA = False", "SAVE_DERIVED_DATA = True")
    script_path = tmp_path / "analysis.py"
    script_path.write_text(script)
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(script_path))
    finally:
        plt.close("all")
    return app_df, namespace


def _app_traces(fig):
    return [trace for trace in fig.data if trace.text is not None
            and "markers" in (getattr(trace, "mode", None) or "")]


def _collections(axis):
    return [collection for collection in axis.collections if len(collection.get_offsets())]


def _phasor_styles(fig, axis):
    app, exported = [], []
    for trace in _app_traces(fig):
        color = np.array([float(value) for value in trace.marker.color[5:-1].split(",")])
        opacity = np.broadcast_to(trace.marker.opacity, len(trace.x))
        for x, y, alpha in zip(trace.x, trace.y, opacity):
            app.append([x, y, *(color[:3] / 255), color[3] * alpha])
    for collection in _collections(axis):
        colors = collection.get_facecolors()
        for i, (x, y) in enumerate(collection.get_offsets()):
            exported.append([x, y, *colors[i % len(colors)]])
    return np.array(sorted(app)), np.array(sorted(exported))


@pytest.mark.parametrize("overlay", ["None", "Boxplot", "SuperPlot"])
def test_comparison_preserves_a_feature_named_like_the_internal_group(
    tmp_path, monkeypatch, overlay
):
    source = _source().rename(columns={"x": "_color_group"})
    source["_color_group_2"] = source["y"] + 100  # A second collision also survives.
    state = _state("Feature Comparison", method_params={
        "selected_var": "_color_group", "overlay": overlay,
        "collapse_by": "dish" if overlay == "SuperPlot" else None,
    })
    app_df, ns = _run(tmp_path, monkeypatch, source, state)
    primary = app_df
    row_id = "id"
    if overlay == "SuperPlot":
        primary, row_id, _ = collapse_rows(app_df, "dish", ["treatment"], "id")
        np.testing.assert_allclose(ns["source_df"]["_color_group"], app_df["_color_group"])
    monkeypatch.setattr(univar, "comparison_overlay_widget", lambda *a: overlay)
    fig = univar.feature_comparison_plot(
        primary, row_id, None, "_color_group", ["treatment"],
        collapse_by=state["method_params"]["collapse_by"], source_df=app_df,
        source_row_id_col="id")
    np.testing.assert_allclose(ns["df"]["_color_group"], primary["_color_group"])
    np.testing.assert_allclose(ns["df"]["_color_group_2"], primary["_color_group_2"])
    app_y = sorted(value for trace in _app_traces(fig) for value in trace.y)
    export_y = sorted(y for collection in _collections(ns["ax"])
                      for _, y in collection.get_offsets())
    np.testing.assert_allclose(export_y, app_y)


@pytest.mark.parametrize("channel", ["shape_by", "opacity_by", "subcolor_by", "separate_by"])
def test_comparison_preserves_a_category_named_like_the_internal_group(
    tmp_path, monkeypatch, channel
):
    source = _source().rename(columns={"shape": "_color_group"})
    state = _state("Feature Comparison", **{channel: "_color_group"},
                   categorical_cols=["treatment", "_color_group", "opacity", "dish"],
                   method_params={"selected_var": "x"})
    app_df, ns = _run(tmp_path, monkeypatch, source, state)
    pd.testing.assert_series_equal(ns["df"]["_color_group"], app_df["_color_group"])
    fig = univar.feature_comparison_plot(app_df, "id", None, "x", ["treatment"],
                                        **{channel: "_color_group"})
    app_points = sorted((x, y) for trace in _app_traces(fig) for x, y in zip(trace.x, trace.y))
    export_points = sorted(tuple(point) for collection in _collections(ns["ax"])
                           for point in collection.get_offsets())
    np.testing.assert_allclose(export_points, app_points)


@pytest.mark.parametrize("channels", [{}, {"shape_by": "shape"}, {"opacity_by": "opacity"}])
def test_phasor_encodings_describe_only_complete_observations(tmp_path, monkeypatch, channels):
    source = _source()
    source.loc[source["treatment"] == "A", G] = np.nan
    state = _state("Phasor Plot", **channels, method_params={
        "selected_channel": "Ch1", "phasor_harmonic": 1, "phasor_f": .08})
    app_df, ns = _run(tmp_path, monkeypatch, source, state)
    fig, _ = bivar.phasor_plot(app_df, "id", None, "Ch1", color_by=["treatment"], **channels)
    assert list(ns["color_map"]) == ["B", "C"]
    assert "lost" not in ns["shape_map"]
    assert "0" not in ns["opacity_map"]
    expected, actual = _phasor_styles(fig, ns["ax"])
    np.testing.assert_allclose(actual[:, :2], expected[:, :2])
    np.testing.assert_allclose(actual[:, 2:5], expected[:, 2:5], atol=1/255)
    np.testing.assert_allclose(actual[:, 5], expected[:, 5])
    assert not any("n=0" in label for label in ns["ax"].get_legend_handles_labels()[1])


@pytest.mark.parametrize("color_by", [[], ["treatment"]])
@pytest.mark.parametrize("shape_by", [None, "shape"])
def test_phasor_opacity_matches_the_apps_effective_alpha(
    tmp_path, monkeypatch, color_by, shape_by
):
    state = _state("Phasor Plot", color_by=color_by, shape_by=shape_by, opacity_by="opacity",
                   method_params={"selected_channel": "Ch1", "phasor_harmonic": 1, "phasor_f": .08})
    app_df, ns = _run(tmp_path, monkeypatch, _source(), state)
    fig, _ = bivar.phasor_plot(app_df, "id", None, "Ch1", color_by=color_by,
                              shape_by=shape_by, opacity_by="opacity")
    expected, actual = _phasor_styles(fig, ns["ax"])
    np.testing.assert_allclose(actual[:, 5], expected[:, 5])


@pytest.mark.parametrize("point_size", [5, 12])
@pytest.mark.parametrize("method,overlay", [
    ("Feature Comparison", "None"), ("Feature Comparison", "Boxplot"),
    ("2D Feature Distribution", "None"), ("Phasor Plot", "None"),
])
def test_ordinary_point_sizes_follow_the_app_diameter_control(
    tmp_path, monkeypatch, point_size, method, overlay
):
    params = dict(selected_var="x", selected_x="x", selected_y="y", overlay=overlay,
                  marginal_plot_type="None", selected_channel="Ch1", phasor_harmonic=1, phasor_f=.08)
    app_df, ns = _run(tmp_path, monkeypatch, _source(),
                      _state(method, point_size=point_size, method_params=params))
    if method == "Feature Comparison":
        monkeypatch.setattr(univar, "comparison_overlay_widget", lambda *a: overlay)
        fig = univar.feature_comparison_plot(app_df, "id", None, "x", ["treatment"])
    elif method == "2D Feature Distribution":
        fig, _, _ = bivar.feature_2d_distribution_plot(
            app_df, "id", None, "x", "y", color_by=["treatment"],
            analysis_options={"marginal_plot_type": "None"})
    else:
        fig, _ = bivar.phasor_plot(app_df, "id", None, "Ch1", color_by=["treatment"])
    fig = apply_plot_styling(fig, point_size, 12, 10)
    assert {trace.marker.size for trace in _app_traces(fig)} == {point_size}
    axis = ns["ax_main"] if method == "2D Feature Distribution" else ns["ax"]
    assert all(collection.get_sizes() == pytest.approx([point_size ** 2])
               for collection in _collections(axis))


def test_unseparated_2d_reports_each_groups_regression_and_gmm_statistics(
    tmp_path, monkeypatch, capsys
):
    rng = np.random.default_rng(23)
    source = pd.DataFrame({
        "id": [f"r{i}" for i in range(120)], "treatment": ["A"] * 60 + ["B"] * 60,
        "x": np.tile(np.repeat([1., 5.], 30), 2) + rng.normal(0, .1, 120),
        "y": np.repeat([2., 8., 5., 15.], 30) + rng.normal(0, .1, 120),
    })
    state = _state("2D Feature Distribution", method_params={
        "selected_x": "x", "selected_y": "y", "fit_regression": True,
        "fit_gmm_2d": True, "gmm_max_components": 2, "gmm_min_weight_threshold": .1,
        "marginal_plot_type": "None"})
    app_df, ns = _run(tmp_path, monkeypatch, source, state, save=True)
    _, _, expected_df = bivar.feature_2d_distribution_plot(
        app_df, "id", None, "x", "y", color_by=["treatment"], analysis_options={
            "fit_regression": True, "fit_gmm": True, "max_components": 2,
            "min_weight_threshold": .1, "marginal_plot_type": "None"})
    pd.testing.assert_series_equal(ns["df"]["2D_GMM_group"], expected_df["2D_GMM_group"])
    output = capsys.readouterr().out
    assert output.count("slope=") == output.count("intercept=") == 2
    # Read expected values from the app's actual shared fit computation.
    app_df["group"] = app_df["treatment"]
    results, _ = bivar.distribution_fit_groups(
        app_df, "x", "y", "group", ["A", "B"], [(None, np.arange(len(app_df)))],
        fit_regression=True, fit_gmm=True, max_components=2, color_by=["treatment"])
    rows = re.findall(r"^\s*\| ([12]) \| ([-\d.]+) \| ([-\d.]+) \| ([-\d.]+) \| ([-\d.]+) \| ([-\d.]+) \|$",
                      output, flags=re.MULTILINE)
    assert len(rows) == 4, output
    for result in results:
        regression = result["regression"]
        assert f"slope={regression['slope']:.4f}" in output
        assert f"intercept={regression['intercept']:.4f}" in output
        assert f"{result['color_group']}: GMM components" in output
    expected = [(component["mean"][0], np.sqrt(component["covariance"][0, 0]),
                 component["mean"][1], np.sqrt(component["covariance"][1, 1]), component["weight"])
                for result in results for component in result["components"]]
    np.testing.assert_allclose(np.array(rows, dtype=float)[:, 1:], expected, atol=.0005)
