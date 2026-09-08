"""Bivariate exports preserve the app's shuffled point batches and draw order."""

import contextlib
import re
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.dataset_io import check_and_fix_df
from src.export_script import generate_script
from src.vis import bivar


G = "Lifetime fit free_Ch1: G(1st)"
S = "Lifetime fit free_Ch1: S(1st)"
METHODS = ["2D Feature Distribution", "Phasor Plot"]
MARKERS = {"circle": "o", "square": "s", "diamond": "D"}


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(st, "session_state", {"plot_show_group_counts": True})
    monkeypatch.setattr(st, "container", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(st, "columns", lambda spec, **kw: [
        contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))])
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw: value)
    monkeypatch.setattr(bivar, "get_context_theme_color", lambda: "black")


def _source():
    rows = []
    # Input order differs from natural color order. The two large groups yield
    # several unequal batches; tiny groups must still appear exactly once.
    for color_index, (color, count) in enumerate(
        [("color10", 73), ("color2", 41), ("tiny", 3), ("singleton", 1)]
    ):
        for index in range(count):
            position = len(rows)
            x = .08 + (position + 1) * .006
            y = .1 + ((position * 37) % 127) / 127 * .28
            rows.append({
                "id": f"cell{position}", "color": color,
                "day": ["Day 10", "Day 2", None][(index + color_index) % 3],
                "shape": ["round", "square", None][index % 3],
                "opacity": ["low", None, "high"][(index // 3) % 3],
                "x": x, "y": y, G: x, S: y,
            })
    for axis, phasor_axis in [("x", G), ("y", S)]:
        missing = dict(rows[0], id=f"missing-{axis}", color="discarded",
                       day="discarded-day", shape="discarded-shape",
                       opacity="discarded-opacity")
        missing[axis] = missing[phasor_axis] = np.nan
        rows.append(missing)
    return pd.DataFrame(rows)


def _state(method, category, color_by, mixed_encodings):
    params = ({
        "selected_x": "x", "selected_y": "y",
        "marginal_plot_type": "none", "fit_regression": False,
        "fit_gmm_2d": False, "log_x": False, "log_y": False,
        "distribution_category": category,
    } if method == "2D Feature Distribution" else {
        "selected_channel": "Ch1", "phasor_harmonic": 1,
        "phasor_f": .08, "phasor_category": category,
    })
    return {
        "method": method, "csv_filename": "data.csv", "unique_row_id_col": "id",
        "fov_name_col": None, "categorical_cols": ["color", "day", "shape", "opacity"],
        "categorical_filters": {}, "numerical_filters": [],
        "color_by": color_by, "separate_by": "day" if category is not None else None,
        "shape_by": "shape" if mixed_encodings else None,
        "opacity_by": "opacity" if mixed_encodings else None,
        "show_group_counts": True, "point_size": 5, "axis_label_size": 12,
        "legend_size": 10, "colormap": "tab10", "method_params": params,
    }


def _run(tmp_path, monkeypatch, state):
    csv_path = tmp_path / "data.csv"
    _source().to_csv(csv_path, index=False)
    # Replay the same CSV read before asking the real app for expected traces.
    frame = pd.read_csv(csv_path, index_col=False, low_memory=False)
    frame, _, error = check_and_fix_df(
        frame, state["categorical_cols"], state["unique_row_id_col"], None)
    assert not error
    script_path = tmp_path / "analysis.py"
    script_path.write_text(generate_script(state))
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(script_path))
    finally:
        plt.close("all")
    return frame, namespace


def _app_plot(frame, state, *, separated):
    channels = {
        "color_by": state["color_by"], "shape_by": state["shape_by"],
        "opacity_by": state["opacity_by"],
        "separate_by": state["separate_by"] if separated else None,
    }
    if state["method"] == "2D Feature Distribution":
        figure, _, _ = bivar.feature_2d_distribution_plot(
            frame.copy(), "id", None, "x", "y", **channels,
            analysis_options={
                "marginal_plot_type": "none", "fit_regression": False,
                "fit_gmm": False, "log_x": False, "log_y": False,
            })
        if separated:
            bivar.select_distribution_category(
                figure, state["method_params"]["distribution_category"])
    else:
        figure, _ = bivar.phasor_plot(frame.copy(), "id", None, "Ch1", **channels)
        if separated:
            bivar.select_phasor_category(figure, state["method_params"]["phasor_category"])
    return figure


def _app_traces(figure):
    return [trace for trace in figure.data
            if trace.text is not None and trace.visible is not False
            and "markers" in (trace.mode or "")]


def _coordinate_key(point):
    return tuple(np.round(np.asarray(point, float), 12))


def _exported_points(axis, coordinate_ids):
    ordered_ids, styles = [], {}
    # Matplotlib draws artists by zorder, retaining insertion order within a layer.
    for collection in sorted(axis.collections, key=lambda artist: artist.get_zorder()):
        if collection.get_zorder() != 2 or not len(collection.get_offsets()):
            continue
        colors = collection.get_facecolors()
        paths = collection.get_paths()
        for index, point in enumerate(collection.get_offsets()):
            identifier = coordinate_ids[_coordinate_key(point)]
            ordered_ids.append(identifier)
            styles[identifier] = (colors[index % len(colors)], paths[index % len(paths)])
    return ordered_ids, styles


def _plain_label(label):
    return re.sub(r"<[^>]+>", "", label.replace("<br>", "\n"))


@pytest.mark.parametrize("method", METHODS, ids=["2d", "phasor"])
@pytest.mark.parametrize("category", [None, "Day 10", "N/A"],
                         ids=["unseparated", "day10", "missing-category"])
@pytest.mark.parametrize("color_by", [["color"], []], ids=["colors", "no-color"])
@pytest.mark.parametrize("mixed_encodings", [False, True], ids=["plain", "shape-opacity"])
def test_export_preserves_app_point_draw_order_and_global_category_batches(
    tmp_path, monkeypatch, method, category, color_by, mixed_encodings
):
    state = _state(method, category, color_by, mixed_encodings)
    frame, namespace = _run(tmp_path, monkeypatch, state)
    columns = ["x", "y"] if method == "2D Feature Distribution" else [G, S]
    retained = frame.dropna(subset=columns)
    assert len(retained) == 118
    coordinate_ids = {
        _coordinate_key(row[columns]): row["id"] for _, row in retained.iterrows()
    }
    assert len(coordinate_ids) == len(retained), "Coordinates must identify individual rows"

    global_figure = _app_plot(frame, state, separated=False)
    global_traces = _app_traces(global_figure)
    assert len(global_traces) > 2, "Fixture must exercise multiple interleaved batches"
    if mixed_encodings:
        assert any(len(set(trace.marker.symbol)) > 1 for trace in global_traces)
        assert "N/A" in namespace["shape_map"]
        assert namespace["opacity_map"]["N/A"] == .15
    assert "discarded" not in namespace["color_map"]

    if category is None:
        figure, visible = global_figure, retained
    else:
        figure = _app_plot(frame, state, separated=True)
        visible = retained[retained["day"] == category]
        visible_ids = set(visible["id"])
        # Read the global app batches and subset them; never recreate its shuffle.
        subset_batches = [
            [identifier for identifier in trace.text if identifier in visible_ids]
            for trace in global_traces]
        assert [list(trace.text) for trace in _app_traces(figure)] == [
            batch for batch in subset_batches if batch]

    traces = _app_traces(figure)
    expected = [identifier for trace in traces for identifier in trace.text]
    axis = namespace["ax_main"] if method == "2D Feature Distribution" else namespace["ax"]
    drawn, styles = _exported_points(axis, coordinate_ids)
    assert len(drawn) == len(expected) == len(visible)
    assert set(drawn) == set(expected) == set(visible["id"])

    # A mixed-shape collection must retain the correct alpha and marker per row.
    for trace in traces:
        color = np.asarray([float(value) for value in trace.marker.color[5:-1].split(",")])
        symbols = np.broadcast_to(trace.marker.symbol, len(trace.text))
        opacities = np.broadcast_to(trace.marker.opacity, len(trace.text))
        for identifier, symbol, opacity in zip(trace.text, symbols, opacities):
            rgba, path = styles[identifier]
            np.testing.assert_allclose(rgba[:3], color[:3] / 255, rtol=0, atol=1 / 255)
            assert rgba[3] == pytest.approx(color[3] * opacity, abs=1e-12)
            marker = MarkerStyle(MARKERS[symbol])
            expected_path = marker.get_path().transformed(marker.get_transform())
            np.testing.assert_allclose(path.vertices, expected_path.vertices)
            np.testing.assert_array_equal(path.codes, expected_path.codes)

    # Plotly sorts legend items by rank even when the first nonempty category
    # batches arrive in a different order. Point order remains unsorted below.
    legend_traces = sorted((trace for trace in traces if trace.showlegend),
                           key=lambda trace: trace.legendrank)
    expected_labels = [_plain_label(trace.name) for trace in legend_traces]
    assert len(expected_labels) == len({trace.legendgroup for trace in traces})
    legend = axis.get_legend()
    assert legend is not None
    counted_labels = [text.get_text() for text in legend.get_texts() if "\nn=" in text.get_text()]
    assert counted_labels == expected_labels

    if category is not None:
        context_ids = {
            coordinate_ids[_coordinate_key(point)]
            for collection in axis.collections if collection.get_zorder() < 2
            for point in collection.get_offsets()}
        assert context_ids == set(retained["id"]) - set(visible["id"])

    assert drawn == expected, "Export must retain the app's global shuffled batch draw order"
