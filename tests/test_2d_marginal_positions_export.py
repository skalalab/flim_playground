"""2D exports keep each available color marginal on its own categorical position."""

import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.collapse import collapse_rows
from src.dataset_io import check_and_fix_df
from src.export_script import generate_script
from src.vis import bivar


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(st, "session_state", {"plot_show_group_counts": True})
    monkeypatch.setattr(bivar, "get_context_theme_color", lambda: "black")


def _source():
    rows = []
    for day in ["Day 10", "Day 2", None]:
        for group_index, treatment in enumerate(["drug10", "drug2", "flat-x", "flat-y"]):
            if day == "Day 10" and treatment == "drug2":
                continue
            for index in range(6):
                rows.append({
                    "id": f"row{len(rows)}", "day": day, "treatment": treatment,
                    "dish": f"dish{index % 3}", "shape": f"shape{index % 2}",
                    "opacity": ["low", "high", None][index % 3],
                    "x": 10. if treatment == "flat-x" else float(index + 1 + group_index),
                    "y": 20. if treatment == "flat-y" else float(2 * index + 2 + group_index),
                })
    rows.append(dict(rows[0], id="missing-x", treatment="discarded", x=np.nan))
    rows.append(dict(rows[0], id="missing-y", treatment="discarded", y=np.nan))
    return pd.DataFrame(rows)


def _run(tmp_path, monkeypatch, marginal, category, *, collapse=False, logged=False):
    state = {
        "method": "2D Feature Distribution", "csv_filename": "data.csv",
        "unique_row_id_col": "id", "fov_name_col": None,
        "categorical_cols": ["day", "treatment", "dish", "shape", "opacity"],
        "color_by": ["treatment"], "shape_by": "shape", "opacity_by": "opacity",
        "separate_by": "day" if category is not None else None,
        "show_group_counts": True,
        "method_params": {
            "selected_x": "x", "selected_y": "y",
            "marginal_plot_type": marginal, "fit_regression": False, "fit_gmm_2d": False,
            "collapse_by": "dish" if collapse else None,
            "log_x": logged, "log_y": logged,
        },
    }
    _source().to_csv(tmp_path / "data.csv", index=False)
    frame = pd.read_csv(tmp_path / "data.csv", index_col=False, low_memory=False)
    frame, _, error = check_and_fix_df(frame, state["categorical_cols"], "id", None)
    assert not error
    script_path = tmp_path / "analysis.py"
    script_path.write_text(generate_script(state))
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(script_path))
    finally:
        plt.close("all")

    frame = frame.dropna(subset=["x", "y"]).copy()
    row_id, shape, opacity = "id", "shape", "opacity"
    if collapse:
        frame, row_id, varied = collapse_rows(
            frame, "dish", ["treatment", state["separate_by"]], "id")
        shape = None if shape in varied else shape
        opacity = None if opacity in varied else opacity
    app, _, _ = bivar.feature_2d_distribution_plot(
        frame, row_id, None, "x", "y", color_by=["treatment"],
        shape_by=shape, opacity_by=opacity, separate_by=state["separate_by"],
        analysis_options={"marginal_plot_type": marginal, "fit_regression": False,
                          "fit_gmm": False, "log_x": logged, "log_y": logged})
    return app, namespace


def _assert_separate_positions(app, namespace, marginal):
    trace_type = "box" if marginal == "boxplot" else "violin"
    for axis_name, category_dimension, app_axis, app_axis_name in [
        ("ax_top", 1, "yaxis", "y2"), ("ax_right", 0, "xaxis", "x2")
    ]:
        axis = namespace[axis_name]
        # The grid hides nothing: every colour group's marginal describes the
        # whole dataset and lives on the overview's strips.
        traces = [trace for trace in app.data if trace.type == trace_type
                  and getattr(trace, app_axis) == app_axis_name]
        # Plotly gives distinct trace names consecutive category positions when
        # their box/violin coordinate is omitted. The app deliberately uses that layout.
        assert len({trace.name for trace in traces}) == len(traces) >= 2
        bodies = (list(axis.patches) if marginal == "boxplot" else
                  [collection for collection in axis.collections
                   if isinstance(collection, PolyCollection)])
        assert len(bodies) == len(traces)
        intervals = []
        for position, (trace, body) in enumerate(zip(traces, bodies)):
            path = body.get_path() if marginal == "boxplot" else body.get_paths()[0]
            coordinates = path.vertices[:, category_dimension]
            low, high = float(coordinates.min()), float(coordinates.max())
            assert (low + high) / 2 == pytest.approx(position)
            intervals.append((low, high))
            app_color = trace.marker.color if marginal == "boxplot" else trace.line.color
            rgb = np.array([float(value) for value in app_color[5:-1].split(",")][:3]) / 255
            actual_color = np.asarray(body.get_facecolor()).reshape(-1, 4)[0, :3]
            np.testing.assert_allclose(actual_color, rgb, rtol=0, atol=1 / 255)
        assert all(first[1] < second[0] for first, second in zip(intervals, intervals[1:]))
        category_ticks = axis.get_yticklabels() if axis_name == "ax_top" else axis.get_xticklabels()
        assert not any(tick.get_visible() for tick in category_ticks)


@pytest.mark.parametrize("marginal", ["boxplot", "violin"])
@pytest.mark.parametrize("category", [None, "Day 2", "Day 10", "N/A"])
def test_color_marginals_keep_distinct_positions_on_each_axis(
    tmp_path, monkeypatch, marginal, category
):
    app, namespace = _run(tmp_path, monkeypatch, marginal, category)
    _assert_separate_positions(app, namespace, marginal)


@pytest.mark.parametrize("marginal", ["boxplot", "violin"])
@pytest.mark.parametrize("logged", [False, True])
def test_marginal_positions_survive_replicate_collapse_and_log_transforms(
    tmp_path, monkeypatch, marginal, logged
):
    app, namespace = _run(tmp_path, monkeypatch, marginal, "Day 2",
                          collapse=True, logged=logged)
    _assert_separate_positions(app, namespace, marginal)


def test_marginal_none_exports_a_single_full_frame_axes(tmp_path, monkeypatch):
    app, namespace = _run(tmp_path, monkeypatch, "None", None)
    assert namespace["MARGINAL_PLOT_TYPE"] is None
    assert namespace["ax_top"] is None
    assert namespace["ax_right"] is None
    assert "MARGINAL_PLOT_TYPE = None" in (tmp_path / "analysis.py").read_text()
    assert not [trace for trace in app.data
                if getattr(trace, "yaxis", None) in ("y2", "y3")
                or getattr(trace, "xaxis", None) == "x2"]
