"""Feature Comparison box overlays reproduce the app's capped-IQR whiskers."""

import contextlib
import re
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.collapse import collapse_rows
from src.dataset_io import check_and_fix_df
from src.export_script import generate_script
from src.vis import univar


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(st, "session_state", {"plot_show_group_counts": True})
    monkeypatch.setattr(st, "container", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")
    monkeypatch.setattr(univar, "comparison_overlay_widget", lambda *args: "Boxplot")


def _state(*, separated=False, logged=False, collapsed=False, encoded=False, color_by=None):
    return {
        "method": "Feature Comparison", "csv_filename": "data.csv",
        "unique_row_id_col": "id", "fov_name_col": None,
        "categorical_cols": ["treatment", "day", "dish", "shape", "opacity", "keep"],
        "categorical_filters": {"keep": ["yes"]}, "numerical_filters": [],
        "color_by": ["treatment"] if color_by is None else color_by,
        "separate_by": "day" if separated else None,
        "shape_by": "shape" if encoded else None,
        "opacity_by": "opacity" if encoded else None, "subcolor_by": None,
        "show_group_counts": True, "point_size": 8, "axis_label_size": 12,
        "legend_size": 10, "colormap": "tab10",
        "method_params": {
            "selected_var": "value", "overlay": "Boxplot", "add_boxplot": True,
            "collapse_by": "dish" if collapsed else None, "log_y": logged,
            "connect_means": False, "effect_size_method": "None",
            "statistical_test": "None",
        },
    }


def _run(tmp_path, monkeypatch, source, state):
    source.to_csv(tmp_path / "data.csv", index=False)
    frame = pd.read_csv(tmp_path / "data.csv", index_col=False, low_memory=False)
    frame, _, error = check_and_fix_df(frame, state["categorical_cols"], "id", None)
    assert not error
    primary = frame[frame["keep"] == "yes"].dropna(subset=["value"]).copy()
    row_id = "id"
    collapse_by = state["method_params"]["collapse_by"]
    if collapse_by:
        primary, row_id, _ = collapse_rows(
            primary, collapse_by, [*state["color_by"], state["separate_by"]], row_id)
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kwargs:
                        state["method_params"]["log_y"] if label == "Log Y" else value)
    app = univar.feature_comparison_plot(
        primary, row_id, None, "value", state["color_by"],
        separate_by=state["separate_by"], shape_by=state["shape_by"],
        opacity_by=state["opacity_by"], collapse_by=collapse_by)
    script_path = tmp_path / "analysis.py"
    script_path.write_text(generate_script(state))
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(script_path))
    finally:
        plt.close("all")
    return app, namespace, primary


def _assert_box_artists_match_app(app, axis):
    boxes = [trace for trace in app.data if trace.type == "box"]
    patches = [patch for patch in axis.patches if isinstance(patch, PathPatch)]
    assert len(patches) == len(boxes)
    for box in boxes:
        center = float(box.x[0])
        patch, = [patch for patch in patches if np.isclose(
            np.mean([patch.get_path().vertices[:, 0].min(),
                     patch.get_path().vertices[:, 0].max()]), center)]
        vertices = patch.get_path().vertices
        np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()],
                                   [box.q1[0], box.q3[0]])
        assert patch.get_facecolor()[3] == 0
        np.testing.assert_allclose(patch.get_edgecolor()[:3], [0, 0, 0])

        local_lines = [line for line in axis.lines if len(line.get_xdata()) == 2
                       and np.isclose(np.mean(line.get_xdata()), center)]
        assert len(local_lines) == 6  # Two whiskers, two caps, median, and mean.
        width = np.ptp(vertices[:, 0])
        summaries = [line for line in local_lines
                     if np.isclose(np.ptp(line.get_xdata()), width)]
        median, = [line for line in summaries if line.get_linestyle() == "-"]
        mean, = [line for line in summaries if line.get_linestyle() == "--"]
        np.testing.assert_allclose(median.get_ydata(), [box.median[0]] * 2)
        np.testing.assert_allclose(mean.get_ydata(), [box.mean[0]] * 2)

        whiskers = [line for line in local_lines if np.ptp(line.get_xdata()) == 0]
        assert len(whiskers) == 2
        np.testing.assert_allclose(
            sorted(float(line.get_ydata()[1]) for line in whiskers),
            [box.lowerfence[0], box.upperfence[0]],
            err_msg="Exported whiskers must end at the same capped IQR fences as the app")


@pytest.mark.parametrize("values, endpoints", [
    ([0., 1., 2., 3., 100.], [0., 6.]),
    ([0., 97., 98., 99., 100.], [94., 100.]),
    ([5., 5., 5.], [5., 5.]),
    ([7.], [7., 7.]),
], ids=["upper-outlier", "lower-outlier", "constant", "single-observation"])
def test_boxplot_whiskers_use_the_apps_fences_including_degenerate_groups(
    tmp_path, monkeypatch, values, endpoints
):
    source = pd.DataFrame({
        "id": [f"cell{i}" for i in range(len(values))], "value": values,
        "treatment": "only", "day": "Day 2", "dish": "D1",
        "shape": "round", "opacity": "low", "keep": "yes",
    })
    app, namespace, _ = _run(tmp_path, monkeypatch, source, _state(color_by=[]))
    box, = [trace for trace in app.data if trace.type == "box"]
    np.testing.assert_allclose([box.lowerfence[0], box.upperfence[0]], endpoints)
    _assert_box_artists_match_app(app, namespace["ax"])


def _grouped_source():
    rows = []
    for day, shift in [("Day 10", 10.), ("Day 2", 0.)]:
        for treatment, means in [
            ("upper", [1., 2., 3., 4., 1000.]),
            ("lower", [1., 997., 998., 999., 1000.]),
        ]:
            for replicate, mean in enumerate(means):
                for offset in [-.1, .1]:
                    rows.append({
                        "id": f"cell{len(rows)}", "value": shift + mean + offset,
                        "treatment": treatment, "day": day, "dish": f"D{replicate}",
                        "shape": ["round", "square"][replicate % 2],
                        "opacity": ["low", "high"][replicate % 2], "keep": "yes",
                    })
    rows.append(dict(rows[0], id="missing", value=np.nan))
    rows.append(dict(rows[0], id="filtered", value=-100000., keep="no"))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("separated", [False, True], ids=["unseparated", "separated"])
@pytest.mark.parametrize("logged", [False, True], ids=["raw", "logged"])
@pytest.mark.parametrize("collapsed", [False, True], ids=["cells", "replicates"])
def test_grouped_box_whiskers_match_after_filters_logs_and_collapse(
    tmp_path, monkeypatch, separated, logged, collapsed
):
    state = _state(separated=separated, logged=logged, collapsed=collapsed, encoded=True)
    app, namespace, primary = _run(tmp_path, monkeypatch, _grouped_source(), state)
    axis = namespace["ax"]
    assert len(namespace["df"]) == len(primary)
    drawn = [collection for collection in axis.collections if len(collection.get_offsets())]
    assert sum(len(collection.get_offsets()) for collection in drawn) == len(primary)
    app_labels = [re.sub(r"<[^>]+>", "", trace.name.replace("<br>", "\n"))
                  for trace in app.data if trace.text is not None and trace.showlegend]
    legend = axis.get_legend()
    assert legend is not None
    counted = [text.get_text() for text in legend.get_texts() if "\nn=" in text.get_text()]
    assert counted == app_labels
    _assert_box_artists_match_app(app, axis)
