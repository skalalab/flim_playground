"""Separated 2D exports draw one overview beside a column of highlight maps."""
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Ellipse

from src.export_script import generate_script


def _frame():
    rows = []
    for day, offset in [("Day 10", 10.), ("Day 2", 0.), (None, 20.)]:
        for treatment in ["ctrl", "drug"]:
            for index in range(8):
                rows.append({
                    "id": f"cell{len(rows)}", "day": day, "treatment": treatment,
                    "shape": f"s{index % 2}", "opacity": f"o{index % 2}",
                    "x": 1. + index + offset,
                    "y": 2. + (index if day == "Day 2" else -index) + offset})
    return pd.DataFrame(rows)


def _state(separate_by, *, marginal="gaussian fit", fit=True, focus=None):
    return {
        "method": "2D Feature Distribution", "csv_filename": "data.csv",
        "unique_row_id_col": "id", "fov_name_col": None,
        "categorical_cols": ["day", "treatment", "shape", "opacity"],
        "color_by": ["treatment"], "shape_by": "shape", "opacity_by": "opacity",
        "separate_by": separate_by, "show_group_counts": True,
        "point_size": 7, "axis_label_size": 12, "legend_size": 10,
        "method_params": {
            "selected_x": "x", "selected_y": "y", "log_x": False, "log_y": False,
            "marginal_plot_type": marginal, "fit_regression": fit, "fit_gmm_2d": fit,
            "gmm_max_components": 2, "gmm_min_weight_threshold": .1,
            "facet_focus": focus,
        },
    }


def _run(tmp_path, monkeypatch, state):
    _frame().to_csv(tmp_path / "data.csv", index=False)
    path = tmp_path / "analysis.py"
    path.write_text(generate_script(state))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        return runpy.run_path(str(path))
    finally:
        plt.close("all")


def _points(ax, zorder=2):
    return np.vstack([collection.get_offsets() for collection in ax.collections
                      if len(collection.get_offsets()) and collection.get_zorder() == zorder])


def test_one_panel_per_level_in_natural_order_with_grey_context(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day"))
    assert "DISTRIBUTION_CATEGORY" not in (tmp_path / "analysis.py").read_text()
    frame = namespace["df"]
    levels = [level for level, _positions in namespace["distribution_panels"]]
    assert levels == ["Day 2", "Day 10", "N/A"]
    facet_axes = namespace["facet_axes"]
    assert len(facet_axes) == 3
    for level, panel_ax in zip(levels, facet_axes):
        expected = frame[frame["day"] == level]
        highlighted = _points(panel_ax)
        assert len(highlighted) == len(expected)
        context = _points(panel_ax, zorder=1)
        assert len(context) == len(frame) - len(expected)
        assert np.all(np.isfinite(context))
    overview = _points(namespace["ax_main"])
    assert len(overview) == len(frame)
    assert not [patch for patch in namespace["ax_main"].patches
                if isinstance(patch, Ellipse)]


def test_models_are_drawn_thinner_inside_their_own_panel(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day"))
    levels = [level for level, _positions in namespace["distribution_panels"]]
    for level, panel_ax in zip(levels, namespace["facet_axes"]):
        ellipses = [patch for patch in panel_ax.patches if isinstance(patch, Ellipse)]
        regressions = [line for line in panel_ax.lines if line.get_linestyle() == '--']
        assert ellipses or regressions
        assert all(patch.get_linewidth() == 1 for patch in ellipses)
        assert all(line.get_linewidth() == 1 for line in regressions)
    assert not [line for line in namespace["ax_main"].lines
                if line.get_linestyle() == '--']


def test_every_levels_statistics_are_printed(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, _state("day"))
    printed = capsys.readouterr().out
    for level in ["Day 2", "Day 10", "N/A"]:
        for group in ["ctrl", "drug"]:
            assert f"day={level} | {group}" in printed


def test_marginals_only_reach_the_overview_and_cover_the_whole_dataset(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day"))
    curves = [line for line in namespace["ax_top"].lines if len(line.get_xdata()) > 2]
    assert len(curves) == 2
    # A panel's own dashed regression is also a long line, so density curves are
    # identified by their solid style.
    for panel_ax in namespace["facet_axes"]:
        assert not [line for line in panel_ax.lines
                    if line.get_linestyle() == "-" and len(line.get_xdata()) > 2]


def test_overview_and_the_panel_column_share_edges_and_stay_square(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day", marginal="None", fit=False))
    fig = namespace["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overview = namespace["ax_main"].get_window_extent(renderer)
    panels = [panel_ax.get_window_extent(renderer) for panel_ax in namespace["facet_axes"]]
    assert overview.y0 == pytest.approx(min(panel.y0 for panel in panels), abs=.5)
    assert overview.y1 == pytest.approx(max(panel.y1 for panel in panels), abs=.5)
    assert overview.width / overview.height == pytest.approx(1., abs=.01)
    for panel in panels:
        assert panel.width / panel.height == pytest.approx(1., abs=.01)
    assert namespace["ax_top"] is None
    assert namespace["ax_right"] is None


def test_the_unseparated_export_keeps_its_single_square_axes(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state(None))
    assert namespace["facet_axes"] == []
    assert namespace["ax_top"] is not None
    assert len(_points(namespace["ax_main"])) == len(namespace["df"])
    assert [patch for patch in namespace["ax_main"].patches if isinstance(patch, Ellipse)]
    fig = namespace["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    main = namespace["ax_main"].get_window_extent(renderer)
    assert main.width / main.height == pytest.approx(1., abs=.01)


def _sizes(ax, zorder=2):
    return {collection.get_sizes()[0] for collection in ax.collections
            if len(collection.get_offsets()) and collection.get_zorder() == zorder}


def test_a_promoted_level_takes_the_overview_slot(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day", focus="Day 10"))
    assert "FOCUS_CATEGORY = 'Day 10'" in (tmp_path / "analysis.py").read_text()
    frame, ax_main = namespace["df"], namespace["ax_main"]
    promoted = frame[frame["day"] == "Day 10"]
    assert len(_points(ax_main)) == len(promoted)
    assert len(_points(ax_main, zorder=1)) == len(frame) - len(promoted)
    assert _sizes(ax_main) == {namespace["POINT_SIZE"] ** 2}
    # The overview takes the vacated slot: every cell, no grey, and a new name.
    vacated = namespace["facet_axes"][1]
    assert len(_points(vacated)) == len(frame)
    assert not [c for c in vacated.collections if c.get_zorder() == 1]
    assert _sizes(vacated) == {namespace["PANEL_POINT_SIZE"] ** 2}
    assert [text.get_text() for text in vacated.texts] == ["Main plot"]
    assert [text.get_text() for text in namespace["facet_axes"][0].texts] == ["Day 2"]


def test_the_promotion_augments_the_plot_title(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day", focus="Day 10"))
    plain = _run(tmp_path, monkeypatch, _state("day"))
    assert namespace["_2d_title"] == f"{plain['_2d_title']} (day: Day 10)"
    assert [text.get_text() for text in namespace["fig"].texts] == [namespace["_2d_title"]]
    assert "day:" not in plain["_2d_title"]


def test_the_promoted_levels_strips_describe_it(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day", focus="Day 10"))
    frame = namespace["df"]
    promoted = frame[frame["day"] == "Day 10"]
    curves = namespace["ax_top"].lines
    assert curves
    assert min(line.get_xdata().min() for line in curves) == pytest.approx(promoted.x.min())
    assert max(line.get_xdata().max() for line in curves) == pytest.approx(promoted.x.max())


def _models(axis):
    """Whatever this frame's groups support: ellipses, regression lines, or both."""
    return ([patch for patch in axis.patches if isinstance(patch, Ellipse)]
            + [line for line in axis.lines if line.get_linestyle() == "--"])


def test_models_stay_with_their_own_level_when_one_is_promoted(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state("day", focus="Day 10"))
    # Day 10's models moved with it; the vacated slot shows the overview, which
    # never carried models, and the other panels keep their own.
    assert _models(namespace["ax_main"])
    assert not _models(namespace["facet_axes"][1])
    assert _models(namespace["facet_axes"][0]) and _models(namespace["facet_axes"][2])
