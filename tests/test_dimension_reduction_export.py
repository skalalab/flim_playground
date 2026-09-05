"""Standalone DR exports reproduce one embedding and its membership facets."""
import inspect
import runpy
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.export_script import generate_script


def _frame():
    return pd.DataFrame({
        "id": [f"cell{i}" for i in range(9)],
        "row": ["row10", "row2", "row2", "row10", "row2", None,
                "row99", "row20", "row30"],
        "column": ["col2", "col1", "col2", "col2", "col1", "col1",
                   "col99", "col20", "col30"],
        "color": ["A", "B", "A", "B", "A", "B", "C", "D", "E"],
        "shape": ["shape10", "shape2", "shape10", "shape2", "shape2", "shape2",
                  "shape99", "shape99", "shape99"],
        "opacity": ["day10", "day2", "day2", "day10", None, "day2",
                    "day99", "day99", "day99"],
        "keep": ["yes"] * 8 + ["no"],
        "feature_a": [1., 2., 3., 4., 5., 6., np.nan, -1., 9.],
        "feature_b": [2., 4., 1., 8., 7., 3., 1., 1., 9.],
    })


def _state(separate_by=(), *, counts=True, method="PCA"):
    return {
        "method": "Dimension Reduction", "csv_filename": "data.csv",
        "unique_row_id_col": "id", "fov_name_col": None,
        "categorical_cols": ["row", "column", "color", "shape", "opacity", "keep"],
        "categorical_filters": {"keep": ["yes"]},
        "numerical_filters": [("feature_a", ">", 0)],
        "color_by": ["color"], "shape_by": "shape", "opacity_by": "opacity",
        "separate_by": list(separate_by), "show_group_counts": counts,
        "point_size": 7, "axis_label_size": 12, "legend_size": 10,
        "method_params": {
            "selected_features": ["feature_a", "feature_b"],
            "dr_method": method,
            "hyperParam_dict": {"n_neighbors": 3, "min_dist": .2,
                                "perplexity": 2, "early_exaggeration": 8},
        },
    }


def _run(tmp_path, monkeypatch, state, frame=None):
    (_frame() if frame is None else frame).to_csv(tmp_path / "data.csv", index=False)
    script = generate_script(state)
    path = tmp_path / "analysis.py"
    path.write_text(script)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        namespace = runpy.run_path(str(path))
    finally:
        plt.close("all")
    return namespace


def _points(ax, zorder=2):
    collections = [c for c in ax.collections
                   if len(c.get_offsets()) and c.get_zorder() == zorder]
    return np.vstack([c.get_offsets() for c in collections]) if collections else np.empty((0, 2))


def _rows(points):
    return sorted(tuple(np.round(point, 7)) for point in points)


def _point_styles(ax):
    """Compare rendered RGB, effective alpha, marker path, and size per coordinate."""
    styles = {}
    for collection in ax.collections:
        if collection.get_zorder() != 2:
            continue
        colors = collection.get_facecolors()
        for i, point in enumerate(collection.get_offsets()):
            color = colors[i % len(colors)]
            shape = tuple(collection.get_paths()[i % len(collection.get_paths())].vertices.ravel())
            styles[tuple(np.round(point, 7))] = (tuple(color), shape,
                                                tuple(collection.get_sizes()))
    return styles


def test_unfaceted_export_builds_encodings_only_from_retained_observations(tmp_path, monkeypatch):
    state = _state()
    # Keep NaNs through the numerical filter so the DR-specific removal is exercised.
    state["numerical_filters"] = []
    state["categorical_filters"] = {"row": ["row2", "row10", "row99", "N/A"]}
    namespace = _run(tmp_path, monkeypatch, state)
    assert namespace["color_groups"] == ["A", "B"]
    assert list(namespace["shape_map"]) == ["shape2", "shape10"]
    assert list(namespace["opacity_map"]) == ["day2", "day10", "N/A"]
    assert len(namespace["fig"].axes) == 1
    assert len(_points(namespace["ax"])) == 6
    labels = namespace["ax"].get_legend_handles_labels()[1]
    assert labels == ["A\nn=3", "B\nn=3", "day2", "day10", "N/A", "shape2", "shape10"]


def test_one_separator_stacks_retained_natural_levels_in_one_column(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state(["row"]))
    overview, *facets = namespace["fig"].axes
    assert len(facets) == 3
    df = namespace["df"]
    overview_styles = _point_styles(overview)
    for ax, level in zip(facets, ["row2", "row10", "N/A"]):
        membership = df["row"].eq(level)
        assert _rows(_points(ax)) == _rows(df.loc[membership, ["_dr_x", "_dr_y"]].values)
        assert _rows(_points(ax, 1)) == _rows(df.loc[~membership, ["_dr_x", "_dr_y"]].values)
        for point, style in _point_styles(ax).items():
            assert style[:2] == overview_styles[point][:2]
            assert style[2] == (max(1, namespace["POINT_SIZE"] - 2),)
        assert ax.get_legend_handles_labels()[1] == []
        assert ax.get_legend() is None
        assert ax.get_xlabel() == ax.get_ylabel() == ""
        assert not any(text.get_text().startswith("n=") for text in ax.texts)
        label = next(text for text in ax.texts if text.get_text() == level)
        assert label.get_position()[0] > 1
        assert label.get_position()[1] == .5
        assert label.get_horizontalalignment() == "left"
        assert ax.get_xlim() == overview.get_xlim()
        assert ax.get_ylim() == overview.get_ylim()
        assert ax.get_aspect() == overview.get_aspect() == 1.0
    first, second, third = [ax.get_position() for ax in facets]
    assert first.x0 == pytest.approx(second.x0)
    assert first.x0 == pytest.approx(third.x0)
    assert first.y0 > second.y0 > third.y0
    assert overview.get_position().width > first.width * 1.5
    assert len(list(tmp_path.glob("*.svg"))) == 1
    assert (tmp_path / "dimension_reduction.svg").stat().st_size > 1000


def test_two_separators_form_matrix_including_empty_intersections(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state(["row", "column"]))
    overview, *facets = namespace["fig"].axes
    assert len(facets) == 6
    expected = [(row, col) for row in ["row2", "row10", "N/A"] for col in ["col1", "col2"]]
    df = namespace["df"]
    for ax, (row, col) in zip(facets, expected):
        membership = df["row"].eq(row) & df["column"].eq(col)
        assert _rows(_points(ax)) == _rows(df.loc[membership, ["_dr_x", "_dr_y"]].values)
        assert len(_points(ax, 1)) == len(df) - membership.sum()
        assert not any(text.get_text().startswith("n=") for text in ax.texts)
        assert ax.get_xlim() == overview.get_xlim()
        assert ax.get_ylim() == overview.get_ylim()
    assert len(_points(facets[2])) == len(_points(facets[5])) == 0
    # Column headings occur once above each column, row labels once on its right.
    text_objects = [text for ax in facets for text in ax.texts]
    for label in ["col1", "col2", "row2", "row10", "N/A"]:
        matches = [text for text in text_objects if text.get_text() == label]
        assert len(matches) == 1
        x, y = matches[0].get_position()
        assert y > 1 if label.startswith("col") else x > 1


@pytest.mark.parametrize("separate_by,row_count,column_count", [
    (["row"], 3, 2),
    (["row", "column"], 3, 2),
    (["column", "row"], 3, 2),
    (["row", "column"], 5, 2),
])
def test_rendered_overview_matches_full_grid_height_and_each_maps_aspect(
        tmp_path, monkeypatch, separate_by, row_count, column_count):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame([
        {"id": f"cell_{row}_{column}_{repeat}", "row": f"row{row}",
         "column": f"col{column}", "color": "A" if repeat == 0 else "B",
         "shape": f"shape{repeat}", "opacity": f"day{repeat}", "keep": "yes",
         "feature_a": float(rng.uniform(1, 10)),
         "feature_b": float(rng.uniform(1, 10))}
        for row in range(row_count) for column in range(column_count)
        for repeat in range(2)
    ])
    namespace = _run(tmp_path, monkeypatch, _state(separate_by), frame)
    fig = namespace["fig"]
    fig.canvas.draw()
    overview, *panels = [ax.get_window_extent() for ax in fig.axes]
    assert len(panels) == (row_count if len(separate_by) == 1 else row_count * column_count)
    for panel in panels:
        assert panel.width / panel.height == pytest.approx(overview.width / overview.height)
    grid_columns = 1 if len(separate_by) == 1 else (
        column_count if separate_by[0] == "row" else row_count)
    for index, panel in enumerate(panels):
        if index % grid_columns < grid_columns - 1:
            assert panel.x1 == pytest.approx(panels[index + 1].x0, abs=.01)
        if index + grid_columns < len(panels):
            assert panel.y0 == pytest.approx(panels[index + grid_columns].y1, abs=.01)
    assert overview.y0 == pytest.approx(min(panel.y0 for panel in panels), abs=.01)
    assert overview.y1 == pytest.approx(max(panel.y1 for panel in panels), abs=.01)


@pytest.mark.parametrize("counts", [True, False])
def test_context_remains_visible_with_either_counts_setting(tmp_path, monkeypatch, counts):
    namespace = _run(tmp_path, monkeypatch,
                     _state(["row", "column"], counts=counts))
    overview, *facets = namespace["fig"].axes
    assert len(facets) == 6
    for ax in facets:
        assert len(_points(ax, 1)) > 0
        assert not any(text.get_text().startswith("n=") for text in ax.texts)
    color_labels = overview.get_legend_handles_labels()[1][:2]
    assert all(("n=" in label) is counts for label in color_labels)


def test_legacy_disabled_background_setting_cannot_hide_context(tmp_path, monkeypatch):
    state = _state(["row", "column"])
    state["method_params"]["show_facet_background"] = False
    namespace = _run(tmp_path, monkeypatch, state)
    overview, *facets = namespace["fig"].axes
    assert len(facets) == 6
    assert len(_points(overview, 1)) == 0
    for ax in facets:
        assert len(_points(ax, 1)) == 6 - len(_points(ax))


def test_effective_opacity_is_global_color_alpha_times_opacity_channel(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state(["row"]))
    styles = _point_styles(namespace["ax"])
    for _, row in namespace["df"].iterrows():
        key = tuple(np.round([row["_dr_x"], row["_dr_y"]], 7))
        rgba = styles[key][0]
        expected_alpha = .6 * namespace["opacity_map"][row["opacity"]]
        assert rgba[3] == pytest.approx(expected_alpha)


def test_gray_context_matches_app_styling(tmp_path, monkeypatch):
    namespace = _run(tmp_path, monkeypatch, _state(["row"]))
    for ax in namespace["fig"].axes[1:]:
        context = [collection for collection in ax.collections if collection.get_zorder() == 1]
        assert len(context) == 1
        np.testing.assert_allclose(context[0].get_facecolors()[0], [184 / 255] * 3 + [.25])


def test_all_axes_hide_gridlines_even_when_matplotlib_style_enables_them(tmp_path, monkeypatch):
    monkeypatch.setitem(matplotlib.rcParams, "axes.grid", True)
    namespace = _run(tmp_path, monkeypatch, _state(["row", "column"]))
    for ax in namespace["fig"].axes:
        assert not any(line.get_visible() for line in ax.get_xgridlines())
        assert not any(line.get_visible() for line in ax.get_ygridlines())


@pytest.mark.parametrize("axis_font,legend_font", [(12, 10), (28, 28), (40, 40)])
def test_legend_stays_below_x_title_for_short_grid_and_larger_fonts(
        tmp_path, monkeypatch, axis_font, legend_font):
    frame = _frame().iloc[:6].copy()
    frame["row"] = [f"row{i % 5}" for i in range(6)]
    frame["column"] = [f"col{i % 2}" for i in range(6)]
    state = _state(["column", "row"])
    state.update(axis_label_size=axis_font, legend_size=legend_font)
    namespace = _run(tmp_path, monkeypatch, state, frame)
    fig, ax = namespace["fig"], namespace["ax"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title = ax.xaxis.label.get_window_extent(renderer)
    legend = ax.get_legend().get_window_extent(renderer)
    assert legend.y1 + 8 <= title.y0
    assert legend.y0 >= 0
    overview, *panels = [panel.get_window_extent(renderer) for panel in fig.axes]
    assert overview.y0 == pytest.approx(min(panel.y0 for panel in panels), abs=.01)
    assert overview.y1 == pytest.approx(max(panel.y1 for panel in panels), abs=.01)
    assert all(panel.width / panel.height == pytest.approx(overview.width / overview.height)
               for panel in panels)


@pytest.mark.parametrize("separate_by", [[], ["row", "column"]])
def test_black_left_and_bottom_axes_override_matplotlib_spine_style(tmp_path, monkeypatch, separate_by):
    for side in ("left", "bottom", "right", "top"):
        monkeypatch.setitem(matplotlib.rcParams, f"axes.spines.{side}", side in ("right", "top"))
    monkeypatch.setitem(matplotlib.rcParams, "axes.edgecolor", "red")
    monkeypatch.setitem(matplotlib.rcParams, "axes.linewidth", 3)
    namespace = _run(tmp_path, monkeypatch, _state(separate_by))
    for ax in namespace["fig"].axes:
        for side in ("left", "bottom"):
            assert ax.spines[side].get_visible()
            assert ax.spines[side].get_edgecolor() == (0., 0., 0., 1.)
            assert ax.spines[side].get_linewidth() == 1
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()


def test_one_column_layout_also_works_without_color_shape_or_opacity(tmp_path, monkeypatch):
    state = _state(["row"])
    state.update(color_by=[], shape_by=None, opacity_by=None)
    namespace = _run(tmp_path, monkeypatch, state)
    overview, *facets = namespace["fig"].axes
    assert len(facets) == 3
    assert overview.get_legend_handles_labels()[1] == ["all_data\nn=6"]
    assert all(style[0][3] == .8 for style in _point_styles(overview).values())
    assert facets[0].get_position().y0 > facets[1].get_position().y0 > facets[2].get_position().y0
    assert facets[0].get_position().x0 == pytest.approx(facets[2].get_position().x0)


def test_generated_coordinates_and_groups_do_not_overwrite_uploaded_categories(tmp_path, monkeypatch):
    state = _state(["_dr_x"])
    frame = _frame().rename(columns={"row": "_dr_x", "color": "_color_group", "shape": "_dr_y"})
    state["categorical_cols"] = ["_dr_x", "column", "_color_group", "_dr_y", "opacity", "keep"]
    state["color_by"] = ["_color_group"]
    state["shape_by"] = "_dr_y"
    namespace = _run(tmp_path, monkeypatch, state, frame)
    assert list(namespace["df"]["_dr_x"]) == ["row10", "row2", "row2", "row10", "row2", "N/A"]
    assert list(namespace["df"]["_dr_y"]) == list(frame["_dr_y"][:6])
    assert list(namespace["df"]["_color_group"]) == list(frame["_color_group"][:6])
    assert len(namespace["fig"].axes) == 4


def test_faceted_export_preserves_global_interleaved_app_draw_order(tmp_path, monkeypatch):
    import streamlit as st
    from src.vis import multivar

    frame = pd.concat([_frame().iloc[:6].assign(
        id=lambda df: df["id"] + f"_{repeat}",
        feature_a=lambda df: df["feature_a"] + repeat / 10,
        feature_b=lambda df: df["feature_b"] - repeat / 20,
    ) for repeat in range(4)], ignore_index=True)
    state = _state(["row"])
    namespace = _run(tmp_path, monkeypatch, state, frame)
    monkeypatch.setattr(multivar, "get_context_theme_color", lambda: "black")
    monkeypatch.setitem(st.session_state, "plot_show_group_counts", True)
    app = multivar.dimension_reduction_plot(
        frame, "id", None, ["feature_a", "feature_b"],
        colored_by=["color"], shape_by="shape", opacity_by="opacity",
        method="PCA", separate_by=["row"],
    )
    coordinate_ids = {
        tuple(np.round([row["_dr_x"], row["_dr_y"]], 7)): row["id"]
        for _, row in namespace["df"].iterrows()
    }
    for index, ax in enumerate(namespace["fig"].axes, 1):
        axis = "x" if index == 1 else f"x{index}"
        expected = [str(identifier) for trace in app.data
                    if trace.xaxis == axis and trace.text is not None
                    for identifier in trace.text]
        drawn = [coordinate_ids[tuple(np.round(point, 7))] for point in _points(ax)]
        assert drawn == expected


@pytest.mark.parametrize("separate_by", [[], ["row", "column"]])
def test_all_reducers_fit_once_and_match_the_apps_global_embedding(tmp_path, monkeypatch, separate_by):
    from src.vis.multivar import dimension_reduction
    import sklearn.decomposition
    import sklearn.manifold
    import umap

    retained = _frame().iloc[:6]
    method_sizes = []
    for method in ("PCA", "UMAP", "t-SNE"):
        state = _state(separate_by, method=method)
        expected, _ = dimension_reduction.__wrapped__(
            retained[["feature_a", "feature_b"]], method=method,
            hyperParam_dict=state["method_params"]["hyperParam_dict"])
        reducer_type = {"PCA": sklearn.decomposition.PCA, "UMAP": umap.UMAP,
                        "t-SNE": sklearn.manifold.TSNE}[method]
        fit_transform = reducer_type.fit_transform
        fitted_rows = []

        def track_fit(self, X, *args, **kwargs):
            fitted_rows.append(len(X))
            return fit_transform(self, X, *args, **kwargs)

        with monkeypatch.context() as fit_patch:
            fit_patch.setattr(reducer_type, "fit_transform", track_fit)
            namespace = _run(tmp_path, monkeypatch, state)
        assert fitted_rows == [6]
        fig = namespace["fig"]
        assert len(fig.axes) == (7 if separate_by else 1)
        np.testing.assert_allclose(namespace["X_reduced"], expected.values, atol=1e-6)
        fig.canvas.draw()
        sizes = []
        for ax in fig.axes:
            box = ax.get_window_extent()
            x_span = np.diff(ax.get_xlim())[0]
            y_span = np.diff(ax.get_ylim())[0]
            assert box.height == pytest.approx(.72 * box.width, abs=.01)
            assert y_span == pytest.approx(.72 * x_span)
            assert box.width / x_span == pytest.approx(box.height / y_span)
            sizes.append((box.width, box.height))
        method_sizes.append(sizes)
    for sizes in method_sizes[1:]:
        np.testing.assert_allclose(sizes, method_sizes[0], atol=.01)


def test_export_inlines_the_apps_shared_facet_helpers():
    from src.vis.dimension_facets import (
        dimension_facet_groups, dimension_facet_layout,
        dimension_interleaved_indices, dimension_ranges, normalize_dimension_categories,
    )
    script = generate_script(_state(["row", "column"]))
    for helper in (normalize_dimension_categories, dimension_facet_groups,
                   dimension_ranges, dimension_facet_layout, dimension_interleaved_indices):
        assert textwrap.dedent(inspect.getsource(helper)).strip() in script
    assert "SHOW_FACET_BACKGROUND" not in script
    assert "FACET_COLUMNS" not in script
    assert "SEPARATE_BY = ['row', 'column']" in script
