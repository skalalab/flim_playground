"""Compare rendered graph details against the app, beyond plotted values."""

import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.dataset_io import check_and_fix_df
from src.export_script import generate_script
from src.vis import bivar, helpers, multivar, univar
from src.vis.dimension_facets import focus_facet_figure
from src.vis.histogram import prepare_histogram


def _source():
    rng = np.random.default_rng(614)
    rows = []
    for day in ["Day 2", "Day 10"]:
        for group in ["ctrl", "drug"]:
            for i in range(40):
                x = (2 if i < 20 else 8) + rng.normal(0, .15)
                y = .5 * x + rng.normal(0, .2)
                rows.append({
                    "id": f"row{len(rows)}", "day": day, "group": group,
                    "shape": f"shape{i % 2}", "opacity": f"dose{i % 3}",
                    "x": x, "y": y,
                    "Lifetime fit free_Ch1: G(1st)": x / 10,
                    "Lifetime fit free_Ch1: S(1st)": y / 10,
                })
    return pd.DataFrame(rows)


def _case(tmp_path, monkeypatch, method, *, separate=False, focus=False,
          gmm=False, regression=False, constant=False, colormap="tab10",
          marginal="None", source=None, gmm_components=2):
    """Run both renderers on the same parsed CSV and settled display controls."""
    source = _source() if source is None else source.copy()
    if constant:
        source["x"] = 2.
    source.to_csv(tmp_path / "data.csv", index=False)
    frame = pd.read_csv(tmp_path / "data.csv", index_col=False, low_memory=False)
    cats = ["day", "group", "shape", "opacity"]
    frame, _, error = check_and_fix_df(frame, cats, "id", None)
    assert not error
    separator = "day" if separate else None
    state = {
        "method": method, "csv_filename": "data.csv",
        "unique_row_id_col": "id", "fov_name_col": None,
        "categorical_cols": cats, "color_by": ["group"],
        "shape_by": "shape", "opacity_by": "opacity", "separate_by": separator,
        "point_size": 9, "axis_label_size": 18, "legend_size": 11,
        "show_group_counts": True, "colormap": colormap,
    }
    monkeypatch.setitem(st.session_state, "plot_show_group_counts", True)
    monkeypatch.setattr(plt, "show", lambda: None)
    for module in (bivar, helpers, multivar, univar):
        monkeypatch.setattr(module, "get_context_theme_color", lambda: "black")
    if method == "Feature Histogram":
        state.update(shape_by=None, opacity_by=None)
        params = {"selected_var": "x", "apply_gmm": gmm,
                  "intersection_threshold": gmm, "gmm_max_components": gmm_components,
                  "gmm_min_weight_threshold": .1}
        prepared = prepare_histogram(
            frame, "x", ["group"], separator, apply_gmm=gmm,
            max_components=gmm_components, intersection_threshold=gmm)
        app = univar._histogram_figure(prepared, colormap, False)
    elif method == "2D Feature Distribution":
        params = {"selected_x": "x", "selected_y": "y",
                  "marginal_plot_type": marginal, "fit_regression": regression,
                  "fit_gmm_2d": gmm, "gmm_max_components": 2,
                  "gmm_min_weight_threshold": .1,
                  "facet_focus": "Day 10" if focus else None}
        app, _, _ = bivar.feature_2d_distribution_plot(
            frame, "id", None, "x", "y", color_by=["group"],
            shape_by="shape", opacity_by="opacity", colormap=colormap,
            separate_by=separator, analysis_options={
                "marginal_plot_type": marginal, "fit_regression": regression,
                "fit_gmm": gmm, "max_components": 2, "min_weight_threshold": .1})
        focus_facet_figure(app, params["facet_focus"])
    elif method == "Phasor Plot":
        params = {"selected_channel": "Ch1", "phasor_harmonic": 1,
                  "phasor_f": .08, "phasor_category": "Day 10" if separate else None}
        app, _ = bivar.phasor_plot(
            frame, "id", None, "Ch1", color_by=["group"],
            shape_by="shape", opacity_by="opacity", colormap=colormap,
            separate_by=separator)
        bivar.select_phasor_category(app, params["phasor_category"])
    else:
        state["separate_by"] = [separator] if separator else []
        params = {"selected_features": ["x", "y"], "dr_method": "PCA",
                  "facet_focus": ("Day 10",) if focus else None}
        app = multivar.dimension_reduction_plot(
            frame, "id", None, ["x", "y"], colored_by=["group"],
            shape_by="shape", opacity_by="opacity", colormap=colormap,
            method="PCA", separate_by=state["separate_by"])
        focus_facet_figure(app, params["facet_focus"])
    app = helpers.apply_plot_styling(
        app, state["point_size"], state["axis_label_size"], state["legend_size"])
    state["method_params"] = params
    path = tmp_path / "analysis.py"
    path.write_text(generate_script(state), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(path))
    finally:
        plt.close("all")
    return app, namespace


@pytest.mark.parametrize("separate", [False, True])
def test_histogram_component_strokes_match_the_app(tmp_path, monkeypatch, separate):
    app, ns = _case(tmp_path, monkeypatch, "Feature Histogram", gmm=True, separate=separate)
    reference = [trace for trace in app.data if "Component" in trace.name]
    actual = [line for axis in ns["histogram_axes"] for line in axis.lines
              if "Component" in line.get_label()]
    assert reference and len(actual) == len(reference)
    for trace, line in zip(reference, actual):
        assert line.get_linewidth() == trace.line.width
        assert (line.get_alpha() if line.get_alpha() is not None else 1.) == (
            trace.opacity if trace.opacity is not None else 1.)


def test_histogram_threshold_labels_match_the_app(tmp_path, monkeypatch):
    app, ns = _case(tmp_path, monkeypatch, "Feature Histogram", gmm=True)
    assert app.layout.annotations
    assert [text.get_text() for text in ns["ax"].texts] == [
        item.text for item in app.layout.annotations]
    for item, text in zip(app.layout.annotations, ns["ax"].texts):
        assert text.get_position() == pytest.approx((item.x, item.y))
        assert text.get_color() == item.font.color


@pytest.mark.parametrize("separate", [False, True])
def test_histogram_dashes_match_the_plotly_renderer(tmp_path, monkeypatch, separate):
    source = _source()
    source["x"] = (np.tile(np.repeat(np.arange(5) * 5., 8), 4)
                   + np.random.default_rng(16).normal(0, .04, len(source)))
    app, ns = _case(tmp_path, monkeypatch, "Feature Histogram", gmm=True,
                    separate=separate, source=source, gmm_components=5)
    # Plotly.js 3.0.1's dashStyle uses a minimum unit of 3 at widths 1 and 2.
    patterns = {"dash": [9., 9.], "dot": [3., 3.], "dashdot": [9., 3., 3., 3.],
                "longdash": [15., 15.], "longdashdot": [15., 6., 3., 6.]}
    components = [trace for trace in app.data if "Component" in trace.name]
    assert {trace.line.dash for trace in components} == set(patterns)
    lines = [line for axis in ns["histogram_axes"] for line in axis.lines
             if "Component" in line.get_label()]
    assert len(lines) == len(components)
    for trace, line in zip(components, lines):
        assert line._dash_pattern[0] == 0
        assert line._dash_pattern[1] == pytest.approx(patterns[trace.line.dash])
    thresholds = [line for axis in ns["histogram_axes"] for line in axis.lines
                  if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
    assert len(thresholds) == len(app.layout.shapes) > 0
    assert all(line._dash_pattern[1] == pytest.approx(patterns["dash"])
               for line in thresholds)


def test_constant_histogram_marker_size_matches_the_styled_app(tmp_path, monkeypatch):
    app, ns = _case(tmp_path, monkeypatch, "Feature Histogram", constant=True)
    for trace, line in zip(app.data, ns["ax"].lines):
        assert trace.mode == "lines+markers"
        assert line.get_markersize() == trace.marker.size
        assert trace.marker.line.width in (None, 0)
        assert line.get_markeredgewidth() == 0


@pytest.mark.parametrize("separate", [False, True])
def test_2d_regression_line_style_matches_the_app(tmp_path, monkeypatch, separate):
    app, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution",
                    regression=True, separate=separate)
    reference = [trace for trace in app.data if "Regression Line" in (trace.hovertemplate or "")]
    actual = [line for axis in [ns["ax_main"], *ns["facet_axes"]] for line in axis.lines]
    assert reference and len(actual) == len(reference)
    for trace, line in zip(reference, actual):
        np.testing.assert_array_equal(line.get_xdata(), trace.x)
        np.testing.assert_array_equal(line.get_ydata(), trace.y)
        assert trace.line.dash in (None, "solid")
        assert line.get_linestyle() == "-"
        assert line.get_linewidth() == trace.line.width


@pytest.mark.parametrize("separate", [False, True])
def test_2d_gmm_has_the_same_overlays_without_extra_center_markers(tmp_path, monkeypatch, separate):
    app, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution", gmm=True, separate=separate)
    reference = [trace for trace in app.data if trace.mode == "lines"]
    axes = [ns["ax_main"], *ns["facet_axes"]]
    assert reference and sum(len(axis.patches) for axis in axes) == len(reference)
    assert not [line for axis in axes for line in axis.lines if line.get_marker() == "+"]


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("webgl", [False, True])
def test_2d_gmm_dashes_match_the_apps_selected_renderer(tmp_path, monkeypatch, separate, webgl):
    source = _source()
    if webgl:
        source = pd.concat([source] * 32, ignore_index=True)
        source["id"] = [f"row{i}" for i in range(len(source))]
    app, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution", gmm=True,
                    separate=separate, source=source)
    traces = [trace for trace in app.data if trace.mode == "lines"]
    assert traces and {trace.type for trace in traces} == {"scattergl" if webgl else "scatter"}
    ellipses = [patch for axis in [ns["ax_main"], *ns["facet_axes"]] for patch in axis.patches]
    assert len(ellipses) == len(traces)
    for trace, ellipse in zip(traces, ellipses):
        assert trace.line.dash == "dash"
        assert ellipse.get_linewidth() == trace.line.width
        expected = [4 * trace.line.width, trace.line.width] if webgl else [9., 9.]
        assert ellipse._dash_pattern[1] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["2D Feature Distribution", "Dimension Reduction"])
def test_promoted_context_keeps_the_apps_smaller_markers(tmp_path, monkeypatch, method):
    app, ns = _case(tmp_path, monkeypatch, method, separate=True, focus=True)
    sizes = {trace.marker.size for trace in app.data
             if isinstance(trace.meta, dict) and trace.meta.get("facet_role") == "context"
             and trace.xaxis in (None, "x")}
    assert len(sizes) == 1
    ax = ns["ax_main"] if method == "2D Feature Distribution" else ns["ax"]
    context = [collection for collection in ax.collections
               if collection.get_zorder() == 1 and len(collection.get_offsets())]
    assert context
    for collection in context:
        np.testing.assert_array_equal(np.sqrt(collection.get_sizes()), list(sizes))


@pytest.mark.parametrize("separate", [False, True])
def test_phasor_reference_curve_matches_every_app_vertex(tmp_path, monkeypatch, separate):
    app, ns = _case(tmp_path, monkeypatch, "Phasor Plot", separate=separate)
    trace, = [trace for trace in app.data if trace.name == "Curve"]
    curve = ns["ax"].lines[0]
    np.testing.assert_array_equal(curve.get_xdata(), trace.x)
    np.testing.assert_array_equal(curve.get_ydata(), trace.y)


@pytest.mark.parametrize("separate", [False, True])
def test_phasor_reference_marker_diameters_match_the_app(tmp_path, monkeypatch, separate):
    app, ns = _case(tmp_path, monkeypatch, "Phasor Plot", separate=separate)
    trace, = [trace for trace in app.data if trace.name == "Lifetime Markers"]
    markers = [line for line in ns["ax"].lines if line.get_marker() == "o"]
    assert len(markers) == len(trace.x)
    assert {line.get_markersize() for line in markers} == {trace.marker.size}


@pytest.mark.parametrize("separate", [False, True])
def test_phasor_reference_annotations_match_text_and_data_positions(tmp_path, monkeypatch, separate):
    app, ns = _case(tmp_path, monkeypatch, "Phasor Plot", separate=separate)
    actual = {text.get_text(): text for text in ns["ax"].texts}
    for item in app.layout.annotations:
        label = item.text.replace("<br>", "\n")
        assert label in actual
        assert actual[label].get_position() == pytest.approx((item.x, item.y))
        assert actual[label].get_fontsize() == item.font.size


@pytest.mark.parametrize("separate", [False, True])
def test_phasor_hides_numeric_ticks_as_the_app_does(tmp_path, monkeypatch, separate):
    app, ns = _case(tmp_path, monkeypatch, "Phasor Plot", separate=separate)
    assert app.layout.xaxis.showticklabels is False
    assert app.layout.yaxis.showticklabels is False
    assert not any(text.get_visible() for text in
                   [*ns["ax"].get_xticklabels(), *ns["ax"].get_yticklabels()])


@pytest.mark.parametrize("method", ["2D Feature Distribution", "Phasor Plot", "Dimension Reduction"])
def test_scatter_strokes_match_the_apps_borderless_points(tmp_path, monkeypatch, method):
    app, ns = _case(tmp_path, monkeypatch, method)
    points = [trace for trace in app.data if trace.text is not None]
    assert points and all(trace.marker.line.width in (None, 0) for trace in points)
    ax = ns["ax_main"] if method == "2D Feature Distribution" else ns["ax"]
    collections = [collection for collection in ax.collections if len(collection.get_offsets())]
    assert collections
    assert all(np.all(collection.get_linewidths() == 0) for collection in collections)


@pytest.mark.parametrize("method", ["2D Feature Distribution", "Phasor Plot", "Dimension Reduction"])
def test_encoding_legend_markers_match_the_apps_borderless_markers(tmp_path, monkeypatch, method):
    app, ns = _case(tmp_path, monkeypatch, method)
    reference = [trace for trace in app.data
                 if trace.legendgroup in ("opacity_legend", "shape_legend")]
    assert reference and all(trace.marker.line.width in (None, 0) for trace in reference)
    ax = ns["ax_main"] if method == "2D Feature Distribution" else ns["ax"]
    handles = ax.get_legend().legend_handles
    assert len(handles) == len(reference) + len(ns["color_groups"])
    assert all(np.all(handle.get_linewidths() == 0) for handle in handles)


@pytest.mark.parametrize("method", ["Feature Histogram", "2D Feature Distribution", "Phasor Plot", "Dimension Reduction"])
@pytest.mark.parametrize("colormap", ["tab10", "viridis"])
def test_group_colors_match_the_apps_byte_rgb_values_exactly(tmp_path, monkeypatch, method, colormap):
    app, ns = _case(tmp_path, monkeypatch, method, colormap=colormap)
    traces = [trace for trace in app.data if trace.text is not None or method == "Feature Histogram"]
    for trace in traces:
        group = trace.legendgroup.split(":", 1)[1] if method == "Feature Histogram" else trace.legendgroup
        color = trace.line.color if method == "Feature Histogram" else trace.marker.color
        rgb = tuple(float(value) / 255 for value in color[5:-1].split(",")[:3])
        assert ns["color_map"][group][:3] == rgb


def _skewed_source():
    """Data inspected through Plotly.js 3.0.1's actual browser calcdata."""
    values = [0., 1., 2., 3., 4., 5., 6., 8., 13., 40.]
    return pd.DataFrame({
        "id": [f"c{i}" for i in range(20)], "group": ["ctrl"] * 10 + ["drug"] * 10,
        "day": ["Day 2"] * 10 + ["Day 10"] * 10,
        "shape": ["shape0"] * 20, "opacity": ["dose0"] * 20,
        "x": values + [v + 5 for v in values],
        "y": list(reversed(values)) + [v * .4 for v in values],
    })


def test_2d_box_quartiles_match_the_plotly_renderer(tmp_path, monkeypatch):
    _, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution",
                 marginal="boxplot", source=_skewed_source())
    # Browser calcdata: q1/median/q3 = 2/4.5/8 for the first population.
    for axis, value_dimension, expected in [
            (ns["ax_top"], 0, [(2., 8.), (7., 13.)]),
            (ns["ax_right"], 1, [(2., 8.), (.8, 3.2)])]:
        assert len(axis.patches) == len(expected)
        for box, (q1, q3) in zip(axis.patches, expected):
            values = box.get_path().vertices[:, value_dimension]
            assert (min(values), max(values)) == pytest.approx((q1, q3))


@pytest.mark.parametrize("marginal", ["boxplot", "violin"])
def test_2d_marginal_fill_and_outline_match_the_plotly_renderer(tmp_path, monkeypatch, marginal):
    app, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution",
                    marginal=marginal, source=_skewed_source())
    for axis, axis_id in [(ns["ax_top"], "y2"), (ns["ax_right"], "y3")]:
        reference = [trace for trace in app.data if getattr(trace, "yaxis", None) == axis_id]
        bodies = list(axis.patches) if marginal == "boxplot" else [
            collection for collection in axis.collections if isinstance(collection, PolyCollection)]
        assert reference and len(reference) == len(bodies)
        for trace, body in zip(reference, bodies):
            color = trace.marker.color if marginal == "boxplot" else trace.line.color
            rgb = [float(value) / 255 for value in color[5:-1].split(",")[:3]]
            # Resolved Plotly defaults are fill alpha .5, opaque colored line, width 2.
            np.testing.assert_allclose(np.asarray(body.get_facecolor()).reshape(-1, 4)[0], [*rgb, .5])
            np.testing.assert_allclose(np.asarray(body.get_edgecolor()).reshape(-1, 4)[0], [*rgb, 1.])
            np.testing.assert_array_equal(np.atleast_1d(body.get_linewidth()), [2.])


def test_2d_violin_support_and_density_match_the_plotly_renderer(tmp_path, monkeypatch):
    _, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution",
                 marginal="violin", source=_skewed_source())
    body, *_ = [collection for collection in ns["ax_top"].collections
                if isinstance(collection, PolyCollection)]
    vertices = body.get_paths()[0].vertices
    # Browser calcdata: bandwidth 2.9719073141787606, soft span +/- 2 bandwidths,
    # 54 density samples. The body has no median or extrema bars in the app.
    assert (vertices[:, 0].min(), vertices[:, 0].max()) == pytest.approx(
        (-5.943814628357521, 45.94381462835752))
    coords = np.unique(vertices[:, 0])
    assert len(coords) == 54
    # Densities at the first sample and peak, normalized to the renderer's half-width .245.
    assert np.abs(vertices[vertices[:, 0] == coords[0], 1]).max() == pytest.approx(
        .245 * .003283981579148548 / .07960438847918506)
    assert np.abs(vertices[:, 1]).max() == pytest.approx(.245)


def test_2d_violins_draw_only_the_bodies_shown_in_the_app(tmp_path, monkeypatch):
    app, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution",
                    marginal="violin", source=_skewed_source())
    assert all(trace.meanline.visible in (None, False) for trace in app.data
               if trace.type == "violin")
    for axis in (ns["ax_top"], ns["ax_right"]):
        assert len(axis.collections) == 2
        assert all(isinstance(collection, PolyCollection) for collection in axis.collections)


def test_2d_box_outliers_match_the_apps_markers(tmp_path, monkeypatch):
    _, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution",
                 marginal="boxplot", source=_skewed_source())
    for axis in (ns["ax_top"], ns["ax_right"]):
        outliers = [line for line in axis.lines if line.get_marker() not in (None, "None", "")]
        assert outliers
        assert {line.get_marker() for line in outliers} == {"o"}
        assert {line.get_markersize() for line in outliers} == {ns["POINT_SIZE"]}
        assert {line.get_markeredgewidth() for line in outliers} == {0}


def test_2d_gaussian_marginal_strokes_match_the_plotly_renderer(tmp_path, monkeypatch):
    _, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution", marginal="gaussian fit")
    # go.Scatter's resolved default line width is 2; both app density traces use it.
    assert {line.get_linewidth() for axis in (ns["ax_top"], ns["ax_right"])
            for line in axis.lines} == {2}


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("marginal", ["gaussian fit", "boxplot", "violin"])
def test_2d_marginals_touch_the_main_plot_as_in_the_app(tmp_path, monkeypatch, separate, marginal):
    app, ns = _case(tmp_path, monkeypatch, "2D Feature Distribution",
                    separate=separate, marginal=marginal)
    assert app.layout.xaxis.domain[1] == app.layout.xaxis2.domain[0]
    assert app.layout.yaxis.domain[1] == app.layout.yaxis2.domain[0]
    ns["fig"].canvas.draw()
    main = ns["ax_main"].get_position()
    top = ns["ax_top"].get_position()
    right = ns["ax_right"].get_position()
    assert top.y0 == pytest.approx(main.y1)
    assert right.x0 == pytest.approx(main.x1)
    assert (top.x0, top.x1) == pytest.approx((main.x0, main.x1))
    assert (right.y0, right.y1) == pytest.approx((main.y0, main.y1))
