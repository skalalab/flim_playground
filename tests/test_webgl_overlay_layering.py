"""Overlays drawn after the points must use the points' own renderer.

Plotly composites WebGL and SVG by DOM layer, not by trace order. Every SVG cartesian
trace lands in the first ``<svg class="main-svg">``; the whole WebGL canvas sits in
``div.gl-container``, which the DOM places *after* that svg. So once a figure crosses
``WEBGL_POINT_THRESHOLD`` and the points become ``go.Scattergl``, any SVG trace drawn
after them on the same axes is painted underneath -- silently, with no error and no
warning. ``zorder`` cannot rescue it: it only reorders traces within the SVG layer.

This bit five overlays at once -- the box plot, the mean connector, the 2D regression
line, the 2D GMM ellipses and the phasor k-means hulls -- every one of them behind an
optional checkbox that the plot functions' defaults leave off, so nothing else in the
suite renders them. This module turns those checkboxes on, forces WebGL, and fails if an
overlay ever lands in the SVG layer after the points again.

The single deliberate exception is ``feature_comparison_plot``'s ``go.Box``: Plotly has
no WebGL box trace, so the trace is kept (buried) for its hover statistics and legend
entry, and its outline is redrawn as ``layer="above"`` shapes, which are the one thing
that paints on top of the canvas.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis import helpers
from src.vis.bivar import feature_2d_distribution_plot, phasor_plot
from src.vis.multivar import dimension_reduction_plot
from src.vis.univar import feature_comparison_plot

# Every checkbox that gates an overlay drawn after the points. Enabled by label rather
# than by turning all checkboxes on, so a future "Log Y" (which changes the data, not the
# layering) cannot silently alter what this module measures.
OVERLAY_TOGGLES = {
    "Add boxplot",
    "Connect means",
    "2D Gaussian Mixture Model",
    "Regression line",
    "Perform K-Means clustering",
}

PHASOR_CHANNEL = "nadh"
_PHASOR_PREFIX = f"Lifetime fit free_{PHASOR_CHANNEL}: "
G_COL = f"{_PHASOR_PREFIX}G(1st)"
S_COL = f"{_PHASOR_PREFIX}S(1st)"


@pytest.fixture
def webgl(monkeypatch):
    """Force the WebGL renderer and switch every overlay on.

    The threshold is patched to 0 rather than feeding in 5000+ synthetic rows: the bug is
    about which renderer each trace uses, not about point count, and a tiny frame keeps
    the GMM and k-means fits fast and deterministic.
    """
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", 0)

    real_checkbox = st.checkbox
    monkeypatch.setattr(
        st, "checkbox",
        lambda label, *a, **k: True if label in OVERLAY_TOGGLES else real_checkbox(label, *a, **k),
    )
    for key, value in (("plot_point_size", 5), ("plot_axis_label_size", 18),
                       ("plot_legend_size", 16)):
        st.session_state[key] = value


def _blobs_df():
    """Two colour groups, each bimodal so the GMM finds more than one component.

    k-means, the GMM and the sina KDE all need real spread; two separated blobs per group
    give every fit something to find without depending on a real dataset.
    """
    rng = np.random.default_rng(0)
    rows = []
    for group in ("A", "B"):
        offset = 0.0 if group == "A" else 6.0
        for blob in (0.0, 4.0):
            for i in range(40):
                x = rng.normal(10.0 + offset + blob, 0.8)
                y = rng.normal(20.0 + offset + blob * 1.5, 0.8)
                rows.append({
                    "cell_id": f"{group}_{blob}_{i}",
                    "image_name": f"fov_{i % 3}",
                    "group": group,
                    "feat_x": x,
                    "feat_y": y,
                    # Phasor coordinates must sit inside the universal semicircle.
                    G_COL: float(np.clip(rng.normal(0.4 + blob * 0.03, 0.02), 0.05, 0.95)),
                    S_COL: float(np.clip(rng.normal(0.35 + blob * 0.02, 0.02), 0.05, 0.49)),
                })
    return pd.DataFrame(rows)


def _figure(result):
    """The plot functions return either a Figure or a tuple containing one."""
    if isinstance(result, tuple):
        return next(item for item in result if isinstance(item, go.Figure))
    return result


def _svg_overlays_after_points(fig):
    """SVG traces on the MAIN axes drawn after the first WebGL trace.

    Traces on a secondary axis (the 2D plot's ``x2``/``y2`` marginal densities) are
    excluded: they occupy their own strips and never overlap the point cloud, so being in
    the SVG layer costs them nothing.
    """
    first_gl = next((i for i, t in enumerate(fig.data) if t.type == "scattergl"), None)
    assert first_gl is not None, "expected the points to render as WebGL"
    return [
        (i, t.type, t.name)
        for i, t in enumerate(fig.data)
        if i > first_gl
        and t.type != "scattergl"
        and getattr(t, "xaxis", None) in (None, "x")
        and getattr(t, "yaxis", None) in (None, "y")
    ]


def _shapes_above(fig):
    return [s for s in (fig.layout.shapes or []) if s.layer == "above"]


def _run_feature_comparison(df):
    return _figure(feature_comparison_plot(
        df, unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_var="feat_x", color_by=["group"],
    ))


def _run_2d(df):
    return _figure(feature_2d_distribution_plot(
        df, unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_x="feat_x", selected_y="feat_y", color_by=["group"],
    ))


def _run_phasor(df):
    return _figure(phasor_plot(
        df, unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_channel=PHASOR_CHANNEL, color_by=["group"],
    ))


def _run_dimension_reduction(df):
    return _figure(dimension_reduction_plot(
        df, unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_features=["feat_x", "feat_y", G_COL], colored_by=["group"], method="PCA",
    ))


def test_feature_comparison_box_outline_is_drawn_above_the_canvas(webgl):
    """The go.Box stays buried by design; the shapes carry the visible outline."""
    fig = _run_feature_comparison(_blobs_df())

    leftover = _svg_overlays_after_points(fig)
    assert all(kind == "box" for _, kind, _ in leftover), leftover

    # 7 shapes per box (body, median, mean, 2 whiskers, 2 caps), one box per colour group.
    boxes = [t for t in fig.data if t.type == "box"]
    assert boxes, "the boxplot toggle did not produce any box trace"
    assert len(_shapes_above(fig)) == 7 * len(boxes)


def test_feature_comparison_mean_connector_uses_the_point_renderer(webgl):
    fig = _run_feature_comparison(_blobs_df())

    means = [t for t in fig.data if str(getattr(t, "name", "")).startswith("Mean")]
    assert means, "the connect-means toggle did not produce a mean trace"
    assert all(t.type == "scattergl" for t in means)


def test_two_d_distribution_overlays_use_the_point_renderer(webgl):
    """Covers both the regression line and the dashed GMM ellipses."""
    fig = _run_2d(_blobs_df())

    assert _svg_overlays_after_points(fig) == []
    ellipses = [t for t in fig.data if "GMM" in str(getattr(t, "name", ""))]
    assert ellipses, "the GMM toggle did not produce any ellipse trace"
    assert all(t.type == "scattergl" for t in ellipses)


def test_phasor_kmeans_overlays_use_the_point_renderer(webgl):
    """The semicircle is drawn BEFORE the points, so only the hulls are at risk."""
    fig = _run_phasor(_blobs_df())

    assert _svg_overlays_after_points(fig) == []


def test_dimension_reduction_has_no_overlay_after_the_points(webgl):
    fig = _run_dimension_reduction(_blobs_df())

    assert _svg_overlays_after_points(fig) == []


def test_below_the_threshold_nothing_changes(monkeypatch):
    """Under the threshold every trace stays SVG and no overlay shapes are added.

    The shapes exist only to escape the WebGL canvas; adding them in SVG mode would
    double-draw each box outline.
    """
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", 10**9)
    real_checkbox = st.checkbox
    monkeypatch.setattr(
        st, "checkbox",
        lambda label, *a, **k: True if label in OVERLAY_TOGGLES else real_checkbox(label, *a, **k),
    )
    for key, value in (("plot_point_size", 5), ("plot_axis_label_size", 18),
                       ("plot_legend_size", 16)):
        st.session_state[key] = value

    fig = _run_feature_comparison(_blobs_df())

    assert not any(t.type == "scattergl" for t in fig.data)
    assert _shapes_above(fig) == []
    means = [t for t in fig.data if str(getattr(t, "name", "")).startswith("Mean")]
    assert means and all(t.type == "scatter" for t in means)
    # Above the points (zorder=1), below the boxes (zorder=10).
    assert all(t.zorder == 2 for t in means)
