"""Dimension Reduction computation shared by the app's cached wrapper and exported scripts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis.dimension_reduction import dimension_axis_titles, reduce_dimensions


def test_reduce_dimensions_pca_names_components_and_reports_variance():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    X["c"] = 2 * X["a"]
    coordinates, explained_variance = reduce_dimensions(X, "PCA", {})
    assert list(coordinates.columns) == ["PC1", "PC2"] and len(coordinates) == 40
    assert explained_variance.shape == (2,) and 0 < explained_variance.sum() <= 100 + 1e-9
    pd.testing.assert_frame_equal(reduce_dimensions(X, "PCA", {})[0], coordinates)  # seeded
    # Features are standardized first, so rescaling one column leaves the embedding unchanged.
    rescaled, _ = reduce_dimensions(X.assign(b=X["b"] * 1000), "PCA", {})
    np.testing.assert_allclose(rescaled.to_numpy(), coordinates.to_numpy(), atol=1e-8)


def test_dimension_axis_titles_append_pca_variance_only():
    assert dimension_axis_titles(pd.DataFrame(columns=["PC1", "PC2"]), np.array([60.0, 25.5])) == [
        "PC1(60.00%)", "PC2(25.50%)"]
    assert dimension_axis_titles(pd.DataFrame(columns=["UMAP1", "UMAP2"]), None) == ["UMAP1", "UMAP2"]


def test_focus_facet_label_escapes_for_plotly_unless_told_not_to():
    from src.vis.dimension_facets import focus_facet_label

    assert focus_facet_label(["day", "line"], ("D<1", "x")) == "day: D&lt;1 · line: x"
    assert focus_facet_label("day", "D<1", escape=False) == "day: D<1"
