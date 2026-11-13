"""Dimension Reduction computation shared by the app's cached wrapper and exported scripts."""
import sys
import threading
import time
import types
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


def test_parallel_umap_runs_unseeded_on_every_core(monkeypatch):
    # umap-learn falls back to one thread whenever it has a seed.
    calls = []

    class FakeUMAP:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit_transform(self, X):
            return np.zeros((len(X), 2))

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 3)))
    reduce_dimensions(X, "UMAP", {})
    reduce_dimensions(X, "UMAP", {"parallel": True})
    assert [(c["random_state"], c["n_jobs"]) for c in calls] == [(42, 1), (None, -1)]


def test_cached_wrapper_runs_one_reduction_at_a_time_across_sessions(monkeypatch):
    # numba's workqueue layer aborts the process when two threads run UMAP at once,
    # so the lock must hold across sessions, not only within one.
    from src.vis import dimension_reduction as dr

    active, peak, guard = [0], [0], threading.Lock()

    def slow_reduce(*args):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return pd.DataFrame(), None

    class SessionStatePerThread(threading.local):
        # Each thread stands in for a browser session with its own Session State.
        def __contains__(self, key):
            return key in self.__dict__

    monkeypatch.setattr(dr, "reduce_dimensions", slow_reduce)
    monkeypatch.setattr(dr, "st", types.SimpleNamespace(session_state=SessionStatePerThread()))
    threads = [threading.Thread(target=dr.dimension_reduction.__wrapped__,
                                args=(pd.DataFrame({"a": [i]}),)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 1
