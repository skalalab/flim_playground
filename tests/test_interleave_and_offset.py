"""Point interleaving uses a deterministic local RNG without changing global state.
Offset estimation remains the mean of the last 10% of decay bins.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vis.helpers import add_interleaved_points_trace
from src.fov_extraction import get_offset, get_prerise_offset


def _grouped(n=60):
    df = pd.DataFrame({
        "X": np.arange(n, dtype=float),
        "Y": np.arange(n, dtype=float) * 2,
        "id": [f"cell_{i}" for i in range(n)],
        "cd": np.arange(n),
    })
    grouped = [(("A", None, None), df)]
    color_map = {"A": "rgba(0,0,0,0.6)"}
    return grouped, color_map


def _point_order(fig):
    """Flatten the (text) sequence across all scatter traces, in add order."""
    seq = []
    for tr in fig.data:
        if tr.text is not None:
            seq.extend(list(tr.text))
    return seq


def test_interleave_order_is_reproducible():
    grouped, color_map = _grouped()
    orders = []
    for _ in range(2):
        fig = go.Figure()
        add_interleaved_points_trace(fig, grouped, color_map, None, None,
                                     ["X", "Y"], "id", "cd")
        orders.append(_point_order(fig))
    assert orders[0] == orders[1], "interleave order must be reproducible across calls"
    # Sanity: it actually shuffled (not identity order) for 60 points.
    assert orders[0] != [f"cell_{i}" for i in range(60)]


def test_interleave_does_not_touch_global_random_state():
    grouped, color_map = _grouped()
    random.seed(123)
    expected = [random.random() for _ in range(5)]
    random.seed(123)
    add_interleaved_points_trace(go.Figure(), grouped, color_map, None, None,
                                 ["X", "Y"], "id", "cd")
    got = [random.random() for _ in range(5)]
    assert got == expected, "function must not consume/reset the global random stream"


def test_get_offset_is_mean_of_last_10_percent():
    rng = np.random.default_rng(0)
    curve = rng.poisson(50, size=200).astype(float)
    expected = np.mean(curve[int(200 * 0.9):])  # mean of last 10% of bins
    assert get_offset(curve) == pytest.approx(expected)
    # Distinguish the expected mean from the median.
    assert get_offset(curve) != pytest.approx(np.median(curve[180:]))


def test_prerise_offset_is_flimkit_pre_peak_median():
    curve = np.r_[np.full(39, 3.0), [50.0], [9.0, 4.0, 7.0, 6.0, 5.0], [1000.0], np.linspace(500, 20, 80)]
    # Peak at bin 45, so bins 0-39 count: thirty-nine 3s and one 50 (median 3, mean 4.2).
    assert get_prerise_offset(curve) == 3.0
    # Fewer than 5 bins before the peak: median of the last 30 bins.
    early = np.r_[[50.0, 1000.0], np.linspace(500, 10, 60)]
    assert get_prerise_offset(early) == pytest.approx(np.median(early[-30:]))
    # A peak at bin 10 leaves exactly 5 bins (0-4); one bin earlier falls back.
    at_ten = np.r_[np.full(5, 2.0), np.full(5, 7.0), [1000.0], np.linspace(500, 20, 40)]
    assert get_prerise_offset(at_ten) == 2.0
    assert get_prerise_offset(at_ten[1:]) == pytest.approx(np.median(at_ten[-30:]))
    # Never below zero (offset-removed input can have a negative baseline).
    assert get_prerise_offset(np.r_[np.full(20, -2.0), [100.0], np.full(20, 1.0)]) == 0.0
