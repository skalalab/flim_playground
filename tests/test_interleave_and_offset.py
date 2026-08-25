"""Regression guards for two reproducibility/clarity fixes.

1. add_interleaved_points_trace seeded its shuffle only when a caller passed
   random_seed (no caller did), and it seeded the GLOBAL random module. Result:
   plot point draw-order differed run-to-run and the call perturbed global RNG
   state. Fix: a local RNG seeded by default.
2. get_offset's docstring/variable name said "median" but it computes the mean.
   Doc-only fix; behavior (mean of last 10% of bins) must be unchanged.
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
from src.fov_extraction import get_offset


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
    # And it is NOT the median (guards against an accidental np.median "fix").
    assert get_offset(curve) != pytest.approx(np.median(curve[180:]))
