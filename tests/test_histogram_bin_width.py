"""histogram_bin_width_widget returns valid bin edges for constant features,
including numpy's single-bin auto result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets.visualization_widgets import histogram_bin_width_widget


def test_constant_feature_returns_valid_bin_edges():
    edges = histogram_bin_width_widget(pd.Series([5.0, 5.0, 5.0]), key="const_feature")
    edges = np.asarray(edges)
    assert edges.ndim == 1
    assert len(edges) >= 2  # at least one bin
    assert edges[0] <= 5.0 <= edges[-1]
