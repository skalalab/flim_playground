"""GMM subpopulation labels (`GMM_group`) must be numbered by ascending-mean
rank — group1 is the smallest-mean component — so they line up with the
component table shown to the user in `feature_gmm_plot`. The bug was that the
hard-assignment path used `best_gmm.predict` (original component indices) and
the intersection path remapped digitize buckets back to original indices, so
the labels disagreed with the rank-ordered table whenever the fitted components
were not already in ascending-mean order.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis.univar import _assign_subpopulation_labels


class _FakeGMM:
    """Minimal GMM stand-in exposing only what the labeler reads."""

    def __init__(self, means, predictions=None):
        self.means_ = np.asarray(means, dtype=float).reshape(-1, 1)
        self._predictions = None if predictions is None else np.asarray(predictions)

    def predict(self, X):
        return self._predictions


def test_hard_assignment_labels_smallest_mean_component_as_group1():
    # Component 0 has the LARGEST mean, component 1 the smallest. predict()
    # returns original component indices [0, 1, 0, 1].
    gmm = _FakeGMM(means=[5.0, 1.0], predictions=[0, 1, 0, 1])
    labels = _assign_subpopulation_labels([4.9, 1.1, 5.1, 0.9], gmm, None, "GroupA")
    # Smallest-mean component (orig idx 1) -> group1; largest (orig idx 0) -> group2.
    assert labels == ["GroupA_group2", "GroupA_group1", "GroupA_group2", "GroupA_group1"]


def test_intersection_thresholds_bucket_points_in_ascending_mean_order():
    # Two components; the single intersection threshold sits at 3.0. Points below
    # it fall in the smallest-mean bucket (group1), points above in group2.
    gmm = _FakeGMM(means=[5.0, 1.0])
    labels = _assign_subpopulation_labels([0.5, 4.0, 2.0, 6.0], gmm, [3.0], "GroupA")
    assert labels == ["GroupA_group1", "GroupA_group2", "GroupA_group1", "GroupA_group2"]


def test_three_component_hard_assignment_orders_all_by_mean():
    # means: orig 0 -> 2.0 (rank 1), orig 1 -> 0.5 (rank 0), orig 2 -> 9.0 (rank 2)
    gmm = _FakeGMM(means=[2.0, 0.5, 9.0], predictions=[0, 1, 2])
    labels = _assign_subpopulation_labels([2.0, 0.5, 9.0], gmm, None, "G")
    assert labels == ["G_group2", "G_group1", "G_group3"]
