"""Guards for degenerate (constant / all-zero) columns in univariate plotting.

Selecting a feature whose values are all identical (e.g. an all-zero derived
column) used to crash feature_comparison_plot: gaussian_kde raises LinAlgError on
the singular covariance of a zero-variance sample, and the effect-size functions
divide by a zero spread. These lock in the graceful fallbacks so a constant column
renders flat instead of aborting the whole page.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis.helpers import _estimate_density_1d, glass_delta, cohens_d


def test_density_constant_column_returns_zero_density_not_crash():
    # All-zero (and any constant) column: no KDE, but must not raise.
    for values in ([0.0, 0.0, 0.0, 0.0], [5, 5, 5], np.zeros(50)):
        kde = _estimate_density_1d(values)
        out = kde(np.asarray(values, dtype=float))
        assert np.all(out == 0), "constant column should yield zero density"


def test_density_too_few_points_returns_zero_density():
    for values in ([], [3.14]):
        kde = _estimate_density_1d(values)
        assert np.all(kde(np.array([1.0, 2.0])) == 0)


def test_density_drops_non_finite_and_still_degenerate():
    # After dropping NaN/inf only one distinct finite value remains -> zero density.
    kde = _estimate_density_1d([2.0, 2.0, np.nan, np.inf, -np.inf])
    assert np.all(kde(np.array([2.0])) == 0)


def test_density_normal_column_produces_real_positive_density():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    kde = _estimate_density_1d(values)
    dens = kde(values)
    assert dens.shape == values.shape
    assert np.any(dens > 0), "non-degenerate data should have positive density somewhere"


def test_glass_delta_constant_control_is_nan_not_inf():
    control = [1.0, 1.0, 1.0, 1.0]   # zero spread
    treat = [2.0, 3.0, 4.0, 5.0]
    assert np.isnan(glass_delta(control, treat, "Mean"))
    assert np.isnan(glass_delta(control, treat, "Median"))


def test_cohens_d_both_constant_is_nan_not_inf():
    g1 = [0.0, 0.0, 0.0, 0.0]
    g2 = [0.0, 0.0, 0.0, 0.0]
    assert np.isnan(cohens_d(g1, g2, "Mean"))
    assert np.isnan(cohens_d(g1, g2, "Median"))


def test_effect_sizes_still_finite_on_normal_data():
    control = [1.0, 2.0, 3.0, 4.0]
    treat = [2.0, 3.0, 4.0, 5.0]
    assert np.isfinite(glass_delta(control, treat, "Mean"))
    assert np.isfinite(cohens_d(control, treat, "Mean"))
