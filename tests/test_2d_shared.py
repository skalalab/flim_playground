"""2D Feature Distribution geometry shared by the app's figure and exported scripts."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis.bivar import gmm_ellipse_params
from src.vis.plot_defaults import panel_point_size


def test_gmm_ellipse_params_give_the_95_percent_ellipse_of_a_covariance():
    angle, width, height = gmm_ellipse_params(np.array([[4.0, 0.0], [0.0, 1.0]]))
    radius = np.sqrt(chi2.ppf(0.95, df=2))
    assert abs(angle) == 90.0                 # the smaller axis (variance 1) points along y; sign is LAPACK's
    assert width == 2 * radius * 1.0          # 2 * radius * sqrt(smaller eigenvalue)
    assert height == 2 * radius * 2.0         # 2 * radius * sqrt(larger eigenvalue)


def test_panel_point_size_is_two_below_the_main_size_and_at_least_one():
    assert panel_point_size(5) == 3
    assert panel_point_size(2) == 1
