"""Degenerate extraction inputs return deliberate results without division warnings.
Dark-cell displacement and zero-signal phasors are undefined; nonpositive fit
degrees of freedom return NaN. Failed fits omit derived amplitude features, and
an all-zero IRF yields NaN fit results with one channel-level warning.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cell_texture import mass_displacement
from src.fit import fit_curves
from src.fit_helper import reduced_chi_square, irf_shift
from src.fov_extraction import get_raw_phasor, extract_fit_results


# Dark or empty cell displacement

def test_mass_displacement_dark_cell_returns_nan():
    # A fully-dark cell (no pixel with intensity > 0) has no defined centroid, so
    # mass displacement is undefined -> deliberate NaN (not 0.0, which would
    # falsely assert the geometric and intensity-weighted centroids coincide).
    cell_image = np.zeros((5, 5))
    result = mass_displacement(cell_image)
    assert np.isnan(result)


def test_mass_displacement_normal_cell_unchanged():
    # An asymmetric two-pixel cell retains its exact displacement.
    cell_image = np.zeros((5, 5))
    cell_image[1, 1] = 1.0
    cell_image[1, 3] = 100.0
    # geometric centroid_x = mean([1, 3]) = 2.0
    # intensity-weighted centroid_x = (1*1 + 3*100) / 101 = 301/101
    expected = abs(2.0 - 301.0 / 101.0)
    result = mass_displacement(cell_image)
    assert result == pytest.approx(expected)


# Reduced chi-square with nonpositive degrees of freedom

def test_reduced_chi_square_zero_dof_returns_nan():
    # len(data_slice) == num_free_params gives zero degrees of freedom.
    fitted = np.array([1.0, 1.0, 1.0])
    data = np.array([2.0, 2.0, 2.0])
    result = reduced_chi_square(fitted, data, start=0, end=3, num_free_params=3)
    assert np.isnan(result)


def test_reduced_chi_square_negative_dof_returns_nan():
    # len(data_slice) < num_free_params gives negative degrees of freedom.
    fitted = np.array([1.0, 1.0])
    data = np.array([2.0, 2.0])
    result = reduced_chi_square(fitted, data, start=0, end=2, num_free_params=5)
    assert np.isnan(result)


def test_reduced_chi_square_normal_case_unchanged():
    # Positive degrees of freedom use the ordinary reduced chi-square.
    fitted = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    # residuals**2 / fitted = 1 each, sum = 5; dof = 5 - 3 = 2; chiq = 2.5
    result = reduced_chi_square(fitted, data, start=0, end=5, num_free_params=3)
    assert result == pytest.approx(2.5)


# Phasors for cells with no signal

def test_get_raw_phasor_zero_signal_no_warning_and_nan():
    # A cell whose (offset-subtracted, clipped) decay sums to 0 -> undefined
    # phasor. Must return NaN deliberately, with no 0/0 RuntimeWarning.
    decay = np.zeros(10)
    time_axis = np.linspace(0.0, 9.0, 10)
    w = 2 * np.pi * 0.08
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        g, s = get_raw_phasor(decay, h=1, w=w, time_axis=time_axis, full_period=False)
    assert np.isnan(g) and np.isnan(s)


def test_get_raw_phasor_normal_signal_matches_formula():
    # A nonzero decay follows the raw-phasor formula.
    decay = np.array([0.0, 5.0, 3.0, 1.0])
    time_axis = np.array([0.0, 1.0, 2.0, 3.0])
    w = 0.5
    total = decay.sum()
    expected_g = np.dot(decay, np.cos(1 * w * time_axis)) / total
    expected_s = np.dot(decay, np.sin(1 * w * time_axis)) / total
    g, s = get_raw_phasor(decay, h=1, w=w, time_axis=time_axis, full_period=False)
    assert g == pytest.approx(expected_g)
    assert s == pytest.approx(expected_s)


# Derived features for failed fits

def _fit_results_kwargs(results, num_components):
    return dict(
        channel_name="ch1",
        decay_curves={"cell_1": np.array([100.0, 50.0, 25.0])},
        results=results,
        num_components=num_components,
        shifted_irf=np.array([1.0, 0.0, 0.0]),
        time_axis=np.array([0.0, 1.0, 2.0]),
        start=0,
        end=3,
        fixed_lifetimes=None,
    )


def test_extract_fit_results_nan_amps_skips_derived():
    # A failed fit yields NaN amps. The derived alpha / mean-lifetime features
    # must be skipped (absent -> NaN in the DataFrame), not computed as NaN/NaN.
    results = {
        "amp1": [np.nan], "amp2": [np.nan],
        "t1": [np.nan], "t2": [np.nan], "offset": [np.nan],
    }
    warning_msg, feats = extract_fit_results(**_fit_results_kwargs(results, 2))
    cell = feats["cell_1"]
    assert "Lifetime fit_ch1: a1" not in cell
    assert "Lifetime fit_ch1: tm" not in cell
    assert "Lifetime fit_ch1: tm_iw" not in cell
    assert warning_msg != ""


def test_extract_fit_results_finite_amps_unchanged():
    # A valid fit produces its derived features.
    results = {
        "amp1": [80.0], "amp2": [20.0],
        "t1": [0.4], "t2": [2.5], "offset": [5.0],
    }
    warning_msg, feats = extract_fit_results(**_fit_results_kwargs(results, 2))
    cell = feats["cell_1"]
    assert "Lifetime fit_ch1: a1" in cell
    assert np.isfinite(cell["Lifetime fit_ch1: a1"])
    assert np.isfinite(cell["Lifetime fit_ch1: tm"])
    assert warning_msg == ""


# Degenerate IRF normalization

def test_irf_shift_zero_irf_no_nan():
    # An all-zero IRF must not produce NaN via the 0/0 normalisation.
    irf = np.zeros(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = irf_shift(irf, shift=0)
    assert not np.any(np.isnan(result))


def test_irf_shift_normal_irf_normalized():
    # A nonzero IRF normalizes to a finite unit sum.
    irf = np.zeros(20)
    irf[5] = 10.0
    irf[6] = 5.0
    result = irf_shift(irf, shift=0)
    assert result.sum() == pytest.approx(1.0)
    assert not np.any(np.isnan(result))


# All-zero IRFs yield NaN fits

def test_fit_curves_all_zero_irf_returns_nan():
    # A zero IRF gives a zero reconvolution model, so the fit returns NaN results.
    duration, time_bins = 12.5, 64
    decay = np.linspace(100.0, 1.0, time_bins)
    results = fit_curves(
        duration, time_bins, [decay, decay], np.zeros(time_bins),
        num_components=1, fitting_algo="MLE", fitting_mode="Local",
        start=0, end=time_bins,
    )
    assert np.all(np.isnan(results["t1"]))
    assert np.all(np.isnan(results["amp1"]))
    assert np.all(np.isnan(results["offset"]))


def test_extract_fit_results_all_zero_irf_flags_nan_once():
    # With an all-zero IRF (so fit_curves returned NaN params), the fit features
    # must be NaN and flagged ONCE at the channel level, not per cell.
    channel_name = "ch1"
    decay_curves = {
        "cell_1": np.array([100.0, 50.0, 25.0]),
        "cell_2": np.array([80.0, 40.0, 20.0]),
    }
    nan2 = [np.nan, np.nan]
    results = {"amp1": nan2, "amp2": nan2, "t1": nan2, "t2": nan2, "offset": nan2}
    warning_msg, feats = extract_fit_results(
        channel_name, decay_curves, results, num_components=2,
        shifted_irf=np.zeros(3), time_axis=np.array([0.0, 1.0, 2.0]),
        start=0, end=3, fixed_lifetimes=None,
    )
    # one channel-level flag, no per-cell amplitude warnings
    assert "IRF is all zeros" in warning_msg
    assert "total amplitude" not in warning_msg
    # every fit feature is NaN for every cell
    for cell in ("cell_1", "cell_2"):
        assert np.isnan(feats[cell]["Lifetime fit_ch1: t1"])
        assert np.isnan(feats[cell]["ch1_amp1"])
