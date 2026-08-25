"""Degenerate-cell NaN/inf guards for per-cell feature extraction.

Each test pins a degenerate input that previously produced NaN/inf (often via an
unguarded division emitting a RuntimeWarning) to a deliberate, documented value:

1. `mass_displacement` (src/cell_texture.py): a fully-dark cell hit `np.mean([])`
   and a 0/0 division -> NaN with RuntimeWarnings. Now returns a deliberate NaN
   (a dark cell has no centroid, so the displacement is undefined), warning-free.
2. `reduced_chi_square` (src/fit_helper.py): an unguarded degrees-of-freedom term
   `len(data_slice) - num_free_params` <= 0 gave +inf / a negative reduced chi-
   square for a too-narrow gate. Now returns NaN.
3. `get_raw_phasor` (src/fov_extraction.py): a signal-less cell (decay sums to 0)
   divided 0/0. Now returns deliberate (NaN, NaN), no RuntimeWarning.
4. `extract_fit_results` (src/fov_extraction.py): a failed fit returns NaN amps;
   `total_amp == 0` did not catch them (NaN == 0 is False), so a1/a2/tm/tm_iw
   were computed as NaN/NaN. Now the cell is skipped (those keys absent -> NaN).
5. `irf_shift` (src/fit_helper.py): a degenerate (all-zero) IRF made the
   `/= np.sum(...)` normalisation 0/0 -> all NaN. Now left unnormalised.
6. `fit_curves` + `extract_fit_results` (src/fit.py, src/fov_extraction.py): an
   all-zero IRF makes the reconvolution model identically zero, so a fit returns
   finite-but-meaningless params (silent garbage). Now `fit_curves` returns NaN
   results and `extract_fit_results` flags it once at the channel level instead
   of emitting per-cell warnings.
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


# --- Bug 1: mass_displacement on a dark/empty cell ---------------------------

def test_mass_displacement_dark_cell_returns_nan():
    # A fully-dark cell (no pixel with intensity > 0) has no defined centroid, so
    # mass displacement is undefined -> deliberate NaN (not 0.0, which would
    # falsely assert the geometric and intensity-weighted centroids coincide).
    cell_image = np.zeros((5, 5))
    result = mass_displacement(cell_image)
    assert np.isnan(result)


def test_mass_displacement_normal_cell_unchanged():
    # Non-regression: an asymmetric two-pixel cell keeps its exact value.
    cell_image = np.zeros((5, 5))
    cell_image[1, 1] = 1.0
    cell_image[1, 3] = 100.0
    # geometric centroid_x = mean([1, 3]) = 2.0
    # intensity-weighted centroid_x = (1*1 + 3*100) / 101 = 301/101
    expected = abs(2.0 - 301.0 / 101.0)
    result = mass_displacement(cell_image)
    assert result == pytest.approx(expected)


# --- Bug 2: reduced_chi_square with non-positive degrees of freedom ----------

def test_reduced_chi_square_zero_dof_returns_nan():
    # len(data_slice) == num_free_params -> dof 0 -> currently +inf.
    fitted = np.array([1.0, 1.0, 1.0])
    data = np.array([2.0, 2.0, 2.0])
    result = reduced_chi_square(fitted, data, start=0, end=3, num_free_params=3)
    assert np.isnan(result)


def test_reduced_chi_square_negative_dof_returns_nan():
    # len(data_slice) < num_free_params -> dof negative -> currently a
    # nonsensical negative reduced chi-square.
    fitted = np.array([1.0, 1.0])
    data = np.array([2.0, 2.0])
    result = reduced_chi_square(fitted, data, start=0, end=2, num_free_params=5)
    assert np.isnan(result)


def test_reduced_chi_square_normal_case_unchanged():
    # Non-regression: dof > 0 computes the ordinary reduced chi-square.
    fitted = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    data = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    # residuals**2 / fitted = 1 each, sum = 5; dof = 5 - 3 = 2; chiq = 2.5
    result = reduced_chi_square(fitted, data, start=0, end=5, num_free_params=3)
    assert result == pytest.approx(2.5)


# --- Bug 3: get_raw_phasor on a signal-less cell -----------------------------

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
    # Non-regression: a real decay still matches the raw-phasor formula.
    decay = np.array([0.0, 5.0, 3.0, 1.0])
    time_axis = np.array([0.0, 1.0, 2.0, 3.0])
    w = 0.5
    total = decay.sum()
    expected_g = np.dot(decay, np.cos(1 * w * time_axis)) / total
    expected_s = np.dot(decay, np.sin(1 * w * time_axis)) / total
    g, s = get_raw_phasor(decay, h=1, w=w, time_axis=time_axis, full_period=False)
    assert g == pytest.approx(expected_g)
    assert s == pytest.approx(expected_s)


# --- Bug 4: extract_fit_results propagating failed-fit NaN amps --------------

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
    # Non-regression: a normal fit still produces the derived features.
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


# --- Bug 5: irf_shift normalisation of a degenerate IRF ----------------------

def test_irf_shift_zero_irf_no_nan():
    # An all-zero IRF must not produce NaN via the 0/0 normalisation.
    irf = np.zeros(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = irf_shift(irf, shift=0)
    assert not np.any(np.isnan(result))


def test_irf_shift_normal_irf_normalized():
    # Non-regression: a normal IRF is normalised to sum 1, no NaN.
    irf = np.zeros(20)
    irf[5] = 10.0
    irf[6] = 5.0
    result = irf_shift(irf, shift=0)
    assert result.sum() == pytest.approx(1.0)
    assert not np.any(np.isnan(result))


# --- Bug 6: all-zero IRF flagged as NaN (not silently fit to garbage) --------

def test_fit_curves_all_zero_irf_returns_nan():
    # A zero IRF -> zero reconvolution model -> a fit would return finite garbage.
    # fit_curves must instead return its NaN-initialised results.
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
