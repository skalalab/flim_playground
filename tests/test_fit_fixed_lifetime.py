"""
Unit tests for fixed-lifetime component fitting in fit_curves().

Synthesises simple 1- and 2-component decays with a delta-function IRF
(no convolution blur) and verifies that fixed τ truly stays constant while
amplitudes/free τ are still optimised.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fit import fit_curves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delta_irf(n_bins: int) -> np.ndarray:
    """Delta-function IRF: all weight at bin 0."""
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    return irf


def _synth_decay(time_axis, amps, taus, offset=5.0) -> np.ndarray:
    """Sum of exponentials + flat offset (no IRF blur needed for delta IRF)."""
    decay = np.zeros_like(time_axis)
    for amp, tau in zip(amps, taus):
        decay += amp * np.exp(-time_axis / tau)
    decay += offset
    # Add mild Poisson noise to mimic real data
    rng = np.random.default_rng(42)
    return rng.poisson(np.maximum(decay, 1e-6)).astype(float)


# Shared acquisition parameters
DURATION = 12.5   # ns
TIME_BINS = 256
PERIOD = DURATION / TIME_BINS
TIME_AXIS = np.linspace(0, (TIME_BINS - 1) * PERIOD, TIME_BINS)
IRF = _delta_irf(TIME_BINS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllFree:
    """Regression guard: no fixed_lifetimes produces reasonable free-fit."""

    def test_single_component_no_fixed(self):
        decay = _synth_decay(TIME_AXIS, [5000], [0.5])
        results = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=1, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes=None,
        )
        assert results["t1"][0] == pytest.approx(0.5, rel=0.15), (
            f"Expected τ1 ≈ 0.5 ns, got {results['t1'][0]:.4f}"
        )

    def test_two_component_no_fixed(self):
        decay = _synth_decay(TIME_AXIS, [8000, 3000], [0.4, 2.5])
        results = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes=None,
        )
        # Sorted: t1 < t2
        assert results["t1"][0] < results["t2"][0]


class TestFixedT1:
    """Fix τ1 — should be held exactly; amplitudes and free τ still optimise."""

    def test_t1_stays_fixed(self):
        FIXED_T1 = 0.4  # ns
        decay = _synth_decay(TIME_AXIS, [8000, 3000], [0.4, 2.5])
        results = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes={"t1": FIXED_T1},
        )
        assert results["t1"][0] == pytest.approx(FIXED_T1, abs=1e-9), (
            f"τ1 should be exactly {FIXED_T1}, got {results['t1'][0]}"
        )

    def test_t2_still_free_when_t1_fixed(self):
        FIXED_T1 = 0.4
        decay = _synth_decay(TIME_AXIS, [8000, 3000], [0.4, 2.5])
        results = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes={"t1": FIXED_T1},
        )
        # τ2 should converge toward 2.5 ns (not be stuck at any fixed value)
        assert results["t2"][0] == pytest.approx(2.5, rel=0.20), (
            f"Free τ2 should converge near 2.5 ns, got {results['t2'][0]:.4f}"
        )


class TestFixedT2:
    """Fix τ2 — τ1 (shorter, free) should still converge."""

    def test_t2_stays_fixed(self):
        FIXED_T2 = 2.5  # ns
        decay = _synth_decay(TIME_AXIS, [8000, 3000], [0.4, 2.5])
        results = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes={"t2": FIXED_T2},
        )
        assert results["t2"][0] == pytest.approx(FIXED_T2, abs=1e-9), (
            f"τ2 should be exactly {FIXED_T2}, got {results['t2'][0]}"
        )

    def test_t1_still_free_when_t2_fixed(self):
        FIXED_T2 = 2.5
        decay = _synth_decay(TIME_AXIS, [8000, 3000], [0.4, 2.5])
        results = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes={"t2": FIXED_T2},
        )
        assert results["t1"][0] == pytest.approx(0.4, rel=0.20), (
            f"Free τ1 should converge near 0.4 ns, got {results['t1'][0]:.4f}"
        )


class TestZeroOrNoneIsIgnored:
    """Passing 0 or None for a component leaves it free (same as not passing it)."""

    def test_zero_val_is_free(self):
        decay = _synth_decay(TIME_AXIS, [8000, 3000], [0.4, 2.5])
        results_free = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
        )
        results_zero = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes={"t1": 0.0, "t2": 0.0},
        )
        assert results_zero["t1"][0] == pytest.approx(results_free["t1"][0], rel=1e-6)
        assert results_zero["t2"][0] == pytest.approx(results_free["t2"][0], rel=1e-6)

    def test_none_val_is_free(self):
        decay = _synth_decay(TIME_AXIS, [8000, 3000], [0.4, 2.5])
        results_free = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
        )
        results_none = fit_curves(
            DURATION, TIME_BINS, [decay], IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes={"t1": None, "t2": None},
        )
        assert results_none["t1"][0] == pytest.approx(results_free["t1"][0], rel=1e-6)
        assert results_none["t2"][0] == pytest.approx(results_free["t2"][0], rel=1e-6)


class TestMultipleCurves:
    """Fixed τ is consistent across all cells in a batch."""

    def test_fixed_t1_all_cells(self):
        FIXED_T1 = 0.4
        rng = np.random.default_rng(7)
        decays = [
            rng.poisson(np.maximum(
                8000 * np.exp(-TIME_AXIS / 0.4) + 3000 * np.exp(-TIME_AXIS / 2.5) + 5, 1e-6
            )).astype(float)
            for _ in range(5)
        ]
        results = fit_curves(
            DURATION, TIME_BINS, decays, IRF,
            num_components=2, fitting_algo="WLS", fitting_mode="Local",
            fixed_lifetimes={"t1": FIXED_T1},
        )
        for i, t1_val in enumerate(results["t1"]):
            assert t1_val == pytest.approx(FIXED_T1, abs=1e-9), (
                f"Cell {i}: τ1 should be exactly {FIXED_T1}, got {t1_val}"
            )
