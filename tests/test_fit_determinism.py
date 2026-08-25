"""Reproducibility guarantees for fit_curves().

The lifetime fit must be deterministic: the same decays fit twice must give
identical results, and the sequential and parallel code paths must agree.

Regression guard for the warm-start seeding bug — the global-search warm-start
(and the Hybrid per-curve global step) were seeded with `rng=`, which lmfit does
not forward to scipy.differential_evolution, leaving the search unseeded and the
whole fit nondeterministic call-to-call.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.fit as fitmod
from src.fit import fit_curves

DATA_DIR = os.path.join(os.path.dirname(__file__), "..",
                        "example_data", "Data_Extraction", "T_cell_activation")
DURATION = 12.5
N_CURVES = 12  # > _MIN_CURVES_FOR_PARALLEL so the parallel path is exercised


def _load_real(n=N_CURVES):
    """Real TCSPC decays — ill-conditioned enough to expose nondeterminism."""
    curves = np.loadtxt(os.path.join(DATA_DIR, "Tcell_Act_filtered.csv"), delimiter=",")
    irf = np.loadtxt(os.path.join(DATA_DIR, "IRF.txt"))
    decays = [curves[i] for i in range(n)]
    return decays, irf, curves.shape[1]


def _common_kwargs(mode="Local"):
    return dict(num_components=2, fitting_algo="WLS", fitting_mode=mode)


@pytest.mark.parametrize("mode", ["Local", "Hybrid"])
def test_fit_reproducible_across_calls(mode):
    """Two identical calls must produce identical results.

    Local exercises the global-search warm-start seed; Hybrid exercises the
    per-curve global-search seed. Both were broken by the `rng=` bug.
    """
    decays, irf, time_bins = _load_real()
    r1 = fit_curves(DURATION, time_bins, decays, irf, **_common_kwargs(mode))
    r2 = fit_curves(DURATION, time_bins, decays, irf, **_common_kwargs(mode))
    for k in r1:
        np.testing.assert_allclose(
            r1[k], r2[k], rtol=0, atol=1e-9,
            err_msg=f"'{k}' is not reproducible across calls ({mode} mode)",
        )


def test_sequential_matches_parallel():
    """The sequential and parallel paths must agree on the same data."""
    decays, irf, time_bins = _load_real()
    orig = fitmod._MIN_CURVES_FOR_PARALLEL
    try:
        fitmod._MIN_CURVES_FOR_PARALLEL = 10**9  # force sequential
        seq = fit_curves(DURATION, time_bins, decays, irf, **_common_kwargs())
        fitmod._MIN_CURVES_FOR_PARALLEL = 1       # force parallel
        par = fit_curves(DURATION, time_bins, decays, irf, **_common_kwargs())
    finally:
        fitmod._MIN_CURVES_FOR_PARALLEL = orig
    for k in seq:
        np.testing.assert_allclose(
            seq[k], par[k], rtol=0, atol=1e-6,
            err_msg=f"'{k}' differs between sequential and parallel",
        )


@pytest.mark.parametrize("num_components", [2, 3])
def test_fit_is_batch_size_invariant(num_components):
    """A cell's fit must not depend on how many cells share its batch.

    Regression guard for the Local-mode warm-start scale bug: the warm-start fit the
    SUMMED decay, whose offset (~N x a single cell's, then clamped to the peak) and
    peak-derived differential-evolution bounds both grew with the batch size N. That
    leaked batch size into the ill-conditioned per-cell fits, so the same cell fit to
    different lifetimes depending on how many cells were co-fit (e.g. tau2 swinging
    1.8 -> 11.5 ns across N = 3 / 12 / 48). Fixed by warm-starting on the MEAN decay.

    Fitting K identical copies of one real curve in different batch sizes isolates the
    scale dependence (the mean of identical copies is the curve itself, independent of
    K), so the lifetimes must be identical across batch sizes.
    """
    decays, irf, time_bins = _load_real()
    curve = decays[0]
    kwargs = dict(num_components=num_components, fitting_algo="WLS", fitting_mode="Local")

    orig = fitmod._MIN_CURVES_FOR_PARALLEL
    fitmod._MIN_CURVES_FOR_PARALLEL = 10**9  # sequential, for a clean comparison
    try:
        reference = None
        for n in (4, 12, 40):
            batch = [curve.copy() for _ in range(n)]
            res = fit_curves(DURATION, time_bins, batch, irf, **kwargs)
            taus = {k: res[k][0] for k in ("t1", "t2", "t3") if k in res}
            if reference is None:
                reference = taus
                continue
            for k, val in taus.items():
                np.testing.assert_allclose(
                    val, reference[k], rtol=0, atol=1e-6,
                    err_msg=f"'{k}' for the same cell changed with batch size "
                            f"(n={n}, {num_components}-comp) — warm-start is not "
                            f"batch-size invariant",
                )
    finally:
        fitmod._MIN_CURVES_FOR_PARALLEL = orig
