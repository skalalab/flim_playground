"""Lifetime fits are reproducible across repeated calls, execution modes, and
batch sizes for identical curves.
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
    """Repeated calls are identical. Local exercises the global warm-start seed;
    Hybrid exercises the per-curve global-search seed.
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
    """A cell's fit is independent of batch size when its batch contains identical curves.
    The mean decay is unchanged across these batches, so warm-start scaling and
    fitted lifetimes must also remain unchanged.
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
