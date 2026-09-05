"""A numpy float32 duration remains compatible with parallel fitting.
SDT repetition times can arrive as float32; lmfit parameter bounds must serialize
to JSON for transfer to workers.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.fit as fitmod
from src.fit import _init_params, fit_curves


def _synth_decays(n_curves, time_bins=256, duration=12.5, seed=0):
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins)
    rng = np.random.default_rng(seed)
    base = 8000 * np.exp(-time_axis / 0.5) + 2000 * np.exp(-time_axis / 2.5) + 5
    return [rng.poisson(np.maximum(base, 1e-6)).astype(float) for _ in range(n_curves)]


def _delta_irf(n_bins):
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    return irf


def test_init_params_float32_duration_is_json_serializable():
    """A float32 duration produces JSON-serializable parameter bounds."""
    dur32 = np.float32(0.05) / np.float32(4.0955) * 1e9  # float32, as from an SDT header
    assert isinstance(dur32, np.float32)
    params, _arrays, _fixed = _init_params(dur32, 256, 2, 12, False, None, None, None)
    params.dumps()  # Parameter serialization must succeed.
    assert type(params["t1"].max) is float


def test_parallel_path_not_disabled_by_float32_duration(capsys):
    """fit_curves with a float32 duration must stay on the parallel path."""
    dur32 = np.float32(12.5)
    time_bins = 256
    decays = _synth_decays(12, time_bins, float(dur32))  # >= _MIN_CURVES_FOR_PARALLEL
    irf = _delta_irf(time_bins)

    orig = fitmod._MIN_CURVES_FOR_PARALLEL
    fitmod._MIN_CURVES_FOR_PARALLEL = 1  # force the parallel path
    try:
        results = fit_curves(dur32, time_bins, decays, irf,
                             num_components=2, fitting_algo="WLS", fitting_mode="Local")
    finally:
        fitmod._MIN_CURVES_FOR_PARALLEL = orig

    out = capsys.readouterr().out
    assert "Parallel fitting unavailable" not in out, (
        f"parallel fitting fell back to sequential:\n{out}"
    )
    assert np.isfinite(results["t1"]).any()
