"""Benchmark sequential and parallel fitting of synthetic two-component decays.

Compare timing and numerical results on Poisson-noisy data. Run manually:
    PYTHONPATH=. uv run python tests/bench_parallel_fit.py
    PYTHONPATH=. uv run python tests/bench_parallel_fit.py --curves 200 --mode Hybrid --fit-shift
Automated correctness coverage lives in test_fit_determinism.py.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fit import fit_curves, _MIN_CURVES_FOR_PARALLEL


def _delta_irf(n_bins):
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    return irf


def _gaussian_irf(n_bins, center=10, sigma=3):
    """Realistic Gaussian-shaped IRF for shift fitting tests."""
    x = np.arange(n_bins)
    irf = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    irf /= irf.sum()
    return irf


def _synth_decays(n_curves, time_axis, seed=42):
    rng = np.random.default_rng(seed)
    decays = []
    for _ in range(n_curves):
        amp1 = rng.uniform(3000, 10000)
        amp2 = rng.uniform(1000, 5000)
        t1 = rng.uniform(0.3, 0.8)
        t2 = rng.uniform(1.5, 4.0)
        offset = rng.uniform(2, 10)
        decay = amp1 * np.exp(-time_axis / t1) + amp2 * np.exp(-time_axis / t2) + offset
        decay = rng.poisson(np.maximum(decay, 1e-6)).astype(float)
        decays.append(decay)
    return decays


def run_benchmark(n_curves, fitting_algo, fitting_mode, fit_shift):
    duration = 12.5
    time_bins = 256
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins)
    irf = _gaussian_irf(time_bins) if fit_shift else _delta_irf(time_bins)

    decays = _synth_decays(n_curves, time_axis)

    shift_str = " + fit_shift" if fit_shift else ""
    print(f"\n{'='*60}")
    print(f"  Curves: {n_curves}  |  Algo: {fitting_algo}  |  Mode: {fitting_mode}{shift_str}")
    print(f"  Parallel threshold: {_MIN_CURVES_FOR_PARALLEL} curves")
    print(f"{'='*60}")

    common_kwargs = dict(
        num_components=2, fitting_algo=fitting_algo, fitting_mode=fitting_mode,
        fit_shift=fit_shift, shift_guess=0.0 if fit_shift else None,
    )

    # --- Sequential ---
    from src import fit as fit_module
    original_threshold = fit_module._MIN_CURVES_FOR_PARALLEL
    fit_module._MIN_CURVES_FOR_PARALLEL = n_curves + 1  # force sequential

    t0 = time.perf_counter()
    results_seq = fit_curves(duration, time_bins, decays, irf, **common_kwargs)
    t_seq = time.perf_counter() - t0

    fit_module._MIN_CURVES_FOR_PARALLEL = original_threshold  # restore

    # --- Parallel ---
    fit_module._MIN_CURVES_FOR_PARALLEL = 1  # force parallel

    t0 = time.perf_counter()
    results_par = fit_curves(duration, time_bins, decays, irf, **common_kwargs)
    t_par = time.perf_counter() - t0

    fit_module._MIN_CURVES_FOR_PARALLEL = original_threshold  # restore

    # --- Timing ---
    speedup = t_seq / t_par if t_par > 0 else float('inf')
    print(f"\n  Sequential: {t_seq:.2f}s")
    print(f"  Parallel:   {t_par:.2f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    # --- Result comparison ---
    # Note: Differential Evolution is stochastic, so results may differ slightly
    # between runs even with identical inputs. We check that the fits are
    # functionally equivalent, not bit-identical.
    all_close = True
    for key in results_seq:
        seq_vals = results_seq[key]
        par_vals = results_par[key]
        nan_seq = np.isnan(seq_vals)
        nan_par = np.isnan(par_vals)
        nan_match = np.array_equal(nan_seq, nan_par)
        if not nan_match:
            n_only_seq = np.sum(nan_seq & ~nan_par)
            n_only_par = np.sum(~nan_seq & nan_par)
            print(f"  WARN [{key}]: NaN mismatch — seq_only={n_only_seq}, par_only={n_only_par}")
            # Compare the subset where both succeeded
            mask = ~nan_seq & ~nan_par
        else:
            mask = ~nan_seq
        if mask.sum() == 0:
            continue
        max_abs = np.max(np.abs(seq_vals[mask] - par_vals[mask]))
        abs_tol = 0.01
        if max_abs > abs_tol:
            all_close = False
            print(f"  MISMATCH [{key}]: max_abs_diff={max_abs:.2e} (tol={abs_tol})")
        else:
            print(f"  OK [{key}]: max_abs_diff={max_abs:.2e}")

    n_nan_seq = sum(np.isnan(results_seq[k]).sum() for k in results_seq)
    n_nan_par = sum(np.isnan(results_par[k]).sum() for k in results_par)
    print(f"\n  Failed fits: sequential={n_nan_seq}, parallel={n_nan_par}")

    if all_close:
        print(f"  RESULT: All values match (abs_tol=0.01)")
    else:
        print(f"  RESULT: Some values differ — see above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=int, default=50)
    parser.add_argument("--algo", choices=["MLE", "WLS"], default="MLE")
    parser.add_argument("--mode", choices=["Hybrid", "Global", "Local"], default="Hybrid")
    parser.add_argument("--fit-shift", action="store_true")
    args = parser.parse_args()

    run_benchmark(args.curves, args.algo, args.mode, args.fit_shift)
