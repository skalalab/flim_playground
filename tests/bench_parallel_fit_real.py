"""Benchmark sequential and parallel fitting on the T-cell activation example data.

Uses 342 cells, 200 time bins, and a measured IRF. Run manually:
    PYTHONPATH=. uv run python tests/bench_parallel_fit_real.py
    PYTHONPATH=. uv run python tests/bench_parallel_fit_real.py --mode Local --components 1
Automated correctness coverage lives in test_fit_determinism.py.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fit import fit_curves, _MIN_CURVES_FOR_PARALLEL

DATA_DIR = os.path.join(os.path.dirname(__file__), "..",
                        "example_data", "Data_Extraction", "T_cell_activation")


def load_real_data():
    curves = np.loadtxt(os.path.join(DATA_DIR, "Tcell_Act_filtered.csv"), delimiter=",")
    irf = np.loadtxt(os.path.join(DATA_DIR, "IRF.txt"))
    # Also load quiescent if available for more curves
    qui_path = os.path.join(DATA_DIR, "Tcell_Qui_filtered.csv")
    if os.path.exists(qui_path):
        qui = np.loadtxt(qui_path, delimiter=",")
        curves = np.vstack([curves, qui])
    return [curves[i] for i in range(curves.shape[0])], irf


def run_benchmark(fitting_algo, fitting_mode, num_components):
    decays, irf = load_real_data()
    n_curves = len(decays)
    time_bins = len(decays[0])
    duration = 12.5  # typical for TCSPC

    print(f"\n{'='*60}")
    print(f"  REAL DATA — {n_curves} cells, {time_bins} bins")
    print(f"  Algo: {fitting_algo}  |  Mode: {fitting_mode}  |  Components: {num_components}")
    print(f"  Parallel threshold: {_MIN_CURVES_FOR_PARALLEL} curves")
    print(f"{'='*60}")

    common_kwargs = dict(
        num_components=num_components, fitting_algo=fitting_algo,
        fitting_mode=fitting_mode,
    )

    from src import fit as fit_module
    original = fit_module._MIN_CURVES_FOR_PARALLEL

    # --- Sequential ---
    fit_module._MIN_CURVES_FOR_PARALLEL = n_curves + 1
    t0 = time.perf_counter()
    results_seq = fit_curves(duration, time_bins, decays, irf, **common_kwargs)
    t_seq = time.perf_counter() - t0
    fit_module._MIN_CURVES_FOR_PARALLEL = original

    # --- Parallel ---
    fit_module._MIN_CURVES_FOR_PARALLEL = 1
    t0 = time.perf_counter()
    results_par = fit_curves(duration, time_bins, decays, irf, **common_kwargs)
    t_par = time.perf_counter() - t0
    fit_module._MIN_CURVES_FOR_PARALLEL = original

    # --- Timing ---
    speedup = t_seq / t_par if t_par > 0 else float("inf")
    print(f"\n  Sequential: {t_seq:.2f}s  ({t_seq/n_curves*1000:.0f} ms/curve)")
    print(f"  Parallel:   {t_par:.2f}s  ({t_par/n_curves*1000:.0f} ms/curve)")
    print(f"  Speedup:    {speedup:.2f}x")

    # --- Result comparison ---
    all_close = True
    for key in results_seq:
        s, p = results_seq[key], results_par[key]
        nan_s, nan_p = np.isnan(s), np.isnan(p)
        mask = ~nan_s & ~nan_p
        if mask.sum() == 0:
            continue
        max_abs = np.max(np.abs(s[mask] - p[mask]))
        if max_abs > 0.01:
            all_close = False
            print(f"  MISMATCH [{key}]: max_abs_diff={max_abs:.2e}")
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
    parser.add_argument("--algo", choices=["MLE", "WLS"], default="MLE")
    parser.add_argument("--mode", choices=["Hybrid", "Local"], default="Hybrid")
    parser.add_argument("--components", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()

    run_benchmark(args.algo, args.mode, args.components)
