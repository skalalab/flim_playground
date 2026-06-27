import multiprocessing
import os
from os import cpu_count

import numpy as np
import psutil
from lmfit import minimize as lmfit_minimize
from lmfit import Parameters
from src.fit_helper import objective, upsample_irf, irf_fwhm_bins

_MIN_CURVES_FOR_PARALLEL = 10


def _init_params(duration, time_bins, num_components, num_curves, fit_shift, shift_guess, shift_halfwidth, fixed_lifetimes):
    """Build lmfit Parameters and allocate result arrays."""
    params = Parameters()
    _fl = fixed_lifetimes or {}

    def _get_fixed(key):
        val = _fl.get(key)
        return float(val) if (val is not None and float(val) > 0) else None

    fixed = {}
    arrays = {
        "amp1": np.full(num_curves, np.nan),
        "t1": np.full(num_curves, np.nan),
        "offset": np.full(num_curves, np.nan),
    }

    params.add('amp1', min=0.001)
    fixed['t1'] = _get_fixed('t1')
    if fixed['t1'] is not None:
        params.add('t1', value=fixed['t1'], vary=False)
    else:
        params.add('t1', min=0.001, max=duration)
    params.add('offset', min=0.0)

    if num_components > 1:
        arrays["amp2"] = np.full(num_curves, np.nan)
        arrays["t2"] = np.full(num_curves, np.nan)
        params.add('amp2', min=0.001)
        fixed['t2'] = _get_fixed('t2')
        if fixed['t2'] is not None:
            params.add('t2', value=fixed['t2'], vary=False)
        else:
            params.add('t2', min=0.001, max=duration)

    if num_components > 2:
        arrays["amp3"] = np.full(num_curves, np.nan)
        arrays["t3"] = np.full(num_curves, np.nan)
        params.add('amp3', min=0.001)
        fixed['t3'] = _get_fixed('t3')
        if fixed['t3'] is not None:
            params.add('t3', value=fixed['t3'], vary=False)
        else:
            params.add('t3', min=0.001, max=duration)

    if fit_shift:
        arrays["shift"] = np.full(num_curves, np.nan)
        center = float(shift_guess) if shift_guess is not None else 0.0
        params.add('shift', value=center,
                   min=center - shift_halfwidth, max=center + shift_halfwidth)

    return params, arrays, fixed


def _set_amplitude_guesses(current_params, decay_curve, num_components):
    """Set per-curve amplitude and offset initial values and bounds."""
    peak = np.max(decay_curve)
    current_params['amp1'].value = peak
    current_params['amp1'].max = peak * 10
    current_params['offset'].max = peak
    if num_components > 1:
        current_params['amp2'].value = peak / 2
        current_params['amp2'].max = peak * 10
    if num_components > 2:
        current_params['amp3'].value = peak / 2
        current_params['amp3'].max = peak * 10


def _fit_single_curve(decay_curve, current_params, irf, time_axis, start, end, fitting_algo, fitting_mode, irf_upsampled, optimizers, seed=None):
    """Run the appropriate optimizer(s) on a single decay curve.

    Modes:
      - Hybrid: global search (DE) → local refinement
      - Local:  local optimizer only (caller should warm-start params)
    """
    args = (decay_curve, irf, time_axis, start, end, fitting_algo, irf_upsampled)

    if fitting_mode == "Hybrid":
        global_opts = dict(optimizers["global_opts"])
        if seed is not None:
            # lmfit forwards `seed` (not `rng`) to scipy.differential_evolution,
            # so seed via `seed` to make the global search reproducible.
            global_opts["seed"] = seed
        result_global = lmfit_minimize(objective, current_params, args=args, method=optimizers["global"], **global_opts)
        current_params = result_global.params

    if fitting_algo == "MLE":
        return lmfit_minimize(objective, current_params, args=args, method=optimizers["mle"], options=optimizers["mle_opts"])
    elif fitting_algo == "WLS":
        return lmfit_minimize(objective, current_params, args=args, method=optimizers["wls"], **optimizers["wls_opts"])
    else:
        raise ValueError(f"Unsupported fitting algorithm: {fitting_algo}. Use 'MLE' or 'WLS'.")


def _extract_result(result, arrays, i, num_components, fit_shift, fixed):
    """Extract fitted parameters into result arrays, sorting free components by lifetime."""
    if fit_shift:
        arrays["shift"][i] = result.params['shift'].value
    arrays["offset"][i] = result.params['offset'].value

    if num_components == 1:
        arrays["amp1"][i] = result.params['amp1'].value
        arrays["t1"][i] = result.params['t1'].value

    elif num_components == 2:
        t1, t2 = result.params['t1'].value, result.params['t2'].value
        amp1, amp2 = result.params['amp1'].value, result.params['amp2'].value
        if fixed.get('t1') is None and fixed.get('t2') is None:
            if t1 > t2:
                t1, t2 = t2, t1
                amp1, amp2 = amp2, amp1
        arrays["amp1"][i], arrays["t1"][i] = amp1, t1
        arrays["amp2"][i], arrays["t2"][i] = amp2, t2

    elif num_components == 3:
        taus = [result.params['t1'].value, result.params['t2'].value, result.params['t3'].value]
        amps = [result.params['amp1'].value, result.params['amp2'].value, result.params['amp3'].value]
        fixed_flags = [fixed.get('t1'), fixed.get('t2'), fixed.get('t3')]
        free_indices = [j for j, f in enumerate(fixed_flags) if f is None]
        if len(free_indices) > 1:
            free_pairs = sorted([(taus[j], amps[j]) for j in free_indices], key=lambda x: x[0])
            for k, idx in enumerate(free_indices):
                taus[idx], amps[idx] = free_pairs[k]
        for j, key in enumerate(['t1', 't2', 't3']):
            arrays[key][i] = taus[j]
            arrays[f"amp{j+1}"][i] = amps[j]



def _extract_result_dict(result, num_components, fit_shift, fixed):
    """Extract fitted parameters as a plain dict (for multiprocessing return)."""
    d = {"offset": result.params['offset'].value}
    if fit_shift:
        d["shift"] = result.params['shift'].value

    if num_components == 1:
        d["amp1"] = result.params['amp1'].value
        d["t1"] = result.params['t1'].value

    elif num_components == 2:
        t1, t2 = result.params['t1'].value, result.params['t2'].value
        amp1, amp2 = result.params['amp1'].value, result.params['amp2'].value
        if fixed.get('t1') is None and fixed.get('t2') is None:
            if t1 > t2:
                t1, t2 = t2, t1
                amp1, amp2 = amp2, amp1
        d["amp1"], d["t1"] = amp1, t1
        d["amp2"], d["t2"] = amp2, t2

    elif num_components == 3:
        taus = [result.params['t1'].value, result.params['t2'].value, result.params['t3'].value]
        amps = [result.params['amp1'].value, result.params['amp2'].value, result.params['amp3'].value]
        fixed_flags = [fixed.get('t1'), fixed.get('t2'), fixed.get('t3')]
        free_indices = [j for j, f in enumerate(fixed_flags) if f is None]
        if len(free_indices) > 1:
            free_pairs = sorted([(taus[j], amps[j]) for j in free_indices], key=lambda x: x[0])
            for k, idx in enumerate(free_indices):
                taus[idx], amps[idx] = free_pairs[k]
        for j, key in enumerate(['t1', 't2', 't3']):
            d[key] = taus[j]
            d[f"amp{j+1}"] = amps[j]

    return d


_worker_shared = {}


def _worker_init(params_dumps, irf, time_axis, start, end,
                 fitting_algo, fitting_mode, irf_upsampled, optimizers,
                 num_components, fit_shift, fixed):
    """Called once per worker process to store shared read-only data."""
    _worker_shared["params_dumps"] = params_dumps
    _worker_shared["irf"] = irf
    _worker_shared["time_axis"] = time_axis
    _worker_shared["start"] = start
    _worker_shared["end"] = end
    _worker_shared["fitting_algo"] = fitting_algo
    _worker_shared["fitting_mode"] = fitting_mode
    _worker_shared["irf_upsampled"] = irf_upsampled
    _worker_shared["optimizers"] = optimizers
    _worker_shared["num_components"] = num_components
    _worker_shared["fit_shift"] = fit_shift
    _worker_shared["fixed"] = fixed


def _fit_single_curve_worker(args):
    """Worker function for parallel fitting. Only receives (index, decay_curve)."""
    i, decay_curve = args
    s = _worker_shared

    current_params = Parameters()
    current_params.loads(s["params_dumps"])
    _set_amplitude_guesses(current_params, decay_curve, s["num_components"])

    try:
        result = _fit_single_curve(
            decay_curve, current_params, s["irf"], s["time_axis"],
            s["start"], s["end"], s["fitting_algo"], s["fitting_mode"],
            s["irf_upsampled"], s["optimizers"], seed=i)
        return (i, _extract_result_dict(result, s["num_components"], s["fit_shift"], s["fixed"]))
    except Exception as e:
        print(f"Error fitting curve {i}: {e}")
        return (i, None)


def fit_curves(duration, time_bins, decay_curves, irf, num_components, fitting_algo, fitting_mode="Hybrid", fit_shift=False, shift_guess=None, shift_halfwidth=None, start=0, end=-1, fixed_lifetimes=None, _progress_callback=None):
    """
    fixed_lifetimes: optional dict mapping 't1'/'t2'/'t3' to a fixed value in ns,
                     or None/0 to leave that component free.
                     Example: {'t1': 0.4, 't2': None}  → fix τ1, fit τ2 freely.
    shift_halfwidth: optional float, half-width of the shift parameter's bounds
                     in bins (bounds = [shift_guess ± halfwidth]). When None and
                     fit_shift=True, defaults to the IRF's FWHM in bins —
                     adapts to detector physics with no hardcoded constant.
    """
    num_curves = len(decay_curves)
    if fit_shift and shift_halfwidth is None:
        shift_halfwidth = irf_fwhm_bins(irf)
    params, arrays, fixed = _init_params(duration, time_bins, num_components, num_curves, fit_shift, shift_guess, shift_halfwidth, fixed_lifetimes)

    # A degenerate (all-zero) IRF makes the reconvolution model identically zero,
    # so the fit is meaningless; return the NaN-initialised results to flag it
    # explicitly instead of fitting a zero model to finite-but-garbage params.
    if irf is not None and not np.any(irf):
        return arrays

    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)

    optimizers = {
        "mle": "nelder",
        "mle_opts": {'maxfev': 100000, 'xatol': 1e-8, 'fatol': 1e-8},
        "wls": "leastsq",
        "wls_opts": {'max_nfev': 100000, 'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8},
        "global": "differential_evolution",
        "global_opts": {'popsize': 25, 'tol': 1e-8, 'max_nfev': 100000},
    }

    irf_upsampled = upsample_irf(irf) if fit_shift else None

    # Warm-start: fit the AVERAGE (mean) decay with Global to seed initial lifetimes
    # for Local mode. Use the mean, NOT the sum: lifetimes are scale-invariant so the
    # τ seeds are identical either way, but the summed decay's offset is ~N× a single
    # cell's (then clamped to the peak) and its peak sets ~N× the DE parameter bounds —
    # both leak batch size/composition into the ill-conditioned per-cell fits, making a
    # cell's result depend on how many cells share its batch. The mean keeps the warm
    # fit at single-cell scale, so the seeded offset and DE bounds are batch invariant.
    if fitting_mode == "Local" and num_curves >= 1:
        mean_decay = np.mean(np.array(decay_curves), axis=0)
        warm_params = params.copy()
        _set_amplitude_guesses(warm_params, mean_decay, num_components)
        try:
            # `seed` (not `rng`) is what lmfit forwards to differential_evolution;
            # required for a reproducible warm-start (see test_fit_determinism.py).
            warm_global_opts = dict(optimizers["global_opts"], seed=0)
            warm_result = lmfit_minimize(objective, warm_params, args=(mean_decay, irf, time_axis, start, end, fitting_algo, irf_upsampled), method=optimizers["global"], **warm_global_opts)
            for key in ['t1', 't2', 't3']:
                if key in params and params[key].vary:
                    params[key].value = warm_result.params[key].value
            if 'offset' in params and params['offset'].vary:
                params['offset'].value = warm_result.params['offset'].value
            if fit_shift and 'shift' in warm_result.params:
                params['shift'].value = warm_result.params['shift'].value
        except Exception:
            pass

    use_parallel = num_curves >= _MIN_CURVES_FOR_PARALLEL
    n_workers = min(psutil.cpu_count(logical=False) or (cpu_count() or 1), num_curves)
    ran_parallel = False

    if use_parallel and n_workers > 1:
        try:
            params_dumps = params.dumps()
            work_items = [(i, decay_curves[i]) for i in range(num_curves)]
            init_args = (params_dumps, irf, time_axis, start, end,
                         fitting_algo, fitting_mode, irf_upsampled, optimizers,
                         num_components, fit_shift, fixed)

            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=n_workers, maxtasksperchild=500,
                          initializer=_worker_init, initargs=init_args) as pool:
                for completed, (i, result_dict) in enumerate(
                    pool.imap_unordered(_fit_single_curve_worker, work_items, chunksize=1)
                ):
                    if _progress_callback:
                        _progress_callback(completed, num_curves)
                    if result_dict is not None:
                        for key, val in result_dict.items():
                            arrays[key][i] = val
            ran_parallel = True
        except Exception as e:
            print(f"Parallel fitting unavailable ({e}), falling back to sequential.")
            ran_parallel = False

    if not ran_parallel:
        for i in range(num_curves):
            decay_curve = decay_curves[i]

            if _progress_callback:
                _progress_callback(i, num_curves)

            current_params = params.copy()
            _set_amplitude_guesses(current_params, decay_curve, num_components)

            try:
                result = _fit_single_curve(decay_curve, current_params, irf, time_axis, start, end, fitting_algo, fitting_mode, irf_upsampled, optimizers, seed=i)
            except Exception as e:
                print(f"Error fitting curve {i}: {e}")
                continue

            _extract_result(result, arrays, i, num_components, fit_shift, fixed)

    return arrays
