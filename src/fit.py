import numpy as np
from lmfit import minimize as lmfit_minimize
from lmfit import Parameters
from src.fit_helper import objective, upsample_irf


def _init_params(duration, time_bins, num_components, num_curves, fit_shift, shift_guess, fixed_lifetimes):
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
        params.add('shift', value=shift_guess or 0, min=-time_bins / 2, max=time_bins / 2)

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


def _fit_single_curve(decay_curve, current_params, irf, time_axis, start, end, fitting_algo, fitting_mode, irf_upsampled, optimizers):
    """Run the appropriate optimizer(s) on a single decay curve."""
    args = (decay_curve, irf, time_axis, start, end, fitting_algo, irf_upsampled)

    if fitting_mode != "Local":
        result_global = lmfit_minimize(objective, current_params, args=args, method=optimizers["global"], **optimizers["global_opts"])

    if fitting_algo == "MLE":
        if fitting_mode == "Local":
            return lmfit_minimize(objective, current_params, args=args, method=optimizers["mle"], options=optimizers["mle_opts"])
        elif fitting_mode == "Hybrid":
            return lmfit_minimize(objective, result_global.params, args=args, method=optimizers["mle"], options=optimizers["mle_opts"])
        else:
            return result_global
    elif fitting_algo == "LS":
        if fitting_mode == "Local":
            return lmfit_minimize(objective, current_params, args=args, method=optimizers["ls"], **optimizers["ls_opts"])
        elif fitting_mode == "Hybrid":
            return lmfit_minimize(objective, result_global.params, args=args, method=optimizers["ls"], **optimizers["ls_opts"])
        else:
            return result_global
    else:
        raise ValueError(f"Unsupported fitting algorithm: {fitting_algo}. Use 'MLE' or 'LS'.")


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



def fit_curves(duration, time_bins, decay_curves, irf, num_components, fitting_algo, fitting_mode="hybrid", fit_shift=False, shift_guess=None, start=0, end=-1, fixed_lifetimes=None, _progress_callback=None):
    """
    fixed_lifetimes: optional dict mapping 't1'/'t2'/'t3' to a fixed value in ns,
                     or None/0 to leave that component free.
                     Example: {'t1': 0.4, 't2': None}  → fix τ1, fit τ2 freely.
    """
    num_curves = len(decay_curves)
    params, arrays, fixed = _init_params(duration, time_bins, num_components, num_curves, fit_shift, shift_guess, fixed_lifetimes)

    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)

    optimizers = {
        "mle": "nelder",
        "mle_opts": {'maxfev': 100000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True},
        "ls": "leastsq",
        "ls_opts": {'max_nfev': 100000, 'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8},
        "global": "differential_evolution",
        "global_opts": {'popsize': 25, 'tol': 1e-8, 'max_nfev': 100000},
    }

    irf_upsampled = upsample_irf(irf) if fit_shift else None

    # Warm-start: fit the summed decay with Global to get better initial lifetimes for Local mode
    if not fit_shift and fitting_mode == "Local" and num_curves > 1:
        summed_decay = np.sum(np.array(decay_curves), axis=0)
        warm_params = params.copy()
        _set_amplitude_guesses(warm_params, summed_decay, num_components)
        try:
            warm_result = lmfit_minimize(objective, warm_params, args=(summed_decay, irf, time_axis, start, end, fitting_algo, irf_upsampled), method=optimizers["global"], **optimizers["global_opts"])
            for key in ['t1', 't2', 't3']:
                if key in params and params[key].vary:
                    params[key].value = warm_result.params[key].value
            if 'offset' in params and params['offset'].vary:
                params['offset'].value = warm_result.params['offset'].value
        except Exception:
            pass

    for i in range(num_curves):
        decay_curve = decay_curves[i]

        if _progress_callback:
            _progress_callback(i, num_curves)

        current_params = params.copy()
        _set_amplitude_guesses(current_params, decay_curve, num_components)

        try:
            result = _fit_single_curve(decay_curve, current_params, irf, time_axis, start, end, fitting_algo, fitting_mode, irf_upsampled, optimizers)
        except Exception as e:
            print(f"Error fitting curve {i}: {e}")
            continue

        _extract_result(result, arrays, i, num_components, fit_shift, fixed)

    return arrays
