import numpy as np
from lmfit import minimize as lmfit_minimize
from lmfit import Parameters
from src.fit_helper import objective

def fit_curves(duration, time_bins, decay_curves, irf, num_components, fitting_algo, fitting_mode="hybrid", fit_shift=False, shift_guess=None, start=0, end=-1, _progress_callback=None):
    
    num_curves = len(decay_curves)
    params = Parameters()
    # initialize the parameters
    amp1_data = np.zeros(num_curves)
    params.add('amp1', min=0)
    t1_data = np.zeros(num_curves)
    params.add('t1', value=0.400, min=0.0, max=duration)
    offset_data = np.zeros(num_curves)
    params.add('offset', min=0, max=1000000)
    if num_components > 1:
        amp2_data = np.zeros(num_curves)
        params.add('amp2', min=0)
        t2_data = np.zeros(num_curves)
        params.add('t2', value=2.5, min=0.0, max=duration)
        
    if num_components > 2:
        amp3_data = np.zeros(num_curves)
        params.add('amp3', min=0)
        t3_data = np.zeros(num_curves)
        params.add('t3', min=0.0, max=duration)

    if fit_shift:
        shift_data = np.zeros(num_curves)
        if shift_guess is None:
            shift_guess = 0
        params.add('shift', value=shift_guess, min=-100, max=100)
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
    mle_fit_options = { 'maxfev': 100000,      # Maximum function evaluations
            'xatol': 1e-8,        # Absolute parameter tolerance
            'fatol': 1e-8,        # Absolute objective tolerance
            'disp': True, } 
    mle_optimizer = "nelder"
    wls_optimizer = "leastsq"
    wls_fit_options = {
        'max_nfev': 100000,      # Maximum function evaluations
        'ftol':   1e-8,
        'xtol':   1e-8,
        'gtol':   1e-8,
    }
    global_optimizer = "differential_evolution"
    global_fit_options = {
        'popsize': 25,    # Population size
        'tol': 1e-8,      # Convergence tolerance
        'max_nfev': 10000   # Maximum function evaluations
    }
    for i in range(num_curves):
        decay_curve = decay_curves[i]
        
        # Update progress if callback is provided
        if _progress_callback:
            _progress_callback(i, num_curves)

        current_params = params.copy()
        current_params['amp1'].value = np.max(decay_curve) 
        current_params['amp1'].max = np.max(decay_curve) * 10
        if num_components > 1:
            current_params['amp2'].value = np.max(decay_curve) / 2
            current_params['amp2'].max = np.max(decay_curve) * 10
        if num_components > 2:
            current_params['amp3'].value = np.max(decay_curve) / 2
            current_params['amp3'].max = np.max(decay_curve) * 10
        try: 
            if fitting_mode != "Local":
                result_global = lmfit_minimize(objective, current_params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=global_optimizer, **global_fit_options)
            if fitting_algo == "MLE": 
                if fitting_mode == "Local":
                    result = lmfit_minimize(objective, current_params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=mle_optimizer, options=mle_fit_options)
                elif fitting_mode == "Hybrid":
                    result = lmfit_minimize(objective, result_global.params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=mle_optimizer, options=mle_fit_options)
                else: # global
                    result = result_global
            elif fitting_algo == "LS":
                if fitting_mode == "Local":
                    result = lmfit_minimize(objective, current_params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=wls_optimizer, **wls_fit_options)
                elif fitting_mode == "Hybrid":
                    result = lmfit_minimize(objective, result_global.params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=wls_optimizer, **wls_fit_options)
                else: # global
                    result = result_global
        except Exception as e:
            print(f"Error fitting curve {i}: {e}")
            result = None
            continue
        if fit_shift:
            shift_data[i] = result.params['shift'].value
        offset_data[i] = result.params['offset'].value
        if num_components == 1:
            amp1_data[i] = result.params['amp1'].value
            t1_data[i] = result.params['t1'].value
        elif num_components == 2:
            t1 = result.params['t1'].value
            t2 = result.params['t2'].value
            amp1 = result.params['amp1'].value
            amp2 = result.params['amp2'].value
            # make sure t1 is the shorter lifetime component and its amplitude
            if t1 > t2:
                t1, t2 = t2, t1
                amp1, amp2 = amp2, amp1
            amp1_data[i] = amp1
            t1_data[i] = t1
            amp2_data[i] = amp2
            t2_data[i] = t2
        elif num_components == 3:
            t1 = result.params['t1'].value
            t2 = result.params['t2'].value
            t3 = result.params['t3'].value
            amp1 = result.params['amp1'].value
            amp2 = result.params['amp2'].value
            amp3 = result.params['amp3'].value
            # sort the lifetimes and keep amplitudes aligned
            lifetime_amp_pairs = sorted([(t1, amp1), (t2, amp2), (t3, amp3)], key=lambda x: x[0])
            (t1, amp1), (t2, amp2), (t3, amp3) = lifetime_amp_pairs

            amp1_data[i] = amp1
            t1_data[i] = t1
            amp2_data[i] = amp2
            t2_data[i] = t2
            amp3_data[i] = amp3
            t3_data[i] = t3

    # assemble results dynamically
    results = {"amp1": amp1_data, "t1": t1_data, "offset": offset_data}
    
    if fit_shift:
        results["shift"] = shift_data
    if num_components > 1:
        results["amp2"] = amp2_data
        results["t2"] = t2_data
    if num_components > 2:
        results["amp3"] = amp3_data
        results["t3"] = t3_data
   
    return results




