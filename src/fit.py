import numpy as np
from lmfit import minimize as lmfit_minimize
from lmfit import Parameters
from src.sdt_io import read_sdt150
from src.fit_helper import guess_shift, objective
from src.file_io import load_image
import streamlit as st
import pandas as pd 

def create_progress_callback(progress_bar):
    def progress_callback(current, total):
        progress = (current + 1) / total
        progress_bar.progress(progress)
    return progress_callback

def fit_curves(duration, time_bins, decay_curves, irf, num_components, fitting_algo, fitting_mode="hybrid", fit_shift=False, shift_guess=None, start=0, end=-1, _progress_callback=None):
    
    # to make sure the irf is normalized
    irf = irf / np.sum(irf)
    num_curves = len(decay_curves)
    params = Parameters()
    # initialize the parameters
    amp1_data = np.zeros(num_curves)
    params.add('amp1', min=0)
    t1_data = np.zeros(num_curves)
    params.add('t1', value=0.400, min=0.100, max=1.0)
    offset_data = np.zeros(num_curves)
    params.add('offset', min=0, max=1000000)
    if num_components > 1:
        amp2_data = np.zeros(num_curves)
        params.add('amp2', min=0)
        t2_data = np.zeros(num_curves)
        params.add('t2', value=2.5, min=1.0, max=5.0)
        
    if num_components > 2:
        amp3_data = np.zeros(num_curves)
        params.add('amp3', min=0)
        t3_data = np.zeros(num_curves)
        params.add('t3', value=5.0, min=4.0, max=duration)

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
        if fitting_mode != "Local":
            result_global = lmfit_minimize(objective, current_params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=global_optimizer, **global_fit_options)
        if fitting_algo == "MLE": 
            if fitting_mode == "Local":
                result = lmfit_minimize(objective, current_params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=mle_optimizer, options=mle_fit_options)
            elif fitting_mode == "Hybrid":
                result = lmfit_minimize(objective, result_global.params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=mle_optimizer, options=mle_fit_options)
            else: # global
                result = result_global
        elif fitting_algo == "WLS":
            if fitting_mode == "Local":
                result = lmfit_minimize(objective, current_params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=wls_optimizer, **wls_fit_options)
            elif fitting_mode == "Hybrid":
                result = lmfit_minimize(objective, result_global.params, args=(decay_curve, irf, time_axis, start, end, fitting_algo), method=wls_optimizer, **wls_fit_options)
            else: # global
                result = result_global
        amp1_data[i] = result.params['amp1'].value
        t1_data[i] = result.params['t1'].value
        offset_data[i] = result.params['offset'].value
        if fit_shift:
            shift_data[i] = result.params['shift'].value
        if num_components > 1:
            amp2_data[i] = result.params['amp2'].value
            t2_data[i] = result.params['t2'].value
        if num_components > 2:
            amp3_data[i] = result.params['amp3'].value
            t3_data[i] = result.params['t3'].value
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


def roi_summing_choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, channel):
    error_msg = ""
    decay_curves = []
    for i, row in metadata_df.iterrows():
        image_name = row['image_name']
        mask_path = row.get('mask', None)
        if channel == "NADH":
            irf_path = row.get('nadh irf', None)
            decay_path = row.get('nadh decay', None)
        else:
            irf_path = row.get('fad irf', None)
            decay_path = row.get('fad decay', None)
        try:
            irf = np.loadtxt(irf_path)
        except Exception as e:
            error_msg = f"Error reading the IRF file for image {image_name} at {irf_path}: {e}"
            return error_msg, None
        try:
            mask = load_image(mask_path)
        except Exception as e:
            error_msg = f"Error reading the mask file for image {image_name} at {mask_path}: {e}"
            return error_msg, None
        try:
            decay = read_sdt150(decay_path)
        except Exception as e:
            error_msg = f"Error reading the decay file for image {image_name} at {decay_path}: {e}"
            return error_msg, None
        
        if len(irf) != time_bins:
            error_msg = f"IRF length mismatch with specified time bins. IRF length: {len(irf)}, time bins: {time_bins}."
            return error_msg, None
        if len(decay.shape) != 3:
            error_msg = f"Decay data mismatch. Expected 3D data (XYT), got {decay.shape}."
            return error_msg, None
        if decay.shape[2] != time_bins:
            error_msg = f"Decay time bins mismatch. Decay time bins: {decay.shape[2]}, time bins: {time_bins}."
            return error_msg, None
        if len(mask.shape) != 2:
            error_msg = f"Mask data mismatch. Expected 2D data, got {mask.shape}."
            return error_msg, None
        if decay.shape[0] != mask.shape[0] or decay.shape[1] != mask.shape[1]:
            error_msg = f"Dimension mismatch: Decay data {decay.shape[:2]} vs mask {mask.shape}"
            return error_msg, None
        
        # binarize the mask
        binary_mask = np.where(mask > 0, 1, 0)
        # image_level ROI summing: sum the time axis of all non-zero pixels in the decay
        summed_decay_curve = np.sum(decay * binary_mask[:, :, np.newaxis], axis=(0, 1))
        decay_curves.append(summed_decay_curve)
    original_decay_curves = decay_curves.copy()
    decay_curves = _floor_decay_curves(decay_curves)
    shift_guess = guess_shift(irf, decay_curves)
    
    # Create progress callback for shift estimation
    st.info(f"Estimating shifts for {channel} channel across {len(decay_curves)} images...")
    shift_progress = st.progress(0)
    
    shift_progress_callback = create_progress_callback(shift_progress)
    
    results = fit_curves(duration, time_bins, decay_curves, irf, num_components, fitting_algo, fitting_mode, fit_shift=True, shift_guess=shift_guess, start=0, end=-1, _progress_callback=shift_progress_callback)
    shift_progress.empty()  # Remove progress bar when done
    
    results["decay_curves"] = decay_curves
    results["original_decay_curves"] = original_decay_curves
    results["fitted_images"] = metadata_df['image_name'].values
    results["irf"] = irf
    return error_msg, results

@st.cache_data
def choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, analysis_type, channel):
    if analysis_type == "ROI Summing Fit":
        error_msg, results = roi_summing_choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, channel)
    elif analysis_type == "K-Flow":
        error_msg, results = k_flow_choose_shift(metadata_df.iloc[0], duration, time_bins, num_components, fitting_algo, fitting_mode, channel)
    return error_msg, results 


def k_flow_choose_shift(metadata, duration, time_bins, num_components, fitting_algo, fitting_mode, channel, num_samples=20, max_intensity=100000):
    error_msg = ""
    kflow_exp_name = metadata['kflow_exp_name']
    if channel == "NADH":
        decay_path = metadata.get('nadh histogram', None)
        irf_path = metadata.get('nadh irf', None)
    elif channel == "FAD":
        decay_path = metadata.get('red histogram', None)
        irf_path = metadata.get('red irf', None)
    try: 
        decays = pd.read_csv(decay_path)
    except Exception as e:
        error_msg = f"Error reading the decay histogram for image {kflow_exp_name} at {decay_path}: {e}"
        return error_msg, None
    try:
        irf = np.loadtxt(irf_path)
    except Exception as e:
        error_msg = f"Error reading the IRF file for image {kflow_exp_name} at {irf_path}: {e}"
        return error_msg, None
    # check the dimension
    if decays.ndim != 2 or decays.shape[1] != time_bins:
        return f"The dimension of the decay histogram for {kflow_exp_name} at {decay_path} is not correct. Expected 2D data (XYT), and T = {time_bins}, got {decays.shape}."
    if len(irf) != time_bins:
        return f"The dimension of the IRF for {kflow_exp_name} at {irf_path} is not correct. Expected 1D data (T), and T = {time_bins}, got {len(irf)}."
    # get sample decay curves
    sample_decays = _get_sample_decay_curves(decays, num_samples, max_intensity)
    original_sample_decays = sample_decays.copy()
    sample_decays = _floor_decay_curves(sample_decays)
    # get the shift guess
    shift_guess = guess_shift(irf, sample_decays)
    # get the shift progress bar
    st.info(f"Estimating shifts for {channel} channel for {kflow_exp_name} using {len(sample_decays)} sample curves...")
    shift_progress = st.progress(0)
    shift_progress_callback = create_progress_callback(shift_progress)
    results = fit_curves(duration, time_bins, sample_decays, irf, num_components, fitting_algo, fitting_mode, fit_shift=True, shift_guess=shift_guess, _progress_callback=shift_progress_callback)
    shift_progress.empty()  # Remove progress bar when done
    results["decay_curves"] = sample_decays
    results["original_decay_curves"] = original_sample_decays
    results["irf"] = irf
    return error_msg, results 

def _floor_decay_curves(decay_curves):
    for i, decay_curve in enumerate(decay_curves):
        # find the minimum value non-zero value from the start of the decay curve
        min_value = np.min(decay_curve[decay_curve > 0])
        decay_curves[i] = decay_curve - min_value
        # clip the decay curve to be non-negative
        decay_curves[i] = np.clip(decay_curves[i], 0, None)
    return decay_curves

def _get_sample_decay_curves(decays, n_samples, max_intensity):
    # use the top n_samples that have the highest intensity less than max_intensity
    # return the decay curves
    decay_intensity = np.sum(decays, axis=1)
    sorted_indices = np.argsort(decay_intensity)[::-1]
    filtered_indices = [idx for idx in sorted_indices if decay_intensity[idx] < max_intensity]

    if len(filtered_indices) < n_samples:
        top_indices = filtered_indices
    else:
        top_indices = filtered_indices[:n_samples]
    return decays.iloc[top_indices].values