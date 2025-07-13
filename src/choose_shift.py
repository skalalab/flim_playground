import numpy as np
import pandas as pd
import streamlit as st
from src.fit import fit_curves
from src.fit_helper import create_progress_callback
from src.file_io import load_image
from src.sdt_io import read_sdt


def guess_shift(irf, curves):
    def align_irf(irf, curve):
        # Cross-correlation to find the optimal shift
        correlation = np.correlate(curve, irf, mode='full')
        shift = np.argmax(correlation) - (len(irf) - 1)
        return shift
    shifts = []
    for i in range(len(curves)):
        shift = align_irf(irf, curves[i])
        shifts.append(shift)
        
    shift_guess = np.median(shifts)
    return shift_guess
def roi_summing_choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, channel_name):
    error_msg = ""
    decay_curves = []
    for i, row in metadata_df.iterrows():
        image_name = row['image_name']
        mask_path = row.get('mask', None)
        if channel == "NADH":
            irf_path = row.get('nadh irf', None)
            decay_path = row.get('nadh decay', None)
            channel_no = row.get('nadh_channel', None)
        else:
            irf_path = row.get('fad irf', None)
            decay_path = row.get('fad decay', None)
            channel_no = row.get('fad_channel', None)
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
            decay = read_sdt(decay_path, channel=channel_no)
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
def choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, input_type, channel_name):
    if input_type == "ROI Summing Fit" or input_type == "SPCImage":
        error_msg, results = roi_summing_choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, channel_name)
    elif input_type == "K-Flow":
        error_msg, results = k_flow_choose_shift(metadata_df.iloc[0], duration, time_bins, num_components, fitting_algo, fitting_mode, channel_name)
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
        try:
            min_value = np.min(decay_curve[decay_curve > 0])
        except Exception as e:
            min_value = 0
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