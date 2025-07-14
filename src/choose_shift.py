import numpy as np
import pandas as pd
import streamlit as st
from src.fit import fit_curves
from src.fit_helper import create_progress_callback
from src.file_io import get_decay_curves_shift


def guess_shift(irf, curves, fit_free=False):
    # if fit_free, return the shifts as they are, otherwise return the median shift as the initialization to the fit routine
    def align_irf(irf, curve):
        # Cross-correlation to find the optimal shift
        correlation = np.correlate(curve, irf, mode='full')
        shift = np.argmax(correlation) - (len(irf) - 1)
        return shift
    shifts = []
    for i in range(len(curves)):
        shift = align_irf(irf, curves[i])
        shifts.append(shift)
    if fit_free:
        return shifts
    else:
        return np.median(shifts)

def choose_shift_fit_free(metadata_df, time_bins, input_type, channel_name):

    error_msg, decay_curves, irf = get_decay_curves_shift(metadata_df, input_type, channel_name, time_bins)
    if error_msg != "":
        return error_msg, None

    shift_guess = guess_shift(irf, decay_curves, fit_free=True)
    results = {"shift": shift_guess, "decay_id": decay_curves.keys()}
    return error_msg, results


@st.cache_data
def choose_shift_fit(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, input_type, channel_name):
    error_msg, decay_curves, irf = get_decay_curves_shift(metadata_df, input_type, channel_name, time_bins)
    if error_msg != "":
        return error_msg, None
    shift_guess = guess_shift(irf, decay_curves.values())
    sample_decays = _floor_decay_curves(decay_curves.values())
    # get the shift progress bar
    st.info(f"Estimating shifts for {channel_name} channel using {len(decay_curves)} sample curves...")
    shift_progress = st.progress(0)
    shift_progress_callback = create_progress_callback(shift_progress)
    results = fit_curves(duration, time_bins, sample_decays, irf, num_components, fitting_algo, fitting_mode, fit_shift=True, shift_guess=shift_guess, _progress_callback=shift_progress_callback)
    shift_progress.empty()  # Remove progress bar when done
    results["decay_curves"] = sample_decays
    results["original_decay_curves"] = decay_curves.values()
    results["decay_id"] = decay_curves.keys()
    results["irf"] = irf

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

