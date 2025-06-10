import streamlit as st
import numpy as np
from src.feature_groups import feature_groups_prefix
"""
This file contains widgets that are specific to certain visualization or classification methods."""

def umap_hyperParams_widget():
    col1, col2 = st.columns(2)
    # First number incrementor in the first column
    with col1:
        n_neighbors = st.number_input(
            "n_neighbors",
            value=15,  # Initial value
            step=5,             # Increment/Decrement step
            format="%d"            # Integer format
        )

    # Second number incrementor in the second column
    with col2:
        min_dist = st.number_input(
            "min_dist",
            value=0.1,  # Initial value
            step=0.1,            
        )

    return n_neighbors, min_dist

def stats_comparison_pair_widget(available_pairs):
    selected_pairs = st.multiselect(
        "Select statistical tests compare pairs",
        available_pairs,
        default=available_pairs,
        key="compare_pairs"
    )
    return selected_pairs


def histogram_bin_width_widget(x_data): 
    # x_data: 1D numpy array of data to be binned (already na dropped)
    # use np's automatic binning logic by not specifying nbins explicitly in np.histogram
    # Calculate bins using numpy based on the overall data range
    counts_all, bin_edges_all = np.histogram(x_data, bins='auto') # Use 'auto' as default
    nbins = len(bin_edges_all) - 1
    if nbins > 1:
        default_bin_width = bin_edges_all[1] - bin_edges_all[0]
        # add a widget to adjust the bin_width, use text input to get the bin width
        # Ensure bin_width is positive to avoid errors in np.arange
        min_val = x_data.min()
        max_val = x_data.max()
        range = max_val - min_val
        bin_width = st.number_input(label="Bin Width", min_value=0.01, max_value=range/3, value=default_bin_width, step=range/50,)
        # Add a small epsilon to max_val to ensure the rightmost edge includes the max value
        epsilon = 1e-9
        # Calculate common bin edges based on the user-provided bin_width
        common_bin_edges = np.arange(min_val, max_val + bin_width + epsilon, bin_width)
    
    return common_bin_edges

def gmm_hyperParams_widget():
    col3, col4 = st.columns(2)
    with col3:
        fit_gmm_max_components = st.slider("Max Components", min_value=2, max_value=5, value=3, step=1) 
    with col4:
        fit_gmm_min_weight_threshold = st.slider("Min Weight Threshold", min_value=0.0, max_value=0.3, value=0.1, step=0.1)
    return fit_gmm_max_components, fit_gmm_min_weight_threshold

def phasor_params_widget(feature_cols_dict):

    available_harmonics = {}
    if "Fit Free Nadh" in feature_cols_dict and len(feature_cols_dict["Fit Free Nadh"]) > 0:
        nadh_fit_free_features_prefix = feature_groups_prefix["Fit Free Nadh"]
        nadh_fit_free_features = feature_cols_dict["Fit Free Nadh"]
        available_harmonics["Nadh"] = []
        if f"{nadh_fit_free_features_prefix}G(1st)" in nadh_fit_free_features and f"{nadh_fit_free_features_prefix}S(1st)" in nadh_fit_free_features:
            available_harmonics["Nadh"].append(1)
        if f"{nadh_fit_free_features_prefix}G(2nd)" in nadh_fit_free_features and f"{nadh_fit_free_features_prefix}S(2nd)" in nadh_fit_free_features:
            available_harmonics["Nadh"].append(2)
    if "Fit Free Fad" in feature_cols_dict and len(feature_cols_dict["Fit Free Fad"]) > 0:

        fad_fit_free_features_prefix = feature_groups_prefix["Fit Free Fad"]
        fad_fit_free_features = feature_cols_dict["Fit Free Fad"]
        available_harmonics["Fad"] = []
        if f"{fad_fit_free_features_prefix}G(1st)" in fad_fit_free_features and f"{fad_fit_free_features_prefix}S(1st)" in fad_fit_free_features:
            available_harmonics["Fad"].append(1)    
        if f"{fad_fit_free_features_prefix}G(2nd)" in fad_fit_free_features and f"{fad_fit_free_features_prefix}S(2nd)" in fad_fit_free_features:
            available_harmonics["Fad"].append(2)
    
    if len(available_harmonics.keys()) > 1:
        selected_channel = st.selectbox("Select a channel", available_harmonics.keys())
    elif len(available_harmonics.keys()) == 1:
        selected_channel = list(available_harmonics.keys())[0]
    else:
        selected_channel = None
        st.warning("No available channels found for phasor plot")
    selected_harmonic = None
    for channel in available_harmonics.keys():
        if available_harmonics[channel] == []:
            st.warning(f"No available harmonics found for {channel}")     
        elif len(available_harmonics[channel]) == 1:
            selected_harmonic = available_harmonics[channel][0]
        else:
            selected_harmonic = st.selectbox(f"Select a harmonic for {channel}", available_harmonics[channel])
    f = None
    if selected_channel is not None and selected_harmonic is not None:
        if selected_harmonic == 1:
            harmonic_str = "1st"    
        elif selected_harmonic == 2:
            harmonic_str = "2nd"
        f = st.number_input(f"Enter the laser repetition rate in GHz for {selected_channel} {harmonic_str} harmonic", value=0.08, min_value=0.0, step=0.01)
    return selected_channel, selected_harmonic, f
