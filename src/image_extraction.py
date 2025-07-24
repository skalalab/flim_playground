import pandas as pd
import numpy as np
from skimage.measure import regionprops
from src.file_io import get_decay_curves, load_image
from src.fit import fit_curves
from src.fit_helper import create_progress_callback, irf_shift
from src.fit_free import get_phasor_features
import streamlit as st

def get_offset(decay_curve):
    """
    Get the offset of a decay curve by taking the minimum of:
    - Median of the first 20% of bins
    - Median of the last 10% of bins
    """
    head_bins_percentile = 20
    tail_bins_percentile = 90
    
    # Calculate the number of bins for each segment
    total_bins = len(decay_curve)
    head_bins = int(total_bins * head_bins_percentile / 100)
    tail_start_bin = int(total_bins * tail_bins_percentile / 100)
    
    # Get the first 20% of bins and calculate median
    head_segment = decay_curve[:head_bins]
    head_median = np.median(head_segment)
    
    # Get the last 10% of bins and calculate median  
    tail_segment = decay_curve[tail_start_bin:]
    tail_median = np.median(tail_segment)
    
    # Return the minimum of the two medians
    return min(head_median, tail_median)

def get_intensity_morphology_features(metadata, channel_name, fov_col_name):
    # get mask morphology features
    mask_morphology_features = ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length", "circularity"]
    mask = load_image(metadata[f"{channel_name}_Mask"])
    mask_props = regionprops(label_image=mask)
    fov_name = metadata[fov_col_name]
    single_cell_features_fov = {}
    for region in mask_props:
        cell_id = f"{fov_name}_{region.label}"
        if cell_id not in single_cell_features_fov:
            single_cell_features_fov[cell_id] = {}
        # Add centroid x and y: image data is indexed in NumPy and most image processing libraries in "reverse"
        single_cell_features_fov[cell_id]['centroid_x'] = region.centroid[1]
        single_cell_features_fov[cell_id]['centroid_y'] = region.centroid[0]
        for feature in mask_morphology_features:
            feature_name = f"{feature}"
            if feature in region:
                single_cell_features_fov[cell_id][feature_name] = region[feature]
            elif feature == "circularity":
                single_cell_features_fov[cell_id][feature_name] = 4 * np.pi * region.area / region.perimeter**2 if region.perimeter > 0 else 0
    return single_cell_features_fov

def spcimage_fit_extraction(metadata, channel_name, num_components, fov_colname):
    fit_feature_prefix = f"Lifetime fit_{channel_name}: "
    image_props = {}
    try:
        a1 = load_image(metadata[f"{channel_name}_a1"])
        # SPC image will output 0 for the thresholded pixels (background), so we need to mask them
        a1 = np.ma.masked_array(a1, mask=a1==0, fill_value=np.nan)
    except Exception as e:
        return f"Error reading the {channel_name} a1 file: {metadata[f'{channel_name}_a1']}: {e}", None
    try:
        mask = load_image(metadata[f"{channel_name}_Mask"])
    except Exception as e:
        return f"Error reading the {channel_name} mask file: {metadata[f'{channel_name}_Mask']}: {e}", None
    if mask.shape != a1.shape:
        return f"Error: {channel_name} a1 file has a different shape than the mask file: {a1.shape} != {mask.shape}", None
    try:
        t1 = load_image(metadata[f"{channel_name}_t1"])
        t1 = np.ma.masked_array(t1, mask=t1==0, fill_value=np.nan)
    except Exception as e:
        return f"Error reading the {channel_name} t1 file: {metadata[f'{channel_name}_t1']}: {e}", None
    if mask.shape != t1.shape:
        return f"Error: {channel_name} t1 file has a different shape than the mask file: {t1.shape} != {mask.shape}", None
    
    image_props[f"{fit_feature_prefix}a1"] = regionprops(label_image=mask, intensity_image=a1)
    image_props[f"{fit_feature_prefix}t1"] = regionprops(label_image=mask, intensity_image=t1)
    tm = a1 * t1

    if num_components == 2:
        try:
            t2 = load_image(metadata[f"{channel_name}_t2"])
            t2 = np.ma.masked_array(t2, mask=t2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the {channel_name} t2 file: {metadata[f'{channel_name}_t2']}: {e}", None
        if mask.shape != t2.shape:
            return f"Error: {channel_name} t2 file has a different shape than the mask file: {t2.shape} != {mask.shape}", None
        tm= (a1 / 100 * t1) + ((100 - a1) / 100 * t2)
        image_props[f"{fit_feature_prefix}t2"] = regionprops(label_image=mask, intensity_image=t2)
    
    elif num_components == 3:
        try:
            a2 = load_image(metadata[f"{channel_name}_a2"])
            a2 = np.ma.masked_array(a2, mask=a2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the {channel_name} a2 file: {metadata[f'{channel_name}_a2']}: {e}", None
        if mask.shape != a2.shape:
            return f"Error: {channel_name} a2 file has a different shape than the mask file: {a2.shape} != {mask.shape}", None
        try:
            t3 = load_image(metadata[f"{channel_name}_t3"])
            t3 = np.ma.masked_array(t3, mask=t3==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the {channel_name} t3 file: {metadata[f'{channel_name}_t3']}: {e}", None
        if mask.shape != t3.shape:
            return f"Error: {channel_name} t3 file has a different shape than the mask file: {t3.shape} != {mask.shape}", None
        tm= (a1 / 100 * t1) + ((100 - a1) / 100 * t2) + ((100 - a1 - a2) / 100 * t3)
        image_props[f"{fit_feature_prefix}a2"] = regionprops(label_image=mask, intensity_image=a2)
        image_props[f"{fit_feature_prefix}t3"] = regionprops(label_image=mask, intensity_image=t3)
    image_props[f"{fit_feature_prefix}tm"] = regionprops(label_image=mask, intensity_image=tm)

    image_name = metadata[fov_colname]
    single_cell_features_img = {}
    for prop in image_props:
        for region in image_props[prop]:
            cell_id = f"{image_name}_{region.label}"
            if cell_id not in single_cell_features_img:
                single_cell_features_img[cell_id] = {}
            single_cell_features_img[cell_id][prop] = region.intensity_mean
            single_cell_features_img[cell_id][f"{prop}_stdev"] = region.intensity_std
         
   # convert single_cell_features_img to a dataframe
    single_cell_fit_features_fov = pd.DataFrame(single_cell_features_img).T
    if single_cell_fit_features_fov.empty:
        return "Error: No cells found in the mask", None
   
    return "", single_cell_fit_features_fov

def extract_fit_results(channel_name, decay_curves, results, num_components):
    """
    Extract fitting results for a specific channel and store them in single_cell_features_img
    
    Args:
        channel_name
        cell_ids: list of cell ids
        single_cell_features_img: dictionary of single cell features
        results: fitting result dictionary for one cell   
        num_components: number of fitting components
    """
    single_cell_features_fov = {}
    fit_feature_prefix = f"Lifetime fit_{channel_name}: "

    for i, cell_id in enumerate(decay_curves.keys()):
        if cell_id not in single_cell_features_fov:
            single_cell_features_fov[cell_id] = {}
        # amplitutes and offsets are just bookkeeping, should be default to uncategorized features (i.e. without prefix)
        single_cell_features_fov[cell_id][f"{channel_name}_amp1"] = results["amp1"][i]
        single_cell_features_fov[cell_id][f"{fit_feature_prefix}t1"] = results["t1"][i] * 1000  # Convert to ps
        single_cell_features_fov[cell_id][f"{channel_name}_offset"] = results["offset"][i]
        if num_components == 2:
            single_cell_features_fov[cell_id][f"{channel_name}_amp2"] = results["amp2"][i]
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}t2"] = results["t2"][i] * 1000  # Convert to ps
            # Calculate alpha values
            amp1, amp2 = results["amp1"][i], results["amp2"][i]
            total_amp = amp1 + amp2
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}a1"] = (amp1 / total_amp) * 100
            # single_cell_features_fov[cell_id][f"{channel_name}_a2"] = (amp2 / total_amp) * 100
            # Calculate mean lifetime (in original units, not converted)
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}tm"] = ((amp1 / total_amp) * results["t1"][i] + (amp2 / total_amp) * results["t2"][i]) * 1000
            
        elif num_components == 3:
            single_cell_features_fov[cell_id][f"{channel_name}_amp2"] = results["amp2"][i]
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}t2"] = results["t2"][i] * 1000  # Convert to ps
            single_cell_features_fov[cell_id][f"{channel_name}_amp3"] = results["amp3"][i]
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}t3"] = results["t3"][i] * 1000  # Convert to ps
            # Calculate alpha values for 3 components
            amp1, amp2, amp3 = results["amp1"][i], results["amp2"][i], results["amp3"][i]
            total_amp = amp1 + amp2 + amp3
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}a1"] = (amp1 / total_amp) * 100
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}a2"] = (amp2 / total_amp) * 100
           # single_cell_features_fov[cell_id][f"{lifetime_feature_prefix}_a3"] = (amp3 / total_amp) * 100
            # Calculate mean lifetime for 3 components (in original units, not converted)
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}tm"] = ((amp1 / total_amp) * results["t1"][i] + (amp2 / total_amp) * results["t2"][i] + (amp3 / total_amp) * results["t3"][i]) * 1000

    return single_cell_features_fov

def extract_fit_free_results(channel_name, decay_curves, shifted_irf, time_axis, laser_rate):
    """
    Extract fit free results for a specific channel and store them in single_cell_features_img (for now, only phasor is implemented)
    Args:
        channel_name
        decay_curves: dictionary of decay curves: key is cell_id, value is decay curve
        shifted_irf: shifted IRF
        time_axis: time axis
        laser_rate: laser repetition rate
    """
    fit_free_feature_prefix = f"Lifetime fit free_{channel_name}: "
    single_cell_features_fov = {}
    for cell_id in decay_curves.keys():
        if cell_id not in single_cell_features_fov:
            single_cell_features_fov[cell_id] = {}
        
        # get the offset for this curve
        offset = get_offset(decay_curves[cell_id])
        # 1st harmonic
        g1, s1, g2, s2, tau_phase, tau_m = get_phasor_features(decay_curves[cell_id], shifted_irf, time_axis, f=laser_rate, offset=offset)
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}G(1st)"] = g1
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}S(1st)"] = s1
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}Tau_phase"] = tau_phase
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}Tau_m"] = tau_m
        # 2nd harmonic
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}G(2nd)"] = g2
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}S(2nd)"] = s2

    return single_cell_features_fov

@st.cache_data
def extract_lifetime_features(metadata, channel_name, input_type, fit, fit_free, metadata_dict):
    need_to_fit = False
    time_bins = metadata["time_bins"]
    duration = metadata["duration"]
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
    if "prefitted" not in input_type or fit_free:
        # get the decay curves and irf
        error_msg, decay_curves, irf = get_decay_curves(metadata, input_type, channel_name, time_bins, shift=False)
        shift = metadata[f"{channel_name}_shift"]
        shifted_irf = irf_shift(irf, shift)
        if error_msg != "":
            return error_msg, None
    if fit:
        num_components = metadata_dict[channel_name]["num_components"]
        if "prefitted" not in input_type:
            fitting_algo = metadata_dict["fitting_algo"]
            fitting_mode = metadata_dict["fitting_mode"]
            start = metadata_dict[channel_name]["start"]
            end = metadata_dict[channel_name]["end"]
            need_to_fit = True
        else: # prefitted
            single_cell_fit_features_fov = spcimage_fit_extraction(metadata, channel_name, num_components, metadata_dict["fov_name_col"])

    channel_container = st.empty()
    with channel_container.container():
        st.info(f"Extracting Lifetime Features (fitting/fit free) for {channel_name}: for {len(decay_curves)} cells...")
        channel_progress = st.progress(0)
    
    channel_progress_callback = create_progress_callback(channel_progress)
    if need_to_fit:
        results = fit_curves(duration, time_bins, list(decay_curves.values()), shifted_irf, num_components, fitting_algo, fitting_mode, start=start, end=end, _progress_callback=channel_progress_callback)
        single_cell_fit_features_fov = extract_fit_results(channel_name, decay_curves, results, num_components)
        # convert to dataframe
        single_cell_fit_features_fov = pd.DataFrame.from_dict(single_cell_fit_features_fov, orient='index')
    channel_container.empty()  # Remove both text and progress bar when done
    if fit_free:
        laser_rate = metadata_dict["laser_rate"]
        single_cell_fit_free_features_fov = extract_fit_free_results(channel_name, decay_curves, shifted_irf, time_axis, laser_rate)
        single_cell_fit_free_features_fov = pd.DataFrame.from_dict(single_cell_fit_free_features_fov, orient='index')
    if fit and fit_free:
        return pd.concat([single_cell_fit_features_fov, single_cell_fit_free_features_fov], axis=1)
    elif fit:
        return single_cell_fit_features_fov
    elif fit_free:
        return single_cell_fit_free_features_fov

def fov_extraction(metadata, metadata_dict):
    """
    Extract single cell features from one fov
    """
    fov_name = metadata[metadata_dict["fov_name_col"]]
    # unique cell id colname
    unique_cell_id_colname = metadata_dict["unique_cell_id_col"]
    # Collect DataFrames from each channel
    fov_feature_dfs = []
    for channel_name in metadata_dict["channel_names"]:
        input_type = metadata_dict[channel_name]["input_type"]
        selected_feature_extractors = metadata_dict[channel_name]["selected_feature_extractors"]
        if metadata_dict[channel_name]["imaging_modality"] == "FLIM":  
            fit = "Lifetime fit" in selected_feature_extractors
            fit_free = "Lifetime fit free" in selected_feature_extractors
            if fit or fit_free:
                single_cell_lifetime_features = extract_lifetime_features(metadata, channel_name, input_type, fit, fit_free, metadata_dict)
                fov_feature_dfs.append(single_cell_lifetime_features)
    
    # Combine all channel DataFrames in one operation
    single_cell_features_fov = pd.concat(fov_feature_dfs, axis=1) if fov_feature_dfs else pd.DataFrame()
    if not single_cell_features_fov.empty:
        single_cell_features_fov[metadata_dict["fov_name_col"]] = fov_name
        single_cell_features_fov.index.name = unique_cell_id_colname
    else:
        return f"Error: No cells found in the {fov_name}", None

    return "", single_cell_features_fov
            