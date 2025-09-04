import pandas as pd
import numpy as np
from skimage.measure import regionprops
from src.file_io import get_decay_curves, load_image, get_irf
from src.decay_io import read_decay
from src.fit import fit_curves
from src.fit_helper import create_progress_callback, irf_shift
import streamlit as st
from phasorpy import phasor, lifetime
from src.cell_texture import granularity, radial_distribution, mass_displacement

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

def get_intensity_texture_features(metadata, channel_name, fov_col_name, mask, input_type):
    feature_prefix = f"Intensity texture_{channel_name}: "
    intensity_texture_features = ["intensity_sum", "granularity", "radial_distribution", "mass_displacement"]
    granularity_values = [1,3,5,7]
    radial_distribution_values = [1,2,3,4]
    fov_name = metadata[fov_col_name]
    single_cell_texture_features_fov = {}
    # get cell ids from the mask
    mask_ids = np.unique(mask)
    mask_ids = mask_ids[mask_ids != 0]
    # get the intensity image from the metadata
    if input_type == "Intensity (2D)":
        try:
            intensity_image = load_image(metadata[f"{channel_name}_Intensity (2D)"])
        except Exception as e:
            return f"Error reading the {channel_name} intensity image: {metadata[f'{channel_name}_Intensity (2D)']}: {e}", pd.DataFrame()
    elif "Decay (3/4D)" in input_type:
        try:
            decay_path = metadata[f"{channel_name}_Decay"]
            channel_no = metadata[f"{channel_name}_channel"]
            error_msg, decay = read_decay(decay_path, channel_no)
            if error_msg != "":
                return error_msg, pd.DataFrame()
            if len(decay.shape) != 3:
                return f"Error: {channel_name} decay file is not a 3D array", pd.DataFrame()
        except Exception as e:
            return f"Error reading the {channel_name} decay file: {metadata[f'{channel_name}_Decay']}: {e}", pd.DataFrame()
        intensity_image = np.sum(decay, axis=-1)
    if intensity_image.shape != mask.shape:
        return f"Error: {channel_name} intensity image has a different shape than the mask: {intensity_image.shape} != {mask.shape}", pd.DataFrame()
    for mask_id in mask_ids:
        cell_id = f"{fov_name}_{mask_id}"
        single_cell_texture_features_fov[cell_id] = {}
        cell_mask = mask == mask_id
        cell_image = intensity_image * cell_mask
        for feature in intensity_texture_features:
            if feature == "intensity_sum":
                single_cell_texture_features_fov[cell_id][f"{feature_prefix}{feature}"] = np.sum(cell_image)
            elif feature == "granularity":
                for n in granularity_values:
                    single_cell_texture_features_fov[cell_id][f"{feature_prefix}{feature}_{n}"] = granularity(cell_image, n)
            elif feature == "radial_distribution":
                for ring_number in radial_distribution_values:
                    single_cell_texture_features_fov[cell_id][f"{feature_prefix}{feature}_ring{ring_number}"] = radial_distribution(cell_image, ring_number)
            elif feature == "mass_displacement":
                single_cell_texture_features_fov[cell_id][f"{feature_prefix}{feature}"] = mass_displacement(cell_image)
    single_cell_texture_features_fov = pd.DataFrame.from_dict(single_cell_texture_features_fov, orient='index')
    return "", single_cell_texture_features_fov
 
def get_intensity_morphology_features(metadata, channel_name, fov_col_name, mask):
    # get mask morphology features
    feature_prefix = f"Intensity morphology_{channel_name}: "
    mask_morphology_features = ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length", "circularity"]
    
    try:
        mask_props = regionprops(label_image=mask)
    except TypeError as e:
        error_msg = f"Error processing mask for {channel_name}: Mask appears to be in boolean format. Please ensure the mask is properly labeled with integer values for different regions."
        return error_msg, pd.DataFrame()
    
    fov_name = metadata[fov_col_name]
    single_cell_morph_features_fov = {}
    for region in mask_props:
        cell_id = f"{fov_name}_{region.label}"
        if cell_id not in single_cell_morph_features_fov:
            single_cell_morph_features_fov[cell_id] = {}
        # Add centroid x and y: image data is indexed in NumPy and most image processing libraries in "reverse"
        single_cell_morph_features_fov[cell_id][f'{channel_name}_centroid_x'] = region.centroid[1]
        single_cell_morph_features_fov[cell_id][f'{channel_name}_centroid_y'] = region.centroid[0]
        for feature in mask_morphology_features:
            feature_name = f"{feature}"
            if feature in region:
                single_cell_morph_features_fov[cell_id][f"{feature_prefix}{feature_name}"] = region[feature]
            elif feature == "circularity":
                single_cell_morph_features_fov[cell_id][f"{feature_prefix}{feature_name}"] = 4 * np.pi * region.area / region.perimeter**2 if region.perimeter > 0 else 0
    single_cell_morph_features_fov = pd.DataFrame.from_dict(single_cell_morph_features_fov, orient='index')
    return "", single_cell_morph_features_fov

def extract_spcimage_fit_results(metadata, channel_name, num_components, fov_colname):
    fit_feature_prefix = f"Lifetime fit_{channel_name}: "
    image_props = {}
    try:
        a1 = load_image(metadata[f"{channel_name}_a1"])
        # SPC image will output 0 for the thresholded pixels (background), so we need to mask them
        a1 = np.ma.masked_array(a1, mask=a1==0, fill_value=np.nan)
    except Exception as e:
        return f"Error reading the {channel_name} a1 file: {metadata[f'{channel_name}_a1']}: {e}", pd.DataFrame()
    try:
        mask = load_image(metadata[f"{channel_name}_Mask"])
    except Exception as e:
        return f"Error reading the {channel_name} mask file: {metadata[f'{channel_name}_Mask']}: {e}", pd.DataFrame()
    if mask.shape != a1.shape:
        return f"Error: {channel_name} a1 file has a different shape than the mask file: {a1.shape} != {mask.shape}", pd.DataFrame()
    try:
        t1 = load_image(metadata[f"{channel_name}_t1"])
        t1 = np.ma.masked_array(t1, mask=t1==0, fill_value=np.nan)
    except Exception as e:
        return f"Error reading the {channel_name} t1 file: {metadata[f'{channel_name}_t1']}: {e}", pd.DataFrame()
    if mask.shape != t1.shape:
        return f"Error: {channel_name} t1 file has a different shape than the mask file: {t1.shape} != {mask.shape}", pd.DataFrame()
    
    try:
        image_props[f"{fit_feature_prefix}a1"] = regionprops(label_image=mask, intensity_image=a1)
        image_props[f"{fit_feature_prefix}t1"] = regionprops(label_image=mask, intensity_image=t1)
        tm = a1 * t1
    except Exception as e:
        return f"Error: {channel_name} a1 or t1 file is not valid: {e}", pd.DataFrame()

    if num_components == 2:
        try:
            t2 = load_image(metadata[f"{channel_name}_t2"])
            t2 = np.ma.masked_array(t2, mask=t2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the {channel_name} t2 file: {metadata[f'{channel_name}_t2']}: {e}", pd.DataFrame()
        if mask.shape != t2.shape:
            return f"Error: {channel_name} t2 file has a different shape than the mask file: {t2.shape} != {mask.shape}", pd.DataFrame()
        tm= (a1 / 100 * t1) + ((100 - a1) / 100 * t2)
        try:
            image_props[f"{fit_feature_prefix}t2"] = regionprops(label_image=mask, intensity_image=t2)
        except Exception as e:
            return f"Error: {channel_name} t2 file is not valid: {e}", pd.DataFrame()
    
    elif num_components == 3:
        try:
            a2 = load_image(metadata[f"{channel_name}_a2"])
            a2 = np.ma.masked_array(a2, mask=a2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the {channel_name} a2 file: {metadata[f'{channel_name}_a2']}: {e}", pd.DataFrame()
        if mask.shape != a2.shape:
            return f"Error: {channel_name} a2 file has a different shape than the mask file: {a2.shape} != {mask.shape}", pd.DataFrame()
        try:
            t3 = load_image(metadata[f"{channel_name}_t3"])
            t3 = np.ma.masked_array(t3, mask=t3==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the {channel_name} t3 file: {metadata[f'{channel_name}_t3']}: {e}", pd.DataFrame()
        if mask.shape != t3.shape:
            return f"Error: {channel_name} t3 file has a different shape than the mask file: {t3.shape} != {mask.shape}", pd.DataFrame()
        tm= (a1 / 100 * t1) + ((100 - a1) / 100 * t2) + ((100 - a1 - a2) / 100 * t3)
        try:
            image_props[f"{fit_feature_prefix}a2"] = regionprops(label_image=mask, intensity_image=a2)
        except Exception as e:
            return f"Error: {channel_name} a2 file is not valid: {e}", pd.DataFrame()
        try:
            image_props[f"{fit_feature_prefix}t3"] = regionprops(label_image=mask, intensity_image=t3)
        except Exception as e:
            return f"Error: {channel_name} t3 file is not valid: {e}", pd.DataFrame()
    try:
        image_props[f"{fit_feature_prefix}tm"] = regionprops(label_image=mask, intensity_image=tm)
    except Exception as e:
        return f"Error: {channel_name} tm file is not valid: {e}", pd.DataFrame()

    image_name = metadata[fov_colname]
    single_cell_features_img = {}
    for prop in image_props:
        for region in image_props[prop]:
            cell_id = f"{image_name}_{region.label}"
            if cell_id not in single_cell_features_img:
                single_cell_features_img[cell_id] = {}
            single_cell_features_img[cell_id][prop] = region.intensity_mean
            #single_cell_features_img[cell_id][f"{prop}_stdev"] = region.intensity_std
         
   # convert single_cell_features_img to a dataframe
    single_cell_fit_features_fov = pd.DataFrame(single_cell_features_img).T
    if single_cell_fit_features_fov.empty:
        return "Error: No cells found in the mask", pd.DataFrame()
   
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

def get_raw_phasor(decay_curve, h, w, time_axis=None, full_period=False):
    # the truncated time axis case
    if not full_period:
        g_raw = np.dot(np.transpose(decay_curve) , np.cos(h*w*time_axis)) / np.sum(decay_curve)
        s_raw = np.dot(np.transpose(decay_curve) , np.sin(h*w*time_axis)) / np.sum(decay_curve)
    else:
        _, g_raw, s_raw = phasor.phasor_from_signal(decay_curve, harmonic=h)
    return g_raw, s_raw

def extract_fit_free_results(channel_name, decay_curves, laser_rate, duration, calibration_method, shifted_irf=None,reference_dye_image=None, reference_dye_lifetime=None, reference_time_axis=None):
    """
    Extract fit free results for a specific channel and store them in single_cell_features_img (for now, only phasor is implemented)
    Args:
        channel_name
        decay_curves: dictionary of decay curves: key is cell_id, value is decay curve
        shifted_irf: shifted IRF
        laser_rate: laser repetition rate
        duration: time between two laser pulses
        calibration_method: calibration method
        reference_dye_file: reference dye file
        reference_dye_lifetime: reference dye lifetime
    """
    if len(decay_curves) == 0:
        return f"Error: No decay curves found for {channel_name}", pd.DataFrame()

    fit_free_feature_prefix = f"Lifetime fit free_{channel_name}: "
    single_cell_features_fov = {}
    
    # Pre-calculate time_axis and w for reuse across all decay curves
    time_axis = None
    w = 2*np.pi*laser_rate
    full_period = np.isclose(laser_rate * duration, 1.0, rtol=1e-12, atol=1e-12)
    if not full_period:
        # Use the first decay curve to determine time_bins
        first_decay_curve = next(iter(decay_curves.values()))
        time_bins = len(first_decay_curve)
        period = duration / time_bins
        time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
    
    if calibration_method == "Reference Dye":
        if reference_time_axis is None:
            return f"Error: Reference time axis is not provided", pd.DataFrame()
        try:
            ref_mean, ref_real, ref_imag = phasor.phasor_from_signal(reference_dye_image, axis=reference_time_axis)
        except Exception as e:
            return f"Error calculating the phasor of reference dye: {e}", pd.DataFrame()
    else:
        if shifted_irf is not None: 
            # calculate the phasor of irf
            g_irf, s_irf = get_raw_phasor(shifted_irf, h=1,  w=w, time_axis=time_axis, full_period=full_period)
            g_irf_2nd, s_irf_2nd = get_raw_phasor(shifted_irf, h=2,  w=w, time_axis=time_axis, full_period=full_period)
        else: 
            return f"Error: Shifted IRF is not provided for {channel_name}", pd.DataFrame()
    for cell_id, decay_curve in decay_curves.items():
        if cell_id not in single_cell_features_fov:
            single_cell_features_fov[cell_id] = {}

        if calibration_method != "Reference Dye":
            # subtract the esetimated offset and clip the timebin to above or equal to 0
            offset = get_offset(decay_curve)
            decay_curve = decay_curve - offset
            # clip the timebin to above or equal to 0
            decay_curve = np.clip(decay_curve, 0, None)

         # calculate the raw phasor coordinates    
        g_raw, s_raw = get_raw_phasor(decay_curve, h=1, w=w, time_axis=time_axis, full_period=full_period)
        g_raw_2nd, s_raw_2nd = get_raw_phasor(decay_curve, h=2, w=w, time_axis=time_axis, full_period=full_period)
        
        if calibration_method == "Reference Dye":
            G, S = lifetime.phasor_calibrate(g_raw, s_raw, ref_mean, ref_real, ref_imag, frequency=laser_rate, lifetime=reference_dye_lifetime)
            G_2nd, S_2nd = lifetime.phasor_calibrate(g_raw_2nd, s_raw_2nd, ref_mean, ref_real, ref_imag, frequency=laser_rate, lifetime=reference_dye_lifetime, harmonic=2)
        else:
            if shifted_irf is not None:
                G, S = phasor.phasor_divide(g_raw, s_raw, g_irf, s_irf)
                G_2nd, S_2nd = phasor.phasor_divide(g_raw_2nd, s_raw_2nd, g_irf_2nd, s_irf_2nd)
            else:
                return f"Error: Shifted IRF is not provided for {channel_name}", pd.DataFrame()

        phi = np.arctan2(S, G) 
        m = np.sqrt(G**2 + S**2)
        tau_phase = 1/w * np.tan(phi)
        tau_m = 1/w * np.sqrt(1/m**2 - 1)
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}G(1st)"] = G
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}S(1st)"] = S
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}Tau_phase"] = tau_phase
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}Tau_m"] = tau_m
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}G(2nd)"] = G_2nd
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}S(2nd)"] = S_2nd

    return "", single_cell_features_fov

def extract_lifetime_features(metadata, channel_name, input_type, fit, fit_free, fov_col_name, calibration_method=None, reference_dye_image=None, reference_dye_lifetime=None, reference_time_axis=None):
    need_to_fit = False
    time_bins = metadata["time_bins"]
    duration = metadata["duration"]
    if "prefitted" not in input_type or fit_free:
        # get the decay curves and irf
        error_msg, decay_curves = get_decay_curves(metadata, input_type, channel_name, time_bins, shift=False)
        if error_msg != "":
            return error_msg, pd.DataFrame()
        
        if f"{channel_name}_shift" in metadata.index:
            error_msg, irf = get_irf(metadata, channel_name, time_bins)
            if error_msg != "":
                return error_msg, pd.DataFrame()
            shift = metadata[f"{channel_name}_shift"]
            shifted_irf = irf_shift(irf, shift)
        else:
            shifted_irf = None
    if fit:
        num_components = metadata[f"{channel_name}_num_components"]
        if "prefitted" not in input_type:
            fitting_algo = metadata["fitting_algo"]
            fitting_mode = metadata["fitting_mode"]
            start = metadata[f"{channel_name}_start"]
            end = metadata[f"{channel_name}_end"]
            need_to_fit = True
        else: # prefitted
            error_msg, single_cell_fit_features_fov = extract_spcimage_fit_results(metadata, channel_name, num_components, fov_col_name)
            if error_msg != "":
                return error_msg, pd.DataFrame()

    channel_container = st.empty()
    with channel_container.container():
        if "prefitted" not in input_type or fit_free:
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
        laser_rate = metadata["laser_rate"]
        if calibration_method == None:
            return f"Error: Calibration method is not provided for {channel_name}", pd.DataFrame()
        error_msg, single_cell_fit_free_features_fov = extract_fit_free_results(channel_name, decay_curves, laser_rate, duration, calibration_method, shifted_irf, reference_dye_image, reference_dye_lifetime, reference_time_axis)
        if error_msg != "":
            return error_msg, pd.DataFrame()
        single_cell_fit_free_features_fov = pd.DataFrame.from_dict(single_cell_fit_free_features_fov, orient='index')
    if fit and fit_free:
        return "", pd.concat([single_cell_fit_features_fov, single_cell_fit_free_features_fov], axis=1)
    elif fit:
        return "", single_cell_fit_features_fov
    elif fit_free:
        return "", single_cell_fit_free_features_fov

@st.cache_data
def fov_extraction(metadata, metadata_dict):
    """
    Extract single cell features from one fov
    """
    fov_col_name = metadata_dict["fov_name_col"]
    fov_name = metadata[fov_col_name]
    # unique cell id colname
    unique_cell_id_colname = metadata_dict["unique_cell_id_col"]
    # Collect DataFrames from each channel
    fov_feature_dfs = []
    extracted_morphology_masks = []

    for channel_name in metadata_dict["channel_names"]:
        input_type = metadata_dict[channel_name]["input_type"]
        imaging_modality = metadata_dict[channel_name]["imaging_modality"]
        selected_feature_extractors = metadata_dict[channel_name]["selected_feature_extractors"]
        if imaging_modality == "FLIM":  
            fit = "Lifetime fit" in selected_feature_extractors
            fit_free = "Lifetime fit free" in selected_feature_extractors
            if fit_free:
                calibration_method = metadata_dict["fit_free_calibration_method"]
                if calibration_method == "Reference Dye":
                    reference_dye_file = metadata_dict["reference_dye_file"]
                    try:
                        reference_dye_image = load_image(reference_dye_file)
                    except Exception as e:
                        return f"Error reading the reference dye file: {reference_dye_file}: {e}", pd.DataFrame() 
                    # calculate the phasor of reference dye
                    if len(reference_dye_image.shape) != 3:
                        return f"Error: Reference dye file should be a 3D array", pd.DataFrame()
                    reference_dye_lifetime = metadata_dict["reference_dye_lifetime"]   
                    reference_time_axis = metadata["reference_dye_time_axis"]
                else:
                    reference_dye_image = None
                    reference_dye_lifetime = None
                    reference_time_axis = None
            else: 
                calibration_method = None
                reference_dye_file = None
                reference_dye_lifetime = None
                reference_time_axis = None

            if fit or fit_free:
                error_msg, single_cell_lifetime_features = extract_lifetime_features(metadata, channel_name, input_type, fit, fit_free, fov_col_name, calibration_method, reference_dye_image, reference_dye_lifetime, reference_time_axis)
                if error_msg != "":
                    st.error(error_msg)
                    continue
                fov_feature_dfs.append(single_cell_lifetime_features)
            int_morph = "Intensity morphology" in selected_feature_extractors
            int_texture = "Intensity texture" in selected_feature_extractors
            if int_morph or int_texture:
                try:
                    mask = load_image(metadata[f"{channel_name}_Mask"])
                except Exception as e:
                    st.error(f"Error reading the {channel_name} mask file: {metadata[f'{channel_name}_Mask']}: {e}")
                    continue
                if int_morph:
                    # check if the morphology features are already extracted for this mask
                    if metadata[f"{channel_name}_Mask"] not in extracted_morphology_masks:
                        extracted_morphology_masks.append(metadata[f"{channel_name}_Mask"])
                        error_msg, single_cell_morph_features_fov = get_intensity_morphology_features(metadata, channel_name, fov_col_name, mask)
                        if error_msg != "":
                            st.error(error_msg)
                        else:
                            fov_feature_dfs.append(single_cell_morph_features_fov)
                if int_texture:
                        error_msg, single_cell_texture_features_fov = get_intensity_texture_features(metadata, channel_name, fov_col_name, mask, input_type)
                        if error_msg != "":
                            st.error(error_msg)
                        else:
                            fov_feature_dfs.append(single_cell_texture_features_fov)
        elif imaging_modality == "Intensity-only":
            if input_type == "Intensity (2D)":
                int_morph = "Intensity morphology" in selected_feature_extractors
                int_texture = "Intensity texture" in selected_feature_extractors
                try:
                    mask = load_image(metadata[f"{channel_name}_Mask"])
                except Exception as e:
                    st.error(f"Error reading the {channel_name} mask file: {metadata[f'{channel_name}_Mask']}: {e}")
                    continue
                if int_morph:
                    if metadata[f"{channel_name}_Mask"] not in extracted_morphology_masks:
                        extracted_morphology_masks.append(metadata[f"{channel_name}_Mask"])
                        error_msg, single_cell_morph_features_fov = get_intensity_morphology_features(metadata, channel_name, fov_col_name, mask)
                        if error_msg != "":
                            st.error(error_msg)
                        else:
                            fov_feature_dfs.append(single_cell_morph_features_fov)
                if int_texture:
                    error_msg, single_cell_texture_features_fov = get_intensity_texture_features(metadata, channel_name, fov_col_name, mask, input_type)
                    if error_msg != "":
                        st.error(error_msg)
                    else:
                        fov_feature_dfs.append(single_cell_texture_features_fov)

    # Combine all channel DataFrames in one operation
    single_cell_features_fov = pd.concat(fov_feature_dfs, axis=1) if fov_feature_dfs else pd.DataFrame()
    if not single_cell_features_fov.empty:
        single_cell_features_fov[fov_col_name] = fov_name
        single_cell_features_fov.index.name = unique_cell_id_colname
    else:
        return f"Error: No cells found in the {fov_name}", pd.DataFrame()

    return "", single_cell_features_fov
            