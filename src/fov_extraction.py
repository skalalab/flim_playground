import pandas as pd
import numpy as np
from skimage.measure import regionprops
from src.file_io import get_decay_curves, load_image, get_irf
from src.decay_io import read_decay
from src.fit import fit_curves
from src.fit_helper import create_progress_callback, irf_shift, forward_pass, reduced_chi_square
import streamlit as st
from phasorpy import phasor, lifetime
from src.cell_texture import granularity, radial_distribution, mass_displacement

def get_offset(decay_curve):
    """
    Get the offset of a decay curve using: median of the last 10% of bins
    """
   # head_bins_percentile = 20
    tail_bins_percentile = 90
    
    # Calculate the number of bins for each segment
    total_bins = len(decay_curve)
    #head_bins = int(total_bins * head_bins_percentile / 100)
    tail_start_bin = int(total_bins * tail_bins_percentile / 100)
    
    # Get the first 20% of bins and calculate median
  #  head_segment = decay_curve[:head_bins]
  #  head_median = np.median(head_segment)
    
    # Get the last 10% of bins and calculate median  
    tail_segment = decay_curve[tail_start_bin:]
    tail_median = np.mean(tail_segment)
    
    # Return the minimum of the two medians
    return tail_median

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
    else:
        return f"Error: Unsupported input type '{input_type}' for intensity texture features on {channel_name}.", pd.DataFrame()
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

    if num_components > 3 or num_components < 1:
        return f"Error: {num_components} are not yet supported. ", pd.DataFrame()

    fit_feature_prefix = f"Lifetime fit_{channel_name}: "
    intensity_images = {}
   
    try:
        mask = load_image(metadata[f"{channel_name}_Mask"])
    except Exception as e:
        return f"Error: failed to read the {channel_name} mask file: {metadata[f'{channel_name}_Mask']}: {e}", pd.DataFrame()
   
    try:
        t1 = load_image(metadata[f"{channel_name}_SPCImage t1"])
        t1 = np.ma.masked_array(t1, mask=t1==0, fill_value=np.nan)
    except Exception as e:
        return f"Error: failed to read the {channel_name} t1 file: {metadata[f'{channel_name}_SPCImage t1']}: {e}", pd.DataFrame()
    if mask.shape != t1.shape:
        return f"Error: {channel_name} t1 file has a different shape than the mask file: {t1.shape} != {mask.shape}", pd.DataFrame()
    
    intensity_images[f"{fit_feature_prefix}t1"] = t1

    if num_components >= 2:
        try:
            a1 = load_image(metadata[f"{channel_name}_a1"])
            # SPC image will output 0 for the thresholded pixels (background), so we need to mask them
            a1 = np.ma.masked_array(a1, mask=a1==0, fill_value=np.nan)
        except Exception as e:
            return f"Error: failed to read the {channel_name} a1 file: {metadata[f'{channel_name}_a1']}: {e}", pd.DataFrame()
        if mask.shape != a1.shape:
            return f"Error: {channel_name} a1 file has a different shape than the mask file: {a1.shape} != {mask.shape}", pd.DataFrame()
        try:
            t2 = load_image(metadata[f"{channel_name}_t2"])
            t2 = np.ma.masked_array(t2, mask=t2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error: failed to read the {channel_name} t2 file: {metadata[f'{channel_name}_t2']}: {e}", pd.DataFrame()
        if mask.shape != t2.shape:
            return f"Error: {channel_name} t2 file has a different shape than the mask file: {t2.shape} != {mask.shape}", pd.DataFrame()
        
        intensity_images[f"{fit_feature_prefix}a1"] = a1
        intensity_images[f"{fit_feature_prefix}t2"] = t2
        if num_components == 2:
            intensity_images[f"{channel_name}_a2"] = 100 - a1

    if num_components == 3:
        try:
            a2 = load_image(metadata[f"{channel_name}_a2"])
            a2 = np.ma.masked_array(a2, mask=a2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error: failed to read the {channel_name} a2 file: {metadata[f'{channel_name}_a2']}: {e}", pd.DataFrame()
        if mask.shape != a2.shape:
            return f"Error: {channel_name} a2 file has a different shape than the mask file: {a2.shape} != {mask.shape}", pd.DataFrame()
        try:
            t3 = load_image(metadata[f"{channel_name}_t3"])
            t3 = np.ma.masked_array(t3, mask=t3==0, fill_value=np.nan)
        except Exception as e:
            return f"Error: failed to read the {channel_name} t3 file: {metadata[f'{channel_name}_t3']}: {e}", pd.DataFrame()
        if mask.shape != t3.shape:
            return f"Error: {channel_name} t3 file has a different shape than the mask file: {t3.shape} != {mask.shape}", pd.DataFrame()
        
        intensity_images[f"{fit_feature_prefix}a2"] = a2
        intensity_images[f"{fit_feature_prefix}t3"] = t3
        intensity_images[f"{channel_name}_a3"] = 100 - a1 - a2

    if num_components == 1:
        tm = t1 
        tm_iw = t1
    elif num_components == 2:
        tm = (a1 / 100 * t1) + ((100 - a1) / 100 * t2)
        alpha1 = a1 / 100.0
        alpha2 = (100.0 - a1) / 100.0
        tm_iw = (alpha1 * (t1 ** 2) + alpha2 * (t2 ** 2)) / tm
    elif num_components == 3:
        tm = (a1 / 100 * t1) + (a2 / 100 * t2) + ((100 - a1 - a2) / 100 * t3)
        alpha1 = a1 / 100.0
        alpha2 = a2 / 100.0
        alpha3 = (100.0 - a1 - a2) / 100.0
        tm_iw = (alpha1 * (t1 ** 2) + alpha2 * (t2 ** 2) + alpha3 * (t3 ** 2)) / tm
    
    intensity_images[f"{fit_feature_prefix}tm"] = tm
    intensity_images[f"{fit_feature_prefix}tm_iw"] = tm_iw

    # Get unique region labels from the mask (excluding background label 0)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0:
        return "Error: No cells found in the mask", pd.DataFrame()

    image_name = metadata[fov_colname]
    single_cell_features_img = {}
    
    # Calculate mean intensity for each region and each intensity image
    for region_label in unique_labels:
        cell_id = f"{image_name}_{region_label}"
        single_cell_features_img[cell_id] = {}
        
        for prop_name, intensity_image in intensity_images.items():
            region_mask = (mask == region_label)
            region_intensities = intensity_image[region_mask]
            mean_intensity = np.ma.average(region_intensities)
            # Convert masked array to regular array with NaN for masked values
            mean_intensity = np.nan if np.ma.is_masked(mean_intensity) else float(mean_intensity)
            single_cell_features_img[cell_id][prop_name] = mean_intensity
         
   # convert single_cell_features_img to a dataframe
    single_cell_fit_features_fov = pd.DataFrame(single_cell_features_img).T
    if single_cell_fit_features_fov.empty:
        return "Error: No cells found in the mask", pd.DataFrame()
   
    return "", single_cell_fit_features_fov

def extract_fit_results(channel_name, decay_curves, results, num_components, shifted_irf, time_axis, start, end, fixed_lifetimes=None):
    """
    Extract fitting results for a specific channel and store them in single_cell_features_img

    Args:
        channel_name
        decay_curves: dict of cell_id -> decay curve
        results: fitting result dictionary from fit_curves
        num_components: number of fitting components
        shifted_irf: the shifted IRF used for fitting
        time_axis: time axis array
        start: start time gate index
        end: end time gate index
        fixed_lifetimes: dict of fixed lifetime values
    """
    single_cell_features_fov = {}
    fit_feature_prefix = f"Lifetime fit_{channel_name}: "
    num_fixed = sum(1 for v in (fixed_lifetimes or {}).values() if v is not None and v > 0)
    num_free_params = num_components * 2 + 1 - num_fixed  # k amps + k taus + offset, minus fixed
    warning_msg = ""
    for i, cell_id in enumerate(decay_curves.keys()):
        if cell_id not in single_cell_features_fov:
            single_cell_features_fov[cell_id] = {}
        # amplitudes and offsets are just bookkeeping, should be default to uncategorized features (i.e. without prefix)
        single_cell_features_fov[cell_id][f"{channel_name}_amp1"] = results["amp1"][i]
        single_cell_features_fov[cell_id][f"{fit_feature_prefix}t1"] = results["t1"][i] * 1000  # Convert to ps
        single_cell_features_fov[cell_id][f"{channel_name}_offset"] = results["offset"][i]
        # Compute reduced chi-square from fitted curve
        fitted_curve = forward_pass(
            amp1=results["amp1"][i], t1=results["t1"][i], offset=results["offset"][i],
            shifted_irf=shifted_irf, time_axis=time_axis,
            amp2=results["amp2"][i] if num_components > 1 else None,
            t2=results["t2"][i] if num_components > 1 else None,
            amp3=results["amp3"][i] if num_components > 2 else None,
            t3=results["t3"][i] if num_components > 2 else None,
        )
        single_cell_features_fov[cell_id][f"{channel_name}_reduced_chi_square"] = reduced_chi_square(
            fitted_curve, decay_curves[cell_id], start, end, num_free_params
        )
        if num_components == 2:
            single_cell_features_fov[cell_id][f"{channel_name}_amp2"] = results["amp2"][i]
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}t2"] = results["t2"][i] * 1000  # Convert to ps
            # Calculate alpha values
            amp1, amp2 = results["amp1"][i], results["amp2"][i]
            total_amp = amp1 + amp2
            # guard against division by zero
            if total_amp == 0:
                warning_msg += f"Warning: {cell_id} has a total amplitude of 0. "
                continue
            alpha1 = amp1 / total_amp
            alpha2 = amp2 / total_amp
            a1_pct = alpha1 * 100
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}a1"] = a1_pct
            single_cell_features_fov[cell_id][f"{channel_name}_a2"] = 100 - a1_pct
            # Calculate mean lifetime (in original units, not converted)
            # Calculate intensity weighted average lifetime (in original units, not converted)
            t1_val = results["t1"][i]
            t2_val = results["t2"][i]
            tm_ns = alpha1 * t1_val + alpha2 * t2_val
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}tm"] = tm_ns * 1000
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}tm_iw"] = (
                ((alpha1 * (t1_val ** 2) + alpha2 * (t2_val ** 2)) / tm_ns) * 1000 if tm_ns != 0 else 0.0
            )
            
        elif num_components == 3:
            single_cell_features_fov[cell_id][f"{channel_name}_amp2"] = results["amp2"][i]
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}t2"] = results["t2"][i] * 1000  # Convert to ps
            single_cell_features_fov[cell_id][f"{channel_name}_amp3"] = results["amp3"][i]
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}t3"] = results["t3"][i] * 1000  # Convert to ps
            # Calculate alpha values for 3 components
            amp1, amp2, amp3 = results["amp1"][i], results["amp2"][i], results["amp3"][i]
            total_amp = amp1 + amp2 + amp3
            # guard against division by zero
            if total_amp == 0:
                warning_msg += f"Warning: {cell_id} has a total amplitude of 0. "
                continue
            alpha1 = amp1 / total_amp
            alpha2 = amp2 / total_amp
            alpha3 = amp3 / total_amp
            a1_pct = alpha1 * 100
            a2_pct = alpha2 * 100
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}a1"] = a1_pct
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}a2"] = a2_pct
            single_cell_features_fov[cell_id][f"{channel_name}_a3"] = 100 - a1_pct - a2_pct
            t1_val = results["t1"][i]
            t2_val = results["t2"][i]
            t3_val = results["t3"][i]
            # Calculate mean lifetime for 3 components (in original units, not converted)
            tm_ns = alpha1 * t1_val + alpha2 * t2_val + alpha3 * t3_val
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}tm"] = tm_ns * 1000
            # Calculate intensity weighted average lifetime for 3 components (in original units, not converted)
            single_cell_features_fov[cell_id][f"{fit_feature_prefix}tm_iw"] = (
                ((alpha1 * (t1_val ** 2) + alpha2 * (t2_val ** 2) + alpha3 * (t3_val ** 2)) / tm_ns) * 1000 if tm_ns != 0 else 0.0
            )

    return warning_msg, single_cell_features_fov

def get_raw_phasor(decay_curve, h, w, time_axis=None, full_period=False):
    # the truncated time axis case
    if not full_period:
        g_raw = np.dot(np.transpose(decay_curve) , np.cos(h*w*time_axis)) / np.sum(decay_curve)
        s_raw = np.dot(np.transpose(decay_curve) , np.sin(h*w*time_axis)) / np.sum(decay_curve)
    else:
        _, g_raw, s_raw = phasor.phasor_from_signal(decay_curve, harmonic=h)
    return g_raw, s_raw

def extract_fit_free_results(channel_name, decay_curves, laser_rate, duration, calibration_method, shifted_irf=None,fluorescence_lifetime_standard_image=None, fluorescence_lifetime_standard_lifetime=None, fluorescence_lifetime_standard_time_axis=None):
    """
    Extract fit free results for a specific channel and store them in single_cell_features_img (for now, only phasor is implemented)
    """
    if len(decay_curves) == 0:
        return f"Error: No decay curves found for {channel_name}", pd.DataFrame()

    fit_free_feature_prefix = f"Lifetime fit free_{channel_name}: "
    single_cell_features_fov = {}
    
    # Pre-calculate time_axis and w for reuse across all decay curves
    time_axis = None
    w = 2*np.pi*laser_rate
    full_period = np.isclose(laser_rate * duration, 1.0, rtol=1e-5, atol=1e-5)
    if not full_period:
        # Use the first decay curve to determine time_bins
        first_decay_curve = next(iter(decay_curves.values()))
        time_bins = len(first_decay_curve)
        period = duration / time_bins
        time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)

    if calibration_method == "Fluorescence Lifetime Standard":
        if fluorescence_lifetime_standard_time_axis is None:
            return "Error: Fluorescence lifetime standard time axis is not provided", pd.DataFrame()
        try:
            if not full_period:
                phi = w * time_axis
                ref_mean, ref_real, ref_imag = phasor.phasor_from_signal(fluorescence_lifetime_standard_image, axis=fluorescence_lifetime_standard_time_axis, sample_phase=phi, use_fft=False)
            else:
                ref_mean, ref_real, ref_imag = phasor.phasor_from_signal(fluorescence_lifetime_standard_image, axis=fluorescence_lifetime_standard_time_axis)
        except Exception as e:
            return f"Error calculating the phasor of fluorescence lifetime standard: {e}", pd.DataFrame()
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

        # subtract the esetimated offset and clip the timebin to above or equal to 0
        offset = get_offset(decay_curve)
        decay_curve = decay_curve - offset
        # clip the timebin to above or equal to 0
        decay_curve = np.clip(decay_curve, 0, None)

         # calculate the raw phasor coordinates    
        g_raw, s_raw = get_raw_phasor(decay_curve, h=1, w=w, time_axis=time_axis, full_period=full_period)
        g_raw_2nd, s_raw_2nd = get_raw_phasor(decay_curve, h=2, w=w, time_axis=time_axis, full_period=full_period)
        
        if calibration_method == "Fluorescence Lifetime Standard":
            G, S = lifetime.phasor_calibrate(g_raw, s_raw, ref_mean, ref_real, ref_imag, frequency=laser_rate, lifetime=fluorescence_lifetime_standard_lifetime)
            G_2nd, S_2nd = lifetime.phasor_calibrate(g_raw_2nd, s_raw_2nd, ref_mean, ref_real, ref_imag, frequency=laser_rate, lifetime=fluorescence_lifetime_standard_lifetime, harmonic=2)
        else:
            if shifted_irf is not None:
                G, S = phasor.phasor_divide(g_raw, s_raw, g_irf, s_irf)
                G_2nd, S_2nd = phasor.phasor_divide(g_raw_2nd, s_raw_2nd, g_irf_2nd, s_irf_2nd)
            else:
                return f"Error: Shifted IRF is not provided for {channel_name}", pd.DataFrame()

        phi = np.arctan2(S, G) 
        m = np.sqrt(G**2 + S**2)
        tau_phase = 1/w * np.tan(phi)
        if m > 0 and m < 1:
            tau_m = 1/w * np.sqrt(1/m**2 - 1)
        else:
            tau_m = np.nan
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}G(1st)"] = G
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}S(1st)"] = S
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}Tau_phase"] = tau_phase
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}Tau_m"] = tau_m
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}G(2nd)"] = G_2nd
        single_cell_features_fov[cell_id][f"{fit_free_feature_prefix}S(2nd)"] = S_2nd

    return "", single_cell_features_fov

def extract_intensity_sum_2d(channel_name, decay_curves, single_cell_features_fov):
    intensity_sum_feature_name = f"Intensity texture_{channel_name}: intensity_sum"
    for _, cell_id in enumerate(decay_curves.keys()):
        single_cell_features_fov[cell_id][intensity_sum_feature_name] = np.sum(decay_curves[cell_id])
    return single_cell_features_fov


def extract_lifetime_features(metadata, channel_name, input_type, fit, fit_free, fov_col_name, calibration_method=None, fluorescence_lifetime_standard_image=None, fluorescence_lifetime_standard_lifetime=None, fluorescence_lifetime_standard_time_axis=None, fixed_lifetimes=None):
    need_to_fit = False

    if "prefitted" not in input_type or fit_free:
        time_bins = metadata["time_bins"]
        duration = metadata["duration"]
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
        try:
            num_components = metadata[f"{channel_name}_num_components"]
        except Exception as e:
            return f"Error: Number of components not found for {channel_name}: {e}", pd.DataFrame()
        if "prefitted" not in input_type:
            try:
                fitting_algo = metadata["fitting_algo"]
                fitting_mode = metadata["fitting_mode"]
                start = metadata[f"{channel_name}_start"]
                end = metadata[f"{channel_name}_end"]
            except Exception as e:
                return f"Error: Fitting algorithm or mode or start or end not found for {channel_name}: {e}", pd.DataFrame()
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
        results = fit_curves(duration, time_bins, list(decay_curves.values()), shifted_irf, num_components, fitting_algo, fitting_mode, start=start, end=end, fixed_lifetimes=fixed_lifetimes, _progress_callback=channel_progress_callback)
        period = duration / time_bins
        time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
        warning_msg, single_cell_fit_features_fov = extract_fit_results(channel_name, decay_curves, results, num_components, shifted_irf, time_axis, start, end, fixed_lifetimes)
        if warning_msg != "":
            st.warning(warning_msg)
        # convert to dataframe
        single_cell_fit_features_fov = pd.DataFrame.from_dict(single_cell_fit_features_fov, orient='index')
    channel_container.empty()  # Remove both text and progress bar when done
    if fit_free:
        try:
            laser_rate = metadata["laser_rate"]
        except (KeyError, TypeError):
            return "Error: laser_rate not found in metadata. Ensure it was set during FOV Metadata Extraction.", pd.DataFrame()
        if calibration_method == None:
            return f"Error: Calibration method is not provided for {channel_name}", pd.DataFrame()
        error_msg, single_cell_fit_free_features_fov = extract_fit_free_results(channel_name, decay_curves, laser_rate, duration, calibration_method, shifted_irf, fluorescence_lifetime_standard_image, fluorescence_lifetime_standard_lifetime, fluorescence_lifetime_standard_time_axis)
        if error_msg != "":
            return error_msg, pd.DataFrame()
        if "2D" in input_type:
            single_cell_fit_free_features_fov = extract_intensity_sum_2d(channel_name, decay_curves, single_cell_fit_free_features_fov)
        single_cell_fit_free_features_fov = pd.DataFrame.from_dict(single_cell_fit_free_features_fov, orient='index')
    if fit and fit_free:
        return "", pd.concat([single_cell_fit_features_fov, single_cell_fit_free_features_fov], axis=1)
    elif fit:
        return "", single_cell_fit_features_fov
    elif fit_free:
        return "", single_cell_fit_free_features_fov
    else:
        return "Error: Neither lifetime fit nor fit free analysis was requested.", pd.DataFrame()


def extract_intensity_features(metadata, channel_name, fov_col_name, input_type,
                                selected_feature_extractors, extracted_morphology_masks):
    """Extract intensity morphology and/or texture features for a channel.

    Shared by both FLIM and Intensity-only branches so the logic lives in one
    place.  Returns (feature_dfs, error_occurred) where *feature_dfs* is a list
    of DataFrames (possibly empty) and *error_occurred* is True if the mask
    could not be loaded.
    """
    int_morph = "Intensity morphology" in selected_feature_extractors
    int_texture = "Intensity texture" in selected_feature_extractors
    feature_dfs = []

    if not (int_morph or int_texture):
        return feature_dfs, False

    try:
        mask = load_image(metadata[f"{channel_name}_Mask"])
    except Exception as e:
        st.error(f"Error reading the {channel_name} mask file: {metadata[f'{channel_name}_Mask']}: {e}")
        return feature_dfs, True  # signal caller to skip this channel

    if int_morph:
        # Avoid re-extracting morphology if the same mask file was already processed
        if metadata[f"{channel_name}_Mask"] not in extracted_morphology_masks:
            extracted_morphology_masks.append(metadata[f"{channel_name}_Mask"])
            error_msg, morph_df = get_intensity_morphology_features(metadata, channel_name, fov_col_name, mask)
            if error_msg != "":
                st.error(error_msg)
            else:
                feature_dfs.append(morph_df)

    if int_texture:
        error_msg, texture_df = get_intensity_texture_features(metadata, channel_name, fov_col_name, mask, input_type)
        if error_msg != "":
            st.error(error_msg)
        else:
            feature_dfs.append(texture_df)

    return feature_dfs, False

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
                if calibration_method == "Fluorescence Lifetime Standard":
                    # channel-specific
                    try:
                        fluorescence_lifetime_standard_file = metadata_dict[channel_name]["fluorescence_lifetime_standard_file"]
                    except KeyError:
                        return f"Error: Fluorescence lifetime standard file not found for channel {channel_name}.", pd.DataFrame()
                    try:
                        fluorescence_lifetime_standard_image = load_image(fluorescence_lifetime_standard_file)
                    except Exception as e:
                        return f"Error reading the fluorescence lifetime standard file for {channel_name}: {fluorescence_lifetime_standard_file}: {e}", pd.DataFrame() 
                    if len(fluorescence_lifetime_standard_image.shape) != 3:
                        return f"Error: Fluorescence lifetime standard file for {channel_name} should be a 3D array", pd.DataFrame()
                    fluorescence_lifetime_standard_lifetime = metadata_dict["fluorescence_lifetime_standard_lifetime"]
                    try:
                        fluorescence_lifetime_standard_time_axis = metadata[f"{channel_name}_fluorescence_lifetime_standard_time_axis"]
                    except KeyError:
                        return f"Error: Fluorescence lifetime standard time axis not found for {channel_name}", pd.DataFrame()
                else:
                    fluorescence_lifetime_standard_image = None
                    fluorescence_lifetime_standard_lifetime = None
                    fluorescence_lifetime_standard_time_axis = None
            else: 
                calibration_method = None
                fluorescence_lifetime_standard_file = None
                fluorescence_lifetime_standard_image = None
                fluorescence_lifetime_standard_lifetime = None
                fluorescence_lifetime_standard_time_axis = None

            if fit or fit_free:
                fixed_lifetimes = metadata_dict[channel_name].get("fixed_lifetimes", {})
                error_msg, single_cell_lifetime_features = extract_lifetime_features(metadata, channel_name, input_type, fit, fit_free, fov_col_name, calibration_method, fluorescence_lifetime_standard_image=fluorescence_lifetime_standard_image, fluorescence_lifetime_standard_lifetime=fluorescence_lifetime_standard_lifetime, fluorescence_lifetime_standard_time_axis=fluorescence_lifetime_standard_time_axis, fixed_lifetimes=fixed_lifetimes)
                if error_msg != "":
                    st.error(error_msg)
                    continue
                fov_feature_dfs.append(single_cell_lifetime_features)
            intensity_dfs, mask_error = extract_intensity_features(
                metadata, channel_name, fov_col_name, input_type,
                selected_feature_extractors, extracted_morphology_masks
            )
            if mask_error:
                continue
            fov_feature_dfs.extend(intensity_dfs)
        elif imaging_modality == "Intensity-only":
            intensity_dfs, mask_error = extract_intensity_features(
                metadata, channel_name, fov_col_name, input_type,
                selected_feature_extractors, extracted_morphology_masks
            )
            if mask_error:
                continue
            fov_feature_dfs.extend(intensity_dfs)

    # Combine all channel DataFrames in one operation
    single_cell_features_fov = pd.concat(fov_feature_dfs, axis=1) if fov_feature_dfs else pd.DataFrame()
    if not single_cell_features_fov.empty:
        single_cell_features_fov[fov_col_name] = fov_name
        single_cell_features_fov.index.name = unique_cell_id_colname
    else:
        return f"Error: No cells found in the {fov_name}", pd.DataFrame()

    return "", single_cell_features_fov
            