import pandas as pd
import numpy as np
from skimage.measure import regionprops
from src.feature_types import all_numerical_feature_groups
from src.file_io import load_image
from src.sdt_io import read_sdt
from src.fit import fit_curves
from src.fit_helper import create_progress_callback, irf_shift
from src.fit_free import get_phasor_features
import streamlit as st

def get_mask_morphology_features(mask, image_name, cell_dict):
        # get mask morphology features
    mask_morphology_features = all_numerical_feature_groups["Mask Morphology"]
    mask_props = regionprops(label_image=mask)
    for region in mask_props:
        cell_id = f"{image_name}_{region.label}"
        if cell_id not in cell_dict:
            cell_dict[cell_id] = {}
        # Add centroid x and y: image data is indexed in NumPy and most image processing libraries in "reverse"
        cell_dict[cell_id]['centroid_x'] = region.centroid[1]
        cell_dict[cell_id]['centroid_y'] = region.centroid[0]
        for feature in mask_morphology_features:
            feature_name = f"{feature}"
            if feature in region:
                cell_dict[cell_id][feature_name] = region[feature]
            elif feature == "circularity":
                cell_dict[cell_id][feature_name] = 4 * np.pi * region.area / region.perimeter**2 if region.perimeter > 0 else 0
    return cell_dict

def spcimage_fit_extraction(metadata, channel_name, num_components):

    image_props = {}
    try:
        a1 = load_image(metadata[f"{channel_name} a1"])
        # SPC image will output 0 for the thresholded pixels (background), so we need to mask them
        a1 = np.ma.masked_array(a1, mask=a1==0, fill_value=np.nan)
    except Exception as e:
        return f"Error reading the {channel_name} a1 file: {metadata[f'{channel_name} a1']}: {e}", None
    try:
        mask = load_image(metadata[f"{channel_name} Mask"])
    except Exception as e:
        return f"Error reading the {channel_name} mask file: {metadata[f'{channel_name} Mask']}: {e}", None
    if mask.shape != a1.shape:
        return f"Error: {channel_name} a1 file has a different shape than the mask file: {a1.shape} != {mask.shape}", None
    try:
        t1 = load_image(metadata[f"{channel_name} t1"])
        t1 = np.ma.masked_array(t1, mask=t1==0, fill_value=np.nan)
    except Exception as e:
        return f"Error reading the {channel_name} t1 file: {metadata[f'{channel_name} t1']}: {e}", None
    if mask.shape != t1.shape:
        return f"Error: {channel_name} t1 file has a different shape than the mask file: {t1.shape} != {mask.shape}", None
    
    image_props[f"{channel_name}_a1"] = regionprops(label_image=mask, intensity_image=a1)
    image_props[f"{channel_name} t1"] = regionprops(label_image=mask, intensity_image=t1)
    
    try:
        a2 = load_image(metadata[f"{channel_name} a2"])
        a2 = np.ma.masked_array(a2, mask=a2==0, fill_value=np.nan)
    except Exception as e:
        return f"Error reading the {channel_name} a2 file: {metadata[f'{channel_name} a2']}: {e}", None
    if mask.shape != a2.shape:
        return f"Error: {channel_name} a2 file has a different shape than the mask file: {a2.shape} != {mask.shape}", None

        try:
            nadh_t2 = load_image(metadata['nadh t2'])
            nadh_t2 = np.ma.masked_array(nadh_t2, mask=nadh_t2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the NADH t2 file: {metadata['nadh t2']}: {e}", None
        if mask.shape != nadh_t2.shape:
            return f"Error: NADH t2 file has a different shape than the mask file: {nadh_t2.shape} != {mask.shape}", None
        nadh_tm= (nadh_a1 / 100 * nadh_t1) + (nadh_a2 / 100 * nadh_t2)

        try: 
            intensity = load_image(metadata[f"{channel_name} intensity"])
        except Exception as e:
            return f"Error reading the {channel_name} intensity file: {metadata[f'{channel_name} intensity']}: {e}", None
        if mask.shape != nadh_intensity.shape:
            return f"Error: NADH intensity file has a different shape than the mask file: {nadh_intensity.shape} != {mask.shape}", None
        nadh_feature_prefix = all_numerical_feature_groups['Nadh Fit']
        image_props[f"{nadh_feature_prefix}a1"] = regionprops(label_image=mask, intensity_image=nadh_a1)
        image_props[f"{nadh_feature_prefix}a2"] = regionprops(label_image=mask, intensity_image=nadh_a2)
        image_props[f"{nadh_feature_prefix}t1"] = regionprops(label_image=mask, intensity_image=nadh_t1)
        image_props[f"{nadh_feature_prefix}t2"] = regionprops(label_image=mask, intensity_image=nadh_t2)
        image_props[f"{nadh_feature_prefix}tm"] = regionprops(label_image=mask, intensity_image=nadh_tm)
        image_props[f"{nadh_feature_prefix}intensity"] = regionprops(label_image=mask, intensity_image=nadh_intensity)

    # calculate redox
    if has_nadh and has_fad:
        norm_redox = nadh_intensity / (nadh_intensity + fad_intensity + 1e-10)
        image_props[f"{nadh_feature_prefix}norm_redox"] = regionprops(label_image=mask, intensity_image=norm_redox)
    image_name = metadata['image_name']
    single_cell_features_img = {}
    fit_fd_prefix = all_numerical_feature_groups["Feature Distribution Fit"]
    for prop in image_props:
        for region in image_props[prop]:
            cell_id = f"{image_name}_{region.label}"
            if cell_id not in single_cell_features_img:
                single_cell_features_img[cell_id] = {}
            single_cell_features_img[cell_id][prop] = region.intensity_mean
            single_cell_features_img[cell_id][f"{prop}_stdev"] = region.intensity_std
            # add fit variable feature distribution features for this fit 
            for feature_distribution_var in texture_features:
                if feature_distribution_var == "polarity":
                    geometric_centroid = region.centroid
                    weighted_centroid = region.centroid_weighted
                    # example: fit_nadh: a1 -> nadh_a1
                    feature_name = prop.replace("fit_", "").replace(": ", "_") 
                    feature_name = f"{fit_fd_prefix}{feature_name}_{feature_distribution_var}"
                    single_cell_features_img[cell_id][feature_name] = np.sqrt(
                        (geometric_centroid[0] - weighted_centroid[0]) ** 2 +
                        (geometric_centroid[1] - weighted_centroid[1]) ** 2)
                    
    
    # convert single_cell_features_img to a dataframe
    single_cell_features_img = pd.DataFrame(single_cell_features_img).T
    if single_cell_features_img.empty:
        return "Error: No cells found in the mask", None
    # name the index as cell_id
    single_cell_features_img.index.name = "cell_id"
   
    return "", single_cell_features_img

@st.cache_data
def roi_summing_fit_extraction(metadata, has_nadh, has_fad, fit_free):

    image_name = metadata['image_name']

    if has_nadh:
        try:
             # check if the shift is provided
            nadh_shift = metadata['nadh_shift']
        except Exception as e:
            return "Error: NADH shift is not provided", None

        shifted_nadh_irf = irf_shift(nadh_irf, nadh_shift)
        try:
            nadh_start = metadata['nadh_start']
            nadh_end = metadata['nadh_end']
        except Exception as e:
            return "Error: NADH start and end are not provided", None
        
    # get fitting config from metadata
    try: 
        fitting_mode = metadata['fitting_mode']
        fitting_algo = metadata['fitting_algo']
        time_bins = metadata['time_bins']
        duration = metadata['duration']
        num_components = metadata['num_components']
        if fit_free:
            laser_rate = metadata['laser_rate']
    except Exception as e:
        return "Error: Some of the fitting config is not provided", None
    
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
    single_cell_features_img = {}
    unique_cells = np.sort(np.unique(mask))
    # Remove background (0)
    unique_cells = unique_cells[unique_cells != 0]
    # Collect all decay curves first, ordered by cell_id
    nadh_decay_curves = []
    fad_decay_curves = []
    cell_ids = []
    
    for cell in unique_cells:
        cell_mask = mask == cell
        cell_id = f"{image_name}_{cell}"
        cell_ids.append(cell_id)
        
        if has_nadh:
            nadh_decay_cell = nadh_decay[cell_mask, :].sum(axis=0)
            nadh_decay_curves.append(nadh_decay_cell)
        if has_fad:
            fad_decay_cell = fad_decay[cell_mask, :].sum(axis=0)
            fad_decay_curves.append(fad_decay_cell)
    if has_nadh and nadh_decay_curves is not None and shifted_nadh_irf is not None and nadh_start is not None and nadh_end is not None: 
        single_cell_features_img = fit_and_extract_results("NADH", duration, time_bins, num_components, fitting_algo, fitting_mode, fit_free, laser_rate, time_axis, cell_ids, single_cell_features_img, nadh_start, nadh_end, nadh_decay_curves, shifted_nadh_irf, extract_fit)
    if has_fad and fad_decay_curves is not None and shifted_fad_irf is not None and fad_start is not None and fad_end is not None:
        single_cell_features_img = fit_and_extract_results("FAD", duration, time_bins, num_components, fitting_algo, fitting_mode, fit_free, laser_rate, time_axis, cell_ids, single_cell_features_img, fad_start, fad_end, fad_decay_curves, shifted_fad_irf, extract_fit)
    # calculate redox
    if extract_fit:
        if has_nadh and has_fad:
            # get the intensity of NADH and FAD
            for cell_id in single_cell_features_img:
                nadh_intensity = single_cell_features_img[cell_id][f"{all_numerical_feature_groups['Nadh Fit']}intensity"]
                fad_intensity = single_cell_features_img[cell_id][f"{all_numerical_feature_groups['Fad Fit']}intensity"]
                normalized_redox = nadh_intensity / (nadh_intensity + fad_intensity + 1e-10)
                single_cell_features_img[cell_id][f"{all_numerical_feature_groups['Nadh Fit']}norm_redox"] = normalized_redox
        # Add morphology features and convert to DataFrame
        single_cell_features_img = get_mask_morphology_features(mask, image_name, single_cell_features_img)
    single_cell_features_img = pd.DataFrame(single_cell_features_img).T
    single_cell_features_img.index.name = "cell_id"
    
    return "", single_cell_features_img

def extract_fit_results(channel, cell_ids, single_cell_features_img, results, decay_curves, num_components):
    """
    Extract fitting results for a specific channel and store them in single_cell_features_img
    
    Args:
        channel: channel name ("nadh" or "fad")
        cell_ids: list of cell ids
        single_cell_features_img: dictionary of single cell features
        results: fitting result dictionary for one cell   
        num_components: number of fitting components
    """
    # Basic parameters (always present)
    if channel == "NADH":
        feature_prefix = all_numerical_feature_groups['Nadh Fit']
    else:
        feature_prefix = all_numerical_feature_groups['Fad Fit']

    for i, cell_id in enumerate(cell_ids):
        if cell_id not in single_cell_features_img:
            single_cell_features_img[cell_id] = {}
        
        single_cell_features_img[cell_id][f"{feature_prefix}amp1"] = results["amp1"][i]
        single_cell_features_img[cell_id][f"{feature_prefix}t1"] = results["t1"][i] * 1000  # Convert to ps
        single_cell_features_img[cell_id][f"{feature_prefix}offset"] = results["offset"][i]
        single_cell_features_img[cell_id][f"{feature_prefix}intensity"] = decay_curves[i].sum()
        if num_components == 2:
            single_cell_features_img[cell_id][f"{feature_prefix}amp2"] = results["amp2"][i]
            single_cell_features_img[cell_id][f"{feature_prefix}t2"] = results["t2"][i] * 1000  # Convert to ps
            # Calculate alpha values
            amp1, amp2 = results["amp1"][i], results["amp2"][i]
            total_amp = amp1 + amp2
            single_cell_features_img[cell_id][f"{feature_prefix}a1"] = (amp1 / total_amp) * 100
            single_cell_features_img[cell_id][f"{feature_prefix}a2"] = (amp2 / total_amp) * 100
            # Calculate mean lifetime (in original units, not converted)
            single_cell_features_img[cell_id][f"{feature_prefix}tm"] = ((amp1 / total_amp) * results["t1"][i] + (amp2 / total_amp) * results["t2"][i]) * 1000
            
        elif num_components == 3:
            single_cell_features_img[cell_id][f"{feature_prefix}amp2"] = results["amp2"][i]
            single_cell_features_img[cell_id][f"{feature_prefix}t2"] = results["t2"][i] * 1000  # Convert to ps
            single_cell_features_img[cell_id][f"{feature_prefix}amp3"] = results["amp3"][i]
            single_cell_features_img[cell_id][f"{feature_prefix}t3"] = results["t3"][i] * 1000  # Convert to ps
            # Calculate alpha values for 3 components
            amp1, amp2, amp3 = results["amp1"][i], results["amp2"][i], results["amp3"][i]
            total_amp = amp1 + amp2 + amp3
            single_cell_features_img[cell_id][f"{feature_prefix}a1"] = (amp1 / total_amp) * 100
            single_cell_features_img[cell_id][f"{feature_prefix}a2"] = (amp2 / total_amp) * 100
            single_cell_features_img[cell_id][f"{feature_prefix}a3"] = (amp3 / total_amp) * 100
            # Calculate mean lifetime for 3 components (in original units, not converted)
            single_cell_features_img[cell_id][f"{feature_prefix}tm"] = ((amp1 / total_amp) * results["t1"][i] + (amp2 / total_amp) * results["t2"][i] + (amp3 / total_amp) * results["t3"][i]) * 1000

    return single_cell_features_img
def extract_fit_free_results(channel, cell_ids, single_cell_features_img, decay_curves, shifted_irf, offsets, time_axis, laser_rate):
    """
    Extract fit free results for a specific channel and store them in single_cell_features_img (for now, only phasor is implemented)
    Args:
        channel: channel name ("nadh" or "fad")
        cell_ids: list of cell ids
        single_cell_features_img: dictionary of single cell features
        decay_curves: list of decay curves
        irf: IRF
        offsets: list of offsets
        time_axis: time axis
        laser_rate: laser repetition rate
    """
    if channel == "NADH":
        feature_prefix = feature_groups_prefix["Fit Free Nadh"]
    else:
        feature_prefix = feature_groups_prefix["Fit Free Fad"]

    for i, cell_id in enumerate(cell_ids):
        if cell_id not in single_cell_features_img:
            single_cell_features_img[cell_id] = {}
            
        # 1st harmonic
        g1, s1, g2, s2, tau_phase, tau_m = get_phasor_features(decay_curves[i], shifted_irf, time_axis, f=laser_rate, offset=offsets[i])
        single_cell_features_img[cell_id][f"{feature_prefix}G(1st)"] = g1
        single_cell_features_img[cell_id][f"{feature_prefix}S(1st)"] = s1
        single_cell_features_img[cell_id][f"{feature_prefix}Tau_phase"] = tau_phase
        single_cell_features_img[cell_id][f"{feature_prefix}Tau_m"] = tau_m
        # 2nd harmonic
        single_cell_features_img[cell_id][f"{feature_prefix}G(2nd)"] = g2
        single_cell_features_img[cell_id][f"{feature_prefix}S(2nd)"] = s2

        
    return single_cell_features_img

def k_flow_fit_extraction(metadata, has_nadh, has_fad):
    error_msg = ""
    image_name = metadata['image_name']
    if has_nadh:
        nadh_decay_path = metadata.get('nadh histogram', None)
        nadh_irf_path = metadata.get('nadh irf', None)
  
        try: 
            nadh_decays = pd.read_csv(nadh_decay_path)
        except Exception as e:
            error_msg = f"Error reading the decay histogram for experiment {image_name} at {nadh_decay_path}: {e}"
            return error_msg, None
        try:
            nadh_irf = np.loadtxt(nadh_irf_path)
        except Exception as e:              
            error_msg = f"Error reading the IRF file for image {image_name} at {nadh_irf_path}: {e}"
            return error_msg, None
        try:
             # check if the shift is provided
            nadh_shift = metadata['nadh_shift']
        except Exception as e:
            return "Error: NADH shift is not provided", None
        try:
            nadh_start = metadata['nadh_start']
            nadh_end = metadata['nadh_end']
        except Exception as e:
            return "Error: NADH start and end are not provided", None
        nadh_shifted_irf = irf_shift(nadh_irf, nadh_shift)
    if has_fad:
        try:
            fad_shift = metadata['fad_shift']
        except Exception as e:
            return "Error: FAD shift is not provided", None
        try:
            fad_irf_path = metadata['red irf']
            fad_irf = np.loadtxt(fad_irf_path)
        except Exception as e:
            error_msg = f"Error reading the IRF file for image {image_name} at {fad_irf_path}: {e}"
            return error_msg, None
        try:
            fad_decay_path = metadata['red histogram']
            fad_decay = pd.read_csv(fad_decay_path)
        except Exception as e:
            error_msg = f"Error reading the decay file for image {image_name} at {fad_decay_path}: {e}"
            return error_msg, None
        shifted_fad_irf = irf_shift(fad_irf, fad_shift)
        try:
            fad_start = metadata['fad_start']
            fad_end = metadata['fad_end']
        except Exception as e:
            return "Error: FAD start and end are not provided", None
    if has_nadh and has_fad:
        if nadh_decays.shape[0] != fad_decay.shape[0]:
            return "Error: NADH and FAD decay curves have different number of cells", None
    if has_nadh:
        cell_ids = [f"{image_name}_{i}" for i in nadh_decays.index]
    elif has_fad:
        cell_ids = [f"{image_name}_{i}" for i in fad_decay.index]
    else:
        return "Error: No NADH or FAD decay curves provided", None
    # get fitting config from metadata
    fitting_mode = metadata['fitting_mode']
    fitting_algo = metadata['fitting_algo']
    time_bins = metadata['time_bins']
    duration = metadata['duration']
    num_components = metadata['num_components']

    laser_rate = metadata['laser_rate']
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
    single_cell_features_img = {}
    if has_nadh:
        single_cell_features_img = fit_and_extract_results("NADH", duration, time_bins, num_components, fitting_algo, fitting_mode, True, laser_rate, time_axis, cell_ids, single_cell_features_img, nadh_start, nadh_end, nadh_decays.values, nadh_shifted_irf)
    if has_fad:
        single_cell_features_img = fit_and_extract_results("FAD", duration, time_bins, num_components, fitting_algo, fitting_mode, True, laser_rate, time_axis, cell_ids, single_cell_features_img, fad_start, fad_end, fad_decay.values, shifted_fad_irf)
    single_cell_features_img = pd.DataFrame(single_cell_features_img).T
    single_cell_features_img.index.name = "cell_id" 
    return "", single_cell_features_img

def image_extraction(metadata, metadata_dict):
    """
    Extract single cell fitting parameters from spc image output files
    """
    for channel_name in metadata_dict["modules"]:
        if "Lifetime" in metadata_dict["modules"][channel_name]:
            if "fit" in metadata_dict["modules"][channel_name]["Lifetime"]:
                num_components = metadata_dict[channel_name]["num_components"]
                if input_type == "SPCImage":
                    
            if "fit_free" in metadata_dict["modules"][channel_name]["Lifetime"]:
                pass

        if "Intensity" in metadata_dict["modules"][channel_name]:
            if "morphology" in metadata_dict["modules"][channel_name]["Intensity"]:
                pass
            if "texture" in metadata_dict["modules"][channel_name]["Intensity"]:
                pass



def fit_and_extract_results(channel, duration, time_bins, num_components, fitting_algo, fitting_mode, fit_free, laser_rate, time_axis, cell_ids, single_cell_features_img, start, end, decay_curves, shifted_irf, extract_fit=True):
    channel_container = st.empty()
    with channel_container.container():
        st.info(f"Fitting {channel} curves for {len(decay_curves)} cells...")
        channel_progress = st.progress(0)
    
    channel_progress_callback = create_progress_callback(channel_progress)
        
    results = fit_curves(duration, time_bins, decay_curves, shifted_irf, num_components, fitting_algo, fitting_mode, start=start, end=end, _progress_callback=channel_progress_callback)
    if extract_fit:
        single_cell_features_img = extract_fit_results(channel, cell_ids, single_cell_features_img, results, decay_curves, num_components)
    channel_container.empty()  # Remove both text and progress bar when done
    if fit_free:
        single_cell_features_img = extract_fit_free_results(channel, cell_ids, single_cell_features_img, decay_curves, shifted_irf, results["offset"], time_axis, laser_rate)

    return single_cell_features_img