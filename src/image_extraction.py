import pandas as pd
import numpy as np
from skimage.measure import regionprops
from src.feature_groups import feature_groups_prefix, feature_groups_features, feature_distribution_vars
from src.file_io import load_image
from src.sdt_io import read_sdt150
from src.fit import fit_curves
from src.fit_helper import irf_shift
from src.phasor import get_phasor_features
import streamlit as st

def get_mask_morphology_features(mask, image_name, cell_dict):
        # get mask morphology features
    mask_morphology_features = feature_groups_features["Mask Morphology"]
    mask_props = regionprops(label_image=mask)
    for region in mask_props:
        cell_id = f"{image_name}_{region.label}"
        if cell_id not in cell_dict:
            cell_dict[cell_id] = {}
        for feature in mask_morphology_features:
            feature_name = f"{feature_groups_prefix['Mask Morphology']}{feature}"
            if feature in region:
                cell_dict[cell_id][feature_name] = region[feature]
            elif feature == "circularity":
                cell_dict[cell_id][feature_name] = 4 * np.pi * region.area / region.perimeter**2 if region.perimeter > 0 else 0
    return cell_dict

def spcimage_fit_extraction(metadata, has_nadh, has_fad, mask):

    image_props = {}
    if has_nadh:
        try:
            nadh_a1 = load_image(metadata['nadh a1'])
            # SPC image will output 0 for the thresholded pixels (background), so we need to mask them
            nadh_a1 = np.ma.masked_array(nadh_a1, mask=nadh_a1==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the NADH a1 file: {metadata['nadh a1']}: {e}", None
        if mask.shape != nadh_a1.shape:
            return f"Error: NADH a1 file has a different shape than the mask file: {nadh_a1.shape} != {mask.shape}", None
        try:
            nadh_a2 = load_image(metadata['nadh a2'])
            nadh_a2 = np.ma.masked_array(nadh_a2, mask=nadh_a2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the NADH a2 file: {metadata['nadh a2']}: {e}", None
        if mask.shape != nadh_a2.shape:
            return f"Error: NADH a2 file has a different shape than the mask file: {nadh_a2.shape} != {mask.shape}", None
        try:
            nadh_t1 = load_image(metadata['nadh t1'])
            nadh_t1 = np.ma.masked_array(nadh_t1, mask=nadh_t1==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the NADH t1 file: {metadata['nadh t1']}: {e}", None
        if mask.shape != nadh_t1.shape:
            return f"Error: NADH t1 file has a different shape than the mask file: {nadh_t1.shape} != {mask.shape}", None
        try:
            nadh_t2 = load_image(metadata['nadh t2'])
            nadh_t2 = np.ma.masked_array(nadh_t2, mask=nadh_t2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the NADH t2 file: {metadata['nadh t2']}: {e}", None
        if mask.shape != nadh_t2.shape:
            return f"Error: NADH t2 file has a different shape than the mask file: {nadh_t2.shape} != {mask.shape}", None
        nadh_tm= (nadh_a1 / 100 * nadh_t1) + (nadh_a2 / 100 * nadh_t2)

        try: 
            nadh_intensity = load_image(metadata['nadh intensity'])
        except Exception as e:
            return f"Error reading the NADH intensity file: {metadata['nadh intensity']}: {e}", None
        if mask.shape != nadh_intensity.shape:
            return f"Error: NADH intensity file has a different shape than the mask file: {nadh_intensity.shape} != {mask.shape}", None
        nadh_feature_prefix = feature_groups_prefix['Nadh Fit']
        image_props[f"{nadh_feature_prefix}a1"] = regionprops(label_image=mask, intensity_image=nadh_a1)
        image_props[f"{nadh_feature_prefix}a2"] = regionprops(label_image=mask, intensity_image=nadh_a2)
        image_props[f"{nadh_feature_prefix}t1"] = regionprops(label_image=mask, intensity_image=nadh_t1)
        image_props[f"{nadh_feature_prefix}t2"] = regionprops(label_image=mask, intensity_image=nadh_t2)
        image_props[f"{nadh_feature_prefix}tm"] = regionprops(label_image=mask, intensity_image=nadh_tm)
        image_props[f"{nadh_feature_prefix}intensity"] = regionprops(label_image=mask, intensity_image=nadh_intensity)

    if has_fad:
        try:
            fad_a1 = load_image(metadata['fad a1'])
            fad_a1 = np.ma.masked_array(fad_a1, mask=fad_a1==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the FAD a1 file: {metadata['fad a1']}: {e}", None
        if mask.shape != fad_a1.shape:
            return f"Error: FAD a1 file has a different shape than the mask file: {fad_a1.shape} != {mask.shape}", None 
        try:
            fad_a2 = load_image(metadata['fad a2'])
            fad_a2 = np.ma.masked_array(fad_a2, mask=fad_a2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the FAD a2 file: {metadata['fad a2']}: {e}", None
        if mask.shape != fad_a2.shape:
            return f"Error: FAD a2 file has a different shape than the mask file: {fad_a2.shape} != {mask.shape}", None
        try:
            fad_t1 = load_image(metadata['fad t1'])
            fad_t1 = np.ma.masked_array(fad_t1, mask=fad_t1==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the FAD t1 file: {metadata['fad t1']}: {e}", None
        if mask.shape != fad_t1.shape:
            return f"Error: FAD t1 file has a different shape than the mask file: {fad_t1.shape} != {mask.shape}", None
        try:
            fad_t2 = load_image(metadata['fad t2'])
            fad_t2 = np.ma.masked_array(fad_t2, mask=fad_t2==0, fill_value=np.nan)
        except Exception as e:
            return f"Error reading the FAD t2 file: {metadata['fad t2']}: {e}", None
        if mask.shape != fad_t2.shape:
            return f"Error: FAD t2 file has a different shape than the mask file: {fad_t2.shape} != {mask.shape}", None
        fad_tm = (fad_a1 / 100 * fad_t1) + (fad_a2 / 100 * fad_t2)
        try: 
            fad_intensity = load_image(metadata['fad intensity'])
        except Exception as e:
            return f"Error reading the FAD intensity file: {metadata['fad intensity']}: {e}", None
        if mask.shape != fad_intensity.shape:
            return f"Error: FAD intensity file has a different shape than the mask file: {fad_intensity.shape} != {mask.shape}", None
        fad_feature_prefix = feature_groups_prefix['Fad Fit']
        image_props[f"{fad_feature_prefix}a1"] = regionprops(label_image=mask, intensity_image=fad_a1)
        image_props[f"{fad_feature_prefix}a2"] = regionprops(label_image=mask, intensity_image=fad_a2)
        image_props[f"{fad_feature_prefix}t1"] = regionprops(label_image=mask, intensity_image=fad_t1)
        image_props[f"{fad_feature_prefix}t2"] = regionprops(label_image=mask, intensity_image=fad_t2)
        image_props[f"{fad_feature_prefix}tm"] = regionprops(label_image=mask, intensity_image=fad_tm)
        image_props[f"{fad_feature_prefix}intensity"] = regionprops(label_image=mask, intensity_image=fad_intensity)

    # calculate redox
    if has_nadh and has_fad:
        norm_redox = nadh_intensity / (nadh_intensity + fad_intensity + 1e-10)
        image_props[f"{nadh_feature_prefix}norm_redox"] = regionprops(label_image=mask, intensity_image=norm_redox)
    image_name = metadata['image_name']
    single_cell_features_img = {}
    fit_fd_prefix = feature_groups_prefix["Feature Distribution Fit"]
    for prop in image_props:
        for region in image_props[prop]:
            cell_id = f"{image_name}_{region.label}"
            if cell_id not in single_cell_features_img:
                single_cell_features_img[cell_id] = {}
            single_cell_features_img[cell_id][prop] = region.intensity_mean
            single_cell_features_img[cell_id][f"{prop}_stdev"] = region.intensity_std
            # add fit variable feature distribution features for this fit 
            for feature_distribution_var in feature_distribution_vars:
                if feature_distribution_var == "polarity":
                    geometric_centroid = region.centroid
                    weighted_centroid = region.centroid_weighted
                    # example: fit_nadh: a1 -> nadh_a1
                    feature_name = prop.replace("fit_", "").replace(": ", "_") 
                    feature_name = f"{fit_fd_prefix}{feature_name}_{feature_distribution_var}"
                    single_cell_features_img[cell_id][feature_name] = np.sqrt(
                        (geometric_centroid[0] - weighted_centroid[0]) ** 2 +
                        (geometric_centroid[1] - weighted_centroid[1]) ** 2)
                    
    single_cell_features_img = get_mask_morphology_features(mask, image_name, single_cell_features_img)
    
    # convert single_cell_features_img to a dataframe
    single_cell_features_img = pd.DataFrame(single_cell_features_img).T
    # name the index as cell_id
    single_cell_features_img.index.name = "cell_id"
   
    return "", single_cell_features_img

@st.cache_data
def roi_summing_fit_extraction(metadata, has_nadh, has_fad, mask, fit_free):

    # checks
    image_name = metadata['image_name']
    if has_nadh:
        try:
             # check if the shift is provided
            nadh_shift = metadata['nadh_shift']
        except Exception as e:
            return "Error: NADH shift is not provided", None

        try:
            nadh_irf_path = metadata['nadh irf']
            nadh_irf = np.loadtxt(nadh_irf_path)
        except Exception as e:
            error_msg = f"Error reading the IRF file for image {image_name} at {nadh_irf_path}: {e}"
            return error_msg, None

        try:
            nadh_decay_path = metadata['nadh decay']
            nadh_decay = read_sdt150(nadh_decay_path)
        except Exception as e:
            error_msg = f"Error reading the decay file for image {image_name} at {nadh_decay_path}: {e}"
            return error_msg, None
        shifted_nadh_irf = irf_shift(nadh_irf, nadh_shift)
        try:
            nadh_start = metadata['nadh_start']
            nadh_end = metadata['nadh_end']
        except Exception as e:
            return "Error: NADH start and end are not provided", None
        
    if has_fad:
        try:
            fad_shift = metadata['fad_shift']
        except Exception as e:
            return "Error: FAD shift is not provided", None
        try:
            fad_irf_path = metadata['fad irf']
            fad_irf = np.loadtxt(fad_irf_path)
        except Exception as e:
            error_msg = f"Error reading the IRF file for image {image_name} at {fad_irf_path}: {e}"
            return error_msg, None
        try:
            fad_decay_path = metadata['fad decay']
            fad_decay = read_sdt150(fad_decay_path)
        except Exception as e:
            error_msg = f"Error reading the decay file for image {image_name} at {fad_decay_path}: {e}"
            return error_msg, None
        shifted_fad_irf = irf_shift(fad_irf, fad_shift)
        try:
            fad_start = metadata['fad_start']
            fad_end = metadata['fad_end']
        except Exception as e:
            return "Error: FAD start and end are not provided", None
        
    # get fitting config from metadata
    fitting_mode = metadata['fitting_mode']
    fitting_algo = metadata['fitting_algo']
    time_bins = metadata['time_bins']
    duration = metadata['duration']
    num_components = metadata['num_components']
    if fit_free:
        laser_rate = metadata['laser_rate']
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
    
    # Fit all curves at once
    if has_nadh and nadh_decay_curves:
        nadh_container = st.empty()
        with nadh_container.container():
            st.info(f"Fitting NADH curves for {len(nadh_decay_curves)} cells...")
            nadh_progress = st.progress(0)
        
        def nadh_progress_callback(current, total):
            progress = (current + 1) / total
            nadh_progress.progress(progress)
            
        nadh_results = fit_curves(duration, time_bins, nadh_decay_curves, shifted_nadh_irf, num_components, fitting_algo, fitting_mode, start=nadh_start, end=nadh_end, _progress_callback=nadh_progress_callback)
        single_cell_features_img = extract_fit_results("nadh", cell_ids, single_cell_features_img, nadh_results, nadh_decay_curves, num_components)
        nadh_container.empty()  # Remove both text and progress bar when done
        if fit_free:
            single_cell_features_img = extract_fit_free_results("nadh", cell_ids, single_cell_features_img, nadh_decay_curves, shifted_nadh_irf, nadh_results["offset"], time_axis, laser_rate)
        
    if has_fad and fad_decay_curves:
        fad_container = st.empty()
        with fad_container.container():
            st.info(f"Fitting FAD curves for {len(fad_decay_curves)} cells...")
            fad_progress = st.progress(0)
        
        def fad_progress_callback(current, total):
            progress = (current + 1) / total
            fad_progress.progress(progress)
            
        fad_results = fit_curves(duration, time_bins, fad_decay_curves, shifted_fad_irf, num_components, fitting_algo, fitting_mode, start=fad_start, end=fad_end, _progress_callback=fad_progress_callback)
        single_cell_features_img = extract_fit_results("fad", cell_ids, single_cell_features_img, fad_results, fad_decay_curves, num_components)    
        fad_container.empty()  # Remove both text and progress bar when done

        if fit_free:
            single_cell_features_img = extract_fit_free_results("fad", cell_ids, single_cell_features_img, fad_decay_curves, shifted_fad_irf, fad_results["offset"], time_axis, laser_rate)
            
    # calculate redox
    if has_nadh and has_fad:
        # get the intensity of NADH and FAD
        for cell_id in single_cell_features_img:
            nadh_intensity = single_cell_features_img[cell_id][f"{feature_groups_prefix['Nadh Fit']}intensity"]
            fad_intensity = single_cell_features_img[cell_id][f"{feature_groups_prefix['Fad Fit']}intensity"]
            normalized_redox = nadh_intensity / (nadh_intensity + fad_intensity + 1e-10)
            single_cell_features_img[cell_id][f"{feature_groups_prefix['Nadh Fit']}norm_redox"] = normalized_redox
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
    if channel == "nadh":
        feature_prefix = feature_groups_prefix['Nadh Fit']
    else:
        feature_prefix = feature_groups_prefix['Fad Fit']

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
    if channel == "nadh":
        feature_prefix = feature_groups_prefix["Fit Free Nadh"]
    else:
        feature_prefix = feature_groups_prefix["Fit Free Fad"]

    for i, cell_id in enumerate(cell_ids):
        if cell_id not in single_cell_features_img:
            single_cell_features_img[cell_id] = {}
        g, s, tau_phase, tau_m = get_phasor_features(decay_curves[i], shifted_irf, time_axis, f=laser_rate, offset=offsets[i])
        single_cell_features_img[cell_id][f"{feature_prefix}G(1st)"] = g
        single_cell_features_img[cell_id][f"{feature_prefix}S(1st)"] = s
        single_cell_features_img[cell_id][f"{feature_prefix}Tau_phase"] = tau_phase
        single_cell_features_img[cell_id][f"{feature_prefix}Tau_m"] = tau_m
    return single_cell_features_img

def image_fit_extraction(metadata, analysis_type, has_nadh, has_fad, fit_free):
    """
    Extract single cell fitting parameters from spc image output files
    """
    try: 
        mask = load_image(metadata['mask'])
    except Exception as e:
        return f"Error reading the mask file: {metadata['mask']}: {e}", None
    if analysis_type == "SPCImage":
        error_msg, single_cell_features_img = spcimage_fit_extraction(metadata, has_nadh, has_fad, mask)
        if error_msg != "":
            return error_msg, None
    elif analysis_type == "ROI Summing Fit":
        error_msg, single_cell_features_img = roi_summing_fit_extraction(metadata, has_nadh, has_fad, mask, fit_free)
        if error_msg != "":
            return error_msg, None
    return "", single_cell_features_img
