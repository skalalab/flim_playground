import pandas as pd
import numpy as np
from skimage.measure import regionprops
from src.feature_groups import feature_groups_prefix, feature_groups_features, feature_distribution_vars
from src.file_io import load_image

def spcimage_fit_extraction(metadata, has_nadh, has_fad, mask):

    image_props = {}
    if has_nadh:
        try:
            nadh_a1 = load_image(metadata['nadh a1'])
            # SPC image will output 0 for the thresholded pixels (background), so we need to mask them
            nadh_a1 = np.ma.masked_array(nadh_a1, mask=nadh_a1==0)

        except Exception as e:
            return f"Error reading the NADH a1 file: {metadata['nadh a1']}: {e}", None
        if mask.shape != nadh_a1.shape:
            return f"Error: NADH a1 file has a different shape than the mask file: {nadh_a1.shape} != {mask.shape}", None
        try:
            nadh_a2 = load_image(metadata['nadh a2'])
            nadh_a2 = np.ma.masked_array(nadh_a2, mask=nadh_a2==0)
        except Exception as e:
            return f"Error reading the NADH a2 file: {metadata['nadh a2']}: {e}", None
        if mask.shape != nadh_a2.shape:
            return f"Error: NADH a2 file has a different shape than the mask file: {nadh_a2.shape} != {mask.shape}", None
        try:
            nadh_t1 = load_image(metadata['nadh t1'])
            nadh_t1 = np.ma.masked_array(nadh_t1, mask=nadh_t1==0)
        except Exception as e:
            return f"Error reading the NADH t1 file: {metadata['nadh t1']}: {e}", None
        if mask.shape != nadh_t1.shape:
            return f"Error: NADH t1 file has a different shape than the mask file: {nadh_t1.shape} != {mask.shape}", None
        try:
            nadh_t2 = load_image(metadata['nadh t2'])
            nadh_t2 = np.ma.masked_array(nadh_t2, mask=nadh_t2==0)
        except Exception as e:
            return f"Error reading the NADH t2 file: {metadata['nadh t2']}: {e}", None
        if mask.shape != nadh_t2.shape:
            return f"Error: NADH t2 file has a different shape than the mask file: {nadh_t2.shape} != {mask.shape}", None
        nadh_tm= (nadh_a1 / 100 * nadh_t1) + (nadh_a2 / 100 * nadh_t2)
        nadh_feature_prefix = feature_groups_prefix['Nadh Fit']
        image_props[f"{nadh_feature_prefix}a1"] = regionprops(label_image=mask, intensity_image=nadh_a1)
        image_props[f"{nadh_feature_prefix}a2"] = regionprops(label_image=mask, intensity_image=nadh_a2)
        image_props[f"{nadh_feature_prefix}t1"] = regionprops(label_image=mask, intensity_image=nadh_t1)
        image_props[f"{nadh_feature_prefix}t2"] = regionprops(label_image=mask, intensity_image=nadh_t2)
        image_props[f"{nadh_feature_prefix}tm"] = regionprops(label_image=mask, intensity_image=nadh_tm)

    if has_fad:
        try:
            fad_a1 = load_image(metadata['fad a1'])
            fad_a1 = np.ma.masked_array(fad_a1, mask=fad_a1==0)
        except Exception as e:
            return f"Error reading the FAD a1 file: {metadata['fad a1']}: {e}", None
        if mask.shape != fad_a1.shape:
            return f"Error: FAD a1 file has a different shape than the mask file: {fad_a1.shape} != {mask.shape}", None 
        try:
            fad_a2 = load_image(metadata['fad a2'])
            fad_a2 = np.ma.masked_array(fad_a2, mask=fad_a2==0)
        except Exception as e:
            return f"Error reading the FAD a2 file: {metadata['fad a2']}: {e}", None
        if mask.shape != fad_a2.shape:
            return f"Error: FAD a2 file has a different shape than the mask file: {fad_a2.shape} != {mask.shape}", None
        try:
            fad_t1 = load_image(metadata['fad t1'])
            fad_t1 = np.ma.masked_array(fad_t1, mask=fad_t1==0)
        except Exception as e:
            return f"Error reading the FAD t1 file: {metadata['fad t1']}: {e}", None
        if mask.shape != fad_t1.shape:
            return f"Error: FAD t1 file has a different shape than the mask file: {fad_t1.shape} != {mask.shape}", None
        try:
            fad_t2 = load_image(metadata['fad t2'])
            fad_t2 = np.ma.masked_array(fad_t2, mask=fad_t2==0)
        except Exception as e:
            return f"Error reading the FAD t2 file: {metadata['fad t2']}: {e}", None
        if mask.shape != fad_t2.shape:
            return f"Error: FAD t2 file has a different shape than the mask file: {fad_t2.shape} != {mask.shape}", None
        fad_tm = (fad_a1 / 100 * fad_t1) + (fad_a2 / 100 * fad_t2)
        fad_feature_prefix = feature_groups_prefix['Fad Fit']
        image_props[f"{fad_feature_prefix}a1"] = regionprops(label_image=mask, intensity_image=fad_a1)
        image_props[f"{fad_feature_prefix}a2"] = regionprops(label_image=mask, intensity_image=fad_a2)
        image_props[f"{fad_feature_prefix}t1"] = regionprops(label_image=mask, intensity_image=fad_t1)
        image_props[f"{fad_feature_prefix}t2"] = regionprops(label_image=mask, intensity_image=fad_t2)
        image_props[f"{fad_feature_prefix}tm"] = regionprops(label_image=mask, intensity_image=fad_tm)
    
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


    # get mask morphology features
    mask_morphology_features = feature_groups_features["Mask Morphology"]
    mask_props = regionprops(label_image=mask)
    for region in mask_props:
        cell_id = f"{image_name}_{region.label}"
        for feature in mask_morphology_features:
            feature_name = f"{feature_groups_prefix['Mask Morphology']}{feature}"
            if feature in region:
                single_cell_features_img[cell_id][feature_name] = region[feature]
            elif feature == "circularity":
                single_cell_features_img[cell_id][feature_name] = 4 * np.pi * region.area / region.perimeter**2 if region.perimeter > 0 else 0
    
    # convert single_cell_features_img to a dataframe
    single_cell_features_img = pd.DataFrame(single_cell_features_img).T
    # name the index as cell_id
    single_cell_features_img.index.name = "cell_id"

   
    return "", single_cell_features_img

def image_fit_extraction(metadata, analysis_type, has_nadh, has_fad):
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
        
        
    return "", single_cell_features_img
