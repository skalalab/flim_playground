import pandas as pd
import numpy as np
from src.feature_groups import feature_groups_prefix
from src.file_io import load_image

def spcimage_fit_extraction(metadata, has_nadh, has_fad, mask):
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
        nadh_tau_mean = (nadh_a1 / 100 * nadh_t1) + (nadh_a2 / 100 * nadh_t2)
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
        fad_tau_mean = (fad_a1 / 100 * fad_t1) + (fad_a2 / 100 * fad_t2)
    image_name = metadata['image_name']
    single_cell_features_img = pd.DataFrame()

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
