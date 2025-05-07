from pathlib import Path

from src.feature_groups import subset_feature_group_features
file_suffix_default = {
    'mask': '_mask.tiff',
    'nadh decay': 'n.sdt',
    'fad decay': 'f.sdt',
    'single cell features': '.csv',
    'nadh histogram': '_ch1.csv',
    'red histogram': '_ch2.csv',
    "nadh a1": "n_a1[%].asc",
    "fad a1": "f_a1[%].asc",
    'nadh irf': '.txt',
    'fad irf': '.txt',
    'red irf': '.txt', 
}

spc_output_suffix = {
    "a1": "_a1[%].asc",
    "a2": "_a2[%].asc",
    "t1": "_t1.asc",
    "t2": "_t2.asc",
    "shift": "_shift.asc",
   # "intensity": "_photons.asc",
}

def list_files_with_suffix(folder_path, suffix):
    path = Path(folder_path)
    # rglob searches files recursively
    return [str(file) for file in path.rglob("*") if file.name.endswith(suffix)]

def list_files_with_filename(folder_path, filename):
    path = Path(folder_path)
    # rglob searches files recursively
    return [str(file) for file in path.rglob("*") if file.name == filename]

def parse_metadata_file(metadata_df):
    """
    Parse the metadata file and return a dictionary of metadata.
    metadata_df: pandas dataframe of metadata
    returns: 
    - feature_groups_features: dictionary of feature groups and their features that are a subset of the full feature groups features
    """
    error_msg = ""
    available_feature_groups_features = {}
    analysis_type = ""
    # check if the metadata file has the required columns
    if "image_name" not in metadata_df.columns:
        error_msg += "The required column `image_name` not found in the metadata file! "
        return error_msg, None, None, None, None, None

    if "fit_free" not in metadata_df.columns:
        error_msg += "The required column `fit_free` not found in the metadata file! "
        return error_msg, None, None, None, None, None
    if len(metadata_df) == 0:
        error_msg += "The metadata file is empty! "
        return error_msg, None, None, None, None, None
    fit_free = bool(metadata_df["fit_free"].iloc[0])
    # determine the avilable feature groups based on the metadata file
    has_nadh = has_fad =  has_mask = feature_distribution = False

    if "nadh histogram" in metadata_df.columns or "red histogram" in metadata_df.columns:
        # k-flow
        if "nadh histogram" in metadata_df.columns:
            if "nadh irf" not in metadata_df.columns:
                error_msg += "The required column `nadh irf` not found in the metadata file! "
                return error_msg, None, None, None, None, None
            has_nadh = True
        if "red histogram" in metadata_df.columns:
            if "red irf" not in metadata_df.columns:
                error_msg += "The required column `red irf` not found in the metadata file! "
                return error_msg, None, None, None, None, None
            has_fad = True
        analysis_type = "K-Flow"
    elif "mask" in metadata_df.columns:
        # for other analysis types requires mask
        has_mask = True
        if "nadh decay" in metadata_df.columns or "fad decay" in metadata_df.columns:
            if "nadh shift" in metadata_df.columns or "fad shift" in metadata_df.columns:
                # spc image and fit free
                feature_distribution = True
                has_nadh = "nadh shift" in metadata_df.columns
                has_fad = "fad shift" in metadata_df.columns
                analysis_type = "SPCImage"
            else:
                # ROI summing fit
                has_nadh = "nadh decay" in metadata_df.columns
                has_fad = "fad decay" in metadata_df.columns
                analysis_type = "ROI Summing Fit"
            if has_nadh:
                if "nadh irf" not in metadata_df.columns:
                    error_msg += "The required column `nadh irf` not found in the metadata file! "
                    return error_msg, None, None, None, None, None
            if has_fad:
                if "fad irf" not in metadata_df.columns:
                    error_msg += "The required column `fad irf` not found in the metadata file! "
                    return error_msg, None, None, None, None, None
        else: # SPCImage without fit free
            if "nadh a1" in metadata_df.columns or "fad a1" in metadata_df.columns:
                feature_distribution = True
                has_nadh = "nadh a1" in metadata_df.columns
                has_fad = "fad a1" in metadata_df.columns
                analysis_type = "SPCImage"
            else:
                error_msg += "Cannot determine the analysis type from the metadata file! "
                return error_msg, None, None, None, None, None
    else: 
        error_msg += "Cannot determine the analysis type from the metadata file! "
        return error_msg, None, None, None, None, None
   
    available_feature_groups_features = subset_feature_group_features(
        has_nadh=has_nadh,
        has_fad=has_fad,
        fit_free=fit_free,
        has_mask=has_mask,
        feature_distribution=feature_distribution
    )
    
    return error_msg, available_feature_groups_features, analysis_type, fit_free, has_nadh, has_fad
   