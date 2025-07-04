from pathlib import Path
from src.feature_groups import subset_feature_group_features

unique_cell_id_col = "cell_id"
required_cols = [unique_cell_id_col, "image_name"]

spc_output_suffix = {
    "a1": "_a1[%].asc",
    "a2": "_a2[%].asc",
    "t1": "_t1.asc",
    "t2": "_t2.asc",
    "intensity": "_photons.asc",
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
    if "image_name" not in metadata_df.columns and "kflow_exp_name" not in metadata_df.columns:
        error_msg += "The required column `image_name` or `kflow_exp_name` not found in the metadata file! "
        return error_msg, None, None, None, None, None

    if "fit_free" not in metadata_df.columns:
        error_msg += "The required column `fit_free` not found in the metadata file! "
        return error_msg, None, None, None, None, None
    
    if "analysis_type" not in metadata_df.columns:
        error_msg += "The required column `analysis_type` not found in the metadata file! "
        return error_msg, None, None, None, None, None
    # check if analysis_type consistent for all rows
    if not metadata_df["analysis_type"].nunique() == 1:
        error_msg += "The analysis type is not consistent for all rows! "
        return error_msg, None, None, None, None, None
    analysis_type = metadata_df["analysis_type"].iloc[0]
    if len(metadata_df) == 0:
        error_msg += "The metadata file is empty! "
        return error_msg, None, None, None, None, None
    fit_free = bool(metadata_df["fit_free"].iloc[0])
    # determine the avilable feature groups based on the metadata file
    has_nadh = has_fad =  has_mask = feature_distribution = False

    if "nadh histogram" in metadata_df.columns or "red histogram" in metadata_df.columns:
        # k-flow
        if analysis_type != "K-Flow":
            error_msg += f"The analysis type should be K-Flow but got {analysis_type}."
            return error_msg, None, None, None, None, None
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

    elif "mask" in metadata_df.columns:
        # for other analysis types requires mask
        has_mask = True
        if "nadh a1" in metadata_df.columns or "fad a1" in metadata_df.columns:
            # spc image and fit free
            feature_distribution = True
            has_nadh = "nadh a1" in metadata_df.columns
            has_fad = "fad a1" in metadata_df.columns
            if analysis_type != "SPCImage":
                error_msg += f"The analysis type should be SPCImage but got {analysis_type}."
                return error_msg, None, None, None, None, None
            if fit_free:
                if "nadh decay" not in metadata_df.columns and "fad decay" not in metadata_df.columns:
                    error_msg += "The required columns `nadh decay` and `fad decay` not found in the metadata file! "
                    return error_msg, None, None, None, None, None
        else:
            # ROI summing fit
            has_nadh = "nadh decay" in metadata_df.columns
            has_fad = "fad decay" in metadata_df.columns
            if analysis_type != "ROI Summing Fit":
                error_msg += f"The analysis type should be ROI Summing Fit but got {analysis_type}."
                return error_msg, None, None, None, None, None
        if fit_free or analysis_type == "ROI Summing Fit":
            if has_nadh:
                if "nadh irf" not in metadata_df.columns:
                    error_msg += "The required column `nadh irf` not found in the metadata file! "
                    return error_msg, None, None, None, None, None
                if "nadh_channel" not in metadata_df.columns:
                    error_msg += "The required column `nadh_channel` not found in the metadata file! "
                    return error_msg, None, None, None, None, None
            if has_fad:
                if "fad irf" not in metadata_df.columns:
                    error_msg += "The required column `fad irf` not found in the metadata file! "
                    return error_msg, None, None, None, None, None
                if "fad_channel" not in metadata_df.columns:
                    error_msg += "The required column `fad_channel` not found in the metadata file! "
                    return error_msg, None, None, None, None, None
    else: 
        error_msg += "Cannot determine the analysis type from the metadata file! "
        return error_msg, None, None, None, None, None
    
    if not has_nadh and not has_fad:
        error_msg += "Neither NADH nor FAD found in the metadata file! "
        return error_msg, None, None, None, None, None
   
    available_feature_groups_features = subset_feature_group_features(
        has_nadh=has_nadh,
        has_fad=has_fad,
        fit_free=fit_free,
        has_mask=has_mask,
        feature_distribution=feature_distribution
    )
    
    return error_msg, available_feature_groups_features, analysis_type, fit_free, has_nadh, has_fad
   