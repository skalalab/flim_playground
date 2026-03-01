from pathlib import Path
from src.config import get_available_feature_extractors, get_file_types, get_fov_name_col, get_unique_cell_id_col

def get_ch_info(metadata_df):
    # get available channels in the metadata file
    # use the {channel_name}_input_type column name to get available channels
    available_channels = [col for col in metadata_df.columns if col.endswith("_input_type")]
    available_channels = [col.split("_input_type")[0] for col in available_channels]
    available_channels = list(dict.fromkeys(available_channels))
    if len(available_channels) == 0:
        return f"No channels found in metadata file.", None
    metadata_dict = {}
    metadata_dict["channels_shift"] = {}
    metadata_dict["channel_names"] = []
    fit_free = False
    for channel_name in available_channels:
        if channel_name not in metadata_dict:
            metadata_dict[channel_name] = {}
            metadata_dict["channel_names"].append(channel_name)
       
        # get input type
        input_type_col = f"{channel_name}_input_type"
        # check for consistency of input type
        if metadata_df[input_type_col].nunique() != 1:
            return f"Input type column {input_type_col} is not consistent.", None
        input_type = metadata_df[input_type_col].iloc[0]
        metadata_dict[channel_name]["input_type"] = input_type

         # get imaging modality
        imaging_modality_col = f"{channel_name}_imaging_modality"
        if imaging_modality_col not in metadata_df.columns:
            return f"Imaging modality column {imaging_modality_col} not found in metadata file.", None
        metadata_dict[channel_name]["imaging_modality"] = metadata_df[imaging_modality_col].iloc[0]
        if metadata_dict[channel_name]["imaging_modality"] == "FLIM":
            # get decay input type
            if "decay_input_type" not in metadata_dict:
                metadata_dict["decay_input_type"] = input_type
            else:
                if metadata_dict["decay_input_type"] != input_type:
                    return "Decay input type should be consistent across all channels.", None

        # get selected feature extractors
        available_feature_extractors = get_available_feature_extractors(input_type)
        selected_feature_extractors = []
        for feature_extractor in available_feature_extractors:
            feature_extractor_col = f"{channel_name}_{feature_extractor}"
            if feature_extractor_col in metadata_df.columns:
                if feature_extractor not in metadata_dict:
                    metadata_dict[feature_extractor] = [channel_name]
                else:
                    metadata_dict[feature_extractor].append(channel_name)
                selected_feature_extractors.append(feature_extractor)
        if len(selected_feature_extractors) == 0:
            return f"No feature extractors found for channel {channel_name}.", None
        metadata_dict[channel_name]["selected_feature_extractors"] = selected_feature_extractors
        # get num_components
        if "Lifetime fit" in selected_feature_extractors:
            num_components_col = f"{channel_name}_num_components"
            if num_components_col not in metadata_df.columns:
                return f"Num components column {num_components_col} not found in metadata file.", None
            metadata_dict[channel_name]["num_components"] = metadata_df[num_components_col].iloc[0]
            if "prefitted" not in input_type:
                # use fitting to find the shift, if it is already fitted, then do not use fitting to find shift (if needed)
                metadata_dict["channels_shift"][channel_name] = "fit"
            # Read fixed-lifetime columns (optional — may or may not be present in CSV)
            import pandas as _pd
            fixed_lifetimes = {}
            for t_key in ["t1", "t2", "t3"]:
                col = f"{channel_name}_fixed_{t_key}"
                if col in metadata_df.columns:
                    val = metadata_df[col].iloc[0]
                    fixed_lifetimes[t_key] = None if (_pd.isna(val) or val == 0) else float(val)
                else:
                    fixed_lifetimes[t_key] = None
            metadata_dict[channel_name]["fixed_lifetimes"] = fixed_lifetimes
        if "Lifetime fit free" in selected_feature_extractors:
            fit_free = True
            if channel_name not in metadata_dict["channels_shift"]:
                # no need to shift if channel-specific fluorescence lifetime standard file is provided
                channel_ref_col = f"{channel_name}_Fluorescence Lifetime Standard"
                if channel_ref_col in metadata_df.columns:
                    continue
                else:
                    metadata_dict["channels_shift"][channel_name] = "fit free"
        
        if "Decay (3/4D)" in input_type:
            if "prefitted" in input_type:
                if len(selected_feature_extractors) == 1 and "Lifetime fit" in selected_feature_extractors:
                    # only lifetime fit is selected, and data is prefitted, no decay channel needed
                    continue
            if f"{channel_name}_channel" not in metadata_df.columns:
                return f"Channel number column {channel_name}_channel not found in metadata file.", None
            else:
                metadata_dict[channel_name]["channel_no"] = metadata_df[f"{channel_name}_channel"].iloc[0]

    metadata_dict["unique_cell_id_col"] = get_unique_cell_id_col()
    metadata_dict["fov_name_col"] = get_fov_name_col()
    
    if fit_free:    # laser rate is only needed when fit free 
        if "laser_rate" in metadata_df.columns:
            if metadata_df["laser_rate"].nunique() != 1:
                return "Laser rate column laser_rate is not consistent.", None
            metadata_dict["laser_rate"] = metadata_df["laser_rate"].iloc[0]
        else:
            return f"Laser rate column laser_rate not found in metadata file.", None
        
        if "fit_free_calibration_method" in metadata_df.columns:
            if metadata_df["fit_free_calibration_method"].nunique() != 1:
                return "Fit free calibration method column fit_free_calibration_method is not consistent.", None
            metadata_dict["fit_free_calibration_method"] = metadata_df["fit_free_calibration_method"].iloc[0]
            if metadata_dict["fit_free_calibration_method"] == "Fluorescence Lifetime Standard":
                # lifetime is global and must be present
                if "fluorescence_lifetime_standard_lifetime" in metadata_df.columns:
                    if metadata_df["fluorescence_lifetime_standard_lifetime"].nunique() != 1:
                        return "Fluorescence lifetime standard's lifetime column fluorescence_lifetime_standard_lifetime is not consistent.", None
                    metadata_dict["fluorescence_lifetime_standard_lifetime"] = metadata_df["fluorescence_lifetime_standard_lifetime"].iloc[0]
                else:
                    return "Fluorescence lifetime standard's lifetime column fluorescence_lifetime_standard_lifetime not found in metadata file.", None
                # channel-specific fluorescence lifetime standard file and time axis
                for channel_name in metadata_dict["channel_names"]:
                    if "Lifetime fit free" not in metadata_dict[channel_name]["selected_feature_extractors"]:
                        continue
                    ref_col = f"{channel_name}_Fluorescence Lifetime Standard"
                    if ref_col not in metadata_df.columns:
                        return f"Fluorescence lifetime standard's file column {ref_col} not found in metadata file.", None
                    # Must be consistent across rows
                    if metadata_df[ref_col].nunique() != 1:
                        return f"Fluorescence lifetime standard's file column {ref_col} is not consistent.", None
                    metadata_dict[channel_name]["fluorescence_lifetime_standard_file"] = metadata_df[ref_col].iloc[0]
                    time_axis_col = f"{channel_name}_fluorescence_lifetime_standard_time_axis"
                    if time_axis_col not in metadata_df.columns:
                        return f"Fluorescence lifetime standard's time axis column `{time_axis_col}` not found in metadata file.", None
                    if metadata_df[time_axis_col].nunique() != 1:
                        return f"Fluorescence lifetime standard's time axis column {time_axis_col} is not consistent.", None
                    metadata_dict[channel_name]["fluorescence_lifetime_standard_time_axis"] = metadata_df[time_axis_col].iloc[0]
        else:
            return "Fit free calibration method column fit_free_calibration_method not found in metadata file.", None

    return "", metadata_dict


def parse_metadata_file(metadata_df, fov_name_col):
    """
    Parse the metadata file and return a dictionary of metadata.
    metadata_df: pandas dataframe of metadata
    """
    # check for required column
    if fov_name_col not in metadata_df.columns:
       return f"Column of field of view names `{fov_name_col}` not found in metadata file.", None
    # check for unique image name
    if metadata_df[fov_name_col].duplicated().any():
        return f"Field of view names are not unique. Check the column `{fov_name_col}`.", None
    has_flim = False
    error_msg, metadata_dict = get_ch_info(metadata_df)
    # If channel info parsing failed, return early to avoid subscripting None
    if error_msg != "" or metadata_dict is None:
        return error_msg if error_msg != "" else "Channel info parsing failed.", None
    for channel_name in metadata_dict["channel_names"]:
        # check for file paths
        input_type = metadata_dict[channel_name]["input_type"]
        if metadata_dict[channel_name]["imaging_modality"] == "FLIM":
            if "prefitted" in input_type:
                feature_extractors = metadata_dict[channel_name]["selected_feature_extractors"]
                if "Lifetime fit" in feature_extractors and len(feature_extractors) == 1:
                    has_flim = False
                else:
                    has_flim = True
            else:
                has_flim = True
        available_file_types = get_file_types(input_type)
        for file_type in available_file_types:
            if f"{channel_name}_{file_type}" in metadata_df.columns and file_type != "IRF":
               # then this is a column storing file paths 
               # check if all file paths are valid and if they are unique
               if metadata_df[f"{channel_name}_{file_type}"].duplicated().any():
                   return f"File paths for {channel_name}_{file_type} are not unique.", None

               # check if the file paths are valid
               for file_path in metadata_df[f"{channel_name}_{file_type}"]:
                   if not Path(file_path).exists():
                       return f"File path {file_path} for {channel_name}_{file_type} is not valid.", None
    
    if has_flim:
        # check for time bins, duration
        if "time_bins" in metadata_df.columns:
            if metadata_df["time_bins"].nunique() != 1:
                return f"Time bins column time_bins is not consistent.", None
            metadata_dict["time_bins"] = metadata_df["time_bins"].iloc[0]
        else:
            return f"Time bins column time_bins not found in metadata file.", None
        if "duration" in metadata_df.columns:
            if metadata_df["duration"].nunique() != 1:
                return f"Duration column duration is not consistent.", None
            metadata_dict["duration"] = metadata_df["duration"].iloc[0]
        else:
            return f"Duration column duration not found in metadata file.", None

    return error_msg, metadata_dict