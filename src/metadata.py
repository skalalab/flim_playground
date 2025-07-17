from pathlib import Path
from src.config import get_available_feature_extractors, get_file_types

def get_ch_info(metadata_df):
    # get available channels in the metadata file
    # use the {channel_name}_input_type column name to get available channels
    available_channels = [col for col in metadata_df.columns if col.endswith("_input_type")]
    available_channels = [col.split("_input_type")[0] for col in available_channels]
    available_channels = list(set(available_channels))
    if len(available_channels) == 0:
        return f"No channels found in metadata file.", None
    metadata_dict = {}
    metadata_dict["channels_shift"] = {}
    metadata_dict["channel_names"] = []
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
                    return f"Decay input type should be consistent across all channels.", None

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
        if "Lifetime fit free" in selected_feature_extractors:
            if channel_name not in metadata_dict["channels_shift"]:
                # if not using fitting to find the shift, then use fit free to find the shift
                metadata_dict["channels_shift"][channel_name] = "fit free"
        
        if "Decay (3/4D)" in input_type:
            metadata_dict[channel_name]["channel_no"] = metadata_df[f"{channel_name}_channel"].iloc[0] 

    return "", metadata_dict


def parse_metadata_file(metadata_df, fov_name_col):
    """
    Parse the metadata file and return a dictionary of metadata.
    metadata_df: pandas dataframe of metadata
    """
    # check for required column
    if fov_name_col not in metadata_df.columns:
       return f"Image name column {fov_name_col} not found in metadata file.", None
    # check for unique image name
    if metadata_df[fov_name_col].duplicated().any():
        return f"Image name column {fov_name_col} is not unique.", None

    error_msg, metadata_dict = get_ch_info(metadata_df)
    for channel_name in metadata_dict["channel_names"]:
        # check for file paths
        input_type = metadata_dict[channel_name]["input_type"]
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
 
    # check for time bins, duration, laser rate
    if "time_bins" in metadata_df.columns:
        if metadata_df["time_bins"].nunique() != 1:
            return f"Time bins column {f"{channel_name}_time_bins"} is not consistent.", None
        metadata_dict["time_bins"] = metadata_df["time_bins"].iloc[0]
    else:
        return f"Time bins column {f"{channel_name}_time_bins"} not found in metadata file.", None
    if "duration" in metadata_df.columns:
        if metadata_df["duration"].nunique() != 1:
            return f"Duration column {f"{channel_name}_duration"} is not consistent.", None
        metadata_dict["duration"] = metadata_df["duration"].iloc[0]
    else:
        return f"Duration column {f"{channel_name}_duration"} not found in metadata file.", None
    
    if "laser_rate" in metadata_df.columns:
        if metadata_df["laser_rate"].nunique() != 1:
            return f"Laser rate column {f"{channel_name}_laser_rate"} is not consistent.", None
        metadata_dict["laser_rate"] = metadata_df["laser_rate"].iloc[0]
    else:
        return f"Laser rate column {f"{channel_name}_laser_rate"} not found in metadata file.", None

    return error_msg, metadata_dict