from pathlib import Path
from src.config import get_all_channel_names, get_all_feature_extractors, get_all_file_types
import streamlit as st
def get_ch_modules(metadata_df):
    available_channels = get_all_channel_names()
    available_feature_extractors = get_all_feature_extractors()
    ch_modules = {}
    column_names = metadata_df.columns
    for channel_name in available_channels:
        for feature_extractor in available_feature_extractors.keys():
            available_modules = available_feature_extractors[feature_extractor]
            for module in available_modules:
                if f"{channel_name}_{feature_extractor}_{module}" in column_names:             
                    if channel_name not in ch_modules:
                        ch_modules[channel_name] = {}
                    if feature_extractor not in ch_modules[channel_name]:    
                        ch_modules[channel_name][feature_extractor] = []
                    ch_modules[channel_name][feature_extractor].append(module)
    
    return ch_modules


def parse_metadata_file(metadata_df, image_name_col):
    """
    Parse the metadata file and return a dictionary of metadata.
    metadata_df: pandas dataframe of metadata
    """
    error_msg = ""
    metadata_dict = {}
    # check for required column
    if image_name_col not in metadata_df.columns:
       return f"Image name column {image_name_col} not found in metadata file.", None
    # check for unique image name
    if metadata_df[image_name_col].duplicated().any():
        return f"Image name column {image_name_col} is not unique.", None
    # check for input type
    if "input_type" not in metadata_df.columns:
        return f"Input type column not found in metadata file.", None
    # check for consistent input type
    if metadata_df["input_type"].nunique() != 1:
        return f"Input type column is not consistent.", None
    
    input_type = metadata_df["input_type"].iloc[0]
   
    metadata_dict["input_type"] = input_type
    channel_modules = get_ch_modules(metadata_df)
    metadata_dict["modules"] = channel_modules
    metadata_dict["channels_fit"] = []
    metadata_dict["channels_fit_free"] = []
    channels = channel_modules.keys()
    available_file_types = get_all_file_types()
    # check for num_components
    for channel_name in channels:
        # spcimage is already fitted 
        if "Lifetime" in channel_modules[channel_name] and "fit" in channel_modules[channel_name]["Lifetime"]:
            metadata_dict["channels_fit"].append(channel_name)
        if "Lifetime" in channel_modules[channel_name] and "fit free" in channel_modules[channel_name]["Lifetime"]:
            metadata_dict["channels_fit_free"].append(channel_name)
        metadata_dict[channel_name] = {}
        if "Lifetime" in channel_modules[channel_name] and "fit" in channel_modules[channel_name]["Lifetime"]:
            component_col = f"{channel_name}_num_components"
            if component_col not in metadata_df.columns:
                return f"Component column {component_col} not found in metadata file.", None
            else:
                if metadata_df[component_col].nunique() != 1:
                    return f"Component column {component_col} is not consistent.", None
                else:
                    metadata_dict[channel_name]["num_components"] = metadata_df[component_col].iloc[0]
        if input_type == "ROI Summing Fit" or input_type == "SPCImage":
            # get decay info from the metadata file
            # get channel number, time bins, duration
            if f"{channel_name}_channel" in metadata_df.columns:
                if metadata_df[f"{channel_name}_channel"].nunique() != 1:
                    return f"Channel number column {f"{channel_name}_channel"} is not consistent.", None
                else:
                    metadata_dict[channel_name]["channel_number"] = metadata_df[f"{channel_name}_channel"].iloc[0]
            else:
                return f"Channel number column {f"{channel_name}_channel"} not found in metadata file.", None
        # check for file paths
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
    # Create channels_shift as a dictionary with channel names as keys
    metadata_dict["channels_shift"] = {}
    channels_to_shift = set(metadata_dict["channels_fit"] + metadata_dict["channels_fit_free"])
    for channel_name in channels_to_shift:
        if channel_name in metadata_dict["channels_fit"] and input_type != "SPCImage":
            metadata_dict["channels_shift"][channel_name] = "fit"
        else:
            metadata_dict["channels_shift"][channel_name] = "fit free"
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