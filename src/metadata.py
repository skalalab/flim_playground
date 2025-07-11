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
        error_msg += f"Image name column {image_name_col} not found in metadata file."
    # check for unique image name
    if metadata_df[image_name_col].duplicated().any():
        error_msg += f"Image name column {image_name_col} is not unique."
    # check for input type
    if "input_type" not in metadata_df.columns:
        error_msg += f"Input type column not found in metadata file."
    # check for consistent input type
    if metadata_df["input_type"].nunique() != 1:
        error_msg += f"Input type column is not consistent."
    
    input_type = metadata_df["input_type"].iloc[0]
   
    metadata_dict["input_type"] = input_type
    channel_modules = get_ch_modules(metadata_df)
    metadata_dict["modules"] = channel_modules

    channels = channel_modules.keys()
    available_file_types = get_all_file_types()
    # check for num_components
    for channel_name in channels:
        metadata_dict[channel_name] = {}
        if "Lifetime" in channel_modules[channel_name] and "fit" in channel_modules[channel_name]["Lifetime"]:
            component_col = f"{channel_name}_num_components"
            if component_col not in metadata_df.columns:
                error_msg += f"Component column {component_col} not found in metadata file."
            else:
                if metadata_df[component_col].nunique() != 1:
                    error_msg += f"Component column {component_col} is not consistent."
                else:
                    metadata_dict[channel_name]["num_components"] = metadata_df[component_col].iloc[0]
        if input_type == "ROI Summing Fit" or input_type == "SPCImage":
            # get decay info from the metadata file
            # get channel number, time bins, duration
            if f"{channel_name}_channel" in metadata_df.columns:
                if metadata_df[f"{channel_name}_channel"].nunique() != 1:
                    error_msg += f"Channel number column {f"{channel_name}_channel"} is not consistent."
                else:
                    metadata_dict[channel_name]["channel_number"] = metadata_df[f"{channel_name}_channel"].iloc[0]
            else:
                error_msg += f"Channel number column {f"{channel_name}_channel"} not found in metadata file."
            if f"{channel_name}_time_bins" in metadata_df.columns:
                if metadata_df[f"{channel_name}_time_bins"].nunique() != 1:
                    error_msg += f"Time bins column {f"{channel_name}_time_bins"} is not consistent."
                metadata_dict[channel_name]["time_bins"] = metadata_df[f"{channel_name}_time_bins"].iloc[0]
            else:
                error_msg += f"Time bins column {f"{channel_name}_time_bins"} not found in metadata file."
            if f"{channel_name}_duration" in metadata_df.columns:
                if metadata_df[f"{channel_name}_duration"].nunique() != 1:
                    error_msg += f"Duration column {f"{channel_name}_duration"} is not consistent."
                metadata_dict[channel_name]["duration"] = metadata_df[f"{channel_name}_duration"].iloc[0]
            else:
                error_msg += f"Duration column {f"{channel_name}_duration"} not found in metadata file."


        # check for file paths
        for file_type in available_file_types:
            if f"{channel_name}_{file_type}" in metadata_df.columns:
               # then this is a column storing file paths 
               # check if all file paths are valid and if they are unique
               if metadata_df[f"{channel_name}_{file_type}"].duplicated().any():
                   error_msg += f"File paths for {channel_name}_{file_type} are not unique."

               # check if the file paths are valid
               for file_path in metadata_df[f"{channel_name}_{file_type}"]:
                   if not Path(file_path).exists():
                       error_msg += f"File path {file_path} for {channel_name}_{file_type} is not valid."
    st.write(metadata_dict)
    return error_msg, metadata_dict