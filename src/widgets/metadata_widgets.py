import streamlit as st
import os 
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import get_default_file_suffixes, get_spc_output_suffix, get_default_2D_decay_config, get_default_laser_rate
from src.dataset_io import happy_emoji, sad_emoji
from src.decay_io import read_decay, read_decay_metadata
from collections import Counter
from datetime import datetime

def load_data_suffix_widget(input_types, selected_channels, selected_ch_num_components, selected_feature_extractors):
    """
    """
    actual_file_suffix = {}
    error_msg = ""
    a1_suffix_list = []
    mask_suffix_list = {}
    if "Decay (3/4D) pixel-prefitted" in input_types.values():
        spc_output_suffix = get_spc_output_suffix()
    for i, (channel_key, channel_name) in enumerate(selected_channels.items()):
        input_type = input_types[channel_key]
        file_suffixes = get_default_file_suffixes(channel_key, input_type, selected_feature_extractors[channel_key])
        if len(file_suffixes) == 0:
            error_msg += f"No file suffixes found for {channel_name} {sad_emoji}"
            return "", error_msg
        else: 
            actual_file_suffix[channel_name] = file_suffixes

        st.subheader(f"File suffixes: {channel_name}")
        num_cols = 3
        cols = st.columns(num_cols)
        for j, (file_type, default_suffix) in enumerate(actual_file_suffix[channel_name].items()):
            if file_type == "a1":
                a1_suffix_list.append(default_suffix)
            elif file_type == "Mask":
                mask_suffix_list[channel_name] = default_suffix
            col = cols[j % num_cols]
            with col:
                # only show the help message for the first file type of the first channel
                if i == 0 and j == 0:
                    help_msg = "The filenames are expected to have *exactly* two parts: *image_name + suffix*. All files from the same image should share the **same** image_name, with the only difference being the suffix."
                elif i == 0 and input_type == "Decay (3/4D) pixel-prefitted" and file_type == "a1":
                    help_msg = f"For other SPCImage output files (e.g. t1, t2), the suffixes are automatically generated based on the provided a1 suffix by replacing {spc_output_suffix['a1']} to get the others."
                else:
                    help_msg = None
                suffix = st.text_input(f"{file_type}", default_suffix, key=f"{channel_name}_{input_type}_{file_type}_suffix", help=help_msg)
                if suffix == "":
                    error_msg += f"Please provide a suffix for {file_type}! "
                else:
                    actual_file_suffix[channel_name][file_type] = suffix
        if input_type == "Decay (3/4D) pixel-prefitted" and error_msg == "": # write the spc outputs' suffixes for this channel
            if channel_name in selected_ch_num_components and selected_ch_num_components[channel_name] != 0:
                num_components = selected_ch_num_components[channel_name]
                if num_components == 1:
                    needed_suffix = ["t1"]
                elif num_components == 2:
                    needed_suffix = ["t1", "t2"]
                elif num_components == 3:
                    needed_suffix = ["t1", "a2", "t2", "t3"]
                for key in needed_suffix:
                    actual_file_suffix[channel_name][key] = actual_file_suffix[channel_name]["a1"].replace(spc_output_suffix["a1"], spc_output_suffix[key])

    # check for duplicates in a1_suffix_list
    if len(set(a1_suffix_list)) != len(a1_suffix_list):
        error_msg += f"Duplicate a1 suffixes found: {a1_suffix_list} {sad_emoji}"

    # output info message for channels that share the same mask suffix
    mask_suffix_seen = {}
    for channel_name, mask_suffix in mask_suffix_list.items():
        if mask_suffix not in mask_suffix_seen:
            mask_suffix_seen[mask_suffix] = [channel_name]
        else:
            mask_suffix_seen[mask_suffix].append(channel_name)
    for mask_suffix, channel_names in mask_suffix_seen.items():
        if len(channel_names) > 1:
            st.info(f"Channels {channel_names} share the same mask suffix: {mask_suffix}")

    # flatten the actual_file_suffix dictionary
    actual_file_suffix_dict = {}
    for channel_name, file_suffix_dict in actual_file_suffix.items():
        for file_type, file_suffix in file_suffix_dict.items():
            actual_file_suffix_dict[f"{channel_name}_{file_type}"] = file_suffix
    return actual_file_suffix_dict, error_msg

@st.cache_data
def load_list_data_from_folder_widget(folder_path, file_suffix, num_cols=3):    
    """
    Load data from a folder and check its validity. Display the file sets for each image group. 
    file_names = image_name + suffix (exactly that, no more, no less)
    image_group: keyed by image_name, and the value is a list of all the files that belong to that image
    """
    
    valid_image_groups = {}

    # Single recursive scan to get all files

    path = Path(folder_path)
    all_files = [str(file) for file in path.rglob("*") if file.is_file()]
    
    if len(all_files) == 0:
        st.warning(f"No files found in folder: **{folder_path}**.")
        return {}
    
    # Build lookup dictionaries for fast access
    files_by_name = {}  # exact filename -> list of file paths
    files_by_suffix = {}  # suffix -> list of file paths
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Index by exact filename
        if filename not in files_by_name:
            files_by_name[filename] = []
        files_by_name[filename].append(file_path)
        
        # Index by suffix for each suffix we care about
        for suffix in set(file_suffix.values()):
            if filename.endswith(suffix):
                if suffix not in files_by_suffix:
                    files_by_suffix[suffix] = []
                files_by_suffix[suffix].append(file_path)

    # use the first key to get the list of images (it does not matter which key to use, since they are all required, they should all be there)
    image_search_suffix = list(file_suffix.values())[0]
    image_files = files_by_suffix.get(image_search_suffix, [])
    if len(image_files) == 0:
        st.warning(f"No image files found with suffix: **{image_search_suffix}**.")
        return {}
    
    # get the image names from the file names by removing the suffix
    image_names = [os.path.basename(file).removesuffix(image_search_suffix) for file in image_files]
    # for each image name, build a widget card with the image name and the files that belong to it
    num_images = len(image_names)
    num_cols = min(num_cols, num_images)
    rows = (num_images + num_cols - 1) // num_cols

    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row * num_cols + col_idx
            if img_idx >= num_images:
                break
            image_name = image_names[img_idx]
            image_group = {}
            missing_keys = []
            duplicate_keys = []
            # get the list of files that belong to this image
            for key, suffix in file_suffix.items():
                # find the file with the exact name: image_name + suffix recursively within the folder (except for IRF)
                if "IRF" not in key:
                    filename = image_name + suffix
                    matched_files = files_by_name.get(filename, [])
                else:
                    matched_files = files_by_suffix.get(suffix, [])
                    
                if len(matched_files) != 1:
                    if len(matched_files) > 1:
                        duplicate_keys.append(key) # more than one file found
                    else:
                        missing_keys.append(key) # no matching file found
                else:
                    image_group[key] = matched_files[0]

            # create the card 
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"Image name: **{image_name}**")
                    if missing_keys or duplicate_keys:
                        st.write("❌ Missing or duplicate files:")
                        for key in missing_keys:
                            if "IRF" not in key:
                                st.write(f"- Missing {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Missing {key} with suffix: {file_suffix[key]}")
                        for key in duplicate_keys:
                            if "IRF" not in key:
                                st.write(f"- Duplicate {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Duplicate {key} with suffix: {file_suffix[key]}")

                    else:
                        st.write("✅ All files found.")
                   

            if missing_keys == [] and duplicate_keys == []:
                valid_image_groups[image_name] = image_group

    return valid_image_groups

def display_feature_groups_widget(metadata_df, num_cols=3):
    """
    Display the feature groups to be extracted.
    """
    # if there are more than 3 rows, write the first 3 rows, else write all rows
    if len(metadata_df) > 3:
        st.write(metadata_df.head(3))
    else:
        st.write(metadata_df)

    # cols = st.columns(num_cols)
    # keys = list(selected_feature_types.keys())
    # chunk_size = (len(keys) + num_cols - 1) // num_cols  # split into 3 roughly equal parts

    # for i, col in enumerate(cols):
    #     for key in keys[i * chunk_size : (i + 1) * chunk_size]:
    #         values = selected_feature_types[key]
    #         # Option 1: use a Markdown newline
    #         col.markdown(
    #             f"""
    #             <div style="
    #                 border:1px solid #ccc;
    #                 padding:8px;
    #                 border-radius:4px;
    #                 margin-bottom:8px;
    #             ">
    #                 <strong style="color: orange;">{key}</strong><br>
    #                 { ', '.join(values) }
    #             </div>
    #             """,
    #             unsafe_allow_html=True
    #         )

def export_metadata_widget(images_df, folder_path):
    # use a botton to export the images as one csv file (one image per row) to the folder_path 
    confirm_export = st.button("Export Image Metadata as CSV", help=f"Export the image meta as one csv file (one image per row) to {folder_path}")
    if confirm_export:
        # convert the dictionary to a dataframe     
        # save the dataframe to a csv file
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = os.path.join(folder_path, f"image_metadata_{time_stamp}.csv")
        try:
            images_df.to_csv(csv_file_path) # Save the DataFrame
        except Exception as e:
            st.error(f"Error exporting the image metadata: {e}. Is the previous metadata file open in another program?")
            return
        st.success(f"Image metadata exported successfully to {csv_file_path} {happy_emoji}")
        st.session_state["last_extracted_metadata"] = images_df
        st.session_state["last_extracted_metadata_filepath"] = csv_file_path

@st.cache_data
def check_raw_decay_data(images_df, channel_name):
    """
    Check if the sdt data is available.
    """
    column_name = f"{channel_name}_Decay"
    if column_name not in images_df.columns:
        return "Error: No sdt data found. Please check the data.", []

    shape_list = []
    laser_rep_time_list = []
    for i, row in images_df.iterrows():
        error_msg, decay_data = read_decay(row[column_name])
        if error_msg != "":
            return error_msg, [], None, None
        shape_list.append(decay_data.shape)
        error_msg, laser_rep_time = read_decay_metadata(row[column_name])
        if error_msg != "":
            return error_msg, [], None, None
        laser_rep_time_list.append(laser_rep_time)
        shape_list.append(decay_data.shape)

    
    # check for the consistency of the shape, a tuple
    if len(set(shape_list)) > 1:
        shape_counts = Counter(shape_list)
        error_msg = f"Inconsistent sdt data shapes found for {channel_name} decay: \n"
        for shape, count in shape_counts.items():
            error_msg += f"- Shape {shape} appears {count} times.\n"
        return error_msg, [], None, None
    if len(set(laser_rep_time_list)) > 1:
        error_msg = f"Inconsistent laser rep time found for {channel_name} decay: \n"
        for laser_rep_time, count in laser_rep_time_list.items():
            error_msg += f"- Laser rep time {laser_rep_time} appears {count} times.\n"
        return error_msg, [], None, None
    else:
        # get the first shape
        shape = shape_list[0]
        time_bins = shape[-1]
        laser_rep_time = laser_rep_time_list[0]
        if len(shape) == 3:
            return "", [-1], time_bins, laser_rep_time
        elif len(shape) == 4:
            # get all non-zero channels
            non_zero_channels = []
            for i in range(shape[0]):
                if np.any(decay_data[i]):
                    non_zero_channels.append(i)
            return "", non_zero_channels, time_bins, laser_rep_time

def check_raw_2D_decay_data(images_df, channel_name):
    for i, row in images_df.iterrows():
        try:
            decay_data = pd.read_csv(row[f"{channel_name}_Decay"], header=None)
        except Exception as e:
            return f"Error reading decay data for {channel_name}: {e}", None
        return "", decay_data.shape[1]

def check_assign_channel_widget(images_df, selected_channels, flim_decay_input_type, imaging_modalities, duration=None, time_bins=None):   
    error_msg = ""
    time_bins_list = []
    laser_rep_time_list = []
    num_cols = len(selected_channels)
    cols = st.columns(num_cols)
    for i, (channel_key, channel_name) in enumerate(selected_channels.items()):
        imaging_modality = imaging_modalities[channel_key]
        if imaging_modality != "FLIM":
            continue
        else:
            decay_col_name = f"{channel_name}_Decay"
            if decay_col_name not in images_df.columns:
                return "Error: File paths for decay data are not provided.", None
            if flim_decay_input_type == "Decay (2D)":
                if duration is not None:
                    images_df["duration"] = duration
                else:
                    return "Error: Duration is not provided.", None
                if time_bins is not None:
                    error_msg, time_bins = check_raw_2D_decay_data(images_df, channel_name)
                    if error_msg == "":
                        time_bins_list.append(time_bins)
                    else:
                        return error_msg, None
            else: # 3/4D decay
                with cols[i]:
                    error_msg, available_channels, time_bins, laser_rep_time = check_raw_decay_data(images_df, channel_name)
                    if error_msg == "":
                        if len(available_channels) == 1:
                            images_df[f"{channel_name}_channel"] = available_channels[0]
                            time_bins_list.append(time_bins)
                            laser_rep_time_list.append(laser_rep_time)
                    else:
                        images_df[f"{channel_name}_channel"] = st.selectbox("Select the sdt channel for nadh decay", available_channels)
                    time_bins_list.append(time_bins)
                    laser_rep_time_list.append(laser_rep_time)
            
    if len(set(time_bins_list)) > 1:
        return "Inconsistent time bins found for the selected channels. Please check the data.", None
    elif len(time_bins_list) == 0:
        return "No time bins found for the selected channels. Please check the data.", None
    else:
        images_df["time_bins"] = time_bins_list[0]

    if len(set(laser_rep_time_list)) > 1:
        return "Inconsistent laser rep time found for the selected channels. Please check the data.", None
    elif len(laser_rep_time_list) == 0:
        return "No laser rep time found for the selected channels. Please check the data.", None
    else:
        images_df["duration"] = laser_rep_time_list[0]

    return error_msg, images_df

def lifetime_data_config_widget(selected_feature_extractors, input_type):
    fit_free = False
    duration = time_bins = laser_rate = None
    for _, extractors in selected_feature_extractors.items():
        if "Lifetime fit free" in extractors:
            fit_free = True
    if input_type == "Decay (2D)":
        default_2D_decay_duration, default_2D_decay_time_bins = get_default_2D_decay_config()
        cols = st.columns(3 if fit_free else 2)
        with cols[0]:
            duration = st.number_input("Duration (s)", value=default_2D_decay_duration, min_value=0.0, max_value=100.0, key="2D_decay_duration")
        with cols[1]:
            time_bins = st.number_input("Time bins", value=default_2D_decay_time_bins, min_value=256, max_value=2048, key="2D_decay_time_bins")
        if fit_free:
            default_laser_rate = get_default_laser_rate(input_type)
            with cols[2]:
                laser_rate = st.number_input("Laser rate (GHz)", value=default_laser_rate, min_value=0.0, max_value=2.0, key="2D_decay_laser_rate")
    else: 
        if fit_free:
            default_laser_rate = get_default_laser_rate(input_type)
            laser_rate = st.number_input("Laser rate (GHz)", value=default_laser_rate, min_value=0.0, max_value=2.0, key="laser_rate")
    return duration, time_bins, laser_rate