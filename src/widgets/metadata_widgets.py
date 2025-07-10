import streamlit as st
import os 
import numpy as np
from pathlib import Path
from src.metadata import parse_metadata_file
from src.config import get_file_suffixes, get_spc_output_suffix
from src.widgets.data_widgets import happy_emoji, sad_emoji
from src.sdt_io import read_sdt150, read_sdt_metadata
from collections import Counter
def load_data_suffix_widget(input_type, selected_channels, selected_ch_num_components):
    """
    """
    actual_file_suffix = {}
    error_msg = ""
    if input_type == "SPCImage":
        spc_output_suffix = get_spc_output_suffix()
    for i, (channel_key, channel_name) in enumerate(selected_channels.items()):
        file_suffixes = get_file_suffixes(channel_key, input_type)
        if len(file_suffixes) == 0:
            error_msg += f"No file suffixes found for {channel_name} {sad_emoji}"
            return "", error_msg
        else:
            actual_file_suffix[channel_name] = file_suffixes

        st.subheader(f"File suffixes: {channel_name}")
        num_cols = 3
        cols = st.columns(num_cols)
        for j, (file_type, default_suffix) in enumerate(file_suffixes.items()):
            col = cols[j % num_cols]
            with col:
                # only show the help message for the first file type of the first channel
                if i == 0 and j == 0:
                    help_msg = "The filenames are expected to have *exactly* two parts: *image_name + suffix*. All files from the same image should share the **same** image_name, with the only difference being the suffix."
                elif i == 0 and input_type == "SPCImage" and file_type == "a1":
                    help_msg = f"For other SPCImage output files (e.g. t1, a2, t2), the suffixes are automatically generated based on the provided a1 suffix by replacing {spc_output_suffix['a1']} to get the others."
                else:
                    help_msg = None
                suffix = st.text_input(f"{file_type}", default_suffix, key=f"{channel_key}_{input_type}_{file_type}_suffix", help=help_msg)
                if suffix == "":
                    error_msg += f"Please provide a suffix for {file_type}! "
                else:
                    actual_file_suffix[channel_name][file_type] = suffix
        if input_type == "SPCImage" and error_msg == "": # write the spc outputs' suffixes for this channel
            if channel_key in selected_ch_num_components and selected_ch_num_components[channel_key] != 0:
                num_components = selected_ch_num_components[channel_key]
                if num_components == 1:
                    needed_suffix = ["t1"]
                elif num_components == 2:
                    needed_suffix = ["t1", "a2", "t2"]
                elif num_components == 3:
                    needed_suffix = ["t1", "a2", "t2", "a3", "t3"]
                for key in needed_suffix:
                    actual_file_suffix[channel_name][key] = actual_file_suffix[channel_name]["a1"].replace(spc_output_suffix["a1"], spc_output_suffix[key])
        

    return actual_file_suffix, error_msg



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
        for suffix in file_suffix.values():
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
                if "irf" not in key:
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
                            if "irf" not in key:
                                st.write(f"- Missing {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Missing {key} with suffix: {file_suffix[key]}")
                        for key in duplicate_keys:
                            if "irf" not in key:
                                st.write(f"- Duplicate {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Duplicate {key} with suffix: {file_suffix[key]}")

                    else:
                        st.write("✅ All files found.")
                   

            if missing_keys == [] and duplicate_keys == []:
                valid_image_groups[image_name] = image_group

    return valid_image_groups

def export_metadata_widget(images_df, folder_path):
    # use a botton to export the images as one csv file (one image per row) to the folder_path 
    confirm_export = st.button("Export Image Metadata as CSV", help=f"Export the image meta as one csv file (one image per row) to {folder_path}")
    if confirm_export:
        # convert the dictionary to a dataframe     
        # save the dataframe to a csv file
        csv_file_path = os.path.join(folder_path, "image_metadata.csv")
        try:
            images_df.to_csv(csv_file_path) # Save the DataFrame
        except Exception as e:
            st.error(f"Error exporting the image metadata: {e}. Is the previous metadata file open in another program?")
            return
        st.success(f"Image metadata exported successfully to {csv_file_path} {happy_emoji}")
        st.session_state["last_extracted_metadata"] = images_df
        st.session_state["last_extracted_metadata_filepath"] = csv_file_path

def parse_metadata_display_feature_widget(metadata_df, num_cols=3): 
    """
    Parse the metadata and display the features available to be extracted later for user to choose. 
    """
    error_msg, available_feature_groups_features, analysis_type, fit_free, has_nadh, has_fad =  parse_metadata_file(metadata_df)
    if error_msg != "":
        st.error(error_msg)
        return 

    # display the available features in a multi select widget, one group per widget
    cols = st.columns(num_cols)
    keys = list(available_feature_groups_features.keys())
    chunk_size = (len(keys) + num_cols - 1) // num_cols  # split into 3 roughly equal parts

    for i, col in enumerate(cols):
        for key in keys[i * chunk_size : (i + 1) * chunk_size]:
            values = available_feature_groups_features[key]
            # Option 1: use a Markdown newline
            col.markdown(
                f"""
                <div style="
                    border:1px solid #ccc;
                    padding:8px;
                    border-radius:4px;
                    margin-bottom:8px;
                ">
                    <strong style="color: orange;">{key}</strong><br>
                    { ', '.join(values) }
                </div>
                """,
                unsafe_allow_html=True
            )


@st.cache_data
def check_sdt_data(images_df, channel):
    """
    Check if the sdt data is available.
    """
    if channel == "nadh":
        column_name = "nadh decay"
    elif channel == "fad":
        column_name = "fad decay"
    else:
        return "Error: Invalid channel", []
    if column_name not in images_df.columns:
        return "Error: No sdt data found. Please check the data.", []

    shape_list = []
    laser_rep_time_list = []
    for i, row in images_df.iterrows():
        sdt_data = read_sdt150(row[column_name])
        shape_list.append(sdt_data.shape)
        laser_rep_time = read_sdt_metadata(row[column_name])
        laser_rep_time_list.append(laser_rep_time)
        shape_list.append(sdt_data.shape)

    
    # check for the consistency of the shape, a tuple
    if len(set(shape_list)) > 1:
        shape_counts = Counter(shape_list)
        error_msg = f"Inconsistent sdt data shapes found for {channel} decay: \n"
        for shape, count in shape_counts.items():
            error_msg += f"- Shape {shape} appears {count} times.\n"
        return error_msg, [], None, None
    if len(set(laser_rep_time_list)) > 1:
        error_msg = f"Inconsistent laser rep time found for {channel} decay: \n"
        for laser_rep_time, count in laser_rep_time_list.items():
            error_msg += f"- Laser rep time {laser_rep_time} appears {count} times.\n"
        return error_msg, [], None, None
    else:
        # get the first shape
        shape = shape_list[0]
        time_bins = shape[2]
        laser_rep_time = laser_rep_time_list[0]
        if len(shape) == 3:
            return "", [-1], time_bins, laser_rep_time
        elif len(shape) == 4:
            # get all non-zero channels
            non_zero_channels = []
            for i in range(shape[0]):
                if np.any(sdt_data[i]):
                    non_zero_channels.append(i)
            return "", non_zero_channels, time_bins, laser_rep_time


def check_sdt_channel_widget(images_df):   
    error_msg = ""
    col1, col2 = st.columns(2)
    with col1:
        if "nadh decay" in images_df.columns:
            error_msg, available_nadh_sdt_channels, nadh_time_bins, nadh_laser_rep_time = check_sdt_data(images_df, "nadh")
            if error_msg == "":
                if len(available_nadh_sdt_channels) == 1:
                    images_df["nadh_channel"] = available_nadh_sdt_channels[0]
                else:
                    images_df["nadh_channel"] = st.selectbox("Select the sdt channel for nadh decay", available_nadh_sdt_channels)
                images_df["time_bins"] = nadh_time_bins
                images_df["duration"] = nadh_laser_rep_time
            else:
                return error_msg, None
    with col2:
        if "fad decay" in images_df.columns:
            error_msg, available_fad_sdt_channels, fad_time_bins, fad_laser_rep_time = check_sdt_data(images_df, "fad")
            if error_msg == "":
                if len(available_fad_sdt_channels) == 1:
                    images_df["fad_channel"] = available_fad_sdt_channels[0]
                else:   
                    images_df["fad_channel"] = st.selectbox("Select the sdt channel for fad decay", available_fad_sdt_channels)
                images_df["time_bins"] = fad_time_bins
                images_df["duration"] = fad_laser_rep_time
            else:
                return error_msg, None

    if "nadh_decay" in images_df.columns and "fad_decay" in images_df.columns:
        if nadh_time_bins != fad_time_bins:
            return "Inconsistent time bins found for nadh and fad decay. Please check the data.", None
        if nadh_laser_rep_time != fad_laser_rep_time:
            return "Inconsistent laser rep time found for nadh and fad decay. Please check the data.", None
      

    return error_msg, images_df
