import streamlit as st
import os
import pandas as pd
import time
from src.navigation import render_top_menu
from src.dataset_io import happy_emoji, sad_emoji
from src.widgets.numeric_extraction_widgets import fov_extraction_widget
from src.widgets.metadata_widgets import load_list_data_from_folder_widget, load_data_suffix_widget, export_metadata_widget, display_feature_groups_widget, check_assign_channel_widget, lifetime_data_config_widget
from src.widgets.category_widgets import map_categories_to_labels_widget, find_available_dfs_widget, check_and_merge_df_widget
from src.widgets.lifetime_widgets import fit_options_widget, choose_shift_widget
from src.metadata import parse_metadata_file
from src.config import get_imaging_modality, get_input_types, get_channel_names, get_num_components, get_selected_feature_extractors, get_fov_name_col, get_decay_input_type

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

if "last_extracted_metadata" not in st.session_state:
    st.session_state["last_extracted_metadata"] = None
if "last_extracted_metadata_filepath" not in st.session_state:
    st.session_state["last_extracted_metadata_filepath"] = None
if "choosing_shift" not in st.session_state:
    st.session_state["choosing_shift"] = False
if "shift_ready" not in st.session_state:
    st.session_state["shift_ready"] = False
st.title("Data Extraction")

col1, col2 = st.columns([0.4, 1])
steps = ["Image Metadata Extraction", "Numeric Feature Extraction", "Categorical Feature Extraction"]
channel_names = get_channel_names()
input_types = get_input_types(channel_names.keys())
imaging_modalities = get_imaging_modality(channel_names.keys())
has_flim = "FLIM" in imaging_modalities.values()
decay_input_type = get_decay_input_type()
ch_num_components = get_num_components(input_types, channel_names.keys())
selected_ch_feature_extractors = get_selected_feature_extractors(input_types, channel_names.keys())
fov_name_col = get_fov_name_col()
with col1:
    # first select the step to perform
    selected_step = st.radio(
        "Select a step to perform",
        steps,
        index=0,
        help="Image Metadata Extraction: Extracts metadata from the images. Numeric Feature Extraction: Extracts single cell numeric features from the images. Categorical Feature Extraction: Extracts categorical features from the images. \n ",
    )
    if selected_step == "Image Metadata Extraction":
        checkbox_cols = st.columns(len(channel_names))
        actual_file_suffix = None
        selected_channels = {}
        selected_ch_num_components = {}
        for index, (channel_key, channel_name) in enumerate(channel_names.items()):       
            with checkbox_cols[index]:
                has_channel = st.checkbox(f"has {channel_name}", value=True, key=f"has_channel_{channel_key}")
                if has_channel:
                    # have a help text to show the planned features to be extracted
                    with st.expander(f"Feature extractors for {channel_name}", expanded=False):
                        st.write(selected_ch_feature_extractors[channel_key])
                    selected_channels[channel_key] = channel_name
                    if ch_num_components[channel_key] != 0 and "prefitted" in input_types[channel_key]: # if equals to 0, it means this channel does not have any lifetime fit analysis; only prefitted needs to be specified to get all the files. 
                        selected_ch_num_components[channel_name] = st.number_input(f"No. component", value=ch_num_components[channel_key], min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis" if index == 0 else None, key=f"num_component_{channel_name}")
                    elif ch_num_components[channel_key] != 0: # do not ask now, will ask later when fitting
                        selected_ch_num_components[channel_name] = ch_num_components[channel_key]
        if len(selected_channels) == 0:
            st.error(f"Please check at least one of the channels {sad_emoji}")
        else:
            if has_flim:
                duration, time_bins, laser_rate = lifetime_data_config_widget(selected_ch_feature_extractors, decay_input_type)
            else: # for later, we will add other imaging modalities and this will ask for those imaging modality specific config
                duration, time_bins, laser_rate = None, None, None
            actual_file_suffix, error_msg = load_data_suffix_widget(input_types, selected_channels, selected_ch_num_components, selected_ch_feature_extractors)
            if error_msg != "":
                st.error(error_msg)
            else:
                folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the raw data that is needed for the selected data extraction type." , key="image_metadata_folder_path")
    elif selected_step == "Numeric Feature Extraction":
        metadata_df = None
        if st.session_state["last_extracted_metadata"] is not None:
            metadata_df = st.session_state["last_extracted_metadata"]
            file_path = st.session_state["last_extracted_metadata_filepath"]
            st.info(f"Using the latest extracted metadata file: {file_path}. Refresh the page to use a different file.")
        else: 
            uploaded_file = st.file_uploader("Upload the image metadata csv", type=["csv"], help="The metadata file should be from the image metadata extraction step. ")
            if uploaded_file is not None:
                try:
                    metadata_df = pd.read_csv(uploaded_file) 
                except Exception as e:
                    st.error(f"Error reading the uploaded CSV file: {e}")
                    metadata_df = None # Ensure metadata_df is None if reading fail

        if metadata_df is not None:
            error_msg, metadata_dict = parse_metadata_file(metadata_df, fov_name_col)
            if error_msg == "":
                st.success(f"✅ Features to be extracted confirmed.")
                decay_input_type = metadata_dict["decay_input_type"]
                shift_needed = len(metadata_dict["channels_shift"]) > 0
                # if there are channels to be fitted, show the fitting options: spcimage is already fitted
                if "Lifetime fit" in metadata_dict and len(metadata_dict["Lifetime fit"]) > 0 and "prefitted" not in decay_input_type:
                    st.info("Please specify the following fitting options.")
                    metadata_dict= fit_options_widget(decay_input_type, metadata_dict)
                
                shifts_are_present = all(f"{ch}_shift" in metadata_df.columns for ch in metadata_dict["channels_shift"])
                if shift_needed and not shifts_are_present:
                    if st.button("Start Finding Shifts"):
                        st.session_state["choosing_shift"] = True
                        st.session_state["shift_ready"] = False
                else:
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        if st.button("Confirm and Start Analysis", use_container_width=True):
                            st.session_state["choosing_shift"] = False
                            st.session_state["shift_ready"] = True
                            st.rerun()
                    
                    if shift_needed and shifts_are_present:
                        with col1_2:
                            if st.button("Go back and find shift", use_container_width=True):
                                st.session_state["choosing_shift"] = True
                                st.session_state["shift_ready"] = False
                                # remove shift columns from metadata_df in session state
                                for ch in metadata_dict["channels_shift"]:
                                    if f"{ch}_shift" in metadata_df.columns:
                                        metadata_df = metadata_df.drop(columns=[f"{ch}_shift"])
                                st.session_state["last_extracted_metadata"] = metadata_df
                                st.rerun()
                        # have a download button to download the metadata file
                        if st.session_state["last_extracted_metadata_filepath"] is not None:
                            download = st.button("Download updated metadata", use_container_width=True, help="Download the augmented metadata with the calculated shifts and selected time gates as a CSV file.")
                            if download:
                                metadata_df.to_csv(st.session_state["last_extracted_metadata_filepath"], index=False)
                        else:
                            st.download_button(label="Download updated metadata", data=metadata_df.to_csv(index=False), file_name=f"metadata_{time.strftime('%Y%m%d_%H%M%S')}.csv", key=f"download_metadata_{time.time()}", use_container_width=True, help="Download the augmented metadata with the calculated shifts and selected time gates as a CSV file.")
            else:
                st.error(f"Error: {error_msg}")

    else:   
        # Categorical features extraction
        df_folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the csv files that you want to assign categories to.")
        delimiter = st.text_input("Cell ID Delimiter", "_", max_chars=2, help="The delimiter used to split the cell ID/base_name column.")
        if df_folder_path != "":
            available_dfs = find_available_dfs_widget(df_folder_path, delimiter)
            if len(available_dfs) > 0:
                st.write(f"Found {len(available_dfs)} available csv files ready to be assigned categories {happy_emoji}:")
                st.write(available_dfs)
            else:
                st.error(f"No available csv files found at {df_folder_path} {sad_emoji}")

with col2: 
    # check if the folder exists
    if selected_step == "Image Metadata Extraction" and error_msg == "": 
        if os.path.isdir(folder_path): 
            images = load_list_data_from_folder_widget(folder_path, file_suffix=actual_file_suffix)
            if len(images) != 0:
                st.success(f"Images with ✅ are loaded successfully {happy_emoji}. Images with ❌ (if any) are not loaded. The following features will be extracted: ")
                images_df = pd.DataFrame.from_dict(images, orient="index")

                # Set index name and reset to column (do this once, outside the loop)
                images_df.index.name = fov_name_col  # Set index name 
                images_df.reset_index(inplace=True)  # Reset index to make it a column
                
                # For each channel, add the feature_types and num_components
                for channel_key, channel_name in selected_channels.items():
                    # assign input type to the channel
                    images_df[f"{channel_name}_input_type"] = input_types[channel_key]
                    images_df[f"{channel_name}_imaging_modality"] = imaging_modalities[channel_key]
                    for feature_extractor in selected_ch_feature_extractors[channel_key]:
                        images_df[f"{channel_name}_{feature_extractor}"] = True
                    if has_flim:
                        if channel_name in selected_ch_num_components: 
                            images_df[f"{channel_name}_num_components"] = selected_ch_num_components[channel_name]
                    # ROI Summing Fit and SPCImage takes in raw decay that maybe multiple channels. need to assign data channel to each image channel
                    # K-flow already knows the duration and time bins and do not need to assign channel
                
                error_msg, images_df = check_assign_channel_widget(images_df, selected_channels, flim_decay_input_type=decay_input_type, imaging_modalities=imaging_modalities, duration=duration, time_bins=time_bins)

                if error_msg != "":
                    st.error(f"Error: {error_msg}")        
                else:   
                    if laser_rate is not None:
                        images_df["laser_rate"] = laser_rate
                    display_feature_groups_widget(images_df)
                    export_metadata_widget(metadata_df=images_df, folder_path=folder_path)
            else: 
                st.warning("No data found in the folder. Please check the path and the file suffixes.")
        elif folder_path != "":
            st.error(f"Folder not found! Please check the path. {sad_emoji}")
        else:
            st.info(f"Please provide a folder path.")
    elif selected_step == "Numeric Feature Extraction" and st.session_state["choosing_shift"] and metadata_df is not None:
        channel_shifts = {}
        for channel_name in metadata_dict["channels_shift"]:
            error_msg, shifts = choose_shift_widget(metadata_df, metadata_dict, channel_name=channel_name)
            if error_msg != "":
                st.error(error_msg)
            else:
                channel_shifts[channel_name] = shifts
        shift_finished = st.button("Confirm Shift and Choose Time Gates (if applicable) for each channel")
        if shift_finished:
            # write the shift to the metadata file
            for channel_name in channel_shifts:
                metadata_df[f"{channel_name}_shift"] = channel_shifts[channel_name]
                if "start" in metadata_dict[channel_name]:
                    metadata_df[f"{channel_name}_start"] = metadata_dict[channel_name]["start"]
                if "end" in metadata_dict[channel_name]:
                    metadata_df[f"{channel_name}_end"] = metadata_dict[channel_name]["end"]
 
            # Store the updated metadata_df in session state so it persists across rerun
            st.session_state["last_extracted_metadata"] = metadata_df
            st.session_state["choosing_shift"] = False
            st.session_state["shift_ready"] = False
            st.rerun()
                
    elif selected_step == "Numeric Feature Extraction" and st.session_state["shift_ready"] and metadata_df is not None:
        single_cell_features = fov_extraction_widget(metadata_df, metadata_dict)
        if not single_cell_features.empty:
            st.success(f"Image features with ✅ are extracted successfully {happy_emoji}! Images with ❌ (if any) are excluded. The first few rows of the features are shown below.")
            st.write(single_cell_features.head())
            # get the current timestamp 
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # get the folder path from the file path
            if st.session_state["last_extracted_metadata_filepath"] is not None:
                folder_path = os.path.dirname(st.session_state["last_extracted_metadata_filepath"])
                csv_path = os.path.join(folder_path, f"single_cell_features_{timestamp}.csv")
    
            # save the features to a csv file
                confirm_export = st.button("Download single cell features as CSV")
                if confirm_export:
                    try:
                        single_cell_features.to_csv(csv_path) # Save the DataFrame
                    except Exception as e:
                        st.error(f"Error exporting the image metadata: {e}. Is the previous metadata file open in another program?")
                    st.success(f"Image metadata exported successfully to {csv_path} {happy_emoji}")
            else:
                st.download_button(label="Download single cell features as CSV", data=single_cell_features.to_csv(), file_name= f"single_cell_features_{timestamp}.csv")
    elif selected_step == "Categorical Feature Extraction" and df_folder_path != "" and len(available_dfs) > 0:
        combined_df, available_categories = check_and_merge_df_widget(available_dfs)
        map_categories_to_labels_widget(available_categories, combined_df, delimiter, df_folder_path)