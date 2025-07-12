import streamlit as st
import os
import pandas as pd
import time
from src.navigation import render_top_menu
from src.dataframe_io import happy_emoji, sad_emoji
from src.widgets.numeric_extraction_widgets import image_extraction_widget
from src.widgets.metadata_widgets import load_list_data_from_folder_widget, load_data_suffix_widget, export_metadata_widget, display_feature_groups_widget, check_assign_channel_widget, lifetime_data_config_widget
from src.widgets.category_widgets import map_categories_to_labels_widget, find_available_dfs_widget, check_and_merge_df_widget
from src.widgets.lifetime_widgets import fit_options_widget, choose_shift_widget
from src.metadata import parse_metadata_file
from src.config import get_available_input_types, get_channel_names, get_num_components, get_feature_extractors, get_image_name_col

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
ch_num_components = get_num_components(channel_names.values())
ch_feature_extractors = get_feature_extractors(channel_names.values())
image_name_col = get_image_name_col()
with col1:
    # first select the step to perform
    selected_step = st.selectbox("Select a step to perform", steps, index=0, help="Image Metadata Extraction: Extracts metadata from the images. Numeric Feature Extraction: \
    Extracts numeric features from the images. Categorical Feature Extraction: Extracts categorical features from the images. \n ")
    # select input type
    if selected_step == "Image Metadata Extraction":
        input_types, preferred_input_type_index =  get_available_input_types()
        selected_input_type = st.selectbox("Select input type", input_types, index=preferred_input_type_index, help="ROI Summing Fit: Performs \
        ROI summing on raw lifetime decay file, and fit the summed decay curve for each cell. SPCImage: extracts single cell fitting data from outputs of SPCImage. K-Flow: \
        Fit K-Flow decay curves for each cell. Categorical Features: augment categorical columns to your existing data file")
        checkbox_cols = st.columns(len(channel_names))
        actual_file_suffix = None
        selected_channels = []
        selected_ch_num_components = {}
        for index, channel_name in enumerate(channel_names.values()):        
            with checkbox_cols[index]:
                has_channel = st.checkbox(f"has {channel_name}", value=True)
                if has_channel:
                    selected_channels.append(channel_name)
                    if ch_num_components[channel_name] != 0: # if equals to 0, it means this channel does not have any lifetime fit/fit free analysis
                        selected_ch_num_components[channel_name] = st.number_input(f"No. component", value=ch_num_components[channel_name], min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis" if index == 0 else None, key=f"num_component_{channel_name}")
        if len(selected_channels) == 0:
            st.error(f"Please check at least one of the channels {sad_emoji}")
        else:
            duration, time_bins, laser_rate = lifetime_data_config_widget(ch_feature_extractors, selected_input_type)
            actual_file_suffix, error_msg = load_data_suffix_widget(selected_input_type, selected_channels, selected_ch_num_components)
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
            error_msg, metadata_dict = parse_metadata_file(metadata_df, image_name_col)
            if error_msg == "":
                st.success(f"✅ Features to be extracted confirmed.")
                input_type = metadata_dict["input_type"]

                shift_needed = len(metadata_dict["channels_shift"]) > 0
                # if there are channels to be fitted, show the fitting options
                if len(metadata_dict["channels_fit"]) > 0:
                    st.info("Please specify the following fitting options.")
                    metadata_dict= fit_options_widget(input_type, metadata_dict)
                st.write(metadata_dict) 
                if shift_needed:
                    if st.button("Start Finding Shifts"):
                        st.session_state["choosing_shift"] = True
                        st.session_state["shift_ready"] = False
                else:
                    if st.button("Confirm and Start Analysis"):
                        st.session_state["choosing_shift"] = False
                        st.session_state["shift_ready"] = True
                        st.rerun()
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
                # augment metadata
                images_df['input_type'] = selected_input_type
                # For each channel, add the feature_types and num_components
                for channel_name in selected_channels:
                    if ch_num_components[channel_name] != 0: # if equals to 0, it means this channel does not care about lifetime analysis at all
                        images_df[f"{channel_name}_num_components"] = ch_num_components[channel_name]
                    for feature_extractor in ch_feature_extractors[channel_name]:
                        for module in ch_feature_extractors[channel_name][feature_extractor]:
                            images_df[f"{channel_name}_{feature_extractor}_{module}"] = True
        
                images_df.index.name = image_name_col  # Set index name 
                images_df.reset_index(inplace=True)  # Reset index to make it a column
                if selected_input_type == "K-Flow":
                    # copy the image_name column to kflow_exp_name
                    images_df["kflow_exp_name"] = images_df[image_name_col]
                
                # ROI Summing Fit and SPCImage takes in raw decay that maybe multiple channels. need to assign data channel to each image channel
                # K-flow already knows the duration and time bins and do not need to assign channel
                error_msg, images_df = check_assign_channel_widget(images_df, selected_channels, input_type=selected_input_type, duration=duration, time_bins=time_bins)
                
                if error_msg != "":
                    st.error(f"Error: {error_msg}")
                else:   
                    if laser_rate is not None:
                        images_df["laser_rate"] = laser_rate
                    display_feature_groups_widget(images_df)
                    export_metadata_widget(images_df=images_df, folder_path=folder_path)
            else: 
                st.warning("No data found in the folder. Please check the path and the file suffixes.")
        elif folder_path != "":
            st.error(f"Folder not found! Please check the path. {sad_emoji}")
        else:
            st.info(f"Please provide a folder path.")
    elif selected_step == "Numeric Feature Extraction" and st.session_state["choosing_shift"] and metadata_df is not None:
        channel_shifts = {}
        for channel_name in metadata_dict["channels_shift"]:
            error_msg, shifts = choose_shift_widget(metadata_df, metadata_dict, channel=channel_name)
            if error_msg != "":
                st.error(f"Error: {error_msg}")
            else:
                channel_shifts[channel_name] = shifts
        shift_finished = st.button("Confirm the Shift and Start the Analysis")
        if shift_finished:
            # write the shift to the metadata file
            for channel_name in channel_shifts:
                metadata_df[f"{channel_name}_shift"] = channel_shifts[channel_name]        
            # Store the updated metadata_df in session state so it persists across rerun
            st.session_state["last_extracted_metadata"] = metadata_df
            st.session_state["choosing_shift"] = False
            st.session_state["shift_ready"] = True
            st.rerun()
                
    elif selected_step == "Numeric Feature Extraction" and st.session_state["shift_ready"] and metadata_df is not None:


        single_cell_features = image_extraction_widget(metadata_df, metadata_dict)
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