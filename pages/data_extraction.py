import streamlit as st
import os
import pandas as pd
import time
from src.navigation import render_top_menu
from src.widgets.data_widgets import happy_emoji, sad_emoji, image_extraction_widget
from src.widgets.metadata_widgets import load_list_data_from_folder_widget, load_data_suffix_widget, export_metadata_widget, parse_metadata_display_feature_widget, check_sdt_channel_widget
from src.widgets.category_widgets import map_categories_to_labels_widget, find_available_dfs_widget, check_and_merge_df_widget
from src.widgets.fit_widgets import fit_options_widget, choose_shift_widget, start_end_widget
from src.metadata import parse_metadata_file

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
analysis_types =  ["ROI Summing Fit", "SPCImage", "K-Flow"]
with col1:
    # first select the step to perform
    selected_step = st.selectbox("Select a step to perform", steps, index=0, help="Image Metadata Extraction: Extracts metadata from the images. Numeric Feature Extraction: \
    Extracts numeric features from the images. Categorical Feature Extraction: Extracts categorical features from the images. \n ")
    # select analysis type
    if selected_step == "Image Metadata Extraction":
        selected_analysis_type = st.selectbox("Select analysis type", analysis_types, index=0, help="ROI Summing Fit: Performs \
        ROI summing on raw lifetime decay file, and fit the summed decay curve for each cell. SPCImage: extracts single cell fitting data from outputs of SPCImage. K-Flow: \
        Fit K-Flow decay curves for each cell. Categorical Features: augment categorical columns to your existing data file")
        checkbox_col1, checkbox_col2, checkbox_col3 = st.columns(3)
        suffix_correct = False
    
        with checkbox_col1:
            has_nadh = st.checkbox("Has NAD(P)H Data", value=True)
        with checkbox_col2:
            has_fad =  st.checkbox("Has FAD Data", value=True)   
        with checkbox_col3:
            fit_free = st.checkbox("Fit Free Analysis", value=True, help="If checked, Fit free (e.g. Phasor) features will be extracted.")
        if has_nadh or has_fad:
            actual_file_suffix, error_msg = load_data_suffix_widget(selected_analysis_type, fit_free, has_nadh, has_fad)
            if error_msg != "":
                st.error(error_msg)
            else:
                suffix_correct = True
                folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the raw data that is needed for the selected data extraction type. " \
                , key="folder_path")
        else: st.error(f"Please check at least one of the channels {sad_emoji}")
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
            error_msg, selected_feature_groups_features, analysis_type, fit_free, has_nadh, has_fad = parse_metadata_file(metadata_df)
            fitting = True if (analysis_type == "ROI Summing Fit" or analysis_type == "K-Flow" or (analysis_type == "SPCImage" and fit_free)) else False
            if error_msg == "":
                st.success(f"✅ Features to be extracted confirmed. Analysis type: {analysis_type}. Fit free: {fit_free}. Channels: NADH: {has_nadh}, FAD/red: {has_fad}.") 
                if fitting:
                    st.info("Please specify the following fitting options.")
                    default_laser_rate = 0.05 if analysis_type == "K-Flow" else 0.08
                    duration, time_bins, num_components, fitting_algo, fitting_mode, fix_shift, laser_rate = fit_options_widget(analysis_type, fit_free, default_laser_rate=default_laser_rate)
                    # based pm the time_bins, add the start and end for NADH and FAD widget 
                    if analysis_type != "K-Flow":
                        time_bins = metadata_df["time_bins"].iloc[0]
                        duration = metadata_df["duration"].iloc[0]
                    if has_nadh:
                        nadh_start, nadh_end = start_end_widget(time_bins, "NADH")
                    if has_fad:
                        fad_start, fad_end = start_end_widget(time_bins, "FAD")
                
                    if st.button("Start Finding Shifts"):
                        st.session_state["choosing_shift"] = True
                        st.session_state["shift_ready"] = False
                            

                if analysis_type == "SPCImage" and not fit_free:
                    if st.button("Confirm and Start Analysis"):
                        st.session_state["choosing_shift"] = False
                        st.session_state["shift_ready"] = True

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
    if selected_step == "Image Metadata Extraction" and suffix_correct: 
        if os.path.isdir(folder_path): 
            images = load_list_data_from_folder_widget(folder_path, file_suffix=actual_file_suffix)
            if len(images) != 0:
                st.success(f"Images with ✅ are loaded successfully {happy_emoji}. Images with ❌ (if any) are not loaded. The following features will be extracted: ")
                images_df = pd.DataFrame.from_dict(images, orient="index")
                images_df['fit_free'] = fit_free
                images_df['analysis_type'] = selected_analysis_type
                images_df.index.name = "image_name"  # Set index name 
                images_df.reset_index(inplace=True)  # Reset index to make it a column
                if selected_analysis_type == "K-Flow":
                    # copy the image_name column to kflow_exp_name
                    images_df["kflow_exp_name"] = images_df["image_name"]
              
                # before exporting, check for the sdt channel (dimension)
                error_msg, images_df = check_sdt_channel_widget(images_df)
                if error_msg != "":
                    st.error(f"Error: {error_msg}")
                else:   
                    parse_metadata_display_feature_widget(images_df)
                    export_metadata_widget(images_df=images_df, folder_path=folder_path)
            else: 
                st.warning("No data found in the folder. Please check the path and the file suffixes.")
        elif folder_path != "":
            st.error(f"Folder not found! Please check the path. {sad_emoji}")
    elif selected_step == "Numeric Feature Extraction" and st.session_state["choosing_shift"] and metadata_df is not None:
        st.info(f"Fitting for shift on {len(metadata_df)} image(s).")
        # first NADH, then FAD/red
        if has_nadh: 
            #st.info("Preproceessing step: choose the shift for all images on channel NADH.")
            error_msg, nadh_shifts = choose_shift_widget(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, analysis_type, fix_shift, channel="NADH")
            if error_msg != "":
                st.error(f"Error: {error_msg}")
        if has_fad:
            #st.info("Preproceessing step: choose the shift for all images on channel FAD/red.")
            error_msg, fad_shifts = choose_shift_widget(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, analysis_type, fix_shift, channel="FAD")
            if error_msg != "":
                st.error(f"Error: {error_msg}")
        
        shift_finished = st.button("Confirm the Shift and Start the Analysis")
        if shift_finished:
            # write the shift to the metadata file
            if has_nadh:
                metadata_df["nadh_shift"] = nadh_shifts
                
            if has_fad:
                metadata_df["fad_shift"] = fad_shifts
            
            # Store the updated metadata_df in session state so it persists across rerun
            st.session_state["last_extracted_metadata"] = metadata_df
            st.session_state["choosing_shift"] = False
            st.session_state["shift_ready"] = True
            st.rerun()
                
    elif selected_step == "Numeric Feature Extraction" and st.session_state["shift_ready"] and metadata_df is not None:
        if fitting:
         # adding the fitting config to the metadata
            metadata_df["fitting_algo"] = fitting_algo
            metadata_df["fitting_mode"] = fitting_mode
            if analysis_type == "K-Flow": # other types get the duration and time bins from the sdt file automatically
                metadata_df["duration"] = duration
                metadata_df["time_bins"] = time_bins
            metadata_df["num_components"] = num_components
            metadata_df["laser_rate"] = laser_rate

            if has_nadh:
                metadata_df["nadh_start"] = nadh_start
                metadata_df["nadh_end"] = nadh_end
            if has_fad:
                metadata_df["fad_start"] = fad_start
                metadata_df["fad_end"] = fad_end

        single_cell_features = image_extraction_widget(metadata_df, analysis_type, fit_free, has_nadh, has_fad)
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