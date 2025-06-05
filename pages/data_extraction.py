import streamlit as st
import os
import pandas as pd
import numpy as np
import time
from src.navigation import render_top_menu
from src.widgets.data_widgets import happy_emoji, sad_emoji, image_extraction_widget
from src.widgets.metadata_widgets import load_list_data_from_folder_widget, load_data_suffix_widget, export_metadata_widget, parse_metadata_display_feature_widget
from src.widgets.fit_widgets import fit_options_widget, choose_shift_widget, start_end_widget
from src.metadata import parse_metadata_file

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

if "last_extracted_metadata" not in st.session_state:
    st.session_state["last_extracted_metadata"] = None
if "last_extracted_metadata_filepath" not in st.session_state:
    st.session_state["last_extracted_metadata_filepath"] = None
if "last_analysis_type" not in st.session_state:
    st.session_state["last_analysis_type"] = None
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
            andalysis_type = st.session_state["last_analysis_type"]
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
            fitting = True if analysis_type == "ROI Summing Fit" or analysis_type == "K-flow" or (analysis_type == "SPCImage" and fit_free) else False
            if error_msg == "":
                st.success(f"✅ Features to be extracted confirmed. Analysis type: {analysis_type}. Fit free: {fit_free}. Channels: NADH: {has_nadh}, FAD/red: {has_fad}.") 
                                
                if fitting:
                    st.info("Please specify the following fitting options.")
                 
                    duration, time_bins, num_components, fitting_algo, fitting_mode, fix_shift, laser_rate = fit_options_widget(analysis_type, fit_free)
                    # based pm the time_bins, add the start and end for NADH and FAD widget 
                    if has_nadh:
                        nadh_start, nadh_end = start_end_widget(time_bins, "NADH")
                    if has_fad:
                        fad_start, fad_end = start_end_widget(time_bins, "FAD")
                        
                    if st.button("Confirm and Start Fitting"):
                        st.session_state["choosing_shift"] = True
                        st.session_state["shift_ready"] = False
                
                if analysis_type == "SPCImage": # spc image and fit free needs fitting but does not need to choose shift
                    st.session_state["choosing_shift"] = False
                    st.session_state["shift_ready"] = True

            else:
                st.error(f"Error: {error_msg}")

    else:   
        # Categorical features extraction
        pass

with col2: 
    # check if the folder exists
    if selected_step == "Image Metadata Extraction" and suffix_correct: 
        if os.path.isdir(folder_path): 
            images = load_list_data_from_folder_widget(folder_path, file_suffix=actual_file_suffix)
            if len(images) != 0:
                st.success(f"Images with ✅ are loaded successfully {happy_emoji}. Images with ❌ (if any) are not loaded. The following features will be extracted: ")
                images_df = pd.DataFrame.from_dict(images, orient="index")
                images_df['fit_free'] = fit_free
                images_df.index.name = "image_name"  # Set index name 
                images_df.reset_index(inplace=True)  # Reset index to make it a column
                parse_metadata_display_feature_widget(images_df)
                export_metadata_widget(images_df=images_df, folder_path=folder_path)
            else: 
                st.warning("No data found in the folder. Please check the path and the file suffixes.")
        elif folder_path != "":
            st.error(f"Folder not found! Please check the path. {sad_emoji}")
    elif selected_step == "Numeric Feature Extraction" and st.session_state["choosing_shift"]:
        st.info(f"Applying {analysis_type} on {len(metadata_df)} images.")
        # first NADH, then FAD/red
        if has_nadh: 
            #st.info("Preproceessing step: choose the shift for all images on channel NADH.")
            error_msg, nadh_shifts = choose_shift_widget(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, analysis_type, channel="NADH")
            if error_msg != "":
                st.error(f"Error: {error_msg}")
        if has_fad:
            #st.info("Preproceessing step: choose the shift for all images on channel FAD/red.")
            error_msg, fad_shifts = choose_shift_widget(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, analysis_type, channel="FAD")
            if error_msg != "":
                st.error(f"Error: {error_msg}")
        
        if error_msg == "":
            if fix_shift:
                # let user choose the shift
                col1, col2 = st.columns(2)
                with col1:
                    if has_nadh:
                        nadh_shift = st.number_input("NADH Shift", value=np.median(nadh_shifts), step=0.1, help="The shift for NADH channel. The provided default value is the median of the shifts. You can change it to a specific value.")
                with col2:
                    if has_fad:
                        fad_shift = st.number_input("FAD Shift", value=np.median(fad_shifts), step=0.1, help="The shift for FAD/red channel. The provided default value is the median of the shifts. You can change it to a specific value.")

            shift_finished = st.button("Confirm the Shift and Start the Analysis")
            if shift_finished:
                # write the shift to the metadata file
                if has_nadh:
                    if fix_shift:
                        metadata_df["nadh_shift"] = nadh_shift
                    else:
                        metadata_df["nadh_shift"] = nadh_shifts
                if has_fad:
                    if fix_shift:
                        metadata_df["fad_shift"] = fad_shift
                    else:
                        metadata_df["fad_shift"] = fad_shifts
                
                # Store the updated metadata_df in session state so it persists across rerun
                st.session_state["last_extracted_metadata"] = metadata_df
                
                st.session_state["choosing_shift"] = False
                st.session_state["shift_ready"] = True
                st.rerun()
                
    elif selected_step == "Numeric Feature Extraction" and st.session_state["shift_ready"]:
        if fitting:
         # adding the fitting config to the metadata
            metadata_df["fitting_algo"] = fitting_algo
            metadata_df["fitting_mode"] = fitting_mode
            metadata_df["duration"] = duration
            metadata_df["time_bins"] = time_bins
            metadata_df["num_components"] = num_components
            if fit_free:
                metadata_df["laser_rate"] = laser_rate
            if has_nadh:
                metadata_df["nadh_start"] = nadh_start
                metadata_df["nadh_end"] = nadh_end
            if has_fad:
                metadata_df["fad_start"] = fad_start
                metadata_df["fad_end"] = fad_end
        if analysis_type == "ROI Summing Fit" or analysis_type == "SPCImage":
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
