import streamlit as st
import os
import pandas as pd
from navigation import render_top_menu
from widgets.data_widgets import happy_emoji, sad_emoji, load_list_data_from_folder_widget, load_data_suffix_widget, export_data_widget, parse_metadata_display_feature_widget
from widgets.fit_widgets import fit_options
from file_util import parse_metadata_file
from fit import choose_shift
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

if "last_extracted_metadata" not in st.session_state:
    st.session_state["last_extracted_metadata"] = None
if "last_extracted_metadata_filepath" not in st.session_state:
    st.session_state["last_extracted_metadata_filepath"] = None


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
        analysis_ready = False
        metadata_df = None
        if st.session_state["last_extracted_metadata"] is not None:
            metadata_df = st.session_state["last_extracted_metadata"]
            file_path = st.session_state["last_extracted_metadata_filepath"]
            st.info(f"Using the latest extracted metadata file: {file_path}. Refresh the page to use a different file.")
        else: 
            uploaded_metadata = st.file_uploader("Upload the image metadata csv", type=["csv"], help="The metadata file should be from the image metadata extraction step. ")
            if uploaded_metadata is not None:
                try:
                    metadata_df = pd.read_csv(uploaded_metadata) 
                except Exception as e:
                    st.error(f"Error reading the uploaded CSV file: {e}")
                    metadata_df = None # Ensure metadata_df is None if reading fail

        if metadata_df is not None:
            error_msg, selected_feature_groups_features, analysis_type, fit_free = parse_metadata_file(metadata_df)

            if error_msg == "":
                st.success(f"✅ Features to be extracted confirmed. Analysis type: {analysis_type}. Fit free: {fit_free}.") 
                                
                if analysis_type == "ROI Summing Fit" or analysis_type == "K-flow":
                    st.info("Please specify the following fitting options.")
                 
                    duration, time_bins, num_components, fitting_algo, fix_shift = fit_options(analysis_type)
                    if st.button("Confirm and Start Fitting"):
                        analysis_ready = True
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
                export_data_widget(images_df=images_df, folder_path=folder_path)
            else: 
                st.warning("No data found in the folder. Please check the path and the file suffixes.")
        elif folder_path != "":
            st.error(f"Folder not found! Please check the path. {sad_emoji}")
    elif selected_step == "Numeric Feature Extraction" and analysis_ready:
        st.info(f"Applying {analysis_type} on {len(metadata_df)} images.")
        st.info("Preproceessing step: choose the shift for all images.")
        shifts = choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, analysis_type)
            
