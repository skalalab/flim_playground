import streamlit as st
import os
from navigation import render_top_menu
from feature_groups import feature_groups_features, get_feature_name
from widgets.load_data_widgets import happy_emoji, sad_emoji, load_data_from_folder_widget
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()
st.title("Data Extraction")

col1, col2 = st.columns([0.4, 1])
numeric_feature_extraction_types =  ["ROI Summing Fit", "SPCImage (former Regionprops)", "K-Flow"]
with col1:
    # select data extraction type
    extraction_types = numeric_feature_extraction_types + ["Categorical Features"]
    
    # Full width for the extraction type selectbox
    selected_extraction_type = st.selectbox("Select extraction type", extraction_types, index=0, help="ROI Summing Fit: Performs \
    ROI summing on raw lifetime decay file, and fit the summed decay curve for each cell. SPCImage: extracts single cell fitting data from outputs of SPCImage. K-Flow: \
    Fit K-Flow decay curves for each cell. Categorical Features: augment categorical columns to your existing data file")
    
    # Create two columns for the checkboxes (2x2 grid)
    if selected_extraction_type in numeric_feature_extraction_types:
        checkbox_col1, checkbox_col2 = st.columns(2)
        
        with checkbox_col1:
            if selected_extraction_type == "ROI Summing Fit" or selected_extraction_type == "K-Flow":
                fix_shift = st.checkbox("Fix Shift", value=True, help="If checked, the shift will be inferred one time and fixed for all the images/cells in the folder (i.e. they were imaged suring the same session). \
                If unchecked, the shift will be inferred for each image separately.")
            fit_free = st.checkbox("Fit Free Analysis", value=True, help="If checked, Fit free (e.g. Phasor) features will be extracted.")
        
        with checkbox_col2:
            has_nadh = st.checkbox("Has NAD(P)H Data", value=True)
            has_fad =  st.checkbox("Has FAD Data", value=True)    
            # Add an empty placeholder in the second column to maintain grid symmetry if needed
            # You can add another checkbox here in the future if required

    folder_ready = False
    folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the raw data that is needed for the selected data extraction type. " \
    , key="folder_path")
    # check if the folder exists
    if os.path.isdir(folder_path):
        load_data_from_folder_widget(folder_path, selected_extraction_type, fit_free, has_nadh, has_fad)
    elif folder_path != "":
        st.warning(f"Folder not found! Please check the path. {sad_emoji}")