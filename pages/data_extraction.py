import streamlit as st
import os
import pandas as pd
import time
from src.navigation import render_top_menu
from src.dataset_io import happy_emoji, sad_emoji
from src.widgets.numeric_extraction_widgets import fov_extraction_widget
from src.widgets.metadata_widgets import load_list_data_from_folder_widget, load_data_suffix_widget, export_metadata_widget, preview_metadata_widget, check_assign_channel_widget, lifetime_data_config_widget
from src.widgets.category_widgets import map_categories_to_labels_widget, find_available_dfs_widget, check_and_merge_df_widget
from src.widgets.lifetime_widgets import fit_options_widget, choose_shift_widget
from src.metadata import parse_metadata_file
from src.config import get_imaging_modality, get_input_types, get_channel_names, get_num_components, get_selected_feature_extractors, get_fov_name_col, get_decay_input_type, get_fit_free_calibration_method
from src.file_io import find_file_in_folder, load_image

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
steps = ["FOV Metadata Extraction", "Numeric Feature Extraction (fitting, phasor, etc.)", "Categorical Feature Extraction (e.g. treatment)"]
channel_names = get_channel_names()
input_types = get_input_types(channel_names.keys())
imaging_modalities = get_imaging_modality(channel_names.keys())
has_flim = "FLIM" in imaging_modalities.values()
decay_input_type = get_decay_input_type()
ch_num_components = get_num_components(input_types, channel_names.keys())
selected_ch_feature_extractors = get_selected_feature_extractors(input_types, channel_names.keys())
fov_name_col = get_fov_name_col()
fit_free_calibration_method, reference_dye_file, reference_dye_lifetime = get_fit_free_calibration_method(decay_input_type)
with col1:
    # first select the step to perform
    selected_step = st.radio(
        "Select a step to perform",
        steps,
        index=0,
        help="FOV Metadata Extraction: Extracts metadata from the field of views. Numeric Feature Extraction: Extracts single cell numeric features from the FOVs. Categorical Feature Extraction: Extracts categorical features from the FOVs. \n ",
    )
    if "FOV Metadata Extraction" in selected_step:
         # show decay input type
        if has_flim:
            st.write(f"Decay input type: {decay_input_type}")
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
            # laser rate is none means there is no fit free analysis
            if laser_rate is not None and fit_free_calibration_method == "Reference Dye":
                cols = st.columns(2)
                with cols[0]:
                    reference_dye_file = st.text_input("Reference dye file name", value=reference_dye_file, key="reference_dye_file")
                with cols[1]:
                    reference_dye_lifetime = st.number_input("Reference dye lifetime in ns", value=reference_dye_lifetime, min_value=0.1, max_value=20.0, step=0.1, key="reference_dye_lifetime")
                if reference_dye_file == "":
                    st.error(f"Please enter a valid reference dye file name.")
                if reference_dye_lifetime == "":
                    st.error(f"Please enter a valid reference dye lifetime.")
            actual_file_suffix, error_msg = load_data_suffix_widget(input_types, selected_channels, selected_ch_num_components, selected_ch_feature_extractors)
            if error_msg != "":
                st.error(error_msg)
            else:
                folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the raw data that is needed for the selected data extraction type." , key="fov_metadata_folder_path")
    elif "Numeric Feature Extraction" in selected_step:
        metadata_df = None
        if st.session_state["last_extracted_metadata_filepath"] is not None:
            file_path = st.session_state["last_extracted_metadata_filepath"]
            st.info(f"Using the latest extracted metadata file: {file_path}. Refresh the page to use a different file.")
        if st.session_state["last_extracted_metadata"] is not None:
            metadata_df = st.session_state["last_extracted_metadata"]
        else: 
            uploaded_file = st.file_uploader("Upload the field of view metadata csv", type=["csv"], help="The metadata file should be from the FOV metadata extraction step.")
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
                shifts_are_present = all(f"{ch}_shift" in metadata_df.columns for ch in metadata_dict["channels_shift"])
                if shift_needed and not shifts_are_present:
                        # if there are channels to be fitted, show the fitting options: spcimage is already fitted
                    if "Lifetime fit" in metadata_dict and len(metadata_dict["Lifetime fit"]) > 0 and "prefitted" not in decay_input_type:
                        st.info("Please specify the following fitting options.")
                        metadata_dict= fit_options_widget(metadata_dict)
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        metadata_dict["fix_shift"] = st.checkbox(
                            "Fix the Shift", 
                            value=True, 
                            key="fix_shift_checkbox",
                            help="If True, the shift will be fixed for all images. If False, the shift will be estimated for each image."
                        )
                    with col1_2:
                        if st.button("Start Finding Shifts"):
                            st.session_state["choosing_shift"] = True
                            st.session_state["shift_ready"] = False
                            st.rerun()
                else:
                    if "fitting_mode" in metadata_df.columns: 
                        metadata_df["fitting_mode"] = st.selectbox(
                            "Fitting Mode", 
                            ["Hybrid", "Global", "Local"], 
                            index=0, 
                            key="fitting_mode_update",
                            help="Hybrid: use global fit to get a good initial guess, then use local fit to refine the fit. Global: use global fit to get the best fit. Local: use local fit to get the best fit."
                        )
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        if st.button("Confirm and Start Analysis", use_container_width=True):
                            # Update metadata_df in session state
                            st.session_state["last_extracted_metadata"] = metadata_df
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
                                try:
                                    metadata_df.to_csv(st.session_state["last_extracted_metadata_filepath"], index=False)
                                    st.success(f"✅ Metadata updated successfully at {st.session_state['last_extracted_metadata_filepath']}")
                                except PermissionError:
                                    st.error(f"❌ Cannot save file - it may be open in another program (like Excel). Please close the file and try again.")
                                except Exception as e:
                                    st.error(f"❌ Error saving file: {str(e)}")
                        else:
                            st.download_button(label="Download updated metadata", data=metadata_df.to_csv(index=False), file_name=f"fov_metadata_{time.strftime('%Y%m%d_%H%M%S')}.csv", key=f"download_metadata_{time.time()}", use_container_width=True, help="Download the augmented metadata with the calculated shifts and selected time gates as a CSV file.")
            else:
                st.error(f"Error: {error_msg}")

    else:   
        # Categorical features extraction
        df_folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the csv files that you want to assign categories to.")
        delimiter = st.text_input("Field of View Name Delimiter", "_", max_chars=2, help="The delimiter used to split the fov_name column.")
        if df_folder_path != "":
            available_dfs = find_available_dfs_widget(df_folder_path, delimiter)
            if len(available_dfs) > 0:
                st.write(f"Found {len(available_dfs)} available csv files ready to be assigned categories {happy_emoji}:")
                st.write(available_dfs)
            else:
                st.error(f"No available csv files found at {df_folder_path} {sad_emoji}")

def validate_folder_path(folder_path):
    """Validate folder path and return appropriate error message"""
    if folder_path == "":
        st.info("Please provide a folder path.")
        return False
    if not os.path.isdir(folder_path):
        st.error(f"Folder not found! Please check the path. {sad_emoji}")
        return False
    return True

def load_and_validate_fovs(folder_path, actual_file_suffix):
    """Load FOVs from folder and validate"""
    fovs = load_list_data_from_folder_widget(folder_path, file_suffix=actual_file_suffix)
    if len(fovs) == 0:
        st.warning("No data found in the folder. Please check the path and the file suffixes.")
        return None
    
    st.success(f"Field of Views with ✅ are loaded successfully {happy_emoji}. FOVs with ❌ (if any) will **not** be recorded. Here is the preview of the FOVs and metadata recorded:")
    return fovs

def prepare_fov_dataframe(fovs, selected_channels, selected_ch_num_components):
    """Prepare FOV dataframe with channel information"""
    fov_df = pd.DataFrame.from_dict(fovs, orient="index")
    
    # Set index name and reset to column
    fov_df.index.name = fov_name_col
    fov_df.reset_index(inplace=True)
    
    # Add channel information
    for channel_key, channel_name in selected_channels.items():
        fov_df[f"{channel_name}_input_type"] = input_types[channel_key]
        fov_df[f"{channel_name}_imaging_modality"] = imaging_modalities[channel_key]
        for feature_extractor in selected_ch_feature_extractors[channel_key]:
            fov_df[f"{channel_name}_{feature_extractor}"] = True
        if has_flim and channel_name in selected_ch_num_components:
            fov_df[f"{channel_name}_num_components"] = selected_ch_num_components[channel_name]
    
    return fov_df

def validate_reference_dye(folder_path, fit_free_calibration_method, reference_dye_file, fov_df, time_bins, reference_dye_lifetime):
    """Validate and add reference dye file if needed"""
    fov_df["fit_free_calibration_method"] = fit_free_calibration_method
    if fit_free_calibration_method != "Reference Dye":
        return "", fov_df
    
    error_msg, reference_dye_file_path = find_file_in_folder(folder_path, reference_dye_file)
    if error_msg != "":
        return error_msg, fov_df
    
    # Check dimensions of reference dye file
    try:
        # Try to load the reference dye file to check dimensions
        ref_dye_data = load_image(reference_dye_file_path)
        ref_dye_shape = ref_dye_data.shape
        
        # Check if it's 3D
        if len(ref_dye_shape) != 3:
            return f"Reference dye file must be 3-dimensional, but got {len(ref_dye_shape)} dimensions with shape {ref_dye_shape}", fov_df
        
        # Check if any dimension matches time_bins
        matched_time_bins = ref_dye_shape.count(time_bins)
        if matched_time_bins == 0:
            return f"Cannot find the time axis ({time_bins} time bins) in the reference dye file dimensions: {ref_dye_shape}", fov_df
        # if there are more than one matched, return an error
        elif matched_time_bins > 1:
            return f"Ambiguous time axis based on the reference dye file dimension: {ref_dye_shape}", fov_df
        else:
            fov_df["reference_dye_time_axis"] = ref_dye_shape.index(time_bins)
            
    except Exception as e:
        return f"Error reading reference dye file for validation: {str(e)}", fov_df
    
    fov_df["reference_dye_file"] = reference_dye_file_path
    fov_df["reference_dye_lifetime"] = reference_dye_lifetime

    return "", fov_df

def finalize_fov_processing(error_msg, fov_df, selected_channels, decay_input_type, imaging_modalities, duration, time_bins, folder_path, fit_free_calibration_method=None, reference_dye_file=None, reference_dye_lifetime=None):
    """Final processing steps for FOV data"""
    if error_msg != "":
        st.error(error_msg)
        return
    
    # Check and assign channels
    error_msg, fov_df = check_assign_channel_widget(
        fov_df, selected_channels, 
        flim_decay_input_type=decay_input_type, 
        imaging_modalities=imaging_modalities, 
        duration=duration, time_bins=time_bins
    )
    
    if error_msg != "":
        st.error(error_msg)
        return
    
    # Validate reference dye after channel assignment
    if fit_free_calibration_method is not None:
        time_bins = fov_df["time_bins"].iloc[0]
        error_msg, fov_df = validate_reference_dye(folder_path, fit_free_calibration_method, reference_dye_file, fov_df, time_bins, reference_dye_lifetime)
        if error_msg != "":
            st.error(error_msg)
            return
    
    # Display and export
    preview_metadata_widget(fov_df)
    export_metadata_widget(metadata_df=fov_df, folder_path=folder_path)

with col2: 
    # FOV Metadata Extraction workflow
    if "FOV Metadata Extraction" in selected_step and error_msg == "":
        # Step 1: Validate folder path
        if not validate_folder_path(folder_path):
            pass  # Error already displayed in function
        else:
            # Step 2: Load and validate FOVs
            fovs = load_and_validate_fovs(folder_path, actual_file_suffix)
            if fovs is None:
                pass  # Error already displayed in function
            else:
                # Step 3: Prepare dataframe
                fov_df = prepare_fov_dataframe(fovs, selected_channels, selected_ch_num_components)
                if laser_rate is not None:
                    fov_df["laser_rate"] = laser_rate
                
                # Step 4: Finalize processing (reference dye validation moved here)
                finalize_fov_processing("", fov_df, selected_channels, decay_input_type, imaging_modalities, duration, time_bins, folder_path, fit_free_calibration_method, reference_dye_file, reference_dye_lifetime)
    elif "Numeric Feature Extraction" in selected_step and st.session_state["choosing_shift"] and metadata_df is not None:
        channel_shifts = {}
        for channel_name in metadata_dict["channels_shift"]:
            error_msg, shifts = choose_shift_widget(metadata_df, metadata_dict, fov_name_col, channel_name=channel_name)
            if error_msg != "":
                st.error(error_msg)
            else:
                channel_shifts[channel_name] = shifts
        shift_finished = st.button("Confirm Time Gates (if applicable) and Shift for each channel")
        if shift_finished:
            # write the shift, time gates and fitting options to the metadata file
            for channel_name in channel_shifts:
                metadata_df[f"{channel_name}_shift"] = channel_shifts[channel_name]
                if "start" in metadata_dict[channel_name]:
                    metadata_df[f"{channel_name}_start"] = metadata_dict[channel_name]["start"]
                if "end" in metadata_dict[channel_name]:
                    metadata_df[f"{channel_name}_end"] = metadata_dict[channel_name]["end"]
                if "num_components" in metadata_dict[channel_name]:
                    metadata_df[f"{channel_name}_num_components"] = metadata_dict[channel_name]["num_components"]
            
            if "fitting_algo" in metadata_dict:
                metadata_df["fitting_algo"] = metadata_dict["fitting_algo"]
            if "fitting_mode" in metadata_dict:
                metadata_df["fitting_mode"] = metadata_dict["fitting_mode"]
 
            # Store the updated metadata_df in session state so it persists across rerun
            st.session_state["last_extracted_metadata"] = metadata_df
            st.session_state["choosing_shift"] = False
            st.session_state["shift_ready"] = False
            st.rerun()
                
    elif "Numeric Feature Extraction" in selected_step and st.session_state["shift_ready"] and metadata_df is not None:
        single_cell_features = fov_extraction_widget(metadata_df, metadata_dict)
        if not single_cell_features.empty:
            st.success(f"Field of view features with ✅ are extracted successfully {happy_emoji}! FOVs with ❌ (if any) are excluded. The first few rows of the features are shown below.")
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
                        st.success(f"✅ Single cell features exported successfully to {csv_path} {happy_emoji}")
                    except Exception as e:
                        st.error(f"❌ Error exporting the single cell features: {str(e)}")
            else:
                st.download_button(label="Download single cell features as CSV", data=single_cell_features.to_csv(), file_name= f"single_cell_features_{timestamp}.csv")
    elif "Categorical Feature Extraction" in selected_step and df_folder_path != "" and len(available_dfs) > 0:
        combined_df, available_categories = check_and_merge_df_widget(available_dfs)
        map_categories_to_labels_widget(available_categories, combined_df, delimiter, df_folder_path)