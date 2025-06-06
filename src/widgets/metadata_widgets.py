import streamlit as st
import os 
import numpy as np
from src.metadata import list_files_with_suffix, list_files_with_filename, parse_metadata_file, spc_output_suffix, file_suffix_default
from src.widgets.data_widgets import happy_emoji
from src.file_io import load_image

def load_data_suffix_widget(analysis_type, fit_free, has_nadh, has_fad):
    """
    ROI summing fit: requires mask, IRF, and raw lifetime decay files for nadh and fad
    SPCImage: requires mask and SPC fitting outputs (a1, a2, t1, t2, and shift) for nadh and fad, and raw lifetime decay files for nadh and fad and IRF (if fit_free)
    K-Flow: requires cell histograms and IRF 
    Categorical Features: requires single cell features csv files
    """
    # based on the analysis type, display the default suffix for each require file type
    actual_file_suffix = {}
    error_msg = ""

    if analysis_type == "ROI Summing Fit":
        # required files: mask, IRF
        actual_file_suffix["mask"] = ""
        if has_nadh:
            actual_file_suffix["nadh decay"] = ""
            actual_file_suffix["nadh irf"] = ""
        if has_fad:
            actual_file_suffix["fad decay"] = ""
            actual_file_suffix["fad irf"] = ""
        
    elif "SPCImage" in analysis_type:
        # required files: mask, SPC fitting outputs (a1, a2, t1, t2, and shift)
        actual_file_suffix["mask"] = ""
        if has_nadh:
            # only use the a1. The rest will be deduced from the suffix of the shift 
            actual_file_suffix["nadh a1"] = ""
        if has_fad:
            actual_file_suffix["fad a1"] = ""
        if fit_free:
            if has_nadh:
                actual_file_suffix["nadh decay"] = ""
                actual_file_suffix["nadh irf"] = ""
            if has_fad:
                actual_file_suffix["fad decay"] = ""
                actual_file_suffix["fad irf"] = ""
            
    elif analysis_type == "K-Flow":
        # required files: cell histograms and IRF
        if has_nadh:
            actual_file_suffix["nadh histogram"] = ""
            actual_file_suffix["nadh irf"] = ""
        if has_fad:
            actual_file_suffix["red histogram"] = ""
            actual_file_suffix["red irf"] = ""
    # create a text input widget for each suffix in the dictionary, maximum 2 per row
    # dynamically determine how many rows are needed
    num_rows = (len(actual_file_suffix) + 1) // 2
    cols = st.columns(2)
    for i, (key, value) in enumerate(actual_file_suffix.items()):
        col = cols[i % 2]
        with col:
            # create a text input for the suffix
            suffix = st.text_input(f"Suffix for {key}", file_suffix_default[key], key=f"{key}_suffix", help=f"The filenames are expected to have *exactly* two parts: \
            *image_name + suffix*. All files from the same image should share the **same** image_name, with the only difference being the suffix. This is the \
                                   suffix for the {key} file")
            if suffix == "":
                error_msg += f"Please provide a suffix for {key}! "
            else:
                actual_file_suffix[key] = suffix
    if error_msg == "" and "SPCImage" in analysis_type:
        # load nadh and fad spc image output files suffixes
        suffix_info = f"For other SPCImage output files (a2, t1, t2, shift), the suffixes are automatically generated based on the provided a1 suffix \
            by replacing {spc_output_suffix['a1']} to get the followings: \n"
        for key, suffix in spc_output_suffix.items():
            if key == "a1": 
                # skip a1, since it is already in the actual_file_suffix dictionary
                continue
            if not fit_free and key == "shift":
                continue
            if has_nadh: 
                nadh_a1_suffix = actual_file_suffix["nadh a1"]
                actual_file_suffix["nadh " + key] = nadh_a1_suffix.replace(spc_output_suffix['a1'], suffix)
                # Use Markdown list syntax for line breaks in st.info
                # Prepend "- " to make it a list item and add backticks for clarity
                suffix_info += f"- nadh {key}: `{actual_file_suffix['nadh ' + key]}`\n"
            if has_fad:
                fad_a1_suffix = actual_file_suffix["fad a1"]
                actual_file_suffix["fad " + key] = fad_a1_suffix.replace(spc_output_suffix['a1'], suffix)
                suffix_info += f"fad {key}: `{actual_file_suffix['fad ' + key]}`\n"
                
        st.info(suffix_info)
        
    # check if the suffixes are valid
    return actual_file_suffix, error_msg

def find_shift_value(shift_path):
    """
    Find the shift value in the shift image.
    """
    shift_image = load_image(shift_path)
    # the shift image should at most contain 2 unique values, 0 and the true shift value
    unique_values = np.unique(shift_image)
    # remove 0 from the unique values
    unique_values = [val for val in unique_values if val != 0]
    if len(unique_values) == 1:
        return unique_values[0]
    else:
        raise ValueError(f"Cannot find the shift value in the shift image {shift_path}.")
    

@st.cache_data
def load_list_data_from_folder_widget(folder_path, file_suffix, num_cols=3):    
    """
    Load data from a folder and check its validity. Display the file sets for each image group. 
    file_names = image_name + suffix (exactly that, no more, no less)
    image_group: keyed by image_name, and the value is a list of all the files that belong to that image
    """
    
    valid_image_groups = {}

    # use the first key to get the list of images (it does not matter which key to use, since they are all required, they should all be there)
    image_search_suffix = list(file_suffix.values())[0]
    image_files = list_files_with_suffix(folder_path, image_search_suffix) # returned file paths are absolute paths in string format
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
                    matched_files = list_files_with_filename(folder_path, image_name + suffix)
                else:
                    matched_files = list_files_with_suffix(folder_path, suffix)
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
                        shift_found = True
                        st.write("✅ All files found.")
                        if "nadh shift" in image_group:
                            try: 
                               image_group["nadh shift value"] = find_shift_value(image_group["nadh shift"])
                            except Exception as e:
                                st.write(f"Error loading the nadh shift: {image_group["nadh shift"]}")
                                shift_found = False
                        if "fad shift" in image_group:
                            try: 
                                image_group["fad shift value"] = find_shift_value(image_group["fad shift"])
                            except Exception as e:
                                st.write(f"Error loading the fad shift: {image_group["fad shift"]}")
                                shift_found = False

            if missing_keys == [] and duplicate_keys == [] and shift_found:
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
    st.session_state["last_analysis_type"] = analysis_type
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