import streamlit as st
import pandas as pd
import random
import os 
from features import get_features, check_and_fix_df
from feature_groups import get_full_feature_name, feature_groups_prefix
from file_util import list_files_with_suffix, list_files_with_filename, parse_metadata_file, spc_output_suffix, file_suffix_default
from widgets.selection_widgets import multi_feature_select_widget
happy_celebratory_emojis = [
    "🥳",  # Partying Face
    "🎉",  # Party Popper
    "🎊",  # Confetti Ball
    "✨",  # Sparkles
    "🎈",  # Balloon
    "🎆",  # Fireworks
    "🎇",  # Sparkler
    "🤩",  # Star-Struck
    "😊",  # Smiling Face with Smiling Eyes
    "😃",  # Grinning Face with Big Eyes
    "😁",  # Beaming Face with Smiling Eyes
    "😄",  # Grinning Face with Smiling Eyes
    "🥰",  # Smiling Face with Hearts
    "🙌",  # Raising Hands
    "🥂",  # Clinking Glasses
    "🍾",  # Bottle with Popping Cork
    "👍",  # Thumbs Up
    "😉",
]

# List of sad, regretful, and remorseful emojis
sad_regretful_emojis = [
    "😥",  # Sad but Relieved Face
    "😢",  # Crying Face
    "😭",  # Loudly Crying Face
    "😞",  # Disappointed Face
    "😟",  # Worried Face
    "🥺",  # Pleading Face (can imply regret or sadness)
    "💔",  # Broken Heart
    "😔",  # Pensive Face (can imply contemplation after a mistake)
    "😬",
    "😮‍💨",
    "😶‍🌫️",
    "🤔",
    "🤒",
    "🥶",
]

# Choose a random happy/celebratory emoji
happy_emoji = random.choice(happy_celebratory_emojis)

# Choose a random sad/regretful emoji
sad_emoji = random.choice(sad_regretful_emojis)

@st.cache_data
def load_csv(uploaded_csv):
    """
    Load a CSV file and check its validity.
    """
    upload_complete = False
    df = feature_cols_dict = None
        # check and fix the uploaded csv 
    if uploaded_csv is not None:
        # Read the uploaded data, explicitly preventing the first column from being used as the index
        df = pd.read_csv(uploaded_csv, index_col=False)
        df, warning_msg, error_msg = check_and_fix_df(df)

        if error_msg != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
            st.write(f"Therefore, we cannot extract data from your uploaded file {sad_emoji}")
        else:
            if warning_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
            # then we can extract the single cell features
            df, feature_cols_dict, warning_msg, error_msg = get_features(df)
            if error_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
                st.write(f"Therefore, we cannot extract data from your uploaded file {sad_emoji}")
            else:
                if warning_msg != "":
                    st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
                st.write(f"Data uploaded successfully {happy_emoji}")
                upload_complete = True
    return df, feature_cols_dict, upload_complete

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
        if has_fad:
            actual_file_suffix["fad decay"] = ""
        actual_file_suffix["irf"] = ""
    elif "SPCImage" in analysis_type:
        # required files: mask, SPC fitting outputs (a1, a2, t1, t2, and shift)
        actual_file_suffix["mask"] = ""
        if has_nadh:
            # only use the shift. The rest will be deduced from the suffix of the shift 
            actual_file_suffix["nadh shift"] = ""
        if has_fad:
            actual_file_suffix["fad shift"] = ""
        if fit_free:
            if has_nadh:
                actual_file_suffix["nadh decay"] = ""
            if has_fad:
                actual_file_suffix["fad decay"] = ""
            actual_file_suffix["irf"] = ""
    elif analysis_type == "K-Flow":
        # required files: cell histograms and IRF
        if has_nadh:
            actual_file_suffix["nadh histogram"] = ""
        if has_fad:
            actual_file_suffix["red histogram"] = ""
        actual_file_suffix["irf"] = ""
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
        suffix_info = "For other SPCImage output files (a1, a2, t1, t2), the suffixes are automatically generated based on the shift suffix: \n"
        for key, suffix in spc_output_suffix.items():
            if key == "shift":
                continue
            if has_nadh: 
                nadh_shift_suffix = actual_file_suffix["nadh shift"]
                actual_file_suffix["nadh " + key] = nadh_shift_suffix.replace("_shift.asc", suffix)
                # Use Markdown list syntax for line breaks in st.info
                # Prepend "- " to make it a list item and add backticks for clarity
                suffix_info += f"- nadh {key}: `{actual_file_suffix['nadh ' + key]}`\n"
            if has_fad:
                fad_shift_suffix = actual_file_suffix["fad shift"]
                actual_file_suffix["fad " + key] = fad_shift_suffix.replace("_shift.asc", suffix)
                suffix_info += f"fad {key}: `{actual_file_suffix['fad ' + key]}`\n"
                
        st.info(suffix_info)
        
    # check if the suffixes are valid
    return actual_file_suffix, error_msg
        

def load_list_data_from_folder_widget(folder_path, file_suffix, show_files=True):    
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
    max_cols = 3
    num_images = len(image_names)
    num_cols = min(max_cols, num_images)
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
                if key != "irf":
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
                    if missing_keys :
                        st.write("❌ Missing or duplicate files:")
                        for key in missing_keys:
                            if key != "irf":
                                st.write(f"- Missing {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Missing {key} with suffix: {file_suffix[key]}")
                        for key in duplicate_keys:
                            if key != "irf":
                                st.write(f"- Duplicate {key}: {image_name + file_suffix[key]}")
                            else:
                                st.write(f"- Duplicate {key} with suffix: {file_suffix[key]}")

                    else:
                        st.write("✅ All files found:")
                        for key, path in image_group.items():
                            st.write(f"- {key}: {os.path.basename(path)}")
            if missing_keys == [] and duplicate_keys == []:
                valid_image_groups[image_name] = image_group

    return valid_image_groups

def export_data_widget(images_df, folder_path):
    # use a botton to export the images as one csv file (one image per row) to the folder_path 
    confirm_export = st.button("Export Image Metadata as CSV", help=f"Export the image meta as one csv file (one image per row) to {folder_path}")
    if confirm_export:
        # convert the dictionary to a dataframe     
        # save the dataframe to a csv file
        csv_file_path = os.path.join(folder_path, "image_metadata.csv")
        images_df.to_csv(csv_file_path) # Save the DataFrame
        st.success(f"Image metadata exported successfully to {csv_file_path} {happy_emoji}")
        st.session_state["last_extracted_metadata"] = images_df
        st.session_state["last_extracted_metadata_filepath"] = csv_file_path

def parse_metadata_display_feature_widget(metadata_df, num_cols=3): 
    """
    Parse the metadata and display the features available to be extracted later for user to choose. 
    """
    error_msg, available_feature_groups_features, analysis_type, _ = parse_metadata_file(metadata_df)
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