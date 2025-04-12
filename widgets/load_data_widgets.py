import streamlit as st
import pandas as pd
from features import get_features, check_and_fix_df
from folder_util import list_files_with_suffix, file_suffix_default
import random

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
        # Read the uploaded data
        df = pd.read_csv(uploaded_csv)
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

def load_data_suffix_widget(extraction_type, fit_free, has_nadh, has_fad):
    """
    ROI summing fit: requires mask, IRF, and raw lifetime decay files for nadh and fad
    SPCImage: requires mask and SPC fitting outputs (a1, a2, t1, t2, and shift) for nadh and fad, and raw lifetime decay files for nadh and fad (if fit_free)
    K-Flow: requires cell histograms and IRF 
    Categorical Features: requires single cell features csv files
    """
    # based on the extraction type, display the default suffix for each require file type
    actual_file_suffix = {}
    error_msg = ""
    if extraction_type == "Categorical Features": 
        pass
    elif extraction_type == "ROI Summing Fit":
        # required files: mask, IRF
        actual_file_suffix["mask"] = ""
        actual_file_suffix["irf"] = ""
        if has_nadh:
            actual_file_suffix["nadh decay"] = ""
        if has_fad:
            actual_file_suffix["fad decay"] = ""
    elif "SPCImage" in extraction_type:
        # required files: mask, SPC fitting outputs (a1, a2, t1, t2, and shift)
        actual_file_suffix["mask"] = ""
        # only use the shift. The rest will be deduced from the suffix of the shift 
        actual_file_suffix["shift"] = ""
        if fit_free:
            if has_nadh:
                actual_file_suffix["nadh decay"] = ""
            if has_fad:
                actual_file_suffix["fad decay"] = ""
    elif extraction_type == "K-Flow":
        # required files: cell histograms and IRF
        actual_file_suffix["irf"] = ""
        if has_nadh:
            actual_file_suffix["nadh histogram"] = ""
        if has_fad:
            actual_file_suffix["red histogram"] = ""
    
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
                                   suffix for the {key} file.")
            if suffix == "":
                error_msg += f"Please provide a suffix for {key}! "
            else:
                actual_file_suffix[key] = suffix
    return actual_file_suffix, error_msg
        

def load_data_from_folder_widget(folder_path, extraction_type, fit_free, has_nadh, has_fad, file_suffix):    
    """
    Load data from a folder and check its validity. Display the file sets for each image group.
    file_names = image_name + suffix (exactly that, no more, no less)
    """
    error_msg = ""
    image_groups = {}

    return image_groups, error_msg
