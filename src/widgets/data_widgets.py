import streamlit as st
import pandas as pd
import random

from src.features import get_features, check_and_fix_df
from src.image_extraction import image_fit_extraction
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

def check_img_features(single_img_cell_features, image_name):
    """
    Drop the cells that have '--' or NaN values.
    """
    total_cells = len(single_img_cell_features)
    
    # Create boolean masks for invalid cells
    dash_mask = single_img_cell_features.astype(str).apply(lambda x: x.str.contains("--").any(), axis=1)
    nan_mask = single_img_cell_features.isna().any(axis=1)
    
    # Combine masks
    invalid_mask = dash_mask | nan_mask
    invalid_cells = invalid_mask.sum()
    
    if invalid_cells > 0:
        st.warning(
            f"The image {image_name} has **{invalid_cells}** cell(s) with '--' or NaN values out of {total_cells} cell(s). "
            "They are removed from the data because **all** of the pixels of those cells are masked by SPC image output files."
        )
        # Filter out invalid cells using the combined mask
        single_img_cell_features = single_img_cell_features[~invalid_mask]
    
    return single_img_cell_features
            


@st.cache_data
def image_extraction_widget(metadata_df, analysis_type, fit_free, has_nadh, has_fad, num_cols=3):

    single_cell_features = pd.DataFrame()
    image_names = metadata_df['image_name'].tolist()
   
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
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"Image name: **{image_name}**")
                    if analysis_type == "SPCImage" or analysis_type == "ROI Summing Fit":
                        metadata = metadata_df[metadata_df['image_name'] == image_name].iloc[0]
                        error_msg, single_cell_features_img = image_fit_extraction(metadata, analysis_type, has_nadh, has_fad, fit_free)
                        if error_msg != "":
                            st.error(error_msg)
                        else:
                            st.success("✅ Success!")
                            single_cell_features_img = check_img_features(single_cell_features_img, image_name)
                            single_cell_features = pd.concat([single_cell_features, single_cell_features_img])

    return single_cell_features

