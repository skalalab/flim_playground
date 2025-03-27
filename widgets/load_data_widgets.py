import streamlit as st
import pandas as pd
from features import get_features, check_and_fix_df
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