import pandas as pd
from src.widgets.analysis_config_widgets import unique_row_id_col, fov_name_col, categorical_cols, all_numeric_col_groups
import streamlit as st
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
def load_csv(uploaded_csv, use_data_extraction=True):
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
            df, feature_cols_dict, warning_msg, error_msg = get_features(df, use_data_extraction=True)
            if error_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
                st.write(f"Therefore, we cannot extract data from your uploaded file {sad_emoji}")
            else:
                if warning_msg != "":
                    st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
                st.write(f"Data uploaded successfully {happy_emoji}")
                upload_complete = True
    return df, feature_cols_dict, upload_complete

def match_col_name(col, col_list):
    """
    match_col_name: a function that takes a column name and a list of canonical column names and returns the first canonical column name that matches the column name
    """
    for col_name in col_list:
        # fuzzy match the column name with the canonical column name
        # e.g. "cell_line", "cell line", "cell-line", "Cell line", "Cell_line", "cell_Lines" all match "cell_line"
        # "treatments", "Treatment", "Treatments" all match "treatment"
        col_processed = col.lower().replace(" ", "_").replace("-", "_")
        col_name_processed = col_name.lower() # Canonical names are assumed to be already processed (lowercase, underscores)

        # Check for direct match, match after removing/adding 's'
        if (col_processed == col_name_processed or
            (col_processed.endswith('s') and col_processed[:-1] == col_name_processed) or
            (col_name_processed.endswith('s') and col_processed == col_name_processed[:-1])):
            return col_name
    return None

def safe_split_with_logging(cell_id):
    try:
        if "_" not in cell_id:
            return "missing image name"
        else:
            return cell_id.rsplit('_', 1)[0]
    except Exception as e:   
        return "missing image name"

def get_feature_cols(cols, use_data_extraction=True):
    """
    feature_cols_dict: a dictionary. Keys are the names of the feature group and values are a list of columns that belong to the group.
    Only feature groups that have at least one column are included in the dictionary.
    """
    feature_cols_dict = {}
    feature_cols_dict["Uncategorized Features"] = []
    for col in cols:
        if use_data_extraction:
        # column format: extractor_channelName:feature_name
        # e.g. "Lifetime fit_Channel 1: G(1st)"
        # first split by ":"
            try:
                extractor_channel, feature = col.split(": ")
            except:
                feature_cols_dict["Uncategorized Features"].append(col)
                continue
            try:
                extractor, channel = extractor_channel.split("_")
            except:
                feature_cols_dict["Uncategorized Features"].append(col)
                continue
            if extractor in all_numeric_col_groups:
                if extractor_channel not in feature_cols_dict:
                    feature_cols_dict[extractor_channel] = []
                feature_cols_dict[extractor_channel].append(col)
            else:
                feature_cols_dict["Uncategorized Features"].append(col)
        else:
            pass
    # Move "Uncategorized Features" to the end of the dictionary
    if "Uncategorized Features" in feature_cols_dict:
        uncategorized = feature_cols_dict.pop("Uncategorized Features")
        if uncategorized:
            feature_cols_dict["Uncategorized Features"] = uncategorized
            
    return feature_cols_dict

def get_features(df, use_data_extraction=True):
    """
    Extract all numeric features from the dataframe. Group them (by channel) based on the feature extractors:
    - morphology (mask morphology)
    - texture (texture features)
    - lifetime fit variables
    - lifetime fit free variables
    """
    warning_msg = error_msg = ""
    # convert 
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    feature_cols_dict = get_feature_cols(numeric_cols, use_data_extraction)
    all_numerical_features_cols = []
    for extractor_channel, cols in feature_cols_dict.items():
        all_numerical_features_cols.extend(cols)

    if len(all_numerical_features_cols) == 0:
        error_msg += "Error: No feature found in the uploaded file.\n"
        return None, None, None, error_msg

    # keep only the columns that are later used in downstream analysis
    avilable_categorical_cols = [col for col in categorical_cols if col in df.columns]
    required_cols = [unique_row_id_col, fov_name_col] if fov_name_col not in avilable_categorical_cols else [unique_row_id_col]
    columns_to_keep = required_cols + avilable_categorical_cols + all_numerical_features_cols
    df = df[columns_to_keep]  
   
    # Print columns that contain NaN values
    columns_with_na = df.columns[df.isna().any()].tolist()
    if columns_with_na:
        num_na_columns = len(columns_with_na)
        if num_na_columns <= 5:
            warning_msg += f"Warning: {', '.join(columns_with_na)} column{'s' if num_na_columns > 1 else ''} contain{'s' if num_na_columns == 1 else ''} NaN values. "
        else:
            warning_msg += f"Warning: {', '.join(columns_with_na[:5])} and {num_na_columns - 5} more columns contain NaN values. "
    
    return df, feature_cols_dict, warning_msg, error_msg

def check_and_fix_df(df):
    """
    check for df's metadata: 
    - single-cell unique_identifier
    - the image the cell comes from: `fov_name`: unique_row_id_col = {fov_name}_{cell_label}
    - fill in na values for categorical columns
    """
    warning_msg = error_msg = ""
    df = df.reset_index(drop=True)
   
    # drop off the all empty columns
    empty_cols = df.columns[df.isnull().all()]
    if len(empty_cols) > 0:
        if len(empty_cols) <= 5:
            warning_msg += f"Warning: {empty_cols} columns are all empty. They were removed.\n"
        else:
            warning_msg += f"Warning: {empty_cols[:5]} columns and {len(empty_cols) - 5} more are all empty. They were removed.\n"
        df.drop(columns=empty_cols, inplace=True)
        if df.empty:
            error_msg += "Error: No data available after removing empty columns.\n"
            return None, warning_msg, error_msg

    # check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()]
    if len(duplicate_cols) > 0:
        if len(duplicate_cols) <= 5:
            warning_msg += f"Warning: {duplicate_cols} columns were duplicated. "
        else:
            warning_msg += f"Warning: {duplicate_cols[:5]} columns and {len(duplicate_cols) - 5} more were duplicated. "
        warning_msg += "The duplicate columns were dropped, only the first was kept.\n"
        # drop the duplicate columns, only keep the first one
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
    # handle the required unique cell identifier column

    if unique_row_id_col not in df.columns:
        error_msg += f"Error: {unique_row_id_col} column is missing in the uploaded file. It is required. \n"
        return None, warning_msg, error_msg
    
    if df[unique_row_id_col].duplicated().any():
        original_row_count = len(df)
        first_duplicate = df[unique_row_id_col].duplicated()
        first_duplicate_value = df[unique_row_id_col][first_duplicate].iloc[0]
        first_duplicate_index = df.loc[first_duplicate].index[0]
        warning_msg += f"Warning: Duplicate values found in `{unique_row_id_col}` column. First duplicate found with {unique_row_id_col}: '{first_duplicate_value}' at row {first_duplicate_index}.\
            The duplicate rows were dropped, only the first was kept.\n"
        # drop the duplicate rows, only keep the first one
        df = df.drop_duplicates(subset=[unique_row_id_col], keep="first")    
        # after fixing the df, print out the number of rows removed
        rows_removed = original_row_count - len(df)
        if rows_removed > 0:
            warning_msg += f"{rows_removed} rows were removed."
        
    # make sure unique_row_id_col is of type str
    df[unique_row_id_col] = df[unique_row_id_col].astype(str)
    if fov_name_col not in df.columns:
        df[fov_name_col] = df[unique_row_id_col].apply(safe_split_with_logging)
    else: 
        df[fov_name_col] = df[fov_name_col].fillna("missing image name")

    for col in df.columns:
        matched_categorical_col = match_col_name(col, categorical_cols)
        if matched_categorical_col is not None:
            # rename the column to match the canonical categorical column name
            df.rename(columns={col: matched_categorical_col}, inplace=True)
            # fix na values
            df[matched_categorical_col] = df[matched_categorical_col].fillna("N/A")
            df[matched_categorical_col] = df[matched_categorical_col].astype(str) # make sure all the values are not numbers

    return df, warning_msg, error_msg