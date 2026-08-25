import pandas as pd
import streamlit as st

from src.config import get_all_feature_extractors
from src.emojis import happy_emoji, sad_emoji
from src.widgets.analysis_config_widgets import (
    get_all_feature_groups,
    get_fov_name_col_analysis,
    get_unique_row_id_col,
)


@st.cache_data(show_spinner=False)
def _read_csv_cached(uploaded_csv):
    """Cache only the raw CSV parse — the expensive, per-rerun cost for large tables.

    Keyed on the uploaded file's *content* (st.cache_data hashes the buffer by
    bytes), so it is independent of the analysis-profile config that the rest of
    load_csv reads (row-id / FOV / feature grouping). That validation stays live
    and always reflects the active profile, avoiding stale results on a profile
    switch. cache_data returns a fresh copy each call, so downstream mutation of
    the frame is safe.
    """
    return pd.read_csv(uploaded_csv, index_col=False, low_memory=False)


def load_csv(uploaded_csv, categorical_cols, use_data_extraction=True):
    """
    Load a CSV file and check its validity.
    """
    upload_complete = False
    df = feature_groups_dict = None
        # check and fix the uploaded csv
    if uploaded_csv is not None:
        # Read the uploaded data (cached parse; index_col=False keeps the first column as data)
        df = _read_csv_cached(uploaded_csv)
        unique_row_id_col = get_unique_row_id_col(use_data_extraction)
        fov_name_col = get_fov_name_col_analysis(use_data_extraction)
        df, warning_msg, error_msg = check_and_fix_df(df, categorical_cols, unique_row_id_col, fov_name_col)

        if error_msg != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
            st.write(f"Therefore, we cannot extract data from your uploaded file {sad_emoji}")
        else:
            if warning_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
            # then we can extract the numeric features
            df, feature_groups_dict, warning_msg, error_msg = get_features(df, categorical_cols, use_data_extraction=use_data_extraction)
            if error_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
                st.write(f"Therefore, we cannot extract data from your uploaded file {sad_emoji}")
            else:
                if warning_msg != "":
                    st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
                st.write(f"Data uploaded successfully {happy_emoji}")
                upload_complete = True
    return df, feature_groups_dict, upload_complete

def match_col_name(col, col_list):
    """
    match_col_name: a function that takes a column name and a list of canonical column names and returns the first canonical column name that matches the column name
    """
    for col_name in col_list:
        # fuzzy match the column name with the canonical column name
        # e.g. "cell_line", "cell line", "cell-line", "Cell line", "Cell_line", "cell_Lines" all match "cell_line"
        # "treatments", "Treatment", "Treatments" all match "treatment"
        col_processed = col.lower().replace(" ", "_").replace("-", "_")
        # Both sides get the same normalisation, so a configured name may carry the
        # spacing and hyphens of the header it names ("IL-18") and still match.
        col_name_processed = col_name.lower().replace(" ", "_").replace("-", "_")

        # Check for direct match, match after removing/adding 's'
        if (col_processed == col_name_processed or
            (col_processed.endswith('s') and col_processed[:-1] == col_name_processed) or
            (col_name_processed.endswith('s') and col_processed == col_name_processed[:-1])):
            return col_name
    return None

_MISSING_FOV_NAME = "missing fov name"


def safe_split_with_logging(cell_id):
    try:
        if "_" not in cell_id:
            return _MISSING_FOV_NAME
        else:
            return cell_id.rsplit('_', 1)[0]
    except Exception:
        return _MISSING_FOV_NAME

def get_feature_groups_data_extraction(cols):
    """
    feature_groups_dict: a dictionary. Keys are the names of the feature group and values are a list of columns that belong to the group.
    Only feature groups that have at least one column are included in the dictionary.
    """
    all_feature_extractors = get_all_feature_extractors()
    feature_groups_dict = {}
    feature_groups_dict["Uncategorized Features"] = []
    for col in cols:
        # column format: extractor_channelName:feature_name
        # e.g. "Lifetime fit_Channel 1: G(1st)"
        # Derived features form a single cross-channel group; their name has no
        # "{extractor}_{channel}" structure, so bucket them before the splits.
        if col.startswith("Derived: "):
            feature_groups_dict.setdefault("Derived Features", []).append(col)
            continue
        # first split by ":"
        try:
            extractor_channel, feature = col.split(": ")
        except Exception:
            feature_groups_dict["Uncategorized Features"].append(col)
            continue
        try:
            extractor, channel = extractor_channel.split("_", 1)
        except Exception:
            feature_groups_dict["Uncategorized Features"].append(col)
            continue
        if extractor in all_feature_extractors:
            if extractor_channel not in feature_groups_dict:
                feature_groups_dict[extractor_channel] = []
            feature_groups_dict[extractor_channel].append(col)
        else:
            feature_groups_dict["Uncategorized Features"].append(col)
    # Move "Uncategorized Features" to the end of the dictionary
    if "Uncategorized Features" in feature_groups_dict:
        uncategorized = feature_groups_dict.pop("Uncategorized Features")
        if uncategorized:
            feature_groups_dict["Uncategorized Features"] = uncategorized

    return feature_groups_dict

def get_feature_groups_user_defined(cols):
    all_feature_groups = get_all_feature_groups()
    feature_groups_dict = {}
    feature_groups_dict["Uncategorized Features"] = []

    for col in cols:
        found_group = False
        for feature_group in all_feature_groups:
            cols_in_group = all_feature_groups[feature_group]
            if col in cols_in_group:
                if feature_group not in feature_groups_dict:
                    feature_groups_dict[feature_group] = []
                feature_groups_dict[feature_group].append(col)
                found_group = True
                break  # Column found in this group, no need to check other groups

        # Only add to uncategorized if it wasn't found in any group
        if not found_group:
            feature_groups_dict["Uncategorized Features"].append(col)

    # Move "Uncategorized Features" to the end of the dictionary
    if "Uncategorized Features" in feature_groups_dict:
        uncategorized = feature_groups_dict.pop("Uncategorized Features")
        if uncategorized:
            feature_groups_dict["Uncategorized Features"] = uncategorized
    return feature_groups_dict

def coerce_majority_numeric_cols(df, skip_cols):
    """
    Attempt to convert non-categorical object columns to numeric.
    Only accept the conversion when <= 1% of non-null values are
    non-numeric (i.e. the column is overwhelmingly numeric with a few
    stray strings like "N/A").  Columns with more than 1% non-numeric
    values are left untouched (likely genuinely categorical/text).

    Must stay free of Streamlit/config dependencies — it is embedded verbatim
    into exported analysis scripts via inspect.getsource().
    """
    warning_msg = ""
    for col in df.columns:
        if col not in skip_cols and not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_original = int(df[col].notna().sum())
            if non_null_original == 0:
                continue
            num_coerced = non_null_original - int(converted.notna().sum())
            coerced_pct = num_coerced / non_null_original
            if coerced_pct <= 0.01:
                if num_coerced > 0:
                    warning_msg += f"Warning: {num_coerced} non-numeric value{'s' if num_coerced > 1 else ''} in '{col}' were converted to NaN.<br>"
                df[col] = converted
    return df, warning_msg

def get_features(df, categorical_cols, use_data_extraction=True):
    """
    Extract all numeric features from the dataframe. Group them (by channel) based on the feature extractors:
    - morphology (mask morphology)
    - texture (texture features)
    - lifetime fit variables
    - lifetime fit free variables
    """
    unique_row_id_col = get_unique_row_id_col(use_data_extraction)
    fov_name_col = get_fov_name_col_analysis(use_data_extraction)
    error_msg = ""

    skip_cols = set([unique_row_id_col, fov_name_col] + list(categorical_cols))
    df, warning_msg = coerce_majority_numeric_cols(df, skip_cols)

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if use_data_extraction:
        feature_groups_dict = get_feature_groups_data_extraction(numeric_cols)
    else:
        feature_groups_dict = get_feature_groups_user_defined(numeric_cols)
    all_numerical_features_cols = []
    for feature_group, cols in feature_groups_dict.items():
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

    return df, feature_groups_dict, warning_msg, error_msg

def check_and_fix_df(df, categorical_cols, unique_row_id_col, fov_name_col):
    """
    check for df's metadata:
    - single-cell unique_identifier
    - fill in na values for categorical columns

    Must stay free of Streamlit/config dependencies — it is embedded verbatim
    into exported analysis scripts via inspect.getsource().
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
        df[fov_name_col] = df[fov_name_col].fillna("missing fov name")

    for col in df.columns:
        matched_categorical_col = match_col_name(col, categorical_cols)
        if matched_categorical_col is not None:
            # Two headers can normalise to the same canonical name ("IL-18" and "IL_18"
            # both do). Renaming the second would leave two columns sharing a name and
            # every df[name] lookup would return a frame, so it keeps its own name and is
            # reported. An exactly-spelled column is never skipped, so it wins the name
            # regardless of column order.
            if matched_categorical_col != col and matched_categorical_col in df.columns:
                warning_msg += (f"Warning: column '{col}' also reads as the categorical column "
                                f"'{matched_categorical_col}', which is already present. "
                                f"'{col}' was left under its own name.\n")
                continue
            # rename the column to match the canonical categorical column name
            df.rename(columns={col: matched_categorical_col}, inplace=True)
            # fix na values, and make sure all the values are labels rather than numbers
            series = df[matched_categorical_col]
            # A numeric column that has blanks is read as float, so a plate or day number
            # would label itself "1.0" instead of the "1" that was typed. Whole numbers go
            # through a nullable integer cast first to keep the label intact; genuinely
            # fractional values keep their decimals.
            if pd.api.types.is_float_dtype(series):
                real = series.dropna()
                if not real.empty and (real % 1 == 0).all():
                    series = series.astype("Int64")
            # Fill from the original null mask, not by matching the stringified "nan" /
            # "<NA>" - a column with a genuine "nan" label would be caught by that.
            df[matched_categorical_col] = series.astype(str).where(series.notna(), "N/A")

    return df, warning_msg, error_msg