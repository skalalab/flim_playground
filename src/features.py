import pandas as pd
from src.feature_groups import required_cols, categorical_cols, feature_groups_default, feature_groups_prefix, feature_groups
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


def safe_split_with_logging(base_name):
    try:
        return base_name.rsplit('_', 1)[0]
    except Exception as e:   
        return "missing image name"

def get_feature_cols(cols, weighted_cols = False):
    """
    feature_cols_dict: a dictionary. Keys are the names of the feature group and values are a list of columns that belong to the group.
    Only feature groups that have at least one column are included in the dictionary.
    """
    feature_cols_dict = {}
    already_matched_cols = []
    for feature_group in feature_groups:
        # if the column is in the default list, add it to the group_cols
        # or if the column starts with any of the prefixes in the prefix list, add it to the group_cols
        group_cols = [c for c in cols if (feature_group in feature_groups_default and c in feature_groups_default[feature_group]) or 
        c.startswith(feature_groups_prefix[feature_group])]
        # remove the stdev columns from the group_cols
        # and remove the weighted columns if weighted_cols is False
        group_cols = [c for c in group_cols if "stdev" not in c and (weighted_cols or "weighted" not in c) and "Unnamed" not in c]
        # remove the already matched columns from the group_cols
        group_cols = [c for c in group_cols if c not in already_matched_cols]
        # only add non-empty feature groups to the dictionary
        if len(group_cols) > 0:
            feature_cols_dict[feature_group] = group_cols
            # remove the matched columns from the cols list
            already_matched_cols.extend(group_cols)
    
    return feature_cols_dict

def get_features(df):
    """
    Extract all numeric features from the dataframe. Categorize them into:
    - NADH fit variables 
    - FAD fit variables
    - morphology (mask morphology and feature distribution)
    - fit-free variables (e.g. phasor coordinates)
    
    """
    warning_msg = error_msg = ""
    # convert 
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    feature_cols_dict = get_feature_cols(numeric_cols)
    all_features_cols = []
    for feature_group, cols in feature_cols_dict.items():
        all_features_cols.extend(cols)

    if len(all_features_cols) == 0:
        error_msg += "Error: No feature found in the uploaded file.\n"
        return None, None, None, error_msg

    # keep only the columns that are later used in downstream analysis
    avilable_categorical_cols = [col for col in categorical_cols if col in df.columns]
    columns_to_keep = required_cols + avilable_categorical_cols + all_features_cols
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
    - single-cell unique_identifier: `cell_id` (required)
    - the image the cell comes from: `image_name`: base_name = {image_name}_{cell_label}
   
    - fill in na values for categorical columns
    """
    warning_msg = error_msg = ""
    df = df.reset_index(drop=True)

    # drop off the all empty columns
    empty_cols = df.columns[df.isnull().all()]
    if len(empty_cols) > 0:
        if len(empty_cols) <= 5:
            warning_msg += f"Warning: {empty_cols} columns are all empty. They will be removed.\n"
        else:
            warning_msg += f"Warning: {empty_cols[:5]} columns and {len(empty_cols) - 5} more are all empty. They will be removed.\n"
        df.drop(columns=empty_cols, inplace=True)
        if df.empty:
            error_msg += "Error: No data available after removing empty columns.\n"
            return None, warning_msg, error_msg

    # check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()]
    if len(duplicate_cols) > 0:
        if len(duplicate_cols) <= 5:
            warning_msg += f"Warning: {duplicate_cols} columns are duplicated. "
        else:
            warning_msg += f"Warning: {duplicate_cols[:5]} columns and {len(duplicate_cols) - 5} more are duplicated. "
        warning_msg += "The duplicate columns will be dropped, only the first one will be kept.\n"
        # drop the duplicate columns, only keep the first one
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
    # handle the required column: cell_id 
    # for backward compatibility, we also check for base_name and base
    if "base" in df.columns or "base_name" in df.columns:
        if "cell_id" in df.columns:
            # drop the cell_id column if it is not the same as base_name
            df.drop(columns=["cell_id"], inplace=True)
        df.rename(columns={"base": "base_name"}, inplace=True)
        df.rename(columns={"base_name": "cell_id"}, inplace=True)
    if "cell_id" not in df.columns:
        error_msg += "Error: cell_id/base_name column is missing in the uploaded file. It is required. \n"
        return None, warning_msg, error_msg
    
    if df["cell_id"].duplicated().any():
        first_duplicate = df["cell_id"].duplicated()
        first_duplicate_value = df["cell_id"][first_duplicate].iloc[0]
        first_duplicate_index = df.loc[first_duplicate].index[0]
        warning_msg += f"Warning: Duplicate values found in `cell_id` column. First duplicate found with cell_id: '{first_duplicate_value}' at row {first_duplicate_index}.\
            The duplicate rows will be dropped, only the first one will be kept.\n"
        # drop the duplicate rows, only keep the first one
        df = df.drop_duplicates(subset=["cell_id"], keep="first")
    if "image_name" not in df.columns:
        df['image_name'] = df["cell_id"].apply(safe_split_with_logging)
    else: 
        df["image_name"] = df["image_name"].fillna("missing image name")

    for col in df.columns:
        matched_categorical_col = match_col_name(col, categorical_cols)
        if matched_categorical_col is not None:
            # rename the column to match the canonical categorical column name
            df.rename(columns={col: matched_categorical_col}, inplace=True)
            # fix na values
            df[matched_categorical_col] = df[matched_categorical_col].fillna("N/A")
            df[matched_categorical_col] = df[matched_categorical_col].astype(str) # make sure all the values are not numbers

    # remove rows with "--" in any column
    original_row_count = len(df)
    df = df[df.apply(lambda row: all(cell != "--" for cell in row), axis=1)]
    # print out the number of rows removed
    rows_removed = original_row_count - len(df)
    if rows_removed > 0:
        warning_msg += f"Warning: {rows_removed} rows containing '--' were removed."
   
    return df, warning_msg, error_msg