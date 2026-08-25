import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import get_categorical_cols, get_fov_name_col, get_unique_cell_id_col
from src.emojis import happy_emoji


def map_categories_to_labels_widget(available_categories, combined_df, delimiter, df_folder_path):
    fov_name_col = get_fov_name_col()
    if fov_name_col not in combined_df.columns:
        st.warning(f"The {fov_name_col} column is not found in the combined dataset. Please check the {fov_name_col} column.")
        return None
    exp_fov_name = combined_df.iloc[0][fov_name_col]
    if delimiter == "":
        slots = [exp_fov_name]
    else:
        slots = exp_fov_name.split(delimiter)
    st.write("--------------------------------")
    st.write("Now your task is to map the categories to (combination of) slots.")
    st.info(f"Example fov_name: {exp_fov_name} has slots: {slots}")

    chosen_categories = st.multiselect("Choose Categorical features (specfied in Configuration @ Home tab) to populate", available_categories)

    if len(chosen_categories) > len(slots):
        st.warning(f"⚠️ Maximum {len(slots)} categories can be selected. Only the first {len(slots)} will be processed.")
        chosen_categories = chosen_categories[: len(slots)]

    cat_label_map = {}

    # Render multiselect widgets in columns (n_per_row)
    n_per_row = 3
    num_categories = len(chosen_categories)

    if num_categories > 0:
        num_rows = (num_categories + n_per_row - 1) // n_per_row  # Ceiling division

        for row in range(num_rows):
            start_idx = row * n_per_row
            end_idx = min(start_idx + n_per_row, num_categories)

            # Create columns for this row
            cols = st.columns(end_idx - start_idx)

            # Add multiselect widgets to this row
            for i, cat_idx in enumerate(range(start_idx, end_idx)):
                cat = chosen_categories[cat_idx]

                with cols[i]:
                    selected_slots = st.multiselect(
                        f"Slots for **{cat}**",
                        options=slots,
                        key=f"slot_{cat}",
                    )
                    # Store the indices of selected slots instead of the actual values
                    selected_indices = [slots.index(slot) for slot in selected_slots]
                    cat_label_map[cat] = selected_indices
    # preview the change by loading the first 5 unique values of fov_name_col or all unique values if less than 5
    # construct a new df with the selected categories and slots and cell_id from the first 5 unique fov_name_col values
    unique_fov_values = combined_df[fov_name_col].unique()
    if len(unique_fov_values) <= 5:
        preview_fov_values = unique_fov_values
    else:
        # use only the first 5 unique values
        preview_fov_values = unique_fov_values[:5]

    # filter the dataframe to only include one row for each selected unique fov_name_col value
    preview_df = combined_df[combined_df[fov_name_col].isin(preview_fov_values)].drop_duplicates(subset=[fov_name_col])[[fov_name_col]].copy()

    # add the chosen categories to the preview df and assign values based on the selected_indices from that category and concatenate them using delimiter
    for cat in chosen_categories:
        if cat_label_map[cat]:  # Only if user has selected slots for this category
            if delimiter == "":
                preview_df[cat] = preview_df[fov_name_col]  # no parsing when delimiter absent
            else:
                preview_df[cat] = preview_df[fov_name_col].apply(
                    lambda x: delimiter.join([x.split(delimiter)[i] for i in cat_label_map[cat]])
                )
        else:
            preview_df[cat] = ""  # Empty string if no slots selected

    st.write("**Preview of category mapping:**")
    st.write(preview_df)

    # let user confirm the mapping
    if st.button("Confirm category mapping & export the combined dataset"):
        # use the cat_label_map to map the categories to the combined df
        for cat in chosen_categories:
            if cat_label_map[cat]:  # Only if user has selected slots for this category
                if delimiter == "":
                    combined_df[cat] = combined_df[fov_name_col]  # no parsing when delimiter absent
                else:
                    combined_df[cat] = combined_df[fov_name_col].apply(
                        lambda x: delimiter.join([x.split(delimiter)[i] for i in cat_label_map[cat]])
                    )
            else:
                combined_df[cat] = ""  # Empty string if no slots selected
        # export the combined df
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_df_path = f"{df_folder_path}/{timestamp}_combined.csv"
        combined_df.to_csv(combined_df_path, index=False)
        st.success(f"Combined dataset exported to {combined_df_path}. {happy_emoji}")

    return cat_label_map

def _dup_values_msg(col, file):
    return f"The {col} column in {file} has duplicate values. Please check the {col} column."


def _fov_parts_msg(col, file, example):
    return f"The {col} column in {file} has different number of parts. For example, check fov_name: {example}."


def find_available_dfs_widget(df_folder_path, delimiter):
    unique_cell_id_col = get_unique_cell_id_col()
    fov_name_col = get_fov_name_col()
    # use glob to recursively find all the csv files in the folder that does end with _merged.csv and _metadata.csv
    if not os.path.isdir(df_folder_path):
        st.warning("Please provide a valid folder path.")
        return []

    path = Path(df_folder_path)
    # Find all CSV files recursively
    all_csv_files = [str(file) for file in path.rglob("*.csv")]

    # Filter out files ending with _merged.csv and _metadata.csv
    all_csv_files = [
        file for file in all_csv_files
        if not (file.endswith('_combined.csv') or file.endswith('_metadata.csv'))
    ]
    available_csv_files = []
    existing_cell_ids = []
    prev_num_parts = 0
    for file in all_csv_files:
        try:
            df = pd.read_csv(file)
        except Exception:
            st.warning(f"Failed to read the file {file}.")
            continue
        if unique_cell_id_col in df.columns and fov_name_col in df.columns:
            if df[unique_cell_id_col].duplicated().any():
                st.warning(_dup_values_msg(unique_cell_id_col, file))
                continue
            # check if the unique_row_id_col has value in all rows
            if df[unique_cell_id_col].isna().any():
                st.warning(f"The {unique_cell_id_col} column in {file} has NaN values. Please check the {unique_cell_id_col} column.")
                continue
            # check if all rows of this column can be split by the delimiter in equal number of parts
            # reject if not
            cell_ids = df[unique_cell_id_col].tolist()
            fov_names = df[fov_name_col].unique()
            # A cell_id already seen in a previously-loaded file means the ids collide
            # across files: warn once and skip the whole file, as the checks above do.
            existing_ids = set(existing_cell_ids)
            if any(cell_id in existing_ids for cell_id in cell_ids):
                st.warning(_dup_values_msg(unique_cell_id_col, file))
                continue
            existing_cell_ids.extend(cell_ids)

            if delimiter == "":
                available_csv_files.append(file)
                continue

            fov_names_parts = [len(fov_name.split(delimiter)) for fov_name in fov_names]
            if fov_names_parts == []:
                st.warning(f"The {unique_cell_id_col} column in {file} is empty.")
                continue
            elif len(set(fov_names_parts)) > 1:
                # find the first row that has different number of parts
                first_row_with_different_parts = fov_names_parts.index(max(fov_names_parts))
                first_row_with_different_parts_fov_name = fov_names[first_row_with_different_parts]
                st.warning(_fov_parts_msg(fov_name_col, file, first_row_with_different_parts_fov_name))
                continue
            elif fov_names_parts[0] == 1:
                st.warning(f"Playground failed to parse the {fov_name_col} column based on the delimiter: {delimiter}. For example, check fov_name: {fov_names[0]}.")
                continue
            if prev_num_parts == 0:
                prev_num_parts = fov_names_parts[0]
            elif prev_num_parts != fov_names_parts[0]:
                st.warning(_fov_parts_msg(fov_name_col, file, fov_names[0]))
                continue

            available_csv_files.append(file)

    return available_csv_files

def check_and_merge_df_widget(available_dfs):
    # what we can assume about each df:
    # it is openable, has a unique and no-nan cell_id column
    # between df, the cell_id is unique
    first_df = pd.read_csv(available_dfs[0])
    combined = first_df.dropna(axis=1, how='all').copy()
    st.info("Merging datasets...")
    for i, nxt in enumerate(available_dfs[1:], start=1):
        st.write(f"Merging dataset {i+1} ({nxt}) into the combined dataset. ")
        nxt_df = pd.read_csv(nxt)
        # remove all empty columns
        nxt_df = nxt_df.dropna(axis=1, how='all')
         # --- 1.  Compute column differences -----------------------------
        only_left  = combined.columns.difference(nxt_df.columns)      # in combined, not in nxt
        only_right = nxt_df.columns.difference(combined.columns)
        if len(only_left) != 0:
            st.write(f"Columns dropped from combined : {list(only_left)}")
        if len(only_right) != 0:
            st.write(f"Columns dropped from {nxt} : {list(only_right)}")
        common = combined.columns.intersection(nxt_df.columns)        # fast set-intersection
        combined = pd.concat(
            [combined[common], nxt_df[common]],        # input list
            axis=0,                                # stack rows
            ignore_index=True,                     # re-number the index
            verify_integrity=False,                # no duplicate-index check; ignore_index renumbers
            join="inner"                           # redundant but explicit
        )
    st.write(f"Finished combining all datasets and got {len(combined)} cells!")
    categorical_cols = get_categorical_cols()
    available_categories = [category for category in categorical_cols if category not in combined.columns]
    return combined, available_categories

