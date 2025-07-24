import streamlit as st
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from src.feature_types import categorical_cols, unique_cell_id_col

def map_categories_to_labels_widget(available_categories, combined_df, delimiter, df_folder_path):
    exp_cell_id = combined_df.iloc[0][unique_cell_id_col]
    slots = exp_cell_id.split(delimiter)
    st.write("--------------------------------")
    st.write("Now your task is to map the categories to (combination of) slots.")
    st.info(f"Example cell_id: {exp_cell_id} has slots: {slots}")
    
    chosen_categories = st.multiselect("Choose Categorical features to populate", available_categories)
    
    # Limit to maximum 4 categories
    if len(chosen_categories) > len(slots) - 1: # because one of the slot has to be cell_label
        st.warning(f"⚠️ Maximum {len(slots) - 1} categories can be selected. Only the first {len(slots) - 1} will be processed.")
        chosen_categories = chosen_categories[: len(slots) - 1]
    
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
    # preview the change by loading the first 5 rows or all rows if less than 5 of the combined df
    # construct a new df with the selected categories and slots and cell_id from the first 5 rows
    if len(combined_df) <= 5:
        # use only the cell_id column
        preview_df = combined_df[['cell_id']].copy()
    else:
        preview_df = combined_df[['cell_id']].iloc[:5].copy()
    
    # add the chosen categories to the preview df and assign values based on the selected_indices from that category and concatenate them using delimiter
    for cat in chosen_categories:
        if cat_label_map[cat]:  # Only if user has selected slots for this category
            preview_df[cat] = preview_df['cell_id'].apply(lambda x: delimiter.join([x.split(delimiter)[i] for i in cat_label_map[cat]]))
        else:
            preview_df[cat] = ""  # Empty string if no slots selected

    st.write("**Preview of category mapping:**")
    st.write(preview_df)

    # let user confirm the mapping
    if st.button("Confirm category mapping & export the combined dataset"):
        # use the cat_label_map to map the categories to the combined df
        for cat in chosen_categories:
            if cat_label_map[cat]:  # Only if user has selected slots for this category
                combined_df[cat] = combined_df['cell_id'].apply(lambda x: delimiter.join([x.split(delimiter)[i] for i in cat_label_map[cat]]))
            else:
                combined_df[cat] = ""  # Empty string if no slots selected
        # export the combined df
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_df_path = f"{df_folder_path}/{timestamp}_combined.csv"
        combined_df.to_csv(combined_df_path, index=False)
        st.success(f"Combined dataset exported to {combined_df_path}.")

    return cat_label_map

def find_available_dfs_widget(df_folder_path, delimiter):
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
        except Exception as e:
            st.warning(f"Failed to read the file {file}.")
            continue
        if unique_cell_id_col in df.columns:
            if df[unique_cell_id_col].duplicated().any():
                st.warning(f"The {unique_cell_id_col} column in {file} has duplicate values. Please check the {unique_cell_id_col} column.")
                continue
            # check if the unique_cell_id_col has value in all rows
            if df[unique_cell_id_col].isna().any():
                st.warning(f"The {unique_cell_id_col} column in {file} has NaN values. Please check the {unique_cell_id_col} column.")
                continue
            # check if all rows of this column can be split by the delimiter in equal number of parts
            # reject if not
            cell_ids = df[unique_cell_id_col].tolist()
            # check if every cell_id is not in existing_cell_ids
            for cell_id in cell_ids:
                if cell_id in existing_cell_ids:
                    st.warning(f"The {unique_cell_id_col} column in {file} has duplicate values. Please check the {unique_cell_id_col} column.")
                    continue
            existing_cell_ids.extend(cell_ids)

            cell_ids_parts = [len(cell_id.split(delimiter)) for cell_id in cell_ids]
            if cell_ids_parts == []:
                st.warning(f"The {unique_cell_id_col} column in {file} is empty.")
                continue
            elif len(set(cell_ids_parts)) > 1:
                # find the first row that has different number of parts
                first_row_with_different_parts = cell_ids_parts.index(max(cell_ids_parts))
                first_row_with_different_parts_cell_id = cell_ids[first_row_with_different_parts]
                st.warning(f"The {unique_cell_id_col} column in {file} has different number of parts. For example, check cell_id: {first_row_with_different_parts_cell_id}.")
                continue
            elif cell_ids_parts[0] == 1:
                st.warning(f"Playground failed to parse the {unique_cell_id_col} column based on the delimiter: {delimiter}. For example, check cell_id: {cell_ids[0]}.")
                continue
            if prev_num_parts == 0:
                prev_num_parts = cell_ids_parts[0]
            elif prev_num_parts != cell_ids_parts[0]:
                st.warning(f"The {unique_cell_id_col} column in {file} has different number of parts. For example, check cell_id: {cell_ids[0]}.")
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
            verify_integrity=False,                # raise if rows double-counted
            join="inner"                           # redundant but explicit
        )
    st.write(f"Finished combining all datasets and got {len(combined)} cells!")
    available_categories = [category for category in categorical_cols if category not in combined.columns]
    return combined, available_categories

    