import pandas as pd
import streamlit as st

from src.derived_features import compute_derived_features
from src.emojis import sad_emoji
from src.fov_extraction import fov_extraction


def check_fov_features(single_fov_cell_features):
    """
    Give warnings for cells that have NaN values.
    """
    total_cells = len(single_fov_cell_features)
    # Create boolean masks for nan cells
    nan_mask = single_fov_cell_features.isna().any(axis=1)
    
    # Count nan cells
    nan_cells = nan_mask.sum()
    
    if nan_cells > 0:
        st.warning(
            f"It has **{nan_cells}** cell(s) with NaN values out of {total_cells} cell(s). "
        )
    
    return single_fov_cell_features
            

def fov_extraction_widget(metadata_df, metadata_dict, num_cols=3):

    single_cell_features = pd.DataFrame()
    fov_name_col = metadata_dict["fov_name_col"]
    fov_names = metadata_df[fov_name_col].tolist()
   
    num_fovs = len(fov_names)
    if num_fovs > 0:
        st.markdown("##### :green[Fields of view:] \n")
    num_cols = min(num_cols, num_fovs)
    rows = (num_fovs + num_cols - 1) // num_cols

    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row * num_cols + col_idx
            if img_idx >= num_fovs:
                break
            fov_name = fov_names[img_idx]
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"**{fov_name}**")
                    metadata = metadata_df[metadata_df[fov_name_col] == fov_name].iloc[0]
                    error_msg, single_cell_features_fov = fov_extraction(metadata, metadata_dict) 
                    if error_msg != "":
                        st.error(f"{error_msg} {sad_emoji}")
                    else:
                        single_cell_features_fov = check_fov_features(single_cell_features_fov)
                        st.success("✅ Success!")
                        single_cell_features = pd.concat([single_cell_features, single_cell_features_fov])

    # Append derived-feature columns computed from the baked metadata definitions
    # (not live config), so a replayed metadata CSV reproduces the same output.
    if not single_cell_features.empty:
        cols_before = set(single_cell_features.columns)
        single_cell_features, derived_warnings = compute_derived_features(
            single_cell_features, metadata_dict.get("derived_features", [])
        )
        for warning in derived_warnings:
            st.warning(warning)
        # Check derived-column NaNs separately: per-FOV validation runs before
        # these columns are computed.
        total_cells = len(single_cell_features)
        new_derived_cols = [c for c in single_cell_features.columns
                            if c.startswith("Derived: ") and c not in cols_before]
        for col in new_derived_cols:
            nan_cells = int(single_cell_features[col].isna().sum())
            if nan_cells > 0:
                name = col.split(": ", 1)[1]
                st.warning(
                    f"Derived feature **{name}** has **{nan_cells}** NaN value(s) out "
                    f"of {total_cells} cell(s) (e.g. from divide-by-zero or a NaN operand)."
                )

    return single_cell_features