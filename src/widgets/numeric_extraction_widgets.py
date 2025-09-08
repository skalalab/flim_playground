import streamlit as st
import pandas as pd
from src.fov_extraction import fov_extraction
import numpy as np

def check_fov_features(single_fov_cell_features, fov_name, fov_name_col):
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
            f"The {fov_name_col} {fov_name} has **{nan_cells}** cell(s) with NaN values out of {total_cells} cell(s). "
        )
    
    return single_fov_cell_features
            

def fov_extraction_widget(metadata_df, metadata_dict, num_cols=3):

    single_cell_features = pd.DataFrame()
    fov_name_col = metadata_dict["fov_name_col"]
    fov_names = metadata_df[fov_name_col].tolist()
   
    num_fovs = len(fov_names)
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
                    st.markdown(f"{fov_name_col}: **{fov_name}**")
                    metadata = metadata_df[metadata_df[fov_name_col] == fov_name].iloc[0]
                    error_msg, single_cell_features_fov = fov_extraction(metadata, metadata_dict) 
                    if error_msg != "":
                        st.error(error_msg)
                    else:
                        single_cell_features_fov = check_fov_features(single_cell_features_fov, fov_name, fov_name_col)
                        st.success("✅ Success!")
                        single_cell_features = pd.concat([single_cell_features, single_cell_features_fov])

    return single_cell_features