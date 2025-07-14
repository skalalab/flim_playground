import streamlit as st
import pandas as pd
from src.image_extraction import image_extraction

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
def image_extraction_widget(metadata_df, metadata_dict, num_cols=3):

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
                    metadata = metadata_df[metadata_df['image_name'] == image_name].iloc[0]
                    error_msg, single_cell_features_img = image_extraction(metadata, metadata_dict) 
                    if error_msg != "":
                        st.error(error_msg)
                    else:
                        single_cell_features_img = check_img_features(single_cell_features_img, image_name)
                        st.success("✅ Success!")
                        single_cell_features = pd.concat([single_cell_features, single_cell_features_img])

    return single_cell_features