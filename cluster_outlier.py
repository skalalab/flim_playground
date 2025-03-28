import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path
from dimension_reduction import dimension_reduction, create_dim_reduction_figure
from navigation import render_top_menu
from roi_sum import roi_sum_dimensionReduction
from input import sdt_folder_check

# Initialize session state so that 
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "raw_exp_var" not in st.session_state:
    st.session_state.raw_exp_var = None

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Clustering")
    method = st.selectbox(
        "Select a clustering & outlier detection method",
        ["PCA: fitted features", "UMAP: raw data", "Image Level Boxplots"],
    )  

    if "raw data" in method:
        folder_path = st.text_input("Copy and paste the *path* to the folder containing the sdt files *and* masks:")
        images, selected_channel, upload_complete = sdt_folder_check(folder_path)

       # print("I am here2: " + str(upload_complete))    
        if images is not None and len(images) > 0:   
            st.write(images)
        if upload_complete and images is not None:
            dr_method = "PCA" if "PCA" in method else "UMAP"

            df, exp_var, error_message = roi_sum_dimensionReduction(images, selected_channel=selected_channel, method=dr_method)
            
            if df is not None:
                st.session_state.raw_df = df
                st.session_state.raw_exp_var = exp_var
            else:
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_message}</h5>",unsafe_allow_html=True)
        #else: upload_complete = False
    if upload_complete is False:
        st.write("Please upload a file/folder path to begin.")

with col2:
    if "raw data" in method and st.session_state.raw_df is not None:
    # Create filters on df (does not re-upload or re-create df)
        dr_method = "PCA" if "PCA" in method else "UMAP"
        fig = create_dim_reduction_figure(filtered_df, method=dr_method, colored_by=["color_category"], exp_var=st.session_state.raw_exp_var)
        st.plotly_chart(fig, use_container_width=True)
    elif upload_complete: 

        if method == "Image Level Boxplots":
            if selected_var != "Select": 
                if (df["image_name"] == "missing image name").any():
                    st.markdown("<h5 style='text-align: center; color: Red;'>Warning: We cannot infer some/all image names from you base_name. We assume that the image name is the base_name without the cell number (which is found after the last underscore) </h5>", unsafe_allow_html=True)
                # Create a boxplot for the selected variable
                fig = px.box(filtered_df, x="image_name", y=selected_var, title=f"Boxplot for {selected_var}")
                st.plotly_chart(fig, use_container_width=True)
  
            else:
                st.markdown("<h5 style='text-align: center;'>Please select one variable to plot.</h5>", unsafe_allow_html=True)

    else:
        st.write("Waiting for file/folder path upload")