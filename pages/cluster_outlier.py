import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import os
from pathlib import Path
from dimension_reduction import dimension_reduction, create_dim_reduction_figure
from navigation import render_top_menu
from features import get_features, fix_df
from widgets import create_filters, create_singleSelects_vars, create_multiSelects_vars, create_checkboxes
from roi_sum import roi_sum_dimensionReduction
from input import sdts_in_dir, fad_suffix, nadh_suffix, mask_suffix

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Clustering")
    method = st.selectbox(
        "Select a clustering & outlier detection method",
        ["PCA: fitted features", "PCA: raw data", "UMAP: fitted features", "UMAP: raw data", "Image Level Boxplots"],
    )  
    upload_complete = False 
    if method == "Image Level Boxplots" or "fitted features" in method:
        uploaded_csv = st.file_uploader("Upload the CSV file from Region Props", type=["csv"])
    
        if uploaded_csv is not None:
        # Read the uploaded data
            df = pd.read_csv(uploaded_csv)
            numeric_cols, nadh_cols, fad_cols, morphology_cols, error_msg = get_features(df)
            if error_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
                upload_complete = False
            else:
                df = fix_df(df)
               # st.markdown("<h6 style='text-align: center;'>File uploaded successfully.</h6>", unsafe_allow_html=True)
                upload_complete = True
            if method == "Image Level Boxplots":
                selected_var = create_singleSelects_vars(nadh_cols, fad_cols, morphology_cols)
            else:
                nadh_vars, fad_vars, morphology_vars = create_multiSelects_vars(nadh_cols, fad_cols, morphology_cols)
        
    elif "raw data" in method:
        st.markdown("**Copy and paste the *path* to the folder containing the sdt files *and* masks in the text box below.**")
        st.markdown("<h7 style='text-align: center; color: red;'>Note: this tool only works ***offline***, as the online app does not have access to your files.</h7>", unsafe_allow_html=True)
        
        folder_path = st.text_input("Enter a folder path:")
        selected_raw = st.selectbox("Select NADH or FAD", ["NADH", "FAD"])
        if folder_path and st.button("List Files & Run"):
            if os.path.isdir(folder_path):
                images, error_msg, has_nadh, has_fad  = sdts_in_dir(folder_path)
                if len(images) > 0:
                    upload_complete = True
                if error_msg != "":
                    st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
                if not has_nadh and selected_raw == "NADH":
                    st.markdown(f"<h5 style='text-align: center; color: red'>No NADH sdts found in the folder.</h5>", unsafe_allow_html=True)
                    upload_complete = False
                if not has_fad and selected_raw == "FAD":
                    st.markdown(f"<h5 style='text-align: center; color: red'>No FAD sdts found in the folder.</h5>", unsafe_allow_html=True)
                    upload_complete = False
                st.write(images)

                if upload_complete is False: 
                    st.markdown(f"<h7 style='text-align: center;'>See error msgs. No sdt found or no mask associated with sdts found. \
                                It looks for {nadh_suffix} suffix for nadh sdts, {fad_suffix} for fad sdts, and {mask_suffix} suffix \
                                and 'mask' keyword for mask files. </h7>", unsafe_allow_html=True)
            else:
                st.markdown("***Warning: The provided path is not a directory or doesn't exist.***")

    if upload_complete is False:
        st.write("Please upload a file/folder path to begin.")

with col2:
    if upload_complete: 
        if "fitted features" in method: 
            if "All NADH Variables" in nadh_vars:
                nadh_vars = nadh_cols
            if "All FAD Variables" in fad_vars:
                fad_vars = fad_cols
            if "All Morphology Variables" in morphology_vars:
                morphology_vars = morphology_cols
            if len(nadh_vars + fad_vars + morphology_vars) > 1:
                # Step 1: Filter the data
                filtered_df, color_by_options, cols = create_filters(df)

                if "removed_images" not in st.session_state:
                    st.session_state["removed_images"] = []
                if "removed_cells" not in st.session_state:
                    st.session_state["removed_cells"] = []
                if "remove_images" not in st.session_state:
                    st.session_state.remove_images = True  # Initialize 'Remove Images' checked
                if "remove_cells" not in st.session_state:
                    st.session_state.remove_cells = False  # Initialize 'Remove Cells' unchecked

                st.session_state["df_removed"] = filtered_df[
                    (~filtered_df["image_name"].isin(st.session_state["removed_images"])) &
                    (~filtered_df["base_name"].isin(st.session_state["removed_cells"]))
                ].reset_index(drop=True)

                ## Step 2: Dimension reduction
                selected_vars = nadh_vars + fad_vars + morphology_vars
                X = st.session_state["df_removed"][selected_vars]
                # Make sure that after filtering, the data is not empty
                if not X.empty:
                    method = "PCA" if "PCA" in method else "UMAP"
                    df_reduced, exp_var  = dimension_reduction(X, n_components=2, method=method)
                
                ## Step 3: Plotting with the interactivity of removing outliers
                    df_reduced["base_name"] = st.session_state["df_removed"]["base_name"]
                    df_reduced["image_name"] = st.session_state["df_removed"]["image_name"]

                    for col in color_by_options:
                        df_reduced[col] = st.session_state["df_removed"][col]
                    fig = create_dim_reduction_figure(df_reduced, method=method, colored_by=color_by_options, exp_var=exp_var)
                    clicked_points = plotly_events(
                        fig, 
                        click_event=True, 
                        hover_event=False, 
                        select_event=False
                    )
                    checkbox1, checkbox2 = create_checkboxes()
                    if clicked_points:
                        clicked_point = clicked_points[0]
                        point_index =  clicked_point["pointIndex"]
                        trace_index = clicked_point["curveNumber"]
                        if st.session_state.remove_cells:
                            clicked_data = fig.data[trace_index]['text'][point_index]
                            st.write(f"You clicked on cell: {clicked_data}. Do you want to remove this cell?")
                        else:
                            clicked_data = fig.data[trace_index]['customdata'][point_index]
                            st.write(f"You clicked on image: {clicked_data}. Do you want to remove this image?")

                        if st.button("Confirm Removal"):
                            # Remove rows with the clicked base_name
                            if st.session_state.remove_cells:
                                st.session_state["removed_cells"].append(clicked_data)
                            else: 
                                st.session_state["removed_images"].append(clicked_data)
                            st.rerun()

                    if len(st.session_state["removed_images"]) > 0 or len(st.session_state["removed_cells"]) > 0:
                        show_images, show_cells = st.columns([0.5, 0.5])
                        with show_images: 
                            st.write("Removed images:")
                            st.write(st.session_state["removed_images"])
                        with show_cells:
                            st.write("Removed cells:")
                            st.write(st.session_state["removed_cells"])

                        col1, col2 = st.columns([0.2, 1])
                        with col1:
                            if st.button("Reset"):
                                st.session_state["removed_images"] = []
                                st.session_state["removed_cells"] = []
                                st.rerun()
                        with col2:
                            df_outliers_removed = df[
                                (~df["image_name"].isin(st.session_state["removed_images"])) &
                                (~df["base_name"].isin(st.session_state["removed_cells"]))
                            ]
                            st.download_button(
                                label="Download Outliers Removed CSV",
                                data=df_outliers_removed.to_csv(index=False),
                                file_name=f"{uploaded_csv.name}_outliers_removed.csv",
                                mime="text/csv"
                            )

                    st.markdown("<h5 style='text-align: center;'>Click on points to remove outlier cells or images to where the outliers belong</h5>", unsafe_allow_html=True)

                else: 
                    st.write("No data to plot")
            else:
                st.markdown("<h5 style='text-align: center;'>Please select at least two numeric variables for performing dimension reduction.</h5>", unsafe_allow_html=True)
        elif method == "Image Level Boxplots":
            filtered_df, color_by_options, cols = create_filters(df, color=False)
            if selected_var != "Select": 
                if (df["image_name"] == "missing image name").any():
                    st.markdown("<h5 style='text-align: center; color: Red;'>Warning: We cannot infer some/all image names from you base_name. We assume that the image name is the base_name without the cell number (which is found after the last underscore) </h5>", unsafe_allow_html=True)
                # Create a boxplot for the selected variable
                fig = px.box(filtered_df, x="image_name", y=selected_var, title=f"Boxplot for {selected_var}")
                st.plotly_chart(fig, use_container_width=True)
  
            else:
                st.markdown("<h5 style='text-align: center;'>Please select one variable to plot.</h5>", unsafe_allow_html=True)

        elif "raw data" in method:
            method = "PCA" if "PCA" in method else "UMAP"
            nadh_df, nadh_exp_var, fad_df, fad_exp_var, error_message = roi_sum_dimensionReduction(images, method=method)

            if nadh_df is not None and fad_df is not None:
                if selected_raw == "NADH":
                    fig = create_dim_reduction_figure(nadh_df, method=method, colored_by=["color_category"], exp_var=nadh_exp_var)
                else:
                    fig = create_dim_reduction_figure(fad_df, method=method, colored_by=["color_category"], exp_var=fad_exp_var)
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("Waiting for file/folder path upload")