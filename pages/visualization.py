import streamlit as st
import pandas as pd
from streamlit_plotly_events import plotly_events

from features import get_features, check_and_fix_df
from widgets.selection_widgets import single_feature_select_widget, multi_feature_select_widget
from widgets.custom_widgets import umap_hyperParams_widget
from widgets.filter_widgets import filters_widget
from widgets.outlier_removal_widgets import remove_image_or_cell_widget
from navigation import render_top_menu
from visualization_functions import feature_comparison_plot, dimension_reduction_plot
from dimension_reduction import dimension_reduction

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
render_top_menu()

# initialize session_states
if "vis_df" not in st.session_state:
    st.session_state.vis_df = None
if "removed_images" not in st.session_state:
    st.session_state["removed_images"] = []
if "removed_cells" not in st.session_state:
    st.session_state["removed_cells"] = []
if "remove_image" not in st.session_state:
    st.session_state.remove_image = True  # Initialize 'Remove Images' checked
if "remove_cell" not in st.session_state:
    st.session_state.remove_cell = False  # Initialize 'Remove Cells' unchecked


col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Visualizations")
    method = st.selectbox(
        "Select a visualization method",
        ["Feature Comparison", "Principal Component Analysis", "UMAP", "Phasor Plot", "Image Comparison"],
    )  
    dimension_reduction_methods = ["UMAP", "Principal Component Analysis"]
    uploaded_csv = st.file_uploader("Upload the CSV file obtained from [Data Extraction](/data_extraction)", type=["csv"])
    upload_complete = False
    # check and fix the uploaded csv 
    if uploaded_csv is not None:
        # Read the uploaded data
        df = pd.read_csv(uploaded_csv)
        df, warning_msg, error_msg = check_and_fix_df(df)

        if error_msg != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
            st.write("Therefore, we cannot extract data from your uploaded file.")
        else:
            if warning_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
            # then we can extract the single cell features
            df, feature_cols_dict, warning_msg, error_msg = get_features(df)
            if error_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
                st.write("Therefore, we cannot extract data from your uploaded file.")
            else:
                if warning_msg != "":
                    st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
                st.write("Data uploaded successfully. Please select a feature to visualize.")
                upload_complete = True
                st.session_state.vis_df = df
    if upload_complete:
        if method == "Feature Comparison" or method == "Image Comparison":
            # single feature selection widget 
            selected_var = single_feature_select_widget(feature_cols_dict, n_per_row=2)
            if method == "Feature Comparison":
                selected_test = st.selectbox("Select a statistical test", ["None", "Mann-Whitney", "t-test_ind"], index=0)
        elif method in dimension_reduction_methods:
            # multiple features selection widget 
            selected_features = multi_feature_select_widget(feature_cols_dict, n_per_row=2)
            if method == "UMAP":
                n_neighbors, min_dist = umap_hyperParams_widget()

with col2:
    if upload_complete:
        filtered_df, color_by_options, cols = filters_widget(st.session_state.vis_df, color=True)
        st.session_state["df_outlier_removed"] = filtered_df[
                    (~filtered_df["image_name"].isin(st.session_state["removed_images"])) &
                    (~filtered_df["cell_id"].isin(st.session_state["removed_cells"]))
                ].reset_index(drop=True)
        if method == "Feature Comparison":
            if selected_var != "Select": 
                # Plot the filtered dataframe
                fig = feature_comparison_plot(st.session_state["df_removed"], selected_var, color_by_options, stats_test=selected_test)
                st.pyplot(fig, use_container_width=True)
        
        elif method in dimension_reduction_methods:
            if len(selected_features) < 2:
                st.write("Please select at least two features for dimension reduction methods like PCA or UMAP.")
            else: 
                X = st.session_state["df_outlier_removed"][selected_features]
                # perform dimension reduction
                
                
        
        image_removal, cell_removal = remove_image_or_cell_widget()
    else:
        st.write("Please upload a file to begin.")

