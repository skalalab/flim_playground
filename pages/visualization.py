import streamlit as st
import pandas as pd
from streamlit_plotly_events import plotly_events

from widgets.load_data_widgets import load_csv, happy_emoji, sad_emoji
from widgets.selection_widgets import single_feature_select_widget, multi_feature_select_widget
from widgets.custom_widgets import umap_hyperParams_widget
from widgets.filter_widgets import filters_widget
from widgets.click_plot_widgets import add_image_or_cell_widget, add_img_cell_widget, display_infoList_widget, reset_widget, add_img_widget
from navigation import render_top_menu
from visualization_functions import feature_comparison_plot, dimension_reduction_plot, image_comparison_plot, feature_histogram_plot, feature_gmm_plot
from dimension_reduction import dimension_reduction

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
render_top_menu()

# initialize session_states
if "vis_df" not in st.session_state:
    st.session_state.vis_df = None
if "added_images" not in st.session_state:
    st.session_state["added_images"] = []
if "added_cells" not in st.session_state:
    st.session_state["added_cells"] = []
if "last_processed_click" not in st.session_state:
    st.session_state.last_processed_click = None
if "last_processed_click_img" not in st.session_state:
    st.session_state.last_processed_click_img = None

dimension_reduction_methods = ["UMAP", "Principal Component Analysis"]
# methods to visualize based on a single feature
feature_visualization_methods = ["Feature Comparison", "Image Comparison", "Feature Histogram (GMM optional)"]
col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Visualizations")
    method = st.selectbox(
        "Select a visualization method",
        dimension_reduction_methods + feature_visualization_methods + ["Phasor Plot"],
    )  
    uploaded_csv = st.file_uploader("Upload the CSV file obtained from [Data Extraction](/data_extraction)", type=["csv"])
    df, feature_cols_dict, upload_complete = load_csv(uploaded_csv)
    st.session_state.vis_df = df

    if upload_complete:
        if method in feature_visualization_methods:
            # single feature selection widget 
            selected_var = single_feature_select_widget(feature_cols_dict, n_per_row=2)
            if method == "Feature Comparison":
                selected_test = st.selectbox("Select a statistical test", ["None", "Glass's Delta"], index=0)
        elif method in dimension_reduction_methods:
            hyperParam_dict = {}
            # multiple features selection widget 
            selected_features = multi_feature_select_widget(feature_cols_dict, n_per_row=2)
            if method == "UMAP":
                n_neighbors, min_dist = umap_hyperParams_widget()
                hyperParam_dict["n_neighbors"] = n_neighbors
                hyperParam_dict["min_dist"] = min_dist

with col2:
    if upload_complete:
        # click_ready: boolean to check if the plot is ready for click events
        click_ready = False
        filtered_df, color_by_options = filters_widget(st.session_state.vis_df, wildcard=True)

        # check if the df is empty after filtering
        if not filtered_df.empty:
            if method in feature_visualization_methods:
                if selected_var != "Select": 
                    # Plot the filtered dataframe
                    if method == "Feature Comparison":
                        fig = feature_comparison_plot(filtered_df, selected_var, color_by_options, stats_test=selected_test)
                        click_ready = True
                    elif method == "Image Comparison":
                        fig = image_comparison_plot(filtered_df, selected_var)
                        click_ready = True
                    elif method == "Feature Histogram (GMM optional)":
                        # create a switch to select between GMM and histogram
                        apply_gmm = st.checkbox("Apply GMM to the feature distribution", value=False)
                        if apply_gmm:
                            fig, h_index_msg = feature_gmm_plot(filtered_df, selected_var, color_by_options)
                            if h_index_msg != "": 
                                st.info(h_index_msg)
                        else: 
                            fig = feature_histogram_plot(filtered_df, selected_var, color_by_options)
                        st.plotly_chart(fig, use_container_width=True)
            
            elif method in dimension_reduction_methods:
                if len(selected_features) < 2:
                    st.write("Please select at least two features for dimension reduction methods like PCA or UMAP.")
                else: 
                    X = filtered_df[selected_features]
                    # perform dimension reduction
                    df_reduced, exp_var = dimension_reduction(X, n_components=2, method=method, hyperParam_dict=hyperParam_dict)
                    # augment df_reduced with required columns and categorical columns used for coloring
                    df_reduced["cell_id"] = filtered_df["cell_id"].values
                    df_reduced["image_name"] = filtered_df["image_name"].values
                    # Add all color columns at once if there are any
                    if color_by_options:
                        df_reduced[color_by_options] = filtered_df[color_by_options].values
                    # plot the reduced data
                    fig = dimension_reduction_plot(df_reduced, method=method, colored_by=color_by_options, exp_var=exp_var)
                    click_ready = True
            elif method == "Phasor Plot":
                st.write("Will be available once the Data Extraction Playground is ready.")
                                      
            if click_ready: 
                if method == "Image Comparison":
                    current_clicked_points_img = plotly_events(
                        fig, click_event=True, hover_event=False, select_event=False, key="image_removal_only" # Use a unique key
                    )                                 
                    # --- Specific logic for Image Comparison outlier removal ---
                    # Process only if it's a new, non-empty click
                    if current_clicked_points_img and current_clicked_points_img != st.session_state.last_processed_click_img:
                        add_img_widget(current_clicked_points_img, fig)
                       
                    # Reset if the current event is empty (no click)
                    elif not current_clicked_points_img:
                         st.session_state.last_processed_click_img = None
                else:
                    current_clicked_points = plotly_events(
                        fig, click_event=True, hover_event=False, select_event=False, key="image_and_cell_removal" # Use a unique key
                    )
                    image_removal, cell_removal = add_image_or_cell_widget()
                
                    # Process only if it's a new, non-empty click
                    if current_clicked_points and current_clicked_points != st.session_state.last_processed_click:
                        # Standard outlier removal widget call
                        add_img_cell_widget(current_clicked_points, fig) 

                    # Reset if the current event is empty (no click)
                    elif not current_clicked_points:
                         st.session_state.last_processed_click = None
        
                if method == "Image Comparison":
                    st.markdown(f"<h5 style='text-align: center;'>Click on a boxplot to show the detailed info of the selected image {happy_emoji}</h5>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h5 style='text-align: center;'>Click on a point to show the detailed info of the selected image or cell {happy_emoji}</h5>", unsafe_allow_html=True)
        else: 
            st.markdown(f"<h5 style='text-align: center; color: red'>No data available after removing outliers and/or filtering {sad_emoji}</h5>", unsafe_allow_html=True)
        # Display added items and reset options (common logic)
        if len(st.session_state["added_images"]) > 0 or len(st.session_state["added_cells"]) > 0:
            display_infoList_widget()
            reset_widget() 
    else:
        st.write("Please upload a file to begin.")
