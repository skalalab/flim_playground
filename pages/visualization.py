import streamlit as st
from streamlit_plotly_events import plotly_events

from src.widgets.data_widgets import load_csv, happy_emoji, sad_emoji
from src.widgets.selection_widgets import single_feature_select_widget, multi_feature_select_widget, twod_single_feature_select_widget
from src.widgets.visualization_widgets import umap_hyperParams_widget, phasor_params_widget, visual_encoding_channels_widget
from src.widgets.filter_widgets import filters_widget
from src.widgets.click_plot_widgets import add_image_or_cell_widget, add_img_cell_widget, display_infoList_widget, reset_widget, add_img_widget
from src.navigation import render_top_menu
from src.vis.multivar import dimension_reduction_plot
from src.vis.bivar import feature_2d_distribution_plot, phasor_plot
from src.vis.univar import image_comparison_plot, feature_histogram_plot, feature_gmm_plot, feature_comparison_plot
from src.vis.helpers import _ensure_aspect_ratio

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

multivar_methods = ["UMAP", "Principal Component Analysis"]
# methods to visualize based on a single feature
univar_methods = ["Feature Comparison", "Feature Histogram (GMM optional)", "Image Comparison"]
bivar_methods = ["2D Feature Distribution", "Phasor Plot"]
col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Visualizations")
    col1_1, col1_2 = st.columns([1, 1])
    with col1_1:
        visualization_type = st.selectbox(
            "Select a visualization type",
            ["Univariate", "Bivariate", "Multivariate"],
            help="Univariate: Visualize the distribution of a single feature. \
            Bivariate: Visualize the relationship between two features. \
            Multivariate: Visualize the relationship between multiple features."
        )
    available_methods = univar_methods if visualization_type == "Univariate" else bivar_methods if visualization_type == "Bivariate" else multivar_methods
    with col1_2:
        method = st.selectbox(
            "Select a visualization method",
            available_methods,
        )
    uploaded_csv = st.file_uploader("Upload the CSV file obtained from [Data Extraction](/data_extraction)", type=["csv"])
    df, feature_cols_dict, upload_complete = load_csv(uploaded_csv)
    st.session_state.vis_df = df

    if upload_complete:
        if method in univar_methods:
            selected_var = single_feature_select_widget(feature_cols_dict, n_per_row=2)
            if method == "Feature Comparison":
                selected_effect_size_method = st.selectbox("Select an effect size method", ["None", "Glass's Delta", "Cohen's Distance"], index=0)
        elif method in bivar_methods:
            if "2D" in method:
                selected_x, selected_y = twod_single_feature_select_widget(feature_cols_dict, n_per_row=2)
            elif method == "Phasor Plot":
                selected_channel, selected_harmonic, f = phasor_params_widget(feature_cols_dict)
        elif method in multivar_methods:
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
        filtered_df = filters_widget(st.session_state.vis_df)
        # for visualization that are point-based, provides the options for other visual encoding channels: opacity, shape, and separate by
        point_based = method not in ["Image Comparison", "Feature Histogram (GMM optional)"]
        color_based = method not in ["Image Comparison"]

        color_by, opacity_by, shape_by, separate_by = visual_encoding_channels_widget(filtered_df, color_based=color_based, point_based=point_based)

        # check if the df is empty after filtering
        if not filtered_df.empty:
            if method in univar_methods and selected_var != "Select":
                # drop rows with NaN values in the selected_var column
                filtered_df = filtered_df[filtered_df[selected_var].notna()]
                if len(filtered_df) > 0:
                    # Plot the filtered dataframe
                    if method == "Feature Comparison":
                        fig = feature_comparison_plot(filtered_df, selected_var, color_by, effect_size_method=selected_effect_size_method)
                        click_ready = True
                    elif method == "Image Comparison":
                        fig = image_comparison_plot(filtered_df, selected_var)
                        click_ready = True
                    elif method == "Feature Histogram (GMM optional)":
                        # create a switch to select between GMM and histogram
                        apply_gmm = st.checkbox("Apply Gaussian Mixture Model to the feature distribution", value=False, help="Fit Gaussian Mixture Models\
                        for each color group on the selected feature with 1, 2, and 3 components (fit on raw distribution, not on the histograms). \
                        Choose the one in which all the components are at least of 10% weight and has the lowest BIC score.")
                        if apply_gmm:
                            feature_gmm_plot(filtered_df, selected_var, color_by)
                        else: 
                            fig = feature_histogram_plot(filtered_df, selected_var, color_by)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No data available after removing rows with missing values {sad_emoji}")
            elif method in bivar_methods:
                if "2D" in method and selected_x != "Select" and selected_y != "Select":
                    # drop rows with NaN values in the selected_x and selected_y columns
                    filtered_df = filtered_df[filtered_df[selected_x].notna() & filtered_df[selected_y].notna()]
                    if len(filtered_df) > 0:
                        _ensure_aspect_ratio(aspect_ratio="1 / 1")
                        fig, table_md, gmm_df = feature_2d_distribution_plot(filtered_df, selected_x, selected_y, color_by)
                        col2_1, col2_2 = st.columns([1, 1])
                        with col2_1:
                            st.plotly_chart(fig, use_container_width=True)
                        with col2_2:
                            if table_md != []:
                                st.markdown(table_md)
                        if "2D_GMM_group" in gmm_df.columns:
                            # available for download
                            st.download_button(label="Download 2D GMM data", data=gmm_df.to_csv(index=False), file_name="2D_gmm_data.csv")
                            
                    else:
                        st.write("No data available after removing rows with missing values {sad_emoji}")
                elif method == "Phasor Plot":
                    if selected_channel is not None and selected_harmonic is not None and f is not None:
                        fig = phasor_plot(filtered_df, selected_channel, color_by=color_by, f=f, harmonic=selected_harmonic)
                        click_ready = True
                    else:
                        st.write("Your data does not contain the required features for phasor plot.")
                                   
            elif method in multivar_methods:
                if len(selected_features) < 2:
                    st.write("Please select at least two features for dimension reduction methods like PCA or UMAP.")
                else: 
                    # drop rows with NaN values in the selected_features columns
                    filtered_df = filtered_df[filtered_df[selected_features].notna().all(axis=1)]
                    
                    if len(filtered_df) > 0:
                        # plot the reduced data
                        fig = dimension_reduction_plot(filtered_df, selected_features, method=method, hyperParam_dict=hyperParam_dict, colored_by=color_by)
                        click_ready = True
                    else:
                        st.write(f"No data available after removing rows with missing values {sad_emoji}")
             
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
            st.markdown(f"<h5 style='text-align: center; color: red'>No data available after filtering {sad_emoji}</h5>", unsafe_allow_html=True)
        # Display added items and reset options (common logic)
        if len(st.session_state["added_images"]) > 0 or len(st.session_state["added_cells"]) > 0:
            display_infoList_widget()
            reset_widget() 
    else:
        st.write("Please upload a file to begin.")
