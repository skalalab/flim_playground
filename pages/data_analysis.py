import streamlit as st
import sys
from pathlib import Path
# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.dataset_io import load_csv, happy_emoji, sad_emoji
from src.widgets.selection_widgets import single_feature_select_widget, multi_feature_select_widget, twod_single_feature_select_widget
from src.widgets.visualization_widgets import umap_hyperParams_widget, phasor_params_widget, visual_encoding_channels_widget, plot_config_widget, tsne_hyperParams_widget
from src.widgets.filter_widgets import filters_widget
from src.navigation import render_top_menu
from src.vis.multivar import dimension_reduction_plot
from src.vis.bivar import feature_2d_distribution_plot, phasor_plot
from src.vis.univar import image_comparison_plot, feature_histogram_plot, feature_gmm_plot, feature_comparison_plot
from src.vis.helpers import apply_plot_styling
from src.widgets.analysis_config_widgets import dataset_config_widget, get_fov_name_col_analysis, get_unique_row_id_col
from src.widgets.classfication_widgets import classifier_options_widget, classification_plot_widget
from src.classify import run_classification
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
render_top_menu()

# initialize session_states
if "vis_df" not in st.session_state:
    st.session_state.vis_df = None
if "plot_point_size" not in st.session_state:
    st.session_state.plot_point_size = 5
if "plot_axis_label_size" not in st.session_state:
    st.session_state.plot_axis_label_size = 18
if "plot_legend_size" not in st.session_state:
    st.session_state.plot_legend_size = 16

multivar_methods = ["Dimension Reduction", "Classification"] #"Align Modalities"]
# methods to visualize based on a single feature
univar_methods = ["Feature Comparison", "Feature Histogram", "Image Comparison"]
bivar_methods = ["2D Feature Distribution", "Phasor Plot"]
col1, col2 = st.columns([0.4, 1])
with col1:
    cols = st.columns([0.6, 1])
    with cols[0]:
        analysis_type = st.radio(
            "### **Data Analysis**",
            [
            "### **Univariate**",
            "### **Bivariate**",
            "### **Multivariate**",
            ],
        )
    with cols[1]:
        available_methods = (
            univar_methods
            if "Univariate" in analysis_type
            else bivar_methods
            if "Bivariate" in analysis_type
            else multivar_methods
        )
        method = st.radio(
            "Methods",
            available_methods,
        )
    use_data_extraction = st.checkbox("Use Dataset from Data Extraction", value=True)
    unique_row_id_col = get_unique_row_id_col(use_data_extraction)
    fov_name_col = get_fov_name_col_analysis(use_data_extraction)
    instruction_text = "Upload the CSV file obtained from [Data Extraction](/data_extraction) directly." if use_data_extraction else "Use the right panel to configure so that your data is properly loaded."
    uploaded_csv = st.file_uploader(
        instruction_text,
        type=["csv"],
    )
    df, feature_groups_dict, upload_complete = load_csv(uploaded_csv, use_data_extraction=use_data_extraction)
    st.session_state.vis_df = df

    if upload_complete:
        if method in univar_methods:
            selected_var = single_feature_select_widget(feature_groups_dict, data_extraction=use_data_extraction, n_per_row=2)
            if method == "Feature Comparison":
                selected_effect_size_method = st.selectbox("Effect size method", ["None", "Glass's Delta", "Cohen's Distance"], index=0)
        elif method in bivar_methods:
            if "2D" in method:
                selected_x, selected_y = twod_single_feature_select_widget(feature_groups_dict, data_extraction=use_data_extraction, n_per_row=2)
            elif method == "Phasor Plot":
                selected_channel, selected_harmonic, f = phasor_params_widget(feature_groups_dict)
        elif method in multivar_methods:
            selected_features = multi_feature_select_widget(feature_groups_dict, data_extraction=use_data_extraction, n_per_row=2)
            if method == "Dimension Reduction":                
                dr_method = st.radio("Dimension Reduction Method", ["UMAP", "PCA", "t-SNE"], horizontal=True)
                if dr_method == "UMAP":
                    hyperParam_dict = umap_hyperParams_widget()
                elif dr_method == "t-SNE":
                    hyperParam_dict = tsne_hyperParams_widget()
            elif method == "Classification":
                cols = st.columns(2)
                with cols[0]:
                    classification_method = st.radio("Classifier", ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"])
                with cols[1]:
                    splits = st.slider("Train size (percentage of training data)", 0.5, 0.9, 0.7, 0.1)
    
with col2:
    if upload_complete:
        # click_ready: boolean to check if the plot is ready for click events
        data_export_ready = False
        filtered_df = filters_widget(st.session_state.vis_df)
        # for visualization that are point-based, provides the options for other visual encoding channels: opacity, shape, and separate by
        point_based = method not in ["Image Comparison", "Feature Histogram", "Classification"]
        color_based = method not in ["Image Comparison", "Classification"]
        image_based = method in ["Image Comparison"]
        separate_by_available = method in ["Feature Comparison"]
        fig = None
        # check if the df is empty after filtering
        if not filtered_df.empty:
            color_by, opacity_by, shape_by, separate_by = visual_encoding_channels_widget(filtered_df, color_based=color_based, point_based=point_based, separate_by_available=separate_by_available)
            if method in univar_methods and selected_var != "Select":
                # drop rows with NaN values in the selected_var column
                filtered_df = filtered_df[filtered_df[selected_var].notna()]
                if len(filtered_df) > 0:
                    # Plot the filtered dataframe
                    if method == "Feature Comparison":
                        fig = feature_comparison_plot(filtered_df, cell_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_var=selected_var, color_by=color_by, opacity_by=opacity_by, shape_by=shape_by, separate_by=separate_by, effect_size_method=selected_effect_size_method)
                    elif method == "Image Comparison":
                        fig = image_comparison_plot(filtered_df, fov_name_col=fov_name_col, selected_var=selected_var)
                    elif method == "Feature Histogram":
                        # create a switch to select between GMM and histogram
                        apply_gmm = st.checkbox("Apply Gaussian Mixture Model to the feature distribution", value=False, help="Fit Gaussian Mixture Models\
                        for each color group on the selected feature with 1, 2, and 3 components (fit on raw distribution, not on the histograms). \
                        Choose the one in which all the components are at least of 10% weight and has the lowest BIC score.")
                        if apply_gmm:
                            fig, df = feature_gmm_plot(filtered_df, selected_var, color_by)
                            data_export_ready = True
                        else: 
                            fig = feature_histogram_plot(filtered_df, selected_var, color_by)    
                else:
                    st.write("No data available after removing rows with missing values {sad_emoji}")
            elif method in bivar_methods:
                if "2D" in method and selected_x != "Select" and selected_y != "Select":
                    # drop rows with NaN values in the selected_x and selected_y columns
                    filtered_df = filtered_df[filtered_df[selected_x].notna() & filtered_df[selected_y].notna()]
                    if len(filtered_df) > 0:
                        fig, table_md, gmm_df = feature_2d_distribution_plot(filtered_df, row_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_x=selected_x, selected_y=selected_y, color_by=color_by, shape_by=shape_by, opacity_by=opacity_by)
                        data_export_ready = True
                    else:
                        st.write("No data available after removing rows with missing values {sad_emoji}")
                elif method == "Phasor Plot":
                    if selected_channel is not None and selected_harmonic is not None and f is not None:
                        fig = phasor_plot(filtered_df, unique_row_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_channel=selected_channel, color_by=color_by, shape_by=shape_by, opacity_by=opacity_by, f=f, harmonic=selected_harmonic)
                    else:
                        st.write("Your data does not contain the required features for phasor plot.")
                                   
            elif method in multivar_methods:
                if method == "Dimension Reduction":
                    if len(selected_features) < 2:
                        st.write("Please select at least two features for dimension reduction methods like PCA or UMAP.")
                    else: 
                        # drop rows with NaN values in the selected_features columns
                        filtered_df = filtered_df[filtered_df[selected_features].notna().all(axis=1)]
                        
                        if len(filtered_df) > 0:
                            # plot the reduced data
                            fig = dimension_reduction_plot(filtered_df, unique_row_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_features=selected_features, method=dr_method, hyperParam_dict=hyperParam_dict, colored_by=color_by, opacity_by=opacity_by, shape_by=shape_by)
                        else:
                            st.write(f"No data available after removing rows with missing values {sad_emoji}")
                elif method == "Classification":
                    error_msg, df_classify, sampling_method = classifier_options_widget(filtered_df, selected_features, classification_method, splits)
                    if error_msg:
                        st.error(error_msg)
                    else:
                        error_msg, results = run_classification(df_classify, classification_method, splits, sampling_method, random_state=42)
                        if error_msg:
                            st.error(error_msg)
                        else:
                            classification_plot_widget(results, classification_method)
                    
            if fig is not None: 
                fig = apply_plot_styling(fig, st.session_state.plot_point_size, st.session_state.plot_axis_label_size, st.session_state.plot_legend_size) 
                if method == "2D Feature Distribution":
                    col2_1, col2_2 = st.columns([1, 1])
                    with col2_1:
                       st.plotly_chart(fig, use_container_width=True)
                    with col2_2:
                        if table_md != []:
                            st.markdown(table_md, unsafe_allow_html=True)
                else:
                    st.plotly_chart(fig, use_container_width=True)
                # 1. Data export (if applicable)
                if data_export_ready:
                    # available for download
                    if method == "2D Feature Distribution" and "2D_GMM_group" in gmm_df.columns:
                        st.download_button(label="Download 2D GMM data", data=gmm_df.to_csv(index=False), file_name="2D_gmm_data.csv")
                    elif method == "Feature Histogram" and "GMM_group" in df.columns:
                        st.download_button(label="Download GMM Grouped Data", data=df.to_csv(index=False), file_name="gmm_grouped_data.csv", mime="text/csv", key="gmm_download")
                # 2. Plot configuration widget at the bottom - allows users to adjust styling after seeing plots 
                st.subheader("📊 Plot Styling")
                # Get current values from session state as defaults for the widgets
                new_point_size, new_axis_label_size, new_legend_size = plot_config_widget(point_based=point_based)
                style_changed = False
                if new_point_size != st.session_state.plot_point_size:
                    st.session_state.plot_point_size = new_point_size
                    style_changed = True
                if new_axis_label_size != st.session_state.plot_axis_label_size:
                    st.session_state.plot_axis_label_size = new_axis_label_size
                    style_changed = True
                if new_legend_size != st.session_state.plot_legend_size:
                    st.session_state.plot_legend_size = new_legend_size
                    style_changed = True   
                if style_changed:
                    st.rerun()
                               
        else: 
            st.markdown(f"<h5 style='text-align: center; color: red'>No data available after filtering {sad_emoji}</h5>", unsafe_allow_html=True)

    else:
        dataset_config_widget(use_data_extraction=use_data_extraction)


