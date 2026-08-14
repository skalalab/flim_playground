import sys
from pathlib import Path

import streamlit as st

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.classify import run_classification
from src.dataset_io import load_csv
from src.emojis import sad_emoji
from src.export_script import generate_script, get_effect_size_threshold_capture
from src.navigation import render_top_menu
from src.vis.bivar import feature_2d_distribution_plot, phasor_plot
from src.vis.helpers import apply_plot_styling, log_negative_error
from src.vis.multivar import dimension_reduction_plot
from src.vis.plot_defaults import (
    DEFAULT_AXIS_LABEL_FONT_SIZE,
    DEFAULT_COLORMAP,
    DEFAULT_LEGEND_FONT_SIZE,
    DEFAULT_POINT_SIZE,
)
from src.vis.univar import (
    feature_comparison_plot,
    feature_gmm_plot,
    feature_histogram_plot,
    fov_comparison_plot,
)
from src.widgets.analysis_config_widgets import (
    dataset_config_widget,
    get_categorical_cols_analysis,
    get_fov_name_col_analysis,
    get_unique_row_id_col,
)
from src.widgets.classification_widgets import (
    CLASSIFIER_OPTIONS,
    classification_plot_widget,
    classifier_hyperparams_widget,
    classifier_options_widget,
)
from src.widgets.filter_widgets import filters_widget
from src.widgets.selection_widgets import (
    multi_feature_select_widget,
    single_feature_select_widget,
    twod_single_feature_select_widget,
)
from src.widgets.visualization_widgets import (
    get_visual_group_keys,
    phasor_params_widget,
    plot_config_widget,
    reorder_x_axis_widget,
    tsne_hyperParams_widget,
    umap_hyperParams_widget,
    visual_encoding_channels_widget,
)

st.set_page_config(layout="wide", page_icon="📊")
render_top_menu()

# initialize session_states
if "vis_df" not in st.session_state:
    st.session_state.vis_df = None
if "plot_point_size" not in st.session_state:
    st.session_state.plot_point_size = DEFAULT_POINT_SIZE
if "plot_axis_label_size" not in st.session_state:
    st.session_state.plot_axis_label_size = DEFAULT_AXIS_LABEL_FONT_SIZE
if "plot_legend_size" not in st.session_state:
    st.session_state.plot_legend_size = DEFAULT_LEGEND_FONT_SIZE
if "plot_colormap" not in st.session_state:
    st.session_state.plot_colormap = DEFAULT_COLORMAP
if "plot_show_group_counts" not in st.session_state:
    st.session_state.plot_show_group_counts = False

def _collect_categorical_filters(categorical_cols, df):
    """Read categorical filter selections from session state."""
    filters = {}
    categories_to_filter = [c for c in categorical_cols if c in df.columns and df[c].nunique() > 1]
    for cat in categories_to_filter:
        sel = st.session_state.get(f"{cat}_multiselect", ["All"])
        if "All" not in sel and sel:
            filters[cat] = list(sel)
    return filters

def _collect_numerical_filters():
    """Read numerical filter conditions from session state."""
    num_filters = []
    i = 0
    while True:
        feat = st.session_state.get(f"num_filter_feature_{i}")
        if feat is None or feat == "None":
            break
        op = st.session_state.get(f"num_filter_operator_{i}_{feat}", ">")
        thresh = st.session_state.get(f"num_filter_threshold_{i}_{feat}")
        if thresh is not None:
            num_filters.append((feat, op, float(thresh)))
        if not st.session_state.get(f"add_another_num_filter_{i}", False):
            break
        i += 1
    return num_filters

def _export_script_button(method, uploaded_csv, categorical_cols, color_by, opacity_by, shape_by, separate_by, **extra_params):
    """Render the export-as-script download button with full state collection."""
    # Shared state
    state = {
        "csv_filename": uploaded_csv.name if uploaded_csv else "data.csv",
        "unique_row_id_col": get_unique_row_id_col(st.session_state.get("_use_data_extraction", True)),
        "fov_name_col": get_fov_name_col_analysis(st.session_state.get("_use_data_extraction", True)),
        "method": method,
        "categorical_filters": _collect_categorical_filters(categorical_cols, st.session_state.vis_df) if st.session_state.vis_df is not None else {},
        "numerical_filters": _collect_numerical_filters(),
        "color_by": color_by if color_by else [],
        "opacity_by": opacity_by,
        "shape_by": shape_by,
        "separate_by": separate_by,
        "categorical_cols": list(categorical_cols) if categorical_cols else [],
        "analysis_columns": st.session_state.get("analysis_columns"),
        "point_size": st.session_state.plot_point_size,
        "axis_label_size": st.session_state.plot_axis_label_size,
        "legend_size": st.session_state.plot_legend_size,
        "colormap": st.session_state.plot_colormap,
    }

    # Method-specific params
    mp = {}
    if method == "Feature Comparison":
        sv = extra_params.get("selected_var") or st.session_state.get("_fc_selected_var")
        es_method = extra_params.get("effect_size_method", st.session_state.get("_fc_effect_size", "None"))
        key_suffix = f"_{sv}_{'_'.join(color_by)}_{separate_by or ''}" if sv else ""
        # Capture custom x-axis order from session state
        custom_order = {}
        if st.session_state.vis_df is not None and sv:
            from src.widgets.visualization_widgets import get_visual_group_keys
            try:
                session_key_sep, session_key_cmp = get_visual_group_keys(
                    st.session_state.vis_df, sv, color_by, separate_by
                )
                if session_key_sep in st.session_state:
                    custom_order["separate_groups"] = list(st.session_state[session_key_sep])
                if session_key_cmp in st.session_state:
                    custom_order["compare_groups"] = list(st.session_state[session_key_cmp])
            except Exception:
                pass
        mp = {
            "selected_var": sv,
            "effect_size_method": es_method,
            "mean_or_median": extra_params.get("mean_or_median"),
            "statistical_test": extra_params.get("statistical_test", "None"),
            "log_y": st.session_state.get(f"log_y{key_suffix}", False),
            "add_boxplot": st.session_state.get(f"add_boxplot{key_suffix}", False),
            "connect_means": st.session_state.get(f"connect_means{key_suffix}", False),
            "effect_size_threshold": get_effect_size_threshold_capture(st.session_state, es_method, sv, separate_by),
            # "A vs B" labels from comparison_pair_widget; None → all pairs
            "selected_pairs": list(st.session_state["compare_pairs"]) if "compare_pairs" in st.session_state else None,
            "custom_order": custom_order if custom_order else None,
        }
    elif method == "Feature Histogram":
        sv = extra_params.get("selected_var") or st.session_state.get("_fh_selected_var")
        mp = {
            "selected_var": sv,
            "log_x": st.session_state.get(f"log_x_hist_{sv}", False) if sv else False,
            "apply_gmm": extra_params.get("apply_gmm", False),
            "intersection_threshold": st.session_state.get("intersection_threshold", False),
            "bin_width": (float(st.session_state[f"hist_bin_width_{sv}"])
                          if sv and f"hist_bin_width_{sv}" in st.session_state else None),
            "gmm_max_components": int(st.session_state.get("fit_gmm_max_components", 3)),
            "gmm_min_weight_threshold": float(st.session_state.get("fit_gmm_min_weight_threshold", 0.1)),
        }
    elif method == "FOV Comparison":
        mp = {"selected_var": extra_params.get("selected_var")}
    elif method == "2D Feature Distribution":
        sx = extra_params.get("selected_x")
        sy = extra_params.get("selected_y")
        mp = {
            "selected_x": sx,
            "selected_y": sy,
            "log_x": st.session_state.get(f"log_x_2d_{sx}_{sy}", False) if sx and sy else False,
            "log_y": st.session_state.get(f"log_y_2d_{sx}_{sy}", False) if sx and sy else False,
            "marginal_plot_type": st.session_state.get(f"marginal_plot_type_selector_{sx}_{sy}", "gaussian fit") if sx and sy else "gaussian fit",
            "fit_regression": st.session_state.get(f"fit_regression_2d_{sx}_{sy}", False) if sx and sy else False,
            "fit_gmm_2d": st.session_state.get(f"fit_gmm_2d_{sx}_{sy}", False) if sx and sy else False,
            "gmm_max_components": int(st.session_state.get("fit_gmm_max_components", 3)),
            "gmm_min_weight_threshold": float(st.session_state.get("fit_gmm_min_weight_threshold", 0.1)),
        }
    elif method == "Phasor Plot":
        ch = extra_params.get("selected_channel")
        mp = {
            "selected_channel": ch,
            "phasor_harmonic": extra_params.get("phasor_harmonic"),
            "phasor_f": extra_params.get("phasor_f"),
            "k_means": st.session_state.get(f"k_means_phasor_{ch}", False) if ch else False,
            "k_means_clusters": st.session_state.get(f"k_means_clusters_phasor_{ch}", 2) if ch else 2,
        }
    elif method == "Dimension Reduction":
        mp = {
            "selected_features": extra_params.get("selected_features", []),
            "dr_method": extra_params.get("dr_method", "PCA"),
            "hyperParam_dict": extra_params.get("hyperParam_dict", {}),
        }
    elif method == "Classification":
        # Read classify_by from the classification widget's session state
        classify_by = list(st.session_state.get("classify_by_multiselect", []))
        mp = {
            "selected_features": extra_params.get("selected_features", []),
            "classification_method": extra_params.get("classification_method"),
            "splits": extra_params.get("splits", 0.7),
            "sampling_method": extra_params.get("sampling_method", "None"),
            "class_weight": extra_params.get("class_weight", "None"),
            "threshold_method": extra_params.get("threshold_method", "None"),
            "classifier_params": extra_params.get("classifier_params", {}),
            "classify_by": classify_by,
            "classify_classes": extra_params.get("classify_classes", []),
        }

    state["method_params"] = mp

    @st.fragment
    def _render_export_button(state, method):
        script_text = generate_script(state)
        dataset_name = state["csv_filename"].rsplit(".", 1)[0]
        fname = f"{dataset_name}_{method.lower().replace(' ', '_')}_analysis.py"
        st.download_button(
            label="Export the entire analysis as a Python script",
            data=script_text,
            file_name=fname,
            mime="text/x-python",
            key=f"export_script_{method}",
        )

    _render_export_button(state, method)

multivar_methods = ["Dimension Reduction", "Classification"] #"Align Modalities"]
# methods to visualize based on a single feature
univar_methods = ["Feature Comparison", "Feature Histogram", "FOV Comparison"]
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
    use_data_extraction = st.checkbox("**Use Dataset from Data Extraction**", value=True)
    st.session_state._use_data_extraction = use_data_extraction
    unique_row_id_col = get_unique_row_id_col(use_data_extraction)
    fov_name_col = get_fov_name_col_analysis(use_data_extraction)
    categorical_cols = get_categorical_cols_analysis(use_data_extraction)
    instruction_text = "Upload the CSV file obtained from [Data Extraction](/data_extraction) directly." if use_data_extraction else "**Use the right panel to configure before loading your data ===>**"
    uploaded_csv = st.file_uploader(
        instruction_text,
        type=["csv"],
    )
    try:
        df, feature_groups_dict, upload_complete = load_csv(uploaded_csv, categorical_cols, use_data_extraction=use_data_extraction)
    except Exception as e:
        st.error(f"Failed to process the uploaded CSV: {e} {sad_emoji}")
        df, feature_groups_dict, upload_complete = None, None, False
    st.session_state.vis_df = df
    # Snapshot the column universe get_features() pruned to, before any plot adds
    # derived columns (GMM_group, _color_group, ...). The exported script replays
    # this same prune so its derived CSVs carry the app's columns, not the raw file's.
    st.session_state.analysis_columns = list(df.columns) if df is not None else None

    if upload_complete:
        if method in univar_methods:
            selected_var = single_feature_select_widget(feature_groups_dict, data_extraction=use_data_extraction, n_per_row=2)
            if method == "Feature Comparison":
                ef_col1, ef_col2 = st.columns(2)
                mean_or_median = None
                with ef_col1:
                    selected_effect_size_method = st.radio(
                        "Effect size method",
                        ["None", "Glass's Delta", "Absolute Cohen's d"],
                        index=0,
                    )
                with ef_col2:
                    if selected_effect_size_method != "None":
                        mean_or_median = st.radio("Mean or Median", ["Mean", "Median"])
                statistical_test = st.radio("Statistical Comparison between Two Groups", ["None", "Independent t-test", "Welch's t-test"], index=0)
               
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
                else:
                    hyperParam_dict = {}
            elif method == "Classification":
                cols = st.columns(2)
                with cols[0]:
                    classification_method = st.radio("Classifier", CLASSIFIER_OPTIONS)
                with cols[1]:
                    splits = st.slider("Train size (proportion of training data)", 0.5, 0.9, 0.7, 0.1)
                st.markdown("**Classifier Hyperparameters**")
                classifier_params = classifier_hyperparams_widget(classification_method)
    
with col2:
    if upload_complete:
        # click_ready: boolean to check if the plot is ready for click events
        data_export_ready = False
        filtered_df = filters_widget(st.session_state.vis_df, categorical_cols)
        # for visualization that are point-based, provides the options for other visual encoding channels: opacity, shape, and separate by
        point_based = method not in ["FOV Comparison", "Feature Histogram", "Classification"]
        color_based = method not in [ "Classification"]
        separate_by_available = method in ["Feature Comparison"]
        fig = None
        # check if the df is empty after filtering
        if not filtered_df.empty:
            color_by, opacity_by, shape_by, separate_by = visual_encoding_channels_widget(filtered_df, categorical_cols, color_based=color_based, point_based=point_based, separate_by_available=separate_by_available)
            if method in univar_methods and selected_var != "Select":
                # drop rows with NaN values in the selected_var column
                filtered_df = filtered_df[filtered_df[selected_var].notna()]
                if len(filtered_df) > 0:
                    # Plot the filtered dataframe
                    if method == "Feature Comparison":
                        # Prepare groups for reordering
                        custom_order = {}
                        # We need to access the logic for determining groups to propose them for sorting
                        # Since we can't easily run the internal logic of feature_comparison_plot without calling it,
                        # we can infer the groups from the dataframe directly here.
                        
                        # Logic to determine keys for "separate_by" and "color_by"
                        # We mimic the logic in feature_comparison_plot somewhat or we just pass the raw data
                        # But to save state, we need to know what the groups are.
                        
                        # Define the reordering UI *before* the plot or *after*? 
                        # User said: "below the actual plot, I want to have an interactive setup... confirm button. After confirm the plot is rerendered"
                        # So we render the plot first (with default or current session state order), then show widgets below.
                        
                        # Check session state for existing custom order
                        session_key_sep, session_key_cmp = get_visual_group_keys(filtered_df, selected_var, color_by, separate_by)
                        
                        current_custom_order = {}
                        if session_key_sep in st.session_state:
                            current_custom_order['separate_groups'] = st.session_state[session_key_sep]
                        if session_key_cmp in st.session_state:
                            current_custom_order['compare_groups'] = st.session_state[session_key_cmp]

                        fig = feature_comparison_plot(filtered_df, cell_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_var=selected_var, color_by=color_by, opacity_by=opacity_by, shape_by=shape_by, separate_by=separate_by, colormap=st.session_state.plot_colormap, effect_size_method=selected_effect_size_method, mean_or_median=mean_or_median, statistical_test=statistical_test, custom_order=current_custom_order)
                        

                    elif method == "FOV Comparison":
                        fig = fov_comparison_plot(filtered_df, fov_name_col=fov_name_col, selected_var=selected_var, color_by=color_by, colormap=st.session_state.plot_colormap)
                    elif method == "Feature Histogram":
                        # Log transform and GMM checkboxes on same row
                        col_log, col_gmm = st.columns([0.15, 0.85])
                        with col_log:
                            log_x = st.checkbox("Log X", value=False, key=f"log_x_hist_{selected_var}")
                        with col_gmm:
                            apply_gmm = st.checkbox("Apply Gaussian Mixture Model to the feature distribution", value=False, help="Fit Gaussian Mixture Models\
                            for each color group on the selected feature with 1 to 5 components (fit on raw distribution, not on the histograms). \
                            Choose the one in which all the components are at least of x% weight and has the lowest BIC score. \
                            The default x% is 10%.")
                        
                        # Apply log transform if requested (consistent with bivar.py)
                        if log_x:
                            import numpy as np
                            if (filtered_df[selected_var] < 0).any():
                                st.error(log_negative_error(selected_var))
                            else:
                                filtered_df = filtered_df.copy()
                                filtered_df[selected_var] = np.log10(filtered_df[selected_var] + 1e-6)
                        if apply_gmm:
                            fig, gmm_df = feature_gmm_plot(filtered_df, selected_var, color_by, colormap=st.session_state.plot_colormap, log_x=log_x)
                            data_export_ready = True
                        else: 
                            fig = feature_histogram_plot(filtered_df, selected_var, color_by, colormap=st.session_state.plot_colormap, log_x=log_x)    
                else:
                    st.write(f"No data available after removing rows with missing values {sad_emoji}")
            elif method in bivar_methods:
                if "2D" in method and selected_x != "Select" and selected_y != "Select":
                    # drop rows with NaN values in the selected_x and selected_y columns
                    filtered_df = filtered_df[filtered_df[selected_x].notna() & filtered_df[selected_y].notna()]
                    if len(filtered_df) > 0:
                        fig, table_md, gmm_df = feature_2d_distribution_plot(filtered_df, unique_row_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_x=selected_x, selected_y=selected_y, color_by=color_by, shape_by=shape_by, opacity_by=opacity_by, colormap=st.session_state.plot_colormap)
                        data_export_ready = True
                    else:
                        st.write(f"No data available after removing rows with missing values {sad_emoji}")
                elif method == "Phasor Plot":
                    if selected_channel is not None and selected_harmonic is not None and f is not None:
                        feature_prefix = "Lifetime fit free_" + selected_channel + ": "
                        g_col = f"{feature_prefix}G(1st)" if selected_harmonic == 1 else f"{feature_prefix}G(2nd)"
                        s_col = f"{feature_prefix}S(1st)" if selected_harmonic == 1 else f"{feature_prefix}S(2nd)"
                        if g_col not in filtered_df.columns or s_col not in filtered_df.columns:
                            st.error(f"Required phasor columns ({g_col}, {s_col}) not found in your data. {sad_emoji}")
                        else:
                            fig, kmeans_df = phasor_plot(filtered_df, unique_row_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_channel=selected_channel, color_by=color_by, shape_by=shape_by, opacity_by=opacity_by, colormap=st.session_state.plot_colormap, f=f, harmonic=selected_harmonic)
                            data_export_ready = True
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
                            try:
                                fig = dimension_reduction_plot(filtered_df, unique_row_id_col=unique_row_id_col, fov_name_col=fov_name_col, selected_features=selected_features, colored_by=color_by, opacity_by=opacity_by, shape_by=shape_by, colormap=st.session_state.plot_colormap, method=dr_method, hyperParam_dict=hyperParam_dict)
                            except Exception as e:
                                st.error(f"Dimension reduction failed: {e}. Check that selected features don't contain constant or all-NaN columns. {sad_emoji}")
                                fig = None
                        else:
                            st.write(f"No data available after removing rows with missing values {sad_emoji}")
                elif method == "Classification":
                    error_msg, df_classify, sampling_method, apply_class_weight, threshold_method = classifier_options_widget(
                        filtered_df,
                        categorical_cols,
                        fov_name_col=fov_name_col,
                        selected_features=selected_features,
                        classifier=classification_method,
                        splits=splits,
                    )
                    if error_msg:
                        st.error(error_msg)
                    else:
                        error_msg, results = run_classification(
                            df_classify,
                            classification_method,
                            splits,
                            sampling_method,
                            apply_class_weight,
                            threshold_method,
                            classifier_params=classifier_params,
                            random_state=42,
                        )
                        if error_msg:
                            st.error(f"{error_msg} {sad_emoji}")
                        else:
                            classification_plot_widget(results, classification_method, threshold_method)
                            _export_script_button(method, uploaded_csv, categorical_cols, color_by, opacity_by, shape_by, separate_by,
                                                  classification_method=classification_method, splits=splits,
                                                  sampling_method=sampling_method, class_weight=apply_class_weight,
                                                  threshold_method=threshold_method, classifier_params=classifier_params,
                                                  selected_features=selected_features,
                                                  classify_classes=df_classify['classes'].unique().tolist())

            if fig is not None: 
                fig = apply_plot_styling(fig, st.session_state.plot_point_size, st.session_state.plot_axis_label_size, st.session_state.plot_legend_size) 
                if method == "2D Feature Distribution":
                    col2_1, col2_2 = st.columns([1, 1])
                    with col2_1:
                       st.plotly_chart(fig, width='stretch')
                    with col2_2:
                        if table_md != []:
                            st.markdown(table_md, unsafe_allow_html=True)
                else:
                    st.plotly_chart(fig, width='stretch')
                # 1. Data export (if applicable)
                if data_export_ready:
                    # available for download
                    if method == "2D Feature Distribution" and "2D_GMM_group" in gmm_df.columns:
                        st.download_button(label="Download 2D GMM data", data=gmm_df.to_csv(index=False), file_name="2D_gmm_data.csv")
                    elif method == "Feature Histogram" and "GMM_group" in gmm_df.columns:
                        st.download_button(label="Download GMM Grouped Data", data=gmm_df.to_csv(index=False), file_name="gmm_grouped_data.csv", mime="text/csv", key="gmm_download")
                    elif method == "Phasor Plot" and "k_means_cluster" in kmeans_df.columns:
                        st.download_button(label="Download K-Means Clustered Data", data=kmeans_df.to_csv(index=False), file_name="kmeans_clustered_data.csv", mime="text/csv", key="kmeans_download")
                if method == "Feature Comparison":
                    # Widgets for reordering below
                    reorder_x_axis_widget(filtered_df, selected_var, color_by, separate_by)

                # 2. Plot configuration widget at the bottom - allows users to adjust styling after seeing plots
                # Widgets use key= to write directly to session state; Streamlit reruns naturally on change
                st.subheader("📊 Plot Styling")
                show_colormap = len(color_by) > 0
                plot_config_widget(point_based=point_based, show_colormap=show_colormap,
                                   show_count_toggle=show_colormap)

                # 3. Export as Python Script
                _extra = {}
                if method in univar_methods:
                    _extra["selected_var"] = selected_var
                    if method == "Feature Comparison":
                        _extra["effect_size_method"] = selected_effect_size_method
                        _extra["mean_or_median"] = mean_or_median
                        _extra["statistical_test"] = statistical_test
                    elif method == "Feature Histogram":
                        try:
                            _extra["apply_gmm"] = apply_gmm
                        except NameError:
                            _extra["apply_gmm"] = False
                elif method in bivar_methods:
                    if "2D" in method:
                        _extra["selected_x"] = selected_x
                        _extra["selected_y"] = selected_y
                    elif method == "Phasor Plot":
                        _extra["selected_channel"] = selected_channel
                        _extra["phasor_harmonic"] = selected_harmonic
                        _extra["phasor_f"] = f
                elif method == "Dimension Reduction":
                    _extra["selected_features"] = selected_features
                    _extra["dr_method"] = dr_method
                    _extra["hyperParam_dict"] = hyperParam_dict
                _export_script_button(method, uploaded_csv, categorical_cols, color_by, opacity_by, shape_by, separate_by, **_extra)

        else:
            st.markdown(f"<h5 style='text-align: center; color: red'>No data available after filtering {sad_emoji}</h5>", unsafe_allow_html=True)

    else:
        dataset_config_widget(use_data_extraction=use_data_extraction)
