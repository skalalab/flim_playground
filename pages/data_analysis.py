import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.classify import run_classification
from src.collapse import collapse_rows
from src.export_labels import EXPORT_METHODS, apply_export_labels, available_label_column
from src.dataset_io import (
    SUPPORTED_SUFFIXES,
    _render_reject,
    _render_warning,
    interpret_table,
    load_table,
    read_table,
    resolve_effective_fov_col,
)
from src.emojis import happy_emoji, sad_emoji
from src.export_script import generate_script, get_effect_size_threshold_capture
from src.navigation import render_top_menu
from src.vis.bivar import (
    distribution_controls, feature_2d_distribution_plot, phasor_plot,
    select_distribution_category, select_phasor_category,
)
from src.widgets.plot_layout import dimension_reduction_chart, phasor_chart, square_2d_plot
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
    render_histogram_summaries,
)
from src.widgets.analysis_config_widgets import (
    get_categorical_cols_analysis,
    get_fov_name_col_analysis,
    get_unique_row_id_col,
    working_copy_arguments,
)
from src.widgets.classification_widgets import (
    CLASSIFIER_OPTIONS,
    classification_plot_widget,
    classifier_hyperparams_widget,
    classifier_options_widget,
)
from src.widgets.encoding_state import (
    drop_varying_channels,
    dropped_channel_note,
)
from src.widgets.filter_widgets import filters_widget, selection_key
from src.widgets.export_labels_widgets import export_labels_widget
from src.widgets.analysis_widget_state import control_default, derived_fit_control_keys, preserve_analysis_controls
from src.widgets.multiselect_modes import ALL_LABEL, chosen_items
from src.widgets.review_table_widget import (
    applied_summary,
    configured_row_id,
    ignored_columns,
    review_gate,
)
from src.widgets.selection_widgets import (
    multi_feature_select_widget,
    single_feature_select_widget,
    twod_single_feature_select_widget,
)
from src.widgets.visualization_widgets import (
    HISTOGRAM_BIN_WIDTH_PREFIX,
    SEPARATION_KEYS,
    _compute_channel_harmonics,
    distribution_category_widget,
    get_visual_group_keys,
    histogram_bin_width_key,
    phasor_category_widget,
    phasor_params_widget,
    plot_config_widget,
    reorder_x_axis_widget,
    tsne_hyperParams_widget,
    umap_hyperParams_widget,
    visual_encoding_channels_widget,
)

st.set_page_config(layout="wide", page_icon="📊")
render_top_menu()

# Preserve fit inputs as well as layout when a method hides their widgets.
# Otherwise a method switch can reset the fit and discard its saved export names.
preserve_analysis_controls(st.session_state, (*SEPARATION_KEYS, *derived_fit_control_keys(st.session_state)))
# GMM and other methods hide Bin Width; keep each feature's raw/log choices.
preserve_analysis_controls(st.session_state, tuple(
    key for key in st.session_state if key.startswith(HISTOGRAM_BIN_WIDTH_PREFIX)))

# Keep shared styling even when a module has no plot or styling widgets yet.
if "vis_df" not in st.session_state:
    st.session_state.vis_df = None
for state_key, default in (("plot_point_size", DEFAULT_POINT_SIZE),
                           ("plot_axis_label_size", DEFAULT_AXIS_LABEL_FONT_SIZE),
                           ("plot_legend_size", DEFAULT_LEGEND_FONT_SIZE),
                           ("plot_colormap", DEFAULT_COLORMAP),
                           ("plot_show_group_counts", False)):
    st.session_state[state_key] = st.session_state.get(state_key, default)

def _collect_categorical_filters(categorical_cols, df):
    """Read categorical filter selections from session state.

    Expand exclusion selections with the widget's chosen_items helper. None means
    unrestricted; an empty list exports as isin([]), keeping no rows.
    """
    filters = {}
    categories_to_filter = [c for c in categorical_cols if c in df.columns and df[c].nunique() > 1]
    for cat in categories_to_filter:
        sel = st.session_state.get(selection_key(cat), [ALL_LABEL])
        chosen = chosen_items(sel, df[cat].unique().tolist())
        if chosen is not None:
            filters[cat] = list(chosen)
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

def _distribution_options(selected_x, selected_y):
    """Snapshot settings that change FD computations, excluding display category."""
    suffix = f"{selected_x}_{selected_y}"
    return dict(
        log_x=st.session_state.get(f"log_x_2d_{suffix}", False),
        log_y=st.session_state.get(f"log_y_2d_{suffix}", False),
        marginal_plot_type=st.session_state.get(f"marginal_plot_type_selector_{suffix}", "gaussian fit"),
        fit_regression=st.session_state.get(f"fit_regression_2d_{suffix}", False),
        fit_gmm=st.session_state.get(f"fit_gmm_2d_{suffix}", False),
        max_components=int(st.session_state.get("fit_gmm_max_components", 3)),
        min_weight_threshold=float(st.session_state.get("fit_gmm_min_weight_threshold", .1)),
    )


def _export_script_button(method, uploaded_file, categorical_cols, color_by, opacity_by, shape_by, separate_by, subcolor_by, delimiter=",", **extra_params):
    """Render the export-as-script download button with full state collection."""
    # Shared state
    state = {
        "csv_filename": uploaded_file.name if uploaded_file else "data.csv",
        # Reuse the detected separator without reading the upload again.
        "delimiter": delimiter,
        # Export the applied review's configured ID. A blank name lets the script
        # generate the same row IDs instead of requiring a generated column in the file.
        "unique_row_id_col": configured_row_id()
        if not st.session_state.get("_use_data_extraction", True)
        else get_unique_row_id_col(True),
        # Include the configured FOV column only when it exists in the data.
        "fov_name_col": st.session_state.get("effective_fov_name_col"),
        "method": method,
        "categorical_filters": _collect_categorical_filters(categorical_cols, st.session_state.vis_df) if st.session_state.vis_df is not None else {},
        "numerical_filters": _collect_numerical_filters(),
        "color_by": color_by if color_by else [],
        "opacity_by": opacity_by,
        "shape_by": shape_by,
        "separate_by": separate_by,
        "subcolor_by": subcolor_by,
        "categorical_cols": list(categorical_cols) if categorical_cols else [],
        # Match the app's coercion skip set and removal of ignored columns.
        "ignored_cols": ignored_columns()
        if not st.session_state.get("_use_data_extraction", True)
        else [],
        "analysis_columns": st.session_state.get("analysis_columns"),
        "point_size": st.session_state.plot_point_size,
        "axis_label_size": st.session_state.plot_axis_label_size,
        "legend_size": st.session_state.plot_legend_size,
        "colormap": st.session_state.plot_colormap,
        "show_group_counts": st.session_state.get("plot_show_group_counts", False),
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
        overlay = st.session_state.get(
            f"comparison_overlay{key_suffix}",
            "Boxplot" if st.session_state.get(f"add_boxplot{key_suffix}", False) else "None")
        mp = {
            "selected_var": sv,
            "effect_size_method": es_method,
            "mean_or_median": extra_params.get("mean_or_median"),
            "statistical_test": extra_params.get("statistical_test", "None"),
            "log_y": st.session_state.get(f"log_y{key_suffix}", False),
            "overlay": overlay,
            "add_boxplot": overlay == "Boxplot",
            "connect_means": st.session_state.get(f"connect_means{key_suffix}", False),
            "effect_size_threshold": get_effect_size_threshold_capture(st.session_state, es_method, sv, separate_by),
            # "A vs B" labels from comparison_pair_widget; None → all pairs
            "selected_pairs": list(st.session_state["compare_pairs"]) if "compare_pairs" in st.session_state else None,
            "custom_order": custom_order if custom_order else None,
            "collapse_by": extra_params.get("collapse_by"),
        }
    elif method == "Feature Histogram":
        sv = extra_params.get("selected_var") or st.session_state.get("_fh_selected_var")
        histogram_log_x = extra_params.get(
            "log_x", st.session_state.get(f"log_x_hist_{sv}", False) if sv else False)
        width_key = histogram_bin_width_key(sv, histogram_log_x) if sv else None
        bin_width = st.session_state.get(width_key) if width_key else None
        mp = {
            "selected_var": sv,
            "log_x": histogram_log_x,
            "apply_gmm": extra_params.get("apply_gmm", False),
            "intersection_threshold": st.session_state.get("intersection_threshold", False),
            "bin_width": float(bin_width) if bin_width is not None else None,
            "gmm_max_components": int(st.session_state.get("fit_gmm_max_components", 3)),
            "gmm_min_weight_threshold": float(st.session_state.get("fit_gmm_min_weight_threshold", 0.1)),
        }
    elif method == "2D Feature Distribution":
        sx = extra_params.get("selected_x")
        sy = extra_params.get("selected_y")
        mp = {
            "selected_x": sx,
            "selected_y": sy,
            "collapse_by": extra_params.get("collapse_by"),
            "distribution_category": extra_params.get("distribution_category"),
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
            "phasor_category": extra_params.get("phasor_category"),
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
    state["derived_export"] = extra_params.get("derived_export")
    export_names_valid = extra_params.get("export_names_valid", True)

    @st.fragment
    def _render_export_button(state, method):
        script_text = generate_script(state) if export_names_valid else ""
        dataset_name = state["csv_filename"].rsplit(".", 1)[0]
        fname = f"{dataset_name}_{method.lower().replace(' ', '_')}_analysis.py"
        st.download_button(
            label="Export the entire analysis as a Python script",
            data=script_text,
            file_name=fname,
            mime="text/x-python",
            key=f"export_script_{method}",
            disabled=not export_names_valid,
        )

    _render_export_button(state, method)

multivar_methods = ["Dimension Reduction", "Classification"]
# Show Phasor Plot until a loaded frame rules it out. Loading below updates this
# availability and reruns if the method list needs to change.
_frame_loaded = st.session_state.get("vis_df") is not None
_show_phasor = bool(st.session_state.get("phasor_available")) or not _frame_loaded
univar_methods = ["Feature Comparison", "Feature Histogram"]
bivar_methods = ["2D Feature Distribution"]
if _show_phasor:
    bivar_methods.append("Phasor Plot")
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
    # Loading resolves a blank configured ID to a generated row-number column.
    configured_row_id_col = get_unique_row_id_col(use_data_extraction)
    unique_row_id_col = configured_row_id_col
    # Hover labels use the user's column name, or "ID" for generated row numbers.
    row_id_label = "Cell ID" if use_data_extraction else (configured_row_id_col or "ID")
    categorical_cols = get_categorical_cols_analysis(use_data_extraction)
    instruction_text = ("Upload the file obtained from [Data Extraction](/data_extraction) directly."
                        if use_data_extraction else "Upload your table")
    uploaded_file = st.file_uploader(
        instruction_text,
        label_visibility="visible" if use_data_extraction else "collapsed",
        # Kept in sync with SUPPORTED_SUFFIXES and dataset_io._diagnose_table.
        help="CSV, tab-, semicolon- or pipe-separated text (.tsv, .txt), Excel (.xlsx, .xlsm) "
             "or OpenDocument (.ods). The table must be a plain grid: column names on the first "
             "row, one row per data point, and — in a spreadsheet — on the first sheet.",
        type=[suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES],
        # Keep the upload when switching between extraction data and user tables.
        key="analysis_file_upload",
    )
    decision = None
    # Only extraction data has a designated FOV column for hover text.
    configured_fov_col = get_fov_name_col_analysis(use_data_extraction)
    try:
        if use_data_extraction:
            df, feature_groups_dict, upload_complete, delimiter, unique_row_id_col = load_table(
                uploaded_file, categorical_cols)
        else:
            # Review the raw headers and dtypes before interpreting column roles.
            df, feature_groups_dict, upload_complete, delimiter = None, None, False, ","
            if uploaded_file is not None:
                raw, _read_meta, delimiter, scope_warning, error_msg = read_table(uploaded_file)
                if error_msg != "":
                    _render_reject(error_msg, scope_warning)
                else:
                    # Show sheet-scope warnings before review, including when Save is blocked.
                    _render_warning(scope_warning)
                    # Review uses the wide column; loader messages stay beside the upload.
                    with col2:
                        decision = review_gate(uploaded_file, raw)
                    if decision is None:
                        # The review screen owns this run. Clear analysis/export data,
                        # then stop before rendering any analysis controls or plots.
                        st.session_state.vis_df = None
                        st.session_state.analysis_columns = None
                        st.stop()
                    if decision is not None:
                        args = working_copy_arguments(
                            decision["roles"], decision["groups"], decision["group_names"])
                        categorical_cols = args["categorical_cols"]
                        row_id_label = args["unique_row_id_col"] or "ID"
                        df, feature_groups_dict, upload_complete, unique_row_id_col = interpret_table(
                            # configured_fov_col is "" on this branch: a field-of-view
                            # column here is an ordinary categorical, named by no role.
                            raw, categorical_cols, args["unique_row_id_col"], configured_fov_col,
                            ignored_cols=args["ignored_cols"], feature_groups=args["feature_groups"],
                            use_data_extraction=False)
                        # Identify the applied profile and allow reopening review after a reject.
                        applied_summary(decision)
    except Exception as e:
        st.error(f"Failed to process the uploaded file: {e} {sad_emoji}")
        df, feature_groups_dict, upload_complete, delimiter = None, None, False, ","
    st.session_state.vis_df = df
    # Capture the analysis columns before plotting adds derived columns, for export parity.
    st.session_state.analysis_columns = list(df.columns) if df is not None else None
    # Resolve hover metadata and phasor availability from the current frame.
    fov_name_col = resolve_effective_fov_col(df, configured_fov_col)
    st.session_state.effective_fov_name_col = fov_name_col
    _channel_harmonics = _compute_channel_harmonics(feature_groups_dict) if feature_groups_dict else {}
    st.session_state.phasor_available = any(
        harmonics for harmonics in _channel_harmonics.values())
    _phasor_now = st.session_state.phasor_available or df is None
    if _phasor_now != _show_phasor:
        st.rerun()

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
                        key="analysis_control_effect_size",
                    )
                with ef_col2:
                    if selected_effect_size_method != "None":
                        mean_or_median = st.radio("Mean or Median", ["Mean", "Median"], key="analysis_control_mean_or_median")
                statistical_test = st.radio("Statistical Comparison between Two Groups", ["None", "Independent t-test", "Welch's t-test"], index=0, key="analysis_control_statistical_test")

        elif method in bivar_methods:
            if "2D" in method:
                selected_x, selected_y = twod_single_feature_select_widget(feature_groups_dict, data_extraction=use_data_extraction, n_per_row=2)
            elif method == "Phasor Plot":
                selected_channel, selected_harmonic, f = phasor_params_widget(feature_groups_dict)
        elif method in multivar_methods:
            selected_features = multi_feature_select_widget(feature_groups_dict, data_extraction=use_data_extraction, n_per_row=2)
            if method == "Dimension Reduction":                
                dr_method = st.radio("Dimension Reduction Method", ["UMAP", "PCA", "t-SNE"], horizontal=True, key="analysis_control_dr_method")
                if dr_method == "UMAP":
                    hyperParam_dict = umap_hyperParams_widget()
                elif dr_method == "t-SNE":
                    hyperParam_dict = tsne_hyperParams_widget()
                else:
                    hyperParam_dict = {}
            elif method == "Classification":
                cols = st.columns(2)
                with cols[0]:
                    classification_method = st.radio("Classifier", CLASSIFIER_OPTIONS, key="analysis_control_classifier")
                with cols[1]:
                    splits = st.slider("Train size (proportion of training data)", 0.5, 0.9, control_default(st.session_state, "analysis_control_train_size", 0.7), 0.1, key="analysis_control_train_size")
                st.markdown("**Classifier Hyperparameters**")
                classifier_params = classifier_hyperparams_widget(classification_method)

with col2:
    if upload_complete:
        data_export_ready = False
        filtered_df = filters_widget(st.session_state.vis_df, categorical_cols)
        # Offer encoding controls supported by the selected plot.
        point_based = method not in ["Feature Histogram", "Classification"]
        color_based = method not in [ "Classification"]
        separate_by_available = method in ["Feature Comparison", "Feature Histogram", "Dimension Reduction", "Phasor Plot", "2D Feature Distribution"]
        # Feature Comparison preserves group identity on the x axis when subcolor is used.
        subcolor_available = method in ["Feature Comparison"]
        # Collapse averages replicate measurements within each color group.
        collapse_available = method in ["Feature Comparison", "2D Feature Distribution"]
        fig = None
        # check if the df is empty after filtering
        if not filtered_df.empty:
            color_by, opacity_by, shape_by, separate_by, subcolor_by, collapse_by = visual_encoding_channels_widget(
                filtered_df, categorical_cols, color_based=color_based,
                point_based=point_based, separate_by_available=separate_by_available,
                subcolor_available=subcolor_available, collapse_available=collapse_available,
                separate_by_mode={"Dimension Reduction": "facets", "Phasor Plot": "subplots",
                                  "2D Feature Distribution": "distribution",
                                  "Feature Histogram": "histogram"}.get(method, "sections"),
            )
            # Keep any notices about channels dropped by collapse beside their controls.
            collapse_note = st.container()
            # Category selector and analysis controls precede the full-size plot.
            bivar_display_area = st.container() if method in bivar_methods else None

            # Use complete observations for means, counts, and every downstream model.
            selected_columns = []
            if method in univar_methods and selected_var != "Select":
                selected_columns = [selected_var]
            elif method == "2D Feature Distribution" and selected_x != "Select" and selected_y != "Select":
                selected_columns = [selected_x, selected_y]
            if selected_columns:
                filtered_df = filtered_df.dropna(subset=selected_columns)

            plot_df = filtered_df
            plot_row_id_col, plot_row_id_label = unique_row_id_col, row_id_label
            plot_fov_name_col = fov_name_col
            if collapse_by and selected_columns and not plot_df.empty:
                plot_df, plot_row_id_col, _varied = collapse_rows(
                    filtered_df, collapse_by, [*color_by, separate_by],
                    unique_row_id_col)
                plot_row_id_label = collapse_by
                # Collapse drops metadata that varies within a replicate group.
                plot_fov_name_col = resolve_effective_fov_col(plot_df, fov_name_col)
                _channels, _dropped = drop_varying_channels(
                    {"shape": shape_by, "opacity": opacity_by,
                     "subcolor": subcolor_by}, _varied)
                shape_by = _channels["shape"]
                opacity_by = _channels["opacity"]
                subcolor_by = _channels["subcolor"]
                with collapse_note:
                    for _role, _col in _dropped.items():
                        st.caption(dropped_channel_note(_role, collapse_by, _col))

            # Choose the result column after collapse and before fitting so a
            # previous annotation remains available as an input and in the CSV.
            label_column = (available_label_column(plot_df.columns, EXPORT_METHODS[method][0])
                            if method in EXPORT_METHODS else None)

            if method in univar_methods and selected_var != "Select":
                if len(filtered_df) > 0:
                    # Plot the filtered dataframe
                    if method == "Feature Comparison":
                        # Share saved group-order keys with the controls below the plot.
                        session_key_sep, session_key_cmp = get_visual_group_keys(plot_df, selected_var, color_by, separate_by)

                        current_custom_order = {}
                        if session_key_sep in st.session_state:
                            current_custom_order['separate_groups'] = st.session_state[session_key_sep]
                        if session_key_cmp in st.session_state:
                            current_custom_order['compare_groups'] = st.session_state[session_key_cmp]

                        fig = feature_comparison_plot(
                            plot_df, unique_row_id_col=plot_row_id_col,
                            fov_name_col=plot_fov_name_col, selected_var=selected_var,
                            color_by=color_by, opacity_by=opacity_by, shape_by=shape_by,
                            separate_by=separate_by, colormap=st.session_state.plot_colormap,
                            effect_size_method=selected_effect_size_method,
                            mean_or_median=mean_or_median, statistical_test=statistical_test,
                            custom_order=current_custom_order, subcolor_by=subcolor_by,
                            row_id_label=plot_row_id_label, collapse_by=collapse_by,
                            source_df=filtered_df, source_row_id_col=unique_row_id_col,
                            source_row_id_label=row_id_label, source_fov_name_col=fov_name_col)

                    elif method == "Feature Histogram":
                        # Log transform and GMM checkboxes on same row
                        col_log, col_gmm = st.columns([0.15, 0.85])
                        with col_log:
                            log_x = st.checkbox("Log X", value=False, key=f"log_x_hist_{selected_var}")
                        with col_gmm:
                            apply_gmm = st.checkbox("Apply Gaussian Mixture Model to the feature distribution", value=False, key="analysis_control_apply_gmm", help="Fit Gaussian Mixture Models\
                            for each separation category and color group on the selected feature with 1 to 5 components (fit on raw distribution, not on the histograms). \
                            Choose the one in which all the components are at least of x% weight and has the lowest BIC score. \
                            The default x% is 10%.")

                        # Apply log transform if requested (consistent with bivar.py)
                        if log_x:
                            import numpy as np
                            if (plot_df[selected_var] < 0).any():
                                st.error(log_negative_error(selected_var))
                                log_x = False
                            else:
                                plot_df = plot_df.copy()
                                plot_df[selected_var] = np.log10(plot_df[selected_var] + 1e-6)
                        if apply_gmm:
                            fig, gmm_df = feature_gmm_plot(plot_df, selected_var, color_by, colormap=st.session_state.plot_colormap, log_x=log_x, separate_by=separate_by, label_column=label_column)
                            data_export_ready = True
                        else:
                            fig = feature_histogram_plot(plot_df, selected_var, color_by, colormap=st.session_state.plot_colormap, log_x=log_x, separate_by=separate_by)
                else:
                    st.write(f"No data available after removing rows with missing values {sad_emoji}")
            elif method in bivar_methods:
                if "2D" in method and selected_x != "Select" and selected_y != "Select":
                    if len(filtered_df) > 0:
                        fig, table_md, gmm_df = feature_2d_distribution_plot(
                            plot_df, unique_row_id_col=plot_row_id_col, fov_name_col=plot_fov_name_col,
                            selected_x=selected_x, selected_y=selected_y, color_by=color_by,
                            shape_by=shape_by, opacity_by=opacity_by, colormap=st.session_state.plot_colormap,
                            row_id_label=plot_row_id_label, separate_by=separate_by,
                            analysis_options=_distribution_options(selected_x, selected_y),
                            label_column=label_column)
                        data_export_ready = _distribution_options(selected_x, selected_y)["fit_gmm"]
                    else:
                        st.write(f"No data available after removing rows with missing values {sad_emoji}")
                elif method == "Phasor Plot":
                    if selected_channel is not None and selected_harmonic is not None and f is not None:
                        feature_prefix = "Lifetime fit free_" + selected_channel + ": "
                        g_col = f"{feature_prefix}G(1st)" if selected_harmonic == 1 else f"{feature_prefix}G(2nd)"
                        s_col = f"{feature_prefix}S(1st)" if selected_harmonic == 1 else f"{feature_prefix}S(2nd)"
                        if g_col not in filtered_df.columns or s_col not in filtered_df.columns:
                            st.error(f"Required phasor columns ({g_col}, {s_col}) not found in your data. {sad_emoji}")
                        elif filtered_df[[g_col, s_col]].dropna().empty:
                            st.info("No complete G/S observations remain after filtering.")
                        else:
                            fig, _ = phasor_plot(
                                filtered_df, unique_row_id_col=unique_row_id_col,
                                fov_name_col=fov_name_col, selected_channel=selected_channel,
                                color_by=color_by, shape_by=shape_by, opacity_by=opacity_by,
                                colormap=st.session_state.plot_colormap, f=f,
                                harmonic=selected_harmonic, row_id_label=row_id_label,
                                separate_by=separate_by)
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
                                fig = dimension_reduction_plot(
                                    filtered_df, unique_row_id_col=unique_row_id_col,
                                    fov_name_col=fov_name_col, selected_features=selected_features,
                                    colored_by=color_by, opacity_by=opacity_by, shape_by=shape_by,
                                    colormap=st.session_state.plot_colormap, method=dr_method,
                                    hyperParam_dict=hyperParam_dict, row_id_label=row_id_label,
                                    separate_by=separate_by,
                                )
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
                            @st.fragment
                            def _render_classification_results(results):
                                # Reuse the fitted model for styling changes and
                                # keep the exported script's styling in sync.
                                classification_plot_widget(results, classification_method, threshold_method)
                                _export_script_button(method, uploaded_file, categorical_cols, color_by, opacity_by, shape_by, separate_by, subcolor_by,
                                                      delimiter=delimiter,
                                                      classification_method=classification_method, splits=splits,
                                                      sampling_method=sampling_method, class_weight=apply_class_weight,
                                                      threshold_method=threshold_method, classifier_params=classifier_params,
                                                      selected_features=selected_features,
                                                      classify_classes=df_classify['classes'].unique().tolist())

                            _render_classification_results(results)

            if fig is not None:
                if data_export_ready:
                    export_frame = gmm_df
                    fit_settings = dict(
                        max_components=st.session_state.get("fit_gmm_max_components", 3),
                        min_weight=st.session_state.get("fit_gmm_min_weight_threshold", .1))
                    if method == "Feature Histogram":
                        fit_settings.update(log_x=log_x, intersection=st.session_state.get("intersection_threshold", False))
                    else:
                        options = _distribution_options(selected_x, selected_y)
                        # Negative values make 2D skip the requested log transform.
                        # Describe the scale actually used for these means.
                        fit_settings.update(
                            log_x=options["log_x"] and not (plot_df[selected_x] < 0).any(),
                            log_y=options["log_y"] and not (plot_df[selected_y] < 0).any())
                    export_context = dict(
                        dataset=(getattr(uploaded_file, "name", None), getattr(uploaded_file, "file_id", None)),
                        features=selected_columns, color_by=color_by, separate_by=separate_by,
                        collapse_by=collapse_by, fit=fit_settings,
                        categorical_filters=_collect_categorical_filters(categorical_cols, st.session_state.vis_df),
                        numerical_filters=_collect_numerical_filters())

                # Colormap, group counts and section-header sizes require rebuilding the
                # figure. Point and legend sizes can be restyled within the fragment.
                def _plot_build_params():
                    return (
                        st.session_state.plot_colormap,
                        st.session_state.get("plot_show_group_counts", False),
                        st.session_state.plot_axis_label_size,
                        _distribution_options(selected_x, selected_y)
                        if method == "2D Feature Distribution" else None,
                    )

                _build_params = _plot_build_params()

                @st.fragment
                def _render_plot_and_controls(base_fig, build_params):
                    # Styling mutates traces and adds legend entries; copy the base figure
                    # so fragment reruns do not accumulate those changes.
                    fig = go.Figure(base_fig)
                    phasor_category = None
                    distribution_category = None
                    display_table = table_md if method == "2D Feature Distribution" else None
                    if method == "2D Feature Distribution":
                        if separate_by:
                            distribution_category = distribution_category_widget(
                                fig.layout.meta["distribution_categories"], separate_by)
                            select_distribution_category(fig, distribution_category)
                            display_table = fig.layout.meta["distribution_summary"]
                        distribution_controls(selected_x, selected_y)
                    if method == "Phasor Plot" and separate_by:
                        phasor_category = phasor_category_widget(
                            fig.layout.meta["phasor_categories"], separate_by)
                        select_phasor_category(fig, phasor_category)
                    fig = apply_plot_styling(fig, st.session_state.plot_point_size, st.session_state.plot_axis_label_size, st.session_state.plot_legend_size)
                    if method == "2D Feature Distribution":
                        square_2d_plot(fig, key=f"plot_chart_2d_{method}")
                        if display_table:
                            st.markdown(display_table, unsafe_allow_html=True)
                    elif method == "Dimension Reduction":
                        dimension_reduction_chart(fig, key=f"plot_chart_{method}")
                    elif method == "Phasor Plot":
                        phasor_chart(fig, key=f"plot_chart_{method}")
                    else:
                        st.plotly_chart(fig, width='stretch', key=f"plot_chart_{method}")
                    if method == "Feature Histogram":
                        render_histogram_summaries(fig)
                    # 1. Data export (if applicable)
                    derived_export, export_names_valid = None, True
                    if data_export_ready:
                        derived_export, export_names_valid = export_labels_widget(
                            export_frame, label_column, method=method, context=export_context)
                        if label_column in export_frame and export_frame[label_column].notna().any():
                            _, filename, download_label = EXPORT_METHODS[method]
                            csv_data = (apply_export_labels(export_frame, label_column, derived_export).to_csv(index=False)
                                        if export_names_valid else "")
                            st.download_button(label=download_label, data=csv_data, file_name=filename,
                                               mime="text/csv", key=f"derived_data_{method}",
                                               disabled=not export_names_valid)
                    if method == "Feature Comparison":
                        # Widgets for reordering below
                        reorder_x_axis_widget(filtered_df, selected_var, color_by, separate_by)

                    # 2. Plot styling. Widgets write to session state via key=, so
                    # Streamlit reruns on change.
                    st.subheader("📊 Plot Styling")
                    show_colormap = len(color_by) > 0
                    plot_config_widget(point_based=point_based, show_colormap=show_colormap,
                                       show_count_toggle=show_colormap or method == "Feature Histogram" or (
                                           method in ("Dimension Reduction", "Phasor Plot", "2D Feature Distribution") and bool(separate_by)))

                    # 3. Export as Python Script
                    _extra = {"derived_export": derived_export, "export_names_valid": export_names_valid}
                    if method in univar_methods:
                        _extra["selected_var"] = selected_var
                        if method == "Feature Comparison":
                            _extra["effect_size_method"] = selected_effect_size_method
                            _extra["mean_or_median"] = mean_or_median
                            _extra["statistical_test"] = statistical_test
                            _extra["collapse_by"] = collapse_by
                        elif method == "Feature Histogram":
                            _extra["log_x"] = log_x
                            try:
                                _extra["apply_gmm"] = apply_gmm
                            except NameError:
                                _extra["apply_gmm"] = False
                    elif method in bivar_methods:
                        if "2D" in method:
                            _extra["selected_x"] = selected_x
                            _extra["selected_y"] = selected_y
                            _extra["collapse_by"] = collapse_by
                            _extra["distribution_category"] = distribution_category
                        elif method == "Phasor Plot":
                            _extra["selected_channel"] = selected_channel
                            _extra["phasor_harmonic"] = selected_harmonic
                            _extra["phasor_f"] = f
                            _extra["phasor_category"] = phasor_category
                    elif method == "Dimension Reduction":
                        _extra["selected_features"] = selected_features
                        _extra["dr_method"] = dr_method
                        _extra["hyperParam_dict"] = hyperParam_dict
                    _export_script_button(method, uploaded_file, categorical_cols, color_by, opacity_by, shape_by, separate_by, subcolor_by, delimiter=delimiter, **_extra)

                    # Rerun only after all fragment widgets register; an earlier rerun
                    # lets Streamlit clean up their state and reset their values.
                    if st.session_state.pop("_plot_needs_rebuild", False) or _plot_build_params() != build_params:
                        st.rerun(scope="app")

                if bivar_display_area is not None:
                    with bivar_display_area:
                        _render_plot_and_controls(fig, _build_params)
                else:
                    _render_plot_and_controls(fig, _build_params)

        else:
            st.markdown(f"<h5 style='text-align: center; color: red'>No data available after filtering {sad_emoji}</h5>", unsafe_allow_html=True)

    elif uploaded_file is None and not use_data_extraction:
        st.caption("Upload a dataset to get started. Its columns will be "
                   f"automatically parsed {happy_emoji}")


# A closing review may trigger another rerun when method availability changes.
# Keep settings until every analysis widget has rendered, then resume normal cleanup.
st.session_state.pop("_review_preserve_controls", None)
