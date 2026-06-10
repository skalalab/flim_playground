import streamlit as st
import numpy as np

from streamlit_sortables import sort_items

from src.vis.plot_defaults import (
    DEFAULT_AXIS_LABEL_FONT_SIZE,
    DEFAULT_COLORMAP,
    DEFAULT_LEGEND_FONT_SIZE,
    DEFAULT_POINT_SIZE,
)

def visual_encoding_channels_widget(filtered_df, categorical_cols, color_based=True, point_based=True, separate_by_available=False):
    available_categories = [category for category in categorical_cols if category in filtered_df.columns and filtered_df[category].nunique() > 1]
    color_by = []
    opacity_by = shape_by = separate_by = None
    
    if len(available_categories) == 0:
        return color_by, opacity_by, shape_by, separate_by
    
    if not color_based:
        return color_by, opacity_by, shape_by, separate_by
    
    # Determine number of columns
    num_cols = 4 if point_based and separate_by_available else 3 if point_based else 1
    cols = st.columns(num_cols)

    # Separate by widget
    if separate_by_available and point_based:
        with cols[0]:
            separate_by = st.selectbox("Separate by", available_categories, index=None, placeholder="Choose an option...")

    # Color by widget (exclude separate_by option)
        available_for_color = [cat for cat in available_categories if cat != separate_by]
        with cols[1]:
            color_by = st.multiselect("Color by", available_for_color, default=[available_for_color[0]] if available_for_color else [])
    else: 
        with cols[0]:
            color_by = st.multiselect("Color by", available_categories, default=[available_categories[0]] if available_categories else [])

    # Initialize defaults
    opacity_by = shape_by = None

    # Point-based widgets (opacity and shape)
    if point_based:
        with cols[2 if separate_by_available else 1]:
            opacity_by = st.selectbox("Opacity by", available_categories, index=None, placeholder="Choose an option...")
        with cols[3 if separate_by_available else 2]:
            shape_by = st.selectbox("Shape by", available_categories, index=None, placeholder="Choose an option...")

    return color_by, opacity_by, shape_by, separate_by

def umap_hyperParams_widget():
    col1, col2 = st.columns(2)
    # First number incrementor in the first column
    umap_hyperParams_dict = {}
    with col1:
        n_neighbors = st.number_input(
            "n_neighbors",
            value=15,  # Initial value
            step=5,             # Increment/Decrement step
            format="%d"            # Integer format
        )
        umap_hyperParams_dict["n_neighbors"] = n_neighbors
    # Second number incrementor in the second column
    with col2:
        min_dist = st.number_input(
            "min_dist",
            value=0.1,  # Initial value
            step=0.1,            
        )
        umap_hyperParams_dict["min_dist"] = min_dist

    return umap_hyperParams_dict

def tsne_hyperParams_widget():
    col1, col2 = st.columns(2)
    tsne_hyperParams_dict = {}
    with col1:
        perplexity = st.number_input("perplexity", value=15, step=10, min_value=5, max_value=1000)
        tsne_hyperParams_dict["perplexity"] = perplexity
    with col2:
        early_exaggeration = st.number_input("early_exaggeration", value=1, step=1, min_value=1, max_value=15)
        tsne_hyperParams_dict["early_exaggeration"] = early_exaggeration
    return tsne_hyperParams_dict

def comparison_pair_widget(available_pairs):
    # Create more descriptive labels for each pair
    pair_labels = []
    for pair in available_pairs:
        if isinstance(pair, tuple) and len(pair) == 2:
            # Format as "Group1 vs Group2"
            label = f"{pair[0]} vs {pair[1]}"
        else:
            # Fallback for other formats
            label = str(pair)
        pair_labels.append(label)
    
    # Create a mapping from labels back to original pairs
    label_to_pair = dict(zip(pair_labels, available_pairs))
    
    selected_labels = st.multiselect(
        "Select comparison pairs",
        pair_labels,
        default=pair_labels,
        key="compare_pairs"
    )
    
    # Convert selected labels back to original pairs
    selected_pairs = [label_to_pair[label] for label in selected_labels]
    return selected_pairs


def histogram_bin_width_widget(x_data, key=None):
    # x_data: 1D numpy array of data to be binned (already na dropped)
    # use np's automatic binning logic by not specifying nbins explicitly in np.histogram
    # Calculate bins using numpy based on the overall data range
    counts_all, bin_edges_all = np.histogram(x_data, bins='auto') # Use 'auto' as default
    nbins = len(bin_edges_all) - 1
    if nbins > 1:
        default_bin_width = bin_edges_all[1] - bin_edges_all[0]
        # add a widget to adjust the bin_width, use text input to get the bin width
        # Ensure bin_width is positive to avoid errors in np.arange
        min_val = x_data.min()
        max_val = x_data.max()
        range = max_val - min_val
        bin_width = st.number_input(label="Bin Width", max_value=range/3, value=default_bin_width, step=range/50, key=key)
        # Add a small epsilon to max_val to ensure the rightmost edge includes the max value
        epsilon = 1e-9
        # Calculate common bin edges based on the user-provided bin_width
        common_bin_edges = np.arange(min_val, max_val + bin_width + epsilon, bin_width)
    
    return common_bin_edges

def gmm_hyperParams_widget():
    col3, col4 = st.columns(2)
    with col3:
        fit_gmm_max_components = st.slider("Max Components", min_value=2, max_value=5, value=3, step=1, key="fit_gmm_max_components")
    with col4:
        fit_gmm_min_weight_threshold = st.slider("Min Weight Threshold", min_value=0.0, max_value=0.3, value=0.1, step=0.1, key="fit_gmm_min_weight_threshold")
    return fit_gmm_max_components, fit_gmm_min_weight_threshold

def phasor_params_widget(feature_groups_dict):

    channel_harmonics = {}
    
    for extractor_channel in feature_groups_dict.keys():
        try:
            extractor, channel = extractor_channel.split("_", 1)
        except Exception as e:
            continue
        if extractor == "Lifetime fit free":
            channel_harmonics[channel] = []
            features = feature_groups_dict[extractor_channel]
            
            # Check if any feature contains G(1st) AND any feature contains S(1st)
            if any("G(1st)" in feature for feature in features) and any("S(1st)" in feature for feature in features):
                channel_harmonics[channel].append(1)
            
            # Check if any feature contains G(2nd) AND any feature contains S(2nd)
            if any("G(2nd)" in feature for feature in features) and any("S(2nd)" in feature for feature in features):
                channel_harmonics[channel].append(2)
                    
    if len(channel_harmonics.keys()) > 1:
        selected_channel = st.selectbox("Channel", channel_harmonics.keys())
    elif len(channel_harmonics.keys()) == 1:
        selected_channel = list(channel_harmonics.keys())[0]
    else:
        selected_channel = None
        st.error("No available channels found for phasor plot")
        return None, None, None
    selected_harmonic = st.selectbox(f"{selected_channel} harmonic No. ", channel_harmonics[selected_channel])
    if selected_channel is not None and selected_harmonic is not None:
        f = st.number_input("Laser repetition rate (**GHz**)", value=0.08, min_value=0.0, step=0.01)
    return selected_channel, selected_harmonic, f

def plot_config_widget(point_based=True, show_colormap=False):
    """
    Widgets to change point, axis label font, legend font size, and colormap.
    Uses key= parameters to write directly to session state — no manual
    compare-and-rerun needed. Streamlit reruns naturally when the user changes a value.
    """
    # Determine number of columns based on what widgets to show
    num_cols = 0
    if point_based:
        num_cols += 1
    num_cols += 2  # axis label size and legend size
    if show_colormap:
        num_cols += 1

    cols = st.columns(num_cols)
    col_idx = 0

    # Point size widget — key= binds directly to session state
    if point_based:
        with cols[col_idx]:
            st.number_input("Point Size", value=st.session_state.get("plot_point_size", DEFAULT_POINT_SIZE),
                          min_value=1, step=1, key="plot_point_size")
        col_idx += 1

    # Axis label size widget
    with cols[col_idx]:
        st.number_input("Axis Label Font Size", value=st.session_state.get("plot_axis_label_size", DEFAULT_AXIS_LABEL_FONT_SIZE),
                       min_value=8, step=1, key="plot_axis_label_size")
    col_idx += 1

    # Legend size widget
    with cols[col_idx]:
        st.number_input("Legend Font Size", value=st.session_state.get("plot_legend_size", DEFAULT_LEGEND_FONT_SIZE),
                       min_value=8, step=1, key="plot_legend_size")
    col_idx += 1

    # Colormap widget
    if show_colormap:
        colormap_options = [
            "tab10", "tab20", "colorblind", "Set1", "Set2", "Set3", "Pastel1", "Pastel2",
            "Accent", "viridis", "plasma", "inferno", "magma", "cividis"
        ]
        current_colormap = st.session_state.get("plot_colormap", DEFAULT_COLORMAP)
        with cols[col_idx]:
            st.selectbox("Color Map", colormap_options,
                        index=colormap_options.index(current_colormap) if current_colormap in colormap_options else 0,
                        key="plot_colormap",
                        help="Choose color palette for categorical data")

def get_custom_order_widget(items, key):
    if sort_items is None:
        st.warning("streamlit-sortables is not installed. Please install it to use this feature.")
        return items
    
    # Custom CSS to ensure visibility in both light and dark modes
    # We force a light card look with black text
    custom_style = """
    .sortable-item {
        background-color: #f0f2f6;
        color: black;
        border: 1px solid #d6d6d6;
        border-radius: 4px;
        padding: 8px;
        margin-bottom: 4px;
    }
    """
    
    sorted_items = sort_items(items, key=key)
    return sorted_items

def get_visual_group_keys(filtered_df, selected_var, color_by, separate_by):
    """
    Constructs the session state keys for retrieving and storing custom order.
    Returns keys for separate groups and compare groups.
    """
    session_key_sep = f"custom_order_sep_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}"
    session_key_cmp = f"custom_order_cmp_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}"
    return session_key_sep, session_key_cmp

def reorder_x_axis_widget(filtered_df, selected_var, color_by, separate_by):
    """
    Renders the widget for reordering x-axis groups and updates session state.
    Use this function *after* plotting to allow the user to adjust the order for next render.
    """
    from src.vis.helpers import natural_tuple_sort
    
    session_key_sep, session_key_cmp = get_visual_group_keys(filtered_df, selected_var, color_by, separate_by)

    # specific to streamlit-sortables: it may not render correctly inside a collapsed expander because of 0 height
    # We use a checkbox to trigger a rerun and render it only when visible
    show_order_config = st.checkbox("Reorder X-axis Groups", value=False)
    
    if show_order_config:
        # Color groups (Compare groups)
        # Grouping logic
        group_by_cols = color_by if color_by else []
        if not group_by_cols:
            temp_df_groups = ["all_data"]
        else:
            temp_df_groups = filtered_df[group_by_cols].astype(str).agg('::'.join, axis=1).unique()
        
        cmp_groups = natural_tuple_sort(temp_df_groups, delimiter='::')
        
        # Merge with existing stored order to preserve relative ordering of known items while adding new ones
        if session_key_cmp in st.session_state:
            stored = st.session_state[session_key_cmp]
            cmp_groups = [g for g in stored if g in cmp_groups] + [g for g in cmp_groups if g not in stored]

        # Initialize version in session state if not present
        version_key = f"sort_version_{'_'.join(color_by)}"
        if version_key not in st.session_state:
            st.session_state[version_key] = 0
            
        st.write(f"**Reorder Groups ({', '.join(color_by) if color_by else 'All Data'})**")
        
        # Use version in key to force re-mount when confirmed
        widget_key = f"sort_cmp_{'_'.join(color_by)}_{st.session_state[version_key]}"
        new_cmp_order = get_custom_order_widget(cmp_groups, key=widget_key)
        
        if st.button("Confirm Reordering"):
            if new_cmp_order:
                st.session_state[session_key_cmp] = new_cmp_order
                # Increment version to force re-render next time
                st.session_state[version_key] += 1
            st.rerun()