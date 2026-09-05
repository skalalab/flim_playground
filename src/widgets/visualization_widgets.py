import numpy as np
import streamlit as st
from streamlit_sortables import sort_items

from src.emojis import sad_emoji
from src.vis.plot_defaults import (
    DEFAULT_AXIS_LABEL_FONT_SIZE,
    DEFAULT_COLORMAP,
    DEFAULT_LEGEND_FONT_SIZE,
    DEFAULT_POINT_SIZE,
)
from src.widgets.encoding_state import color_multiselect_label, prune_to_options
from src.widgets.analysis_widget_state import control_default, number_input_default

# Explicit keys preserve selections when labels or option lists change.
COLOR_BY_KEY = "vis_encoding_color_by"
AS_COLOUR_KEY = "vis_encoding_as_colour"

# Shape and subcolor share a selection across point-based methods. Methods without
# this picker allow Streamlit to clear it through normal widget cleanup.
PICKER_COL_KEY = "vis_encoding_picker_col"
OPACITY_BY_KEY = "vis_encoding_opacity_by"

# Collapse controls row aggregation and is not passed to get_point_visual_mappings.
COLLAPSE_BY_KEY = "vis_encoding_collapse_by"


def _pruned_selectbox(label, options, key, **kwargs):
    """Preserve a keyed selection while clearing values absent from current options.

    Filters and grouping controls change these lists. Prune before rendering so
    Streamlit never receives a saved value that its selectbox cannot offer.
    """
    stored = st.session_state.get(key)
    pruned = prune_to_options(stored, options)
    if stored != pruned:
        st.session_state[key] = pruned
    return st.selectbox(label, options, index=None,
                        placeholder="Choose an option...", key=key, **kwargs)


def _picker_selectbox(label, options, **kwargs):
    """Select a column shared by the shape and subcolor modes."""
    return _pruned_selectbox(label, options, PICKER_COL_KEY, **kwargs)


# Native labels remain available to screen readers when the visible label is custom.
PICKER_LABELS = {True: "Subcolor by", False: "Shape by"}

# The label reads "Shape [switch] subcolor by" above the column picker.
# Both modes encode nominal values; opacity has its own ordinal control.
SWITCH_LEAD = "Shape"
SWITCH_TRAIL = "subcolor by"


def _shape_selectbox(available_categories, **kwargs):
    """Render the shared picker in shape mode, with optional native-label hiding."""
    return _picker_selectbox(PICKER_LABELS[False], available_categories, **kwargs)


def _picker_label(text):
    """Align the custom label and toggle above their shared picker.

    Match Streamlit's widget-label font and toggle height. Keep the CSS in the
    same markdown element to avoid adding a child and gap to the label row.
    """
    st.markdown(
        f"<style>.st-key-{AS_COLOUR_KEY} p{{font-size:0.875rem;line-height:1.6}}</style>"
        f"<div style='font-size:0.875rem; margin:0; white-space:nowrap;"
        f" display:flex; align-items:center; min-height:1.5rem;'>{text}</div>",
        unsafe_allow_html=True,
    )


def visual_encoding_channels_widget(filtered_df, categorical_cols, color_based=True, point_based=True, separate_by_available=False, subcolor_available=False, collapse_available=False):
    available_categories = [category for category in categorical_cols if category in filtered_df.columns and filtered_df[category].nunique() > 1]
    color_by = []
    opacity_by = shape_by = separate_by = subcolor_by = collapse_by = None

    if len(available_categories) == 0:
        return color_by, opacity_by, shape_by, separate_by, subcolor_by, collapse_by

    if not color_based:
        return color_by, opacity_by, shape_by, separate_by, subcolor_by, collapse_by

    # Named, equal-width slots keep the layout consistent as method-specific controls
    # appear. The shape/subcolor switch shares the final picker's label row.
    slots = []
    if separate_by_available and point_based:
        slots.append("separate")
    slots.append("color")
    if collapse_available and point_based:
        slots.append("collapse")
    if point_based:
        slots += ["opacity", "picker"]
    cols = st.columns(len(slots))
    at = {name: index for index, name in enumerate(slots)}

    # Separate by widget
    if "separate" in at:
        with cols[at["separate"]]:
            separate_by = _pruned_selectbox(
                "Separate by", available_categories, key="analysis_control_separate_by")

    available_for_color = [cat for cat in available_categories if cat != separate_by]
    show_subcolor = subcolor_available and point_based

    # Resolve grouping before the shape/subcolor control. Seed and prune through
    # session state only, avoiding duplicate-default warnings from the multiselect.
    default_color = [available_for_color[0]] if available_for_color else []
    if COLOR_BY_KEY not in st.session_state:
        st.session_state[COLOR_BY_KEY] = default_color
    pruned_color = prune_to_options(
        st.session_state[COLOR_BY_KEY], available_for_color, fallback=default_color,
    )
    if st.session_state[COLOR_BY_KEY] != pruned_color:
        st.session_state[COLOR_BY_KEY] = pruned_color
    effective_color = pruned_color

    # Grouping columns already identify each x-axis slot. Offer the remaining
    # categories to shape, subcolor and opacity through one shared option list.
    available_for_decoration = [cat for cat in available_categories
                                if cat != separate_by and cat not in effective_color]

    as_colour = False
    if point_based:
        with cols[at["picker"]]:
            if show_subcolor:
                # Evaluate the switch before Color by so its label reflects this run.
                # Match native label/input spacing while keeping the picker full width.
                with st.container(gap="xxsmall"):
                    with st.container(horizontal=True, vertical_alignment="top",
                                      gap="xsmall"):
                        with st.container(width="content"):
                            _picker_label(SWITCH_LEAD)
                        as_colour = st.toggle(
                            SWITCH_TRAIL, key=AS_COLOUR_KEY, width="content",
                            help="**Off** — the column below sets point shape."
                                 "\n\n**On** — it sets color: each value keeps one "
                                 "color plot-wide, so you can spot it in every group it "
                                 "appears in and see the spread within each. Best for a "
                                 "column nested inside the grouping one (e.g. donors "
                                 "within each treatment).",
                        )
                    # Keep the active channel in the native label for screen readers.
                    if as_colour:
                        subcolor_by = _picker_selectbox(
                            PICKER_LABELS[True], available_for_decoration,
                            disabled=not effective_color, label_visibility="collapsed",
                        )
                        # A disabled selectbox can still return its stored value.
                        if not effective_color:
                            subcolor_by = None
                    else:
                        shape_by = _shape_selectbox(available_for_decoration,
                                                    label_visibility="collapsed")
            else:
                shape_by = _shape_selectbox(available_for_decoration)

    with cols[at["color"]]:
        color_by = st.multiselect(
            color_multiselect_label(show_subcolor, as_colour),
            available_for_color,
            key=COLOR_BY_KEY,
            help="Groups compared along the x axis. These groups set the color too, "
                 "unless the Shape/subcolor switch is on — then that column sets the "
                 "color and these only set the x positions.",
        )

    # Subcolor requires an active grouping, including after this run's multiselect change.
    if not color_by:
        subcolor_by = None

    # Grouping narrows Collapse by, never the reverse. Decoration columns remain
    # eligible: each survives collapse when constant within the replicate group.
    if "collapse" in at:
        available_for_collapse = [cat for cat in available_for_color if cat not in color_by]
        with cols[at["collapse"]]:
            collapse_by = _pruned_selectbox(
                "Collapse by", available_for_collapse, COLLAPSE_BY_KEY,
                help="One point per value of this column, inside each x group, holding "
                     "the MEAN of the cells it covers -- so the box, the mean line and "
                     "every statistic describe replicates (dishes, patients, images) "
                     "rather than cells. Pair it with Subcolor by on the same column to "
                     "trace one replicate across every group.",
            )

    if point_based:
        with cols[at["opacity"]]:
            opacity_by = _pruned_selectbox("Opacity by", available_for_decoration,
                                           OPACITY_BY_KEY)

    return color_by, opacity_by, shape_by, separate_by, subcolor_by, collapse_by

def umap_hyperParams_widget():
    col1, col2 = st.columns(2)
    umap_hyperParams_dict = {}
    with col1:
        n_neighbors = st.number_input(
            "n_neighbors",
            value=number_input_default(st.session_state, "analysis_control_umap_neighbors", 15),
            step=5,
            format="%d",
            key="analysis_control_umap_neighbors",
        )
        umap_hyperParams_dict["n_neighbors"] = n_neighbors
    with col2:
        min_dist = st.number_input(
            "min_dist",
            value=number_input_default(st.session_state, "analysis_control_umap_min_dist", 0.1),
            step=0.1,
            key="analysis_control_umap_min_dist",
        )
        umap_hyperParams_dict["min_dist"] = min_dist

    return umap_hyperParams_dict

def tsne_hyperParams_widget():
    col1, col2 = st.columns(2)
    tsne_hyperParams_dict = {}
    with col1:
        perplexity = st.number_input("perplexity", value=number_input_default(st.session_state, "analysis_control_tsne_perplexity", 15), step=10, min_value=5, max_value=1000, key="analysis_control_tsne_perplexity")
        tsne_hyperParams_dict["perplexity"] = perplexity
    with col2:
        early_exaggeration = st.number_input("early_exaggeration", value=number_input_default(st.session_state, "analysis_control_tsne_exaggeration", 1), step=1, min_value=1, max_value=15, key="analysis_control_tsne_exaggeration")
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
        default=control_default(st.session_state, "compare_pairs", pair_labels),
        key="compare_pairs"
    )

    # Convert selected labels back to original pairs
    selected_pairs = [label_to_pair[label] for label in selected_labels]
    return selected_pairs


def histogram_bin_width_widget(x_data, key=None):
    """Choose common bin edges for a 1D array with missing values removed."""
    _, bin_edges_all = np.histogram(x_data, bins='auto')
    nbins = len(bin_edges_all) - 1
    if nbins > 1:
        default_bin_width = bin_edges_all[1] - bin_edges_all[0]
        min_val = x_data.min()
        max_val = x_data.max()
        range = max_val - min_val
        default_bin_width = min(default_bin_width, range / 3)
        if key is not None and key in st.session_state:
            stored_width = st.session_state[key]
            if stored_width is None or not 0 < stored_width <= range / 3:
                st.session_state[key] = default_bin_width
        bin_width = st.number_input(label="Bin Width", max_value=range/3, value=number_input_default(st.session_state, key, default_bin_width), step=range/50, key=key)
        # Add a small epsilon to max_val to ensure the rightmost edge includes the max value
        epsilon = 1e-9
        # Calculate common bin edges based on the user-provided bin_width
        common_bin_edges = np.arange(min_val, max_val + bin_width + epsilon, bin_width)

    else:
        # Use numpy's single bin for constant or near-constant data.
        common_bin_edges = bin_edges_all

    return common_bin_edges

def gmm_hyperParams_widget():
    col3, col4 = st.columns(2)
    with col3:
        fit_gmm_max_components = st.slider("Max Components", min_value=2, max_value=5, value=control_default(st.session_state, "fit_gmm_max_components", 3), step=1, key="fit_gmm_max_components")
    with col4:
        fit_gmm_min_weight_threshold = st.slider("Min Weight Threshold", min_value=0.0, max_value=0.3, value=control_default(st.session_state, "fit_gmm_min_weight_threshold", 0.1), step=0.1, key="fit_gmm_min_weight_threshold")
    return fit_gmm_max_components, fit_gmm_min_weight_threshold

def _compute_channel_harmonics(feature_groups_dict):
    """Map each fit-free channel to the phasor harmonics it can plot.

    A harmonic is available only when BOTH its G and S coordinates are present
    among the channel's features. Returns ``{channel: [harmonics...]}`` where the
    list may be empty when no complete G/S pair exists.
    """
    channel_harmonics = {}
    for extractor_channel in feature_groups_dict.keys():
        try:
            extractor, channel = extractor_channel.split("_", 1)
        except Exception:
            continue
        if extractor == "Lifetime fit free":
            channel_harmonics[channel] = []
            features = feature_groups_dict[extractor_channel]
            # A harmonic needs both its G and S coordinates present.
            if any("G(1st)" in feature for feature in features) and any("S(1st)" in feature for feature in features):
                channel_harmonics[channel].append(1)
            if any("G(2nd)" in feature for feature in features) and any("S(2nd)" in feature for feature in features):
                channel_harmonics[channel].append(2)
    return channel_harmonics


def phasor_params_widget(feature_groups_dict):
    channel_harmonics = _compute_channel_harmonics(feature_groups_dict)

    if len(channel_harmonics.keys()) > 1:
        selected_channel = st.selectbox("Channel", channel_harmonics.keys(), key="analysis_control_phasor_channel")
    elif len(channel_harmonics.keys()) == 1:
        selected_channel = list(channel_harmonics.keys())[0]
    else:
        st.error(f"No available channels found for phasor plot {sad_emoji}")
        return None, None, None
    if not channel_harmonics.get(selected_channel):
        # A fit-free channel may have no complete pair even when another is plottable.
        st.error(f"No phasor harmonics available for {selected_channel}: both G and S coordinates are required. {sad_emoji}")
        return None, None, None
    selected_harmonic = st.selectbox(f"{selected_channel} harmonic No. ", channel_harmonics[selected_channel], key=f"analysis_control_phasor_harmonic_{selected_channel}")
    f = None
    if selected_channel is not None and selected_harmonic is not None:
        f = st.number_input("Laser repetition rate (**GHz**)", value=number_input_default(st.session_state, "analysis_control_phasor_frequency", 0.08), min_value=0.0, step=0.01, key="analysis_control_phasor_frequency")
    return selected_channel, selected_harmonic, f

def plot_config_widget(point_based=True, show_colormap=False, show_count_toggle=False):
    """Render plot styling controls bound directly to session state."""
    # Determine number of columns based on what widgets to show
    num_cols = 0
    if point_based:
        num_cols += 1
    num_cols += 2  # axis label size and legend size
    if show_colormap:
        num_cols += 1

    # Reassign on the rendering run so Streamlit sends saved values to remounted
    # browser widgets. Supplying values through state alone avoids default warnings.
    for state_key, default in (("plot_point_size", DEFAULT_POINT_SIZE),
                               ("plot_axis_label_size", DEFAULT_AXIS_LABEL_FONT_SIZE),
                               ("plot_legend_size", DEFAULT_LEGEND_FONT_SIZE),
                               ("plot_colormap", DEFAULT_COLORMAP),
                               ("plot_show_group_counts", False)):
        st.session_state[state_key] = st.session_state.get(state_key, default)

    cols = st.columns(num_cols)
    col_idx = 0

    # Point size widget — key= binds directly to session state
    if point_based:
        with cols[col_idx]:
            st.number_input("Point Size", min_value=1, step=1, key="plot_point_size")
        col_idx += 1

    # Axis label size widget
    with cols[col_idx]:
        st.number_input("Axis Label Font Size", min_value=8, step=1, key="plot_axis_label_size")
    col_idx += 1

    # Legend size widget
    with cols[col_idx]:
        st.number_input("Legend Font Size", min_value=8, step=1, key="plot_legend_size")
    col_idx += 1

    # Colormap widget
    if show_colormap:
        colormap_options = [
            "tab10", "tab20", "colorblind", "Set1", "Set2", "Set3", "Pastel1", "Pastel2",
            "Accent", "viridis", "plasma", "inferno", "magma", "cividis"
        ]
        # Prune a saved palette before rendering the keyed selectbox.
        if st.session_state["plot_colormap"] not in colormap_options:
            st.session_state["plot_colormap"] = colormap_options[0]
        with cols[col_idx]:
            st.selectbox("Color Map", colormap_options,
                        key="plot_colormap",
                        help="Choose color palette for categorical data")

    # Optional toggle: append per-color-group sample size to legend entries
    if show_count_toggle:
        st.checkbox(
            "Show group counts (n) in legend",
            key="plot_show_group_counts",
            help="Append each color group's sample size to its legend entry, e.g. 'Control (n=42)'.",
        )

def get_custom_order_widget(items, key):
    if sort_items is None:
        st.warning("streamlit-sortables is not installed. Please install it to use this feature.")
        return items

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

    # Mount the component only when visible; a collapsed expander gives it zero height.
    show_order_config = st.checkbox("Reorder X-axis Groups", value=False, key="analysis_control_reorder_groups")

    if show_order_config:
        group_by_cols = color_by if color_by else []
        if not group_by_cols:
            temp_df_groups = ["all_data"]
        else:
            temp_df_groups = filtered_df[group_by_cols].astype(str).agg('::'.join, axis=1).unique()

        cmp_groups = natural_tuple_sort(temp_df_groups, delimiter='::')

        # Retain the saved relative order and append newly available groups.
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
            # Defer the rebuild until the fragment's remaining widgets have registered.
            st.session_state["_plot_needs_rebuild"] = True
