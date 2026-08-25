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

# Explicit keys. Without them the state of the Color by multiselect lives under an
# auto-generated ID that includes its label -- so renaming it to "Group by" would wipe
# the user's selection on every toggle flip.
COLOR_BY_KEY = "vis_encoding_color_by"
AS_COLOUR_KEY = "vis_encoding_as_colour"

# ONE key for the shape/subcolor picker's column, shared by both of its roles. The switch chooses
# which channel the column drives; it does not choose a different column, so flipping it
# leaves the selection alone rather than swapping in whatever that channel held last.
#
# One key also avoids any restore machinery: Streamlit purges the state of a widget that
# stops rendering, and under a shared key one role or the other always renders on a
# POINT-BASED method. Methods with no point channel (FOV Comparison, Feature Histogram,
# Classification) render neither, so leaving for one of those does clear the column.
#
# The cost: a column picked as shape on Scatter arrives preselected in Feature
# Comparison's picker, and vice versa.
PICKER_COL_KEY = "vis_encoding_picker_col"


def _picker_selectbox(label, options, **kwargs):
    """The shape/subcolor picker, in whichever role the switch has put it.

    Two things can retire the held column: the row offers only categories with
    ``nunique() > 1``, so a filter can disqualify it; and the subcolor role excludes
    whatever Color by is grouping on, so a column legal as opacity can be illegal as
    subcolor. Streamlit raises on an unoffered value under an explicit key, so it must be
    dropped here -- the one case where a flip clears the selection.
    """
    stored = st.session_state.get(PICKER_COL_KEY)
    pruned = prune_to_options(stored, options)
    if stored != pruned:
        st.session_state[PICKER_COL_KEY] = pruned
    return st.selectbox(label, options, index=None,
                        placeholder="Choose an option...", key=PICKER_COL_KEY, **kwargs)


# The name handed to the shape/subcolor picker WIDGET, keyed on the switch. Its visible label is
# collapsed in favour of the switch phrase below, so this is what a screen reader reads.
PICKER_LABELS = {True: "Subcolor by", False: "Shape by"}

# The switch reads as one phrase with the channels either side of it:
#
#     Opacity [o---] subcolor by     <- opacity drives the picker below
#     Opacity [---o] subcolor by     <- subcolor does
#
# Static, so both channels stay on screen and the knob alone says which is active. Ends
# on "by" so it runs into the picker below: "shape/subcolor by <column>". Shape rather
# than opacity is the partner because shape and colour are both NOMINAL -- either is a
# sensible encoding for the same column, so the flip is a real choice. Opacity is the one
# ordinal channel (create_opacity_mapping ranks values on a ramp), so pairing it here
# would have offered to put an unordered column on an ordered scale.
SWITCH_LEAD = "Shape"
SWITCH_TRAIL = "subcolor by"


def _shape_selectbox(available_categories, **kwargs):
    """The Shape by picker, shared by the switch's Shape role and every method that has no
    switch. ``kwargs`` carries ``label_visibility`` where the switch draws the label
    itself; methods with no switch keep the native one."""
    return _picker_selectbox(PICKER_LABELS[False], available_categories, **kwargs)


def _picker_label(text):
    """Draw the shape/subcolor picker's label, so the switch can sit after it.

    Streamlit puts a widget's label in the same block as its input, so a switch placed
    beside the picker lands beside the *input*. Hence the hand-drawn label and the
    collapsed native one.

    Every number here was measured in a browser and nothing re-checks it, so a Streamlit
    upgrade drifts it silently:

    * 0.875rem is Streamlit's widget-label size (theme ``fontSizes.sm``). Colour is left
      to inherit so it follows the active theme.
    * A toggle's box carries trailing space under the visible switch, so aligning boxes
      leaves the switch above the label: 7.5px on ``vertical_alignment="center"``, 16.4px
      on ``"bottom"``. A box padded to the switch's height with ``"top"`` gives 0.5px.
    * Toggle labels render at body size, 16px against this row's 14px, so the phrase came
      out in two sizes. The ``st-key-`` rule fixing that is why the switch needs an
      explicit key, and it rides in this same markdown call because a separate
      ``<style>`` would be another child of a gapped container and widen the row.
    """
    st.markdown(
        f"<style>.st-key-{AS_COLOUR_KEY} p{{font-size:0.875rem;line-height:1.6}}</style>"
        f"<div style='font-size:0.875rem; margin:0; white-space:nowrap;"
        f" display:flex; align-items:center; min-height:1.5rem;'>{text}</div>",
        unsafe_allow_html=True,
    )


def visual_encoding_channels_widget(filtered_df, categorical_cols, color_based=True, point_based=True, separate_by_available=False, subcolor_available=False):
    available_categories = [category for category in categorical_cols if category in filtered_df.columns and filtered_df[category].nunique() > 1]
    color_by = []
    opacity_by = shape_by = separate_by = subcolor_by = None

    if len(available_categories) == 0:
        return color_by, opacity_by, shape_by, separate_by, subcolor_by

    if not color_based:
        return color_by, opacity_by, shape_by, separate_by, subcolor_by

    # A plain count, so every column is equal. The channel switch is absorbed inside the
    # LAST column -- sharing its picker's top line -- rather than given a column of its
    # own, so this row still lays out identically on Scatter, Phasor and Dimension
    # Reduction, which have no switch at all. If the column is too narrow to hold picker
    # and switch side by side the switch wraps underneath, which is the old layout and
    # still readable; widening that column instead would misalign every other method.
    if point_based:
        num_cols = 4 if separate_by_available else 3
    else:
        num_cols = 1
    cols = st.columns(num_cols)

    # Separate by widget
    if separate_by_available and point_based:
        with cols[0]:
            separate_by = st.selectbox("Separate by", available_categories, index=None, placeholder="Choose an option...")

    available_for_color = [cat for cat in available_categories if cat != separate_by]
    show_subcolor = subcolor_available and point_based

    # Resolved before the switch's slot, because that slot needs to know whether anything
    # is grouped by. Independent of every other control, so hoisting it changes nothing
    # except making it available early. `effective_color` is what the multiselect will
    # hold once it renders -- always the stored selection, because the first-run default
    # is SEEDED into state here rather than handed to the multiselect as `default=`.
    #
    # Seeding is what `default=` amounted to anyway: under an explicit key the stored
    # value wins, so the argument only ever decided the run with nothing stored -- the
    # one this branch covers. Passing both is what Streamlit warns about ("created with
    # a default value but also had its value set via the Session State API"), and the
    # pruning below writes this same key in the same run the multiselect renders, so the
    # warning fires for real: on a filter that retires a chosen column, or on Separate by
    # claiming it. Same shape as the feature multiselects in selection_widgets.py.
    default_color = [available_for_color[0]] if available_for_color else []
    if COLOR_BY_KEY not in st.session_state:
        st.session_state[COLOR_BY_KEY] = default_color
    pruned_color = prune_to_options(
        st.session_state[COLOR_BY_KEY], available_for_color, fallback=default_color,
    )
    if st.session_state[COLOR_BY_KEY] != pruned_color:
        st.session_state[COLOR_BY_KEY] = pruned_color
    effective_color = pruned_color

    # The switch's slot is the LAST column, but it is EVALUATED here, before the Color by
    # multiselect renders -- st.columns hands back containers, so writing into them out of
    # order still lays them out left to right. That is what lets the multiselect's label
    # read the switch and picker values from this run. Reading them from session state
    # instead left "Group by" a run behind whenever the picker changed for a reason other
    # than a click: pruning dropping the column, the selection being cleared, or the
    # disabled branch forcing subcolor_by to None.
    as_colour = False
    if point_based:
        with cols[3 if separate_by_available else 2]:
            if show_subcolor:
                # The switch sits immediately after the phrase's first word, on the same
                # line, with the picker at full width underneath -- see SWITCH_LEAD for
                # the shape. Both halves take only their own width in a horizontal row,
                # so the switch lands right after the words rather than out at the far
                # edge. That is why the label is drawn by hand and the picker's own
                # collapsed: Streamlit puts a widget's label in the same block as its
                # input, so a switch set beside the picker is pushed out to wherever the
                # *input* ends. The picker keeps the whole column width, which the long
                # column names this row offers need.
                # gap="xsmall" (0.5rem) rather than the "small" default (1rem): the
                # switch is meant to read as attached to the label it modifies, and a
                # full 1rem sets it adrift halfway to the next column.
                # gap="xxsmall" (0.25rem) reunites the drawn label with its picker.
                # Streamlit puts a native label and its input in ONE child of the column
                # with 4px between them; drawing the label separately makes it a second
                # child, so the column's own 1rem gap opens up and drops the picker below
                # the ones either side of it -- measured at 12px against a sibling
                # selectbox. Wrapping both in a container whose gap is 4px restores the
                # native spacing exactly (0px displacement); gap=None overshoots to -4px.
                with st.container(gap="xxsmall"):
                    with st.container(horizontal=True, vertical_alignment="top",
                                      gap="xsmall"):
                        with st.container(width="content"):
                            _picker_label(SWITCH_LEAD)
                        as_colour = st.toggle(
                            SWITCH_TRAIL, key=AS_COLOUR_KEY, width="content",
                            # Leads with the two states, because that is the question
                            # someone opens this tooltip to answer. The one fact worth
                            # the words is that a colour means the VALUE, figure-wide --
                            # that is what makes the channel useful rather than merely
                            # colourful, and nothing on screen says it. The mechanism is
                            # described against "values" and "groups" because this text
                            # ships to every dataset; the column names appear only behind
                            # an explicit "e.g.", which reads as an illustration rather
                            # than as an assumption about what the user's columns are.
                            help="**Off** \u2014 the column below sets point shape."
                                 "\n\n**On** \u2014 it sets color: each value keeps one "
                                 "color plot-wide, so you can spot it in every group it "
                                 "appears in and see the spread within each. Best for a "
                                 "column nested inside the grouping one (e.g. donors "
                                 "within each treatment).",
                        )
                    # Still named for the channel it drives even though that name is
                    # collapsed: it is what a screen reader announces, and the switch
                    # phrase above is decoration to anything that does not render CSS.
                    if as_colour:
                        available_for_subcolor = [cat for cat in available_for_color if cat not in effective_color]
                        subcolor_by = _picker_selectbox(
                            PICKER_LABELS[True], available_for_subcolor,
                            disabled=not effective_color, label_visibility="collapsed",
                        )
                        # A disabled selectbox still returns what it last held, so the
                        # forcing is not redundant with `disabled` above.
                        if not effective_color:
                            subcolor_by = None
                    else:
                        shape_by = _shape_selectbox(available_categories,
                                                    label_visibility="collapsed")
            else:
                shape_by = _shape_selectbox(available_categories)

    with cols[1 if separate_by_available and point_based else 0]:
        color_by = st.multiselect(
            color_multiselect_label(show_subcolor, as_colour),
            available_for_color,
            key=COLOR_BY_KEY,
            help="Groups compared along the x axis. These groups set the color too, "
                 "unless the Shape/subcolor switch is on \u2014 then that column sets the "
                 "color and these only set the x positions.",
        )

    # subcolor_by was decided against the stored selection above; re-check it against what
    # the multiselect actually returned, so clearing every group in this same run does
    # not leave a subcolor column setting colour for a plot that has no groups to draw.
    if not color_by:
        subcolor_by = None

    # Opacity has no switch: it is the only ordinal channel, so it shares a column with
    # nothing. Unkeyed, as it was before this row took explicit keys.
    if point_based:
        with cols[2 if separate_by_available else 1]:
            opacity_by = st.selectbox("Opacity by", available_categories, index=None, placeholder="Choose an option...")

    return color_by, opacity_by, shape_by, separate_by, subcolor_by

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
    _, bin_edges_all = np.histogram(x_data, bins='auto') # Use 'auto' as default
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

    else:
        # Constant / near-constant feature: numpy's 'auto' yields a single bin;
        # fall back to those edges instead of leaving common_bin_edges unbound.
        common_bin_edges = bin_edges_all

    return common_bin_edges

def gmm_hyperParams_widget():
    col3, col4 = st.columns(2)
    with col3:
        fit_gmm_max_components = st.slider("Max Components", min_value=2, max_value=5, value=3, step=1, key="fit_gmm_max_components")
    with col4:
        fit_gmm_min_weight_threshold = st.slider("Min Weight Threshold", min_value=0.0, max_value=0.3, value=0.1, step=0.1, key="fit_gmm_min_weight_threshold")
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
        selected_channel = st.selectbox("Channel", channel_harmonics.keys())
    elif len(channel_harmonics.keys()) == 1:
        selected_channel = list(channel_harmonics.keys())[0]
    else:
        st.error(f"No available channels found for phasor plot {sad_emoji}")
        return None, None, None
    if not channel_harmonics.get(selected_channel):
        # No complete G/S pair for this channel -> nothing to plot. Guard the
        # otherwise-empty selectbox (which returns None) so f stays defined.
        st.error(f"No phasor harmonics available for {selected_channel}: both G and S coordinates are required. {sad_emoji}")
        return None, None, None
    selected_harmonic = st.selectbox(f"{selected_channel} harmonic No. ", channel_harmonics[selected_channel])
    f = None
    if selected_channel is not None and selected_harmonic is not None:
        f = st.number_input("Laser repetition rate (**GHz**)", value=0.08, min_value=0.0, step=0.01)
    return selected_channel, selected_harmonic, f

def plot_config_widget(point_based=True, show_colormap=False, show_count_toggle=False):
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

    # Every default is SEEDED into session state rather than handed to its widget as
    # `value=`/`index=`. Under an explicit key the stored value wins, so those arguments
    # only ever decided a run with nothing stored -- and passing both is what makes
    # Streamlit warn ("created with a default value but also had its value set via the
    # Session State API") on any run that also WRITES the key, which seeding does.
    #
    # data_analysis.py seeds these at the page top, which normally lands a run before this
    # row -- the row renders only once a dataset is loaded. But leaving the page purges
    # widget state while `vis_df`, a plain session key, survives, so a return visit seeds
    # and renders in ONE run and the warning fires. Seeding here as well keeps each
    # default beside the widget that owns it and covers the classification page's call,
    # which does not run data_analysis.py's block.
    for state_key, default in (("plot_point_size", DEFAULT_POINT_SIZE),
                               ("plot_axis_label_size", DEFAULT_AXIS_LABEL_FONT_SIZE),
                               ("plot_legend_size", DEFAULT_LEGEND_FONT_SIZE),
                               ("plot_colormap", DEFAULT_COLORMAP),
                               ("plot_show_group_counts", False)):
        if state_key not in st.session_state:
            st.session_state[state_key] = default

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
        # The dropped `index=` carried a fallback for a stored colormap this list does not
        # offer, and that fallback is still needed: Streamlit RAISES on an unoffered value
        # under an explicit key, where an out-of-range index would only have been ignored.
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
            # A new order is a build-time change, but do not st.rerun() here: this widget
            # runs inside the plot fragment (pages/data_analysis.py) and the styling
            # controls render after it, so rerunning now would reset them to their module
            # defaults. Flag it and let the escalation at the end of the fragment fire.
            st.session_state["_plot_needs_rebuild"] = True