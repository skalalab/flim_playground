import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, median_abs_deviation, norm, ttest_ind
from sklearn.mixture import GaussianMixture

from src.emojis import sad_emoji
from src.vis.plot_defaults import WEBGL_POINT_THRESHOLD
from src.widgets.visualization_widgets import comparison_pair_widget


def log_negative_error(var_name):
    return f"Cannot apply log to {var_name}: contains negative values. {sad_emoji}"


def get_context_theme_color():
    """Plot color for the current theme: 'black' in light mode, 'white' otherwise.

    ``st.context.theme.type`` is a plain server-side read, so it costs no rerun. It is
    None until the browser reports in, which falls through to the dark-mode color.
    """
    return "black" if st.context.theme.type == "light" else "white"


def point_trace_class(n_points):
    """``go.Scattergl`` once a figure draws ``WEBGL_POINT_THRESHOLD`` points, else ``go.Scatter``.

    SVG costs one ``<path>`` DOM node per point, so a 14k-cell figure makes the browser
    walk and re-rasterise 14k nodes every time the page scrolls. WebGL holds the same
    points in typed-array buffers and adds no nodes.

    Decide this ONCE per figure from its total point count, never per trace: Plotly paints
    every WebGL trace beneath every SVG one, so a figure mixing the two would layer its
    colour groups by renderer rather than by draw order.
    """
    return go.Scattergl if n_points >= WEBGL_POINT_THRESHOLD else go.Scatter

def find_intersection(pi1, mu1, sigma1, pi2, mu2, sigma2):
    """
    Find the intersection point between two weighted Gaussian components where
    pi1 * N(x; mu1, sigma1) = pi2 * N(x; mu2, sigma2).
    """
    f = lambda x: pi1 * norm.pdf(x, mu1, sigma1) - pi2 * norm.pdf(x, mu2, sigma2)
    # The root must lie between the two means
    return brentq(f, min(mu1, mu2), max(mu1, mu2))

def glass_delta(group1, group2, mean_or_median):
    # group1 should be the control
    if mean_or_median == "Mean": 
        diff = np.mean(group2) - np.mean(group1)
        group1_sd = np.std(group1, ddof=1)  # Using Bessel's correction with ddof=1
    else:
        diff = np.median(group2) - np.median(group1)
        # use MAD (median_absolute_deviation)
        # scale: normal: divides by 0.67449 → multiplies by 1.4826 internally
        group1_sd = median_abs_deviation(group1, scale='normal')

    if group1_sd == 0:  # constant control group -> effect size undefined (avoid inf/nan divide)
        return np.nan
    return diff / group1_sd

def cohens_d(group1, group2, mean_or_median):
    """Compute absolute Cohen's d (|d|) for two independent samples."""
    # sample sizes
    n1, n2 = len(group1), len(group2)

    if mean_or_median == "Mean": 
        # mean difference
        diff = np.mean(group1) - np.mean(group2)
         # unbiased sample variances
        s1_2, s2_2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        # pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1)*s1_2 + (n2 - 1)*s2_2) / (n1 + n2 - 2))
    else:
        diff = np.median(group1) - np.median(group2)
        mad_1, mad_2 = median_abs_deviation(group1, scale="normal"), median_abs_deviation(group2, scale="normal")
        pooled_sd = np.sqrt(((n1 - 1) * mad_1**2 + (n2 - 1) * mad_2**2) /
                     (n1 + n2 - 2))
    if pooled_sd == 0:  # no pooled spread -> effect size undefined (avoid inf/nan divide)
        return np.nan
    return abs(diff / pooled_sd)

def create_opacity_mapping(groups, min_opacity=0.3, max_opacity=1.0,
                           na_value="N/A", na_opacity=0.15):
    """Opacity mapping for groups, evenly spaced in natural order, with ``na_value`` held out.

    Opacity is the only ordinal channel, so a ramp slot is a rank and missing data has no
    claim to one. ``na_value`` is pinned below ``min_opacity``, leaving the real levels the
    full spread. "N/A" is the loader's marker for a missing categorical (check_and_fix_df
    in src/dataset_io.py).

    Keep ``na_opacity`` a default argument, not a module constant: export_script's
    _extract_source copies this function's source and defaults but no module state.
    """
    groups = list(groups)
    real = [group for group in groups if group != na_value]
    real = natural_tuple_sort(real) if len(real) > 1 else real
    if len(real) == 1:
        mapping = {real[0]: max_opacity}
    else:
        # len(real) == 0 (an all-N/A column) yields an empty linspace and an empty map,
        # which is the honest answer: no real level to rank.
        opacity_values = np.linspace(min_opacity, max_opacity, len(real))
        mapping = {group: opacity_values[i] for i, group in enumerate(real)}
    if na_value in groups:
        # Added last so N/A sits at the end of the legend's opacity block --
        # add_point_legend_traces ranks entries by dict order.
        mapping[na_value] = na_opacity
    return mapping

def create_shape_mapping(groups):
    """Create shape mapping for groups using different plotly symbols, preserving natural order"""
    groups = list(groups)
    groups = natural_tuple_sort(groups) if len(groups) > 1 else groups
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
               'triangle-down', 'pentagon', 'hexagon', 'octagon', 'star', 'diamond-tall']
    return {group: symbols[i % len(symbols)] for i, group in enumerate(groups)}

def _palette_rgb(colormap, count):
    """``count`` (r, g, b) float triples from a named colormap, falling back to tab10.

    Shared with the subcolor palette (``create_subcolor_map``), which needs the same
    colours by the same rules; both are drawn in one figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    count = max(int(count), 1)
    try:
        if colormap in ["viridis", "plasma", "inferno", "magma", "cividis"]:
            # Continuous colormaps have no discrete entries to take, so sample them.
            cmap = plt.colormaps[colormap]
            colors = [cmap(0.5)] if count == 1 else [cmap(i / (count - 1)) for i in range(count)]
            return [(color[0], color[1], color[2]) for color in colors]
        return [tuple(color[:3]) for color in sns.color_palette(colormap, n_colors=count)]
    # ValueError for a name seaborn does not know; KeyError should the whitelist above
    # drift from plt.colormaps. Not ImportError: the imports sit above the try so
    # _extract_source carries them into the exported script, and the fallback needs
    # seaborn anyway.
    except (ValueError, KeyError):
        return [tuple(color[:3]) for color in sns.color_palette("tab10", n_colors=count)]

def create_color_map(groups, overlap_point, colormap="tab10"):
    # if points in the visualization is going to overlap, use a transparent color
    if overlap_point:
        alpha = 0.6 if len(groups) > 1 else 1.0
    else:
        alpha = 1.0

    palette = _palette_rgb(colormap, len(groups))
    color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {alpha})" for color in palette]
    color_map = {t: color_sequence[i] for i, t in enumerate(groups)}
    return color_map

def _prepare_group_data(df, group_by_cols, new_group_col_name, overlap_point=True, colormap="tab10"):
    """
    Prepares group data by creating a new group column, sorting unique groups,
    and generating a color map.
    Modifies the DataFrame in place by adding the new group column.
    """
    # Ensure group_by_cols is a list, even if a single string is passed
    if isinstance(group_by_cols, str):
        group_by_cols = [group_by_cols]

    if not group_by_cols: # Handle empty list for group_by_cols
        # Create a dummy group if no columns are provided for grouping
        df[new_group_col_name] = "all_data"
    else:
        df[new_group_col_name] = df[group_by_cols].astype(str).agg('::'.join, axis=1)
    unique_groups = df[new_group_col_name].unique()
    unique_groups = natural_tuple_sort(unique_groups, delimiter='::')
    color_map = create_color_map(unique_groups, overlap_point=overlap_point, colormap=colormap)
    return unique_groups, color_map

def _calculate_effect_size(group1_data, group2_data, method: str, mean_or_median):
    """
    Calculates the effect size between two groups using the specified method.
    """
    if group1_data.empty or group2_data.empty:
        return None
    if method == "Glass's Delta":
        return glass_delta(group1_data, group2_data, mean_or_median)
    elif method == "Absolute Cohen's d":
        # Ensure cohens_d function is available and handles data appropriately
        return cohens_d(group1_data, group2_data, mean_or_median)
    else:
        st.warning(f"Unsupported effect size method: {method}")
        return None

def _compute_bracket_position(x_start, x_end, region_max_y, positioning_metrics, drawn_annotations_list):
    """Compute bracket y-position with collision detection. Framework-agnostic (pure math).

    Args:
        x_start, x_end: Horizontal span of the bracket.
        region_max_y: Maximum y value in the spanned data region.
        positioning_metrics: Dict with offset_from_data_abs, vertical_spacing_abs,
            bracket_vertical_length_abs, text_offset_from_bracket_abs,
            text_height_allowance_for_collision_abs.
        drawn_annotations_list: List of previously drawn annotations (modified in-place).

    Returns:
        (y_bracket_top, y_text_center, bracket_length) or None if positioning fails.
    """
    offset_abs = positioning_metrics['offset_from_data_abs']
    spacing_abs = positioning_metrics['vertical_spacing_abs']
    bracket_h = positioning_metrics['bracket_vertical_length_abs']
    text_offset = positioning_metrics['text_offset_from_bracket_abs']
    text_h = positioning_metrics['text_height_allowance_for_collision_abs']

    y_candidate = region_max_y + offset_abs

    for _ in range(50):
        y_text_center = y_candidate + text_offset + (text_h / 2)
        y_bottom = y_candidate - bracket_h
        y_top = y_text_center + (text_h / 2)

        collision = False
        for ann in drawn_annotations_list:
            x_overlap = max(x_start, ann['x_start']) < min(x_end, ann['x_end'])
            if x_overlap:
                y_overlap = max(y_bottom, ann['y_bottom']) < min(y_top, ann['y_top'])
                if y_overlap:
                    y_candidate = ann['y_top'] + spacing_abs
                    collision = True
                    break

        if not collision:
            drawn_annotations_list.append({
                'x_start': x_start, 'x_end': x_end,
                'y_bottom': y_bottom, 'y_top': y_top,
                'y_bracket_draw': y_candidate, 'y_text_draw': y_text_center
            })
            return y_candidate, y_text_center, bracket_h

    # Fallback
    y_text_center = y_candidate + text_offset + (text_h / 2)
    y_bottom = y_candidate - bracket_h
    y_top = y_text_center + (text_h / 2)
    drawn_annotations_list.append({
        'x_start': x_start, 'x_end': x_end,
        'y_bottom': y_bottom, 'y_top': y_top,
        'y_bracket_draw': y_candidate, 'y_text_draw': y_text_center
    })
    return y_candidate, y_text_center, bracket_h


def _annotate_single_effect_size(fig, pair_strings, effect_size_value, compare_groups_list,
                                 drawn_annotations_list, positioning_metrics,
                                 original_df, data_column_name, group_column_name_in_df,
                                 overall_min_y_val, data_range_y, annotation_color, position_map=None,
                                 star_text: str = None, show_effect_size: bool = True):
    """
    Adds a single effect size annotation (bracket and text) to the figure,
    handling y-positioning and collision detection.
    """
    if position_map is not None:
        x_positions = [position_map[pair_strings[0]], position_map[pair_strings[1]]]
        x_start_new = min(x_positions)
        x_end_new = max(x_positions)
        spanned_group_names = [group for group, pos in position_map.items()
                             if pos >= x_start_new and pos <= x_end_new]
    else:
        x_indices = [compare_groups_list.index(pair_strings[0]), compare_groups_list.index(pair_strings[1])]
        x_start_new = min(x_indices)
        x_end_new = max(x_indices)
        spanned_group_names = compare_groups_list[x_start_new : x_end_new + 1]
    df_in_span = original_df[original_df[group_column_name_in_df].isin(spanned_group_names)]
    current_region_max_y = df_in_span[data_column_name].max(skipna=True)

    if pd.isna(current_region_max_y):
        current_region_max_y = overall_min_y_val
        if pd.isna(current_region_max_y):
            current_region_max_y = 0

    result = _compute_bracket_position(
        x_start_new, x_end_new, current_region_max_y,
        positioning_metrics, drawn_annotations_list
    )
    if result is None:
        return

    final_y_bracket_top, final_y_text_annotation_center, bracket_vertical_length_abs = result

    # Add bracket lines (Plotly)
    fig.add_shape(
        type="line", x0=x_start_new, y0=final_y_bracket_top,
        x1=x_end_new, y1=final_y_bracket_top,
        line=dict(width=1.5, color=annotation_color)
    )
    for x_pos_single in [x_start_new, x_end_new]:
        fig.add_shape(
            type="line", x0=x_pos_single, y0=final_y_bracket_top,
            x1=x_pos_single, y1=final_y_bracket_top - bracket_vertical_length_abs,
            line=dict(width=1.5, color=annotation_color)
        )
    # Build annotation text
    if show_effect_size and star_text:
        annotation_text = f"{effect_size_value:.2f}{star_text}"
    elif show_effect_size and not star_text:
        annotation_text = f"Δ={effect_size_value:.2f}"
    elif (not show_effect_size) and star_text:
        annotation_text = star_text
    else:
        annotation_text = ""

    fig.add_annotation(
        x=(x_start_new + x_end_new) / 2, y=final_y_text_annotation_center,
        text=annotation_text, showarrow=False, font=dict(size=12, color=annotation_color),
        align="center"
    )

def _add_effect_size_annotations(fig, df, selected_var, compare_groups, group_col_name, all_possible_pairs, annotation_color, effect_size_method="None", mean_or_median=None, position_map=None, selected_pairs=None, threshold=None, statistical_test: str = "None", global_data_range=None):
    """
    Adds effect size annotations to the figure.
    Manages selection of pairs, calculation of effect sizes, and calls annotation plotting.
    
    Args:
        position_map: Optional dict mapping group names to actual x-positions for separate sections
        selected_pairs: Optional pre-selected pairs to avoid showing the widget again
        threshold: Optional pre-set threshold to avoid showing the widget again
        global_data_range: Optional tuple (global_min, global_max) for consistent spacing across sections
    """
    if not all_possible_pairs:
        return

    # Only show widget if pairs aren't pre-selected
    if selected_pairs is None:
        selected_pairs = comparison_pair_widget(all_possible_pairs)

    # Precompute star texts if a statistical test is requested
    pair_to_star = {}
    if selected_pairs and statistical_test != "None":
        for pair in selected_pairs:
            group1 = df[df[group_col_name] == pair[0]][selected_var].dropna()
            group2 = df[df[group_col_name] == pair[1]][selected_var].dropna()
            if group1.empty or group2.empty:
                continue
            equal_var = (statistical_test == "Independent t-test")
            try:
                _, pval = ttest_ind(group1, group2, equal_var=equal_var, nan_policy='omit')
            except Exception:
                pval = np.nan
            # Map p-value to stars
            if pd.isna(pval):
                stars = ""
            elif pval <= 0.0001:
                stars = "****"
            elif pval <= 0.001:
                stars = "***"
            elif pval <= 0.01:
                stars = "**"
            elif pval <= 0.05:
                stars = "*"
            else:
                stars = ""
            pair_to_star[pair] = stars

    if selected_pairs and effect_size_method != "None":
        drawn_annotations = []  # List to store details of drawn annotations for collision detection

        # --- Define vertical spacing parameters (relative to data range) ---
        # Use global_data_range if provided for consistent spacing across separate sections
        if global_data_range is not None:
            global_min_y, global_max_y = global_data_range
        else:
            global_max_y = df[selected_var].max(skipna=True)
            global_min_y = df[selected_var].min(skipna=True)

        if pd.isna(global_max_y) or pd.isna(global_min_y) or len(df[selected_var].dropna()) < 2:
            data_range_y = 1  # Default if overall data is all NaN or not enough points
        else:
            data_range_y = global_max_y - global_min_y

        if data_range_y == 0:  # Avoid division by zero or if all values are the same
            data_range_y = 1 # Use a nominal range to prevent zero spacing

        # Calculate absolute positioning metrics once
        positioning_metrics = {
            'offset_from_data_abs': 0.05 * data_range_y,
            'vertical_spacing_abs': 0.08 * data_range_y,
            'bracket_vertical_length_abs': 0.03 * data_range_y,
            'text_offset_from_bracket_abs': 0.03 * data_range_y,
            'text_height_allowance_for_collision_abs': 0.04 * data_range_y
        }

        # Sort pairs for consistent annotation order and simpler collision logic
        # Sorting key ensures that pairs are processed from left-to-right, and shorter spans before longer ones if they start at the same point.
        if position_map is not None:
            # Use actual positions for sorting when position_map is provided
            sorted_pairs = sorted(selected_pairs,
                                  key=lambda p: (min(position_map[p[0]], position_map[p[1]]),
                                                 max(position_map[p[0]], position_map[p[1]])))
        else:
            # Use group indices for sorting in regular plots
            sorted_pairs = sorted(selected_pairs,
                                  key=lambda p: (min(compare_groups.index(p[0]), compare_groups.index(p[1])),
                                                 max(compare_groups.index(p[0]), compare_groups.index(p[1]))))

        # --- Threshold input based on selected method ---
        if threshold is None:
            threshold = 0.0
            threshold_key_suffix = selected_var
            if effect_size_method == "Glass's Delta":
                threshold = st.number_input("Glass's Delta Threshold", value=0.7, min_value=0.0, max_value=3.0, step=0.05, 
                                            key=f"glass_delta_thresh_{threshold_key_suffix}")
            elif effect_size_method == "Absolute Cohen's d":
                threshold = st.number_input(
                    "Absolute Cohen's d threshold",
                    value=0.5,
                    min_value=0.0,
                    max_value=3.0,
                    step=0.1,
                    key=f"cohens_d_thresh_{threshold_key_suffix}",
                )

        for pair in sorted_pairs:
            group1_data = df[df[group_col_name] == pair[0]][selected_var].dropna()
            group2_data = df[df[group_col_name] == pair[1]][selected_var].dropna()

            if group1_data.empty or group2_data.empty:
                st.warning(f"Skipping pair {pair} due to empty data for one or both groups.")
                continue
            effect_size_value = _calculate_effect_size(group1_data, group2_data, effect_size_method, mean_or_median)

            if effect_size_value is not None and abs(effect_size_value) >= threshold:
                stars = pair_to_star.get(pair)
                _annotate_single_effect_size(
                    fig=fig,
                    pair_strings=pair,
                    effect_size_value=effect_size_value,
                    compare_groups_list=compare_groups,
                    drawn_annotations_list=drawn_annotations, # This list is modified in-place
                    positioning_metrics=positioning_metrics,
                    original_df=df,
                    data_column_name=selected_var,
                    group_column_name_in_df=group_col_name,
                    overall_min_y_val=global_min_y, # Pass global_min_y for fallback
                    data_range_y=data_range_y, # Pass data_range_y for context if needed inside, though metrics are now absolute
                    annotation_color=annotation_color,
                    position_map=position_map,
                    star_text=stars,
                    show_effect_size=True
                )


        # No explicit "else" for unsupported methods here as _calculate_effect_size handles the warning,
        # and effect_size_value would be None, thus skipping annotation.

    # If only statistical tests were requested (no effect size), add star-only annotations
    if selected_pairs and effect_size_method == "None" and statistical_test != "None":
        drawn_annotations = []

        # Use global_data_range if provided for consistent spacing across separate sections
        if global_data_range is not None:
            global_min_y, global_max_y = global_data_range
        else:
            global_max_y = df[selected_var].max(skipna=True)
            global_min_y = df[selected_var].min(skipna=True)
        if pd.isna(global_max_y) or pd.isna(global_min_y) or len(df[selected_var].dropna()) < 2:
            data_range_y = 1
        else:
            data_range_y = global_max_y - global_min_y
        if data_range_y == 0:
            data_range_y = 1

        positioning_metrics = {
            'offset_from_data_abs': 0.05 * data_range_y,
            'vertical_spacing_abs': 0.08 * data_range_y,
            'bracket_vertical_length_abs': 0.03 * data_range_y,
            'text_offset_from_bracket_abs': 0.03 * data_range_y,
            'text_height_allowance_for_collision_abs': 0.04 * data_range_y
        }

        if position_map is not None:
            sorted_pairs = sorted(selected_pairs,
                                  key=lambda p: (min(position_map[p[0]], position_map[p[1]]),
                                                 max(position_map[p[0]], position_map[p[1]])))
        else:
            sorted_pairs = sorted(selected_pairs,
                                  key=lambda p: (min(compare_groups.index(p[0]), compare_groups.index(p[1])),
                                                 max(compare_groups.index(p[0]), compare_groups.index(p[1]))))

        for pair in sorted_pairs:
            stars = pair_to_star.get(pair, "")
            # Skip if both groups empty or no stars and you prefer not to show ns; we'll annotate even if empty string to keep brackets consistent if desired
            _annotate_single_effect_size(
                fig=fig,
                pair_strings=pair,
                effect_size_value=0.0,
                compare_groups_list=compare_groups,
                drawn_annotations_list=drawn_annotations,
                positioning_metrics=positioning_metrics,
                original_df=df,
                data_column_name=selected_var,
                group_column_name_in_df=group_col_name,
                overall_min_y_val=global_min_y,
                data_range_y=data_range_y,
                annotation_color=annotation_color,
                position_map=position_map,
                star_text=stars,
                show_effect_size=False
            )

@st.cache_data(show_spinner=False)
def _find_best_gmm(data, max_components=3, min_weight_threshold=0.1, random_state=42):
    """
    Finds the best Gaussian Mixture Model (GMM) based on BIC, subject to constraints.

    Args:
        data (np.ndarray): Input data (1D or 2D).
        max_components (int): Maximum number of components to try.
        min_weight_threshold (float): Minimum weight for a component to be considered valid.
        random_state (int): Random state for GMM initialization.

    Returns:
        sklearn.mixture.GaussianMixture or None: The best GMM found, or None if no valid model.
    """
    if data.ndim == 1:
        data_reshaped = data.reshape(-1, 1)
    elif data.ndim == 2:
        data_reshaped = data
    else:
        raise ValueError("Input data must be 1D or 2D.")

    best_gmm = None
    lowest_bic = np.inf

    # Pin the BLAS/OpenMP pool to one thread: a 1-D or 2-D feature gives each thread too
    # little work to pay for synchronising them, so the sweep runs ~9x faster single-
    # threaded. Fitted means then differ by ~1e-12 from the multi-threaded reduction
    # order, which can only matter on a near-tie in BIC between two component counts.
    # Imported inside the function so the source stays self-contained when
    # export_script.py inlines it via inspect.getsource().
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k, random_state=random_state)
            gmm.fit(data_reshaped)
            bic = gmm.bic(data_reshaped)

            if gmm.weights_.min() >= min_weight_threshold:
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
    return best_gmm

def natural_key(s):
    """Return a tuple for natural sorting: (is_number, number or string)"""
    # Match decimal or integer numbers (including negative numbers)
    match = re.search(r'([-+]?\d*\.\d+|\d+)', s)
    if match:
        return (0, float(match.group(1)), s)
    else:
        return (1, s)

def tuple_natural_key(tup):
    """Return a tuple of natural keys for each element in the tuple."""
    return tuple(natural_key(str(x)) for x in tup)

def natural_tuple_sort(strings, delimiter='::'):
    """
    Sort a list of delimited strings using natural sort for each column.
    Values must already be strings: check_and_fix_df astype(str)s every categorical column
    at load, and the callers that group on something else go through dropna/astype(str)
    first. The split runs before tuple_natural_key, so its str() cannot cover for a
    non-string here.
    :param strings: list of strings to sort
    :param delimiter: delimiter to split columns (default '::')
    :return: sorted list of strings
    """
    return sorted(strings, key=lambda x: tuple_natural_key(x.split(delimiter)))

def _sorted_levels(values):
    """The distinct values of a Series, in natural-sort order."""
    return natural_tuple_sort(values.unique())

def _channel_groups_and_map(df, column, mapper):
    """Shared body of the shape and opacity channels: their levels, and ``mapper``'s map.

    Keep the raw values and the ``dropna()``: these maps' keys are looked up against a
    ``df.groupby`` on the raw column in ``get_point_visual_mappings``, so each key must
    equal a real groupby key. Stringifying makes every lookup miss; folding nulls to
    "N/A" invents a level no group matches, since groupby drops NaN keys. To include
    nulls, fill the column before the groupby rather than relabelling levels after.
    """
    if not column or column not in df.columns:
        return [], None
    groups = _sorted_levels(df[column].dropna())
    return groups, mapper(groups)

def create_opacity_groups_and_map(df, opacity_by_col):
    """Sorted unique groups of ``opacity_by_col`` and their opacity map."""
    return _channel_groups_and_map(df, opacity_by_col, create_opacity_mapping)

def create_shape_groups_and_map(df, shape_by_col):
    """Sorted unique groups of ``shape_by_col`` and their symbol map."""
    return _channel_groups_and_map(df, shape_by_col, create_shape_mapping)

def interleave_point_batches(index_by_level, num_batches=15, random_seed=42):
    """Split each level's point indices into batches and cycle through the levels.

    Returns ``[(level, indices), ...]`` in the order the traces should be added.

    Plotly paints later traces over earlier ones, so emitting one trace per level draws
    each level entirely on top of the previous — in a sina plot, where the levels share
    one jittered x band, the last level ends up systematically the most visible.
    Cycling through the levels a batch at a time spreads that bias evenly. All batches
    of a level keep its legendgroup, so the legend still holds one entry per level.

    Indices are shuffled within a level before batching, so a batch is a sample across
    the level rather than a contiguous run of rows: contiguous batches would be
    correlated with whatever the frame happens to be sorted by, which puts the same
    bias back in a different guise. The shuffle is a fixed-seed permutation of the
    index array, so it reorders only the paint order — every point keeps the x it was
    already assigned.

    Batches hold at least ~5 points, matching add_interleaved_points_trace, below which
    the extra traces cost more than the interleaving buys.
    """
    import math

    rng = np.random.default_rng(random_seed)
    batched = []
    for level, indices in index_by_level.items():
        indices = np.asarray(indices)
        if not len(indices):
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        count = min(num_batches, max(1, len(shuffled) // 5))
        size = math.ceil(len(shuffled) / count)
        batched.append((level, [shuffled[i:i+size] for i in range(0, len(shuffled), size)]))

    ordered = []
    for round_index in range(max((len(b) for _level, b in batched), default=0)):
        for level, batches in batched:
            if round_index < len(batches):
                ordered.append((level, batches[round_index]))
    return ordered

def create_subcolor_map(df, subcolor_by, group_col, color_groups, engine="plotly",
                                  colormap="tab10"):
    """Colour a nested categorical column, one colour per value across the whole figure.

    Returns ``{value: colour}``, or None when the channel is off.

    Colour encodes the nested value while the colour group keeps its x position. The map
    is global -- one colour per distinct value for the whole figure -- so a value in
    several groups is the same colour in each and one legend entry serves it.

    The keys are in natural-sort order and are the figure's value list, which callers
    iterate to decide what to draw in a group's x band; a value absent from a group
    yields an empty mask the batcher skips.

    Colours come from ``make_palette`` seeded by the first entry of ``colormap``, not from
    the colormap directly: entries are scored composited at the alpha points are drawn
    with, and generating for the requested count avoids seaborn cycling a qualitative
    palette past its length into duplicates. Nulls fold to "N/A" (check_and_fix_df in
    src/dataset_io.py) so those rows still plot.

    ``engine`` follows format_group_label: "plotly" returns "rgba(...)" strings, "mpl"
    returns (r, g, b) tuples for the exported Matplotlib script.
    """
    from src.vis.subcolor_palette import make_palette_cached

    if (not subcolor_by or subcolor_by not in df.columns
            or group_col not in df.columns or not len(color_groups)):
        return None

    values = df[subcolor_by].fillna("N/A").astype(str)
    groups = df[group_col].astype(str)
    # Restricted to the groups the figure actually draws, not every row in the frame, so
    # a group filtered out upstream cannot spend a colour nothing displays. Equivalent to
    # unioning each group's own values, which is how this was written when the per-group
    # breakdown was also returned.
    in_figure = groups.isin([str(group) for group in color_groups])
    # Every distinct value in the figure, not per group: that is what makes the colour
    # mean the value itself, the same one wherever the value appears.
    all_values = _sorted_levels(values[in_figure])
    if not len(all_values):
        return None
    # alpha=0.7 is not a default worth inheriting silently -- it has to match the
    # opacity the points are actually drawn at (univar.feature_comparison_plot, and
    # scatter_with_encodings' base_alpha in the export), because make_palette scores
    # candidates on how they look composited at that alpha over the background. Passed
    # explicitly so changing one side shows up as disagreeing with the other.
    palette = make_palette_cached(_palette_rgb(colormap, 1)[0], len(all_values), alpha=0.7)

    colour_of = {}
    for index, value in enumerate(all_values):
        red, green, blue = palette[index]
        if engine == "mpl":
            colour_of[value] = (red, green, blue)
        else:
            # round(), not int(): int() truncates, so a channel that should be 31
            # arrives as 30 whenever the float sits a hair below it -- landing a byte off
            # the same palette's unrounded floats, which is what the Matplotlib export
            # draws from. check_subcolor.py pins the two engines against each other.
            # Opaque here: the points carry their own alpha through marker.opacity, and
            # baking a second one in would compound with it.
            colour_of[value] = (
                f"rgba({round(red * 255)}, {round(green * 255)}, {round(blue * 255)}, 1.0)"
            )

    return colour_of

def format_group_label(group, count=None, show_count=False, engine="plotly"):
    """Legend label for a color group.

    When ``show_count`` is enabled and a count is given, the count is placed on a
    second line below the group name, e.g. "Control" with "n=42" beneath it.

    ``engine`` selects the markup, following ``format_feature_label``
    (src/feature_labels.py): ``"plotly"`` (the app, default) breaks the line with
    ``<br>`` and shrinks the count with a relative ``em`` so it tracks the legend font
    size; ``"mpl"`` (the exported Matplotlib script) uses a plain newline, since
    Matplotlib legend labels take no markup and cannot mix sizes within one entry.
    The wording is written once here so the exported legend reads the same as the
    screen — export_script.py inlines this function rather than reproducing the text.
    """
    label = str(group)
    if show_count and count is not None:
        if engine == "mpl":
            label = f"{label}\nn={count}"
        else:
            label = f"{label}<br><span style='font-size: 0.75em'>n={count}</span>"
    return label

def get_point_visual_mappings(
    df,
    color_by=None,
    shape_by=None,
    opacity_by=None,
    separate_by=None,
    group_col_name="group",
    overlap_point=True,
    colormap="tab10"
):
    """
    General helper for point-based visualizations to handle color_by, shape_by, and opacity_by.
    Returns grouped DataFrame, color_map, shape_map, opacity_map, and group keys.
    """
    # Prepare color grouping
    color_by = color_by or []
    if isinstance(color_by, str):
        color_by = [color_by]
    unique_color_groups, color_map = _prepare_group_data(df, color_by, group_col_name, overlap_point=overlap_point, colormap=colormap)
    # Prepare shape mapping
    _, shape_map = create_shape_groups_and_map(df, shape_by)
    # Prepare opacity mapping
    _, opacity_map = create_opacity_groups_and_map(df, opacity_by)
    if separate_by and separate_by.strip() != "" and separate_by in df.columns:
        separate_groups = natural_tuple_sort(df[separate_by].dropna().unique())
    else:
        separate_groups = None

    # Build ordered group keys from mapping dicts
    from itertools import product
    color_keys = list(color_map.keys())
    shape_keys = list(shape_map.keys()) if shape_map else [None]
    opacity_keys = list(opacity_map.keys()) if opacity_map else [None]
    separate_keys = list(separate_groups) if separate_groups is not None else [None]

    # Which of the four channels actually select rows, and which slot each occupies
    # in the group_key tuple below. Inactive channels keep a None placeholder in the
    # key (callers unpack it positionally) but take no part in the grouping.
    key_cols = [group_col_name]
    key_positions = [0]
    if shape_by and shape_by in df.columns:
        key_cols.append(shape_by)
        key_positions.append(1)
    if opacity_by and opacity_by in df.columns:
        key_cols.append(opacity_by)
        key_positions.append(2)
    if separate_by and separate_by in df.columns:
        key_cols.append(separate_by)
        key_positions.append(3)

    def ordered_group_iter():
        # Group once, then walk the Cartesian product and look each combination up, so
        # the frame is scanned once rather than once per combination (most of which match
        # no rows). Walking `product` rather than the groupby is what fixes trace, legend
        # and legendrank order. groupby drops NaN keys, and the shape/opacity/separate key
        # lists come from `.dropna().unique()`, so a NaN key is never looked up.
        groups = {}
        for key, group_df in df.groupby(key_cols, sort=False, observed=True):
            if not isinstance(key, tuple):
                key = (key,)
            groups[key] = group_df

        for group_key in product(color_keys, shape_keys, opacity_keys, separate_keys):
            group_df = groups.get(tuple(group_key[pos] for pos in key_positions))
            if group_df is not None and len(group_df) > 0:
                yield group_key, group_df

    # Return an iterable like a groupby object, but ordered as specified
    grouped_ordered = ordered_group_iter()

    return grouped_ordered, color_map, shape_map, opacity_map, separate_groups

def apply_plot_styling(fig, point_size, axis_label_size, legend_size):
    """Apply consistent styling to plotly figures.

    Data-point markers are sized by ``point_size``, while legend markers are
    sized by ``legend_size`` (matching the legend font). Plotly normally ties
    the legend marker size to each trace's ``marker.size`` (``itemsizing='trace'``),
    so we decouple the two by hiding the legend on real data traces and
    adding "ghost" traces (no plotted points) whose ``marker.size`` is set
    to ``legend_size``. Existing ghost legend traces (e.g. shape/opacity
    entries) are also resized to ``legend_size``.
    """
    # Names of traces that should keep their original marker sizes
    skip_trace_names = {'Lifetime Markers'}

    # Update marker sizes for all scatter and box traces
    for trace in fig.data:
        # Skip traces that should maintain their original sizes
        if hasattr(trace, 'name') and trace.name in skip_trace_names:
            continue
        if hasattr(trace, 'marker') and trace.marker:
            # 'scattergl' as well as 'scatter': point_trace_class swaps in the WebGL trace
            # above WEBGL_POINT_THRESHOLD, and a gate that named only 'scatter' would walk
            # straight past those traces and silently leave Point Size doing nothing.
            if trace.type in ('scatter', 'scattergl') or trace.type == 'box' and trace.marker:
                trace.marker.size = point_size

    # Update annotation font sizes to match axis label size
    if fig.layout.annotations:
        for annotation in fig.layout.annotations:
            if annotation.font:
                annotation.font.size = axis_label_size
            else:
                annotation.font = dict(size=axis_label_size)

    # Theme-aware hover tooltips: black-on-white in light mode, white-on-dark in dark.
    theme_color = get_context_theme_color()

    # Update layout with axis and legend font sizes
    fig.update_layout(
        xaxis=dict(
            title=dict(font=dict(size=axis_label_size)),
            tickfont=dict(size=axis_label_size-2)
        ),
        yaxis=dict(
            title=dict(font=dict(size=axis_label_size)),
            tickfont=dict(size=axis_label_size-2)
        ),
        legend=dict(
            font=dict(size=legend_size)
        ),
        hoverlabel=dict(
            bgcolor="white" if theme_color == "black" else "rgb(30, 30, 30)",
            font=dict(color=theme_color),
            bordercolor=theme_color,
        )
    )

    # Replace each color-trace legend entry with a ghost trace whose marker
    # size equals legend_size, and resize any pre-existing ghost legend
    # traces (shape/opacity entries) to match.
    ghost_traces = []
    seen_legendgroups = set()
    for trace in fig.data:
        # WebGL data traces must reach this block too, or a large figure keeps its
        # trace-tied legend markers and never gets the decoupled ghost entries. The ghosts
        # themselves stay go.Scatter below: they draw a single None point, so SVG is right
        # for a legend swatch and costs no WebGL trace.
        if trace.type not in ('scatter', 'scattergl'):
            continue
        if hasattr(trace, 'name') and trace.name in skip_trace_names:
            continue
        if not getattr(trace, 'showlegend', True):
            continue

        x_attr = getattr(trace, 'x', None)
        x_data = list(x_attr) if x_attr is not None else []
        is_ghost = len(x_data) > 0 and all(v is None for v in x_data)
        if is_ghost:
            if trace.marker:
                trace.marker.size = legend_size
            continue

        legendgroup = trace.legendgroup or trace.name
        if legendgroup in seen_legendgroups:
            trace.showlegend = False
            continue
        seen_legendgroups.add(legendgroup)

        marker_color = trace.marker.color if trace.marker else None
        marker_symbol = trace.marker.symbol if trace.marker else 'circle'
        if isinstance(marker_symbol, (list, tuple, np.ndarray)):
            marker_symbol = marker_symbol[0] if len(marker_symbol) > 0 else 'circle'
        marker_opacity = trace.marker.opacity if trace.marker else 1.0
        if isinstance(marker_opacity, (list, tuple, np.ndarray)):
            marker_opacity = marker_opacity[0] if len(marker_opacity) > 0 else 1.0

        ghost_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name=trace.name,
            legendgroup=trace.legendgroup,
            marker=dict(
                color=marker_color,
                symbol=marker_symbol,
                opacity=marker_opacity,
                size=legend_size,
            ),
            showlegend=True,
            legendrank=getattr(trace, 'legendrank', None),
            hoverinfo='skip',
        ))
        trace.showlegend = False

    for ghost in ghost_traces:
        fig.add_trace(ghost)

    return fig

def _estimate_density_1d(y_values, bw_method='scott'):
    """
    Estimate the density of y_values using KDE. Returns a function that maps y to density.

    Degenerate inputs have no well-defined KDE and make gaussian_kde raise
    LinAlgError on a singular covariance matrix: fewer than 2 finite points, or zero
    variance (a constant column — e.g. an all-zero feature). In those cases return a
    zero-density fallback so callers still render points/box without a density shape,
    instead of the LinAlgError aborting the whole plot.
    """
    y_values = np.asarray(y_values, dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    zero_density = lambda y: np.zeros_like(np.asarray(y, dtype=float))
    if len(y_values) < 2 or np.ptp(y_values) == 0:
        return zero_density
    try:
        return gaussian_kde(y_values, bw_method=bw_method)
    except np.linalg.LinAlgError:
        # Near-singular covariance (e.g. all-but-one identical) can still slip past
        # the ptp check; fall back rather than crash.
        return zero_density

def _density_at_points(y_values, grid_size=1024, bw_method='scott'):
    """Density at each of ``y_values``, without evaluating the KDE at every point.

    ``gaussian_kde.evaluate`` costs O(n_train x n_eval), so asking it for the density at
    its own training points is O(n^2) -- 60 s for a single 113k-point group. Evaluating on
    a small grid and interpolating back is O(n x grid_size) and takes that to under a
    second.

    The grid deliberately unions a uniform spacing with a quantile spacing. Uniform alone
    is accurate through the bulk but collapses on heavy-tailed features, where nearly every
    grid point lands in empty space and the dense core goes unresolved -- measured at
    several pixels of drift in the sina jitter on a Cauchy-distributed feature, which is
    visible. Quantile spacing puts resolution wherever the points actually are. Together
    they stayed sub-pixel on every distribution shape tested (normal, bimodal, lognormal,
    exponential, spiky mixture, Cauchy).

    Returns zeros for the degenerate inputs ``_estimate_density_1d`` rejects, which is what
    its zero-density fallback produced when it was called per point.
    """
    y_values = np.asarray(y_values, dtype=float)
    finite = y_values[np.isfinite(y_values)]
    # Same guards as _estimate_density_1d, checked here too so the grid below never
    # degenerates to a single repeated value (np.interp needs an increasing xp).
    if len(finite) < 2 or np.ptp(finite) == 0:
        return np.zeros_like(y_values)

    kde = _estimate_density_1d(y_values, bw_method=bw_method)
    if not isinstance(kde, gaussian_kde):
        # _estimate_density_1d hit its LinAlgError fallback and returned zero_density.
        return np.zeros_like(y_values)

    half = max(2, grid_size // 2)
    grid = np.unique(np.concatenate([
        np.linspace(finite.min(), finite.max(), half),
        np.quantile(finite, np.linspace(0, 1, half)),
    ]))
    return np.interp(y_values, grid, kde(grid))

def add_interleaved_points_trace(
    fig,
    grouped,
    color_map,
    shape_map,
    opacity_map,
    axis_labels,
    text_col,
    customdata_col,
    shape_by=None,
    opacity_by=None,
    hovertemplate="<b>%{text}</b>",
    random_seed=42,
    num_batches=15,
    show_counts=False
):
    """
    Adds multiple interleaved traces per color group to minimize occlusion
    while maintaining legend interactivity.
    
    Strategy: Split each color group into N batches, then add traces in an 
    interleaved pattern (batch1 of each color, batch2 of each color, etc.).
    All batches of the same color share a legendgroup so clicking the legend
    entry shows/hides all batches together.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to add traces to
    grouped : iterable
        Iterator of (group_key, group_df) tuples from get_point_visual_mappings
    color_map : dict
        Mapping of color_group to color values
    shape_map : dict or None
        Mapping of shape_group to symbol values
    opacity_map : dict or None
        Mapping of opacity_group to opacity values
    axis_labels : list
        List of [x_label, y_label] for the axes
    text_col : str
        Column name for text/hover labels
    customdata_col : str
        Column name for customdata
    shape_by : str or None
        Column name used for shape grouping
    opacity_by : str or None
        Column name used for opacity grouping
    hovertemplate : str
        Hover template string
    random_seed : int or None
        Seed for the per-color shuffle. Defaults to a fixed value so the
        interleave order is reproducible across reruns; pass None for a
        nondeterministic order.
    num_batches : int
        Number of batches to split each color group into (default: 15)
    
    Returns:
    --------
    scatter_cls : type
        The trace class used for the points (``go.Scatter`` or ``go.Scattergl``). Overlays
        drawn on the same axes after this call must use it too -- see the note at the
        return statement.
    """
    import math
    import random

    # Local RNG so the per-color shuffle is reproducible run-to-run without
    # touching (or being perturbed by) the global `random` state. Seeded by
    # default; pass random_seed=None for a nondeterministic order.
    rng = random.Random(random_seed)

    # Collect each colour's points as columns rather than one dict per cell. A single
    # colour can receive points from several (shape, opacity, separate) subgroups, so
    # gather the chunks in iteration order and concatenate once at the end.
    chunks_by_color = {}
    for group_key, group_df in grouped:
        chunks_by_color.setdefault(group_key[0], []).append(
            (group_df, group_key[1], group_key[2])
        )

    points_by_color = {}
    for color_group, chunks in chunks_by_color.items():
        lengths = [len(chunk_df) for chunk_df, _, _ in chunks]
        points_by_color[color_group] = {
            'x': np.concatenate([c[axis_labels[0]].to_numpy() for c, _, _ in chunks]),
            'y': np.concatenate([c[axis_labels[1]].to_numpy() for c, _, _ in chunks]),
            'text': np.concatenate([c[text_col].to_numpy() for c, _, _ in chunks]),
            'customdata': np.concatenate([c[customdata_col].to_numpy() for c, _, _ in chunks]),
            # The shape and opacity groups are constant within a chunk (they come from
            # the group key), so repeat them instead of reading one per row. Object
            # dtype is what keeps an inactive channel's None group as None rather than
            # coercing it to nan.
            'shape_group': np.repeat(
                np.array([s for _, s, _ in chunks], dtype=object), lengths),
            'opacity_group': np.repeat(
                np.array([o for _, _, o in chunks], dtype=object), lengths),
        }

    # Shuffle points within each color group. An index permutation is shuffled rather
    # than the rows, so `rng` is drawn from in the same sequence either way.
    for color_group, columns in points_by_color.items():
        order = list(range(len(columns['x'])))
        rng.shuffle(order)
        order = np.asarray(order, dtype=int)
        points_by_color[color_group] = {k: v[order] for k, v in columns.items()}

    # Split each color group into batches. Batches are (start, end) bounds into the
    # colour's arrays; slicing at trace-build time avoids copying the points twice.
    batches_by_color = {}
    for color_group, columns in points_by_color.items():
        num_points = len(columns['x'])
        # Adjust num_batches if there are fewer points
        actual_batches = min(num_batches, max(1, num_points // 5))  # At least 5 points per batch
        batch_size = math.ceil(num_points / actual_batches)

        batches = []
        for i in range(actual_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            if start_idx < num_points:
                batches.append((start_idx, end_idx))

        batches_by_color[color_group] = batches

    # Get color groups in their natural order
    color_group_order = {color_group: i for i, color_group in enumerate(color_map.keys())}
    sorted_color_groups = sorted(points_by_color.keys(), key=lambda x: color_group_order.get(x, float('inf')))

    # Add <extra></extra> to hovertemplate to hide trace name from hover
    if "<extra>" not in hovertemplate:
        hover_without_trace = hovertemplate + "<extra></extra>"
    else:
        hover_without_trace = hovertemplate

    # Find maximum number of batches across all colors
    max_batches = max(len(batches) for batches in batches_by_color.values())

    # One renderer for every batch of every colour, decided from the figure's total. Chosen
    # once rather than per batch because Plotly paints every WebGL trace beneath every SVG
    # one: a mixed figure would stack its colours by renderer and undo the interleaving this
    # function exists to produce.
    scatter_cls = point_trace_class(sum(len(c['x']) for c in points_by_color.values()))

    # Interleave batches: add batch i from each color before moving to batch i+1
    for batch_idx in range(max_batches):
        for color_group in sorted_color_groups:
            batches = batches_by_color[color_group]

            # Skip if this color doesn't have this many batches
            if batch_idx >= len(batches):
                continue

            start_idx, end_idx = batches[batch_idx]
            columns = points_by_color[color_group]

            # Build arrays for this batch. Left as numpy rather than .tolist(): Plotly 6
            # serialises a numpy array to base64 binary, which measured ~30% off the wire
            # payload and skips building a Python list per batch, while a list round-trips
            # as JSON numbers. The string columns gain nothing from it but stay numpy so
            # every column is sliced the same way.
            x_vals = columns['x'][start_idx:end_idx]
            y_vals = columns['y'][start_idx:end_idx]
            text_vals = columns['text'][start_idx:end_idx]
            customdata_vals = columns['customdata'][start_idx:end_idx]

            # Map visual properties to arrays. These stay full-length lists even though
            # each batch has a single shape/opacity group: apply_plot_styling reads
            # marker.symbol/marker.opacity and branches on whether they are sequences
            # when it builds the ghost legend traces.
            batch_shape_groups = columns['shape_group'][start_idx:end_idx]
            batch_opacity_groups = columns['opacity_group'][start_idx:end_idx]
            marker_symbols = [shape_map[g] if g is not None and shape_map else 'circle' for g in batch_shape_groups]
            marker_opacities = [opacity_map[g] if g is not None and opacity_map else 0.8 for g in batch_opacity_groups]

            # Only show legend for the first batch of each color
            show_in_legend = (batch_idx == 0)

            fig.add_trace(
                scatter_cls(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    text=text_vals,
                    customdata=customdata_vals,
                    hovertemplate=hover_without_trace,
                    name=format_group_label(color_group, len(points_by_color[color_group]['x']), show_counts),
                    legendgroup=str(color_group),  # All batches of same color share legendgroup
                    marker=dict(
                        color=color_map[color_group],
                        symbol=marker_symbols,
                        opacity=marker_opacities
                    ),
                    showlegend=show_in_legend,
                    legendrank=color_group_order[color_group]
                )
            )

    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)

    fig.update_layout(
        hovermode='closest'
    )

    # The renderer, not the figure: callers mutate the figure they passed in, but any
    # overlay they draw AFTER this call (regression line, cluster hulls, centroids) must
    # use the SAME class. Plotly paints the whole WebGL canvas above every SVG trace, so
    # trace order cannot rescue an SVG overlay -- it lands under the cloud it is meant to
    # sit on. A GL overlay added after the points paints above them, because GL traces do
    # stack in trace order within the canvas.
    return scatter_cls

def add_point_legend_traces(fig, shape_map, opacity_map, shape_by=None, opacity_by=None):
    """
    Adds legend traces for shape and opacity visual channels to the given figure.
    """
    # Add opacity legend traces
    if opacity_map:
        for i, (opacity_group, opacity_value) in enumerate(opacity_map.items()):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='gray',
                        opacity=opacity_value,
                        symbol='circle'
                    ),
                    name=f'{opacity_group}',
                    legendgroup='opacity_legend',
                    showlegend=True,
                    legendrank=100 + i,
                    hoverinfo='skip'
                )
            )
    # Add shape legend traces
    if shape_map:
        for i, (shape_group, shape_symbol) in enumerate(shape_map.items()):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='gray',
                        opacity=0.8,
                        symbol=shape_symbol
                    ),
                    name=f'{shape_group}',
                    legendgroup='shape_legend',
                    showlegend=True,
                    legendrank=200 + i,
                    hoverinfo='skip'
                )
            )
    return fig