import seaborn as sns
import streamlit as st
import numpy as np
from scipy.stats import norm, gaussian_kde, median_abs_deviation, ttest_ind
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
import pandas as pd
from src.widgets.visualization_widgets import comparison_pair_widget
import re
import plotly.graph_objects as go
from streamlit_theme import st_theme

def get_theme_color(key="theme_color_detector"):
    """Get plot color based on current system theme (light/dark mode).
    
    Uses a unique key for st_theme() to avoid duplicate component errors,
    and always fetches the current theme to respond to theme changes.
    
    Args:
        key (str): Unique key for the st_theme component. Defaults to "theme_color_detector".
    
    Returns:
        str: 'black' for light mode, 'white' for dark mode.
    """
    # Use a unique key to avoid duplicate element ID errors
    theme = st_theme(key=key)
    # Default to dark mode color if theme detection fails
    if theme is None or theme.get("base") == "dark":
        return "white"
    else:
        return "black"

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
    
    return diff / group1_sd

def cohens_d(group1, group2, mean_or_median):
    """Compute Cohen's d for two independent samples."""
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
    return diff / pooled_sd

def create_opacity_mapping(groups, min_opacity=0.3, max_opacity=1.0):
    """Create opacity mapping for groups with evenly spaced values, preserving natural order"""
    groups = list(groups)
    groups = natural_tuple_sort(groups) if len(groups) > 1 else groups
    if len(groups) == 1:
        return {groups[0]: max_opacity}
    opacity_values = np.linspace(min_opacity, max_opacity, len(groups))
    return {group: opacity_values[i] for i, group in enumerate(groups)}

def create_shape_mapping(groups):
    """Create shape mapping for groups using different plotly symbols, preserving natural order"""
    groups = list(groups)
    groups = natural_tuple_sort(groups) if len(groups) > 1 else groups
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
               'triangle-down', 'pentagon', 'hexagon', 'octagon', 'star', 'diamond-tall']
    return {group: symbols[i % len(symbols)] for i, group in enumerate(groups)}

def create_color_map(groups, overlap_point, colormap="tab10"):
    # if points in the visualization is going to overlap, use a transparent color
    if overlap_point: 
        alpha = 0.6 if len(groups) > 1 else 1.0
    else:
        alpha = 1.0
    
    # Handle different colormap types
    try:
        if colormap in ["viridis", "plasma", "inferno", "magma", "cividis"]:
            # Use matplotlib colormaps for these
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap)
            if len(groups) == 1:
                colors = [cmap(0.5)]
            else:
                colors = [cmap(i / (len(groups) - 1)) for i in range(len(groups))]
            palette = [(color[0], color[1], color[2]) for color in colors]
        else:
            # Use seaborn for all other colormap names
            palette = sns.color_palette(colormap, n_colors=len(groups))
    except (ValueError, ImportError):
        # Fallback to tab10 if colormap is not available
        palette = sns.color_palette("tab10", n_colors=len(groups))
    
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
    elif method == "Cohen's d":
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

    for iteration in range(50):
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
            elif effect_size_method == "Cohen's d":
                threshold = st.number_input("Cohen's d Threshold", value=0.7, min_value=0.0, max_value=3.0, step=0.05,
                                            key=f"cohens_d_thresh_{threshold_key_suffix}")

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
    :param strings: list of strings to sort
    :param delimiter: delimiter to split columns (default '::')
    :return: sorted list of strings
    """
    return sorted(strings, key=lambda x: tuple_natural_key(x.split(delimiter)))

def create_opacity_groups_and_map(df, opacity_by_col):
    """
    Given a DataFrame and a column name, return sorted unique groups and an opacity map.
    """
    if opacity_by_col and opacity_by_col in df.columns:
        groups = df[opacity_by_col].dropna().unique()
        groups = natural_tuple_sort(groups)
        opacity_map = create_opacity_mapping(groups)
        return groups, opacity_map
    else:
        return [], None

def create_shape_groups_and_map(df, shape_by_col):
    """
    Given a DataFrame and a column name, return sorted unique groups and a shape map.
    """
    if shape_by_col and shape_by_col in df.columns:
        groups = df[shape_by_col].dropna().unique()
        groups = natural_tuple_sort(groups)
        shape_map = create_shape_mapping(groups)
        return groups, shape_map
    else:
        return [], None

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
    shape_groups, shape_map = create_shape_groups_and_map(df, shape_by)
    # Prepare opacity mapping
    opacity_groups, opacity_map = create_opacity_groups_and_map(df, opacity_by)
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

    def ordered_group_iter():
        for group_key in product(color_keys, shape_keys, opacity_keys, separate_keys):
            # Build boolean mask for each group
            mask = (
                (df[group_col_name] == group_key[0])
            )
            if shape_by and shape_by in df.columns:
                mask &= (df[shape_by] == group_key[1])
            if opacity_by and opacity_by in df.columns:
                mask &= (df[opacity_by] == group_key[2])
            if separate_by and separate_by in df.columns:
                mask &= (df[separate_by] == group_key[3])
            group_df = df[mask]
            if len(group_df) > 0:
                yield group_key, group_df

    # Return an iterable like a groupby object, but ordered as specified
    grouped_ordered = ordered_group_iter()

    return grouped_ordered, color_map, shape_map, opacity_map, separate_groups

# Function to apply plot styling to any figure
def apply_plot_styling(fig, point_size, axis_label_size, legend_size):
    """Apply consistent styling to plotly figures"""
    # Names of traces that should keep their original marker sizes
    skip_trace_names = {'Lifetime Markers'}
    
    # Update marker sizes for all scatter and box traces
    for trace in fig.data:
        # Skip traces that should maintain their original sizes
        if hasattr(trace, 'name') and trace.name in skip_trace_names:
            continue
        if hasattr(trace, 'marker') and trace.marker:
            if trace.type == 'scatter':
                trace.marker.size = point_size
            elif trace.type == 'box' and trace.marker:
                trace.marker.size = point_size
    
    # Update annotation font sizes to match axis label size
    if fig.layout.annotations:
        for annotation in fig.layout.annotations:
            if annotation.font:
                annotation.font.size = axis_label_size
            else:
                annotation.font = dict(size=axis_label_size)
    
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
        )
    )
    return fig

def _estimate_density_1d(y_values, bw_method='scott'):
    """
    Estimate the density of y_values using KDE. Returns a function that maps y to density.
    """

    y_values = np.asarray(y_values)
    if len(y_values) < 2:
        return lambda y: np.zeros_like(np.asarray(y))
    kde = gaussian_kde(y_values, bw_method=bw_method)
    return kde

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
    random_seed=None,
    num_batches=15
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
        Random seed for shuffling (for reproducibility)
    num_batches : int
        Number of batches to split each color group into (default: 15)
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The figure with traces added
    """
    import random
    import math
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Collect all points, grouped by color
    points_by_color = {}
    
    for group_key, group_df in grouped:
        color_group = group_key[0]
        shape_group = group_key[1]
        opacity_group = group_key[2]
        
        if color_group not in points_by_color:
            points_by_color[color_group] = []
            
        for idx, row in group_df.iterrows():
            points_by_color[color_group].append({
                'x': row[axis_labels[0]],
                'y': row[axis_labels[1]],
                'text': row[text_col],
                'customdata': row[customdata_col],
                'shape_group': shape_group,
                'opacity_group': opacity_group,
            })
    
    # Shuffle points within each color group
    for color_group in points_by_color:
        random.shuffle(points_by_color[color_group])
    
    # Split each color group into batches
    batches_by_color = {}
    for color_group, points in points_by_color.items():
        num_points = len(points)
        # Adjust num_batches if there are fewer points
        actual_batches = min(num_batches, max(1, num_points // 5))  # At least 5 points per batch
        batch_size = math.ceil(num_points / actual_batches)
        
        batches = []
        for i in range(actual_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            if start_idx < num_points:
                batches.append(points[start_idx:end_idx])
        
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
    
    # Interleave batches: add batch i from each color before moving to batch i+1
    for batch_idx in range(max_batches):
        for color_group in sorted_color_groups:
            batches = batches_by_color[color_group]
            
            # Skip if this color doesn't have this many batches
            if batch_idx >= len(batches):
                continue
            
            batch_points = batches[batch_idx]
            
            # Build arrays for this batch
            x_vals = [p['x'] for p in batch_points]
            y_vals = [p['y'] for p in batch_points]
            text_vals = [p['text'] for p in batch_points]
            customdata_vals = [p['customdata'] for p in batch_points]
            
            # Map visual properties to arrays
            marker_symbols = [shape_map[p['shape_group']] if p['shape_group'] is not None and shape_map else 'circle' for p in batch_points]
            marker_opacities = [opacity_map[p['opacity_group']] if p['opacity_group'] is not None and opacity_map else 0.8 for p in batch_points]
            
            # Only show legend for the first batch of each color
            show_in_legend = (batch_idx == 0)
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    text=text_vals,
                    customdata=customdata_vals,
                    hovertemplate=hover_without_trace,
                    name=str(color_group),
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
    
    # Add shape/opacity legends if needed
    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)
    
    # Update layout with hovermode
    fig.update_layout(
        hovermode='closest'
    )
    
    return fig

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