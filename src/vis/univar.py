from itertools import combinations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.emojis import sad_emoji
from src.feature_labels import format_feature_label
from src.widgets.visualization_widgets import (
    comparison_pair_widget,
    gmm_hyperParams_widget,
    histogram_bin_width_widget,
)

from .helpers import (
    _add_effect_size_annotations,
    _density_at_points,
    _find_best_gmm,
    _prepare_group_data,
    add_point_legend_traces,
    create_subcolor_map,
    find_intersection,
    format_group_label,
    get_context_theme_color,
    get_point_visual_mappings,
    hover_field,
    interleave_point_batches,
    log_negative_error,
    point_trace_class,
)


def fov_comparison_plot(df, fov_name_col, selected_var, color_by, colormap="tab10"):
    fig = go.Figure()
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False, colormap=colormap)
    show_counts = st.session_state.get("plot_show_group_counts", False)
    group_counts = df.dropna(subset=[selected_var]).groupby(GROUP_COL_NAME).size().to_dict()

    fov_names = df[fov_name_col].unique()

    legend_added = set()

    # Group by color first, then by FOV within each color group
    for color_group in unique_color_groups:
        group_df = df[df[GROUP_COL_NAME] == color_group]
        for fov_name in fov_names:
            fov_group_df = group_df[group_df[fov_name_col] == fov_name]
            if fov_group_df.empty:
                continue

            # Show legend only for the first occurrence of each color group
            show_legend = color_group not in legend_added
            if show_legend:
                legend_added.add(color_group)

            fig.add_trace(go.Box(
                y=fov_group_df[selected_var],
                name=format_group_label(color_group, group_counts.get(color_group), show_counts),  # color group name (optionally with count)
                x=[fov_name] * len(fov_group_df),  # Explicitly set x values for grouping
                boxpoints=False, # Only show the box
                marker_color=color_map[color_group],
                showlegend=show_legend,  # Show legend only once per color group
                legendgroup=color_group,  # Group legend entries by color
                hovertemplate=(
                    hover_field(fov_name_col, fov_name)
                    + hover_field("Group", color_group)
                )
            ))

    fig.update_layout(
        title=f'Distribution of {format_feature_label(selected_var)} by Field of View',
        xaxis_title=fov_name_col,
        yaxis_title=format_feature_label(selected_var),
        showlegend=True,
        hovermode='closest',
        margin=dict(l=50, r=20, t=50, b=max(80, len(max(fov_names, key=len, default=''))*5)) # Adjust bottom margin for long names
    )
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    return fig

def feature_histogram_plot(df, selected_var, color_by=[], colormap="tab10", log_x=False):
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False, colormap=colormap)
    show_counts = st.session_state.get("plot_show_group_counts", False)

    fig = go.Figure()

    bin_edges = histogram_bin_width_widget(df[selected_var], key=f"hist_bin_width_{selected_var}")

    for color_group in unique_color_groups:
        group_df = df[df[GROUP_COL_NAME] == color_group]
        x_data = group_df[selected_var].dropna()
        x_data_skewness = x_data.skew()
        # Determine skewness interpretation based on rule of thumb
        if x_data_skewness < -1:
            direction = "strongly left-skewed"
        elif -1 <= x_data_skewness < -0.5:
            direction = "moderately left-skewed"
        elif -0.5 <= x_data_skewness < -0.25:
            direction = "approximately symmetric"
        elif -0.25 <= x_data_skewness <= 0.25:
            direction = "almost symmetric"
        elif 0.25 < x_data_skewness <= 0.5:
            direction = "approximately symmetric"
        elif 0.5 < x_data_skewness <= 1:
            direction = "moderately right-skewed"
        else:  # x_data_skewness > 1
            direction = "strongly right-skewed"

        # Color the text using the same color as the plot
        st.markdown(f'<span style="color: {color_map[color_group]}"><strong>{color_group}</strong> skewness: {x_data_skewness:.3f} → {direction}</span>', unsafe_allow_html=True)

        if x_data.empty:
            continue # Skip empty groups
        # Calculate histogram counts using the common bin edges derived from bin_width
        counts, bin_edges = np.histogram(x_data, bins=bin_edges)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Add line trace connecting bin centers
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=counts,
            mode='lines', # Use lines instead of markers+lines
            name=format_group_label(color_group, len(x_data), show_counts),
            line=dict(color=color_map[color_group], width=2),
            hovertemplate=(
                f"<b>Group:</b> {color_group}<br>"
                f"<b>Count:</b> %{{y}}<extra></extra>"
            )
        ))

    theme_color = get_context_theme_color()
    # data is log-transformed upstream (data_analysis.py) when log_x is set; wrap the label to match
    pretty_var = format_feature_label(selected_var)
    x_axis_label = f"log₁₀({pretty_var})" if log_x else pretty_var
    fig.update_layout(
        title=dict(
            text=f'Frequency histogram of {pretty_var} by {", ".join(color_by)}',
            font=dict(color=theme_color)
        ),
        xaxis=dict(
            title=dict(text=x_axis_label, font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='Count', font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            showgrid=True,
            zeroline=False
        ),
        hovermode='x unified', # Good for comparing counts at specific x-values
        # hover tooltip styling is applied centrally in apply_plot_styling (theme-aware)
        margin=dict(l=50, r=20, t=50, b=80)
    )
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)
    return fig

def _assign_subpopulation_labels(values, best_gmm, thresholds, color_group):
    """Label each point with its GMM subpopulation, numbered by ascending-mean rank.

    ``group1`` is always the smallest-mean component, so the labels line up with
    the component table rendered in ``feature_gmm_plot`` (which is sorted by mean).

    - With intersection ``thresholds`` (ascending), ``np.digitize`` already
      returns the ascending-mean bucket index, so it is used directly.
    - Without thresholds, ``best_gmm.predict`` returns *original* component
      indices, which are remapped to ascending-mean rank.
    """
    values = np.asarray(values)
    if thresholds is not None:
        ranks = np.digitize(values, bins=thresholds)
    else:
        sorted_indices = np.argsort(best_gmm.means_.flatten())
        rank_of = {int(orig): rank for rank, orig in enumerate(sorted_indices)}
        ranks = [rank_of[int(c)] for c in best_gmm.predict(values.reshape(-1, 1))]
    return [f"{color_group}_group{int(r) + 1}" for r in ranks]


def feature_gmm_plot(df, selected_var, color_by=[], colormap="tab10", log_x=False):
    h_index_msg = ""    
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False, colormap=colormap)
    show_counts = st.session_state.get("plot_show_group_counts", False)
    fit_gmm_max_components, fit_gmm_min_weight_threshold = gmm_hyperParams_widget()
    # add the choice to do "intersection thresholding" or "hard assignment"
    intersection_threshold = st.checkbox("Use intersection as threshold", value=False, key="intersection_threshold", help="If checked, the point where the two Gaussian distributions intersect will be used as the threshold. If not checked, each data will be assigned to the component with the highest posterior probability.")
    fig = go.Figure()
    theme_color = get_context_theme_color()
    # fit a Gaussian Mixture Model (GMM) to each color group

    # Collect tables for two-column display
    gmm_tables = []
    for color_group in unique_color_groups:
        group_df = df[df[GROUP_COL_NAME] == color_group]
        x_data = group_df[selected_var].dropna()

        if x_data.empty:
            continue # Skip empty groups

        # --- Fit GMMs with 1 to 3 components ---
        best_gmm = _find_best_gmm(x_data.values, max_components=fit_gmm_max_components, min_weight_threshold=fit_gmm_min_weight_threshold) # Use x_data.values for 1D

        if best_gmm is None: # if no valid model is found, skip this group
            st.warning(f"No valid GMM found for group {color_group} with current constraints.")
            continue

        # use plotly to plot curve of the best gmm
        x = np.linspace(x_data.min(), x_data.max(), 1000).reshape(-1, 1)
        logprob = best_gmm.score_samples(x)
        pdf = np.exp(logprob)
        responsibilities = best_gmm.predict_proba(x)  # Component weights per point
        pdf_individual = responsibilities * pdf[:, np.newaxis]  # Individual component densities
        fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=pdf,
            mode='lines',
            name=format_group_label(f'{color_group} GMM', len(x_data), show_counts),
            line=dict(color=color_map[color_group], width=2),
            hovertemplate=(
                f"<b>Group:</b> {color_group}<br>"
            )
        ))
        if best_gmm.n_components > 1:
            pi = best_gmm.weights_
            mu = best_gmm.means_.flatten()
            sigma = np.sqrt(best_gmm.covariances_.ravel())
            gmm_overall_mean = np.sum(pi * mu)
            # iteratively print out the mean and standard deviation of each component in a table
            # Sort components by ascending mu (mean) values
            sorted_indices = np.argsort(mu)
            table_md = [f"**GMM Components for {color_group}:**"]
            table_md.append("| Component | Mean  | Std. Dev. | Weight |")
            table_md.append("|-----------|-------|-----------|--------|")
            for rank, i in enumerate(sorted_indices):
                table_md.append(f"| {rank+1}       | {mu[i]:.2f} | {sigma[i]:.2f}    | {pi[i]:.2f}  |")

            gmm_tables.append("\n".join(table_md))

            h_index = 0
            # Calculate standard deviation of component means once before the loop
            means_std = np.std([best_gmm.means_[j][0] for j in range(best_gmm.n_components)], ddof=1)
            dash_styles = ['dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
            for rank, i in enumerate(sorted_indices):
                fig.add_trace(go.Scatter(
                    x=x.flatten(),
                    y=pdf_individual[:, i],
                    mode='lines',
                    name=f'{color_group} Component {rank+1}',
                    line=dict(color=color_map[color_group], width=1, dash=dash_styles[rank % len(dash_styles)]),
                    hovertemplate=(
                        f"<b>Group:</b> {color_group}<br>"
                    )
                ))
                # Calculate H-index for this subpopulation
                # Calculate entropy term for this component
                entropy_term = -best_gmm.weights_[i] * np.log(best_gmm.weights_[i])
                # Calculate normalized distance from overall mean
                distance_term = np.abs(best_gmm.means_[i][0] - gmm_overall_mean) / means_std
                # Combine terms to get component contribution to H-index
                h_index += entropy_term * distance_term
            # Add H-index message
            h_index_msg += f"H-index for {color_group}: {h_index:.3f}. "
            data_indices = x_data.index
            intersection_threshold_possible = intersection_threshold
            if intersection_threshold:
                # predict the component membership for each point (intersection thresholding)
                # find the intersection point of the component distributions
                # Sort components by mean to ensure that the intersection is calculated between the correct pairs
                pi, mu, sigma = pi[sorted_indices], mu[sorted_indices], sigma[sorted_indices]
                thresholds = []
                for i in range(len(mu) - 1):
                    try:
                        t = find_intersection(pi[i], mu[i], sigma[i],
                              pi[i+1], mu[i+1], sigma[i+1])
                        thresholds.append(t)
                    except Exception:
                        st.error(f"Error finding intersection between {color_group} component {i+1} and component {i+2}: either there is no intersection or there are more than one intersection. {sad_emoji}")
                        st.warning("Intersection threshold is not possible, so we resort to hard assignment in this group.")
                        intersection_threshold_possible = False
                        break

                if intersection_threshold_possible:
                    thresholds = np.sort(thresholds)
                    # plot the thresholds
                    for i, threshold in enumerate(thresholds):
                     # Replace the alpha value with 0.5
                        transparent_color = color_map[color_group].replace(color_map[color_group].split(',')[-1], ' 0.5)')
                        fig.add_shape(type="line",
                            x0=threshold, y0=0, x1=threshold, y1=max(pdf),
                            line=dict(color=transparent_color, width=2, dash="dash"),
                            name=f"{color_group} Threshold", 
                        )
                        # Add annotation above the threshold line
                        fig.add_annotation(
                            x=threshold, y=max(pdf) * 1.05, text=f"Threshold ({threshold:.2f})", showarrow=False, align="center",
                            font=dict(color=theme_color)
                        )
                        st.markdown(f"Threshold for <span style='color:{color_map[color_group]}'>{color_group}</span> between component {i+1} and component {i+2}: **{threshold:.2f}**", unsafe_allow_html=True)

                    assigned_labels = _assign_subpopulation_labels(x_data.values, best_gmm, thresholds, color_group)
            if not intersection_threshold_possible:
                # Hard assignment: each point joins its highest-posterior component,
                # numbered by ascending-mean rank to match the component table above.
                assigned_labels = _assign_subpopulation_labels(x_data.values, best_gmm, None, color_group)
            df.loc[data_indices, "GMM_group"] = assigned_labels

    # Display tables in two columns using modular arithmetic
    if gmm_tables:
        col1, col2 = st.columns(2)
        for i, table in enumerate(gmm_tables):
            if i % 2 == 0:  # Even indices (0, 2, 4, ...) go to column 1
                with col1:
                    st.markdown(table)
            else:  # Odd indices (1, 3, 5, ...) go to column 2
                with col2:
                    st.markdown(table)

    if h_index_msg != "": 
        st.info(h_index_msg)

    # data is log-transformed upstream (data_analysis.py) when log_x is set; wrap the label to match
    pretty_var = format_feature_label(selected_var)
    x_axis_label = f"log₁₀({pretty_var})" if log_x else pretty_var
    fig.update_layout(
        title=dict(
            text=f'Gaussian Mixture Model fit of {pretty_var} by {", ".join(color_by)}',
            font=dict(color=theme_color)
        ),
        xaxis=dict(
            title=dict(text=x_axis_label, font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='Probability Density', font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            showgrid=True,
            zeroline=False
        ),
        # hover tooltip styling is applied centrally in apply_plot_styling (theme-aware)
        margin=dict(l=50, r=20, t=50, b=80)
    )

    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    return fig, df

# Plotly's automatic box width for these plots. Adjacent groups sit exactly 1.0 apart (x
# positions are consecutive integers within a section), and the default boxgap and
# boxgroupgap of 0.3 each give (1 - 0.3) * (1 - 0.3) = 0.49 data units; whiskerwidth
# defaults to half of that. Both were measured off the rendered SVG box to confirm.
_BOX_WIDTH = 0.49
_WHISKER_CAP_WIDTH = _BOX_WIDTH * 0.5


def _add_box_outline_above_gl(fig, x, q1, median, q3, lower_fence, upper_fence, mean_val, color):
    """Redraw a box outline as ``layer="above"`` shapes so it paints over WebGL points.

    Plotly puts the WebGL canvas *above* every SVG cartesian layer -- the points live in
    div.gl-container, which the DOM places after the svg holding the box -- so once the
    figure crosses the WebGL threshold no zorder can lift a go.Box over the cloud (its
    trace only moves between zorder subplots inside that same svg). Shapes declared
    layer="above" land in g.layer-above, the one group painted on top of the canvas.

    The go.Box trace itself is left untouched, hidden beneath the points, so its hover
    statistics and legend entry still work.
    """
    half, cap = _BOX_WIDTH / 2, _WHISKER_CAP_WIDTH / 2
    outline = dict(color=color, width=3)
    common = dict(xref="x", yref="y", layer="above")
    fig.add_shape(type="rect", x0=x - half, x1=x + half, y0=q1, y1=q3,
                  line=outline, fillcolor="rgba(0,0,0,0)", **common)
    fig.add_shape(type="line", x0=x - half, x1=x + half, y0=median, y1=median,
                  line=outline, **common)
    # Dashed mean line, matching boxmean=True on the trace.
    fig.add_shape(type="line", x0=x - half, x1=x + half, y0=mean_val, y1=mean_val,
                  line=dict(color=color, width=3, dash="dash"), **common)
    for base, fence in ((q1, lower_fence), (q3, upper_fence)):
        fig.add_shape(type="line", x0=x, x1=x, y0=base, y1=fence, line=outline, **common)
        fig.add_shape(type="line", x0=x - cap, x1=x + cap, y0=fence, y1=fence,
                      line=outline, **common)


def feature_comparison_plot(df, unique_row_id_col, fov_name_col, selected_var, color_by, opacity_by=None, shape_by=None, separate_by=None, colormap="tab10", effect_size_method="None", mean_or_median=None, statistical_test="None", custom_order=None, subcolor_by=None, row_id_label="ID"):

    # Get theme color once at the start for all theme-aware elements
    theme_color = get_context_theme_color()
    # Pretty FLIM label (Greek notation) reused for hover, title, and y-axis
    pretty_var = format_feature_label(selected_var)

    col1, col2, col3 = st.columns([0.15, 0.2, 0.65])
    with col1:
        log_y = st.checkbox("Log Y", value=False, key=f"log_y_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")
    with col2:
        add_boxplot = st.checkbox("Add boxplot", value=False, key=f"add_boxplot_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")
    with col3:
        connect_means = st.checkbox("Connect means", value=False, key=f"connect_means_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")

    # Create a working copy to avoid modifying the original dataframe
    df = df.copy()

    # Apply log transform if requested (consistent with bivar.py)
    if log_y:
        if (df[selected_var] < 0).any():
            st.error(log_negative_error(selected_var))
        else:
            df[selected_var] = np.log10(df[selected_var] + 1e-6)

    fig = go.Figure()
    COLOR_GROUP_COL_NAME = 'compare_group'
    # Use the new helper for color, shape, opacity
    grouped_sep, color_map, shape_map, opacity_map, separate_groups = get_point_visual_mappings(
        df,
        color_by=color_by,
        shape_by=shape_by,
        opacity_by=opacity_by,
        separate_by=separate_by,
        group_col_name=COLOR_GROUP_COL_NAME,
        overlap_point=False,
        colormap=colormap
    )
    grouped_list = list(grouped_sep)
    group_keys = [group_key for group_key, _ in grouped_list]
    compare_groups = list(color_map.keys())
    show_counts = st.session_state.get("plot_show_group_counts", False)
    plotted = df.dropna(subset=[selected_var])
    group_counts = plotted.groupby(COLOR_GROUP_COL_NAME).size().to_dict()

    # Subcolor takes the colour channel AWAY from the colour group: colour comes
    # to mean the nested value while x keeps encoding the group. The map is global -- one
    # colour per distinct value -- so a value appearing in several groups wears one colour
    # and its single bare legend entry is true everywhere. Positions, tick labels, the box
    # overlay and every statistic stay at the colour-group level either way.
    subcolor_of = create_subcolor_map(
        plotted, subcolor_by, COLOR_GROUP_COL_NAME, list(color_map.keys()), colormap=colormap,
    )
    # Counted per value across the whole frame, matching the global mapping: one legend
    # entry covers every group the value appears in, so its count has to as well.
    subcolor_counts = {}
    if subcolor_of:
        subcolor_counts = (plotted[subcolor_by].fillna("N/A").astype(str)
                          .value_counts().to_dict())

    # Apply custom order early so compare_pairs uses the reordered groups
    # This is critical for Glass's Delta where the first group in the pair is the control
    ordered_compare_groups = list(compare_groups)
    if custom_order and 'compare_groups' in custom_order:
        custom_cmp = [g for g in custom_order['compare_groups'] if g in ordered_compare_groups]
        remaining_cmp = [g for g in ordered_compare_groups if g not in custom_cmp]
        ordered_compare_groups = custom_cmp + remaining_cmp

    # Generate pairs from ordered groups so Glass's Delta uses correct control group
    compare_pairs = list(combinations(ordered_compare_groups, 2))
    point_size = 5

    # Track legend entries to avoid duplicates
    legend_entries = set()

    # Calculate x-positions for separate sections - only for existing combinations

    if separate_groups:
        # First, find which combinations actually exist in the data
        existing_combinations = []

        # Apply custom order to separate_groups if provided
        ordered_separate_groups = list(separate_groups)
        if custom_order and 'separate_groups' in custom_order:
            # Filter to only include groups that are actually in the data, preserving data integrity
            custom_sep = [g for g in custom_order['separate_groups'] if g in ordered_separate_groups]
            # Add any missing groups at the end
            remaining = [g for g in ordered_separate_groups if g not in custom_sep]
            ordered_separate_groups = custom_sep + remaining

        for separate_group in ordered_separate_groups:
            section_combinations = []

            # Use ordered_compare_groups (already set with custom order at function start)
            for color_group in ordered_compare_groups:
                combo_exists = any(
                    (separate_group in group_key if isinstance(group_key, tuple) else group_key == separate_group) and
                    (color_group in group_key if isinstance(group_key, tuple) else group_key == color_group)
                    for group_key in group_keys
                )
                if combo_exists:
                    section_combinations.append((separate_group, color_group))
            existing_combinations.append(section_combinations)

        # Now calculate positions only for existing combinations
        x_positions = {}
        x_tick_positions_actual = []
        x_tick_labels_actual = []
        section_spacing = 0.5  # Reduced gap size between sections
        current_x = 0

        for section_idx, section_combinations in enumerate(existing_combinations):
            if section_idx > 0:
                current_x += section_spacing  # Add spacing between sections

            for separate_group, color_group in section_combinations:
                x_positions[(separate_group, color_group)] = current_x
                x_tick_positions_actual.append(current_x)
                x_tick_labels_actual.append("" if color_group == "all_data" else color_group)
                current_x += 1

        separate_sections_info = []
        current_x = 0
        for section_idx, section_combinations in enumerate(existing_combinations):
            if section_idx > 0:
                current_x += section_spacing
            section_start = current_x
            section_end = current_x + len(section_combinations) - 1
            section_center = (section_start + section_end) / 2 if section_combinations else current_x
            # existing_combinations is built in ordered_separate_groups order,
            # so section_idx indexes both.
            group_name = ordered_separate_groups[section_idx]

            separate_sections_info.append({
                'group': group_name,
                'center': section_center,
                'combinations': section_combinations
            })
            current_x += len(section_combinations)
    else:
        # Standard x-positions when no separate_by
        # ordered_compare_groups is already set with custom order at function start
        x_positions = {color_group: idx for idx, color_group in enumerate(ordered_compare_groups)}

    # Sort grouped_list based on the custom order to ensure legend matches x-axis
    # Sort keys: (separate_group index, color_group index)
    def group_sort_key(item):
        g_key, _ = item
        color_g = g_key[0]
        separate_g = g_key[3]

        c_idx = float('inf')
        if color_g in ordered_compare_groups:
            c_idx = ordered_compare_groups.index(color_g)

        s_idx = float('inf')
        if separate_groups and separate_g in ordered_separate_groups:
            s_idx = ordered_separate_groups.index(separate_g)
        elif not separate_groups:
            s_idx = 0

        return (s_idx, c_idx)

    grouped_list.sort(key=group_sort_key)

    # Row positions drawn in each (separate section, colour group) cell, pooled from the
    # subgroups so a row with a NaN shape/opacity value is excluded here too. Positions
    # rather than index labels, sorted, so the pooled order is row order.
    all_y = df[selected_var].to_numpy(dtype=float)
    pooled_rows = {}
    for group_key, group_df in grouped_list:
        if group_key[0] not in compare_groups:
            continue
        drawn = group_df.dropna(subset=[selected_var])
        if drawn.empty:
            continue
        cell = (group_key[3], group_key[0])
        pooled_rows.setdefault(cell, []).append(df.index.get_indexer(drawn.index))
    pooled_rows = {cell: np.sort(np.concatenate(chunks))
                   for cell, chunks in pooled_rows.items()}

    def cell_x_position(separate_group, color_group):
        if separate_groups:
            return x_positions.get((separate_group, color_group))
        return x_positions.get(color_group)

    # Sina jitter: horizontal spread proportional to local density. The KDE and the rng
    # are scoped to the colour group, which owns the x position, so shape_by/opacity_by
    # change point styling only and never move a point.
    x_of_row = np.full(len(df), np.nan)
    for (separate_group, color_group), rows in pooled_rows.items():
        x_position = cell_x_position(separate_group, color_group)
        if x_position is None:
            continue
        y_data = all_y[rows]
        densities = _density_at_points(y_data)
        # Normalize densities to a reasonable jitter width
        max_jitter = 0.35  # Controls the max horizontal spread
        if len(densities) > 0 and np.max(densities) > 0:
            norm_densities = densities / np.max(densities)
        else:
            # Degenerate density (a constant column has no KDE): spread points with
            # uniform jitter so they stay visible instead of stacking into one dot.
            norm_densities = np.ones_like(densities)
        # Randomly assign sign to spread points left/right
        rng = np.random.default_rng(seed=42)
        x_of_row[rows] = x_position + (rng.uniform(-1, 1, size=len(y_data))
                                       * norm_densities * max_jitter)

    # Filled by the subgroup loop, drained after it: drawing is deferred so a colour
    # group's shape/opacity subgroups can be pooled into one trace.
    point_buckets = {}

    for group_key, group_df in grouped_list:
        # Always unpack group_key by position
        color_group = group_key[0]
        shape_group = group_key[1]
        opacity_group = group_key[2]
        separate_group = group_key[3]

        # Skip if not in our color groups (shouldn't happen but safety check)
        if color_group not in compare_groups:
            continue

        # Drop rows where the variable to plot is NaN
        group_df = group_df.dropna(subset=[selected_var])
        if group_df.empty:
            continue

        # --- Prepare Hover Information ---
        # Every label is the column's own name -- for the identifier, the configured
        # column or "ID" for the row numbers invented for a table that has none.
        hovertemplate_parts = [hover_field(pretty_var, "%{y:.3f}")]
        hovertemplate_parts.append(hover_field(row_id_label, "%{text}"))
        if fov_name_col is not None:
            point_customdata = group_df[fov_name_col]
            # Add the corresponding part to the hovertemplate, referencing customdata
            hovertemplate_parts.append(hover_field(fov_name_col, "%{customdata}"))
        hovertemplate_parts.append("<extra></extra>") # Hide the default trace info box
        final_hovertemplate = "".join(hovertemplate_parts)

        # --- Determine x-position and visual properties ---
        # Same skip as the jitter pass: a subgroup with no x position is not accumulated.
        if cell_x_position(separate_group, color_group) is None:
            continue

        marker_color = color_map[color_group]
        marker_opacity = opacity_map.get(opacity_group, 0.7) if opacity_map and opacity_group is not None else 0.7
        marker_symbol = shape_map.get(shape_group, 'circle') if shape_map and shape_group is not None else 'circle'

        # Each row's x was assigned per colour group above; look it up rather than
        # recomputing, so a subgroup cannot get a spread of its own.
        y_data = group_df[selected_var].values
        x_jittered = x_of_row[df.index.get_indexer(group_df.index)]

        # Accumulate instead of drawing: the shape/opacity subgroups of one colour group
        # share an x band, so they are pooled into one trace carrying symbol and opacity
        # as per-point arrays, the same way helpers.add_interleaved_points_trace does.
        bucket = point_buckets.setdefault((separate_group, color_group), {
            "x": [], "y": [], "text": [], "customdata": [],
            "symbol": [], "opacity": [], "subcolor": [],
        })
        bucket["x"].append(x_jittered)
        bucket["y"].append(y_data)
        bucket["text"].append(group_df[unique_row_id_col].to_numpy())
        # Only when the frame has a FOV column. The concat below skips a key with no
        # chunks, so an omitted one simply never reaches the trace.
        if fov_name_col is not None:
            bucket["customdata"].append(point_customdata.to_numpy())
        # Constant within a subgroup because they come from the group key, so repeat
        # rather than read one per row.
        bucket["symbol"].append(np.repeat(marker_symbol, len(y_data)))
        bucket["opacity"].append(np.repeat(marker_opacity, len(y_data)))
        # Only under subcolor: this is the one channel read in just that
        # branch, and filling it otherwise allocates a full-length array of empty
        # strings per subgroup on every ordinary sina plot for nothing.
        if subcolor_of:
            bucket["subcolor"].append(group_df[subcolor_by].fillna("N/A").astype(str).to_numpy())

    # One renderer for the whole figure, decided from the total the buckets hold. Chosen
    # once rather than per trace: Plotly paints every WebGL trace beneath every SVG one, so
    # a figure mixing the two would stack its colour groups by renderer, not by draw order.
    scatter_cls = point_trace_class(
        sum(len(chunk) for bucket in point_buckets.values() for chunk in bucket["y"])
    )

    for (separate_group, color_group), bucket in point_buckets.items():
        columns = {name: np.concatenate(chunks)
                   for name, chunks in bucket.items() if chunks}
        # `columns` is built with `if chunks`, so the key is simply absent when the
        # frame has no FOV column.
        customdata_all = columns.get("customdata")
        # Only what both branches share. The matched branch draws a masked subset, so
        # symbol and opacity are passed per call rather than baked in here.
        marker_kwargs = dict(size=point_size, line=dict(width=0.5, color='DarkSlateGrey'))
        trace_kwargs = dict(mode='markers', hovertemplate=final_hovertemplate)
        if scatter_cls is go.Scatter:
            # Scattergl rejects zorder outright, and has no use for it: the WebGL layer
            # already sits beneath the SVG boxes, which is exactly what zorder=1 buys
            # against the boxes' zorder=10 below.
            trace_kwargs['zorder'] = 1

        if not subcolor_of:
            # One trace per colour group: symbol and opacity vary per point, colour does
            # not, so there is no paint order for a batcher to fix.
            show_legend = color_group not in legend_entries
            if show_legend:
                legend_entries.add(color_group)
            fig.add_trace(scatter_cls(
                x=columns["x"], y=columns["y"],
                name=format_group_label(color_group, group_counts.get(color_group), show_counts),
                marker=dict(color=color_map[color_group], symbol=columns["symbol"],
                            opacity=columns["opacity"], **marker_kwargs),
                showlegend=show_legend,
                legendgroup=color_group,
                text=columns["text"],
                customdata=customdata_all,
                **trace_kwargs
            ))
            continue

        # Subcolor: colour varies by value, so each value needs its own trace to stay
        # clickable in the legend. Batch them and cycle, so no value paints over another.
        for value, mask in interleave_point_batches({
            value: np.flatnonzero(columns["subcolor"] == value)
            # The map's keys are the figure's value list in natural-sort order; a value
            # this group has no points for yields an empty mask, which the batcher skips.
            for value in subcolor_of
        }):
            # One legend entry per value, and every batch of it -- in this colour group
            # and in every other -- repeats the value, so the entry appears once and its
            # legendgroup toggles all of them.
            show_entry = value not in legend_entries
            if show_entry:
                legend_entries.add(value)
            fig.add_trace(scatter_cls(
                x=columns["x"][mask], y=columns["y"][mask],
                name=format_group_label(value, subcolor_counts.get(value), show_counts),
                marker=dict(color=subcolor_of[value],
                            symbol=columns["symbol"][mask],
                            opacity=columns["opacity"][mask], **marker_kwargs),
                showlegend=show_entry,
                # \x1f rather than '::' because apply_plot_styling dedupes on the joined
                # legendgroup string, and '::' would let a real group name collide with
                # this synthetic one.
                legendgroup=f"subcolor\x1f{value}",
                text=columns["text"][mask],
                customdata=None if customdata_all is None else customdata_all[mask],
                **trace_kwargs
            ))

    if connect_means:
        # Above the points in BOTH modes. In WebGL the connector is a GL trace added after
        # them, and GL traces stack in trace order; in SVG it needs an explicit zorder,
        # since the points carry zorder=1 and it would otherwise default to 0 and hide
        # under the cloud. Kept below the boxes' zorder=10.
        mean_line_kwargs = {} if scatter_cls is go.Scattergl else {'zorder': 2}
        if separate_groups:
            # Create a line for each separate group
            for section_info in separate_sections_info:
                section_group_name = section_info['group']

                # These are the color groups that exist in this section
                section_color_groups = [combo[1] for combo in section_info['combinations']]

                # Get the relevant data for this section
                section_df = df[df[separate_by] == section_group_name]

                means_to_plot = []
                for color_group in section_color_groups:
                    x_pos = x_positions.get((section_group_name, color_group))
                    if x_pos is None: continue

                    group_data = section_df[section_df[COLOR_GROUP_COL_NAME] == color_group]
                    if not group_data.empty:
                        mean_y = group_data[selected_var].mean()
                        means_to_plot.append({'x': x_pos, 'y': mean_y})

                # Sort by x position to draw line correctly
                if len(means_to_plot) > 1:
                    means_to_plot.sort(key=lambda p: p['x'])
                    x_coords = [p['x'] for p in means_to_plot]
                    y_coords = [p['y'] for p in means_to_plot]

                    fig.add_trace(scatter_cls(
                        x=x_coords,
                        y=y_coords,
                        mode='lines+markers',
                        name=f'Mean ({section_group_name})',
                        line=dict(width=1, dash='solid', color=theme_color),
                        marker=dict(size=8, symbol='x', color=theme_color),
                        showlegend=True,
                        **mean_line_kwargs
                    ))
        else: # No separate_by
            means_to_plot = []
            for color_group in compare_groups:
                x_pos = x_positions.get(color_group)
                if x_pos is None: continue

                group_data = df[df[COLOR_GROUP_COL_NAME] == color_group]
                if not group_data.empty:
                    mean_y = group_data[selected_var].mean()
                    means_to_plot.append({'x': x_pos, 'y': mean_y})

            # Sort by x position to draw line correctly
            if len(means_to_plot) > 1:
                means_to_plot.sort(key=lambda p: p['x'])
                x_coords = [p['x'] for p in means_to_plot]
                y_coords = [p['y'] for p in means_to_plot]

                fig.add_trace(scatter_cls(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name='Mean',
                    line=dict(width=2, dash='solid', color=theme_color),
                    marker=dict(size=8, symbol='x', color=theme_color),
                    showlegend=False,
                    **mean_line_kwargs
                ))

    # --- 2. Add legend traces for opacity and shape mappings ---
    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)

    # --- 3. Add vertical dashed lines between separate sections ---
    if separate_groups and len(separate_groups) > 1:
        # Add vertical lines between sections using actual section boundaries
        for section_idx in range(len(separate_sections_info) - 1):
            current_section = separate_sections_info[section_idx]
            next_section = separate_sections_info[section_idx + 1]

            # Calculate the end of current section and start of next section
            current_section_end = max([x_positions[(combo[0], combo[1])] for combo in current_section['combinations']])
            next_section_start = min([x_positions[(combo[0], combo[1])] for combo in next_section['combinations']])

            # Place line at the center of the gap
            line_x = (current_section_end + next_section_start) / 2
            fig.add_vline(
                x=line_x,
                line_dash="dash",
                line_color=theme_color,
                line_width=2
            )

    # Build title with visual encoding information
    title_parts = [f'Distribution of {pretty_var} by {", ".join(color_by)}']
    if separate_by and separate_by.strip() != "":
        title_parts.append(f'separated by: {separate_by}')
    if opacity_by and opacity_by.strip() != "":
        title_parts.append(f'opacity: {opacity_by}')
    if shape_by and shape_by.strip() != "":
        title_parts.append(f'shape: {shape_by}')
    if subcolor_by and subcolor_by.strip() != "":
        title_parts.append(f'subcolor: {subcolor_by}')

    full_title = title_parts[0]
    if len(title_parts) > 1:
        full_title += f' ({", ".join(title_parts[1:])})'

    # Configure x-axis labels and layout
    if separate_groups:
        # Section headers are annotations directly under the axis line; the group tick
        # labels are pushed below them by ticklabelstandoff. The header slot is one line
        # of text at a size set here, so reserving room for it is arithmetic on that size.
        header_font_size = st.session_state.get("plot_axis_label_size", 12)
        longest_tick_label = max((len(str(t)) for t in x_tick_labels_actual), default=0)
        # One bold line plus padding. ticklabelstandoff pushes the group labels this far
        # from the axis, on top of Plotly's own default standoff; the header sits in that gap.
        header_slot_px = round(1.6 * header_font_size)
        # Pinned rather than left to Plotly, which picks 0/45/90° from a container width
        # the server cannot know (the chart renders with width='stretch'); the Matplotlib
        # export (src/export_script.py) can only reproduce an angle decided here. Labels of
        # up to 4 characters fit upright at any width; longer ones slant. Negative is the
        # uphill slant, matching the export's rotation=45, ha='right'.
        tick_angle = 0 if longest_tick_label <= 4 else -45
        for section_info in separate_sections_info:
            if section_info['combinations']:  # Only add annotation if section has data
                fig.add_annotation(
                    x=section_info['center'],
                    y=0,  # Bottom of the plot area
                    # Padding between the axis line and the header, not a measurement of
                    # anything below it.
                    yshift=-round(0.25 * header_font_size),
                    text=f"<b>{section_info['group']}</b>",
                    showarrow=False,
                    xref="x",
                    yref="paper",
                    # apply_plot_styling() overwrites every annotation's size with
                    # plot_axis_label_size but leaves the colour, so that must be theme-aware.
                    font=dict(size=header_font_size, color=theme_color),
                    xanchor="center",
                    yanchor="top"
                )

        xaxis_config = dict(
            tickvals=x_tick_positions_actual,
            ticktext=x_tick_labels_actual,
            tickangle=tick_angle,  # fixed above so the export can reproduce it
            ticklabelstandoff=header_slot_px,  # the section header sits in this gap
            zeroline=False,
            tickfont=dict(color=theme_color),
            # The tick labels are the lowest element, so a bottom margin sized from them
            # also covers the header above. Plotly measures them in the browser.
            automargin=True,
        )
    else:
        # Standard x-axis configuration
        xaxis_config = dict(
            tickvals=list(range(len(compare_groups))),
            ticktext=ordered_compare_groups,
            zeroline=False,
            tickfont=dict(color=theme_color),
            # As in the separate_by branch above; here Plotly rotates the labels itself.
            automargin=True,
        )

    # --- Loop 2: Boxplots (on top of points) ---
    # One box per (section, colour group) from that cell's pooled rows — not per
    # shape/opacity subgroup, which would stack several boxes on one x position.
    if add_boxplot:
        for (separate_group, color_group), rows in pooled_rows.items():
            x_position = cell_x_position(separate_group, color_group)
            if x_position is None:
                continue

            marker_color = color_map[color_group]
            trace_name = color_group
            y_data = all_y[rows]

            # Calculate boxplot statistics manually to enforce 1.5 IQR whiskers
            q1 = np.percentile(y_data, 25)
            median = np.percentile(y_data, 50)
            q3 = np.percentile(y_data, 75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            lower_fence = max(lower_fence, np.min(y_data))
            upper_fence = q3 + 1.5 * iqr
            upper_fence = min(upper_fence, np.max(y_data))
            mean_val = np.mean(y_data)

            # Add Box trace (Mean as dashed line, Median as solid line)
            fig.add_trace(go.Box(
                x=[x_position], # Align with x-position
                q1=[q1],
                median=[median],
                q3=[q3],
                lowerfence=[lower_fence],
                upperfence=[upper_fence],
                mean=[mean_val],
                name=trace_name,
                marker_color=marker_color,
                fillcolor='rgba(0,0,0,0)', # Transparent fill
                line=dict(color=theme_color, width=3), # Theme-aware color outlines
                boxpoints=False, # Hide points in box trace
                boxmean=True, # Show mean as dashed line
                showlegend=False,
                hoverinfo='y', # Show stats on hover
                zorder=10
            ))

            if scatter_cls is go.Scattergl:
                _add_box_outline_above_gl(fig, x_position, q1, median, q3,
                                          lower_fence, upper_fence, mean_val, theme_color)

    # Set y-axis label based on log transform (pretty_var defined at top)
    y_axis_label = f"log₁₀({pretty_var})" if log_y else pretty_var

    # A floor only, so short labels still get a gutter; xaxis.automargin grows the
    # margin past it to whatever the rendered labels need.
    bottom_margin = 120

    fig.update_layout(
        title=dict(text=full_title, font=dict(color=theme_color)),
        xaxis=xaxis_config,
        yaxis=dict(
            title=dict(text=y_axis_label, font=dict(color=theme_color)),
            showline=False,
            tickfont=dict(color=theme_color)
        ),
        showlegend=True, # Show legend entries based on the 'name' of each go.Box trace
        hovermode='closest', # Hover behavior
        margin=dict(l=50, r=20, t=50, b=bottom_margin), # Floor; xaxis.automargin grows it
    )

    # --- 4. Add statistical annotations ---
    if compare_pairs != [] and ((effect_size_method != "None" and mean_or_median is not None) or (statistical_test != "None")):
        if separate_groups:
            # Get user selection for statistical comparisons once (to avoid duplicate widget keys)
            selected_pairs = comparison_pair_widget(compare_pairs)

            # Get threshold once to avoid duplicate widgets across sections
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

            if selected_pairs:  # Only proceed if user selected some pairs
                # Calculate global data range ONCE for consistent spacing across all sections
                global_min_y = df[selected_var].min(skipna=True)
                global_max_y = df[selected_var].max(skipna=True)
                global_data_range = (global_min_y, global_max_y)

                # Apply statistical annotations within each separate section
                for section_info in separate_sections_info:
                    if len(section_info['combinations']) > 1:  # Need at least 2 groups for comparison
                        # Extract just the color groups that exist in this section
                        section_color_groups = [combo[1] for combo in section_info['combinations']]
                        section_compare_pairs = list(combinations(section_color_groups, 2))

                        # Filter to only include pairs that user selected and exist in this section
                        filtered_section_pairs = []
                        for pair in section_compare_pairs:
                            if pair in selected_pairs or tuple(reversed(pair)) in selected_pairs:
                                filtered_section_pairs.append(pair)

                        if filtered_section_pairs:
                            # Filter the dataframe to only include this separate_by group
                            section_df = df[df[separate_by] == section_info['group']].copy()

                            # Create position mapping for this section
                            section_position_map = {}
                            actual_positions = []
                            for combo in section_info['combinations']:
                                separate_group, color_group = combo
                                actual_x = x_positions[(separate_group, color_group)]
                                section_position_map[color_group] = actual_x
                                actual_positions.append(actual_x)

                            # Apply annotations for this section with actual positions
                            _add_effect_size_annotations(
                                fig=fig,
                                df=section_df,
                                selected_var=selected_var,
                                compare_groups=section_color_groups,
                                group_col_name=COLOR_GROUP_COL_NAME,
                                all_possible_pairs=section_compare_pairs,
                                annotation_color=theme_color,
                                effect_size_method=effect_size_method,
                                mean_or_median=mean_or_median,
                                position_map=section_position_map,
                                selected_pairs=filtered_section_pairs,
                                threshold=threshold,
                                statistical_test=statistical_test,
                                global_data_range=global_data_range  # Pass global range for consistent spacing
                            )
        else:
            # Standard statistical annotations when no separate_by
            _add_effect_size_annotations(
                fig=fig,
                df=df,
                selected_var=selected_var,
                compare_groups=ordered_compare_groups,  # Use ordered groups for correct bracket positioning
                group_col_name=COLOR_GROUP_COL_NAME,
                all_possible_pairs=compare_pairs,
                annotation_color=theme_color,
                effect_size_method=effect_size_method,
                mean_or_median=mean_or_median,
                statistical_test=statistical_test,
                position_map=x_positions  # Pass position map for correct y-range calculation after reordering
            )

    # Drop the temporary group column if it exists
    if COLOR_GROUP_COL_NAME in df.columns:
        df.drop(columns=[COLOR_GROUP_COL_NAME], inplace=True)
    return fig