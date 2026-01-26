import streamlit as st
import plotly.graph_objects as go
from itertools import combinations
import numpy as np
from src.widgets.visualization_widgets import histogram_bin_width_widget, gmm_hyperParams_widget, comparison_pair_widget
from .helpers import _prepare_group_data, find_intersection, _add_effect_size_annotations, _find_best_gmm, _estimate_density_1d, get_point_visual_mappings, add_point_legend_traces, get_theme_color

def fov_comparison_plot(df, fov_name_col, selected_var, color_by, colormap="tab10"):
    if (df[fov_name_col] == "missing fov name").any():
        st.markdown("<h5 style='text-align: center; color: Orange;'>Warning: We cannot find the fov column from your dataset.  </h5>", unsafe_allow_html=True)
    
    fig = go.Figure()
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False, colormap=colormap)
    
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
                name=color_group,  # Use color group name for legend
                x=[fov_name] * len(fov_group_df),  # Explicitly set x values for grouping
                boxpoints=False, # Only show the box
                marker_color=color_map[color_group],
                showlegend=show_legend,  # Show legend only once per color group
                legendgroup=color_group,  # Group legend entries by color
                hovertemplate=(
                    f"<b>FOV:</b> {fov_name}<br>"
                    f"<b>Group:</b> {color_group}<br>"
                )
            ))
    
    fig.update_layout(
        title=f'Distribution of {selected_var} by Field of View',
        xaxis_title=fov_name_col,
        yaxis_title=selected_var,
        showlegend=True, # Hide legend 
        hovermode='closest',
       # xaxis={'categoryorder':'array', 'categoryarray': sorted(fov_names)}, # Sort boxes by name
        margin=dict(l=50, r=20, t=50, b=max(80, len(max(fov_names, key=len, default=''))*5)) # Adjust bottom margin for long names
    )
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True) 
    
    return fig

def feature_histogram_plot(df, selected_var, color_by=[], colormap="tab10"):
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False, colormap=colormap)
   
    fig = go.Figure()

    bin_edges = histogram_bin_width_widget(df[selected_var])

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

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Add line trace connecting bin centers
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=counts,
            mode='lines', # Use lines instead of markers+lines
            name=color_group,
            line=dict(color=color_map[color_group], width=2),
            hovertemplate=(
                f"<b>Group:</b> {color_group}<br>"
                f"<b>Count:</b> %{{y}}<extra></extra>"
            )
        ))

    theme_color = get_theme_color(key=f"theme_hist_{selected_var}")
    fig.update_layout(
        title=dict(
            text=f'Frequency histogram of {selected_var} by {", ".join(color_by)}',
            font=dict(color=theme_color)
        ),
        xaxis=dict(
            title=dict(text=selected_var, font=dict(color=theme_color)),
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
        hoverlabel=dict(
            bgcolor="white" if theme_color == "black" else "rgb(30, 30, 30)",
            font=dict(color=theme_color, size=13),
            bordercolor=theme_color
        ),
        margin=dict(l=50, r=20, t=50, b=80)
    )
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)
    return fig

def feature_gmm_plot(df, selected_var, color_by=[], colormap="tab10"):
    h_index_msg = ""    
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False, colormap=colormap)
    fit_gmm_max_components, fit_gmm_min_weight_threshold = gmm_hyperParams_widget()
    # add the choice to do "intersection thresholding" or "hard assignment"
    intersection_threshold = st.checkbox("Use intersection as threshold", value=False, key="intersection_threshold", help="If checked, the point where the two Gaussian distributions intersect will be used as the threshold. If not checked, each data will be assigned to the component with the highest posterior probability.")
    fig = go.Figure()
    # fit a Gaussian Mixture Model (GMM) to each color group
    
    # Collect tables for two-column display
    gmm_tables = []
    for color_group in unique_color_groups:
        group_df = df[df[GROUP_COL_NAME] == color_group]
        x_data = group_df[selected_var].dropna()

        if x_data.empty:
            continue # Skip empty groups

        # Fit GMM to the data
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
        # Plot the GMM
        fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=pdf,
            mode='lines',
            name=f'{color_group} GMM',
            line=dict(color=color_map[color_group], width=2),
            hovertemplate=(
                f"<b>Group:</b> {color_group}<br>"
            )
        ))
        # # add histogram plot
        # fig.add_trace(go.Histogram(
        #     x=x_data,
        #     histnorm='probability density',
        #     name=f'{color_group} Histogram',
        #     opacity=0.5,
        #     marker_color="gray",
        #     hovertemplate=(
        #         f"<b>Group:</b> {color_group}<br>"
        #         f"<b>Count:</b> %{{y}}<extra></extra>"
        #     ),
        #     # not showing the legend
        #     showlegend=False,
        # ))
        # Plot individual components if more than one
        if best_gmm.n_components > 1:
            
            # Plot individual components
            # pdf_individual is already calculated above
            # Plot each component with a different color
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
            
            # Store table for later display
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
                    except Exception as e:
                        st.error(f"Error finding intersection between {color_group} component {sorted_indices[i]+1} and component {sorted_indices[i+1]+1}: either there is no intersection or there are more than one intersection.")
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
                        st.markdown(f"Threshold for <span style='color:{color_map[color_group]}'>{color_group}</span> between component {sorted_indices[i]+1} and component {sorted_indices[i+1]+1}: **{threshold:.2f}**", unsafe_allow_html=True)

                    subpopulation_labels = np.digitize(x_data, bins=thresholds)
                    # restore the original order of the labels
                    subpopulation_labels = sorted_indices[subpopulation_labels]
            if not intersection_threshold_possible:
                # Predict the component membership for each point (soft thresholding)
                data_2d = x_data.values.reshape(-1, 1)
                subpopulation_labels = best_gmm.predict(data_2d)
            # Assign the predicted labels (0-based) to the new column in the original DataFrame
            # Add 1 to have 1-based component indexing (e.g., group1, 2, ...)
            assigned_labels = [f"{color_group}_group{label + 1}" for label in subpopulation_labels]
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
    
    theme_color = get_theme_color(key=f"theme_gmm_{selected_var}")
    fig.update_layout(
        title=dict(
            text=f'Gaussian Mixture Model fit of {selected_var} by {", ".join(color_by)}',
            font=dict(color=theme_color)
        ),
        xaxis=dict(
            title=dict(text=selected_var, font=dict(color=theme_color)),
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
        hoverlabel=dict(
            bgcolor="white" if theme_color == "black" else "rgb(30, 30, 30)",
            font=dict(color=theme_color, size=13),
            bordercolor=theme_color
        ),
        margin=dict(l=50, r=20, t=50, b=80)
    )
    
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    return fig, df

def feature_comparison_plot(df, cell_id_col, fov_name_col, selected_var, color_by, opacity_by=None, shape_by=None, separate_by=None, colormap="tab10", effect_size_method="None", mean_or_median=None, statistical_test="None", custom_order=None):

    # Get theme color once at the start for all theme-aware elements
    theme_color = get_theme_color(key=f"theme_compare_{selected_var}")
    
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        add_boxplot = st.checkbox("Add boxplot", value=False, key=f"add_boxplot_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")
    with col2:
        connect_means = st.checkbox("Connect means", value=False, key=f"connect_means_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")
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
    compare_pairs = list(combinations(compare_groups, 2))
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
            
            # Apply custom order to compare_groups if provided
            ordered_compare_groups = list(compare_groups)
            if custom_order and 'compare_groups' in custom_order:
                custom_cmp = [g for g in custom_order['compare_groups'] if g in ordered_compare_groups]
                remaining_cmp = [g for g in ordered_compare_groups if g not in custom_cmp]
                ordered_compare_groups = custom_cmp + remaining_cmp

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
            
            for local_idx, (separate_group, color_group) in enumerate(section_combinations):
                x_positions[(separate_group, color_group)] = current_x
                x_tick_positions_actual.append(current_x)
                x_tick_labels_actual.append("" if color_group == "all_data" else color_group)
                current_x += 1
        
        # Store for later use in x-axis configuration
        separate_sections_info = []
        current_x = 0
        for section_idx, section_combinations in enumerate(existing_combinations):
            if section_idx > 0:
                current_x += section_spacing
            section_start = current_x
            section_end = current_x + len(section_combinations) - 1
            section_center = (section_start + section_end) / 2 if section_combinations else current_x
            # The group name comes from the first element of the first combination in the section, 
            # or we need to track it from ordered_separate_groups. 
            # Since existing_combinations aligns with ordered_separate_groups...
            group_name = ordered_separate_groups[section_idx]
            
            separate_sections_info.append({
                'group': group_name,
                'center': section_center,
                'combinations': section_combinations
            })
            current_x += len(section_combinations)
    else:
        # Standard x-positions when no separate_by
        ordered_compare_groups = list(compare_groups)
        if custom_order and 'compare_groups' in custom_order:
            custom_cmp = [g for g in custom_order['compare_groups'] if g in ordered_compare_groups]
            remaining_cmp = [g for g in ordered_compare_groups if g not in custom_cmp]
            ordered_compare_groups = custom_cmp + remaining_cmp

        x_positions = {color_group: idx for idx, color_group in enumerate(ordered_compare_groups)}

    # Sort grouped_list based on the custom order to ensure legend matches x-axis
    # Sort keys: (separate_group index, color_group index)
    def group_sort_key(item):
        g_key, _ = item
        color_g = g_key[0]
        separate_g = g_key[3]
        
        # Get indices with default fallback
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
        hovertemplate_parts = [
            f"<b>{selected_var}:</b> %{{y:.3f}}<br>" # Display the Y value
        ]
        hovertemplate_parts.append("<b>Cell ID:</b> %{text}<br>")
        point_customdata = group_df[fov_name_col]
        # Add the corresponding part to the hovertemplate, referencing customdata
        hovertemplate_parts.append("<b>fov:</b> %{customdata}<br>")
        hovertemplate_parts.append("<extra></extra>") # Hide the default trace info box
        final_hovertemplate = "".join(hovertemplate_parts)
        
        # --- Determine x-position and visual properties ---
        if separate_groups:
            x_position = x_positions.get((separate_group, color_group))
            if x_position is None:
                continue  # Skip if x_position not found
        else:
            x_position = x_positions.get(color_group)
            if x_position is None:
                continue  # Skip if x_position not found
        
        marker_color = color_map[color_group]
        marker_opacity = opacity_map.get(opacity_group, 0.7) if opacity_map and opacity_group is not None else 0.7
        marker_symbol = shape_map.get(shape_group, 'circle') if shape_map and shape_group is not None else 'circle'
        
        # --- Determine trace name and legend visibility ---
        # Only show in legend if this color group hasn't been shown yet
        show_legend = color_group not in legend_entries
        if show_legend:
            legend_entries.add(color_group)
        
        trace_name = color_group
        
        # --- Sina plot: density-based horizontal jitter ---
        y_data = group_df[selected_var].values
        kde = _estimate_density_1d(y_data)
        densities = kde(y_data)
        # Normalize densities to a reasonable jitter width
        max_jitter = 0.35  # Controls the max horizontal spread
        if len(densities) > 0 and np.max(densities) > 0:
            norm_densities = densities / np.max(densities)
        else:
            norm_densities = np.zeros_like(densities)
        # Randomly assign sign to spread points left/right
        rng = np.random.default_rng(seed=42)
        jitter_offsets = (rng.uniform(-1, 1, size=len(y_data))) * norm_densities * max_jitter
        x_jittered = x_position + jitter_offsets
        # Plot points (no violin, just sina)
        fig.add_trace(go.Scatter(
            x=x_jittered,
            y=y_data,
            mode='markers',
            name=trace_name,
            marker=dict(
                color=marker_color,
                size=point_size,
                opacity=marker_opacity,
                symbol=marker_symbol,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            showlegend=show_legend,
            legendgroup=color_group,
            text=group_df[cell_id_col],
            customdata=point_customdata,
            hovertemplate=final_hovertemplate,
            zorder=1
        ))

    if connect_means:
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
                    # Get x position
                    x_pos = x_positions.get((section_group_name, color_group))
                    if x_pos is None: continue

                    # Calculate mean
                    group_data = section_df[section_df[COLOR_GROUP_COL_NAME] == color_group]
                    if not group_data.empty:
                        mean_y = group_data[selected_var].mean()
                        means_to_plot.append({'x': x_pos, 'y': mean_y})
                
                # Sort by x position to draw line correctly
                if len(means_to_plot) > 1:
                    means_to_plot.sort(key=lambda p: p['x'])
                    x_coords = [p['x'] for p in means_to_plot]
                    y_coords = [p['y'] for p in means_to_plot]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines+markers',
                        name=f'Mean ({section_group_name})',
                        line=dict(width=1, dash='solid', color=theme_color),
                        marker=dict(size=8, symbol='x', color=theme_color),
                        showlegend=True
                    ))
        else: # No separate_by
            means_to_plot = []
            for color_group in compare_groups:
                # Get x position
                x_pos = x_positions.get(color_group)
                if x_pos is None: continue

                # Calculate mean
                group_data = df[df[COLOR_GROUP_COL_NAME] == color_group]
                if not group_data.empty:
                    mean_y = group_data[selected_var].mean()
                    means_to_plot.append({'x': x_pos, 'y': mean_y})
            
            # Sort by x position to draw line correctly
            if len(means_to_plot) > 1:
                means_to_plot.sort(key=lambda p: p['x'])
                x_coords = [p['x'] for p in means_to_plot]
                y_coords = [p['y'] for p in means_to_plot]

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name='Mean',
                    line=dict(width=2, dash='solid', color=theme_color),
                    marker=dict(size=8, symbol='x', color=theme_color),
                    showlegend=False
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
    title_parts = [f'Distribution of {selected_var} by {", ".join(color_by)}']
    if separate_by and separate_by.strip() != "":
        title_parts.append(f'separated by: {separate_by}')
    if opacity_by and opacity_by.strip() != "":
        title_parts.append(f'opacity: {opacity_by}')
    if shape_by and shape_by.strip() != "":
        title_parts.append(f'shape: {shape_by}')
    
    full_title = title_parts[0]
    if len(title_parts) > 1:
        full_title += f' ({", ".join(title_parts[1:])})'
    
    # Configure x-axis labels and layout
    if separate_groups:
        # Use the actual positions and labels we calculated
        # Add section headers using annotations
        for section_info in separate_sections_info:
            if section_info['combinations']:  # Only add annotation if section has data
                fig.add_annotation(
                    x=section_info['center'],
                    y=-0.20,  # Position below x-axis labels
                    text=f"<b>{section_info['group']}</b>",
                    showarrow=False,
                    xref="x",
                    yref="paper",
                    font=dict(size=12, color="black"),
                    xanchor="center"
                )
        
        xaxis_config = dict(
            tickvals=x_tick_positions_actual,
            ticktext=x_tick_labels_actual,
            zeroline=False,
            tickfont=dict(color=theme_color),
        )
    else:
        # Standard x-axis configuration
        xaxis_config = dict(
            tickvals=list(range(len(compare_groups))),
            ticktext=ordered_compare_groups,
            zeroline=False,
            tickfont=dict(color=theme_color),
        )
    
    # --- Loop 2: Boxplots (on top of points) ---
    if add_boxplot:
        for group_key, group_df in grouped_list:
            # Always unpack group_key by position
            color_group = group_key[0]
            shape_group = group_key[1]
            opacity_group = group_key[2]
            separate_group = group_key[3]
            
            # Skip if not in our color groups
            if color_group not in compare_groups:
                continue
                    
            # Drop rows where the variable to plot is NaN
            group_df = group_df.dropna(subset=[selected_var])
            if group_df.empty:
                continue

            # --- Determine x-position ---
            if separate_groups:
                x_position = x_positions.get((separate_group, color_group))
                if x_position is None: continue
            else:
                x_position = x_positions.get(color_group)
                if x_position is None: continue
            
            marker_color = color_map[color_group]
            trace_name = color_group
            y_data = group_df[selected_var].values

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

    fig.update_layout(
        title=dict(text=full_title, font=dict(color=theme_color)),
        xaxis=xaxis_config,
        yaxis=dict(
            title=dict(text=selected_var, font=dict(color=theme_color)),
            showline=False,
            tickfont=dict(color=theme_color)
        ),
        showlegend=True, # Show legend entries based on the 'name' of each go.Box trace
        hovermode='closest', # Hover behavior
        margin=dict(l=50, r=20, t=50, b=max(120, len(max(compare_groups, key=len, default=''))*5)), # Adjust bottom margin for section headers
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
            elif effect_size_method == "Cohen's d":
                threshold = st.number_input("Cohen's d Threshold", value=0.5, min_value=0.0, max_value=3.0, step=0.05,
                                            key=f"cohens_d_thresh_{threshold_key_suffix}")
            
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
                compare_groups=compare_groups,
                group_col_name=COLOR_GROUP_COL_NAME,
                all_possible_pairs=compare_pairs,
                annotation_color=theme_color,
                effect_size_method=effect_size_method,
                mean_or_median=mean_or_median,
                statistical_test=statistical_test
            )

    # Drop the temporary group column if it exists
    if COLOR_GROUP_COL_NAME in df.columns:
        df.drop(columns=[COLOR_GROUP_COL_NAME], inplace=True)
    return fig