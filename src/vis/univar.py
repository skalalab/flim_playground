import streamlit as st
import plotly.graph_objects as go
from itertools import combinations
import numpy as np

from src.widgets.visualization_widgets import histogram_bin_width_widget, gmm_hyperParams_widget, stats_comparison_pair_widget

from .helpers import _prepare_group_data, find_intersection, _add_effect_size_annotations, _find_best_gmm, create_opacity_mapping, create_shape_mapping, _calculate_effect_size, _annotate_single_effect_size

def image_comparison_plot(df, selected_var):
    if (df["image_name"] == "missing image name").any():
        st.markdown("<h5 style='text-align: center; color: Red;'>Warning: We cannot infer some/all image names from you cell_id column. We assume that the image name is the cell_id without the cell number (which is found after the last underscore) </h5>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    image_names = df['image_name'].unique()
    
    for image_name in image_names:
        image_df = df[df['image_name'] == image_name]
        fig.add_trace(go.Box(
            y=image_df[selected_var],
            name=image_name, # Store image_name here to retrieve on click
            boxpoints=False, # Only show the box
            # customdata=[image_name] * len(image_df), # Alternative if name doesn't work reliably
            # hovertemplate=f"<b>Image:</b> {image_name}<br><b>{selected_var}:</b> %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        title=f'Distribution of {selected_var} by Image',
        xaxis_title='Image Name',
        yaxis_title=selected_var,
        showlegend=False, # Hide legend if too many images
        hovermode='closest',
        xaxis={'categoryorder':'array', 'categoryarray': sorted(image_names)}, # Sort boxes by name
        margin=dict(l=50, r=20, t=50, b=max(80, len(max(image_names, key=len, default=''))*5)) # Adjust bottom margin for long names
    )
    
    return fig

def feature_histogram_plot(df, selected_var, color_by=[]):
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False)
   
    fig = go.Figure()

    bin_edges = histogram_bin_width_widget(df[selected_var])

    for color_group in unique_color_groups:
        group_df = df[df[GROUP_COL_NAME] == color_group]
        x_data = group_df[selected_var].dropna()

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

    fig.update_layout(
        title=f'Frequency histogram of {selected_var} by {", ".join(color_by)}',
        xaxis_title=selected_var,
        yaxis_title='Count',
        legend_title_text='Groups',
        hovermode='x unified', # Good for comparing counts at specific x-values
        margin=dict(l=50, r=20, t=50, b=80)
        # Removed barmode='overlay'
    )
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)
    return fig

def feature_gmm_plot(df, selected_var, color_by=[]):
    h_index_msg = ""    
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False)
    fit_gmm_max_components, fit_gmm_min_weight_threshold = gmm_hyperParams_widget()
    # add the choice to do "hard thresholding" or "soft thresholding"
    hard_thresholding = st.checkbox("Use hard thresholding", value=False, key="hard_thresholding", help="If checked, the point where the two Gaussian distributions intersect will be used as the threshold. If not checked, each data will be assigned to the component with the highest posterior probability.")
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
            dash_styles = ['dash', 'dot', 'dashdot']
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
                h_index += -best_gmm.weights_[i] * np.log(best_gmm.weights_[i]) * np.abs(best_gmm.means_[i][0] - gmm_overall_mean)
            # Add H-index message
            h_index_msg += f"H-index for {color_group}: {h_index:.3f}. "
            data_indices = x_data.index
            hard_thresholding_possible = hard_thresholding
            if hard_thresholding:
                # predict the component membership for each point (hard thresholding)
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
                        st.warning("Hard thresholding is not possible, so we resort to soft thresholding in this group.")
                        hard_thresholding_possible = False
                        break

                if hard_thresholding_possible:
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
                        )
                        st.markdown(f"Threshold for <span style='color:{color_map[color_group]}'>{color_group}</span> between component {sorted_indices[i]+1} and component {sorted_indices[i+1]+1}: **{threshold:.2f}**", unsafe_allow_html=True)

                    subpopulation_labels = np.digitize(x_data, bins=thresholds)
                    # restore the original order of the labels
                    subpopulation_labels = sorted_indices[subpopulation_labels]
            if not hard_thresholding_possible:
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
   
    st.plotly_chart(fig, use_container_width=True, key=f"gmm_plot_{selected_var}_{', '.join(color_by)}")
    
    # have a button to export the GMM group augmented dataframe
    downloaded = st.download_button(label="Download GMM Grouped Data", data=df.to_csv(index=False), file_name="gmm_grouped_data.csv", mime="text/csv", key="gmm_download")
    if downloaded:
        if "GMM_group" in df.columns:
            df.drop(columns=['GMM_group'], inplace=True)

    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    return fig, h_index_msg



def feature_comparison_plot(df, selected_var, color_by, opacity_by=None, shape_by=None, separate_by=None, effect_size_method="None"):
    fig = go.Figure()
    GROUP_COL_NAME = 'compare_group'
    compare_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False)
    compare_pairs = list(combinations(compare_groups, 2))
    jitter_amount = 1
    point_size = 5
    
    # Handle separate_by grouping
    separate_groups = None
    if separate_by and separate_by.strip() != "" and separate_by in df.columns:
        separate_groups = sorted(df[separate_by].dropna().unique())
    
    # Create opacity and shape mappings if provided
    opacity_map = None
    shape_map = None
    
    if opacity_by and opacity_by.strip() != "" and opacity_by in df.columns:
        opacity_groups = df[opacity_by].dropna().unique()
        opacity_map = create_opacity_mapping(opacity_groups)
    
    if shape_by and shape_by.strip() != "" and shape_by in df.columns:
        shape_groups = df[shape_by].dropna().unique() 
        shape_map = create_shape_mapping(shape_groups)

    # --- 1. Create all unique combinations for separate traces ---
    # Build grouping columns list
    all_grouping_cols = [GROUP_COL_NAME]
    if separate_by and separate_by.strip() != "" and separate_by in df.columns:
        all_grouping_cols.append(separate_by)
    if opacity_by and opacity_by.strip() != "" and opacity_by in df.columns:
        all_grouping_cols.append(opacity_by)
    if shape_by and shape_by.strip() != "" and shape_by in df.columns:
        all_grouping_cols.append(shape_by)
    
    # Group by all visual encoding variables
    grouped = df.groupby(all_grouping_cols, dropna=False)
    
    # Track legend entries to avoid duplicates
    legend_entries = set()
    
    # Calculate x-positions for separate sections - only for existing combinations
    if separate_groups:
        # First, find which combinations actually exist in the data
        existing_combinations = []
        for separate_group in separate_groups:
            section_combinations = []
            for color_group in compare_groups:
                # Check if this combination exists in the grouped data
                combo_exists = any(
                    (separate_group in group_key if isinstance(group_key, tuple) else group_key == separate_group) and
                    (color_group in group_key if isinstance(group_key, tuple) else group_key == color_group)
                    for group_key in grouped.groups.keys()
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
            
            section_start_x = current_x
            for local_idx, (separate_group, color_group) in enumerate(section_combinations):
                x_positions[(separate_group, color_group)] = current_x
                x_tick_positions_actual.append(current_x)
                x_tick_labels_actual.append(color_group)
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
            separate_sections_info.append({
                'group': separate_groups[section_idx],
                'center': section_center,
                'combinations': section_combinations
            })
            current_x += len(section_combinations)
    else:
        # Standard x-positions when no separate_by
        x_positions = {color_group: idx for idx, color_group in enumerate(compare_groups)}
    
    for group_key, group_df in grouped:
        # Extract group information from group_key
        color_group = group_key[0]
        
        # Handle separate_by, opacity_by, and shape_by extraction
        separate_group = None
        opacity_group = None  
        shape_group = None
        
        key_idx = 1  # Start from index 1 (after color_group)
        
        if separate_by and separate_by.strip() != "" and separate_by in df.columns:
            separate_group = group_key[key_idx] if len(group_key) > key_idx else None
            key_idx += 1
            
        if opacity_by and opacity_by.strip() != "" and opacity_by in df.columns:
            opacity_group = group_key[key_idx] if len(group_key) > key_idx else None
            key_idx += 1
            
        if shape_by and shape_by.strip() != "" and shape_by in df.columns:
            shape_group = group_key[key_idx] if len(group_key) > key_idx else None
        
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
        point_customdata = group_df['image_name']
        # Add the corresponding part to the hovertemplate, referencing customdata
        hovertemplate_parts.append("<b>Image:</b> %{customdata}<br>")
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
        marker_opacity = opacity_map.get(opacity_group, 0.8) if opacity_map and opacity_group is not None else 0.8
        marker_symbol = shape_map.get(shape_group, 'circle') if shape_map and shape_group is not None else 'circle'
        
        # --- Determine trace name and legend visibility ---
        # Only show in legend if this color group hasn't been shown yet
        show_legend = color_group not in legend_entries
        if show_legend:
            legend_entries.add(color_group)
        
        trace_name = color_group
        
        # --- Add the go.Box Trace ---
        fig.add_trace(go.Box(
            # Core data and category assignment
            y=group_df[selected_var],
            x=[x_position] * len(group_df),  # Custom x-position for this group
            name=trace_name,
            # Point display settings
            boxpoints='all',
            jitter=jitter_amount,
            pointpos=0,
            # Styling for the individual points (marker)
            marker=dict(
                color=marker_color,
                size=point_size,
                opacity=marker_opacity,
                symbol=marker_symbol,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            # Make the actual box plot elements invisible
            fillcolor='rgba(0,0,0,0)',
            line_color='rgba(0,0,0,0)',
            # Legend control
            showlegend=show_legend,
            legendgroup=color_group,  # Group all traces with same color
            # --- Hover Info for Points ---
            text=group_df['cell_id'],
            customdata=point_customdata,
            hovertemplate=final_hovertemplate
        ))
    
    # --- 2. Add legend traces for opacity and shape mappings ---
    # Add opacity legend traces
    if opacity_map:
        for opacity_group, opacity_value in opacity_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],  # No data points
                mode='markers',
                marker=dict(
                    size=12,
                    color='gray',
                    opacity=opacity_value,
                    symbol='circle'
                ),
                name=f'{opacity_by}: {opacity_group}',
                legendgroup='opacity_legend',
                showlegend=True,
                hoverinfo='skip'
            ))
    
    # Add shape legend traces  
    if shape_map:
        for shape_group, shape_symbol in shape_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],  # No data points
                mode='markers', 
                marker=dict(
                    size=12,
                    color='gray',
                    opacity=0.8,
                    symbol=shape_symbol
                ),
                                name=f'{shape_by}: {shape_group}',
                legendgroup='shape_legend', 
                showlegend=True,
                hoverinfo='skip'
            ))
     
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
                line_color="gray",
                line_width=2,
                opacity=0.7
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
                    y=-0.15,  # Position below x-axis
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
            title=f'{", ".join(color_by)} (grouped by {separate_by})'
        )
    else:
        # Standard x-axis configuration
        xaxis_config = dict(title=', '.join(color_by))
    
    fig.update_layout(
        title=full_title,
        xaxis=xaxis_config,
        yaxis_title=selected_var,
        showlegend=True, # Show legend entries based on the 'name' of each go.Box trace
        hovermode='closest', # Hover behavior
      #  template='plotly_white',
        margin=dict(l=50, r=20, t=50, b=max(120, len(max(compare_groups, key=len, default=''))*5)), # Adjust bottom margin for section headers
        # Ensure boxplot elements like mean lines or whiskers are not shown if they somehow sneak through
        # (though transparent colors should be sufficient)
       # boxmode='group' 
    )

    # --- 4. Add statistical annotations ---
    if compare_pairs != [] and effect_size_method != "None":
        if separate_groups:
            # Get user selection for statistical comparisons once (to avoid duplicate widget keys)
            selected_pairs = stats_comparison_pair_widget(compare_pairs)
            
            # Get threshold once to avoid duplicate widgets across sections
            threshold = 0.0
            threshold_key_suffix = selected_var
            if effect_size_method == "Glass's Delta":
                threshold = st.number_input("Glass's Delta Threshold", value=0.7, min_value=0.0, max_value=3.0, step=0.05, 
                                            key=f"glass_delta_thresh_{threshold_key_suffix}")
            elif effect_size_method == "Cohen's Distance":
                threshold = st.number_input("Cohen's Distance Threshold", value=0.5, min_value=0.0, max_value=3.0, step=0.05,
                                            key=f"cohens_d_thresh_{threshold_key_suffix}")
            
            if selected_pairs:  # Only proceed if user selected some pairs
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
                                group_col_name=GROUP_COL_NAME,
                                all_possible_pairs=section_compare_pairs,
                                effect_size_method=effect_size_method,
                                position_map=section_position_map,
                                selected_pairs=filtered_section_pairs,
                                threshold=threshold
                            )
        else:
            # Standard statistical annotations when no separate_by
            _add_effect_size_annotations(
                fig=fig,
                df=df,
                selected_var=selected_var,
                compare_groups=compare_groups,
                group_col_name=GROUP_COL_NAME,
                all_possible_pairs=compare_pairs,
                effect_size_method=effect_size_method
            )

    df.drop(columns=[GROUP_COL_NAME], inplace=True)
    return fig