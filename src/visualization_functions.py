import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, gaussian_kde
from scipy.optimize import brentq
from src.widgets.custom_widgets import stats_comparison_pair_widget, histogram_bin_width_widget
from src.dimension_reduction import dimension_reduction
import pandas as pd

def find_intersection(pi1, mu1, sigma1, pi2, mu2, sigma2):
    """
    Find the intersection point between two weighted Gaussian components where
    pi1 * N(x; mu1, sigma1) = pi2 * N(x; mu2, sigma2).
    """
    f = lambda x: pi1 * norm.pdf(x, mu1, sigma1) - pi2 * norm.pdf(x, mu2, sigma2)
    # The root must lie between the two means
    return brentq(f, min(mu1, mu2), max(mu1, mu2))

def glass_delta(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    group2_sd = np.std(group2, ddof=1)  # Using Bessel's correction with ddof=1
    return mean_diff / group2_sd

def cohens_d(group1, group2):
    """Compute Cohen's d for two independent samples."""
    # sample sizes
    n1, n2 = len(group1), len(group2)
    # unbiased sample variances
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    # mean difference
    mean_diff = np.mean(group1) - np.mean(group2)
    return mean_diff / pooled_sd

def create_color_map(groups, overlap_point):
    # if points in the visulization is going to overlap, use a transparent color
    if overlap_point: 
        alpha = 0.6 if len(groups) > 1 else 1.0
    else:
        alpha = 1.0
    palette = sns.color_palette("tab10", n_colors=len(groups))
    color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {alpha})" for color in palette]
    color_map = {t: color_sequence[i] for i, t in enumerate(groups)}
    return color_map

def _prepare_group_data(df, group_by_cols, new_group_col_name, overlap_point=True):
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
        df[new_group_col_name] = df[group_by_cols].astype(str).agg('_'.join, axis=1)
    
    unique_groups = df[new_group_col_name].unique()
    unique_groups = sorted(unique_groups, key=lambda x: tuple(x.split('_')))
    color_map = create_color_map(unique_groups, overlap_point=overlap_point)
    return unique_groups, color_map

def _calculate_effect_size(group1_data, group2_data, method: str):
    """
    Calculates the effect size between two groups using the specified method.
    """
    if group1_data.empty or group2_data.empty:
        return None

    if method == "Glass's Delta":
        return glass_delta(group1_data, group2_data)
    elif method == "Cohen's Distance":
        # Ensure cohens_d function is available and handles data appropriately
        return cohens_d(group1_data, group2_data)
    else:
        st.warning(f"Unsupported effect size method: {method}")
        return None

def _annotate_single_effect_size(fig, pair_strings, effect_size_value, compare_groups_list, 
                                 drawn_annotations_list, positioning_metrics, 
                                 original_df, data_column_name, group_column_name_in_df, 
                                 overall_min_y_val, data_range_y):
    """
    Adds a single effect size annotation (bracket and text) to the figure,
    handling y-positioning and collision detection.
    """
    x_indices = [compare_groups_list.index(pair_strings[0]), compare_groups_list.index(pair_strings[1])]
    x_start_new = min(x_indices)
    x_end_new = max(x_indices)

    spanned_group_names = compare_groups_list[x_start_new : x_end_new + 1]
    df_in_span = original_df[original_df[group_column_name_in_df].isin(spanned_group_names)]
    current_region_max_y = df_in_span[data_column_name].max(skipna=True)

    if pd.isna(current_region_max_y):
        current_region_max_y = overall_min_y_val
        if pd.isna(current_region_max_y): # Fallback if overall_min_y is also NaN
            current_region_max_y = 0


    # Use pre-calculated absolute positioning metrics
    offset_from_data_abs = positioning_metrics['offset_from_data_abs']
    vertical_spacing_abs = positioning_metrics['vertical_spacing_abs']
    bracket_vertical_length_abs = positioning_metrics['bracket_vertical_length_abs']
    text_offset_from_bracket_abs = positioning_metrics['text_offset_from_bracket_abs']
    text_height_allowance_for_collision_abs = positioning_metrics['text_height_allowance_for_collision_abs']

    y_candidate_bracket_top = current_region_max_y + offset_from_data_abs
    final_y_bracket_top = None
    final_y_text_annotation_center = None
    max_iterations = 50 # Max attempts to find a clear spot

    for iteration in range(max_iterations):
        proposed_y_bracket_top = y_candidate_bracket_top
        proposed_y_text_center = proposed_y_bracket_top + text_offset_from_bracket_abs + (text_height_allowance_for_collision_abs / 2)
        
        # Calculate the bounding box of the new annotation
        new_ann_y_bottom = proposed_y_bracket_top - bracket_vertical_length_abs
        new_ann_y_top = proposed_y_text_center + (text_height_allowance_for_collision_abs / 2)

        collision_found = False
        for existing_ann in drawn_annotations_list:
            # Check for x-overlap: True if the horizontal spans of annotations overlap
            x_overlap = max(x_start_new, existing_ann['x_start']) < min(x_end_new, existing_ann['x_end'])
            if x_overlap:
                # Check for y-overlap: True if the vertical spans of annotations overlap
                y_overlap = max(new_ann_y_bottom, existing_ann['y_bottom']) < min(new_ann_y_top, existing_ann['y_top'])
                if y_overlap:
                    # Collision detected, propose a new y_candidate_bracket_top above the existing annotation
                    y_candidate_bracket_top = existing_ann['y_top'] + vertical_spacing_abs
                    collision_found = True
                    break 
        
        if not collision_found:
            final_y_bracket_top = proposed_y_bracket_top
            final_y_text_annotation_center = proposed_y_text_center
            drawn_annotations_list.append({
                'x_start': x_start_new, 'x_end': x_end_new,
                'y_bottom': new_ann_y_bottom, 'y_top': new_ann_y_top,
                'y_bracket_draw': final_y_bracket_top, 
                'y_text_draw': final_y_text_annotation_center
            })
            break # Found a spot
        
        if iteration == max_iterations - 1:
            # Fallback if no optimal position is found after max_iterations
            st.warning(f"Could not find optimal position for annotation {pair_strings}, using fallback.")
            # Use the last proposed y_candidate_bracket_top, which might overlap
            final_y_bracket_top = y_candidate_bracket_top 
            final_y_text_annotation_center = y_candidate_bracket_top + text_offset_from_bracket_abs + (text_height_allowance_for_collision_abs / 2)
            # Recalculate y_bottom and y_top for the fallback annotation
            new_ann_y_bottom_fb = final_y_bracket_top - bracket_vertical_length_abs
            new_ann_y_top_fb = final_y_text_annotation_center + (text_height_allowance_for_collision_abs / 2)
            drawn_annotations_list.append({
                'x_start': x_start_new, 'x_end': x_end_new,
                'y_bottom': new_ann_y_bottom_fb, 'y_top': new_ann_y_top_fb,
                'y_bracket_draw': final_y_bracket_top, 
                'y_text_draw': final_y_text_annotation_center
            })

    if final_y_bracket_top is None: # Should not happen if fallback is implemented
        return

    # Add bracket lines
    fig.add_shape(
        type="line", x0=x_start_new, y0=final_y_bracket_top,
        x1=x_end_new, y1=final_y_bracket_top,
        line=dict(color="black", width=1.5)
    )
    for x_pos_single in [x_start_new, x_end_new]:
        fig.add_shape(
            type="line", x0=x_pos_single, y0=final_y_bracket_top,
            x1=x_pos_single, y1=final_y_bracket_top - bracket_vertical_length_abs,
            line=dict(color="black", width=1.5)
        )
    # Add effect size text
    fig.add_annotation(
        x=(x_start_new + x_end_new) / 2, y=final_y_text_annotation_center,
        text=f"Δ={effect_size_value:.2f}", showarrow=False, font=dict(size=12),
        align="center"
    )

def _add_effect_size_annotations(fig, df, selected_var, compare_groups, group_col_name, all_possible_pairs, effect_size_method="None"):
    """
    Adds effect size annotations to the figure.
    Manages selection of pairs, calculation of effect sizes, and calls annotation plotting.
    """
    if not all_possible_pairs:
        return

    selected_pairs = stats_comparison_pair_widget(all_possible_pairs)

    if selected_pairs and effect_size_method != "None":
        drawn_annotations = []  # List to store details of drawn annotations for collision detection

        # --- Define vertical spacing parameters (relative to data range) ---
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
            'text_offset_from_bracket_abs': 0.02 * data_range_y,
            'text_height_allowance_for_collision_abs': 0.04 * data_range_y
        }

        # Sort pairs for consistent annotation order and simpler collision logic
        # Sorting key ensures that pairs are processed from left-to-right, and shorter spans before longer ones if they start at the same point.
        sorted_pairs = sorted(selected_pairs,
                              key=lambda p: (min(compare_groups.index(p[0]), compare_groups.index(p[1])),
                                             max(compare_groups.index(p[0]), compare_groups.index(p[1]))))
        
        # --- Threshold input based on selected method ---
        threshold = 0.0
        threshold_key_suffix = selected_var
        if effect_size_method == "Glass's Delta":
            threshold = st.number_input("Glass's Delta Threshold", value=0.7, min_value=0.0, max_value=3.0, step=0.05, 
                                        key=f"glass_delta_thresh_{threshold_key_suffix}")
        elif effect_size_method == "Cohen's Distance":
            threshold = st.number_input("Cohen's Distance Threshold", value=0.5, min_value=0.0, max_value=3.0, step=0.05,
                                        key=f"cohens_d_thresh_{threshold_key_suffix}")

        for pair in sorted_pairs:
            group1_data = df[df[group_col_name] == pair[0]][selected_var].dropna()
            group2_data = df[df[group_col_name] == pair[1]][selected_var].dropna()

            if group1_data.empty or group2_data.empty:
                st.debug(f"Skipping pair {pair} due to empty data for one or both groups.")
                continue

            effect_size_value = _calculate_effect_size(group1_data, group2_data, effect_size_method)

            if effect_size_value is not None and abs(effect_size_value) >= threshold:
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
                    data_range_y=data_range_y # Pass data_range_y for context if needed inside, though metrics are now absolute
                )
          

        # No explicit "else" for unsupported methods here as _calculate_effect_size handles the warning,
        # and effect_size_value would be None, thus skipping annotation.

def feature_comparison_plot(df, selected_var, compared_by, effect_size_method="None"):
    fig = go.Figure()
    GROUP_COL_NAME = 'compare_group'
    compare_groups, color_map = _prepare_group_data(df, compared_by, GROUP_COL_NAME, overlap_point=False)
    compare_pairs = list(combinations(compare_groups, 2))
    jitter_amount = 1
    point_size = 5

    # --- 1. Plotting Traces using go.Box (with hidden box) ---
    for group in compare_groups:
        # Filter data for the current group
        g_df = df[df[GROUP_COL_NAME] == group].copy()
        # Drop rows where the variable to plot is NaN, as they cannot be plotted
        g_df = g_df.dropna(subset=[selected_var])
        # Skip this group if no data remains after filtering/dropping NaNs
        if g_df.empty:
            continue
        # --- Prepare Hover Information ---
        # Initialize data containers
        point_text_data = None
        point_customdata = None
        hovertemplate_parts = [
            f"<b>Group:</b> {group}<br>", # Display the specific group name
            f"<b>{selected_var}:</b> %{{y:.3f}}<br>" # Display the Y value
        ]
        hovertemplate_parts.append("<b>Cell ID:</b> %{text}<br>")
        point_text_data = g_df['cell_id'] # Assign the series to 'text'
        point_customdata = g_df['image_name']
        # Add the corresponding part to the hovertemplate, referencing customdata
        hovertemplate_parts.append("<b>Image:</b> %{customdata}<br>")
        hovertemplate_parts.append("<extra></extra>") # Hide the default trace info box
        final_hovertemplate = "".join(hovertemplate_parts)

        # --- Add the go.Box Trace ---
        fig.add_trace(go.Box(
            # Core data and category assignment
            y=g_df[selected_var],       # Y values for this group
            name=group,                 # Assigns to category & legend label
            # Point display settings
            boxpoints='all',            # Show all individual points (strip/swarm)
            jitter=jitter_amount,       # Control horizontal spread of points
            pointpos=0,                 # Center points horizontally in category space

            # Styling for the individual points (marker)
            marker=dict(
                color=color_map[group], # Use pre-defined color for this group
                size=point_size,
                opacity=0.8,            # Slight transparency helps with dense areas
                line=dict(width=0.5, color='DarkSlateGrey') # Optional outline
            ),

            # Make the actual box plot elements invisible
           # fillcolor='rgba(0,0,0,0)',  # Transparent fill
           # line_color='rgba(0,0,0,0)', # Transparent box outline
            # --- Hover Info for Points ---
            # Assign the prepared data arrays/series
            text=point_text_data,       # Data referenced by %{text} in template
            customdata=point_customdata,# Data referenced by %{customdata}
            # Assign the dynamically built hovertemplate string
            hovertemplate=final_hovertemplate
        ))
    
    fig.update_layout(
        title=f'Distribution of {selected_var} by {", ".join(compared_by)}',
        xaxis_title=', '.join(compared_by),
        yaxis_title=selected_var,
        showlegend=True, # Show legend entries based on the 'name' of each go.Box trace
        hovermode='closest', # Hover behavior
      #  template='plotly_white',
        margin=dict(l=50, r=20, t=50, b=max(80, len(max(compare_groups, key=len, default=''))*5)), # Adjust bottom margin
        # Ensure boxplot elements like mean lines or whiskers are not shown if they somehow sneak through
        # (though transparent colors should be sufficient)
       # boxmode='group' 
    )

    # --- 2. Add statistical annotations ---
    if compare_pairs != [] and effect_size_method != "None":
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

def dimension_reduction_plot(df, selected_features, method="UMAP", hyperParam_dict={}, colored_by=[], exp_var=None):
    """create a plotly plot to visualize the dimension-reduced data
    """
    X = df[selected_features]
                    # perform dimension reduction
    df_reduced, exp_var = dimension_reduction(X, n_components=2, method=method, hyperParam_dict=hyperParam_dict)
    # augment df_reduced with required columns and categorical columns used for coloring
    df_reduced["cell_id"] = df["cell_id"].values
    df_reduced["image_name"] = df["image_name"].values
    # Add all color columns at once if there are any
    if not colored_by.empty:
        df_reduced[colored_by] = df[colored_by].values
    # plot the reduced data
                       
    fig = go.Figure()
    if method == "Principal Component Analysis":
        axis_labels = ["PC1", "PC2"]
    elif method == "UMAP":
        axis_labels = ["UMAP1", "UMAP2"]
    else:
        axis_labels = ["dim1", "dim2"]

    # colored by unique combinations of the selected categorical columns
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df_reduced, colored_by, GROUP_COL_NAME, overlap_point=True)

    # plot scatter plot iteratively, once for each color group
    for g in unique_color_groups:
        g_df =  df_reduced[df_reduced[GROUP_COL_NAME] == g]
        fig.add_trace(
            go.Scatter(
                x=g_df[axis_labels[0]],
                y=g_df[axis_labels[1]],
                mode='markers',
                name=f'{g}',
                text=g_df["cell_id"],   
                customdata=g_df["image_name"],
                hovertemplate="<b>%{text}</b>",
                marker=dict(color=color_map[g])
            ),
    )               
        
    fig.update_layout(
        hovermode='closest'
    )

    # Update axis labels to include explained variance
    if exp_var is not None: 
        fig.update_xaxes(title_text=f"{axis_labels[0]}({exp_var[0]:.2f}%)")
        fig.update_yaxes(title_text=f"{axis_labels[1]}({exp_var[1]:.2f}%)")
    else:
        fig.update_xaxes(title_text=f"{axis_labels[0]}")
        fig.update_yaxes(title_text=f"{axis_labels[1]}")

    return fig

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
   
    # add the choice to do "hard thresholding" or "soft thresholding"
    hard_thresholding = st.checkbox("Use hard thresholding", value=False, key="hard_thresholding", help="If checked, the point where the two Gaussian distributions intersect will be used as the threshold. If not checked, each data will be assigned to the component with the highest posterior probability.")
    fig = go.Figure()
    export_available = False
    # fit a Gaussian Mixture Model (GMM) to each color group
    for color_group in unique_color_groups:
        group_df = df[df[GROUP_COL_NAME] == color_group]
        x_data = group_df[selected_var].dropna()

        if x_data.empty:
            continue # Skip empty groups

        # Fit GMM to the data
        # --- Fit GMMs with 1 to 3 components ---
        data_2d = x_data.values.reshape(-1, 1)  # GMM expects 2D array
        best_gmm = None
        lowest_bic = np.inf
        max_components = 3 # Or determine dynamically
        valid_models = {} # Store valid models (k: gmm_model)
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(data_2d)
            bic = gmm.bic(data_2d)

            # Check the constraint: all weights must be >= 0.10
            min_weight = gmm.weights_.min() 
            if min_weight >= 0.2:
                valid_models[k] = gmm
                # Keep track of the best model among valid ones based on BIC
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    
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
        # add histogram plot
        fig.add_trace(go.Histogram(
            x=x_data,
            histnorm='probability density',
            name=f'{color_group} Histogram',
            opacity=0.5,
            marker_color="gray",
            hovertemplate=(
                f"<b>Group:</b> {color_group}<br>"
                f"<b>Count:</b> %{{y}}<extra></extra>"
            ),
            # not showing the legend
            showlegend=False,
        ))
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
            table_md = [f"**GMM Components for {color_group}:**"]
            table_md.append("| Component | Mean  | Std. Dev. | Weight |")
            table_md.append("|-----------|-------|-----------|--------|")
            for i in range(best_gmm.n_components):
                table_md.append(f"| {i+1}       | {mu[i]:.2f} | {sigma[i]:.2f}    | {pi[i]:.2f}  |")
            st.markdown("\n".join(table_md))

            h_index = 0
            dash_styles = ['dash', 'dot', 'dashdot']
            for i in range(best_gmm.n_components):
                fig.add_trace(go.Scatter(
                    x=x.flatten(),
                    y=pdf_individual[:, i],
                    mode='lines',
                    name=f'{color_group} Component {i+1}',
                    line=dict(color=color_map[color_group], width=1, dash=dash_styles[i % len(dash_styles)]),
                    hovertemplate=(
                        f"<b>Group:</b> {color_group}<br>"
                    )
                ))
                # Calculate H-index for this subpopulation
                h_index += -best_gmm.weights_[i] * np.log(best_gmm.weights_[i]) * np.abs(best_gmm.means_[i][0] - gmm_overall_mean)
            # Add H-index message
            h_index_msg += f"H-index for {color_group}: {h_index:.3f}. "
            data_indices = x_data.index
            if hard_thresholding:
                # predict the component membership for each point (hard thresholding)
                # find the intersection point of the component distributions
                
                # Sort components by mean to ensure that the intersection is calculated between the correct pairs
                sorted_idx = np.argsort(mu)
                pi, mu, sigma = pi[sorted_idx], mu[sorted_idx], sigma[sorted_idx]
                thresholds = []
                for i in range(len(mu) - 1):
                    try: 
                        t = find_intersection(pi[i], mu[i], sigma[i],
                              pi[i+1], mu[i+1], sigma[i+1])
                        thresholds.append(t)
                    except Exception as e:
                        st.error(f"Error finding intersection between {color_group} component {sorted_idx[i]+1} and {sorted_idx[i+1]+1}: either there is no intersection or there are more than one intersection.")
                        #thresholds.append(None)
                # Ensure thresholds are in ascending order
                thresholds = np.sort(thresholds)
                # plot the thresholds
                for threshold in thresholds:
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
                   
                subpopulation_labels = np.digitize(x_data, bins=thresholds)
                # restore the original order of the labels
                subpopulation_labels = sorted_idx[subpopulation_labels]
            else:
                # Predict the component membership for each point (soft thresholding)
                subpopulation_labels = best_gmm.predict(data_2d)
            # Assign the predicted labels (0-based) to the new column in the original DataFrame
            # Add 1 to have 1-based component indexing (e.g., group1, 2, ...)
            assigned_labels = [f"{color_group}_group{label + 1}" for label in subpopulation_labels]
            df.loc[data_indices, "GMM_group"] = assigned_labels
            export_available = True
    if h_index_msg != "": 
        st.info(h_index_msg)
   

    st.plotly_chart(fig, use_container_width=True, key=f"gmm_plot_{selected_var}_{', '.join(color_by)}")
    # have a button to export the GMM group augmented dataframe
    if export_available:
        st.download_button(
            label="Download GMM Grouped Data",
            data=df.to_csv(index=False),
            file_name="gmm_grouped_data.csv",
            mime="text/csv",
            key="gmm_download"
        )
        df.drop(columns=['GMM_group'], inplace=True)

    
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    return fig, h_index_msg


def feature_2d_distribution_plot(df, selected_x, selected_y, color_by=[]):
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=False)
   
    fig = go.Figure()
    for color_group in unique_color_groups:
        group_df = df[df[GROUP_COL_NAME] == color_group]
        if group_df.empty or group_df[selected_x].nunique() < 2 or group_df[selected_y].nunique() < 2:
            # Skip if group is empty or has insufficient data for KDE
            continue
        # Main scatter plot
        fig.add_trace(go.Scatter(
            x=group_df[selected_x],
            y=group_df[selected_y],
            mode='markers',
            name=color_group,
            marker=dict(color=color_map[color_group], size=5, opacity=0.7),
            hovertemplate=(
                f"<b>Group:</b> {color_group}<br>"
                f"<b>{selected_x}:</b> %{{x}}<br>"
                f"<b>{selected_y}:</b> %{{y}}<extra></extra>"
            )
        ))

        # Marginal density for X-axis
        x_data = group_df[selected_x].dropna()
        if not x_data.empty and x_data.nunique() > 1:
            kde_x = gaussian_kde(x_data)
            x_range = np.linspace(x_data.min(), x_data.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_x(x_range),
                mode='lines',
                name=f'{color_group}_x_density',
                line=dict(color=color_map[color_group]),
                yaxis='y2',
                showlegend=False,
                opacity=0.7
            ))

        # Marginal density for Y-axis
        y_data = group_df[selected_y].dropna()
        if not y_data.empty and y_data.nunique() > 1:
            kde_y = gaussian_kde(y_data)
            y_range = np.linspace(y_data.min(), y_data.max(), 200)
            fig.add_trace(go.Scatter(
                x=kde_y(y_range), # X values for the density curve on y-axis marginal
                y=y_range,        # Y values for the density curve
                mode='lines',
                name=f'{color_group}_y_density',
                line=dict(color=color_map[color_group]),
                xaxis='x2',
                showlegend=False,
                opacity=0.7,
                #fill='tozerox', # Fill area to the x-axis (which is x2)
            ))

    fig.update_layout(
        title=f'2D Distribution of {selected_x} and {selected_y} by {", ".join(color_by)}',
        xaxis_title=selected_x,
        yaxis_title=selected_y,
        hovermode='closest',
        # Configure axes for marginal plots
        xaxis=dict(domain=[0, 0.83], showgrid=False, zeroline=False), # Main x-axis, reduced slightly
        yaxis=dict(domain=[0, 0.83], showgrid=False, zeroline=False), # Main y-axis, reduced slightly
        xaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False, showticklabels=False), # Marginal y-density's x-axis
        yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False, showticklabels=False), # Marginal x-density's y-axis
        # Removed bargap and barmode as they are for histograms
    )
    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    return fig

