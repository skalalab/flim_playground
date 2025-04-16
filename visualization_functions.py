import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.mixture import GaussianMixture
from widgets.custom_widgets import stats_comparison_pair_widget, histogram_bin_width_widget

def glass_delta(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    group2_sd = np.std(group2, ddof=1)  # Using Bessel's correction with ddof=1
    return mean_diff / group2_sd

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


def feature_comparison_plot(df, selected_var, compared_by, stats_test="None"):
    fig = go.Figure()
    df['compare_group'] = df[compared_by].agg('_'.join, axis=1)
    compare_groups = df['compare_group'].unique()
    compare_pairs = list(combinations(compare_groups, 2))
    color_map = create_color_map(compare_groups, overlap_point=False)
    jitter_amount = 1
    point_size = 5

    # --- 1. Plotting Traces using go.Box (with hidden box) ---
    for group in compare_groups:
        # Filter data for the current group
        g_df = df[df['compare_group'] == group].copy()
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
            fillcolor='rgba(0,0,0,0)',  # Transparent fill
            line_color='rgba(0,0,0,0)', # Transparent box outline
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
    if compare_pairs != [] and stats_test != "None":
        selected_pairs = stats_comparison_pair_widget(compare_pairs)
        if selected_pairs != []:
            if stats_test == "Glass's Delta":
                # Calculate glasser's delta for each pair
                
                # Keep track of the highest y-position used for annotations so far
                max_annotation_y = -np.inf 
                # Define vertical spacing parameters
                offset_from_data = 0.05 * (df[selected_var].max() - df[selected_var].min()) # Initial offset based on data range
                vertical_spacing = 0.08 * (df[selected_var].max() - df[selected_var].min()) # Space between annotations
                bracket_vertical_length = 0.03 * (df[selected_var].max() - df[selected_var].min()) # Length of bracket arms
                text_offset_from_bracket = 0.02 * (df[selected_var].max() - df[selected_var].min()) # Space between bracket and text

                # Sort pairs based on the x-position to draw lower annotations first (optional but can help)
                sorted_pairs = sorted(selected_pairs, key=lambda p: max(compare_groups.tolist().index(p[0]), compare_groups.tolist().index(p[1])))

                for pair in sorted_pairs:
                    group1 = df[df['compare_group'] == pair[0]][selected_var]
                    group2 = df[df['compare_group'] == pair[1]][selected_var]
                    # Skip if either group is empty
                    if group1.empty or group2.empty:
                        continue
                        
                    delta = glass_delta(group1, group2)
                    # Add annotation to the figure
                    # Get indices for positioning
                    x_indices = [compare_groups.tolist().index(pair[0]), compare_groups.tolist().index(pair[1])]
                    x_positions = sorted(x_indices)
                    
                    # Determine the highest data point under this annotation range
                    current_pair_max_y = max(group1.max(), group2.max())
                    
                    # Calculate initial desired position for the bracket top
                    y_bracket_top_initial = current_pair_max_y + offset_from_data
                    
                    # Check if this position is below the highest annotation drawn so far
                    # If so, place it above the highest one with spacing
                    y_bracket_top = max(y_bracket_top_initial, max_annotation_y + vertical_spacing)
                    
                    # Calculate the final text position
                    y_text_annotation = y_bracket_top + text_offset_from_bracket

                    # Update the highest y position used
                    max_annotation_y = y_text_annotation 

                    # Add horizontal line for the top of the square bracket
                    fig.add_shape(
                        type="line",
                        x0=x_positions[0],
                        y0=y_bracket_top,
                        x1=x_positions[1],
                        y1=y_bracket_top,
                        line=dict(color="black", width=1.5),
                    )
                    
                    # Add vertical lines for the sides of square brackets
                    for x_pos in x_positions:
                        fig.add_shape(
                            type="line",
                            x0=x_pos,
                            y0=y_bracket_top,
                            x1=x_pos,
                            y1=y_bracket_top - bracket_vertical_length, 
                            line=dict(color="black", width=1.5),
                        )
                    
                    # Add text annotation above the bracket
                    fig.add_annotation(
                        x=(x_positions[0] + x_positions[1])/2,
                        y=y_text_annotation, 
                        text=f"Δ={delta:.2f}",
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        align="center"
                    )

    df.drop(columns=['compare_group'], inplace=True)
    return fig

def dimension_reduction_plot(df, method="UMAP", colored_by=[], exp_var=None):
    """create a plotly plot to visualize the dimension-reduced data
    """
    fig = go.Figure()
    if method == "Principal Component Analysis":
        axis_labels = ["PC1", "PC2"]
    elif method == "UMAP":
        axis_labels = ["UMAP1", "UMAP2"]
    else:
        axis_labels = ["dim1", "dim2"]

    # colored by unique combinations of the selected categorical columns
    df['unique_color_group'] = df[colored_by].agg('_'.join, axis=1)
    unique_color_groups = df['unique_color_group'].unique()

    color_map = create_color_map(unique_color_groups, overlap_point=True)

    # plot scatter plot iteratively, once for each color group
    for g in unique_color_groups:
        g_df =  df[df['unique_color_group'] == g]
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
    # remove the column after plotting
    df.drop(columns=['unique_color_group'], inplace=True)
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
    df['unique_color_group'] = df[color_by].agg('_'.join, axis=1)
    unique_color_groups = df['unique_color_group'].unique()
    # Using solid colors for lines
   
    color_map = create_color_map(unique_color_groups, overlap_point=False)
    fig = go.Figure()

    # Determine common binning for all groups to make lines comparable
    # Handle potential NaN values before calculating min/max
    valid_data = df[selected_var].dropna()
    if valid_data.empty:
        # Handle case where there's no valid data at all
        st.warning(f"No valid data found for variable '{selected_var}' to plot.")
        # remove the column before returning an empty figure
        df.drop(columns=['unique_color_group'], inplace=True)
        return fig # Return empty figure

    bin_edges = histogram_bin_width_widget(valid_data)

    for color_group in unique_color_groups:
        group_df = df[df['unique_color_group'] == color_group]
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
    df.drop(columns=['unique_color_group'], inplace=True)
    return fig

def feature_gmm_plot(df, selected_var, color_by=[]):
    df['unique_color_group'] = df[color_by].agg('_'.join, axis=1)
    unique_color_groups = df['unique_color_group'].unique()
    # Using solid colors for lines
   
    color_map = create_color_map(unique_color_groups, overlap_point=False)
    fig = go.Figure()

    # Determine common binning for all groups to make lines comparable
    # Handle potential NaN values before calculating min/max
    valid_data = df[selected_var].dropna()
    if valid_data.empty:
        # Handle case where there's no valid data at all
        st.warning(f"No valid data found for variable '{selected_var}' to plot.")
        # remove the column before returning an empty figure
        df.drop(columns=['unique_color_group'], inplace=True)
        return fig # Return empty figure
    
    # fit a Gaussian Mixture Model (GMM) to each color group
    for color_group in unique_color_groups:
        group_df = df[df['unique_color_group'] == color_group]
        x_data = group_df[selected_var].dropna()

        if x_data.empty:
            continue # Skip empty groups

        # Fit GMM to the data
        # --- Fit GMMs with 1 to 3 components ---
        data_2d = x_data.values.reshape(-1, 1)  # GMM expects 2D array
        aic_scores = []
        gmms = []
        for n in range(1, 4): 
            gmm = GaussianMixture(n_components=n, random_state=42).fit(data_2d)
            gmms.append(gmm)
            aic_scores.append(gmm.aic(data_2d))
        
        # Select best model based on AIC
        best_idx = np.argmin(aic_scores)
        best_gmm = gmms[best_idx]
        n_components = best_idx + 1
        print(f"Best GMM for {color_group} has {n_components} components with AIC: {aic_scores[best_idx]}")
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
                f"<b>Count:</b> %{{y}}<extra></extra>"
            )
        ))
        # Plot individual components
        for i in range(n_components):
            fig.add_trace(go.Scatter(
                x=x.flatten(),
                y=pdf_individual[:, i],
                mode='lines',
                name=f'{color_group} Component {i+1}',
                line=dict(color=color_map[color_group], width=1, dash='dash'),
                hovertemplate=(
                    f"<b>Group:</b> {color_group}<br>"
                    f"<b>Count:</b> %{{y}}<extra></extra>"
                )
            ))
    fig.update_layout(
        title=f'Gaussian Mixture Model of {selected_var} by {", ".join(color_by)}',
        xaxis_title=selected_var,
        yaxis_title='Density',
        legend_title_text='Groups',
        hovermode='x unified', # Good for comparing counts at specific x-values
        margin=dict(l=50, r=20, t=50, b=80)
    )
    # remove the column after plotting
    df.drop(columns=['unique_color_group'], inplace=True)
    return fig