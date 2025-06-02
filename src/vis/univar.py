import streamlit as st
import plotly.graph_objects as go
from itertools import combinations
import numpy as np

from src.widgets.visualization_widgets import histogram_bin_width_widget
from src.widgets.visualization_widgets import gmm_hyperParams_widget
from .helpers import _prepare_group_data, find_intersection, _add_effect_size_annotations, _find_best_gmm

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
            hard_thresholding_possible = hard_thresholding
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
                        st.error(f"Error finding intersection between {color_group} component {sorted_idx[i]+1} and component {sorted_idx[i+1]+1}: either there is no intersection or there are more than one intersection.")
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
                        # write out the thresholds
                        st.write(f"Threshold for {color_group} between component {sorted_idx[i]+1} and component {sorted_idx[i+1]+1}: **{threshold:.2f}**")

                    subpopulation_labels = np.digitize(x_data, bins=thresholds)
                    # restore the original order of the labels
                    subpopulation_labels = sorted_idx[subpopulation_labels]
            if not hard_thresholding_possible:
                # Predict the component membership for each point (soft thresholding)
                data_2d = x_data.values.reshape(-1, 1)
                subpopulation_labels = best_gmm.predict(data_2d)
            # Assign the predicted labels (0-based) to the new column in the original DataFrame
            # Add 1 to have 1-based component indexing (e.g., group1, 2, ...)
            assigned_labels = [f"{color_group}_group{label + 1}" for label in subpopulation_labels]
            df.loc[data_indices, "GMM_group"] = assigned_labels
            
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