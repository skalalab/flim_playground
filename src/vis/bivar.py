from .helpers import _prepare_group_data, _find_best_gmm
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from src.widgets.visualization_widgets import gmm_hyperParams_widget
import streamlit as st
from src.vis.helpers import get_point_visual_mappings, add_point_legend_traces
from src.feature_types import unique_cell_id_col, fov_name_col

def _plot_marginal_density(fig, data, axis_type, color, name_prefix, plot_type, plotly_axis_params):
    """Helper function to plot marginal densities."""
    if data.empty or data.nunique() < 2:
        return

    if plot_type == 'gaussian fit':
        kde = gaussian_kde(data)
        data_range = np.linspace(data.min(), data.max(), 200)
        if axis_type == 'x':
            fig.add_trace(go.Scatter(
                x=data_range,
                y=kde(data_range),
                mode='lines',
                name=f'{name_prefix}_x_density',
                line=dict(color=color),
                yaxis=plotly_axis_params['yaxis'],
                showlegend=False,
                opacity=0.7
            ))
        elif axis_type == 'y':
            fig.add_trace(go.Scatter(
                x=kde(data_range), # For y-axis marginal, KDE values are x
                y=data_range,    # and original data range is y
                mode='lines',
                name=f'{name_prefix}_y_density',
                line=dict(color=color),
                xaxis=plotly_axis_params['xaxis'],
                showlegend=False,
                opacity=0.7
            ))
    elif plot_type == 'boxplot':
        if axis_type == 'x':
            fig.add_trace(go.Box(
                x=data,
                name=f'{name_prefix}_x_box',
                marker_color=color,
                yaxis=plotly_axis_params['yaxis'],
                showlegend=False
            ))
        elif axis_type == 'y':
            fig.add_trace(go.Box(
                y=data,
                name=f'{name_prefix}_y_box',
                marker_color=color,
                xaxis=plotly_axis_params['xaxis'],
                showlegend=False
            ))
    elif plot_type == 'violin':
        if axis_type == 'x':
            fig.add_trace(go.Violin(
                x=data,
                name=f'{name_prefix}_x_violin',
                line_color=color,
                yaxis=plotly_axis_params['yaxis'],
                showlegend=False,
                points=False # Hide points for a cleaner look
            ))
        elif axis_type == 'y':
            fig.add_trace(go.Violin(
                y=data,
                name=f'{name_prefix}_y_violin',
                line_color=color,
                xaxis=plotly_axis_params['xaxis'],
                showlegend=False,
                points=False # Hide points for a cleaner look
            ))

def _plot_gmm_ellipse(fig, mean_x, mean_y, cov, color, name_prefix, i):
    """Helper function to plot GMM ellipses."""
   # Calculate eigenvalues and eigenvectors for ellipse orientation
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

    # Create confidence ellipse (e.g., 2-sigma ~ 95% confidence)
    confidence_level = 2  # 2-sigma
    width = 2 * confidence_level * np.sqrt(eigenvals[0])
    height = 2 * confidence_level * np.sqrt(eigenvals[1])

    # Generate ellipse points
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = (width/2) * np.cos(theta)
    ellipse_y = (height/2) * np.sin(theta)

    # Rotate ellipse
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    ellipse_x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle + mean_x
    ellipse_y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle + mean_y

    # Add ellipse to plot
    fig.add_trace(go.Scatter(
        x=ellipse_x_rot,
        y=ellipse_y_rot,
        mode='lines',
        line=dict(color=color, width=2, dash='dash'),
        name=f'{name_prefix} GMM {i+1}',
        showlegend=False,  # Don't clutter legend with ellipses
        hoverinfo='skip'   # Don't show hover for ellipse lines
    ))


def feature_2d_distribution_plot(df, selected_x, selected_y, color_by=[], shape_by=None, opacity_by=None, marginal_plot_type='gaussian fit'):
    GROUP_COL_NAME = 'unique_color_group'
    # Use the new helper for color, shape, opacity
    grouped, color_map, shape_map, opacity_map, group_keys = get_point_visual_mappings(
        df,
        color_by=color_by,
        shape_by=shape_by,
        opacity_by=opacity_by,
        group_col_name=GROUP_COL_NAME,
        overlap_point=False
    )
    fig = go.Figure()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_marginal_plot_type = st.selectbox(
            'Marginal Plot Type',
            ['gaussian fit', 'boxplot', 'violin'],
            index=['gaussian fit', 'boxplot', 'violin'].index(marginal_plot_type), # Set default based on function arg
            key=f'marginal_plot_type_selector_{selected_x}_{selected_y}' # More unique key
        )
    with col2:
        st.write("")
        st.write("")
        fit_gmm = st.checkbox("Fit a 2D Gaussian Mixture Model", value=True)
    if fit_gmm:
        fit_gmm_max_components, fit_gmm_min_weight_threshold = gmm_hyperParams_widget()

    table_md = []
    for group_key, group_df in grouped:
        # Unpack group_key for color, shape, opacity
        color_group = group_key[0] if isinstance(group_key, tuple) else group_key
        shape_group = None
        opacity_group = None
        key_idx = 1
        if shape_by and shape_by in df.columns:
            shape_group = group_key[key_idx] if len(group_key) > key_idx else None
            key_idx += 1
        if opacity_by and opacity_by in df.columns:
            opacity_group = group_key[key_idx] if len(group_key) > key_idx else None
        if group_df.empty or group_df[selected_x].nunique() < 2 or group_df[selected_y].nunique() < 2:
            continue
        # Main scatter plot
        fig.add_trace(go.Scatter(
            x=group_df[selected_x],
            y=group_df[selected_y],
            mode='markers',
            name=color_group,
            text=group_df[unique_cell_id_col],
            customdata=group_df[fov_name_col],
            marker=dict(
                color=color_map.get(color_group, 'gray'),
                size=5,
                opacity=opacity_map.get(opacity_group, 0.7) if opacity_map else 0.7,
                symbol=shape_map.get(shape_group, 'circle') if shape_map else 'circle',
            ),
            hovertemplate=(
                f"<b>Group:</b> {color_group}<br>"
                f"<b>Cell ID:</b> %{{text}}<br>"
                f"<b>Image:</b> %{{customdata}}<br>"
                f"<b>{selected_x}:</b> %{{x}}<br>"
                f"<b>{selected_y}:</b> %{{y}}<extra></extra>"
            )
        ))

        # annotate the correlation coefficient and p-value of the current group
        corr_coef, p_value = pearsonr(group_df[selected_x], group_df[selected_y])
        table_md += [f"\n**{color_group}:**"]
        table_md.append(f"Correlation Coefficient b/w {selected_x} and {selected_y}: **{corr_coef:.2f}** (p-value: {p_value:.2f})")

        if fit_gmm: 
            # Fit GMM for the current group
            group_data_2d = group_df[[selected_x, selected_y]]
            if len(group_data_2d) > 1: # Need at least 2 points for GMM, ideally more
                best_gmm = _find_best_gmm(group_data_2d, max_components=fit_gmm_max_components, min_weight_threshold=fit_gmm_min_weight_threshold) # Example: try up to 2 components
                if best_gmm and best_gmm.n_components > 1:
                    table_md += ["\n**GMM Components:**"]
                    table_md.append("")
                    table_md.append(f"| Component | **{selected_x}** | **{selected_x}** | **{selected_y}** | **{selected_y}** | Weight |")
                    table_md.append(f"|------|-----|-----|-----|-----|------|")
                    table_md.append(f"| | **Mean** | **Std Dev** | **Mean** | **Std Dev** | |")

                    for i in range(best_gmm.n_components):
                        mean = best_gmm.means_[i]
                        cov = best_gmm.covariances_[i]
                        mean_x, mean_y = mean
                        std_x = np.sqrt(cov[0][0])
                        std_y = np.sqrt(cov[1][1])
                        weight = best_gmm.weights_[i]

                        table_md.append(f"| {i+1} | {mean_x:.2f} | {std_x:.2f} | {mean_y:.2f} | {std_y:.2f} | {weight:.2f} |")
                        # plot the gmm component using Ellipse
                        _plot_gmm_ellipse(fig, mean_x, mean_y, cov, color_map[color_group], color_group, i+1)
                    # use the best gmm model to predict the component membership of the current group
                    data_indices = group_data_2d.index
                    subpopulation_labels = best_gmm.predict(group_data_2d)
                    assigned_labels = [f"{color_group}_group{label + 1}" for label in subpopulation_labels]
                    df.loc[data_indices, "2D_GMM_group"] = assigned_labels
                elif best_gmm and best_gmm.n_components == 1:
                    st.write(f"Only one GMM component found for {color_group} with current constraints.")
                else:
                    st.write(f"No suitable GMM found for {color_group} with current constraints.")
            else:
                st.write(f"\nSkipping GMM for group: {color_group} due to insufficient data (points: {len(group_data_2d)})")

        # Marginal density for X-axis
        x_data = group_df[selected_x].dropna()
        _plot_marginal_density(fig, x_data, 'x', color_map.get(color_group, 'gray'), color_group, selected_marginal_plot_type, plotly_axis_params={'yaxis': 'y2'})

        # Marginal density for Y-axis
        y_data = group_df[selected_y].dropna()
        _plot_marginal_density(fig, y_data, 'y', color_map.get(color_group, 'gray'), color_group, selected_marginal_plot_type, plotly_axis_params={'xaxis': 'x2'})

    # Add legend traces for opacity and shape
    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)

    fig.update_layout(
        title=f'2D Distribution of {selected_x} and {selected_y} by {", ".join(color_by)}',
        xaxis_title=selected_x,
        yaxis_title=selected_y,
        hovermode='closest',
        # Configure axes for marginal plots
        xaxis=dict(domain=[0, 0.9], showgrid=False, zeroline=False), # Main x-axis, reduced slightly
        yaxis=dict(domain=[0, 0.9], showgrid=False, zeroline=False), # Main y-axis, reduced slightly
        xaxis2=dict(domain=[0.9, 1], showgrid=False, zeroline=False, showticklabels=False), # Marginal y-density's x-axis
        yaxis2=dict(domain=[0.9, 1], showgrid=False, zeroline=False, showticklabels=False), # Marginal x-density's y-axis
        # Removed bargap and barmode as they are for histograms
    )

    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    table_md = "\n".join(table_md)
    return fig, table_md, df


def phasor_plot(df, channel_name,color_by=[], shape_by=None, opacity_by=None, f=0.08, harmonic=1):

    # Create the figure
    fig = go.Figure()

    # Consolidate all layout settings into one call
    if harmonic == 1:
        harmonic_str = "1st"
    elif harmonic == 2:
        harmonic_str = "2nd"
    fig.update_layout(
        title=f'{channel_name} {harmonic_str} Harmonic Phasor',
        xaxis=dict(
            range=[-0.05, 1.05],
            title='g',
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            scaleanchor="y"
        ),
        yaxis=dict(
            range=[0.15, 0.55],
            title='s',
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False
        ),
        font=dict(size=15),
        title_font=dict(size=20, family='Arial', color='black'),
        autosize=True,
        margin=dict(l=0, r=10, t=50, b=60),  # Increased bottom margin for x-axis title
        hovermode='closest'
    )

    # Plot the curve
    u = np.arange(0, 100, 0.01)
    x_curve = 1 / (1 + u**2)
    y_curve = u / (1 + u**2)

    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        line=dict(color='black'),
        name='Curve', 
        hoverinfo='skip',# Hide the hover info for this trace
        showlegend=False 
    ))

    # Calculate and plot specific points
    wt = 2 * np.pi * f * np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    x_points = 1 / (1 + wt**2)
    y_points = wt / (1 + wt**2)

    fig.add_trace(go.Scatter(
        x=x_points,
        y=y_points,
        mode='markers',
        marker=dict(size=8, color='black'),
        name='Lifetime Markers', 
        hoverinfo='skip', # Hide the hover info for this trace,
        showlegend=False
    ))

    # Annotate the points
    lifetime_labels = ['0.5 ns', '1 ns', '2 ns', '3 ns', '4 ns', '5 ns']
    labels = len(lifetime_labels)
    label_coords = list(zip(x_points - 0.02, y_points + 0.03))[:labels]

    for i in range(labels):
        fig.add_annotation(
            x=label_coords[i][0],
            y=label_coords[i][1],
            text=lifetime_labels[i],
            showarrow=False,
            font=dict(size=12),
            xanchor='left'
        )
    
    # Add text inside the plot
    fig.add_annotation(
        x=0.8,
        y=0.5,
        text=f"f = {f * 1000} MHz",
        showarrow=False,
        font=dict(size=15, color='black'),
        xanchor='left'
    )
    
    # plot the phasor coordinates
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df, color_by, GROUP_COL_NAME, overlap_point=True)

    # Add support for shape_by and opacity_by
    shape_map = None
    opacity_map = None
    shape_groups = None
    opacity_groups = None
    if shape_by and shape_by in df.columns:
        shape_groups = sorted(df[shape_by].dropna().unique())
        # Use helpers to assign symbols
        symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'star']
        shape_map = {group: symbols[i % len(symbols)] for i, group in enumerate(shape_groups)}
    if opacity_by and opacity_by in df.columns:
        opacity_groups = sorted(df[opacity_by].dropna().unique())
        opacities = np.linspace(0.4, 1.0, num=len(opacity_groups))
        opacity_map = {group: opacities[i] for i, group in enumerate(opacity_groups)}

    feature_prefix = "Lifetime fit free_" + channel_name + ": "
    if harmonic == 1:
        g_feature = f"{feature_prefix}G(1st)"
        s_feature = f"{feature_prefix}S(1st)"
    elif harmonic == 2:
        g_feature = f"{feature_prefix}G(2nd)"
        s_feature = f"{feature_prefix}S(2nd)"
    for g in unique_color_groups:
        g_df =  df[df[GROUP_COL_NAME] == g]   
        # For each shape/opacity group within this color group
        if shape_map or opacity_map:
            # Group by shape and/or opacity
            group_cols = []
            if shape_map:
                group_cols.append(shape_by)
            if opacity_map:
                group_cols.append(opacity_by)
            if group_cols:
                grouped = g_df.groupby(group_cols)
                for group_keys, sub_df in grouped:
                    # Unpack group_keys
                    if shape_map and opacity_map:
                        shape_group, opacity_group = group_keys
                    elif shape_map:
                        shape_group = group_keys
                        opacity_group = None
                    elif opacity_map:
                        shape_group = None
                        opacity_group = group_keys
                    else:
                        shape_group = opacity_group = None
                    marker_symbol = shape_map[shape_group] if shape_group is not None else 'circle'
                    marker_opacity = opacity_map[opacity_group] if opacity_group is not None else 0.8
                    fig.add_trace(
                        go.Scatter(
                            x=sub_df[g_feature],
                            y=sub_df[s_feature],
                            mode='markers',
                            name=f'{g}',
                            text=sub_df[unique_cell_id_col],
                            customdata=sub_df[fov_name_col],
                            hovertemplate="<b>%{text}</b>",
                            marker=dict(color=color_map[g], size=5, symbol=marker_symbol, opacity=marker_opacity)
                        ),
                    )
        else:
            fig.add_trace(
                go.Scatter(
                    x=g_df[g_feature],
                    y=g_df[s_feature],
                    mode='markers',
                    name=f'{g}',
                    text=g_df[unique_cell_id_col],
                    customdata=g_df[fov_name_col],
                    hovertemplate="<b>%{text}</b>",
                    marker=dict(color=color_map[g],size=5)
                ),
            )
    # Add shape/opacity legends if needed
    from .helpers import add_point_legend_traces
    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)
    return fig
