

from .helpers import _prepare_group_data
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde

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