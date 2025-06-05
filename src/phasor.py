import plotly.graph_objects as go
import numpy as np
import seaborn as sns

def get_phasor_features(decay_curve, shifted_irf, time_axis, f=0.08, offset=0, harmonic=1):

    """
    Calculate the phasor features for a given decay curve
    Args:
        decay_curve: the decay curve to be fitted
        shifted_irf: the shifted irf
        time_axis: the time axis
        f: laser repetition rate in [GHz]
        offset: the offset of the decay curve
        harmonic: the harmonic of the decay curve
    """
    decay_curve = decay_curve - offset
    # clip the timebin to above or equal to 0
    decay_curve = np.clip(decay_curve, 0, None)
    w = 2*np.pi*f
    G_IRF = np.dot(np.transpose(shifted_irf) , np.cos(w*time_axis)) / np.sum(shifted_irf)
    S_IRF = np.dot(np.transpose(shifted_irf) , np.sin(w*time_axis)) / np.sum(shifted_irf)
    cos_coeff = np.cos(w*time_axis)
    sin_coeff = np.sin(w*time_axis)
    
    # corrected coefficients
    corrected_cos_coeff = (G_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (S_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff
    corrected_sin_coeff = (-S_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (G_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff

    decay_curve_sum = np.sum(decay_curve)
    G = np.dot(decay_curve, corrected_cos_coeff) / decay_curve_sum
    S = np.dot(decay_curve, corrected_sin_coeff) / decay_curve_sum

    phi = np.arctan2(G, S) 
    m = np.sqrt(G**2 + S**2)
    tau_phase = 1/w * np.tan(phi)
    tau_m = 1/w * np.sqrt(1/m**2 - 1)
    return G,S, tau_phase, tau_m

def phasor_plot(df, f=0.08):

    # Create the figure
    fig = go.Figure()

    # Set axis limits
    fig.update_layout(
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 0.55]), 
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
            font=dict(size=10),
            xanchor='left'
        )
    
    # Add titles and axis labels
    title = 'Phasor Plot'
    fig.update_layout(
        title=title,
        xaxis_title='g',
        yaxis_title='s',
        font=dict(size=15),
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15, family='Arial', color='black')),
        yaxis=dict(title_font=dict(size=15, family='Arial', color='black'))
    )

    # Add text inside the plot
    fig.add_annotation(
        x=0.8,
        y=0.5,
        text=f"{f * 1000} MHz",
        showarrow=False,
        font=dict(size=15, color='black'),
        xanchor='left'
    )

    # Maintain 1:2 aspect ratio with dynamic sizing
    fig.update_layout(
        autosize=True,  # Let Plotly automatically resize based on the screen
        xaxis=dict(scaleanchor="y"),  # Maintain aspect ratio (1:2)
        margin=dict(l=10, r=10, t=50, b=10),  # Adjust margins as needed
        hovermode='closest'
    )
    
    # plot the phasor coordinates
    
    unique_color_groups = df["color_category"].unique()
    alpha = 0.6 if len(unique_color_groups) > 1 else 1.0
    palette = sns.color_palette("tab10", n_colors=len(unique_color_groups))
    color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {alpha})" for color in palette]
    color_map = {t: color_sequence[i] for i, t in enumerate(unique_color_groups)}

    for g in unique_color_groups:
        g_df =  df[df["color_category"] == g]
        fig.add_trace(
            go.Scatter(
                x=g_df['G'],
                y=g_df['S'],
                mode='markers',
                name=f'{g}',
                text=g_df["base_name"],
                customdata=g_df["image_name"],
                hovertemplate="<b>%{text}</b>",
                marker=dict(color=color_map[g],size=3)
            ),
        )
    return fig
