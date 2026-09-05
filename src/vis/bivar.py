import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import ConvexHull
from scipy.stats import chi2, gaussian_kde, pearsonr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.feature_labels import format_feature_label
from src.widgets.analysis_widget_state import number_input_default
from src.widgets.gmm_tables import gmm_component_table, gmm_tables_html
from src.widgets.visualization_widgets import gmm_hyperParams_widget

from .helpers import (
    _find_best_gmm,
    add_interleaved_points_trace,
    get_context_theme_color,
    get_point_visual_mappings,
    hover_field,
    log_negative_error,
)


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
                yaxis=plotly_axis_params.get('yaxis', 'y'),
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
                yaxis=plotly_axis_params.get('yaxis', 'y'),
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
                yaxis=plotly_axis_params.get('yaxis', 'y'),
                showlegend=False,
                points=False # Hide points for a cleaner look
            ))

def _create_phasor_background(fig, theme_color, f=0.08, harmonic=1):
    """
    Helper function to create the phasor semicircle, axes, annotations, and lifetime markers.
    Adds these elements to the provided figure.

    ``harmonic`` is the harmonic the plotted G/S coordinates were computed at
    (see fov_extraction.py, which passes ``harmonic=h`` to
    ``phasor.phasor_from_signal``). The n-th harmonic phasor is evaluated at
    n·ω, so the reference geometry must use n·2πf as well.
    """
    # Plot the curve
    u = np.arange(0, 100, 0.01)
    x_curve = 1 / (1 + u**2)
    y_curve = u / (1 + u**2)

    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        line=dict(color=theme_color),
        name='Curve', 
        hoverinfo='skip',# Hide the hover info for this trace
        showlegend=False
    ))

    # Add S axis line (vertical line from (0,0) to (0,0.5))
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, 0.5],
        mode='lines',
        line=dict(color=theme_color, width=2),
        name='S Axis',
        hoverinfo='skip',
        showlegend=False
    ))

    # Add G axis line (horizontal line from (0,0) to (1,0))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 0],
        mode='lines',
        line=dict(color=theme_color, width=2),
        name='G Axis',
        hoverinfo='skip',
        showlegend=False
    ))

    # Add axis annotations
    # S axis annotation at 0.5
    fig.add_annotation(
        x=-0.02,
        y=0.5,
        text="0.5",
        showarrow=False,
        font=dict(size=12, color=theme_color),
        xanchor='right',
        yanchor='middle'
    )

    # G axis annotation at 0
    fig.add_annotation(
        x=0,
        y=-0.02,
        text="0",
        showarrow=False,
        font=dict(size=12, color=theme_color),
        xanchor='center',
        yanchor='top'
    )

    # G axis annotation at 1
    fig.add_annotation(
        x=1,
        y=-0.02,
        text="1",
        showarrow=False,
        font=dict(size=12, color=theme_color),
        xanchor='center',
        yanchor='top'
    )

    # Lifetime markers. The n-th harmonic phasor is evaluated at n*omega, so a marker
    # for tau belongs at n*2*pi*f*tau. The semicircle is parameterised by omega*tau and
    # needs no harmonic correction.
    wt = 2 * np.pi * f * harmonic * np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    x_points = 1 / (1 + wt**2)
    y_points = wt / (1 + wt**2)

    fig.add_trace(go.Scatter(
        x=x_points,
        y=y_points,
        mode='markers',
        marker=dict(size=7, color=theme_color),
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
            font=dict(size=12, color=theme_color),
            xanchor='left'
        )

    # Add text inside the plot. Report the frequency the geometry is drawn at, which
    # for harmonic n is n x the laser repetition rate.
    freq_text = f"f = {f * harmonic * 1000} MHz"
    if harmonic != 1:
        freq_text += f"<br>({harmonic} x {f * 1000} MHz)"
    fig.add_annotation(
        x=0.8,
        y=0.5,
        text=freq_text,
        showarrow=False,
        font=dict(size=15, color=theme_color),
        xanchor='left'
    )

def _plot_gmm_ellipse(fig, mean_x, mean_y, cov, color, name_prefix, i, scatter_cls=go.Scatter):
    """Helper function to plot GMM ellipses."""
   # Calculate eigenvalues and eigenvectors for ellipse orientation
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    # Scale the ellipse to 95% probability for a 2D Gaussian.
    r = np.sqrt(chi2.ppf(0.95, df=2))
    width = 2 * r * np.sqrt(eigenvals[0])
    height = 2 * r * np.sqrt(eigenvals[1])

    # Generate ellipse points
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = (width/2) * np.cos(theta)
    ellipse_y = (height/2) * np.sin(theta)

    # Rotate and center ellipse
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    ellipse_x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle + mean_x
    ellipse_y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle + mean_y

    # Use the points' renderer so the ellipse can layer above them in WebGL.
    fig.add_trace(scatter_cls(
        x=ellipse_x_rot,
        y=ellipse_y_rot,
        mode='lines',
        line=dict(color=color, width=2, dash='dash'),
        name=f'{name_prefix} GMM {i+1}',
        showlegend=False,  # Don't clutter legend with ellipses
        hoverinfo='skip'   # Don't show hover for ellipse lines
    ))

def feature_2d_distribution_plot(df, unique_row_id_col, fov_name_col, selected_x, selected_y, color_by=[], shape_by=None, opacity_by=None, marginal_plot_type='gaussian fit', colormap="tab10", row_id_label="ID"):
    GROUP_COL_NAME = 'unique_color_group'
    # Create valid copy to allow modification
    df = df.copy()

    # Pretty FLIM labels (Greek notation) reused for hover tooltips and axis titles
    pretty_x = format_feature_label(selected_x)
    pretty_y = format_feature_label(selected_y)

    # Squeezed columns for log checks: smaller ratio for the first two
    col_log_x, col_log_y, col1, col2, col3 = st.columns([0.8, 0.8, 2, 2, 2])
    with col_log_x:
        st.write("")
        st.write("")
        log_x = st.checkbox("Log X", value=False, key=f"log_x_2d_{selected_x}_{selected_y}")
    with col_log_y:
        st.write("")
        st.write("")
        log_y = st.checkbox("Log Y", value=False, key=f"log_y_2d_{selected_x}_{selected_y}")
    with col1:
        selected_marginal_plot_type = st.selectbox(
            'Marginal Plot Type',
            ['gaussian fit', 'boxplot', 'violin'],
            index=['gaussian fit', 'boxplot', 'violin'].index(marginal_plot_type),
            key=f'marginal_plot_type_selector_{selected_x}_{selected_y}'
        )
    with col2:
        st.write("")
        st.write("")
        fit_gmm = st.checkbox("2D Gaussian Mixture Model", value=False, key=f"fit_gmm_2d_{selected_x}_{selected_y}")
    with col3:
        st.write("")
        st.write("")
        fit_regression = st.checkbox("Regression line", value=False, key=f"fit_regression_2d_{selected_x}_{selected_y}")

    if log_x:
        if (df[selected_x] < 0).any():
            st.error(log_negative_error(selected_x))
        else:
            df[selected_x] = np.log10(df[selected_x] + 1e-6)
    if log_y:
        if (df[selected_y] < 0).any():
            st.error(log_negative_error(selected_y))
        else:
            df[selected_y] = np.log10(df[selected_y] + 1e-6)

    grouped, color_map, shape_map, opacity_map, group_keys = get_point_visual_mappings(
        df,
        color_by=color_by,
        shape_by=shape_by,
        opacity_by=opacity_by,
        group_col_name=GROUP_COL_NAME,
        overlap_point=False,
        colormap=colormap
    )
    fig = go.Figure()



    if fit_gmm:
        fit_gmm_max_components, fit_gmm_min_weight_threshold = gmm_hyperParams_widget()

    # Convert grouped iterator to list so we can iterate multiple times
    grouped_list = list(grouped)

    # Use original column names and escape them with hover_field for Plotly markup.
    hover_lines = [hover_field(row_id_label, "%{text}")]
    if fov_name_col is not None:
        hover_lines.append(hover_field(fov_name_col, "%{customdata}"))
    hover_lines.append(hover_field(pretty_x, "%{x}"))
    hover_lines.append(hover_field(pretty_y, "%{y}"))
    hover_lines.append("<extra></extra>")   # hide the default trace info box
    point_cls = add_interleaved_points_trace(
        fig=fig,
        grouped=grouped_list,
        color_map=color_map,
        shape_map=shape_map,
        opacity_map=opacity_map,
        axis_labels=[selected_x, selected_y],
        text_col=unique_row_id_col,
        customdata_col=fov_name_col,
        hovertemplate="".join(hover_lines),
        show_counts=st.session_state.get("plot_show_group_counts", False)
    )

    table_md = []
    gmm_tables = []
    # Per-group analysis keys on colour groups only, never per shape/opacity subgroup:
    # those two channels affect point styling and nothing else.
    for color_group in color_map.keys():
        group_df = df[df[GROUP_COL_NAME] == color_group]
        if group_df.empty or group_df[selected_x].nunique() < 2 or group_df[selected_y].nunique() < 2:
            continue

        # annotate the correlation coefficient and p-value of the current group
        corr_coef, p_value = pearsonr(group_df[selected_x], group_df[selected_y])
        table_md.append(f"\n**{color_group}:** Pearson r = **{corr_coef:.2f}** (p = {p_value:.3g})")
        x_data = group_df[selected_x].dropna()
        y_data = group_df[selected_y].dropna()

        if len(x_data) >= 2 and len(y_data) >= 2:
            # Ensure x_data and y_data have the same indices (both drop NaN)
            valid_indices = x_data.index.intersection(y_data.index)
            x_clean = x_data.loc[valid_indices].values.reshape(-1, 1)
            y_clean = y_data.loc[valid_indices].values

        if fit_regression:
            if len(x_clean) >= 2:  # Need at least 2 points for regression
                # Fit linear regression
                reg_model = LinearRegression()
                reg_model.fit(x_clean, y_clean)

                # Calculate R²
                y_pred = reg_model.predict(x_clean)
                r2 = r2_score(y_clean, y_pred)

                # Create regression line points for plotting
                x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_range = reg_model.predict(x_range.reshape(-1, 1))

                # Use the points' renderer so the regression line can layer above them.
                fig.add_trace(point_cls(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color=color_map.get(color_group, 'black'), width=2),
                    showlegend=False,
                    hovertemplate=f'<b>Regression Line</b><br>R² = {r2:.3f}<br>Slope = {reg_model.coef_[0]:.3f}<br>Intercept = {reg_model.intercept_:.3f}<extra></extra>'
                ))

                # Keep correlation and regression together in one line per group.
                table_md[-1] += f" · Regression R² = **{r2:.3f}** (slope = {reg_model.coef_[0]:.3f}, intercept = {reg_model.intercept_:.3f})"

        _plot_marginal_density(fig, x_data, 'x', color_map.get(color_group, 'gray'), color_group, selected_marginal_plot_type, plotly_axis_params={'yaxis': 'y2'})

        _plot_marginal_density(fig, y_data, 'y', color_map.get(color_group, 'gray'), color_group, selected_marginal_plot_type, plotly_axis_params={'xaxis': 'x2', 'yaxis': 'y3'})

    # --- GMM fitting per color group (not per shape/opacity) ---
    if fit_gmm:
        for color_group in color_map.keys():
            # Filter data for this color group using the helper column
            group_df = df[df[GROUP_COL_NAME] == color_group]

            if group_df.empty or group_df[selected_x].nunique() < 2 or group_df[selected_y].nunique() < 2:
                continue

            # Fit GMM for the current COLOR group (aggregating all shapes/opacities)
            group_data_2d = group_df[[selected_x, selected_y]]
            if len(group_data_2d) > 1: # Need at least 2 points for GMM
                best_gmm = _find_best_gmm(group_data_2d, max_components=fit_gmm_max_components, min_weight_threshold=fit_gmm_min_weight_threshold)
                if best_gmm and best_gmm.n_components > 1:
                    component_rows = []
                    for i in range(best_gmm.n_components):
                        mean = best_gmm.means_[i]
                        cov = best_gmm.covariances_[i]
                        mean_x, mean_y = mean
                        std_x = np.sqrt(cov[0][0])
                        std_y = np.sqrt(cov[1][1])
                        weight = best_gmm.weights_[i]

                        component_rows.append((
                            i + 1, f"{mean_x:.2f} ± {std_x:.2f}",
                            f"{mean_y:.2f} ± {std_y:.2f}", f"{weight:.2f}",
                        ))
                        # plot the gmm component using Ellipse
                        _plot_gmm_ellipse(fig, mean_x, mean_y, cov, color_map[color_group], color_group, i+1, scatter_cls=point_cls)
                    gmm_tables.append(gmm_component_table(
                        color_group, component_rows, [selected_x, selected_y]))
                    # use the best gmm model to predict the component membership of the current group
                    data_indices = group_data_2d.index
                    subpopulation_labels = best_gmm.predict(group_data_2d)
                    assigned_labels = [f"{color_group}_group{label + 1}" for label in subpopulation_labels]
                    df.loc[data_indices, "2D_GMM_group"] = assigned_labels
                elif best_gmm and best_gmm.n_components == 1:
                    table_md.append(f"\nOnly one GMM component found for {color_group} with current constraints.")
                else:
                    table_md.append(f"\nNo suitable GMM found for {color_group} with current constraints.")
            else:
                table_md.append(f"\nSkipping GMM for group: {color_group} due to insufficient data (points: {len(group_data_2d)})")

    # Note: Legend traces and hovermode are already added by add_interleaved_points_trace
    # Just update layout with additional settings
    theme_color = get_context_theme_color()

    # Set axis labels based on log transform (pretty_x/pretty_y defined above)
    x_axis_label = f"log₁₀({pretty_x})" if log_x else pretty_x
    y_axis_label = f"log₁₀({pretty_y})" if log_y else pretty_y

    fig.update_layout(
        title=dict(
            text=f'2D Distribution of {pretty_x} and {pretty_y} by {", ".join(color_by)}',
            font=dict(color=theme_color)
        ),
        xaxis=dict(
            title=dict(text=x_axis_label, font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            domain=[0, 0.9],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text=y_axis_label, font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            domain=[0, 0.9],
            showgrid=True,
            zeroline=False
        ),
        # Configure axes for marginal plots
        xaxis2=dict(domain=[0.9, 1], anchor='y3', showgrid=False, zeroline=False, showticklabels=False), # Marginal y-density's x-axis
        yaxis2=dict(domain=[0.9, 1], showgrid=False, zeroline=False, showticklabels=False), # Marginal x-density's y-axis
        # Match the main Y range without extending its grid through the marginal.
        yaxis3=dict(domain=[0, 0.9], anchor='x2', matches='y', showgrid=False,
                    zeroline=False, showline=False, showticklabels=False),
    )

    # remove the column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    if gmm_tables:
        table_md.append(gmm_tables_html(gmm_tables))
    table_md = "\n".join(table_md)
    return fig, table_md, df

def _cluster_hull_polygon(pts):
    """Return one cluster's boundary polygon (unclosed) for its (x, y) points.

    Deduplicates first because Qhull fails on repeated points, and falls back to a
    small circle around the mean when there are fewer than three unique points,
    which have no hull. Kept free of Plotly/Streamlit so the exported script can
    inline this exact source and draw identical boundaries.
    """
    uniq = np.unique(np.asarray(pts, dtype=float), axis=0)
    if uniq.shape[0] >= 3:
        hull = ConvexHull(uniq)
        return uniq[hull.vertices]
    center = uniq.mean(axis=0)
    r = max(np.linalg.norm(uniq - center, axis=1).max(initial=0.0), 0.01)
    theta = np.linspace(0, 2 * np.pi, 80)
    return np.c_[center[0] + 1.2 * r * np.cos(theta),
                 center[1] + 1.2 * r * np.sin(theta)]


def _plot_convex_hull(
    fig,
    df,
    g_col,
    s_col,
    theme_color,
    label_col="k_means_cluster",
    polygon_color="#1f77b4",
    centers_raw=None,
    line_width=2,
    scatter_cls=go.Scatter,
):
    """
    Overlay per-cluster convex hull polygons (same color) and black × centroids
    onto an existing Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Existing figure that already has your (G,S) scatter.
    df : DataFrame
        Must contain columns g_col, s_col, and label_col (int cluster labels).
    g_col, s_col : str
        Column names for raw G and S used in your scatter.
    label_col : str
        Column with integer labels from k-means (e.g., "k_means_cluster").
    polygon_color : str
        Single color for all polygons (hex like "#1f77b4" or a named color).
    centers_raw : array-like, shape (k,2), optional
        Centroids in RAW (G,S) units (e.g., `scaler.inverse_transform(kmeans.cluster_centers_)`).
        If None, centroid positions are computed as the mean (G,S) per cluster.
    line_width : int
        Polygon outline width.
    """
    # 1) Draw a convex hull polygon for each cluster
    unique_clusters = sorted([c for c in df[label_col].unique() if c >= 0])

    for c in unique_clusters:
        sub = df.loc[df[label_col] == c, [g_col, s_col]].dropna()
        if sub.empty:
            continue

        poly = _cluster_hull_polygon(sub.to_numpy())

        fig.add_trace(scatter_cls(
            x=np.r_[poly[:, 0], poly[0, 0]],
            y=np.r_[poly[:, 1], poly[0, 1]],
            mode="lines",
            line=dict(color=polygon_color, width=line_width),
            name=f"Cluster {int(c)} boundary",
            showlegend=False,
        ))

    # 2) Centroids as black crosses
    if centers_raw is None:
        centers_raw = (df.groupby(label_col)[[g_col, s_col]]
                       .mean().reindex(unique_clusters).to_numpy())

    fig.add_trace(scatter_cls(
        x=centers_raw[:, 0],
        y=centers_raw[:, 1],
        mode="markers",
        marker=dict(symbol="x", size=14, line=dict(width=1.5, color=theme_color), color=polygon_color),
        hovertemplate="<b>Centroid</b><br>g: %{x:.2f}<br>s: %{y:.2f}<extra></extra>",
        name="Centroids",
        showlegend=False
    ))
    return fig

def phasor_kmeans(X_raw, n_clusters, random_state=42):
    """Standardize raw (G, S) coordinates, K-Means cluster them, and return
    (labels, cluster centers transformed back to raw coordinates).

    n_init=10 keeps the best of 10 seeded k-means++ restarts so clusters are
    stable. Must stay free of Streamlit dependencies — it is embedded verbatim
    into exported analysis scripts via inspect.getsource(), which is also why the
    threadpoolctl import below sits inside the function rather than at module scope.
    """
    # A single OpenMP thread avoids synchronization overhead for two-column coordinates.
    from threadpoolctl import threadpool_limits

    scaler = StandardScaler().fit(X_raw)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    with threadpool_limits(1):
        kmeans.fit(scaler.transform(X_raw))
    centers_raw = scaler.inverse_transform(kmeans.cluster_centers_)
    return kmeans.labels_, centers_raw


def phasor_plot(df, unique_row_id_col, fov_name_col, selected_channel, color_by=[], shape_by=None, opacity_by=None, colormap="tab10", f=0.08, harmonic=1, row_id_label="ID"):

    # Get theme color once at the start for all theme-aware elements
    theme_color = get_context_theme_color()

    # Create the figure
    fig = go.Figure()

    feature_prefix = "Lifetime fit free_" + selected_channel + ": "
    if harmonic == 1:
        harmonic_str = "1st"
        g_feature = f"{feature_prefix}G(1st)"
        s_feature = f"{feature_prefix}S(1st)"
    elif harmonic == 2:
        harmonic_str = "2nd"
        g_feature = f"{feature_prefix}G(2nd)"
        s_feature = f"{feature_prefix}S(2nd)"

    # drop rows with NaN values in the g_feature and s_feature columns
    df = df[df[g_feature].notna() & df[s_feature].notna()]

    fig.update_layout(
        title=dict(text=f'{selected_channel} {harmonic_str} Harmonic Phasor', font=dict(size=20, family='Arial', color=theme_color)),
        xaxis=dict(
            range=[-0.05, 1.05],
            title=dict(text='g', font=dict(color=theme_color), standoff=5),
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            scaleanchor="y"
        ),
        yaxis=dict(
            range=[-0.05, 0.55],
            title=dict(text='s', font=dict(color=theme_color), standoff=0),
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False
        ),
        font=dict(size=15),
        autosize=True,
        margin=dict(l=30, r=10, t=50, b=40),
        hovermode='closest'
    )

    # Create phasor background (semicircle, axes, annotations, lifetime markers)
    _create_phasor_background(fig, theme_color, f, harmonic)

    # plot the phasor coordinates
    GROUP_COL_NAME = 'unique_color_group'
    # Use the unified helper for color, shape, opacity
    grouped, color_map, shape_map, opacity_map, group_keys = get_point_visual_mappings(
        df,
        color_by=color_by,
        shape_by=shape_by,
        opacity_by=opacity_by,
        group_col_name=GROUP_COL_NAME,
        overlap_point=True,
        colormap=colormap
    )
    col1, col2 = st.columns(2)
    with col1:
        st.write("")
        st.write("")
        k_means = st.checkbox("Perform K-Means clustering", value=False, key=f"k_means_phasor_{selected_channel}")
    if k_means:
        with col2:
            k_means_clusters = st.number_input("Number of clusters", value=number_input_default(st.session_state, f"k_means_clusters_phasor_{selected_channel}", 2), min_value=1, max_value=8, step=1, key=f"k_means_clusters_phasor_{selected_channel}")

    # Convert grouped iterator to list so we can iterate multiple times
    grouped_list = list(grouped)

    # Add all points using interleaved plotting function
    point_cls = add_interleaved_points_trace(
        fig=fig,
        grouped=grouped_list,
        color_map=color_map,
        shape_map=shape_map,
        opacity_map=opacity_map,
        axis_labels=[g_feature, s_feature],
        text_col=unique_row_id_col,
        customdata_col=fov_name_col,
        # Label identifiers because they may be generated row numbers.
        hovertemplate=hover_field(row_id_label, "%{text}"),
        show_counts=st.session_state.get("plot_show_group_counts", False)
    )


    # --- K-Means clustering per color group (not per shape/opacity) ---
    if k_means:
        for color_group in color_map.keys():
            # Filter data for this color group using the helper column
            group_df = df[df[GROUP_COL_NAME] == color_group]

            if group_df.empty:
                continue

            # cluster on a standardized copy — raw G,S stay untouched
            X_raw = group_df[[g_feature, s_feature]].to_numpy(copy=True)
            labels, centers_raw = phasor_kmeans(X_raw, k_means_clusters)

            # plot the convex hull
            # Create a temporary dataframe with cluster labels for plotting
            group_df_with_clusters = group_df.copy()
            group_df_with_clusters["k_means_cluster"] = labels

            _plot_convex_hull(fig, group_df_with_clusters, g_feature, s_feature, theme_color, "k_means_cluster", color_map.get(color_group, 'gray'), centers_raw, line_width=2, scatter_cls=point_cls)

            # Update the main dataframe with cluster labels
            assigned_labels = [f"{color_group}_group{label + 1}" for label in labels]
            df.loc[group_df.index, "k_means_cluster"] = assigned_labels

    # Note: Legend traces and hovermode are already added by add_interleaved_points_trace
    # Remove the temporary group column after plotting
    df.drop(columns=[GROUP_COL_NAME], inplace=True)

    return fig, df
