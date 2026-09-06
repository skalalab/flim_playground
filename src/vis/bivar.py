import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2, gaussian_kde, pearsonr
from sklearn.linear_model import LinearRegression

from src.feature_labels import format_feature_label
from src.export_labels import available_label_column, format_export_group_labels
from src.widgets.gmm_tables import gmm_component_table, gmm_tables_html
from src.widgets.visualization_widgets import gmm_hyperParams_widget

from .helpers import (
    _find_best_gmm,
    add_interleaved_points_trace,
    format_group_label,
    get_context_theme_color,
    get_point_visual_mappings,
    hover_field,
    log_negative_error,
    natural_tuple_sort,
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

def distribution_controls(selected_x, selected_y, marginal_plot_type='gaussian fit'):
    """Render analysis settings separately so changing category never refits."""
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
        marginal = st.selectbox(
            'Marginal Plot Type', ['gaussian fit', 'boxplot', 'violin'],
            index=['gaussian fit', 'boxplot', 'violin'].index(marginal_plot_type),
            key=f'marginal_plot_type_selector_{selected_x}_{selected_y}')
    with col2:
        st.write("")
        st.write("")
        fit_gmm = st.checkbox("2D Gaussian Mixture Model", value=False,
                              key=f"fit_gmm_2d_{selected_x}_{selected_y}")
    with col3:
        st.write("")
        st.write("")
        regression = st.checkbox("Regression line", value=False,
                                 key=f"fit_regression_2d_{selected_x}_{selected_y}")
    max_components, min_weight = gmm_hyperParams_widget() if fit_gmm else (3, .1)
    return dict(log_x=log_x, log_y=log_y, marginal_plot_type=marginal,
                fit_gmm=fit_gmm, fit_regression=regression,
                max_components=max_components, min_weight_threshold=min_weight)


def distribution_fit_groups(df, x_col, y_col, group_col, color_groups, panels,
                            separate_by=None, fit_regression=False, fit_gmm=False,
                            max_components=3, min_weight_threshold=.1, color_by=None):
    """Analyze category × color populations once; shared verbatim with export.

    Shape/opacity never enter the memberships. Position-based assignments preserve
    rows even when dataframe indices or collapsed hover labels are duplicated.
    """
    results = []
    assignments = np.full(len(df), None, dtype=object)
    for level, panel_rows in panels:
        for color_group in color_groups:
            positions = panel_rows[df.iloc[panel_rows][group_col].to_numpy() == color_group]
            if not len(positions):
                continue
            group = df.iloc[positions]
            x, y = group[x_col].to_numpy(), group[y_col].to_numpy()
            result = dict(category=level, color_group=color_group, positions=positions,
                          pearson=None, regression=None, components=[], notices=[])
            valid = len(x) >= 2 and len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2
            if valid:
                coefficient, p_value = pearsonr(x, y)
                result['pearson'] = (float(coefficient), float(p_value))
                if fit_regression:
                    model = LinearRegression().fit(x.reshape(-1, 1), y)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    result['regression'] = dict(
                        x=x_line, y=model.predict(x_line.reshape(-1, 1)),
                        r2=float(model.score(x.reshape(-1, 1), y)),
                        slope=float(model.coef_[0]), intercept=float(model.intercept_))
            else:
                reason = 'fewer than two observations' if len(x) < 2 else 'constant X or Y'
                result['notices'].append(f"Pearson r and regression unavailable: {reason}.")
            if fit_gmm:
                if not valid:
                    result['notices'].append('Skipping GMM: insufficient data or constant X or Y.')
                else:
                    try:
                        model = _find_best_gmm(
                            group[[x_col, y_col]], max_components=max_components,
                            min_weight_threshold=min_weight_threshold)
                    except (ValueError, np.linalg.LinAlgError):
                        model = None
                    if model is None:
                        result['notices'].append('No suitable GMM found with current constraints.')
                    elif model.n_components == 1:
                        result['notices'].append('Only one GMM component found with current constraints.')
                    else:
                        result['components'] = [
                            dict(mean=mean, covariance=cov, weight=float(weight))
                            for mean, cov, weight in zip(model.means_, model.covariances_, model.weights_)]
                        labels = model.predict(group[[x_col, y_col]])
                        assignments[positions] = format_export_group_labels(
                            labels, color_group, separate_by=separate_by,
                            category=level, color_by=color_by)
            results.append(result)
    return results, assignments


def distribution_ranges(df, x_col, y_col, results):
    """Common independent X/Y bounds include all data and fitted overlays."""
    values = [list(df[x_col]), list(df[y_col])]
    radius = np.sqrt(chi2.ppf(.95, df=2))
    for result in results:
        regression = result['regression']
        if regression:
            values[0].extend(regression['x'])
            values[1].extend(regression['y'])
        for component in result['components']:
            for axis in (0, 1):
                center = component['mean'][axis]
                span = radius * np.sqrt(component['covariance'][axis][axis])
                values[axis].extend([center - span, center + span])
    bounds = []
    for axis_values in values:
        low, high = float(np.min(axis_values)), float(np.max(axis_values))
        padding = (high - low) * .05 or max(abs(low) * .05, .5)
        bounds.append([low - padding, high + padding])
    return bounds


def select_distribution_category(fig, category=None):
    """Select prepared points, marginals, fits, and summary without computation."""
    meta = fig.layout.meta
    if not isinstance(meta, dict) or not meta.get('distribution_categories'):
        return fig
    categories = meta['distribution_categories']
    category = category if category in categories else categories[0]
    for trace in fig.data:
        info = trace.meta
        if not isinstance(info, dict) or 'distribution_role' not in info:
            continue
        active = info['category'] == category
        role = info['distribution_role']
        trace.visible = not active if role == 'context' else active
        trace.showlegend = active and role == 'points' and info.get('legend', False)
    fig.update_layout(
        meta={**meta, 'distribution_category': category,
              'distribution_summary': meta['distribution_summaries'][category]},
        legend_uirevision=f"{meta['distribution_separate_by']}:{category}")
    return fig


def feature_2d_distribution_plot(df, unique_row_id_col, fov_name_col, selected_x,
                                 selected_y, color_by=None, shape_by=None, opacity_by=None,
                                 marginal_plot_type='gaussian fit', colormap="tab10",
                                 row_id_label="ID", separate_by=None, analysis_options=None,
                                 label_column=None):
    """One joint distribution per category, with shared coordinates and encodings."""
    import html

    label_column = available_label_column(df.columns, label_column or "2D_GMM_group")
    color_by = [color_by] if isinstance(color_by, str) else list(color_by or [])
    df = df.dropna(subset=[selected_x, selected_y]).copy()
    panels = category_panel_rows(df, separate_by, color_by)
    if df.empty:
        raise ValueError('No complete X/Y observations remain for 2D Feature Distribution.')
    options = (distribution_controls(selected_x, selected_y, marginal_plot_type)
               if analysis_options is None else analysis_options)
    marginal = options.get('marginal_plot_type', marginal_plot_type)
    pretty_x, pretty_y = format_feature_label(selected_x), format_feature_label(selected_y)
    for column, enabled in [(selected_x, options.get('log_x')), (selected_y, options.get('log_y'))]:
        if enabled:
            if (df[column] < 0).any():
                st.error(log_negative_error(column))
            else:
                df[column] = np.log10(df[column] + 1e-6)
    # Keep the returned metadata unchanged; normalize only the rendering copy.
    result_df = df.copy()
    if separate_by:
        for column in dict.fromkeys([*color_by, shape_by, opacity_by]):
            if column:
                df[column] = df[column].astype(str).where(df[column].notna(), 'N/A')
    group_column = 'unique_color_group'
    while group_column in df.columns:
        group_column += '_'
    point_id = unique_row_id_col
    if separate_by:
        point_id = '__distribution_position__'
        while point_id in df.columns:
            point_id += '_'
        df[point_id] = np.arange(len(df))
    grouped, color_map, shape_map, opacity_map, _ = get_point_visual_mappings(
        df, color_by=color_by, shape_by=shape_by, opacity_by=opacity_by,
        group_col_name=group_column, overlap_point=False, colormap=colormap)
    results, assignments = distribution_fit_groups(
        df, selected_x, selected_y, group_column, list(color_map), panels,
        separate_by=separate_by, fit_regression=options.get('fit_regression', False),
        fit_gmm=options.get('fit_gmm', False), max_components=options.get('max_components', 3),
        min_weight_threshold=options.get('min_weight_threshold', .1), color_by=color_by)
    if options.get('fit_gmm') and (separate_by or any(value is not None for value in assignments)):
        result_df[label_column] = assignments

    hover = [hover_field(row_id_label, '%{text}')]
    if fov_name_col is not None:
        hover.append(hover_field(fov_name_col, '%{customdata}'))
    hover.extend([hover_field(pretty_x, '%{x}'), hover_field(pretty_y, '%{y}'), '<extra></extra>'])
    base = go.Figure()
    show_counts = st.session_state.get('plot_show_group_counts', False)
    point_cls = add_interleaved_points_trace(
        base, grouped, color_map, shape_map, opacity_map, [selected_x, selected_y],
        point_id, fov_name_col, hovertemplate=''.join(hover), show_counts=show_counts)
    fig = go.Figure() if separate_by else base
    if separate_by:
        # A single copy per category makes context linear in the number of rows.
        for level, positions in panels:
            fig.add_trace(point_cls(
                x=df.iloc[positions][selected_x], y=df.iloc[positions][selected_y],
                mode='markers', marker=dict(color='#b8b8b8', opacity=.18, symbol='circle', size=3),
                hoverinfo='skip', showlegend=False,
                meta=dict(distribution_role='context', category=level)))
        for level, positions in panels:
            counts = {result['color_group']: len(result['positions']) for result in results
                      if result['category'] == level}
            seen = set()
            for trace in base.data:
                if trace.text is None:
                    continue
                row_positions = np.asarray(trace.text, dtype=int)
                keep = np.isin(row_positions, positions)
                if not keep.any():
                    continue
                spec = trace.to_plotly_json()
                spec.pop('type')
                for field in ('x', 'y', 'customdata'):
                    value = getattr(trace, field)
                    if value is not None:
                        spec[field] = np.asarray(value)[keep]
                spec['text'] = result_df.iloc[row_positions[keep]][unique_row_id_col].to_numpy()
                for field in ('symbol', 'opacity'):
                    spec['marker'][field] = np.asarray(getattr(trace.marker, field))[keep]
                group = trace.legendgroup
                spec.update(name=format_group_label(group, counts[group], show_counts),
                            showlegend=False, meta=dict(distribution_role='points', category=level,
                                                        legend=group not in seen))
                seen.add(group)
                fig.add_trace(type(trace)(**spec))
        for trace in base.data:
            if trace.text is None and trace.showlegend:
                fig.add_trace(trace)

    summaries = {}
    for level, _positions in panels:
        lines, tables = [], []
        for result in results:
            if result['category'] != level:
                continue
            group = result['color_group']
            label = html.escape(str(group))
            group_df = df.iloc[result['positions']]
            if result['pearson'] is not None:
                coefficient, p_value = result['pearson']
                lines.append(f"\n**{label}:** Pearson r = **{coefficient:.2f}** (p = {p_value:.3g})")
            regression = result['regression']
            if regression:
                lines[-1] += (f" · Regression R² = **{regression['r2']:.3f}** "
                              f"(slope = {regression['slope']:.3f}, intercept = {regression['intercept']:.3f})")
                fig.add_trace(point_cls(
                    x=regression['x'], y=regression['y'], mode='lines',
                    line=dict(color=color_map[group], width=2), showlegend=False, legendgroup=str(group),
                    hovertemplate=(f"<b>Regression Line</b><br>R² = {regression['r2']:.3f}"
                                   f"<br>Slope = {regression['slope']:.3f}"
                                   f"<br>Intercept = {regression['intercept']:.3f}<extra></extra>"),
                    meta=dict(distribution_role='regression', category=level) if separate_by else None))
            start = len(fig.data)
            _plot_marginal_density(fig, group_df[selected_x], 'x', color_map[group], group,
                                   marginal, {'yaxis': 'y2'})
            _plot_marginal_density(fig, group_df[selected_y], 'y', color_map[group], group,
                                   marginal, {'xaxis': 'x2', 'yaxis': 'y3'})
            for trace in fig.data[start:]:
                trace.legendgroup = str(group)
                if separate_by:
                    trace.meta = dict(distribution_role='marginal', category=level)
            component_rows = []
            for index, component in enumerate(result['components'], 1):
                mean_x, mean_y = component['mean']
                covariance, weight = component['covariance'], component['weight']
                component_rows.append((index, f"{mean_x:.2f} ± {np.sqrt(covariance[0][0]):.2f}",
                                       f"{mean_y:.2f} ± {np.sqrt(covariance[1][1]):.2f}", f"{weight:.2f}"))
                start = len(fig.data)
                _plot_gmm_ellipse(fig, mean_x, mean_y, covariance, color_map[group], group, index,
                                  scatter_cls=point_cls)
                for trace in fig.data[start:]:
                    trace.legendgroup = str(group)
                    if separate_by:
                        trace.meta = dict(distribution_role='fit', category=level)
            if component_rows:
                tables.append(gmm_component_table(group, component_rows, [selected_x, selected_y]))
            for notice in result['notices']:
                lines.append(f"\n**{label}:** {notice}")
        if tables:
            lines.append(gmm_tables_html(tables))
        summaries[level] = '\n'.join(lines)

    theme_color = get_context_theme_color()
    x_label = f"log₁₀({pretty_x})" if options.get('log_x') else pretty_x
    y_label = f"log₁₀({pretty_y})" if options.get('log_y') else pretty_y
    fig.update_layout(
        title=dict(text=f'2D Distribution of {pretty_x} and {pretty_y} by {", ".join(color_by)}',
                   font=dict(color=theme_color)),
        xaxis=dict(title=dict(text=x_label, font=dict(color=theme_color)),
                   tickfont=dict(color=theme_color), domain=[0, .9], showgrid=False, zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(color=theme_color)),
                   tickfont=dict(color=theme_color), domain=[0, .9], showgrid=True, zeroline=False),
        xaxis2=dict(domain=[.9, 1], anchor='y3', showgrid=False, zeroline=False, showticklabels=False),
        yaxis2=dict(domain=[.9, 1], showgrid=False, zeroline=False, showticklabels=False),
        yaxis3=dict(domain=[0, .9], anchor='x2', matches='y', showgrid=False,
                    zeroline=False, showline=False, showticklabels=False),
        hovermode='closest', legend=dict(groupclick='togglegroup'))
    if separate_by:
        x_range, y_range = distribution_ranges(df, selected_x, selected_y, results)
        fig.update_layout(
            xaxis=dict(range=x_range), yaxis=dict(range=y_range),
            uirevision=f'distribution:{separate_by}:{selected_x}:{selected_y}:{bool(options.get("log_x"))}:{bool(options.get("log_y"))}',
            meta=dict(distribution_categories=[level for level, _ in panels],
                      distribution_separate_by=separate_by, distribution_summaries=summaries))
        # Keep density amplitudes comparable as well as measurement coordinates.
        if marginal == 'gaussian fit':
            for marginal_axis, coordinate in [('yaxis2', 'y'), ('xaxis2', 'x')]:
                peaks = [max(getattr(trace, coordinate)) for trace in fig.data
                         if isinstance(trace.meta, dict) and trace.meta.get('distribution_role') == 'marginal'
                         and getattr(trace, coordinate + 'axis') == marginal_axis.replace('axis', '')]
                if peaks:
                    fig.update_layout(**{marginal_axis: dict(range=[0, max(peaks) * 1.05])})
        select_distribution_category(fig)
        table_md = fig.layout.meta['distribution_summary']
    else:
        table_md = summaries[None]
    return fig, table_md, result_df

def category_panel_rows(df, separate_by=None, color_by=None):
    """Ordered positional memberships, shared verbatim with script export."""
    if separate_by is None:
        return [(None, np.arange(len(df)))]
    if not isinstance(separate_by, str) or separate_by not in df.columns:
        raise ValueError("Separate by must be one categorical column present in the data.")
    colors = [color_by] if isinstance(color_by, str) else list(color_by or [])
    if separate_by in colors:
        raise ValueError("Separate by cannot also be used for Color by.")
    values = df[separate_by].astype(str).where(df[separate_by].notna(), "N/A")
    return [(level, np.flatnonzero(values.to_numpy() == level))
            for level in natural_tuple_sort(values.unique())]


def _phasor_panel_rows(df, separate_by=None, color_by=None):
    """Compatibility entry point for Phasor's shared category memberships."""
    return category_panel_rows(df, separate_by, color_by)


def select_phasor_category(fig, category=None):
    """Select prepared traces without refitting models or changing the G/S axes.

    Apply to a fresh copy of the base figure before shared plot styling adds
    legend swatches. Category controls can then rerun just the display fragment.
    """
    meta = fig.layout.meta
    if not isinstance(meta, dict) or not meta.get("phasor_categories"):
        return fig
    categories = meta["phasor_categories"]
    category = category if category in categories else categories[0]
    for trace in fig.data:
        trace_meta = trace.meta
        if not isinstance(trace_meta, dict) or "phasor_role" not in trace_meta:
            continue
        active = trace_meta["category"] == category
        role = trace_meta["phasor_role"]
        trace.visible = not active if role == "context" else active
        trace.showlegend = active and role == "points" and trace_meta["legend"]
    fig.update_layout(meta={**meta, "phasor_category": category},
                      legend_uirevision=f"{meta['phasor_separate_by']}:{category}")
    return fig


def _compose_phasor_categories(base, df, panels, separate_by, row_id_col,
                              g_col, s_col, point_cls, show_counts):
    """Prepare category foregrounds and context on one full-size set of axes."""
    fig = go.Figure(layout=base.layout)
    fig.update_xaxes(domain=[0, 1], anchor="y", scaleanchor="y", scaleratio=1,
                     constrain="domain")
    fig.update_yaxes(domain=[0, 1], anchor="x", constrain="domain")
    for trace in base.data:
        if trace.text is None and not trace.showlegend:
            fig.add_trace(trace)  # One copy of the semicircle and lifetime references.
    # Context always lies beneath the selected category's encoded points.
    for level, positions in panels:
        fig.add_trace(point_cls(
            x=df.iloc[positions][g_col].to_numpy(), y=df.iloc[positions][s_col].to_numpy(),
            mode="markers", marker=dict(color="#b8b8b8", opacity=.18, symbol="circle", size=3),
            showlegend=False, hoverinfo="skip",
            meta=dict(phasor_role="context", category=level)))
    for level, positions in panels:
        batches, counts = [], {}
        for trace in base.data:
            if trace.text is None:
                continue
            row_positions = np.asarray(trace.text, dtype=int)
            keep = np.isin(row_positions, positions)
            if not keep.any():
                continue
            counts[trace.legendgroup] = counts.get(trace.legendgroup, 0) + int(keep.sum())
            batches.append((trace, row_positions, keep))
        seen_colors = set()
        for trace, row_positions, keep in batches:
            spec = trace.to_plotly_json()
            spec.pop("type")
            for field in ("x", "y", "customdata"):
                value = getattr(trace, field)
                if value is not None:
                    spec[field] = np.asarray(value)[keep]
            spec["text"] = df.iloc[row_positions[keep]][row_id_col].to_numpy()
            for field in ("symbol", "opacity"):
                spec["marker"][field] = np.asarray(getattr(trace.marker, field))[keep]
            group = trace.legendgroup
            spec.update(name=format_group_label(group, counts[group], show_counts),
                        xaxis="x", yaxis="y", showlegend=False,
                        meta=dict(phasor_role="points", category=level,
                                  legend=group not in seen_colors))
            seen_colors.add(group)
            fig.add_trace(type(trace)(**spec))
    # Shape/opacity swatches retain the global mappings when a category changes.
    for trace in base.data:
        if trace.text is None and trace.showlegend:
            fig.add_trace(trace)
    plot_height = .6 / 1.1
    fig.update_layout(height=round(1000 * plot_height + 90),
                      margin=dict(l=30, r=10, t=50, b=40),
                      legend=dict(orientation="v", yref="paper", y=1, yanchor="auto",
                                  xref="paper", x=1.02, xanchor="left", groupclick="togglegroup"),
                      uirevision=f"phasor:{separate_by}:{g_col}:{s_col}",
                      meta={"phasor_subplot_layout": {"plot_height": plot_height},
                            "phasor_categories": [level for level, _ in panels],
                            "phasor_separate_by": separate_by})
    return select_phasor_category(fig)


def phasor_plot(df, unique_row_id_col, fov_name_col, selected_channel, color_by=[], shape_by=None, opacity_by=None, colormap="tab10", f=0.08, harmonic=1, row_id_label="ID", separate_by: str | None = None):
    color_by = [color_by] if isinstance(color_by, str) else list(color_by or [])
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
    df = df[df[g_feature].notna() & df[s_feature].notna()].copy()
    panels = _phasor_panel_rows(df, separate_by, color_by)
    if df.empty:
        raise ValueError("No complete G/S observations remain for Phasor Plot.")
    # Keep metadata in the returned data, normalizing only the rendering copy.
    original_df = df.copy()
    if separate_by:
        for column in dict.fromkeys([*color_by, shape_by, opacity_by]):
            if column:
                df[column] = df[column].astype(str).where(df[column].notna(), "N/A")

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
    while GROUP_COL_NAME in df.columns:
        GROUP_COL_NAME += "_"
    point_id_col = unique_row_id_col
    if separate_by:
        point_id_col = "__phasor_position__"
        while point_id_col in df.columns:
            point_id_col += "_"
        df[point_id_col] = np.arange(len(df))
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
    # Convert grouped iterator to list so we can iterate multiple times
    grouped_list = list(grouped)

    hover = hover_field(row_id_label, "%{text}")
    if fov_name_col is not None:
        hover += hover_field(fov_name_col, "%{customdata}")
    hover += hover_field("g", "%{x:.4g}") + hover_field("s", "%{y:.4g}")

    # Add all points using interleaved plotting function
    point_cls = add_interleaved_points_trace(
        fig=fig,
        grouped=grouped_list,
        color_map=color_map,
        shape_map=shape_map,
        opacity_map=opacity_map,
        axis_labels=[g_feature, s_feature],
        text_col=point_id_col,
        customdata_col=fov_name_col,
        # Label identifiers because they may be generated row numbers.
        hovertemplate=hover,
        show_counts=st.session_state.get("plot_show_group_counts", False)
    )
    if separate_by:
        return (_compose_phasor_categories(fig, original_df, panels, separate_by, unique_row_id_col,
                                          g_feature, s_feature, point_cls,
                                          st.session_state.get("plot_show_group_counts", False)),
                original_df)
    return fig, original_df
