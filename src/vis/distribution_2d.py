"""2D Feature Distribution: scatter with marginals, regression line and 2D GMM.

The pure half (`gmm_ellipse_params`, `distribution_fit_groups`, `distribution_ranges`)
is shared with standalone export and embedded with inspect.getsource(); the Plotly half
draws it in the app.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde, pearsonr
from sklearn.linear_model import LinearRegression

from src.column_roles import code_span
from src.export_labels import available_label_column, format_export_group_labels
from src.feature_labels import format_feature_label
from src.widgets.visualization_widgets import gmm_hyperParams_widget
from .dimension_facets import category_facet_groups, category_panel_rows, dimension_facet_layout
from .gmm_tables import gmm_component_table, gmm_group_title, gmm_tables_html
from .helpers import (
    _find_best_gmm,
    add_interleaved_points_trace,
    get_point_visual_mappings,
    hover_field,
    log10_columns,
    log_negative_error,
)
from .plot_defaults import OVERVIEW_MODEL_WIDTH, PANEL_CONTEXT_ALPHA, PANEL_CONTEXT_COLOR, PANEL_MODEL_WIDTH, get_context_theme_color

MARGINAL_OPTIONS = ['None', 'gaussian fit', 'boxplot', 'violin']


def _plot_marginal_density(fig, data, axis_type, color, name_prefix, plot_type, plotly_axis_params):
    """Helper function to plot marginal densities."""
    if plot_type in (None, 'None') or data.empty or data.nunique() < 2:
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


def gmm_ellipse_params(covariance):
    """Angle in degrees, width and height of a component's 95% probability ellipse.

    Shared by the app's Plotly outline and the exported script's Ellipse patch.
    """
    from scipy.stats import chi2

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues * chi2.ppf(0.95, df=2))
    return angle, width, height


def _plot_gmm_ellipse(fig, mean_x, mean_y, cov, color, name_prefix, i, scatter_cls=go.Scatter):
    """Helper function to plot GMM ellipses."""
    angle, width, height = gmm_ellipse_params(cov)

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

def distribution_controls(selected_x, selected_y, marginal_plot_type='None'):
    """Render analysis settings separately so changing a setting never refits twice."""
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
        marginal_key = f'marginal_plot_type_selector_{selected_x}_{selected_y}'
        # Prune before the keyed widget renders: a session value Streamlit cannot
        # offer raises instead of falling back.
        if st.session_state.get(marginal_key) not in (None, *MARGINAL_OPTIONS):
            st.session_state[marginal_key] = 'None'
        marginal = st.selectbox(
            'Marginal Plot Type', MARGINAL_OPTIONS,
            index=MARGINAL_OPTIONS.index(marginal_plot_type), key=marginal_key)
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
    from scipy.stats import chi2

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


def _join_level_blocks(blocks, panels, separate_by):
    """List every level in panel order; each line already names its own level."""
    if not separate_by:
        return blocks[None]

    return '\n'.join(f"\n{blocks[level]}" for level, _positions in panels)


def render_distribution_component_tables(tables, separate_by, component_editor):
    """Edit each level's component tables in turn, keeping a level's tables together.

    The grid shows every level at once, and ``gmm_group_title`` names the level
    in each table's own title, so no heading is needed above them.
    """
    names = {}
    for level in dict.fromkeys(table['category'] for table in tables):
        if component_editor is not None:
            names.update(component_editor(
                [table for table in tables if table['category'] == level]))
    return names


def feature_2d_distribution_plot(df, unique_row_id_col, fov_name_col, selected_x,
                                 selected_y, color_by=None, shape_by=None, opacity_by=None,
                                 marginal_plot_type='None', colormap="tab10",
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
    df, skipped = log10_columns(df, [column for column, enabled in
                                     [(selected_x, options.get('log_x')), (selected_y, options.get('log_y'))]
                                     if enabled])
    for column in skipped:
        st.error(log_negative_error(column))
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
    composition = None
    panel_axis = {}
    if separate_by:
        x_range, y_range = distribution_ranges(df, selected_x, selected_y, results)
        composition = dimension_facet_layout(
            category_facet_groups(panels), x_range, y_range, aspect=1.)
        # Overview first, then one panel per level. Panel p owns x{p+4}/y{p+4};
        # 2 and 3 stay reserved for the overview's marginal strips.
        slots = [(None, np.arange(len(df)), 'x', 'y', True)]
        for index, (level, positions) in enumerate(panels):
            suffix = str(index + 4)
            panel_axis[level] = (f'x{suffix}', f'y{suffix}')
            slots.append((level, positions, f'x{suffix}', f'y{suffix}', False))
        panel_slot = {level: index + 1 for index, (level, _positions) in enumerate(panels)}
        for slot, (level, positions, xaxis, yaxis, overview) in enumerate(slots):
            if not overview:
                # Each panel gets its own "everyone else" context trace of
                # N - n_level points, rather than one shared trace toggled by
                # visibility. That is O(levels x rows), not O(rows): the figure
                # ends up drawing (levels + 1) x N points across the overview
                # and every panel's own copy plus its context.
                other = np.setdiff1d(np.arange(len(df)), positions, assume_unique=False)
                if len(other):
                    fig.add_trace(point_cls(
                        x=df.iloc[other][selected_x], y=df.iloc[other][selected_y],
                        xaxis=xaxis, yaxis=yaxis, mode='markers',
                        marker=dict(color=PANEL_CONTEXT_COLOR, opacity=PANEL_CONTEXT_ALPHA, symbol='circle'),
                        hoverinfo='skip', showlegend=False,
                        meta=dict(facet_role='context', facet_slot=slot)))
            for trace in base.data:
                if trace.text is None:  # Shape/opacity legend swatches carry no points.
                    continue
                row_positions = np.asarray(trace.text, dtype=int)
                keep = np.isin(row_positions, positions)
                if not keep.any():
                    continue
                spec = trace.to_plotly_json()
                spec.pop('type')
                # Read the original arrays: to_plotly_json may serialize numerics.
                for field in ('x', 'y', 'customdata'):
                    value = getattr(trace, field)
                    if value is not None:
                        spec[field] = np.asarray(value)[keep]
                spec['text'] = result_df.iloc[row_positions[keep]][unique_row_id_col].to_numpy()
                for field in ('symbol', 'opacity'):
                    spec['marker'][field] = np.asarray(getattr(trace.marker, field))[keep]
                # One shared legend on the overview reports whole-dataset counts,
                # so the base trace's name and legend flag are kept verbatim.
                spec.update(xaxis=xaxis, yaxis=yaxis,
                            showlegend=bool(overview and trace.showlegend),
                            meta=dict(facet_role='points', facet_slot=slot))
                fig.add_trace(type(trace)(**spec))
        for trace in base.data:
            if trace.text is None and trace.showlegend:
                fig.add_trace(trace)

    summaries = {}
    statistics_summaries = {}
    component_tables = []
    for level, _positions in panels:
        lines, tables = [], []
        for result in results:
            if result['category'] != level:
                continue
            group = result['color_group']
            # The combined title carries a level name read from the file, which
            # the removed heading used to neutralise; a code span still does.
            label = (code_span(gmm_group_title(level, group)) if separate_by
                     else html.escape(str(group)))
            # A model belongs to its own panel; colour already encodes Color by
            # and leaves no channel to attribute an overlay on the overview.
            overlay_axes = dict(zip(('xaxis', 'yaxis'), panel_axis[level])) if separate_by else {}
            overlay_width = PANEL_MODEL_WIDTH if separate_by else OVERVIEW_MODEL_WIDTH
            if result['pearson'] is not None:
                coefficient, p_value = result['pearson']
                lines.append(f"\n**{label}:** Pearson r = **{coefficient:.2f}** (p = {p_value:.3g})")
            regression = result['regression']
            if regression:
                lines[-1] += (f" · Regression R² = **{regression['r2']:.3f}** "
                              f"(slope = {regression['slope']:.3f}, intercept = {regression['intercept']:.3f})")
                fig.add_trace(point_cls(
                    x=regression['x'], y=regression['y'], mode='lines', **overlay_axes,
                    line=dict(color=color_map[group], width=overlay_width),
                    showlegend=False, legendgroup=str(group),
                    hovertemplate=(f"<b>Regression Line</b><br>R² = {regression['r2']:.3f}"
                                   f"<br>Slope = {regression['slope']:.3f}"
                                   f"<br>Intercept = {regression['intercept']:.3f}<extra></extra>"),
                    meta=dict(facet_role='regression',
                              facet_slot=panel_slot[level]) if separate_by else None))
            component_rows = []
            metadata_rows = []
            for index, component in enumerate(result['components'], 1):
                mean_x, mean_y = component['mean']
                covariance, weight = component['covariance'], component['weight']
                x_mean_sd = f"{mean_x:.2f} ± {np.sqrt(covariance[0][0]):.2f}"
                y_mean_sd = f"{mean_y:.2f} ± {np.sqrt(covariance[1][1]):.2f}"
                component_rows.append((index, x_mean_sd, y_mean_sd, f"{weight:.2f}"))
                metadata_rows.append({
                    'source_label': format_export_group_labels(
                        [index - 1], group, separate_by=separate_by,
                        category=level, color_by=color_by)[0],
                    'component': index,
                    'x_mean_sd': x_mean_sd,
                    'y_mean_sd': y_mean_sd,
                    'weight': float(weight),
                })
                start = len(fig.data)
                _plot_gmm_ellipse(fig, mean_x, mean_y, covariance, color_map[group], group, index,
                                  scatter_cls=point_cls)
                for trace in fig.data[start:]:
                    trace.legendgroup = str(group)
                    trace.line.width = overlay_width
                    if separate_by:
                        trace.update(**{f'{name}axis': value
                                        for name, value in zip('xy', panel_axis[level])})
                        trace.meta = dict(facet_role='fit', facet_slot=panel_slot[level])
            if component_rows:
                tables.append(gmm_component_table(gmm_group_title(level, group),
                                                  component_rows, [selected_x, selected_y]))
                component_tables.append({
                    'category': level,
                    'group': group,
                    'features': [selected_x, selected_y],
                    'rows': metadata_rows,
                })
            for notice in result['notices']:
                lines.append(f"\n**{label}:** {notice}")
        statistics_summaries[level] = '\n'.join(lines)
        if tables:
            lines.append(gmm_tables_html(tables))
        summaries[level] = '\n'.join(lines)

    # Strips belong to the main block and describe whatever occupies it, so a
    # separated figure prepares one set per level beside the whole-dataset set
    # and promotion only changes which one is visible. No cross-level amplitude
    # scaling is needed: one set is shown at a time.
    strip_sets = [(0, df)]
    if separate_by:
        strip_sets += [(index + 1, df.iloc[positions])
                       for index, (_level, positions) in enumerate(panels)]
    for slot, subset in strip_sets:
        for group in color_map:
            group_df = subset[subset[group_column] == group]
            start = len(fig.data)
            _plot_marginal_density(fig, group_df[selected_x], 'x', color_map[group], group,
                                   marginal, {'yaxis': 'y2'})
            _plot_marginal_density(fig, group_df[selected_y], 'y', color_map[group], group,
                                   marginal, {'xaxis': 'x2', 'yaxis': 'y3'})
            for trace in fig.data[start:]:
                trace.legendgroup = str(group)
                if separate_by:
                    trace.meta = dict(facet_role='marginal', facet_slot=slot)
                    trace.visible = slot == 0

    theme_color = get_context_theme_color()
    x_label = f"log₁₀({pretty_x})" if options.get('log_x') else pretty_x
    y_label = f"log₁₀({pretty_y})" if options.get('log_y') else pretty_y
    block_x = composition['overview']['x_domain'] if separate_by else [0., 1.]
    block_y = composition['overview']['y_domain'] if separate_by else [0., 1.]
    span = .9 if marginal not in (None, 'None') else 1.
    main_x = block_x[0] + span * (block_x[1] - block_x[0])
    main_y = block_y[0] + span * (block_y[1] - block_y[0])
    axis_layout = dict(
        xaxis=dict(title=dict(text=x_label, font=dict(color=theme_color)),
                   tickfont=dict(color=theme_color), domain=[block_x[0], main_x],
                   showgrid=False, zeroline=False),
        yaxis=dict(title=dict(text=y_label, font=dict(color=theme_color)),
                   tickfont=dict(color=theme_color), domain=[block_y[0], main_y],
                   showgrid=True, zeroline=False))
    if marginal not in (None, 'None'):
        axis_layout.update(
            xaxis2=dict(domain=[main_x, block_x[1]], anchor='y3', showgrid=False,
                        zeroline=False, showticklabels=False),
            yaxis2=dict(domain=[main_y, block_y[1]], showgrid=False, zeroline=False,
                        showticklabels=False),
            yaxis3=dict(domain=[block_y[0], main_y], anchor='x2', matches='y',
                        showgrid=False, zeroline=False, showline=False,
                        showticklabels=False))
    if separate_by:
        axis_layout['xaxis']['range'] = x_range
        axis_layout['yaxis']['range'] = y_range
        for index, panel in enumerate(composition['panels']):
            suffix = str(index + 4)
            for dimension, bounds, domain in (('x', x_range, panel['x_domain']),
                                              ('y', y_range, panel['y_domain'])):
                other = 'y' if dimension == 'x' else 'x'
                axis_layout[f'{dimension}axis{suffix}'] = dict(
                    domain=domain, anchor=f'{other}{suffix}', range=bounds, matches=dimension,
                    tickfont=dict(color=theme_color), showticklabels=False,
                    showgrid=False, zeroline=False,
                    showline=True, linecolor='black', linewidth=1, mirror=False, ticks='')
    fig.update_layout(
        title=dict(text=f'2D Distribution of {pretty_x} and {pretty_y} by {", ".join(color_by)}',
                   font=dict(color=theme_color)),
        hovermode='closest', legend=dict(groupclick='togglegroup'), **axis_layout)
    statistics = _join_level_blocks(statistics_summaries, panels, separate_by)
    table_md = _join_level_blocks(summaries, panels, separate_by)
    meta = dict(gmm_component_tables=component_tables, distribution_statistics=statistics)
    if separate_by:
        for panel in composition['panels']:
            fig.add_annotation(
                x=panel['x_domain'][1], y=sum(panel['y_domain']) / 2,
                xref='paper', yref='paper', text=html.escape(str(panel['values'][0])),
                showarrow=False, xanchor='left', yanchor='middle', xshift=6,
                font=dict(color=theme_color, size=14))
        # Promotion writes into this one; it is created empty so the annotation
        # count and indices stay frozen for the resize script.
        fig.add_annotation(
            x=.5, y=1., xref='paper', yref='paper', text='', showarrow=False,
            xanchor='center', yanchor='top', yshift=-4,
            font=dict(color=theme_color, size=14),
            bgcolor=('rgba(255,255,255,0.75)' if theme_color == 'black'
                     else 'rgba(30,30,30,0.75)'))
        fig.update_layout(
            height=round(1000 * composition['plot_height'] + 160),
            margin=dict(l=80, r=140, t=70, b=90, pad=0, autoexpand=True),
            # Reserve the legend's measured height below the axis title.
            legend=dict(orientation='h', yref='container', y=0, yanchor='bottom',
                        xref='paper', x=0, groupclick='togglegroup'),
            uirevision=f'distribution:{separate_by}:{selected_x}:{selected_y}:'
                       f'{bool(options.get("log_x"))}:{bool(options.get("log_y"))}')
        meta.update(
            distribution_categories=[level for level, _ in panels],
            distribution_separate_by=separate_by,
            facet_focus={
                'layout_key': 'distribution_facet_layout',
                'separate_by': separate_by,
                'keys': [level for level, _ in panels],
                'labels': [html.escape(str(level)) for level, _ in panels],
                'axes': [['x', 'y'], *([f'x{index + 4}', f'y{index + 4}']
                                       for index in range(len(panels)))],
                'slot_labels': list(range(len(panels))),
                'title': fig.layout.title.text,
                'stamp_annotation': len(panels),
                'applied': None},
            # Canonical geometry lets the chart wrapper refit the composition
            # without accumulating drift or touching the user's zoom ranges.
            distribution_facet_layout={
                'plot_height': composition['plot_height'],
                # Insertion order of axis_layout is the composition's own order;
                # iterating fig.layout would hand back an unordered property set.
                'axes': {name: list(fig.layout[name].domain) for name in axis_layout},
                'annotations': [{'x': item.x, 'y': item.y,
                                 'xref': item.xref, 'yref': item.yref}
                                for item in fig.layout.annotations]})
    fig.update_layout(meta=meta)
    return fig, table_md, result_df
