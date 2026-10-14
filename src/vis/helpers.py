import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.mixture import GaussianMixture

from src.emojis import sad_emoji
from src.natural_sort import natural_tuple_sort
from src.vis.plot_defaults import (
    panel_point_size,
    get_context_theme_color,
    SUPERPLOT_OBSERVATION_MIN_SIZE,
    SUPERPLOT_OBSERVATION_SIZE_SCALE,
    SUPERPLOT_REPLICATE_SIZE_SCALE,
    WEBGL_POINT_THRESHOLD,
)


def log_negative_error(var_name):
    return f"Cannot apply log to {var_name}: contains negative values. {sad_emoji}"


def log10_columns(df, columns):
    """Return a copy with log10(x + 1e-6) applied to ``columns``, and the columns skipped.

    A column holding a negative value is left untouched and reported, so the caller
    shows ``log_negative_error`` for it. Columns are handled in order.
    """
    df = df.copy()
    skipped = []
    for column in columns:
        if (df[column] < 0).any():
            skipped.append(column)
        else:
            df[column] = np.log10(df[column] + 1e-6)
    return df, skipped


def hover_field(label, value_ref):
    """Return a bold ``label: value`` hover line with the label escaped.

    Column names can contain markup. Escape them at interpolation while keeping
    the template's own HTML tags intact.
    """
    return f"<b>{html.escape(str(label))}:</b> {value_ref}<br>"


def point_trace_class(n_points):
    """Use Scattergl at WEBGL_POINT_THRESHOLD points, otherwise Scatter.

    Choose once per figure to reduce large SVG workloads while keeping point traces
    in one renderer; mixing SVG and WebGL changes their relative layering.
    """
    return go.Scattergl if n_points >= WEBGL_POINT_THRESHOLD else go.Scatter


def create_opacity_mapping(groups, min_opacity=0.3, max_opacity=1.0,
                           na_value="N/A", na_opacity=0.15):
    """Assign evenly spaced opacities in natural order, excluding missing values.

    The loader's missing-category label ``na_value`` receives ``na_opacity``
    below the ranked levels. Keep it a default argument: script export copies
    this function without module state.
    """
    groups = list(groups)
    real = [group for group in groups if group != na_value]
    real = natural_tuple_sort(real) if len(real) > 1 else real
    if len(real) == 1:
        mapping = {real[0]: max_opacity}
    else:
        # An all-missing column has no ranked levels.
        opacity_values = np.linspace(min_opacity, max_opacity, len(real))
        mapping = {group: opacity_values[i] for i, group in enumerate(real)}
    if na_value in groups:
        # Added last so N/A sits at the end of the legend's opacity block --
        # add_point_legend_traces ranks entries by dict order.
        mapping[na_value] = na_opacity
    return mapping

def create_shape_mapping(groups):
    """Create shape mapping for groups using different plotly symbols, preserving natural order"""
    groups = list(groups)
    groups = natural_tuple_sort(groups) if len(groups) > 1 else groups
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
               'triangle-down', 'pentagon', 'hexagon', 'octagon', 'star', 'diamond-tall']
    return {group: symbols[i % len(symbols)] for i, group in enumerate(groups)}

def _palette_rgb(colormap, count):
    """``count`` (r, g, b) float triples from a named colormap, falling back to tab10.

    Shared with the subcolor palette (``create_subcolor_map``), which needs the same
    colours by the same rules; both are drawn in one figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    count = max(int(count), 1)
    try:
        if colormap in ["viridis", "plasma", "inferno", "magma", "cividis"]:
            # Continuous colormaps have no discrete entries to take, so sample them.
            cmap = plt.colormaps[colormap]
            colors = [cmap(0.5)] if count == 1 else [cmap(i / (count - 1)) for i in range(count)]
            return [(color[0], color[1], color[2]) for color in colors]
        return [tuple(color[:3]) for color in sns.color_palette(colormap, n_colors=count)]
    # Fall back for unknown palettes. Keep imports outside the try so script
    # export captures them; the fallback also requires seaborn.
    except (ValueError, KeyError):
        return [tuple(color[:3]) for color in sns.color_palette("tab10", n_colors=count)]

def create_color_map(groups, overlap_point, colormap="tab10"):
    # if points in the visualization is going to overlap, use a transparent color
    if overlap_point:
        alpha = 0.6 if len(groups) > 1 else 1.0
    else:
        alpha = 1.0

    palette = _palette_rgb(colormap, len(groups))
    color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {alpha})" for color in palette]
    color_map = {t: color_sequence[i] for i, t in enumerate(groups)}
    return color_map

def _prepare_group_data(df, group_by_cols, new_group_col_name, overlap_point=True, colormap="tab10"):
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
        df[new_group_col_name] = df[group_by_cols].astype(str).agg('::'.join, axis=1)
    unique_groups = df[new_group_col_name].unique()
    unique_groups = natural_tuple_sort(unique_groups, delimiter='::')
    color_map = create_color_map(unique_groups, overlap_point=overlap_point, colormap=colormap)
    return unique_groups, color_map


@st.cache_data(show_spinner=False)
def _find_best_gmm(data, max_components=3, min_weight_threshold=0.1, random_state=42):
    """
    Finds the best Gaussian Mixture Model (GMM) based on BIC, subject to constraints.

    Args:
        data (np.ndarray): Input data (1D or 2D).
        max_components (int): Maximum number of components to try, capped at the sample count.
        min_weight_threshold (float): Minimum weight for a component to be considered valid.
        random_state (int): Random state for GMM initialization.

    Returns:
        sklearn.mixture.GaussianMixture or None: The best GMM found, or None if no valid model.
    """
    if data.ndim == 1:
        data_reshaped = data.reshape(-1, 1)
    elif data.ndim == 2:
        data_reshaped = data
    else:
        raise ValueError("Input data must be 1D or 2D.")

    n_samples = len(data_reshaped)
    if n_samples < 2:
        return None

    best_gmm = None
    lowest_bic = np.inf

    # Use one BLAS/OpenMP thread: synchronization dominates these 1D/2D fits.
    # Keep the import local so script export captures it with the function.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        for k in range(1, min(max_components, n_samples) + 1):
            gmm = GaussianMixture(n_components=k, random_state=random_state)
            gmm.fit(data_reshaped)
            bic = gmm.bic(data_reshaped)

            if gmm.weights_.min() >= min_weight_threshold:
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
    return best_gmm

def _sorted_levels(values):
    """The distinct values of a Series, in natural-sort order."""
    return natural_tuple_sort(values.unique())

def _channel_groups_and_map(df, column, mapper):
    """Return nonmissing channel levels and their visual mapping.

    Preserve values so map keys match the groupby keys in get_point_visual_mappings.
    Fill nulls upstream if they should participate in grouping.
    """
    if not column or column not in df.columns:
        return [], None
    groups = _sorted_levels(df[column].dropna())
    return groups, mapper(groups)

def create_opacity_groups_and_map(df, opacity_by_col):
    """Sorted unique groups of ``opacity_by_col`` and their opacity map."""
    return _channel_groups_and_map(df, opacity_by_col, create_opacity_mapping)

def create_shape_groups_and_map(df, shape_by_col):
    """Sorted unique groups of ``shape_by_col`` and their symbol map."""
    return _channel_groups_and_map(df, shape_by_col, create_shape_mapping)


def format_group_label(group, count=None, show_count=False, engine="plotly"):
    """Format a color-group legend label with an optional count on a second line.

    Plotly uses smaller HTML text for the count; Matplotlib uses a plain newline.
    This function is inlined into exported scripts to keep legend wording consistent.
    """
    label = str(group)
    if show_count and count is not None:
        if engine == "mpl":
            label = f"{label}\nn={count}"
        else:
            label = f"{label}<br><span style='font-size: 0.75em'>n={count}</span>"
    return label

def get_point_visual_mappings(
    df,
    color_by=None,
    shape_by=None,
    opacity_by=None,
    separate_by=None,
    group_col_name="group",
    overlap_point=True,
    colormap="tab10"
):
    """
    General helper for point-based visualizations to handle color_by, shape_by, and opacity_by.
    Returns grouped DataFrame, color_map, shape_map, opacity_map, and group keys.
    """
    # Prepare color grouping
    color_by = color_by or []
    if isinstance(color_by, str):
        color_by = [color_by]
    unique_color_groups, color_map = _prepare_group_data(df, color_by, group_col_name, overlap_point=overlap_point, colormap=colormap)
    # Prepare shape mapping
    _, shape_map = create_shape_groups_and_map(df, shape_by)
    # Prepare opacity mapping
    _, opacity_map = create_opacity_groups_and_map(df, opacity_by)
    if separate_by and separate_by.strip() != "" and separate_by in df.columns:
        separate_groups = natural_tuple_sort(df[separate_by].dropna().unique())
    else:
        separate_groups = None

    # Build ordered group keys from mapping dicts
    from itertools import product
    color_keys = list(color_map.keys())
    shape_keys = list(shape_map.keys()) if shape_map else [None]
    opacity_keys = list(opacity_map.keys()) if opacity_map else [None]
    separate_keys = list(separate_groups) if separate_groups is not None else [None]

    # Which of the four channels actually select rows, and which slot each occupies
    # in the group_key tuple below. Inactive channels keep a None placeholder in the
    # key (callers unpack it positionally) but take no part in the grouping.
    key_cols = [group_col_name]
    key_positions = [0]
    if shape_by and shape_by in df.columns:
        key_cols.append(shape_by)
        key_positions.append(1)
    if opacity_by and opacity_by in df.columns:
        key_cols.append(opacity_by)
        key_positions.append(2)
    if separate_by and separate_by in df.columns:
        key_cols.append(separate_by)
        key_positions.append(3)

    def ordered_group_iter():
        # Group once, then use the Cartesian product to preserve trace and legend order.
        # Channel key lists exclude nulls, matching groupby's default null handling.
        groups = {}
        for key, group_df in df.groupby(key_cols, sort=False, observed=True):
            if not isinstance(key, tuple):
                key = (key,)
            groups[key] = group_df

        for group_key in product(color_keys, shape_keys, opacity_keys, separate_keys):
            group_df = groups.get(tuple(group_key[pos] for pos in key_positions))
            if group_df is not None and len(group_df) > 0:
                yield group_key, group_df

    # Return an iterable like a groupby object, but ordered as specified
    grouped_ordered = ordered_group_iter()

    return grouped_ordered, color_map, shape_map, opacity_map, separate_groups

def apply_plot_styling(fig, point_size, axis_label_size, legend_size):
    """Apply consistent styling to plotly figures.

    Data-point markers are sized by ``point_size``, while legend markers are
    sized by ``legend_size`` (matching the legend font). Plotly normally ties
    the legend marker size to each trace's ``marker.size`` (``itemsizing='trace'``),
    so we decouple the two by hiding the legend on real data traces and
    adding "ghost" traces (no plotted points) whose ``marker.size`` is set
    to ``legend_size``. Existing ghost legend traces (e.g. shape/opacity
    entries) are also resized to ``legend_size``.
    """
    meta = fig.layout.meta
    theme_color = get_context_theme_color()
    # Either facet grid: one overview beside smaller panels on extra axes.
    facet_grid = isinstance(meta, dict) and (
        'dimension_reduction_layout' in meta or 'distribution_facet_layout' in meta)
    if isinstance(meta, dict) and meta.get('histogram'):
        panels = meta.get('histogram_summaries', [])
        legend_heights = []
        for row in range(len(panels)):
            legend_id = 'legend' + (str(row + 1) if row else '')
            entries = [trace for trace in fig.data
                       if trace.legend == legend_id and trace.showlegend]
            lines = sum((trace.name or '').count('<br>') + 1 for trace in entries)
            groups = len({trace.legendgroup for trace in entries})
            legend_heights.append(lines * (1.4 * legend_size + 2) + groups * 6 + 12)
        row_height = max(220, max(legend_heights, default=0))
        bottom_margin = 2 * axis_label_size + 40
        if meta.get('histogram_separator') and meta.get('histogram_summaries'):
            # Keep category headings and local legends within their own rows,
            # including large fonts and fits with several component entries.
            rows = len(meta['histogram_summaries'])
            gap = max(80, 2 * axis_label_size + 32)
            plot_height = rows * row_height + (rows - 1) * gap
            top_margin = max(80, 2 * axis_label_size + 24)
            fig.update_layout(height=plot_height + top_margin + bottom_margin,
                              margin=dict(t=top_margin))
            for row in range(rows):
                top = 1 - row * (row_height + gap) / plot_height
                bottom = top - row_height / plot_height
                axis = 'yaxis' + (str(row + 1) if row else '')
                fig.layout[axis].domain = [max(0, bottom), top]
                fig.layout.annotations[row].y = top
        else:
            fig.update_layout(height=max(fig.layout.height or 450,
                                         row_height + (fig.layout.margin.t or 80) + bottom_margin))
        fig.update_layout(margin=dict(b=bottom_margin), title_font_color=theme_color)
        fig.update_xaxes(title_font_color=theme_color, tickfont_color=theme_color)
        fig.update_yaxes(title_font_color=theme_color, tickfont_color=theme_color)
        for row in range(len(panels)):
            suffix = str(row + 1) if row else ''
            fig.update_layout({'legend' + suffix: dict(
                x=1.02 if meta.get('histogram_gmm') else 1,
                xanchor='left' if meta.get('histogram_gmm') else 'right',
                y=fig.layout['yaxis' + suffix].domain[1],
                font=dict(size=legend_size, color=theme_color),
                bgcolor=('rgba(255,255,255,0.85)' if theme_color == 'black'
                         else 'rgba(30,30,30,0.85)'))})
    if facet_grid:
        # Keep method-specific numeric tick widths from changing the frame.
        # The shared font setting determines room for the vertical axis/title.
        fig.update_layout(margin=dict(l=max(80, 4 * axis_label_size + 8)))

    # Update marker sizes for all scatter and box traces
    for trace in fig.data:
        # Skip traces that should maintain their original sizes
        if isinstance(trace.meta, dict) and trace.meta.get('is_lifetime_reference'):
            continue
        if hasattr(trace, 'marker') and trace.marker:
            # Style both SVG and WebGL point traces.
            if trace.type in ('scatter', 'scattergl') or trace.type == 'box' and trace.marker:
                is_facet = facet_grid and getattr(trace, 'xaxis', None) not in (None, 'x')
                is_context = isinstance(trace.meta, dict) and (
                    trace.meta.get('phasor_role') == 'context'
                    or trace.meta.get('facet_role') == 'context')
                superplot_role = trace.meta.get('superplot_role') if isinstance(trace.meta, dict) else None
                trace.marker.size = (max(SUPERPLOT_OBSERVATION_MIN_SIZE,
                                        point_size * SUPERPLOT_OBSERVATION_SIZE_SCALE)
                                     if superplot_role == 'observation' else
                                     point_size * SUPERPLOT_REPLICATE_SIZE_SCALE
                                     if superplot_role == 'replicate' else
                                     panel_point_size(point_size) if is_facet or is_context else point_size)

    # Facet-grid row labels share the legend's font control.
    annotation_size = legend_size if facet_grid else axis_label_size
    if isinstance(meta, dict) and meta.get('histogram_separator'):
        # Category headings remain compact as the axes grow.
        annotation_size = min(axis_label_size, 24)
    if fig.layout.annotations:
        for annotation in fig.layout.annotations:
            if annotation.font:
                annotation.font.size = annotation_size
            else:
                annotation.font = dict(size=annotation_size)
            if isinstance(meta, dict) and meta.get('histogram'):
                annotation.font.color = theme_color

    # Update layout with axis and legend font sizes
    fig.update_layout(
        xaxis=dict(
            title=dict(font=dict(size=axis_label_size)),
            tickfont=dict(size=axis_label_size-2)
        ),
        yaxis=dict(
            title=dict(font=dict(size=axis_label_size)),
            tickfont=dict(size=axis_label_size-2)
        ),
        legend=dict(
            font=dict(size=legend_size)
        ),
        hoverlabel=dict(
            bgcolor="white" if theme_color == "black" else "rgb(30, 30, 30)",
            font=dict(color=theme_color),
            bordercolor=theme_color,
        )
    )

    fig.update_xaxes(title_font_size=axis_label_size, tickfont_size=axis_label_size-2)
    fig.update_yaxes(title_font_size=axis_label_size, tickfont_size=axis_label_size-2)

    # Replace each color-trace legend entry with a ghost trace whose marker
    # size equals legend_size, and resize any pre-existing ghost legend
    # traces (shape/opacity entries) to match.
    ghost_traces = []
    seen_legendgroups = set()
    for trace in fig.data:
        # Create independent legend markers for both SVG and WebGL data traces.
        # Ghosts use SVG because they draw only a legend swatch.
        if trace.type not in ('scatter', 'scattergl'):
            continue
        if isinstance(trace.meta, dict) and trace.meta.get('is_lifetime_reference'):
            continue
        if not getattr(trace, 'showlegend', True):
            continue

        # Line legends already carry their color and dash pattern. Marker ghosts
        # would lose both and merge component entries sharing a color group.
        if trace.mode == 'lines' or isinstance(meta, dict) and meta.get('histogram'):
            continue

        x_attr = getattr(trace, 'x', None)
        x_data = list(x_attr) if x_attr is not None else []
        is_ghost = len(x_data) > 0 and all(v is None for v in x_data)
        if is_ghost:
            if trace.marker:
                trace.marker.size = legend_size
            continue

        legendgroup = trace.legendgroup or trace.name
        if legendgroup in seen_legendgroups:
            trace.showlegend = False
            continue
        seen_legendgroups.add(legendgroup)

        marker_color = trace.marker.color if trace.marker else None
        marker_symbol = trace.marker.symbol if trace.marker else 'circle'
        if isinstance(marker_symbol, (list, tuple, np.ndarray)):
            marker_symbol = marker_symbol[0] if len(marker_symbol) > 0 else 'circle'
        marker_opacity = trace.marker.opacity if trace.marker else 1.0
        if isinstance(marker_opacity, (list, tuple, np.ndarray)):
            marker_opacity = marker_opacity[0] if len(marker_opacity) > 0 else 1.0

        ghost_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name=trace.name,
            legendgroup=trace.legendgroup,
            marker=dict(
                color=marker_color,
                symbol=marker_symbol,
                opacity=marker_opacity,
                size=legend_size,
            ),
            showlegend=True,
            legendrank=getattr(trace, 'legendrank', None),
            hoverinfo='skip',
        ))
        trace.showlegend = False

    for ghost in ghost_traces:
        fig.add_trace(ghost)

    return fig


def add_interleaved_points_trace(
    fig,
    grouped,
    color_map,
    shape_map,
    opacity_map,
    axis_labels,
    text_col,
    customdata_col,
    shape_by=None,
    opacity_by=None,
    hovertemplate="<b>%{text}</b>",
    random_seed=42,
    num_batches=15,
    show_counts=False
):
    """
    Adds multiple interleaved traces per color group to minimize occlusion
    while maintaining legend interactivity.
    
    Strategy: Split each color group into N batches, then add traces in an 
    interleaved pattern (batch1 of each color, batch2 of each color, etc.).
    All batches of the same color share a legendgroup so clicking the legend
    entry shows/hides all batches together.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to add traces to
    grouped : iterable
        Iterator of (group_key, group_df) tuples from get_point_visual_mappings
    color_map : dict
        Mapping of color_group to color values
    shape_map : dict or None
        Mapping of shape_group to symbol values
    opacity_map : dict or None
        Mapping of opacity_group to opacity values
    axis_labels : list
        List of [x_label, y_label] for the axes
    text_col : str
        Column name for text/hover labels
    customdata_col : str or None
        Column name for customdata, or None when the frame has no FOV column —
        every trace is then built with customdata=None.
    shape_by : str or None
        Column name used for shape grouping
    opacity_by : str or None
        Column name used for opacity grouping
    hovertemplate : str
        Hover template string
    random_seed : int or None
        Seed for the per-color shuffle. Defaults to a fixed value so the
        interleave order is reproducible across reruns; pass None for a
        nondeterministic order.
    num_batches : int
        Number of batches to split each color group into (default: 15)
    
    Returns:
    --------
    scatter_cls : type
        The trace class used for the points (``go.Scatter`` or ``go.Scattergl``). Overlays
        drawn on the same axes after this call must use it too -- see the note at the
        return statement.
    """
    import math
    import random

    # Use a local seeded RNG for reproducible order without changing global state.
    rng = random.Random(random_seed)

    # Collect array chunks from all styling subgroups of each colour, then
    # concatenate once in group iteration order.
    chunks_by_color = {}
    for group_key, group_df in grouped:
        chunks_by_color.setdefault(group_key[0], []).append(
            (group_df, group_key[1], group_key[2])
        )

    points_by_color = {}
    for color_group, chunks in chunks_by_color.items():
        lengths = [len(chunk_df) for chunk_df, _, _ in chunks]
        points_by_color[color_group] = {
            'x': np.concatenate([c[axis_labels[0]].to_numpy() for c, _, _ in chunks]),
            'y': np.concatenate([c[axis_labels[1]].to_numpy() for c, _, _ in chunks]),
            'text': np.concatenate([c[text_col].to_numpy() for c, _, _ in chunks]),
            # Repeat the subgroup's constant shape/opacity keys; object dtype preserves None.
            'shape_group': np.repeat(
                np.array([s for _, s, _ in chunks], dtype=object), lengths),
            'opacity_group': np.repeat(
                np.array([o for _, _, o in chunks], dtype=object), lengths),
        }
        # Omitted rather than set to None when there is no FOV column: the shuffle
        # below reindexes every value in this dict with v[order].
        if customdata_col is not None:
            points_by_color[color_group]['customdata'] = np.concatenate(
                [c[customdata_col].to_numpy() for c, _, _ in chunks])

    # Shuffle indices once per colour and apply the same permutation to every column.
    for color_group, columns in points_by_color.items():
        order = list(range(len(columns['x'])))
        rng.shuffle(order)
        order = np.asarray(order, dtype=int)
        points_by_color[color_group] = {k: v[order] for k, v in columns.items()}

    # Split each color group into batches. Batches are (start, end) bounds into the
    # colour's arrays; slicing at trace-build time avoids copying the points twice.
    batches_by_color = {}
    for color_group, columns in points_by_color.items():
        num_points = len(columns['x'])
        # Adjust num_batches if there are fewer points
        actual_batches = min(num_batches, max(1, num_points // 5))  # At least 5 points per batch
        batch_size = math.ceil(num_points / actual_batches)

        batches = []
        for i in range(actual_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            if start_idx < num_points:
                batches.append((start_idx, end_idx))

        batches_by_color[color_group] = batches

    # Get color groups in their natural order
    color_group_order = {color_group: i for i, color_group in enumerate(color_map.keys())}
    sorted_color_groups = sorted(points_by_color.keys(), key=lambda x: color_group_order.get(x, float('inf')))

    # Add <extra></extra> to hovertemplate to hide trace name from hover
    if "<extra>" not in hovertemplate:
        hover_without_trace = hovertemplate + "<extra></extra>"
    else:
        hover_without_trace = hovertemplate

    # Find maximum number of batches across all colors
    max_batches = max(len(batches) for batches in batches_by_color.values())

    # Choose one renderer for the full figure; mixed SVG/WebGL layering
    # would override the batches' intended interleaving order.
    scatter_cls = point_trace_class(sum(len(c['x']) for c in points_by_color.values()))

    # Interleave batches: add batch i from each color before moving to batch i+1
    for batch_idx in range(max_batches):
        for color_group in sorted_color_groups:
            batches = batches_by_color[color_group]

            # Skip if this color doesn't have this many batches
            if batch_idx >= len(batches):
                continue

            start_idx, end_idx = batches[batch_idx]
            columns = points_by_color[color_group]

            # Keep numeric arrays for Plotly's binary serialization and avoid list copies.
            # String arrays use the same slicing path.
            x_vals = columns['x'][start_idx:end_idx]
            y_vals = columns['y'][start_idx:end_idx]
            text_vals = columns['text'][start_idx:end_idx]
            customdata_vals = (columns['customdata'][start_idx:end_idx]
                               if 'customdata' in columns else None)

            # Keep per-point symbol and opacity arrays: batches can mix styling subgroups,
            # and apply_plot_styling uses these sequences to build legend entries.
            batch_shape_groups = columns['shape_group'][start_idx:end_idx]
            batch_opacity_groups = columns['opacity_group'][start_idx:end_idx]
            marker_symbols = [shape_map[g] if g is not None and shape_map else 'circle' for g in batch_shape_groups]
            marker_opacities = [opacity_map[g] if g is not None and opacity_map else 0.8 for g in batch_opacity_groups]

            # Only show legend for the first batch of each color
            show_in_legend = (batch_idx == 0)

            fig.add_trace(
                scatter_cls(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    text=text_vals,
                    customdata=customdata_vals,
                    hovertemplate=hover_without_trace,
                    name=format_group_label(color_group, len(points_by_color[color_group]['x']), show_counts),
                    legendgroup=str(color_group),  # All batches of same color share legendgroup
                    marker=dict(
                        color=color_map[color_group],
                        symbol=marker_symbols,
                        opacity=marker_opacities
                    ),
                    showlegend=show_in_legend,
                    legendrank=color_group_order[color_group]
                )
            )

    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)

    fig.update_layout(
        hovermode='closest'
    )

    # Return the renderer for overlays on these axes. Later overlays must use
    # the same trace class to appear above the points in WebGL as well as SVG.
    return scatter_cls

def add_point_legend_traces(fig, shape_map, opacity_map, shape_by=None, opacity_by=None):
    """
    Adds legend traces for shape and opacity visual channels to the given figure.
    """
    # Add opacity legend traces
    if opacity_map:
        for i, (opacity_group, opacity_value) in enumerate(opacity_map.items()):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='gray',
                        opacity=opacity_value,
                        symbol='circle'
                    ),
                    name=f'{opacity_group}',
                    legendgroup='opacity_legend',
                    showlegend=True,
                    legendrank=100 + i,
                    hoverinfo='skip'
                )
            )
    # Add shape legend traces
    if shape_map:
        for i, (shape_group, shape_symbol) in enumerate(shape_map.items()):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='gray',
                        opacity=0.8,
                        symbol=shape_symbol
                    ),
                    name=f'{shape_group}',
                    legendgroup='shape_legend',
                    showlegend=True,
                    legendrank=200 + i,
                    hoverinfo='skip'
                )
            )
    return fig
