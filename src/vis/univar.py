import html
from itertools import combinations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.column_roles import code_span
from .histogram import histogram_legend_label, prepare_histogram
from .histogram import _assign_subpopulation_labels as _assign_subpopulation_labels

from src.feature_labels import format_feature_label
from src.widgets.analysis_widget_state import number_input_default
from src.widgets.gmm_tables import gmm_component_table, gmm_tables_html
from src.widgets.visualization_widgets import (
    comparison_pair_widget,
    gmm_hyperParams_widget,
    histogram_bin_width_widget,
)

from .helpers import (
    _add_effect_size_annotations,
    _density_at_points,
    add_point_legend_traces,
    create_subcolor_map,
    create_color_map,
    format_group_label,
    get_context_theme_color,
    get_point_visual_mappings,
    hover_field,
    interleave_point_batches,
    log_negative_error,
    point_trace_class,
)


def feature_histogram_plot(df, selected_var, color_by=None, colormap="tab10", log_x=False,
                           separate_by=None):
    """Count curves in full-width category rows, with one set of bin controls."""
    from src.widgets.analysis_widget_state import preserve_analysis_controls
    from src.widgets.visualization_widgets import histogram_bin_width_key

    width_keys = [histogram_bin_width_key(selected_var, scale) for scale in (False, True)]
    width_key = histogram_bin_width_key(selected_var, log_x)
    legacy_key = f"hist_bin_width_{selected_var}"
    if not any(key in st.session_state for key in width_keys) and legacy_key in st.session_state:
        # The previous shared value belongs to the scale currently on screen.
        st.session_state[width_key] = st.session_state[legacy_key]
    preserve_analysis_controls(st.session_state, width_keys)
    edges = histogram_bin_width_widget(df[selected_var].dropna(), key=width_key)
    prepared = prepare_histogram(df, selected_var, color_by, separate_by, bin_edges=edges)
    return _histogram_figure(prepared, colormap, log_x)


def feature_gmm_plot(df, selected_var, color_by=None, colormap="tab10", log_x=False,
                     separate_by=None):
    """Category-local GMM curves and a positionally labeled analyzed dataframe."""
    max_components, min_weight = gmm_hyperParams_widget()
    intersection = st.checkbox(
        "Use intersection as threshold", value=False, key="intersection_threshold",
        help="Use intersections between adjacent components as thresholds within each "
             "category and color group. If unavailable, use the highest posterior probability.")
    prepared = prepare_histogram(
        df, selected_var, color_by, separate_by, apply_gmm=True,
        max_components=max_components, min_weight_threshold=min_weight,
        intersection_threshold=intersection)
    return _histogram_figure(prepared, colormap, log_x), prepared["df"]


def _histogram_figure(prepared, colormap, log_x):
    """Render panel legends and serialize optional GMM details for display below."""
    theme = get_context_theme_color()
    panels = prepared["panels"]
    separate_by = prepared["separate_by"]
    rows = max(1, len(panels))
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=min(0.16, 80 / (300 * rows)) if rows > 1 else 0,
        subplot_titles=[html.escape(str(p['category']))
                        for p in panels] if separate_by else None)
    fig.update_annotations(font_color=theme)
    colors = create_color_map(prepared["color_groups"], overlap_point=False, colormap=colormap)
    show_counts = st.session_state.get("plot_show_group_counts", False)
    gmm_mode = prepared["apply_gmm"]
    dash_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot"]
    summaries = []
    for row, panel in enumerate(panels, 1):
        summary = dict(category=panel["category"], groups=[])
        legend_id = "legend" + (str(row) if row > 1 else "")
        yaxis = "yaxis" + (str(row) if row > 1 else "")
        fig.update_layout({legend_id: dict(
            orientation="v", x=1.02 if gmm_mode else 1, y=fig.layout[yaxis].domain[1],
            xanchor="left" if gmm_mode else "right", yanchor="top", xref="paper", yref="paper",
            font=dict(color=theme),
            bgcolor="rgba(255,255,255,0.85)" if theme == "black" else "rgba(30,30,30,0.85)",
            groupclick="togglegroup", tracegroupgap=4)})
        for group in panel["groups"]:
            color = group["color_group"]
            rank = prepared["color_groups"].index(color) * 10
            label = f"{color} GMM" if gmm_mode else color
            name = html.escape(histogram_legend_label(
                label, group["count"], group["skewness"], show_counts,
                show_skewness=not gmm_mode)).replace("\n", "<br>")
            legend_group = f"{row}:{color}"
            hover = hover_field("Group", html.escape(str(group["label"]))) + f"n={group['count']}<br>"
            if not gmm_mode or len(group["pdf"]):
                fig.add_trace(go.Scatter(
                    x=group["x"] if gmm_mode else prepared["bin_centers"],
                    y=group["pdf"] if gmm_mode else group["counts"],
                    mode="lines+markers" if not gmm_mode and len(prepared["bin_centers"]) == 1 else "lines",
                    marker=dict(color=colors[color], size=6), name=name,
                    line=dict(color=colors[color], width=2), legendgroup=legend_group,
                    legend=legend_id, showlegend=True, legendrank=rank,
                    hovertemplate=hover + hover_field("Density" if gmm_mode else "Count", "%{y}") + "<extra></extra>"),
                    row=row, col=1)
            else:
                # Sparse fits still contribute their optional local count.
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines", name=name,
                    legendgroup=legend_group, legend=legend_id, legendrank=rank,
                    line=dict(color=colors[color], width=2), showlegend=True,
                    hoverinfo="skip"), row=row, col=1)
            component_rows = []
            if gmm_mode:
                for component in group["components"]:
                    component_rank = component["rank"]
                    component_rows.append((component_rank,
                                           f"{component['mean']:.2f} ± {component['std']:.2f}",
                                           f"{component['weight']:.2f}"))
                    if len(group["components"]) <= 1:
                        continue
                    fig.add_trace(go.Scatter(
                        x=group["x"], y=component["density"], mode="lines",
                        name=f"{html.escape(str(color))} Component {component_rank}",
                        legendgroup=legend_group, legend=legend_id,
                        legendrank=rank + component_rank, showlegend=True,
                        line=dict(color=colors[color], width=1,
                                  dash=dash_styles[(component_rank - 1) % len(dash_styles)]),
                        hovertemplate=hover + f"Component {component_rank}<br>Density: %{{y}}<extra></extra>"),
                        row=row, col=1)
                for threshold in group["thresholds"] if group["thresholds"] is not None else []:
                    fig.add_shape(type="line", x0=threshold, x1=threshold,
                                  y0=0, y1=max(group["pdf"]),
                                  line=dict(color=colors[color], width=2, dash="dash"), opacity=0.5,
                                  row=row, col=1)
                    if not separate_by:
                        fig.add_annotation(x=threshold, y=max(group["pdf"]) * 1.05,
                                           text=f"Threshold ({threshold:.2f})", showarrow=False,
                                           font=dict(color=theme), row=row, col=1)
            summary["groups"].append(dict(
                label=group["label"], count=group["count"], color=colors[color],
                skewness=float(group["skewness"]) if np.isfinite(group["skewness"]) else None,
                components=component_rows, h_index=group.get("h_index"),
                thresholds=list(group["thresholds"]) if group.get("thresholds") is not None else [],
                notices=group.get("notices", [])))
        summaries.append(summary)
    pretty_var = format_feature_label(prepared["selected_var"])
    x_label = f"log₁₀({pretty_var})" if log_x else pretty_var
    title = "Gaussian Mixture Model fit" if gmm_mode else "Frequency histogram"
    suffix = f" by {html.escape(', '.join(prepared['color_by']))}" if prepared["color_by"] else ""
    fig.update_xaxes(title=dict(text="", font=dict(color=theme)),
                     tickfont=dict(color=theme), showgrid=False, zeroline=False,
                     range=prepared["x_range"], visible=False)
    fig.update_xaxes(title_text=x_label, visible=True, showticklabels=True, row=rows, col=1)
    y_label = ("Density" if separate_by else "Probability Density") if gmm_mode else "Count"
    fig.update_yaxes(title=dict(text=y_label, font=dict(color=theme)),
                     tickfont=dict(color=theme), showgrid=True, zeroline=False,
                     range=prepared["y_range"])
    fig.update_layout(
        title=dict(text=f"{title} of {pretty_var}{suffix}", font=dict(color=theme)),
        height=300 * rows + 130 if separate_by else 450,
        hovermode="x unified", margin=dict(l=60, r=20, t=80, b=100, autoexpand=True),
        meta=dict(histogram=True, histogram_summaries=summaries,
                  histogram_gmm=gmm_mode, histogram_separator=separate_by,
                  histogram_feature=prepared["selected_var"]))
    return fig


def render_histogram_summaries(fig):
    """Render optional GMM details separately from the compact panel legends."""
    meta = fig.layout.meta
    if not meta["histogram_gmm"]:
        return
    for panel in meta["histogram_summaries"]:
        label = (f"{meta['histogram_separator']}={panel['category']}"
                 if meta["histogram_separator"] else "All observations")
        container = st.expander(f"GMM details · {code_span(label)}")
        with container:
            tables = []
            for group in panel["groups"]:
                for notice in group["notices"]:
                    st.info(f"{code_span(group['label'])}: {notice}")
                if group["components"]:
                    tables.append(gmm_component_table(group["label"], group["components"], [meta["histogram_feature"]]))
                if group["h_index"] is not None:
                    st.markdown(f"H-index for {code_span(group['label'])}: **{group['h_index']:.3f}**")
                for i, threshold in enumerate(group["thresholds"], 1):
                    st.markdown(f"Threshold for {code_span(group['label'])} between component {i} and {i+1}: **{threshold:.2f}**")
            if tables:
                st.markdown(gmm_tables_html(tables), unsafe_allow_html=True)


# Match Plotly's box width for unit-spaced groups: (1 - boxgap) * (1 - boxgroupgap).
# Both gaps default to 0.3; whiskers span half the box width.
_BOX_WIDTH = 0.49
_WHISKER_CAP_WIDTH = _BOX_WIDTH * 0.5


def _add_box_outline_above_gl(fig, x, q1, median, q3, lower_fence, upper_fence, mean_val, color):
    """Redraw box outlines as above-layer shapes over WebGL points.

    SVG box zorder cannot lift a box over the WebGL canvas. Keep the original box
    trace for hover statistics and its legend entry.
    """
    half, cap = _BOX_WIDTH / 2, _WHISKER_CAP_WIDTH / 2
    outline = dict(color=color, width=3)
    common = dict(xref="x", yref="y", layer="above")
    fig.add_shape(type="rect", x0=x - half, x1=x + half, y0=q1, y1=q3,
                  line=outline, fillcolor="rgba(0,0,0,0)", **common)
    fig.add_shape(type="line", x0=x - half, x1=x + half, y0=median, y1=median,
                  line=outline, **common)
    # Dashed mean line, matching boxmean=True on the trace.
    fig.add_shape(type="line", x0=x - half, x1=x + half, y0=mean_val, y1=mean_val,
                  line=dict(color=color, width=3, dash="dash"), **common)
    for base, fence in ((q1, lower_fence), (q3, upper_fence)):
        fig.add_shape(type="line", x0=x, x1=x, y0=base, y1=fence, line=outline, **common)
        fig.add_shape(type="line", x0=x - cap, x1=x + cap, y0=fence, y1=fence,
                      line=outline, **common)


def feature_comparison_plot(df, unique_row_id_col, fov_name_col, selected_var, color_by, opacity_by=None, shape_by=None, separate_by=None, colormap="tab10", effect_size_method="None", mean_or_median=None, statistical_test="None", custom_order=None, subcolor_by=None, row_id_label="ID"):

    # Get theme color once at the start for all theme-aware elements
    theme_color = get_context_theme_color()
    # Pretty FLIM label (Greek notation) reused for hover, title, and y-axis
    pretty_var = format_feature_label(selected_var)

    col1, col2, col3 = st.columns([0.15, 0.2, 0.65])
    with col1:
        log_y = st.checkbox("Log Y", value=False, key=f"log_y_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")
    with col2:
        add_boxplot = st.checkbox("Add boxplot", value=False, key=f"add_boxplot_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")
    with col3:
        connect_means = st.checkbox("Connect means", value=False, key=f"connect_means_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")

    # Create a working copy to avoid modifying the original dataframe
    df = df.copy()

    # Apply log transform if requested (consistent with bivar.py)
    if log_y:
        if (df[selected_var] < 0).any():
            st.error(log_negative_error(selected_var))
        else:
            df[selected_var] = np.log10(df[selected_var] + 1e-6)

    fig = go.Figure()
    COLOR_GROUP_COL_NAME = 'compare_group'
    grouped_sep, color_map, shape_map, opacity_map, separate_groups = get_point_visual_mappings(
        df,
        color_by=color_by,
        shape_by=shape_by,
        opacity_by=opacity_by,
        separate_by=separate_by,
        group_col_name=COLOR_GROUP_COL_NAME,
        overlap_point=False,
        colormap=colormap
    )
    grouped_list = list(grouped_sep)
    group_keys = [group_key for group_key, _ in grouped_list]
    compare_groups = list(color_map.keys())
    show_counts = st.session_state.get("plot_show_group_counts", False)
    plotted = df.dropna(subset=[selected_var])
    group_counts = plotted.groupby(COLOR_GROUP_COL_NAME).size().to_dict()

    # Subcolor gives each nested value a figure-wide color. X positions, boxes and
    # statistics continue to describe the comparison groups.
    subcolor_of = create_subcolor_map(
        plotted, subcolor_by, COLOR_GROUP_COL_NAME, list(color_map.keys()), colormap=colormap,
    )
    # Count each subcolor value across the frame to match its shared legend entry.
    subcolor_counts = {}
    if subcolor_of:
        subcolor_counts = (plotted[subcolor_by].fillna("N/A").astype(str)
                          .value_counts().to_dict())

    # Apply custom order early so compare_pairs uses the reordered groups
    # This is critical for Glass's Delta where the first group in the pair is the control
    ordered_compare_groups = list(compare_groups)
    if custom_order and 'compare_groups' in custom_order:
        custom_cmp = [g for g in custom_order['compare_groups'] if g in ordered_compare_groups]
        remaining_cmp = [g for g in ordered_compare_groups if g not in custom_cmp]
        ordered_compare_groups = custom_cmp + remaining_cmp

    # Generate pairs from ordered groups so Glass's Delta uses correct control group
    compare_pairs = list(combinations(ordered_compare_groups, 2))
    point_size = 5

    # Track legend entries to avoid duplicates
    legend_entries = set()

    # Calculate x-positions for separate sections - only for existing combinations

    if separate_groups:
        # First, find which combinations actually exist in the data
        existing_combinations = []

        # Apply custom order to separate_groups if provided
        ordered_separate_groups = list(separate_groups)
        if custom_order and 'separate_groups' in custom_order:
            # Filter to only include groups that are actually in the data, preserving data integrity
            custom_sep = [g for g in custom_order['separate_groups'] if g in ordered_separate_groups]
            # Add any missing groups at the end
            remaining = [g for g in ordered_separate_groups if g not in custom_sep]
            ordered_separate_groups = custom_sep + remaining

        for separate_group in ordered_separate_groups:
            section_combinations = []

            # Use ordered_compare_groups (already set with custom order at function start)
            for color_group in ordered_compare_groups:
                combo_exists = any(
                    (separate_group in group_key if isinstance(group_key, tuple) else group_key == separate_group) and
                    (color_group in group_key if isinstance(group_key, tuple) else group_key == color_group)
                    for group_key in group_keys
                )
                if combo_exists:
                    section_combinations.append((separate_group, color_group))
            existing_combinations.append(section_combinations)

        # Now calculate positions only for existing combinations
        x_positions = {}
        x_tick_positions_actual = []
        x_tick_labels_actual = []
        section_spacing = 0.5  # Reduced gap size between sections
        current_x = 0

        for section_idx, section_combinations in enumerate(existing_combinations):
            if section_idx > 0:
                current_x += section_spacing  # Add spacing between sections

            for separate_group, color_group in section_combinations:
                x_positions[(separate_group, color_group)] = current_x
                x_tick_positions_actual.append(current_x)
                x_tick_labels_actual.append("" if color_group == "all_data" else color_group)
                current_x += 1

        separate_sections_info = []
        current_x = 0
        for section_idx, section_combinations in enumerate(existing_combinations):
            if section_idx > 0:
                current_x += section_spacing
            section_start = current_x
            section_end = current_x + len(section_combinations) - 1
            section_center = (section_start + section_end) / 2 if section_combinations else current_x
            # existing_combinations is built in ordered_separate_groups order,
            # so section_idx indexes both.
            group_name = ordered_separate_groups[section_idx]

            separate_sections_info.append({
                'group': group_name,
                'center': section_center,
                'combinations': section_combinations
            })
            current_x += len(section_combinations)
    else:
        # Standard x-positions when no separate_by
        # ordered_compare_groups is already set with custom order at function start
        x_positions = {color_group: idx for idx, color_group in enumerate(ordered_compare_groups)}

    # Sort grouped_list based on the custom order to ensure legend matches x-axis
    # Sort keys: (separate_group index, color_group index)
    def group_sort_key(item):
        g_key, _ = item
        color_g = g_key[0]
        separate_g = g_key[3]

        c_idx = float('inf')
        if color_g in ordered_compare_groups:
            c_idx = ordered_compare_groups.index(color_g)

        s_idx = float('inf')
        if separate_groups and separate_g in ordered_separate_groups:
            s_idx = ordered_separate_groups.index(separate_g)
        elif not separate_groups:
            s_idx = 0

        return (s_idx, c_idx)

    grouped_list.sort(key=group_sort_key)

    # Pool row positions for each section/colour cell, excluding missing styling keys.
    # Sort positions to retain the input row order.
    all_y = df[selected_var].to_numpy(dtype=float)
    pooled_rows = {}
    for group_key, group_df in grouped_list:
        if group_key[0] not in compare_groups:
            continue
        drawn = group_df.dropna(subset=[selected_var])
        if drawn.empty:
            continue
        cell = (group_key[3], group_key[0])
        pooled_rows.setdefault(cell, []).append(df.index.get_indexer(drawn.index))
    pooled_rows = {cell: np.sort(np.concatenate(chunks))
                   for cell, chunks in pooled_rows.items()}

    def cell_x_position(separate_group, color_group):
        if separate_groups:
            return x_positions.get((separate_group, color_group))
        return x_positions.get(color_group)

    # Sina jitter: horizontal spread proportional to local density. The KDE and the rng
    # are scoped to the colour group, which owns the x position, so shape_by/opacity_by
    # change point styling only and never move a point.
    x_of_row = np.full(len(df), np.nan)
    for (separate_group, color_group), rows in pooled_rows.items():
        x_position = cell_x_position(separate_group, color_group)
        if x_position is None:
            continue
        y_data = all_y[rows]
        densities = _density_at_points(y_data)
        # Normalize densities to a reasonable jitter width
        max_jitter = 0.35  # Controls the max horizontal spread
        if len(densities) > 0 and np.max(densities) > 0:
            norm_densities = densities / np.max(densities)
        else:
            # Degenerate density (a constant column has no KDE): spread points with
            # uniform jitter so they stay visible instead of stacking into one dot.
            norm_densities = np.ones_like(densities)
        # Randomly assign sign to spread points left/right
        rng = np.random.default_rng(seed=42)
        x_of_row[rows] = x_position + (rng.uniform(-1, 1, size=len(y_data))
                                       * norm_densities * max_jitter)

    # Collect styling subgroups before drawing each colour group's trace.
    point_buckets = {}

    for group_key, group_df in grouped_list:
        # Always unpack group_key by position
        color_group = group_key[0]
        shape_group = group_key[1]
        opacity_group = group_key[2]
        separate_group = group_key[3]

        # Skip if not in our color groups (shouldn't happen but safety check)
        if color_group not in compare_groups:
            continue

        # Drop rows where the variable to plot is NaN
        group_df = group_df.dropna(subset=[selected_var])
        if group_df.empty:
            continue

        # --- Prepare Hover Information ---
        # Every label is the column's own name -- for the identifier, the configured
        # column or "ID" for the row numbers invented for a table that has none.
        hovertemplate_parts = [hover_field(pretty_var, "%{y:.3f}")]
        hovertemplate_parts.append(hover_field(row_id_label, "%{text}"))
        if fov_name_col is not None:
            point_customdata = group_df[fov_name_col]
            # Add the corresponding part to the hovertemplate, referencing customdata
            hovertemplate_parts.append(hover_field(fov_name_col, "%{customdata}"))
        hovertemplate_parts.append("<extra></extra>") # Hide the default trace info box
        final_hovertemplate = "".join(hovertemplate_parts)

        # --- Determine x-position and visual properties ---
        # Same skip as the jitter pass: a subgroup with no x position is not accumulated.
        if cell_x_position(separate_group, color_group) is None:
            continue

        marker_color = color_map[color_group]
        marker_opacity = opacity_map.get(opacity_group, 0.7) if opacity_map and opacity_group is not None else 0.7
        marker_symbol = shape_map.get(shape_group, 'circle') if shape_map and shape_group is not None else 'circle'

        # Each row's x was assigned per colour group above; look it up rather than
        # recomputing, so a subgroup cannot get a spread of its own.
        y_data = group_df[selected_var].values
        x_jittered = x_of_row[df.index.get_indexer(group_df.index)]

        # Pool styling subgroups into a shared x band with per-point symbol and opacity.
        bucket = point_buckets.setdefault((separate_group, color_group), {
            "x": [], "y": [], "text": [], "customdata": [],
            "symbol": [], "opacity": [], "subcolor": [],
        })
        bucket["x"].append(x_jittered)
        bucket["y"].append(y_data)
        bucket["text"].append(group_df[unique_row_id_col].to_numpy())
        # Only when the frame has a FOV column. The concat below skips a key with no
        # chunks, so an omitted one simply never reaches the trace.
        if fov_name_col is not None:
            bucket["customdata"].append(point_customdata.to_numpy())
        # Constant within a subgroup because they come from the group key, so repeat
        # rather than read one per row.
        bucket["symbol"].append(np.repeat(marker_symbol, len(y_data)))
        bucket["opacity"].append(np.repeat(marker_opacity, len(y_data)))
        # Allocate value arrays only when subcolor is active.
        if subcolor_of:
            bucket["subcolor"].append(group_df[subcolor_by].fillna("N/A").astype(str).to_numpy())

    # Choose one renderer for the full figure; mixed SVG/WebGL layering
    # would override colour-group draw order.
    scatter_cls = point_trace_class(
        sum(len(chunk) for bucket in point_buckets.values() for chunk in bucket["y"])
    )

    for (separate_group, color_group), bucket in point_buckets.items():
        columns = {name: np.concatenate(chunks)
                   for name, chunks in bucket.items() if chunks}
        # `columns` is built with `if chunks`, so the key is simply absent when the
        # frame has no FOV column.
        customdata_all = columns.get("customdata")
        # Only what both branches share. The matched branch draws a masked subset, so
        # symbol and opacity are passed per call rather than baked in here.
        marker_kwargs = dict(size=point_size, line=dict(width=0.5, color='DarkSlateGrey'))
        trace_kwargs = dict(mode='markers', hovertemplate=final_hovertemplate)
        if scatter_cls is go.Scatter:
            # Only SVG traces support zorder; WebGL box outlines use above-layer shapes.
            trace_kwargs['zorder'] = 1

        if not subcolor_of:
            # A single-colour group needs one trace with per-point symbol and opacity.
            show_legend = color_group not in legend_entries
            if show_legend:
                legend_entries.add(color_group)
            fig.add_trace(scatter_cls(
                x=columns["x"], y=columns["y"],
                name=format_group_label(color_group, group_counts.get(color_group), show_counts),
                marker=dict(color=color_map[color_group], symbol=columns["symbol"],
                            opacity=columns["opacity"], **marker_kwargs),
                showlegend=show_legend,
                legendgroup=color_group,
                text=columns["text"],
                customdata=customdata_all,
                **trace_kwargs
            ))
            continue

        # Interleave subcolor values in batches while retaining per-value legend control.
        for value, mask in interleave_point_batches({
            value: np.flatnonzero(columns["subcolor"] == value)
            # The map's keys are the figure's value list in natural-sort order; a value
            # this group has no points for yields an empty mask, which the batcher skips.
            for value in subcolor_of
        }):
            # All batches of a subcolor value share one legend entry across comparison groups.
            show_entry = value not in legend_entries
            if show_entry:
                legend_entries.add(value)
            fig.add_trace(scatter_cls(
                x=columns["x"][mask], y=columns["y"][mask],
                name=format_group_label(value, subcolor_counts.get(value), show_counts),
                marker=dict(color=subcolor_of[value],
                            symbol=columns["symbol"][mask],
                            opacity=columns["opacity"][mask], **marker_kwargs),
                showlegend=show_entry,
                # Use a control-character separator to prevent ordinary group names from
                # colliding with synthetic legend groups during styling.
                legendgroup=f"subcolor\x1f{value}",
                text=columns["text"][mask],
                customdata=None if customdata_all is None else customdata_all[mask],
                **trace_kwargs
            ))

    if connect_means:
        # Draw connectors above points: later traces in WebGL, explicit zorder in SVG.
        # Keep them below the boxes at zorder=10.
        mean_line_kwargs = {} if scatter_cls is go.Scattergl else {'zorder': 2}
        if separate_groups:
            # Create a line for each separate group
            for section_info in separate_sections_info:
                section_group_name = section_info['group']

                # These are the color groups that exist in this section
                section_color_groups = [combo[1] for combo in section_info['combinations']]

                # Get the relevant data for this section
                section_df = df[df[separate_by] == section_group_name]

                means_to_plot = []
                for color_group in section_color_groups:
                    x_pos = x_positions.get((section_group_name, color_group))
                    if x_pos is None: continue

                    group_data = section_df[section_df[COLOR_GROUP_COL_NAME] == color_group]
                    if not group_data.empty:
                        mean_y = group_data[selected_var].mean()
                        means_to_plot.append({'x': x_pos, 'y': mean_y})

                # Sort by x position to draw line correctly
                if len(means_to_plot) > 1:
                    means_to_plot.sort(key=lambda p: p['x'])
                    x_coords = [p['x'] for p in means_to_plot]
                    y_coords = [p['y'] for p in means_to_plot]

                    fig.add_trace(scatter_cls(
                        x=x_coords,
                        y=y_coords,
                        mode='lines+markers',
                        name=f'Mean ({section_group_name})',
                        line=dict(width=1, dash='solid', color=theme_color),
                        marker=dict(size=8, symbol='x', color=theme_color),
                        showlegend=True,
                        **mean_line_kwargs
                    ))
        else: # No separate_by
            means_to_plot = []
            for color_group in compare_groups:
                x_pos = x_positions.get(color_group)
                if x_pos is None: continue

                group_data = df[df[COLOR_GROUP_COL_NAME] == color_group]
                if not group_data.empty:
                    mean_y = group_data[selected_var].mean()
                    means_to_plot.append({'x': x_pos, 'y': mean_y})

            # Sort by x position to draw line correctly
            if len(means_to_plot) > 1:
                means_to_plot.sort(key=lambda p: p['x'])
                x_coords = [p['x'] for p in means_to_plot]
                y_coords = [p['y'] for p in means_to_plot]

                fig.add_trace(scatter_cls(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name='Mean',
                    line=dict(width=2, dash='solid', color=theme_color),
                    marker=dict(size=8, symbol='x', color=theme_color),
                    showlegend=False,
                    **mean_line_kwargs
                ))

    # --- 2. Add legend traces for opacity and shape mappings ---
    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)

    # --- 3. Add vertical dashed lines between separate sections ---
    if separate_groups and len(separate_groups) > 1:
        # Add vertical lines between sections using actual section boundaries
        for section_idx in range(len(separate_sections_info) - 1):
            current_section = separate_sections_info[section_idx]
            next_section = separate_sections_info[section_idx + 1]

            # Calculate the end of current section and start of next section
            current_section_end = max([x_positions[(combo[0], combo[1])] for combo in current_section['combinations']])
            next_section_start = min([x_positions[(combo[0], combo[1])] for combo in next_section['combinations']])

            # Place line at the center of the gap
            line_x = (current_section_end + next_section_start) / 2
            fig.add_vline(
                x=line_x,
                line_dash="dash",
                line_color=theme_color,
                line_width=2
            )

    # Build title with visual encoding information
    title_parts = [f'Distribution of {pretty_var} by {", ".join(color_by)}']
    if separate_by and separate_by.strip() != "":
        title_parts.append(f'separated by: {separate_by}')
    if opacity_by and opacity_by.strip() != "":
        title_parts.append(f'opacity: {opacity_by}')
    if shape_by and shape_by.strip() != "":
        title_parts.append(f'shape: {shape_by}')
    if subcolor_by and subcolor_by.strip() != "":
        title_parts.append(f'subcolor: {subcolor_by}')

    full_title = title_parts[0]
    if len(title_parts) > 1:
        full_title += f' ({", ".join(title_parts[1:])})'

    # Configure x-axis labels and layout
    if separate_groups:
        # Section headers are annotations directly under the axis line; the group tick
        # labels are pushed below them by ticklabelstandoff. The header slot is one line
        # of text at a size set here, so reserving room for it is arithmetic on that size.
        header_font_size = st.session_state.get("plot_axis_label_size", 12)
        longest_tick_label = max((len(str(t)) for t in x_tick_labels_actual), default=0)
        # One bold line plus padding. ticklabelstandoff pushes the group labels this far
        # from the axis, on top of Plotly's own default standoff; the header sits in that gap.
        header_slot_px = round(1.6 * header_font_size)
        # Fix the angle for app/export parity: short labels stay upright; longer labels
        # slant uphill, matching Matplotlib's rotation=45 with right alignment.
        tick_angle = 0 if longest_tick_label <= 4 else -45
        for section_info in separate_sections_info:
            if section_info['combinations']:  # Only add annotation if section has data
                fig.add_annotation(
                    x=section_info['center'],
                    y=0,  # Bottom of the plot area
                    # Padding between the axis line and the header, not a measurement of
                    # anything below it.
                    yshift=-round(0.25 * header_font_size),
                    text=f"<b>{section_info['group']}</b>",
                    showarrow=False,
                    xref="x",
                    yref="paper",
                    # apply_plot_styling() overwrites every annotation's size with
                    # plot_axis_label_size but leaves the colour, so that must be theme-aware.
                    font=dict(size=header_font_size, color=theme_color),
                    xanchor="center",
                    yanchor="top"
                )

        xaxis_config = dict(
            tickvals=x_tick_positions_actual,
            ticktext=x_tick_labels_actual,
            tickangle=tick_angle,  # fixed above so the export can reproduce it
            ticklabelstandoff=header_slot_px,  # the section header sits in this gap
            zeroline=False,
            tickfont=dict(color=theme_color),
            # The tick labels are the lowest element, so a bottom margin sized from them
            # also covers the header above. Plotly measures them in the browser.
            automargin=True,
        )
    else:
        # Standard x-axis configuration
        xaxis_config = dict(
            tickvals=list(range(len(compare_groups))),
            ticktext=ordered_compare_groups,
            zeroline=False,
            tickfont=dict(color=theme_color),
            # As in the separate_by branch above; here Plotly rotates the labels itself.
            automargin=True,
        )

    # --- Loop 2: Boxplots (on top of points) ---
    # One box per (section, colour group) from that cell's pooled rows — not per
    # shape/opacity subgroup, which would stack several boxes on one x position.
    if add_boxplot:
        for (separate_group, color_group), rows in pooled_rows.items():
            x_position = cell_x_position(separate_group, color_group)
            if x_position is None:
                continue

            marker_color = color_map[color_group]
            trace_name = color_group
            y_data = all_y[rows]

            # Calculate boxplot statistics manually to enforce 1.5 IQR whiskers
            q1 = np.percentile(y_data, 25)
            median = np.percentile(y_data, 50)
            q3 = np.percentile(y_data, 75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            lower_fence = max(lower_fence, np.min(y_data))
            upper_fence = q3 + 1.5 * iqr
            upper_fence = min(upper_fence, np.max(y_data))
            mean_val = np.mean(y_data)

            # Add Box trace (Mean as dashed line, Median as solid line)
            fig.add_trace(go.Box(
                x=[x_position], # Align with x-position
                q1=[q1],
                median=[median],
                q3=[q3],
                lowerfence=[lower_fence],
                upperfence=[upper_fence],
                mean=[mean_val],
                name=trace_name,
                marker_color=marker_color,
                fillcolor='rgba(0,0,0,0)', # Transparent fill
                line=dict(color=theme_color, width=3), # Theme-aware color outlines
                boxpoints=False, # Hide points in box trace
                boxmean=True, # Show mean as dashed line
                showlegend=False,
                hoverinfo='y', # Show stats on hover
                zorder=10
            ))

            if scatter_cls is go.Scattergl:
                _add_box_outline_above_gl(fig, x_position, q1, median, q3,
                                          lower_fence, upper_fence, mean_val, theme_color)

    # Set y-axis label based on log transform (pretty_var defined at top)
    y_axis_label = f"log₁₀({pretty_var})" if log_y else pretty_var

    # A floor only, so short labels still get a gutter; xaxis.automargin grows the
    # margin past it to whatever the rendered labels need.
    bottom_margin = 120

    fig.update_layout(
        title=dict(text=full_title, font=dict(color=theme_color)),
        xaxis=xaxis_config,
        yaxis=dict(
            title=dict(text=y_axis_label, font=dict(color=theme_color)),
            showline=False,
            tickfont=dict(color=theme_color)
        ),
        showlegend=True, # Show legend entries based on the 'name' of each go.Box trace
        hovermode='closest', # Hover behavior
        margin=dict(l=50, r=20, t=50, b=bottom_margin), # Floor; xaxis.automargin grows it
    )

    # --- 4. Add statistical annotations ---
    if compare_pairs != [] and ((effect_size_method != "None" and mean_or_median is not None) or (statistical_test != "None")):
        if separate_groups:
            # Get user selection for statistical comparisons once (to avoid duplicate widget keys)
            selected_pairs = comparison_pair_widget(compare_pairs)

            # Get threshold once to avoid duplicate widgets across sections
            threshold = 0.0
            threshold_key_suffix = selected_var
            if effect_size_method == "Glass's Delta":
                threshold = st.number_input("Glass's Delta Threshold", value=number_input_default(st.session_state, f"glass_delta_thresh_{threshold_key_suffix}", 0.7), min_value=0.0, max_value=3.0, step=0.05,
                                            key=f"glass_delta_thresh_{threshold_key_suffix}")
            elif effect_size_method == "Absolute Cohen's d":
                threshold = st.number_input(
                    "Absolute Cohen's d threshold",
                    value=number_input_default(st.session_state, f"cohens_d_thresh_{threshold_key_suffix}", 0.5),
                    min_value=0.0,
                    max_value=3.0,
                    step=0.1,
                    key=f"cohens_d_thresh_{threshold_key_suffix}",
                )

            if selected_pairs:  # Only proceed if user selected some pairs
                # Calculate global data range ONCE for consistent spacing across all sections
                global_min_y = df[selected_var].min(skipna=True)
                global_max_y = df[selected_var].max(skipna=True)
                global_data_range = (global_min_y, global_max_y)

                # Apply statistical annotations within each separate section
                for section_info in separate_sections_info:
                    if len(section_info['combinations']) > 1:  # Need at least 2 groups for comparison
                        # Extract just the color groups that exist in this section
                        section_color_groups = [combo[1] for combo in section_info['combinations']]
                        section_compare_pairs = list(combinations(section_color_groups, 2))

                        # Filter to only include pairs that user selected and exist in this section
                        filtered_section_pairs = []
                        for pair in section_compare_pairs:
                            if pair in selected_pairs or tuple(reversed(pair)) in selected_pairs:
                                filtered_section_pairs.append(pair)

                        if filtered_section_pairs:
                            # Filter the dataframe to only include this separate_by group
                            section_df = df[df[separate_by] == section_info['group']].copy()

                            # Create position mapping for this section
                            section_position_map = {}
                            actual_positions = []
                            for combo in section_info['combinations']:
                                separate_group, color_group = combo
                                actual_x = x_positions[(separate_group, color_group)]
                                section_position_map[color_group] = actual_x
                                actual_positions.append(actual_x)

                            # Apply annotations for this section with actual positions
                            _add_effect_size_annotations(
                                fig=fig,
                                df=section_df,
                                selected_var=selected_var,
                                compare_groups=section_color_groups,
                                group_col_name=COLOR_GROUP_COL_NAME,
                                all_possible_pairs=section_compare_pairs,
                                annotation_color=theme_color,
                                effect_size_method=effect_size_method,
                                mean_or_median=mean_or_median,
                                position_map=section_position_map,
                                selected_pairs=filtered_section_pairs,
                                threshold=threshold,
                                statistical_test=statistical_test,
                                global_data_range=global_data_range,  # Pass global range for consistent spacing
                                section_label=section_info['group']
                            )
        else:
            # Standard statistical annotations when no separate_by
            _add_effect_size_annotations(
                fig=fig,
                df=df,
                selected_var=selected_var,
                compare_groups=ordered_compare_groups,  # Use ordered groups for correct bracket positioning
                group_col_name=COLOR_GROUP_COL_NAME,
                all_possible_pairs=compare_pairs,
                annotation_color=theme_color,
                effect_size_method=effect_size_method,
                mean_or_median=mean_or_median,
                statistical_test=statistical_test,
                position_map=x_positions  # Pass position map for correct y-range calculation after reordering
            )

    # Drop the temporary group column if it exists
    if COLOR_GROUP_COL_NAME in df.columns:
        df.drop(columns=[COLOR_GROUP_COL_NAME], inplace=True)
    return fig
