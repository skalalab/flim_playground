"""Feature Comparison: box, violin, strip and sina plots with effect sizes and statistics.

Pure computation (layout, labels, effect sizes, bracket planning, densities, SuperPlot
summaries) is shared with standalone export and embedded with inspect.getsource(); the
Plotly half draws it in the app. It knows nothing about Collapse by: the page hands it an
already-collapsed frame.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde, median_abs_deviation, ttest_ind

from src.column_roles import code_span
from src.feature_labels import format_feature_label
from src.widgets.analysis_widget_state import number_input_default
from src.widgets.visualization_widgets import comparison_overlay_widget, comparison_pair_widget
from .helpers import (
    _palette_rgb,
    _sorted_levels,
    add_point_legend_traces,
    format_group_label,
    get_point_visual_mappings,
    hover_field,
    log_negative_error,
    point_trace_class,
)
from .plot_defaults import (
    SUPERPLOT_OBSERVATION_MIN_SIZE,
    SUPERPLOT_OBSERVATION_OPACITY_SCALE,
    SUPERPLOT_OBSERVATION_SIZE_SCALE,
    SUPERPLOT_REPLICATE_JITTER_WIDTH,
    SUPERPLOT_REPLICATE_LINE_WIDTH,
    SUPERPLOT_REPLICATE_SIZE_SCALE,
    get_context_theme_color,
)


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


def comparison_layout(compare_groups, separate_groups, present_cells, custom_order=None,
                      section_gap=0.5):
    """Slots, sections, ticks and dividers of a Feature Comparison figure.

    ``present_cells`` holds the (section, colour group) pairs with observations and
    is read only when ``separate_groups`` is given. A custom order moves the listed
    groups first and keeps the rest in their natural order. Positions are keyed
    (section, group) when separated, else by group; the ``all_data`` placeholder
    group gets a blank tick. Shared verbatim with exported scripts.
    """
    from itertools import combinations

    def ordered(groups, key):
        groups = list(groups)
        if custom_order and key in custom_order:
            first = [group for group in custom_order[key] if group in groups]
            return first + [group for group in groups if group not in first]
        return groups

    compare = ordered(compare_groups, "compare_groups")
    separate = ordered(separate_groups or [], "separate_groups")
    layout = {"compare_groups": compare, "separate_groups": separate,
              "compare_pairs": list(combinations(compare, 2)),
              "x_positions": {}, "tick_positions": [], "tick_labels": [],
              "sections": [], "dividers": []}
    if not separate:
        for index, group in enumerate(compare):
            layout["x_positions"][group] = index
            layout["tick_positions"].append(index)
            layout["tick_labels"].append("" if group == "all_data" else group)
        return layout
    position = 0
    for index, section in enumerate(separate):
        cells = [(section, group) for group in compare if (section, group) in present_cells]
        if index > 0:
            # The divider sits at the centre of the gap between neighbouring sections.
            layout["dividers"].append(position + (section_gap - 1) / 2)
            position += section_gap
        start = position
        for cell in cells:
            layout["x_positions"][cell] = position
            layout["tick_positions"].append(position)
            layout["tick_labels"].append("" if cell[1] == "all_data" else cell[1])
            position += 1
        layout["sections"].append({"group": section, "combinations": cells,
                                   "center": (start + position - 1) / 2 if cells else start})
    return layout


def comparison_labels(selected_var, color_by, *, log_y, separate_by, opacity_by, shape_by,
                      subcolor_by, engine="plotly"):
    """Title and y-axis label shared by the app's figure and exported scripts."""
    pretty_var = format_feature_label(selected_var, engine=engine)
    encodings = [f"{name}: {column}" for name, column in (
        ("separated by", separate_by), ("opacity", opacity_by),
        ("shape", shape_by), ("subcolor", subcolor_by)) if column and str(column).strip()]
    title = f"Distribution of {pretty_var} by {', '.join(color_by)}"
    if encodings:
        title += f" ({', '.join(encodings)})"
    return {"title": title, "y": f"log₁₀({pretty_var})" if log_y else pretty_var}


def feature_comparison_plot(df, unique_row_id_col, fov_name_col, selected_var, color_by, opacity_by=None, shape_by=None, separate_by=None, colormap="tab10", effect_size_method="None", mean_or_median=None, statistical_test="None", custom_order=None, subcolor_by=None, row_id_label="ID", *, collapse_by=None, source_df=None, source_row_id_col=None, source_row_id_label="ID", source_fov_name_col=None):

    # Get theme color once at the start for all theme-aware elements
    theme_color = get_context_theme_color()
    # Pretty FLIM label (Greek notation) reused for hover, title, and y-axis
    pretty_var = format_feature_label(selected_var)

    # Parent columns add margins to their first/last checkbox, even inside this row.
    st.html("""
        <style>
        .st-key-feature_comparison_controls [data-testid="stCheckbox"] {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        </style>
    """)
    with st.container(key="feature_comparison_controls", horizontal=True,
                      vertical_alignment="center", gap="medium"):
        log_y = st.checkbox("Log Y", value=False, key=f"log_y_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")
        overlay = comparison_overlay_widget(selected_var, color_by, separate_by, collapse_by)
        add_boxplot = overlay == "Boxplot"
        connect_means = st.checkbox("Connect means", value=False, key=f"connect_means_{selected_var}_{'_'.join(color_by)}_{separate_by or ''}")

    # Create a working copy to avoid modifying the original dataframe
    df = df.copy()
    superplot = overlay == "SuperPlot"
    if superplot:
        if source_df is None:
            raise ValueError("SuperPlot requires the original observations before Collapse by.")
        source_df = source_df.dropna(subset=[selected_var]).copy().reset_index(drop=True)

    # Apply log transform if requested (consistent with distribution_2d.py)
    if log_y:
        if (df[selected_var] < 0).any() or (superplot and (source_df[selected_var] < 0).any()):
            st.error(log_negative_error(selected_var))
            log_y = False
        else:
            df[selected_var] = np.log10(df[selected_var] + 1e-6)
            if superplot:
                source_df[selected_var] = np.log10(source_df[selected_var] + 1e-6)

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
    # Group keys are (color, shape, opacity, section); keep column roles distinct.
    section_color_pairs = {(key[3], key[0]) for key, _ in grouped_list}
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

    layout = comparison_layout(compare_groups, separate_groups, section_color_pairs, custom_order)
    ordered_compare_groups = layout["compare_groups"]
    # Pairs come from the ordered groups so Glass's Delta uses the right control group.
    compare_pairs = layout["compare_pairs"]
    x_positions = layout["x_positions"]
    point_size = 5

    # Track legend entries to avoid duplicates
    legend_entries = set()

    if separate_groups:
        ordered_separate_groups = layout["separate_groups"]
        x_tick_positions_actual = layout["tick_positions"]
        x_tick_labels_actual = layout["tick_labels"]
        separate_sections_info = layout["sections"]

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
        # Keep SuperPlot replicate centers within the width of the SEM caps.
        max_jitter = SUPERPLOT_REPLICATE_JITTER_WIDTH if superplot else 0.35
        x_of_row[rows] = x_position + sina_offsets(all_y[rows], max_jitter)

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
        default_opacity = 1.0 if superplot else 0.7
        marker_opacity = (opacity_map.get(opacity_group, default_opacity)
                          if opacity_map and opacity_group is not None else default_opacity)
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
        + (len(source_df) if superplot else 0)
    )

    display_df = df
    if superplot:
        add_superplot_observations(
            fig, source_df, selected_var, color_by, separate_by, shape_by, opacity_by,
            subcolor_by, color_map, shape_map, opacity_map, subcolor_of,
            cell_x_position, scatter_cls, point_size, source_row_id_col,
            source_row_id_label, source_fov_name_col, pretty_var)
        summary_cols = [COLOR_GROUP_COL_NAME] + ([separate_by] if separate_by else [])
        summary = summarize_superplot(plotted, selected_var, summary_cols)
        display_df = superplot_display_frame(source_df, summary, selected_var, summary_cols)
        if (summary["count"] < 2).any():
            st.caption("SEM is unavailable for groups with fewer than two replicates; "
                       "their mean bars are still shown.")
        # Draw bars before replicate dots so the caps cannot obscure their centers.
        add_superplot_summary(fig, summary, separate_by, cell_x_position, scatter_cls, theme_color)

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
        if superplot:
            marker_kwargs.update(size=point_size * SUPERPLOT_REPLICATE_SIZE_SCALE,
                                 line=dict(width=SUPERPLOT_REPLICATE_LINE_WIDTH, color='DarkSlateGrey'))
            trace_kwargs['meta'] = {"superplot_role": "replicate"}
        if scatter_cls is go.Scatter:
            # Only SVG traces support zorder; WebGL box outlines use above-layer shapes.
            trace_kwargs['zorder'] = 2 if superplot else 1

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
    for line_x in layout["dividers"]:
        fig.add_vline(x=line_x, line_dash="dash", line_color=theme_color, line_width=2)

    # Title and y label name the feature and every active encoding (shared with the export).
    labels = comparison_labels(selected_var, color_by, log_y=log_y, separate_by=separate_by,
                               opacity_by=opacity_by, shape_by=shape_by, subcolor_by=subcolor_by)
    full_title = labels["title"]

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
            tickvals=layout["tick_positions"],
            ticktext=layout["tick_labels"],
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

            # 1.5 IQR whiskers capped at the observed extrema (shared with the export).
            stats = box_statistics(y_data)
            q1, median, q3 = stats["q1"], stats["median"], stats["q3"]
            lower_fence, upper_fence, mean_val = stats["lower_fence"], stats["upper_fence"], stats["mean"]

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

    y_axis_label = labels["y"]

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
                global_min_y = display_df[selected_var].min(skipna=True)
                global_max_y = display_df[selected_var].max(skipna=True)
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
                                section_label=section_info['group'],
                                display_df=display_df[display_df[separate_by] == section_info['group']]
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
                position_map=x_positions,  # Keep statistical samples distinct from drawing bounds.
                display_df=display_df
            )

    # Drop the temporary group column if it exists
    if COLOR_GROUP_COL_NAME in df.columns:
        df.drop(columns=[COLOR_GROUP_COL_NAME], inplace=True)
    return fig


# SuperPlot overlay: observation and summary layers for replicate points.
def summarize_superplot(df, selected_var, group_cols):
    """One equally weighted mean and sample SEM per group of replicate points.

    Self-contained apart from pandas/numpy so the standalone export can inline it.
    A singleton has an undefined SEM; its mean remains available for plotting.
    Grouping keys stay in the index so user column names cannot collide with
    the count, mean, and sem statistic columns.
    """
    return (df.groupby(group_cols, sort=False, observed=True, dropna=False)[selected_var]
            .agg(count="count", mean="mean", sem="sem"))


def add_superplot_observations(fig, source, selected_var, color_by, separate_by,
                              shape_by, opacity_by, subcolor_by, color_map,
                              shape_map, opacity_map, subcolor_map, x_position,
                              scatter_cls, point_size, row_id_col, row_id_label,
                              fov_name_col, pretty_var, group_col="compare_group"):
    """Draw source observations using the main layer's maps and group positions.

    The caller owns this source copy. Add its grouping columns for later visual
    bounds calculations, while keeping source values out of statistical inputs.
    """
    get_point_visual_mappings(
        source, color_by=color_by, shape_by=shape_by, opacity_by=opacity_by,
        separate_by=separate_by, group_col_name=group_col, overlap_point=False)
    group_cols = [group_col] + ([separate_by] if separate_by else [])
    hover = hover_field(pretty_var, "%{y:.3f}")
    hover += hover_field(row_id_label, "%{text}")
    has_fov = fov_name_col is not None and fov_name_col in source.columns
    if has_fov:
        hover += hover_field(fov_name_col, "%{customdata}")
    hover += "<extra></extra>"

    for key, rows in source.groupby(group_cols, sort=False, observed=True):
        color_group, *section = key if isinstance(key, tuple) else (key,)
        x = x_position(section[0] if section else None, color_group)
        if x is None or color_group not in color_map:
            continue
        values = rows[selected_var].to_numpy()
        xs = x + sina_offsets(values, 0.35)
        shapes = (rows[shape_by].map(shape_map).fillna("circle").to_numpy()
                  if shape_by and shape_map else np.repeat("circle", len(rows)))
        opacities = (rows[opacity_by].map(opacity_map).fillna(1.0).to_numpy(dtype=float)
                     if opacity_by and opacity_map else np.ones(len(rows))) * SUPERPLOT_OBSERVATION_OPACITY_SCALE
        labels = rows[row_id_col].to_numpy() if row_id_col in rows else rows.index.astype(str).to_numpy()
        fovs = rows[fov_name_col].to_numpy() if has_fov else None
        if subcolor_map:
            levels = rows[subcolor_by].fillna("N/A").astype(str).to_numpy()
            batches = interleave_point_batches({level: np.flatnonzero(levels == level) for level in subcolor_map})
        else:
            batches = [(color_group, np.arange(len(rows)))]
        for value, mask in batches:
            fig.add_trace(scatter_cls(
                x=xs[mask], y=values[mask], text=labels[mask],
                customdata=fovs[mask] if has_fov else None,
                mode="markers", showlegend=False, name=str(value),
                legendgroup=f"subcolor\x1f{value}" if subcolor_map else color_group,
                marker=dict(size=max(SUPERPLOT_OBSERVATION_MIN_SIZE,
                                     point_size * SUPERPLOT_OBSERVATION_SIZE_SCALE),
                            color=subcolor_map[value] if subcolor_map else color_map[color_group],
                            symbol=shapes[mask], opacity=opacities[mask],
                            line=dict(width=0)),
                hovertemplate=hover, meta={"superplot_role": "observation"},
                **({"zorder": 0} if scatter_cls is go.Scatter else {})))


def add_superplot_summary(fig, summary, separate_by, x_position, scatter_cls, color,
                          group_col="compare_group"):
    """Draw mean bars and capped SEM above observations, below replicate dots."""
    for key, row in summary.iterrows():
        keys = key if isinstance(key, tuple) else (key,)
        group = dict(zip(summary.index.names, keys))
        x = x_position(group.get(separate_by) if separate_by else None, group[group_col])
        if x is None:
            continue
        mean, sem = float(row["mean"]), float(row["sem"])
        xs, ys = [x - 0.2, x + 0.2], [mean, mean]
        if np.isfinite(sem):
            low, high = mean - sem, mean + sem
            xs += [None, x, x, None, x - 0.1, x + 0.1, None, x - 0.1, x + 0.1]
            ys += [None, low, high, None, low, low, None, high, high]
        fig.add_trace(scatter_cls(
            x=xs, y=ys, mode="lines", line=dict(color=color, width=2),
            name="Mean ± SEM", showlegend=False, hoverinfo="skip",
            meta={"superplot_role": "summary", "count": int(row["count"]),
                  "mean": mean, "sem": sem if np.isfinite(sem) else None},
            **({"zorder": 1} if scatter_cls is go.Scatter else {})))


def superplot_display_frame(source, summary, selected_var, group_cols):
    """Bounds for annotation placement include cells and both ends of each SEM.

    These synthetic rows are strictly for geometry, never statistical analysis.
    """
    ends = []
    for sign in (-1, 1):
        frame = summary.index.to_frame(index=False)
        frame[selected_var] = (summary["mean"] + sign * summary["sem"].fillna(0)).to_numpy()
        ends.append(frame)
    return pd.concat([source[[*group_cols, selected_var]], *ends], ignore_index=True)


# Effect sizes, bracket planning, density and sina helpers: only Feature Comparison and its export use them.
def glass_delta(group1, group2, mean_or_median):
    # Glass's delta uses group1 as control; its spread needs at least two samples.
    if len(group1) < 2:
        return np.nan
    if mean_or_median == "Mean": 
        diff = np.mean(group2) - np.mean(group1)
        group1_sd = np.std(group1, ddof=1)  # Using Bessel's correction with ddof=1
    else:
        diff = np.median(group2) - np.median(group1)
        # use MAD (median_absolute_deviation)
        # scale: normal: divides by 0.67449 → multiplies by 1.4826 internally
        group1_sd = median_abs_deviation(group1, scale='normal')

    if group1_sd == 0:  # constant control group -> effect size undefined (avoid inf/nan divide)
        return np.nan
    return diff / group1_sd


def cohens_d(group1, group2, mean_or_median):
    """Compute absolute Cohen's d (|d|) for two independent samples."""
    # sample sizes
    n1, n2 = len(group1), len(group2)

    # Each group needs at least two samples for its sample variance,
    # regardless of the combined sample count.
    if n1 < 2 or n2 < 2:
        return np.nan

    if mean_or_median == "Mean": 
        # mean difference
        diff = np.mean(group1) - np.mean(group2)
         # unbiased sample variances
        s1_2, s2_2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        # pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1)*s1_2 + (n2 - 1)*s2_2) / (n1 + n2 - 2))
    else:
        diff = np.median(group1) - np.median(group2)
        mad_1, mad_2 = median_abs_deviation(group1, scale="normal"), median_abs_deviation(group2, scale="normal")
        pooled_sd = np.sqrt(((n1 - 1) * mad_1**2 + (n2 - 1) * mad_2**2) /
                     (n1 + n2 - 2))
    if pooled_sd == 0:  # no pooled spread -> effect size undefined (avoid inf/nan divide)
        return np.nan
    return abs(diff / pooled_sd)


def _calculate_effect_size(group1_data, group2_data, method: str, mean_or_median):
    """
    Calculates the effect size between two groups using the specified method.
    """
    if group1_data.empty or group2_data.empty:
        return None
    if method == "Glass's Delta":
        return glass_delta(group1_data, group2_data, mean_or_median)
    elif method == "Absolute Cohen's d":
        return cohens_d(group1_data, group2_data, mean_or_median)
    raise ValueError(f"Unsupported effect size method: {method}")


def p_value_stars(pval):
    """Significance stars: four at 1e-4, three at 1e-3, two at 0.01, one at 0.05, none above."""
    if pd.isna(pval):
        return ""
    for stars, cutoff in (("****", 0.0001), ("***", 0.001), ("**", 0.01), ("*", 0.05)):
        if pval <= cutoff:
            return stars
    return ""


def bracket_positioning_metrics(data_range):
    """Bracket spacing as fractions of the data range, one scale across every section."""
    return {
        'offset_from_data_abs': 0.05 * data_range,
        'vertical_spacing_abs': 0.08 * data_range,
        'bracket_vertical_length_abs': 0.03 * data_range,
        'text_offset_from_bracket_abs': 0.03 * data_range,
        'text_height_allowance_for_collision_abs': 0.04 * data_range,
    }


def _compute_bracket_position(x_start, x_end, region_max_y, positioning_metrics, drawn_annotations_list):
    """Compute bracket y-position with collision detection. Framework-agnostic (pure math).

    Args:
        x_start, x_end: Horizontal span of the bracket.
        region_max_y: Maximum y value in the spanned data region.
        positioning_metrics: Dict with offset_from_data_abs, vertical_spacing_abs,
            bracket_vertical_length_abs, text_offset_from_bracket_abs,
            text_height_allowance_for_collision_abs.
        drawn_annotations_list: List of previously drawn annotations (modified in-place).

    Returns:
        (y_bracket_top, y_text_center, bracket_length) or None if positioning fails.
    """
    offset_abs = positioning_metrics['offset_from_data_abs']
    spacing_abs = positioning_metrics['vertical_spacing_abs']
    bracket_h = positioning_metrics['bracket_vertical_length_abs']
    text_offset = positioning_metrics['text_offset_from_bracket_abs']
    text_h = positioning_metrics['text_height_allowance_for_collision_abs']

    y_candidate = region_max_y + offset_abs

    for _ in range(50):
        y_text_center = y_candidate + text_offset + (text_h / 2)
        y_bottom = y_candidate - bracket_h
        y_top = y_text_center + (text_h / 2)

        collision = False
        for ann in drawn_annotations_list:
            x_overlap = max(x_start, ann['x_start']) < min(x_end, ann['x_end'])
            if x_overlap:
                y_overlap = max(y_bottom, ann['y_bottom']) < min(y_top, ann['y_top'])
                if y_overlap:
                    y_candidate = ann['y_top'] + spacing_abs
                    collision = True
                    break

        if not collision:
            drawn_annotations_list.append({
                'x_start': x_start, 'x_end': x_end,
                'y_bottom': y_bottom, 'y_top': y_top,
                'y_bracket_draw': y_candidate, 'y_text_draw': y_text_center
            })
            return y_candidate, y_text_center, bracket_h

    # Fallback
    y_text_center = y_candidate + text_offset + (text_h / 2)
    y_bottom = y_candidate - bracket_h
    y_top = y_text_center + (text_h / 2)
    drawn_annotations_list.append({
        'x_start': x_start, 'x_end': x_end,
        'y_bottom': y_bottom, 'y_top': y_top,
        'y_bracket_draw': y_candidate, 'y_text_draw': y_text_center
    })
    return y_candidate, y_text_center, bracket_h


def plan_effect_size_brackets(df, selected_var, group_column, pairs, position_map, *,
                              effect_size_method, mean_or_median, threshold, statistical_test,
                              data_range, fallback_y=None, display_df=None):
    """Plan the comparison brackets of one section without drawing them.

    Samples come from ``df``; bracket heights clear the drawn frame ``display_df``
    (replicate means plus observations and SEM ends in a SuperPlot). Pairs are
    processed left to right, shorter spans first, and stacked where they collide.
    Returns ``(brackets, skipped)``: one dict per drawn bracket (``pair``, ``x_start``,
    ``x_end``, ``y_top``, ``y_text``, ``height``, ``text``, ``effect_size``, ``p_value``)
    and the pairs with an empty group. Shared verbatim with exported scripts.
    """
    frame = df if display_df is None else display_df
    metrics = bracket_positioning_metrics(data_range)
    ordered = sorted(pairs, key=lambda pair: (min(position_map[pair[0]], position_map[pair[1]]),
                                              max(position_map[pair[0]], position_map[pair[1]])))
    brackets, skipped, drawn = [], [], []
    for pair in ordered:
        group1 = df.loc[df[group_column] == pair[0], selected_var].dropna()
        group2 = df.loc[df[group_column] == pair[1], selected_var].dropna()
        if group1.empty or group2.empty:
            skipped.append(pair)
            continue
        p_value = np.nan
        if statistical_test != "None":
            try:
                _, p_value = ttest_ind(group1, group2, equal_var=statistical_test == "Independent t-test",
                                       nan_policy='omit')
            except Exception:
                p_value = np.nan
        stars = p_value_stars(p_value)
        effect_size = None
        if effect_size_method != "None":
            effect_size = _calculate_effect_size(group1, group2, effect_size_method, mean_or_median)
            if not (effect_size is not None and abs(effect_size) >= threshold):
                continue
            text = f"{effect_size:.2f}{stars}" if stars else f"Δ={effect_size:.2f}"
        elif stars:
            text = stars
        else:
            continue
        x_start, x_end = sorted((position_map[pair[0]], position_map[pair[1]]))
        spanned = [group for group, pos in position_map.items() if x_start <= pos <= x_end]
        region_max = frame.loc[frame[group_column].isin(spanned), selected_var].max(skipna=True)
        if pd.isna(region_max):
            region_max = 0 if fallback_y is None or pd.isna(fallback_y) else fallback_y
        result = _compute_bracket_position(x_start, x_end, region_max, metrics, drawn)
        if result is None:
            continue
        y_top, y_text, height = result
        brackets.append({"pair": pair, "x_start": x_start, "x_end": x_end, "y_top": y_top,
                         "y_text": y_text, "height": height, "text": text,
                         "effect_size": effect_size, "p_value": p_value})
    return brackets, skipped


def _draw_plotly_bracket(fig, bracket, annotation_color):
    """Draw one planned bracket: a top line, two legs and the centred label."""
    x_start, x_end, y_top = bracket["x_start"], bracket["x_end"], bracket["y_top"]
    fig.add_shape(type="line", x0=x_start, y0=y_top, x1=x_end, y1=y_top,
                  line=dict(width=1.5, color=annotation_color))
    for x_pos_single in [x_start, x_end]:
        fig.add_shape(type="line", x0=x_pos_single, y0=y_top, x1=x_pos_single,
                      y1=y_top - bracket["height"], line=dict(width=1.5, color=annotation_color))
    fig.add_annotation(x=(x_start + x_end) / 2, y=bracket["y_text"], text=bracket["text"],
                       showarrow=False, font=dict(size=12, color=annotation_color), align="center")


def _add_effect_size_annotations(fig, df, selected_var, compare_groups, group_col_name, all_possible_pairs, annotation_color, effect_size_method="None", mean_or_median=None, position_map=None, selected_pairs=None, threshold=None, statistical_test: str = "None", global_data_range=None, section_label=None, display_df=None):
    """
    Adds effect size annotations to the figure.
    Manages selection of pairs, calculation of effect sizes, and calls annotation plotting.
    
    Args:
        position_map: Optional dict mapping group names to actual x-positions for separate sections
        selected_pairs: Optional pre-selected pairs to avoid showing the widget again
        threshold: Optional pre-set threshold to avoid showing the widget again
        global_data_range: Optional tuple (global_min, global_max) for consistent spacing across sections
        section_label: Optional separate_by section name, so the thin-group notice below
            can say which section it is talking about (this runs once per section)
    """
    # Geometry may include observation/summary overlays; samples always come from df.
    annotation_df = df if display_df is None else display_df
    if not all_possible_pairs:
        return

    # Only show widget if pairs aren't pre-selected
    if selected_pairs is None:
        selected_pairs = comparison_pair_widget(all_possible_pairs)

    # Warn about requested pairs with fewer than two points in a group,
    # so an uncomputable comparison is distinguishable from no difference.
    if selected_pairs and (effect_size_method != "None" or statistical_test != "None"):
        counts = {}
        for pair in selected_pairs:
            for group in pair:
                if group not in counts:
                    counts[group] = int(
                        df[df[group_col_name] == group][selected_var].notna().sum())
        thin = [group for group in compare_groups if counts.get(group, 2) < 2]
        thin += [group for group in counts if counts[group] < 2 and group not in thin]
        if thin:
            # Escape file-provided group and section names for Markdown warnings.
            named = ", ".join(
                f"{code_span(group)} ({counts[group]} point"
                f"{'' if counts[group] == 1 else 's'})"
                for group in thin)
            where = f" in {code_span(section_label)}" if section_label else ""
            asked = []
            if effect_size_method != "None":
                asked.append("effect size")
            if statistical_test != "None":
                asked.append("p-value")
            st.warning(
                f"No {' or '.join(asked)} for pairs involving {named}{where} — a spread "
                f"needs at least two points, so nothing is drawn for them.")

    if not selected_pairs or (effect_size_method == "None" and statistical_test == "None"):
        return

    # One scale for every section: the global data range when the caller supplies it.
    if global_data_range is not None:
        global_min_y, global_max_y = global_data_range
    else:
        global_max_y = annotation_df[selected_var].max(skipna=True)
        global_min_y = annotation_df[selected_var].min(skipna=True)
    if pd.isna(global_max_y) or pd.isna(global_min_y) or len(df[selected_var].dropna()) < 2:
        data_range_y = 1  # Default if overall data is all NaN or not enough points
    else:
        data_range_y = global_max_y - global_min_y
    if data_range_y == 0:  # Avoid zero spacing when every value is the same
        data_range_y = 1

    # --- Threshold input based on selected method ---
    if effect_size_method != "None" and threshold is None:
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

    if position_map is None:
        position_map = {group: index for index, group in enumerate(compare_groups)}
    brackets, skipped = plan_effect_size_brackets(
        df, selected_var, group_col_name, selected_pairs, position_map,
        effect_size_method=effect_size_method, mean_or_median=mean_or_median,
        threshold=0.0 if threshold is None else threshold, statistical_test=statistical_test,
        data_range=data_range_y, fallback_y=global_min_y, display_df=annotation_df)
    if effect_size_method != "None":
        for pair in skipped:
            st.warning(f"Skipping pair {pair} due to empty data for one or both groups.")
    for bracket in brackets:
        _draw_plotly_bracket(fig, bracket, annotation_color)


def interleave_point_batches(index_by_level, num_batches=15, random_seed=42):
    """Split each level's point indices into batches and cycle through the levels.

    Return ``[(level, indices), ...]`` in trace order. Interleaving limits occlusion
    bias between levels. A seeded shuffle avoids grouping batches by input row order
    without changing point coordinates. Aim for at least five points per batch to
    limit trace overhead.
    """
    import math

    rng = np.random.default_rng(random_seed)
    batched = []
    for level, indices in index_by_level.items():
        indices = np.asarray(indices)
        if not len(indices):
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        count = min(num_batches, max(1, len(shuffled) // 5))
        size = math.ceil(len(shuffled) / count)
        batched.append((level, [shuffled[i:i+size] for i in range(0, len(shuffled), size)]))

    ordered = []
    for round_index in range(max((len(b) for _level, b in batched), default=0)):
        for level, batches in batched:
            if round_index < len(batches):
                ordered.append((level, batches[round_index]))
    return ordered


def create_subcolor_map(df, subcolor_by, group_col, color_groups, engine="plotly",
                                  colormap="tab10"):
    """Map nested categorical values to consistent colors across the figure.

    Return a naturally ordered ``{value: color}`` map, or None when inactive. Nulls
    become "N/A". Generate a palette for the value count, seeded by the first colormap
    entry, to avoid cycling into duplicate colors.

    "plotly" returns rgba strings; "mpl" returns RGB tuples for script export.
    """
    from src.vis.subcolor_palette import make_palette_cached

    if (not subcolor_by or subcolor_by not in df.columns
            or group_col not in df.columns or not len(color_groups)):
        return None

    values = df[subcolor_by].fillna("N/A").astype(str)
    groups = df[group_col].astype(str)
    # Allocate colors only to values in groups drawn by the figure.
    in_figure = groups.isin([str(group) for group in color_groups])
    all_values = _sorted_levels(values[in_figure])
    if not len(all_values):
        return None
    # Score palette colors at the same alpha used by comparison points and script export.
    palette = make_palette_cached(_palette_rgb(colormap, 1)[0], len(all_values), alpha=0.7)

    colour_of = {}
    for index, value in enumerate(all_values):
        red, green, blue = palette[index]
        if engine == "mpl":
            colour_of[value] = (red, green, blue)
        else:
            # Round RGB bytes to match the exported palette. Marker opacity supplies alpha.
            colour_of[value] = (
                f"rgba({round(red * 255)}, {round(green * 255)}, {round(blue * 255)}, 1.0)"
            )

    return colour_of


def _estimate_density_1d(y_values, bw_method='scott'):
    """Return a KDE callable, or zero density when no valid KDE exists.

    Fewer than two finite points, constant values, or a singular covariance
    use the fallback so points and boxes can render without a density shape.
    """
    y_values = np.asarray(y_values, dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    zero_density = lambda y: np.zeros_like(np.asarray(y, dtype=float))
    if len(y_values) < 2 or np.ptp(y_values) == 0:
        return zero_density
    try:
        return gaussian_kde(y_values, bw_method=bw_method)
    except np.linalg.LinAlgError:
        # Near-singular covariance (e.g. all-but-one identical) can still slip past
        # the ptp check; fall back rather than crash.
        return zero_density


def _density_at_points(y_values, grid_size=1024, bw_method='scott'):
    """Estimate density at each point by interpolating a KDE evaluated on a grid.

    Grid evaluation costs O(n * grid_size), avoiding O(n²) evaluation at all
    training points. Combine uniform and quantile spacing to resolve both the
    full range and dense regions of heavy-tailed data.

    Return zeros for inputs with no valid KDE.
    """
    y_values = np.asarray(y_values, dtype=float)
    finite = y_values[np.isfinite(y_values)]
    # Reject degenerate inputs before building the increasing grid np.interp needs.
    if len(finite) < 2 or np.ptp(finite) == 0:
        return np.zeros_like(y_values)

    kde = _estimate_density_1d(y_values, bw_method=bw_method)
    if not isinstance(kde, gaussian_kde):
        # _estimate_density_1d hit its LinAlgError fallback and returned zero_density.
        return np.zeros_like(y_values)

    half = max(2, grid_size // 2)
    grid = np.unique(np.concatenate([
        np.linspace(finite.min(), finite.max(), half),
        np.quantile(finite, np.linspace(0, 1, half)),
    ]))
    return np.interp(y_values, grid, kde(grid))


def sina_offsets(values, max_jitter, seed=42):
    """Horizontal sina offsets: random signs scaled by local density.

    The rng is scoped to one colour group, which owns the x position, so shape
    and opacity change point styling only and never move a point. A degenerate
    density (a constant column has no KDE) spreads points with uniform jitter so
    they stay visible instead of stacking into one dot.
    """
    densities = _density_at_points(values)
    if len(densities) > 0 and np.max(densities) > 0:
        normalized = densities / np.max(densities)
    else:
        normalized = np.ones_like(densities)
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(-1, 1, size=len(values)) * normalized * max_jitter


def box_statistics(values):
    """Quartiles, 1.5 IQR fences capped at the observed extrema, and the mean of one box."""
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    return dict(q1=q1, median=median, q3=q3,
                lower_fence=max(q1 - 1.5 * iqr, np.min(values)),
                upper_fence=min(q3 + 1.5 * iqr, np.max(values)),
                mean=np.mean(values))
