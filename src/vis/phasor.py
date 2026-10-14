"""Phasor plot: reference geometry, category selection and the Plotly figure.

The pure half (`phasor_reference_geometry`, `phasor_columns`, `resolve_phasor_category`,
`_phasor_panel_rows`) is shared with standalone export and embedded with
inspect.getsource(); `phasor_plot` draws it in the app, with faint context points for the
categories not selected.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .dimension_facets import category_panel_rows
from .helpers import add_interleaved_points_trace, format_group_label, get_point_visual_mappings, hover_field
from .plot_defaults import PANEL_CONTEXT_COLOR, get_context_theme_color


def phasor_reference_geometry(f=0.08, harmonic=1):
    """Reference coordinates and labels shared with standalone Phasor exports."""
    u = np.arange(0, 100, 0.01)
    lifetimes = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    wt = 2 * np.pi * f * harmonic * lifetimes
    marker_x = 1 / (1 + wt**2)
    marker_y = wt / (1 + wt**2)
    frequency = f"f = {f * harmonic * 1000} MHz"
    if harmonic != 1:
        frequency += f"\n({harmonic} x {f * 1000} MHz)"
    return dict(
        curve_x=1 / (1 + u**2), curve_y=u / (1 + u**2),
        marker_x=marker_x, marker_y=marker_y,
        lifetime_labels=[f"{tau:g} ns" for tau in lifetimes[:6]],
        label_x=marker_x[:6] - .02, label_y=marker_y[:6] + .03,
        frequency=frequency,
    )


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
    geometry = phasor_reference_geometry(f, harmonic)

    fig.add_trace(go.Scatter(
        x=geometry['curve_x'],
        y=geometry['curve_y'],
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
    fig.add_trace(go.Scatter(
        x=geometry['marker_x'],
        y=geometry['marker_y'],
        mode='markers',
        marker=dict(size=7, color=theme_color),
        name='Lifetime Markers',
        meta=dict(is_lifetime_reference=True),
        hoverinfo='skip', # Hide the hover info for this trace,
        showlegend=False
    ))

    # Annotate the points
    lifetime_labels = geometry['lifetime_labels']
    labels = len(lifetime_labels)
    label_coords = list(zip(geometry['label_x'], geometry['label_y']))

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
    freq_text = geometry['frequency'].replace('\n', '<br>')
    fig.add_annotation(
        x=0.8,
        y=0.5,
        text=freq_text,
        showarrow=False,
        font=dict(size=15, color=theme_color),
        xanchor='left'
    )


def _phasor_panel_rows(df, separate_by=None, color_by=None):
    """Compatibility entry point for Phasor's shared category memberships."""
    return category_panel_rows(df, separate_by, color_by)


def phasor_columns(channel, harmonic):
    """The G/S column names and harmonic label of a channel's phasor features.

    Harmonic 1 is "1st"; any other is "2nd", the only other harmonic the picker offers.
    """
    label = "1st" if harmonic == 1 else "2nd"
    prefix = f"Lifetime fit free_{channel}: "
    return f"{prefix}G({label})", f"{prefix}S({label})", label


def resolve_phasor_category(levels, requested=None):
    """The category to show: the requested one when present, else the first, else None."""
    if not levels:
        return None
    return requested if requested in levels else levels[0]


def select_phasor_category(fig, category=None):
    """Select prepared traces without refitting models or changing the G/S axes.

    Apply to a fresh copy of the base figure before shared plot styling adds
    legend swatches. Category controls can then rerun just the display fragment.
    """
    meta = fig.layout.meta
    if not isinstance(meta, dict) or not meta.get("phasor_categories"):
        return fig
    categories = meta["phasor_categories"]
    category = resolve_phasor_category(categories, category)
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
            mode="markers", marker=dict(color=PANEL_CONTEXT_COLOR, opacity=.18, symbol="circle", size=3),
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

    g_feature, s_feature, harmonic_str = phasor_columns(selected_channel, harmonic)

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
