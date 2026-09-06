"""Observation and summary layers for a plot whose main points are replicates."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .helpers import _density_at_points, get_point_visual_mappings, hover_field, interleave_point_batches
from .plot_defaults import (
    SUPERPLOT_OBSERVATION_MIN_SIZE,
    SUPERPLOT_OBSERVATION_OPACITY_SCALE,
    SUPERPLOT_OBSERVATION_SIZE_SCALE,
)


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
        density = _density_at_points(values)
        norm = density / np.max(density) if len(density) and np.max(density) > 0 else np.ones(len(values))
        xs = x + np.random.default_rng(42).uniform(-1, 1, len(values)) * norm * 0.35
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
