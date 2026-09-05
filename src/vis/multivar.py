import warnings
import html

import numpy as np

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.utils.deprecation')
import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .helpers import (
    add_interleaved_points_trace,
    get_context_theme_color,
    get_point_visual_mappings,
    hover_field,
    point_trace_class,
)

from .dimension_facets import (
    dimension_facet_groups, dimension_facet_layout, dimension_ranges,
    normalize_dimension_categories,
)


@st.cache_data()
def dimension_reduction(X, n_components=2, method="UMAP", hyperParam_dict={}, random_state=42):
    exp_var = None
    if 'dr_lock' not in st.session_state:
        st.session_state.dr_lock = threading.Lock()
    with st.session_state.dr_lock:
        # Standardize features before dimensionality reduction.
        X_std = StandardScaler().fit_transform(X)
        if method == "PCA":
            # Seeded like UMAP/t-SNE below: svd_solver="auto" can pick the randomized
            # solver on larger inputs, which varies between reruns unless seeded.
            pca = PCA(n_components=n_components, random_state=random_state)
            principal_components = pca.fit_transform(X_std)
            df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
            exp_var = pca.explained_variance_ratio_ * 100
        elif method == "UMAP":
            umap_neighbors = hyperParam_dict.get('n_neighbors', 15)
            umap_min_dist = hyperParam_dict.get('min_dist', 0.1)
            reducer = umap.UMAP(n_neighbors=umap_neighbors,min_dist=umap_min_dist,
                metric='euclidean', n_components=n_components, random_state=random_state)
            df = pd.DataFrame(reducer.fit_transform(X_std), columns=["UMAP1", "UMAP2"])
        elif method == "t-SNE":
            perplexity = hyperParam_dict.get('perplexity', 15)
            early_exaggeration = hyperParam_dict.get('early_exaggeration', 12)
            tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, random_state=random_state)
            df = pd.DataFrame(tsne.fit_transform(X_std), columns=["t-SNE1", "t-SNE2"])
    return df, exp_var

def dimension_reduction_plot(df, unique_row_id_col, fov_name_col, selected_features, colored_by=None, opacity_by=None, shape_by=None, colormap="tab10", method="UMAP", hyperParam_dict=None, row_id_label="ID", separate_by=None):
    """Show one global embedding beside optional categorical highlight maps."""
    # Only measurements determine the cache key. Display controls and metadata do
    # not move coordinates, and category levels come from complete observations.
    df = df.dropna(subset=selected_features)
    if df.empty:
        raise ValueError("No complete observations remain for dimension reduction.")
    groups = dimension_facet_groups(df, separate_by)
    colored_by = [colored_by] if isinstance(colored_by, str) else list(colored_by or [])
    coordinates, exp_var = dimension_reduction(
        df[selected_features], n_components=2, method=method,
        hyperParam_dict=hyperParam_dict or {},
    )
    df_reduced = normalize_dimension_categories(
        df.reset_index(drop=True), [*colored_by, shape_by, opacity_by, *groups["separate_by"]],
    )
    # Internal axis columns must not overwrite an uploaded identifier or category.
    axis_columns = []
    for label in ("_dr_x", "_dr_y"):
        while label in df_reduced.columns:
            label += "_"
        axis_columns.append(label)
    df_reduced[axis_columns] = coordinates.iloc[:, :2].to_numpy()
    axis_labels = list(coordinates.columns[:2])
    x_range, y_range = dimension_ranges(*coordinates.iloc[:, :2].to_numpy().T)
    composition = dimension_facet_layout(groups, x_range, y_range)

    group_column = "unique_color_group"
    while group_column in df_reduced.columns:
        group_column += "_"
    grouped, color_map, shape_map, opacity_map, _ = get_point_visual_mappings(
        df_reduced, color_by=colored_by, shape_by=shape_by, opacity_by=opacity_by,
        group_col_name=group_column, overlap_point=True, colormap=colormap,
    )
    show_counts = st.session_state.get("plot_show_group_counts", False)
    # Build the global interleave once, then subset its traces for every panel.
    # This retains per-point encodings and draw order, with one shared legend.
    overview = go.Figure()
    hover = hover_field(row_id_label, "%{text}")
    if fov_name_col is not None:
        hover += hover_field(fov_name_col, "%{customdata}")
    add_interleaved_points_trace(
        fig=overview, grouped=grouped, color_map=color_map, shape_map=shape_map,
        opacity_map=opacity_map, axis_labels=axis_columns, text_col=unique_row_id_col,
        customdata_col=fov_name_col, hovertemplate=hover, show_counts=show_counts,
    )
    displayed_points = len(df) * (1 + len(groups["panels"]))
    scatter_cls = point_trace_class(displayed_points)
    fig = go.Figure()
    for trace in overview.data:
        if trace.text is None:  # Shape/opacity legend swatches contain no points.
            fig.add_trace(trace)
        else:
            spec = trace.to_plotly_json()
            spec.pop("type")
            fig.add_trace(scatter_cls(**spec, xaxis="x", yaxis="y"))

    theme_color = get_context_theme_color()
    for index, panel in enumerate([composition["overview"], *composition["panels"]], 1):
        suffix = "" if index == 1 else str(index)
        xaxis, yaxis = f"x{suffix}", f"y{suffix}"
        small = index > 1
        for dimension, bounds, domain, anchor, title in (
                ("x", x_range, panel["x_domain"], yaxis, axis_labels[0]),
                ("y", y_range, panel["y_domain"], xaxis, axis_labels[1])):
            if exp_var is not None:
                title += f"({exp_var[0 if dimension == 'x' else 1]:.2f}%)"
            fig.update_layout(**{f"{dimension}axis{suffix}": dict(
                domain=domain, anchor=anchor, range=bounds, matches=dimension if small else None,
                title=dict(text=None if small else title, font=dict(color=theme_color)),
                tickfont=dict(color=theme_color), showticklabels=not small,
                showgrid=False, zeroline=False,
                showline=True, linecolor="black", linewidth=1, mirror=False,
                ticks="" if small else None,
            )})
        if not small:
            continue
        mask = panel["mask"]
        if (~mask).any():
            context = df_reduced.loc[~mask]
            fig.add_trace(scatter_cls(
                x=context[axis_columns[0]], y=context[axis_columns[1]],
                xaxis=xaxis, yaxis=yaxis, mode="markers", name="Other groups",
                marker=dict(color="#b8b8b8", opacity=0.25),
                hoverinfo="skip", showlegend=False,
            ))
        identifiers = df_reduced.loc[mask, unique_row_id_col].to_numpy()
        for trace in overview.data:
            if trace.text is None:
                continue
            keep = np.isin(trace.text, identifiers)
            if not keep.any():
                continue
            spec = trace.to_plotly_json()
            spec.pop("type")
            # Read original arrays: to_plotly_json may serialize numeric arrays.
            for field in ("x", "y", "text", "customdata"):
                value = getattr(trace, field)
                if value is not None:
                    spec[field] = np.asarray(value)[keep]
            for field in ("symbol", "opacity"):
                spec["marker"][field] = np.asarray(getattr(trace.marker, field))[keep]
            spec.update(xaxis=xaxis, yaxis=yaxis, showlegend=False)
            fig.add_trace(scatter_cls(**spec))

    if groups["panels"]:
        for panel in composition["panels"]:
            if len(groups["separate_by"]) == 2 and panel["row"] == 0:
                label = panel["values"][-1]
                fig.add_annotation(
                    x=sum(panel["x_domain"]) / 2, y=panel["y_domain"][1],
                    xref="paper", yref="paper", text=html.escape(str(label)),
                    showarrow=False, xanchor="center", yanchor="bottom", yshift=4,
                    font=dict(color=theme_color, size=14),
                )
            if panel["col"] == groups["ncols"] - 1:
                fig.add_annotation(
                    x=panel["x_domain"][1], y=sum(panel["y_domain"]) / 2,
                    xref="paper", yref="paper", text=html.escape(str(panel["values"][0])),
                    showarrow=False, xanchor="left", yanchor="middle", xshift=6,
                    font=dict(color=theme_color, size=14),
                )
        fig.update_layout(height=round(1000 * composition["plot_height"] + 160),
                          margin=dict(l=80, r=140, t=70, b=90),
                          # Reserve the legend's measured height below the axis
                          # title, independent of the embedding/grid height.
                          legend=dict(orientation="h", yref="container", y=0,
                                      yanchor="bottom", xref="paper", x=0,
                                      groupclick="togglegroup"))
    else:
        fig.update_layout(legend=dict(groupclick="togglegroup"))
    # Streamlit's theme pads axes by 8 px. Touching facets need their lines on
    # the domain edges so neighboring points cannot paint over the boundaries.
    fig.update_layout(hovermode="closest", margin=dict(pad=0, autoexpand=True),
                      xaxis=dict(automargin="height"), yaxis=dict(automargin=False))
    # Canonical domains also let the native chart fit the whole composition in
    # fullscreen without changing the linked coordinate ranges.
    fig.update_layout(meta={"dimension_reduction_layout": {
        "plot_height": composition["plot_height"],
        "axes": {name: list(fig.layout[name].domain) for name in fig.layout
                 if name.startswith(("xaxis", "yaxis"))},
        "annotations": [{"x": item.x, "y": item.y,
                         "xref": item.xref, "yref": item.yref}
                        for item in fig.layout.annotations],
    }})
    return fig
