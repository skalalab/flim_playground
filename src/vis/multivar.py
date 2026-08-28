import warnings

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
)


@st.cache_data()
def dimension_reduction(X, n_components=2, method="UMAP", hyperParam_dict={}, random_state=42):
    exp_var = None
    if 'dr_lock' not in st.session_state:
        st.session_state.dr_lock = threading.Lock()
    with st.session_state.dr_lock:
        # Standardize features before PCA and umap
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

def dimension_reduction_plot(df, unique_row_id_col, fov_name_col, selected_features, colored_by=[], opacity_by=None, shape_by=None, colormap="tab10", method="UMAP", hyperParam_dict={}):
    """create a plotly plot to visualize the dimension-reduced data"""
    X = df[selected_features]
    df_reduced, exp_var = dimension_reduction(X, n_components=2, method=method, hyperParam_dict=hyperParam_dict)
    # augment df_reduced with required columns and categorical columns used for coloring
    df_reduced[unique_row_id_col] = df[unique_row_id_col].values
    if fov_name_col is not None:
        df_reduced[fov_name_col] = df[fov_name_col].values
    # Add all color columns at once if there are any
    if len(colored_by) > 0:
        df_reduced[colored_by] = df[colored_by].values
    if shape_by:
        df_reduced[shape_by] = df[shape_by].values
    if opacity_by:
        df_reduced[opacity_by] = df[opacity_by].values
    # plot the reduced data
    fig = go.Figure()
    if method == "PCA":
        axis_labels = ["PC1", "PC2"]
    elif method == "UMAP":
        axis_labels = ["UMAP1", "UMAP2"]
    elif method == "t-SNE":
        axis_labels = ["t-SNE1", "t-SNE2"]
    else:
        axis_labels = ["dim1", "dim2"]

    GROUP_COL_NAME = 'unique_color_group'
    grouped, color_map, shape_map, opacity_map, _ = get_point_visual_mappings(
        df_reduced,
        color_by=colored_by,
        shape_by=shape_by,
        opacity_by=opacity_by,
        group_col_name=GROUP_COL_NAME,
        overlap_point=True,
        colormap=colormap
    )

    # Use the reusable function to add interleaved points and legend
    add_interleaved_points_trace(
        fig=fig,
        grouped=grouped,
        color_map=color_map,
        shape_map=shape_map,
        opacity_map=opacity_map,
        axis_labels=axis_labels,
        text_col=unique_row_id_col,
        customdata_col=fov_name_col,
        hovertemplate="<b>%{text}</b>",
        show_counts=st.session_state.get("plot_show_group_counts", False)
    )

    theme_color = get_context_theme_color()

    # Update axis labels to include explained variance
    if exp_var is not None:
        fig.update_xaxes(
            title=dict(text=f"{axis_labels[0]}({exp_var[0]:.2f}%)", font=dict(color=theme_color)),
            tickfont=dict(color=theme_color)
        )
        fig.update_yaxes(
            title=dict(text=f"{axis_labels[1]}({exp_var[1]:.2f}%)", font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            showgrid=True
        )
    else:
        fig.update_xaxes(
            title=dict(text=f"{axis_labels[0]}", font=dict(color=theme_color)),
            tickfont=dict(color=theme_color)
        )
        fig.update_yaxes(
            title=dict(text=f"{axis_labels[1]}", font=dict(color=theme_color)),
            tickfont=dict(color=theme_color),
            showgrid=True
        )

    # Lock axis range to prevent rescaling when toggling legend items
    x_data = df_reduced[axis_labels[0]]
    y_data = df_reduced[axis_labels[1]]
    x_padding = (x_data.max() - x_data.min()) * 0.05
    y_padding = (y_data.max() - y_data.min()) * 0.05
    fig.update_xaxes(range=[x_data.min() - x_padding, x_data.max() + x_padding])
    fig.update_yaxes(range=[y_data.min() - y_padding, y_data.max() + y_padding])

    return fig