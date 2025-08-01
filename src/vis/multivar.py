from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.utils.deprecation')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import umap
import plotly.graph_objects as go
import streamlit as st
from .helpers import get_point_visual_mappings, add_point_legend_traces
import threading    
@st.cache_data()
def dimension_reduction(X, n_components=2, method="UMAP", hyperParam_dict={}):
    exp_var = None
    if 'dr_lock' not in st.session_state:
        st.session_state.dr_lock = threading.Lock()
    with st.session_state.dr_lock:
        # Standardize features before PCA and umap
        X_std = StandardScaler().fit_transform(X)
        if method == "PCA":
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(X_std)
            df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
            exp_var = pca.explained_variance_ratio_ * 100
        elif method == "UMAP":
            umap_neighbors = hyperParam_dict.get('n_neighbors', 15)
            umap_min_dist = hyperParam_dict.get('min_dist', 0.1)
            reducer = umap.UMAP(n_neighbors=umap_neighbors,min_dist=umap_min_dist,   
                metric='euclidean', n_components=n_components)
            df = pd.DataFrame(reducer.fit_transform(X_std), columns=["UMAP1", "UMAP2"])
        elif method == "t-SNE":
            perplexity = hyperParam_dict.get('perplexity', 15)
            early_exaggeration = hyperParam_dict.get('early_exaggeration', 1)
            tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration)
            df = pd.DataFrame(tsne.fit_transform(X_std), columns=["t-SNE1", "t-SNE2"])
    return df, exp_var

def dimension_reduction_plot(df, unique_row_id_col, fov_name_col, selected_features, method="UMAP", hyperParam_dict={}, colored_by=[], opacity_by=None, shape_by=None, exp_var=None, colormap="colorblind"):
    """create a plotly plot to visualize the dimension-reduced data"""
    X = df[selected_features]
    # perform dimension reduction
    df_reduced, exp_var = dimension_reduction(X, n_components=2, method=method, hyperParam_dict=hyperParam_dict)
    # augment df_reduced with required columns and categorical columns used for coloring
    df_reduced[unique_row_id_col] = df[unique_row_id_col].values
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
    for group_key, group_df in grouped:
        color_group = group_key[0]
        shape_group = group_key[1] if shape_by else None
        opacity_group = group_key[2] if shape_by and opacity_by else (group_key[1] if opacity_by else None)
        marker_color = color_map[color_group]
        marker_symbol = shape_map[shape_group] if shape_group is not None and shape_map else 'circle'
        marker_opacity = opacity_map[opacity_group] if opacity_group is not None and opacity_map else 0.8
        fig.add_trace(
            go.Scatter(
                x=group_df[axis_labels[0]],
                y=group_df[axis_labels[1]],
                mode='markers',
                name=f'{color_group}',
                text=group_df[unique_row_id_col],
                customdata=group_df[fov_name_col],
                hovertemplate="<b>%{text}</b>",
                marker=dict(color=marker_color, symbol=marker_symbol, opacity=marker_opacity)
            ),
        )
    # Add shape/opacity legends if needed
    add_point_legend_traces(fig, shape_map, opacity_map, shape_by=shape_by, opacity_by=opacity_by)
    fig.update_layout(
        hovermode='closest'
    )

    # Update axis labels to include explained variance
    if exp_var is not None: 
        fig.update_xaxes(title_text=f"{axis_labels[0]}({exp_var[0]:.2f}%)")
        fig.update_yaxes(title_text=f"{axis_labels[1]}({exp_var[1]:.2f}%)")
    else:
        fig.update_xaxes(title_text=f"{axis_labels[0]}")
        fig.update_yaxes(title_text=f"{axis_labels[1]}")

    return fig