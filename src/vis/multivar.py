from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.utils.deprecation')
from sklearn.decomposition import PCA
import pandas as pd
import umap
import plotly.graph_objects as go
import streamlit as st
from .helpers import _prepare_group_data

@st.cache_data
def dimension_reduction(X, n_components=2, method="UMAP", hyperParam_dict={}):
    # Standardize features before PCA and umap
    exp_var = None
    X_std = StandardScaler().fit_transform(X)
    if method == "Principal Component Analysis":
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_std)
        df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
        exp_var = pca.explained_variance_ratio_ * 100
    elif method == "UMAP":
        umap_neighbors = 15  # Default value if n_neighbors is not provided
        umap_min_dist = 0.1  # Default value if min_dist is not provided
        # Safely access hyperparameters if they exist
        if hyperParam_dict:
            umap_neighbors = hyperParam_dict.get('n_neighbors', umap_neighbors)
            umap_min_dist = hyperParam_dict.get('min_dist', umap_min_dist)
        reducer = umap.UMAP(n_neighbors=umap_neighbors,min_dist=umap_min_dist,   
               metric='euclidean', n_components=n_components)
        df = pd.DataFrame(reducer.fit_transform(X_std), columns=["UMAP1", "UMAP2"])
    return df, exp_var

def dimension_reduction_plot(df, selected_features, method="UMAP", hyperParam_dict={}, colored_by=[], exp_var=None):
    """create a plotly plot to visualize the dimension-reduced data
    """
    X = df[selected_features]
                    # perform dimension reduction
    df_reduced, exp_var = dimension_reduction(X, n_components=2, method=method, hyperParam_dict=hyperParam_dict)
    # augment df_reduced with required columns and categorical columns used for coloring
    df_reduced["cell_id"] = df["cell_id"].values
    df_reduced["image_name"] = df["image_name"].values
    # Add all color columns at once if there are any
    if len(colored_by) > 0:
        df_reduced[colored_by] = df[colored_by].values
    # plot the reduced data
    fig = go.Figure()
    if method == "Principal Component Analysis":
        axis_labels = ["PC1", "PC2"]
    elif method == "UMAP":
        axis_labels = ["UMAP1", "UMAP2"]
    else:
        axis_labels = ["dim1", "dim2"]

    # colored by unique combinations of the selected categorical columns
    GROUP_COL_NAME = 'unique_color_group'
    unique_color_groups, color_map = _prepare_group_data(df_reduced, colored_by, GROUP_COL_NAME, overlap_point=True)

    # plot scatter plot iteratively, once for each color group
    for g in unique_color_groups:
        g_df =  df_reduced[df_reduced[GROUP_COL_NAME] == g]
        fig.add_trace(
            go.Scatter(
                x=g_df[axis_labels[0]],
                y=g_df[axis_labels[1]],
                mode='markers',
                name=f'{g}',
                text=g_df["cell_id"],   
                customdata=g_df["image_name"],
                hovertemplate="<b>%{text}</b>",
                marker=dict(color=color_map[g])
            ),
    )               
        
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