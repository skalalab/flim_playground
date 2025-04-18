from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.utils.deprecation')
from sklearn.decomposition import PCA
import pandas as pd
import umap

import streamlit as st

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

