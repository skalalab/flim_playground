from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import umap

def dimension_reduction(X, n_components=2, method="PCA", umap_neighbors=15, umap_min_dist=0.1):
    # Standardize features before PCA and umap
    exp_var = None
    X_std = StandardScaler().fit_transform(X)
    if method == "PCA":
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_std)
        df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
        exp_var = pca.explained_variance_ratio_ * 100
    elif method == "UMAP":
        reducer = umap.UMAP(n_neighbors=umap_neighbors,min_dist=umap_min_dist,   
               metric='euclidean', n_components=n_components)
        df = pd.DataFrame(reducer.fit_transform(X_std), columns=["UMAP1", "UMAP2"])
    return df, exp_var

