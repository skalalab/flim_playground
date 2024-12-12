from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import umap
def dimension_reduction(X, n_components=2, method="PCA"):
    # Standardize features before PCA
    X_std = StandardScaler().fit_transform(X)
    if method == "PCA":
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_std)
        df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
        exp_var = pca.explained_variance_ratio_ * 100
    elif method == "UMAP":
        reducer = umap.UMAP( n_neighbors=15,
               min_dist=0.1,   
               metric='euclidean', n_components=n_components)
        df = pd.DataFrame(reducer.fit_transform(X_std), columns=["UMAP1", "UMAP2"])
        exp_var = None
    return df, exp_var