from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import umap
import plotly.graph_objects as go
import seaborn as sns

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


def create_figure(df, axis_labels=["PC1", "PC2"], colored_by="treatment", exp_var=None):
    colored_by = colored_by[0]
    unique_color_groups = df[colored_by].unique()
    palette = sns.color_palette("tab20", n_colors=len(unique_color_groups))
    color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.6)" for color in palette]
    color_map = {t: color_sequence[i] for i, t in enumerate(unique_color_groups)}

    # Create scatter plot
    fig = go.Figure()
    for g in unique_color_groups:
        g_df =  df[df[colored_by] == g]
        fig.add_trace(
            go.Scatter(
                x=g_df[axis_labels[0]],
                y=g_df[axis_labels[1]],
                mode='markers',
                name=f'{g}',
                text=g_df["base_name"],
                customdata=g_df["image_name"],
                hovertemplate="<b>%{text}</b>",
                marker=dict(color=color_map[g])
            ),
    )

    # Update axis labels to include explained variance
    if exp_var is not None: 
        fig.update_xaxes(title_text=f"{axis_labels[0]}({exp_var[0]:.2f}%)")
        fig.update_yaxes(title_text=f"{axis_labels[1]}({exp_var[1]:.2f}%)")
    else:
        fig.update_xaxes(title_text=f"{axis_labels[0]}")
        fig.update_yaxes(title_text=f"{axis_labels[1]}")

    return fig