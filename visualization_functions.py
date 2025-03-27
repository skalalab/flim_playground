
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from itertools import combinations
import streamlit as st
import plotly.graph_objects as go

def feature_comparison_plot(df, selected_var, compared_by, stats_test="None"): 
    # create a new copy of df 
    df['compare_group'] = df[compared_by].agg('_'.join, axis=1)
    compare_groups = df['compare_group'].unique()
    compare_pairs = list(combinations(compare_groups, 2))
    # assign a different color to each compare_group
    alpha = 1 
    palette = sns.color_palette("tab10", n_colors=len(compare_groups))
    color_map = {group: (color[0], color[1], color[2], alpha) for group, color in zip(compare_groups, palette)}
   
    fig, ax = plt.subplots()
    sns.boxplot(x="compare_group", y=selected_var, data=df, showfliers=False, palette=color_map, hue="compare_group", ax=ax, boxprops=dict(facecolor="none", edgecolor="black"),)
    sns.swarmplot(x="compare_group", y=selected_var, data=df, palette=color_map,  hue="compare_group", ax=ax, size =2)

    # Add statistical annotations
    if compare_pairs != [] and stats_test != "None":
        pair_chose = st.multiselect("Select statistical tests compare pairs", compare_pairs, default=compare_pairs, key="compare_pairs")
        if pair_chose != []:
            annotator = Annotator(ax, pair_chose, data=df, x="compare_group", y=selected_var)
            annotator.configure(test=stats_test, text_format="star", loc="outside", verbose=2)
            annotator.apply_and_annotate()
    # dynmically adjust the font size of x-axis labels
  #  ax.set_xticklabels(ax.get_xticklabels(), fontsize=12 if len(compare_groups) < 4 else (6 if len(compare_groups) <= 8 else 4))
   # ax.tick_params(axis='x', labelsize=12 if len(compare_groups) < 4 else (6 if len(compare_groups) <= 8 else 4))
    plt.tight_layout()
    df.drop(columns=['compare_group'], inplace=True)
    return fig


def dimension_reduction_plot(df, method="UMAP", colored_by=[], exp_var=None):
    
    fig = go.Figure()

    if method == "Principal Component Analysis":
        axis_labels = ["PC1", "PC2"]
    elif method == "UMAP":
        axis_labels = ["UMAP1", "UMAP2"]
    else:
        axis_labels = ["dim1", "dim2"]
    # create a new copy of df 
    df['unique_color_group'] = df[colored_by].agg('_'.join, axis=1)
    unique_color_groups = df['unique_color_group'].unique()
    alpha = 0.6 if len(unique_color_groups) > 1 else 1.0
    palette = sns.color_palette("tab10", n_colors=len(unique_color_groups))
    color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {alpha})" for color in palette]
    color_map = {t: color_sequence[i] for i, t in enumerate(unique_color_groups)}

    # plot scatter plot iteratively, once for each color group
    for g in unique_color_groups:
        g_df =  df[df['unique_color_group'] == g]
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
    # remove the column after plotting
    df.drop(columns=['unique_color_group'], inplace=True)
    return fig