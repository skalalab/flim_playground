
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from itertools import combinations
import streamlit as st
import plotly.graph_objects as go
import numpy as np
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

def interactive_feature_comparison_plot(df, selected_var, compared_by, stats_test="None"):
    fig = go.Figure()
    df['compare_group'] = df[compared_by].agg('_'.join, axis=1)
    compare_groups = df['compare_group'].unique()
    compare_pairs = list(combinations(compare_groups, 2))
    palette = sns.color_palette("tab10", n_colors=len(compare_groups))
    color_sequence = [f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 1)" for c in palette]
    color_map = {group: color for group, color in zip(compare_groups, color_sequence)}
    # Calculate positions for the swarm plot effect
    # We'll create x-coordinates that spread points out horizontally based on density
    group_positions = {}
    for i, group in enumerate(compare_groups):
        group_df = df[df['compare_group'] == group]
        group_values = group_df[selected_var].values
        
        # Sort values for swarm-like arrangement
        sorted_indices = np.argsort(group_values)
        group_data_sorted = group_values[sorted_indices]
        
        # Calculate spread based on density (similar to swarm plot)
        x_positions = np.zeros_like(group_data_sorted, dtype=float)
        bin_size = (group_values.max() - group_values.min()) / 20 if len(group_values) > 1 else 1
        
        if bin_size > 0:
            # Group points by bin to create swarm-like spread
            for j in range(len(group_data_sorted)):
                # Find nearby points
                nearby = np.abs(group_data_sorted - group_data_sorted[j]) < bin_size
                nearby_count = np.sum(nearby[:j])  # Only consider points we've already placed
                
                # Offset based on how many nearby points we've already placed
                x_positions[j] = i + (nearby_count % 2) * 0.1 * (-1 if nearby_count % 4 < 2 else 1) * (nearby_count // 2 + 1)
        
        # Store positions with original indices
        original_indices = sorted_indices
        group_positions[group] = {
            'x': x_positions,
            'y': group_data_sorted,
            'indices': original_indices,
            'ids': group_df.index.values[original_indices]
        }
    # Add points for each group
    for group in compare_groups:
        pos = group_positions[group]
        group_df = df[df['compare_group'] == group]
        
        fig.add_trace(
            go.Scatter(
                x=pos['x'],
                y=pos['y'],
                mode='markers',
                name=group,
                text=[f"ID: {idx}<br>{selected_var}: {val:.4f}" for idx, val in 
                      zip(pos['ids'], pos['y'])],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(color=color_map[group], size=8)
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{selected_var} by {', '.join(compared_by)}",
        xaxis=dict(
            title='',
            tickmode='array',
            tickvals=list(range(len(compare_groups))),
            ticktext=compare_groups
        ),
        yaxis=dict(title=selected_var),
        hovermode="closest"
    )
    
    # Clean up
    df.drop(columns=['compare_group'], inplace=True)
    
    return fig

def dimension_reduction_plot(df, method="UMAP", colored_by=[], exp_var=None):
    """create a plotly plot to visualize the dimension-reduced data
    """
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