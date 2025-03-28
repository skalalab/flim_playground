
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
    jitter_amount = 1
    point_size = 5

    # --- 3. Plotting Traces using go.Box (with hidden box) ---
    for group in compare_groups:
        # Filter data for the current group
        g_df = df[df['compare_group'] == group].copy()
        # Drop rows where the variable to plot is NaN, as they cannot be plotted
        g_df = g_df.dropna(subset=[selected_var])
        # Skip this group if no data remains after filtering/dropping NaNs
        if g_df.empty:
            continue
        # --- Prepare Hover Information ---
        # Initialize data containers
        point_text_data = None
        point_customdata = None
        hovertemplate_parts = [
            f"<b>Group:</b> {group}<br>", # Display the specific group name
            f"<b>{selected_var}:</b> %{{y:.3f}}<br>" # Display the Y value
        ]
        hovertemplate_parts.append("<b>Cell ID:</b> %{text}<br>")
        point_text_data = g_df['cell_id'] # Assign the series to 'text'
        point_customdata = g_df['image_name']
        # Add the corresponding part to the hovertemplate, referencing customdata
        hovertemplate_parts.append("<b>Image:</b> %{customdata}<br>")
        hovertemplate_parts.append("<extra></extra>") # Hide the default trace info box
        final_hovertemplate = "".join(hovertemplate_parts)

        # --- Add the go.Box Trace ---
        fig.add_trace(go.Box(
            # Core data and category assignment
            y=g_df[selected_var],       # Y values for this group
            name=group,                 # Assigns to category & legend label
            # Point display settings
            boxpoints='all',            # Show all individual points (strip/swarm)
            jitter=jitter_amount,       # Control horizontal spread of points
            pointpos=0,                 # Center points horizontally in category space

            # Styling for the individual points (marker)
            marker=dict(
                color=color_map[group], # Use pre-defined color for this group
                size=point_size,
                opacity=0.8,            # Slight transparency helps with dense areas
                line=dict(width=0.5, color='DarkSlateGrey') # Optional outline
            ),

            # Make the actual box plot elements invisible
            fillcolor='rgba(0,0,0,0)',  # Transparent fill
            line_color='rgba(0,0,0,0)', # Transparent box outline
            # --- Hover Info for Points ---
            # Assign the prepared data arrays/series
            text=point_text_data,       # Data referenced by %{text} in template
            customdata=point_customdata,# Data referenced by %{customdata}
            # Assign the dynamically built hovertemplate string
            hovertemplate=final_hovertemplate
        ))
    fig.update_layout(
        title=f'Distribution of {selected_var} by {", ".join(compared_by)}',
        xaxis_title=', '.join(compared_by),
        yaxis_title=selected_var,
        showlegend=True, # Show legend entries based on the 'name' of each go.Box trace
        hovermode='closest', # Hover behavior
      #  template='plotly_white',
        margin=dict(l=50, r=20, t=50, b=max(80, len(max(compare_groups, key=len, default=''))*5)), # Adjust bottom margin
        # Ensure boxplot elements like mean lines or whiskers are not shown if they somehow sneak through
        # (though transparent colors should be sufficient)
       # boxmode='group' 
    )

    # --- 4. Add statistical annotations ---
    if compare_pairs != [] and stats_test != "None":
        pair_chose = st.multiselect("Select statistical tests compare pairs", compare_pairs, default=[compare_pairs[0]],key="compare_pairs")
        if pair_chose != []:
            print(pair_chose)

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