
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from itertools import combinations
import streamlit as st

def feature_comparison_plot(df, selected_var, compared_by, stats_test="N/A"): 
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
    if compare_pairs != [] and stats_test != "N/A":
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