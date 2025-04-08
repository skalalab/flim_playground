import streamlit as st

"""
This file contains widgets that are specific to certain visualization or classification methods."""

def umap_hyperParams_widget():
    col1, col2 = st.columns(2)
    # First number incrementor in the first column
    with col1:
        n_neighbors = st.number_input(
            "n_neighbors",
            value=15,  # Initial value
            step=5,             # Increment/Decrement step
            format="%d"            # Integer format
        )

    # Second number incrementor in the second column
    with col2:
        min_dist = st.number_input(
            "min_dist",
            value=0.1,  # Initial value
            step=0.1,            
        )

    return n_neighbors, min_dist

def stats_comparison_pair_widget(available_pairs):
    selected_pairs = st.multiselect(
        "Select statistical tests compare pairs",
        available_pairs,
        default=available_pairs,
        key="compare_pairs"
    )
    return selected_pairs