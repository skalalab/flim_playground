import streamlit as st
import pandas as pd
from feature_groups import categorical_cols

# Generic callback function to handle "All" logic
def update_multiselect(key, options):
    # Get the current selection from session state
    current_selection = st.session_state[key]
    # If "All" is selected, clear all other selections
    if len(current_selection) > 1:
        # all is just selected
        if "All" in current_selection[-1]:
            st.session_state[key] = ["All"]
        else: 
            # all is selected with other options
            st.session_state[key] = [option for option in current_selection if option != "All"]


def create_filters(df, color=True, compare=False): 
    """
    color: True if we want to color or compare the plot by a categorical variable
    compare: True if we want to compare the plot by a categorical variable
    """
    # Initially, filtered_df is the original df
    filtered_df = df.copy()
    # Keep track of which categories need filter: has more than 1 unique value
    categories_to_filter = [category for category in categorical_cols if category in df.columns and df[category].nunique() > 1]
    
    if len(categories_to_filter) > 0:
        if color or compare:
            cols = st.columns(len(categories_to_filter) + 1) # +1 for color_by/compare_by
        else:
            cols = st.columns(len(categories_to_filter))

    for i, category in enumerate(categories_to_filter):
        with cols[i]:
            # Create a multiselect for each category
            # Try to convert to numeric for sorting if the column contains numbers as strings
            unique_values = df[category].unique().tolist()
            try:
                # Check if the values can be converted to numeric
                # if it can, sort them in their numeric order
                # Values are still in their original string format since we only used float() for the sort key
                unique_values = sorted(unique_values, key=lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)
                
            except Exception as e:
                # If conversion fails, just do regular string sorting
                unique_values = sorted(unique_values)
            # Add "All" option to the list of unique values
            unique_values.append("All")

            key = f"{category}_multiselect"
            # Initialize session state for this key if not already set
            if key not in st.session_state:
                st.session_state[key] = [unique_values[0]]
            selected_values = st.multiselect(
                f"Select {category}(s)", 
                unique_values, 
                key=key, 
                on_change=update_multiselect, args=(key, unique_values))
            
            # Filter the dataframe based on the selected values
            if "All" in selected_values:
                # If "All" is selected, keep all rows
                pass
            else:
                # Otherwise, filter the dataframe
                filtered_df = filtered_df[filtered_df[category].isin(selected_values)] 
            
    if color or compare:
        selectText = "Compare by" if compare else "Color by"
        with cols[-1]:
            colo_compare_by_options = st.multiselect(selectText, categories_to_filter, default=categories_to_filter[-1])                   
    else:
        colo_compare_by_options = []

    return filtered_df, colo_compare_by_options, cols