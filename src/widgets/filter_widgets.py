import streamlit as st
from src.vis.helpers import natural_tuple_sort
# Generic callback function to handle "All" logic
def update_multiselect(key, options):
    current_selection = st.session_state[key]
    if len(current_selection) > 1:
        if "All" in current_selection[-1]:
            st.session_state[key] = ["All"]
        else:
            st.session_state[key] = [option for option in current_selection if option != "All"]

def filters_widget(df, categorical_cols):
    categories_to_filter = [category for category in categorical_cols if category in df.columns and df[category].nunique() > 1]
    
    if not categories_to_filter:
        return df.copy()

    cols = st.columns(len(categories_to_filter))
    
    # This dataframe is progressively filtered to determine the options for subsequent filters.
    options_df = df.copy()
    
    # This dataframe is filtered at the end based on all selections.
    final_filtered_df = df.copy()

    for i, category in enumerate(categories_to_filter):
        with cols[i]:
            # Use the progressively filtered dataframe to get unique values for the current filter
            unique_values_for_current_filter = options_df[category].unique().tolist()
            unique_values_for_current_filter = natural_tuple_sort(unique_values_for_current_filter, delimiter='_')
            unique_values_for_current_filter.append("All")

            key = f"{category}_multiselect"
            
            # Get current selection from session state, defaulting to "All"
            current_selection = st.session_state.get(key, ["All"])

            # Ensure that the current selection is valid given the available options
            valid_selection = [v for v in current_selection if v in unique_values_for_current_filter]
            if not valid_selection:
                valid_selection = ["All"]
            
            # If the selection in the session state is not valid, update it before rendering the widget.
            if st.session_state.get(key) != valid_selection:
                st.session_state[key] = valid_selection

            selected_values = st.multiselect(
                f"Select {category}(s)",
                unique_values_for_current_filter,
                key=key,
                on_change=update_multiselect,
                args=(key, unique_values_for_current_filter),
            )

            # Progressively filter the dataframe for determining the next filter's options.
            if "All" not in selected_values:
                options_df = options_df[options_df[category].isin(selected_values)]

    # After creating all widgets, filter the original dataframe based on all selections.
    for category in categories_to_filter:
        key = f"{category}_multiselect"
        selected_values = st.session_state.get(key, ["All"])
        if "All" not in selected_values:
            final_filtered_df = final_filtered_df[final_filtered_df[category].isin(selected_values)]
            
    return final_filtered_df