import streamlit as st
from src.vis.helpers import natural_tuple_sort
import pandas as pd

# Generic callback function to handle "All" logic and cascade resets
def update_multiselect(key, options, categories_to_filter, current_category_index):
    current_selection = st.session_state.get(key, ["All"])
    if len(current_selection) > 1:
        if "All" in current_selection[-1]:
            st.session_state[key] = ["All"]
        else:
            st.session_state[key] = [option for option in current_selection if option != "All"]
    
    # Reset all downstream filters to "All" when this filter changes
    if current_category_index < len(categories_to_filter) - 1:
        for j in range(current_category_index + 1, len(categories_to_filter)):
            downstream_key = f"{categories_to_filter[j]}_multiselect"
            st.session_state[downstream_key] = ["All"]

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
                args=(key, unique_values_for_current_filter, categories_to_filter, i),
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

    numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numerical_cols:
        # Layout for numerical filter controls
        # Use a while loop to allow for multiple filters
        
        i = 0
        while True:
            # Layout for the current filter row
            num_col1, num_col2, num_col3, num_col4 = st.columns([1, 0.5, 1, 0.5])
            
            with num_col1:
                feature = st.selectbox(f"Select Feature {i+1}", ["None"] + numerical_cols, key=f"num_filter_feature_{i}")
            
            if feature == "None":
                break
            
            with num_col2:
                op = st.selectbox("Operator", [">", "<="], key=f"num_filter_operator_{i}_{feature}")
            
            with num_col3:
                if final_filtered_df.empty:
                    st.warning("No data available.")
                    break

                # Determine min/max for the selected feature based on the CURRENT filtered dataframe
                f_min = float(final_filtered_df[feature].min())
                f_max = float(final_filtered_df[feature].max())
                f_mean = float(final_filtered_df[feature].mean())
                
                # Handle case where min == max (single value)
                if f_min == f_max:
                    f_min -= 0.01
                    f_max += 0.01
                
                # Widget key including feature to ensure uniqueness when feature changes
                threshold_key = f"num_filter_threshold_{i}_{feature}"
                
                # Clamp existing session state value to new range if it exists
                if threshold_key in st.session_state:
                    current_val = st.session_state[threshold_key]
                    if current_val < f_min:
                        st.session_state[threshold_key] = f_min
                    elif current_val > f_max:
                        st.session_state[threshold_key] = f_max
                
                thresh = st.number_input(
                    f"Threshold ({f_min:.2f} - {f_max:.2f})", 
                    value=f_mean, 
                    min_value=f_min, 
                    max_value=f_max, 
                    key=threshold_key
                )
            
            # Apply the filter immediately so the next iteration uses the filtered data
            if op == ">":
                final_filtered_df = final_filtered_df[final_filtered_df[feature] > thresh]
            else:
                final_filtered_df = final_filtered_df[final_filtered_df[feature] <= thresh]
            
            with num_col4:
                st.write("") # Spacer
                st.write("")
                add_another = st.checkbox("Add another", key=f"add_another_num_filter_{i}")
            
            if not add_another:
                break
                
            i += 1

    return final_filtered_df