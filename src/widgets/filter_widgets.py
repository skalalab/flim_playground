import streamlit as st
from src.feature_types import categorical_cols

# Generic callback function to handle "All" logic
def update_multiselect(key, options):
    current_selection = st.session_state[key]
    if len(current_selection) > 1:
        if "All" in current_selection[-1]:
            st.session_state[key] = ["All"]
        else:
            st.session_state[key] = [option for option in current_selection if option != "All"]


def filters_widget(df):
    filtered_df = df.copy()
    categories_to_filter = [category for category in categorical_cols if category in df.columns and df[category].nunique() > 1]
    if len(categories_to_filter) > 0:
        cols = st.columns(len(categories_to_filter))

    # Track selections for each filter
    for i, category in enumerate(categories_to_filter):
        with cols[i]:
            unique_values = filtered_df[category].unique().tolist()
            try:
                unique_values = sorted(unique_values, key=lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)
            except Exception:
                unique_values = sorted(unique_values)
            unique_values.append("All")

            key = f"{category}_multiselect"
            # If current selection is not in new options, reset to first available
            current_selection = st.session_state.get(key, [unique_values[0]])
            # Remove any values not in unique_values
            valid_selection = [v for v in current_selection if v in unique_values]
            if not valid_selection:
                valid_selection = [unique_values[0]]
            # If "All" is in options and nothing is selected, default to "All"
            if "All" in unique_values and not valid_selection:
                valid_selection = ["All"]
            # Update session state if needed
            if st.session_state.get(key) != valid_selection:
                st.session_state[key] = valid_selection

            selected_values = st.multiselect(
                f"Select {category}(s)",
                unique_values,
                default=st.session_state[key],
                key=key,
                on_change=update_multiselect, args=(key, unique_values))

            # Filter the dataframe based on the selected values
            if "All" in selected_values:
                pass
            else:
                filtered_df = filtered_df[filtered_df[category].isin(selected_values)]
    return filtered_df