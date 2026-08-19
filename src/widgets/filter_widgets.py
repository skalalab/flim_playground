import streamlit as st
from src.vis.helpers import natural_tuple_sort
import pandas as pd

ALL_LABEL = "All"


def selection_key(category):
    return f"{category}_multiselect"


# Generic callback function to handle "All" logic
def update_multiselect(key):
    current_selection = st.session_state.get(key, [ALL_LABEL])
    if len(current_selection) > 1:
        if current_selection[-1] == ALL_LABEL:
            st.session_state[key] = [ALL_LABEL]
        else:
            st.session_state[key] = [option for option in current_selection if option != ALL_LABEL]


def reachable_values(df, categories, selections, target):
    """
    The values of `target` that still have rows once every OTHER category's selection is
    applied. Excluding the target's own selection is what makes the filters symmetric:
    each one is narrowed by all the others rather than only by the ones to its left, so
    the order of categorical_cols no longer decides which combinations are reachable.
    """
    subset = df
    for other in categories:
        if other == target:
            continue
        selected = selections.get(other, [ALL_LABEL])
        if ALL_LABEL not in selected:
            subset = subset[subset[other].isin(selected)]
    return set(subset[target].unique().tolist())


def resolve_selections(df, categories, selections):
    """
    Prune the stored selections until each one only holds values that are reachable given
    all the others, then report the option list for every filter.

    Symmetric narrowing means a change to one filter can strip support from another
    filter's selection, and that prune can in turn narrow a third, so this iterates to a
    fixpoint. Every pass recomputes all option sets from the same snapshot and applies the
    prunes together, so the result does not depend on the order of `categories`. Passes
    only ever shrink a selection or fall back to "All" (which is then left alone), so the
    loop terminates.

    Returns (resolved selections, options per category, values dropped per category).
    """
    resolved = {category: list(selections.get(category, [ALL_LABEL])) for category in categories}
    dropped = {category: [] for category in categories}

    # Values that vanished from the data entirely (a newly loaded csv, a profile switch)
    # are not a cross-filtering question, so clear them before the fixpoint runs.
    for category in categories:
        present = set(df[category].unique().tolist())
        if ALL_LABEL in resolved[category]:
            continue
        stale = [value for value in resolved[category] if value not in present]
        if stale:
            kept = [value for value in resolved[category] if value in present]
            resolved[category] = kept or [ALL_LABEL]

    for _ in range(len(categories) + 1):
        options = {category: reachable_values(df, categories, resolved, category) for category in categories}
        changed = False
        for category in categories:
            if ALL_LABEL in resolved[category]:
                continue
            unsupported = [value for value in resolved[category] if value not in options[category]]
            if not unsupported:
                continue
            kept = [value for value in resolved[category] if value in options[category]]
            dropped[category].extend(unsupported)
            resolved[category] = kept or [ALL_LABEL]
            changed = True
        if not changed:
            break

    options = {category: reachable_values(df, categories, resolved, category) for category in categories}
    return resolved, options, dropped


def filters_widget(df, categorical_cols):

    # Filtered at the end based on all selections.
    final_filtered_df = df.copy()

    categories_to_filter = [category for category in categorical_cols if category in df.columns and df[category].nunique() > 1]
    if categories_to_filter:
        # Read every selection before rendering anything: each filter's options depend on
        # the other filters, so they all have to come from one consistent snapshot.
        stored_selections = {
            category: st.session_state.get(selection_key(category), [ALL_LABEL])
            for category in categories_to_filter
        }
        resolved_selections, reachable, dropped = resolve_selections(df, categories_to_filter, stored_selections)

        cols = st.columns(len(categories_to_filter))
        for i, category in enumerate(categories_to_filter):
            with cols[i]:
                unique_values_for_current_filter = natural_tuple_sort(list(reachable[category]), delimiter='_')
                unique_values_for_current_filter.append(ALL_LABEL)

                key = selection_key(category)

                # The fixpoint above already guarantees this is a subset of the options,
                # so writing it back cannot hand the widget a value it does not offer.
                if st.session_state.get(key) != resolved_selections[category]:
                    st.session_state[key] = resolved_selections[category]

                st.multiselect(
                    f"Select {category}(s)",
                    unique_values_for_current_filter,
                    key=key,
                    on_change=update_multiselect,
                    args=(key,),
                )

        # Surface anything the fixpoint had to drop, so a selection is never lost silently.
        # Each value is reported with the selections that eliminated it, since the whole
        # point of symmetric filtering is that another filter caused this.
        explained, unexplained = [], []
        for category, values in dropped.items():
            if not values:
                continue
            causes = [
                f"{other} = {', '.join(map(str, resolved_selections[other]))}"
                for other in categories_to_filter
                if other != category and ALL_LABEL not in resolved_selections[other]
            ]
            deselected = f"**{category}** = {', '.join(map(str, values))}"
            # Naming a cause is only honest while the value is still unreachable. A
            # category that fell back to "All" loosens the constraints, so those drops are
            # grouped as one empty combination instead of blamed on a filter.
            if causes and all(value not in reachable[category] for value in values):
                explained.append(f"{deselected} (no rows with {' and '.join(causes)})")
            else:
                unexplained.append(deselected)
        notices = explained
        if unexplained:
            notices.append(f"{' and '.join(unexplained)} (no rows in that combination)")
        if notices:
            # st.markdown rather than st.caption: caption renders at the theme's `sm` size,
            # which is too quiet for a notice that says a selection was taken away.
            st.markdown(f":primary[Deselected {'; '.join(notices)}.]")

        # Apply every selection to an unfiltered copy. This is a plain conjunction of
        # masks, so the resulting rows do not depend on the order of categorical_cols.
        for category in categories_to_filter:
            selected_values = st.session_state.get(selection_key(category), [ALL_LABEL])
            if ALL_LABEL not in selected_values:
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