import streamlit as st
from src.vis.helpers import natural_tuple_sort
from src.widgets.analysis_widget_state import number_input_default
from src.widgets.multiselect_modes import (
    ALL_LABEL,
    EXCEPT_LABEL,
    chosen_items,
    excluded_items,
    normalize_mode_selection,
)
import pandas as pd


def selection_key(category):
    return f"{category}_multiselect"


def describe_selection(category, stored):
    """Describe a stored selection, listing excluded values for an "Except:" filter."""
    excluded = excluded_items(stored)
    if excluded is not None:
        return f"{category} ≠ {', '.join(map(str, excluded))}"
    return f"{category} = {', '.join(map(str, stored))}"


def reachable_values(df, categories, selections, target):
    """Return target values reachable under all other categories' selections."""
    subset = df
    for other in categories:
        if other == target:
            continue
        selected = selections.get(other, [ALL_LABEL])
        if ALL_LABEL not in selected:
            subset = subset[subset[other].isin(selected)]
    return set(subset[target].unique().tolist())


def resolve_selections(df, categories, selections, exempt=()):
    """Resolve mutually constrained filters until no selections need pruning.

    Each pass uses one snapshot for every option set, independent of category order.
    Selections only shrink or fall back to "All". Exempt categories retain their
    selection while constraining the others; exclusion filters use this because
    their effective values are recomputed from the data each run.

    Return (resolved selections, options per category, dropped values per category).
    """
    resolved = {category: list(selections.get(category, [ALL_LABEL])) for category in categories}
    dropped = {category: [] for category in categories}

    # Clear values absent from the data before resolving cross-filter constraints.
    for category in categories:
        present = set(df[category].unique().tolist())
        if ALL_LABEL in resolved[category] or category in exempt:
            continue
        stale = [value for value in resolved[category] if value not in present]
        if stale:
            kept = [value for value in resolved[category] if value in present]
            resolved[category] = kept or [ALL_LABEL]

    for _ in range(len(categories) + 1):
        options = {category: reachable_values(df, categories, resolved, category) for category in categories}
        changed = False
        for category in categories:
            if ALL_LABEL in resolved[category] or category in exempt:
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
        # Snapshot all selections before rendering. Widgets store mode sentinels;
        # cross-filtering uses effective values expanded from the current data.
        present_values, stored_selections, effective_selections = {}, {}, {}
        for category in categories_to_filter:
            values = df[category].unique().tolist()
            present_values[category] = natural_tuple_sort(values, delimiter='_')

            stored = list(st.session_state.get(selection_key(category), [ALL_LABEL]))
            excluded = excluded_items(stored)
            if excluded is not None:
                # Drop vanished values while keeping the exclusion sentinel first.
                present = set(values)
                stored = [EXCEPT_LABEL, *(value for value in excluded if value in present)]
            stored_selections[category] = stored

            chosen = chosen_items(stored, present_values[category])
            effective_selections[category] = [ALL_LABEL] if chosen is None else chosen

        exclude_mode = {
            category for category in categories_to_filter
            if excluded_items(stored_selections[category]) is not None
        }
        resolved_selections, reachable, dropped = resolve_selections(
            df, categories_to_filter, effective_selections, exempt=exclude_mode
        )

        cols = st.columns(len(categories_to_filter))
        for i, category in enumerate(categories_to_filter):
            with cols[i]:
                offered = reachable[category]
                if category in exclude_mode:
                    # Keep excluded values offered even when unreachable, preserving
                    # the user's exclusions and a valid keyed widget selection.
                    offered = offered | set(excluded_items(stored_selections[category]))
                unique_values_for_current_filter = natural_tuple_sort(list(offered), delimiter='_')
                unique_values_for_current_filter.extend([ALL_LABEL, EXCEPT_LABEL])

                key = selection_key(category)

                # Both stored exclusions and resolved inclusions fit the current options.
                desired = stored_selections[category] if category in exclude_mode else resolved_selections[category]
                if st.session_state.get(key) != desired:
                    st.session_state[key] = desired

                st.multiselect(
                    f"Select {category}(s)",
                    unique_values_for_current_filter,
                    key=key,
                    on_change=normalize_mode_selection,
                    args=(key,),
                    help=f'Pick the {category}s to keep, or pick "{EXCEPT_LABEL}" together with '
                         'the ones to drop to keep everything else.',
                )

        # Explain deselections caused by incompatible filters.
        explained, unexplained = [], []
        for category, values in dropped.items():
            if not values:
                continue
            causes = [
                describe_selection(other, stored_selections[other])
                for other in categories_to_filter
                if other != category and ALL_LABEL not in resolved_selections[other]
            ]
            deselected = f"**{category}** = {', '.join(map(str, values))}"
            # A fallback to "All" can restore reachability. Name individual causes
            # only when the value remains unreachable under the final constraints.
            if causes and all(value not in reachable[category] for value in values):
                explained.append(f"{deselected} (no rows with {' and '.join(causes)})")
            else:
                unexplained.append(deselected)
        notices = explained
        if unexplained:
            notices.append(f"{' and '.join(unexplained)} (no rows in that combination)")
        if notices:
            st.markdown(f":primary[Deselected {'; '.join(notices)}.]")

        # Explain when an exclusion filter removes every row.
        emptied = [
            category for category in categories_to_filter
            if category in exclude_mode and not resolved_selections[category]
        ]
        if emptied:
            st.warning(f"Every value of {', '.join(emptied)} is excluded, so no rows are left.")

        # Apply the intersection of resolved selections, including expanded exclusions.
        for category in categories_to_filter:
            selected_values = resolved_selections[category]
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
                    value=number_input_default(st.session_state, threshold_key, f_mean),
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
