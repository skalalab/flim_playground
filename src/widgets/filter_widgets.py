import streamlit as st
from src.vis.helpers import natural_tuple_sort
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
    """The stored selection as text, for the deselection notice.

    Formatted from the stored (sentinel) form rather than the resolved one so an "Except:"
    filter reads as the one value it drops instead of listing every value it keeps.
    """
    excluded = excluded_items(stored)
    if excluded is not None:
        return f"{category} ≠ {', '.join(map(str, excluded))}"
    return f"{category} = {', '.join(map(str, stored))}"


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


def resolve_selections(df, categories, selections, exempt=()):
    """
    Prune the stored selections until each one only holds values that are reachable given
    all the others, then report the option list for every filter.

    Symmetric narrowing means a change to one filter can strip support from another
    filter's selection, and that prune can in turn narrow a third, so this iterates to a
    fixpoint. Every pass recomputes all option sets from the same snapshot and applies the
    prunes together, so the result does not depend on the order of `categories`. Passes
    only ever shrink a selection or fall back to "All" (which is then left alone), so the
    loop terminates.

    Categories in `exempt` keep their selection verbatim. An "Except:" selection is
    re-derived from the data on every rerun, so it holds nothing stale to prune, and
    pruning it would report values the user never picked as deselected. They still narrow
    every other category through `reachable_values`, which is what keeps a filter that
    drops a value indistinguishable from one that never offered it.

    Returns (resolved selections, options per category, values dropped per category).
    """
    resolved = {category: list(selections.get(category, [ALL_LABEL])) for category in categories}
    dropped = {category: [] for category in categories}

    # Values that vanished from the data entirely (a newly loaded csv, a profile switch)
    # are not a cross-filtering question, so clear them before the fixpoint runs.
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
        # Read every selection before rendering anything: each filter's options depend on
        # the others, so they all come from one consistent snapshot. Each filter keeps two
        # forms — the stored one the widget shows, which may carry the "Except:" sentinel,
        # and the effective one the cross-filtering uses, always a plain list of values
        # (or ["All"] for no constraint). The effective form is re-derived every rerun, so
        # "all except X" tracks the data instead of freezing into a fixed list.
        present_values, stored_selections, effective_selections = {}, {}, {}
        for category in categories_to_filter:
            values = df[category].unique().tolist()
            present_values[category] = natural_tuple_sort(values, delimiter='_')

            stored = list(st.session_state.get(selection_key(category), [ALL_LABEL]))
            excluded = excluded_items(stored)
            if excluded is not None:
                # Same reasoning as the stale pass in resolve_selections: a value that has
                # vanished from the data is not a cross-filtering question. Rebuilt rather
                # than filtered in place so the sentinel keeps leading the chips.
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
                    # Narrowed by the other filters like any other filter, but already-
                    # excluded values stay on the list even once unreachable: dropping them
                    # would hand the widget a session-state value it does not offer
                    # (Streamlit raises) and forget an exclusion the user still wants.
                    offered = offered | set(excluded_items(stored_selections[category]))
                unique_values_for_current_filter = natural_tuple_sort(list(offered), delimiter='_')
                unique_values_for_current_filter.extend([ALL_LABEL, EXCEPT_LABEL])

                key = selection_key(category)

                # Both forms are subsets of the options built just above -- the stored one
                # by construction, the resolved one by the fixpoint -- so writing it back
                # cannot hand the widget a value it does not offer.
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

        # Surface anything the fixpoint had to drop, so a selection is never lost silently.
        # Each value is reported with the selections that eliminated it, since the whole
        # point of symmetric filtering is that another filter caused this.
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

        # Excluding every value is a click away once the sentinel is on, and the generic
        # "no data after filtering" message downstream would not say why.
        emptied = [
            category for category in categories_to_filter
            if category in exclude_mode and not resolved_selections[category]
        ]
        if emptied:
            st.warning(f"Every value of {', '.join(emptied)} is excluded, so no rows are left.")

        # Apply every selection to an unfiltered copy. This is a plain conjunction of
        # masks, so the resulting rows do not depend on the order of categorical_cols.
        # Taken from the resolved selections rather than re-read from session state: the
        # two agree for a plain selection, but only the resolved form carries an "Except:"
        # filter already expanded into the values it keeps.
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