import streamlit as st
from src.widgets.multiselect_modes import (
    ALL_LABEL,
    EXCEPT_LABEL,
    chosen_items,
    normalize_mode_selection,
)
"""
This module contains functions to create single and multiple selection widgets.
"""

def reset_other_menus(selected_menu, menus):
    selected_value = st.session_state.get(selected_menu, "Select")
    if selected_value != "Select":  # Only reset if the selection is not "Select"
        for menu in menus:
            if menu != selected_menu:
                st.session_state[menu] = "Select"
        st.session_state.selected_menu = selected_menu


def feature_display_to_column(feature_list, feature_group, data_extraction=True):
    """Map picker labels to their full DataFrame column names, preserving order.

    Data-extraction labels omit each column's prefix. Resolve them from the
    columns themselves because group names can differ from those prefixes
    (e.g. "Derived Features" contains "Derived: <name>"). Uncategorized
    features and user-table columns retain their full names.
    """
    if data_extraction and "Uncategorized" not in feature_group:
        return {col.split(": ", 1)[1]: col for col in feature_list}
    return {col: col for col in feature_list}


def resolve_pending_selection(feature_groups_dict, key_prefix, data_extraction=True, session_state=None):
    """Return an axis's selected column before rendering, or "Select" if invalid.

    Read each group's keyed selection to choose the grid or expander layout.
    Validate against current options: the x selection is removed before y
    renders, so a saved y selection may no longer be available.
    """
    if session_state is None:
        session_state = st.session_state
    for feature_group, feature_list in feature_groups_dict.items():
        stored = session_state.get(f"{key_prefix}_menu_{feature_group}", "Select")
        if stored == "Select":
            continue
        display_to_col = feature_display_to_column(
            feature_list, feature_group, data_extraction)
        if stored in display_to_col:
            return display_to_col[stored]
    return "Select"

def single_feature_select_widget(feature_groups_dict, data_extraction=True, n_per_row=2, key_prefix=""):
    """Render mutually exclusive feature pickers with ``n_per_row`` groups per row."""

    menus = []
    for feature_group in feature_groups_dict.keys():
        menus.append(f"{key_prefix}_menu_{feature_group}")

    selected_var = "Select"
    feature_groups = list(feature_groups_dict.keys())

    # Calculate number of rows needed
    num_groups = len(feature_groups)
    num_rows = (num_groups + n_per_row - 1) // n_per_row  # Ceiling division

    # Create rows of columns
    for row in range(num_rows):
        start_idx = row * n_per_row
        end_idx = min(start_idx + n_per_row, num_groups)

        # Create columns for this row
        cols = st.columns(end_idx - start_idx)

        # Add menus to this row
        for i, col_idx in enumerate(range(start_idx, end_idx)):
            feature_group = feature_groups[col_idx]
            menu_key = f"{key_prefix}_menu_{feature_group}"
            # Group labels may differ from column prefixes; use the column mapping.
            display_to_col = feature_display_to_column(
                feature_groups_dict[feature_group], feature_group, data_extraction)
            display_list = list(display_to_col.keys())
            with cols[i]:
                current_selection = st.selectbox(
                    f"{feature_group}",
                    ["Select"] + display_list,
                    index=0,
                    key=menu_key,
                    on_change=reset_other_menus,
                    args=(menu_key, menus)
                )

                # If this menu has a non-Select value, it becomes our selected_var
                if current_selection != "Select":
                    selected_var = display_to_col[current_selection]

    return selected_var

def _axis_select_block(feature_groups_dict, axis_name, key_prefix, data_extraction=True, n_per_row=2):
    """Render an axis picker and return its selected column, or "Select".

    Show an unselected grid inline; place a selected grid in an expander named
    for the full column. The expander mounts collapsed: ``expanded`` sets its
    initial state, so a fresh container is needed to close it after a change.
    """
    pending = resolve_pending_selection(feature_groups_dict, key_prefix, data_extraction)

    if pending == "Select":
        st.write(f"**Select the {axis_name}-axis feature:** ")
        return single_feature_select_widget(
            feature_groups_dict, data_extraction=data_extraction,
            n_per_row=n_per_row, key_prefix=key_prefix)

    with st.expander(f"{axis_name.upper()}-axis — {pending}", expanded=False):
        return single_feature_select_widget(
            feature_groups_dict, data_extraction=data_extraction,
            n_per_row=n_per_row, key_prefix=key_prefix)


def twod_single_feature_select_widget(feature_groups_dict, data_extraction=True, n_per_row=2):
    selected_x = _axis_select_block(
        feature_groups_dict, "x", "2d_x", data_extraction, n_per_row)

    # Remove x after its picker renders and before building y options.
    for feature_group in feature_groups_dict.keys():
        if selected_x in feature_groups_dict[feature_group]:
            feature_groups_dict[feature_group].remove(selected_x)

    if selected_x != "Select":
        selected_y = _axis_select_block(
            feature_groups_dict, "y", "2d_y", data_extraction, n_per_row)
    else:
        selected_y = "Select"

    return selected_x, selected_y

def multi_feature_select_widget(feature_groups_dict, data_extraction=True, n_per_row=2):

    selected_features = []
    feature_groups = list(feature_groups_dict.keys())

    # Calculate number of rows needed
    num_groups = len(feature_groups)
    num_rows = (num_groups + n_per_row - 1) // n_per_row  # Ceiling division

    # Create rows of columns
    for row in range(num_rows):
        start_idx = row * n_per_row
        end_idx = min(start_idx + n_per_row, num_groups)

        # Create columns for this row
        cols = st.columns(end_idx - start_idx)

        # Add multiselect widgets to this row
        for i, col_idx in enumerate(range(start_idx, end_idx)):
            feature_group = feature_groups[col_idx]
            # Resolve display labels through the same mapping as the single picker.
            display_to_col = feature_display_to_column(
                feature_groups_dict[feature_group], feature_group, data_extraction)
            feature_list = list(display_to_col.keys())
            key = f"ms_{feature_group}"

            with cols[i]:
                if len(feature_list) > 1:
                    options = [ALL_LABEL, EXCEPT_LABEL] + feature_list
                    default = [ALL_LABEL] if feature_group != "Uncategorized Features" else []
                else:
                    options = feature_list
                    default = feature_list
                if key not in st.session_state:
                    st.session_state[key] = default
                # The callback keeps "All" exclusive and preserves additive exclusions.
                selected = st.multiselect(
                    f"{feature_group}",
                    options=options,
                    key=key,
                    on_change=normalize_mode_selection,
                    args=(key,),
                    help=f'Select one or more columns corresponding to {feature_group} features. '
                         f'Pick "{EXCEPT_LABEL}" together with a few to use all the others.'
                )
                # None selects every feature; [] selects none, including the uncategorized default.
                chosen = chosen_items(selected, feature_list)
                if chosen is None:
                    chosen = feature_list
                selected_features.extend(display_to_col[name] for name in chosen)

    return selected_features
