import streamlit as st
"""
This module contains functions to create single and multiple selection widgets. 
"""

def update_multiselect_feature(key, options):
    """Callback function to handle "All" logic for feature selection widgets"""
    current_selection = st.session_state.get(key, ["All"])
    if len(current_selection) > 1:
        if current_selection[-1] == "All":
            st.session_state[key] = ["All"]
        else:
            st.session_state[key] = [option for option in current_selection if option != "All"]

def reset_other_menus(selected_menu, menus):
    selected_value = st.session_state.get(selected_menu, "Select")
    if selected_value != "Select":  # Only reset if the selection is not "Select"
        for menu in menus:
            if menu != selected_menu:
                st.session_state[menu] = "Select"
        st.session_state.selected_menu = selected_menu


def feature_display_to_column(feature_list, feature_group, data_extraction=True):
    """Map each picker-displayed name back to its real DataFrame column.

    For Data-Extraction columns the picker shows only the part after ``": "`` (e.g.
    "t1" for "Lifetime fit_nadh: t1"), so a selection must be resolved back to the
    full column name. Crucially this keys off each *column itself*, not the group
    name — so groups whose friendly name differs from the column prefix round-trip
    correctly. The cross-channel **"Derived Features"** group is exactly such a case:
    its columns are ``"Derived: <name>"``, so the old ``f"{group}: {name}"`` rebuild
    produced the non-existent ``"Derived Features: <name>"`` and raised KeyError.

    Returns an ordered ``{display_name: column}`` dict. Uncategorized columns and the
    non-Data-Extraction path are shown/returned verbatim (identity mapping).
    """
    if data_extraction and "Uncategorized" not in feature_group:
        return {col.split(": ", 1)[1]: col for col in feature_list}
    return {col: col for col in feature_list}


def resolve_pending_selection(feature_groups_dict, key_prefix, data_extraction=True, session_state=None):
    """Resolve an axis's current pick from session state, before its widgets render.

    ``single_feature_select_widget`` keys each group's selectbox
    ``f"{key_prefix}_menu_{group}"``, so the previous run's choice is readable
    ahead of the widget. ``twod_single_feature_select_widget`` needs it that
    early to decide whether to draw the plain grid or a collapsed expander.

    A stored display value is only honoured while it is still present in its
    group's *current* option list. That matters because the x-axis pick is
    removed from ``feature_groups_dict`` before the y-axis grid renders: when the
    user re-opens x and steals the feature y was showing, Streamlit silently
    resets y's selectbox to "Select". Validating here keeps the container and its
    label agreeing with the value the widget will actually return.

    Returns the full DataFrame column name, or "Select" when the axis has no
    valid selection.
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

# create selectboxs for variables
def single_feature_select_widget(feature_groups_dict, data_extraction=True, n_per_row=2, key_prefix=""):
    """
    n_per_row: number of selectboxs in a row"""
    
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
            # Displayed name -> real column. Resolving the selection through this map
            # (instead of rebuilding "{group}: {name}") round-trips groups whose
            # friendly name differs from the column prefix, e.g. the "Derived
            # Features" group whose columns are "Derived: <name>".
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
    """Render one axis's feature picker.

    With no pick yet, this is the plain prompt + grid of per-group selectboxes.
    Once a feature is chosen the same grid moves inside a collapsed expander
    labelled with the full column name, freeing the vertical space the grid ate.

    The collapse relies on Streamlit *remounting* the element rather than on
    flipping ``expanded``, which only ever initialises an expander's state
    (Streamlit 1.54) and so cannot be trusted to close one that already exists.
    Swapping a markdown+columns block for an expandable block mounts a fresh
    expander, initialised collapsed. Re-opening it and changing the pick then
    re-collapses it on the rerun, showing the new value — verified on 1.54, and
    the behaviour we want either way, so nothing here depends on whether a given
    frontend preserves the user's toggle.

    Returns the axis's selected column, or "Select".
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

    # Remove the x pick so it cannot also be chosen for y. Must happen after the
    # x grid renders (it still needs the feature in its own options) and before
    # the y grid does.
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
    # feature groups that have one or more features available for selection 
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
            # Displayed name -> real column, so selections resolve back to actual
            # columns even when the group's friendly name differs from the column
            # prefix (e.g. "Derived Features" -> "Derived: <name>").
            display_to_col = feature_display_to_column(
                feature_groups_dict[feature_group], feature_group, data_extraction)
            feature_list = list(display_to_col.keys())
            key = f"ms_{feature_group}"

            with cols[i]:
                if len(feature_list) > 1:
                    options = ["All"] + feature_list
                    default = ["All"] if feature_group != "Uncategorized Features" else []
                else:
                    options = feature_list
                    default = feature_list
                if key not in st.session_state:
                    st.session_state[key] = default
                # use update_multiselect_feature to handle the "All" logic: if "All" is selected, clear all other selections
                # if "All" is in selected list, and other options are selected, remove "All"
                selected = st.multiselect(
                    f"{feature_group}",
                    options=options,
                    #default=st.session_state[key],
                    key=key,
                    on_change=update_multiselect_feature,
                    args=(key, options),
                    help=f"Select one or more columns corresponding to {feature_group} features."
                )
                chosen = feature_list if "All" in selected else selected
                selected_features.extend(display_to_col[name] for name in chosen)
               
    return selected_features
