import streamlit as st
"""
This module contains functions to create single and multiple selection widgets. 
"""

def expand_all_selection(selected, feature_list):
    """Expand the sentinel 'All' selection to the full feature list."""
    return list(feature_list) if "All" in selected else list(selected)


def normalize_feature_multiselect_state(current_selection, feature_list):
    """Normalize stored selections so the widget keeps the app's canonical 'All' behavior."""
    if len(feature_list) <= 1:
        if not feature_list:
            return []
        valid = [value for value in (current_selection or []) if value in feature_list]
        return valid or list(feature_list)

    options = ["All"] + list(feature_list)
    valid_selection = [value for value in (current_selection or []) if value in options]
    if not valid_selection:
        return ["All"]
    if "All" in valid_selection and len(valid_selection) > 1:
        if valid_selection[-1] == "All":
            return ["All"]
        valid_selection = [value for value in valid_selection if value != "All"]
    if len(valid_selection) == len(feature_list):
        return ["All"]
    return valid_selection


def feature_multiselect_widget(label, feature_list, key, help=None):
    """Shared multiselect with the app's standard 'All' behavior."""
    if len(feature_list) > 1:
        options = ["All"] + list(feature_list)
        default = ["All"]
    else:
        options = list(feature_list)
        default = list(feature_list)

    current_selection = st.session_state.get(key, default)
    normalized_selection = normalize_feature_multiselect_state(current_selection, feature_list)
    if current_selection != normalized_selection:
        st.session_state[key] = normalized_selection

    selected = st.multiselect(
        label,
        options=options,
        key=key,
        on_change=update_multiselect_feature,
        args=(key, options),
        help=help,
    )
    return expand_all_selection(selected, feature_list)


def update_multiselect_feature(key, options):
    """Callback function to handle "All" logic for feature selection widgets"""
    current_selection = st.session_state.get(key, ["All"])
    if len(current_selection) > 1:
        if "All" in current_selection[-1]:
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
            feature_list = feature_groups_dict[feature_group]
            # strip the extractor_channel from the feature_list
            if data_extraction:
                feature_list = [col.split(": ")[1] for col in feature_list] if "Uncategorized" not in feature_group else feature_list
            with cols[i]:
                current_selection = st.selectbox(
                    f"{feature_group}", 
                    ["Select"] + feature_list, 
                    index=0, 
                    key=menu_key,
                    on_change=reset_other_menus, 
                    args=(menu_key, menus)
                )
                
                # If this menu has a non-Select value, it becomes our selected_var
                if current_selection != "Select":
                    selected_var = current_selection
                    if data_extraction:
                        selected_var = f"{feature_group}: {selected_var}" if "Uncategorized" not in feature_group else selected_var
    
    return selected_var

def twod_single_feature_select_widget(feature_groups_dict, data_extraction=True, n_per_row=2):
    st.write("**Select the x-axis feature:** ")
    selected_x = single_feature_select_widget(feature_groups_dict, data_extraction=data_extraction, n_per_row=n_per_row, key_prefix="2d_x")
    # remove the selected_x from the feature_groups_dict
    # selected_x is not a feature group, it is a feature
    for feature_group in feature_groups_dict.keys():
        if selected_x in feature_groups_dict[feature_group]:
            feature_groups_dict[feature_group].remove(selected_x)
    if selected_x != "Select":
        st.write("**Select the y-axis feature:** ")
        selected_y = single_feature_select_widget(feature_groups_dict, data_extraction=data_extraction, n_per_row=n_per_row, key_prefix="2d_y")
    else:
        selected_y = "Select"
    if selected_x != "Select" and selected_y != "Select":
        st.info(f"Selected features: **{selected_x}** and **{selected_y}**")
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
            feature_list = feature_groups_dict[feature_group]
            if data_extraction:
            # strip the extractor_channel from the feature_list
                feature_list = [col.split(": ")[1] for col in feature_list] if "Uncategorized" not in feature_group else feature_list
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
                if "All" in selected:
                    expanded = expand_all_selection(selected, feature_list)
                    if data_extraction:
                        selected_features.extend([f"{feature_group}: {col}" for col in expanded]) if "Uncategorized" not in feature_group else selected_features.extend(expanded)
                    else:
                        selected_features.extend(expanded)
                else:
                    if data_extraction:
                        selected_features.extend([f"{feature_group}: {col}" for col in selected]) if "Uncategorized" not in feature_group else selected_features.extend(selected)
                    else:
                        selected_features.extend(selected)
               
    return selected_features
