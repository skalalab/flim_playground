import streamlit as st

def reset_other_menus(selected_menu, menus):
    selected_value = st.session_state[selected_menu]
    if selected_value != "Select":  # Only reset if the selection is not "Select"
        for menu in menus:
            if menu != selected_menu:
                st.session_state[menu] = "Select"
        st.session_state.selected_menu = selected_menu

# create selectboxs for variables
def create_singleSelects_vars(feature_cols_dict):
    
    menus = []       
    for feature_group in feature_cols_dict.keys():
        menus.append("menu_" + feature_group)
        
    # Render the dropdowns with callbacks
    # Create columns based on the number of non-empty feature groups
    cols = st.columns(len(menus))
    selected_var = "Select"
    
    # Dynamically create selectboxes for each feature group
    for i, feature_group in enumerate(feature_cols_dict.keys()):
        menu_key = f"menu_{feature_group}"
        feature_list = feature_cols_dict[feature_group]
        
        with cols[i]:
            if len(feature_list) > 0:
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
    return selected_var


def create_multiSelects_vars(nadh_cols, fad_cols, morphology_cols, columns=False):
    vars = [nadh_cols, fad_cols, morphology_cols]
    var_names = ["NADH", "FAD", "Morphology"]
    # Filter out empty lists
    non_empty_vars = [(name, lst) for name, lst in zip(var_names, vars) if len(lst) > 0]
    
    # Initialize selected items
    selected_items = [None] * len(vars)
    if columns:
        cols = st.columns(len(non_empty_vars))
        for col, (name, var_list) in zip(cols, non_empty_vars):
            with col:
                selected_items[var_names.index(name)] = st.multiselect(f"Select from {name} Variables", 
                                options= [f"All {name} Variables"] + var_list if len(var_list) > 1 else var_list,
                                default=[f"All {name} Variables"],
                                help=f"Select one or more columns corresponding to {name} variables."
                                )
    else:
        for name, var_list in non_empty_vars:
            selected_items[var_names.index(name)] = st.multiselect(f"Select from {name} Variables", 
                            options= [f"All {name} Variables"] + var_list if len(var_list) > 1 else var_list,
                            default=[f"All {name} Variables"],
                            help=f"Select one or more columns corresponding to {name} variables."
                            )
    # if nadh_cols != []:
    #     nadh_vars = st.multiselect(
    #         "Select NADH Variables",
    #         options= ["All NADH Variables"] + nadh_cols if len(nadh_cols) > 0 else nadh_cols,
    #         default=["All NADH Variables"],
    #         help="Select one or more columns corresponding to NADH variables."
    #     )
    # else:
    #     nadh_vars = []
    # if fad_cols != []:
    #     fad_vars = st.multiselect(
    #         "Select FAD Variables",
    #         options= ["All FAD Variables"] + fad_cols if len(fad_cols) > 0 else fad_cols,
    #         default=["All FAD Variables"],
    #         help="Select one or more columns corresponding to FAD variables."
    #     )
    # else:
    #     fad_vars = []
    # if morphology_cols != []:
    #     morphology_vars = st.multiselect(
    #         "Select Morphology Variables",
    #         options= ["All Morphology Variables"] + morphology_cols if len(morphology_cols) > 0 else morphology_cols,
    #         default=["All Morphology Variables"],
    #         help="Select one or more columns corresponding to morphology variables."
    #     )
    # else:
    #     morphology_vars = []
    selected_items = (selected if selected is not None else [] for selected in selected_items)
    nadh_vars, fad_vars, morphology_vars = selected_items

    return nadh_vars, fad_vars, morphology_vars

def ensure_exclusive_images():
    # If user tries to uncheck "Remove Images", make sure "Remove Cells" is checked
    if not st.session_state.remove_images:
        st.session_state.remove_cells = True
    else:
        # If "Remove Images" is checked, ensure "Remove Cells" is unchecked
        st.session_state.remove_cells = False

def ensure_exclusive_cells():
    # If user tries to uncheck "Remove Cells", make sure "Remove Images" is checked
    if not st.session_state.remove_cells:
        st.session_state.remove_images = True
    else:
        # If "Remove Cells" is checked, ensure "Remove Images" is unchecked
        st.session_state.remove_images = False
    
def create_checkboxes():
    col1, col2 = st.columns([0.3, 1])
    with col1:
        st.checkbox(
            "Remove Images",
            key="remove_images",
            on_change=ensure_exclusive_images
        )

    with col2:
        st.checkbox(
            "Remove Cells",
            key="remove_cells",
            on_change=ensure_exclusive_cells
        )

    return col1, col2

def create_umap_hyperParams():
    col1, col2 = st.columns(2)
    # First number incrementor in the first column
    with col1:
        n_neighbors = st.number_input(
            "n_neighbors",
            value=15,  # Initial value
            step=5,             # Increment/Decrement step
            format="%d"            # Integer format
        )

    # Second number incrementor in the second column
    with col2:
        min_dist = st.number_input(
            "min_dist",
            value=0.1,  # Initial value
            step=0.1,            
        )

    return n_neighbors, min_dist