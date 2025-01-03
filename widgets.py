import streamlit as st

# Generic callback function to handle "All" logic
def update_multiselect(key, options):
    # Get the current selection from session state
    current_selection = st.session_state[key]
    # If "All" is selected, clear all other selections
    if len(current_selection) > 1:
        # all is just selected
        if "All" in current_selection[-1]:
            st.session_state[key] = ["All"]
        else: 
            # all is selected with other options
            st.session_state[key] = [option for option in current_selection if option != "All"]

def create_filters(df, color=True, compare=False): 
        # Check for existence of columns
        exp_day_exists = "experiment" in df.columns or "day" in df.columns
        cl_exists = "cell_line" in df.columns
        tr_category_exists = "treatment" in df.columns or "color_category" in df.columns
        # Initially, filtered_df is the original df
        filtered_df = df.copy()
        # Keep track of which columns are available for color_by
        available_for_color = []
        ### Handle "experiment" column ###
        cols = st.columns(4)

        if exp_day_exists:
            if "experiment" in df.columns:
                column = "experiment"
            else:
                column = "day"

            values = sorted(df[column].unique().tolist())
            if len(values) > 1:
                values.append("All")  # Add "all" option
                with cols[0]:
                    selected_experiment = st.multiselect(f"Select {column}(s)", values, default=values[0], key="experiment_day_multiselect",on_change=update_multiselect, args=("experiment_day_multiselect", values))
                if "All" not in selected_experiment:
                    filtered_df = filtered_df[filtered_df[column].isin(selected_experiment)]
                available_for_color.append(column)


        ### Handle "cell_line" column ###
        if cl_exists:
            # Based on current filtered_df (which may or may not be filtered by experiment)
            cell_lines = sorted(filtered_df["cell_line"].unique().tolist())
            if len(cell_lines) > 1:
                cell_lines.append("All")
                with cols[1]:
                    selected_cell_lines = st.multiselect("Select cell line(s)", cell_lines, default=cell_lines[0], key="cell_line_multiselect",on_change=update_multiselect, args=("cell_line_multiselect", cell_lines))
                if "All" not in selected_cell_lines:
                    filtered_df = filtered_df[filtered_df["cell_line"].isin(selected_cell_lines)]
                available_for_color.append("cell_line")
            
            else:
                # Only one cell line or none
                # No need to show widget if there's only one possible choice
                pass

        ### Handle "treatment" column ###
        if tr_category_exists:
            # Based on the current filtered_df (which may be filtered by experiment and/or cell line)
            if "color_category" in df.columns:
                column = "color_category"
            else:
                column = "treatment"
            values = sorted(filtered_df[column].unique().tolist())
            # If more than one treatment, show the widget
            if len(values) > 1:
                values.append("All")
                with cols[2]:
                    selected_treatments = st.multiselect(f"Select {column}(s)", values, default=values[-1], key="tr_cat_multiselect",on_change=update_multiselect, args=("tr_cat_multiselect", values))
                if "All" not in selected_treatments:
                    filtered_df = filtered_df[filtered_df[column].isin(selected_treatments)]
                available_for_color.append(column)
            else:
                # Only one treatment or none
                # No widget needed
                pass

        # If more than one of experiment, cell_line, treatment columns exist, add a color_by multiselect
        # Only include columns that actually exist
       
        if len(available_for_color) > 1 and color is True:
            selectText = "Compare by" if compare else "Color by"
            with cols[3]:
                color_by_options = st.multiselect(selectText, available_for_color, default=available_for_color[-1])                   
        else:
            color_by_options = ["treatment"]

        return filtered_df, color_by_options, cols

def reset_other_menus(selected_menu, menus):
    selected_value = st.session_state[selected_menu]
    if selected_value != "Select":  # Only reset if the selection is not "Select"
        for menu in menus:
            if menu != selected_menu:
                st.session_state[menu] = "Select"
        st.session_state.selected_menu = selected_menu

# create selectboxs for variables
def create_singleSelects_vars(nadh_cols, fad_cols, morphology_cols):
    col1, col2, col3 = st.columns(3)
    menus = ["menu_nadh", "menu_fad", "menu_morphology"]           
    # Render the dropdowns with callbacks
    with col1:
        if len(nadh_cols) > 0:
            selected_nadh = st.selectbox(
                "Nadh Variables", 
                ["Select"] + nadh_cols, 
                index=0, 
                key="menu_nadh",
                on_change=reset_other_menus, 
                args=("menu_nadh",menus)
            )
        else: selected_nadh = "Select"
    with col2:
        if len(fad_cols) > 0: 
            selected_fad = st.selectbox(
                "Fad Variables", 
                ["Select"] + fad_cols, 
                index=0, 
                key="menu_fad",
                on_change=reset_other_menus, 
                args=("menu_fad",menus)
            )
        else: selected_fad = "Select"
    with col3:
        if len(morphology_cols) > 0:         
            selected_morphology = st.selectbox(
                "Morphology", 
                ["Select"] + morphology_cols, 
                index=0, 
                key="menu_morphology",
                on_change=reset_other_menus, 
                args=("menu_morphology",menus)
            )
        else: selected_morphology = "Select"

    selected_var =  selected_nadh if selected_nadh and selected_nadh != "Select" else selected_fad if selected_fad and selected_fad != "Select" else selected_morphology
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