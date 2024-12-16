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

def create_filters(df, color=True): 
        # Check for existence of columns
        exp_exists = "experiment" in df.columns
        cl_exists = "cell_line" in df.columns
        tr_exists = "treatment" in df.columns
        # Initially, filtered_df is the original df
        filtered_df = df.copy()
        # Keep track of which columns are available for color_by
        available_for_color = []
        ### Handle "experiment" column ###
        cols = st.columns(4)

        if exp_exists:
            experiments = sorted(df["experiment"].unique().tolist())
            if len(experiments) > 1:
                experiments.append("All")  # Add "all" option
                with cols[0]:
                    selected_experiment = st.multiselect("Select experiment", experiments, default=experiments[0], key="experiment_multiselect",on_change=update_multiselect, args=("experiment_multiselect", experiments))
                if selected_experiment != "All":
                    filtered_df = filtered_df[filtered_df["experiment"].isin(selected_experiment)]
                available_for_color.append("experiment")

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
        if tr_exists:
            # Based on the current filtered_df (which may be filtered by experiment and/or cell line)
            treatments = sorted(filtered_df["treatment"].unique().tolist())
            # If more than one treatment, show the widget
            if len(treatments) > 1:
                treatments.append("All")
                with cols[2]:
                    selected_treatments = st.multiselect("Select treatment(s)", treatments, default=treatments[-1], key="treatment_multiselect",on_change=update_multiselect, args=("treatment_multiselect", treatments))
                if "All" not in selected_treatments:
                    filtered_df = filtered_df[filtered_df["treatment"].isin(selected_treatments)]
                available_for_color.append("treatment")
            else:
                # Only one treatment or none
                # No widget needed
                pass

        # If more than one of experiment, cell_line, treatment columns exist, add a color_by multiselect
        # Only include columns that actually exist
        existing_filter_columns = [col for col in ["experiment", "cell_line", "treatment"] if col in df.columns]
        if len(existing_filter_columns) > 1 and color is True:
            with cols[3]:
                color_by_options = st.multiselect("Color by", existing_filter_columns, default=existing_filter_columns[-1])                   
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
    menus = ["menu_nadh", "menu_fad", "menu_morphology"]           
    # Render the dropdowns with callbacks
    selected_nadh = st.selectbox(
        "Nadh Variables", 
        ["Select"] + nadh_cols, 
        index=0, 
        key="menu_nadh",
        on_change=reset_other_menus, 
        args=("menu_nadh",menus)
    )

    selected_fad = st.selectbox(
        "Fad Variables", 
        ["Select"] + fad_cols, 
        index=0, 
        key="menu_fad",
        on_change=reset_other_menus, 
        args=("menu_fad",menus)
    )

    selected_morphology = st.selectbox(
        "Morphology Variables", 
        ["Select"] + morphology_cols, 
        index=0, 
        key="menu_morphology",
        on_change=reset_other_menus, 
        args=("menu_morphology",menus)
    )

    selected_var =  selected_nadh if selected_nadh != "Select" else selected_fad if selected_fad != "Select" else selected_morphology
    return selected_var


def create_multiSelects_vars(nadh_cols, fad_cols, morphology_cols):
    nadh_vars = st.multiselect(
        "Select NADH Variables",
        options= ["All NADH Variables"] + nadh_cols if len(nadh_cols) > 0 else nadh_cols,
        default=[],
        help="Select one or more columns corresponding to NADH variables."
    )
    fad_vars = st.multiselect(
        "Select FAD Variables",
        options= ["All FAD Variables"] + fad_cols if len(fad_cols) > 0 else fad_cols,
        default=[],
        help="Select one or more columns corresponding to FAD variables."
    )
    morphology_vars = st.multiselect(
        "Select Morphology Variables",
        options= ["All Morphology Variables"] + morphology_cols if len(morphology_cols) > 0 else morphology_cols,
        default=[],
        help="Select one or more columns corresponding to morphology variables."
    )

    return nadh_vars, fad_vars, morphology_vars