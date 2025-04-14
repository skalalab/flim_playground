import streamlit as st

def ensure_exclusive_images():
    # If user tries to uncheck "Remove Images", make sure "Remove Cells" is checked
    if not st.session_state.get_image_info:
        st.session_state.get_cell_info = True
    else:
        # If "Remove Images" is checked, ensure "Remove Cells" is unchecked
        st.session_state.get_cell_info = False

def ensure_exclusive_cells():
    # If user tries to uncheck "Remove Cells", make sure "Remove Images" is checked
    if not st.session_state.get_cell_info:
        st.session_state.get_image_info = True
    else:
        # If "Remove Cells" is checked, ensure "Remove Images" is unchecked
        st.session_state.get_image_info = False
        st.session_state.get_cell_info = True # I wonder why I need this but I do need this
    
def add_image_or_cell_widget():
    # initialize session state variables 
    if "get_image_info" not in st.session_state:
        st.session_state.get_image_info = True  # Initialize 'Remove Images' checked
    if "get_cell_info" not in st.session_state:
        st.session_state.get_cell_info = False  # Initialize 'Remove Cells' unchecked

    image_info, cell_info = st.columns([0.3, 1])
    with image_info:
        st.checkbox(
            "Get Image Info",
            #value=st.session_state.remove_image,
            key="get_image_info",
            on_change=ensure_exclusive_images
        )

    with cell_info:
        st.checkbox(
            "Get Cell Info",
            #value=st.session_state.remove_cell,
            key="get_cell_info",
            on_change=ensure_exclusive_cells
        )

    return image_info, cell_info

def add_img_cell_widget(clicked_points, fig):
    """
    Function to handle the removal of outliers based on user clicked points."""

    clicked_point = clicked_points[0]
    point_index =  clicked_point["pointIndex"]
    trace_index = clicked_point["curveNumber"]
    if st.session_state.get_cell_info:
        clicked_data = fig.data[trace_index]['text'][point_index]
        st.write(f"You clicked on cell: {clicked_data}.")
        if clicked_data not in st.session_state["added_cells"]:
            # Append the cell name to the list of added cells
            st.session_state["added_cells"].append(clicked_data)
        else:
            st.write(f"Cell {clicked_data} is already added.")
    else:
        clicked_data = fig.data[trace_index]['customdata'][point_index]
        st.write(f"You clicked on image: {clicked_data}.")
        if clicked_data not in st.session_state["added_images"]:
            # Append the image name to the list of added images
            st.session_state["added_images"].append(clicked_data)
        else:
            st.write(f"Image {clicked_data} is already added.")     
    st.session_state.last_processed_click = clicked_points # Store the processed click
    
def add_img_widget(current_clicked_points_img, fig):
    clicked_curve_index = current_clicked_points_img[0]["curveNumber"]
    # Retrieve image name from the trace's name property
    clicked_image_name = fig.data[clicked_curve_index].name 
    st.write(f"You clicked on image: {clicked_image_name}.")
    if clicked_image_name not in st.session_state["added_images"]:
        # Append the image name to the list of added images
        st.session_state["added_images"].append(clicked_image_name)
    else:
        st.write(f"Image {clicked_image_name} is already added.")
    st.session_state.last_processed_click_img = current_clicked_points_img # Store the processed clic

def display_infoList_widget():
    display_images, display_cells = st.columns([0.5, 0.5])
    with display_images: 
        st.write("Added images:")
        st.write(st.session_state["added_images"])
    with display_cells:
        st.write("Added cells:")
        st.write(st.session_state["added_cells"])

def reset_widget():
    """"
    "Function to reset the outlier removal or export the outlier_removed CSV.
    """
    if st.button("Reset"):
        st.session_state["added_images"] = []
        st.session_state["added_cells"] = []
        st.rerun()