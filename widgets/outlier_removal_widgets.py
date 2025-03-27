import streamlit as st

def ensure_exclusive_images():
    # If user tries to uncheck "Remove Images", make sure "Remove Cells" is checked
    if not st.session_state.remove_image:
        st.session_state.remove_cell = True
    else:
        # If "Remove Images" is checked, ensure "Remove Cells" is unchecked
        st.session_state.remove_cell = False

def ensure_exclusive_cells():
    # If user tries to uncheck "Remove Cells", make sure "Remove Images" is checked
    if not st.session_state.remove_cell:
        st.session_state.remove_image = True
    else:
        # If "Remove Cells" is checked, ensure "Remove Images" is unchecked
        st.session_state.remove_image = False
        st.session_state.remove_cell = True # I wonder why I need this but I do need this
    
def remove_image_or_cell_widget():
    # initialize session state variables 
    if "remove_image" not in st.session_state:
        st.session_state.remove_image = True  # Initialize 'Remove Images' checked
    if "remove_cell" not in st.session_state:
        st.session_state.remove_cell = False  # Initialize 'Remove Cells' unchecked

    image_removal, cell_removal = st.columns([0.3, 1])
    with image_removal:
        st.checkbox(
            "Remove Images",
            #value=st.session_state.remove_image,
            key="remove_image",
            on_change=ensure_exclusive_images
        )

    with cell_removal:
        st.checkbox(
            "Remove Cells",
            #value=st.session_state.remove_cell,
            key="remove_cell",
            on_change=ensure_exclusive_cells
        )

    return image_removal, cell_removal

def remove_outlier_widget(clicked_points, fig):
    """
    Function to handle the removal of outliers based on user clicked points."""

    clicked_point = clicked_points[0]
    point_index =  clicked_point["pointIndex"]
    trace_index = clicked_point["curveNumber"]
    if st.session_state.remove_cell:
        clicked_data = fig.data[trace_index]['text'][point_index]
        st.write(f"You clicked on cell: {clicked_data}. Do you want to remove this cell?")
    else:
        clicked_data = fig.data[trace_index]['customdata'][point_index]
        st.write(f"You clicked on image: {clicked_data}. Do you want to remove this image?")

    if st.button("Confirm Removal"):
        # Remove rows with the clicked cell_id
        if st.session_state.remove_cell:
            st.session_state["removed_cells"].append(clicked_data)
        else: 
            st.session_state["removed_images"].append(clicked_data)
        st.rerun()

def display_outliers_widget():
    display_images, display_cells = st.columns([0.5, 0.5])
    with display_images: 
        st.write("Removed images:")
        st.write(st.session_state["removed_images"])
    with display_cells:
        st.write("Removed cells:")
        st.write(st.session_state["removed_cells"])

def reset_export_widget(uploaded_csv, df):
    """"
    "Function to reset the outlier removal or export the outlier_removed CSV.
    """
    col1, col2 = st.columns([0.2, 1])
    with col1:
        if st.button("Reset"):
            st.session_state["removed_images"] = []
            st.session_state["removed_cells"] = []
            st.rerun()
    with col2:
        df_outliers_removed = df[
            (~df["image_name"].isin(st.session_state["removed_images"])) &
            (~df["cell_id"].isin(st.session_state["removed_cells"]))
        ]
        st.download_button(
            label="Download Outliers Removed CSV",
            data=df_outliers_removed.to_csv(index=False),
            file_name=f"{uploaded_csv.name}_outliers_removed.csv",
            mime="text/csv"
        )