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
    
def remove_image_or_cell_widget():
    image_removal, cell_removal = st.columns([0.3, 1])
    with image_removal:
        st.checkbox(
            "Remove Images",
            value=st.session_state.remove_image,
            key="remove_image",
            on_change=ensure_exclusive_images
        )

    with cell_removal:
        st.checkbox(
            "Remove Cells",
            value=st.session_state.remove_cell,
            key="remove_cell",
            on_change=ensure_exclusive_cells
        )

    return image_removal, cell_removal
