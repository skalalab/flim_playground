import streamlit as st
from input import sdt_folder_check
from phasor import phasor_plot, calculate_phasor
from widgets import create_filters
from navigation import render_top_menu
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

if "phasor_df" not in st.session_state:
    st.session_state.phasor_df = None
# Render the top menu 
render_top_menu()
col1, col2 = st.columns([0.4, 1])
with col1:
    upload_complete = False 
    st.title("SDT toolbox")
    method = st.selectbox(
        "Select a sdt tool",
        ["Phasor Analysis", "ROI Summing", "SDT Fitting", "SDT Conversion"],
    )  
    if method == "Phasor Analysis":
        folder_path = st.text_input("Copy and paste the *path* to the folder containing the sdt files *and* masks:")
        images, selected_channel, upload_complete = sdt_folder_check(folder_path, irf_check=True)
        if "original_irf" in images:
            st.write(f"The uploaded irf has {len(images["original_irf"])} timebins. If this is not as expected, the tool expects one number a row.")
        if images is not None and len(images) > 0:   
            st.write(images)

        if upload_complete and images is not None:
            # Now compute the df once, store it in session state
            df, error_message = calculate_phasor(images, selected_channel)
            if df is not None:
                st.session_state.phasor_df = df
            else:
                st.markdown(
                    f"<h5 style='text-align: center; color: red'>{error_message}</h5>",
                    unsafe_allow_html=True
                )

    if upload_complete is False:
        st.write("Please upload a file/folder path to begin.")

with col2:
   # if upload_complete:
    if method == "Phasor Analysis" and st.session_state.phasor_df is not None:
        # Create filters on df (does not re-upload or re-create df)
        filtered_df, color_by_options, cols = create_filters(st.session_state.phasor_df, color=False)

        # Plot the filtered dataframe
        fig = phasor_plot(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

        # Provide a download button
        st.download_button(
            label="Download Phasor Coordinates",
            data=st.session_state.phasor_df.to_csv(index=False),
            file_name="phasor.csv",
            mime="text/csv"
        )
           
    else:
        st.write("Waiting for file/folder path upload")
