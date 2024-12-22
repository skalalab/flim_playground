import streamlit as st
from input import sdt_folder_check
from roi_sum import roi_sum_dimensionReduction
from phasor import phasor_plot

from navigation import render_top_menu
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
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
        if "irf" in images:
            st.write(f"The uploaded irf has {len(images["irf"])} timebins. If this is not as expected, the tool expects one number a row.")
        if images is not None and len(images) > 0:   
            st.write(images)

    if upload_complete is False:
        st.write("Please upload a file/folder path to begin.")

with col2:
    if upload_complete:
        if method == "Phasor Analysis":
            df, exp_var, error_message = roi_sum_dimensionReduction(images, selected_channel=selected_channel, method="Phasor")
            if df is not None:
                fig = phasor_plot(df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_message}</h5>", unsafe_allow_html=True)
    else:
        st.write("Waiting for file/folder path upload")