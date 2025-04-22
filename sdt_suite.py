import streamlit as st
from file_util import sdt_folder_check, get_sdts
from phasor import phasor_plot, calculate_phasor
from navigation import render_top_menu
from sdt_io import write_sdt, read_sdt150
from roi_sum import sum_sdt
import tifffile as tiff
import os
from pathlib import Path
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
        ["Phasor Analysis", "ROI Summing", "SDT Fitting"],
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
    elif method == "ROI Summing":
        folder_path = st.text_input("Copy and paste the *path* to the folder containing the sdt files *and* masks:")
        _, images, error_message = get_sdts(folder_path, mask=True)
        if error_message != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_message}</h5>", unsafe_allow_html=True)
        if len(images) > 0:
            st.write(images)
            upload_complete = True

    elif method == "SDT Fitting":
        st.write("Coming soon!")

    # elif method == "SDT Conversion":
    #     folder_path = st.text_input("Copy and paste the *path* to the folder containing the 10-bit sdt files:")
    #     sdts, _,_ = get_sdts(folder_path, mask=False)
    #     if len(sdts) == 0:
    #         st.markdown("<h7 style='text-align: center; color: red;'>We cannot find files that ends in .sdt inside the provided path. Note: this tool only works ***offline***, as the online app does not have access to your files.</h7>", unsafe_allow_html=True)
    #     else: 
    #         st.write([Path(sdt).name for sdt in sdts])
    #         upload_complete = True

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

    elif method == "ROI Summing" and images != {}:
        st.write("Creating a folder called 'summed_sdts' in the same directory as the input folder.")
        os.makedirs(os.path.join(folder_path, "summed_sdts"), exist_ok=True)
        for image in images:
            sdt = images[image]["sdt"]
            if "mask" not in images[image]:
                continue
            
            mask = images[image]["mask"]
            mask = tiff.imread(Path(mask))
            sdt_data = read_sdt150(sdt)
            _, summed_sdt, error_msg = sum_sdt(sdt_data, mask)
            if error_msg != "":
                st.write(f"Error summing {Path(sdt).name}: {error_msg}")
            else:
                resolution = sdt_data.shape[1]
                write_sdt(os.path.join(folder_path, "summed_sdts", f"{image}_summed.sdt"), summed_sdt, resolution=resolution)
                st.write(f"Summed {Path(sdt).name} and wrote to the 'summed_sdts' folder.")

    # elif method == "SDT Conversion" and sdts != []:
    #     st.write("Creating a folder called 'converted_sdts' in the same directory as the input folder.")
    #     os.makedirs(os.path.join(folder_path, "converted_sdts"), exist_ok=True)
    #     for sdt in sdts:
    #         error_message = sdt_convert(sdt, os.path.join(folder_path, "converted_sdts"))
            
    #         if error_message != "":
    #             st.write(f"Error converting {Path(sdt).name}: {error_message}")
    #         else:
    #             st.write(f"Converted {Path(sdt).name} and wrote to the 'converted_sdts' folder.")
           
    else:
        st.write("Waiting for file/folder path upload")
