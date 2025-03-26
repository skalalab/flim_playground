import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

from features import get_features, check_and_fix_df
from selection_widgets import create_singleSelects_vars
from filter_widgets import create_filters
from navigation import render_top_menu
from visualization_functions import feature_comparison_plot

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
render_top_menu()

# initialize session_states
if "vis_df" not in st.session_state:
    st.session_state.vis_df = None
if "removed_images" not in st.session_state:
    st.session_state["removed_images"] = []
if "removed_cells" not in st.session_state:
    st.session_state["removed_cells"] = []
if "remove_images" not in st.session_state:
    st.session_state.remove_images = True  # Initialize 'Remove Images' checked
if "remove_cells" not in st.session_state:
    st.session_state.remove_cells = False  # Initialize 'Remove Cells' unchecked


col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Visualizations")
    method = st.selectbox(
        "Select a visualization method",
        ["Feature Comparison", "Principal Component Analysis", "UMAP", "Phasor Plot", "Image Comparison"],
    )  

    uploaded_csv = st.file_uploader("Upload the CSV file obtained from [Data Extraction](/data_extraction)", type=["csv"])
    upload_complete = False
    # check and fix the uploaded csv 
    if uploaded_csv is not None:
        # Read the uploaded data
        df = pd.read_csv(uploaded_csv)
        df, warning_msg, error_msg = check_and_fix_df(df)

        if error_msg != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
            st.write("Therefore, we cannot extract data from your uploaded file.")
        else:
            if warning_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
            # then we can extract the single cell features
            df, feature_cols_dict, warning_msg, error_msg = get_features(df)
            if error_msg != "":
                st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
                st.write("Therefore, we cannot extract data from your uploaded file.")
            else:
                if warning_msg != "":
                    st.markdown(f"<h5 style='text-align: center; color: orange'>{warning_msg}</h5>", unsafe_allow_html=True)
                st.write("Data uploaded successfully. Please select a feature to visualize.")
                upload_complete = True
                st.session_state.vis_df = df
    if upload_complete:
        if method == "Feature Comparison":
            selected_var = create_singleSelects_vars(feature_cols_dict)
            selected_test = st.selectbox("Select a statistical test", ["N/A", "Mann-Whitney", "t-test_ind"], index=0)

with col2:
    if upload_complete:
        filtered_df, compare_by_options, cols = create_filters(st.session_state.vis_df, color=True, compare=True)
        if selected_var != "Select": 
            print(f"Selected variable: {selected_var}")
            # Plot the filtered dataframe
            # fig = feature_comparison_plot(filtered_df, selected_var, compare_by_options, stats_test=selected_test)
            # st.pyplot(fig, use_container_width=True)
            
    else:
        st.write("Please upload a file to begin.")

