import streamlit as st
import pandas as pd

from features import get_features, fix_df
from widgets import create_singleSelects_vars, create_filters
from navigation import render_top_menu
from visualization_functions import feature_comparison_plot

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
render_top_menu()

if "vis_df" not in st.session_state:
    st.session_state.vis_df = None

col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Visualizations")
    method = st.selectbox(
        "Select a visualization method",
        ["Feature Comparison"],
    )  

    uploaded_csv = st.file_uploader("Upload the CSV file from Region Props", type=["csv"])
    if method == "Feature Comparison" and uploaded_csv is not None:
        # Read the uploaded data
        df = pd.read_csv(uploaded_csv)
        numeric_cols, nadh_cols, fad_cols, morphology_cols, error_msg = get_features(df)
        if error_msg != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
            df = None
        else:
            if df is not None and len(df) > 0:
                df = fix_df(df)
                st.session_state.vis_df = df
                st.write("Data uploaded successfully. Please select a feature to visualize.")
            else:
                st.write("We cannot extract data from your uploaded file.")
        
        selected_var = create_singleSelects_vars(nadh_cols, fad_cols, morphology_cols)
        selected_test = st.selectbox("Select a statistical test", ["N/A", "Mann-Whitney", "t-test_ind"], index=0)

with col2:
    if st.session_state.vis_df is not None:
        filtered_df, compare_by_options, cols = create_filters(st.session_state.vis_df, color=True, compare=True)
        if selected_var != "Select": 
            # Plot the filtered dataframe
            fig = feature_comparison_plot(filtered_df, selected_var, compare_by_options, stats_test=selected_test)
            st.pyplot(fig, use_container_width=True)
            
    else:
        st.write("Please upload a file to begin.")

