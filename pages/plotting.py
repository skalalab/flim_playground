import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from features import get_features, fix_df
from navigation import render_top_menu
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
render_top_menu()
st.write("To be developed")
col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Visualizations")
    method = st.selectbox(
        "Select a visualization method",
        ["Feature Comparison"],
    )  
    upload_complete = False 
    uploaded_csv = st.file_uploader("Upload the CSV file from Region Props", type=["csv"])

    if uploaded_csv is not None:
        # Read the uploaded data
        df = pd.read_csv(uploaded_csv)
        numeric_cols, nadh_cols, fad_cols, morphology_cols, error_msg = get_features(df)
        if error_msg != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
            upload_complete = False
        else:
            df = fix_df(df)
            # st.markdown("<h6 style='text-align: center;'>File uploaded successfully.</h6>", unsafe_allow_html=True)
            upload_complete = True
        if method == "Feature Comparison":
            nadh_vars = st.multiselect(
                "Select NADH Variables",
                options= nadh_cols ,
                default=[],
                help="Select one or more columns corresponding to NADH variables."
            )
            fad_vars = st.multiselect(
                "Select FAD Variables",
                options= fad_cols,
                default=[],
                help="Select one or more columns corresponding to FAD variables."
            )
            morphology_vars = st.multiselect(
                "Select Morphology Variables",
                options= morphology_cols ,
                default=[],
                help="Select one or more columns corresponding to morphology variables."
            )


