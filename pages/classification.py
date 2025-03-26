import streamlit as st
import pandas as pd
from itertools import combinations

from train import classify
from features import get_features, fix_df
from navigation import render_top_menu
from selection_widgets import create_multiSelects_vars
from filter_widgets import create_filters
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu on this page as well
render_top_menu()
st.title("Classification")
col1, col2 = st.columns([0.4, 1])
with col1:
    uploaded_csv = st.file_uploader(" Please Upload the CSV file from Region Props", type=["csv"])
        
    if uploaded_csv is not None:
    # Read the uploaded data
        df = pd.read_csv(uploaded_csv)
        df = fix_df(df)
        numeric_cols, nadh_cols, fad_cols, morphology_cols, error_msg = get_features(df)
        if error_msg != "":
            st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
            df = None

        class_by = "treatment"
        treatments = df[class_by].unique()
        if len(treatments) <= 1:
            st.write("Either no treatment column found or only one treatment found in the uploaded file.")
            df = None
        else:
            # get all possible combinations of treatments as ways to classify the data
            classifications = []
            classification_options = []
            for i in range(len(treatments)+1, 1, -1):
                classification = list(combinations(treatments, i))
                classifications.extend(classification)
                classification_options.extend([" VS ".join(c) for c in classification])
            selected_option = st.selectbox("Select a way to classify", classification_options)
            selected_classification = classifications[classification_options.index(selected_option)]
            cols = st.columns(2)
            with cols[0]:
                classification_method = st.selectbox("Select a classification method", ["Random Forest", "SVM", "Logistic Regression"])
            with cols[1]:
                splits = st.slider("Select the train size (percentage of training data)", 0.5, 0.9, 0.7, 0.1)
            
    else:
        st.write("Please upload a file/folder path to begin.")
        df = None
with col2:
    if df is not None:
        nadh_vars, fad_vars, morphology_vars = create_multiSelects_vars(nadh_cols, fad_cols, morphology_cols, columns=True)
        if "All NADH Variables" in nadh_vars:
            nadh_vars = nadh_cols
        if "All FAD Variables" in fad_vars:
            fad_vars = fad_cols
        if "All Morphology Variables" in morphology_vars:
            morphology_vars = morphology_cols
        selected_vars = nadh_vars + fad_vars + morphology_vars
        df_classify = df[df[class_by].isin(selected_classification)][selected_vars+[class_by]]
        st.write(f"Running {classification_method} to classify between: {selected_option}, trained on {int(splits*100)}% of the data.")
        fig1, accuracy, fig2 = classify(df_classify, classification_method, splits)
        st.pyplot(fig1)
        st.write(f"Accuracy: {accuracy:.2f}")
        if fig2 is not None:
            st.pyplot(fig2)

    else:
        st.write("Waiting for file/folder path upload")