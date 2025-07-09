import streamlit as st
import pandas as pd
from itertools import combinations

from src.classify import run_classification, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from src.navigation import render_top_menu
from src.widgets.selection_widgets import multi_feature_select_widget
from src.widgets.filter_widgets import filters_widget
from src.widgets.data_widgets import load_csv, happy_emoji, sad_emoji
from src.widgets.visualization_widgets import plot_config_widget
from src.feature_groups import categorical_cols

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu on this page as well
render_top_menu()
st.title("Classification")
col1, col2 = st.columns([0.4, 1])

with col1:
    # Read the uploaded data
    uploaded_csv = st.file_uploader("Upload the CSV file obtained from [Data Extraction](/data_extraction)", type=["csv"])
    df, feature_cols_dict, upload_complete = load_csv(uploaded_csv)
    if upload_complete:
        cols = st.columns(2)
        with cols[0]:
            classification_method = st.selectbox("Classifier", ["Random Forest", "SVM", "Logistic Regression"])
        with cols[1]:
            splits = st.slider("Train size (percentage of training data)", 0.5, 0.9, 0.7, 0.1)
        selected_features = multi_feature_select_widget(feature_cols_dict, n_per_row=1)
        filtered_df = df 

with col2:
    if upload_complete:   
        filtered_df = filters_widget(df)
        classify_by_options = [category for category in categorical_cols if category in filtered_df.columns and filtered_df[category].nunique() > 1]
        if len(classify_by_options) > 0:
            classify_by_options = st.multiselect("Classify by", classify_by_options, default=classify_by_options[-1])
        else:
            classify_by_options = []
        filtered_df['classes'] = filtered_df[classify_by_options].agg('_'.join, axis=1)
        classes = filtered_df['classes'].unique()
        if len(classes) <= 1 or len(selected_features) == 0:
            if len(classes) <= 1:
                st.write(f"No more than one class available for classification {sad_emoji}.")
            else:
                st.write("Please select features.")
        else: 
            classification_options = []
            for i in range(len(classes)+1, 1, -1):
                classification_option = list(combinations(classes, i))
                classification_options.extend(classification_option)

            # support classification of 1 class vs rest. If only two classes are available, then no need to show those option
            if len(classes) > 2:
                for cls in classes:
                    classification_options.append([cls, "the rest"])
            classification_options.reverse()
            classification_options_text= [" VS ".join(c) for c in classification_options]
            selected_option_text = st.selectbox("Select a way to classify", classification_options_text)
            selected_option = classification_options[classification_options_text.index(selected_option_text)]

            # handle the case of 1 class vs rest
            if "the rest" in selected_option:
                df_classify = filtered_df[selected_features+['classes']]
                df_classify.loc[:,'classes'] = df_classify['classes'].apply(lambda x: x if x == selected_option[0] else "the rest")
            else:
                df_classify = filtered_df[filtered_df['classes'].isin(selected_option)][selected_features+['classes']]
            
            if "the rest" in selected_option:
                the_rest = [cls for cls in classes if cls != selected_option[0]]
                selected_option_text = f"{selected_option[0]} VS {', '.join(the_rest)}"
            st.write(f"Running {classification_method} to classify between: {selected_option_text}, trained on {int(splits*100)}% of the data {happy_emoji}.")
            results = run_classification(df_classify, classification_method, splits)
            st.markdown("---")  # Add a separator line
            st.subheader("📈 Performance Metrics")
            # Get current values from session state as defaults for the widgets
            point_size, axis_label_size, legend_size = plot_config_widget(point_based=False)

            # Now generate the plots with the selected styles
            cols = st.columns(2)
            with cols[0]:
                fig1 = plot_confusion_matrix(results['y_test'], results['y_pred'], axis_label_size=axis_label_size, legend_size=legend_size)
                st.pyplot(fig1)
            with cols[1]:
                fig2 = plot_roc_curve(results['y_test'], results['y_score'], axis_label_size=axis_label_size, legend_size=legend_size)
                st.pyplot(fig2)
            st.write(f"Accuracy: {results['accuracy']:.2f}")
            if classification_method == "Random Forest":
                fig3 = plot_feature_importance(results['classifier'], results['X_train'].columns, axis_label_size=axis_label_size, bar_label_size=legend_size)
                st.pyplot(fig3)
            st.markdown("---")

    else:
        st.write("Waiting for file/folder path upload")