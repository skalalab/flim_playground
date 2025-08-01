import streamlit as st
from itertools import combinations
from src.dataset_io import happy_emoji, sad_emoji
from src.classify import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from src.widgets.visualization_widgets import plot_config_widget

def classifier_options_widget(df, categorical_cols, fov_name_col, selected_features, classifier, splits):
    classify_by_options = [category for category in categorical_cols if category in df.columns and df[category].nunique() > 1 and category != fov_name_col]
    col1, col2 = st.columns(2)
    with col1:
        if len(classify_by_options) > 0:
            classify_by_options = st.multiselect("Classify by", classify_by_options, default=classify_by_options[-1])
    with col2:
        sampling_method = st.selectbox("Sampling method", ["None", "Undersampling", "Oversampling"], help="Undersampling: Randomly remove samples from the majority class. Oversampling: Randomly duplicate samples from the minority class.")

    df['classes'] = df[classify_by_options].agg('_'.join, axis=1)
    classes = df['classes'].unique()
    if len(classes) <= 1:
        return f"No more than one class available for classification {sad_emoji}.", None, None
    elif len(selected_features) == 0:
        return "Please select features.", None, None
    else: 
        # Check if the number of classes would generate too many combinations
        max_combinations = 2000  # Reasonable limit for UI performance
        total_combinations = 2 ** len(classes) - 1  # Total possible non-empty combinations
        
        if total_combinations > max_combinations:
            return f"Too many classes ({len(classes)}) would generate {total_combinations:,} combinations, which exceeds the limit of {max_combinations:,}. Please reduce the number of classes or group some categories together {sad_emoji}.", None, None
        
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
            df_classify = df[selected_features+['classes']]
            df_classify.loc[:,'classes'] = df_classify['classes'].apply(lambda x: x if x == selected_option[0] else "the rest")
        else:
            df_classify = df[df['classes'].isin(selected_option)][selected_features+['classes']]
        
        if "the rest" in selected_option:
            the_rest = [cls for cls in classes if cls != selected_option[0]]
            selected_option_text = f"{selected_option[0]} VS {', '.join(the_rest)}"
        st.write(f"Running {classifier} to classify between: {selected_option_text}, trained on {int(splits*100)}% of the data {happy_emoji}.")
        return "", df_classify, sampling_method
    
def classification_plot_widget(results, classification_method):
    st.markdown("---")  # Add a separator line
    st.subheader("📈 Performance Metrics")
    # Get current values from session state as defaults for the widgets
    point_size, axis_label_size, legend_size, _ = plot_config_widget(point_based=False)

    # Now generate the plots with the selected styles
    cols = st.columns(2)
    with cols[0]:
        fig1 = plot_confusion_matrix(results['y_test'], results['y_pred'], axis_label_size=axis_label_size, legend_size=legend_size)
        st.pyplot(fig1)
    with cols[1]:
        fig2 = plot_roc_curve(results['y_test'], results['y_score'], axis_label_size=axis_label_size, legend_size=legend_size)
        st.pyplot(fig2)
    st.write(f"Accuracy: {results['accuracy']:.2f}")
    if classification_method in ["Random Forest", "Gradient Boosting"]:
        fig3 = plot_feature_importance(results['classifier'], results['X_train'].columns, axis_label_size=axis_label_size, bar_label_size=legend_size)
        st.pyplot(fig3)
    st.markdown("---")