from itertools import combinations

import streamlit as st

from src.classify import (
    create_overall_accuracy_table,
    create_per_class_metrics_table,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.emojis import happy_emoji, sad_emoji
from src.vis.plot_defaults import DEFAULT_AXIS_LABEL_FONT_SIZE, DEFAULT_LEGEND_FONT_SIZE
from src.widgets.visualization_widgets import plot_config_widget

CLASSIFIER_OPTIONS = ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"]


def _classifier_prefix(classifier):
    return f"clf_{classifier.lower().replace(' ', '_')}"


def _random_forest_hyperparams_widget(prefix):
    params = {}
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        params["n_estimators"] = st.slider("n_estimators", min_value=50, max_value=1000, value=100, step=50, key=f"{prefix}_n_estimators")
    with row1_col2:
        max_depth_raw = st.number_input("max_depth (0=None)", min_value=0, value=0, step=1, key=f"{prefix}_max_depth")
        params["max_depth"] = None if max_depth_raw == 0 else int(max_depth_raw)

    with st.expander("Advanced settings", expanded=False):
        adv1_col1, adv1_col2 = st.columns(2)
        with adv1_col1:
            params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, value=2, step=1, key=f"{prefix}_min_samples_split")
        with adv1_col2:
            params["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, value=1, step=1, key=f"{prefix}_min_samples_leaf")

        adv2_col1, _ = st.columns(2)
        with adv2_col1:
            max_features_option = st.selectbox("max_features", ["sqrt", "log2", "None"], key=f"{prefix}_max_features")
            params["max_features"] = None if max_features_option == "None" else max_features_option

    return params


def _gradient_boosting_hyperparams_widget(prefix):
    params = {}
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        params["n_estimators"] = st.slider("n_estimators", min_value=50, max_value=1000, value=100, step=50, key=f"{prefix}_n_estimators")
    with row1_col2:
        params["learning_rate"] = st.number_input("learning_rate", min_value=0.01, max_value=2.0, value=0.1, step=0.01, format="%.2f", key=f"{prefix}_learning_rate")

    with st.expander("Advanced settings", expanded=False):
        adv1_col1, adv1_col2 = st.columns(2)
        with adv1_col1:
            params["max_depth"] = st.number_input("max_depth", min_value=1, value=3, step=1, key=f"{prefix}_max_depth")
        with adv1_col2:
            params["subsample"] = st.slider("subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.05, key=f"{prefix}_subsample")

    return params


def _svm_hyperparams_widget(prefix):
    params = {}
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        params["kernel"] = st.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"], key=f"{prefix}_kernel")
    with row1_col2:
        c_options = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        params["C"] = st.select_slider(
            "C",
            options=c_options,
            value=1.0,
            help="SVM regularization coefficient. Higher C = weaker regularization (fits training data more closely); lower C = stronger regularization.",
            key=f"{prefix}_C",
        )

    with st.expander("Advanced settings", expanded=False):
        adv1_col1, adv1_col2 = st.columns(2)
        with adv1_col1:
            gamma_mode = st.selectbox("gamma", ["scale", "auto", "custom"], key=f"{prefix}_gamma_mode")
            if gamma_mode == "custom":
                params["gamma"] = st.number_input("gamma value", min_value=1e-06, max_value=10.0, value=0.1, step=0.01, format="%.4f", key=f"{prefix}_gamma_value")
            else:
                params["gamma"] = gamma_mode
        with adv1_col2:
            params["tol"] = st.number_input("tol", min_value=1e-05, max_value=0.1, value=0.001, step=1e-04, format="%.4f", key=f"{prefix}_tol")

        if params["kernel"] == "poly":
            adv2_col1, adv2_col2 = st.columns(2)
            with adv2_col1:
                params["degree"] = st.number_input("degree", min_value=2, max_value=8, value=3, step=1, key=f"{prefix}_degree")
            with adv2_col2:
                params["coef0"] = st.number_input("coef0", min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f"{prefix}_coef0")
        elif params["kernel"] == "sigmoid":
            adv2_col1, _ = st.columns(2)
            with adv2_col1:
                params["coef0"] = st.number_input("coef0", min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f"{prefix}_coef0")

    return params


def _logistic_hyperparams_widget(prefix):
    params = {}
    regularization_options_by_solver = {
        "lbfgs": ["l2", "none"],
        "liblinear": ["l1", "l2"],
        "newton-cg": ["l2", "none"],
        "newton-cholesky": ["l2", "none"],
        "sag": ["l2", "none"],
        "saga": ["l1", "l2", "elasticnet", "none"],
    }

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        params["solver"] = st.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], key=f"{prefix}_solver")
    with row1_col2:
        c_options = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        params["C"] = st.select_slider(
            "C",
            options=c_options,
            value=1.0,
            help="Inverse regularization strength in Logistic Regression. Higher C = weaker regularization; lower C = stronger regularization.",
            key=f"{prefix}_C",
        )

    with st.expander("Advanced settings", expanded=False):
        adv1_col1, adv1_col2 = st.columns(2)
        with adv1_col1:
            params["max_iter"] = st.number_input("max_iter", min_value=100, max_value=50000, value=10000, step=100, key=f"{prefix}_max_iter")
        with adv1_col2:
            params["tol"] = st.number_input("tol", min_value=1e-06, max_value=0.1, value=1e-04, step=1e-04, format="%.5f", key=f"{prefix}_tol")

        adv3_col1, adv3_col2 = st.columns(2)
        with adv3_col1:
            regularization_options = regularization_options_by_solver[params["solver"]]
            default_regularization = "l2" if "l2" in regularization_options else regularization_options[0]
            regularization_key = f"{prefix}_regularization"
            if regularization_key in st.session_state and st.session_state[regularization_key] not in regularization_options:
                st.session_state[regularization_key] = default_regularization
            regularization = st.selectbox(
                "regularization",
                regularization_options,
                index=regularization_options.index(default_regularization),
                key=regularization_key,
            )
            params["regularization"] = regularization
        with adv3_col2:
            params["fit_intercept"] = st.checkbox("fit_intercept", value=True, key=f"{prefix}_fit_intercept")

        if params["solver"] == "saga" and params["regularization"] == "elasticnet":
            adv4_col1, _ = st.columns(2)
            with adv4_col1:
                params["l1_ratio"] = st.slider("l1_ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key=f"{prefix}_l1_ratio")

    return params


def classifier_hyperparams_widget(classifier):
    prefix = _classifier_prefix(classifier)
    if classifier == "Random Forest":
        return _random_forest_hyperparams_widget(prefix)
    if classifier == "Gradient Boosting":
        return _gradient_boosting_hyperparams_widget(prefix)
    if classifier == "SVM":
        return _svm_hyperparams_widget(prefix)
    return _logistic_hyperparams_widget(prefix)


def classifier_options_widget(df, categorical_cols, fov_name_col, selected_features, classifier, splits):
    available_categories = [category for category in categorical_cols if category in df.columns and df[category].nunique() > 1 and category != fov_name_col]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if len(available_categories) > 0:
            classify_by_options = st.multiselect("Classify by", available_categories, default=available_categories[-1], key="classify_by_multiselect")
        else:
            classify_by_options = []
    with col2:
        sampling_method = st.selectbox("Sampling method", ["None", "Undersampling", "Oversampling"], help="Undersampling: Randomly remove samples from the majority class. Oversampling: Randomly duplicate samples from the minority class.")

    with col3:
        # Hide class_weight option for Gradient Boosting (not supported)
        if classifier != "Gradient Boosting":
            class_weight = st.selectbox("Class weight", ["None", "Balanced"], help="Balanced: Assign weights to each class when calculating the loss function. The weights are inversely proportional to the number of samples in each class.")
        else:
            # Set to None for Gradient Boosting (not supported by sklearn)
            class_weight = "None"
            st.caption("Class weight not available for Gradient Boosting")
    with col4:
        threshold_method = st.selectbox("Threshold tuning based on", ["None", "Balanced Accuracy", "F1 Score"])

    if len(available_categories) == 0:
        return f"No categorical feature available for classification. Classification requires a categorical column with at least two distinct values {sad_emoji}.", None, None, None, None
    if len(classify_by_options) == 0:
        return f"Please select at least one category for classification. {sad_emoji}", None, None, None, None

    df = df.copy()
    df['classes'] = df[classify_by_options].agg('_'.join, axis=1)
    classes = df['classes'].unique()
    if len(classes) <= 1:
        return f"No more than one class available for classification {sad_emoji}.", None, None, None, None
    elif len(selected_features) == 0:
        return f"Please select features. {sad_emoji}", None, None, None, None
    else: 
        # Check if the number of classes would generate too many combinations
        max_combinations = 2000  # Reasonable limit for UI performance
        total_combinations = 2 ** len(classes) - 1  # Total possible non-empty combinations
        
        if total_combinations > max_combinations:
            return f"Too many classes ({len(classes)}) would generate {total_combinations:,} combinations, which exceeds the limit of {max_combinations:,}. Please reduce the number of classes or group some categories together {sad_emoji}.", None, None, None, None
        
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
        return "", df_classify, sampling_method, class_weight, threshold_method
    
def classification_plot_widget(results, classification_method, threshold_method):
    st.subheader("📈 Performance Metrics")
    
    # Display metrics tables side by side (moved above plots)
    metrics = results['metrics']
    cols_metrics = st.columns(2)
    with cols_metrics[0]:
        st.markdown("**Overall Metrics**")
        overall_table = create_overall_accuracy_table(metrics)
        st.markdown(overall_table)
    with cols_metrics[1]:
        st.markdown("**Per-Class Metrics**")
        per_class_table = create_per_class_metrics_table(metrics, threshold_method)
        st.markdown(per_class_table, unsafe_allow_html=True)
    
    st.markdown("---")  # Divider between tables and plot controls
    
    plot_config_widget(point_based=False)
    axis_label_size = st.session_state.get("plot_axis_label_size", DEFAULT_AXIS_LABEL_FONT_SIZE)
    legend_size = st.session_state.get("plot_legend_size", DEFAULT_LEGEND_FONT_SIZE)

    # Now generate the plots with the selected styles
    cols = st.columns(2)
    with cols[0]:
        threshold_value = results.get('threshold_values')
        fig1 = plot_roc_curve(results['y_test'], results['y_score'], axis_label_size=axis_label_size, legend_size=legend_size, metrics=metrics, threshold_value=threshold_value)
        st.pyplot(fig1)
    with cols[1]:
        fig2 = plot_confusion_matrix(results['y_test'], results['y_pred'], axis_label_size=axis_label_size, legend_size=legend_size)
        st.pyplot(fig2)
    
    if classification_method in ["Random Forest", "Gradient Boosting"]:
        fig3 = plot_feature_importance(results['classifier'], results['X_train'].columns, axis_label_size=axis_label_size, bar_label_size=legend_size)
        st.pyplot(fig3)
    st.markdown("---")
