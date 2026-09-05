import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from copy import deepcopy
from src.tuned_threshold_classifier import TunedThresholdClassifierCV

def prepare_data(df, splits, sampling_method, random_state):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    try:
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=1-splits, random_state=random_state, stratify=y)   # keeps original class fractions
    except ValueError as e:
        return f"Error splitting data: {e}", None, None, None, None
    sampler = None
    if sampling_method == "Undersampling":
        sampler = RandomUnderSampler(random_state=random_state)
    elif sampling_method == "Oversampling":
        sampler = RandomOverSampler(random_state=random_state)

    if sampler is not None:
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    
    return "", X_train, X_test, y_train, y_test

def calculate_roc_curve(num_classes, y_test, y_score, classes=None):
    if classes is None:
        classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    if num_classes == 2:
        # label_binarize marks class 1 positive; invert to plot class 0's ROC.
        y_test_first_class = 1 - y_test_binarized  
        fpr, tpr, _ = roc_curve(y_test_first_class, y_score[:,0])  # Probabilities for the first class
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc, classes[0]
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        return fpr, tpr, roc_auc, classes

def plot_roc_curve(y_test, y_score, axis_label_size=12, legend_size=12, metrics=None, threshold_value=None):
    classes = np.unique(y_test)
    num_classes = len(classes)
    result = calculate_roc_curve(num_classes, y_test, y_score, classes=classes)
    fig, ax = plt.subplots()
    
    # Define colors for different classes (use a colormap for multi-class)
    if num_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    if num_classes == 2:
        fpr, tpr, roc_auc, positive_class = result
        # Show which class the ROC represents
        ax.plot(fpr, tpr, label=f"{positive_class} (AUC = {roc_auc:.2f})", color=colors[0])
        
        # Plot operating point if metrics are provided
        if metrics is not None and 'per_class' in metrics:
            # Per-class metrics use string keys; this ROC represents class 0.
            pos_class_str = str(positive_class)
            if pos_class_str in metrics['per_class']:
                cls_metrics = metrics['per_class'][pos_class_str]
                # FPR = 1 - Specificity
                op_fpr = 1 - cls_metrics['specificity']
                op_tpr = cls_metrics['recall']
                
                threshold_label = ''
                if threshold_value is not None:
                    # Convert the classifier's class-1 threshold to class-0 probability.
                    threshold_label += f'Threshold ≥ {1 - threshold_value:.2f}'
                
                ax.scatter(op_fpr, op_tpr, c='red', s=100, zorder=5, label=threshold_label)
    else:
        fpr, tpr, roc_auc, classes = result
        
        # Extract threshold values for label display only
        threshold_array = None
        if threshold_value is not None:
            threshold_array = np.asarray(threshold_value)
        
        # Plot ROC curves for each class
        for i in range(num_classes):
            # Build label with AUC and normalization factor if available
            if threshold_array is not None and len(threshold_array) == num_classes and threshold_array[i] is not None:
                label = f"{classes[i]} (AUC={roc_auc[i]:.2f}, norm={threshold_array[i]:.2f})"
            else:
                label = f"{classes[i]} (AUC={roc_auc[i]:.2f})"
            ax.plot(fpr[i], tpr[i], label=label, color=colors[i])

            # Plot actual operating point from metrics
            if metrics is not None and 'per_class' in metrics:
                class_str = str(classes[i])
                if class_str in metrics['per_class']:
                    cls_metrics = metrics['per_class'][class_str]
                    # FPR = 1 - Specificity
                    op_fpr = 1 - cls_metrics['specificity']
                    op_tpr = cls_metrics['recall']
                    
                    ax.scatter(op_fpr, op_tpr, 
                             c=colors[i], s=100, zorder=5, marker='s', 
                             edgecolors='black', linewidths=1.5)
    
    ax.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    ax.set_xlabel("False Positive Rate", fontsize=axis_label_size)
    ax.set_ylabel("True Positive Rate", fontsize=axis_label_size)
    ax.set_title("ROC Curve", fontsize=axis_label_size)
    ax.legend(loc="lower right", fontsize=legend_size)
    ax.tick_params(axis='both', labelsize=legend_size)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_test, y_pred, axis_label_size=12, legend_size=12):
    """
    Plot confusion matrix.
    Computes confusion matrix separately for plotting.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        axis_label_size: Font size for axis labels
        legend_size: Font size for legend/text
    """
    classes = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap='Blues', ax=ax, colorbar=False, text_kw={'fontsize': legend_size})
    ax.set_title("Confusion Matrix", fontsize=axis_label_size)
    ax.set_xlabel("Predicted Label", fontsize=axis_label_size)
    ax.set_ylabel("True Label", fontsize=axis_label_size)
    ax.tick_params(axis='both', labelsize=legend_size)
    ax.set_aspect('equal', adjustable='box')
    return fig

def plot_feature_importance(classifier, feature_names, axis_label_size=12, bar_label_size=12):
    fig, ax = plt.subplots(figsize=(24,12))
    feature_importances = pd.Series(classifier.feature_importances_, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False)
    plt.barh(
        feature_importances.index,
        feature_importances.values,
        color="skyblue",
        edgecolor="black",
        height=0.8
    )
    plt.xlabel("Feature Importance", fontsize=axis_label_size)
    plt.ylabel("Features", fontsize=axis_label_size)
    plt.title('Feature Importances from Random Forest', fontsize=axis_label_size)
    plt.xticks(rotation=45, ha='right', fontsize=bar_label_size)
    plt.yticks(fontsize=bar_label_size)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return fig

def calculate_metrics(y_test, y_pred):
    """
    Calculate classification metrics from one shared confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary containing overall accuracy, per-class metrics, and confusion matrix
    """
    # Get classes and calculate confusion matrix once
    classes, support_per_class = np.unique(y_test, return_counts=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    num_classes = len(classes)
    total_samples = np.sum(cm)
    
    # Initialize arrays for per-class metrics
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    specificity_per_class = np.zeros(num_classes)
    youdens_j_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    
    # Calculate all metrics from confusion matrix
    for i in range(num_classes):
        # True Positives: diagonal element
        tp = cm[i, i]
        
        # False Positives: sum of column i minus TP
        fp = np.sum(cm[:, i]) - tp
        
        # False Negatives: sum of row i minus TP
        fn = np.sum(cm[i, :]) - tp
        
        # True Negatives: all samples minus TP, FP, FN
        tn = total_samples - (tp + fp + fn)
        
        # Precision = TP / (TP + FP)
        if (tp + fp) > 0:
            precision_per_class[i] = tp / (tp + fp)
        else:
            precision_per_class[i] = 0.0
        
        # Recall (Sensitivity) = TP / (TP + FN)
        if (tp + fn) > 0:
            recall_per_class[i] = tp / (tp + fn)
        else:
            recall_per_class[i] = 0.0
        
        # Specificity (TNR) = TN / (TN + FP)
        if (tn + fp) > 0:
            specificity_per_class[i] = tn / (tn + fp)
        else:
            specificity_per_class[i] = 0.0
        
        # Youden's J = Sensitivity + Specificity - 1 = Recall + Specificity - 1
        youdens_j_per_class[i] = recall_per_class[i] + specificity_per_class[i] - 1.0
        
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        if (precision_per_class[i] + recall_per_class[i]) > 0:
            f1_per_class[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i])
        else:
            f1_per_class[i] = 0.0
    
    # Overall accuracy = sum of diagonal / total samples
    accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0.0
    
    # Overall balanced accuracy = average of recall across all classes
    overall_balanced_accuracy = np.mean(recall_per_class)
    
    # Create dictionary with per-class metrics
    per_class_metrics = {}
    for i, cls in enumerate(classes):
        per_class_metrics[str(cls)] = {
            'n': int(support_per_class[i]),
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'specificity': specificity_per_class[i],
            'youdens_j': youdens_j_per_class[i],
            'f1_score': f1_per_class[i]
        }
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': overall_balanced_accuracy,
        'per_class': per_class_metrics,
        'confusion_matrix': cm,
        'classes': classes
    }

def create_overall_accuracy_table(metrics_dict):
    """
    Create a markdown table with overall accuracy.
    
    Args:
        metrics_dict: Dictionary containing metrics (from calculate_metrics)
    
    Returns:
        str: Markdown formatted table
    """
    # Calculate total N
    total_n = sum(metrics['n'] for metrics in metrics_dict['per_class'].values()) if 'per_class' in metrics_dict else 0
    
    table = "    | **Metric** | **Value** |\n"
    table += "    |------------|-----------|\n"
    table += f"    | **Accuracy** | {metrics_dict['accuracy']:.4f} |\n"
    table += f"    | **N** | {total_n} |\n"
    
    return f"\n{table}"

def create_per_class_metrics_table(metrics_dict, threshold_method):
    """
    Create a markdown table with per-class metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics (from calculate_metrics)
        threshold_method: Threshold tuning method
    Returns:
        str: Markdown formatted table
    """
    # Color column header red based on threshold method
    recall_header = '<span style="color:red">**Recall**</span>' if threshold_method == "Balanced Accuracy" else "Recall"
    f1_score_header = '<span style="color:red">**F1 Score**</span>' if threshold_method == "F1 Score" else "F1 Score"
    
    if 'per_class' not in metrics_dict or not metrics_dict['per_class']:
        return f"\n    | **Class** | **N** | **Precision** | {recall_header} | **Specificity** | **Youden's J** | {f1_score_header} |\n    |-----------|-------|---------------|------------|-----------------|----------------|--------------|\n    | No data | - | - | - | - | - | - |\n"
    
    table = f"    | **Class** | **N** | **Precision** | {recall_header} | **Specificity** | **Youden's J** | {f1_score_header} |\n"
    table += "    |-----------|-------|---------------|------------|-----------------|----------------|--------------|\n"
    
    # Calculate averages for multi-class case
    num_classes = len(metrics_dict['per_class'])
    if num_classes > 1:
        # Initialize accumulators
        total_n = sum_precision = sum_recall = sum_specificity = sum_youdens_j = sum_f1_score = 0.0
    
    for class_name, metrics in metrics_dict['per_class'].items():
        # Color recall values red if threshold method is Balanced Accuracy
        recall_value = f'<span style="color:red">{metrics["recall"]:.4f}</span>' if threshold_method == "Balanced Accuracy" else f"{metrics['recall']:.4f}"
        # Color F1 Score values red if threshold method is F1 Score
        f1_score_value = f'<span style="color:red">{metrics["f1_score"]:.4f}</span>' if threshold_method == "F1 Score" else f"{metrics['f1_score']:.4f}"
        youdens_j_value = f"{metrics['youdens_j']:.4f}"
        table += f"    | {class_name} | {metrics['n']} | {metrics['precision']:.4f} | {recall_value} | {metrics['specificity']:.4f} | {youdens_j_value} | {f1_score_value} |\n"
        
        # Accumulate for average calculation
        if num_classes > 1:
            total_n += metrics['n']
            sum_precision += metrics['precision']
            sum_recall += metrics['recall']
            sum_specificity += metrics['specificity']
            sum_youdens_j += metrics['youdens_j']
            sum_f1_score += metrics['f1_score']
    
    # Add average row if more than one class
    if num_classes > 1:
        avg_n = total_n / num_classes
        avg_precision = sum_precision / num_classes
        avg_recall = sum_recall / num_classes
        avg_specificity = sum_specificity / num_classes
        avg_youdens_j = sum_youdens_j / num_classes
        avg_f1_score = sum_f1_score / num_classes
        
        # Color recall average red if threshold method is Balanced Accuracy
        avg_recall_value = f'<span style="color:red">{avg_recall:.4f}</span>' if threshold_method == "Balanced Accuracy" else f"{avg_recall:.4f}"
        # Color F1 Score average red if threshold method is F1 Score
        avg_f1_score_value = f'<span style="color:red">{avg_f1_score:.4f}</span>' if threshold_method == "F1 Score" else f"{avg_f1_score:.4f}"
        
        table += f"    | **Average** | {avg_n:.2f} | {avg_precision:.4f} | {avg_recall_value} | {avg_specificity:.4f} | {avg_youdens_j:.4f} | {avg_f1_score_value} |\n"
    
    return f"\n{table}"


def _build_classifier(method, class_weight, classifier_params, random_state):
    params = dict(classifier_params or {})

    if method == "Random Forest":
        params.pop("class_weight", None)
        params.pop("random_state", None)
        params.pop("n_jobs", None)
        # Fit independently seeded trees across all available cores.
        return RandomForestClassifier(random_state=random_state, class_weight=class_weight,
                                      n_jobs=-1, **params)
    if method == "Gradient Boosting":
        params.pop("random_state", None)
        return GradientBoostingClassifier(random_state=random_state, **params)
    if method == "SVM":
        params.pop("probability", None)
        params.pop("class_weight", None)
        params.pop("random_state", None)
        svm_params = {
            "kernel": "linear",
            "probability": True,
            "random_state": random_state,
            "class_weight": class_weight,
        }
        svm_params.update(params)
        return make_pipeline(StandardScaler(), SVC(**svm_params))
    if method == "Logistic Regression":
        params.pop("class_weight", None)
        params.pop("random_state", None)
        solver = params.get("solver", "lbfgs")
        regularization = str(params.pop("regularization", "l2")).lower()

        allowed_regularization_by_solver = {
            "lbfgs": ["l2", "none"],
            "liblinear": ["l1", "l2"],
            "newton-cg": ["l2", "none"],
            "newton-cholesky": ["l2", "none"],
            "sag": ["l2", "none"],
            "saga": ["l1", "l2", "elasticnet", "none"],
        }
        allowed_regularization = allowed_regularization_by_solver.get(solver, ["l2"])
        if regularization not in allowed_regularization:
            raise ValueError(f"Regularization '{regularization}' is not supported by solver '{solver}'")

        logistic_params = {
            "random_state": random_state,
            "max_iter": 10000,
            "class_weight": class_weight,
        }
        allowed_logistic_params = {"solver", "C", "max_iter", "tol", "fit_intercept", "l1_ratio"}
        logistic_params.update({k: v for k, v in params.items() if k in allowed_logistic_params})

        # scikit-learn 1.8+ path: drive regularization with l1_ratio/C.
        if regularization == "l2":
            logistic_params["l1_ratio"] = 0.0
        elif regularization == "l1":
            logistic_params["l1_ratio"] = 1.0
        elif regularization == "elasticnet":
            logistic_params["l1_ratio"] = float(logistic_params.get("l1_ratio", 0.5))
        elif regularization == "none":
            # Near-unregularized setting without triggering deprecation warnings.
            logistic_params["C"] = 1e12
            logistic_params["l1_ratio"] = 0.0

        return make_pipeline(StandardScaler(), LogisticRegression(**logistic_params))

    raise ValueError(f"Unsupported classification method: {method}")


def run_classification(df, method, splits, sampling_method, class_weight, threshold_method, classifier_params=None, random_state=42):
    error_msg, X_train, X_test, y_train, y_test = prepare_data(df, splits, sampling_method, random_state)
    if error_msg:
        return error_msg, None
    if class_weight == "Balanced":
        class_weight = 'balanced'
    else:
        class_weight = None
    try:
        classifier = _build_classifier(method, class_weight, classifier_params, random_state)
    except Exception as e:
        return f"Error creating {method} model: {e}", None
    
    # adjust thresholds based on the threshold method
    tuned_classifier = None
    threshold_values = None

    try:
        if threshold_method == "None":
            # Use default threshold (0.5 for binary classification)
            threshold_values = 0.5 if len(np.unique(y_train)) == 2 else None
        elif threshold_method in ["Balanced Accuracy", "F1 Score"]:
            # Map threshold method to scoring metric
            scoring_map = {
                "Balanced Accuracy": "balanced_accuracy",
                "F1 Score": "f1_macro"
            }
            # Create a deep copy of the classifier for tuning
            classifier_copy = deepcopy(classifier)
            tuned_classifier = TunedThresholdClassifierCV(
                classifier_copy, 
                scoring=scoring_map[threshold_method],
                random_state=random_state
            ).fit(X_train, y_train)
            # Extract thresholds (binary: best_threshold_, multi-class: best_thresholds_)
            if tuned_classifier.n_classes_ == 2:
                threshold_values = tuned_classifier.best_threshold_
            else:
                threshold_values = tuned_classifier.best_thresholds_
        
        # Fit the original classifier if not already fitted
        if tuned_classifier is None:
            classifier.fit(X_train, y_train)
        
        # Get predictions and probabilities
        if tuned_classifier is not None:
            y_pred = tuned_classifier.predict(X_test)
            y_score = tuned_classifier.predict_proba(X_test)
            # Use the tuned classifier as the main classifier
            classifier = tuned_classifier.estimator
        else:
            y_pred = classifier.predict(X_test)
            y_score = classifier.predict_proba(X_test)
    except Exception as e:
        return f"Error training {method}: {e}", None

    # Calculate comprehensive metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    return "", {
        'classifier': classifier,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_score': y_score,
        'metrics': metrics,
        'threshold_values': threshold_values,
    }
        
