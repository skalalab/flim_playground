import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

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

def calculate_roc_curve(num_classes, y_test, y_score):
    y_test = label_binarize(y_test, classes=np.unique(y_test))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])  # Probabilities for the positive class
        roc_auc = auc(fpr, tpr)
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

def plot_confusion_matrix(y_test, y_pred, axis_label_size=12, legend_size=12):
    classes = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap='Blues', ax=ax, colorbar=False, text_kw={'fontsize': legend_size})
    ax.set_title("Confusion Matrix", fontsize=axis_label_size)
    ax.set_xlabel("Predicted Label", fontsize=axis_label_size)
    ax.set_ylabel("True Label", fontsize=axis_label_size)
    ax.tick_params(axis='both', labelsize=legend_size)
    ax.set_aspect('equal', adjustable='box')
    return fig

def plot_roc_curve(y_test, y_score, axis_label_size=12, legend_size=12):
    classes = np.unique(y_test)
    num_classes = len(classes)
    fpr, tpr, roc_auc = calculate_roc_curve(num_classes, y_test, y_score)
    fig, ax = plt.subplots()
    if num_classes == 2:
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:
        for i in range(num_classes):
            ax.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    ax.set_xlabel("False Positive Rate", fontsize=axis_label_size)
    ax.set_ylabel("True Positive Rate", fontsize=axis_label_size)
    ax.set_title("ROC Curve", fontsize=axis_label_size)
    ax.legend(loc="lower right", fontsize=legend_size)
    ax.tick_params(axis='both', labelsize=legend_size)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig

def plot_feature_importance(classifier, feature_names, axis_label_size=12, bar_label_size=12):
    fig, ax = plt.subplots(figsize=(12,6))
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
    Calculate comprehensive classification metrics including accuracy, precision, recall, and F1 score.
    Returns both overall accuracy and per-class metrics.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary containing overall accuracy and per-class metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    classes, support_per_class = np.unique(y_test, return_counts=True)
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Create dictionary with per-class metrics
    per_class_metrics = {}
    for i, cls in enumerate(classes):
        per_class_metrics[str(cls)] = {
            'n': int(support_per_class[i]),
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1_score': f1_per_class[i]
        }
    
    return {
        'accuracy': accuracy,
        'per_class': per_class_metrics
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

def create_per_class_metrics_table(metrics_dict):
    """
    Create a markdown table with per-class metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics (from calculate_metrics)
    
    Returns:
        str: Markdown formatted table
    """
    if 'per_class' not in metrics_dict or not metrics_dict['per_class']:
        return "\n    | **Class** | **N** | **Precision** | **Recall** | **F1 Score** |\n    |-----------|-------|---------------|------------|--------------|\n    | No data | - | - | - | - |\n"
    
    table = "    | **Class** | **N** | **Precision** | **Recall** | **F1 Score** |\n"
    table += "    |-----------|-------|---------------|------------|--------------|\n"
    
    for class_name, metrics in metrics_dict['per_class'].items():
        table += f"    | {class_name} | {metrics['n']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |\n"
    
    return f"\n{table}"

def create_metrics_table(metrics_dict):
    """
    Create markdown tables with overall accuracy and per-class metrics.
    Returns both tables for side-by-side display.
    
    Args:
        metrics_dict: Dictionary containing metrics (from calculate_metrics)
    
    Returns:
        tuple: (overall_accuracy_table, per_class_metrics_table)
    """
    return create_overall_accuracy_table(metrics_dict), create_per_class_metrics_table(metrics_dict)

def run_classification(df, method, splits, sampling_method, random_state=42):
    error_msg, X_train, X_test, y_train, y_test = prepare_data(df, splits, sampling_method, random_state)
    if error_msg:
        return error_msg, None
    if method == "Random Forest":
        classifier = RandomForestClassifier(random_state=random_state )
    elif method == "Gradient Boosting":
        classifier = GradientBoostingClassifier(random_state=random_state)
    elif method == "SVM":
        classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=random_state))
    elif method == "Logistic Regression":
        classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=random_state, max_iter=10000, class_weight='balanced'))
    
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)

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
    }
        