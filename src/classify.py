import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(df, splits):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # y = label_binarize(y, classes=np.unique(y))
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=1-splits, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

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

def plot_confusion_matrix(y_test, y_pred):
    classes = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    # Create the confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap='Blues', ax=ax,colorbar=False)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return fig

def plot_roc_curve(y_test, y_score):
    classes = np.unique(y_test)
    num_classes = len( classes)

    # calculate roc curve
    fpr, tpr, roc_auc = calculate_roc_curve(num_classes, y_test, y_score)

    # plot the ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    if num_classes == 2:  # Binary classification
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:  # Multiclass classification
        for i in range(num_classes):
            ax.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
        
    ax.plot([0, 1], [0, 1], 'k--', label="Random Chance")  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()

    return fig

def plot_feature_importance(classifier, feature_names):
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
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title('Feature Importances from Random Forest')
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return fig

def classify(df, method, splits):
    X_train, X_test, y_train, y_test = prepare_data(df, splits)
    if method == "Random Forest":
        classifier = RandomForestClassifier(random_state=42)
    elif method == "SVM":
        # for svc and logreg, we need to scale the data
        classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42))
    elif method == "Logistic Regression":
        classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
    
    # y_score is the probability of the sample for each class in the model
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)

    fig1 = plot_confusion_matrix(y_test, y_pred)
    fig2 = plot_roc_curve(y_test, y_score)

    if method == "Random Forest":
        fig3 = plot_feature_importance(classifier, X_train.columns)
        return fig1, fig2, fig3, accuracy_score(y_test, y_pred)
    else:
        return fig1, fig2, None, accuracy_score(y_test, y_pred)
        


