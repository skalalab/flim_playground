"""Classification export: its builder and the statistics printer only it pastes."""
from src.export_script.skeleton import MethodSpec, _extract_source


def _print_classification_statistics(metrics):
    """Report shared metrics with the app table's precision and unweighted averages."""
    per_class = metrics["per_class"]
    total_n = sum(result["n"] for result in per_class.values())
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total N: {total_n}")
    print("| Class | N | Precision | Recall | Specificity | Youden's J | F1 Score |")
    print("|-------|---|-----------|--------|-------------|------------|----------|")
    fields = ("precision", "recall", "specificity", "youdens_j", "f1_score")
    for label, result in per_class.items():
        values = " | ".join(f"{result[field]:.4f}" for field in fields)
        print(f"| {label} | {result['n']} | {values} |")
    class_count = len(per_class)
    if class_count > 1:
        averages = " | ".join(
            f"{sum(result[field] for result in per_class.values()) / class_count:.4f}"
            for field in fields)
        print(f"| Average | {total_n / class_count:.2f} | {averages} |")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")


def _build_classification(state: dict) -> str:
    from src.classify import (
        _build_classifier,
        calculate_metrics,
        calculate_roc_curve,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_roc_curve,
        prepare_data,
        run_classification,
    )
    from src.tuned_threshold_classifier import TunedThresholdClassifierCV

    # Inlined from the app so the script computes exactly what the page did.
    computation_src = _extract_source(
        prepare_data, _build_classifier, calculate_metrics, _print_classification_statistics,
        calculate_roc_curve, plot_roc_curve, plot_confusion_matrix, plot_feature_importance,
        TunedThresholdClassifierCV, run_classification,
    )

    return f"""
# ============================================================
# Classification Pipeline (functions extracted from FLIM Playground source)
# ============================================================
{computation_src}

# --- Prepare data ---
feature_cols = SELECTED_FEATURES
df["classes"] = df[CLASSIFY_BY].astype(str).agg("_".join, axis=1)

if CLASSIFY_CLASSES:
    if "the rest" in CLASSIFY_CLASSES:
        target_class = [c for c in CLASSIFY_CLASSES if c != "the rest"][0]
        df["classes"] = df["classes"].apply(lambda x: x if x == target_class else "the rest")
    else:
        df = df[df["classes"].isin(CLASSIFY_CLASSES)]

# No feature-NaN drop: the app passes rows straight to sklearn (which rejects NaN),
# so the export must not silently train on a reduced subset — it errors the same way.

n_classes = df["classes"].nunique()
if n_classes < 2:
    raise ValueError(f"Need at least 2 classes for classification, but only found {{n_classes}}: {{df['classes'].unique().tolist()}}. "
                     "Check your CLASSIFY_CLASSES and categorical filters.")

df_classify = df[feature_cols + ["classes"]]

# --- Fit, tune and predict through the app's own pipeline (pasted above) ---
error_msg, result = run_classification(
    df_classify, CLASSIFICATION_METHOD, TRAIN_SIZE, SAMPLING_METHOD, CLASS_WEIGHT,
    THRESHOLD_METHOD, CLASSIFIER_PARAMS, RANDOM_STATE)
if error_msg:
    raise RuntimeError(error_msg)
classifier = result["classifier"]
X_train, X_test = result["X_train"], result["X_test"]
y_train, y_test = result["y_train"], result["y_test"]
y_pred, y_score = result["y_pred"], result["y_score"]
metrics, threshold_values = result["metrics"], result["threshold_values"]

print("=" * 60)
print(f"Classification Results: {{CLASSIFICATION_METHOD}}")
print("=" * 60)
_print_classification_statistics(metrics)

# --- Plots ---
fig_roc = plot_roc_curve(y_test, y_score, axis_label_size=AXIS_LABEL_SIZE,
                         legend_size=LEGEND_SIZE, metrics=metrics, threshold_value=threshold_values)
fig_roc.savefig("roc_curve.svg", format="svg", bbox_inches="tight")

fig_cm = plot_confusion_matrix(y_test, y_pred, axis_label_size=AXIS_LABEL_SIZE, legend_size=LEGEND_SIZE)
fig_cm.savefig("confusion_matrix.svg", format="svg", bbox_inches="tight")

# Feature importance
saved_fi = False
actual_clf = classifier
if hasattr(classifier, 'named_steps'):
    actual_clf = list(classifier.named_steps.values())[-1]
if hasattr(actual_clf, 'feature_importances_'):
    fig_fi = plot_feature_importance(actual_clf, feature_cols, axis_label_size=AXIS_LABEL_SIZE, bar_label_size=LEGEND_SIZE)
    fig_fi.savefig("feature_importance.svg", format="svg", bbox_inches="tight")
    saved_fi = True

plt.show()
print("Figures saved: roc_curve.svg, confusion_matrix.svg" + (", feature_importance.svg" if saved_fi else ""))
"""


def _config_lines(state: dict) -> list[str]:
    mp = state.get("method_params", {})
    return [
        f"SELECTED_FEATURES = {mp.get('selected_features', [])!r}",
        f"CLASSIFICATION_METHOD = {mp.get('classification_method')!r}",
        f"TRAIN_SIZE = {mp.get('splits', 0.7)!r}",
        f"SAMPLING_METHOD = {mp.get('sampling_method', 'None')!r}",
        f"CLASS_WEIGHT = {mp.get('class_weight', 'None')!r}",
        f"THRESHOLD_METHOD = {mp.get('threshold_method', 'None')!r}",
        f"CLASSIFIER_PARAMS = {mp.get('classifier_params', {})!r}",
        f"CLASSIFY_BY = {mp.get('classify_by', [])!r}  # Categorical columns used to build class labels",
        f"CLASSIFY_CLASSES = {mp.get('classify_classes', [])!r}  # Selected class values to include",
        "RANDOM_STATE = 42",
    ]


def _imports(state: dict) -> list[str]:
    mp = state.get("method_params", {})
    extra = [
        "from sklearn.model_selection import train_test_split, StratifiedKFold",
        "from sklearn.preprocessing import StandardScaler, label_binarize",
        "from sklearn.pipeline import make_pipeline",
        "from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, f1_score, accuracy_score, get_scorer",
        "from sklearn.base import clone",
        "from scipy.optimize import minimize",
        "from copy import deepcopy",
        # CLASSIFICATION_METHOD is read when the script runs, so every classifier
        # _build_classifier can name is imported; scikit-learn is always declared.
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier",
        "from sklearn.svm import SVC",
        "from sklearn.linear_model import LogisticRegression",
    ]
    sampling = mp.get("sampling_method", "None")
    if sampling == "Oversampling":
        extra.append("from imblearn.over_sampling import RandomOverSampler")
    elif sampling == "Undersampling":
        extra.append("from imblearn.under_sampling import RandomUnderSampler")
    return extra


def _packages(state: dict) -> set[str]:
    packages = {"scipy", "scikit-learn"}
    if state.get("method_params", {}).get("sampling_method", "None") in ("Oversampling", "Undersampling"):
        packages.add("imbalanced-learn")
    return packages


SPEC = MethodSpec(
    name="Classification",
    build=_build_classification,
    config_lines=_config_lines,
    imports=_imports,
    packages=_packages,
    # The builder saves its three figures itself, so there is no shared footer.
    footer=lambda state: None,
    # The app draws these figures with Matplotlib's defaults: no theme, no title size.
    title_size=None,
)
