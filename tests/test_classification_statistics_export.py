"""Classification exports report every app metric without changing the analysis."""

import re
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.classify import (
    create_overall_accuracy_table,
    create_per_class_metrics_table,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    run_classification,
)
from src.dataset_io import check_and_fix_df
from src.export_script import generate_script


@pytest.fixture(params=[
    ("Random Forest", ["control_A", "drug B"], "None", "None", "None",
     {"n_estimators": 50, "max_depth": 3}),
    ("Logistic Regression", ["control_A", "drug B", "3"], "F1 Score",
     "Oversampling", "Balanced", {"C": .1, "solver": "lbfgs"}),
    ("SVM", ["drug B", "the rest"], "Balanced Accuracy", "None", "Balanced",
     {"kernel": "rbf", "C": .5}),
], ids=["binary", "multiclass-tuned", "one-versus-rest-tuned"])
def classification_case(request, tmp_path, monkeypatch, capsys):
    method, selected_classes, threshold, sampling, weight, parameters = request.param
    rng = np.random.default_rng(512)
    counts = [100, 70, 40]
    source = pd.DataFrame({
        "id": [f"cell{i}" for i in range(sum(counts))],
        "target": np.repeat(["control_A", "drug B", "3"], counts),
        "x": rng.normal(size=sum(counts)) + np.repeat([0., .65, 1.4], counts),
        "y": rng.normal(size=sum(counts)),
        "z": rng.normal(size=sum(counts)) * 2,
    })
    csv_path = tmp_path / "data.csv"
    source.to_csv(csv_path, index=False)
    # Both analyses start from the same saved bytes, including parsed floats.
    app_frame = pd.read_csv(csv_path, index_col=False, low_memory=False)
    app_frame, _, error = check_and_fix_df(app_frame, ["target"], "id", None)
    assert not error
    app_frame["classes"] = app_frame["target"]
    if "the rest" in selected_classes:
        app_frame["classes"] = app_frame["classes"].where(
            app_frame["classes"] == selected_classes[0], "the rest")
    else:
        app_frame = app_frame[app_frame["classes"].isin(selected_classes)]
    error, expected = run_classification(
        app_frame[["x", "y", "z", "classes"]], method, .7, sampling, weight,
        threshold, classifier_params=parameters, random_state=42)
    assert not error, error
    state = {
        "method": "Classification", "csv_filename": "data.csv",
        "unique_row_id_col": "id", "fov_name_col": None,
        "categorical_cols": ["target"], "axis_label_size": 16, "legend_size": 11,
        "method_params": {
            "selected_features": ["x", "y", "z"], "classification_method": method,
            "classify_by": ["target"], "classify_classes": selected_classes,
            "splits": .7, "sampling_method": sampling, "class_weight": weight,
            "threshold_method": threshold, "classifier_params": parameters,
        },
    }
    script = generate_script(state)
    assert not re.search(r"^\s*(?:from|import) src(?:\.|\b)", script, re.MULTILINE)
    path = tmp_path / "analysis.py"
    path.write_text(script)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    capsys.readouterr()
    try:
        namespace = runpy.run_path(str(path))
        output = capsys.readouterr().out
        yield expected, namespace, output, state, tmp_path
    finally:
        plt.close("all")


def _table_rows(text):
    """Read visible cells from the app's Markdown/HTML and plain console tables."""
    rows = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [re.sub(r"<[^>]+>", "", cell).replace("**", "").strip()
                 for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def test_export_reports_every_value_shown_in_the_app_metric_tables(classification_case):
    expected, _, output, state, _ = classification_case
    metrics = expected["metrics"]
    overall = dict(_table_rows(create_overall_accuracy_table(metrics))[1:])
    table = _table_rows(create_per_class_metrics_table(
        metrics, state["method_params"]["threshold_method"]))

    assert "Youden's J" in output, "The export must report the app's per-class Youden's J"
    assert _table_rows(output) == table
    assert f"Total N: {overall['N']}" in output
    assert f"Accuracy: {overall['Accuracy']}" in output
    assert f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}" in output
    assert "<span" not in output and "**" not in output
    assert len(table) == len(metrics["per_class"]) + 2  # Header, classes, full Average row.


def test_reporting_preserves_model_outputs_and_saved_figures(classification_case):
    expected, namespace, _, state, directory = classification_case
    for key in ("X_train", "X_test"):
        pd.testing.assert_frame_equal(namespace[key], expected[key])
    for key in ("y_train", "y_test"):
        pd.testing.assert_series_equal(namespace[key], expected[key])
    np.testing.assert_array_equal(namespace["y_pred"], expected["y_pred"])
    np.testing.assert_allclose(namespace["y_score"], expected["y_score"], rtol=1e-12, atol=1e-12)
    if expected["threshold_values"] is None:
        assert namespace["threshold_values"] is None
    else:
        np.testing.assert_array_equal(namespace["threshold_values"], expected["threshold_values"])
    assert namespace["metrics"]["per_class"] == expected["metrics"]["per_class"]
    np.testing.assert_array_equal(namespace["metrics"]["confusion_matrix"],
                                  expected["metrics"]["confusion_matrix"])

    styles = dict(axis_label_size=state["axis_label_size"], legend_size=state["legend_size"])
    app_roc = plot_roc_curve(expected["y_test"], expected["y_score"], **styles,
                              metrics=expected["metrics"], threshold_value=expected["threshold_values"])
    app_cm = plot_confusion_matrix(expected["y_test"], expected["y_pred"], **styles)
    assert len(namespace["fig_roc"].axes[0].lines) == len(app_roc.axes[0].lines)
    for actual, reference in zip(namespace["fig_roc"].axes[0].lines, app_roc.axes[0].lines):
        np.testing.assert_array_equal(actual.get_xydata(), reference.get_xydata())
        assert actual.get_label() == reference.get_label()
    np.testing.assert_array_equal(namespace["fig_cm"].axes[0].images[0].get_array(),
                                  app_cm.axes[0].images[0].get_array())
    saved = {"roc_curve.svg", "confusion_matrix.svg"}
    if hasattr(expected["classifier"], "feature_importances_"):
        np.testing.assert_array_equal(namespace["actual_clf"].feature_importances_,
                                      expected["classifier"].feature_importances_)
        app_fi = plot_feature_importance(
            expected["classifier"], expected["X_train"].columns,
            axis_label_size=styles["axis_label_size"], bar_label_size=styles["legend_size"])
        np.testing.assert_array_equal(
            [patch.get_width() for patch in namespace["fig_fi"].axes[0].patches],
            [patch.get_width() for patch in app_fi.axes[0].patches])
        saved.add("feature_importance.svg")
    assert {path.name for path in directory.glob("*.svg")} == saved
    assert all("<svg" in (directory / name).read_text() for name in saved)
