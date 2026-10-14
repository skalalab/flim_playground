"""The export generator is a package: one module per method, shared modules method-agnostic."""
import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import export_script
from src.export_labels import EXPORT_METHODS

PACKAGE = Path(export_script.__file__).parent
METHOD_NAMES = {"Feature Comparison", "Feature Histogram", "2D Feature Distribution",
                "Phasor Plot", "Dimension Reduction", "Classification"}
SHARED_MODULES = ("__init__.py", "skeleton.py", "mpl_helpers.py")


def test_registry_lists_every_method_once():
    """Each method registers one spec under its display name; nothing else is registered."""
    assert set(export_script.METHODS) == METHOD_NAMES
    for name, spec in export_script.METHODS.items():
        assert spec.name == name
        for field in ("build", "config_lines", "imports", "packages", "footer"):
            assert callable(getattr(spec, field)), (name, field)
    assert set(EXPORT_METHODS) <= set(export_script.METHODS)


@pytest.mark.parametrize("module", SHARED_MODULES)
def test_shared_modules_never_name_a_method(module):
    """Shared code reads the spec; it never branches on a method-name string."""
    tree = ast.parse((PACKAGE / module).read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not (literals & METHOD_NAMES), sorted(literals & METHOD_NAMES)


_STATE = {
    "csv_filename": "data.csv", "delimiter": ",", "unique_row_id_col": "cell_id", "fov_name_col": None,
    "categorical_filters": {}, "numerical_filters": [], "color_by": ["treatment"], "opacity_by": "dish",
    "shape_by": "cell_line", "separate_by": None, "subcolor_by": None,
    "categorical_cols": ["treatment", "cell_line", "dish"], "ignored_cols": [], "analysis_columns": None,
    "point_size": 5, "axis_label_size": 12, "legend_size": 10, "colormap": "tab10",
    "show_group_counts": False, "derived_export": None,
}
_PARAMS = {
    "Feature Comparison": {"selected_var": "F1", "effect_size_method": "Glass's delta",
                           "statistical_test": "t-test", "mean_or_median": "mean"},
    "Feature Histogram": {"selected_var": "F1", "apply_gmm": True},
    "2D Feature Distribution": {"selected_x": "F1", "selected_y": "F2", "marginal_plot_type": "boxplot",
                                "fit_regression": True, "fit_gmm_2d": True},
    "Phasor Plot": {"selected_channel": "nadh", "phasor_harmonic": 1, "phasor_f": 0.08},
    "Dimension Reduction": {"selected_features": ["F1", "F2"], "dr_method": "PCA", "hyperParam_dict": {}},
    "Classification": {"selected_features": ["F1", "F2"], "classification_method": "Random Forest",
                       "classify_by": ["treatment"], "classify_classes": ["a", "b"]},
}


@pytest.mark.parametrize("method", sorted(METHOD_NAMES))
def test_scripts_paste_only_helpers_they_use(method):
    """Every function or class pasted into a script is referenced somewhere else in it."""
    script = export_script.generate_script(dict(_STATE, method=method, method_params=_PARAMS[method]))
    pasted = [node.name for node in ast.parse(script).body
              if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    unused = [name for name in pasted
              if len(re.findall(rf"\b{re.escape(name)}\b", script)) < 2]
    assert unused == [], unused


@pytest.mark.parametrize("method", sorted(EXPORT_METHODS))
def test_derived_csv_names_come_from_export_methods(method):
    """The derived label column and CSV name reach a script once each, from EXPORT_METHODS."""
    label_column, filename, _ = EXPORT_METHODS[method]
    script = export_script.generate_script(dict(_STATE, method=method, method_params=_PARAMS[method]))
    assert f"DERIVED_LABEL_NAME = {label_column!r}" in script
    assert f"DERIVED_DATA_FILE = {filename!r}" in script
    # Pasted app functions may carry the app's own fallback names; the template may not.
    template = [node for node in ast.parse(script).body
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    literals = [node.value for statement in template for node in ast.walk(statement)
                if isinstance(node, ast.Constant) and node.value in (label_column, filename)]
    assert sorted(literals) == sorted([label_column, filename])


@pytest.mark.parametrize("method, extra_params, helper", [
    ("Classification", {}, "run_classification"),
    ("Feature Comparison", {"overlay": "SuperPlot", "collapse_by": "dish"}, "superplot_display_frame"),
    ("Feature Comparison", {"collapse_by": "dish"}, "drop_varying_channels"),
    ("Feature Histogram", {"log_x": True}, "log10_columns"),
    ("2D Feature Distribution", {"log_x": True}, "log10_columns"),
    ("Feature Histogram", {}, "histogram_labels"),
    ("Phasor Plot", {}, "phasor_columns"),
    ("Phasor Plot", {}, "resolve_phasor_category"),
    ("Phasor Plot", {}, "panel_point_size"),
    ("2D Feature Distribution", {}, "gmm_ellipse_params"),
    ("2D Feature Distribution", {}, "gmm_group_title"),
    ("2D Feature Distribution", {}, "panel_point_size"),
    ("Dimension Reduction", {}, "reduce_dimensions"),
    ("Dimension Reduction", {}, "dimension_axis_titles"),
    ("Dimension Reduction", {}, "panel_point_size"),
    ("Dimension Reduction", {}, "focus_facet_label"),
    ("2D Feature Distribution", {}, "focus_facet_label"),
    ("Feature Comparison", {}, "comparison_layout"),
    ("Feature Comparison", {}, "comparison_labels"),
    ("Feature Comparison", {}, "plan_effect_size_brackets"),
    ("Feature Comparison", {}, "p_value_stars"),
    ("Feature Comparison", {}, "box_statistics"),
    ("Feature Comparison", {}, "sina_offsets"),
], ids=lambda value: value if isinstance(value, str) else "")
def test_scripts_compute_through_the_app_function(method, extra_params, helper):
    """A script pastes the app's pure function instead of re-typing its body.

    test_scripts_paste_only_helpers_they_use then requires a second mention of every
    pasted name, so a paste nothing calls fails there.
    """
    params = dict(_PARAMS[method], **extra_params)
    script = export_script.generate_script(dict(_STATE, method=method, method_params=params))
    assert f"def {helper}(" in script


def test_classification_script_runs_after_editing_the_method(tmp_path, monkeypatch):
    """CLASSIFICATION_METHOD is read when the script runs, so every classifier is importable."""
    import runpy

    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = __import__("numpy").random.default_rng(0)
    n = 80
    frame = pd.DataFrame({
        "cell_id": range(1, n + 1), "treatment": ["a", "b"] * (n // 2),
        "cell_line": ["x", "y", "y", "x"] * (n // 4), "dish": ["d1", "d2"] * (n // 2),
        "F1": rng.normal(0, 1, n), "F2": rng.normal(0, 1, n)})
    frame.loc[frame["treatment"] == "b", "F1"] += 2.5
    frame.to_csv(tmp_path / "data.csv", index=False)
    script = export_script.generate_script(
        dict(_STATE, method="Classification", method_params=_PARAMS["Classification"]))
    assert "CLASSIFICATION_METHOD = 'Random Forest'" in script
    script = script.replace("CLASSIFICATION_METHOD = 'Random Forest'", "CLASSIFICATION_METHOD = 'SVM'")
    (tmp_path / "analysis.py").write_text(script, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        namespace = runpy.run_path(str(tmp_path / "analysis.py"))
    finally:
        plt.close("all")
    assert type(namespace["actual_clf"]).__name__ == "SVC"


def test_2d_script_panel_constants_come_from_plot_defaults():
    """The grid constants the 2D script used to restate by hand now come from plot_defaults."""
    from src.vis import plot_defaults

    script = export_script.generate_script(dict(
        _STATE, method="2D Feature Distribution", method_params=_PARAMS["2D Feature Distribution"]))
    for name in ("PANEL_MODEL_WIDTH", "OVERVIEW_MODEL_WIDTH", "PANEL_CONTEXT_COLOR", "PANEL_CONTEXT_ALPHA"):
        assert f"{name} = {getattr(plot_defaults, name)!r}" in script
    assert "PANEL_POINT_SIZE = panel_point_size(POINT_SIZE)" in script
