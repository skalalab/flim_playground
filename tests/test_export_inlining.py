"""Inlined export helpers include every required module-level constant.
inspect.getsource copies function source without its module namespace, so the
generated script must supply any referenced globals.
"""
import runpy
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import classify, collapse, dataset_io
from src.export_script import generate_script
from src.vis import (
    distribution_2d,
    phasor,
    dimension_facets,
    gmm_tables,
    feature_comparison,
    helpers,
    feature_histogram,
    dimension_reduction,
    plot_defaults,
)
from src.widgets import encoding_state

# The app functions src/export_script inlines verbatim: _build_data_loading's four,
# prepare_rows's two, and the computation each method shares with the app.
INLINED_HELPERS = (
    dataset_io.drop_unnamed_columns,
    dataset_io.check_and_fix_df,
    dataset_io.resolve_row_id_col,
    dataset_io.coerce_majority_numeric_cols,
    collapse.collapse_rows,
    encoding_state.drop_varying_channels,
    classify.run_classification,
    feature_comparison.superplot_display_frame,
    helpers.log10_columns,
    feature_comparison.sina_offsets,
    feature_comparison.box_statistics,
    feature_comparison._calculate_effect_size,
    feature_comparison.p_value_stars,
    feature_comparison.bracket_positioning_metrics,
    feature_comparison.plan_effect_size_brackets,
    feature_comparison.comparison_layout,
    feature_comparison.comparison_labels,
    feature_histogram.histogram_labels,
    phasor.phasor_columns,
    phasor.resolve_phasor_category,
    distribution_2d.gmm_ellipse_params,
    distribution_2d.distribution_ranges,
    gmm_tables.gmm_group_title,
    dimension_reduction.reduce_dimensions,
    dimension_reduction.dimension_axis_titles,
    dimension_facets.focus_facet_label,
    plot_defaults.panel_point_size,
)


def _module_constants_read_by(func):
    """Find constants referenced by func in its owning module.
    Modules and functions are supplied by the script's imports and other inlined
    helpers, so they are excluded.
    """
    namespace = vars(sys.modules[func.__module__])
    return {
        name for name in func.__code__.co_names
        if name in namespace
        and not isinstance(namespace[name], (types.ModuleType, types.FunctionType, type))
    }


def _state(filename="data.csv"):
    return {
        "csv_filename": filename, "delimiter": ",", "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name", "method": "Feature Comparison",
        "categorical_filters": {}, "numerical_filters": [], "color_by": ["treatment"],
        "opacity_by": None, "shape_by": None, "separate_by": None, "subcolor_by": None,
        "point_size": 5, "axis_label_size": 12, "legend_size": 10, "colormap": "tab10",
        "categorical_cols": ["treatment"],
        # get_features() keeps the row id, every present categorical (the FOV column
        # among them) and every recognised feature.
        "analysis_columns": ["cell_id", "image_name", "treatment", "Lifetime fit_ch1: T1"],
        "method_params": {"selected_var": "Lifetime fit_ch1: T1",
                          "effect_size_method": "None", "statistical_test": "None"},
    }


@pytest.mark.parametrize("helper", INLINED_HELPERS, ids=lambda h: h.__name__)
def test_no_inlined_helper_reads_a_module_level_constant(helper):
    """Every inlined helper has the module-level constants it references."""
    constants = _module_constants_read_by(helper)
    assert constants == set(), (
        f"{helper.__name__} reads module-level {constants}, which the generated "
        "script never defines -- emit the constant alongside the inlined source, "
        "the way the deleted _extract_constants did."
    )


def test_a_script_runs_on_a_file_with_no_fov_column(tmp_path, monkeypatch):
    """No FOV column, and row ids with no "_": nothing is invented."""
    df = pd.DataFrame({"cell_id": ["a", "b", "c", "d"],
                       "treatment": ["ctrl", "drug", "ctrl", "drug"],
                       "Lifetime fit_ch1: T1": [0.40, 0.55, 0.61, 0.48]})
    df.to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "analysis.py").write_text(generate_script(_state()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(tmp_path / "analysis.py"))
    finally:
        plt.close("all")
    # Nothing invented.
    assert "image_name" not in namespace["df"].columns
