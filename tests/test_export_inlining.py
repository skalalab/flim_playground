"""Completeness of the inlined-helper mechanism in generated analysis scripts.

`src/export_script.py` reproduces the app by `inspect.getsource()`-ing shared
helpers. getsource copies function *bodies* only — so a helper that reads a
module-level constant compiles fine and then raises NameError at runtime, in a
script advertised as behaving identically to the app.

That is not hypothetical: `safe_split_with_logging` read `_MISSING_FOV_NAME`,
which no generated script ever defined. It stayed latent because the function
only runs when the uploaded file has no FOV column, and the constant is only
reached for a row id containing no "_". Both tests below fail without the fix.
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

from src import dataset_io
from src.export_script import generate_script

# The helpers src/export_script.py::_build_data_loading inlines verbatim.
INLINED_HELPERS = (
    dataset_io.match_col_name,
    dataset_io.safe_split_with_logging,
    dataset_io.drop_unnamed_columns,
    dataset_io.check_and_fix_df,
    dataset_io.coerce_majority_numeric_cols,
)


def _module_constants_read_by(func):
    """Module-level constants `func` closes over — not imports, not other helpers.

    Modules and functions are excluded because the generated script gets those
    from its own import block and from the other inlined helpers.
    """
    namespace = vars(dataset_io)
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
        # get_features() always keeps the row id and the FOV column, so the app's
        # captured column universe carries image_name even when the file lacks it.
        "analysis_columns": ["cell_id", "image_name", "treatment", "Lifetime fit_ch1: T1"],
        "method_params": {"selected_var": "Lifetime fit_ch1: T1",
                          "effect_size_method": "None", "statistical_test": "None"},
    }


@pytest.mark.parametrize("helper", INLINED_HELPERS, ids=lambda h: h.__name__)
def test_every_constant_an_inlined_helper_reads_is_defined_in_the_script(helper):
    """Static guard for the whole mechanism, not just the one constant that broke."""
    script = generate_script(_state())
    for constant in _module_constants_read_by(helper):
        assert f"{constant} = " in script, (
            f"{helper.__name__} reads module-level {constant}, which the generated "
            "script never defines — see _extract_constants in src/export_script.py"
        )


def test_an_emitted_constant_carries_the_apps_own_value():
    """Read off the live module, so the two cannot drift apart."""
    assert f"_MISSING_FOV_NAME = {dataset_io._MISSING_FOV_NAME!r}" in generate_script(_state())


def test_a_script_runs_on_data_that_reaches_the_fov_fallback(tmp_path, monkeypatch):
    """The exact shape that reproduced it: no FOV column, row ids with no "_"."""
    df = pd.DataFrame({"cell_id": ["a", "b", "c", "d"],
                       "treatment": ["ctrl", "drug", "ctrl", "drug"],
                       "Lifetime fit_ch1: T1": [0.40, 0.55, 0.61, 0.48]})
    df.to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "analysis.py").write_text(generate_script(_state()))
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(tmp_path / "analysis.py"))
    finally:
        plt.close("all")
    # The fallback ran and labelled every row, rather than raising NameError.
    assert (namespace["df"]["image_name"] == dataset_io._MISSING_FOV_NAME).all()
