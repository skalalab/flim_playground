"""Exported scripts declare the packages needed by their selected analysis."""

import ast
import re
import tomllib

import pytest

from src.export_script import generate_script


_BASE_DEPENDENCIES = {"matplotlib", "numpy", "pandas", "seaborn"}
_METADATA_BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)


def _state(method, **method_params):
    params = {
        "Feature Comparison": {"selected_var": "feature_a"},
        "Feature Histogram": {"selected_var": "feature_a"},
        "2D Feature Distribution": {
            "selected_x": "feature_a", "selected_y": "feature_b",
        },
        "Phasor Plot": {"selected_channel": "Ch1"},
        "Dimension Reduction": {"selected_features": ["feature_a", "feature_b"]},
        "Classification": {
            "selected_features": ["feature_a", "feature_b"],
            "classification_method": "Random Forest",
            "classify_by": ["treatment"], "classify_classes": ["control", "drug"],
        },
    }[method]
    params.update(method_params)
    return {
        "method": method, "csv_filename": "data.csv",
        "unique_row_id_col": "cell_id", "fov_name_col": "image_name",
        "categorical_cols": ["treatment"], "color_by": ["treatment"],
        "categorical_filters": {}, "numerical_filters": [],
        "method_params": params,
    }


def _metadata(script):
    blocks = list(_METADATA_BLOCK.finditer(script))
    assert len(blocks) == 1, "Every export must contain exactly one PEP 723 block"
    block = blocks[0]
    assert block.group("type") == "script"
    assert script.count("# /// script\n") == 1
    content = "\n".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in block.group("content").splitlines()
    )
    metadata = tomllib.loads(content)
    assert metadata["requires-python"] == ">=3.11"
    dependencies = metadata["dependencies"]
    assert dependencies == sorted(set(dependencies))
    assert all(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
               for name in dependencies), "Dependencies must be unpinned package names"
    return metadata


@pytest.mark.parametrize("method,params,extras", [
    pytest.param("Feature Comparison", {}, {"scipy"}, id="comparison"),
    pytest.param("Feature Histogram", {"apply_gmm": False},
                 {"scipy", "scikit-learn", "threadpoolctl"}, id="histogram"),
    pytest.param("Feature Histogram", {"apply_gmm": True},
                 {"scipy", "scikit-learn", "threadpoolctl"}, id="histogram-gmm"),
    pytest.param("2D Feature Distribution", {"fit_regression": False, "fit_gmm_2d": False},
                 {"scipy"}, id="2d"),
    pytest.param("2D Feature Distribution", {"fit_regression": True, "fit_gmm_2d": False},
                 {"scipy", "scikit-learn"}, id="2d-regression"),
    pytest.param("2D Feature Distribution", {"fit_regression": False, "fit_gmm_2d": True},
                 {"scipy", "scikit-learn", "threadpoolctl"}, id="2d-gmm"),
    pytest.param("2D Feature Distribution", {"fit_regression": True, "fit_gmm_2d": True},
                 {"scipy", "scikit-learn", "threadpoolctl"}, id="2d-regression-gmm"),
    pytest.param("Phasor Plot", {}, set(), id="phasor"),
    pytest.param("Dimension Reduction", {}, {"scikit-learn"}, id="dr-default"),
    pytest.param("Dimension Reduction", {"dr_method": "PCA"},
                 {"scikit-learn"}, id="dr-pca"),
    pytest.param("Dimension Reduction", {"dr_method": "UMAP"},
                 {"scikit-learn", "umap-learn"}, id="dr-umap"),
    pytest.param("Dimension Reduction", {"dr_method": "t-SNE"},
                 {"scikit-learn"}, id="dr-tsne"),
    pytest.param("Classification", {"sampling_method": "None"},
                 {"scipy", "scikit-learn"}, id="classification"),
    pytest.param("Classification", {"sampling_method": "Oversampling"},
                 {"scipy", "scikit-learn", "imbalanced-learn"}, id="classification-over"),
    pytest.param("Classification", {"sampling_method": "Undersampling"},
                 {"scipy", "scikit-learn", "imbalanced-learn"}, id="classification-under"),
])
def test_metadata_declares_the_selected_analysis_dependencies(method, params, extras):
    script = generate_script(_state(method, **params))
    assert _metadata(script)["dependencies"] == sorted(_BASE_DEPENDENCIES | extras)


def test_metadata_follows_the_citation_and_precedes_the_script_header():
    script = generate_script(_state("Phasor Plot"))
    _metadata(script)
    block = _METADATA_BLOCK.search(script)
    assert script.startswith("# Citation\n")
    assert script.index("# Software (the latest version):") < block.start()
    assert block.end() < script.index('"""')
    assert script[block.end():].lstrip().startswith('"""')
    assert "uv run analysis.py" in ast.get_docstring(ast.parse(script))


@pytest.mark.parametrize("filename", [
    "book.xlsx", "book.xlsm", "book.xlsb", "book.xls", "book.ods",
    "BOOK.XLSX", "BOOK.XLSM", "BOOK.XLSB", "BOOK.XLS", "BOOK.ODS",
])
def test_spreadsheet_exports_declare_calamine_and_explain_installation(filename):
    state = _state("Phasor Plot")
    state["csv_filename"] = filename
    script = generate_script(state)
    assert _metadata(script)["dependencies"] == sorted(
        _BASE_DEPENDENCIES | {"python-calamine"})
    assert 'df = pd.read_excel(DATA_PATH, sheet_name=0, engine="calamine")' in script
    read_comments = script[script.index("# Reading a spreadsheet"):
                           script.index("df = pd.read_excel(DATA_PATH")]
    assert "uv run" in read_comments and "automatically" in read_comments
    assert "python analysis.py" in read_comments
    assert "pip install python-calamine" in read_comments


@pytest.mark.parametrize("filename", [
    None, "data.csv", "DATA.CSV", "data.tsv", "data.txt", "data", "book.xlsx.csv",
], ids=["default", "csv", "uppercase-csv", "tsv", "txt", "no-suffix", "final-suffix"])
def test_text_exports_do_not_declare_calamine(filename):
    state = _state("Phasor Plot")
    if filename is None:
        state.pop("csv_filename")
    else:
        state["csv_filename"] = filename
    script = generate_script(state)
    assert _metadata(script)["dependencies"] == sorted(_BASE_DEPENDENCIES)
    assert "df = pd.read_csv(DATA_PATH" in script
    if filename is None:
        assert "DATA_PATH = 'data.csv'" in script
