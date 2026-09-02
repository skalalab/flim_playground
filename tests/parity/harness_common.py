"""Shared plumbing for app-vs-export parity checks on the real example datasets.

The app side loads a CSV through the same get_features() pipeline pages/data_analysis.py
uses, then calls the plotting function directly. The export side generates the standalone
script from an equivalent `state` dict and runs it with runpy. Both sides then get poked
for the numbers that are supposed to agree.

See README.md in this directory for how to run these.
"""
import os
import runpy
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# tests/parity/harness_common.py -> tests/parity -> tests -> repo root
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

EXAMPLES = REPO / "example_data" / "Data_Analysis"
# Scratch space for generated scripts and their outputs. Under tests/, so already
# gitignored; safe to delete at any time.
WORK_ROOT = HERE / "_work"


def load_app_df(csv_path, categorical_cols, unique_row_id_col, fov_name_col):
    """Replicate the app's load path: read -> check_and_fix -> get_features prune.

    Mirrors src/dataset_io.py::get_features without the Streamlit/config layer, so the
    frame handed to the plot functions matches what the live app holds in vis_df.
    """
    from src.dataset_io import (
        check_and_fix_df,
        coerce_majority_numeric_cols,
        get_feature_groups_data_extraction,
        resolve_row_id_col,
    )

    df = pd.read_csv(csv_path, index_col=False)
    df, _w, err = check_and_fix_df(df, categorical_cols, unique_row_id_col, fov_name_col)
    assert err == "", err
    # A no-op for every fixture here, which all name a real identifier -- but the app
    # runs it between these two steps, and this function's job is to be that path.
    df, unique_row_id_col = resolve_row_id_col(df, unique_row_id_col)
    skip = set([unique_row_id_col] + list(categorical_cols))
    df, _ = coerce_majority_numeric_cols(df, skip)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    groups = get_feature_groups_data_extraction(numeric_cols)
    all_num = [c for g in groups.values() for c in g]
    avail_cat = [c for c in categorical_cols if c in df.columns]
    return df[[unique_row_id_col] + avail_cat + all_num], groups


def base_state(method, csv_name, categorical_cols, unique_row_id_col="cell_id",
               fov_name_col="image_name", analysis_columns=None, **overrides):
    """Mirror the dict pages/data_analysis.py::_export_script_button collects.

    Keep this in sync with that function — if it grows a key, add it here too, or the
    harness will silently stop exercising it.

    Method-only controls do NOT belong here: they ride `method_params`, the way `log_y`,
    `add_boxplot` and `collapse_by` do for Feature Comparison. Adding one in both places
    is the mistake this note exists to prevent.
    """
    import streamlit as st

    state = {
        "csv_filename": csv_name,
        "unique_row_id_col": unique_row_id_col,
        "fov_name_col": fov_name_col,
        "method": method,
        "categorical_filters": {},
        "numerical_filters": [],
        "color_by": [],
        "opacity_by": None,
        "shape_by": None,
        "separate_by": None,
        "subcolor_by": None,
        "categorical_cols": list(categorical_cols),
        "analysis_columns": analysis_columns,
        "point_size": 5,
        "axis_label_size": 12,
        "legend_size": 10,
        "colormap": "tab10",
        # Read from session state, exactly as _export_script_button does, so a harness
        # case that flips the toggle is captured without also passing it here.
        "show_group_counts": st.session_state.get("plot_show_group_counts", False),
        "method_params": {},
    }
    state.update(overrides)
    return state


def run_export(state, csv_path, workdir, transform=None):
    """Generate the script for `state`, run it against `csv_path`, return its namespace.

    Returns (namespace, script_text). The generated script reads DATA_PATH relative to
    its own working directory, so the CSV is copied in beside it.
    """
    from src.export_script import generate_script

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / state["csv_filename"]
    if not target.exists():
        target.write_bytes(Path(csv_path).read_bytes())
    script = generate_script(state)
    if transform:
        script = transform(script)
    script_path = workdir / "analysis.py"
    script_path.write_text(script)
    cwd = Path.cwd()
    sys.path.insert(0, str(workdir))
    try:
        os.chdir(workdir)
        ns = runpy.run_path(str(script_path))
    finally:
        os.chdir(cwd)
        sys.path.remove(str(workdir))
        plt.close("all")
    return ns, script


def apply_filters(df, state):
    """Apply a state's categorical/numerical filters app-side.

    Mirrors src/export_script.py::_build_filters, which the script emits right after
    loading. The app reaches the same frame through filters_widget(); doing it here keeps
    both sides on identical rows without needing the widget.
    """
    # The app's Operator selectbox offers exactly these two (filter_widgets.py). Anything
    # else cannot come out of the UI, so accepting it here would only let a harness test a
    # combination that never occurs — and quietly pass.
    ops = {">": "gt", "<=": "le"}
    for col, values in (state.get("categorical_filters") or {}).items():
        df = df[df[col].isin(values)]
    for feat, op, thresh in (state.get("numerical_filters") or []):
        if op not in ops:
            raise ValueError(
                f"numerical filter operator {op!r} is not offered by the app; "
                f"use one of {sorted(ops)}")
        # Applied in order, each on the already-filtered frame, as both the widget and
        # the generated script do.
        df = df[getattr(df[feat], ops[op])(thresh)]
    return df


def page_collectors():
    """The real `_collect_categorical_filters` / `_collect_numerical_filters` from
    pages/data_analysis.py, without importing the page.

    Importing the module would execute the whole Streamlit page. Instead the two function
    definitions are lifted out by AST and compiled on their own, so the harness exercises
    the actual capture code the export button relies on rather than a copy of it.
    """
    import ast

    src = (REPO / "pages" / "data_analysis.py").read_text()
    tree = ast.parse(src)
    wanted = {"_collect_categorical_filters", "_collect_numerical_filters"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in defs}
    if missing:
        raise RuntimeError(f"pages/data_analysis.py no longer defines {sorted(missing)}")
    # The page's own module-level imports come along, so a collector that starts calling a
    # newly imported helper keeps working instead of NameError-ing here (selection_key /
    # chosen_items / ALL_LABEL arrived that way with the "Except:" filter mode).
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]

    ns = {}
    code = compile(ast.Module(body=imports + defs, type_ignores=[]),
                   "<page-collectors>", "exec")
    exec(code, ns)  # noqa: S102 - compiling known nodes out of a repo file, not input
    return ns["_collect_categorical_filters"], ns["_collect_numerical_filters"]


def enable_derived(script):
    """Flip the exported script's opt-in constant so it writes its derived-data CSV.

    In the app that CSV comes from a download button; the script ships it behind
    SAVE_DERIVED_DATA = False so running it has no side effects.
    """
    assert "SAVE_DERIVED_DATA = False" in script
    return script.replace("SAVE_DERIVED_DATA = False", "SAVE_DERIVED_DATA = True")


def scatter_points(ax):
    """All scattered (x, y) points on a matplotlib axis."""
    pts = [np.asarray(c.get_offsets(), float) for c in ax.collections
           if len(c.get_offsets())]
    return np.vstack(pts) if pts else np.empty((0, 2))


def sorted_rows(a):
    """Order-insensitive view of an (n, 2) point set, for comparing two clouds."""
    a = np.asarray(a, float)
    if a.size == 0:
        return a
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def app_point_traces(fig, main_axis_only=False):
    """Plotly traces that carry real plotted points.

    Robust to Box/Violin traces, which have no `.mode` attribute at all, and skips the
    shape/opacity legend traces add_point_legend_traces() adds as `x=[None], y=[None]`
    — the export's add_encoding_legend_entries() draws those as empty scatters, so
    counting them makes the app look like it has one extra point per encoding level.
    """
    out = []
    for t in fig.data:
        if "markers" not in (getattr(t, "mode", None) or ""):
            continue
        if t.x is None or not len(t.x):
            continue
        if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in t.x):
            continue
        if main_axis_only and getattr(t, "yaxis", None) not in (None, "y"):
            continue
        out.append(t)
    return out


def mpl_label(var):
    """The axis/title text the export uses (Matplotlib mathtext, not Plotly HTML)."""
    from src.feature_labels import format_feature_label
    return format_feature_label(var, engine="mpl")


class Results:
    """Collects check outcomes and prints them as they happen.

    `known_gap=True` marks a difference that is understood and accepted: it is
    reported but does not fail the run. If such a check starts passing, it is
    reported as FIXED so the flag can be removed.
    """

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.gaps = 0
        self.fixed = 0

    def check(self, label, ok, detail="", known_gap=False):
        ok = bool(ok)
        if known_gap:
            if ok:
                self.fixed += 1
                status = "FIXED"
            else:
                self.gaps += 1
                status = "KNOWN GAP"
        elif ok:
            self.passed += 1
            status = "PASS"
        else:
            self.failed += 1
            status = "FAIL"
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
        return ok

    def summary(self, title):
        bits = [f"{self.passed} passed"]
        if self.failed:
            bits.append(f"{self.failed} FAILED")
        if self.gaps:
            bits.append(f"{self.gaps} known gap(s)")
        if self.fixed:
            bits.append(f"{self.fixed} known gap(s) now FIXED — promote to a normal check")
        print(f"\n{title}: " + ", ".join(bits))
        return self.failed == 0

    @property
    def ok(self):
        return self.failed == 0
