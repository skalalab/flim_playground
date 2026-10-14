"""The script skeleton every method shares: source pasting, header, config, data loading, filters, footer.

Nothing here branches on a method name; method modules compose these pieces.
"""
import inspect
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from src.export_labels import EXPORT_METHODS
from src.export_script.mpl_helpers import (
    add_encoding_legend_entries,
    create_color_map,
    create_shape_map,
    scatter_dimension_batch,
    scatter_interleaved_points,
)
from src.version import get_app_version
from src.vis.plot_defaults import (
    DEFAULT_AXIS_LABEL_FONT_SIZE,
    DEFAULT_COLORMAP,
    DEFAULT_LEGEND_FONT_SIZE,
    DEFAULT_POINT_SIZE,
)

# Prepended to every exported analysis script (same message as README.md # Citation).
_EXPORT_SCRIPT_CITATION = """\
# Citation
#
# If FLIM Playground contributed to your research—whether through Data Extraction for single-cell
# feature extraction or through Data Analysis for data exploration, visualization, selection of
# analysis methods, or hyperparameter tuning (UMAP, clustering, classification, etc.)—please cite
# both the software version you used and the publication. Your citation directly supports us in
# maintaining and improving it ✨🎈🍾.
#
# Publication:
#   Zhao, W., Samimi, K., Skala, M.C., and Datta, R. (2026). FLIM Playground: An interactive,
#   end-to-end graphical user interface for analyzing single cells with fluorescence lifetime
#   imaging microscopy. Cell Rep. Methods. https://doi.org/10.1016/j.crmeth.2026.101484
#
# Software (the latest version):
#   Zhao W., Samimi K., Skala M.C., Datta R. FLIM Playground [Computer software]. Zenodo.
#   https://doi.org/10.5281/zenodo.19744706
#
"""


# Streamlit's Plotly theme draws chart titles at fontSizes.md (1rem = 16 px) whatever the
# axis label size, and its frontend wraps every Plotly title in <b>. A method whose app
# figure sets its own title font declares that size on its spec instead.
_THEME_TITLE_FONT_SIZE = 16


# Follows the imports of every script whose app chart is Plotly: all but Classification,
# which the app draws with Matplotlib's defaults. The app's text renders in a Helvetica-like
# sans that Matplotlib's default DejaVu Sans outgrows by 10-17% at the same size; Arial
# matches within 4%. A family list (not font.sans-serif) lets the subscript digits Arial
# lacks fall back per glyph to DejaVu Sans. Helvetica has them, but hanging below the
# baseline, so it stays out of the list. A missing family would log findfont warnings.
# The axes styling itself is mpl_helpers.style_axes_like_app, pasted by the methods that call it.
_CHART_THEME_SETUP = """
# Match the app's Helvetica-like text; DejaVu Sans (bundled) draws the subscripts Arial lacks.
if "Arial" in {font.name for font in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
"""


@dataclass(frozen=True)
class MethodSpec:
    """What one analysis method tells the shared skeleton about its script.

    The skeleton reads these fields instead of branching on a method name, so a new
    method is one module exposing a SPEC plus one entry in the package registry.
    """
    name: str                                   # display name, the registry key
    build: Callable[[dict], str]                # the method body, emitted after the filters
    config_lines: Callable[[dict], list[str]]   # the method's constants, after the shared ones
    imports: Callable[[dict], list[str]]        # extra import lines; sorted and deduplicated here
    packages: Callable[[dict], set[str]]        # extra PEP 723 dependencies
    footer: Callable[[dict], str | None]        # layout call before savefig; None = the method saves itself
    title_size: int | None = _THEME_TITLE_FONT_SIZE  # None = Matplotlib defaults: no theme, no TITLE_SIZE


def _extract_source(*funcs_or_classes, strip_src_imports=True) -> str:
    """Extract dedented source of functions/classes for injection into generated scripts.

    Args:
        *funcs_or_classes: Functions or classes to extract source from.
        strip_src_imports: If True, remove 'from src.xxx import ...' lines since
            dependencies will be inlined in the same generated script.
    """
    parts = []
    for obj in funcs_or_classes:
        src = inspect.getsource(obj)
        src = textwrap.dedent(src)
        # Standalone scripts have no Streamlit runtime for decorators such as cache_data.
        src = re.sub(r"\A(?:@[^\n]*\n)+", "", src)
        if strip_src_imports:
            # Remove function-local project imports too; their dependencies are inlined.
            src = re.sub(r"^[ \t]*from src\..*$", "", src, flags=re.MULTILINE)
        parts.append(src.strip())
    return "\n\n".join(parts) + "\n"


def _extract_module_source(module) -> str:
    """Inline a module with its imports, constants, and internal dependencies.

    Drop future imports, which cannot occur mid-script, and the module docstring.
    """
    src = inspect.getsource(module)
    src = re.sub(r"^from __future__ import .*$\n?", "", src, flags=re.MULTILINE)
    src = re.sub(r'\A\s*(?:"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')[ \t]*\n', "", src)
    return src.strip() + "\n"


def _script_dependencies(spec: MethodSpec, state: dict) -> list[str]:
    """Select packages for the script's imports and embedded computation helpers."""
    from src.dataset_io import SPREADSHEET_SUFFIXES, suffix_of_name

    dependencies = {"matplotlib", "numpy", "pandas", "seaborn"} | spec.packages(state)
    if suffix_of_name(state.get("csv_filename", "data.csv")) in SPREADSHEET_SUFFIXES:
        dependencies.add("python-calamine")
    return sorted(dependencies)


def _build_preamble(spec: MethodSpec, state: dict) -> str:
    method = state["method"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    base_imports = [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import re",
    ]
    if spec.title_size is not None:
        base_imports.insert(3, "from matplotlib import font_manager")

    imports = "\n".join(base_imports + sorted(set(spec.imports(state))))
    dependencies = ", ".join(f'"{package}"' for package in _script_dependencies(spec, state))
    return (
        f"{_EXPORT_SCRIPT_CITATION}\n"
        '# /// script\n'
        '# requires-python = ">=3.11"\n'
        f'# dependencies = [{dependencies}]\n'
        '# ///\n\n'
        # Record the generating app's version, independent of where the script later runs.
        f'"""\nAuto-generated by FLIM Playground {get_app_version()} \u2014 {method}\nDate: {ts}\n'
        '\nRun with automatic dependency installation: uv run analysis.py\n'
        f'"""\n{imports}\n'
        f"{_CHART_THEME_SETUP if spec.title_size is not None else ''}"
    )


def _build_config_section(spec: MethodSpec, state: dict) -> str:
    lines = [
        "",
        "# " + "=" * 60,
        "# Configuration \u2014 edit these values to customize the analysis",
        "# " + "=" * 60,
        f'DATA_PATH = {state.get("csv_filename", "data.csv")!r}  # Run this script in the same directory as your data',
        f"POINT_SIZE = {state.get('point_size', DEFAULT_POINT_SIZE)}",
        f"AXIS_LABEL_SIZE = {state.get('axis_label_size', DEFAULT_AXIS_LABEL_FONT_SIZE)}",
        f"LEGEND_SIZE = {state.get('legend_size', DEFAULT_LEGEND_FONT_SIZE)}",
        # A method drawn with Matplotlib's defaults (no title size) gets no theme line.
        *([f"TITLE_SIZE = {spec.title_size}  # the app's chart title, bold and left-aligned"]
          if spec.title_size is not None else []),
        f"SHOW_GROUP_COUNTS = {state.get('show_group_counts', False)!r}  # 'Show group counts (n) in legend'",
        f"COLORMAP = {state.get('colormap', DEFAULT_COLORMAP)!r}",
        f"COLOR_BY = {state.get('color_by', [])!r}",
        f"SHAPE_BY = {state.get('shape_by')!r}",
        f"OPACITY_BY = {state.get('opacity_by')!r}",
        # Keep a blank configured ID so the script generates its own row numbers.
        f"UNIQUE_ROW_ID_COL = {state.get('unique_row_id_col', '')!r}",
        f"FOV_NAME_COL = {state.get('fov_name_col')!r}",
        f"CATEGORICAL_COLS = {state.get('categorical_cols', [])!r}",
        # Skip numeric coercion of ignored columns before pruning them, matching the app.
        f"IGNORED_COLS = {state.get('ignored_cols', [])!r}",
    ]

    # Capture the app's retained columns so derived exports match without config files.
    analysis_columns = state.get("analysis_columns")
    if analysis_columns:
        lines.append("ANALYSIS_COLUMNS = [")
        lines.extend(f"    {col!r}," for col in analysis_columns)
        lines.append("]")
    else:
        lines.append("ANALYSIS_COLUMNS = None  # not captured — keep every column")

    if spec.name in EXPORT_METHODS:
        lines.append(
            f"DERIVED_EXPORT = {state.get('derived_export')!r}  # Optional column_name and value_names for derived CSV labels"
        )
    lines.extend(spec.config_lines(state))
    return "\n".join(lines) + "\n"


def _build_read_call(filename: str, delimiter: str = ",") -> str:
    """Build a read call matching dataset_io._read_table_cached's parser and options.

    Keep CSV-only index_col/low_memory settings out of the spreadsheet branch.
    """
    from src.dataset_io import SPREADSHEET_SUFFIXES, suffix_of_name

    suffix = suffix_of_name(filename)
    if suffix in SPREADSHEET_SUFFIXES:
        return ('# Reading a spreadsheet: uv run installs python-calamine automatically.\n'
                '# For python analysis.py, install it manually: pip install python-calamine\n'
                'df = pd.read_excel(DATA_PATH, sheet_name=0, engine="calamine")\n'
                '# A spreadsheet yields each header cell\'s native type; the app stringifies\n'
                '# them so a numeric header is a str everywhere downstream — the categorical\n'
                '# lookup and every df[name] access assume it.\n'
                'df.columns = [str(col) for col in df.columns]')
    # Delimited text reuses the app's detected separator.
    return (f"SEPARATOR = {delimiter!r}  # the separator the app detected\n"
            "df = pd.read_csv(DATA_PATH, index_col=False, sep=SEPARATOR, low_memory=False)")


def _build_data_loading(state: dict) -> str:
    from src.dataset_io import (
        check_and_fix_df,
        coerce_majority_numeric_cols,
        drop_unnamed_columns,
        resolve_row_id_col,
    )
    read_call = _build_read_call(state.get("csv_filename", "data.csv"),
                                 state.get("delimiter", ","))
    loading_src = _extract_source(drop_unnamed_columns,
                                  check_and_fix_df, resolve_row_id_col,
                                  coerce_majority_numeric_cols)
    divider = "# " + "=" * 60
    return f"""
{divider}
# Data Loading — runs the same normalization functions as the app (src/dataset_io.py)
{divider}
{loading_src}

{read_call}
# Same as the app: columns whose header cell was blank are not analysed.
df = drop_unnamed_columns(df)
df, _warning_msg, _error_msg = check_and_fix_df(df, CATEGORICAL_COLS, UNIQUE_ROW_ID_COL, FOV_NAME_COL)
if _error_msg:
    raise SystemExit(_error_msg.strip())
# Same as the app: a blank UNIQUE_ROW_ID_COL means the table has no identifier of its
# own, so one is invented here under the same name the app invented.
df, ROW_ID_COL = resolve_row_id_col(df, UNIQUE_ROW_ID_COL)
# Skip coercion of resolved IDs, categoricals, and ignored columns, matching the app.
df, _coerce_warning = coerce_majority_numeric_cols(
    df, set([ROW_ID_COL] + list(CATEGORICAL_COLS) + list(IGNORED_COLS)))
_warning_msg += _coerce_warning
if ANALYSIS_COLUMNS is not None:
    # Retain captured analysis columns that are present in this file.
    _missing = [col for col in ANALYSIS_COLUMNS if col not in df.columns]
    if _missing:
        print("Warning: analysed column(s) missing from the data file: " + ", ".join(_missing))
    df = df[[col for col in ANALYSIS_COLUMNS if col in df.columns]]
if _warning_msg:
    print(_warning_msg.strip())
"""


def _build_filters(state: dict) -> str:
    cat_filters = state.get("categorical_filters", {})
    num_filters = state.get("numerical_filters", [])
    if not cat_filters and not num_filters:
        return ""
    lines = ["# " + "=" * 60, "# Filters", "# " + "=" * 60]
    for col, values in cat_filters.items():
        lines.append(f"df = df[df[{col!r}].isin({values!r})]")
    for feat, op, thresh in num_filters:
        lines.append(f"df = df[df[{feat!r}] {op} {thresh!r}]")
    return "\n".join(lines) + "\n"

# The run-time check of an edited SEPARATE_BY, shared by the methods that separate panels.
SEPARATE_BY_VALIDATION = """\
if SEPARATE_BY is not None:
    if not isinstance(SEPARATE_BY, str) or SEPARATE_BY not in df.columns:
        raise ValueError("Separate by must be one categorical column present in the data.")
    if SEPARATE_BY in COLOR_BY:
        raise ValueError("Separate by cannot also be used for Color by.")
"""


def _build_derived_export_helpers(spec: MethodSpec) -> str:
    """Share generated labels, name validation and copying with the app."""
    if spec.name not in EXPORT_METHODS:
        return ""
    from src.export_labels import (
        apply_export_labels,
        available_label_column,
        format_export_group_labels,
        normalize_export_labels,
    )

    label_column, filename, _ = EXPORT_METHODS[spec.name]
    return ("\n# Derived CSV labels (extracted from FLIM Playground source)\n"
            + _extract_source(format_export_group_labels, available_label_column, normalize_export_labels,
                              apply_export_labels)
            + f"DERIVED_LABEL_NAME = {label_column!r}\nDERIVED_DATA_FILE = {filename!r}\n")


def _build_visual_encoding(*, overlap_point: bool = True, group_column_expr: str = '"_color_group"',
                           byte_colors: bool = True, interleave: bool = False) -> str:
    """Build visual encoding section by extracting real functions.

    interleave adds the shuffled-batch scatter helpers for methods that draw all groups
    into one axes in the app's global point order.
    """
    from src.natural_sort import natural_key, natural_tuple_sort, tuple_natural_key
    from src.vis.helpers import create_opacity_mapping, format_group_label

    # Include natural-sort dependencies and shared legend wording in Matplotlib format.
    helpers_src = _extract_source(natural_key, tuple_natural_key, natural_tuple_sort,
                                  create_opacity_mapping, format_group_label)
    # Extract Matplotlib-adapted color/shape maps and scatter helpers from this module
    mpl_helpers = [create_color_map, create_shape_map, add_encoding_legend_entries]
    if interleave:
        from src.vis.dimension_facets import dimension_interleaved_indices

        mpl_helpers.extend([dimension_interleaved_indices, scatter_dimension_batch,
                            scatter_interleaved_points])
    mpl_src = _extract_source(*mpl_helpers)

    alpha_expr = "0.6 if len(color_groups) > 1 else 1.0" if overlap_point else "1.0"
    # Effective point alpha = color alpha × 0.8 marker opacity (the app's
    # add_interleaved_points_trace default when no opacity channel is set).
    base_alpha_expr = "(0.6 if len(color_groups) > 1 else 1.0) * 0.8" if overlap_point else "0.8"

    group_code = f"""
# ============================================================
# Visual Encoding Helpers
# ============================================================
{helpers_src}
{mpl_src}

# --- Build groups ---
if COLOR_BY:
    df[{group_column_expr}] = df[COLOR_BY].astype(str).agg("::".join, axis=1)
else:
    df[{group_column_expr}] = "all_data"

color_groups = natural_tuple_sort(df[{group_column_expr}].unique().tolist())
color_map = create_color_map(color_groups, COLORMAP, alpha={alpha_expr},
                             byte_colors={byte_colors})
BASE_ALPHA = {base_alpha_expr}

shape_map = {{}}
if SHAPE_BY:
    shape_groups = natural_tuple_sort(df[SHAPE_BY].unique().astype(str).tolist())
    shape_map = create_shape_map(shape_groups)

opacity_map = {{}}
if OPACITY_BY:
    opacity_groups = natural_tuple_sort(df[OPACITY_BY].unique().astype(str).tolist())
    opacity_map = create_opacity_mapping(opacity_groups)
"""
    return group_code


def _build_footer(spec: MethodSpec, state: dict) -> str:
    layout = spec.footer(state)
    if layout is None:
        return ""
    fname = state["method"].lower().replace(" ", "_")
    return f"""
# ============================================================
# Save & Show
# ============================================================
{layout}
plt.savefig("{fname}.svg", format="svg", bbox_inches="tight")
plt.show()
print("Figure saved to {fname}.svg")
"""


def prepare_rows(state: dict, header: str, channels: tuple, log_block: str) -> str:
    """Emit a method's row preparation: its header, the optional collapse, then its log step.

    Run before _build_visual_encoding so groups and encodings describe the retained
    rows or replicates. Logging after collapse gives log10(mean), matching the app.
    Plain string concatenation preserves the generated f-strings' braces.
    """
    from src.collapse import collapse_rows
    from src.widgets.encoding_state import drop_varying_channels

    if not state.get("method_params", {}).get("collapse_by"):
        return header + log_block

    # Revalidate decoration channels if the user edits COLLAPSE_BY in the script.
    channel_map = ", ".join(f'"{name}": {name}' for name in channels)
    return header + """
# Collapse to one point per replicate (extracted from FLIM Playground source)
""" + _extract_source(collapse_rows, drop_varying_channels) + f"""

df, _label_col, _varied = collapse_rows(
    df, COLLAPSE_BY, [*COLOR_BY, SEPARATE_BY], ROW_ID_COL)
_channels, _dropped = drop_varying_channels({{{channel_map}}}, _varied)
for _channel, _name in _dropped.items():
    print("NOTE: " + _channel + " is off -- one " + str(COLLAPSE_BY)
          + " point covers several " + str(_name)
          + " values, so it cannot be further divided.")
""" + "".join(f'{name} = _channels["{name}"]\n' for name in channels) + log_block
