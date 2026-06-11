"""
Script generator for exporting FLIM Playground analyses as self-contained Python scripts.

Each generated script uses Matplotlib for plotting, references the CSV by file path,
and includes all filters, parameters, and analysis logic as editable constants.

Architecture: Uses inspect.getsource() to extract computation functions from the actual
codebase modules (helpers.py, classify.py, tuned_threshold_classifier.py) so algorithm
changes are automatically reflected in exported scripts. Only Matplotlib-specific rendering
code (which has no Plotly equivalent in the codebase) is written as string templates.
"""
import inspect
import re
import textwrap
from datetime import datetime

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


# ---------------------------------------------------------------------------
# Source extraction utility
# ---------------------------------------------------------------------------

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
        if strip_src_imports:
            src = re.sub(r"^from src\..*$", "", src, flags=re.MULTILINE)
        parts.append(src.strip())
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Matplotlib-adapted helper functions (real Python, extracted via getsource)
# These exist because the app's versions return Plotly-specific formats.
# ---------------------------------------------------------------------------

def create_color_map(groups, colormap, alpha=0.8):
    """Map group names to RGBA tuples for Matplotlib."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        if colormap in ("viridis", "plasma", "inferno", "magma", "cividis"):
            cmap = plt.colormaps[colormap]
            if len(groups) == 1:
                palette = [cmap(0.5)]
            else:
                palette = [cmap(i / (len(groups) - 1)) for i in range(len(groups))]
        else:
            palette = sns.color_palette(colormap, n_colors=len(groups))
    except (ValueError, KeyError):
        palette = sns.color_palette("tab10", n_colors=len(groups))
    return {g: (*palette[i][:3], alpha) for i, g in enumerate(groups)}


def create_shape_map(groups):
    """Map group names to Matplotlib marker symbols."""
    MPL_MARKERS = ['o', 's', 'D', 'P', 'X', '^', 'v', 'p', 'h', '8', '*', 'd']
    return {g: MPL_MARKERS[i % len(MPL_MARKERS)] for i, g in enumerate(groups)}


def scatter_with_encodings(ax, x, y, color, label, point_size,
                           shape_vals=None, shape_map=None,
                           opacity_vals=None, opacity_map=None,
                           base_alpha=0.7, linewidths=0.3, zorder=2):
    """Scatter one color group's points, sub-grouped by shape/opacity values.

    Matplotlib cannot vary marker style within a single scatter call, so each
    (shape, opacity) combination becomes its own call — mirroring the app's
    per-point symbol/opacity arrays (helpers.add_interleaved_points_trace).
    Only the first non-empty sub-group carries the legend label.
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        ax.scatter([], [], c=[color], s=point_size, alpha=base_alpha,
                   edgecolors='DarkSlateGrey', linewidths=linewidths,
                   label=label, zorder=zorder)
        return
    use_shape = shape_vals is not None and bool(shape_map)
    use_opacity = opacity_vals is not None and bool(opacity_map)
    shape_arr = np.asarray([str(v) for v in shape_vals]) if use_shape else None
    opacity_arr = np.asarray([str(v) for v in opacity_vals]) if use_opacity else None
    labeled = False
    for shape_key in (list(shape_map) if use_shape else [None]):
        for opacity_key in (list(opacity_map) if use_opacity else [None]):
            mask = np.ones(len(x), dtype=bool)
            if use_shape:
                mask &= shape_arr == shape_key
            if use_opacity:
                mask &= opacity_arr == opacity_key
            if not mask.any():
                continue
            ax.scatter(x[mask], y[mask], c=[color], s=point_size,
                       alpha=opacity_map[opacity_key] if use_opacity else base_alpha,
                       marker=shape_map[shape_key] if use_shape else 'o',
                       edgecolors='DarkSlateGrey', linewidths=linewidths,
                       label=label if not labeled else None, zorder=zorder)
            labeled = True


def add_encoding_legend_entries(ax, shape_map, opacity_map, point_size):
    """Add gray proxy legend entries for opacity and shape groups
    (mirrors the app's helpers.add_point_legend_traces: opacity first, then shape)."""
    for group, alpha in (opacity_map or {}).items():
        ax.scatter([], [], c='gray', alpha=alpha, marker='o', s=point_size, label=str(group))
    for group, marker in (shape_map or {}).items():
        ax.scatter([], [], c='gray', alpha=0.8, marker=marker, s=point_size, label=str(group))


# ---------------------------------------------------------------------------
# State-capture helpers (used by pages/data_analysis.py)
# ---------------------------------------------------------------------------

def get_effect_size_threshold_capture(session_state, effect_size_method, selected_var, separate_by):
    """Read the effect-size threshold the app's widgets wrote to session state.

    Mirrors the widget keys and defaults in src/vis/helpers.py (non-separate
    path) and src/vis/univar.py (separate path): the key suffix is the selected
    variable; Glass's Delta defaults to 0.7 on both paths, Absolute Cohen's d
    defaults to 0.7 (0.5 when separate_by is active).
    """
    if effect_size_method == "Glass's Delta":
        return float(session_state.get(f"glass_delta_thresh_{selected_var}", 0.7))
    if effect_size_method == "Absolute Cohen's d":
        default = 0.5 if separate_by else 0.7
        return float(session_state.get(f"cohens_d_thresh_{selected_var}", default))
    return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_script(state: dict) -> str:
    """Generate a self-contained Python script from the current analysis state."""
    method = state["method"]
    parts = [
        _build_preamble(state),
        _build_config_section(state),
        _build_data_loading(state),
        _build_filters(state),
    ]

    builders = {
        "Feature Comparison": _build_feature_comparison,
        "Feature Histogram": _build_feature_histogram,
        "FOV Comparison": _build_fov_comparison,
        "2D Feature Distribution": _build_2d_distribution,
        "Phasor Plot": _build_phasor_plot,
        "Dimension Reduction": _build_dimension_reduction,
        "Classification": _build_classification,
    }
    builder = builders.get(method)
    if builder is None:
        parts.append(f'\nprint("Export not yet supported for method: {method}")\n')
    else:
        parts.append(builder(state))

    parts.append(_build_footer(state))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _build_preamble(state: dict) -> str:
    method = state["method"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    base_imports = [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import re",
    ]

    extra = []
    mp = state.get("method_params", {})

    if method == "Feature Comparison":
        extra.append("from scipy.stats import gaussian_kde, ttest_ind, median_abs_deviation")
    elif method == "Feature Histogram":
        if mp.get("apply_gmm"):
            extra += ["from sklearn.mixture import GaussianMixture",
                      "from scipy.stats import norm", "from scipy.optimize import brentq"]
    elif method == "2D Feature Distribution":
        extra.append("from scipy.stats import gaussian_kde")
        # Pearson r + p is reported for every color group (like the app), so import
        # it unconditionally; only the regression line itself is gated.
        extra.append("from scipy.stats import pearsonr")
        if mp.get("fit_regression"):
            extra.append("from sklearn.linear_model import LinearRegression")
        if mp.get("fit_gmm_2d"):
            extra += ["from sklearn.mixture import GaussianMixture",
                      "from matplotlib.patches import Ellipse", "from scipy.stats import chi2"]
    elif method == "Phasor Plot":
        if mp.get("k_means"):
            extra += ["from sklearn.cluster import KMeans",
                      "from sklearn.preprocessing import StandardScaler",
                      "from scipy.spatial import ConvexHull"]
    elif method == "Dimension Reduction":
        extra.append("from sklearn.preprocessing import StandardScaler")
        dr = mp.get("dr_method", "PCA")
        if dr == "PCA":
            extra.append("from sklearn.decomposition import PCA")
        elif dr == "UMAP":
            extra.append("import umap")
        elif dr == "t-SNE":
            extra.append("from sklearn.manifold import TSNE")
    elif method == "Classification":
        extra += [
            "from sklearn.model_selection import train_test_split, StratifiedKFold",
            "from sklearn.preprocessing import StandardScaler, label_binarize",
            "from sklearn.pipeline import make_pipeline",
            "from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, f1_score, accuracy_score, get_scorer",
            "from sklearn.base import clone",
            "from scipy.optimize import minimize",
            "from copy import deepcopy",
        ]
        clf = mp.get("classification_method", "")
        if "Random Forest" in clf:
            extra.append("from sklearn.ensemble import RandomForestClassifier")
        elif "Gradient Boosting" in clf:
            extra.append("from sklearn.ensemble import GradientBoostingClassifier")
        elif "SVM" in clf:
            extra.append("from sklearn.svm import SVC")
        elif "Logistic Regression" in clf:
            extra.append("from sklearn.linear_model import LogisticRegression")
        sampling = mp.get("sampling_method", "None")
        if sampling == "Oversampling":
            extra.append("from imblearn.over_sampling import RandomOverSampler")
        elif sampling == "Undersampling":
            extra.append("from imblearn.under_sampling import RandomUnderSampler")

    imports = "\n".join(base_imports + sorted(set(extra)))
    return (
        f"{_EXPORT_SCRIPT_CITATION}\n"
        f'"""\nAuto-generated by FLIM Playground \u2014 {method}\nDate: {ts}\n"""\n{imports}\n'
    )


def _build_config_section(state: dict) -> str:
    mp = state.get("method_params", {})
    method = state["method"]
    lines = [
        "",
        "# " + "=" * 60,
        "# Configuration \u2014 edit these values to customize the analysis",
        "# " + "=" * 60,
        f'CSV_PATH = {state.get("csv_filename", "data.csv")!r}  # Run this script in the same directory as your data',
        f"POINT_SIZE = {state.get('point_size', DEFAULT_POINT_SIZE)}",
        f"AXIS_LABEL_SIZE = {state.get('axis_label_size', DEFAULT_AXIS_LABEL_FONT_SIZE)}",
        f"LEGEND_SIZE = {state.get('legend_size', DEFAULT_LEGEND_FONT_SIZE)}",
        f"COLORMAP = {state.get('colormap', DEFAULT_COLORMAP)!r}",
        f"COLOR_BY = {state.get('color_by', [])!r}",
        f"SHAPE_BY = {state.get('shape_by')!r}",
        f"OPACITY_BY = {state.get('opacity_by')!r}",
        f"UNIQUE_ROW_ID_COL = {state.get('unique_row_id_col', 'cell_id')!r}",
        f"FOV_NAME_COL = {state.get('fov_name_col', 'image_name')!r}",
        f"CATEGORICAL_COLS = {state.get('categorical_cols', [])!r}",
    ]

    if method in ("Feature Comparison", "Feature Histogram", "FOV Comparison"):
        lines.append(f"SELECTED_VAR = {mp.get('selected_var')!r}")
    if method == "Feature Comparison":
        lines.append(f"SEPARATE_BY = {state.get('separate_by')!r}")
        lines.append(f"EFFECT_SIZE_METHOD = {mp.get('effect_size_method', 'None')!r}")
        lines.append(f"MEAN_OR_MEDIAN = {mp.get('mean_or_median')!r}")
        lines.append(f"STATISTICAL_TEST = {mp.get('statistical_test', 'None')!r}")
        lines.append(f"LOG_Y = {mp.get('log_y', False)!r}")
        lines.append(f"ADD_BOXPLOT = {mp.get('add_boxplot', False)!r}")
        lines.append(f"CONNECT_MEANS = {mp.get('connect_means', False)!r}")
        lines.append(f"EFFECT_SIZE_THRESHOLD = {mp.get('effect_size_threshold', 0.0)!r}")
        lines.append(f"SELECTED_PAIRS = {mp.get('selected_pairs')!r}  # None → annotate all pairs; else list of 'group1 vs group2' labels")
        custom_order = mp.get("custom_order")
        if custom_order:
            lines.append(f"CUSTOM_ORDER = {custom_order!r}  # Reorder: compare_groups and/or separate_groups")
        else:
            lines.append("CUSTOM_ORDER = None  # Set to {'compare_groups': ['group1', 'group2', ...]} to reorder x-axis")
    elif method == "Feature Histogram":
        lines.append(f"LOG_X = {mp.get('log_x', False)!r}")
        lines.append(f"APPLY_GMM = {mp.get('apply_gmm', False)!r}")
        lines.append(f"INTERSECTION_THRESHOLD = {mp.get('intersection_threshold', False)!r}")
        lines.append(f"BIN_WIDTH = {mp.get('bin_width')!r}  # None → numpy 'auto' bin width")
        lines.append(f"GMM_MAX_COMPONENTS = {mp.get('gmm_max_components', 3)!r}")
        lines.append(f"GMM_MIN_WEIGHT_THRESHOLD = {mp.get('gmm_min_weight_threshold', 0.1)!r}")
        if mp.get("apply_gmm"):
            lines.append("SAVE_DERIVED_DATA = False  # True → also write gmm_grouped_data.csv (the app's download button)")
    elif method == "2D Feature Distribution":
        lines.append(f"SELECTED_X = {mp.get('selected_x')!r}")
        lines.append(f"SELECTED_Y = {mp.get('selected_y')!r}")
        lines.append(f"LOG_X = {mp.get('log_x', False)!r}")
        lines.append(f"LOG_Y = {mp.get('log_y', False)!r}")
        lines.append(f"MARGINAL_PLOT_TYPE = {mp.get('marginal_plot_type', 'gaussian fit')!r}")
        lines.append(f"FIT_REGRESSION = {mp.get('fit_regression', False)!r}")
        lines.append(f"FIT_GMM_2D = {mp.get('fit_gmm_2d', False)!r}")
        lines.append(f"GMM_MAX_COMPONENTS = {mp.get('gmm_max_components', 3)!r}")
        lines.append(f"GMM_MIN_WEIGHT_THRESHOLD = {mp.get('gmm_min_weight_threshold', 0.1)!r}")
        if mp.get("fit_gmm_2d"):
            lines.append("SAVE_DERIVED_DATA = False  # True → also write 2D_gmm_data.csv (the app's download button)")
    elif method == "Phasor Plot":
        lines.append(f"PHASOR_CHANNEL = {mp.get('selected_channel')!r}")
        lines.append(f"PHASOR_HARMONIC = {mp.get('phasor_harmonic', 1)!r}")
        lines.append(f"PHASOR_F = {mp.get('phasor_f', 0.08)!r}")
        lines.append(f"K_MEANS = {mp.get('k_means', False)!r}")
        lines.append(f"K_MEANS_CLUSTERS = {mp.get('k_means_clusters', 2)!r}")
        if mp.get("k_means"):
            lines.append("SAVE_DERIVED_DATA = False  # True → also write kmeans_clustered_data.csv (the app's download button)")
    elif method == "Dimension Reduction":
        lines.append(f"SELECTED_FEATURES = {mp.get('selected_features', [])!r}")
        lines.append(f"DR_METHOD = {mp.get('dr_method', 'PCA')!r}")
        lines.append(f"HYPER_PARAMS = {mp.get('hyperParam_dict', {})!r}")
    elif method == "Classification":
        lines.append(f"SELECTED_FEATURES = {mp.get('selected_features', [])!r}")
        lines.append(f"CLASSIFICATION_METHOD = {mp.get('classification_method')!r}")
        lines.append(f"TRAIN_SIZE = {mp.get('splits', 0.7)!r}")
        lines.append(f"SAMPLING_METHOD = {mp.get('sampling_method', 'None')!r}")
        lines.append(f"CLASS_WEIGHT = {mp.get('class_weight', 'None')!r}")
        lines.append(f"THRESHOLD_METHOD = {mp.get('threshold_method', 'None')!r}")
        lines.append(f"CLASSIFIER_PARAMS = {mp.get('classifier_params', {})!r}")
        lines.append(f"CLASSIFY_BY = {mp.get('classify_by', [])!r}  # Categorical columns used to build class labels")
        lines.append(f"CLASSIFY_CLASSES = {mp.get('classify_classes', [])!r}  # Selected class values to include")
        lines.append("RANDOM_STATE = 42")

    return "\n".join(lines) + "\n"


def _build_data_loading(state: dict) -> str:
    from src.dataset_io import (
        check_and_fix_df,
        coerce_majority_numeric_cols,
        match_col_name,
        safe_split_with_logging,
    )

    loading_src = _extract_source(match_col_name, safe_split_with_logging,
                                  check_and_fix_df, coerce_majority_numeric_cols)
    divider = "# " + "=" * 60
    return f"""
{divider}
# Data Loading — runs the same normalization functions as the app (src/dataset_io.py)
{divider}
{loading_src}

df = pd.read_csv(CSV_PATH, index_col=False, low_memory=False)
df, _warning_msg, _error_msg = check_and_fix_df(df, CATEGORICAL_COLS, UNIQUE_ROW_ID_COL, FOV_NAME_COL)
if _error_msg:
    raise SystemExit(_error_msg.strip())
df, _coerce_warning = coerce_majority_numeric_cols(
    df, set([UNIQUE_ROW_ID_COL, FOV_NAME_COL] + list(CATEGORICAL_COLS)))
_warning_msg += _coerce_warning
if _warning_msg:
    print(_warning_msg.replace("<br>", "\\n").strip())
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


def _build_visual_encoding(state: dict, overlap_point: bool = True) -> str:
    """Build visual encoding section by extracting real functions."""
    from src.vis.helpers import natural_key, tuple_natural_key, natural_tuple_sort, create_opacity_mapping

    # Extract computation from actual source (include tuple_natural_key which natural_tuple_sort depends on)
    helpers_src = _extract_source(natural_key, tuple_natural_key, natural_tuple_sort, create_opacity_mapping)
    # Extract Matplotlib-adapted color/shape maps and scatter helpers from this module
    mpl_src = _extract_source(create_color_map, create_shape_map,
                              scatter_with_encodings, add_encoding_legend_entries)

    alpha_expr = "0.6 if len(color_groups) > 1 else 1.0" if overlap_point else "1.0"

    group_code = f"""
# ============================================================
# Visual Encoding Helpers
# ============================================================
{helpers_src}
{mpl_src}

# --- Build groups ---
if COLOR_BY:
    df["_color_group"] = df[COLOR_BY].astype(str).agg("::".join, axis=1)
else:
    df["_color_group"] = "all_data"

color_groups = natural_tuple_sort(df["_color_group"].unique().tolist())
color_map = create_color_map(color_groups, COLORMAP, alpha={alpha_expr})

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


def _build_footer(state: dict) -> str:
    method = state["method"]
    if method == "Classification":
        return ""
    fname = method.lower().replace(" ", "_")
    return f"""
# ============================================================
# Save & Show
# ============================================================
plt.tight_layout()
plt.savefig("{fname}.svg", format="svg", bbox_inches="tight")
plt.show()
print("Figure saved to {fname}.svg")
"""


# ---------------------------------------------------------------------------
# Method builders — use _extract_source() for computation, string templates
# only for Matplotlib-specific rendering that has no codebase equivalent.
# ---------------------------------------------------------------------------

def _build_fov_comparison(state: dict) -> str:
    fov_col = state.get("fov_name_col", "image_name")
    return _build_visual_encoding(state, overlap_point=False) + f"\nfov_col = {fov_col!r}\n" + """
# ============================================================
# FOV Comparison — Box Plots per FOV
# ============================================================

df = df[df[SELECTED_VAR].notna()]

fig, ax = plt.subplots(figsize=(12, 6))

# FOVs in CSV appearance order, matching the app (univar.py: df[fov_name_col].unique())
fovs = df[fov_col].unique().tolist()
positions = []
tick_labels = []

offset = 0
for fov_i, fov in enumerate(fovs):
    fov_df = df[df[fov_col] == fov]
    group_data = []
    group_colors = []
    for g in color_groups:
        gdf = fov_df[fov_df["_color_group"] == g]
        group_data.append(gdf[SELECTED_VAR].dropna().values)
        group_colors.append(color_map[g][:3])

    bp = ax.boxplot(group_data, positions=list(range(offset, offset + len(color_groups))),
                    widths=0.6, patch_artist=True, manage_ticks=False)
    for patch, c in zip(bp['boxes'], group_colors):
        patch.set_facecolor((*c, 0.5))
        patch.set_edgecolor(c)
    for element in ('whiskers', 'caps', 'medians'):
        for line in bp[element]:
            line.set_color('black')

    tick_labels.extend([f"{fov}\\n{g}" for g in color_groups])
    positions.extend(range(offset, offset + len(color_groups)))
    offset += len(color_groups) + 1

ax.set_xticks(positions)
ax.set_xticklabels(tick_labels, fontsize=max(6, AXIS_LABEL_SIZE - 6), rotation=45, ha='right')
ax.set_ylabel(SELECTED_VAR, fontsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='y', labelsize=LEGEND_SIZE)

for g in color_groups:
    ax.scatter([], [], c=[color_map[g][:3]], label=g, s=50)
ax.legend(fontsize=LEGEND_SIZE)
"""


def _build_feature_histogram(state: dict) -> str:
    from src.vis.helpers import _find_best_gmm, find_intersection

    has_gmm = state.get("method_params", {}).get("apply_gmm", False)

    if has_gmm:
        gmm_src = _extract_source(_find_best_gmm, find_intersection)
        return _build_visual_encoding(state, overlap_point=False) + f"""
# ============================================================
# Gaussian Mixture Model Fit (extracted from FLIM Playground)
# ============================================================
{gmm_src}

df = df[df[SELECTED_VAR].notna()]

if LOG_X:
    if (df[SELECTED_VAR] < 0).any():
        print(f"WARNING: Cannot apply log to {{SELECTED_VAR}}: contains negative values.")
    else:
        df[SELECTED_VAR] = np.log10(df[SELECTED_VAR] + 1e-6)

fig, ax = plt.subplots(figsize=(10, 6))

# "GMM_group" is created by the .loc assignments below (object dtype, NaN for
# unassigned rows) — same as the app; pre-seeding it as float would make the
# string-label assignment a pandas FutureWarning/error.

for g in color_groups:
    group_mask = df["_color_group"] == g
    gdata = df.loc[group_mask, SELECTED_VAR].dropna()
    if len(gdata) < 3:
        continue
    gmm = _find_best_gmm(gdata.values, max_components=GMM_MAX_COMPONENTS,
                         min_weight_threshold=GMM_MIN_WEIGHT_THRESHOLD)
    if gmm is None:
        print(f"  {{g}}: No valid GMM found with current constraints.")
        continue

    x_range = np.linspace(gdata.min(), gdata.max(), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x_range)
    pdf = np.exp(logprob)
    responsibilities = gmm.predict_proba(x_range)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.plot(x_range.flatten(), pdf, color=color_map[g][:3], linewidth=2, label=f"{{g}} GMM")

    pi = gmm.weights_
    mu = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_.ravel())
    sorted_idx = np.argsort(mu)

    print(f"  {{g}}: Best GMM has {{gmm.n_components}} component(s)")
    print(f"    | Component | Mean     | Std. Dev. | Weight |")
    print(f"    |-----------|----------|-----------|--------|")
    for rank, idx in enumerate(sorted_idx):
        print(f"    | {{rank+1}}         | {{mu[idx]:.4f}} | {{sigma[idx]:.4f}}  | {{pi[idx]:.3f}}  |")

    if gmm.n_components > 1:
        dash_styles = ['--', ':', '-.', (0, (5, 10)), (0, (3, 5, 1, 5))]
        gmm_overall_mean = np.sum(pi * mu)
        means_std = np.std(mu, ddof=1)
        h_index = 0.0

        for rank, idx in enumerate(sorted_idx):
            ax.plot(x_range.flatten(), pdf_individual[:, idx],
                   linestyle=dash_styles[rank % len(dash_styles)],
                   color=color_map[g][:3], alpha=0.6, linewidth=1.5,
                   label=f"{{g}} Component {{rank+1}}")
            entropy_term = -pi[idx] * np.log(pi[idx])
            distance_term = np.abs(mu[idx] - gmm_overall_mean) / means_std if means_std > 0 else 0
            h_index += entropy_term * distance_term

        print(f"    H-index: {{h_index:.3f}}")

        pi_sorted, mu_sorted, sigma_sorted = pi[sorted_idx], mu[sorted_idx], sigma[sorted_idx]
        data_indices = gdata.index

        intersection_ok = INTERSECTION_THRESHOLD
        thresholds = []
        if INTERSECTION_THRESHOLD:
            for i in range(len(mu_sorted) - 1):
                try:
                    t = find_intersection(pi_sorted[i], mu_sorted[i], sigma_sorted[i],
                                          pi_sorted[i+1], mu_sorted[i+1], sigma_sorted[i+1])
                    thresholds.append(t)
                except Exception:
                    print(f"    Warning: No intersection found between components {{i+1}} and {{i+2}}, using hard assignment.")
                    intersection_ok = False
                    break

            if intersection_ok:
                thresholds = np.sort(thresholds)
                for i, t in enumerate(thresholds):
                    ax.axvline(x=t, color=color_map[g][:3], linestyle='--', alpha=0.5, linewidth=2)
                    ax.text(t, ax.get_ylim()[1] * 0.95, f"Threshold: {{t:.2f}}",
                           ha='center', fontsize=9, color=color_map[g][:3])
                    print(f"    Threshold between component {{i+1}} and {{i+2}}: {{t:.4f}}")
                subpopulation_labels = np.digitize(gdata.values, bins=thresholds)
                subpopulation_labels = sorted_idx[subpopulation_labels]
        if not intersection_ok:
            data_2d = gdata.values.reshape(-1, 1)
            subpopulation_labels = gmm.predict(data_2d)

        assigned_labels = [f"{{g}}_group{{label + 1}}" for label in subpopulation_labels]
        df.loc[data_indices, "GMM_group"] = assigned_labels
    # No else: a single-component group is left unlabeled, matching the app
    # (univar.py assigns GMM_group only inside the n_components>1 branch).

if SAVE_DERIVED_DATA:
    df.drop(columns=["_color_group"]).to_csv("gmm_grouped_data.csv", index=False)
    print("GMM grouped data saved to gmm_grouped_data.csv")

ax.set_xlabel(SELECTED_VAR, fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("Probability Density", fontsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=LEGEND_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
"""

    # No GMM — plain histogram
    return _build_visual_encoding(state, overlap_point=False) + """
# ============================================================
# Feature Histogram
# ============================================================
df = df[df[SELECTED_VAR].notna()]

if LOG_X:
    if (df[SELECTED_VAR] < 0).any():
        print(f"WARNING: Cannot apply log to {SELECTED_VAR}: contains negative values.")
    else:
        df[SELECTED_VAR] = np.log10(df[SELECTED_VAR] + 1e-6)

fig, ax = plt.subplots(figsize=(10, 6))

all_vals = df[SELECTED_VAR].dropna().values
if BIN_WIDTH is not None:
    bin_width = float(BIN_WIDTH)
else:
    _, _auto_edges = np.histogram(all_vals, bins='auto')
    bin_width = _auto_edges[1] - _auto_edges[0] if len(_auto_edges) > 1 else 1.0
# Common bin edges shared by all groups (mirrors the app's histogram widget)
bin_edges = np.arange(all_vals.min(), all_vals.max() + bin_width + 1e-9, bin_width)

for g in color_groups:
    gdata = df[df["_color_group"] == g][SELECTED_VAR].dropna().values
    if len(gdata) == 0:
        continue
    counts, edges = np.histogram(gdata, bins=bin_edges)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, counts, label=g, color=color_map[g][:3], linewidth=2)

    # Bias-corrected skewness (pandas .skew()) + the app's 7-way label ladder
    # (univar.py:74-87) — keep both identical to the app.
    sk = pd.Series(gdata).skew()
    if sk < -1:
        desc = "strongly left-skewed"
    elif sk < -0.5:
        desc = "moderately left-skewed"
    elif sk < -0.25:
        desc = "approximately symmetric"
    elif sk <= 0.25:
        desc = "almost symmetric"
    elif sk <= 0.5:
        desc = "approximately symmetric"
    elif sk <= 1:
        desc = "moderately right-skewed"
    else:
        desc = "strongly right-skewed"
    print(f"  {g}: skewness = {sk:.3f} ({desc})")

ax.set_xlabel(SELECTED_VAR, fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=LEGEND_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
"""


def _build_feature_comparison(state: dict) -> str:
    from src.vis.helpers import glass_delta, cohens_d, _compute_bracket_position

    effect_size_src = _extract_source(glass_delta, cohens_d, _compute_bracket_position)

    return _build_visual_encoding(state) + f"""
# ============================================================
# Feature Comparison — Sina Plot
# ============================================================
from scipy.stats import gaussian_kde, ttest_ind, median_abs_deviation

# Effect size + bracket positioning functions (extracted from FLIM Playground source)
{effect_size_src}

df = df[df[SELECTED_VAR].notna()]
if LOG_Y:
    df = df.copy()
    df[SELECTED_VAR] = np.log10(df[SELECTED_VAR] + 1e-6)

# --- Separate_by logic ---
separate_groups = [None]
if SEPARATE_BY:
    separate_groups = natural_tuple_sort(df[SEPARATE_BY].unique().astype(str).tolist())

# --- Apply custom ordering ---
ordered_color_groups = list(color_groups)
if CUSTOM_ORDER and 'compare_groups' in CUSTOM_ORDER:
    custom_cmp = [g for g in CUSTOM_ORDER['compare_groups'] if g in ordered_color_groups]
    remaining = [g for g in ordered_color_groups if g not in custom_cmp]
    ordered_color_groups = custom_cmp + remaining

ordered_separate_groups = list(separate_groups)
if CUSTOM_ORDER and 'separate_groups' in CUSTOM_ORDER and separate_groups != [None]:
    custom_sep = [g for g in CUSTOM_ORDER['separate_groups'] if g in ordered_separate_groups]
    remaining = [g for g in ordered_separate_groups if g not in custom_sep]
    ordered_separate_groups = custom_sep + remaining

# Build x-positions using ordered groups
x_positions = {{}}
x_labels = []
tick_positions = []
pos = 0
section_boundaries = []

for sec_i, sec_group in enumerate(ordered_separate_groups):
    if sec_group is not None:
        sec_df = df[df[SEPARATE_BY] == sec_group]
        sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df["_color_group"].unique()]
    else:
        sec_df = df
        sec_color_groups = ordered_color_groups

    if sec_i > 0:
        section_boundaries.append(pos - 0.5)
        pos += 1

    for cg in sec_color_groups:
        key = (sec_group, cg) if sec_group is not None else cg
        x_positions[key] = pos
        label = f"{{sec_group}}\\n{{cg}}" if sec_group is not None else cg
        x_labels.append(label)
        tick_positions.append(pos)
        pos += 1

fig, ax = plt.subplots(figsize=(max(10, len(tick_positions) * 1.2), 6))

# --- Plot points (Sina jitter) ---
legend_entries = set()
for sec_group in ordered_separate_groups:
    if sec_group is not None:
        sec_df = df[df[SEPARATE_BY] == sec_group]
        sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df["_color_group"].unique()]
    else:
        sec_df = df
        sec_color_groups = ordered_color_groups

    for cg in sec_color_groups:
        key = (sec_group, cg) if sec_group is not None else cg
        x_pos = x_positions[key]
        group_df = sec_df[sec_df["_color_group"] == cg]
        y_data = group_df[SELECTED_VAR].values

        if len(y_data) < 2:
            scatter_with_encodings(ax, [x_pos] * len(y_data), y_data, color_map[cg][:3],
                                   cg if cg not in legend_entries else None, POINT_SIZE,
                                   shape_vals=group_df[SHAPE_BY] if SHAPE_BY else None, shape_map=shape_map,
                                   opacity_vals=group_df[OPACITY_BY] if OPACITY_BY else None, opacity_map=opacity_map,
                                   base_alpha=1.0, linewidths=0.5)
            legend_entries.add(cg)
            continue

        # KDE-based jitter (Sina plot)
        try:
            kde = gaussian_kde(y_data)
            densities = kde(y_data)
            max_d = densities.max()
            norm_d = densities / max_d if max_d > 0 else np.zeros_like(densities)
        except Exception:
            norm_d = np.zeros(len(y_data))

        rng = np.random.default_rng(42)
        jitter = rng.uniform(-1, 1, len(y_data)) * norm_d * 0.35

        scatter_with_encodings(ax, x_pos + jitter, y_data, color_map[cg][:3],
                               cg if cg not in legend_entries else None, POINT_SIZE,
                               shape_vals=group_df[SHAPE_BY] if SHAPE_BY else None, shape_map=shape_map,
                               opacity_vals=group_df[OPACITY_BY] if OPACITY_BY else None, opacity_map=opacity_map,
                               linewidths=0.5)
        legend_entries.add(cg)

# --- Boxplot overlay ---
if ADD_BOXPLOT:
    for sec_group in ordered_separate_groups:
        if sec_group is not None:
            sec_df = df[df[SEPARATE_BY] == sec_group]
            sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df["_color_group"].unique()]
        else:
            sec_df = df
            sec_color_groups = ordered_color_groups

        for cg in sec_color_groups:
            key = (sec_group, cg) if sec_group is not None else cg
            x_pos = x_positions[key]
            gdata = sec_df[sec_df["_color_group"] == cg][SELECTED_VAR].dropna().values
            if len(gdata) == 0:
                continue
            bp = ax.boxplot([gdata], positions=[x_pos], widths=0.5, patch_artist=True,
                          manage_ticks=False, showfliers=False, zorder=1)
            bp['boxes'][0].set_facecolor('none')
            bp['boxes'][0].set_edgecolor('black')
            bp['medians'][0].set_color('black')
            bp['medians'][0].set_linewidth(2)

# --- Connect means ---
if CONNECT_MEANS:
    for sec_group in ordered_separate_groups:
        if sec_group is not None:
            sec_df = df[df[SEPARATE_BY] == sec_group]
            sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df["_color_group"].unique()]
        else:
            sec_df = df
            sec_color_groups = ordered_color_groups

        means_x, means_y = [], []
        for cg in sec_color_groups:
            key = (sec_group, cg) if sec_group is not None else cg
            x_pos = x_positions[key]
            gdata = sec_df[sec_df["_color_group"] == cg][SELECTED_VAR].dropna()
            if len(gdata) > 0:
                means_x.append(x_pos)
                means_y.append(gdata.mean())
        if len(means_x) > 1:
            ax.plot(means_x, means_y, 'k-o', linewidth=2, markersize=6, zorder=3)

# --- Section dividers ---
for boundary in section_boundaries:
    ax.axvline(x=boundary, linestyle='--', color='gray', alpha=0.5, linewidth=1)

# --- Effect size / statistical test annotations ---
if EFFECT_SIZE_METHOD != "None" or STATISTICAL_TEST != "None":
    from itertools import combinations
    for sec_group in ordered_separate_groups:
        if sec_group is not None:
            sec_df = df[df[SEPARATE_BY] == sec_group]
            sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df["_color_group"].unique()]
        else:
            sec_df = df
            sec_color_groups = ordered_color_groups

        if len(sec_color_groups) < 2:
            continue

        pairs = list(combinations(sec_color_groups, 2))
        if SELECTED_PAIRS is not None:
            pairs = [p for p in pairs
                     if f"{{p[0]}} vs {{p[1]}}" in SELECTED_PAIRS
                     or f"{{p[1]}} vs {{p[0]}}" in SELECTED_PAIRS]
        all_y = sec_df[SELECTED_VAR].dropna()
        data_range = all_y.max() - all_y.min()
        if data_range == 0:
            data_range = 1.0

        positioning_metrics = {{
            'offset_from_data_abs': 0.05 * data_range,
            'vertical_spacing_abs': 0.08 * data_range,
            'bracket_vertical_length_abs': 0.03 * data_range,
            'text_offset_from_bracket_abs': 0.03 * data_range,
            'text_height_allowance_for_collision_abs': 0.04 * data_range,
        }}

        drawn = []

        for pair in pairs:
            g1_data = sec_df[sec_df["_color_group"] == pair[0]][SELECTED_VAR].dropna().values
            g2_data = sec_df[sec_df["_color_group"] == pair[1]][SELECTED_VAR].dropna().values
            if len(g1_data) == 0 or len(g2_data) == 0:
                continue

            star = ""
            if STATISTICAL_TEST != "None":
                equal_var = (STATISTICAL_TEST == "Independent t-test")
                try:
                    _, pval = ttest_ind(g1_data, g2_data, equal_var=equal_var)
                    if pval <= 0.0001: star = "****"
                    elif pval <= 0.001: star = "***"
                    elif pval <= 0.01: star = "**"
                    elif pval <= 0.05: star = "*"
                except Exception:
                    pass

            if EFFECT_SIZE_METHOD != "None":
                if EFFECT_SIZE_METHOD == "Glass's Delta":
                    es = glass_delta(g1_data, g2_data, MEAN_OR_MEDIAN)
                else:
                    es = cohens_d(g1_data, g2_data, MEAN_OR_MEDIAN)
                if abs(es) < EFFECT_SIZE_THRESHOLD:
                    continue
                txt = f"{{es:.2f}}{{star}}" if star else f"\\u0394={{es:.2f}}"
            else:
                txt = star  # statistical test only — star-only annotation

            key1 = (sec_group, pair[0]) if sec_group is not None else pair[0]
            key2 = (sec_group, pair[1]) if sec_group is not None else pair[1]
            x1, x2 = x_positions[key1], x_positions[key2]
            x_start, x_end = min(x1, x2), max(x1, x2)

            spanned = [cg for cg in sec_color_groups
                       if x_positions.get((sec_group, cg) if sec_group is not None else cg, -1) >= x_start
                       and x_positions.get((sec_group, cg) if sec_group is not None else cg, -1) <= x_end]
            region_df = sec_df[sec_df["_color_group"].isin(spanned)]
            region_max = region_df[SELECTED_VAR].max() if not region_df.empty else all_y.max()

            # Use extracted collision detection function
            result = _compute_bracket_position(x_start, x_end, region_max, positioning_metrics, drawn)
            if result is None:
                continue
            y_bracket_top, y_text_center, bracket_h = result

            # Draw bracket (Matplotlib)
            ax.plot([x_start, x_start, x_end, x_end],
                   [y_bracket_top - bracket_h, y_bracket_top, y_bracket_top, y_bracket_top - bracket_h],
                   color='black', linewidth=1.5, zorder=4)

            ax.text((x_start + x_end) / 2, y_text_center,
                   txt, ha='center', va='bottom', fontsize=10, zorder=4)

# --- Axis setup ---
ax.set_xticks(tick_positions)
ax.set_xticklabels(x_labels, fontsize=max(8, AXIS_LABEL_SIZE - 4))
ax.set_ylabel(SELECTED_VAR, fontsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='y', labelsize=LEGEND_SIZE)
add_encoding_legend_entries(ax, shape_map, opacity_map, POINT_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
"""


def _build_2d_distribution(state: dict) -> str:
    from src.vis.helpers import _find_best_gmm
    gmm_src = _extract_source(_find_best_gmm) if state.get("method_params", {}).get("fit_gmm_2d") else ""

    return _build_visual_encoding(state) + f"""
# ============================================================
# 2D Feature Distribution
# ============================================================
{gmm_src}

df = df[df[SELECTED_X].notna() & df[SELECTED_Y].notna()]

if LOG_X:
    if (df[SELECTED_X] < 0).any():
        print(f"WARNING: Cannot apply log to {{SELECTED_X}}: contains negative values.")
    else:
        df = df.copy()
        df[SELECTED_X] = np.log10(df[SELECTED_X] + 1e-6)
if LOG_Y:
    if (df[SELECTED_Y] < 0).any():
        print(f"WARNING: Cannot apply log to {{SELECTED_Y}}: contains negative values.")
    else:
        df = df.copy()
        df[SELECTED_Y] = np.log10(df[SELECTED_Y] + 1e-6)

# Create figure with marginal axes
if MARGINAL_PLOT_TYPE != 'none':
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
else:
    fig, ax_main = plt.subplots(figsize=(10, 8))
    ax_top = None
    ax_right = None

legend_entries = set()
for g in color_groups:
    gdf = df[df["_color_group"] == g]
    label = g if g not in legend_entries else None
    scatter_with_encodings(ax_main, gdf[SELECTED_X], gdf[SELECTED_Y],
                           color_map[g][:3], label, POINT_SIZE,
                           shape_vals=gdf[SHAPE_BY] if SHAPE_BY else None, shape_map=shape_map,
                           opacity_vals=gdf[OPACITY_BY] if OPACITY_BY else None, opacity_map=opacity_map)
    legend_entries.add(g)

    if ax_top is not None and len(gdf) > 1:
        from scipy.stats import gaussian_kde
        try:
            x_vals = gdf[SELECTED_X].dropna().values
            kde_x = gaussian_kde(x_vals)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
            if MARGINAL_PLOT_TYPE == 'gaussian fit':
                ax_top.plot(x_range, kde_x(x_range), color=color_map[g][:3], linewidth=1.5)
            elif MARGINAL_PLOT_TYPE == 'boxplot':
                ax_top.boxplot(x_vals, vert=False, positions=[0], widths=0.5,
                             patch_artist=True, boxprops=dict(facecolor=(*color_map[g][:3], 0.3)))
            elif MARGINAL_PLOT_TYPE == 'violin':
                parts = ax_top.violinplot(x_vals, vert=False, positions=[0], showmedians=True)
                for pc in parts.get('bodies', []):
                    pc.set_facecolor((*color_map[g][:3], 0.3))
        except Exception:
            pass

    if ax_right is not None and len(gdf) > 1:
        try:
            y_vals = gdf[SELECTED_Y].dropna().values
            kde_y = gaussian_kde(y_vals)
            y_range = np.linspace(y_vals.min(), y_vals.max(), 200)
            if MARGINAL_PLOT_TYPE == 'gaussian fit':
                ax_right.plot(kde_y(y_range), y_range, color=color_map[g][:3], linewidth=1.5)
            elif MARGINAL_PLOT_TYPE == 'boxplot':
                ax_right.boxplot(y_vals, vert=True, positions=[0], widths=0.5,
                               patch_artist=True, boxprops=dict(facecolor=(*color_map[g][:3], 0.3)))
            elif MARGINAL_PLOT_TYPE == 'violin':
                parts = ax_right.violinplot(y_vals, vert=True, positions=[0], showmedians=True)
                for pc in parts.get('bodies', []):
                    pc.set_facecolor((*color_map[g][:3], 0.3))
        except Exception:
            pass

# Pearson r + p is reported per color group unconditionally, matching the app's
# always-on correlation readout; the regression line + R² stay gated.
for g in color_groups:
    gdf = df[df["_color_group"] == g].dropna(subset=[SELECTED_X, SELECTED_Y])
    if len(gdf) < 2:
        continue
    r_val, p_val = pearsonr(gdf[SELECTED_X], gdf[SELECTED_Y])
    print(f"  {{g}}: Pearson r={{r_val:.4f}}, p={{p_val:.2e}}")
    if FIT_REGRESSION:
        X_reg = gdf[SELECTED_X].values.reshape(-1, 1)
        y_reg = gdf[SELECTED_Y].values
        model = LinearRegression().fit(X_reg, y_reg)
        r2 = model.score(X_reg, y_reg)
        x_line = np.linspace(X_reg.min(), X_reg.max(), 100)
        ax_main.plot(x_line, model.predict(x_line.reshape(-1, 1)), '--',
                    color=color_map[g][:3], linewidth=2)
        print(f"    R\\u00b2={{r2:.4f}}")

if FIT_GMM_2D:
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2

    for g in color_groups:
        gdf = df[df["_color_group"] == g].dropna(subset=[SELECTED_X, SELECTED_Y])
        if len(gdf) < 3:
            continue
        X_gmm = gdf[[SELECTED_X, SELECTED_Y]].values
        best_gmm = _find_best_gmm(X_gmm, max_components=GMM_MAX_COMPONENTS,
                                  min_weight_threshold=GMM_MIN_WEIGHT_THRESHOLD)

        # Ellipses + per-point labels only when the GMM has >1 component (bivar.py);
        # a unimodal best-fit draws no overlay.
        if best_gmm is not None and best_gmm.n_components > 1:
            for i in range(best_gmm.n_components):
                mean = best_gmm.means_[i]
                cov = best_gmm.covariances_[i]
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                chi2_val = chi2.ppf(0.95, 2)
                width = 2 * np.sqrt(eigenvalues[0] * chi2_val)
                height = 2 * np.sqrt(eigenvalues[1] * chi2_val)
                ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                fill=False, edgecolor=color_map[g][:3], linewidth=2, linestyle='--')
                ax_main.add_patch(ellipse)
                ax_main.plot(*mean, '+', color=color_map[g][:3], markersize=15, markeredgewidth=2)

            # per-point component membership, as the app assigns it
            subpopulation_labels = best_gmm.predict(X_gmm)
            df.loc[gdf.index, "2D_GMM_group"] = [f"{{g}}_group{{label + 1}}" for label in subpopulation_labels]

    if SAVE_DERIVED_DATA:
        df.drop(columns=["_color_group"]).to_csv("2D_gmm_data.csv", index=False)
        print("2D GMM data saved to 2D_gmm_data.csv")

ax_main.set_xlabel(SELECTED_X, fontsize=AXIS_LABEL_SIZE)
ax_main.set_ylabel(SELECTED_Y, fontsize=AXIS_LABEL_SIZE)
ax_main.tick_params(axis='both', labelsize=LEGEND_SIZE)
add_encoding_legend_entries(ax_main, shape_map, opacity_map, POINT_SIZE)
ax_main.legend(fontsize=LEGEND_SIZE)
"""


def _build_phasor_plot(state: dict) -> str:
    # Rendering is Matplotlib-specific; the K-Means clustering is extracted from
    # the app (src/vis/bivar.py) so both run the identical computation.
    kmeans_src = ""
    if state.get("method_params", {}).get("k_means"):
        from src.vis.bivar import phasor_kmeans
        kmeans_src = ("\n# K-Means clustering (extracted from FLIM Playground source)\n"
                      + _extract_source(phasor_kmeans))
    return _build_visual_encoding(state) + kmeans_src + """
# ============================================================
# Phasor Plot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Universal semicircle: G = 1/(1+u^2), S = u/(1+u^2)
u = np.linspace(0, 100, 5000)
G_semi = 1.0 / (1.0 + u**2)
S_semi = u / (1.0 + u**2)
ax.plot(G_semi, S_semi, 'k-', linewidth=1.5, zorder=1)

# Lifetime markers
w = 2 * np.pi * PHASOR_F
for tau in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    wt = w * tau
    g_marker = 1.0 / (1.0 + wt**2)
    s_marker = wt / (1.0 + wt**2)
    ax.plot(g_marker, s_marker, 'ko', markersize=5, zorder=3)
    ax.annotate(f"{tau}ns", (g_marker, s_marker), textcoords="offset points",
               xytext=(5, 5), fontsize=8, zorder=3)

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

harmonic_label = "1st" if PHASOR_HARMONIC == 1 else "2nd"
g_col = f"Lifetime fit free_{PHASOR_CHANNEL}: G({harmonic_label})"
s_col = f"Lifetime fit free_{PHASOR_CHANNEL}: S({harmonic_label})"

if g_col not in df.columns or s_col not in df.columns:
    print(f"ERROR: Columns {g_col} and/or {s_col} not found in data.")
else:
    keep_cols = [g_col, s_col, "_color_group"] + [col for col in (SHAPE_BY, OPACITY_BY) if col]
    plot_df = df[list(dict.fromkeys(keep_cols))].dropna()

    for g in color_groups:
        gdf = plot_df[plot_df["_color_group"] == g]
        scatter_with_encodings(ax, gdf[g_col], gdf[s_col], color_map[g][:3], g, POINT_SIZE,
                               shape_vals=gdf[SHAPE_BY] if SHAPE_BY else None, shape_map=shape_map,
                               opacity_vals=gdf[OPACITY_BY] if OPACITY_BY else None, opacity_map=opacity_map)

    add_encoding_legend_entries(ax, shape_map, opacity_map, POINT_SIZE)

    if K_MEANS:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial import ConvexHull

        for g in color_groups:
            gdf = plot_df[plot_df["_color_group"] == g]
            if len(gdf) < K_MEANS_CLUSTERS:
                continue
            coords = gdf[[g_col, s_col]].values
            labels, centers = phasor_kmeans(coords, K_MEANS_CLUSTERS)
            # per-point cluster membership, as the app assigns it
            df.loc[gdf.index, "k_means_cluster"] = [f"{g}_group{label + 1}" for label in labels]

            # All of a group's hulls/centroids share that group's color
            # (bivar.py _plot_convex_hull passes the group color), not a per-cluster ramp.
            group_color = color_map[g][:3]
            for ci in range(K_MEANS_CLUSTERS):
                cluster_pts = coords[labels == ci]
                if len(cluster_pts) >= 3:
                    try:
                        hull = ConvexHull(cluster_pts)
                        hull_pts = cluster_pts[hull.vertices]
                        hull_pts = np.vstack([hull_pts, hull_pts[0]])
                        ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                               alpha=0.15, color=group_color, zorder=1)
                        ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                               color=group_color, linewidth=1.5, zorder=1)
                    except Exception:
                        pass
                ax.plot(centers[ci, 0], centers[ci, 1], 'x',
                       color=group_color, markersize=12, markeredgewidth=2, zorder=4)

        if SAVE_DERIVED_DATA:
            df.drop(columns=["_color_group"]).to_csv("kmeans_clustered_data.csv", index=False)
            print("K-Means clustered data saved to kmeans_clustered_data.csv")

ax.set_xlabel("G", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("S", fontsize=AXIS_LABEL_SIZE)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 0.55)
ax.set_aspect('equal')
ax.tick_params(axis='both', labelsize=LEGEND_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
"""


def _build_dimension_reduction(state: dict) -> str:
    return _build_visual_encoding(state) + """
# ============================================================
# Dimension Reduction
# ============================================================
from sklearn.preprocessing import StandardScaler

df = df[df[SELECTED_FEATURES].notna().all(axis=1)]
X = df[SELECTED_FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if DR_METHOD == "PCA":
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)
    xlabel = f"PC1 ({reducer.explained_variance_ratio_[0]*100:.1f}%)"
    ylabel = f"PC2 ({reducer.explained_variance_ratio_[1]*100:.1f}%)"
elif DR_METHOD == "UMAP":
    import umap
    n_neighbors = HYPER_PARAMS.get("n_neighbors", 15)
    min_dist = HYPER_PARAMS.get("min_dist", 0.1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric='euclidean', random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)
    xlabel, ylabel = "UMAP 1", "UMAP 2"
elif DR_METHOD == "t-SNE":
    from sklearn.manifold import TSNE
    perplexity = HYPER_PARAMS.get("perplexity", 15)
    early_exaggeration = HYPER_PARAMS.get("early_exaggeration", 12)
    reducer = TSNE(n_components=2, perplexity=perplexity,
                   early_exaggeration=early_exaggeration, random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)
    xlabel, ylabel = "t-SNE 1", "t-SNE 2"

df = df.copy()
df["_dr_x"] = X_reduced[:, 0]
df["_dr_y"] = X_reduced[:, 1]

fig, ax = plt.subplots(figsize=(10, 8))

for g in color_groups:
    gdf = df[df["_color_group"] == g]
    scatter_with_encodings(ax, gdf["_dr_x"], gdf["_dr_y"], color_map[g][:3], g, POINT_SIZE,
                           shape_vals=gdf[SHAPE_BY] if SHAPE_BY else None, shape_map=shape_map,
                           opacity_vals=gdf[OPACITY_BY] if OPACITY_BY else None, opacity_map=opacity_map)

ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=LEGEND_SIZE)
add_encoding_legend_entries(ax, shape_map, opacity_map, POINT_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
"""


def _build_classification(state: dict) -> str:
    from src.classify import (
        prepare_data, _build_classifier, calculate_metrics,
        calculate_roc_curve, plot_roc_curve, plot_confusion_matrix, plot_feature_importance,
    )
    from src.tuned_threshold_classifier import TunedThresholdClassifierCV

    # Extract all computation source from actual codebase
    computation_src = _extract_source(
        prepare_data, _build_classifier, calculate_metrics,
        calculate_roc_curve, plot_roc_curve, plot_confusion_matrix, plot_feature_importance,
        TunedThresholdClassifierCV,
    )

    mp = state.get("method_params", {})
    clf_method = mp.get("classification_method", "Random Forest")

    # Build classifier call code — uses _build_classifier from extracted source
    clf_call = f'classifier = _build_classifier({clf_method!r}, cw, CLASSIFIER_PARAMS, RANDOM_STATE)'

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

df = df[df[feature_cols].notna().all(axis=1) & df["classes"].notna()]

n_classes = df["classes"].nunique()
if n_classes < 2:
    raise ValueError(f"Need at least 2 classes for classification, but only found {{n_classes}}: {{df['classes'].unique().tolist()}}. "
                     "Check your CLASSIFY_CLASSES and categorical filters.")

df_classify = df[feature_cols + ["classes"]]

error_msg, X_train, X_test, y_train, y_test = prepare_data(
    df_classify, TRAIN_SIZE, SAMPLING_METHOD, RANDOM_STATE
)
if error_msg:
    raise RuntimeError(error_msg)

cw = "balanced" if CLASS_WEIGHT == "Balanced" else None
{clf_call}

# --- Threshold optimization ---
threshold_values = None
tuned_classifier = None

if THRESHOLD_METHOD == "None":
    threshold_values = 0.5 if len(np.unique(y_train)) == 2 else None
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)
elif THRESHOLD_METHOD in ["Balanced Accuracy", "F1 Score"]:
    scoring_map = {{"Balanced Accuracy": "balanced_accuracy", "F1 Score": "f1_macro"}}
    classifier_copy = deepcopy(classifier)
    tuned_classifier = TunedThresholdClassifierCV(
        classifier_copy,
        scoring=scoring_map[THRESHOLD_METHOD],
        random_state=RANDOM_STATE
    ).fit(X_train, y_train)

    if tuned_classifier.n_classes_ == 2:
        threshold_values = tuned_classifier.best_threshold_
    else:
        threshold_values = tuned_classifier.best_thresholds_

    classifier.fit(X_train, y_train)
    if tuned_classifier is not None:
        y_pred = tuned_classifier.predict(X_test)
        y_score = tuned_classifier.predict_proba(X_test)
    else:
        y_pred = classifier.predict(X_test)
        y_score = classifier.predict_proba(X_test)
else:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)

# --- Metrics ---
metrics = calculate_metrics(y_test, y_pred)

print("=" * 60)
print(f"Classification Results: {{CLASSIFICATION_METHOD}}")
print("=" * 60)
print(f"Accuracy: {{metrics['accuracy']:.4f}}")
for cls, m in metrics['per_class'].items():
    print(f"  {{cls}}: Precision={{m['precision']:.3f}}, Recall={{m['recall']:.3f}}, "
          f"Specificity={{m['specificity']:.3f}}, F1={{m['f1_score']:.3f}}, N={{m['n']}}")
print(f"Balanced Accuracy: {{metrics['balanced_accuracy']:.4f}}")

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
