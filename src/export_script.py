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
# FLIM Playground is currently on bioRxiv (https://www.biorxiv.org/content/10.1101/2025.09.30.679625).
# If it contributed to your research—whether through Data Extraction for single-cell feature extraction
# or through Data Analysis for data exploration, visualization, selection of analysis methods, or
# hyperparameter tuning (UMAP, clustering, classification, etc.)—please cite this work in your
# publication. Your citation directly supports us in maintaining and improving it ✨🎈🍾.
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
        extra.append("from scipy.stats import skew")
        if mp.get("apply_gmm"):
            extra += ["from sklearn.mixture import GaussianMixture",
                      "from scipy.stats import norm", "from scipy.optimize import brentq"]
    elif method == "2D Feature Distribution":
        extra.append("from scipy.stats import gaussian_kde")
        if mp.get("fit_regression"):
            extra += ["from sklearn.linear_model import LinearRegression",
                      "from scipy.stats import pearsonr"]
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
        custom_order = mp.get("custom_order")
        if custom_order:
            lines.append(f"CUSTOM_ORDER = {custom_order!r}  # Reorder: compare_groups and/or separate_groups")
        else:
            lines.append("CUSTOM_ORDER = None  # Set to {'compare_groups': ['group1', 'group2', ...]} to reorder x-axis")
    elif method == "Feature Histogram":
        lines.append(f"LOG_X = {mp.get('log_x', False)!r}")
        lines.append(f"APPLY_GMM = {mp.get('apply_gmm', False)!r}")
        lines.append(f"INTERSECTION_THRESHOLD = {mp.get('intersection_threshold', False)!r}")
    elif method == "2D Feature Distribution":
        lines.append(f"SELECTED_X = {mp.get('selected_x')!r}")
        lines.append(f"SELECTED_Y = {mp.get('selected_y')!r}")
        lines.append(f"LOG_X = {mp.get('log_x', False)!r}")
        lines.append(f"LOG_Y = {mp.get('log_y', False)!r}")
        lines.append(f"MARGINAL_PLOT_TYPE = {mp.get('marginal_plot_type', 'gaussian fit')!r}")
        lines.append(f"FIT_REGRESSION = {mp.get('fit_regression', False)!r}")
        lines.append(f"FIT_GMM_2D = {mp.get('fit_gmm_2d', False)!r}")
    elif method == "Phasor Plot":
        lines.append(f"PHASOR_CHANNEL = {mp.get('selected_channel')!r}")
        lines.append(f"PHASOR_HARMONIC = {mp.get('phasor_harmonic', 1)!r}")
        lines.append(f"PHASOR_F = {mp.get('phasor_f', 0.08)!r}")
        lines.append(f"K_MEANS = {mp.get('k_means', False)!r}")
        lines.append(f"K_MEANS_CLUSTERS = {mp.get('k_means_clusters', 2)!r}")
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
    return "\n# " + "=" * 60 + "\n# Data Loading\n# " + "=" * 60 + "\ndf = pd.read_csv(CSV_PATH)\n"


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
    # Extract Matplotlib-adapted color/shape maps from this module
    mpl_src = _extract_source(create_color_map, create_shape_map)

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

fovs = natural_tuple_sort(df[fov_col].unique().tolist())
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

if "GMM_group" not in df.columns:
    df["GMM_group"] = np.nan

for g in color_groups:
    group_mask = df["_color_group"] == g
    gdata = df.loc[group_mask, SELECTED_VAR].dropna()
    if len(gdata) < 3:
        continue
    gmm = _find_best_gmm(gdata.values)
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
    else:
        df.loc[gdata.index, "GMM_group"] = f"{{g}}_group1"

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
_, bin_edges = np.histogram(all_vals, bins='auto')
bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0

for g in color_groups:
    gdata = df[df["_color_group"] == g][SELECTED_VAR].dropna().values
    if len(gdata) == 0:
        continue
    bins = np.arange(gdata.min(), gdata.max() + bin_width, bin_width)
    counts, edges = np.histogram(gdata, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, counts, label=g, color=color_map[g][:3], linewidth=2)

    from scipy.stats import skew as scipy_skew
    sk = scipy_skew(gdata)
    direction = "right" if sk > 0 else "left"
    if abs(sk) > 1:
        desc = f"strongly {direction}-skewed"
    elif abs(sk) > 0.5:
        desc = f"moderately {direction}-skewed"
    else:
        desc = "approximately symmetric"
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
            ax.scatter([x_pos] * len(y_data), y_data, c=[color_map[cg][:3]],
                      s=POINT_SIZE, edgecolors='DarkSlateGrey', linewidths=0.5,
                      label=cg if cg not in legend_entries else None, zorder=2)
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

        marker = 'o'
        if SHAPE_BY and SHAPE_BY in group_df.columns:
            shape_vals = group_df[SHAPE_BY].astype(str).unique()
            if len(shape_vals) == 1 and shape_vals[0] in shape_map:
                marker = shape_map[shape_vals[0]]

        alpha = 0.7
        if OPACITY_BY and OPACITY_BY in group_df.columns:
            opacity_vals = group_df[OPACITY_BY].astype(str).unique()
            if len(opacity_vals) == 1 and opacity_vals[0] in opacity_map:
                alpha = opacity_map[opacity_vals[0]]

        ax.scatter(x_pos + jitter, y_data, c=[color_map[cg][:3]], s=POINT_SIZE,
                  alpha=alpha, marker=marker, edgecolors='DarkSlateGrey', linewidths=0.5,
                  label=cg if cg not in legend_entries else None, zorder=2)
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

# --- Effect size annotations ---
if EFFECT_SIZE_METHOD != "None":
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

            if EFFECT_SIZE_METHOD == "Glass's Delta":
                es = glass_delta(g1_data, g2_data, MEAN_OR_MEDIAN)
            else:
                es = cohens_d(g1_data, g2_data, MEAN_OR_MEDIAN)

            if abs(es) < EFFECT_SIZE_THRESHOLD:
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

            if star:
                txt = f"{{es:.2f}}{{star}}"
            else:
                txt = f"\\u0394={{es:.2f}}"
            ax.text((x_start + x_end) / 2, y_text_center,
                   txt, ha='center', va='bottom', fontsize=10, zorder=4)

# --- Axis setup ---
ax.set_xticks(tick_positions)
ax.set_xticklabels(x_labels, fontsize=max(8, AXIS_LABEL_SIZE - 4))
ax.set_ylabel(SELECTED_VAR, fontsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='y', labelsize=LEGEND_SIZE)
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
    c = [color_map[g][:3]]
    label = g if g not in legend_entries else None
    ax_main.scatter(gdf[SELECTED_X], gdf[SELECTED_Y], c=c, s=POINT_SIZE,
                   alpha=0.7, edgecolors='DarkSlateGrey', linewidths=0.3,
                   label=label, zorder=2)
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

if FIT_REGRESSION:
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr
    for g in color_groups:
        gdf = df[df["_color_group"] == g].dropna(subset=[SELECTED_X, SELECTED_Y])
        if len(gdf) < 2:
            continue
        X_reg = gdf[SELECTED_X].values.reshape(-1, 1)
        y_reg = gdf[SELECTED_Y].values
        model = LinearRegression().fit(X_reg, y_reg)
        r2 = model.score(X_reg, y_reg)
        r_val, p_val = pearsonr(gdf[SELECTED_X], gdf[SELECTED_Y])
        x_line = np.linspace(X_reg.min(), X_reg.max(), 100)
        ax_main.plot(x_line, model.predict(x_line.reshape(-1, 1)), '--',
                    color=color_map[g][:3], linewidth=2)
        print(f"  {{g}}: R\\u00b2={{r2:.4f}}, Pearson r={{r_val:.4f}}, p={{p_val:.2e}}")

if FIT_GMM_2D:
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2

    for g in color_groups:
        gdf = df[df["_color_group"] == g].dropna(subset=[SELECTED_X, SELECTED_Y])
        if len(gdf) < 3:
            continue
        X_gmm = gdf[[SELECTED_X, SELECTED_Y]].values
        best_gmm = _find_best_gmm(X_gmm, max_components=3)

        if best_gmm is not None:
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

ax_main.set_xlabel(SELECTED_X, fontsize=AXIS_LABEL_SIZE)
ax_main.set_ylabel(SELECTED_Y, fontsize=AXIS_LABEL_SIZE)
ax_main.tick_params(axis='both', labelsize=LEGEND_SIZE)
ax_main.legend(fontsize=LEGEND_SIZE)
"""


def _build_phasor_plot(state: dict) -> str:
    # Phasor plot is entirely Matplotlib-specific rendering, no extractable computation
    return _build_visual_encoding(state) + """
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
    plot_df = df[[g_col, s_col, "_color_group"]].dropna()

    for g in color_groups:
        gdf = plot_df[plot_df["_color_group"] == g]
        ax.scatter(gdf[g_col], gdf[s_col], c=[color_map[g][:3]], s=POINT_SIZE,
                  alpha=0.7, edgecolors='DarkSlateGrey', linewidths=0.3,
                  label=g, zorder=2)

    if K_MEANS:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial import ConvexHull

        for g in color_groups:
            gdf = plot_df[plot_df["_color_group"] == g]
            if len(gdf) < K_MEANS_CLUSTERS:
                continue
            coords = gdf[[g_col, s_col]].values
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            km = KMeans(n_clusters=K_MEANS_CLUSTERS, random_state=42, n_init=10)
            labels = km.fit_predict(coords_scaled)
            centers = scaler.inverse_transform(km.cluster_centers_)

            cluster_colors = plt.colormaps['tab10'](np.linspace(0, 1, K_MEANS_CLUSTERS))
            for ci in range(K_MEANS_CLUSTERS):
                cluster_pts = coords[labels == ci]
                if len(cluster_pts) >= 3:
                    try:
                        hull = ConvexHull(cluster_pts)
                        hull_pts = cluster_pts[hull.vertices]
                        hull_pts = np.vstack([hull_pts, hull_pts[0]])
                        ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                               alpha=0.15, color=cluster_colors[ci], zorder=1)
                        ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                               color=cluster_colors[ci], linewidth=1.5, zorder=1)
                    except Exception:
                        pass
                ax.plot(centers[ci, 0], centers[ci, 1], 'x',
                       color=cluster_colors[ci], markersize=12, markeredgewidth=2, zorder=4)

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
    ax.scatter(gdf["_dr_x"], gdf["_dr_y"], c=[color_map[g][:3]], s=POINT_SIZE,
              alpha=0.7, edgecolors='DarkSlateGrey', linewidths=0.3,
              label=g, zorder=2)

ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
ax.tick_params(axis='both', labelsize=LEGEND_SIZE)
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
