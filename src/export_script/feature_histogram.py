"""Feature Histogram export: its builder."""
from src.export_labels import EXPORT_METHODS
from src.export_script.mpl_helpers import (
    create_color_map,
    plotly_dash_style,
    style_axes_like_app,
)
from src.export_script.skeleton import (
    SEPARATE_BY_VALIDATION,
    MethodSpec,
    _extract_source,
)


def _build_feature_histogram(state: dict) -> str:
    """Render every prepared category, sharing calculations with the app."""
    from src.feature_labels import format_feature_label
    from src.natural_sort import natural_key, natural_tuple_sort, tuple_natural_key
    from src.vis.feature_histogram import (
        _assign_subpopulation_labels,
        find_intersection,
        histogram_bin_edges,
        histogram_bin_settings,
        histogram_gmm,
        histogram_labels,
        histogram_legend_label,
        histogram_skewness,
        prepare_histogram,
    )
    from src.vis.helpers import _find_best_gmm, log10_columns

    helpers = [natural_key, tuple_natural_key, natural_tuple_sort,
               histogram_legend_label, create_color_map, plotly_dash_style, histogram_bin_settings,
               histogram_bin_edges, histogram_skewness, _find_best_gmm,
               find_intersection, _assign_subpopulation_labels, histogram_gmm,
               prepare_histogram, format_feature_label, histogram_labels, style_axes_like_app]
    return ("\n# Log transform (extracted from FLIM Playground source)\n" + _extract_source(log10_columns)
            + _PREPARATION
            + "\n# Shared Histogram preparation (extracted from FLIM Playground)\n"
            + _extract_source(*helpers) + """
# ============================================================
# Feature Histogram
# ============================================================
DERIVED_LABEL_COLUMN = available_label_column(df.columns, DERIVED_LABEL_NAME)
histogram_data = prepare_histogram(
    df, SELECTED_VAR, COLOR_BY, SEPARATE_BY, bin_width=BIN_WIDTH,
    apply_gmm=APPLY_GMM, max_components=GMM_MAX_COMPONENTS,
    min_weight_threshold=GMM_MIN_WEIGHT_THRESHOLD,
    intersection_threshold=INTERSECTION_THRESHOLD, label_column=DERIVED_LABEL_COLUMN)
df = histogram_data["df"]
color_groups = histogram_data["color_groups"]
color_map = create_color_map(color_groups, COLORMAP, alpha=1.0)
bin_edges = histogram_data["bin_edges"]
bin_width = float(bin_edges[1] - bin_edges[0])
panels = histogram_data["panels"]
panel_levels = [panel["category"] for panel in panels]
rows = max(1, len(panels))
row_height = 4
if APPLY_GMM:
    # An outside legend must fit beside its own row even with large legend fonts.
    legend_rows = max((sum(1 + (len(group["components"]) if len(group["components"]) > 1 else 0)
                           for group in panel["groups"]) for panel in panels), default=0)
    row_height = max(row_height, (legend_rows + 2) * LEGEND_SIZE * 1.3 / 72)
fig, _axes = plt.subplots(
    rows, 1, squeeze=False, sharex=True, sharey=True,
    figsize=(10, row_height * rows + 1 if SEPARATE_BY else max(6, row_height + 1)),
    layout="constrained" if SEPARATE_BY else None)
histogram_axes = list(_axes[:, 0])
dash_styles = ['dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

for ax, panel in zip(histogram_axes, panels):
    legend_handles = []
    if SEPARATE_BY:
        # The app's category headings stop growing at 24 (apply_plot_styling).
        ax.set_title(str(panel['category']), fontsize=min(AXIS_LABEL_SIZE, 24))
    for group in panel["groups"]:
        g = group["color_group"]
        color = color_map[g][:3]
        label = f"{g} GMM" if APPLY_GMM else g
        sk = group["skewness"]
        legend_label = histogram_legend_label(
            label, group["count"], sk, SHOW_GROUP_COUNTS, show_skewness=not APPLY_GMM)
        skew_text = f"{sk:.3f}" if np.isfinite(sk) else "undefined"
        print(f"  {group['label']} (n={group['count']}): skewness = {skew_text}")
        if not APPLY_GMM or len(group["pdf"]):
            line, = ax.plot(
                group["x"] if APPLY_GMM else histogram_data["bin_centers"],
                group["pdf"] if APPLY_GMM else group["counts"],
                color=color, linewidth=2, label=legend_label,
                marker='o' if not APPLY_GMM and len(histogram_data["bin_centers"]) == 1 else None,
                markersize=POINT_SIZE, markeredgewidth=0)
        else:
            # Sparse and failed fits retain their local count/skewness legend entry.
            line = Line2D([], [], color=color, linewidth=2, label=legend_label)
        legend_handles.append(line)
        if APPLY_GMM:
            gmm = group["gmm"]
            for notice in group["notices"]:
                print(f"    {group['label']}: {notice}")
            if gmm is not None:
                print(f"    Best GMM has {gmm.n_components} component(s)")
                print("    | Component | Mean     | Std. Dev. | Weight |")
                print("    |-----------|----------|-----------|--------|")
            for component in group["components"]:
                rank = component["rank"]
                print(f"    | {rank}         | {component['mean']:.4f} | "
                      f"{component['std']:.4f}  | {component['weight']:.3f}  |")
                if len(group["components"]) <= 1:
                    continue
                line, = ax.plot(
                    group["x"], component["density"], color=color,
                    linestyle=plotly_dash_style(dash_styles[(rank - 1) % len(dash_styles)], 1),
                    linewidth=1, label=f"{g} Component {rank}")
                legend_handles.append(line)
            if group["h_index"] is not None:
                print(f"    H-index: {group['h_index']:.3f}")
            thresholds = group["thresholds"]
            for index, threshold in enumerate(thresholds if thresholds is not None else [], 1):
                peak = float(max(group["pdf"]))
                ax.plot([threshold, threshold], [0, peak], color=color,
                        linestyle=plotly_dash_style('dash', 2), alpha=0.5, linewidth=2)
                if not SEPARATE_BY:
                    ax.text(threshold, peak * 1.05, f"Threshold ({threshold:.2f})",
                            ha='center', fontsize=AXIS_LABEL_SIZE, color='black')
                print(f"    Threshold between component {index} and {index + 1}: {threshold:.4f}")
    if legend_handles:
        # The app's legends have no border; the one over the data is translucent white.
        if APPLY_GMM:
            ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                      borderaxespad=0, fontsize=LEGEND_SIZE, frameon=False)
        else:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_SIZE,
                      edgecolor='none', framealpha=0.85)
labels = histogram_labels(histogram_data, LOG_X, engine='mpl')
x_label, y_label, title = labels["x"], labels["y"], labels["title"]
for ax in histogram_axes:
    show_x_axis = ax is histogram_axes[-1]
    ax.set_xlabel(x_label if show_x_axis else "", fontsize=AXIS_LABEL_SIZE)
    ax.xaxis.set_visible(show_x_axis)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(histogram_data["x_range"])
    ax.set_ylim(histogram_data["y_range"])
    ax.tick_params(axis='both', labelsize=AXIS_LABEL_SIZE - 2, labelbottom=show_x_axis)
    style_axes_like_app(ax)
ax = histogram_axes[0]
if SEPARATE_BY:
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight='bold', x=0.01, ha='left')
else:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', loc='left')

if APPLY_GMM and SAVE_DERIVED_DATA:
    apply_export_labels(df, DERIVED_LABEL_COLUMN, DERIVED_EXPORT).to_csv(
        DERIVED_DATA_FILE, index=False)
    print("GMM grouped data saved to " + DERIVED_DATA_FILE)
""")


# Histogram retains individual units to explore their variability and heterogeneity.
_PREPARATION = """
# Histogram retains individual units to explore their variability and heterogeneity.
""" + SEPARATE_BY_VALIDATION + """df = df[df[SELECTED_VAR].notna()].copy()
if LOG_X:
    df, _skipped_log = log10_columns(df, [SELECTED_VAR])
    if _skipped_log:
        print("WARNING: Cannot apply log to " + str(SELECTED_VAR) + ": contains negative values.")
        LOG_X = False
"""


def _config_lines(state: dict) -> list[str]:
    mp = state.get("method_params", {})
    return [
        f"SELECTED_VAR = {mp.get('selected_var')!r}",
        f"SEPARATE_BY = {state.get('separate_by')!r}  # one full-width row per category",
        f"LOG_X = {mp.get('log_x', False)!r}",
        f"APPLY_GMM = {mp.get('apply_gmm', False)!r}",
        f"INTERSECTION_THRESHOLD = {mp.get('intersection_threshold', False)!r}",
        f"BIN_WIDTH = {mp.get('bin_width')!r}  # None → numpy 'auto' bin width",
        f"GMM_MAX_COMPONENTS = {mp.get('gmm_max_components', 3)!r}",
        f"GMM_MIN_WEIGHT_THRESHOLD = {mp.get('gmm_min_weight_threshold', 0.1)!r}",
        f"SAVE_DERIVED_DATA = False  # True → also write {EXPORT_METHODS[SPEC.name][1]} when APPLY_GMM is enabled (the app's download button)",
    ]


SPEC = MethodSpec(
    name="Feature Histogram",
    build=_build_feature_histogram,
    config_lines=_config_lines,
    imports=lambda state: ["from matplotlib.lines import Line2D",
                           "from sklearn.mixture import GaussianMixture",
                           "from scipy.stats import norm", "from scipy.optimize import brentq"],
    # The GMM helpers are embedded even when fitting is disabled.
    packages=lambda state: {"scipy", "scikit-learn", "threadpoolctl"},
    # Preserve constrained layout even if separation is enabled by editing the script.
    footer=lambda state: "if not SEPARATE_BY:\n    plt.tight_layout()",
)
