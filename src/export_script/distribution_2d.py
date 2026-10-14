"""2D Feature Distribution export: its builder and the helpers only it pastes."""
from src.export_labels import EXPORT_METHODS
from src.export_script.mpl_helpers import (
    facet_canvas,
    place_facet_legend,
    plotly_dash_style,
    style_axes_like_app,
    widen_to_fit,
)
from src.export_script.skeleton import (
    SEPARATE_BY_VALIDATION,
    MethodSpec,
    _build_visual_encoding,
    _extract_source,
    prepare_rows,
)


def marginal_box_statistics(values):
    """Plotly's default linear quartiles (Hazen ranks) and observed whiskers.

    Matplotlib's default percentiles use different ranks, so its boxplot cannot
    reproduce the app without explicitly supplied statistics.
    """
    import numpy as np

    values = np.sort(np.asarray(values, dtype=float))
    q1, median, q3 = np.percentile(values, [25, 50, 75], method='hazen')
    spread = q3 - q1
    lower = max(values[0], min(q1, values[values >= q1 - 1.5 * spread][0]))
    upper = min(values[-1], max(q3, values[values <= q3 + 1.5 * spread][-1]))
    return dict(q1=q1, med=median, q3=q3, whislo=lower, whishi=upper,
                fliers=values[(values < lower) | (values > upper)])


def marginal_violin_statistics(values):
    """Plotly's Gaussian KDE, robust Silverman bandwidth, and default soft span."""
    import numpy as np

    values = np.sort(np.asarray(values, dtype=float))
    box = marginal_box_statistics(values)
    extent = values[-1] - values[0]
    bandwidth = max(1.059 * min(np.std(values, ddof=1), (box['q3'] - box['q1']) / 1.349)
                    * len(values) ** -.2, extent / 100)
    low, high = values[0] - 2 * bandwidth, values[-1] + 2 * bandwidth
    intervals = int(np.ceil((high - low) / (bandwidth / 3)))
    coords = np.linspace(low, high, intervals + 1)
    density = np.array([np.exp(-.5 * ((x - values) / bandwidth) ** 2).mean()
                        / (bandwidth * np.sqrt(2 * np.pi)) for x in coords])
    return dict(coords=coords, vals=density, mean=values.mean(), median=box['med'],
                min=values[0], max=values[-1])


def draw_distribution_marginal(ax, values, orientation, color, plot_type, position, point_size):
    """Render one available marginal using the app renderer's statistics and styles."""
    import numpy as np
    from scipy.stats import gaussian_kde

    if plot_type == 'gaussian fit':
        coords = np.linspace(min(values), max(values), 200)
        density = gaussian_kde(values)(coords)
        x, y = (coords, density) if orientation == 'horizontal' else (density, coords)
        ax.plot(x, y, color=color, linewidth=2, alpha=.7, gid='marginal')
        return
    if plot_type == 'boxplot':
        stroke = dict(color=color, linewidth=2)
        ax.bxp([marginal_box_statistics(values)], orientation=orientation,
               positions=[position], widths=.49, capwidths=.245, patch_artist=True,
               boxprops=dict(facecolor=(*color, .5), edgecolor=color, linewidth=2),
               whiskerprops=stroke, capprops=stroke, medianprops=stroke,
               flierprops=dict(marker='o', markersize=point_size,
                               markerfacecolor=color, markeredgewidth=0))
    elif plot_type == 'violin':
        parts = ax.violin([marginal_violin_statistics(values)], orientation=orientation,
                          positions=[position], widths=.49, showmeans=False,
                          showmedians=False, showextrema=False)
        for body in parts['bodies']:
            body.set_alpha(None)
            body.set_facecolor((*color, .5))
            body.set_edgecolor(color)
            body.set_linewidth(2)
    # Plotly pads categorical positions by half a slot regardless of body width.
    if orientation == 'horizontal':
        ax.set_ylim(-.5, position + .5)
    else:
        ax.set_xlim(-.5, position + .5)


def _print_distribution_statistics(result, label):
    """Report the shared 2D fit results for either exported category layout."""
    import numpy as np

    if result["pearson"] is not None:
        coefficient, p_value = result["pearson"]
        print(f"  {label}: Pearson r={coefficient:.4f}, p={p_value:.2e}")
    regression = result["regression"]
    if regression is not None:
        print(f"    R²={regression['r2']:.4f}, slope={regression['slope']:.4f}, "
              f"intercept={regression['intercept']:.4f}")
    if result["components"]:
        print(f"  {label}: GMM components")
        print("    | Component | Mean X | Std. Dev. X | Mean Y | Std. Dev. Y | Weight |")
    for index, component in enumerate(result["components"], 1):
        mean, covariance, weight = component["mean"], component["covariance"], component["weight"]
        std_x, std_y = np.sqrt(np.diag(covariance))
        print(f"    | {index} | {mean[0]:.4f} | {std_x:.4f} | "
              f"{mean[1]:.4f} | {std_y:.4f} | {weight:.3f} |")
    for notice in result["notices"]:
        print(f"  {label}: {notice}")


def _build_2d_distribution(state: dict) -> str:
    """One builder for both layouts: the separation only adds panels beside the overview."""
    from src.feature_labels import format_feature_label
    from src.vis.bivar import (
        category_panel_rows,
        distribution_fit_groups,
        distribution_ranges,
        gmm_ellipse_params,
    )
    from src.vis.dimension_facets import (
        MAIN_PLOT_LABEL,
        category_facet_groups,
        dimension_facet_layout,
        focus_facet_label,
        focus_slot_keys,
        normalize_dimension_categories,
    )
    from src.vis.gmm_tables import gmm_group_title
    from src.vis.plot_defaults import (
        OVERVIEW_MODEL_WIDTH,
        PANEL_CONTEXT_ALPHA,
        PANEL_CONTEXT_COLOR,
        PANEL_MODEL_WIDTH,
        WEBGL_POINT_THRESHOLD,
        panel_point_size,
    )

    helpers = [category_panel_rows, distribution_fit_groups, distribution_ranges,
               category_facet_groups, dimension_facet_layout, focus_slot_keys, focus_facet_label,
               widen_to_fit, place_facet_legend, _print_distribution_statistics,
               marginal_box_statistics, marginal_violin_statistics, draw_distribution_marginal,
               plotly_dash_style, format_feature_label, style_axes_like_app, facet_canvas,
               gmm_ellipse_params, gmm_group_title, panel_point_size]
    if state.get("method_params", {}).get("fit_gmm_2d"):
        from src.vis.helpers import _find_best_gmm

        helpers.append(_find_best_gmm)
    fit_src = (_extract_source(*helpers)
               + f"\nWEBGL_POINT_THRESHOLD = {WEBGL_POINT_THRESHOLD}\n")

    # Pasted ahead of the preparation block that calls it.
    preparation = "\n" + _extract_source(normalize_dimension_categories) + """
if df.empty:
    raise ValueError("No complete X/Y observations remain for 2D Feature Distribution.")
DERIVED_LABEL_COLUMN = available_label_column(df.columns, DERIVED_LABEL_NAME)
# Preserve retained metadata for CSV; normalize only the rendering copy.
distribution_data = df.copy()
if SEPARATE_BY:
    df = normalize_dimension_categories(df, [*COLOR_BY, SHAPE_BY, OPACITY_BY])
FD_GROUP_COLUMN = available_label_column(df.columns, "_color_group")
"""
    from src.vis.helpers import log10_columns

    log_block = ("\n# Log transform (extracted from FLIM Playground source)\n"
                 + _extract_source(log10_columns) + _LOG_2D_BLOCK)
    return (prepare_rows(state, _HEADER, ("SHAPE_BY", "OPACITY_BY"), log_block) + preparation
            + _build_visual_encoding(overlap_point=False,
                                     group_column_expr="FD_GROUP_COLUMN", interleave=True)
            + "\n# Shared 2D panel and fit helpers\n" + fit_src + """
# ============================================================
# 2D Feature Distribution
# ============================================================
# --- Grid constants ------------------------------------------------------
# Faceted and context points, model widths and the overview label are the app's
# own values (src/vis/plot_defaults.py and src/vis/dimension_facets.py), written
# here by the export so they cannot drift.
PANEL_POINT_SIZE = panel_point_size(POINT_SIZE)
PANEL_MODEL_WIDTH = """ + repr(PANEL_MODEL_WIDTH) + """
OVERVIEW_MODEL_WIDTH = """ + repr(OVERVIEW_MODEL_WIDTH) + """
PANEL_CONTEXT_COLOR = """ + repr(PANEL_CONTEXT_COLOR) + """
PANEL_CONTEXT_ALPHA = """ + repr(PANEL_CONTEXT_ALPHA) + """
MAIN_PLOT_LABEL = """ + repr(MAIN_PLOT_LABEL) + """
# -------------------------------------------------------------------------

distribution_panels = category_panel_rows(df, SEPARATE_BY, COLOR_BY)

# Shape and opacity describe points; they never partition a statistical fit.
distribution_results, distribution_assignments = distribution_fit_groups(
    df, SELECTED_X, SELECTED_Y, FD_GROUP_COLUMN, color_groups,
    distribution_panels, separate_by=SEPARATE_BY, fit_regression=FIT_REGRESSION,
    fit_gmm=FIT_GMM_2D, max_components=GMM_MAX_COMPONENTS,
    min_weight_threshold=GMM_MIN_WEIGHT_THRESHOLD, color_by=COLOR_BY,
)
x_range, y_range = distribution_ranges(df, SELECTED_X, SELECTED_Y, distribution_results)
facet_layout = dimension_facet_layout(
    category_facet_groups(distribution_panels if SEPARATE_BY else []),
    x_range, y_range, aspect=1.0,
)

if FIT_GMM_2D and (SEPARATE_BY or any(value is not None for value in distribution_assignments)):
    # distribution_data is the CSV copy; df keeps the same labels for inspection.
    distribution_data[DERIVED_LABEL_COLUMN] = distribution_assignments
    df[DERIVED_LABEL_COLUMN] = distribution_assignments
    if SAVE_DERIVED_DATA:
        apply_export_labels(
            distribution_data, DERIVED_LABEL_COLUMN, DERIVED_EXPORT
        ).to_csv(DERIVED_DATA_FILE, index=False)
        print("2D GMM data saved to " + DERIVED_DATA_FILE)

facet_axes = []
if SEPARATE_BY:
    # One shared legend below the whole composition; the layout keeps its own geometry.
    fig, place_panel, left_margin, figure_width, figure_height = facet_canvas(
        facet_layout, len(color_groups) + len(shape_map) + len(opacity_map))

    overview_x = facet_layout["overview"]["x_domain"]
    overview_y = facet_layout["overview"]["y_domain"]
    if MARGINAL_PLOT_TYPE is not None:
        # Strips take the outer tenth of the overview block in both directions,
        # so the main axes stay square.
        main_x = overview_x[0] + 0.9 * (overview_x[1] - overview_x[0])
        main_y = overview_y[0] + 0.9 * (overview_y[1] - overview_y[0])
        ax_main = place_panel(overview_x[0], main_x, overview_y[0], main_y)
        ax_top = place_panel(overview_x[0], main_x, main_y, overview_y[1])
        ax_right = place_panel(main_x, overview_x[1], overview_y[0], main_y)
        ax_top.sharex(ax_main)
        ax_right.sharey(ax_main)
        ax_top.tick_params(labelbottom=False, labelleft=False)
        ax_right.tick_params(labelleft=False, labelbottom=False)
    else:
        ax_main = place_panel(*overview_x, *overview_y)
        ax_top = ax_right = None
    for panel in facet_layout["panels"]:
        facet_axes.append(place_panel(*panel["x_domain"], *panel["y_domain"]))
elif MARGINAL_PLOT_TYPE is not None:
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 9], width_ratios=[9, 1],
                  hspace=0, wspace=0)
    # Match the app's square main axes and 10% marginal domains. Anchor the
    # shared edges so the marginals stay aligned even if figsize is edited.
    ax_main = fig.add_subplot(gs[1, 0], box_aspect=1, anchor='NE')
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main, box_aspect=1/9, anchor='SE')
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main, box_aspect=9, anchor='NW')
    ax_top.tick_params(labelbottom=False, labelleft=False)
    ax_right.tick_params(labelleft=False, labelbottom=False)
else:
    fig, ax_main = plt.subplots(figsize=(10, 10), subplot_kw=dict(box_aspect=1))
    ax_top = None
    ax_right = None

# Each slot draws one membership: the level promoted into the overview block if
# any, every other level in its own map, and the whole dataset wherever the
# overview sits. focus_slot_keys is the app's own arrangement rule.
panel_positions = {level: positions for level, positions in distribution_panels}
slot_keys = focus_slot_keys(list(panel_positions), FOCUS_CATEGORY if SEPARATE_BY else None)
promoted_category = slot_keys[0]
slot_axes = [ax_main, *facet_axes]
panel_point_area = PANEL_POINT_SIZE ** 2
point_legend_handles = []
for slot, (slot_ax, slot_key) in enumerate(zip(slot_axes, slot_keys)):
    # Points follow the slot, not the level: the main block keeps POINT_SIZE.
    slot_area = POINT_SIZE ** 2 if slot == 0 else panel_point_area
    positions = None if slot_key is None else panel_positions[slot_key]
    if positions is not None:
        membership = np.zeros(len(df), dtype=bool)
        membership[positions] = True
        if (~membership).any():
            background = df.loc[~membership]
            slot_ax.scatter(background[SELECTED_X], background[SELECTED_Y],
                            color=PANEL_CONTEXT_COLOR, alpha=PANEL_CONTEXT_ALPHA,
                            s=panel_point_area, edgecolors='none', linewidths=0, zorder=1)
    handles = scatter_interleaved_points(
        slot_ax, df, SELECTED_X, SELECTED_Y, FD_GROUP_COLUMN, color_groups, color_map,
        slot_area, shape_by=SHAPE_BY, shape_map=shape_map,
        opacity_by=OPACITY_BY, opacity_map=opacity_map,
        base_alpha=BASE_ALPHA, show_counts=SHOW_GROUP_COUNTS,
        active_positions=positions,
    )
    if slot_key is None:
        # One shared legend, counted over the whole dataset, wherever it sits.
        point_legend_handles = handles

# Marginals belong to the main block and describe whatever occupies it: the
# whole dataset, or the promoted category. One set is drawn, so no cross-level
# density rescaling is needed (the app draws one set at a time as well).
# Each visible box/violin gets its own categorical position on each axis.
marginal_frame = (df if promoted_category is None
                  else df.iloc[panel_positions[promoted_category]])
marginal_positions = {"x": 0, "y": 0}
for g in color_groups:
    gdf = marginal_frame[marginal_frame[FD_GROUP_COLUMN] == g]

    # Guard each marginal on its own axis, as the app does (_plot_marginal_density
    # in src/vis/bivar.py returns early per axis), so a constant y still draws x.
    if ax_top is not None and gdf[SELECTED_X].nunique() > 1:
        try:
            draw_distribution_marginal(
                ax_top, gdf[SELECTED_X].dropna().values, 'horizontal', color_map[g][:3],
                MARGINAL_PLOT_TYPE, marginal_positions["x"], POINT_SIZE)
            if MARGINAL_PLOT_TYPE in ('boxplot', 'violin'):
                marginal_positions["x"] += 1
        except Exception:
            pass

    if ax_right is not None and gdf[SELECTED_Y].nunique() > 1:
        try:
            draw_distribution_marginal(
                ax_right, gdf[SELECTED_Y].dropna().values, 'vertical', color_map[g][:3],
                MARGINAL_PLOT_TYPE, marginal_positions["y"], POINT_SIZE)
            if MARGINAL_PLOT_TYPE in ('boxplot', 'violin'):
                marginal_positions["y"] += 1
        except Exception:
            pass

# Every level's model is drawn in that level's own panel, never on the overview,
# and every level's statistics are printed (the app lists them all as well).
panel_by_level = {slot_key: slot_ax
                  for slot_ax, slot_key in zip(slot_axes, slot_keys)
                  if slot_key is not None}
model_width = PANEL_MODEL_WIDTH if SEPARATE_BY else OVERVIEW_MODEL_WIDTH
model_dash = plotly_dash_style('dash', model_width, webgl=len(df) >= WEBGL_POINT_THRESHOLD)
for result in distribution_results:
    color = color_map[result["color_group"]][:3]
    label = gmm_group_title(result["category"], result["color_group"])
    _print_distribution_statistics(result, label)
    model_ax = panel_by_level.get(result["category"], ax_main)
    regression = result["regression"]
    if regression is not None:
        model_ax.plot(regression["x"], regression["y"], color=color, linewidth=model_width,
                      gid='regression')
    for component in result["components"]:
        mean, covariance = component["mean"], component["covariance"]
        angle, width, height = gmm_ellipse_params(covariance)
        model_ax.add_patch(Ellipse(
            xy=mean, width=width, height=height, angle=angle, fill=False,
            edgecolor=color, linewidth=model_width, linestyle=model_dash,
        ))

if SEPARATE_BY:
    for panel_ax, slot_key in zip(facet_axes, slot_keys[1:]):
        panel_ax.text(1.04, 0.5,
                      MAIN_PLOT_LABEL if slot_key is None else str(slot_key),
                      transform=panel_ax.transAxes,
                      ha='left', va='center', fontsize=LEGEND_SIZE)
    for panel_ax in [ax_main, *facet_axes]:
        panel_ax.set_xlim(x_range)
        panel_ax.set_ylim(y_range)
    # The small maps keep the app's left and bottom axis lines; the overview block
    # follows the chart theme below.
    for panel_ax in facet_axes:
        panel_ax.grid(False, which='both')
        for side in ('left', 'bottom'):
            panel_ax.spines[side].set_visible(True)
            panel_ax.spines[side].set_color('black')
            panel_ax.spines[side].set_linewidth(1)
        for side in ('top', 'right'):
            panel_ax.spines[side].set_visible(False)
        panel_ax.tick_params(axis='both', which='both', bottom=False, left=False,
                             labelbottom=False, labelleft=False)

# Horizontal gridlines on the main axes only; the marginal strips are bare.
style_axes_like_app(ax_main)
for marginal_ax in (ax_top, ax_right):
    if marginal_ax is not None:
        style_axes_like_app(marginal_ax, grid_axis=None)
ax_main.set_xlabel(f"log₁₀({format_feature_label(SELECTED_X, engine='mpl')})" if LOG_X else format_feature_label(SELECTED_X, engine='mpl'), fontsize=AXIS_LABEL_SIZE)
ax_main.set_ylabel(f"log₁₀({format_feature_label(SELECTED_Y, engine='mpl')})" if LOG_Y else format_feature_label(SELECTED_Y, engine='mpl'), fontsize=AXIS_LABEL_SIZE)
ax_main.tick_params(axis='both', labelsize=AXIS_LABEL_SIZE - 2)
_2d_title = (f"2D Distribution of {format_feature_label(SELECTED_X, engine='mpl')} and "
             f"{format_feature_label(SELECTED_Y, engine='mpl')} by {', '.join(COLOR_BY)}")
if promoted_category is not None:
    # A promotion names itself in the title, as it does in the app.
    _2d_title += f" ({focus_facet_label(SEPARATE_BY, promoted_category, escape=False)})"
encoding_legend_handles = add_encoding_legend_entries(
    ax_main, shape_map, opacity_map, POINT_SIZE ** 2, borderless=True)
if SEPARATE_BY:
    fig.suptitle(_2d_title, fontsize=TITLE_SIZE, fontweight='bold', x=0.01, ha='left')
    figure_height = place_facet_legend(fig, ax_main,
                                       point_legend_handles + encoding_legend_handles,
                                       LEGEND_SIZE, left_margin, figure_width, figure_height)
else:
    # The app's title sits above the whole chart, top marginal included.
    (ax_main if ax_top is None else ax_top).set_title(
        _2d_title, fontsize=TITLE_SIZE, fontweight='bold', loc='left')
    # Plotly's default legend sits beside the plot, clear of the right marginal.
    legend = ax_main.legend(handles=point_legend_handles + encoding_legend_handles,
                            fontsize=LEGEND_SIZE, loc='upper left', frameon=False,
                            borderaxespad=0, bbox_to_anchor=(1.02, 1),
                            bbox_transform=(ax_right or ax_main).transAxes)
    for handle in legend.legend_handles:
        handle.set_sizes([LEGEND_SIZE ** 2])
    if ax_right is not None:
        # tight_layout cannot move the marginal grid, so the canvas makes the room.
        widen_to_fit(fig, fit_height=True)
""")


_HEADER = """
# Validate the editable layout before collapse can remove any metadata columns.
""" + SEPARATE_BY_VALIDATION + """df = df[df[SELECTED_X].notna() & df[SELECTED_Y].notna()].copy()
"""


# Log transform after optional replicate aggregation; the flags stay set, as in the app.
_LOG_2D_BLOCK = """
df, _skipped_log = log10_columns(
    df, [column for column, enabled in ((SELECTED_X, LOG_X), (SELECTED_Y, LOG_Y)) if enabled])
for _column in _skipped_log:
    print(f"WARNING: Cannot apply log to {_column}: contains negative values.")
"""


def _config_lines(state: dict) -> list[str]:
    mp = state.get("method_params", {})
    lines = []
    if mp.get("collapse_by"):
        lines.append(f"COLLAPSE_BY = {mp['collapse_by']!r}  # one point per replicate within each plot group, holding the MEAN of its cells")
    lines.append(f"SEPARATE_BY = {state.get('separate_by')!r}  # one highlight map per category, beside the overview")
    lines.append(f"SELECTED_X = {mp.get('selected_x')!r}")
    lines.append(f"SELECTED_Y = {mp.get('selected_y')!r}")
    lines.append(f"LOG_X = {mp.get('log_x', False)!r}")
    lines.append(f"LOG_Y = {mp.get('log_y', False)!r}")
    # The script reads better with a real None, and its guards test identity.
    marginal = mp.get('marginal_plot_type', 'None')
    lines.append(f"MARGINAL_PLOT_TYPE = {None if marginal == 'None' else marginal!r}")
    lines.append(f"FIT_REGRESSION = {mp.get('fit_regression', False)!r}")
    lines.append(f"FIT_GMM_2D = {mp.get('fit_gmm_2d', False)!r}")
    lines.append(f"GMM_MAX_COMPONENTS = {mp.get('gmm_max_components', 3)!r}")
    lines.append(f"GMM_MIN_WEIGHT_THRESHOLD = {mp.get('gmm_min_weight_threshold', 0.1)!r}")
    lines.append(f"FOCUS_CATEGORY = {mp.get('facet_focus')!r}"
                 "  # category promoted into the overview slot; None → the whole dataset")
    if mp.get("fit_gmm_2d"):
        lines.append(f"SAVE_DERIVED_DATA = False  # True → also write {EXPORT_METHODS[SPEC.name][1]} (the app's download button)")
    return lines


def _imports(state: dict) -> list[str]:
    mp = state.get("method_params", {})
    extra = ["from scipy.stats import gaussian_kde"]
    # distribution_ranges() widens the shared bounds with a chi2 radius in
    # every layout, so chi2 is no longer tied to separation or GMM.
    extra.append("from scipy.stats import chi2")
    # Pearson r + p is reported for every color group (like the app), so import
    # it unconditionally; only the regression line itself is gated.
    extra.append("from scipy.stats import pearsonr")
    if mp.get("fit_regression"):
        extra.append("from sklearn.linear_model import LinearRegression")
    if mp.get("fit_gmm_2d"):
        extra += ["from sklearn.mixture import GaussianMixture",
                  "from matplotlib.patches import Ellipse", "from scipy.stats import chi2"]
    return extra


def _packages(state: dict) -> set[str]:
    mp = state.get("method_params", {})
    packages = {"scipy"}
    if mp.get("fit_regression") or mp.get("fit_gmm_2d"):
        packages.add("scikit-learn")
    if mp.get("fit_gmm_2d"):
        packages.add("threadpoolctl")
    return packages


SPEC = MethodSpec(
    name="2D Feature Distribution",
    build=_build_2d_distribution,
    config_lines=_config_lines,
    imports=_imports,
    packages=_packages,
    # Marginal strips already share exact edges and have room for the legend.
    # tight_layout would rearrange that grid after widen_to_fit freezes it.
    footer=lambda state: "if not SEPARATE_BY and MARGINAL_PLOT_TYPE is None:\n    plt.tight_layout()",
)
