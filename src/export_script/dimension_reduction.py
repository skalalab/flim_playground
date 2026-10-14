"""Dimension Reduction export: its builder."""
from src.export_script.mpl_helpers import (
    facet_canvas,
    place_facet_legend,
    widen_to_fit,
)
from src.export_script.skeleton import (
    MethodSpec,
    _build_visual_encoding,
    _extract_source,
)


def _build_dimension_reduction(state: dict) -> str:
    from src.export_labels import available_label_column
    from src.vis.dimension_facets import (
        MAIN_PLOT_LABEL,
        dimension_facet_groups,
        dimension_facet_layout,
        dimension_ranges,
        focus_facet_label,
        focus_slot_keys,
        normalize_dimension_categories,
    )
    from src.vis.multivar import dimension_axis_titles, reduce_dimensions
    from src.vis.plot_defaults import (
        PANEL_CONTEXT_ALPHA,
        PANEL_CONTEXT_COLOR,
        panel_point_size,
    )

    facet_src = _extract_source(normalize_dimension_categories,
                                dimension_facet_groups, dimension_ranges,
                                dimension_facet_layout, focus_slot_keys, focus_facet_label,
                                available_label_column, facet_canvas,
                                widen_to_fit, place_facet_legend,
                                reduce_dimensions, dimension_axis_titles, panel_point_size)
    # The app's own values, written here by the export so they cannot drift.
    facet_src += (f"\nMAIN_PLOT_LABEL = {MAIN_PLOT_LABEL!r}\n"
                  f"PANEL_CONTEXT_COLOR = {PANEL_CONTEXT_COLOR!r}\n"
                  f"PANEL_CONTEXT_ALPHA = {PANEL_CONTEXT_ALPHA!r}\n")
    # Encoding maps and facet levels describe the observations that are actually
    # reduced. No facet changes the scaler, fit, color, shape, or opacity maps.
    retained = """
df = df.dropna(subset=SELECTED_FEATURES).copy()
df = normalize_dimension_categories(df, [*COLOR_BY, SHAPE_BY, OPACITY_BY, *SEPARATE_BY])
DR_GROUP_COLUMN = available_label_column(df.columns, "_color_group")
"""
    return facet_src + retained + _build_visual_encoding(group_column_expr="DR_GROUP_COLUMN", interleave=True) + """
# ============================================================
# Dimension Reduction
# ============================================================
# The app's own standardize-and-embed step (pasted above); UMAP is imported inside it.
coordinates, explained_variance = reduce_dimensions(df[SELECTED_FEATURES], DR_METHOD, HYPER_PARAMS)
X_reduced = coordinates.to_numpy()
xlabel, ylabel = dimension_axis_titles(coordinates, explained_variance)

df = df.copy()
DR_X_COLUMN = available_label_column(df.columns, "_dr_x")
DR_Y_COLUMN = available_label_column(df.columns, "_dr_y")
df[DR_X_COLUMN] = X_reduced[:, 0]
df[DR_Y_COLUMN] = X_reduced[:, 1]

x_range, y_range = dimension_ranges(df[DR_X_COLUMN], df[DR_Y_COLUMN])
facet_groups = dimension_facet_groups(df, SEPARATE_BY)
facet_layout = dimension_facet_layout(facet_groups, x_range, y_range)
facet_axes = []

if SEPARATE_BY:
    # One shared legend below the whole composition; the layout keeps its own geometry.
    fig, place_panel, left_margin, figure_width, figure_height = facet_canvas(
        facet_layout, len(color_groups) + len(shape_map) + len(opacity_map))
    for panel in [facet_layout["overview"], *facet_layout["panels"]]:
        facet_axes.append(place_panel(*panel["x_domain"], *panel["y_domain"]))
    ax, *facet_axes = facet_axes
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    # Fixed margins keep the frame independent of each method's tick labels.
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.12, top=0.96)

all_rows = np.ones(len(df), dtype=bool)
# Each slot draws one membership: the promoted panel in the overview block if
# any, every other panel its own, and the whole dataset wherever the overview
# sits. focus_slot_keys is the app's own arrangement rule.
panel_keys = [tuple(panel["values"]) for panel in facet_layout["panels"]]
panel_masks = dict(zip(panel_keys, (panel["mask"] for panel in facet_layout["panels"])))
slot_keys = focus_slot_keys(panel_keys, tuple(FOCUS_CATEGORY) if FOCUS_CATEGORY else None)
promoted_key = slot_keys[0]
for slot, (panel_ax, slot_key) in enumerate(zip([ax, *facet_axes], slot_keys)):
    is_overview = slot == 0
    membership = all_rows if slot_key is None else panel_masks[slot_key]
    # Points follow the slot, not the panel: the main block keeps POINT_SIZE.
    slot_point_size = POINT_SIZE if is_overview else panel_point_size(POINT_SIZE)
    if (~membership).any():
        background = df.loc[~membership]
        panel_ax.scatter(background[DR_X_COLUMN], background[DR_Y_COLUMN],
                         color=PANEL_CONTEXT_COLOR, alpha=PANEL_CONTEXT_ALPHA,
                         s=panel_point_size(POINT_SIZE) ** 2,
                         edgecolors='none', linewidths=0, zorder=1)
    # The app's global batch order, restricted to this slot's membership; the
    # whole-dataset slot carries the legend labels, counted over the whole dataset.
    scatter_interleaved_points(
        panel_ax, df, DR_X_COLUMN, DR_Y_COLUMN, DR_GROUP_COLUMN, color_groups, color_map,
        slot_point_size ** 2, shape_by=SHAPE_BY, shape_map=shape_map,
        opacity_by=OPACITY_BY, opacity_map=opacity_map, base_alpha=BASE_ALPHA,
        show_counts=SHOW_GROUP_COUNTS, labels=slot_key is None,
        active_positions=None if slot_key is None else np.flatnonzero(membership))
    panel_ax.set_xlim(x_range)
    panel_ax.set_ylim(y_range)
    panel_ax.set_aspect('equal', adjustable='box')
    panel_ax.grid(False, which='both')
    for side in ('left', 'bottom'):
        panel_ax.spines[side].set_visible(True)
        panel_ax.spines[side].set_color('black')
        panel_ax.spines[side].set_linewidth(1)
    for side in ('top', 'right'):
        panel_ax.spines[side].set_visible(False)
    if not is_overview:
        panel_ax.tick_params(axis='both', which='both', bottom=False, left=False,
                             labelbottom=False, labelleft=False)

for facet_ax, panel, slot_key in zip(facet_axes, facet_layout["panels"], slot_keys[1:]):
    if len(SEPARATE_BY) == 1:
        facet_ax.text(1.04, 0.5,
                      MAIN_PLOT_LABEL if slot_key is None else panel["values"][0],
                      transform=facet_ax.transAxes,
                      ha='left', va='center', fontsize=LEGEND_SIZE)
    else:
        if panel["row"] == 0:
            facet_ax.text(0.5, 1.03, panel["values"][1], transform=facet_ax.transAxes,
                          ha='center', va='bottom', fontsize=LEGEND_SIZE)
        if panel["col"] == facet_groups["ncols"] - 1:
            facet_ax.text(1.04, 0.5, panel["values"][0], transform=facet_ax.transAxes,
                          ha='left', va='center', fontsize=LEGEND_SIZE)
        if slot_key is None:
            # A cell is named by its row and column, so the overview is marked
            # inside the cell it now occupies instead of renaming an edge.
            facet_ax.text(0.5, 0.97, MAIN_PLOT_LABEL, transform=facet_ax.transAxes,
                          ha='center', va='top', fontsize=LEGEND_SIZE,
                          bbox=dict(facecolor='white', alpha=0.75,
                                    edgecolor='none', pad=2))
if promoted_key is not None:
    # Dimension Reduction draws no title of its own, so the promotion is it.
    fig.suptitle(focus_facet_label(SEPARATE_BY, promoted_key, escape=False),
                 fontsize=TITLE_SIZE, fontweight='bold', x=0.01, ha='left')

ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
# The app's axis lines carry no tick marks.
ax.tick_params(axis='both', labelsize=AXIS_LABEL_SIZE - 2, length=0)
# One shared legend, counted over the whole dataset, wherever that map sits.
point_legend_handles = [ax, *facet_axes][slot_keys.index(None)].get_legend_handles_labels()[0]
encoding_legend_handles = add_encoding_legend_entries(
    ax, shape_map, opacity_map, LEGEND_SIZE ** 2, borderless=True)
if SEPARATE_BY:
    figure_height = place_facet_legend(fig, ax, point_legend_handles + encoding_legend_handles,
                                       LEGEND_SIZE, left_margin, figure_width, figure_height)
else:
    legend = ax.legend(handles=point_legend_handles + encoding_legend_handles,
                       fontsize=LEGEND_SIZE, loc='upper left', frameon=False,
                       bbox_to_anchor=(1.02, 1), borderaxespad=0)
    for handle in legend.legend_handles:
        handle.set_sizes([LEGEND_SIZE ** 2])
    # The legend sits beside the fixed frame, past the canvas edge.
    widen_to_fit(fig)
"""


def _config_lines(state: dict) -> list[str]:
    mp = state.get("method_params", {})
    focus = mp.get("facet_focus")
    return [
        f"SELECTED_FEATURES = {mp.get('selected_features', [])!r}",
        f"DR_METHOD = {mp.get('dr_method', 'PCA')!r}",
        f"HYPER_PARAMS = {mp.get('hyperParam_dict', {})!r}",
        f"SEPARATE_BY = {state.get('separate_by') or []!r}  # ordered row and column categories",
        (f"FOCUS_CATEGORY = {tuple(focus) if focus else None!r}"
         "  # panel promoted into the overview slot; None → the whole dataset"),
    ]


def _imports(state: dict) -> list[str]:
    # reduce_dimensions names PCA and TSNE (scikit-learn is always declared) and
    # imports umap inside its own branch, so DR_METHOD stays editable in the script.
    return ["from sklearn.preprocessing import StandardScaler",
            "from sklearn.decomposition import PCA",
            "from sklearn.manifold import TSNE"]


def _packages(state: dict) -> set[str]:
    packages = {"scikit-learn"}
    if state.get("method_params", {}).get("dr_method", "PCA") == "UMAP":
        packages.add("umap-learn")
    return packages


SPEC = MethodSpec(
    name="Dimension Reduction",
    build=_build_dimension_reduction,
    config_lines=_config_lines,
    imports=_imports,
    packages=_packages,
    # The facet composition and the beside-legend fix their own geometry.
    footer=lambda state: "",
)
