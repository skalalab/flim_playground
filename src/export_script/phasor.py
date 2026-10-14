"""Phasor Plot export: its builders and the helpers only it pastes."""
from src.export_script.mpl_helpers import style_axes_like_app
from src.export_script.skeleton import (
    MethodSpec,
    _build_visual_encoding,
    _extract_source,
)


def place_phasor_legend(ax, handles, legend_size, frequency_label):
    """Beside the plot, as in the app, and clear of the frequency label.

    The app's plot area runs on past that label; these equal-aspect axes stop at
    g = 1.05, which a large label overhangs. Its width is fixed in points, so one
    measurement anchors the legend past whichever reaches further right, and the
    anchor follows any later layout.
    """
    from matplotlib.transforms import blended_transform_factory, offset_copy

    fig = ax.figure
    fig.canvas.draw()
    label_box = frequency_label.get_window_extent()
    x, x_transform = 1.02, ax.transAxes
    if label_box.x1 > ax.get_window_extent().x1:
        x = frequency_label.get_position()[0]
        x_transform = offset_copy(ax.transData, fig=fig, units='points',
                                  x=label_box.width * 72 / fig.dpi + legend_size / 2)
    legend = ax.legend(handles=handles, fontsize=legend_size, loc='upper left', frameon=False,
                       borderaxespad=0, bbox_to_anchor=(x, 1),
                       bbox_transform=blended_transform_factory(x_transform, ax.transAxes))
    for handle in legend.legend_handles:
        handle.set_sizes([legend_size ** 2])
    return legend


def draw_phasor_background(ax, f, harmonic, annotation_size):
    """Render the app's reference geometry and its three numeric axis labels."""
    from src.vis.bivar import phasor_reference_geometry

    geometry = phasor_reference_geometry(f, harmonic)
    ax.plot(geometry['curve_x'], geometry['curve_y'], color='black', linewidth=2, zorder=1)
    ax.plot([0, 1], [0, 0], color='black', linewidth=2, zorder=1)
    ax.plot([0, 0], [0, .5], color='black', linewidth=2, zorder=1)
    for x, y in zip(geometry['marker_x'], geometry['marker_y']):
        ax.plot(x, y, 'ko', markersize=7, markeredgewidth=0, zorder=1)
    for index, (text, x, y) in enumerate(zip(
            geometry['lifetime_labels'], geometry['label_x'], geometry['label_y'])):
        ax.annotate(text, (geometry['marker_x'][index], geometry['marker_y'][index]),
                    xytext=(x, y), textcoords='data', ha='left', va='center',
                    fontsize=annotation_size, zorder=3)
    for text, x, y, horizontal, vertical in [
            ('0.5', -.02, .5, 'right', 'center'),
            ('0', 0, -.02, 'center', 'top'),
            ('1', 1, -.02, 'center', 'top')]:
        ax.text(x, y, text, ha=horizontal, va=vertical, fontsize=annotation_size)
    # Plotly hides automatic ticks; only the explicit labels above are drawn.
    ax.tick_params(axis='both', which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    return ax.text(.8, .5, geometry['frequency'], fontsize=annotation_size,
                   ha='left', va='center')


def _build_phasor_plot(state: dict) -> str:
    from src.export_labels import available_label_column
    from src.vis.bivar import (
        _phasor_panel_rows,
        category_panel_rows,
        phasor_columns,
        phasor_reference_geometry,
        resolve_phasor_category,
    )
    from src.vis.plot_defaults import panel_point_size

    preparation = ("\n# Grouping and legend helpers (extracted from FLIM Playground source)\n"
                   + _extract_source(available_label_column, place_phasor_legend, panel_point_size,
                                     phasor_reference_geometry, draw_phasor_background,
                                     style_axes_like_app, phasor_columns,
                                     resolve_phasor_category) + """
# Both Phasor layouts build their encodings from complete G/S observations.
g_col, s_col, harmonic_label = phasor_columns(PHASOR_CHANNEL, PHASOR_HARMONIC)
if g_col not in df.columns or s_col not in df.columns:
    raise SystemExit(f"ERROR: Columns {g_col} and/or {s_col} not found in data.")
df = df.dropna(subset=[g_col, s_col]).copy()
if df.empty:
    raise ValueError("No complete G/S observations remain for Phasor Plot.")
PHASOR_GROUP_COLUMN = available_label_column(df.columns, "_color_group")
""")
    helper_src = "\n# Shared Phasor panel helpers\n" + _extract_source(category_panel_rows, _phasor_panel_rows)

    return preparation + helper_src + _build_visual_encoding(
        group_column_expr="PHASOR_GROUP_COLUMN", interleave=True
    ) + """
# ============================================================
# Phasor Plot
# ============================================================
# SEPARATE_BY names a column: one category is drawn in full and the rest as grey context.
phasor_panels = _phasor_panel_rows(df, SEPARATE_BY, COLOR_BY) if SEPARATE_BY is not None else []
panel_levels = [level for level, _positions in phasor_panels]
phasor_category = resolve_phasor_category(panel_levels, PHASOR_CATEGORY)
active_panel_index = (
    panel_levels.index(phasor_category) if phasor_category is not None else None
)
if active_panel_index is not None:
    active_positions = phasor_panels[active_panel_index][1]
elif SEPARATE_BY is not None:
    active_positions = np.array([], dtype=int)
else:
    active_positions = None  # no separation: every point is foreground

fig, ax = plt.subplots(figsize=(10, 6))
frequency_label = draw_phasor_background(ax, PHASOR_F, PHASOR_HARMONIC, AXIS_LABEL_SIZE)
if active_positions is not None and len(active_positions) < len(df):
    other_positions = np.setdiff1d(
        np.arange(len(df)), active_positions, assume_unique=True
    )
    other_df = df.iloc[other_positions]
    ax.scatter(
        other_df[g_col], other_df[s_col], color=PANEL_CONTEXT_COLOR, alpha=0.18,
        s=panel_point_size(POINT_SIZE) ** 2, edgecolors='none', linewidths=0,
        label='_nolegend_', zorder=0,
    )

point_legend_handles = scatter_interleaved_points(
    ax, df, g_col, s_col, PHASOR_GROUP_COLUMN, color_groups, color_map,
    POINT_SIZE ** 2, shape_by=SHAPE_BY, shape_map=shape_map,
    opacity_by=OPACITY_BY, opacity_map=opacity_map,
    base_alpha=BASE_ALPHA, show_counts=SHOW_GROUP_COUNTS,
    active_positions=active_positions,
)

encoding_legend_handles = add_encoding_legend_entries(
    ax, shape_map, opacity_map, POINT_SIZE ** 2, borderless=True)

ax.set_xlabel("g", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel("s", fontsize=AXIS_LABEL_SIZE)
ax.set_title(
    f"{PHASOR_CHANNEL} {harmonic_label} Harmonic Phasor",
    fontsize=TITLE_SIZE, fontweight='bold', loc='left',
)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 0.55)
ax.set_aspect('equal')
ax.tick_params(axis='both', labelsize=AXIS_LABEL_SIZE - 2)
style_axes_like_app(ax, grid_axis=None)

category_label = None
if phasor_category is not None:
    category_label = ax.text(
        0, 0, f"{SEPARATE_BY}: {phasor_category}", transform=fig.transFigure,
        ha='center', va='top', fontsize=AXIS_LABEL_SIZE,
        fontweight='bold', clip_on=False,
    )
    category_label.set_in_layout(False)

legend_handles = point_legend_handles + encoding_legend_handles
if legend_handles:
    place_phasor_legend(ax, legend_handles, LEGEND_SIZE, frequency_label)

if SEPARATE_BY is not None:
    # Reserve rendered space below the g label for the active category.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    category_height = (
        category_label.get_window_extent(renderer).height if category_label is not None else 0
    )
    layout_bottom = (category_height + 0.30 * fig.dpi) / fig.bbox.height
    for _ in range(4):
        fig.tight_layout(rect=(0, layout_bottom, 1, 1))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if category_label is None:
            break
        xlabel_box = ax.xaxis.label.get_window_extent(renderer)
        axes_box = ax.get_position()
        category_label.set_position((
            axes_box.x0 + axes_box.width / 2,
            (xlabel_box.y0 - 0.15 * fig.dpi) / fig.bbox.height,
        ))
        fig.canvas.draw()
        category_box = category_label.get_window_extent(renderer)
        shortfall = 0.15 * fig.dpi - category_box.y0
        if shortfall <= 0:
            break
        layout_bottom += shortfall / fig.bbox.height
    if category_label is not None:
        # Manual positioning kept this out of tight_layout; include it in the final
        # bbox_inches='tight' calculation so long category names are not clipped.
        category_label.set_in_layout(True)
"""


# The app's Phasor figure sets its own title font (Arial 20, src/vis/bivar.py).
_PHASOR_TITLE_FONT_SIZE = 20


def _config_lines(state: dict) -> list[str]:
    from src.vis.plot_defaults import PANEL_CONTEXT_COLOR

    mp = state.get("method_params", {})
    return [
        f"PANEL_CONTEXT_COLOR = {PANEL_CONTEXT_COLOR!r}",
        f"SEPARATE_BY = {state.get('separate_by')!r}",
        f"PHASOR_CHANNEL = {mp.get('selected_channel')!r}",
        f"PHASOR_HARMONIC = {mp.get('phasor_harmonic', 1)!r}",
        f"PHASOR_F = {mp.get('phasor_f', 0.08)!r}",
        f"PHASOR_CATEGORY = {mp.get('phasor_category')!r}",
    ]


SPEC = MethodSpec(
    name="Phasor Plot",
    build=_build_phasor_plot,
    config_lines=_config_lines,
    imports=lambda state: [],
    packages=lambda state: set(),
    # The separated layout fits its own category label below the axes at run time.
    footer=lambda state: "if SEPARATE_BY is None:\n    plt.tight_layout()",
    title_size=_PHASOR_TITLE_FONT_SIZE,
)
