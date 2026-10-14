"""Matplotlib helpers that SHIP into generated scripts and are used by two or more methods.

Everything in this file is pasted with inspect.getsource: no Streamlit, no relative imports,
pyplot imported inside function bodies. The app's versions return Plotly-specific formats.
"""


def create_color_map(groups, colormap, alpha=0.8, byte_colors=True):
    """Map group names to RGBA, matching Plotly's integer RGB channels."""
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
    return {g: (*(tuple(int(channel * 255) / 255 for channel in palette[i][:3])
                  if byte_colors else palette[i][:3]), alpha)
            for i, g in enumerate(groups)}


def create_shape_map(groups):
    """Map group names to Matplotlib marker symbols."""
    MPL_MARKERS = ['o', 's', 'D', 'P', 'X', '^', 'v', 'p', 'h', '8', '*', 'd']
    return {g: MPL_MARKERS[i % len(MPL_MARKERS)] for i, g in enumerate(groups)}


def scatter_with_encodings(ax, x, y, color, label, point_size,
                           shape_vals=None, shape_map=None,
                           opacity_vals=None, opacity_map=None,
                           base_alpha=0.7, linewidths=0.3, zorder=2):
    """Scatter one color group's points, grouped by shape.

    Matplotlib needs one call per marker shape. Apply opacity as a per-point alpha
    array to avoid layering by opacity level. Only the first nonempty shape group
    carries the legend label.
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
    alphas = (np.asarray([opacity_map[str(v)] for v in opacity_vals], dtype=float)
              if use_opacity else None)
    labeled = False
    for shape_key in (list(shape_map) if use_shape else [None]):
        mask = (shape_arr == shape_key) if use_shape else np.ones(len(x), dtype=bool)
        if not mask.any():
            continue
        ax.scatter(x[mask], y[mask], c=[color], s=point_size,
                   alpha=alphas[mask] if use_opacity else base_alpha,
                   marker=shape_map[shape_key] if use_shape else 'o',
                   edgecolors='DarkSlateGrey', linewidths=linewidths,
                   label=label if not labeled else None, zorder=zorder)
        labeled = True


def add_encoding_legend_entries(ax, shape_map, opacity_map, point_size, borderless=False):
    """Add gray proxy legend entries for opacity and shape groups
    (mirrors the app's helpers.add_point_legend_traces: opacity first, then shape)."""
    handles = []
    marker_style = dict(edgecolors='none', linewidths=0) if borderless else {}
    for group, alpha in (opacity_map or {}).items():
        handles.append(ax.scatter([], [], c='gray', alpha=alpha, marker='o',
                                  s=point_size, label=str(group), **marker_style))
    for group, marker in (shape_map or {}).items():
        handles.append(ax.scatter([], [], c='gray', alpha=0.8, marker=marker,
                                  s=point_size, label=str(group), **marker_style))
    return handles


def style_axes_like_app(ax, grid_axis="y"):
    # Streamlit's chart theme: light gridlines behind the data, no frame or tick marks.
    # The gridline colour is Plotly's gridcolor as the app renders it.
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color='#e6eaf1', linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def plotly_dash_style(dash, linewidth, webgl=False):
    """Match the app renderer's dash lengths despite Matplotlib's scaling."""
    import matplotlib.pyplot as plt

    patterns = {'dot': (1, 1), 'dash': (3, 3), 'dashdot': (3, 1, 1, 1),
                'longdash': (5, 5), 'longdashdot': (5, 2, 1, 2)}
    unit = max(linewidth, 3)
    if webgl:
        # Scattergl uses its own dash texture, with longer solid segments.
        patterns = {'dot': (1, 1), 'dash': (4, 1), 'dashdot': (4, 1, 1, 1),
                    'longdash': (8, 1), 'longdashdot': (8, 1, 1, 1)}
        unit = linewidth
    scale = linewidth if plt.rcParams['lines.scale_dashes'] else 1
    return (0, tuple(length * unit / scale for length in patterns[dash]))


def scatter_dimension_batch(ax, x, y, color, label, point_size,
                            shape_vals=None, shape_map=None,
                            opacity_vals=None, opacity_map=None, base_alpha=0.7):
    """One collection with per-point paths preserves a mixed batch's draw order."""
    from matplotlib.markers import MarkerStyle

    alphas = ([opacity_map[str(value)] for value in opacity_vals]
              if opacity_vals is not None and opacity_map else base_alpha)
    collection = ax.scatter(x, y, c=[color], alpha=alphas, s=point_size,
                            edgecolors='none', linewidths=0,
                            label=label, zorder=2)
    if shape_vals is not None and shape_map:
        paths = {}
        for value, marker in shape_map.items():
            style = MarkerStyle(marker)
            paths[value] = style.get_path().transformed(style.get_transform())
        collection.set_paths([paths[str(value)] for value in shape_vals])
    return collection


def scatter_interleaved_points(ax, df, x_col, y_col, group_column, color_groups,
                               color_map, point_size, shape_by=None, shape_map=None,
                               opacity_by=None, opacity_map=None, base_alpha=0.7,
                               show_counts=False, active_positions=None, labels=True):
    """Draw the app's global batches, optionally restricted to one category.

    Keep mixed shapes in a single collection so every point retains its shuffled
    position. Return one legend handle per visible color in natural group order;
    labels=False draws an axes that must not feed a legend.
    """
    import numpy as np

    from src.vis.dimension_facets import dimension_interleaved_indices
    from src.vis.helpers import format_group_label

    batches = dimension_interleaved_indices(
        df, group_column, color_groups, shape_by, shape_map, opacity_by, opacity_map)
    membership = np.ones(len(df), dtype=bool)
    if active_positions is not None:
        membership[:] = False
        membership[active_positions] = True
    counts = df.loc[membership, group_column].value_counts().to_dict()
    first_handles = {}
    for group, positions in batches:
        positions = positions[membership[positions]]
        if not len(positions):
            continue
        points = df.iloc[positions]
        label = (format_group_label(group, counts[group], show_counts, engine='mpl')
                 if labels and group not in first_handles else None)
        point_opacity_map = {key: alpha * color_map[group][3]
                             for key, alpha in (opacity_map or {}).items()}
        handle = scatter_dimension_batch(
            ax, points[x_col], points[y_col], color_map[group][:3], label, point_size,
            shape_vals=points[shape_by] if shape_by else None, shape_map=shape_map,
            opacity_vals=points[opacity_by] if opacity_by else None,
            opacity_map=point_opacity_map, base_alpha=base_alpha)
        first_handles.setdefault(group, handle)
    return [first_handles[group] for group in color_groups if group in first_handles]


def widen_to_fit(fig, pad=0.1, fit_height=False):
    """Fit the canvas horizontally, and optionally vertically, keeping axes sizes.

    The saved SVG is cropped to whatever was drawn, so it always holds a legend or
    label past the canvas edge; plt.show() draws only the canvas. Returns the inches
    added on the left, which shift anything later placed in figure coordinates.
    """
    fig.canvas.draw()
    content = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    extra_left = max(0., pad - content.x0)
    extra_right = max(0., content.x1 + pad - width)
    extra_bottom = max(0., pad - content.y0) if fit_height else 0.
    extra_top = max(0., content.y1 + pad - height) if fit_height else 0.
    if extra_left or extra_right or extra_bottom or extra_top:
        positions = [panel_ax.get_position().frozen() for panel_ax in fig.axes]
        new_width = width + extra_left + extra_right
        new_height = height + extra_bottom + extra_top
        fig.set_size_inches(new_width, new_height)
        for panel_ax, position in zip(fig.axes, positions):
            panel_ax.set_position([(position.x0 * width + extra_left) / new_width,
                                   (position.y0 * height + extra_bottom) / new_height,
                                   position.width * width / new_width,
                                   position.height * height / new_height])
    return extra_left


def facet_canvas(facet_layout, legend_entries, plot_width=10.0):
    """One figure sized for a facet layout, with room for labels and a shared legend below.

    Keep the layout's physical geometry; return the figure, a place_panel(x0, x1, y0, y1)
    that adds axes at a layout domain, and the margins the legend placement needs.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plot_height = plot_width * facet_layout["plot_height"]
    legend_rows = max(1, int(np.ceil(legend_entries / 5)))
    left_margin, right_margin = 0.85, 0.8
    bottom_margin, top_margin = 0.85 + 0.35 * legend_rows, 0.55
    figure_width = left_margin + plot_width + right_margin
    figure_height = bottom_margin + plot_height + top_margin
    fig = plt.figure(figsize=(figure_width, figure_height))

    def place_panel(x0, x1, y0, y1):
        return fig.add_axes([
            (left_margin + x0 * plot_width) / figure_width,
            (bottom_margin + y0 * plot_height) / figure_height,
            (x1 - x0) * plot_width / figure_width,
            (y1 - y0) * plot_height / figure_height,
        ])

    return fig, place_panel, left_margin, figure_width, figure_height


def place_facet_legend(fig, ax, handles, legend_size, left_margin, figure_width,
                       figure_height, legend_y=0.015):
    """One horizontal legend under the overview's axis title, sized to real text.

    ``handles`` come from the whole-dataset map, wherever a promotion moved it: the
    app's shared legend never follows the promotion.

    The side margins are guesses as well: a long level name, "Main plot" or a large
    y title can outrun them, so the canvas first widens to hold everything drawn.

    Measure before reserving space: large fonts and short grids can make the
    legend taller than the initial margin estimate. Wrap long combination labels
    within the canvas instead of widening the saved SVG, then grow only the
    bottom margin so every panel keeps its physical size and shared geometry.
    """
    left_margin += widen_to_fit(fig)
    figure_width = fig.get_figwidth()

    def make_legend(columns):
        legend = ax.legend(handles=handles, fontsize=legend_size, loc='lower left',
                           ncol=columns, frameon=False,
                           bbox_to_anchor=(left_margin / figure_width, legend_y),
                           bbox_transform=fig.transFigure)
        for handle in legend.legend_handles:
            handle.set_sizes([legend_size ** 2])
        return legend

    legend_columns = min(5, len(handles))
    legend = make_legend(legend_columns)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    while (legend_columns > 1 and
           legend.get_window_extent(renderer).x1 > fig.bbox.width - 0.15 * fig.dpi):
        legend_columns -= 1
        legend = make_legend(legend_columns)
    legend_top = legend.get_window_extent(renderer).y1
    title_bottom = ax.xaxis.label.get_window_extent(renderer).y0
    extra_bottom = max(0., (legend_top + 0.15 * fig.dpi - title_bottom)
                       / (fig.dpi * (1 - legend_y)))
    if extra_bottom:
        positions = [panel_ax.get_position().frozen() for panel_ax in fig.axes]
        old_height = figure_height
        figure_height += extra_bottom
        fig.set_size_inches(figure_width, figure_height)
        for panel_ax, position in zip(fig.axes, positions):
            panel_ax.set_position([
                position.x0, (position.y0 * old_height + extra_bottom) / figure_height,
                position.width, position.height * old_height / figure_height,
            ])
    return figure_height
