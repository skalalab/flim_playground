"""Pure layout and membership shared by Dimension Reduction and script export."""
import numpy as np

from .helpers import natural_tuple_sort


def normalize_dimension_categories(df, columns):
    """Keep missing categorical observations in the loader's existing N/A group."""
    df = df.copy()
    for column in dict.fromkeys(c for c in columns if c):
        values = df[column]
        df[column] = values.astype(str).where(values.notna(), "N/A")
    return df


def dimension_interleaved_indices(df, color_column, color_groups, shape_by=None,
                                  shape_map=None, opacity_by=None, opacity_map=None,
                                  random_seed=42, num_batches=15):
    """Positional row batches matching the app's seeded color interleaving.

    Form each color's rows in shape/opacity subgroup order before shuffling,
    just as add_interleaved_points_trace does. Facets subset these global batches.
    """
    import math
    import random

    columns = [color_column]
    if shape_by:
        columns.append(shape_by)
    if opacity_by:
        columns.append(opacity_by)
    subgroups = {}
    for position, values in enumerate(df[columns].itertuples(index=False, name=None)):
        key = (values[0], values[1] if shape_by else None,
               values[-1] if opacity_by else None)
        subgroups.setdefault(key, []).append(position)
    shape_rank = {value: index for index, value in enumerate(shape_map or {})}
    opacity_rank = {value: index for index, value in enumerate(opacity_map or {})}
    rng = random.Random(random_seed)
    batches = {}
    for group in color_groups:
        keys = sorted((key for key in subgroups if key[0] == group),
                      key=lambda key: (shape_rank.get(key[1], 0), opacity_rank.get(key[2], 0)))
        indices = np.asarray([position for key in keys for position in subgroups[key]], dtype=int)
        if not len(indices):
            continue
        order = list(range(len(indices)))
        rng.shuffle(order)
        indices = indices[order]
        count = min(num_batches, max(1, len(indices) // 5))
        size = math.ceil(len(indices) / count)
        batches[group] = [indices[start:start + size] for start in range(0, len(indices), size)]
    return [(group, batches[group][batch])
            for batch in range(max((len(value) for value in batches.values()), default=0))
            for group in color_groups if group in batches and batch < len(batches[group])]


def dimension_facet_groups(df, separate_by=None):
    """Return ordered panel memberships, including empty matrix intersections."""
    from collections.abc import Sequence

    if separate_by is None:
        separate_by = []
    if isinstance(separate_by, str) or not isinstance(separate_by, Sequence):
        raise ValueError("Separate by must be an ordered sequence of up to two columns.")
    separate_by = list(separate_by)
    if len(separate_by) > 2 or len(set(separate_by)) != len(separate_by):
        raise ValueError("Separate by accepts up to two distinct columns.")
    if any(column not in df.columns for column in separate_by):
        raise ValueError("Separate by columns must be present in the data.")
    result = dict(separate_by=separate_by, row_levels=[], column_levels=[],
                  nrows=0, ncols=0, panels=[])
    if not separate_by or df.empty:
        return result
    categories = normalize_dimension_categories(df[separate_by], separate_by)
    levels = [natural_tuple_sort(categories[column].unique()) for column in separate_by]
    if len(separate_by) == 1:
        result.update(ncols=1, nrows=len(levels[0]), row_levels=levels[0])
        positions = [(i, 0, (value,)) for i, value in enumerate(levels[0])]
    else:
        result.update(nrows=len(levels[0]), ncols=len(levels[1]),
                      row_levels=levels[0], column_levels=levels[1])
        positions = [(r, c, (first, second)) for r, first in enumerate(levels[0])
                     for c, second in enumerate(levels[1])]
    for row, col, values in positions:
        mask = np.ones(len(df), dtype=bool)
        for column, value in zip(separate_by, values):
            mask &= categories[column].to_numpy() == value
        result["panels"].append(dict(row=row, col=col, values=values, mask=mask))
    return result


def dimension_ranges(x, y, panel_aspect=0.72):
    """Centered bounds preserve equal units in a compact, stable method frame.

    Pad both coordinate extents, then expand to a fixed height/width proportion.
    This changes only the displayed bounds, never the embedding coordinates.
    """
    ranges = []
    for values in (x, y):
        low, high = float(np.min(values)), float(np.max(values))
        padding = (high - low) * 0.05 or max(abs(low) * 0.05, 0.5)
        ranges.append([low - padding, high + padding])
    half_width = max(ranges[0][1] - ranges[0][0],
                     (ranges[1][1] - ranges[1][0]) / panel_aspect) / 2
    return tuple([(low + high) / 2 - half_span, (low + high) / 2 + half_span]
                 for (low, high), half_span in zip(ranges, (half_width, half_width * panel_aspect)))


def dimension_facet_layout(groups, x_range, y_range):
    """Proportional panel domains; plot_height is measured in plotting-width units.

    The overview and the complete grid share their top and bottom edges. With
    contiguous rows and the same aspect in every panel, the overview must be
    nrows times as wide as one small map. The chart wrapper uses plot_height to
    preserve equal coordinate scales as the available width changes.
    """
    overview = dict(x_domain=[0., 1.], y_domain=[0., 1.])
    aspect = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
    if not groups["panels"]:
        return dict(overview=overview, panels=[], plot_height=aspect)
    width = 0.96 / (groups["nrows"] + groups["ncols"])
    overview_width = groups["nrows"] * width
    plot_height = overview_width * aspect
    overview = dict(x_domain=[0., overview_width], y_domain=[0., 1.])
    # Shared edges must be bit-identical: Plotly uses strict comparisons to
    # detect overlapping subplots, which changes background/axis layering.
    x_edges = [overview_width + 0.04 + column * width
               for column in range(groups["ncols"] + 1)]
    x_edges[-1] = 1.0
    panels = []
    for panel in groups["panels"]:
        panels.append(dict(panel, x_domain=x_edges[panel["col"]:panel["col"] + 2],
                           y_domain=[1. - (panel["row"] + 1) / groups["nrows"],
                                     1. - panel["row"] / groups["nrows"]]))
    return dict(overview=overview, panels=panels, plot_height=plot_height)
