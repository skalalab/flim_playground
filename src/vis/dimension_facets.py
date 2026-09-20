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


def dimension_facet_layout(groups, x_range, y_range, aspect=None):
    """Proportional panel domains; plot_height is measured in plotting-width units.

    The overview and the complete grid share their top and bottom edges. With
    contiguous rows and the same aspect in every panel, the overview must be
    nrows times as wide as one small map. The chart wrapper uses plot_height to
    preserve equal coordinate scales as the available width changes.

    ``aspect`` overrides the ratio taken from the coordinate ranges. Dimension
    Reduction leaves it None because both axes share one embedding's units; 2D
    Distribution passes 1.0 because X and Y carry independent units and only a
    square frame reads the same in every panel.
    """
    overview = dict(x_domain=[0., 1.], y_domain=[0., 1.])
    if aspect is None:
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


def category_facet_groups(panels):
    """One column of panels from ordered ``category_panel_rows`` memberships.

    2D Distribution separates on exactly one column, so its layout is the
    single-column case of the Dimension Reduction grid. Positions stay with the
    caller; only the geometry needs row/column slots.
    """
    return dict(nrows=len(panels), ncols=1,
                panels=[dict(row=index, col=0, values=(level,))
                        for index, (level, _positions) in enumerate(panels)])


def focus_slot_keys(keys, focus=None):
    """Which membership each slot draws once ``focus`` is promoted.

    Slot 0 is the overview block and slots 1..N the panels in composition order;
    ``None`` means every row. Promotion swaps one panel with the overview, so the
    composition's domains never move and every other panel keeps its level. A
    ``focus`` a filter has removed from ``keys`` leaves the arrangement alone.
    """
    keys = list(keys)
    if focus is None or focus not in keys:
        return [None, *keys]
    return [focus, *(None if key == focus else key for key in keys)]


MAIN_PLOT_LABEL = "Main plot"


def as_facet_key(value):
    """Plotly serialization returns a matrix key as a list; compare them whole."""
    return tuple(value) if isinstance(value, list) else value


def _facet_axis_property(axis):
    """``'x4'`` names the layout domain ``'xaxis4'``; ``'x'`` names ``'xaxis'``."""
    return f"{axis[0]}axis{axis[1:]}"


def focus_facet_label(separate_by, key):
    """Name a promoted panel by its own column(s), e.g. ``day: Day 2``."""
    import html

    columns = [separate_by] if isinstance(separate_by, str) else list(separate_by or [])
    values = key if isinstance(key, tuple) else (key,)
    return " · ".join(html.escape(f"{column}: {value}")
                      for column, value in zip(columns, values))


def focus_facet_figure(fig, focus=None):
    """Promote ``focus`` into the main slot of an already built facet grid.

    Apply to a copy of the base figure, the way ``select_phasor_category`` does.
    Contents move between slots while every domain, range and axis stays where it
    was built, so a promotion refits nothing and recomputes no coordinate. The
    result is always computed from the unpromoted figure, which is what makes
    restoring "promote nothing" rather than an inverse operation.
    """
    meta = fig.layout.meta
    block = meta.get("facet_focus") if isinstance(meta, dict) else None
    if not block or not block.get("keys"):
        return fig
    keys = [as_facet_key(key) for key in block["keys"]]
    axes = [tuple(pair) for pair in block["axes"]]
    slots = focus_slot_keys(keys, as_facet_key(focus))
    promoted = slots[0]
    # Exactly one panel slot is vacated, and it is the promoted one. With no
    # promotion that lookup lands on the overview itself, so nothing trades.
    vacated = slots.index(None)
    traded = {0: axes[vacated], vacated: axes[0]}
    for trace in fig.data:
        trace_meta = trace.meta
        if not isinstance(trace_meta, dict) or "facet_slot" not in trace_meta:
            continue
        slot = trace_meta["facet_slot"]
        if trace_meta.get("facet_role") == "marginal":
            # Strips belong to the main block and describe whatever occupies it.
            trace.visible = slot == vacated
        elif slot in traded:
            trace.update(xaxis=traded[slot][0], yaxis=traded[slot][1])

    annotations = fig.layout.annotations
    slot_labels = list(block["slot_labels"])
    for index, label in zip(slot_labels, block["labels"]):
        if index is not None:
            annotations[index].text = label
    layout_block = dict(meta[block["layout_key"]])
    canonical = [dict(item) for item in layout_block["annotations"]]
    stamped = vacated and slot_labels[vacated - 1] is None
    if vacated and not stamped:
        annotations[slot_labels[vacated - 1]].text = MAIN_PLOT_LABEL
    stamp = block.get("stamp_annotation")
    if stamp is not None:
        placement = dict(text="", x=canonical[stamp]["x"], y=canonical[stamp]["y"])
        if stamped:
            # A matrix cell is named by its row and column, so the overview is
            # marked inside the cell it now occupies instead.
            domains = layout_block["axes"]
            x_domain = domains[_facet_axis_property(axes[vacated][0])]
            y_domain = domains[_facet_axis_property(axes[vacated][1])]
            placement = dict(text=MAIN_PLOT_LABEL, x=sum(x_domain) / 2, y=y_domain[1])
        annotations[stamp].update(**placement)
        canonical[stamp] = {**canonical[stamp], "x": placement["x"], "y": placement["y"]}
    # The promotion names itself in the plot title: the figure's own title,
    # augmented, so restoring is simply the title it was built with.
    base_title = block.get("title") or ""
    if promoted is not None or base_title:
        label = focus_facet_label(block.get("separate_by"), promoted) if promoted is not None else ""
        # A grid with a title of its own is augmented; one without — Dimension
        # Reduction — is titled by the promotion alone.
        fig.update_layout(title=dict(
            text=f"{base_title} ({label})" if base_title and label else label or base_title))
    layout_block["annotations"] = canonical
    fig.update_layout(meta={**meta, block["layout_key"]: layout_block,
                            "facet_focus": {**block, "applied": promoted}})
    return fig


def facet_focus_from_click(fig, selection, current=None):
    """The promotion a Plotly selection asks for, or ``current`` if it asks none.

    Resolved against the figure the click came from, so the curve index always
    describes what was on screen. A click on the overview restores the default
    arrangement; one on a legend swatch or any untagged trace changes nothing.
    """
    points = (selection or {}).get("points") or []
    if not points:
        return current
    curve = points[0].get("curve_number")
    if curve is None or curve >= len(fig.data):
        return current
    trace_meta = fig.data[curve].meta
    if not isinstance(trace_meta, dict) or "facet_slot" not in trace_meta:
        return current
    slot = trace_meta["facet_slot"]
    keys = (fig.layout.meta or {}).get("facet_focus", {}).get("keys") or []
    if slot == 0:
        return None
    return as_facet_key(keys[slot - 1]) if slot <= len(keys) else current
