"""Pure decisions and status messages for visual-encoding controls."""
from src.column_roles import code_span


POINT_MODES = ("opacity", "subcolor", "shape")


def initial_point_encoding(picker_column, as_colour=False, opacity_column=None):
    """Migrate legacy controls, preferring the shared picker's existing intent."""
    if as_colour:
        return "subcolor", picker_column
    if picker_column is not None:
        return "shape", picker_column
    if opacity_column is not None:
        return "opacity", opacity_column
    return "shape", None


def resolve_point_mode(selected, last_mode="shape"):
    """Keep one mode active when a selected native segment is clicked again."""
    if selected in POINT_MODES:
        return selected
    return last_mode if last_mode in POINT_MODES else "shape"


def point_encoding_channels(mode, column, has_color_groups):
    """Return only the active mapping as ``(opacity, shape, subcolor)``."""
    return (
        column if mode == "opacity" else None,
        column if mode == "shape" else None,
        column if mode == "subcolor" and has_color_groups else None,
    )


def prune_to_options(stored, options, fallback=None):
    """Restrict a stored scalar or list to the current widget options.

    An unavailable scalar becomes None. For lists, ``fallback`` replaces a
    nonempty selection only when every item is removed; an empty list stays empty.
    """
    offered = set(options)
    if isinstance(stored, list):
        kept = [value for value in stored if value in offered]
        if stored and not kept and fallback is not None:
            return list(fallback)
        return kept
    return stored if stored in offered else None


def color_multiselect_label(show_subcolor, as_colour):
    """Use "Group by" when the point-encoding selector is in Subcolor mode.

    The mode determines which control offers color, even before a column is picked.
    """
    return "Group by" if (show_subcolor and as_colour) else "Color by"


def drop_varying_channels(channels, varied):
    """Disable channels whose columns varied within a collapse group.

    A collapsed point cannot carry one decoration value for those columns.
    Resolve this before export captures page state so the app and script agree.
    Return ``(kept, dropped)``: a role-to-column map with disabled values set to
    None, and a map containing only the disabled channels.
    """
    varied = set(varied)
    kept = {role: (None if column in varied else column)
            for role, column in channels.items()}
    dropped = {role: column for role, column in channels.items()
               if column and column in varied}
    return kept, dropped


# Display names for decoration controls.
_CHANNEL_LABELS = {"shape": "Shape by", "opacity": "Opacity by", "subcolor": "Subcolor by"}


def dropped_channel_note(role, collapse_by, column):
    """Explain why collapse disabled a decoration channel, as Markdown.

    Escape file-provided column names with ``code_span`` while retaining the
    markup on the control label.
    """
    return (f"**{_CHANNEL_LABELS[role]}** is off — one {code_span(collapse_by)} point "
            f"covers several {code_span(column)} values, so it cannot be further "
            "divided.")
