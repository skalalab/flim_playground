"""Pure decisions behind the visual-encoding row.

Split out of visualization_widgets.py because Streamlit widgets return their defaults
in bare mode and AppTest cannot reach the analysis page, so anything left inside the
widget function can only be checked by hand. These two decisions are the parts with
real branching, so they live here and get tests.
"""


def prune_to_options(stored, options, fallback=None):
    """``stored`` narrowed to ``options``, so a widget is never handed a value it
    does not offer.

    Streamlit *raises* when session state holds an unoffered value (auto-generated
    keys reset silently instead, which is why this only became necessary once the
    controls took explicit keys). ``options`` shrinks whenever a filter collapses a
    column to a single value -- the encoding row offers only categories with
    ``nunique() > 1`` -- or when another control claims the column.

    ``fallback`` applies only when a *non-empty* selection prunes to nothing: an
    already-empty selection is a deliberate state, not damage to repair.
    """
    offered = set(options)
    if isinstance(stored, list):
        kept = [value for value in stored if value in offered]
        if stored and not kept and fallback is not None:
            return list(fallback)
        return kept
    return stored if stored in offered else None


def color_multiselect_label(show_subcolor, as_colour):
    """``"Group by"`` when the third slot has claimed colour, else ``"Color by"``.

    Depends on the switch alone, deliberately -- not on whether a column has been picked
    yet. The moment the switch goes on, the third slot is the one offering colour, so
    gating this on the picker holding something would leave two controls both presenting
    themselves as the colour channel until one was chosen.

    The cost is that with the switch on and nothing picked, colour still comes from these
    groups while the label already reads "Group by". That is the intent the switch
    expresses and it resolves as soon as a column is chosen; two controls claiming one
    channel never resolves.

    The third slot no longer spells "Color by" on screen -- it shows a static
    ``Opacity [switch] subcolor by`` phrase and lets the knob say which half is live -- so
    the two can never collide as identical visible text. What must not collide is the
    CLAIM: exactly one of this label and the switch offers colour, which is what
    ``check_encoding_row`` asserts.
    """
    return "Group by" if (show_subcolor and as_colour) else "Color by"


def drop_varying_channels(channels, varied):
    """Resolve the decoration channels against the columns a collapse dropped.

    A collapsed dot is a mean over several cells, so a decoration only means anything
    when its column holds ONE value in every collapse group. ``varied`` is what
    ``collapse.collapse_rows`` reports it dropped for failing that test, and a channel
    pointing at such a column has nothing to look up -- not a missing value, but
    several crushed into one row.

    Resolving it here rather than inside the plot is what keeps the export honest:
    ``_export_script_button`` captures whatever the page holds, so the generated script
    receives ``shape_by=None`` already decided and needs no collapse logic of its own.

    Returns ``(kept, dropped)`` -- the same ``{role: column}`` mapping with offenders
    set to ``None``, and just the offenders, so the caller can say which channel it
    switched off and why.
    """
    varied = set(varied)
    kept = {role: (None if column in varied else column)
            for role, column in channels.items()}
    dropped = {role: column for role, column in channels.items()
               if column and column in varied}
    return kept, dropped
