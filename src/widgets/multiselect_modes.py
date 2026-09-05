"""Shared "All" / "Except:" selection modes for filters and feature pickers.

Sentinels share the selection list with values. "Except:" treats those values as
exclusions, expanded against the current data on every rerun. Its short label
fits the narrow columns used by grouped pickers.
"""

import streamlit as st

ALL_LABEL = "All"
EXCEPT_LABEL = "Except:"

SENTINELS = (ALL_LABEL, EXCEPT_LABEL)


def normalize_mode_selection(key):
    """``on_change`` callback keeping the two sentinels coherent, last pick wins.

    - picking "All" clears everything else, since it already means the whole set.
    - picking "Except:" drops "All" but keeps the values already chosen, so ticking it
      after picking X turns "just X" into "everything but X" in one click.
    - picking a value drops "All" but *keeps* "Except:", which is what makes the
      exclude list additive.

    The sentinel is hoisted to the front so the chips read left-to-right as "Except: | X".
    """
    current = list(st.session_state.get(key, [ALL_LABEL]))
    if current and current[-1] == ALL_LABEL:
        normalized = [ALL_LABEL]
    else:
        values = [item for item in current if item not in SENTINELS]
        normalized = [EXCEPT_LABEL, *values] if EXCEPT_LABEL in current else values
    if normalized != current:
        st.session_state[key] = normalized


def chosen_items(stored, universe):
    """Desugar a stored selection into the items it actually chooses, in `universe` order.

    Returns ``None`` for "no constraint" -- the callers disagree on what that means (the
    filters skip the mask entirely, the feature pickers take every feature), so neither
    convention is baked in here.

        [ALL_LABEL]              -> None
        [EXCEPT_LABEL]           -> None                (excluding nothing)
        [EXCEPT_LABEL, "B"]      -> universe minus "B"
        ["B"]                    -> ["B"]
        []                       -> []                  (chooses nothing)

    "All" wins over "Except:" if both somehow appear together, being the wider of the
    two. `normalize_mode_selection` never produces that pair.
    """
    if ALL_LABEL in stored:
        return None
    values = [item for item in stored if item not in SENTINELS]
    if EXCEPT_LABEL in stored:
        if not values:
            return None
        excluded = set(values)
        return [item for item in universe if item not in excluded]
    return values


def excluded_items(stored):
    """The values an exclude-mode selection drops, or ``None`` if it is not in that mode."""
    if EXCEPT_LABEL not in stored or ALL_LABEL in stored:
        return None
    return [item for item in stored if item not in SENTINELS]
