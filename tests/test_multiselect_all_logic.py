"""The shared multiselect callback matches All and Except: by exact equality.
Labels containing All remain ordinary values. Tests cover the selection rules used
by categorical filters and feature pickers.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets.multiselect_modes import (
    ALL_LABEL,
    EXCEPT_LABEL,
    normalize_mode_selection,
)


def _normalize(key, stored):
    st.session_state[key] = list(stored)
    normalize_mode_selection(key)
    return st.session_state[key]


# --- "All" is matched by equality, not by substring ---------------------------

def test_keeps_label_that_contains_all_substring():
    # "Allo" (allogeneic) genuinely contains the capital "All" substring.
    assert _normalize("k_feat", ["feat1", "Allo"]) == ["feat1", "Allo"]


def test_keeps_label_that_contains_all_substring_when_picked_last():
    assert _normalize("k_feat_last", ["feat1", "Overall"]) == ["feat1", "Overall"]


def test_collapses_when_all_is_the_real_last_choice():
    assert _normalize("k_feat2", ["feat1", ALL_LABEL]) == [ALL_LABEL]


def test_all_picked_last_clears_an_except_selection_too():
    assert _normalize("k_feat3", [EXCEPT_LABEL, "feat1", ALL_LABEL]) == [ALL_LABEL]


def test_all_picked_first_is_dropped_when_a_value_follows():
    # "All" already means the whole set, so choosing a value after it narrows.
    assert _normalize("k_feat4", [ALL_LABEL, "feat1"]) == ["feat1"]


# --- "Except:" stays additive and leads the chips ----------------------------

def test_except_keeps_values_already_chosen():
    # Ticking "Except:" after picking feat1 turns "just feat1" into "all but feat1".
    assert _normalize("k_exc", ["feat1", EXCEPT_LABEL]) == [EXCEPT_LABEL, "feat1"]


def test_picking_a_value_keeps_except_so_exclusions_are_additive():
    assert _normalize("k_exc2", [EXCEPT_LABEL, "feat1", "feat2"]) == [
        EXCEPT_LABEL,
        "feat1",
        "feat2",
    ]


def test_except_drops_all_but_keeps_the_values():
    assert _normalize("k_exc3", [ALL_LABEL, "feat1", EXCEPT_LABEL]) == [
        EXCEPT_LABEL,
        "feat1",
    ]


def test_already_normalized_selection_is_left_untouched():
    key = "k_stable"
    st.session_state[key] = [EXCEPT_LABEL, "feat1"]
    normalize_mode_selection(key)
    normalize_mode_selection(key)
    assert st.session_state[key] == [EXCEPT_LABEL, "feat1"]
