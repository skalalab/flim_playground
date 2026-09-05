"""Resolve All and Except: selections for feature pickers and categorical filters.
chosen_items returns None for no constraint: filters skip masking, while feature
pickers select every feature. Removed filter values are pruned; an emptied
selection falls back to All.
"""
import pandas as pd

from src.widgets.filter_widgets import resolve_selections
from src.widgets.multiselect_modes import (
    ALL_LABEL,
    EXCEPT_LABEL,
    chosen_items,
    excluded_items,
)

FEATURES = ["feat1", "feat2", "feat3"]


# --- chosen_items -------------------------------------------------------------

def test_all_means_no_constraint():
    assert chosen_items([ALL_LABEL], FEATURES) is None


def test_plain_selection_returns_exactly_those_items():
    assert chosen_items(["feat2", "feat3"], FEATURES) == ["feat2", "feat3"]


def test_except_returns_the_universe_minus_the_excluded():
    assert chosen_items([EXCEPT_LABEL, "feat2"], FEATURES) == ["feat1", "feat3"]


def test_except_follows_universe_order_not_selection_order():
    assert chosen_items([EXCEPT_LABEL, "feat1"], FEATURES) == ["feat2", "feat3"]


def test_except_excluding_nothing_is_no_constraint():
    assert chosen_items([EXCEPT_LABEL], FEATURES) is None


def test_empty_selection_chooses_nothing_and_is_not_no_constraint():
    # Distinct from None: an empty pick is a real state that selects no rows.
    assert chosen_items([], FEATURES) == []


def test_all_wins_over_except_being_the_wider_of_the_two():
    assert chosen_items([ALL_LABEL, EXCEPT_LABEL, "feat2"], FEATURES) is None


# --- excluded_items -----------------------------------------------------------

def test_excluded_items_reports_the_dropped_values():
    assert excluded_items([EXCEPT_LABEL, "feat2"]) == ["feat2"]


def test_excluded_items_is_none_outside_exclude_mode():
    assert excluded_items(["feat2"]) is None
    assert excluded_items([ALL_LABEL]) is None


# --- stale values are pruned, an emptied selection falls back to "All" ---------

def _frame():
    return pd.DataFrame({"cat_a": ["x", "y"], "cat_b": ["p", "q"]})


def test_value_absent_from_the_data_is_dropped():
    resolved, _, _ = resolve_selections(
        _frame(), ["cat_a"], {"cat_a": ["x", "gone"]})
    assert resolved["cat_a"] == ["x"]


def test_selection_emptied_by_pruning_falls_back_to_all():
    resolved, _, _ = resolve_selections(
        _frame(), ["cat_a"], {"cat_a": ["gone", "also_gone"]})
    assert resolved["cat_a"] == [ALL_LABEL]


def test_all_selection_is_left_alone():
    resolved, _, _ = resolve_selections(
        _frame(), ["cat_a"], {"cat_a": [ALL_LABEL]})
    assert resolved["cat_a"] == [ALL_LABEL]
