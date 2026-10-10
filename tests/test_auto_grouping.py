"""Grouping follows known siblings, then existing group names, then shared prefixes. A
prefix ends at the earliest separator.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.column_roles import detect_column_groups

# Shared prefixes

def test_a_prefix_two_columns_share_becomes_a_group():
    cols = ["nadh_t1_mean", "nadh_t2_mean", "nadh_a1", "fad_t1_mean", "fad_a1"]
    assert detect_column_groups(cols) == {
        "nadh_t1_mean": "nadh", "nadh_t2_mean": "nadh", "nadh_a1": "nadh",
        "fad_t1_mean": "fad", "fad_a1": "fad",
    }


def test_a_column_with_no_separator_is_left_ungrouped():
    """Columns without a shared prefix remain ungrouped."""
    groups = detect_column_groups(["Area", "Perimeter", "nadh_t1", "nadh_t2"])
    assert "Area" not in groups and "Perimeter" not in groups
    assert groups["nadh_t1"] == "nadh"


def test_a_prefix_carried_by_one_column_is_not_a_group():
    """Creating a group requires at least two columns with the same prefix."""
    assert detect_column_groups(["nadh_t1_mean", "Area"]) == {}


def test_the_colon_convention_groups_too():
    cols = ["Intensity: mean", "Intensity: std", "Shape: area", "Shape: perimeter"]
    assert detect_column_groups(cols) == {
        "Intensity: mean": "Intensity", "Intensity: std": "Intensity",
        "Shape: area": "Shape", "Shape: perimeter": "Shape",
    }


def test_the_earliest_separator_in_the_name_wins():
    """The earliest separator wins regardless of its type."""
    groups = detect_column_groups(["nadh_t1.mean", "nadh_t2.mean"])
    assert set(groups.values()) == {"nadh"}


def test_a_hyphen_is_not_a_separator():
    """Hyphens within names such as E-cadherin do not define feature groups."""
    assert detect_column_groups(["E-cadherin area", "E-cadherin intensity"]) == {}


def test_a_hyphen_stays_inside_the_prefix_when_a_real_separator_follows():
    groups = detect_column_groups(["anti-PD1_dose", "anti-PD1_response"])
    assert set(groups.values()) == {"anti-PD1"}


def test_a_name_starting_with_a_separator_has_no_prefix():
    """An empty prefix is not a group name."""
    assert detect_column_groups(["_a", "_b"]) == {}


def test_no_columns_is_no_groups():
    assert detect_column_groups([]) == {}


# Existing group names

def test_a_new_column_joins_a_group_the_profile_already_has():
    """A lone new column can join an existing group with a matching prefix."""
    groups = detect_column_groups(["nadh_t3_mean"], {"nadh": ["nadh_t1_mean"]})
    assert groups == {"nadh_t3_mean": "nadh"}


def test_joining_an_existing_group_works_for_a_single_column():
    """Joining an existing group does not require a second matching column."""
    assert detect_column_groups(["nadh_t3_mean"]) == {}
    assert detect_column_groups(["nadh_t3_mean"], {"nadh": []}) == {"nadh_t3_mean": "nadh"}


def test_existing_groups_may_be_given_as_bare_names():
    assert detect_column_groups(["fad_a1"], ["fad"]) == {"fad_a1": "fad"}


# Known siblings take precedence

def test_a_new_column_joins_the_group_its_sibling_was_filed_under():
    """Known siblings match a renamed group even when its name has no matching prefix."""
    groups = detect_column_groups(["nadh_t3_mean"],
                                  known_groups={"nadh_t1_mean": "NADH lifetime"})
    assert groups == {"nadh_t3_mean": "NADH lifetime"}


def test_a_sibling_outranks_a_group_that_merely_shares_the_prefix_name():
    """A known sibling's saved group takes precedence over a matching group name."""
    groups = detect_column_groups(["nadh_t3_mean"], existing_groups=["nadh"],
                                  known_groups={"nadh_t1_mean": "NADH lifetime"})
    assert groups == {"nadh_t3_mean": "NADH lifetime"}


def test_siblings_that_disagree_decide_nothing():
    """Conflicting sibling groups leave a lone new column ungrouped."""
    groups = detect_column_groups(["nadh_t3_mean"],
                                  known_groups={"nadh_t1_mean": "lifetime",
                                                "nadh_intensity": "intensity"})
    assert groups == {}


def test_a_sibling_whose_name_has_no_prefix_decides_nothing():
    """"Area" carries no key to match on, so it cannot attract anything."""
    assert detect_column_groups(["nadh_t3_mean"],
                                known_groups={"Area": "morphology"}) == {}


# Extraction-style column names


@pytest.mark.parametrize("channel", ["NADH", "long_channel_name"])
def test_recognized_measurements_create_full_groups_even_for_singletons(channel):
    name = f"Lifetime fit_{channel}: T1"
    assert detect_column_groups([name], extractor_hints=["Lifetime fit"]) == {name: f"Lifetime fit_{channel}"}


def test_measurement_channels_have_separate_sibling_keys_and_follow_saved_names():
    columns = ["Lifetime fit_NADH: T2", "Lifetime fit_FAD: T2", "foo_bar: baz", "foo_qux: baz"]
    groups = detect_column_groups(columns, extractor_hints=["Lifetime fit"], known_groups={
        "Lifetime fit_NADH: T1": "renamed NADH", "Lifetime fit_FAD: T1": "renamed FAD"})
    assert groups == {columns[0]: "renamed NADH", columns[1]: "renamed FAD", columns[2]: "foo", columns[3]: "foo"}


@pytest.mark.parametrize("known_group", ["custom", None])
@pytest.mark.parametrize("column, sibling, hints", [
    ("Lifetime fit_NADH: T2", "Lifetime fit_NADH: T1", {"extractor_hints": ["Lifetime fit"]}),
    ("Derived: ratio", "Derived: sum", {}),
    ("NADH_offset", "NADH_amp", {"channel_hints": ["NADH"]}),
    ("ordinary_new", "ordinary_old", {}),
])
def test_unambiguous_saved_siblings_include_explicit_ungrouped_choices(column, sibling, hints, known_group):
    expected = {} if known_group is None else {column: known_group}
    assert detect_column_groups([column], known_groups={sibling: known_group}, **hints) == expected


def test_derived_singleton_uses_derived_features_group():
    assert detect_column_groups(["Derived: ratio"]) == {"Derived: ratio": "Derived Features"}


@pytest.mark.parametrize("channel_source", ["hint", "new measurement", "saved measurement"])
def test_bookkeeping_never_creates_a_channel_group(channel_source):
    columns = ["NADH_amp", "NADH_offset"]
    kwargs = {"extractor_hints": ["Lifetime fit"]}
    if channel_source == "hint":
        kwargs["channel_hints"] = ["NADH"]
    elif channel_source == "new measurement":
        columns.append("Lifetime fit_NADH: T1")
    else:
        kwargs["known_groups"] = {"Lifetime fit_NADH: T1": "lifetime"}
    groups = detect_column_groups(columns, **kwargs)
    assert "NADH_amp" not in groups and "NADH_offset" not in groups


def test_bookkeeping_uses_existing_channel_group_and_longest_channel_match():
    columns = ["ch_long_amp", "ch_offset"]
    assert detect_column_groups(columns, existing_groups=["ch", "ch_long"], channel_hints=["ch", "ch_long"]) == {
        "ch_long_amp": "ch_long", "ch_offset": "ch"}


def test_bookkeeping_siblings_do_not_follow_measurement_siblings():
    assert detect_column_groups(["NADH_offset"], extractor_hints=["Lifetime fit"], known_groups={
        "Lifetime fit_NADH: T1": "lifetime"}) == {}


def test_conflicting_measurement_siblings_fall_back_to_the_full_group():
    name = "Lifetime fit_NADH: T3"
    assert detect_column_groups([name], extractor_hints=["Lifetime fit"], known_groups={
        "Lifetime fit_NADH: T1": "one", "Lifetime fit_NADH: T2": "two"}) == {name: "Lifetime fit_NADH"}
