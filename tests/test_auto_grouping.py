"""Grouping follows known siblings, then existing group names, then shared prefixes. A
prefix ends at the earliest separator.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_an_extraction_style_name_is_cut_at_the_earliest_separator_like_any_other():
    """Extraction-style names use the same prefix rule as every other uploaded column.
    """
    groups = detect_column_groups(["Lifetime fit_ch1: T1", "Lifetime fit_ch2: T1",
                                   "foo_bar: baz", "foo_qux: baz"])

    assert groups == {"Lifetime fit_ch1: T1": "Lifetime fit",
                      "Lifetime fit_ch2: T1": "Lifetime fit",
                      "foo_bar: baz": "foo", "foo_qux: baz": "foo"}
