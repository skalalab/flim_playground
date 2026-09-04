"""Guessing which feature group a new column belongs to.

Groups only organise the feature pickers, so this is the one place the design can
guess freely: a wrong role removes a measurement from the analysis, while a wrong
group sorts a dropdown oddly. Nothing is lost and the mistake is visible in the same
table that made it -- so the rules lean towards offering an answer, where the role
rules lean towards abstaining.

Two rules in the pure half (join an existing group, then form groups from shared
prefixes), and one in the dataset_io wrapper that runs ahead of both: a table written
in the extraction naming convention already carries its own grouping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.column_roles import detect_column_groups

# ------------------------------------------------ rule 2: shared prefixes

def test_a_prefix_two_columns_share_becomes_a_group():
    cols = ["nadh_t1_mean", "nadh_t2_mean", "nadh_a1", "fad_t1_mean", "fad_a1"]
    assert detect_column_groups(cols) == {
        "nadh_t1_mean": "nadh", "nadh_t2_mean": "nadh", "nadh_a1": "nadh",
        "fad_t1_mean": "fad", "fad_a1": "fad",
    }


def test_a_column_with_no_separator_is_left_ungrouped():
    """Absent from the result, so it falls to Uncategorized Features as always."""
    groups = detect_column_groups(["Area", "Perimeter", "nadh_t1", "nadh_t2"])
    assert "Area" not in groups and "Perimeter" not in groups
    assert groups["nadh_t1"] == "nadh"


def test_a_prefix_carried_by_one_column_is_not_a_group():
    """A group of one is not a group -- definition, not a tunable cutoff."""
    assert detect_column_groups(["nadh_t1_mean", "Area"]) == {}


def test_the_colon_convention_groups_too():
    cols = ["Intensity: mean", "Intensity: std", "Shape: area", "Shape: perimeter"]
    assert detect_column_groups(cols) == {
        "Intensity: mean": "Intensity", "Intensity: std": "Intensity",
        "Shape: area": "Shape", "Shape: perimeter": "Shape",
    }


def test_the_earliest_separator_in_the_name_wins():
    """No precedence between separators to argue about -- position decides."""
    groups = detect_column_groups(["nadh_t1.mean", "nadh_t2.mean"])
    assert set(groups.values()) == {"nadh"}


def test_a_hyphen_is_not_a_separator():
    """":", "_" and "." mark structure; a hyphen usually sits *inside* a word.

    "E-cadherin", "anti-PD1", "t-SNE", "2026-08-27" -- cutting at the hyphen names the
    group "E" or "2026", and a junk name is worse than no group because it is sticky:
    rule 1 recruits every later column that starts with it, and Save writes it into the
    profile.
    """
    assert detect_column_groups(["E-cadherin area", "E-cadherin intensity"]) == {}


def test_a_hyphen_stays_inside_the_prefix_when_a_real_separator_follows():
    groups = detect_column_groups(["anti-PD1_dose", "anti-PD1_response"])
    assert set(groups.values()) == {"anti-PD1"}


def test_a_name_starting_with_a_separator_has_no_prefix():
    """An empty prefix is not a group name."""
    assert detect_column_groups(["_a", "_b"]) == {}


def test_no_columns_is_no_groups():
    assert detect_column_groups([]) == {}


# ------------------------------------------- rule 1: join an existing group

def test_a_new_column_joins_a_group_the_profile_already_has():
    """Nothing is invented -- `nadh` exists because the user made it."""
    groups = detect_column_groups(["nadh_t3_mean"], {"nadh": ["nadh_t1_mean"]})
    assert groups == {"nadh_t3_mean": "nadh"}


def test_joining_an_existing_group_works_for_a_single_column():
    """The 2+ rule is for *forming* a group, not for joining one that exists."""
    assert detect_column_groups(["nadh_t3_mean"]) == {}
    assert detect_column_groups(["nadh_t3_mean"], {"nadh": []}) == {"nadh_t3_mean": "nadh"}


def test_existing_groups_may_be_given_as_bare_names():
    assert detect_column_groups(["fad_a1"], ["fad"]) == {"fad_a1": "fad"}


# ------------------------------- rule 1, continued: follow the siblings

def test_a_new_column_joins_the_group_its_sibling_was_filed_under():
    """The rename case. "NADH lifetime" matches no prefix, so only the sibling finds it."""
    groups = detect_column_groups(["nadh_t3_mean"],
                                  known_groups={"nadh_t1_mean": "NADH lifetime"})
    assert groups == {"nadh_t3_mean": "NADH lifetime"}


def test_a_sibling_outranks_a_group_that_merely_shares_the_prefix_name():
    """Where the user actually filed a column like this one beats a matching string."""
    groups = detect_column_groups(["nadh_t3_mean"], existing_groups=["nadh"],
                                  known_groups={"nadh_t1_mean": "NADH lifetime"})
    assert groups == {"nadh_t3_mean": "NADH lifetime"}


def test_siblings_that_disagree_decide_nothing():
    """Two groups hold "nadh" columns, so the prefix is not the user's grouping axis.

    Picking one would make the answer depend on an order the user cannot see, and the
    prefix rule below already abstains for a lone column.
    """
    groups = detect_column_groups(["nadh_t3_mean"],
                                  known_groups={"nadh_t1_mean": "lifetime",
                                                "nadh_intensity": "intensity"})
    assert groups == {}


def test_a_sibling_whose_name_has_no_prefix_decides_nothing():
    """"Area" carries no key to match on, so it cannot attract anything."""
    assert detect_column_groups(["nadh_t3_mean"],
                                known_groups={"Area": "morphology"}) == {}


# ------------------------------- the wrapper: the extraction convention first

def _extraction_cols():
    return ["Lifetime fit_ch1: T1", "Lifetime fit_ch1: T2", "Morphology_ch1: Area"]


def test_the_extraction_convention_wins_over_the_prefix_rule(monkeypatch):
    """The case that justifies the wrapper existing at all.

    These columns carry their own grouping. The prefix rule would cut them at the
    underscore and file every channel under "Lifetime fit" / "Morphology".
    """
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors",
                        lambda: ["Lifetime fit", "Morphology"])

    groups = dataset_io.detect_groups(_extraction_cols())

    assert groups == {"Lifetime fit_ch1: T1": "Lifetime fit_ch1",
                      "Lifetime fit_ch1: T2": "Lifetime fit_ch1",
                      "Morphology_ch1: Area": "Morphology_ch1"}
    assert "Lifetime fit" not in groups.values()   # what the prefix rule would say


def test_the_prefix_rule_runs_when_the_convention_does_not_apply(monkeypatch):
    """A table it cannot parse comes back all-Uncategorized, which is the fall-through."""
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors", list)

    assert dataset_io.detect_groups(["nadh_t1", "nadh_t2"]) == {"nadh_t1": "nadh",
                                                               "nadh_t2": "nadh"}


def test_the_prefix_rule_picks_up_what_the_convention_could_not_parse(monkeypatch):
    """The fall-through is per column, not per file.

    The convention runs first because the prefix rule would butcher the names it *can*
    read -- a risk that does not exist for the ones it declined. So a mixed table (an
    extraction CSV with a few hand-added columns) gets both answers.
    """
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors",
                        lambda: ["Lifetime fit"])

    groups = dataset_io.detect_groups(["Lifetime fit_ch1: T1", "Lifetime fit_ch1: T2",
                                       "notes_a", "notes_b"])

    assert groups == {"Lifetime fit_ch1: T1": "Lifetime fit_ch1",
                      "Lifetime fit_ch1: T2": "Lifetime fit_ch1",
                      "notes_a": "notes", "notes_b": "notes"}


def test_a_renamed_convention_group_attracts_its_new_siblings(monkeypatch):
    """Without this, T3 resurrects "Lifetime fit_ch1" beside the name the user chose."""
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors",
                        lambda: ["Lifetime fit"])

    groups = dataset_io.detect_groups(
        ["Lifetime fit_ch1: T3"],
        known_groups={"Lifetime fit_ch1: T1": "NADH lifetime"})

    assert groups == {"Lifetime fit_ch1: T3": "NADH lifetime"}


def test_a_sibling_in_another_channel_does_not_attract(monkeypatch):
    """The convention key is `{extractor}_{channel}`, so ch2 is not ch1's sibling."""
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors",
                        lambda: ["Lifetime fit"])

    groups = dataset_io.detect_groups(
        ["Lifetime fit_ch2: T1"],
        known_groups={"Lifetime fit_ch1: T1": "NADH lifetime"})

    assert groups == {"Lifetime fit_ch2: T1": "Lifetime fit_ch2"}


def test_a_column_the_convention_grouped_does_not_feed_the_prefix_count(monkeypatch):
    """Only the leftovers are counted, or the convention would compete with itself.

    Both names below share the prefix "Lifetime". Counting them together would form a
    group "Lifetime" holding a column the convention had already filed under
    "Lifetime fit_ch1".
    """
    monkeypatch.setattr(dataset_io, "get_all_feature_extractors",
                        lambda: ["Lifetime fit"])

    groups = dataset_io.detect_groups(["Lifetime fit_ch1: T1", "Lifetime fit_extra"])

    assert groups == {"Lifetime fit_ch1: T1": "Lifetime fit_ch1"}
