"""Only a unique nonempty exact column match auto-applies a profile.
Other profiles sharing columns are ranked for the chooser by shared count, missing
count, and name; Auto-detect comes last.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.profile_matching import (
    chooser_is_needed,
    chooser_options,
    compare_columns,
    exact_match,
    rank_profiles,
)

PDL1 = {"cell_id", "image_name", "treatment", "n.t1.mean", "Area", "notes"}


# ------------------------------------------------------------- the identity rule

def test_the_same_column_set_is_an_exact_fit():
    """Arguments are (name, file_cols, profile_cols) -- file first, as everywhere."""
    fit = compare_columns("pdl1", set(PDL1), PDL1)
    assert fit.is_exact
    assert not fit.missing and not fit.new


def test_column_order_does_not_matter():
    """A set comparison, so a reordered file is the same file."""
    reordered = list(PDL1)[::-1]
    assert compare_columns("pdl1", set(reordered), PDL1).is_exact


def test_a_file_with_a_new_measurement_is_not_an_exact_fit():
    """The case that rules out containment."""
    fit = compare_columns("pdl1", PDL1 | {"n.t2.mean"}, PDL1)
    assert not fit.is_exact
    assert fit.new == ("n.t2.mean",)
    assert fit.missing == ()


def test_a_file_missing_a_known_column_is_not_an_exact_fit():
    fit = compare_columns("pdl1", PDL1 - {"Area"}, PDL1)
    assert not fit.is_exact
    assert fit.missing == ("Area",)
    assert fit.new == ()


def test_an_ignored_column_still_counts_toward_the_profile():
    """Ignored columns remain in profile identity so the same file still matches."""
    assert compare_columns("pdl1", set(PDL1), PDL1).is_exact
    assert not compare_columns("pdl1", set(PDL1), PDL1 - {"notes"}).is_exact


# ------------------------------------------------------------------ auto-apply

def test_exactly_one_fit_auto_applies():
    profiles = {"pdl1": PDL1, "iris": {"species", "sepal_length"}}
    assert exact_match(set(PDL1), profiles) == "pdl1"


def test_two_profiles_over_the_same_columns_do_not_auto_apply():
    """Same columns, different groupings -- ambiguous, so the user chooses."""
    profiles = {"pdl1": PDL1, "pdl1-bychannel": set(PDL1)}
    assert exact_match(set(PDL1), profiles) is None


def test_no_fit_does_not_auto_apply():
    assert exact_match({"a", "b"}, {"pdl1": PDL1}) is None


def test_no_saved_profiles_does_not_auto_apply():
    assert exact_match({"a", "b"}, {}) is None


# --------------------------------------------------------------- the chooser

def test_more_shared_columns_ranks_higher():
    profiles = {"few": {"cell_id"}, "many": {"cell_id", "treatment", "Area"}}
    assert [fit.name for fit in rank_profiles(PDL1, profiles)] == ["many", "few"]


def test_fewer_missing_breaks_a_tie_on_shared():
    """Equal shared counts are ordered by fewer missing columns.
    Names sort in the opposite order so the assertion exercises the missing-count key.
    """
    profiles = {
        "a_stale": {"cell_id", "treatment", "gone_a", "gone_b"},
        "z_clean": {"cell_id", "treatment"},
    }
    assert [fit.name for fit in rank_profiles(PDL1, profiles)] == ["z_clean", "a_stale"]


def test_a_profile_sharing_nothing_is_still_offered():
    """`rank_profiles` orders; it never filters. The one cutoff is in `chooser_options`."""
    profiles = {"pdl1": PDL1, "iris": {"species", "sepal_length"}}
    names = [fit.name for fit in rank_profiles(PDL1, profiles)]
    assert names == ["pdl1", "iris"]


def test_the_ranking_reports_all_three_counts():
    """shared / missing / new -- "16 of 18" would be ambiguous about which 18."""
    fit = rank_profiles(PDL1 | {"extra"}, {"pdl1": PDL1 | {"absent"}})[0]
    assert len(fit.shared) == len(PDL1)
    assert fit.missing == ("absent",)
    assert fit.new == ("extra",)


def test_ranking_is_stable_for_identical_fits():
    """Two equally good profiles must not swap places between runs."""
    profiles = {"b_prof": {"cell_id"}, "a_prof": {"cell_id"}}
    first = [fit.name for fit in rank_profiles(PDL1, profiles)]
    assert first == [fit.name for fit in rank_profiles(PDL1, profiles)]
    assert first == ["a_prof", "b_prof"]


def test_no_profiles_ranks_to_nothing():
    assert rank_profiles(PDL1, {}) == []


def test_an_empty_profile_is_not_a_fit_for_an_empty_file():
    """An empty profile is not an exact match, even for an empty file."""
    assert not compare_columns("unsaved", set(), set()).is_exact
    assert exact_match(set(), {"unsaved": set()}) is None


# --------------------------------------------------- what the chooser is handed

def test_the_make_a_new_one_option_comes_last():
    """The chooser lists ranked candidates first and Auto-detect last."""
    profiles = {"partial": {"treatment", "species"}, "pdl1": PDL1}
    assert chooser_options(PDL1, profiles, "NEW") == ["pdl1", "partial", "NEW"]


def test_a_profile_sharing_no_column_is_left_out_of_the_chooser():
    """Profiles with no shared columns are excluded from the chooser."""
    profiles = {"iris": {"species", "sepal_length"}, "pdl1": PDL1}
    assert chooser_options(PDL1, profiles, "NEW") == ["pdl1", "NEW"]


def test_a_profile_that_knows_no_columns_is_left_out_of_the_chooser():
    """An empty profile shares nothing with any file, so the same test drops it."""
    assert chooser_options(PDL1, {"unsaved": set()}, "NEW") == ["NEW"]


def test_a_single_shared_column_is_enough_to_be_listed():
    """The cutoff is zero shared, not a proportion -- there is no tuning knob here."""
    assert chooser_options(PDL1, {"thin": {"notes", "a", "b", "c"}}, "NEW") == ["thin", "NEW"]


def test_with_no_profiles_the_chooser_offers_only_the_new_one():
    assert chooser_options(PDL1, {}, "NEW") == ["NEW"]


# ------------------------------------------------- whether to ask the question

def test_no_chooser_once_the_applied_profile_describes_the_file_exactly():
    """Reopening an exact match suppresses the chooser."""
    assert not chooser_is_needed("pdl1", PDL1, {"pdl1": PDL1, "iris": {"species"}})


def test_the_chooser_is_needed_while_nothing_is_applied():
    assert chooser_is_needed(None, PDL1, {"pdl1": PDL1})


def test_the_chooser_stays_while_the_applied_profile_only_partly_fits():
    """Picking a 16-of-18 profile must not make the list vanish under the cursor."""
    assert chooser_is_needed("pdl1", PDL1 | {"extra"}, {"pdl1": PDL1})


def test_an_applied_name_no_profile_carries_still_needs_the_chooser():
    assert chooser_is_needed("deleted", PDL1, {"pdl1": PDL1})
