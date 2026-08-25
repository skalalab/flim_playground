"""Round-trip for the feature-picker's display-name <-> real-column mapping.

The Data-Analysis feature pickers show only the part after ``": "`` (e.g. "t1" for
"Lifetime fit_nadh: t1") and must resolve a selection back to the full column name.
The old code rebuilt it as ``f"{group}: {name}"``, which assumed the group name
equals the column prefix. The cross-channel "Derived Features" group breaks that
invariant (its columns are "Derived: <name>"), so selecting a derived feature
produced the bogus "Derived Features: <name>" and raised KeyError in data_analysis.

These lock in ``feature_display_to_column``, the single mapping both pickers use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets.selection_widgets import feature_display_to_column


def test_normal_group_strips_prefix_and_maps_back_to_full_column():
    cols = ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: a1"]
    mapping = feature_display_to_column(cols, "Lifetime fit_nadh", data_extraction=True)
    # Picker shows the short names...
    assert list(mapping.keys()) == ["t1", "a1"]
    # ...but each resolves back to the real column.
    assert mapping["t1"] == "Lifetime fit_nadh: t1"
    assert mapping["a1"] == "Lifetime fit_nadh: a1"


def test_derived_features_group_round_trips_to_real_derived_column():
    """Regression: the group name "Derived Features" != the column prefix "Derived"."""
    cols = ["Derived: redox_ratio", "Derived: C"]
    mapping = feature_display_to_column(cols, "Derived Features", data_extraction=True)
    assert list(mapping.keys()) == ["redox_ratio", "C"]
    # The exact bug: this must be "Derived: C", NOT "Derived Features: C".
    assert mapping["C"] == "Derived: C"
    assert mapping["redox_ratio"] == "Derived: redox_ratio"


def test_uncategorized_group_is_identity():
    cols = ["nadh_offset", "nadh_reduced_chi_square"]
    mapping = feature_display_to_column(cols, "Uncategorized Features", data_extraction=True)
    assert mapping == {"nadh_offset": "nadh_offset",
                       "nadh_reduced_chi_square": "nadh_reduced_chi_square"}


def test_non_data_extraction_is_identity_even_with_colon_columns():
    # User-uploaded CSVs: columns are used verbatim, no prefix stripping.
    cols = ["my col: raw", "another"]
    mapping = feature_display_to_column(cols, "Some Group", data_extraction=False)
    assert mapping == {"my col: raw": "my col: raw", "another": "another"}


def test_derived_name_with_only_one_colon_split():
    # split(": ", 1) keeps everything after the first ": " (defensive; the builder
    # already forbids ": " in derived names, but never rely on messy column text).
    mapping = feature_display_to_column(["Derived: a_b_c"], "Derived Features")
    assert mapping == {"a_b_c": "Derived: a_b_c"}
