"""Feature pickers map shortened display names back to exact column names.
Group names can differ from column prefixes: Derived Features contains columns
named Derived: <name>. Both pickers use feature_display_to_column.
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
    """The Derived Features group resolves columns with the Derived: prefix."""
    cols = ["Derived: redox_ratio", "Derived: C"]
    mapping = feature_display_to_column(cols, "Derived Features", data_extraction=True)
    assert list(mapping.keys()) == ["redox_ratio", "C"]
    # Resolve the actual column: Derived: C.
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
    # Split only at the first separator to preserve the rest of an uploaded name.
    mapping = feature_display_to_column(["Derived: a_b_c"], "Derived Features")
    assert mapping == {"a_b_c": "Derived: a_b_c"}
