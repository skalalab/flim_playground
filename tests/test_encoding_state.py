import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.widgets.encoding_state import color_multiselect_label, prune_to_options


class TestPruneToOptions:
    def test_list_keeps_only_offered_entries(self):
        assert prune_to_options(["a", "b"], ["a", "c"]) == ["a"]

    def test_list_preserves_stored_order(self):
        assert prune_to_options(["b", "a"], ["a", "b"]) == ["b", "a"]

    def test_emptied_list_falls_back(self):
        assert prune_to_options(["x"], ["a", "b"], fallback=["a"]) == ["a"]

    def test_emptied_list_without_fallback_stays_empty(self):
        assert prune_to_options(["x"], ["a"]) == []

    def test_already_empty_list_does_not_take_fallback(self):
        # An empty selection is a deliberate user state, not something to repair.
        assert prune_to_options([], ["a"], fallback=["a"]) == []

    def test_scalar_kept_when_offered(self):
        assert prune_to_options("a", ["a", "b"]) == "a"

    def test_scalar_dropped_when_not_offered(self):
        assert prune_to_options("z", ["a", "b"]) is None

    def test_none_stays_none(self):
        assert prune_to_options(None, ["a"]) is None

    def test_missing_state_stays_none(self):
        # st.session_state.get() returns None for a key that was never written.
        assert prune_to_options(None, []) is None


class TestColorMultiselectLabel:
    """Two booleans, so four inputs and four tests. Exactly one control CLAIMS colour at
    any time: the moment the switch goes on, the third slot is the one offering it, so the
    multiselect must give the name up in the same run. Named after the claim rather than
    the wording because the third slot no longer spells "Color by" on screen -- it shows a
    static phrase and reads "Subcolor by" only to a screen reader."""

    def test_switch_on_hands_the_name_over(self):
        assert color_multiselect_label(True, True) == "Group by"

    def test_switch_off_keeps_the_name(self):
        assert color_multiselect_label(True, False) == "Color by"

    def test_no_switch_rendered_keeps_the_name(self):
        # Every method except Feature Comparison; there is no second colour channel to
        # hand the name to, whatever a stale stored value says.
        assert color_multiselect_label(False, True) == "Color by"

    def test_no_switch_and_off_keeps_the_name(self):
        assert color_multiselect_label(False, False) == "Color by"
