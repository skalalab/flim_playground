"""File content in a Markdown-rendered status message.

`st.error`, `st.warning`, `st.info` and `st.toast` render their body as Markdown, and
these messages interpolate column names, cell values, group names and profile names --
text the app does not control. A column called `*note*` used to arrive as *note*, italics
with the asterisks eaten, so the message named a column the file does not contain; a
repeated value of `__x__` came out as a bold `x`.

The rule is `column_roles.code_span` at every interpolation point, never a blanket escape
over a finished message: the `**...**` in the at-the-cap message is ours and deliberate,
and the last test here is what stops a future blanket escape from eating it.
"""
import re

import pandas as pd
import pytest

from src.column_roles import (
    ROLE_CATEGORICAL,
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
    code_span,
    enforce_role_invariants,
)
from src.dataset_io import review_blocking_reason
from src.widgets import review_table_widget as gate

# Names a scientist can plausibly type or export, each carrying Markdown that would
# change or eat the text around it.
MARKUP_NAMES = ("*note*", "__x__", "a_b_c", "[note](x)", "cell`id", "#1", "a*b")

_SPAN = re.compile(r"(`+)(?:.*?)\1", re.DOTALL)


def _prose(message):
    """The message with every code span removed: what Markdown can still interpret."""
    return _SPAN.sub("", message)


def _carries_no_loose_markup(message):
    """No emphasis or link character survives outside a code span."""
    return not set(_prose(message)) & set("*_[]`")


@pytest.fixture
def acw(tmp_path, monkeypatch):
    from src.widgets import analysis_config_widgets as module

    monkeypatch.setattr(module, "_ANALYSIS_CONFIG_PATH", tmp_path / "analysis_config.toml")
    module.st.session_state.pop("current_profile", None)
    return module


# ------------------------------------------------------------------------- code_span

@pytest.mark.parametrize("name", MARKUP_NAMES)
def test_a_span_holds_the_name_exactly_as_spelled(name):
    span = code_span(name)
    assert name in span
    assert _prose(span) == ""          # the whole thing is inside the span


def test_a_name_containing_backticks_gets_a_longer_fence():
    """One backtick would end the span early and spill the rest into the prose."""
    assert code_span("a`b") == "``a`b``"
    assert code_span("a``b") == "```a``b```"


def test_a_name_that_begins_or_ends_with_a_backtick_is_padded():
    """CommonMark strips one leading and one trailing space, so the padding is invisible."""
    assert code_span("`id") == "`` `id ``"
    assert code_span("id`") == "`` id` ``"


def test_an_empty_value_still_makes_a_valid_span():
    """A pair of backticks with nothing between them is not a span at all."""
    assert code_span("") == "` `"


def test_a_non_string_value_is_accepted():
    """Cell values arrive as whatever the column holds."""
    assert code_span(7) == "`7`"


# ------------------------------------------------- the messages that carry file content

@pytest.mark.parametrize("name", MARKUP_NAMES)
def test_the_row_id_reason_spans_both_the_column_and_the_repeated_value(name):
    """It interpolates a column name *and* a raw cell value -- the only message that
    quotes data rather than a header."""
    df = pd.DataFrame({name: ["__x__", "__x__", "c3"],
                       "treatment": ["DMSO", "PD-L1", "DMSO"],
                       "Area": [301.2, 288.5, 340.2]})
    roles = {name: ROLE_ROW_ID, "treatment": ROLE_CATEGORICAL, "Area": ROLE_NUMERICAL}

    message = review_blocking_reason(df, roles)

    assert code_span(name) in message
    assert code_span("__x__") in message
    assert _carries_no_loose_markup(message), message


def test_the_row_id_reason_spans_the_name_in_its_other_two_branches():
    """Empty-everywhere and blank-in-some-rows quote the column but no value."""
    for column in ([None, None, None], ["a", None, "c"]):
        df = pd.DataFrame({"*note*": column, "Area": [1.0, 2.0, 3.0]})
        message = review_blocking_reason(df, {"*note*": ROLE_ROW_ID,
                                              "Area": ROLE_NUMERICAL})
        assert code_span("*note*") in message, message
        assert _carries_no_loose_markup(message), message


def test_the_comma_decimal_note_spans_its_column_for_the_gate():
    """One sentence, two renderers: the gate's Markdown and the reader's plain text.

    `_comma_decimal_hint` is appended to a rejection the reader escapes once through
    `_as_html`, and to this one, which Streamlit renders as Markdown -- hence `mark`.
    """
    from src.dataset_io import _comma_decimal_hint

    df = pd.DataFrame({"cell_id": ["a", "b"], "*val*": ["1,5", "2,5"]})
    # The gate's caller asks for spans; the reader's default keeps the plain quotes that
    # `_as_html` escapes, since a backtick there would just be a backtick on screen.
    assert code_span("*val*") in _comma_decimal_hint(df, mark=code_span)
    assert "'*val*'" in _comma_decimal_hint(df)


def test_the_one_role_per_column_notice_spans_both_names():
    """Two column names in one sentence, from the table's own dropdowns."""
    roles = {"*keeps*": ROLE_ROW_ID, "__loses__": ROLE_ROW_ID}
    _roles, _groups, notices = enforce_role_invariants(
        roles, {}, numeric_cols=set(),
        previous_roles={"*keeps*": ROLE_ROW_ID, "__loses__": ROLE_CATEGORICAL})

    assert notices
    assert code_span("*keeps*") in notices[0]
    assert code_span("__loses__") in notices[0]
    assert _carries_no_loose_markup(notices[0]), notices[0]


def test_the_duplicate_group_warning_spans_the_typed_name():
    gate.st.session_state._review_group_names = ["*lifetime*"]
    message = gate._group_name_error("*lifetime*")
    assert code_span("*lifetime*") in message
    assert _carries_no_loose_markup(message), message


def test_the_ungrouped_marker_warning_spans_the_marker():
    from src.column_roles import NO_GROUP

    gate.st.session_state._review_group_names = []
    message = gate._group_name_error(NO_GROUP)
    assert code_span(NO_GROUP) in message
    assert _carries_no_loose_markup(message), message


@pytest.mark.parametrize("name", MARKUP_NAMES)
def test_the_profile_errors_span_the_profile_name(acw, name):
    """Profile names are typed into the Save-as box, so they are user text too."""
    for message in (acw.rename_profile(name, "other"), acw.delete_profile(name)):
        assert code_span(name) in message
        assert _carries_no_loose_markup(message), message


def test_the_already_exists_error_spans_the_new_name(acw):
    roles = {"cell_id": ROLE_ROW_ID, "Area": ROLE_NUMERICAL}
    acw.save_working_copy("pdl1", roles, {})
    acw.save_working_copy("*taken*", roles, {})

    message = acw.rename_profile("pdl1", "*taken*")

    assert code_span("*taken*") in message
    assert _carries_no_loose_markup(message), message


# --------------------------------------------- the markup that is ours and must survive

def test_the_at_the_cap_message_keeps_its_deliberate_bold(acw):
    """Why the escaping is at the interpolation points and not over whole messages.

    This one interpolates nothing the user wrote -- a count and our own section label --
    and its `**...**` is how it points at the panel that fixes the problem. A blanket
    escape at the render site would turn that into four literal asterisks.
    """
    roles = {"cell_id": ROLE_ROW_ID, "Area": ROLE_NUMERICAL}
    for i in range(acw.MAX_PROFILES):
        acw.save_working_copy(f"p{i}", roles, {})

    message = acw.save_working_copy("one too many", roles, {})

    assert f"**{acw.MANAGE_LABEL}**" in message, message
